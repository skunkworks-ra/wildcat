"""Per-decision LLM orchestrator for wildcat.

The Orchestrator drives the workflow state machine:
  read state → call MCP tools → assemble prompt → LLM → parse decision
  → (optionally) run CASA → transition state → repeat

The LLM is invoked once per decision point (stateless between calls).
Long CASA jobs don't cause timeouts because the orchestrator awaits
an asyncio.Event set by SentinelWatcher, not a HTTP response.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path

from wildcat.llm import LLMBackend
from wildcat.runner import CASARunner
from wildcat.skills import load_system_prompt
from wildcat.state import Stage, StateDB
from wildcat.tools import MSInspectClient

log = logging.getLogger(__name__)

# JSON schema that every LLM response must conform to
_DECISION_SCHEMA_KEYS = {"next_stage", "casa_script", "summary", "reasoning"}

_VALID_NEXT_STAGES = {s.value for s in Stage} - {Stage.IDLE, Stage.ERROR}

# System instruction appended to every prompt asking for structured output
_JSON_INSTRUCTION = """
Respond with a single JSON object and no other text. Schema:
{
  "next_stage": "<one of: PHASE2_RUNNING, PHASE3_RUNNING, HUMAN_CHECKPOINT, CALIBRATION_LOOP, IMAGING_PIPELINE, ERROR>",
  "casa_script": "<python script string, or null if no CASA job is needed>",
  "summary": "<human-readable summary of your findings (2-5 sentences)>",
  "reasoning": "<brief reasoning trace explaining the decision>"
}
"""


class Orchestrator:
    def __init__(
        self,
        db: StateDB,
        tools: MSInspectClient,
        llm: LLMBackend,
        skills_path: str,
        runner: CASARunner,
        checkpoint_event: asyncio.Event,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        self.db = db
        self.tools = tools
        self.llm = llm
        self.skills_path = skills_path
        self.runner = runner
        self.checkpoint_event = checkpoint_event
        self.stop_event = stop_event or asyncio.Event()

    # ── Main loop ──────────────────────────────────────────────────────────

    async def run(self, workflow_id: int) -> None:
        """Main loop. Called on startup (or restart after checkpoint).

        Reads the current stage, dispatches to the correct handler, and
        exits when the workflow reaches a terminal or waiting state.
        """
        while True:
            wf = self.db.get_workflow(workflow_id)
            stage = Stage(wf["stage"])
            log.info("Workflow %d: stage=%s", workflow_id, stage)

            if stage == Stage.IDLE:
                self.db.transition(workflow_id, Stage.PHASE1_RUNNING)

            elif stage == Stage.PHASE1_RUNNING:
                await self._handle_phase(workflow_id, phase=1)

            elif stage == Stage.PHASE2_RUNNING:
                await self._handle_phase(workflow_id, phase=2)

            elif stage == Stage.PHASE3_RUNNING:
                await self._handle_phase(workflow_id, phase=3)

            elif stage == Stage.HUMAN_CHECKPOINT:
                await self._handle_checkpoint(workflow_id)

            elif stage in (Stage.IMAGING_PIPELINE, Stage.CALIBRATION_LOOP):
                log.info("Workflow %d reached %s — handoff complete", workflow_id, stage)
                break

            elif stage == Stage.STOPPED:
                log.info("Workflow %d stopped by user request", workflow_id)
                break

            elif stage == Stage.ERROR:
                log.error("Workflow %d is in ERROR state — manual intervention required", workflow_id)
                break

            else:
                # PHASE*_COMPLETE — advance to the next running state
                next_stage = {
                    Stage.PHASE1_COMPLETE: Stage.PHASE2_RUNNING,
                    Stage.PHASE2_COMPLETE: Stage.PHASE3_RUNNING,
                    Stage.PHASE3_COMPLETE: Stage.HUMAN_CHECKPOINT,
                }.get(stage)
                if next_stage:
                    self.db.transition(workflow_id, next_stage)
                else:
                    log.warning("Unhandled stage %s — stopping", stage)
                    break

    # ── Phase handler ──────────────────────────────────────────────────────

    async def _handle_phase(self, workflow_id: int, phase: int) -> None:
        """Run MCP tools → LLM decision → optional CASA job → transition."""
        wf = self.db.get_workflow(workflow_id)
        ms_path: str = wf["ms_path"]
        stage = Stage(wf["stage"])

        # 0. Check for stop before starting any work
        if self.stop_event.is_set():
            log.info("Stop requested — halting before Phase %d tools", phase)
            self.db.transition(workflow_id, Stage.STOPPED)
            return

        # 1. Call MCP tools
        log.info("Running Phase %d tools on %s", phase, ms_path)
        runner_map = {1: self.tools.run_phase1, 2: self.tools.run_phase2, 3: self.tools.run_phase3}
        tool_outputs: dict[str, dict] = await runner_map[phase](ms_path)

        # 2. Persist tool outputs
        phase_label = f"phase{phase}"
        for tool_name, output in tool_outputs.items():
            self.db.save_tool_output(
                workflow_id, phase_label, tool_name, json.dumps(output)
            )

        # 3. Assemble prompt
        system_prompt = load_system_prompt(self.skills_path, stage)
        user_content = self._format_tool_outputs(tool_outputs, phase)
        messages = [
            {"role": "system", "content": system_prompt + "\n\n" + _JSON_INSTRUCTION},
            {"role": "user",   "content": user_content},
        ]

        # 4. LLM decision — check for stop before invoking LLM
        if self.stop_event.is_set():
            log.info("Stop requested — halting before LLM call in Phase %d", phase)
            self.db.transition(workflow_id, Stage.STOPPED)
            return
        response = await self.llm.complete(messages)
        raw_decision = self._extract_content(response)
        decision = self._parse_decision(raw_decision)

        # 5. Persist LLM decision
        prompt_hash = hashlib.sha256(user_content.encode()).hexdigest()[:16]
        model = response.get("model", "unknown")
        self.db.save_llm_decision(
            workflow_id, stage.value, json.dumps(decision), model, prompt_hash
        )

        # 6. Optional CASA job — check for stop before launching subprocess
        if decision.get("casa_script"):
            if self.stop_event.is_set():
                log.info("Stop requested — skipping CASA job in Phase %d", phase)
                self.db.transition(workflow_id, Stage.STOPPED)
                return
            await self._run_casa_job(workflow_id, stage.value, decision["casa_script"])

        # 7. Transition to the next stage from the LLM decision
        next_stage_str = decision["next_stage"]
        try:
            next_stage = Stage(next_stage_str)
        except ValueError:
            log.error("LLM returned unknown next_stage %r — entering ERROR", next_stage_str)
            self.db.transition(workflow_id, Stage.ERROR)
            return

        self.db.transition(workflow_id, next_stage)

    # ── Checkpoint handler ─────────────────────────────────────────────────

    async def _handle_checkpoint(self, workflow_id: int) -> None:
        """Write checkpoint row, signal UI, await human decision."""
        wf = self.db.get_workflow(workflow_id)

        # Re-use the summary from the last LLM decision — no second LLM call needed.
        last_decision = self.db.conn.execute(
            "SELECT decision FROM llm_decisions WHERE workflow_id = ?"
            " ORDER BY decided_at DESC LIMIT 1",
            (workflow_id,),
        ).fetchone()
        if last_decision:
            llm_summary = json.loads(last_decision["decision"]).get("summary", "No summary available.")
        else:
            llm_summary = "No summary available."

        checkpoint_id = self.db.create_checkpoint(
            workflow_id, Stage.HUMAN_CHECKPOINT.value, llm_summary
        )
        log.info(
            "Checkpoint %d created for workflow %d — awaiting human decision",
            checkpoint_id, workflow_id,
        )

        # Wait for the UI to fire the event (via POST /checkpoint/{id}/decide)
        self.checkpoint_event.clear()
        await self.checkpoint_event.wait()

        # Read what the human decided
        checkpoint = self.db.get_latest_checkpoint(workflow_id)
        human_route = (checkpoint or {}).get("human_route", "imaging")

        if human_route == "calibration":
            self.db.transition(workflow_id, Stage.CALIBRATION_LOOP)
        else:
            self.db.transition(workflow_id, Stage.IMAGING_PIPELINE)

    # ── CASA job helper ────────────────────────────────────────────────────

    async def _run_casa_job(
        self, workflow_id: int, stage_label: str, script_content: str
    ) -> None:
        """Write script to disk, submit to runner, await sentinel."""
        script_dir = Path(self.runner.config.jobs_dir) / str(workflow_id)
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / f"{stage_label}.py"
        script_path.write_text(script_content, encoding="utf-8")

        job_id = self.db.create_job(workflow_id, stage_label, str(script_path))

        # Run CASA in the background; wait for sentinel
        self.checkpoint_event.clear()
        asyncio.create_task(self.runner.submit(job_id, str(script_path)))
        await self.checkpoint_event.wait()

    # ── Parsing helpers ────────────────────────────────────────────────────

    def _parse_decision(self, raw: str) -> dict:
        """Extract and validate the LLM JSON decision.

        Raises ValueError on malformed output — we do not trust the LLM
        blindly; the caller maps this to Stage.ERROR.
        """
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            )

        try:
            decision = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM response is not valid JSON: {exc}\n---\n{raw}") from exc

        missing = _DECISION_SCHEMA_KEYS - decision.keys()
        if missing:
            raise ValueError(f"LLM decision missing required keys: {missing}")

        return decision

    def _extract_content(self, response: dict) -> str:
        """Pull the assistant message text out of an OpenAI response dict."""
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Unexpected LLM response shape: {exc}\n{response}") from exc

    def _format_tool_outputs(self, outputs: dict[str, dict], phase: int) -> str:
        """Format tool outputs into a structured prompt string."""
        lines = [f"## Phase {phase} tool outputs\n"]
        for tool_name, data in outputs.items():
            lines.append(f"### {tool_name}")
            lines.append("```json")
            lines.append(json.dumps(data, indent=2))
            lines.append("```\n")
        return "\n".join(lines)
