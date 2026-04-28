"""FastAPI checkpoint UI for wildcat.

Provides a minimal human-in-the-loop interface:
  GET  /checkpoint/{workflow_id}           — render checkpoint page
  POST /checkpoint/{workflow_id}/decide    — record human decision
  GET  /jobs/<path>                        — static file serving for plots

The checkpoint_event is shared with the Orchestrator — posting a decision
fires the event which unblocks orchestrator._handle_checkpoint().
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from wildcat.orchestrator import _QUESTION_CONFIG_MAP
from wildcat.state import Stage, StateDB

log = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


_LOG_PATH = "/var/log/wildcat.log"


async def _tail_log(log_path: str):
    """Yield SSE-formatted lines by tailing a log file.

    Replays the last 500 lines on connect so the UI is never empty,
    then follows new lines as they arrive.
    """
    try:
        with open(log_path) as f:
            # Replay existing content (last 500 lines) before following
            lines = f.readlines()
            for line in lines[-500:]:
                yield f"data: {line.rstrip()}\n\n"
            # Now follow new lines
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                else:
                    await asyncio.sleep(0.2)
    except FileNotFoundError:
        yield f"data: [log file not found: {log_path}]\n\n"


def build_ui_app(
    db: StateDB,
    checkpoint_event: asyncio.Event,
    stop_event: asyncio.Event | None = None,
    start_event: asyncio.Event | None = None,
    ms_path: str | None = None,
) -> FastAPI:
    """Construct the FastAPI application with all routes wired up."""
    app = FastAPI(title="wildcat checkpoint UI")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    templates.env.filters["tojson"] = lambda v, **kw: json.dumps(v, **kw)

    router = APIRouter()

    @router.get("/start", response_class=HTMLResponse)
    async def start_page(request: Request) -> HTMLResponse:
        """Waiting screen shown when autostart is not set."""
        # Predict the next workflow_id from the current max (SQLite AUTOINCREMENT).
        row = db.conn.execute("SELECT MAX(id) AS max_id FROM workflow").fetchone()
        next_id = (row["max_id"] or 0) + 1
        already_started = start_event is not None and start_event.is_set()
        return templates.TemplateResponse(
            request,
            "start.html",
            {
                "ms_path": ms_path or "unknown",
                "next_workflow_id": next_id,
                "already_started": already_started,
            },
        )

    @router.post("/start")
    async def do_start(request: Request) -> HTMLResponse:
        """Fire the start event and redirect to the starting page."""
        if start_event is not None and not start_event.is_set():
            start_event.set()
            log.info("Pipeline start triggered via UI")
        # next_id prediction: same logic as start_page
        row = db.conn.execute("SELECT MAX(id) AS max_id FROM workflow").fetchone()
        next_id = (row["max_id"] or 0) + 1
        return templates.TemplateResponse(
            request,
            "start.html",
            {
                "ms_path": ms_path or "unknown",
                "next_workflow_id": next_id,
                "already_started": True,
            },
        )

    @router.get("/checkpoint/{workflow_id}", response_class=HTMLResponse)
    async def show_checkpoint(request: Request, workflow_id: int) -> HTMLResponse:
        """Render the checkpoint review page."""
        try:
            wf = db.get_workflow(workflow_id)
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"Workflow {workflow_id} not found"
            )

        checkpoint = db.get_latest_checkpoint(workflow_id)
        if checkpoint is None:
            raise HTTPException(
                status_code=404,
                detail=f"No checkpoint found for workflow {workflow_id}",
            )

        # Collect tool outputs across all phases for the metrics table
        tool_outputs: dict[str, dict] = {}
        for phase in ("phase1", "phase2", "phase3"):
            for row in db.get_tool_outputs(workflow_id, phase):
                try:
                    tool_outputs[row["tool_name"]] = json.loads(row["output_json"])
                except (json.JSONDecodeError, KeyError):
                    tool_outputs[row["tool_name"]] = {"raw": row.get("output_json", "")}

        # Extract checkpoint_questions from the last LLM decision
        checkpoint_questions: list[dict] = []
        last_dec_row = db.conn.execute(
            "SELECT decision FROM llm_decisions WHERE workflow_id = ?"
            " ORDER BY decided_at DESC LIMIT 1",
            (workflow_id,),
        ).fetchone()
        if last_dec_row:
            try:
                last_dec = json.loads(last_dec_row["decision"])
                checkpoint_questions = last_dec.get("checkpoint_questions") or []
            except (json.JSONDecodeError, KeyError):
                checkpoint_questions = []

        # Collect plots from completed jobs
        plots: list[str] = []
        jobs = db.conn.execute(
            "SELECT plots FROM jobs WHERE workflow_id = ? AND plots IS NOT NULL",
            (workflow_id,),
        ).fetchall()
        for j in jobs:
            try:
                plots.extend(json.loads(j["plots"]) if j["plots"] else [])
            except (json.JSONDecodeError, TypeError):
                pass

        return templates.TemplateResponse(
            request,
            "checkpoint.html",
            {
                "workflow": wf,
                "checkpoint": checkpoint,
                "tool_outputs": tool_outputs,
                "checkpoint_questions": checkpoint_questions,
                "plots": plots,
            },
        )

    @router.get("/status/{workflow_id}", response_class=HTMLResponse)
    async def status(request: Request, workflow_id: int) -> HTMLResponse:
        """Live status fragment — polled by the UI every 3s via HTMX."""
        try:
            wf = db.get_workflow(workflow_id)
            stage = wf["stage"]
            updated = wf["updated_at"]
        except KeyError:
            return HTMLResponse("<span style='color:#f87171'>Workflow not found</span>")
        cp = db.get_latest_checkpoint(workflow_id)
        color = "#86efac" if stage == "HUMAN_CHECKPOINT" else "#7dd3fc"
        cp_info = f" · checkpoint #{cp['id']}" if cp else " · awaiting checkpoint"
        reload = ""
        if cp:
            reload = "<script>if(!window._reloaded){window._reloaded=true;location.reload();}</script>"
        return HTMLResponse(
            f"<span style='color:{color}'>● {stage}</span>"
            f"<span style='color:#64748b; font-size:0.8rem; margin-left:1rem'>{updated}{cp_info}</span>"
            f"{reload}"
        )

    @router.get("/checkpoint/{workflow_id}/panel", response_class=HTMLResponse)
    async def checkpoint_panel(request: Request, workflow_id: int) -> HTMLResponse:
        """Fragment for the pipeline monitor checkpoint section.

        While no checkpoint exists, returns a polling div so HTMX keeps checking.
        Once a checkpoint with questions is ready, returns a stable form div
        WITHOUT hx-trigger — HTMX stops swapping and the form is safe to interact with.
        """
        checkpoint = db.get_latest_checkpoint(workflow_id)

        # Pull checkpoint_questions from last LLM decision
        checkpoint_questions: list[dict] = []
        if checkpoint:
            row = db.conn.execute(
                "SELECT decision FROM llm_decisions WHERE workflow_id = ?"
                " ORDER BY decided_at DESC LIMIT 1",
                (workflow_id,),
            ).fetchone()
            if row:
                try:
                    checkpoint_questions = (
                        json.loads(row["decision"]).get("checkpoint_questions") or []
                    )
                except (json.JSONDecodeError, KeyError):
                    pass

        already_decided = checkpoint and checkpoint.get("human_route") is not None

        return templates.TemplateResponse(
            request,
            "checkpoint_panel.html",
            {
                "workflow_id": workflow_id,
                "checkpoint": checkpoint,
                "checkpoint_questions": checkpoint_questions,
                "already_decided": already_decided,
            },
        )

    @router.post("/checkpoint/{workflow_id}/decide")
    async def decide(request: Request, workflow_id: int) -> dict:
        """Record human per-question answers and unblock the orchestrator.

        Form fields:
          notes            — optional free-text notes
          answer_<id>      — one field per checkpoint question, e.g. answer_polcal=proceed
        """
        checkpoint = db.get_latest_checkpoint(workflow_id)
        if checkpoint is None:
            raise HTTPException(
                status_code=404,
                detail=f"No pending checkpoint for workflow {workflow_id}",
            )

        form = await request.form()
        notes: str = form.get("notes", "")  # type: ignore[assignment]

        # Collect answers: {"polcal": "proceed", "aggressive_flagging": "yes", ...}
        answers: dict[str, str] = {
            key[len("answer_") :]: str(value)
            for key, value in form.items()
            if key.startswith("answer_")
        }

        # Route resolution (priority: exit > loop_back > proceed/yes/no)
        # After HUMAN_CHECKPOINT approval: → CALIBRATION_PREFLAG (begin calibration)
        # After CALIBRATION_CHECKPOINT approval: → IMAGING_PIPELINE
        answer_values = set(answers.values())
        checkpoint_stage = checkpoint.get("stage", "")
        if "exit" in answer_values:
            route = Stage.STOPPED.value
        elif "loop_back" in answer_values:
            route = Stage.CALIBRATION_LOOP.value
        elif checkpoint_stage == Stage.HUMAN_CHECKPOINT.value:
            route = Stage.CALIBRATION_PREFLAG.value
        else:
            route = Stage.IMAGING_PIPELINE.value

        # Apply _QUESTION_CONFIG_MAP to derive config updates
        try:
            wf_config = db.get_workflow_config(workflow_id)
        except KeyError:
            wf_config = {"polcal": True, "aggressive_flagging": False}

        for question_id, answer in answers.items():
            mapping = _QUESTION_CONFIG_MAP.get(question_id, {})
            update = mapping.get(answer)
            if update is not None:
                config_key, config_value = update
                wf_config[config_key] = config_value

        db.set_workflow_config(workflow_id, wf_config)

        db.resolve_checkpoint(
            checkpoint["id"],
            human_route=route,
            human_notes=notes,
            question_answers=json.dumps(answers),
        )
        log.info(
            "Checkpoint %d resolved: route=%s answers=%s workflow=%d",
            checkpoint["id"],
            route,
            answers,
            workflow_id,
        )

        # Unblock the orchestrator
        checkpoint_event.set()

        return {"status": "ok", "route": route, "config": wf_config}

    @router.get("/logs", response_class=HTMLResponse)
    async def logs_page(request: Request) -> HTMLResponse:
        """Render the live log stream page."""
        return templates.TemplateResponse(request, "logs.html", {})

    @router.get("/pipeline/{workflow_id}", response_class=HTMLResponse)
    async def pipeline_page(request: Request, workflow_id: int) -> HTMLResponse:
        """Render the pipeline transparency monitor page."""
        try:
            wf = db.get_workflow(workflow_id)
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"Workflow {workflow_id} not found"
            )
        return templates.TemplateResponse(
            request,
            "pipeline.html",
            {"workflow": wf},
        )

    @router.get("/pipeline/{workflow_id}/fragment", response_class=HTMLResponse)
    async def pipeline_fragment(request: Request, workflow_id: int) -> HTMLResponse:
        """HTMX polling fragment — returns the full pipeline data panel."""
        try:
            wf = db.get_workflow(workflow_id)
        except KeyError:
            return HTMLResponse("<p style='color:#f87171'>Workflow not found</p>")

        phase_data = []
        for phase_num, phase_label in [(1, "phase1"), (2, "phase2"), (3, "phase3")]:
            rows = db.get_tool_outputs(workflow_id, phase_label)
            tools_parsed = []
            for row in rows:
                try:
                    parsed = json.loads(row["output_json"])
                except json.JSONDecodeError:
                    parsed = {"raw": row.get("output_json", "")}
                tools_parsed.append(
                    {
                        "name": row["tool_name"],
                        "output": parsed,
                        "collected_at": row["collected_at"],
                    }
                )
            phase_data.append(
                {"phase": phase_num, "label": phase_label, "tools": tools_parsed}
            )

        llm_decisions = db.conn.execute(
            "SELECT stage, decision, model, decided_at FROM llm_decisions"
            " WHERE workflow_id = ? ORDER BY decided_at",
            (workflow_id,),
        ).fetchall()
        decisions_parsed = []
        for row in llm_decisions:
            try:
                dec = json.loads(row["decision"])
            except json.JSONDecodeError:
                dec = {"raw": row["decision"]}
            decisions_parsed.append(
                {
                    "stage": row["stage"],
                    "model": row["model"],
                    "decided_at": row["decided_at"],
                    "decision": dec,
                }
            )

        jobs = db.conn.execute(
            "SELECT stage, script_path, status, stdout, stderr, plots, queued_at, completed_at"
            " FROM jobs WHERE workflow_id = ? ORDER BY queued_at",
            (workflow_id,),
        ).fetchall()
        jobs_data = [dict(j) for j in jobs]

        return templates.TemplateResponse(
            request,
            "pipeline_fragment.html",
            {
                "workflow": wf,
                "phase_data": phase_data,
                "decisions": decisions_parsed,
                "jobs": jobs_data,
            },
        )

    @router.post("/pipeline/{workflow_id}/stop")
    async def stop_pipeline(workflow_id: int) -> dict:
        """Signal the orchestrator to stop at the next safe boundary."""
        try:
            wf = db.get_workflow(workflow_id)
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"Workflow {workflow_id} not found"
            )

        terminal = {"STOPPED", "IMAGING_PIPELINE", "CALIBRATION_LOOP", "ERROR"}
        if wf["stage"] in terminal:
            raise HTTPException(
                status_code=409,
                detail=f"Workflow is already in terminal state {wf['stage']!r}",
            )

        if stop_event is not None:
            stop_event.set()
            log.info("Stop signal sent for workflow %d", workflow_id)
        return {"status": "stop_requested"}

    @router.get("/logs/stream")
    async def logs_stream() -> StreamingResponse:
        """SSE endpoint — tails /var/log/wildcat.log and pushes new lines."""
        return StreamingResponse(
            _tail_log(_LOG_PATH),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    app.include_router(router)

    # Serve plot PNGs from /data/jobs/
    jobs_dir = Path("/data/jobs")
    if jobs_dir.exists():
        app.mount("/jobs", StaticFiles(directory=str(jobs_dir)), name="jobs")

    return app
