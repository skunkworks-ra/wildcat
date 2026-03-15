"""Per-decision LLM orchestrator for wildcat.

The Orchestrator drives the workflow state machine:
  read state → call MCP tools (or read job metrics) → assemble prompt
  → LLM → parse decision → (optionally) run CASA → transition state → repeat

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

# ── Per-stage JSON instructions ─────────────────────────────────────────────

# Inspection phases (Phase 1-3): full schema with optional checkpoint_questions
_JSON_INSTRUCTION = """
Respond with a single JSON object and no other text. Schema:
{
  "next_stage": "<one of: PHASE2_RUNNING, PHASE3_RUNNING, HUMAN_CHECKPOINT, ERROR>",
  "casa_script": "<python script string, or null if no CASA job is needed>",
  "summary": "<human-readable summary of your findings (2-5 sentences)>",
  "reasoning": "<brief reasoning trace explaining the decision>",
  "checkpoint_questions": [
    {
      "id": "polcal",
      "finding": "<one sentence about polarisation calibration feasibility>",
      "severity": "<info|warning|critical>",
      "question": "<question to ask the human expert about polarisation calibration>",
      "options": ["continue_polcal", "stokes_i_only", "exit"]
    },
    {
      "id": "aggressive_flagging",
      "finding": "<one sentence about flagging issues e.g. bad antennas or RFI>",
      "severity": "<info|warning|critical>",
      "question": "<yes/no question about whether to apply aggressive flagging>",
      "options": ["yes", "no"]
    }
  ]
}
The checkpoint_questions array is ONLY included when next_stage is HUMAN_CHECKPOINT. Omit it otherwise.
Always include both questions (polcal and aggressive_flagging) when emitting checkpoint_questions.
"""

# CALIBRATION_PREFLAG: rflag pass on calibrators, decide if clean enough to solve
_JSON_INSTRUCTION_PREFLAG = """
Respond with a single JSON object and no other text. Schema:
{
  "next_stage": "<CALIBRATION_PREFLAG | CALIBRATION_SOLVE>",
  "casa_script": "<completed CASA Python script — fill ALL {PLACEHOLDER} values>",
  "summary": "<what flag fractions you found and what you decided>",
  "reasoning": "<brief trace>"
}
Choose CALIBRATION_SOLVE when overall calibrator flag fraction < 0.15 and no SPW > 0.30 flagged.
Choose CALIBRATION_PREFLAG for another rflag pass when data is still heavily contaminated.
Maximum 3 CALIBRATION_PREFLAG passes — the orchestrator enforces this cap.
"""

# CALIBRATION_SOLVE: delay+BP+gain sequence, flag BP solutions, report metrics
_JSON_INSTRUCTION_SOLVE = """
Respond with a single JSON object and no other text. Schema:
{
  "next_stage": "<CALIBRATION_PREFLAG | CALIBRATION_APPLY>",
  "casa_script": "<completed CASA Python script — fill ALL {PLACEHOLDER} values>",
  "summary": "<bp_flagged_frac and gain_flagged_frac you found and what you decided>",
  "reasoning": "<brief trace>"
}
Choose CALIBRATION_APPLY when bp_flagged_frac < 0.20 AND gain_flagged_frac < 0.15.
Choose CALIBRATION_PREFLAG when solutions are too heavily flagged and more RFI excision is needed.
"""

# CALIBRATION_APPLY: applycal + rflag corrected + flag target, then checkpoint
_JSON_INSTRUCTION_APPLY = """
Respond with a single JSON object and no other text. Schema:
{
  "next_stage": "CALIBRATION_CHECKPOINT",
  "casa_script": "<completed CASA Python script — fill ALL {PLACEHOLDER} values>",
  "summary": "<what applycal did and post-cal flag fractions on corrected data>",
  "reasoning": "<brief trace>",
  "checkpoint_questions": [
    {
      "id": "bp_quality",
      "finding": "<actual bp_flagged_frac value and what it means for data quality>",
      "severity": "<info|warning|critical>",
      "question": "Bandpass solutions: are these acceptable to proceed to imaging?",
      "options": ["proceed", "loop_back", "exit"]
    },
    {
      "id": "gain_quality",
      "finding": "<actual gain_flagged_frac and n_antennas_lost from WILDCAT_METRICS>",
      "severity": "<info|warning|critical>",
      "question": "Gain solutions: proceed to imaging or re-run calibration?",
      "options": ["proceed", "loop_back", "exit"]
    }
  ]
}
Always set next_stage to CALIBRATION_CHECKPOINT. Always include both checkpoint_questions with real numbers.
"""

_JSON_INSTRUCTIONS_BY_STAGE: dict[Stage, str] = {
    Stage.CALIBRATION_PREFLAG: _JSON_INSTRUCTION_PREFLAG,
    Stage.CALIBRATION_SOLVE:   _JSON_INSTRUCTION_SOLVE,
    Stage.CALIBRATION_APPLY:   _JSON_INSTRUCTION_APPLY,
}

# ── Config mapping ────────────────────────────────────────────────────────────

# Maps (question_id, answer) → (config_key, value) to apply to workflow_config.
# None means no config update for that answer — routing-only.
_QUESTION_CONFIG_MAP: dict[str, dict[str, tuple[str, object] | None]] = {
    "polcal": {
        "continue_polcal": ("polcal", True),
        "stokes_i_only":   ("polcal", False),
        "exit":            None,
        "proceed":         ("polcal", True),   # LLM fallback
        "loop_back":       ("polcal", False),  # LLM fallback
    },
    "aggressive_flagging": {
        "yes":       ("aggressive_flagging", True),
        "proceed":   ("aggressive_flagging", True),    # LLM fallback
        "no":        ("aggressive_flagging", False),
        "loop_back": ("aggressive_flagging", False),   # LLM fallback
    },
    # Calibration checkpoint — routing only, no config updates
    "bp_quality":   {"proceed": None, "loop_back": None, "exit": None},
    "gain_quality":  {"proceed": None, "loop_back": None, "exit": None},
}

# ── CASA script templates ────────────────────────────────────────────────────
# Each template has {PLACEHOLDER} sections the LLM fills from Phase 1-3 outputs.
# Sections marked [DETERMINISTIC] must be copied verbatim.

_TEMPLATE_PREFLAG = """\
# wildcat CALIBRATION_PREFLAG — rflag calibrators, report flag fractions
# LLM: fill all {PLACEHOLDER} values from Phase 1-3 tool outputs

import json
from casatasks import flagdata

vis          = '{VIS}'           # absolute path to the measurement set
cal_fields   = '{CAL_FIELDS}'   # comma-sep field IDs for all calibrators, e.g. '0,1,2'
all_spw      = '{ALL_SPW}'      # e.g. '0~15'
corrstring   = '{CORRSTRING}'   # e.g. 'RR,LL' or 'XX,YY'

# [DETERMINISTIC] rflag on calibrators
flagdata(
    vis=vis, mode='rflag',
    field=cal_fields, spw=all_spw,
    correlation='ABS_' + corrstring,
    ntime='scan', combinescans=False,
    datacolumn='data',
    winsize=3, timedevscale=4.0, freqdevscale=4.0,
    extendflags=False, action='apply', flagbackup=True,
)

# [DETERMINISTIC] Extend flags to full polarisation products
flagdata(
    vis=vis, mode='extend',
    field=cal_fields, spw=all_spw,
    growtime=50.0, growfreq=90.0,
    action='apply', flagbackup=False,
)

# [DETERMINISTIC] Collect and report flag fractions
summary = flagdata(
    vis=vis, mode='summary', spwchan=True,
    action='calculate', savepars=False,
)
total   = summary.get('total', 1)
flagged = summary.get('flagged', 0)
overall_frac = flagged / max(total, 1)
per_spw = {
    int(k): round(v['flagged'] / max(v['total'], 1), 4)
    for k, v in summary.get('spw', {}).items()
}
metrics = {
    'stage':              'CALIBRATION_PREFLAG',
    'overall_flag_frac':  round(overall_frac, 4),
    'per_spw_flag_frac':  per_spw,
    'n_spw_heavy':        sum(1 for f in per_spw.values() if f > 0.30),
}
print('WILDCAT_METRICS: ' + json.dumps(metrics))
"""

_TEMPLATE_SOLVE = """\
# wildcat CALIBRATION_SOLVE — delay+BP+gain, flag BP solutions, report metrics
# LLM: fill all {PLACEHOLDER} values from Phase 1-3 tool outputs

import json
from casatools import ms as mstool
from casatasks import gaincal, bandpass, flagdata, rmtables

vis           = '{VIS}'
flux_field    = '{FLUX_FIELD}'        # field ID(s) for flux calibrator
bp_field      = '{BP_FIELD}'          # field ID(s) for bandpass calibrator
delay_field   = '{DELAY_FIELD}'       # field ID(s) for delay calibrator
phase_field   = '{PHASE_FIELD}'       # field ID(s) for phase calibrator
phase_scans   = '{PHASE_SCAN_IDS}'   # comma-sep scan IDs on phase cal, e.g. '5,10,15'
refant        = '{REFANT}'            # reference antenna name, e.g. 'ea01'
all_spw       = '{ALL_SPW}'
corrstring    = '{CORRSTRING}'
flux_standard = '{FLUX_STANDARD}'    # e.g. 'Perley-Butler 2017'
minblperant   = {MINBLPERANT}         # int, e.g. 4
int_time_s    = {INT_TIME_S}          # integration time in seconds, e.g. 2.02

workdir = '/data/jobs/{WORKFLOW_ID}'

# [DETERMINISTIC] Set flux density scale
from casatasks import setjy
setjy(vis=vis, field=flux_field, scalebychan=True, standard=flux_standard)

# [DETERMINISTIC] Compute solution intervals from phase-cal scan durations
_DAY_S = 86400.0
ms_tool = mstool()
ms_tool.open(vis)
scan_summary = ms_tool.getscansummary()
ms_tool.close()

phase_scan_ids = [int(s.strip()) for s in phase_scans.split(',') if s.strip()]
durations = []
for sid in phase_scan_ids:
    sd = scan_summary.get(str(sid), {})
    for sub in sd.values():
        t_end = sub.get('EndTime', 0.0)
        t_beg = sub.get('BeginTime', 0.0)
        if t_end > t_beg:
            durations.append(_DAY_S * (t_end - t_beg))
            break

gain_solint1 = f'{int_time_s:.2f}s'
gain_solint2 = f'{(max(durations) * 1.01 if durations else 30.0):.2f}s'
print(f'Computed solints: gain_solint1={gain_solint1}  gain_solint2={gain_solint2}')

# [DETERMINISTIC] Table paths
t_delay    = workdir + '/delay.cal'
t_initph   = workdir + '/initphase.cal'
t_bp       = workdir + '/bandpass.cal'
t_gain     = workdir + '/gain.cal'
for t in [t_delay, t_initph, t_bp, t_gain]:
    rmtables(t)

# [DETERMINISTIC] Delay calibration
gaincal(
    vis=vis, caltable=t_delay,
    field=delay_field, spw=all_spw,
    gaintype='K', solint='inf', combine='scan',
    refant=refant, minblperant=minblperant, minsnr=3.0,
)

# [DETERMINISTIC] Initial per-integration phase (to de-smear bandpass)
gaincal(
    vis=vis, caltable=t_initph,
    field=bp_field, spw=all_spw,
    gaintype='G', calmode='p', solint=gain_solint1,
    refant=refant, minblperant=minblperant, minsnr=3.0,
    gaintable=[t_delay], gainfield=[''], interp=[''],
)

# [DETERMINISTIC] Bandpass (solint=inf, all data combined)
bandpass(
    vis=vis, caltable=t_bp,
    field=bp_field, spw=all_spw,
    solint='inf', combine='scan',
    refant=refant, minblperant=minblperant, minsnr=3.0,
    bandtype='B', fillgaps=62,
    gaintable=[t_delay, t_initph], gainfield=['', ''], interp=['', ''],
)

# [DETERMINISTIC] Flag bad bandpass solutions with rflag
bp_flag_before = flagdata(caltable=t_bp, mode='summary', action='calculate')
flagdata(
    vis=t_bp, mode='rflag',
    winsize=3, timedevscale=4.0, freqdevscale=4.0,
    action='apply', flagbackup=False,
)
bp_flag_after = flagdata(caltable=t_bp, mode='summary', action='calculate')
bp_flagged_frac = bp_flag_after.get('flagged', 0) / max(bp_flag_after.get('total', 1), 1)

# [DETERMINISTIC] Final amplitude+phase gain solutions
gaincal(
    vis=vis, caltable=t_gain,
    field=phase_field, spw=all_spw,
    gaintype='G', calmode='ap', solint=gain_solint2,
    refant=refant, minblperant=minblperant, minsnr=3.0,
    gaintable=[t_delay, t_bp], gainfield=['', ''], interp=['', 'linear'],
)
gain_flag_after = flagdata(caltable=t_gain, mode='summary', action='calculate')
gain_flagged_frac = gain_flag_after.get('flagged', 0) / max(gain_flag_after.get('total', 1), 1)

# Count antennas that are completely flagged in the gain table
from casatools import table as tbtool
tb = tbtool()
tb.open(t_gain)
ant_col = tb.getcol('ANTENNA1')
flag_col = tb.getcol('FLAG')
tb.close()
import numpy as np
unique_ants = set(ant_col.tolist())
lost_ants = sum(1 for a in unique_ants if flag_col[ant_col == a].all())

# [DETERMINISTIC] Emit metrics
metrics = {
    'stage':              'CALIBRATION_SOLVE',
    'gain_solint1':       gain_solint1,
    'gain_solint2':       gain_solint2,
    'bp_flagged_frac':    round(bp_flagged_frac, 4),
    'gain_flagged_frac':  round(gain_flagged_frac, 4),
    'n_antennas_lost':    int(lost_ants),
    't_delay':            t_delay,
    't_bp':               t_bp,
    't_gain':             t_gain,
}
print('WILDCAT_METRICS: ' + json.dumps(metrics))
"""

_TEMPLATE_APPLY = """\
# wildcat CALIBRATION_APPLY — applycal + rflag corrected + flag target
# LLM: fill all {PLACEHOLDER} values

import json
from casatasks import applycal, flagdata

vis          = '{VIS}'
target_field = '{TARGET_FIELD}'    # field ID(s) for science target(s)
cal_field    = '{CAL_FIELDS}'      # all calibrator field IDs
all_spw      = '{ALL_SPW}'
corrstring   = '{CORRSTRING}'

workdir  = '/data/jobs/{WORKFLOW_ID}'
t_delay  = workdir + '/delay.cal'
t_bp     = workdir + '/bandpass.cal'
t_gain   = workdir + '/gain.cal'

# [DETERMINISTIC] Apply calibration to all fields
applycal(
    vis=vis, field='',
    spw=all_spw,
    gaintable=[t_delay, t_bp, t_gain],
    gainfield=['', '', ''],
    interp=['', 'linear', 'linear'],
    calwt=[False, False, False],
    flagbackup=True,
)

# [DETERMINISTIC] rflag on CORRECTED column for calibrators
flagdata(
    vis=vis, mode='rflag',
    field=cal_field, spw=all_spw,
    correlation='ABS_' + corrstring,
    ntime='scan', combinescans=False,
    datacolumn='corrected',
    winsize=3, timedevscale=4.0, freqdevscale=4.0,
    extendflags=False, action='apply', flagbackup=False,
)

# [DETERMINISTIC] rflag on target corrected column
flagdata(
    vis=vis, mode='rflag',
    field=target_field, spw=all_spw,
    correlation='ABS_' + corrstring,
    ntime='scan', combinescans=False,
    datacolumn='corrected',
    winsize=3, timedevscale=5.0, freqdevscale=5.0,
    extendflags=False, action='apply', flagbackup=False,
)

# [DETERMINISTIC] Final flag summary on corrected data
summary = flagdata(
    vis=vis, mode='summary', spwchan=True,
    action='calculate', savepars=False,
)
total   = summary.get('total', 1)
flagged = summary.get('flagged', 0)
post_cal_flag_frac = flagged / max(total, 1)
per_spw = {
    int(k): round(v['flagged'] / max(v['total'], 1), 4)
    for k, v in summary.get('spw', {}).items()
}

metrics = {
    'stage':                'CALIBRATION_APPLY',
    'post_cal_flag_frac':   round(post_cal_flag_frac, 4),
    'per_spw_flag_frac':    per_spw,
    'n_spw_heavy':          sum(1 for f in per_spw.values() if f > 0.30),
}
print('WILDCAT_METRICS: ' + json.dumps(metrics))
"""

_TEMPLATES_BY_STAGE: dict[Stage, str] = {
    Stage.CALIBRATION_PREFLAG: _TEMPLATE_PREFLAG,
    Stage.CALIBRATION_SOLVE:   _TEMPLATE_SOLVE,
    Stage.CALIBRATION_APPLY:   _TEMPLATE_APPLY,
}

# Valid next_stage values per calibration stage
_VALID_CAL_NEXT_STAGES: dict[Stage, set[str]] = {
    Stage.CALIBRATION_PREFLAG: {Stage.CALIBRATION_PREFLAG.value, Stage.CALIBRATION_SOLVE.value},
    Stage.CALIBRATION_SOLVE:   {Stage.CALIBRATION_PREFLAG.value, Stage.CALIBRATION_APPLY.value},
    Stage.CALIBRATION_APPLY:   {Stage.CALIBRATION_CHECKPOINT.value},
}

# Maximum preflag iterations before escalating to CALIBRATION_CHECKPOINT
_PREFLAG_MAX_ITERATIONS = 3


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
        """Main loop. Reads stage, dispatches to correct handler, exits on terminal/waiting state."""
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

            elif stage in (Stage.CALIBRATION_PREFLAG, Stage.CALIBRATION_SOLVE, Stage.CALIBRATION_APPLY):
                await self._handle_calibration_stage(workflow_id, stage)

            elif stage == Stage.CALIBRATION_CHECKPOINT:
                await self._handle_checkpoint(workflow_id)

            elif stage == Stage.CALIBRATION_LOOP:
                # Reset preflag iteration counter and restart calibration
                wf_config = self.db.get_workflow_config(workflow_id)
                wf_config["_preflag_iterations"] = 0
                wf_config.pop("_preflag_flag_warning", None)
                self.db.set_workflow_config(workflow_id, wf_config)
                self.db.transition(workflow_id, Stage.CALIBRATION_PREFLAG)

            elif stage == Stage.IMAGING_PIPELINE:
                log.info("Workflow %d reached IMAGING_PIPELINE — handoff complete", workflow_id)
                break

            elif stage == Stage.STOPPED:
                log.info("Workflow %d stopped by user request", workflow_id)
                break

            elif stage == Stage.ERROR:
                log.error("Workflow %d is in ERROR state — manual intervention required", workflow_id)
                break

            else:
                # PHASE*_COMPLETE — advance to next running state
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

    # ── Inspection phase handler ───────────────────────────────────────────

    async def _handle_phase(self, workflow_id: int, phase: int) -> None:
        """Run MCP tools → LLM decision → optional CASA job → transition."""
        wf = self.db.get_workflow(workflow_id)
        ms_path: str = wf["ms_path"]
        stage = Stage(wf["stage"])

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
            self.db.save_tool_output(workflow_id, phase_label, tool_name, json.dumps(output))

        # 3. Assemble prompt
        system_prompt = load_system_prompt(self.skills_path, stage)
        user_content = self._format_tool_outputs(tool_outputs, phase)
        if phase > 1:
            wf_config = self.db.get_workflow_config(workflow_id)
            user_content += (
                "\n\n## Human decisions (workflow_config)\n"
                "```json\n" + json.dumps(wf_config, indent=2) + "\n```\n"
                "Adjust tool selection and recommendations to reflect these decisions.\n"
            )
        messages = [
            {"role": "system", "content": system_prompt + "\n\n" + _JSON_INSTRUCTION},
            {"role": "user",   "content": user_content},
        ]

        if self.stop_event.is_set():
            log.info("Stop requested — halting before LLM call in Phase %d", phase)
            self.db.transition(workflow_id, Stage.STOPPED)
            return

        response = await self.llm.complete(messages)
        raw_decision = self._extract_content(response)
        decision = self._parse_decision(raw_decision)

        prompt_hash = hashlib.sha256(user_content.encode()).hexdigest()[:16]
        model = response.get("model", "unknown")
        self.db.save_llm_decision(workflow_id, stage.value, json.dumps(decision), model, prompt_hash)

        if decision.get("casa_script"):
            if self.stop_event.is_set():
                log.info("Stop requested — skipping CASA job in Phase %d", phase)
                self.db.transition(workflow_id, Stage.STOPPED)
                return
            await self._run_casa_job(workflow_id, stage.value, decision["casa_script"])

        next_stage_str = decision["next_stage"]
        try:
            next_stage = Stage(next_stage_str)
        except ValueError:
            log.error("LLM returned unknown next_stage %r — entering ERROR", next_stage_str)
            self.db.transition(workflow_id, Stage.ERROR)
            return

        self.db.transition(workflow_id, next_stage)

    # ── Calibration stage handler ──────────────────────────────────────────

    async def _handle_calibration_stage(self, workflow_id: int, stage: Stage) -> None:
        """Read previous job metrics → LLM fills template → run CASA → transition.

        Shared by CALIBRATION_PREFLAG, CALIBRATION_SOLVE, CALIBRATION_APPLY.
        """
        if self.stop_event.is_set():
            self.db.transition(workflow_id, Stage.STOPPED)
            return

        wf = self.db.get_workflow(workflow_id)
        wf_config = self.db.get_workflow_config(workflow_id)
        preflag_iterations = wf_config.get("_preflag_iterations", 0)

        # Enforce preflag iteration cap
        if stage == Stage.CALIBRATION_PREFLAG and preflag_iterations >= _PREFLAG_MAX_ITERATIONS:
            log.warning(
                "CALIBRATION_PREFLAG hit cap (%d iterations) for workflow %d — escalating",
                _PREFLAG_MAX_ITERATIONS, workflow_id,
            )
            wf_config["_preflag_flag_warning"] = True
            self.db.set_workflow_config(workflow_id, wf_config)
            self.db.transition(workflow_id, Stage.CALIBRATION_CHECKPOINT)
            return

        # 1. Build context: previous CASA job metrics + all Phase 1-3 tool outputs + workflow_config
        prev_metrics = self._read_last_job_metrics(workflow_id)
        phase_outputs = self._load_all_tool_outputs(workflow_id)
        template = _TEMPLATES_BY_STAGE[stage]
        instruction = _JSON_INSTRUCTIONS_BY_STAGE[stage]

        user_content = self._format_calibration_prompt(
            stage, wf["ms_path"], prev_metrics, phase_outputs, wf_config, template, preflag_iterations
        )

        system_prompt = load_system_prompt(self.skills_path, stage)
        messages = [
            {"role": "system", "content": system_prompt + "\n\n" + instruction},
            {"role": "user",   "content": user_content},
        ]

        if self.stop_event.is_set():
            self.db.transition(workflow_id, Stage.STOPPED)
            return

        # 2. LLM call
        response = await self.llm.complete(messages)
        raw_decision = self._extract_content(response)
        decision = self._parse_decision(raw_decision)

        model = response.get("model", "unknown")
        self.db.save_llm_decision(workflow_id, stage.value, json.dumps(decision), model)

        # 3. Validate next_stage is legal for this stage
        next_stage_str = decision["next_stage"]
        valid = _VALID_CAL_NEXT_STAGES.get(stage, set())
        if next_stage_str not in valid:
            log.error(
                "LLM returned illegal next_stage %r for %s (valid: %s) — entering ERROR",
                next_stage_str, stage, valid,
            )
            self.db.transition(workflow_id, Stage.ERROR)
            return

        # 4. Run CASA script
        if decision.get("casa_script"):
            if self.stop_event.is_set():
                self.db.transition(workflow_id, Stage.STOPPED)
                return
            await self._run_casa_job(workflow_id, stage.value, decision["casa_script"])

        # 5. Update preflag iteration counter
        if stage == Stage.CALIBRATION_PREFLAG:
            if next_stage_str == Stage.CALIBRATION_PREFLAG.value:
                wf_config["_preflag_iterations"] = preflag_iterations + 1
            else:
                wf_config["_preflag_iterations"] = 0  # reset on exit
            self.db.set_workflow_config(workflow_id, wf_config)

        # 6. Transition
        self.db.transition(workflow_id, Stage(next_stage_str))

    # ── Checkpoint handler ─────────────────────────────────────────────────

    async def _handle_checkpoint(self, workflow_id: int) -> None:
        """Write checkpoint row, signal UI, await human decision, transition.

        Used for both HUMAN_CHECKPOINT and CALIBRATION_CHECKPOINT — the stage
        value stored in the checkpoint row distinguishes them for the UI.
        """
        wf = self.db.get_workflow(workflow_id)
        current_stage = Stage(wf["stage"])

        last_decision = self.db.conn.execute(
            "SELECT decision FROM llm_decisions WHERE workflow_id = ?"
            " ORDER BY decided_at DESC LIMIT 1",
            (workflow_id,),
        ).fetchone()
        if last_decision:
            llm_summary = json.loads(last_decision["decision"]).get("summary", "No summary available.")
        else:
            llm_summary = "No summary available."

        checkpoint_id = self.db.create_checkpoint(workflow_id, current_stage.value, llm_summary)
        log.info(
            "Checkpoint %d (%s) created for workflow %d — awaiting human decision",
            checkpoint_id, current_stage.value, workflow_id,
        )

        self.checkpoint_event.clear()
        await self.checkpoint_event.wait()

        checkpoint = self.db.get_latest_checkpoint(workflow_id)
        human_route = (checkpoint or {}).get("human_route", Stage.IMAGING_PIPELINE.value)

        route_map = {
            Stage.STOPPED.value:               Stage.STOPPED,
            Stage.CALIBRATION_LOOP.value:      Stage.CALIBRATION_LOOP,
            Stage.CALIBRATION_PREFLAG.value:   Stage.CALIBRATION_PREFLAG,
            Stage.IMAGING_PIPELINE.value:      Stage.IMAGING_PIPELINE,
            # Legacy
            "calibration": Stage.CALIBRATION_LOOP,
            "imaging":     Stage.IMAGING_PIPELINE,
        }
        next_stage = route_map.get(human_route, Stage.IMAGING_PIPELINE)
        self.db.transition(workflow_id, next_stage)

    # ── CASA job helper ────────────────────────────────────────────────────

    async def _run_casa_job(self, workflow_id: int, stage_label: str, script_content: str) -> None:
        """Write script to disk, submit to runner, await sentinel."""
        script_dir = Path(self.runner.config.jobs_dir) / str(workflow_id)
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / f"{stage_label}.py"
        script_path.write_text(script_content, encoding="utf-8")

        job_id = self.db.create_job(workflow_id, stage_label, str(script_path))

        self.checkpoint_event.clear()
        asyncio.create_task(self.runner.submit(job_id, str(script_path)))
        await self.checkpoint_event.wait()

    # ── Calibration prompt helpers ─────────────────────────────────────────

    def _read_last_job_metrics(self, workflow_id: int) -> dict:
        """Parse WILDCAT_METRICS from the most recent completed job's stdout."""
        row = self.db.conn.execute(
            "SELECT stdout FROM jobs WHERE workflow_id = ? AND status = 'done'"
            " ORDER BY completed_at DESC LIMIT 1",
            (workflow_id,),
        ).fetchone()
        if not row or not row["stdout"]:
            return {}
        for line in row["stdout"].splitlines():
            if line.startswith("WILDCAT_METRICS:"):
                try:
                    return json.loads(line[len("WILDCAT_METRICS:"):].strip())
                except json.JSONDecodeError:
                    pass
        return {}

    def _load_all_tool_outputs(self, workflow_id: int) -> dict[str, dict]:
        """Load all Phase 1-3 tool outputs keyed by tool name."""
        outputs: dict[str, dict] = {}
        for phase in ("phase1", "phase2", "phase3"):
            for row in self.db.get_tool_outputs(workflow_id, phase):
                try:
                    outputs[row["tool_name"]] = json.loads(row["output_json"])
                except (json.JSONDecodeError, KeyError):
                    pass
        return outputs

    def _format_calibration_prompt(
        self,
        stage: Stage,
        ms_path: str,
        prev_metrics: dict,
        tool_outputs: dict[str, dict],
        wf_config: dict,
        template: str,
        preflag_iterations: int,
    ) -> str:
        """Build the user_content for a calibration stage LLM call."""
        lines: list[str] = []

        lines.append(f"## Calibration stage: {stage.value}")
        lines.append(f"Measurement set: {ms_path}")
        lines.append(f"Workflow config: {json.dumps(wf_config)}")
        if preflag_iterations:
            lines.append(f"Preflag iterations so far: {preflag_iterations} / {_PREFLAG_MAX_ITERATIONS}")

        if prev_metrics:
            lines.append("\n## Previous job metrics (WILDCAT_METRICS)")
            lines.append("```json")
            lines.append(json.dumps(prev_metrics, indent=2))
            lines.append("```")

        lines.append("\n## Phase 1-3 tool outputs (use to fill template placeholders)")
        for tool_name, data in tool_outputs.items():
            lines.append(f"### {tool_name}")
            lines.append("```json")
            lines.append(json.dumps(data, indent=2))
            lines.append("```")

        lines.append("\n## Script template")
        lines.append("Complete the script by replacing ALL {PLACEHOLDER} values.")
        lines.append("Do NOT modify sections marked [DETERMINISTIC].")
        lines.append("```python")
        lines.append(template)
        lines.append("```")

        return "\n".join(lines)

    # ── Parsing helpers ────────────────────────────────────────────────────

    def _parse_decision(self, raw: str) -> dict:
        """Extract and validate the LLM JSON decision."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(l for l in lines if not l.strip().startswith("```"))

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
        """Format MCP tool outputs into a structured prompt string."""
        lines = [f"## Phase {phase} tool outputs\n"]
        for tool_name, data in outputs.items():
            lines.append(f"### {tool_name}")
            lines.append("```json")
            lines.append(json.dumps(data, indent=2))
            lines.append("```\n")
        return "\n".join(lines)
