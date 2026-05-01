"""Per-decision LLM orchestrator for wildcat.

The Orchestrator drives the workflow state machine:
  read state → call MCP tools (or read job metrics) → assemble prompt
  → LLM → parse decision → (optionally) run CASA → transition state → repeat

The LLM is invoked once per decision point (stateless between calls).
Long CASA jobs don't cause timeouts because the orchestrator awaits
asyncio subprocesses directly — the event loop stays idle while CASA runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
from pathlib import Path

from wildcat.llm import LLMBackend
from wildcat.runner import CASARunner
from wildcat.skills import load_system_prompt
from wildcat.state import Stage, StateDB
from wildcat.tools import MSInspectClient

log = logging.getLogger(__name__)

# JSON schema that every LLM response must conform to
_DECISION_SCHEMA_KEYS = {
    "next_stage",
    "summary",
    "reasoning",
}  # casa_script is optional

# ── Per-stage JSON instructions ─────────────────────────────────────────────

# Inspection phases 1 and 2: advance to next phase or escalate on critical failure
_JSON_INSTRUCTION = """
Respond with a single JSON object and no other text. Schema:
{
  "next_stage": "<one of: PHASE2_RUNNING, PHASE3_RUNNING, HUMAN_CHECKPOINT, ERROR>",
  "casa_script": "<python script string, or null if no CASA job is needed>",
  "summary": "<human-readable summary of your findings (2-5 sentences)>",
  "reasoning": "<brief reasoning trace explaining the decision>"
}
Only choose HUMAN_CHECKPOINT or ERROR for unrecoverable critical failures.
Normal observations should advance to the next phase.
"""

# Phase 3: autonomous decision — always proceed to calibration unless data is fundamentally broken
_JSON_INSTRUCTION_PHASE3 = """
Respond with a single JSON object and no other text. Schema:
{
  "next_stage": "<CALIBRATION_PREFLAG | ERROR>",
  "summary": "<human-readable summary of your findings (2-5 sentences)>",
  "reasoning": "<brief reasoning trace explaining the decision>"
}

Always choose CALIBRATION_PREFLAG.
Only choose ERROR for fundamental data corruption (all calibrators missing or entirely flagged,
no usable data whatsoever).

Do NOT include polcal or aggressive_flagging decisions — these are set automatically by the
orchestrator from ms_pol_cal_feasibility (verdict) and ms_antenna_flag_fraction (per-antenna
flag fractions). Do NOT include checkpoint_questions or config_updates.
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

# CALIBRATION_APPLY: metrics summary + autonomous proceed decision.
# The CASA script is filled deterministically — LLM interprets results and decides whether to
# surface a human checkpoint or proceed directly to imaging.
_JSON_INSTRUCTION_APPLY = """
The CASA calibration script has already run. You are given the WILDCAT_METRICS output.
Respond with a single JSON object and no other text. Schema:
{
  "next_stage": "<IMAGING_PIPELINE if auto_proceed else CALIBRATION_CHECKPOINT>",
  "auto_proceed": <true if all fractions < 0.20 and n_antennas_lost <= 1, else false>,
  "summary": "<2-5 sentences with actual metric values from WILDCAT_METRICS>",
  "reasoning": "<brief trace explaining auto_proceed decision and severity>",
  "checkpoint_questions": [
    {
      "id": "calibration_done",
      "finding": "<actual values + caltable paths from t_bp/t_gain/t_delay for operator inspection>",
      "severity": "<warning|critical>",
      "question": "Calibration complete. Proceed to imaging or loop back for another calibration pass?",
      "options": ["proceed", "loop_back", "exit"],
      "recommendation": "<proceed|loop_back>",
      "timeout_seconds": <300 for warning, 600 for critical>,
      "timeout_default": "proceed"
    }
  ]
}
If auto_proceed is true, set checkpoint_questions to [].
Severity: warning if any fraction 0.20-0.40 or n_antennas_lost 2-3; critical if any > 0.40 or n_antennas_lost > 3.
recommendation: loop_back only if post_cal_flag_frac > 0.40 or n_antennas_lost > 3; otherwise proceed.
"""

# CALIBRATION_SOLVE: CASA ran deterministically; LLM inspects caltable quality and routes.
# Valid next_stage: CALIBRATION_APPLY | CALIBRATION_PREFLAG | CALIBRATION_SOLVE (retry).
_JSON_INSTRUCTION_SOLVE = """
The CASA calibration script has already run. You are given:
  - WILDCAT_METRICS: flag fractions and caltable paths from the solve job
  - CALSOL_STATS: ms_calsol_stats output for delay.cal, bandpass.cal, gain.cal

Respond with a single JSON object and no other text. Schema:
{
  "next_stage": "<CALIBRATION_APPLY | CALIBRATION_PREFLAG | CALIBRATION_SOLVE>",
  "casa_script": "<modified solve script if next_stage == CALIBRATION_SOLVE, else null>",
  "summary": "<2-4 sentences with actual metric values and quality assessment>",
  "reasoning": "<brief trace explaining the routing decision>"
}

Rules:
- CALIBRATION_APPLY: bp and gain flag fractions within band thresholds AND no hard quality failures.
- CALIBRATION_PREFLAG: flag fractions exceed band thresholds — more RFI excision needed.
- CALIBRATION_SOLVE (retry): flag fractions within threshold BUT quality failure detected
  (phase_rms_deg > 30° on bandpass → re-run delay step; SNR < 3 → try alternate refant or combine=scan).
  Provide a modified casa_script. Maximum 2 CALIBRATION_SOLVE retries — orchestrator enforces the cap.
- If PREFLAG cap is already reached and thresholds are not met, choose CALIBRATION_APPLY anyway.
"""

# POLCAL_SOLVE: full solve sequence K + G + B + setjy(polcal) + Kcross + Df + Xf
# Replaces CALIBRATION_SOLVE when polcal=True. CALIBRATION_APPLY follows.
_JSON_INSTRUCTION_POLCAL = """
Respond with a single JSON object and no other text. Schema:
{
  "next_stage": "CALIBRATION_APPLY",
  "casa_script": "<completed CASA Python script — fill ALL {PLACEHOLDER} values>",
  "summary": "<solve steps run, calibrator, bp_flagged_frac, gain_flagged_frac, polindex_c0, polangle_c0_rad>",
  "reasoning": "<brief trace>"
}
Always set next_stage to CALIBRATION_APPLY.
Fill ALL {PLACEHOLDER} values from Phase 1-3 tool outputs.
Choose POLTYPE_DTERM using guidance from wildcat/06-polcal.md (Df vs Df+QU table).
POL_FREQ_LO and POL_FREQ_HI should bracket the usable PA nodes — for 3C48 at S-band use 2.0 and 9.0.
"""

_JSON_INSTRUCTIONS_BY_STAGE: dict[Stage, str] = {
    Stage.CALIBRATION_PREFLAG: _JSON_INSTRUCTION_PREFLAG,
    Stage.CALIBRATION_SOLVE: _JSON_INSTRUCTION_SOLVE,
    Stage.POLCAL_SOLVE: _JSON_INSTRUCTION_POLCAL,
}

# ── Config mapping ────────────────────────────────────────────────────────────

# Maps (question_id, answer) → (config_key, value) to apply to workflow_config.
# None means no config update for that answer — routing-only.
_QUESTION_CONFIG_MAP: dict[str, dict[str, tuple[str, object] | None]] = {
    "polcal": {
        "continue_polcal": ("polcal", True),
        "stokes_i_only": ("polcal", False),
        "exit": None,
        "proceed": ("polcal", True),  # LLM fallback
        "loop_back": ("polcal", False),  # LLM fallback
    },
    "aggressive_flagging": {
        "yes": ("aggressive_flagging", True),
        "proceed": ("aggressive_flagging", True),  # LLM fallback
        "no": ("aggressive_flagging", False),
        "loop_back": ("aggressive_flagging", False),  # LLM fallback
    },
    # Calibration checkpoint — routing only, no config updates
    "calibration_done": {"proceed": None, "loop_back": None, "exit": None},
}

# ── CASA script templates ────────────────────────────────────────────────────
# Each template has {PLACEHOLDER} sections the LLM fills from Phase 1-3 outputs.
# Sections marked [DETERMINISTIC] must be copied verbatim.

_TEMPLATE_PREFLAG = """\
# wildcat CALIBRATION_PREFLAG — split calibrators, tfcrop, report flag fractions
# All values filled deterministically by the orchestrator from Phase 1-3 tool outputs.

import json, os
from casatasks import split, flagdata

original_ms  = '{VIS}'          # original read-only measurement set
cal_fields   = '{CAL_FIELDS}'   # comma-sep field IDs for all calibrators, e.g. '0,1,2'
all_spw      = '{ALL_SPW}'      # e.g. '0~15'
corrstring   = '{CORRSTRING}'   # e.g. 'RR,LL' or 'XX,YY'

workdir = '/data/jobs/{WORKFLOW_ID}'

# [DETERMINISTIC] Split calibrators to writable working copy (idempotent)
vis = workdir + '/calibrators.ms'
if not os.path.exists(vis):
    split(vis=original_ms, outputvis=vis, field=cal_fields, datacolumn='data', keepflags=True)

# [DETERMINISTIC] tfcrop + extend in a single list-mode pass (one MS read)
# tfcrop is appropriate here — raw data lacks Gaussian statistics needed by rflag
flag_cmds = [
    "mode='tfcrop' field='{CAL_FIELDS}' spw='{ALL_SPW}' correlation='ABS_{CORRSTRING}' ntime='scan' combinescans=False datacolumn='data' timecutoff=4.0 freqcutoff=4.0 timefit='line' freqfit='poly' maxnpieces=7 flagdimension='freqtime' extendflags=False",
    "mode='extend' field='{CAL_FIELDS}' spw='{ALL_SPW}' growtime=50.0 growfreq=90.0",
]
flagdata(
    vis=vis, mode='list',
    inpfile=flag_cmds,
    action='apply', flagbackup=False,
)

# [DETERMINISTIC] Discard SPWs above threshold in one pass
_spw_summary = flagdata(vis=vis, mode='summary', spwchan=True, action='calculate', savepars=False)
_bad_spws = ','.join(
    spw for spw, info in _spw_summary.get('spw', {}).items()
    if info['flagged'] / max(info['total'], 1) > {SPW_DISCARD_THRESHOLD}
)
if _bad_spws:
    flagdata(vis=vis, mode='manual', spw=_bad_spws, action='apply', flagbackup=False)

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
# All values filled deterministically by the orchestrator from Phase 1-3 tool outputs.

import json
from casatools import ms as mstool
from casatasks import gaincal, bandpass, flagdata, rmtables

workdir       = '/data/jobs/{WORKFLOW_ID}'
vis           = workdir + '/calibrators.ms'  # [DETERMINISTIC] split from CALIBRATION_PREFLAG
flux_field    = '{FLUX_FIELD}'        # field ID(s) for flux calibrator
bp_field      = '{BP_FIELD}'          # field ID(s) for bandpass calibrator
delay_field   = '{DELAY_FIELD}'       # field ID(s) for delay calibrator
phase_field   = '{PHASE_FIELD}'       # field ID(s) for phase calibrator
phase_scans   = '{PHASE_SCAN_IDS}'   # comma-sep scan IDs on phase cal
refant        = '{REFANT}'            # reference antenna (lowest flag fraction)
all_spw       = '{ALL_SPW}'
flux_standard = '{FLUX_STANDARD}'    # CASA-accepted standard string
minblperant   = {MINBLPERANT}
int_time_s    = {INT_TIME_S}          # integration time in seconds

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
# datacolumn='CPARAM' required: cal tables have no DATA column
flagdata(
    vis=t_bp, mode='rflag',
    datacolumn='CPARAM',
    winsize=3, timedevscale=4.0, freqdevscale=4.0,
    action='apply', flagbackup=False,
)
bp_flag_after = flagdata(vis=t_bp, mode='summary', action='calculate')
bp_flagged_frac = bp_flag_after.get('flagged', 0) / max(bp_flag_after.get('total', 1), 1)

# [DETERMINISTIC] Final amplitude+phase gain solutions
gaincal(
    vis=vis, caltable=t_gain,
    field=phase_field, spw=all_spw,
    gaintype='G', calmode='ap', solint=gain_solint2,
    refant=refant, minblperant=minblperant, minsnr=3.0,
    gaintable=[t_delay, t_bp], gainfield=['', ''], interp=['', 'linear'],
)
gain_flag_after = flagdata(vis=t_gain, mode='summary', action='calculate')
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
lost_ants = sum(1 for a in unique_ants if flag_col[..., ant_col == a].all())

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
# wildcat CALIBRATION_APPLY — single applycal pass (includes polcal tables if present)
# All values filled deterministically by the orchestrator.

import json
import os
from casatasks import applycal, flagdata

workdir   = '/data/jobs/{WORKFLOW_ID}'
vis       = workdir + '/calibrators.ms'  # [DETERMINISTIC] split calibrators only
cal_field = '{CAL_FIELDS}'               # all calibrator field IDs
all_spw   = '{ALL_SPW}'
corrstring = '{CORRSTRING}'              # parallel hands only (e.g. 'XX,YY' or 'RR,LL')
t_delay  = workdir + '/delay.cal'
t_bp     = workdir + '/bandpass.cal'
t_gain   = workdir + '/gain.cal'
t_kcross = workdir + '/kcross.K'
t_dterm  = workdir + '/dterms.D'
t_xfeed  = workdir + '/polangle.X'

# [DETERMINISTIC] Build gaintable list — include polcal tables only if they exist
_base_tables = [t_delay, t_bp, t_gain]
_polcal_tables = [t for t in [t_kcross, t_dterm, t_xfeed] if os.path.exists(t)]
all_gaintable = _base_tables + _polcal_tables
all_gainfield = [''] * len(all_gaintable)
all_interp    = ['', 'linear', 'linear'] + [''] * len(_polcal_tables)
use_parang    = len(_polcal_tables) > 0

# [DETERMINISTIC] Apply calibration to all fields
applycal(
    vis=vis, field='',
    spw=all_spw,
    gaintable=all_gaintable,
    gainfield=all_gainfield,
    interp=all_interp,
    calwt=[False] * len(all_gaintable),
    parang=use_parang,
    flagbackup=True,
)

# [DETERMINISTIC] rflag + extend on CORRECTED column in a single list-mode pass
flag_cmds = [
    "mode='rflag' field='{CAL_FIELDS}' spw='{ALL_SPW}' correlation='ABS_{CORRSTRING}' ntime='scan' combinescans=False datacolumn='corrected' winsize=3 timedevscale=4.0 freqdevscale=4.0 extendflags=False",
    "mode='extend' field='{CAL_FIELDS}' spw='{ALL_SPW}' growtime=50.0 growfreq=90.0",
]
flagdata(
    vis=vis, mode='list',
    inpfile=flag_cmds,
    action='apply', flagbackup=False,
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

_TEMPLATE_POLCAL = """\
# wildcat POLCAL_SOLVE — full solve: K + G + B + setjy(polcal) + Kcross + Df + Xf
# Replaces CALIBRATION_SOLVE when polcal=True. CALIBRATION_APPLY follows.
# LLM: fill all {PLACEHOLDER} values from Phase 1-3 tool outputs.
# See wildcat/05-solve.md for K/G/B guidance, wildcat/06-polcal.md for Df vs Df+QU.

import json
from casatools import ms as mstool
from casatasks import setjy, gaincal, bandpass, polcal, flagdata, rmtables
from ms_inspect.util.polcal_setjy_fit import fit_from_catalogue

vis               = '{VIS}'
flux_field        = '{FLUX_FIELD}'          # field ID(s) for flux calibrator
bp_field          = '{BP_FIELD}'            # field ID(s) for bandpass calibrator
delay_field       = '{DELAY_FIELD}'         # field ID(s) for delay calibrator
phase_field       = '{PHASE_FIELD}'         # field ID(s) for phase calibrator
phase_scans       = '{PHASE_SCAN_IDS}'      # comma-sep scan IDs on phase cal
angle_cal_field   = '{ANGLE_CAL_FIELD}'     # field name for polcal angle calibrator (e.g. '3C48')
leakage_cal_field = '{LEAKAGE_CAL_FIELD}'   # usually same as angle_cal_field
refant            = '{REFANT}'
all_spw           = '{ALL_SPW}'
flux_standard     = '{FLUX_STANDARD}'
minblperant       = {MINBLPERANT}
int_time_s        = {INT_TIME_S}

calibrator_name   = '{CALIBRATOR_NAME}'  # catalogue b1950 name (from ms_pol_cal_feasibility)
reffreq_ghz       = {REFFREQ_GHZ}        # float — band_centre_ghz from ms_pol_cal_feasibility
pol_freq_lo       = {POL_FREQ_LO}        # float — lower GHz bound for pol fit (e.g. 2.0 for S-band)
pol_freq_hi       = {POL_FREQ_HI}        # float — upper GHz bound for pol fit (e.g. 9.0 for S-band)
poltype_dterm     = '{POLTYPE_DTERM}'    # 'Df' or 'Df+QU' — see wildcat/06-polcal.md

workdir  = '/data/jobs/{WORKFLOW_ID}'
t_delay  = workdir + '/delay.cal'
t_initph = workdir + '/initphase.cal'
t_bp     = workdir + '/bandpass.cal'
t_gain   = workdir + '/gain.cal'
t_kcross = workdir + '/kcross.K'
t_dterm  = workdir + '/dterms.D'
t_xfeed  = workdir + '/polangle.X'
for t in [t_delay, t_initph, t_bp, t_gain, t_kcross, t_dterm, t_xfeed]:
    rmtables(t)

# [DETERMINISTIC] Set flux density scale (Stokes I only — polcal model set below)
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

# [DETERMINISTIC] Delay calibration
gaincal(
    vis=vis, caltable=t_delay,
    field=delay_field, spw=all_spw,
    gaintype='K', solint='inf', combine='scan',
    refant=refant, minblperant=minblperant, minsnr=3.0,
)

# [DETERMINISTIC] Initial per-integration phase (de-smear bandpass)
gaincal(
    vis=vis, caltable=t_initph,
    field=bp_field, spw=all_spw,
    gaintype='G', calmode='p', solint=gain_solint1,
    refant=refant, minblperant=minblperant, minsnr=3.0,
    gaintable=[t_delay], gainfield=[''], interp=[''],
)

# [DETERMINISTIC] Bandpass
bandpass(
    vis=vis, caltable=t_bp,
    field=bp_field, spw=all_spw,
    solint='inf', combine='scan',
    refant=refant, minblperant=minblperant, minsnr=3.0,
    bandtype='B', fillgaps=62,
    gaintable=[t_delay, t_initph], gainfield=['', ''], interp=['', ''],
)

# [DETERMINISTIC] Flag bad bandpass solutions
bp_flag_before = flagdata(caltable=t_bp, mode='summary', action='calculate')
flagdata(vis=t_bp, mode='rflag', datacolumn='CPARAM',
         winsize=3, timedevscale=4.0, freqdevscale=4.0,
         action='apply', flagbackup=False)
bp_flag_after = flagdata(caltable=t_bp, mode='summary', action='calculate')
bp_flagged_frac = bp_flag_after.get('flagged', 0) / max(bp_flag_after.get('total', 1), 1)

# [DETERMINISTIC] Amplitude+phase gain solutions
gaincal(
    vis=vis, caltable=t_gain,
    field=phase_field, spw=all_spw,
    gaintype='G', calmode='ap', solint=gain_solint2,
    refant=refant, minblperant=minblperant, minsnr=3.0,
    gaintable=[t_delay, t_bp], gainfield=['', ''], interp=['', 'linear'],
)
gain_flag_after = flagdata(vis=t_gain, mode='summary', action='calculate')
gain_flagged_frac = gain_flag_after.get('flagged', 0) / max(gain_flag_after.get('total', 1), 1)

# [DETERMINISTIC] Set full polarisation calibrator model (Stokes I + polindex + polangle)
pol_freq_range = (pol_freq_lo, pol_freq_hi)
params = fit_from_catalogue(
    calibrator_name,
    reffreq_ghz=reffreq_ghz,
    pol_freq_range_ghz=pol_freq_range,
)
setjy(
    vis=vis,
    field=angle_cal_field,
    standard='manual',
    fluxdensity=[params.flux_jy, 0, 0, 0],
    spix=params.spix,
    reffreq=f'{reffreq_ghz}GHz',
    polindex=params.polindex,
    polangle=params.polangle,
    scalebychan=True,
    usescratch=True,
)

# [DETERMINISTIC] Cross-hand delay (multiband combine for wideband SNR)
gaincal(
    vis=vis, caltable=t_kcross,
    field=angle_cal_field,
    gaintype='KCROSS',
    solint='inf', combine='scan,spw',
    refant=refant,
    smodel=[1, 0, 1, 0],
    gaintable=[t_delay, t_bp, t_gain],
    parang=True,
)

# [DETERMINISTIC] D-term leakage (Df or Df+QU per wildcat/06-polcal.md decision table)
polcal(
    vis=vis, caltable=t_dterm,
    field=leakage_cal_field,
    poltype=poltype_dterm,
    solint='inf', combine='scan',
    refant=refant,
    gaintable=[t_delay, t_bp, t_gain, t_kcross],
    parang=True,
)

# [DETERMINISTIC] Position angle calibration
polcal(
    vis=vis, caltable=t_xfeed,
    field=angle_cal_field,
    poltype='Xf',
    solint='inf', combine='scan',
    refant=refant,
    gaintable=[t_delay, t_bp, t_gain, t_kcross, t_dterm],
    parang=True,
)

# [DETERMINISTIC] Emit metrics
from casatools import table as tbtool
tb = tbtool()
tb.open(t_gain)
ant_col  = tb.getcol('ANTENNA1')
flag_col = tb.getcol('FLAG')
tb.close()
import numpy as np
unique_ants = set(ant_col.tolist())
lost_ants   = sum(1 for a in unique_ants if flag_col[ant_col == a].all())

metrics = {
    'stage':             'POLCAL_SOLVE',
    'gain_solint1':      gain_solint1,
    'gain_solint2':      gain_solint2,
    'bp_flagged_frac':   round(bp_flagged_frac, 4),
    'gain_flagged_frac': round(gain_flagged_frac, 4),
    'n_antennas_lost':   int(lost_ants),
    'calibrator':        calibrator_name,
    'reffreq_ghz':       reffreq_ghz,
    'polindex_c0':       round(params.polindex[0], 6),
    'polangle_c0_rad':   round(params.polangle[0], 6),
    't_delay':           t_delay,
    't_bp':              t_bp,
    't_gain':            t_gain,
    't_kcross':          t_kcross,
    't_dterm':           t_dterm,
    't_xfeed':           t_xfeed,
}
print('WILDCAT_METRICS: ' + json.dumps(metrics))
"""

_TEMPLATE_IMAGING = """\
# wildcat IMAGING_PIPELINE — first-pass continuum imaging
# LLM: fill all {PLACEHOLDER} values from the context provided.
# Do NOT modify sections marked [DETERMINISTIC].

import json
from casatasks import tclean, exportfits

workdir        = '/data/jobs/{WORKFLOW_ID}'
vis            = '{VIS}'                  # full MS path (CORRECTED_DATA column)
target_field   = '{TARGET_FIELD}'        # CASA field selection string for science target(s)
image_name     = workdir + '/target.image'
imsize         = {IMSIZE}                # [npix, npix] — derive from primary beam / resolution
cell           = '{CELL}'               # e.g. '2arcsec' — ~5 pixels per synthesised beam
phasecenter    = '{PHASECENTER}'         # '' for single field; 'ICRS HH:MM:SS.s +/-DD.MM.SS.s' for mosaic
gridder        = '{GRIDDER}'            # 'standard' (single field) or 'mosaic'
wprojplanes    = {WPROJPLANES}           # 1 for standard; -1 for mosaic/wproject
deconvolver    = 'hogbom'
weighting      = 'briggs'
robust         = 0.5
niter          = {NITER}                 # start conservative: 1000–5000
threshold      = '{THRESHOLD}'          # e.g. '0.5mJy' — ~3× expected RMS
stokes         = '{STOKES}'             # 'I' default; 'IQUV' if polcal ran

# [DETERMINISTIC] Run tclean
tclean(
    vis=vis,
    imagename=image_name,
    field=target_field,
    spw='',
    imsize=imsize,
    cell=cell,
    phasecenter=phasecenter,
    gridder=gridder,
    wprojplanes=wprojplanes,
    deconvolver=deconvolver,
    weighting=weighting,
    robust=robust,
    niter=niter,
    threshold=threshold,
    stokes=stokes,
    pbcor=True,
    savemodel='none',
)

# [DETERMINISTIC] Export to FITS for archiving
exportfits(imagename=image_name + '.pbcor', fitsimage=image_name + '.pbcor.fits', overwrite=True)

# [DETERMINISTIC] Emit image paths for ms_image_stats
metrics = {
    'stage':      'IMAGING_PIPELINE',
    'image_path': image_name + '.image',
    'pbcor_path': image_name + '.pbcor',
    'psf_path':   image_name + '.psf',
    'fits_path':  image_name + '.pbcor.fits',
}
print('WILDCAT_METRICS: ' + json.dumps(metrics))
"""

# Instruction for IMAGING_PIPELINE — small-model friendly: key params only
_JSON_INSTRUCTION_IMAGING = """
The Phase 1-3 tool outputs are provided. Fill the tclean template placeholders.
Respond with a single JSON object and no other text:
{
  "next_stage": "IMAGING_PIPELINE",
  "casa_script": "<completed python script — fill ALL {PLACEHOLDER} values>",
  "summary": "<1-3 sentences: what you imaged, key parameter choices>",
  "reasoning": "<brief trace>"
}

Parameter derivation rules (apply in order, no extra reasoning needed):
- VIS: the full ms_path from context
- TARGET_FIELD: field IDs where intent contains TARGET or OBSERVE — comma-separated
- IMSIZE: [round(1.5 * primary_beam_px / cell_px) to nearest 100, same] where
    primary_beam_deg = 45.0 / (dish_m * freq_ghz),
    synth_beam_arcsec = 206265 * lambda_m / max_baseline_m,
    cell_arcsec = synth_beam_arcsec / 5,
    primary_beam_px = primary_beam_deg * 3600 / cell_arcsec
- CELL: "{cell_arcsec:.1f}arcsec"
- PHASECENTER: '' for single pointing; 'ICRS {ra} {dec}' of mosaic centre for multiple
- GRIDDER: 'mosaic' if multiple target pointings else 'standard'
- WPROJPLANES: -1 if mosaic else 1
- NITER: 3000
- THRESHOLD: '0.5mJy'
- STOKES: 'IQUV' if polcal ran (workflow_config polcal=true) else 'I'
"""

_TEMPLATES_BY_STAGE: dict[Stage, str] = {
    Stage.CALIBRATION_PREFLAG: _TEMPLATE_PREFLAG,
    Stage.POLCAL_SOLVE: _TEMPLATE_POLCAL,
    Stage.IMAGING_PIPELINE: _TEMPLATE_IMAGING,
}

# Valid next_stage values per calibration stage (LLM-driven stages only)
_VALID_CAL_NEXT_STAGES: dict[Stage, set[str]] = {
    Stage.CALIBRATION_PREFLAG: {
        Stage.CALIBRATION_PREFLAG.value,
        Stage.CALIBRATION_SOLVE.value,
    },
    Stage.CALIBRATION_SOLVE: {
        Stage.CALIBRATION_APPLY.value,
        Stage.CALIBRATION_PREFLAG.value,
        Stage.CALIBRATION_SOLVE.value,
    },
    Stage.POLCAL_SOLVE: {Stage.CALIBRATION_APPLY.value},
}

# Maximum solve retries before forcing CALIBRATION_APPLY regardless of quality
_SOLVE_MAX_RETRIES = 2

# Polcal feasibility verdicts that allow entering POLCAL_SOLVE
_POLCAL_VIABLE_VERDICTS = {"FULL", "DEGRADED"}

# Maximum preflag iterations before escalating to CALIBRATION_CHECKPOINT
_PREFLAG_MAX_ITERATIONS = 3

# If overall flag fraction exceeds this after a preflag pass, stop looping and proceed to solve.
# At P-band, >50% flagging is endemic RFI — more flagging destroys more data than it saves.
_PREFLAG_FLAG_CAP = 0.50

# SPWs with flag fraction above this after tfcrop are discarded entirely before solve.
_SPW_DISCARD_THRESHOLD = 0.80

# Per-band thresholds for CALIBRATION_SOLVE → CALIBRATION_APPLY routing.
# Keyed by band letter derived from band_centre_ghz in ms_pol_cal_feasibility.
# (bp_flagged_frac_max, gain_flagged_frac_max)
_SOLVE_THRESHOLDS_BY_BAND: dict[str, tuple[float, float]] = {
    "P": (0.60, 0.60),  # 230–470 MHz — endemic RFI environment
    "L": (0.60, 0.60),  # 1–2 GHz
    "S": (0.40, 0.40),  # 2–4 GHz
    "C": (0.20, 0.15),  # 4–8 GHz
    "X": (0.20, 0.15),  # 8–12 GHz
    "default": (0.40, 0.40),
}

# Tool outputs needed per calibration stage — only relevant tools are sent to the LLM.
# Decision context (loop/proceed) comes from workflow_config and prev_metrics, not tool dumps.
_CAL_STAGE_TOOLS: dict[Stage, set[str]] = {
    Stage.CALIBRATION_PREFLAG: {
        "ms_field_list",
        "ms_spectral_window_list",
        "ms_correlator_config",
    },
    Stage.POLCAL_SOLVE: {
        "ms_observation_info",
        "ms_field_list",
        "ms_scan_list",
        "ms_scan_intent_summary",
        "ms_spectral_window_list",
        "ms_correlator_config",
        "ms_refant",
        "ms_pol_cal_feasibility",
    },
}


# ── Internal tools for LLM-driven context querying ───────────────────────
# These are passed to the LLM as OpenAI-compatible function definitions.
# The LLM calls them to pull context it needs for its decision.

_INTERNAL_TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "get_metrics",
            "description": "Get WILDCAT_METRICS from a completed CASA job for a given stage",
            "parameters": {
                "type": "object",
                "properties": {
                    "stage": {
                        "type": "string",
                        "description": "Stage name, e.g. CALIBRATION_PREFLAG or CALIBRATION_SOLVE",
                    },
                },
                "required": ["stage"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_previous_script",
            "description": "Get the CASA script from the most recent completed job for a given stage",
            "parameters": {
                "type": "object",
                "properties": {
                    "stage": {
                        "type": "string",
                        "description": "Stage name, e.g. CALIBRATION_PREFLAG",
                    },
                },
                "required": ["stage"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tool_output",
            "description": "Get a Phase 1-3 inspection tool output by name (e.g. ms_field_list)",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Tool name, e.g. ms_field_list, ms_spectral_window_list",
                    },
                },
                "required": ["tool_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_workflow_config",
            "description": "Get the current workflow configuration (polcal, aggressive_flagging, preflag_iterations)",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_calsol_stats",
            "description": (
                "Get the full ms_calsol_stats output (including per-channel arrays) "
                "for a caltable produced by CALIBRATION_SOLVE. "
                "Use this when the summary in the prompt lacks detail needed for routing. "
                "caltable is one of: 'delay.cal', 'bandpass.cal', 'gain.cal'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "caltable": {
                        "type": "string",
                        "description": "Caltable filename, e.g. 'bandpass.cal'",
                    },
                },
                "required": ["caltable"],
            },
        },
    },
]

# Stages that use the multi-turn tool-use pattern (others keep single-turn)
_TOOL_USE_STAGES = {
    Stage.CALIBRATION_PREFLAG,
    Stage.CALIBRATION_SOLVE,
    Stage.POLCAL_SOLVE,
}


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
        *,
        max_user_tokens: int = 16000,
        max_retries: int = 5,
        max_tool_rounds: int = 5,
    ) -> None:
        self.db = db
        self.tools = tools
        self.llm = llm
        self.skills_path = skills_path
        self.runner = runner
        self.checkpoint_event = checkpoint_event
        self.stop_event = stop_event or asyncio.Event()
        self.max_user_tokens = max_user_tokens
        self.max_retries = max_retries
        self.max_tool_rounds = max_tool_rounds

    # ── Main loop ──────────────────────────────────────────────────────────

    async def run(self, workflow_id: int) -> None:
        """Main loop. Reads stage, dispatches to correct handler, exits on terminal state."""
        while True:
            wf = self.db.get_workflow(workflow_id)
            stage = Stage(wf["stage"])
            log.info("Workflow %d: stage=%s", workflow_id, stage)

            try:
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

                elif stage == Stage.CALIBRATION_APPLY:
                    await self._handle_apply_stage(workflow_id)

                elif stage in (
                    Stage.CALIBRATION_PREFLAG,
                    Stage.CALIBRATION_SOLVE,
                    Stage.POLCAL_SOLVE,
                ):
                    await self._handle_calibration_stage(workflow_id, stage)

                elif stage == Stage.CALIBRATION_CHECKPOINT:
                    await self._handle_checkpoint(workflow_id)

                elif stage == Stage.CALIBRATION_LOOP:
                    wf_config = self.db.get_workflow_config(workflow_id)
                    wf_config["_preflag_iterations"] = 0
                    wf_config["_solve_retries"] = 0
                    wf_config.pop("_preflag_flag_warning", None)
                    self.db.set_workflow_config(workflow_id, wf_config)
                    self.db.transition(workflow_id, Stage.CALIBRATION_PREFLAG)

                elif stage == Stage.IMAGING_PIPELINE:
                    await self._handle_imaging_stage(workflow_id)

                elif stage == Stage.IMAGING_CHECKPOINT:
                    await self._handle_checkpoint(workflow_id)

                elif stage == Stage.STOPPED:
                    log.info("Workflow %d stopped by user request", workflow_id)
                    break

                elif stage == Stage.ERROR:
                    log.error(
                        "Workflow %d is in ERROR state — manual intervention required",
                        workflow_id,
                    )
                    break

                else:
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

            except Exception:
                log.exception(
                    "Unhandled error in stage %s for workflow %d — entering ERROR",
                    stage,
                    workflow_id,
                )
                self.db.transition(workflow_id, Stage.ERROR)
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
        runner_map = {
            1: self.tools.run_phase1,
            2: self.tools.run_phase2,
            3: self.tools.run_phase3,
        }
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
        if phase > 1:
            wf_config = self.db.get_workflow_config(workflow_id)
            user_content += (
                "\n\n## Workflow config\n"
                "```json\n" + json.dumps(wf_config, indent=2) + "\n```\n"
            )
        json_instruction = _JSON_INSTRUCTION_PHASE3 if phase == 3 else _JSON_INSTRUCTION
        messages = [
            {"role": "system", "content": system_prompt + "\n\n" + json_instruction},
            {"role": "user", "content": user_content},
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
        self.db.save_llm_decision(
            workflow_id, stage.value, json.dumps(decision), model, prompt_hash
        )

        if phase != 3 and decision.get("casa_script"):
            if self.stop_event.is_set():
                log.info("Stop requested — skipping CASA job in Phase %d", phase)
                self.db.transition(workflow_id, Stage.STOPPED)
                return
            await self._run_casa_job(workflow_id, stage.value, decision["casa_script"])
            if self.db.get_workflow(workflow_id)["stage"] == Stage.ERROR.value:
                return

        next_stage_str = decision["next_stage"]
        try:
            next_stage = Stage(next_stage_str)
        except ValueError:
            log.error(
                "LLM returned unknown next_stage %r — entering ERROR", next_stage_str
            )
            self.db.transition(workflow_id, Stage.ERROR)
            return

        # Guard: LLM must not escalate to HUMAN_CHECKPOINT from inspection phases.
        # Phase 1/2 should advance to the next phase; Phase 3 applies deterministic
        # config rules and proceeds to calibration.
        _phase_fallback = {
            1: Stage.PHASE2_RUNNING,
            2: Stage.PHASE3_RUNNING,
            3: Stage.CALIBRATION_PREFLAG,
        }
        if next_stage == Stage.HUMAN_CHECKPOINT:
            fallback = _phase_fallback[phase]
            log.warning(
                "Workflow %d: LLM chose HUMAN_CHECKPOINT in Phase %d — overriding to %s",
                workflow_id,
                phase,
                fallback.value,
            )
            next_stage = fallback

        if phase == 3:
            # Apply deterministic config rules after routing is resolved
            self._apply_deterministic_config(workflow_id)

        self.db.transition(workflow_id, next_stage)

    # ── Deterministic config rules ─────────────────────────────────────────

    def _apply_deterministic_config(self, workflow_id: int) -> dict:
        """Apply rule-based config overrides from Phase 2/3 tool outputs.

        Rules:
          - ms_pol_cal_feasibility verdict NOT_FEASIBLE or DEGRADED → polcal=False
          - Any antenna with flag_fraction >= 1.0 → aggressive_flagging=True

        These always override whatever the LLM decided. Returns the overrides dict.
        """
        all_outputs = self._load_all_tool_outputs(workflow_id)
        wf_config = self.db.get_workflow_config(workflow_id)
        overrides: dict = {}

        # Rule 1: polcal feasibility from dedicated tool
        verdict = (
            all_outputs.get("ms_pol_cal_feasibility", {})
            .get("data", {})
            .get("verdict", "")
        )
        if verdict in ("NOT_FEASIBLE", "DEGRADED") and wf_config.get("polcal", True):
            wf_config["polcal"] = False
            overrides["polcal"] = False
            log.warning(
                "Workflow %d: polcal=False — ms_pol_cal_feasibility verdict=%r",
                workflow_id,
                verdict,
            )

        # Rule 2: any antenna fully flagged → enable aggressive flagging
        per_antenna = (
            all_outputs.get("ms_antenna_flag_fraction", {})
            .get("data", {})
            .get("per_antenna", [])
        )
        fully_flagged = [
            a["antenna_name"]
            for a in per_antenna
            if a.get("flag_fraction", {}).get("value", 0) >= 1.0
        ]
        if fully_flagged and not wf_config.get("aggressive_flagging", False):
            wf_config["aggressive_flagging"] = True
            overrides["aggressive_flagging"] = True
            log.warning(
                "Workflow %d: aggressive_flagging=True — antennas at 100%%: %s",
                workflow_id,
                fully_flagged,
            )

        if overrides:
            self.db.set_workflow_config(workflow_id, wf_config)
            log.info(
                "Workflow %d: deterministic config overrides applied: %s",
                workflow_id,
                overrides,
            )
        return overrides

    # ── Calibration stage handler ──────────────────────────────────────────

    async def _handle_calibration_stage(self, workflow_id: int, stage: Stage) -> None:
        """Calibration stage dispatcher.

        CALIBRATION_SOLVE: run CASA deterministically first, then call ms_calsol_stats,
          then LLM interprets quality and routes.
        CALIBRATION_PREFLAG, POLCAL_SOLVE: LLM fills template, then CASA runs.
        """
        if self.stop_event.is_set():
            self.db.transition(workflow_id, Stage.STOPPED)
            return

        wf = self.db.get_workflow(workflow_id)
        wf_config = self.db.get_workflow_config(workflow_id)
        preflag_iterations = wf_config.get("_preflag_iterations", 0)

        # ── CALIBRATION_SOLVE: run first, inspect, then LLM ───────────────
        if stage == Stage.CALIBRATION_SOLVE:
            solve_retries = wf_config.get("_solve_retries", 0)

            # Enforce retry cap — force proceed after too many retries
            if solve_retries >= _SOLVE_MAX_RETRIES:
                log.warning(
                    "Workflow %d: CALIBRATION_SOLVE retry cap (%d) reached — forcing CALIBRATION_APPLY",
                    workflow_id,
                    _SOLVE_MAX_RETRIES,
                )
                self.db.transition(workflow_id, Stage.CALIBRATION_APPLY)
                return

            # Build and run the deterministic solve script
            script = self._build_solve_script(workflow_id)
            await self._run_casa_job(workflow_id, Stage.CALIBRATION_SOLVE.value, script)
            if self.db.get_workflow(workflow_id)["stage"] == Stage.ERROR.value:
                return

            # Call ms_calsol_stats on all three caltables in parallel
            metrics = self.db.get_last_job_metrics(
                workflow_id, Stage.CALIBRATION_SOLVE.value
            )
            t_delay = metrics.get("t_delay", "")
            t_bp = metrics.get("t_bp", "")
            t_gain = metrics.get("t_gain", "")
            delay_stats, bp_stats, gain_stats = await asyncio.gather(
                self._run_calsol_stats(workflow_id, t_delay),
                self._run_calsol_stats(workflow_id, t_bp),
                self._run_calsol_stats(workflow_id, t_gain),
            )

            band = self._get_band(workflow_id)
            bp_max, gain_max = _SOLVE_THRESHOLDS_BY_BAND.get(
                band, _SOLVE_THRESHOLDS_BY_BAND["default"]
            )

            instruction = _JSON_INSTRUCTION_SOLVE
            system_prompt = load_system_prompt(self.skills_path, stage)
            user_content = "\n".join(
                [
                    "## CALIBRATION_SOLVE — post-run inspection",
                    f"Band: {band}  |  bp threshold: {bp_max}  |  gain threshold: {gain_max}",
                    f"Preflag iterations: {preflag_iterations} / {_PREFLAG_MAX_ITERATIONS}",
                    f"Solve retries so far: {solve_retries} / {_SOLVE_MAX_RETRIES}",
                    "",
                    "## WILDCAT_METRICS",
                    "```json",
                    json.dumps(metrics, indent=2),
                    "```",
                    "",
                    "## CALSOL_STATS — delay.cal",
                    "```json",
                    json.dumps(delay_stats, indent=2),
                    "```",
                    "",
                    "## CALSOL_STATS — bandpass.cal",
                    "```json",
                    json.dumps(bp_stats, indent=2),
                    "```",
                    "",
                    "## CALSOL_STATS — gain.cal",
                    "```json",
                    json.dumps(gain_stats, indent=2),
                    "```",
                ]
            )
            messages = [
                {"role": "system", "content": system_prompt + "\n\n" + instruction},
                {"role": "user", "content": user_content},
            ]

            if self.stop_event.is_set():
                self.db.transition(workflow_id, Stage.STOPPED)
                return

            decision, _model = await self._llm_call_with_retry_and_tools(
                workflow_id, stage, messages
            )
            if decision is None:
                return

            next_stage_str = decision["next_stage"]
            valid = _VALID_CAL_NEXT_STAGES[stage]
            if next_stage_str not in valid:
                log.error(
                    "LLM returned illegal next_stage %r for CALIBRATION_SOLVE — entering ERROR",
                    next_stage_str,
                )
                self.db.transition(workflow_id, Stage.ERROR)
                return

            # Track retries; run modified script if LLM wants a retry
            if next_stage_str == Stage.CALIBRATION_SOLVE.value:
                wf_config["_solve_retries"] = solve_retries + 1
                self.db.set_workflow_config(workflow_id, wf_config)
                if decision.get("casa_script"):
                    await self._run_casa_job(
                        workflow_id,
                        Stage.CALIBRATION_SOLVE.value,
                        self._sanitize_llm_script(decision["casa_script"]),
                    )
                    if self.db.get_workflow(workflow_id)["stage"] == Stage.ERROR.value:
                        return

            # Polcal routing on CALIBRATION_APPLY path
            if next_stage_str == Stage.CALIBRATION_APPLY.value and wf_config.get(
                "polcal"
            ):
                polcal_verdict = self._get_polcal_verdict(workflow_id)
                if polcal_verdict in _POLCAL_VIABLE_VERDICTS:
                    next_stage_str = Stage.POLCAL_SOLVE.value

            self.db.transition(workflow_id, Stage(next_stage_str))
            return

        # ── CALIBRATION_PREFLAG / POLCAL_SOLVE: LLM first, CASA second ────

        # Increment counter at entry so it counts all PREFLAG runs, regardless of
        # whether re-entry was driven by the LLM, the flag cap, or SOLVE threshold failure.
        if stage == Stage.CALIBRATION_PREFLAG:
            preflag_iterations += 1
            wf_config["_preflag_iterations"] = preflag_iterations
            self.db.set_workflow_config(workflow_id, wf_config)

        # Enforce preflag iteration cap
        if (
            stage == Stage.CALIBRATION_PREFLAG
            and preflag_iterations >= _PREFLAG_MAX_ITERATIONS
        ):
            log.warning(
                "CALIBRATION_PREFLAG hit cap (%d iterations) for workflow %d — escalating",
                _PREFLAG_MAX_ITERATIONS,
                workflow_id,
            )
            wf_config["_preflag_flag_warning"] = True
            self.db.set_workflow_config(workflow_id, wf_config)
            self.db.transition(workflow_id, Stage.CALIBRATION_CHECKPOINT)
            return

        instruction = _JSON_INSTRUCTIONS_BY_STAGE[stage]
        system_prompt = load_system_prompt(self.skills_path, stage)

        # 1. Build minimal base context — the LLM can query for more via tools
        is_reentry = stage == Stage.CALIBRATION_PREFLAG and preflag_iterations > 1
        base_lines = [
            f"## Calibration stage: {stage.value}",
            f"Measurement set: {wf['ms_path']}",
            f"Preflag iterations: {preflag_iterations} / {_PREFLAG_MAX_ITERATIONS}",
        ]

        # Always include the template (LLM needs it to produce the script)
        template = _TEMPLATES_BY_STAGE.get(stage)
        tmpl: str | None = None
        if template:
            tmpl = template.replace("{WORKFLOW_ID}", str(workflow_id))
            # Pre-fill ALL deterministic placeholders before showing to the LLM.
            # After this call the script is complete and runnable — no blanks remain.
            if stage == Stage.CALIBRATION_PREFLAG:
                tmpl = self._prefill_preflag_template(workflow_id, tmpl)
            base_lines.append("\n## CASA script")
            base_lines.append(
                "All values are already filled. Copy this script verbatim into "
                "the casa_script field of your JSON response. "
                "Do NOT modify sections marked [DETERMINISTIC]."
            )
            base_lines.append(f"```python\n{tmpl}\n```")

        if is_reentry:
            base_lines.append(
                "\nThis is a re-entry: the previous PREFLAG pass didn't produce clean enough "
                "calibration. Use the tools to retrieve the previous script and metrics, then "
                "adjust flagging parameters and return a new script."
            )
        else:
            base_lines.append(
                "\nReview the script above, then return your decision. "
                "Use the tools if you need to inspect Phase 1-3 outputs or workflow config."
            )

        user_content = "\n".join(base_lines)
        system_with_instruction = system_prompt + "\n\n" + instruction
        messages = [
            {"role": "system", "content": system_with_instruction},
            {"role": "user", "content": user_content},
        ]

        if self.stop_event.is_set():
            self.db.transition(workflow_id, Stage.STOPPED)
            return

        # 2. LLM call with tools and retry
        decision, model = await self._llm_call_with_retry_and_tools(
            workflow_id, stage, messages
        )
        if decision is None:
            return

        # 3. Validate next_stage is legal for this stage
        next_stage_str = decision["next_stage"]
        valid = _VALID_CAL_NEXT_STAGES.get(stage, set())
        if next_stage_str not in valid:
            log.error(
                "LLM returned illegal next_stage %r for %s (valid: %s) — entering ERROR",
                next_stage_str,
                stage,
                valid,
            )
            self.db.transition(workflow_id, Stage.ERROR)
            return

        # 4. Run CASA script
        # For the first PREFLAG pass the template is fully deterministic — use it
        # directly to avoid LLM-introduced mutations (e.g. trailing commas that
        # turn list assignments into tuples).  Re-entries need the LLM's script
        # because it adjusts flagging parameters.
        if stage == Stage.CALIBRATION_PREFLAG and tmpl is not None and not is_reentry:
            script = tmpl
        elif decision.get("casa_script"):
            script = decision["casa_script"]
            if stage == Stage.CALIBRATION_PREFLAG and tmpl is not None:
                script = self._prefill_preflag_template(workflow_id, script)
            script = self._sanitize_llm_script(script)
        else:
            script = None

        if script is not None:
            if self.stop_event.is_set():
                self.db.transition(workflow_id, Stage.STOPPED)
                return
            # Hard-fail if any {PLACEHOLDER} patterns remain — do not pass to CASA.
            _KNOWN_PLACEHOLDERS = {
                "{CAL_FIELDS}", "{ALL_SPW}", "{CORRSTRING}",
                "{VIS}", "{WORKFLOW_ID}", "{SPW_DISCARD_THRESHOLD}",
            }
            bad = [p for p in re.findall(r"\{[A-Z_a-z]+\}", script) if p in _KNOWN_PLACEHOLDERS]
            if bad:
                raise RuntimeError(
                    f"Workflow {workflow_id}: casa_script still contains unfilled "
                    f"placeholders after safety substitution: {bad} — "
                    "refusing to pass to CASA"
                )
            await self._run_casa_job(workflow_id, stage.value, script)
            if self.db.get_workflow(workflow_id)["stage"] == Stage.ERROR.value:
                return

        # 4b. Flag fraction cap
        if (
            stage == Stage.CALIBRATION_PREFLAG
            and next_stage_str == Stage.CALIBRATION_PREFLAG.value
        ):
            current_metrics = self.db.get_last_job_metrics(
                workflow_id, Stage.CALIBRATION_PREFLAG.value
            )
            flag_frac = current_metrics.get("overall_flag_frac", 0.0)
            if isinstance(flag_frac, (int, float)) and flag_frac >= _PREFLAG_FLAG_CAP:
                log.warning(
                    "Workflow %d: overall_flag_frac=%.3f >= cap %.2f — forcing CALIBRATION_SOLVE",
                    workflow_id,
                    flag_frac,
                    _PREFLAG_FLAG_CAP,
                )
                next_stage_str = Stage.CALIBRATION_SOLVE.value

        # 4c. Polcal routing
        if (
            stage == Stage.CALIBRATION_PREFLAG
            and next_stage_str == Stage.CALIBRATION_SOLVE.value
        ):
            if wf_config.get("polcal"):
                polcal_verdict = self._get_polcal_verdict(workflow_id)
                if polcal_verdict in _POLCAL_VIABLE_VERDICTS:
                    log.info(
                        "Workflow %d: polcal=True, verdict=%s — routing to POLCAL_SOLVE",
                        workflow_id,
                        polcal_verdict,
                    )
                    next_stage_str = Stage.POLCAL_SOLVE.value
                else:
                    log.info(
                        "Workflow %d: polcal=True but verdict=%r — using CALIBRATION_SOLVE",
                        workflow_id,
                        polcal_verdict,
                    )

        # 5. Transition
        self.db.transition(workflow_id, Stage(next_stage_str))

    # ── Internal tool executor ─────────────────────────────────────────────

    def _execute_internal_tool(
        self, workflow_id: int, tool_name: str, args: dict
    ) -> str:
        """Dispatch an internal tool call and return the result as a JSON string."""
        if tool_name == "get_metrics":
            stage_filter = args.get("stage")
            metrics = self.db.get_last_job_metrics(workflow_id, stage_filter)
            return (
                json.dumps(metrics, indent=2)
                if metrics
                else '{"error": "no metrics found"}'
            )

        if tool_name == "get_previous_script":
            stage_filter = args.get("stage", "CALIBRATION_PREFLAG")
            row = self.db.conn.execute(
                "SELECT script_path FROM jobs"
                " WHERE workflow_id = ? AND stage = ? AND status = 'done'"
                " ORDER BY id DESC LIMIT 1",
                (workflow_id, stage_filter),
            ).fetchone()
            if row and row["script_path"]:
                try:
                    with open(row["script_path"]) as f:
                        return f.read()
                except OSError:
                    pass
            return '{"error": "no script found"}'

        if tool_name == "get_tool_output":
            tn = args.get("tool_name", "")
            outputs = self._load_all_tool_outputs(workflow_id, tools={tn})
            if tn in outputs:
                return json.dumps(outputs[tn], indent=2)
            return f'{{"error": "tool output {tn} not found"}}'

        if tool_name == "get_workflow_config":
            return json.dumps(self.db.get_workflow_config(workflow_id), indent=2)

        if tool_name == "get_calsol_stats":
            caltable = args.get("caltable", "")
            rows = self.db.get_tool_outputs(workflow_id, "calibration_solve")
            for row in rows:
                if row["tool_name"] == f"ms_calsol_stats:{caltable}":
                    return row["output_json"]
            return f'{{"error": "calsol_stats for {caltable!r} not found"}}'

        return f'{{"error": "unknown tool {tool_name}"}}'

    # ── LLM retry with tool use ───────────────────────────────────────────

    async def _llm_call_with_retry_and_tools(
        self, workflow_id: int, stage: Stage, messages: list[dict]
    ) -> tuple[dict | None, str]:
        """Call the LLM with internal tools, retrying on failure.

        For stages in _TOOL_USE_STAGES, uses multi-turn tool calling.
        For other stages, falls back to simple single-turn calls.
        Returns (decision_dict, model_str) or (None, "") on total failure.
        """
        use_tools = stage in _TOOL_USE_STAGES
        last_error = ""

        for attempt in range(1, self.max_retries + 1):
            try:
                if use_tools:
                    response = await self.llm.complete_with_tools(
                        messages,
                        tools=_INTERNAL_TOOLS_OPENAI,
                        tool_executor=lambda name, args: self._execute_internal_tool(
                            workflow_id, name, args
                        ),
                        max_rounds=self.max_tool_rounds,
                        max_result_tokens=self.max_user_tokens,
                    )
                else:
                    response = await self.llm.complete(messages)

                raw = self._extract_content(response)
                if not raw or not raw.strip():
                    raise ValueError("LLM returned empty response")
                decision = self._parse_decision(raw)
                model = response.get("model", "unknown")
                self.db.save_llm_decision(
                    workflow_id, stage.value, json.dumps(decision), model
                )
                if attempt > 1:
                    log.info(
                        "Workflow %d: LLM succeeded on attempt %d/%d for %s",
                        workflow_id,
                        attempt,
                        self.max_retries,
                        stage.value,
                    )
                return decision, model

            except (ValueError, KeyError) as exc:
                last_error = str(exc)
                log.warning(
                    "Workflow %d: LLM attempt %d/%d failed for %s — %s",
                    workflow_id,
                    attempt,
                    self.max_retries,
                    stage.value,
                    last_error,
                )
                self.db.save_llm_decision(
                    workflow_id,
                    stage.value,
                    json.dumps({"_retry_error": last_error, "_attempt": attempt}),
                    "retry",
                )

        log.error(
            "Workflow %d: all %d LLM retries exhausted for %s — entering ERROR. Last: %s",
            workflow_id,
            self.max_retries,
            stage.value,
            last_error,
        )
        self.db.transition(workflow_id, Stage.ERROR)
        return None, ""

    # ── Deterministic CALIBRATION_SOLVE ───────────────────────────────────

    # Maps the tool-output flux_standard identifier to the CASA-accepted string.
    _FLUX_STANDARD_MAP: dict[str, str] = {
        "Perley-Butler-2017": "Perley-Butler 2017",
        "Perley-Butler-2013": "Perley-Butler 2013",
        "Scaife-Heald-2012": "Scaife-Heald 2012",
        "Stevens-Reynolds-2016": "Stevens-Reynolds 2016",
    }

    async def _generate_calsol_plots(self, workflow_id: int) -> None:
        """Call ms_calsol_plot_library for delay/bandpass/gain caltables.

        Runs after CALIBRATION_APPLY so plots are ready at the checkpoint.
        Failures are logged but do not abort the workflow.
        """
        metrics = self.db.get_last_job_metrics(
            workflow_id, Stage.CALIBRATION_SOLVE.value
        )
        workdir = f"/data/jobs/{workflow_id}"
        caltable_paths = [
            p for p in [
                metrics.get("t_delay", f"{workdir}/delay.cal"),
                metrics.get("t_bp",    f"{workdir}/bandpass.cal"),
                metrics.get("t_gain",  f"{workdir}/gain.cal"),
            ]
            if p
        ]
        if not caltable_paths:
            log.warning("Workflow %d: no caltable paths in metrics — skipping plots", workflow_id)
            return

        try:
            result = await self.tools.call_tool(
                "ms_plot_caltable_library",
                {"params": {"caltable_paths": caltable_paths, "output_dir": workdir}},
            )
            plots_list = result.get("data", {}).get("plots", {}).get("value", []) if isinstance(result, dict) else []
            html_paths = [p["html_path"] for p in plots_list if p.get("status") == "ok" and p.get("html_path")]
            if html_paths:
                # Store plot paths in the most recent job row for the UI
                job_row = self.db.conn.execute(
                    "SELECT id FROM jobs WHERE workflow_id = ? AND stage = ?"
                    " ORDER BY completed_at DESC LIMIT 1",
                    (workflow_id, Stage.CALIBRATION_APPLY.value),
                ).fetchone()
                if job_row:
                    self.db.conn.execute(
                        "UPDATE jobs SET plots = ? WHERE id = ?",
                        (json.dumps(html_paths), job_row["id"]),
                    )
                    self.db.conn.commit()
            log.info("Workflow %d: calsol plots generated: %s", workflow_id, html_paths)
        except Exception as exc:
            log.warning("Workflow %d: ms_calsol_plot_library failed: %s", workflow_id, exc)

    async def _run_calsol_stats(
        self, workflow_id: int, caltable_path: str
    ) -> dict:
        """Call ms_calsol_stats via MCP on a single caltable.

        Stores the full raw result in tool_outputs (phase="calibration_solve")
        so the LLM can retrieve per-channel detail on demand. Returns a compact
        summary dict for the initial prompt.
        """
        try:
            raw = await self.tools.call_tool(
                "ms_calsol_stats", {"params": {"caltable_path": caltable_path}}
            )
            tool_name = f"ms_calsol_stats:{Path(caltable_path).name}"
            self.db.save_tool_output(
                workflow_id,
                "calibration_solve",
                tool_name,
                json.dumps(raw),
            )
            return self._summarize_calsol_stats(raw)
        except Exception as exc:
            log.warning("ms_calsol_stats failed for %s: %s", caltable_path, exc)
            return {"error": str(exc)}

    @staticmethod
    def _summarize_calsol_stats(stats: dict) -> dict:
        """Strip raw per-channel arrays from ms_calsol_stats output.

        Replaces large nested arrays with scalar {min, max, mean} summaries
        and compact per-antenna/per-SPW flag fraction dicts so the result fits
        comfortably inside the LLM context window.
        """
        if "error" in stats or "data" not in stats:
            return stats

        data = stats["data"]

        def val(key, default=None):
            v = data.get(key, {})
            return v.get("value", default) if isinstance(v, dict) else (v if v is not None else default)

        def flat_stats(arr) -> dict | None:
            nums: list[float] = []

            def walk(x: object) -> None:
                if isinstance(x, list):
                    for item in x:
                        walk(item)
                elif isinstance(x, (int, float)) and not math.isnan(x):
                    nums.append(float(x))

            walk(arr)
            if not nums:
                return None
            return {
                "min": round(min(nums), 4),
                "max": round(max(nums), 4),
                "mean": round(sum(nums) / len(nums), 4),
            }

        ant_names: list[str] = val("ant_names", [])
        spw_ids: list[int] = val("spw_ids", [])
        flag_raw: list = val("flagged_frac", [])  # shape [n_ant, n_spw, n_field]

        per_spw_flag: dict[str, float | None] = {}
        per_ant_flag: dict[str, float | None] = {}
        if flag_raw:
            n_ant = len(flag_raw)
            n_spw = len(flag_raw[0]) if flag_raw else 0
            for si, spw in enumerate(spw_ids):
                spw_vals = [
                    flag_raw[ai][si][0]
                    for ai in range(n_ant)
                    if si < len(flag_raw[ai]) and flag_raw[ai][si]
                ]
                per_spw_flag[str(spw)] = round(sum(spw_vals) / len(spw_vals), 3) if spw_vals else None
            for ai, ant in enumerate(ant_names):
                if ai < n_ant:
                    ant_vals = [
                        flag_raw[ai][si][0]
                        for si in range(n_spw)
                        if si < len(flag_raw[ai]) and flag_raw[ai][si]
                    ]
                    per_ant_flag[ant] = round(sum(ant_vals) / len(ant_vals), 3) if ant_vals else None

        out: dict = {
            "table_type": val("table_type"),
            "n_antennas": val("n_antennas"),
            "n_spw": val("n_spw"),
            "overall_flagged_frac": val("overall_flagged_frac"),
            "n_antennas_lost": val("n_antennas_lost"),
            "antennas_lost": val("antennas_lost", []),
            "per_spw_flagged_frac": per_spw_flag,
            "per_ant_flagged_frac": per_ant_flag,
            "outliers": val("outliers", {}),
        }

        table_type = out["table_type"]
        if table_type == "K":
            if s := flat_stats(val("delay_ns")):
                out["delay_ns"] = s
            if s := flat_stats(val("delay_rms_ns")):
                out["delay_rms_ns"] = s
        else:
            for metric in ("amp_mean", "amp_std", "phase_mean_deg", "phase_rms_deg", "snr_mean"):
                if s := flat_stats(val(metric)):
                    out[metric] = s

        return out

    @staticmethod
    def _sanitize_llm_script(script: str) -> str:
        """Remove common LLM-introduced Python syntax errors from generated CASA scripts.

        Covers:
        - Trailing commas after list assignments that create accidental tuples,
          e.g.  flag_cmds = [...],  →  flag_cmds = [...]
          The pattern handles list items that themselves contain brackets.
        """
        return re.sub(
            r"(\bflag_cmds\s*=\s*\[(?:[^\[\]]|\[[^\[\]]*\])*\])\s*,",
            r"\1",
            script,
            flags=re.DOTALL,
        )

    def _prefill_preflag_template(self, workflow_id: int, tmpl: str) -> str:
        """Fill deterministic PREFLAG placeholders from stored tool outputs.

        Fills VIS, CAL_FIELDS, ALL_SPW, CORRSTRING — the same derivation used
        by _build_solve_script — so the LLM cannot misidentify target fields.

        Raises RuntimeError if required tool outputs are missing or yield no
        calibrator fields — callers must not proceed with an unfilled template.
        """
        outputs = self._load_all_tool_outputs(workflow_id)

        field_data = outputs.get("ms_field_list", {}).get("data", {})
        if not field_data:
            raise RuntimeError(
                f"Workflow {workflow_id}: ms_field_list tool output not found — "
                "cannot determine calibrator fields for PREFLAG script"
            )

        fields = field_data.get("fields", [])
        cal_ids = [
            str(f["field_id"])
            for f in fields
            if f.get("calibrator_role", {}).get("value")
            or f.get("calibrator_match", {}).get("value")
        ]
        if not cal_ids:
            raise RuntimeError(
                f"Workflow {workflow_id}: no calibrator fields found in ms_field_list — "
                "cannot build PREFLAG script (check Phase 1 tool outputs)"
            )
        cal_fields = ",".join(cal_ids)

        spw_data = outputs.get("ms_spectral_window_list", {}).get("data", {})
        if not spw_data:
            raise RuntimeError(
                f"Workflow {workflow_id}: ms_spectral_window_list tool output not found — "
                "cannot determine SPW range for PREFLAG script"
            )
        n_spw = spw_data.get("n_spw")
        if not n_spw:
            raise RuntimeError(
                f"Workflow {workflow_id}: n_spw missing from ms_spectral_window_list"
            )
        all_spw = f"0~{n_spw - 1}"

        corr_products = (
            outputs.get("ms_correlator_config", {})
            .get("data", {})
            .get("correlation_products", {})
            .get("value", [])
        )
        if not corr_products:
            raise RuntimeError(
                f"Workflow {workflow_id}: ms_correlator_config tool output not found — "
                "cannot determine correlation products for PREFLAG script"
            )
        parallel = [c for c in corr_products if c in ("XX", "YY", "RR", "LL")]
        if not parallel:
            raise RuntimeError(
                f"Workflow {workflow_id}: no parallel-hand correlations found in "
                f"ms_correlator_config (got {corr_products!r})"
            )
        corrstring = ",".join(parallel)

        wf = self.db.get_workflow(workflow_id)
        filled = (
            tmpl.replace("{VIS}", wf["ms_path"])
            .replace("{CAL_FIELDS}", cal_fields)
            .replace("{ALL_SPW}", all_spw)
            .replace("{CORRSTRING}", corrstring)
            .replace("{SPW_DISCARD_THRESHOLD}", str(_SPW_DISCARD_THRESHOLD))
        )

        # Sanity check — no placeholders should remain after deterministic fill
        remaining = re.findall(r"\{[A-Z_]+\}", filled)
        if remaining:
            raise RuntimeError(
                f"Workflow {workflow_id}: PREFLAG template has unfilled placeholders "
                f"after substitution: {remaining}"
            )
        return filled

    def _build_solve_script(self, workflow_id: int) -> str:
        """Fill all CALIBRATION_SOLVE placeholders from Phase 1-3 tool outputs."""
        outputs = self._load_all_tool_outputs(workflow_id)

        flux_field_id = self._solve_flux_field_id(outputs)
        phase_field_id = self._solve_phase_field_id(outputs, flux_field_id)

        n_spw = (
            outputs.get("ms_spectral_window_list", {}).get("data", {}).get("n_spw", 16)
        )
        all_spw = f"0~{n_spw - 1}"

        int_time_s = (
            outputs.get("ms_correlator_config", {})
            .get("data", {})
            .get("dump_time_s", {})
            .get("value", 2.0)
        )

        subs = {
            "{WORKFLOW_ID}": str(workflow_id),
            "{FLUX_FIELD}": str(flux_field_id),
            "{BP_FIELD}": str(flux_field_id),
            "{DELAY_FIELD}": str(flux_field_id),
            "{PHASE_FIELD}": str(phase_field_id),
            "{PHASE_SCAN_IDS}": self._solve_cal_scan_ids(outputs, str(phase_field_id)),
            "{REFANT}": self._solve_best_refant(outputs),
            "{ALL_SPW}": all_spw,
            "{FLUX_STANDARD}": self._solve_flux_standard(outputs, flux_field_id),
            "{MINBLPERANT}": "4",
            "{INT_TIME_S}": f"{float(int_time_s):.2f}",
        }
        script = _TEMPLATE_SOLVE
        for placeholder, value in subs.items():
            script = script.replace(placeholder, value)
        return script

    def _solve_flux_field_id(self, outputs: dict) -> int:
        """Return the field_id of the flux calibrator.

        Prefers a field with calibrator_role containing 'flux'; falls back to
        the first field with any calibrator_match.
        """
        fields = outputs.get("ms_field_list", {}).get("data", {}).get("fields", [])
        for f in fields:
            if "flux" in (f.get("calibrator_role", {}).get("value") or []):
                return f["field_id"]
        for f in fields:
            if f.get("calibrator_match", {}).get("value"):
                return f["field_id"]
        return 0

    def _solve_phase_field_id(self, outputs: dict, flux_field_id: int) -> int:
        """Return the field_id of the phase calibrator.

        Uses a field explicitly marked 'phase'; falls back to the flux calibrator
        field when no separate phase cal exists (common for short snapshots).
        """
        fields = outputs.get("ms_field_list", {}).get("data", {}).get("fields", [])
        for f in fields:
            if "phase" in (f.get("calibrator_role", {}).get("value") or []):
                return f["field_id"]
        return flux_field_id

    def _solve_cal_scan_ids(self, outputs: dict, field_id: str) -> str:
        """Return comma-separated scan numbers on field_id from the scan list."""
        scans = outputs.get("ms_scan_list", {}).get("data", {}).get("scans", [])
        ids = [
            str(s["scan_number"]) for s in scans if str(s.get("field_id")) == field_id
        ]
        return ",".join(ids) if ids else "1"

    def _solve_best_refant(self, outputs: dict) -> str:
        """Return the antenna with the lowest flag fraction (excluding 100% flagged)."""
        per_ant = (
            outputs.get("ms_antenna_flag_fraction", {})
            .get("data", {})
            .get("per_antenna", [])
        )
        valid = [
            a for a in per_ant if (a.get("flag_fraction", {}).get("value") or 1.0) < 1.0
        ]
        if not valid:
            return "ea01"
        return min(valid, key=lambda a: a["flag_fraction"]["value"])["antenna_name"]

    def _solve_flux_standard(self, outputs: dict, flux_field_id: int) -> str:
        """Map the tool-output flux_standard to the CASA-accepted string."""
        fields = outputs.get("ms_field_list", {}).get("data", {}).get("fields", [])
        for f in fields:
            if f["field_id"] == flux_field_id:
                raw = f.get("flux_standard", {}).get("value") or ""
                if raw:
                    return self._FLUX_STANDARD_MAP.get(raw, raw)
                # flux_standard absent from catalogue — safe default for VLA flux cals
                return "Perley-Butler 2017"
        return "Perley-Butler 2017"

    # ── Deterministic CALIBRATION_APPLY ───────────────────────────────────

    async def _handle_apply_stage(self, workflow_id: int) -> None:
        """CALIBRATION_APPLY: fill template deterministically, run CASA, then call LLM
        to interpret the metrics into human-facing checkpoint questions.
        """
        if self.stop_event.is_set():
            self.db.transition(workflow_id, Stage.STOPPED)
            return

        script = self._build_apply_script(workflow_id)
        await self._run_casa_job(workflow_id, Stage.CALIBRATION_APPLY.value, script)
        if self.db.get_workflow(workflow_id)["stage"] == Stage.ERROR.value:
            return

        # Generate calsol plots for all three caltables in parallel
        await self._generate_calsol_plots(workflow_id)

        # Read metrics from both stages for the LLM summary
        apply_metrics = self.db.get_last_job_metrics(
            workflow_id, Stage.CALIBRATION_APPLY.value
        )
        solve_metrics = self.db.get_last_job_metrics(
            workflow_id, Stage.CALIBRATION_SOLVE.value
        )

        user_content = (
            "## CALIBRATION_SOLVE metrics\n```json\n"
            + json.dumps(solve_metrics, indent=2)
            + "\n```\n\n## CALIBRATION_APPLY metrics\n```json\n"
            + json.dumps(apply_metrics, indent=2)
            + "\n```"
        )
        system_prompt = load_system_prompt(self.skills_path, Stage.CALIBRATION_APPLY)
        messages = [
            {
                "role": "system",
                "content": system_prompt + "\n\n" + _JSON_INSTRUCTION_APPLY,
            },
            {"role": "user", "content": user_content},
        ]

        decision, _model = await self._llm_call_with_retry_and_tools(
            workflow_id, Stage.CALIBRATION_APPLY, messages
        )
        if decision is None:
            return

        if decision.get("auto_proceed"):
            log.info(
                "Workflow %d: auto_proceed=True — skipping checkpoint, advancing to IMAGING_PIPELINE",
                workflow_id,
            )
            self.db.transition(workflow_id, Stage.IMAGING_PIPELINE)
        else:
            self.db.transition(workflow_id, Stage.CALIBRATION_CHECKPOINT)

    async def _handle_imaging_stage(self, workflow_id: int) -> None:
        """IMAGING_PIPELINE: LLM fills tclean template, CASA runs, ms_image_stats called.

        Context is assembled deterministically from Phase 1-3 tool outputs so the
        small model only needs to derive tclean parameters — it doesn't query tools.
        """
        if self.stop_event.is_set():
            self.db.transition(workflow_id, Stage.STOPPED)
            return

        wf = self.db.get_workflow(workflow_id)
        wf_config = self.db.get_workflow_config(workflow_id)
        outputs = self._load_all_tool_outputs(workflow_id)

        # Derive key observing parameters for the LLM
        obs = outputs.get("ms_observation_info", {}).get("data", {})
        ants = outputs.get("ms_antenna_list", {}).get("data", {})
        baselines = outputs.get("ms_baseline_lengths", {}).get("data", {})
        spws = outputs.get("ms_spectral_window_list", {}).get("data", {})
        fields = outputs.get("ms_field_list", {}).get("data", {}).get("fields", [])

        target_fields = [
            f for f in fields
            if not (f.get("calibrator_role", {}).get("value") or f.get("calibrator_match", {}).get("value"))
        ]

        # Derive centre freq and total bandwidth from the SPW list
        _spw_list = spws.get("spectral_windows", [])
        _centre_freqs = [
            s["centre_freq_hz"]["value"]
            for s in _spw_list
            if isinstance(s.get("centre_freq_hz"), dict)
        ]
        _centre_freq = round(sum(_centre_freqs) / len(_centre_freqs)) if _centre_freqs else "unknown"
        _total_bw = sum(
            s["total_bw_hz"]["value"]
            for s in _spw_list
            if isinstance(s.get("total_bw_hz"), dict)
        )
        # dish_diameter_m lives inside the per-antenna list, not at top level
        _first_ant = (ants.get("antennas") or [{}])[0]
        _dish_m = _first_ant.get("diameter_m", {}).get("value", "unknown") if isinstance(_first_ant.get("diameter_m"), dict) else _first_ant.get("diameter_m", "unknown")

        user_content = "\n".join([
            "## IMAGING_PIPELINE — fill the tclean template",
            f"MS path: {wf['ms_path']}",
            f"Workflow config: {json.dumps(wf_config)}",
            "",
            "## Observing summary",
            f"Telescope: {obs.get('telescope_name', {}).get('value', 'unknown')}",
            f"Centre freq (Hz): {_centre_freq}",
            f"Bandwidth (Hz): {_total_bw}",
            f"Max baseline (m): {baselines.get('max_baseline_m', {}).get('value', 'unknown')}",
            f"Dish diameter (m): {_dish_m}",
            f"N antennas: {ants.get('n_antennas', 'unknown')}",
            "",
            "## Target fields",
            "```json",
            json.dumps([{
                "field_id": f.get("field_id"),
                "name": f.get("name"),
                "ra_deg": f.get("ra_j2000_deg", {}).get("value"),
                "dec_deg": f.get("dec_j2000_deg", {}).get("value"),
            } for f in target_fields], indent=2),
            "```",
        ])

        system_prompt = load_system_prompt(self.skills_path, Stage.IMAGING_PIPELINE)
        messages = [
            {"role": "system", "content": system_prompt + "\n\n" + _JSON_INSTRUCTION_IMAGING},
            {"role": "user", "content": user_content},
        ]

        decision, _model = await self._llm_call_with_retry_and_tools(
            workflow_id, Stage.IMAGING_PIPELINE, messages
        )
        if decision is None:
            return

        casa_script = decision.get("casa_script") or ""
        if not casa_script.strip():
            log.error("Workflow %d: IMAGING_PIPELINE LLM returned no casa_script", workflow_id)
            self.db.transition(workflow_id, Stage.ERROR)
            return

        await self._run_casa_job(workflow_id, Stage.IMAGING_PIPELINE.value, casa_script)
        if self.db.get_workflow(workflow_id)["stage"] == Stage.ERROR.value:
            return

        # Call ms_image_stats on the pbcor image
        img_metrics = self.db.get_last_job_metrics(workflow_id, Stage.IMAGING_PIPELINE.value)
        pbcor_path = img_metrics.get("pbcor_path", "")
        psf_path = img_metrics.get("psf_path", "")
        if pbcor_path:
            try:
                img_stats = await self.tools.call_tool(
                    "ms_image_stats",
                    {"params": {"image_path": pbcor_path, "psf_path": psf_path or None}},
                )
                log.info("Workflow %d: image stats: %s", workflow_id, img_stats)
                # Store image stats as a plot entry so UI can display them
                job_row = self.db.conn.execute(
                    "SELECT id FROM jobs WHERE workflow_id = ? AND stage = ?"
                    " ORDER BY completed_at DESC LIMIT 1",
                    (workflow_id, Stage.IMAGING_PIPELINE.value),
                ).fetchone()
                if job_row:
                    self.db.conn.execute(
                        "UPDATE jobs SET plots = ? WHERE id = ?",
                        (json.dumps({"image_stats": img_stats, "fits_path": img_metrics.get("fits_path", "")}), job_row["id"]),
                    )
                    self.db.conn.commit()
            except Exception as exc:
                log.warning("Workflow %d: ms_image_stats failed: %s", workflow_id, exc)

        self.db.transition(workflow_id, Stage.IMAGING_CHECKPOINT)

    def _build_apply_script(self, workflow_id: int) -> str:
        """Fill all CALIBRATION_APPLY placeholders from Phase 1-3 tool outputs."""
        outputs = self._load_all_tool_outputs(workflow_id)

        # All calibrator field IDs — any field with a calibrator role or catalogue match
        fields = outputs.get("ms_field_list", {}).get("data", {}).get("fields", [])
        cal_ids = [
            str(f["field_id"])
            for f in fields
            if f.get("calibrator_role", {}).get("value")
            or f.get("calibrator_match", {}).get("value")
        ]
        cal_fields = ",".join(cal_ids) if cal_ids else "0"

        n_spw = (
            outputs.get("ms_spectral_window_list", {}).get("data", {}).get("n_spw", 16)
        )
        all_spw = f"0~{n_spw - 1}"

        # Parallel hands only (e.g. XX,YY or RR,LL) for rflag on corrected data
        all_prods = (
            outputs.get("ms_correlator_config", {})
            .get("data", {})
            .get("correlation_products", {})
            .get("value", ["XX", "YY"])
        )
        parallel = [p for p in all_prods if len(p) == 2 and p[0] == p[1]]
        corrstring = ",".join(parallel) if parallel else "XX,YY"

        subs = {
            "{WORKFLOW_ID}": str(workflow_id),
            "{CAL_FIELDS}": cal_fields,
            "{ALL_SPW}": all_spw,
            "{CORRSTRING}": corrstring,
        }
        script = _TEMPLATE_APPLY
        for placeholder, value in subs.items():
            script = script.replace(placeholder, value)
        return script

    # ── Checkpoint handler ─────────────────────────────────────────────────

    async def _handle_checkpoint(self, workflow_id: int) -> None:
        """Write checkpoint row, signal UI, await human decision, transition.

        Used for both HUMAN_CHECKPOINT and CALIBRATION_CHECKPOINT — the stage
        value stored in the checkpoint row distinguishes them for the UI.
        """
        wf = self.db.get_workflow(workflow_id)
        current_stage = Stage(wf["stage"])

        last_decision_row = self.db.conn.execute(
            "SELECT decision FROM llm_decisions WHERE workflow_id = ?"
            " ORDER BY decided_at DESC LIMIT 1",
            (workflow_id,),
        ).fetchone()
        last_llm = (
            json.loads(last_decision_row["decision"]) if last_decision_row else {}
        )
        llm_summary = last_llm.get("summary", "No summary available.")

        checkpoint_id = self.db.create_checkpoint(
            workflow_id, current_stage.value, llm_summary
        )
        log.info(
            "Checkpoint %d (%s) created for workflow %d — awaiting human decision",
            checkpoint_id,
            current_stage.value,
            workflow_id,
        )

        # Timeout config from LLM decision (CALIBRATION_APPLY sets these per checkpoint_questions).
        questions = last_llm.get("checkpoint_questions") or []
        first_q = questions[0] if questions else {}
        timeout_seconds = first_q.get("timeout_seconds", 300)
        timeout_default = first_q.get("timeout_default", Stage.IMAGING_PIPELINE.value)

        self.checkpoint_event.clear()
        try:
            await asyncio.wait_for(
                self.checkpoint_event.wait(), timeout=float(timeout_seconds)
            )
        except asyncio.TimeoutError:
            log.info(
                "Checkpoint %d timed out after %ss — using default route %r",
                checkpoint_id,
                timeout_seconds,
                timeout_default,
            )
            self.db.resolve_checkpoint(checkpoint_id, timeout_default, "timeout")

        checkpoint = self.db.get_latest_checkpoint(workflow_id)
        human_route = (checkpoint or {}).get(
            "human_route", Stage.IMAGING_PIPELINE.value
        )

        route_map = {
            Stage.STOPPED.value: Stage.STOPPED,
            Stage.CALIBRATION_LOOP.value: Stage.CALIBRATION_LOOP,
            Stage.CALIBRATION_PREFLAG.value: Stage.CALIBRATION_PREFLAG,
            Stage.IMAGING_PIPELINE.value: Stage.IMAGING_PIPELINE,
            # Legacy
            "calibration": Stage.CALIBRATION_LOOP,
            "imaging": Stage.IMAGING_PIPELINE,
        }
        next_stage = route_map.get(human_route, Stage.IMAGING_PIPELINE)
        self.db.transition(workflow_id, next_stage)

    # ── CASA job helper ────────────────────────────────────────────────────

    async def _run_casa_job(
        self, workflow_id: int, stage_label: str, script_content: str
    ) -> None:
        """Write script to disk, submit to runner, await completion.

        Transitions to ERROR if the job fails — callers should check for ERROR
        state and return immediately after calling this method.
        """
        script_dir = Path(self.runner.config.jobs_dir) / str(workflow_id)
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / f"{stage_label}.py"
        script_path.write_text(script_content, encoding="utf-8")

        job_id = self.db.create_job(workflow_id, stage_label, str(script_path))
        await self.runner.submit(job_id, str(script_path))

        job = self.db.conn.execute(
            "SELECT status, stderr FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if job and job["status"] == "failed":
            log.error(
                "Job %d (%s) failed — transitioning workflow %d to ERROR\n%s",
                job_id,
                stage_label,
                workflow_id,
                job["stderr"] or "",
            )
            self.db.transition(workflow_id, Stage.ERROR)

    # ── Calibration prompt helpers ─────────────────────────────────────────

    def _get_band(self, workflow_id: int) -> str:
        """Derive band letter from band_centre_ghz in ms_pol_cal_feasibility.

        Returns 'P', 'L', 'S', 'C', 'X', or 'default' if not available.
        """
        for row in self.db.get_tool_outputs(workflow_id, "phase3"):
            if row["tool_name"] == "ms_pol_cal_feasibility":
                try:
                    output = json.loads(row["output_json"])
                    ghz = float(output.get("data", {}).get("band_centre_ghz", 0))
                    if ghz < 1.0:
                        return "P"
                    elif ghz < 2.0:
                        return "L"
                    elif ghz < 4.0:
                        return "S"
                    elif ghz < 8.0:
                        return "C"
                    else:
                        return "X"
                except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
                    pass
        return "default"

    def _get_polcal_verdict(self, workflow_id: int) -> str | None:
        """Read ms_pol_cal_feasibility verdict from stored Phase 3 tool outputs.

        Returns the verdict string (e.g. 'FULL', 'NOT_FEASIBLE') or None if
        the tool output is not available.
        """
        for row in self.db.get_tool_outputs(workflow_id, "phase3"):
            if row["tool_name"] == "ms_pol_cal_feasibility":
                try:
                    output = json.loads(row["output_json"])
                    return output.get("data", {}).get("verdict")
                except (json.JSONDecodeError, AttributeError):
                    pass
        return None

    def _load_all_tool_outputs(
        self, workflow_id: int, tools: set[str] | None = None
    ) -> dict[str, dict]:
        """Load Phase 1-3 tool outputs keyed by tool name.

        If tools is given, only those tool names are returned — keeps calibration
        prompts focused on the placeholders each stage actually needs to fill.
        """
        outputs: dict[str, dict] = {}
        for phase in ("phase1", "phase2", "phase3"):
            for row in self.db.get_tool_outputs(workflow_id, phase):
                if tools is not None and row["tool_name"] not in tools:
                    continue
                try:
                    outputs[row["tool_name"]] = json.loads(row["output_json"])
                except (json.JSONDecodeError, KeyError):
                    pass
        return outputs

    def _get_previous_preflag_script(self, workflow_id: int) -> str:
        """Read the script content from the most recent completed CALIBRATION_PREFLAG job."""
        row = self.db.conn.execute(
            "SELECT script_path FROM jobs"
            " WHERE workflow_id = ? AND stage = ? AND status = 'done'"
            " ORDER BY id DESC LIMIT 1",
            (workflow_id, Stage.CALIBRATION_PREFLAG.value),
        ).fetchone()
        if row and row["script_path"]:
            try:
                with open(row["script_path"]) as f:
                    return f.read()
            except OSError:
                pass
        return ""

    # ── Parsing helpers ────────────────────────────────────────────────────

    def _parse_decision(self, raw: str) -> dict:
        """Extract and validate the LLM JSON decision."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(ln for ln in lines if not ln.strip().startswith("```"))

        try:
            decision = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM response is not valid JSON: {exc}\n---\n{raw}"
            ) from exc

        missing = _DECISION_SCHEMA_KEYS - decision.keys()
        if missing:
            raise ValueError(f"LLM decision missing required keys: {missing}")

        return decision

    def _extract_content(self, response: dict) -> str:
        """Pull the assistant message text out of an OpenAI response dict."""
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"Unexpected LLM response shape: {exc}\n{response}"
            ) from exc

    def _format_tool_outputs(self, outputs: dict[str, dict], phase: int) -> str:
        """Format MCP tool outputs into a structured prompt string."""
        lines = [f"## Phase {phase} tool outputs\n"]
        for tool_name, data in outputs.items():
            lines.append(f"### {tool_name}")
            lines.append("```json")
            lines.append(json.dumps(data, indent=2))
            lines.append("```\n")
        return "\n".join(lines)
