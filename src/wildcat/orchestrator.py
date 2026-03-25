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
import re
from dataclasses import dataclass
from pathlib import Path

from wildcat.llm import LLMBackend
from wildcat.runner import CASARunner
from wildcat.skills import load_system_prompt
from wildcat.state import Stage, StateDB
from wildcat.tools import MSInspectClient

log = logging.getLogger(__name__)

# Required keys per stage — APPLY has a different schema (checkpoint_questions, no next_stage)
_DECISION_SCHEMA_KEYS = {"next_stage", "summary", "reasoning"}  # default
_DECISION_SCHEMA_KEYS_BY_STAGE: dict[str, set[str]] = {
    Stage.CALIBRATION_APPLY.value: {"summary", "checkpoint_questions"},
}

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

# CALIBRATION_APPLY: metrics summary + checkpoint questions only.
# The CASA script is filled deterministically — LLM only interprets the results for the human.
_JSON_INSTRUCTION_APPLY = """
The CASA calibration script has already run. You are given the WILDCAT_METRICS output.
Respond with a single JSON object and no other text. Schema:
{
  "summary": "<2-5 sentences summarising calibration quality using the actual metric values>",
  "checkpoint_questions": [
    {
      "id": "calibration_done",
      "finding": "<one sentence with actual values: bp_flagged_frac=X, gain_flagged_frac=Y, post_cal_flag_frac=Z, n_antennas_lost=N>",
      "severity": "<info|warning|critical>",
      "question": "Calibration complete. Proceed to imaging or loop back for another calibration pass?",
      "options": ["proceed", "loop_back", "exit"]
    }
  ]
}
Use the real metric values from WILDCAT_METRICS. Severity is info if all fractions < 0.20, warning if any 0.20-0.40, critical if any > 0.40.
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
Choose POLTYPE_DTERM using guidance from 09-polcal-execution.md (Df vs Df+QU table).
POL_FREQ_LO and POL_FREQ_HI should bracket the usable PA nodes — for 3C48 at S-band use 2.0 and 9.0.
"""

_JSON_INSTRUCTIONS_BY_STAGE: dict[Stage, str] = {
    Stage.CALIBRATION_PREFLAG: _JSON_INSTRUCTION_PREFLAG,
    Stage.POLCAL_SOLVE:        _JSON_INSTRUCTION_POLCAL,
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
    "calibration_done": {"proceed": None, "loop_back": None, "exit": None},
}

# ── CASA script templates ────────────────────────────────────────────────────
# Each template has {PLACEHOLDER} sections the LLM fills from Phase 1-3 outputs.
# Sections marked [DETERMINISTIC] must be copied verbatim.

_TEMPLATE_PREFLAG = """\
# wildcat CALIBRATION_PREFLAG — split calibrators, rflag, report flag fractions
# LLM: fill all {PLACEHOLDER} values from Phase 1-3 tool outputs

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
    "mode='tfcrop' field='" + cal_fields + "' spw='" + all_spw + "' correlation='ABS_" + corrstring + "' ntime='scan' combinescans=False datacolumn='data' timecutoff=4.0 freqcutoff=4.0 timefit='line' freqfit='poly' maxnpieces=7 flagdimension='freqtime' extendflags=False",
    "mode='extend' field='" + cal_fields + "' spw='" + all_spw + "' growtime=50.0 growfreq=90.0",
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
    "mode='rflag' field='" + cal_field + "' spw='" + all_spw + "' correlation='ABS_" + corrstring + "' ntime='scan' combinescans=False datacolumn='corrected' winsize=3 timedevscale=4.0 freqdevscale=4.0 extendflags=False",
    "mode='extend' field='" + cal_field + "' spw='" + all_spw + "' growtime=50.0 growfreq=90.0",
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
# See 07-calibration-execution.md for K/G/B guidance, 09-polcal-execution.md for Df vs Df+QU.

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
poltype_dterm     = '{POLTYPE_DTERM}'    # 'Df' or 'Df+QU' — see 09-polcal-execution.md

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

# [DETERMINISTIC] D-term leakage (Df or Df+QU per 09-polcal-execution.md decision table)
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

_TEMPLATES_BY_STAGE: dict[Stage, str] = {
    Stage.CALIBRATION_PREFLAG: _TEMPLATE_PREFLAG,
    Stage.POLCAL_SOLVE:        _TEMPLATE_POLCAL,
}

# Valid next_stage values per calibration stage (LLM-driven stages only)
_VALID_CAL_NEXT_STAGES: dict[Stage, set[str]] = {
    Stage.CALIBRATION_PREFLAG: {Stage.CALIBRATION_PREFLAG.value, Stage.CALIBRATION_SOLVE.value},
    Stage.POLCAL_SOLVE:        {Stage.CALIBRATION_APPLY.value},
}

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
# (bp_flagged_frac_max, gain_flagged_frac_max)
_SOLVE_THRESHOLDS_BY_BAND: dict[str, tuple[float, float]] = {
    "P":       (0.60, 0.60),  # 230–470 MHz — endemic RFI environment
    "L":       (0.60, 0.60),  # 1–2 GHz
    "S":       (0.40, 0.40),  # 2–4 GHz
    "C":       (0.20, 0.15),  # 4–8 GHz
    "X":       (0.20, 0.15),  # 8–12 GHz
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

# ── Context assembly specification ────────────────────────────────────────
# Each calibration stage declares which context objects the LLM receives.
# The orchestrator assembles them from DB, respecting token budget.
#
# source types:
#   "tool_outputs" — Phase 1-3 tool outputs, filtered by tool names
#   "metrics"      — WILDCAT_METRICS from a completed job, filtered by stage
#   "prev_script"  — script content from a completed job, filtered by stage
#   "template"     — CASA script template for the LLM to fill
#   "workflow_config" — current workflow config JSON
#
# condition: "first_iteration" | "reentry" | "from_solve" | None (always)

@dataclass
class ContextSpec:
    source: str
    label: str  # heading in the assembled prompt
    stage_filter: str | None = None  # for metrics/prev_script: which job stage
    tools: tuple[str, ...] | None = None  # for tool_outputs: which tools
    template_key: str | None = None  # for template: key in _TEMPLATES_BY_STAGE
    condition: str | None = None  # when to include this spec


_STAGE_CONTEXT: dict[Stage, list[ContextSpec]] = {
    Stage.CALIBRATION_PREFLAG: [
        # First iteration: tool outputs + template
        ContextSpec(
            source="tool_outputs", label="Phase 1-3 tool outputs",
            tools=("ms_field_list", "ms_spectral_window_list", "ms_correlator_config"),
            condition="first_iteration",
        ),
        ContextSpec(
            source="template", label="Script template",
            template_key="CALIBRATION_PREFLAG",
            condition="first_iteration",
        ),
        # Re-entry: previous script + PREFLAG metrics (not SOLVE metrics)
        ContextSpec(
            source="prev_script", label="Script from previous PREFLAG run",
            stage_filter="CALIBRATION_PREFLAG",
            condition="reentry",
        ),
        ContextSpec(
            source="metrics", label="Previous PREFLAG outcome (WILDCAT_METRICS)",
            stage_filter="CALIBRATION_PREFLAG",
            condition="reentry",
        ),
        # Re-entry from SOLVE failure: include SOLVE metrics so LLM knows why it's back
        ContextSpec(
            source="metrics", label="CALIBRATION_SOLVE result (thresholds not met)",
            stage_filter="CALIBRATION_SOLVE",
            condition="from_solve",
        ),
    ],
    Stage.POLCAL_SOLVE: [
        ContextSpec(
            source="tool_outputs", label="Phase 1-3 tool outputs",
            tools=(
                "ms_observation_info", "ms_field_list", "ms_scan_list",
                "ms_scan_intent_summary", "ms_spectral_window_list",
                "ms_correlator_config", "ms_refant", "ms_pol_cal_feasibility",
            ),
        ),
        ContextSpec(
            source="metrics", label="Previous PREFLAG outcome",
            stage_filter="CALIBRATION_PREFLAG",
        ),
        ContextSpec(
            source="template", label="Script template",
            template_key="POLCAL_SOLVE",
        ),
    ],
    Stage.CALIBRATION_APPLY: [
        ContextSpec(
            source="metrics", label="CALIBRATION_SOLVE metrics",
            stage_filter="CALIBRATION_SOLVE",
        ),
        ContextSpec(
            source="metrics", label="CALIBRATION_APPLY metrics",
            stage_filter="CALIBRATION_APPLY",
        ),
    ],
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

                elif stage == Stage.CALIBRATION_SOLVE:
                    await self._handle_solve_stage(workflow_id)

                elif stage == Stage.CALIBRATION_APPLY:
                    await self._handle_apply_stage(workflow_id)

                elif stage in (
                    Stage.CALIBRATION_PREFLAG,
                    Stage.POLCAL_SOLVE,
                ):
                    await self._handle_calibration_stage(workflow_id, stage)

                elif stage == Stage.CALIBRATION_CHECKPOINT:
                    await self._handle_checkpoint(workflow_id)

                elif stage == Stage.CALIBRATION_LOOP:
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
                    stage, workflow_id,
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
                "\n\n## Workflow config\n"
                "```json\n" + json.dumps(wf_config, indent=2) + "\n```\n"
            )
        json_instruction = _JSON_INSTRUCTION_PHASE3 if phase == 3 else _JSON_INSTRUCTION
        messages = [
            {"role": "system", "content": system_prompt + "\n\n" + json_instruction},
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
            log.error("LLM returned unknown next_stage %r — entering ERROR", next_stage_str)
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
                workflow_id, phase, fallback.value,
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
                workflow_id, verdict,
            )

        # Rule 2: any antenna fully flagged → enable aggressive flagging
        per_antenna = (
            all_outputs.get("ms_antenna_flag_fraction", {})
            .get("data", {})
            .get("per_antenna", [])
        )
        fully_flagged = [
            a["antenna_name"] for a in per_antenna
            if a.get("flag_fraction", {}).get("value", 0) >= 1.0
        ]
        if fully_flagged and not wf_config.get("aggressive_flagging", False):
            wf_config["aggressive_flagging"] = True
            overrides["aggressive_flagging"] = True
            log.warning(
                "Workflow %d: aggressive_flagging=True — antennas at 100%%: %s",
                workflow_id, fully_flagged,
            )

        if overrides:
            self.db.set_workflow_config(workflow_id, wf_config)
            log.info("Workflow %d: deterministic config overrides applied: %s", workflow_id, overrides)
        return overrides

    # ── Calibration stage handler ──────────────────────────────────────────

    async def _handle_calibration_stage(self, workflow_id: int, stage: Stage) -> None:
        """Assemble context → LLM (with retry) → run CASA → transition.

        Handles LLM-driven calibration stages: CALIBRATION_PREFLAG, POLCAL_SOLVE.
        CALIBRATION_SOLVE and CALIBRATION_APPLY are handled by their own deterministic handlers.
        """
        if self.stop_event.is_set():
            self.db.transition(workflow_id, Stage.STOPPED)
            return

        wf = self.db.get_workflow(workflow_id)
        wf_config = self.db.get_workflow_config(workflow_id)
        preflag_iterations = wf_config.get("_preflag_iterations", 0)

        # Increment counter at entry so it counts all PREFLAG runs, regardless of
        # whether re-entry was driven by the LLM, the flag cap, or SOLVE threshold failure.
        if stage == Stage.CALIBRATION_PREFLAG:
            preflag_iterations += 1
            wf_config["_preflag_iterations"] = preflag_iterations
            self.db.set_workflow_config(workflow_id, wf_config)

        # Enforce preflag iteration cap
        if stage == Stage.CALIBRATION_PREFLAG and preflag_iterations > _PREFLAG_MAX_ITERATIONS:
            log.warning(
                "CALIBRATION_PREFLAG hit cap (%d iterations) for workflow %d — escalating",
                _PREFLAG_MAX_ITERATIONS, workflow_id,
            )
            wf_config["_preflag_flag_warning"] = True
            self.db.set_workflow_config(workflow_id, wf_config)
            self.db.transition(workflow_id, Stage.CALIBRATION_CHECKPOINT)
            return

        # 1. Assemble context from the stage's declared context specs
        user_content = self._assemble_context(workflow_id, stage, wf_config, wf["ms_path"], preflag_iterations)

        system_prompt = load_system_prompt(self.skills_path, stage)
        instruction = _JSON_INSTRUCTIONS_BY_STAGE[stage]
        # /no_think disables Qwen3 chain-of-thought for structured JSON outputs
        messages = [
            {"role": "system", "content": system_prompt + "\n\n" + instruction + "\n\n/no_think"},
            {"role": "user",   "content": user_content},
        ]

        if self.stop_event.is_set():
            self.db.transition(workflow_id, Stage.STOPPED)
            return

        # 2. LLM call with retry
        decision, model = await self._llm_call_with_retry(
            workflow_id, stage, messages
        )
        if decision is None:
            # All retries exhausted — _llm_call_with_retry already set ERROR
            return

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
            if self.db.get_workflow(workflow_id)["stage"] == Stage.ERROR.value:
                return

        # 4b. Flag fraction cap: if PREFLAG wants to loop but data is too heavily flagged,
        #     force-advance to CALIBRATION_SOLVE rather than destroying more data.
        if stage == Stage.CALIBRATION_PREFLAG and next_stage_str == Stage.CALIBRATION_PREFLAG.value:
            current_metrics = self.db.get_last_job_metrics(workflow_id, Stage.CALIBRATION_PREFLAG.value)
            flag_frac = current_metrics.get("overall_flag_frac", 0.0)
            if isinstance(flag_frac, (int, float)) and flag_frac >= _PREFLAG_FLAG_CAP:
                log.warning(
                    "Workflow %d: overall_flag_frac=%.3f >= cap %.2f — forcing CALIBRATION_SOLVE",
                    workflow_id, flag_frac, _PREFLAG_FLAG_CAP,
                )
                next_stage_str = Stage.CALIBRATION_SOLVE.value

        # 4c. Polcal routing: when CALIBRATION_PREFLAG advances to CALIBRATION_SOLVE,
        #     redirect to POLCAL_SOLVE if polcal=True and feasibility verdict is viable.
        if stage == Stage.CALIBRATION_PREFLAG and next_stage_str == Stage.CALIBRATION_SOLVE.value:
            if wf_config.get("polcal"):
                polcal_verdict = self._get_polcal_verdict(workflow_id)
                if polcal_verdict in _POLCAL_VIABLE_VERDICTS:
                    log.info(
                        "Workflow %d: polcal=True, verdict=%s — routing to POLCAL_SOLVE",
                        workflow_id, polcal_verdict,
                    )
                    next_stage_str = Stage.POLCAL_SOLVE.value
                else:
                    log.info(
                        "Workflow %d: polcal=True but verdict=%r — using CALIBRATION_SOLVE",
                        workflow_id, polcal_verdict,
                    )

        # 5. Transition
        self.db.transition(workflow_id, Stage(next_stage_str))

    # ── Deterministic CALIBRATION_SOLVE ───────────────────────────────────

    # Maps the tool-output flux_standard identifier to the CASA-accepted string.
    _FLUX_STANDARD_MAP: dict[str, str] = {
        "Perley-Butler-2017":     "Perley-Butler 2017",
        "Perley-Butler-2013":     "Perley-Butler 2013",
        "Scaife-Heald-2012":      "Scaife-Heald 2012",
        "Stevens-Reynolds-2016":  "Stevens-Reynolds 2016",
    }

    # ── Context assembly ────────────────────────────────────────────────

    def _assemble_context(
        self,
        workflow_id: int,
        stage: Stage,
        wf_config: dict,
        ms_path: str,
        preflag_iterations: int,
    ) -> str:
        """Build the LLM user content from the stage's declared context specs.

        Each stage defines what context it needs in _STAGE_CONTEXT. This method
        evaluates conditions, fetches data from the DB, and assembles it into a
        single string — respecting the token budget.
        """
        specs = _STAGE_CONTEXT.get(stage, [])
        is_reentry = stage == Stage.CALIBRATION_PREFLAG and preflag_iterations > 1
        # Detect if we're coming back from a SOLVE failure
        from_solve = is_reentry and bool(
            self.db.get_last_job_metrics(workflow_id, Stage.CALIBRATION_SOLVE.value)
        )

        lines: list[str] = []
        lines.append(f"## Calibration stage: {stage.value}")
        lines.append(f"Measurement set: {ms_path}")
        lines.append(f"Workflow config: {json.dumps(wf_config)}")
        if preflag_iterations:
            lines.append(f"Preflag iterations: {preflag_iterations} / {_PREFLAG_MAX_ITERATIONS}")

        token_estimate = sum(len(l) for l in lines) // 4

        for spec in specs:
            # Evaluate condition
            if spec.condition == "first_iteration" and is_reentry:
                continue
            if spec.condition == "reentry" and not is_reentry:
                continue
            if spec.condition == "from_solve" and not from_solve:
                continue

            section = self._fetch_context_section(workflow_id, spec, wf_config)
            if not section:
                continue

            section_tokens = len(section) // 4
            if token_estimate + section_tokens > self.max_user_tokens:
                log.warning(
                    "Workflow %d: context budget exceeded at section %r (%d tokens used, "
                    "%d section, %d limit) — truncating",
                    workflow_id, spec.label, token_estimate, section_tokens, self.max_user_tokens,
                )
                break

            lines.append(f"\n## {spec.label}")
            lines.append(section)
            token_estimate += section_tokens

        if is_reentry:
            lines.append(
                "\nAdjust the flagging parameters based on the outcome above and return a new script. "
                "Respond using the JSON schema specified in the system prompt."
            )

        log.info(
            "Workflow %d: assembled %d tokens of context for %s",
            workflow_id, token_estimate, stage.value,
        )
        return "\n".join(lines)

    def _fetch_context_section(
        self, workflow_id: int, spec: ContextSpec, wf_config: dict
    ) -> str:
        """Fetch a single context section from the DB based on the spec."""
        if spec.source == "tool_outputs":
            outputs = self._load_all_tool_outputs(workflow_id, tools=set(spec.tools) if spec.tools else None)
            if not outputs:
                return ""
            parts = []
            for tool_name, data in outputs.items():
                parts.append(f"### {tool_name}")
                parts.append("```json")
                parts.append(json.dumps(data, indent=2))
                parts.append("```")
            return "\n".join(parts)

        if spec.source == "metrics":
            metrics = self.db.get_last_job_metrics(workflow_id, spec.stage_filter)
            if not metrics:
                return ""
            return "```json\n" + json.dumps(metrics, indent=2) + "\n```"

        if spec.source == "prev_script":
            return self._get_previous_preflag_script(workflow_id) or ""

        if spec.source == "template":
            stage_key = Stage(spec.template_key) if spec.template_key else None
            tmpl = _TEMPLATES_BY_STAGE.get(stage_key, "") if stage_key else ""
            if not tmpl:
                return ""
            tmpl = tmpl.replace("{WORKFLOW_ID}", str(workflow_id))
            if stage_key == Stage.CALIBRATION_PREFLAG:
                tmpl = self._prefill_preflag_template(workflow_id, tmpl)
            return (
                "Complete the script by replacing ALL {PLACEHOLDER} values.\n"
                "Do NOT modify sections marked [DETERMINISTIC].\n"
                "```python\n" + tmpl + "\n```"
            )

        if spec.source == "workflow_config":
            return "```json\n" + json.dumps(wf_config, indent=2) + "\n```"

        log.warning("Unknown context source %r in spec %r", spec.source, spec.label)
        return ""

    # ── LLM retry ─────────────────────────────────────────────────────────

    async def _llm_call_with_retry(
        self, workflow_id: int, stage: Stage, messages: list[dict]
    ) -> tuple[dict | None, str]:
        """Call the LLM and retry on empty/invalid responses.

        Returns (decision_dict, model_str) on success, or (None, "") if all
        retries exhausted (ERROR state is set before returning).
        """
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self.llm.complete(messages)
                raw = self._extract_content(response)
                if not raw or not raw.strip():
                    raise ValueError("LLM returned empty response")
                decision = self._parse_decision(raw, stage)
                model = response.get("model", "unknown")
                self.db.save_llm_decision(
                    workflow_id, stage.value, json.dumps(decision), model
                )
                if attempt > 1:
                    log.info(
                        "Workflow %d: LLM succeeded on attempt %d/%d for %s",
                        workflow_id, attempt, self.max_retries, stage.value,
                    )
                return decision, model
            except (ValueError, KeyError) as exc:
                last_error = str(exc)
                log.warning(
                    "Workflow %d: LLM attempt %d/%d failed for %s — %s",
                    workflow_id, attempt, self.max_retries, stage.value, last_error,
                )
                # Store the failed attempt for observability
                self.db.save_llm_decision(
                    workflow_id, stage.value,
                    json.dumps({"_retry_error": last_error, "_attempt": attempt}),
                    "retry",
                )

        log.error(
            "Workflow %d: all %d LLM retries exhausted for %s — entering ERROR. Last: %s",
            workflow_id, self.max_retries, stage.value, last_error,
        )
        self.db.transition(workflow_id, Stage.ERROR)
        return None, ""

    # ── Deterministic CALIBRATION_SOLVE ───────────────────────────────────

    async def _handle_solve_stage(self, workflow_id: int) -> None:
        """CALIBRATION_SOLVE: fill template deterministically, run CASA, apply threshold rule.

        No LLM call. All placeholders are derived from Phase 1-3 tool outputs.
        Threshold rule: bp_flagged_frac < 0.20 AND gain_flagged_frac < 0.15 → CALIBRATION_APPLY,
        otherwise → CALIBRATION_PREFLAG for another flagging pass.
        """
        if self.stop_event.is_set():
            self.db.transition(workflow_id, Stage.STOPPED)
            return

        script = self._build_solve_script(workflow_id)
        await self._run_casa_job(workflow_id, Stage.CALIBRATION_SOLVE.value, script)
        if self.db.get_workflow(workflow_id)["stage"] == Stage.ERROR.value:
            return

        metrics = self.db.get_last_job_metrics(workflow_id, Stage.CALIBRATION_SOLVE.value)
        bp_frac   = float(metrics.get("bp_flagged_frac",   1.0))
        gain_frac = float(metrics.get("gain_flagged_frac", 1.0))

        band = self._get_band(workflow_id)
        bp_max, gain_max = _SOLVE_THRESHOLDS_BY_BAND.get(band, _SOLVE_THRESHOLDS_BY_BAND["default"])
        log.info("Workflow %d: band=%s thresholds bp<%.2f gain<%.2f (actual bp=%.3f gain=%.3f)",
                 workflow_id, band, bp_max, gain_max, bp_frac, gain_frac)

        if bp_frac < bp_max and gain_frac < gain_max:
            log.info(
                "Workflow %d: CALIBRATION_SOLVE passed (bp=%.3f gain=%.3f) → CALIBRATION_APPLY",
                workflow_id, bp_frac, gain_frac,
            )
            self.db.transition(workflow_id, Stage.CALIBRATION_APPLY)
        else:
            wf_config = self.db.get_workflow_config(workflow_id)
            preflag_iterations = wf_config.get("_preflag_iterations", 0)
            if preflag_iterations >= _PREFLAG_MAX_ITERATIONS:
                log.warning(
                    "Workflow %d: CALIBRATION_SOLVE thresholds not met (bp=%.3f gain=%.3f) "
                    "and PREFLAG cap reached (%d) — escalating to CALIBRATION_APPLY",
                    workflow_id, bp_frac, gain_frac, preflag_iterations,
                )
                self.db.transition(workflow_id, Stage.CALIBRATION_APPLY)
            else:
                log.warning(
                    "Workflow %d: CALIBRATION_SOLVE thresholds not met (bp=%.3f gain=%.3f) → CALIBRATION_PREFLAG",
                    workflow_id, bp_frac, gain_frac,
                )
                self.db.transition(workflow_id, Stage.CALIBRATION_PREFLAG)

    def _build_solve_script(self, workflow_id: int) -> str:
        """Fill all CALIBRATION_SOLVE placeholders from Phase 1-3 tool outputs."""
        outputs = self._load_all_tool_outputs(workflow_id)

        flux_field_id = self._solve_flux_field_id(outputs)
        phase_field_id = self._solve_phase_field_id(outputs, flux_field_id)

        n_spw = (
            outputs.get("ms_spectral_window_list", {})
            .get("data", {}).get("n_spw", 16)
        )
        all_spw = f"0~{n_spw - 1}"

        int_time_s = (
            outputs.get("ms_correlator_config", {})
            .get("data", {}).get("dump_time_s", {}).get("value", 2.0)
        )

        subs = {
            "{WORKFLOW_ID}":    str(workflow_id),
            "{FLUX_FIELD}":     str(flux_field_id),
            "{BP_FIELD}":       str(flux_field_id),
            "{DELAY_FIELD}":    str(flux_field_id),
            "{PHASE_FIELD}":    str(phase_field_id),
            "{PHASE_SCAN_IDS}": self._solve_cal_scan_ids(outputs, str(phase_field_id)),
            "{REFANT}":         self._solve_best_refant(outputs),
            "{ALL_SPW}":        all_spw,
            "{FLUX_STANDARD}":  self._solve_flux_standard(outputs, flux_field_id),
            "{MINBLPERANT}":    "4",
            "{INT_TIME_S}":     f"{float(int_time_s):.2f}",
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
        ids = [str(s["scan_number"]) for s in scans if str(s.get("field_id")) == field_id]
        return ",".join(ids) if ids else "1"

    def _solve_best_refant(self, outputs: dict) -> str:
        """Return the antenna with the lowest flag fraction (excluding 100% flagged)."""
        per_ant = (
            outputs.get("ms_antenna_flag_fraction", {})
            .get("data", {}).get("per_antenna", [])
        )
        valid = [a for a in per_ant if (a.get("flag_fraction", {}).get("value") or 1.0) < 1.0]
        if not valid:
            return "ea01"
        return min(valid, key=lambda a: a["flag_fraction"]["value"])["antenna_name"]

    def _solve_flux_standard(self, outputs: dict, flux_field_id: int) -> str:
        """Map the tool-output flux_standard to the CASA-accepted string."""
        fields = outputs.get("ms_field_list", {}).get("data", {}).get("fields", [])
        for f in fields:
            if f["field_id"] == flux_field_id:
                raw = f.get("flux_standard", {}).get("value") or ""
                return self._FLUX_STANDARD_MAP.get(raw, raw)
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

        # Read metrics from both stages for the LLM summary
        apply_metrics = self.db.get_last_job_metrics(workflow_id, Stage.CALIBRATION_APPLY.value)
        solve_metrics = self.db.get_last_job_metrics(workflow_id, Stage.CALIBRATION_SOLVE.value)

        user_content = (
            "## CALIBRATION_SOLVE metrics\n```json\n"
            + json.dumps(solve_metrics, indent=2)
            + "\n```\n\n## CALIBRATION_APPLY metrics\n```json\n"
            + json.dumps(apply_metrics, indent=2)
            + "\n```"
        )
        system_prompt = load_system_prompt(self.skills_path, Stage.CALIBRATION_APPLY)
        messages = [
            {"role": "system", "content": system_prompt + "\n\n" + _JSON_INSTRUCTION_APPLY + "\n\n/no_think"},
            {"role": "user",   "content": user_content},
        ]

        decision, _model = await self._llm_call_with_retry(
            workflow_id, Stage.CALIBRATION_APPLY, messages
        )
        if decision is None:
            return

        self.db.transition(workflow_id, Stage.CALIBRATION_CHECKPOINT)

    def _build_apply_script(self, workflow_id: int) -> str:
        """Fill all CALIBRATION_APPLY placeholders from Phase 1-3 tool outputs."""
        outputs = self._load_all_tool_outputs(workflow_id)

        # All calibrator field IDs — any field with a calibrator role or catalogue match
        fields = outputs.get("ms_field_list", {}).get("data", {}).get("fields", [])
        cal_ids = [
            str(f["field_id"]) for f in fields
            if f.get("calibrator_role", {}).get("value")
            or f.get("calibrator_match", {}).get("value")
        ]
        cal_fields = ",".join(cal_ids) if cal_ids else "0"

        n_spw = (
            outputs.get("ms_spectral_window_list", {})
            .get("data", {}).get("n_spw", 16)
        )
        all_spw = f"0~{n_spw - 1}"

        # Parallel hands only (e.g. XX,YY or RR,LL) for rflag on corrected data
        all_prods = (
            outputs.get("ms_correlator_config", {})
            .get("data", {}).get("correlation_products", {}).get("value", ["XX", "YY"])
        )
        parallel = [p for p in all_prods if len(p) == 2 and p[0] == p[1]]
        corrstring = ",".join(parallel) if parallel else "XX,YY"

        subs = {
            "{WORKFLOW_ID}": str(workflow_id),
            "{CAL_FIELDS}": cal_fields,
            "{ALL_SPW}":    all_spw,
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
                job_id, stage_label, workflow_id, job["stderr"] or "",
            )
            self.db.transition(workflow_id, Stage.ERROR)

    # ── Calibration prompt helpers ─────────────────────────────────────────

    def _get_band(self, workflow_id: int) -> str:
        """Derive band letter from band_centre_ghz in ms_pol_cal_feasibility."""
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

    def _prefill_preflag_template(self, workflow_id: int, tmpl: str) -> str:
        """Fill deterministic PREFLAG placeholders from stored tool outputs.

        Fills VIS, CAL_FIELDS, ALL_SPW, CORRSTRING, SPW_DISCARD_THRESHOLD so
        the LLM cannot misidentify target fields or misconfigure the script.
        """
        outputs = self._load_all_tool_outputs(workflow_id)

        fields = outputs.get("ms_field_list", {}).get("data", {}).get("fields", [])
        cal_ids = [
            str(f["field_id"]) for f in fields
            if f.get("calibrator_role", {}).get("value")
            or f.get("calibrator_match", {}).get("value")
        ]
        cal_fields = ",".join(cal_ids) if cal_ids else "0"

        n_spw = outputs.get("ms_spectral_window_list", {}).get("data", {}).get("n_spw", 16)
        all_spw = f"0~{n_spw - 1}"

        corr_products = (
            outputs.get("ms_correlator_config", {})
            .get("data", {}).get("corr_products", [])
        )
        parallel = [c for c in corr_products if c in ("XX", "YY", "RR", "LL")]
        corrstring = ",".join(parallel) if parallel else "XX,YY"

        wf = self.db.get_workflow(workflow_id)
        return (
            tmpl
            .replace("{VIS}", wf["ms_path"])
            .replace("{CAL_FIELDS}", cal_fields)
            .replace("{ALL_SPW}", all_spw)
            .replace("{CORRSTRING}", corrstring)
            .replace("{SPW_DISCARD_THRESHOLD}", str(_SPW_DISCARD_THRESHOLD))
        )

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

    def _parse_decision(self, raw: str, stage: Stage | None = None) -> dict:
        """Extract and validate the LLM JSON decision."""
        # Strip Qwen3 thinking blocks (<think>...</think>) before parsing
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(l for l in lines if not l.strip().startswith("```"))

        try:
            decision = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM response is not valid JSON: {exc}\n---\n{raw}") from exc

        required = _DECISION_SCHEMA_KEYS_BY_STAGE.get(
            stage.value if stage else "", _DECISION_SCHEMA_KEYS
        )
        missing = required - decision.keys()
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
