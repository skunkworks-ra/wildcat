# wildcat — Design Document
## Agentic Radio Interferometry Reduction Pipeline

**Status:** CALIBRATION_PREFLAG→SOLVE loop fixed; PREFLAG re-entry context minimised; first full run in progress
**Last revised:** 2026-03-25 (rev 9)
**Scope:** Philosophy, state machine, LLM contract, deterministic rules, human checkpoint.
**Implementation detail** (DB schema, data flow, module map): see `ARCHITECTURE.md`.

---

## 1. Philosophy

### 1.1 The Problem

Radio interferometric reduction requires running CASA scripts that take minutes to
hours. The workflow is not linear — after inspection a human expert must decide
whether the calibration is good enough to proceed to imaging, or whether to loop
back. This judgment cannot be automated; it requires scientific assessment of
calibration metrics.

The goal is to automate everything *except* that final judgment, and to make the
handoff to the human as frictionless as possible.

### 1.2 Key Design Decisions

- **Stateless LLM per decision point.** The LLM is invoked once per stage with
  exactly the context needed for that decision. SQLite is the durable memory
  between calls. This tolerates arbitrarily long CASA jobs without timeouts.
- **Minimal context, sufficient information.** Each LLM call receives the smallest
  prompt that still enables an autonomous decision. No accumulated history, no
  redundant tool outputs on re-entry. See §2.3.
- **Deterministic config over LLM judgment.** Data quality decisions with
  measurable thresholds (polcal feasibility, antenna flagging) are made by
  the orchestrator from tool outputs — not by the LLM and not by the human.
  The LLM advances the pipeline; the orchestrator enforces constraints.
- **Direct asyncio subprocess await.** `CASARunner.submit()` is awaited directly.
  No sentinel files, no watchdog — these caused silent failures when pre-existing
  `.done` files prevented inotify events from firing.
- **CASA scripts run via python3.** `casatasks` is a Python package. Scripts run
  as `python3 script.py`. No standalone `casa` binary required.
- **Single container.** llama.cpp + CUDA built from source inside the image.
  ms-inspect volume-mounted and pip-installed at entrypoint.
- **Explicit pipeline start gate.** Pipeline does not start on container boot.
  Priority: `--autostart` flag > `WILDCAT_AUTOSTART` env > UI button at `/start`.

---

## 2. Workflow

### 2.1 State Machine

```
IDLE → PHASE1_RUNNING → PHASE2_RUNNING → PHASE3_RUNNING
                                               │
                              (always, unless fundamental data corruption)
                                               │
                                               ▼
                                  CALIBRATION_PREFLAG ◄──────────────┐
                                       │ rflag calibrators            │
                                       │ (max 3 iterations)           │
                                       ▼                              │
                   (polcal=True,   CALIBRATION_SOLVE                  │
                    verdict=FULL/   │ delay+BP+gain                   │
                    DEGRADED)       │ flag BP caltable                │
                        │           │ emit WILDCAT_METRICS            │
                        ▼           ▼                                 │
                  POLCAL_SOLVE  CALIBRATION_APPLY                     │
                   delay+BP+gain    │ applycal                        │
                   setjy(polcal)    │ rflag corrected + target        │
                   Kcross+Df+Xf ───┘ emit WILDCAT_METRICS            │
                   WILDCAT_METRICS  │                                 │
                                    ▼                                 │
                         CALIBRATION_CHECKPOINT                       │
                              │            │                          │
                   (approve)  │            │ (loop_back) ─────────────┘
                              ▼            ▼
                    IMAGING_PIPELINE   CALIBRATION_LOOP
```

### 2.2 LLM Role per Stage

**Inspection phases (1–3):**
- LLM may only advance or return `ERROR` (fundamental data corruption).
- `HUMAN_CHECKPOINT` is not valid — orchestrator overrides it and logs a warning.
- Phase 3 has no `casa_script` in the schema (read-only inspection).

**Calibration stages:**

| Stage | Script source | Next-stage decision | LLM role |
|-------|--------------|---------------------|----------|
| CALIBRATION_PREFLAG | LLM fills template | LLM: loop or proceed | Full |
| CALIBRATION_SOLVE | Orchestrator fills all placeholders | Rule: bp<0.20 AND gain<0.15 → APPLY, else → PREFLAG | None |
| CALIBRATION_APPLY | Orchestrator fills all placeholders | Fixed: → CALIBRATION_CHECKPOINT | Metrics summary + checkpoint question only |
| POLCAL_SOLVE | LLM fills template | Fixed: → CALIBRATION_APPLY | Full |

**Orchestrator-enforced loop guards:**
- 3-iteration cap on CALIBRATION_PREFLAG (counter increments at stage entry, not exit).
- Flag-fraction cap: if overall > 0.50 after a PREFLAG pass, force-advance to CALIBRATION_SOLVE rather than destroying more data.
- If CALIBRATION_SOLVE thresholds not met and PREFLAG cap already reached, force-advance to CALIBRATION_APPLY.
- Failed CASA job → ERROR immediately. No LLM retry on failed CASA.

### 2.3 LLM Context — Minimal but Sufficient

Each LLM call receives only what it needs to decide autonomously. No accumulated
history across calls.

| Call site | System prompt | User content |
|-----------|--------------|--------------|
| Inspection phase 1–2 | Skills partial + JSON schema + `/no_think` | Phase tool outputs (JSON) |
| Inspection phase 3 | Skills partial + JSON schema + `/no_think` | Phase tool outputs (JSON) |
| CALIBRATION_PREFLAG iteration 1 | Skills partial + JSON schema + `/no_think` | Tool outputs (ms_field_list, ms_spectral_window_list, ms_correlator_config) + workflow config + script template |
| CALIBRATION_PREFLAG iteration 2+ | Skills partial + JSON schema + `/no_think` | **Previous filled script** + WILDCAT_METRICS + iteration count. Tool outputs omitted — static, already encoded in prior script. |
| CALIBRATION_APPLY | Skills partial + JSON schema + `/no_think` | WILDCAT_METRICS from SOLVE + APPLY only |

Rationale for re-entry: the LLM needs to know what it flagged last time and what
happened, not re-derive field IDs and SPW ranges it already filled correctly.

### 2.4 Deterministic Config Rules

Applied by the orchestrator after Phase 3, before any LLM call. Override LLM.

| Tool output | Rule | Config effect |
|------------|------|--------------|
| `ms_pol_cal_feasibility.data.verdict` ∈ {NOT_FEASIBLE, DEGRADED} | PA coverage insufficient | `polcal = False` |
| Any `ms_antenna_flag_fraction.data.per_antenna[].flag_fraction >= 1.0` | Antenna fully dead | `aggressive_flagging = True` |

### 2.5 Polcal Routing

Orchestrator-driven, not LLM-driven.
- After CALIBRATION_PREFLAG, if `polcal=True` AND verdict ∈ {FULL, DEGRADED}: redirect to POLCAL_SOLVE.
- Otherwise: CALIBRATION_SOLVE silently.
- CALIBRATION_APPLY auto-detects polcal tables via `os.path.exists`; sets `parang=True` only when present.

### 2.6 Human Checkpoint

The one human decision: are calibration metrics good enough to image?

- LLM generates a single `calibration_done` question with real metric values
  (`bp_flagged_frac`, `gain_flagged_frac`, `post_cal_flag_frac`, `n_antennas_lost`).
- Options: `proceed` → IMAGING_PIPELINE, `loop_back` → CALIBRATION_LOOP, `exit` → STOPPED.
- Polcal and flagging decisions are never human questions — derived deterministically.

---

## 3. LLM JSON Contracts

### 3.1 Inspection Phases 1–2

```json
{
  "next_stage": "<PHASE2_RUNNING | PHASE3_RUNNING | ERROR>",
  "casa_script": "<python string or null>",
  "summary": "<2-5 sentences>",
  "reasoning": "<brief trace>"
}
```

### 3.2 Inspection Phase 3

```json
{
  "next_stage": "<CALIBRATION_PREFLAG | ERROR>",
  "summary": "<2-5 sentences>",
  "reasoning": "<brief trace>"
}
```

### 3.3 CALIBRATION_PREFLAG

```json
{
  "next_stage": "<CALIBRATION_PREFLAG | CALIBRATION_SOLVE>",
  "casa_script": "<completed script>",
  "summary": "<flag fractions and decision>",
  "reasoning": "<brief trace>"
}
```

### 3.4 CALIBRATION_APPLY

```json
{
  "summary": "<2-5 sentences on calibration quality>",
  "checkpoint_questions": [
    {
      "id": "calibration_done",
      "finding": "<bp_flagged_frac=X, gain_flagged_frac=Y, post_cal_flag_frac=Z, n_antennas_lost=N>",
      "severity": "<info|warning|critical>",
      "question": "Calibration complete. Proceed to imaging or loop back?",
      "options": ["proceed", "loop_back", "exit"]
    }
  ]
}
```

### 3.5 Error Handling

- Bad JSON from LLM → `ValueError` → `run()` try/except → ERROR, clean exit.
- Failed CASA job → `_run_casa_job` transitions to ERROR, caller returns immediately.
- Unknown or illegal `next_stage` → transition to ERROR.

---

## 4. Current Status (2026-03-25)

### Working
- Container builds (llama.cpp + CUDA 12.6, sm_86)
- ms-inspect starts via supervisord, 20 tools over SSE
- Phases 1–3 complete against 3C129_1.ms
- Deterministic config fires correctly: polcal=False (poor PA), aggressive_flagging=True (ea19 dead)
- CALIBRATION_PREFLAG: LLM fills script, rflag runs, metrics emitted
- CALIBRATION_SOLVE: all placeholders filled deterministically
- CALIBRATION_APPLY: script deterministic; LLM generates checkpoint question only
- PREFLAG loop cap: counter increments at stage entry; cap fires after 3 entries regardless of exit path
- PREFLAG re-entry: sends previous script + WILDCAT_METRICS only (no tool outputs)
- ms-inspect `{value: null, flag: "UNAVAILABLE"}` envelope handled safely across all helper methods

### Open / Next
- First successful end-to-end run to CALIBRATION_CHECKPOINT (in progress)
- CALIBRATION_CHECKPOINT UI validation with real bp/gain metrics
- Calibration quality plots (Bokeh, post-checkpoint)
- Imaging script generation

---

## 5. Known Issues / Resolved

| Issue | Resolution |
|-------|-----------|
| CUDA 12.9 CCCL macro error | Downgrade build base to 12.6.3 |
| `libcuda.so.1` not found at link time | `BUILD_SHARED_LIBS=OFF` |
| `libcuda.so.1` not found at runtime | CDI spec + `/etc/cdi/nvidia.yaml` |
| `FastMCP.run()` host kwarg missing | Switch to `sse_app()` + uvicorn |
| ms-inspect not on PyPI | pip install from mounted volume in entrypoint.sh |
| LLM escalating inspection phases to HUMAN_CHECKPOINT | Guard + override; not valid from Phase 3 |
| Starlette 1.0.0 breaking TemplateResponse API | All calls updated to (request, name, context) |
| `casa` binary not in container | Execute via `python3`; casatasks is a Python package |
| Failed CASA job not detected — orchestrator loops | `_run_casa_job` checks outcome, transitions to ERROR |
| LLM parse error crashes process, supervisord restarts | try/except in `run()` loop; ERROR + clean exit |
| Pipeline stuck waiting for `/start` | `WILDCAT_AUTOSTART=1` in `podman run` |
| `casa_script` required in schema | Made optional; Phase 3 omits it entirely |
| Sentinel file race condition | Removed sentinel/watchdog; direct `await runner.submit()` |
| LLM hallucinating CALIBRATION_SOLVE placeholders | CALIBRATION_SOLVE made fully deterministic |
| CALIBRATION_APPLY script LLM-generated | Made deterministic; LLM retained for checkpoint question only |
| `flagdata rflag` on cal table — missing data column | Added `datacolumn='CPARAM'` to both SOLVE and POLCAL templates |
| `{value: null, flag: "UNAVAILABLE"}` envelope — `.get("value", default)` returns None | Changed to `(... .get("value") or default)` in all helper methods |
| PREFLAG loop cap never fires — counter reset on exit | Counter now increments at stage entry; removed reset-on-exit logic |
| LLM context overflow on PREFLAG re-entry | Re-entry sends previous script + metrics only; tool outputs omitted |
