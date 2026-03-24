# wildcat — Design Document
## Agentic Radio Interferometry Reduction Pipeline

**Status:** End-to-end calibration run not yet completed — structural fixes complete, first full run pending
**Last revised:** 2026-03-24 (rev 8)
**Scope:** Full pipeline — inspection phases, deterministic config, calibration, human checkpoint.

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

- **Stateless LLM per decision point.** llama-server is invoked once per phase.
  SQLite is the durable memory between calls. This tolerates arbitrarily long
  CASA jobs without timeouts.
- **ms-inspect over SSE.** ms-inspect serves MCP tools over SSE transport
  (`/sse`, `/messages/`) using `mcp.FastMCP.sse_app()` + uvicorn. The MCP
  Python client handles the protocol natively.
- **Direct asyncio subprocess await.** CASARunner.submit() is awaited directly
  by the orchestrator. The asyncio event loop stays idle while CASA runs — no
  sentinel files, no watchdog. Previous sentinel/watchdog approach was replaced
  after diagnosing a failure mode where pre-existing `.done` files from prior
  runs silently prevented inotify events from firing.
- **Single container.** llama.cpp built from source with CUDA inside the image.
  ms-inspect volume-mounted and pip-installed at entrypoint. Model GGUF
  mounted at runtime.
- **Explicit pipeline start gate.** The pipeline does not start automatically
  on container boot. wildcat waits for an explicit start signal before creating
  the workflow or starting the LLM. Priority: `--autostart` CLI flag >
  `WILDCAT_AUTOSTART` env var > UI button at `/start`. `run.sh` passes
  `WILDCAT_AUTOSTART=1` so automated runs are hands-off.
- **Deterministic config over LLM judgment.** Data quality decisions with
  measurable thresholds (polcal feasibility, antenna flagging) are made by
  the orchestrator from tool outputs — not by the LLM and not by the human.
  The LLM advances the pipeline; the orchestrator enforces constraints.
- **CASA scripts run via python3.** `casatasks` is installed as a Python
  package. Scripts are executed as `python3 script.py`, not via the `casa`
  CLI. `config.toml [casa] executable = "python3"`, `args = []`.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│  wildcat container                                  │
│                                                     │
│  supervisord                                        │
│  ├── ms-inspect  (SSE :8000)                        │
│  └── wildcat     (orchestrator + UI :8081)          │
│       └── llama-server subprocess (:8080)           │
│                                                     │
│  Volume mounts:                                     │
│  /opt/ms-inspect  ← data-analyst repo (ro)         │
│  /skills          ← skill partials (ro)             │
│  /data/ms         ← measurement sets (ro)           │
│  /models/model.gguf ← GGUF file (ro)               │
│  /data            ← SQLite DB + job outputs (rw)   │
└─────────────────────────────────────────────────────┘
```

### 2.1 Workflow State Machine

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

**LLM role in inspection phases (1–3):**
- Each phase calls its MCP tool group, passes outputs to LLM, reads JSON decision.
- LLM may only advance to the next phase or return `ERROR` for fundamental data
  corruption (all calibrators missing/corrupt, no usable data).
- `HUMAN_CHECKPOINT` is blocked by the orchestrator in all inspection phases —
  if the LLM returns it, the orchestrator overrides to the next phase and logs
  a warning. Phase 3 does not include `casa_script` in its JSON schema.
- After Phase 3, the orchestrator applies **deterministic config rules** (see §2.3)
  before transitioning to `CALIBRATION_PREFLAG`.

**Calibration phase LLM role (per stage):**

| Stage | Script | Next-stage decision | LLM role |
|-------|--------|---------------------|----------|
| CALIBRATION_PREFLAG | LLM fills template | LLM: loop or proceed | Full LLM |
| CALIBRATION_SOLVE | Orchestrator fills all placeholders | Rule: bp<0.20 AND gain<0.15 → APPLY, else → PREFLAG | None |
| CALIBRATION_APPLY | Orchestrator fills all placeholders | Fixed: → CALIBRATION_CHECKPOINT | Metrics summary + checkpoint questions only |
| POLCAL_SOLVE | LLM fills template | Fixed: → CALIBRATION_APPLY | Full LLM |

Orchestrator enforces: 3-iteration cap on CALIBRATION_PREFLAG, flag-fraction
cap (if overall > 0.50, force-advance rather than flagging away more data), job
failure → ERROR (no LLM retry on failed CASA).

**Polcal routing (orchestrator, not LLM):**
- `ms_pol_cal_feasibility` verdict read from Phase 3 tool outputs (already in DB).
- After CALIBRATION_PREFLAG, if `polcal=True` AND verdict ∈ {FULL, DEGRADED}:
  orchestrator redirects `CALIBRATION_SOLVE → POLCAL_SOLVE`.
- If verdict is NOT_FEASIBLE or LEAKAGE_ONLY: uses CALIBRATION_SOLVE silently.
- CALIBRATION_APPLY auto-detects polcal tables via `os.path.exists`; sets
  `parang=True` only when they are present.

**Human checkpoint (CALIBRATION_CHECKPOINT only):**
- The one remaining human decision: are calibration metrics good enough to image?
- The LLM generates a single `calibration_done` question with real metrics
  (`bp_flagged_frac`, `gain_flagged_frac`, `post_cal_flag_frac`, `n_antennas_lost`).
- Options: `proceed` → IMAGING_PIPELINE, `loop_back` → CALIBRATION_LOOP,
  `exit` → STOPPED.

### 2.2 MCP Tool Groups

| Phase | Tools |
|-------|-------|
| 1 | ms_observation_info, ms_field_list, ms_scan_list, ms_scan_intent_summary, ms_spectral_window_list, ms_correlator_config |
| 2 | ms_antenna_list, ms_baseline_lengths, ms_elevation_vs_time, ms_parallactic_angle_vs_time, ms_shadowing_report, ms_antenna_flag_fraction |
| 3 | ms_rfi_channel_stats, ms_flag_summary, ms_refant, ms_pol_cal_feasibility |

ms-inspect exposes 20 tools total (verified against 3C129_1.ms).

### 2.3 Deterministic Config Rules

Applied by the orchestrator after Phase 3 tool calls, before any LLM decision:

| Tool output | Rule | Config effect |
|------------|------|--------------|
| `ms_pol_cal_feasibility.data.verdict` ∈ {NOT_FEASIBLE, DEGRADED} | polcal not viable | `polcal = False` |
| Any `ms_antenna_flag_fraction.data.per_antenna[].flag_fraction >= 1.0` | antenna fully dead | `aggressive_flagging = True` |

These override any LLM `config_updates`. Logged as warnings. The human is
notified via the pipeline monitor but not asked to confirm — the data speaks.

### 2.4 WILDCAT_METRICS Protocol

CASA scripts emit one line to stdout at the end:

```
WILDCAT_METRICS: {"stage": "CALIBRATION_PREFLAG", "overall_flag_frac": 0.12, ...}
```

The orchestrator parses this in `_read_last_job_metrics()` and passes it to the
next LLM call as context. If a job fails (no metrics line), `_run_casa_job`
transitions to ERROR immediately — no LLM call on failed jobs.

---

## 3. Container

### 3.1 Scripts

| Script | Purpose |
|--------|---------|
| `build.sh` | Builds the container image — run once or when `Containerfile` changes |
| `fetch-model.sh` | Downloads the default GGUF (`unsloth/Qwen3.5-4B-GGUF`) from HuggingFace; uses `huggingface-cli` if available, falls back to `curl`; respects `HF_TOKEN`; outputs the local path to stdout |
| `run.sh` | Starts the container with `WILDCAT_AUTOSTART=1` — fully hands-off; requires image and `MODEL_PATH` to exist |
| `clean.sh` | Kills the running container and removes `wildcat.db` — idempotent; use before a fresh run |

**Typical workflow:**
```bash
./build.sh          # once, or on Containerfile changes
export MODEL_PATH=$(./fetch-model.sh)   # once
./run.sh            # start — hands-off, polls until CALIBRATION_CHECKPOINT
./clean.sh          # reset between runs
```

Override host ports with `PORT_MS_INSPECT` (default 8100), `PORT_LLAMA` (default
8180), `PORT_UI` (default 8181).

### 3.2 Container Image

**Base image:** `docker.io/nvidia/cuda:12.6.3-devel-ubuntu24.04`

Note: CUDA 12.9 host driver is compatible at runtime; 12.6.3 is the build base
because CUDA 12.9 CCCL macros break llama.cpp compilation.

**GPU injection (WSL2):** CDI spec generated via `nvidia-ctk cdi generate` and
placed in `/etc/cdi/nvidia.yaml`. Container launched with `--device nvidia.com/gpu=all`.

**llama.cpp build flags:**
```
-DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_CUDA_ARCHITECTURES=86
```
`BUILD_SHARED_LIBS=OFF` avoids `libcuda.so.1` link-time issues — the static binary
resolves it at runtime from the CDI-injected driver stack.

**entrypoint.sh:** Runs `pip install --upgrade -e /opt/ms-inspect[casa]` at every
container start so the live source is always used. Side effect: this can upgrade
Python dependencies (e.g. Starlette) to versions with breaking API changes. Known
issue: Starlette 1.0.0 changed `TemplateResponse(name, context)` to
`TemplateResponse(request, name, context)` — all UI routes use the new signature.

---

## 4. UI Layer

### 4.1 Pages

| Route | Purpose |
|-------|---------|
| `/start` | Pre-start gate — shows MS path, predicted workflow ID, Start Pipeline button. Bypassed when `WILDCAT_AUTOSTART=1`. |
| `/pipeline/{id}` | Live pipeline transparency monitor — tool outputs, LLM decisions, CASA artifacts. HTMX polling (1s). |
| `/checkpoint/{id}` | Human checkpoint — structured calibration metrics question, route decision. Only reached at CALIBRATION_CHECKPOINT. |
| `/logs` | Raw log stream (SSE tail of `/var/log/wildcat.log`) |

### 4.2 Pipeline Monitor (`/pipeline/{id}`)

HTMX polling (1s) of `/pipeline/{id}/fragment`. All `<details>` elements render
open by default (HTMX replaces the DOM on each poll; `open` attribute ensures
content is always visible). Renders:

- **MCP Tool Calls** — per-phase cards with structured field extraction from the
  `{value, flag}` ms-inspect envelope. Per-tool renderers: field chips
  (ms_field_list), scan table (ms_scan_list), SPW table (ms_spectral_window_list),
  intent bar chart (ms_scan_intent_summary), generic KV table fallback.
- **LLM Decisions** — summary (primary), reasoning trace, next_stage badge, CASA
  script (when present).
- **CASA Jobs & Artifacts** — job status badges, stdout/stderr, plot thumbnails.
- **Stop pipeline** button — cooperative cancellation via `stop_event`; orchestrator
  checks at phase boundaries. Transitions to `STOPPED` at next safe boundary.

### 4.3 Human Checkpoint (CALIBRATION_CHECKPOINT only)

The one human decision point: proceed to imaging or loop back for another
calibration pass. The LLM emits a single `calibration_done` question with real
metric values (`bp_flagged_frac`, `gain_flagged_frac`, `post_cal_flag_frac`,
`n_antennas_lost`). Options: `proceed`, `loop_back`, `exit`.

Polcal and flagging decisions are **not** human questions — they are derived
deterministically from tool outputs (see §2.3). The checkpoint_panel uses an
HTMX outerHTML swap trick to lock the form once questions arrive, preventing
the polling from resetting the form state.

---

## 5. LLM Contract

### 5.1 Inspection Phases (1–2)

```json
{
  "next_stage": "<PHASE2_RUNNING | PHASE3_RUNNING | ERROR>",
  "casa_script": "<python string or null>",
  "summary": "<2-5 sentences>",
  "reasoning": "<brief trace>"
}
```

### 5.2 Inspection Phase 3

```json
{
  "next_stage": "<CALIBRATION_PREFLAG | ERROR>",
  "summary": "<2-5 sentences>",
  "reasoning": "<brief trace>"
}
```

No `casa_script` (Phase 3 is read-only inspection). No `config_updates` or
`checkpoint_questions` (config is set deterministically). `HUMAN_CHECKPOINT` is
not a valid option — the orchestrator overrides it if returned.

### 5.3 Calibration Stages

Only CALIBRATION_PREFLAG and POLCAL_SOLVE involve a full LLM call to fill the script template. CALIBRATION_SOLVE and CALIBRATION_APPLY scripts are filled deterministically by the orchestrator.

**CALIBRATION_PREFLAG** (LLM fills script + decides next stage):
```json
{
  "next_stage": "<CALIBRATION_PREFLAG | CALIBRATION_SOLVE>",
  "casa_script": "<completed script>",
  "summary": "<flag fractions found and decision>",
  "reasoning": "<brief trace>"
}
```

**CALIBRATION_APPLY** (LLM interprets results only — no script, no routing):
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

### 5.4 Error Handling

- Bad JSON from LLM → `ValueError` → caught by `run()` try/except → ERROR state,
  clean exit (no supervisord restart loop).
- Failed CASA job → sentinel `.failed` detected → `_run_casa_job` transitions to
  ERROR → caller returns immediately.
- Unknown `next_stage` value → transition to ERROR.
- Illegal `next_stage` for calibration stage (not in whitelist) → transition to ERROR.

---

## 6. Current Status (2026-03-24)

### Working
- Container builds (llama.cpp + CUDA 12.6, sm_86)
- ms-inspect starts via supervisord, 20 tools over SSE
- Phases 1–3 complete against 3C129_1.ms
- Deterministic config: polcal=False (poor PA), aggressive_flagging=True (ea19 dead)
- Inspection phases never block at HUMAN_CHECKPOINT (guard + override)
- CASA scripts execute via python3 + casatasks (no standalone `casa` binary needed)
- Failed CASA job → ERROR state, no loop
- LLM parse errors → ERROR state, no supervisord restart loop
- All UI routes work with Starlette 1.0.0
- Pipeline monitor cells expand by default
- CALIBRATION_PREFLAG CASA script runs successfully (rflag + flag fraction metrics)
- CALIBRATION_SOLVE: all placeholders filled deterministically (field ID, refant, flux standard, SPWs, scans, int time)
- CALIBRATION_APPLY: CASA script filled deterministically; LLM generates human-facing summary only
- Orchestrator no longer uses sentinel files or watchdog — direct asyncio subprocess await

### Open / Next
- First successful CALIBRATION_SOLVE → CALIBRATION_APPLY → CALIBRATION_CHECKPOINT run (untested)
- CALIBRATION_CHECKPOINT UI validation with real bp/gain metrics
- Imaging script generation post-calibration

---

## 7. Known Issues / Resolved

| Issue | Resolution |
|-------|-----------|
| CUDA 12.9 CCCL macro error | Downgrade build base to 12.6.3 |
| `libcuda.so.1` not found at link time | `BUILD_SHARED_LIBS=OFF` |
| `libcuda.so.1` not found at runtime | CDI spec + `/etc/cdi/nvidia.yaml` |
| `FastMCP.run()` host kwarg missing | Switch to `sse_app()` + uvicorn |
| Second LLM call OOM in `_handle_checkpoint` | Re-use summary from llm_decisions table |
| ms-inspect not on PyPI | pip install from mounted volume in entrypoint.sh |
| LLM escalating inspection phases to HUMAN_CHECKPOINT | Guard + override in orchestrator; not a valid next_stage from Phase 3 |
| Starlette 1.0.0 breaking TemplateResponse API | All calls updated to (request, name, context) |
| `casa` binary not in container | Execute scripts via `python3`; casatasks is the Python package |
| Failed CASA job not detected — orchestrator loops | `_run_casa_job` checks job status post-sentinel; transitions to ERROR |
| LLM parse error crashes process, supervisord restarts as new workflow | try/except in `run()` loop; transitions to ERROR, exits cleanly |
| Pipeline stuck waiting for `/start` in run.sh | `WILDCAT_AUTOSTART=1` added to `podman run` in run.sh |
| `casa_script` required in `_DECISION_SCHEMA_KEYS` | Made optional; Phase 3 schema omits it entirely |
| Orchestrator permanently blocked after CALIBRATION_PREFLAG completes | Sentinel file `1.done` pre-existed from prior container run; `touch()` on existing file produces `FileModifiedEvent`, not `FileCreatedEvent`; watchdog never fired. Fixed: removed sentinel/watchdog, replaced with direct `await runner.submit()`. |
| LLM hallucinating wrong CALIBRATION_SOLVE placeholders (`flux_standard` hyphen format, `corrstring` including cross-hands) | CALIBRATION_SOLVE made fully deterministic — orchestrator fills all placeholders from tool outputs. No LLM involvement in script generation. |
| CALIBRATION_APPLY script still LLM-generated | Made deterministic — orchestrator fills CAL_FIELDS, ALL_SPW, CORRSTRING. LLM retained only for post-run checkpoint question generation. |
