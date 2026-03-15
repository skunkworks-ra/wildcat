# wildcat — Design Document
## Agentic Radio Interferometry Reduction Pipeline

**Status:** Working — Phase 1-2 end-to-end verified, calibration pipeline staged
**Last revised:** 2026-03-15 (rev 4)
**Scope:** Orchestration layer. Phases 1–3 inspection + human checkpoint routing.

---

## 1. Philosophy

### 1.1 The Problem

Radio interferometric reduction requires running CASA scripts that take minutes to
hours. The workflow is not linear — after inspection a human expert must decide
whether to proceed to imaging or return to flagging and calibration. This decision
cannot be automated; it requires scientific judgment.

The goal is to automate everything *except* that judgment call, and to make the
handoff to the human — and back — as frictionless as possible.

### 1.2 Key Design Decisions

- **Stateless LLM per decision point.** llama-server is invoked once per phase.
  SQLite is the durable memory between calls. This tolerates arbitrarily long
  CASA jobs without timeouts.
- **ms-inspect over SSE.** ms-inspect serves MCP tools over SSE transport
  (`/sse`, `/messages/`) using `mcp.FastMCP.sse_app()` + uvicorn. The MCP
  Python client handles the protocol natively.
- **Sentinel files, not polling.** CASARunner writes `<job_id>.done` or
  `<job_id>.failed` on exit. watchdog fires an asyncio.Event to unblock
  the orchestrator instantly.
- **Single container.** llama.cpp built from source with CUDA inside the image.
  ms-inspect volume-mounted and pip-installed at entrypoint. Model GGUF
  mounted at runtime.

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
IDLE → PHASE1_RUNNING → PHASE2_RUNNING → PHASE3_RUNNING → HUMAN_CHECKPOINT
                                                               │
                                              (human approves) │
                                                               ▼
                                              CALIBRATION_PREFLAG ◄──────────┐
                                                       │ rflag calibrators    │
                                                       │ (max 3 iterations)   │
                                                       ▼                      │
                                              CALIBRATION_SOLVE               │
                                                       │ delay+BP+gain        │
                                                       │ flag BP caltable     │
                                                       │ emit WILDCAT_METRICS │
                                                       ▼                      │
                                              CALIBRATION_APPLY               │
                                                       │ applycal             │
                                                       │ rflag corrected      │
                                                       │ flag target          │
                                                       ▼                      │
                                         CALIBRATION_CHECKPOINT               │
                                              │            │                  │
                                   (approve) │            │ (loop_back) ──────┘
                                              ▼            ▼
                                    IMAGING_PIPELINE    CALIBRATION_LOOP
                                                             │
                                                             └──► CALIBRATION_PREFLAG (restart)
```

The LLM is the orchestrator at every calibration stage — it reads
`WILDCAT_METRICS` from CASA stdout and decides whether to loop or advance.
The orchestrator enforces a 3-iteration cap on `CALIBRATION_PREFLAG` and
escalates to `CALIBRATION_CHECKPOINT` if data remains heavily flagged.

### 2.2 MCP Tool Groups

| Phase | Tools |
|-------|-------|
| 1 | ms_observation_info, ms_field_list, ms_scan_list, ms_scan_intent_summary, ms_spectral_window_list, ms_correlator_config |
| 2 | ms_antenna_list, ms_baseline_lengths, ms_elevation_vs_time, ms_parallactic_angle_vs_time, ms_shadowing_report, ms_antenna_flag_fraction |
| 3 | ms_rfi_channel_stats, ms_flag_summary, ms_refant, ms_pol_cal_feasibility |

ms-inspect exposes 20 tools total (verified against 3C129_1.ms).

---

## 3. Container

**Base image:** `docker.io/nvidia/cuda:12.6.3-devel-ubuntu24.04`
Note: CUDA 12.9.1 is incompatible with current llama.cpp (CCCL 2.9 macro
breakage). 12.6.3 produces a binary that runs correctly on the host's 12.9
driver stack.

**GPU injection (WSL2):** CDI spec generated via `nvidia-ctk cdi generate`
and placed in `/etc/cdi/nvidia.yaml`. Container launched with
`--device nvidia.com/gpu=all`.

**llama.cpp build flags:**
```
-DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_CUDA_ARCHITECTURES=86
```
`BUILD_SHARED_LIBS=OFF` avoids the `libcuda.so.1` link-time issue — the
static binary resolves it at runtime from the CDI-injected driver.

---

## 4. UI Layer

### 4.1 Pages

| Route | Purpose |
|-------|---------|
| `/pipeline/{id}` | Live pipeline transparency monitor — tool outputs, LLM decisions, CASA artifacts |
| `/checkpoint/{id}` | Human checkpoint — structured questions, route decision |
| `/logs` | Raw log stream (SSE tail of `/var/log/wildcat.log`) |

### 4.2 Pipeline Monitor (`/pipeline/{id}`)

HTMX polling (1s) of `/pipeline/{id}/fragment`. Renders:
- **MCP Tool Calls** — per-phase cards with structured field extraction from the
  `{value, flag}` ms-inspect envelope. Per-tool renderers: field chips
  (ms_field_list), scan table (ms_scan_list), SPW table (ms_spectral_window_list),
  intent bar chart (ms_scan_intent_summary), generic KV table fallback.
- **LLM Decisions** — summary (primary), reasoning trace (collapsible), next_stage
  badge, CASA script (collapsible).
- **CASA Jobs & Artifacts** — job status badges, plot thumbnails, stdout/stderr.
- **Stop pipeline** button — cooperative cancellation via `stop_event` asyncio.Event;
  orchestrator checks at phase boundaries (before tools, before LLM, before CASA).
  Transitions to `STOPPED` stage on next safe boundary.

### 4.3 Human Checkpoint — Structured Questions (rev 4)

**Problem:** The current checkpoint presents two hard-coded buttons (imaging /
calibration) with no scientific context. This is too blunt for domain experts.

**Design:**

The LLM emits a `checkpoint_questions` array when `next_stage == HUMAN_CHECKPOINT`,
as part of its existing Phase N decision JSON. No second LLM call (empirical
first — may add dedicated checkpoint reasoning call later if same-call quality
is poor).

```json
"checkpoint_questions": [
  {
    "id": "polcal",
    "finding": "PA coverage is 45° — below the 90° threshold for reliable polarization calibration.",
    "severity": "warning",
    "question": "Proceed without polarization calibration?",
    "options": ["proceed", "loop_back", "exit"]
  },
  {
    "id": "aggressive_flagging",
    "finding": "3 scans show elevated RFI in spw 4-6.",
    "severity": "info",
    "question": "Apply aggressive flagging before calibration?",
    "options": ["yes", "no"]
  }
]
```

Options always include `exit` as a valid escape hatch.

**Route resolution (orchestrator-side, deterministic):**

| answers contain | route |
|----------------|-------|
| any `exit` | → `STOPPED` |
| any `loop_back` | → `CALIBRATION_LOOP` |
| all `proceed` / `yes` / `no` | → `IMAGING_PIPELINE` |

Priority: exit > loop_back > proceed.

**Config mapping (fixed rule table in orchestrator, not LLM-generated):**

The tiny model (Qwen3.5-4B) is not asked to emit config key/value mappings —
that is fragile schema generation. Instead a static table in the orchestrator
maps `(question_id, answer) → (config_key, value)`:

```python
_QUESTION_CONFIG_MAP = {
    "polcal":              {"proceed": ("polcal", False),           "loop_back": None, "exit": None},
    "aggressive_flagging": {"yes":     ("aggressive_flagging", True), "no": ("aggressive_flagging", False)},
}
```

**Workflow config state:**

A `workflow_config` JSON column on the `workflow` table carries human decisions
forward. Defaults are source-agnostic pipeline defaults:

```json
{
  "polcal": true,
  "aggressive_flagging": false
}
```

This blob is injected into every subsequent LLM prompt (Phase 2+) so the model
knows what the human has decided and adjusts tool selection and recommendations
accordingly.

**DB changes:**
- `workflow.workflow_config TEXT` — JSON blob, written at workflow creation,
  updated after each checkpoint resolution.
- `checkpoints.question_answers TEXT` — JSON blob recording human answer per
  question_id for audit trail.

**What changes per file:**

| File | Change |
|------|--------|
| `state.py` | `workflow_config` column; `get/set_workflow_config()` methods |
| `orchestrator.py` | Inject config into LLM prompts; parse `checkpoint_questions`; apply `_QUESTION_CONFIG_MAP`; resolve route from answers |
| `_DECISION_SCHEMA_KEYS` | `checkpoint_questions` added as optional key |
| `_JSON_INSTRUCTION` | Document new schema to LLM |
| `ui/app.py` | `POST /checkpoint/{id}/decide` accepts per-question answers, resolves route, writes config |
| `templates/checkpoint.html` | Question cards with option buttons; plot display |

### 4.4 Plots in checkpoint UI

The checkpoint page renders plot thumbnails inline (from `jobs.plots`) with
LLM-provided captions where available. The pipeline monitor also shows them
under CASA Jobs. The LLM may reference specific plot filenames in its
`checkpoint_questions` findings to direct the human's attention.

---

## 5. Current Status (2026-03-15)

### Working
- Container builds successfully (llama.cpp + CUDA 12.6, sm_86)
- ms-inspect starts via supervisord, serves all 20 tools over SSE
- Phase 1 and 2 tools verified against 3C129_1.ms
- LLM (Qwen3.5-4B Q4_K_XL, RTX 3080 Laptop) produces structured JSON decisions
- Workflow transitions through inspection phases to HUMAN_CHECKPOINT
- Structured checkpoint UI — per-question cards with data-driven findings,
  polcal and aggressive_flagging decisions captured in workflow_config
- Pipeline monitor at `/pipeline/{id}` — one page: tool cards, LLM decisions,
  CASA jobs, and checkpoint form in a single HTMX-polled view
- Checkpoint panel uses outerHTML swap trick to lock the form once the
  questions arrive (polling stops, form is stable and clickable)
- Calibration pipeline staged: CALIBRATION_PREFLAG → CALIBRATION_SOLVE →
  CALIBRATION_APPLY → CALIBRATION_CHECKPOINT — LLM routes autonomously via
  WILDCAT_METRICS; 3-iteration preflag cap enforced in orchestrator
- CASA script templates with deterministic sections (solint from scan
  durations, flagcal on BP caltable, rflag, metrics) and LLM-filled
  placeholders (field names, refant, flux standard)

### Open / Next
- End-to-end calibration test (CASA not yet in container)
- CALIBRATION_CHECKPOINT UI validation with real bp/gain metrics
- Imaging script generation post-calibration
- Phase 3 tool calls untested end-to-end

---

## 6. Known Issues / Resolved

| Issue | Resolution |
|-------|-----------|
| CUDA 12.9 CCCL macro error | Downgrade build base to 12.6.3 |
| `libcuda.so.1` not found at link time | `BUILD_SHARED_LIBS=OFF` |
| `libcuda.so.1` not found at runtime | CDI spec + `/etc/cdi/nvidia.yaml` |
| `FastMCP.run()` host kwarg missing | Switch to `sse_app()` + uvicorn |
| Second LLM call OOM in `_handle_checkpoint` | Re-use summary from llm_decisions table |
| ms-inspect not on PyPI | pip install from mounted volume in entrypoint.sh |
