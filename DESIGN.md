# wildcat — Design Document
## Agentic Radio Interferometry Reduction Pipeline

**Status:** Working — Phase 1 end-to-end verified
**Last revised:** 2026-03-14 (rev 2)
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
IDLE → PHASE1_RUNNING → HUMAN_CHECKPOINT → IMAGING_PIPELINE
                                         → CALIBRATION_LOOP
```

Phases 2 and 3 are implemented but Phase 1 currently transitions directly
to HUMAN_CHECKPOINT (LLM decision). Phases 2/3 will be wired once Phase 1
checkpoint UI is validated end-to-end.

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

## 4. Current Status (2026-03-14)

### Working
- Container builds successfully (llama.cpp + CUDA 12.6, sm_86)
- ms-inspect starts via supervisord, serves all 20 tools over SSE
- Phase 1 tools run against 3C129_1.ms — all 6 tools return valid JSON
- LLM (Qwen3.5-4B Q4_K_XL, RTX 3080 Laptop) receives Phase 1 output and
  produces structured JSON decision
- Workflow transitions to HUMAN_CHECKPOINT and is recorded in SQLite
- Checkpoint UI serves on :8081

### Open / Next
- Checkpoint row not yet written before llama-server exits (second LLM call
  removed — fix deployed, needs retest after rebuild)
- Checkpoint UI untested end-to-end (POST /decide not yet exercised)
- Phase 2 and 3 tool calls untested
- run.sh tool smoke test steps produce no output (silent — needs debug)
- CASA runner untested (no CASA in container yet)

---

## 5. Known Issues / Resolved

| Issue | Resolution |
|-------|-----------|
| CUDA 12.9 CCCL macro error | Downgrade build base to 12.6.3 |
| `libcuda.so.1` not found at link time | `BUILD_SHARED_LIBS=OFF` |
| `libcuda.so.1` not found at runtime | CDI spec + `/etc/cdi/nvidia.yaml` |
| `FastMCP.run()` host kwarg missing | Switch to `sse_app()` + uvicorn |
| Second LLM call OOM in `_handle_checkpoint` | Re-use summary from llm_decisions table |
| ms-inspect not on PyPI | pip install from mounted volume in entrypoint.sh |
