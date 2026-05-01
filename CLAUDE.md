# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**wildcat** is an agentic orchestration layer for radio interferometry data reduction. It
connects a local LLM (via llama.cpp or Ollama) to the `ms-inspect` MCP server, runs CASA
calibration scripts, and presents a human checkpoint UI — all inside a single Podman container.

**Repos:**
- `~/src/skunkworks-ra/wildcat` — wildcat itself
- `~/src/skunkworks-ra/radio-analyst` — ms-inspect MCP server + radio skills

The core design principle: automate everything except the scientific judgment call. The human
expert decides whether to proceed to imaging or loop back to calibration; wildcat handles
everything else including LLM-driven MCP tool calls, CASA script generation and execution,
and checkpoint routing.

## Development commands

Uses [pixi](https://pixi.sh) for environment management (Python 3.12).

```bash
pixi run test          # pytest tests/ -v
pixi run lint          # ruff check src/ tests/
pixi run check         # ruff check + ruff format --check
```

Run the orchestrator directly:
```bash
wildcat --ms /path/to/data.ms --config config.toml
wildcat --ms /path/to/data.ms --workflow-id 3   # resume existing workflow
```

Run a single test file:
```bash
pixi run pytest tests/test_tools.py::test_list_tools_returns_schemas -v
```

Integration tests require ms-inspect running and a real `.ms` path:
```bash
MS_INSPECT_URL=http://localhost:8000 TEST_MS_PATH=/data/3C129_1.ms pixi run test
```

Build and run in container:
```bash
./run.sh   # builds Podman container, runs smoke test, waits for checkpoint
# Override defaults: MS_PATH=... OUTPUT_DIR=... ./run.sh
```

## Architecture

```
main.py  ──→  Orchestrator  ──→  MSInspectClient  (MCP streamable-http POST /mcp)
                  │          ──→  LLMBackend        (OpenAI-compatible HTTP)
                  │          ──→  CASARunner         (asyncio subprocess)
                  │          ──→  StateDB            (SQLite /data/wildcat.db)
                  └──→  FastAPI UI  (:8081, checkpoint UI + pipeline monitor)
```

All managed by supervisord inside the container. ms-inspect and wildcat run as separate
supervised programs. CASA runs as child processes of wildcat.

### Workflow state machine

```
IDLE → PHASE1_RUNNING → PHASE1_COMPLETE → PHASE2_RUNNING → PHASE2_COMPLETE
     → PHASE3_RUNNING → PHASE3_COMPLETE → HUMAN_CHECKPOINT
                                               ├── CALIBRATION_PREFLAG → CALIBRATION_SOLVE
                                               │       ↑                      ↓
                                               │   (flag cap)          CALIBRATION_APPLY
                                               │                             ↓
                                               │                   CALIBRATION_CHECKPOINT
                                               │                    ├── IMAGING_PIPELINE
                                               │                    └── CALIBRATION_LOOP
                                               └── IMAGING_PIPELINE
                                               └── STOPPED
```

Phases 1-3 each call a group of MCP tools on ms-inspect, then invoke the LLM once. The LLM
returns a JSON decision (`next_stage`, `casa_script`, etc.). CALIBRATION_PREFLAG →
CALIBRATION_SOLVE is the normal calibration path. POLCAL_SOLVE is an alternate to
CALIBRATION_SOLVE when `polcal=True` in workflow_config.

State lives entirely in SQLite. The LLM is stateless and invoked once per decision point.

### Key modules

- `orchestrator.py` — main loop; dispatches to `_handle_phase()`, `_handle_checkpoint()`,
  and calibration stage handlers; parses and validates LLM JSON; contains all per-stage
  JSON instruction strings and CASA script templates.
- `state.py` — all SQLite access; `Stage` enum defines every valid state; `StateDB` is a
  context manager; never write raw SQL outside this file.
- `llm.py` — `LLMBackend` starts/stops llama-server subprocess or connects to Ollama;
  `complete()` for single-turn, `complete_with_tools()` for multi-turn tool-use loops.
  Health check polls `/health` for up to 180s (model cold-load from disk can take >2 min).
- `tools.py` — `MSInspectClient` wraps MCP streamable-http; opens a fresh session per call.
- `runner.py` — `CASARunner.submit()` spawns CASA as `asyncio.create_subprocess_exec`,
  streams stdout/stderr to SQLite, awaits exit. `WILDCAT_METRICS:` prefix lines in stdout
  are extracted as structured metrics.
- `skills.py` — assembles system prompt from skill partials in the mounted
  `/skills/radio-interferometry/` volume; missing partials are warnings, not errors.
- `config.py` — typed dataclasses loaded from `config.toml` via `tomllib`.
- `ui/app.py` — FastAPI app; key routes: `GET /start`, `GET /pipeline/{id}`,
  `POST /checkpoint/{id}/decide`, `GET /logs/stream` (SSE).

### LLM decision contract

Every LLM response must be a JSON object. `_DECISION_SCHEMA_KEYS` requires `next_stage`,
`summary`, and `reasoning` from **every** stage — per-stage instructions must include all
three. Optional: `casa_script`, `checkpoint_questions`, `auto_proceed`. The orchestrator
validates in `_parse_decision()` and enters ERROR on malformed output.

### Internal tools (multi-turn tool-use stages)

Stages in `_TOOL_USE_STAGES` (CALIBRATION_PREFLAG, CALIBRATION_SOLVE, POLCAL_SOLVE) run a
multi-turn tool-use loop. Available tools:
- `get_metrics(stage)` — WILDCAT_METRICS from a completed CASA job
- `get_previous_script(stage)` — CASA script from the most recent completed job
- `get_tool_output(tool_name)` — Phase 1-3 MCP tool output by name
- `get_workflow_config()` — current polcal/aggressive_flagging/iteration counts
- `get_calsol_stats(caltable)` — full raw `ms_calsol_stats` output for `delay.cal`,
  `bandpass.cal`, or `gain.cal` (stored in DB after CALIBRATION_SOLVE CASA job runs)

### Container ports

| Container | Host (`run.sh`) | Service |
|-----------|-----------------|---------|
| 8000      | 8100            | ms-inspect MCP |
| 8080      | 8180            | llama-server |
| 8081      | 8181            | Checkpoint UI |

### Volume mounts

- `/data` → `$OUTPUT_DIR` — wildcat.db, jobs/, log files
- `/data/ms/` → directory containing the .ms file (read-only)
- `/opt/ms-inspect` → radio-analyst repo (read-only, pip-installed at startup)
- `/skills/radio-interferometry` → `.claude/skills/radio-interferometry`
- `/models/model.gguf` → GGUF model file (read-only)

## Configuration

All settings in `config.toml`. Key sections:

```toml
[llm]
backend = "llamacpp"   # or "ollama"
max_user_tokens = 16000
max_retries     = 5
max_tool_rounds = 5

[llm.llamacpp]
model_path  = "/models/model.gguf"
port        = 8080
ctx_size    = 32768
n_gpu_layers = 99
temp        = 0.0
```

Switch backends by changing `[llm] backend` — no code changes required.

## Known constraints and hard-won fixes

### CALIBRATION_PREFLAG: LLM trailing comma turns flag_cmds into a tuple (fixed)
The LLM was adding a trailing comma when copying the pre-filled template back:
`flag_cmds = [...],` — Python treats this as a tuple, causing CASA `flagdata` to
reject `inpfile` with `AssertionError: {'inpfile': ['must be of cStr type']}`.

Fix: on the first pass (`not is_reentry`) the orchestrator uses `tmpl` directly
and ignores `decision["casa_script"]`. For re-entries a regex sanitizer strips
`flag_cmds = [...],` → `flag_cmds = [...]` before the script reaches CASA.

### CALIBRATION_APPLY: missing next_stage/reasoning in LLM instruction (fixed)
`_JSON_INSTRUCTION_APPLY` defined a schema without `next_stage` or `reasoning`, but
`_DECISION_SCHEMA_KEYS` requires both from every stage. The LLM followed the instruction
faithfully and returned `{auto_proceed, summary, checkpoint_questions}`, causing
`_parse_decision()` to reject all 5 retries and enter ERROR. Fixed by adding `next_stage`
(`IMAGING_PIPELINE` if `auto_proceed` else `CALIBRATION_CHECKPOINT`) and `reasoning` to
the schema. **Rule:** every stage instruction must include `next_stage`, `summary`, and
`reasoning` — verify this whenever adding a new stage.

### CALIBRATION_SOLVE context overflow on ms_calsol_stats (fixed)
`ms_calsol_stats` on `bandpass.cal` returns a 27×8×64 per-channel `amp_array` (~515KB).
All three caltables together hit 347k tokens — way over the 65536-token context window,
causing `exceed_context_size_error` and ERROR state with no LLM decision recorded.

Fix: `_run_calsol_stats()` stores the full raw JSON in `tool_outputs`
(`phase="calibration_solve"`) then returns a compact summary via `_summarize_calsol_stats()`,
which collapses nested arrays to `{min, max, mean}` scalars and per-antenna/per-SPW dicts
(~670 tokens total for all three tables). The LLM can call `get_calsol_stats(caltable)` if
it needs per-channel detail.

### ms-inspect health check in run.sh
ms-inspect returns HTTP 406 for a bare POST to `/mcp` (not 200/400/422). Health check must
include 406 in accepted codes. Use `127.0.0.1` not `localhost` — on this host `localhost`
resolves to `::1`/IPv6 which pasta networking cannot route to the IPv4-only container
service. Use `curl -4`.

### SSH port forwarding
`LocalForward 8181 127.0.0.1:8181` — must use `127.0.0.1`, not `localhost`. Same IPv4/IPv6
issue as above.

### Calibrator lookup fails for CASA "=" naming convention (fixed)
CASA names calibrators as `"B1950=CommonName"` (e.g. `"0137+331=3C48"`). `_normalise()` in
`calibrators.py` and `pol_calibrators.py` strips `+` but not `=`, so the lookup fails.
Fixed in both files to split on `=` and try each part. Root cause of: `calibrator_role=null`
for 3C48, `flux_standard=null` → empty `setjy()` standard → CASA error, pol cal verdict
`NOT_FEASIBLE` when 3C48 is a valid pol angle calibrator.

### CALIBRATION_PREFLAG crash: corr_products key mismatch (fixed)
`_prefill_preflag_template()` called `.get("corr_products", [])` but `ms_correlator_config`
uses the field-wrapped schema key `correlation_products` with `{"value": [...], "flag": "..."}`.
Always unwrap with `.get("value", default)` after the field name lookup.

### Empty flux_standard causes setjy() failure (fixed)
When `ms_field_list` doesn't recognise the flux calibrator, it returns `flux_standard=null`.
`_solve_flux_standard()` now falls back to `"Perley-Butler 2017"` instead of returning `""`.

### asyncio deadlock in ms-inspect (fixed, in radio-analyst)
`ms_antenna_flag_fraction` used `mp.get_context("fork")` from inside an asyncio event loop.
Fix: `asyncio.to_thread()` wrapping in `server.py`; `spawn` context in `tools/flags.py`.

### Per-scan intents empty despite field-level intents available (fixed, in radio-analyst)
For some MSs, `msmd.intentsforscan()` returns empty but `msmd.intentsforfield()` works.
Fixed in `scans.py` to fall back to `intentsforfield(fid)` when per-scan intents are empty.
Note: the correct API is `msmd.intentsforscan(scan_num)` (singular scalar) — the plural
`intentsforscans([scan_num])` doesn't exist in CASA 6.x; `AttributeError` was silently
swallowed by `except Exception`.

### ms_shadowing_report: msmd.shadowedAntennas() doesn't exist in CASA 6.x (fixed)
Fixed to use `casatasks.flagdata(mode='shadow', action='calculate')`. Output format changed:
`shadow_flag_fraction`, `n_shadow_flagged`, `n_total_rows`, `shadowed_antennas` replace the
old fields.

### CUDA and container build
- CUDA architectures: `sm_61` (Pascal/GTX 10xx), `sm_75` (Turing/RTX 2080), `sm_86`
  (Ampere/RTX 3xxx). After changing `CMAKE_CUDA_ARCHITECTURES` a full `./build.sh` rebuild
  is required (~10-15 min).
- `BUILD_SHARED_LIBS=OFF` is required — the static binary resolves `libcuda.so.1` at
  runtime from the host driver.
- CUDA 12.9 host driver works with CUDA 12.6.3 container base image via CDI injection.

### GPU detection order in run.sh
1. `/dev/dxg` present → WSL2 mode
2. `/etc/cdi/nvidia.yaml` present → CDI spec (`--device nvidia.com/gpu=all`)
3. `nvidia-smi` works → direct device passthrough + bind-mount `libcuda.so.1` from host
4. Fallback → CPU only (`WILDCAT_N_GPU_LAYERS=0`)
