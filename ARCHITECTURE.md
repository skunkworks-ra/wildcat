# wildcat — Architecture Reference
## Implementation detail: modules, DB schema, data flow, container

**Last revised:** 2026-03-25
**Design decisions and rationale:** see `DESIGN.md`.

---

## 1. Module Map

| Module | Responsibility |
|--------|---------------|
| `main.py` | Entry point. Parses CLI args, constructs `Orchestrator`, handles start gate. |
| `orchestrator.py` | Main loop. Dispatches to stage handlers, calls LLM, writes DB. All pipeline logic lives here. |
| `state.py` | All SQLite access. `StateDB` is a context manager. `Stage` enum defines every valid workflow state. |
| `llm.py` | `LLMBackend` — starts/stops llama-server subprocess or connects to Ollama. `complete()` is the only LLM call site. |
| `runner.py` | `CASARunner` — writes script to disk, spawns `python3 script.py`, awaits completion directly via asyncio subprocess. |
| `config.py` | Typed `WildcatConfig` dataclasses loaded from `config.toml` via stdlib `tomllib`. No raw TOML elsewhere. |
| `skills.py` | Loads system prompt partials from the mounted `/skills/` volume, keyed by `Stage`. |
| `tools.py` | `MSInspectClient` — speaks MCP over SSE. `run_phase1/2/3()` fan out to the relevant tool groups. |
| `ui/app.py` | FastAPI app. All UI routes. `POST /checkpoint/{id}/decide` resolves the human checkpoint. |

---

## 2. SQLite Schema

Database path: `/data/wildcat.db` (configurable via `config.toml [state] db_path`).

### `workflow`
| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Auto-increment |
| `ms_path` | TEXT | Path to measurement set inside container |
| `stage` | TEXT | Current `Stage` enum value |
| `workflow_config` | TEXT | JSON blob: `{polcal, aggressive_flagging, _preflag_iterations, ...}` |
| `created_at` | TEXT | ISO timestamp |
| `updated_at` | TEXT | Updated on every `transition()` call |

### `tool_outputs`
| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | |
| `workflow_id` | INTEGER | FK → workflow |
| `phase` | INTEGER | 1, 2, or 3 |
| `tool_name` | TEXT | e.g. `ms_field_list` |
| `output_json` | TEXT | Full MCP response envelope as JSON |
| `collected_at` | TEXT | ISO timestamp |

Written once per tool call during inspection phases. Never rewritten. Read by calibration stage handlers to fill CASA script placeholders.

### `jobs`
| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | |
| `workflow_id` | INTEGER | FK → workflow |
| `stage` | TEXT | Stage value at time of submission |
| `script_path` | TEXT | Absolute path to `.py` file on disk |
| `status` | TEXT | `running` → `done` or `failed` |
| `stdout` | TEXT | Full CASA stdout. Contains `WILDCAT_METRICS:` line. |
| `stderr` | TEXT | Full CASA stderr. CASA WARN/SEVERE messages land here. |
| `plots` | TEXT | JSON list of plot file paths (future use) |
| `submitted_at` | TEXT | |
| `completed_at` | TEXT | Set when status transitions to done/failed |

### `llm_decisions`
| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | |
| `workflow_id` | INTEGER | FK → workflow |
| `stage` | TEXT | Stage at time of decision |
| `decision_json` | TEXT | Full parsed JSON from LLM |
| `model` | TEXT | Model identifier from response |
| `created_at` | TEXT | |

One row per LLM call. Written immediately after `_parse_decision()` succeeds.

### `checkpoints`
| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | |
| `workflow_id` | INTEGER | FK → workflow |
| `questions_json` | TEXT | JSON array from LLM `checkpoint_questions` field |
| `human_route` | TEXT | NULL until human decides: `proceed`, `loop_back`, or `exit` |
| `created_at` | TEXT | |

---

## 3. Orchestrator Data Flow per Stage

### Inspection Phases (1–3)

```
orchestrator._handle_phase(workflow_id, phase)
  │
  ├─ tools.run_phaseN(ms_path)          → MCP over SSE → ms-inspect
  │    writes: tool_outputs rows
  │
  ├─ (phase 3 only) _apply_deterministic_config()
  │    reads:  tool_outputs (ms_pol_cal_feasibility, ms_antenna_flag_fraction)
  │    writes: workflow.workflow_config
  │
  ├─ LLM call
  │    system: skills partial + JSON schema + /no_think
  │    user:   phase tool outputs (JSON)
  │    writes: llm_decisions row
  │
  └─ db.transition(next_stage)
       writes: workflow.stage, workflow.updated_at
```

### CALIBRATION_PREFLAG — Iteration 1

```
orchestrator._handle_calibration_stage(workflow_id, CALIBRATION_PREFLAG)
  │
  ├─ increment _preflag_iterations in workflow_config
  ├─ check cap (> 3) → escalate to CALIBRATION_CHECKPOINT if hit
  │
  ├─ _load_all_tool_outputs(tools={ms_field_list, ms_spectral_window_list, ms_correlator_config})
  │    reads: tool_outputs table
  │
  ├─ LLM call  [_format_calibration_prompt]
  │    system: skills partial + JSON schema + /no_think
  │    user:   tool outputs + workflow config + script template
  │    writes: llm_decisions row
  │
  ├─ _run_casa_job(script=decision["casa_script"])
  │    writes: script to /data/jobs/{id}/CALIBRATION_PREFLAG.py
  │    writes: jobs row (script_path, status=running)
  │    awaits: python3 subprocess
  │    writes: jobs row (stdout, stderr, status=done/failed, completed_at)
  │
  └─ db.transition(next_stage from LLM, or CALIBRATION_SOLVE if flag cap hit)
```

### CALIBRATION_PREFLAG — Iteration 2+

```
orchestrator._handle_calibration_stage(workflow_id, CALIBRATION_PREFLAG)
  │
  ├─ increment _preflag_iterations (now 2 or 3)
  ├─ check cap → escalate if hit
  │
  ├─ _get_previous_preflag_script()
  │    reads: jobs table (script_path of last done PREFLAG job)
  │    reads: script file from disk
  │
  ├─ _read_last_job_metrics()
  │    reads: jobs table (stdout of last completed job)
  │    parses: WILDCAT_METRICS: {...} line
  │
  ├─ LLM call  [_format_preflag_reentry_prompt]
  │    system: skills partial + JSON schema + /no_think
  │    user:   previous script + WILDCAT_METRICS + iteration count
  │            (NO tool outputs — static, already encoded in prior script)
  │    writes: llm_decisions row
  │
  ├─ _run_casa_job(...)
  └─ db.transition(next_stage)
```

### CALIBRATION_SOLVE

```
orchestrator._handle_solve_stage(workflow_id)
  │
  ├─ _build_solve_script()
  │    reads: tool_outputs (ms_field_list, ms_spectral_window_list,
  │                         ms_correlator_config, ms_antenna_flag_fraction,
  │                         ms_scan_list, ms_refant)
  │    fills: all {PLACEHOLDER} values deterministically
  │    NO LLM call
  │
  ├─ _run_casa_job(script)
  │    writes: /data/jobs/{id}/CALIBRATION_SOLVE.py
  │    awaits: python3 subprocess (delay + BP + gain calibration, ~4 min)
  │    writes: jobs row with stdout (contains WILDCAT_METRICS)
  │
  ├─ _read_last_job_metrics()  → {bp_flagged_frac, gain_flagged_frac, ...}
  │
  └─ threshold check (no LLM):
       bp < 0.20 AND gain < 0.15  → CALIBRATION_APPLY
       else AND preflag_iterations < cap → CALIBRATION_PREFLAG
       else (cap reached)         → CALIBRATION_APPLY
```

### CALIBRATION_APPLY

```
orchestrator._handle_apply_stage(workflow_id)
  │
  ├─ _build_apply_script()
  │    reads: tool_outputs (ms_field_list, ms_spectral_window_list,
  │                         ms_correlator_config)
  │    detects: polcal tables via os.path.exists → sets parang=True if present
  │    NO LLM call
  │
  ├─ _run_casa_job(script)
  │    writes: /data/jobs/{id}/CALIBRATION_APPLY.py
  │    awaits: python3 subprocess (applycal + rflag corrected, ~3 min)
  │
  ├─ LLM call  (metrics interpretation only — no script, no routing)
  │    system: skills partial + JSON schema + /no_think
  │    user:   WILDCAT_METRICS from SOLVE + APPLY
  │    writes: llm_decisions row
  │    writes: checkpoints row (questions_json, human_route=NULL)
  │
  └─ db.transition(CALIBRATION_CHECKPOINT)
```

---

## 4. WILDCAT_METRICS Protocol

Every CASA script emits exactly one line to stdout at successful completion:

```
WILDCAT_METRICS: {"stage": "CALIBRATION_PREFLAG", "overall_flag_frac": 0.12, ...}
```

The orchestrator parses this in `_read_last_job_metrics()`:
- Queries `jobs` table for the most recent completed job for the workflow
- Scans `stdout` for a line starting with `WILDCAT_METRICS:`
- Returns the parsed JSON dict (small — never the full stdout)

**This dict is the only job output the LLM ever sees.** Raw stdout stays in the DB for the UI only.

If a job fails (no metrics line emitted), `_run_casa_job` transitions to ERROR immediately.

### Metrics fields by stage

| Stage | Fields |
|-------|--------|
| CALIBRATION_PREFLAG | `overall_flag_frac`, `cal_flag_frac`, `n_chunks_flagged` |
| CALIBRATION_SOLVE | `bp_flagged_frac`, `gain_flagged_frac`, `n_antennas_lost`, `gain_solint1`, `gain_solint2`, `t_delay`, `t_bp`, `t_gain` |
| CALIBRATION_APPLY | `post_cal_flag_frac`, `target_flag_frac`, `n_antennas_lost` |
| POLCAL_SOLVE | `bp_flagged_frac`, `gain_flagged_frac`, `polindex_c0`, `polangle_c0_rad` |

---

## 5. MCP Tool Groups

ms-inspect serves tools over SSE at `:8000`. The MCP Python client handles the protocol.

| Phase | Tools called |
|-------|-------------|
| 1 | `ms_observation_info`, `ms_field_list`, `ms_scan_list`, `ms_scan_intent_summary`, `ms_spectral_window_list`, `ms_correlator_config` |
| 2 | `ms_antenna_list`, `ms_baseline_lengths`, `ms_elevation_vs_time`, `ms_parallactic_angle_vs_time`, `ms_shadowing_report`, `ms_antenna_flag_fraction` |
| 3 | `ms_rfi_channel_stats`, `ms_flag_summary`, `ms_refant`, `ms_pol_cal_feasibility` |

Tool outputs use the `{value, flag}` envelope from ms-inspect:
```json
{"value": <any>, "flag": "COMPLETE|INFERRED|PARTIAL|SUSPECT|UNAVAILABLE"}
```
`flag: "UNAVAILABLE"` means `value` is `null` — not a missing key. All orchestrator
helpers guard against this with `(x.get("value") or default)`, not `.get("value", default)`.

---

## 6. Container

### Ports

| Internal | Host (default) | Service |
|----------|---------------|---------|
| `:8000` | `PORT_MS_INSPECT` (8100) | ms-inspect MCP (SSE) |
| `:8080` | `PORT_LLAMA` (8180) | llama-server (OpenAI-compatible) |
| `:8081` | `PORT_UI` (8181) | wildcat UI |

### Volume Mounts

| Container path | Source | Mode |
|---------------|--------|------|
| `/opt/ms-inspect` | data-analyst repo | ro |
| `/skills/radio-interferometry` | data-analyst skills dir | ro |
| `/data/ms` | directory containing `.ms` file | ro |
| `/models/model.gguf` | GGUF model file | ro |
| `/data` | wildcat output dir (DB + job scripts) | rw |

### Build

```
Base: docker.io/nvidia/cuda:12.6.3-devel-ubuntu24.04
llama.cpp flags: -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_CUDA_ARCHITECTURES="61;75;86"
```

`BUILD_SHARED_LIBS=OFF` — static binary resolves `libcuda.so.1` at runtime via CDI injection rather than link time.

CUDA 12.9 host driver works at runtime; 12.6.3 is the build base because CUDA 12.9 CCCL macros break llama.cpp compilation.

### entrypoint.sh

Runs `pip install --upgrade -e /opt/ms-inspect[casa]` at every container start. This ensures the live ms-inspect source is always used but can upgrade Python deps (e.g. Starlette) to versions with breaking APIs. Pin versions in `pyproject.toml` if this becomes a problem.

### Scripts

| Script | Purpose |
|--------|---------|
| `build.sh` | Build container image — run once or on `Containerfile` changes |
| `fetch-model.sh` | Download default GGUF from HuggingFace; respects `HF_TOKEN` |
| `run.sh` | Start container with `WILDCAT_AUTOSTART=1`; polls DB until terminal state |
| `clean.sh` | Kill `wildcat-test` container and remove `wildcat.db` — idempotent |

---

## 7. UI Layer

| Route | Purpose |
|-------|---------|
| `/start` | Pre-start gate — MS path, predicted workflow ID, Start button. Bypassed when `WILDCAT_AUTOSTART=1`. |
| `/pipeline/{id}` | Live transparency monitor — tool outputs, LLM decisions, CASA artifacts. HTMX polling 1s. |
| `/checkpoint/{id}` | Human checkpoint form — calibration metrics, proceed/loop_back/exit. Only at CALIBRATION_CHECKPOINT. |
| `/logs` | Raw log stream via SSE tail of `/var/log/wildcat.log` |

Pipeline monitor polls `/pipeline/{id}/fragment` every 1s. All `<details>` elements have `open` attribute — HTMX replaces the DOM on every poll, collapsing any manually opened elements without it.

`POST /checkpoint/{id}/decide` writes `human_route` to the checkpoints table and sets the asyncio event that unblocks the orchestrator.
