# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**wildcat** is an agentic orchestration layer for radio interferometry data reduction. It connects a local LLM (via llama.cpp or Ollama) to the `ms-inspect` MCP server, runs CASA scripts for data processing, and presents a human checkpoint UI at the end of the inspection phases — all inside a single container.

The core design principle: automate everything except the scientific judgment call. The human expert decides whether to proceed to imaging or loop back to calibration; wildcat handles everything else.

## Development commands

Uses [pixi](https://pixi.sh) for environment management (Python 3.12).

```bash
pixi run test          # pytest tests/ -v
pixi run lint          # ruff check src/ tests/
pixi run check         # ruff check + ruff format --check
pixi run start         # wildcat entrypoint
```

Run the orchestrator directly:
```bash
wildcat --ms /path/to/data.ms --config config.toml
wildcat --ms /path/to/data.ms --workflow-id 3   # resume existing workflow
# or: WILDCAT_MS_PATH=/path/to/data.ms wildcat
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
./run.sh   # builds Podman container, runs smoke test, waits for HUMAN_CHECKPOINT
```

## Architecture

```
main.py  ──→  Orchestrator  ──→  MSInspectClient  (MCP over SSE)
                  │          ──→  LLMBackend        (OpenAI-compatible HTTP)
                  │          ──→  CASARunner         (subprocess + sentinel files)
                  │          ──→  StateDB            (SQLite)
                  └──→  FastAPI UI  (checkpoint UI served on :8081)
```

### Workflow state machine

```
IDLE → PHASE1_RUNNING → [PHASE2_RUNNING] → [PHASE3_RUNNING] → HUMAN_CHECKPOINT
                                                                    ├── IMAGING_PIPELINE
                                                                    └── CALIBRATION_LOOP
```

Phases 2 and 3 are wired but currently Phase 1 transitions directly to `HUMAN_CHECKPOINT`. All state lives in SQLite (`wildcat.db`); the LLM is stateless and invoked once per phase — this tolerates arbitrarily long CASA jobs without HTTP timeouts.

### Key modules

- `orchestrator.py` — main loop; dispatches to `_handle_phase()` and `_handle_checkpoint()`; parses LLM JSON decisions
- `state.py` — all SQLite access; the `Stage` enum defines every valid workflow state; `StateDB` is a context manager
- `llm.py` — `LLMBackend` starts/stops llama-server subprocess or connects to Ollama; `complete()` is the only LLM call site
- `tools.py` — `MSInspectClient` speaks MCP over SSE; `run_phase1/2/3()` fan out to the relevant tool groups
- `runner.py` — `CASARunner` spawns CASA subprocesses; `SentinelWatcher` (watchdog) fires an `asyncio.Event` on `.done`/`.failed` sentinel files instead of polling
- `config.py` — typed `WildcatConfig` dataclasses loaded from `config.toml` via stdlib `tomllib`; no raw TOML elsewhere
- `skills.py` — loads system prompt partials from the mounted `/skills/` volume, keyed by `Stage`
- `ui/app.py` — FastAPI app; `POST /checkpoint/{id}/decide` resolves the human checkpoint and sets the event

### LLM decision contract

Every LLM response must be a JSON object with exactly these keys:
```json
{
  "next_stage": "<Stage value>",
  "casa_script": "<python string or null>",
  "summary": "<2-5 sentences>",
  "reasoning": "<brief trace>"
}
```
The orchestrator validates this in `_parse_decision()` and enters `ERROR` state on malformed output.

### Container ports

| Port | Service |
|------|---------|
| 8000 | ms-inspect MCP (SSE) — internal |
| 8080 | llama-server (OpenAI-compatible) — internal |
| 8081 | Checkpoint UI (host-exposed) |

Host-side ports in `run.sh` are offset: 8100/8180/8181.

## Configuration

All settings in `config.toml`. Key sections: `[llm.llamacpp]`, `[llm.ollama]`, `[llm]` (`backend = "llamacpp" | "ollama"`), `[mcp]`, `[state]`, `[casa]`, `[ui]`, `[skills]`.

Switch backends by changing `[llm] backend` — no code changes required.

## Known constraints

- CUDA 12.9 is incompatible with the current llama.cpp build; container uses 12.6.3 base image (works with 12.9 host driver via CDI injection).
- ms-inspect is not on PyPI; it is pip-installed from the volume-mounted source at container startup (`entrypoint.sh`).
- `BUILD_SHARED_LIBS=OFF` is required for the llama.cpp static binary to resolve `libcuda.so.1` at runtime rather than link time.
