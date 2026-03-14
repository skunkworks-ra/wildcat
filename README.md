# wildcat

Agentic orchestration layer for radio interferometry reduction.

Wires a local LLM (Qwen3.5-4B via llama.cpp) to the ms-inspect MCP server,
a CASA script runner, a SQLite state machine, and a human checkpoint UI —
all inside a single container.

**Status:** Phase 1 end-to-end working against real MS data (3C129_1.ms).

---

## Requirements

- Podman
- NVIDIA GPU (tested: RTX 3080 Laptop, sm_86)
- nvidia-container-toolkit + CDI spec at `/etc/cdi/nvidia.yaml`
- ms-inspect source at `/home/pjaganna/Software/data-analyst`
- Qwen3.5-4B GGUF (cached by llama.cpp at `~/.cache/llama.cpp/`)

## Quick Start

```bash
# One-time: generate CDI spec for GPU injection
mkdir -p ~/.config/cdi
nvidia-ctk cdi generate --output=$HOME/.config/cdi/nvidia.yaml
sudo mkdir -p /etc/cdi && sudo cp ~/.config/cdi/nvidia.yaml /etc/cdi/

# Build and run
cd /home/pjaganna/Software/wildcat
./run.sh
```

`run.sh` builds the container, starts it, waits for ms-inspect to be ready,
runs Phase 1 smoke tests, then polls until the workflow reaches HUMAN_CHECKPOINT.

## Watching a run

Once the container is up (after the ms-inspect readiness check in `run.sh`),
open these in your browser:

| URL | What you see |
|-----|-------------|
| `http://localhost:8181/logs` | Live log stream — tool calls, LLM inference, stage transitions, colour-coded in real time |
| `http://localhost:8181/checkpoint/1` | Human checkpoint UI — LLM summary, key metrics table, imaging/calibration decision |

The log stream auto-scrolls and colour-codes by line type: stage transitions
(green), LLM calls (purple), tool calls (orange), warnings (yellow), errors (red).
Uncheck **auto-scroll** to freeze the view while a run is in progress.

To watch from the terminal instead:

```bash
podman exec wildcat-test tail -f /var/log/wildcat.log
```

## Ports

| Port (host) | Service |
|-------------|---------|
| 8100 | ms-inspect MCP (SSE) |
| 8180 | llama-server (OpenAI-compatible) |
| 8181 | Checkpoint UI + log stream |

## Configuration

`config.toml` — all settings. Key options:

```toml
[llm]
backend = "llamacpp"   # or "ollama"

[llm.llamacpp]
model_path   = "/models/model.gguf"   # mounted at runtime
ctx_size     = 32768
n_gpu_layers = 99

[llm.ollama]
base_url = "http://localhost:11434/v1"
model    = "qwen3:4b"

[mcp]
base_url = "http://localhost:8000"

[state]
db_path = "/data/wildcat.db"          # persisted to output volume
```

## Volume Mounts

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `data-analyst/` | `/opt/ms-inspect` | ms-inspect source |
| `data-analyst/.claude/skills/radio-interferometry/` | `/skills/radio-interferometry` | LLM skill partials |
| `Data/measurement_sets/` | `/data/ms` | Measurement sets (read-only) |
| `~/.cache/llama.cpp/...gguf` | `/models/model.gguf` | LLM model weights |
| `Data/wildcat-output/` | `/data` | SQLite DB + job outputs |

## Inspecting state

```bash
# Workflow history
python3 -c "
import sqlite3
con = sqlite3.connect('/home/pjaganna/Data/wildcat-output/wildcat.db')
for r in con.execute('SELECT id, stage, updated_at FROM workflow ORDER BY id DESC LIMIT 5'):
    print(r)
"

# Shell into container
podman exec -it wildcat-test bash

# Stop
podman rm -f wildcat-test
```

## Architecture

See `DESIGN.md` for full architecture, state machine, and known issues.
