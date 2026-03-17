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

## Quick Start

Three steps, each done once (except the last):

```bash
# 1. One-time: generate CDI spec for GPU injection
mkdir -p ~/.config/cdi
nvidia-ctk cdi generate --output=$HOME/.config/cdi/nvidia.yaml
sudo mkdir -p /etc/cdi && sudo cp ~/.config/cdi/nvidia.yaml /etc/cdi/

# 2. One-time: build the container image (~10-15 min, CUDA compile)
cd /home/pjaganna/Software/wildcat
./build.sh

# 3. One-time: download the model (Qwen3.5-4B GGUF from HuggingFace/unsloth)
export MODEL_PATH=$(./fetch-model.sh)

# 4. Start the container
./run.sh
```

`run.sh` checks the image and model exist, starts the container, and waits for
ms-inspect to be ready. The pipeline does **not** start automatically — visit the
UI to confirm your settings and start the run.

### Starting the pipeline

Once the container is up, open `http://localhost:8181/start`. The waiting screen
shows the configured Measurement Set path and the predicted workflow ID. Click
**Start Pipeline** when ready.

To skip the UI gate (e.g. in CI or automated runs):

```bash
# Via env var
WILDCAT_AUTOSTART=1 ./run.sh

# Via CLI flag (passed through supervisord env)
# set WILDCAT_AUTOSTART=1 in the podman run -e args, or pass --autostart to wildcat directly
```

## Watching a run

Once the pipeline has started, open these in your browser:

| URL | What you see |
|-----|-------------|
| `http://localhost:8181/start` | Pre-start waiting screen — MS path confirmation and Start Pipeline button |
| `http://localhost:8181/pipeline/1` | Live pipeline monitor — phase outputs, LLM decisions, job status |
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
| `$MODEL_PATH` (from `fetch-model.sh`) | `/models/model.gguf` | LLM model weights |
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
