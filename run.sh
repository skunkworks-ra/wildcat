#!/bin/bash
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE=wildcat
MS_PATH=/home/pjaganna/Data/measurement_sets/3C129_1.ms
MS_INSPECT_SRC=/home/pjaganna/Software/data-analyst
SKILLS_SRC=/home/pjaganna/Software/data-analyst/.claude/skills/radio-interferometry
OUTPUT_DIR=/home/pjaganna/Data/wildcat-output

CONTAINER_MS=/data/ms/3C129_1.ms

# MODEL_PATH must be set externally — run: export MODEL_PATH=$(./fetch-model.sh)
MODEL_PATH="${MODEL_PATH:-}"

# ── Helpers ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

step() { echo -e "\n${GREEN}▶ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠  $*${NC}"; }
fail() { echo -e "${RED}✗  $*${NC}"; exit 1; }

# ── Step 1: Preflight ─────────────────────────────────────────────────────────
step "Preflight checks"

cd "$(dirname "$0")"

[ -d "$MS_PATH" ]        || fail "MS not found: $MS_PATH"
[ -d "$MS_INSPECT_SRC" ] || fail "ms-inspect source not found: $MS_INSPECT_SRC"
[ -d "$SKILLS_SRC" ]     || fail "Skills not found: $SKILLS_SRC"
mkdir -p "$OUTPUT_DIR"

podman image exists "$IMAGE" || fail "Image '$IMAGE' not found — run ./build.sh first"

[ -n "$MODEL_PATH" ] || fail "MODEL_PATH is not set — run: export MODEL_PATH=\$(./fetch-model.sh)"
[ -f "$MODEL_PATH" ] || fail "Model file not found: $MODEL_PATH — run: export MODEL_PATH=\$(./fetch-model.sh)"

echo "  MS:         $MS_PATH"
echo "  ms-inspect: $MS_INSPECT_SRC"
echo "  model:      $MODEL_PATH"
echo "  output:     $OUTPUT_DIR"

# ── Step 2: Start container ───────────────────────────────────────────────────
step "Starting container"

# Clean up any previous run
podman rm -f wildcat-test 2>/dev/null || true

MODEL_MOUNT_ARGS=(-v "$MODEL_PATH":/models/model.gguf:ro)

podman run -d \
    --name wildcat-test \
    --device /dev/dxg \
    -v /usr/lib/wsl:/usr/lib/wsl:ro \
    -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
    -e WILDCAT_MS_PATH="$CONTAINER_MS" \
    -v "$MS_INSPECT_SRC":/opt/ms-inspect:ro \
    -v "$SKILLS_SRC":/skills/radio-interferometry:ro \
    -v "$(dirname "$MS_PATH")":/data/ms:ro \
    -v "$OUTPUT_DIR":/data \
    "${MODEL_MOUNT_ARGS[@]+"${MODEL_MOUNT_ARGS[@]}"}" \
    -p 8100:8000 -p 8180:8080 -p 8181:8081 \
    "$IMAGE"

echo "  Container started: wildcat-test"

# ── Step 3: Wait for ms-inspect ───────────────────────────────────────────────
step "Waiting for ms-inspect to be ready on :8000"

for i in $(seq 1 60); do
    HTTP_CODE=$(curl -s --connect-timeout 2 --max-time 3 http://localhost:8100/sse -w "%{http_code}" -o /dev/null 2>/dev/null || true)
    if [ "$HTTP_CODE" = "200" ]; then
        echo "  ms-inspect is up (attempt $i)"
        break
    fi
    if [ "$i" -eq 60 ]; then
        warn "ms-inspect did not start in time — printing logs:"
        podman logs wildcat-test | tail -30
        echo ""
        warn "ms-inspect stderr:"
        podman exec wildcat-test cat /var/log/ms-inspect.log 2>/dev/null || true
        fail "ms-inspect failed to start"
    fi
    echo "  waiting... ($i/60)"
    sleep 2
done

# ── Step 4: List tools ────────────────────────────────────────────────────────
step "Available ms-inspect tools"

podman exec wildcat-test python3 - <<'EOF'
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def main():
    async with sse_client("http://localhost:8000/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            print(f"  {len(result.tools)} tools available:")
            for t in sorted(result.tools, key=lambda x: x.name):
                print(f"    {t.name}")

asyncio.run(main())
EOF

# ── Step 5: Smoke test — Phase 1 tools ───────────────────────────────────────
step "Smoke test: Phase 1 tools against $CONTAINER_MS"

podman exec wildcat-test python3 - <<EOF
import asyncio, json
from mcp import ClientSession
from mcp.client.sse import sse_client

TOOLS = """ms_observation_info ms_field_list ms_scan_list ms_scan_intent_summary ms_spectral_window_list ms_correlator_config""".split()
MS = "$CONTAINER_MS"

async def main():
    async with sse_client("http://localhost:8000/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for tool in TOOLS:
                try:
                    r = await session.call_tool(tool, {"ms_path": MS})
                    text = r.content[0].text if r.content else "{}"
                    data = json.loads(text)
                    print(f"  {tool} ... OK  {list(data.keys())[:3]}")
                except Exception as e:
                    print(f"  {tool} ... FAILED: {e}")

asyncio.run(main())
EOF

# ── Step 6: Deep print — observation info ────────────────────────────────────
step "Full output: ms_observation_info"

podman exec wildcat-test python3 - <<EOF
import asyncio, json
from mcp import ClientSession
from mcp.client.sse import sse_client

async def main():
    async with sse_client("http://localhost:8000/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            r = await session.call_tool("ms_observation_info", {"ms_path": "$CONTAINER_MS"})
            text = r.content[0].text if r.content else "{}"
            print(json.dumps(json.loads(text), indent=2))

asyncio.run(main())
EOF

# ── Step 7: Wait for wildcat to reach HUMAN_CHECKPOINT ───────────────────────
step "Waiting for wildcat to reach HUMAN_CHECKPOINT"

for i in $(seq 1 120); do
    STAGE=$(python3 -c "
import sqlite3
con = sqlite3.connect('/home/pjaganna/Data/wildcat-output/wildcat.db')
row = con.execute('SELECT stage FROM workflow ORDER BY id DESC LIMIT 1').fetchone()
print(row[0] if row else 'NONE')
" 2>/dev/null)
    echo "  stage: $STAGE ($i/120)"
    if [ "$STAGE" = "HUMAN_CHECKPOINT" ]; then
        # Verify checkpoint row exists
        HAS_CP=$(python3 -c "
import sqlite3
con = sqlite3.connect('/home/pjaganna/Data/wildcat-output/wildcat.db')
row = con.execute('SELECT id FROM checkpoints ORDER BY id DESC LIMIT 1').fetchone()
print(row[0] if row else 'none')
" 2>/dev/null)
        echo "  checkpoint id: $HAS_CP"
        break
    fi
    if [ "$i" -eq 120 ]; then
        warn "Timed out — wildcat logs:"
        podman exec wildcat-test cat /var/log/wildcat.log | tail -20
    fi
    sleep 5
done

# ── Done ──────────────────────────────────────────────────────────────────────
step "All checks passed"
echo ""
echo "  Checkpoint UI:  http://localhost:8181/checkpoint/1"
echo "  ms-inspect:     http://localhost:8100/mcp/v1/tools/list"
echo "  llama-server:   http://localhost:8180/health"
echo ""
echo "  Logs:  podman logs -f wildcat-test"
echo "  Shell: podman exec -it wildcat-test bash"
echo "  Stop:  podman rm -f wildcat-test"
