#!/bin/bash
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE=wildcat

# ── Helpers ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

step() { echo -e "\n${GREEN}▶ $*${NC}"; }
fail() { echo -e "${RED}✗  $*${NC}"; exit 1; }

# ── Build ─────────────────────────────────────────────────────────────────────
step "Building container image (first run: ~10-15 min — CUDA compile)"

cd "$(dirname "$0")"
podman build -t "$IMAGE" . 2>&1 | while IFS= read -r line; do
    echo "  [build] $line"
done

echo ""
echo "  Image built: $IMAGE"
echo "  Run ./fetch-model.sh to download the default model, then ./run.sh to start."
