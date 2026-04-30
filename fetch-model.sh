#!/bin/bash
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
# Default: Gemma 4 E2B instruct GGUF from unsloth on HuggingFace
HF_REPO="${HF_REPO:-unsloth/gemma-4-E2B-it-GGUF}"
HF_FILE="${HF_FILE:-gemma-4-E2B-it-Q4_K_M.gguf}"
CACHE_DIR="${MODEL_CACHE_DIR:-$HOME/.cache/wildcat/models}"
MODEL_PATH="$CACHE_DIR/$HF_FILE"

# ── Helpers ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

step()  { echo -e "\n${GREEN}▶ $*${NC}" >&2; }
warn()  { echo -e "${YELLOW}⚠  $*${NC}" >&2; }
fail()  { echo -e "${RED}✗  $*${NC}" >&2; exit 1; }
info()  { echo -e "  $*" >&2; }

# ── Already cached? ───────────────────────────────────────────────────────────
if [ -f "$MODEL_PATH" ]; then
    SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
    warn "Model already cached ($SIZE): $MODEL_PATH"
    echo "$MODEL_PATH"
    exit 0
fi

mkdir -p "$CACHE_DIR"

# ── Download ──────────────────────────────────────────────────────────────────
step "Fetching $HF_FILE from $HF_REPO"
info "Destination: $MODEL_PATH"

# Prefer huggingface-cli if available; fall back to curl
if command -v huggingface-cli &>/dev/null; then
    info "Using huggingface-cli"
    huggingface-cli download "$HF_REPO" "$HF_FILE" --local-dir "$CACHE_DIR" >&2
    # huggingface-cli may nest under a subdirectory — find the file
    FOUND=$(find "$CACHE_DIR" -name "$HF_FILE" -not -path '*/.cache/*' | head -1)
    if [ -z "$FOUND" ]; then
        fail "Download completed but $HF_FILE not found under $CACHE_DIR"
    fi
    if [ "$FOUND" != "$MODEL_PATH" ]; then
        mv "$FOUND" "$MODEL_PATH"
    fi
else
    info "huggingface-cli not found — using curl"
    HF_URL="https://huggingface.co/$HF_REPO/resolve/main/$HF_FILE"
    curl -L --progress-bar -o "$MODEL_PATH" "$HF_URL" \
        ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} || {
        rm -f "$MODEL_PATH"
        fail "Download failed: $HF_URL"
    }
fi

# ── Verify ────────────────────────────────────────────────────────────────────
[ -f "$MODEL_PATH" ] || fail "Model file missing after download: $MODEL_PATH"
SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
info "Downloaded ($SIZE): $MODEL_PATH"

# Print the path to stdout so callers can capture it
echo "$MODEL_PATH"
