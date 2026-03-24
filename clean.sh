#!/bin/bash
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
CONTAINER=wildcat-test
OUTPUT_DIR=/home/pjaganna/Data/wildcat-output

# ── Helpers ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

step() { echo -e "\n${GREEN}▶ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠  $*${NC}"; }

# ── Stop container ────────────────────────────────────────────────────────────
step "Stopping container"

if podman container exists "$CONTAINER" 2>/dev/null; then
    podman rm -f "$CONTAINER"
    echo "  Removed: $CONTAINER"
else
    warn "Container '$CONTAINER' not running — skipping"
fi

# ── Clear database ────────────────────────────────────────────────────────────
step "Clearing database"

DB="$OUTPUT_DIR/wildcat.db"
if [ -f "$DB" ]; then
    rm -f "$DB"
    echo "  Removed: $DB"
else
    warn "Database not found — skipping"
fi

echo -e "\n  Ready. Run ./run.sh to start fresh."
