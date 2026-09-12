#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PORT="${1:-8765}"

python3 "$SCRIPT_DIR/progress_dashboard.py" \
  --root "$WORKSPACE_ROOT" \
  --state-dir "$WORKSPACE_ROOT/.progress_tasks" \
  --port "$PORT"

