#!/usr/bin/env bash
# Clean-build, parity-check, and HIP-event benchmark the paper-scale decoder.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}
else
  PYTHON=${PYTHON:-python3}
fi

exec "$PYTHON" "$SCRIPT_DIR/benchmark.py" "$@"
