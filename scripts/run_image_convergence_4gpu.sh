#!/usr/bin/env bash
set -euo pipefail

echo "The convergence pilot is restricted to physical GPUs 0 and 1." >&2
exec bash "$(dirname "$0")/run_image_convergence_2gpu.sh" "$@"
