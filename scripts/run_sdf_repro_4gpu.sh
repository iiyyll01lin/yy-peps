#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "The four-GPU SDF launcher is disabled for this evidence phase." >&2
echo "Delegating to the manifest-backed physical-GPU-2/3 public subset." >&2
exec bash "${ROOT}/scripts/run_sdf_public_subset_2gpu.sh"
