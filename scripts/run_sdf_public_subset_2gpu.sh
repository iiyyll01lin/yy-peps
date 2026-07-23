#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PEPS_PYTHON:-${ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment not found or not executable: ${PYTHON}" >&2
  exit 2
fi

processed_args=()
if [[ -n "${PEPS_SDF_PROCESSED_ROOT:-}" ]]; then
  processed_args=(--processed-root "${PEPS_SDF_PROCESSED_ROOT}")
fi

exec "${PYTHON}" -m experiments.sdf_public_subset launch \
  "${processed_args[@]}"
