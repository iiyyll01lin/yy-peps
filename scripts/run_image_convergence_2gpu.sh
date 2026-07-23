#!/usr/bin/env bash
set -euo pipefail

python_bin="${PEPS_PYTHON:-.venv/bin/python}"
arguments=()
if [[ -n "${PEPS_PILOT_WALL_SECONDS:-}" ]]; then
  arguments+=(--wall-seconds "${PEPS_PILOT_WALL_SECONDS}")
fi

export HIP_VISIBLE_DEVICES="0,1"
export CUDA_VISIBLE_DEVICES="0,1"
unset ROCR_VISIBLE_DEVICES

exec "${python_bin}" -m experiments.image_convergence launch "${arguments[@]}"
