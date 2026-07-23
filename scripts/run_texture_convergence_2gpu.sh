#!/usr/bin/env bash
set -euo pipefail

python_bin="${PEPS_PYTHON:-.venv/bin/python}"
output_root="${PEPS_OUTPUT_ROOT:-results}"
receipt="${output_root}/texture_repro/dataset_verification.json"
wall_seconds="${PEPS_TEXTURE_PILOT_WALL_SECONDS:-7200}"
log_root="${output_root}/work/texture-repro/launch-logs/convergence-pilot-2gpu"
mkdir -p "${log_root}"
unset ROCR_VISIBLE_DEVICES

run_queue() {
  local physical_device="$1"
  shift
  local rank
  for rank in "$@"; do
    HIP_VISIBLE_DEVICES="${physical_device}" \
    CUDA_VISIBLE_DEVICES="${physical_device}" \
    timeout --signal=TERM --kill-after=120s "$((wall_seconds + 120))s" \
      "${python_bin}" -m experiments.texture_repro pilot-run \
      --output-root "${output_root}" \
      --rank "${rank}" \
      --world-size 4 \
      --device cuda:0 \
      --physical-device-index "${physical_device}" \
      --max-wall-seconds "${wall_seconds}" \
      --verification-receipt "${receipt}" \
      --allow-protocol-assumptions \
      >"${log_root}/rank-${rank}.log" 2>&1
  done
}

run_queue 0 0 2 &
queue_zero_pid="$!"
run_queue 1 1 3 &
queue_one_pid="$!"

terminate_queues() {
  kill "${queue_zero_pid}" "${queue_one_pid}" 2>/dev/null || true
}
trap terminate_queues INT TERM

status=0
if ! wait "${queue_zero_pid}"; then
  status=1
fi
if ! wait "${queue_one_pid}"; then
  status=1
fi
trap - INT TERM

"${python_bin}" -m experiments.texture_repro pilot-status \
  --output-root "${output_root}" \
  --output "${output_root}/texture_repro/convergence_pilot_progress.json"
"${python_bin}" -m experiments.texture_repro pilot-report \
  --output-root "${output_root}" \
  --output "${output_root}/texture_repro/convergence_pilot.json" \
  --csv-output "${output_root}/texture_repro/convergence_pilot_observations.csv"

exit "${status}"
