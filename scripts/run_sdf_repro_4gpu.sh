#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PEPS_PYTHON:-${ROOT}/.venv/bin/python}"
WORK_ROOT="${PEPS_SDF_WORK_ROOT:-${ROOT}/results/work/sdf-repro}"
OUTPUT_ROOT="${PEPS_SDF_OUTPUT_ROOT:-${ROOT}/results/sdf_repro}"
TABLE3="${ROOT}/configs/paper/sdf/table3_mape.toml"
TABLE6="${ROOT}/configs/paper/sdf/table6_l1.toml"
TABLE4="${ROOT}/configs/paper/sdf/table4_deferred.toml"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment not found or not executable: ${PYTHON}" >&2
  exit 2
fi

gpu_count="$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
if [[ "${gpu_count}" -lt 4 ]]; then
  echo "The full SDF matrix requires four visible GPUs; found ${gpu_count}." >&2
  exit 2
fi

processed_args=()
if [[ -n "${PEPS_SDF_PROCESSED_ROOT:-}" ]]; then
  processed_args=(--processed-root "${PEPS_SDF_PROCESSED_ROOT}")
fi

mkdir -p "${WORK_ROOT}/logs" "${OUTPUT_ROOT}"

"${PYTHON}" -m experiments.sdf_repro validate \
  --config "${TABLE3}" \
  --config "${TABLE6}" \
  --table4-config "${TABLE4}" \
  --output "${OUTPUT_ROOT}/volume_validation.json" \
  --table4-receipt "${OUTPUT_ROOT}/table4_deferred_auth_required.json" \
  "${processed_args[@]}" \
  >"${WORK_ROOT}/logs/preflight.json"

"${PYTHON}" -m experiments.sdf_repro estimate \
  --config "${TABLE3}" \
  --config "${TABLE6}" \
  >"${OUTPUT_ROOT}/cost.json"

pids=()
terminate_workers() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap terminate_workers INT TERM

for rank in 0 1 2 3; do
  HIP_VISIBLE_DEVICES="${rank}" \
  ROCR_VISIBLE_DEVICES="${rank}" \
  "${PYTHON}" -m experiments.sdf_repro run \
    --config "${TABLE3}" \
    --config "${TABLE6}" \
    --rank "${rank}" \
    --world-size 4 \
    --device cuda:0 \
    --work-root "${WORK_ROOT}" \
    --render-root "${OUTPUT_ROOT}/renders" \
    --skip-volume-checksums \
    "${processed_args[@]}" \
    >"${WORK_ROOT}/logs/rank-${rank}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
trap - INT TERM
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one SDF shard failed; checkpoints are preserved under ${WORK_ROOT}." >&2
  exit 1
fi

"${PYTHON}" -m experiments.sdf_repro aggregate \
  --config "${TABLE3}" \
  --config "${TABLE6}" \
  --work-root "${WORK_ROOT}" \
  --output-root "${OUTPUT_ROOT}"

echo "SDF three-shape artifacts written to ${OUTPUT_ROOT}"
echo "Pitted Stonefish/Table 4 remains deferred_auth_required."
