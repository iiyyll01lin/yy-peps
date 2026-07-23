#!/usr/bin/env bash
set -euo pipefail

artifact="${1:-table2}"
shift || true

case "${artifact}" in
  table2|sweep) ;;
  *)
    echo "unknown texture artifact: ${artifact}" >&2
    exit 2
    ;;
esac

python_bin="${PEPS_PYTHON:-.venv/bin/python}"
output_root="${PEPS_OUTPUT_ROOT:-results}"
receipt="${output_root}/texture_repro/dataset_verification.json"
log_root="${output_root}/work/texture-repro/launch-logs/${artifact}"
mkdir -p "${log_root}"

if [[ "${PEPS_SKIP_TEXTURE_VERIFY:-0}" != "1" ]]; then
  "${python_bin}" -m experiments.texture_repro manifest \
    --verify-files \
    --output "${receipt}"
fi

extra=("$@")
pids=()
for rank in 0 1 2 3; do
  HIP_VISIBLE_DEVICES="${rank}" \
    "${python_bin}" -m experiments.texture_repro run \
      --artifact "${artifact}" \
      --rank "${rank}" \
      --world-size 4 \
      --device cuda:0 \
      --output-root "${output_root}" \
      --verification-receipt "${receipt}" \
      --allow-protocol-assumptions \
      "${extra[@]}" \
      >"${log_root}/rank-${rank}.log" 2>&1 &
  pids+=("$!")
done

terminate_children() {
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
trap - INT TERM

"${python_bin}" -m experiments.texture_repro status \
  --artifact "${artifact}" \
  --output-root "${output_root}" \
  --output "${output_root}/texture_repro/${artifact}_status.json"
"${python_bin}" -m experiments.texture_repro report \
  --output-root "${output_root}" \
  --destination-dir "${output_root}/texture_repro"

if [[ "${artifact}" == "table2" ]]; then
  "${python_bin}" -m experiments.texture_repro figure8 \
    --device cuda:0 \
    --output-root "${output_root}" \
    --verification-receipt "${receipt}"
fi

exit "${status}"
