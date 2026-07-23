#!/usr/bin/env bash
set -euo pipefail

artifact="${1:-table1}"
shift || true

case "${artifact}" in
  table1|table5|core-ablations|recipe-ablations|smoke|appendix-smoke) ;;
  *)
    echo "unknown image artifact: ${artifact}" >&2
    exit 2
    ;;
esac

python_bin="${PEPS_PYTHON:-.venv/bin/python}"
authorization_receipt=""
extra=()
while (( "$#" )); do
  case "$1" in
    --authorization-receipt)
      if (( "$#" < 2 )); then
        echo "--authorization-receipt requires a path" >&2
        exit 2
      fi
      authorization_receipt="$2"
      shift 2
      ;;
    *)
      extra+=("$1")
      shift
      ;;
  esac
done

if [[ "${artifact}" == "table1" ]]; then
  echo "Table 1 launch and recovery are disabled after the external-recovery incident; a separate full-reproduction gate must deliberately change the code-level interlock." >&2
  exit 3
elif [[ "${artifact}" != "smoke" && "${artifact}" != "appendix-smoke" ]]; then
  extra+=(--allow-protocol-assumptions)
fi

output_root="${PEPS_OUTPUT_ROOT:-results}"
log_root="${output_root}/work/image-repro/launch-logs/${artifact}"
mkdir -p "${log_root}"

pids=()
terminate_workers() {
  trap - HUP INT TERM
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  exit 143
}
trap terminate_workers HUP INT TERM

for rank in 0 1 2 3; do
  HIP_VISIBLE_DEVICES="${rank}" \
    "${python_bin}" -m experiments.image_repro run \
      --artifact "${artifact}" \
      --rank "${rank}" \
      --world-size 4 \
      --device cuda:0 \
      --output-root "${output_root}" \
      "${extra[@]}" \
      >"${log_root}/rank-${rank}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
trap - HUP INT TERM
exit "${status}"
