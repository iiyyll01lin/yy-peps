#!/usr/bin/env bash
# Build an LDS probe. Separate from build_kernel.sh because these have to build
# for CDNA too, where that kernel cannot go.
#
# On the RDNA4 box the distro's HIP 5.7 headers shadow the ROCm 7 ones, so the
# same symlink shim build_kernel.sh explains is needed here. Where hipcc works
# unaided (the CDNA container), it is used directly.
#
# The source is selectable because there are three of these now and the census
# had no build path at all, which left the one measurement carrying the granule
# result reproducible only from memory. The output name follows the source name
# for the reason build_kernel.sh gives about its variants: two probes must not
# be able to overwrite each other and be confused in a receipt.
set -euo pipefail

ARCH="${1:-gfx1201}"
SRC="${2:-lds_occupancy_probe}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/hip/build/${SRC}_$ARCH"
mkdir -p "$REPO/hip/build"

if [ ! -f "$REPO/hip/$SRC.hip" ]; then
  echo "no such probe source: hip/$SRC.hip" >&2
  exit 2
fi

# Word-split on purpose: lds_static_probe needs -DPROBE_LDS_BYTES.
read -r -a EXTRA <<< "${PEPS_EXTRA_FLAGS:-}"

# Guarded rather than piped: pipefail turns a missing .git into an abort, so an
# export without history would fail to build instead of building unwarned.
if git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  if [ -n "$(git -C "$REPO" status --porcelain -- "hip/$SRC.hip")" ]; then
    echo "WARNING: hip/$SRC.hip has uncommitted changes; the receipt will not be reproducible" >&2
  fi
fi

if [ -x "${ROCM_PATH:-/opt/rocm}/bin/amdclang++" ]; then
  ROCM="${ROCM_PATH:-/opt/rocm}"
  SHIM="${TMPDIR:-/tmp}/peps-probe-include-$ARCH"
  mkdir -p "$SHIM"
  for entry in "$ROCM"/include/*; do ln -sf "$entry" "$SHIM/"; done
  set -x
  "$ROCM/bin/amdclang++" -x hip -O2 -I"$SHIM" --offload-arch="$ARCH" \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    "$REPO/hip/$SRC.hip" \
    -L"$ROCM/lib" -lamdhip64 -o "$OUT"
  set +x
else
  set -x
  hipcc --offload-arch="$ARCH" -O2 \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    "$REPO/hip/$SRC.hip" -o "$OUT"
  set +x
fi

echo "built $OUT"
sha256sum "$REPO/hip/$SRC.hip"
sha256sum "$OUT"
