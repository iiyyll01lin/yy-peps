#!/usr/bin/env bash
# Build the LDS occupancy probe. Separate from build_kernel.sh because this one
# has to build for CDNA too, where that kernel cannot go.
#
# On the RDNA4 box the distro's HIP 5.7 headers shadow the ROCm 7 ones, so the
# same symlink shim build_kernel.sh explains is needed here. Where hipcc works
# unaided (the CDNA container), it is used directly.
set -euo pipefail

ARCH="${1:-gfx1201}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/hip/build/lds_probe_$ARCH"
mkdir -p "$REPO/hip/build"

if [ -x "${ROCM_PATH:-/opt/rocm}/bin/amdclang++" ]; then
  ROCM="${ROCM_PATH:-/opt/rocm}"
  SHIM="${TMPDIR:-/tmp}/peps-probe-include-$ARCH"
  mkdir -p "$SHIM"
  for entry in "$ROCM"/include/*; do ln -sf "$entry" "$SHIM/"; done
  set -x
  "$ROCM/bin/amdclang++" -x hip -O2 -I"$SHIM" --offload-arch="$ARCH" \
    "$REPO/hip/lds_occupancy_probe.hip" \
    -L"$ROCM/lib" -lamdhip64 -o "$OUT"
  set +x
else
  set -x
  hipcc --offload-arch="$ARCH" -O2 \
    "$REPO/hip/lds_occupancy_probe.hip" -o "$OUT"
  set +x
fi

echo "built $OUT"
sha256sum "$OUT"
