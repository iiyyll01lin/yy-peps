#!/usr/bin/env bash
# Build the fused PEPS kernel against the ROCm toolchain actually installed.
#
# Why the include shim exists:
#
# This machine carries Ubuntu's ROCm 5.7.1 packages (libamdhip64-dev) alongside
# a ROCm 7.2.3 install under /opt. The distro package puts HIP 5.7 headers in
# /usr/include/hip, and those headers reference __AMDGCN_WAVEFRONT_SIZE, which
# clang 22 no longer predefines. A plain build therefore fails with 19 errors.
#
# The obvious fixes do not work. --rocm-path and --hip-path do not reorder the
# search list, and -I/opt/rocm/include is silently dropped: clang resolves it to
# the same real directory the HIP driver already appends *after* /usr/include,
# deduplicates the two, and keeps the later position. `amdclang++ -x hip -E -v`
# shows /usr/include ahead of /opt/rocm-7.2.3/.../include with no trace of the
# -I. -nogpuinc reorders correctly but also drops headers the runtime needs.
#
# So the shim is a real directory of symlinks. Its own path differs from the
# ROCm include directory, so clang cannot deduplicate it away, and it lands
# ahead of /usr/include where it belongs. No sudo, no system change, and
# removing libamdhip64-dev is not an option because hipcc, librocblas-dev and
# four other distro packages depend on it.
set -euo pipefail

ROCM="${ROCM_PATH:-/opt/rocm}"
ARCH="${1:-gfx1201}"
# Optional variant name. It only changes the output filename, so two
# builds of the same source with different -D flags cannot overwrite
# each other and cannot be confused in a receipt.
VARIANT="${2:-}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHIM="${TMPDIR:-/tmp}/peps-rocm-include-$ARCH"
OUT="$REPO/hip/build/fused_peps_$ARCH${VARIANT:+_$VARIANT}"
# Word-split on purpose: this carries several -D flags.
read -r -a EXTRA <<< "${PEPS_EXTRA_FLAGS:-}"

rm -rf "$SHIM"
mkdir -p "$SHIM" "$REPO/hip/build"
for entry in "$ROCM"/include/*; do ln -s "$entry" "$SHIM/"; done

SHA="$(git -C "$REPO" rev-parse --short HEAD)"
DIRTY="$(git -C "$REPO" status --porcelain -- hip/ | wc -l)"
if [ "$DIRTY" -ne 0 ]; then
  echo "WARNING: hip/ has uncommitted changes; the receipt will not be reproducible" >&2
fi

set -x
"$ROCM/bin/amdclang++" -x hip -O3 -DNDEBUG \
  -I"$SHIM" \
  --offload-arch="$ARCH" \
  -DPEPS_GIT_SHA="\"$SHA\"" \
  -DPEPS_TARGET_ISA="\"$ARCH\"" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  "$REPO/hip/fused_peps_kernel.hip" \
  -L"$ROCM/lib" -lamdhip64 \
  -o "$OUT"
set +x

"$ROCM/bin/roc-obj-ls" "$OUT" | sed -n 2p
echo "built $OUT"
sha256sum "$OUT"
