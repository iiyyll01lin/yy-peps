#!/usr/bin/env bash
# bench_latency.sh — build + run the HIP kernels on the local box, print latency.
# 繁體中文:在本機編譯並執行 HIP kernel,印出延遲。自動偵測 GPU 架構。
set -e

# Detect gfx arch from rocminfo (fallback to gfx1201).
ARCH=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || echo gfx1201)
echo "== Building for $ARCH =="

hipcc --offload-arch="$ARCH" wmma_mlp.hip        -o wmma_mlp
hipcc --offload-arch="$ARCH" fused_peps_kernel.hip -o fused_peps

echo "== WMMA MLP (16x16x16 tiles) =="
HIP_VISIBLE_DEVICES=0 ./wmma_mlp 4096 64 64 200

echo "== Fused sample+MLP =="
HIP_VISIBLE_DEVICES=0 ./fused_peps 262144 200

echo
echo "Arch: $ARCH  (gfx1201=RDNA4 / gfx1151=RDNA3.5)"
echo "Note: paper's ms figures target RDNA4; RDNA3.5 numbers are a comparison point."
