# HIP / WMMA kernels — Part V (W11–W12) / HIP 與 WMMA kernel

**English.** These kernels port the PEPS inference path to AMD HIP and accelerate
the MLP decoder with **WMMA** (Wave Matrix Multiply-Accumulate) intrinsics. They
are written for **two RDNA generations** and merged into one repo:

| Box | GPU | arch | build flag |
|---|---|---|---|
| **B** | 4× Navi 48 | RDNA 4 | `--offload-arch=gfx1201` |
| **A** | Radeon 8060S | RDNA 3.5 | `--offload-arch=gfx1151` |

**繁體中文.** 這些 kernel 把 PEPS 推論路徑移植到 AMD HIP,並用 **WMMA** intrinsics
加速 MLP 解碼器。針對**兩個 RDNA 世代**撰寫並融合到同一個 repo。

## Files / 檔案
- `fused_peps_kernel.hip` — encoder-sample + MLP fused into one kernel (teaching
  example of kernel fusion). 編碼取樣 + MLP 融合成單一 kernel(kernel fusion 教材)。
- `wmma_mlp.hip` — WMMA-accelerated MLP matmul, per-ISA fragment sizes.
  WMMA 加速的 MLP 矩陣乘,依 ISA 選 fragment 大小。
- `bench_latency.sh` — build + run on the local box, print ms latency.

## WMMA note / WMMA 注意
Both RDNA 3.5 (`gfx1151`) and RDNA 4 (`gfx1201`) expose the `__builtin_amdgcn_wmma_*`
intrinsics, but the supported fragment shapes and dtypes differ. RDNA 4 adds
faster/int8 paths used by RTXNTC. The kernels select the intrinsic at compile time
via the `__gfx1201__` / `__gfx1151__` predefined macros. The paper's ms numbers are
RDNA 4; the RDNA 3.5 build is a comparison point, **not** a reproduction.

RDNA 3.5 與 RDNA 4 都提供 `__builtin_amdgcn_wmma_*` intrinsics,但支援的 fragment
形狀與 dtype 不同;RDNA 4 多了 RTXNTC 用的 int8 快路徑。kernel 以編譯期巨集選擇
intrinsic。論文 ms 數字為 RDNA 4;RDNA 3.5 版本是對照點,**非**重現。

## Build / 編譯
```bash
# on Box B (RDNA4):
hipcc --offload-arch=gfx1201 wmma_mlp.hip -o wmma_mlp && ./wmma_mlp
# on Box A (RDNA3.5):
hipcc --offload-arch=gfx1151 wmma_mlp.hip -o wmma_mlp && ./wmma_mlp
# or:
bash bench_latency.sh
```
