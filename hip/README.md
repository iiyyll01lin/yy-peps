# HIP / WMMA kernels — Part V (W11–W12)

The primary path is now an **integrated paper-workload reference**, not the old
“one grid sample + first MLP layer” component benchmark.

## Integrated path

`fused_peps_kernel.hip` executes, in one kernel:

1. the paper projection with layout `(x, S_1..S_L, C_1..C_L)` and
   `phi_i = 2^i pi`;
2. every point's bilinear sample from one shared channel-first grid;
3. baseline, concat PEPS, or paper Algorithm 1 Pink aggregation;
4. a complete four-Linear decoder: **three hidden layers** plus output.

Two implementations share the same fixture contract:

- `fp32`: one-thread scalar correctness oracle;
- `fp16`: one-wave fused rocWMMA path. It builds selected features, runs all
  four Linear layers, applies fp32 biases/GELU, and writes only RGB globally.

The workload geometry is 1024×1024 output and grid, 16 grid features, three
frequencies, three 64-wide hidden layers, GELU, and RGB output.

```bash
# parity fixtures
./fused_peps fixture fp32 in.bin out.bin
./fused_peps fixture fp16 in.bin out.bin

# scalar correctness-reference timing
./fused_peps workload baseline 1024 20
./fused_peps workload peps     1024 20
./fused_peps workload pink     1024 20

# integrated fp16/rocWMMA timing with warmup and distribution statistics
./fused_peps benchmark bi-grid 1024 20 100
./fused_peps benchmark grid-peps-3f 1024 20 100
./fused_peps benchmark grid-pink-peps-3f 1024 20 100
```

Both implementations have baseline/concat/Pink binary parity fixtures. The
benchmark emits JSON receipts with `comparable_to_paper=false`; paper timings
are included only as sourced external references, never as local measurements.
The current fp16 path is functionally complete but still requires performance
work before a repeated 1024² run is practical.

整合路徑包含 projection、所有 shared-grid samples、concat / paper-exact Pink、
三個 hidden layers 與輸出層。scalar fp32 與 fused fp16/rocWMMA 皆完成 parity;
benchmark receipt 固定 `comparable_to_paper=false`,不得冒充 paper comparison。

## Parity fixtures

`tests/test_hip_parity.py` builds the real binary when HIP is available and sends
identical baseline, concat, and Pink fixtures through both precision paths. The
CPU reference is cross-checked against `Projector`, `GridEncoder`,
`ConcatAggregator`, `PinkAggregator`, and the four-layer `MLP`.

The integrated fixture header is 11 little-endian int32 values:

`magic, schema, mode, C, H, W, N, L, hidden, output, activation`

It is followed by float32 `grid, coords, W1,b1,W2,b2,W3,b3,W4,b4`. Matrices use
`[input, output]` row-major layout. `hip/export_fixture.py` is the canonical
exporter for random fixtures and actual PyTorch Grid/PEPS models; it also writes
hashes/manifests and portable weight archives.

## Supplementary diagnostics

- `./fused_peps micro 262144 1000` retains the original sample + first-layer
  microbenchmark.
- `wmma_mlp.hip` retains isolated fp16 and int8 16×16×16 rocWMMA GEMM fixtures
  and microbenchmarks.

These diagnostics locate component regressions; they are not the paper workload.

## Safe build and benchmark

```bash
bash bench_latency.sh
```

The script requires a compatible local HIP compiler and a detected AMD
architecture. It prefers `/opt/rocm/bin/hipcc`, falls back to ROCm's
`amdclang++ -x hip` packaging, and uses a system `hipcc` only last; this avoids
silently selecting an older compiler that cannot target `gfx1201`. It exits
instead of guessing an ISA. Set `RUN_LARGE_WMMA=1` only when the larger
supplementary GEMMs are safe on the machine.

The integrated fp16 paper-scale benchmark is intentionally not part of the
default script yet: on the current gfx1201/ROCm 7.2.3 host, a 30-iteration
1024² attempt exceeded five minutes and was stopped without recording a result.
`hip/benchmark.py` now runs parity and a bounded 64² preflight first. Its current
receipt (`results/hip_benchmark_gfx1201.json`) predicts 280.6 seconds even for a
one-warmup/two-iteration four-method protocol and exits without modifying the
latency CSV. This is a performance blocker, not missing parity or workload
coverage; `--force-slow` is an explicit opt-in.

`gfx1151` and `gfx1201` have different WMMA encodings. rocWMMA selects the target
intrinsic, while the explicit device macros in `wmma_mlp.hip` report generation
11/12 and select the supported int8 path. Any measured row must include ISA,
ROCm version, mode, complete workload dimensions, iteration count, and provenance.
