# HIP / WMMA kernels — Part V (W11–W12)

The primary path is a **paper-scale fused PEPS decoder**, not the old “one grid
sample + first MLP layer” component benchmark.

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

The benchmark geometry is 1024×1024 output and grid, 16 grid features, three
64-wide hidden layers, GELU, and RGB output. It covers BI Grid, Grid-PEPS 3F,
and selective Grid-PinkPEPS 3F/4F.

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
./fused_peps benchmark grid-pink-peps-4f 1024 20 100
```

Both implementations have baseline/concat/Pink binary parity fixtures. The
benchmark emits JSON receipts with `comparable_to_paper=false`; paper timings
are included only as sourced external references, never as local measurements.
The fp16 path is functionally complete. It is not the authors' unreleased
kernel, so paper values remain external references rather than an exact timing
target.

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
architecture. It removes the previous build directory, names the binary with
the ISA and git SHA, and verifies the single embedded target with `roc-obj-ls`.
It then runs fp32/fp16 full-output parity, a bounded preflight, and the four
HIP-event benchmarks before atomically updating:

- `results/hip_benchmark_<isa>.json`: build command/hash, code-object receipt,
  hardware/clock/power snapshots, parity errors, median/p95 and protocol;
- `results/hip_latency.csv`: schema-v3 summary rows while retaining legacy rows.

The exclusive local gfx1201 run (ROCm 7.2.3, 30 warmups, 100 timed iterations)
measured medians of 42.304/48.200/51.070/53.787 ms for BI Grid, Grid-PEPS 3F,
PinkPEPS 3F, and PinkPEPS 4F. The device reports 32 compute units and a generic
GPU name; the paper reports 4.32/5.47/4.86/4.99 ms on an RX 9070 XT without its
precision or timing protocol. The local medians are 8.81–10.78× those external
references, and the receipt therefore marks direct comparison false.
Performance matching remains open; correctness and workload coverage do not.

`gfx1151` and `gfx1201` have different WMMA encodings. rocWMMA selects the target
intrinsic, while the explicit device macros in `wmma_mlp.hip` report generation
11/12 and select the supported int8 path. Any measured row must include ISA,
ROCm version, mode, complete workload dimensions, iteration count, and provenance.
