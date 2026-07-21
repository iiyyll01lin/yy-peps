# Part V — AMD hardware (W11–W12) / AMD 硬體

> `results/hip_latency.csv` is currently `legacy-unverified`. Paper numbers
> below are cited external targets; local rows are not accepted hardware claims.

## Paper runtime target / 論文 runtime 目標

The paper's runtime experiment generates one 1024×1024 RGB texture on an RX
9070 XT. All three methods use a 1024×1024 grid, 16 features, and a decoder with
three 64-wide hidden layers; PEPS and Pink use three frequencies. The paper
reports 4.32 ms for BI-grid, 5.47 ms for Grid-PEPS, and 4.86 ms for
Grid-PinkPEPS.

Those values are **external paper references**. They are not copied into the
local results CSV and are not considered reproduced until precision, fusion,
WMMA use, timing boundaries, and hardware all match.

論文 workload 是 RX 9070 XT 上生成 1024×1024 RGB texture:1024² grid、16 features、
三層 64-wide hidden MLP,PEPS/Pink 使用三個 frequency。論文數字只作外部參考;本地
precision、fusion、WMMA、timing boundary 與硬體未全部配對前,不得稱為重現。

## W11 · Integrated HIP path / 整合 HIP 路徑

`hip/fused_peps_kernel.hip` now includes the complete workload:

1. `phi_i=2^i pi` projection in `(x,S_1..S_L,C_1..C_L)` order;
2. all bilinear samples from the shared channel-first grid;
3. baseline, concat, and paper Algorithm 1 Pink modes;
4. three GELU hidden layers and the RGB output layer.

Pink uses `a_i=max(1,floor(C/2^i))`, cumulative `G_i`, reverse circular sine
slices, and forward circular cosine slices. The CPU reference is independently
cross-checked against the Python projector, encoder, aggregators, and MLP.
When HIP is available, binary fixtures cover all three integrated modes in both
the scalar fp32 path and the fused fp16/rocWMMA path.

目前整合 path 包含 projection、所有 shared-grid samples、baseline/concat/Pink 與完整
三 hidden layer MLP。HIP 可用時,三種 mode 的 scalar fp32 與 fused fp16/rocWMMA
都以相同 fixture 對 PyTorch reference。

The current **measured** integrated rows use the scalar fp32 reference. The same
kernel now has a fused fp16/rocWMMA path through all four Linear layers, and its
baseline/concat/Pink fixtures pass parity, but W11 does not yet record a matched
integrated fp16 latency row. W11 writes a row only after build, parity, execution,
and parsing succeed.

Current Box B rerun (gfx1201, ROCm 7.2.3, 20 iterations):

| integrated mode | implementation | ms/iter | parity |
|---|---|---:|---|
| baseline | scalar fp32 | 246.3032 | passed |
| concat PEPS | scalar fp32 | 411.7999 | passed |
| Pink PEPS | scalar fp32 | 295.3217 | passed |

These are locally measured correctness-reference latencies and every row has
`comparable_to_paper=false`; they must not be compared as if they reproduced
the paper's optimized 4–5 ms implementation.

## W12 · WMMA diagnostics / WMMA 診斷

`hip/wmma_mlp.hip` retains isolated fp16 and int8 rocWMMA matrix fixtures and
microbenchmarks. `fused_peps micro` retains the old sample + first-layer
diagnostic. Existing values in `results/hip_latency.csv` are preserved under
`supplementary_microbenchmark`; they are not integrated results and have
`comparable_to_paper=false`.

WMMA 與舊 sample+first-layer 數字保留為 component diagnostics,但 schema 明確標記
為 supplementary,不得取代完整 workload。

The current Box B isolated 4096×64×64 rerun measures **15.0896 ms fp16** and
**15.4592 ms int8**, with parity passed. Older Box A rows and the large 2048³
Box B rows remain explicitly `legacy_reported`; they are retained for provenance,
not blended into the current integrated result.

## Result contract and remaining blocker / 結果契約與 blocker

`results/hip_latency.schema.json` requires workload kind, mode, implementation,
ISA, ROCm version, dimensions, iteration count, parity state, provenance, and
whether a paper comparison is valid. Integrated fp16 parity now exists on
gfx1201/ROCm 7.2.3. A 1024² run configured for 10 warmups and 30 iterations did
not complete within five minutes and was stopped without recording a latency
row. The bounded runner then passed a clean build and all parity cases, but its
64² preflight projected **280.6 seconds** even for a one-warmup/two-iteration
four-method 1024² protocol, so it refused to time or write a row. The exact
remaining blocker is therefore fp16 integrated-kernel performance and a
practical repeated target-size measurement—not missing projection,
aggregation, decoder depth, or parity. Isolated layer speedups still do not
satisfy the paper-comparison requirement.
