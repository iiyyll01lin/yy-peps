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

The tracked CSV retains scalar fp32 correctness-reference rows and now also
contains a clean-build fused fp16/rocWMMA run through all four Linear layers.
The runner wrote these rows only after code-object inspection, full-output
parity, execution, and receipt validation succeeded.

Current fused Box B receipt (gfx1201, ROCm 7.2.3, 30 warmups and 100 timed
iterations per method):

| integrated mode | implementation | ms/iter | parity |
|---|---|---:|---|
| baseline | fused fp16/rocWMMA | 42.4544 | passed |
| concat PEPS 3F | fused fp16/rocWMMA | 48.2829 | passed |
| Pink PEPS 3F | fused fp16/rocWMMA | 51.1820 | passed |
| Pink PEPS 4F | fused fp16/rocWMMA | 53.9012 | passed |

These are local measurements, not a paper reproduction. Every row and the
bundle set `comparable_to_paper=false` / `directly_comparable=false` because
the paper does not disclose matching precision, timing boundaries,
synchronization, or kernel source.

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

## Measuring latency on a cold card / 冷卡上的延遲量測

The first protocol ran every iteration of method one, then every iteration of
method two, and so on, starting from an idle GPU. `rocm-smi` recorded the card
at 6 W with the shader clock parked before the run. Thirty warmup iterations
were not enough to reach steady clocks, so whichever method went first absorbed
the ramp.

The signature is in `results/hip_benchmark_gfx1201.json`: the first method
measured has a min of 7.30 ms against a median of 42.30 ms and a standard
deviation of 16.9 ms, while the last two methods are stable to 0.07 and 0.50 ms.
**The ordering that receipt reports is an artefact of measurement order, not a
property of the kernels.**

`hip/stable_latency.py` fixes it with two changes. It spins the card until the
shader clock stops climbing, then interleaves the methods round by round with a
rotating start, so any residual drift is shared instead of being charged to
whoever went first. Eight rounds now agree to within 1.03x, and a six-round run
reproduces the same medians to two decimal places.

| method | old median | **new median** | old stddev | new round spread | paper |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bi-grid` | 42.30 | **7.40** | 16.86 | 1.03x | 4.32 |
| `grid-peps-3f` | 48.20 | **16.06** | 10.30 | 1.01x | 5.47 |
| `grid-pink-peps-3f` | 51.07 | **18.36** | 0.07 | 1.01x | 4.86 |
| `grid-pink-peps-4f` | 53.79 | **21.27** | 0.50 | 1.01x | 4.99 |

Most of the apparent 10x gap to the paper was a measurement artefact. What
survives is a real 1.7x to 4.3x gap, and it is ours to close: `rocminfo` reports
**64 compute units** on this part, chip ID 0x7551, the same Navi 48 configuration
as the paper's RX 9070 XT. The `compute_units: 32` field in the older receipt is
`multiProcessorCount`, which counts workgroup processors on RDNA; 32 WGPs are 64
CUs. Reading it as 32 CUs would wrongly excuse half the gap.

One disagreement with the paper survives the fix and is worth a student's
attention: the paper has Grid-PinkPEPS *faster* than Grid-PEPS, 4.86 against
5.47 ms, while this implementation has Pink slower, 18.36 against 16.06. Pink
should do less work, so the Pink path is where the optimisation study starts.

舊協定從閒置卡開始、逐方法連續量測,先跑的方法吸收了時脈爬升。修正後(先讓時脈穩定、
再逐輪交錯輪替)輪間離散度降到 1.03 倍以內,與論文的差距從約 10 倍收斂到 1.7–4.3 倍。
本機為 64 CU,與論文的 RX 9070 XT 同級,故剩餘差距屬於最佳化空間而非硬體劣勢。

## Rebuilding the kernel / 重新建置 kernel

`hip/build_kernel.sh` exists because a plain `amdclang++` invocation no longer
compiles this file. Ubuntu's `libamdhip64-dev` 5.7.1 installs HIP 5.7 headers in
`/usr/include/hip`, and those reference `__AMDGCN_WAVEFRONT_SIZE`, which clang 22
no longer predefines; the build fails with 19 errors.

`--rocm-path` and `--hip-path` do not help because they do not reorder the search
list, and `-I/opt/rocm/include` is silently dropped: clang resolves it to the same
directory the HIP driver already appends *after* `/usr/include`, deduplicates the
pair, and keeps the later position. `amdclang++ -x hip -E -v` shows `/usr/include`
ahead of the ROCm include with no trace of the `-I` at all, which is the only way
to see this.

The script therefore builds a directory of symlinks whose own path differs, so it
cannot be deduplicated away. Removing the distro package is not an option:
`hipcc`, `librocblas-dev`, `librocrand-dev`, `librocprim-dev`, `librocsparse-dev`
and `librccl-dev` all depend on it.

系統的 Ubuntu ROCm 5.7.1 套件把 HIP 5.7 標頭放在 `/usr/include/hip`,會蓋過 ROCm 7.2.3。
`-I` 會被 clang 去重而失效,故改以符號連結目錄繞開;該套件被 hipcc 等六個套件反向依賴,
不宜移除。

## Result contract and comparison blocker / 結果契約與比較限制

`results/hip_latency.schema.json` requires workload kind, mode, implementation,
ISA, ROCm version, dimensions, iteration count, parity state, provenance, and
whether a paper comparison is valid. Integrated fp16 parity and a practical
repeated 1024² measurement now exist on gfx1201/ROCm 7.2.3. The remaining
blocker is **comparison fidelity**, not missing workload coverage: the local
device reports only a generic GPU name, and the paper does not publish its
precision, timing/synchronization protocol, or kernel. The tracked receipt is
therefore useful local hardware evidence but remains `legacy-unverified` and
must not be presented as reproducing the paper's 4–5 ms figures. Isolated layer
speedups still do not satisfy the paper-comparison requirement.
