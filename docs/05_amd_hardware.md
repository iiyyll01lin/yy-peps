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

## Where the remaining gap goes / 剩下的差距去了哪裡

With the measurement fixed, the same kernel source and the same harness run on
two parts answer a question the paper cannot: does this kernel benefit from
RDNA4 at all, beyond having more compute units?

| method | gfx1201 (RDNA4, 64 CU) | gfx1151 (RDNA3.5, 40 CU) | speedup | per CU |
| --- | ---: | ---: | ---: | ---: |
| `bi-grid` | 7.40 | 12.81 | 1.73x | 1.08 |
| `grid-peps-3f` | 16.06 | 27.82 | 1.73x | 1.08 |
| `grid-pink-peps-3f` | 18.36 | 28.99 | 1.58x | 0.99 |
| `grid-pink-peps-4f` | 21.27 | 33.17 | 1.56x | 0.97 |

The compute-unit ratio alone is 64/40 = 1.60. Normalised by it, every method
lands between 0.97 and 1.08. **The kernel scales with width and extracts nothing
else from the generation change**, which for code that issues WMMA instructions
is a result rather than a null.

`rocprofv3` says why. The dispatch asks for **32 KB of LDS per 64-thread
workgroup**, and a compute unit has 64 KB, so only two workgroups fit. That is
four resident waves out of a possible thirty-two: **12.5% occupancy**. Wave slots
would have allowed sixteen workgroups and the register budget nine, so LDS is
about four times more restrictive than the next constraint. With four waves per
CU there is almost nothing to switch to while a memory access is outstanding,
which is exactly the profile of a kernel whose speed follows compute-unit count
and nothing else.

The kernel is also 99.99% of GPU time and its profiled average, 12.59 ms, matches
the HIP-event median of 12.60 ms, so the host-side timing boundary is not hiding
anything.

Three experiments follow directly, and all of them are minutes long:

1. Halve the LDS footprint per workgroup. The arithmetic predicts 25% occupancy
   if wave slots and registers stay clear. **Done — see the next section.**
2. Raise the workgroup above 64 threads so one LDS allocation serves more waves,
   which moves occupancy without changing the footprint.
3. Profile the Pink path on its own. It is slower than plain PEPS here while the
   paper reports the opposite ordering, and it is the one method whose
   cross-generation scaling falls below the compute-unit ratio. **Partly
   answered by the next section:** the ordering was an artefact of the shared
   worst-case LDS cap, not of the Pink path itself.

`results/hip_occupancy.json` records the numbers and states plainly that
occupancy was computed from the launch geometry and the part's advertised
limits, not measured with a hardware counter. Low occupancy is consistent with
the observed scaling; it is not yet proof of the cause.

同一份原始碼、同一套量測在兩張卡上跑:RDNA4 對 RDNA3.5 的加速是 1.56–1.73 倍,而
CU 比是 1.60 倍,正規化後落在 0.97–1.08 之間——**這個 kernel 只從 CU 數量獲益,沒有
從世代差異拿到其他好處**。rocprofv3 指出原因:每個 64 執行緒的 workgroup 要 32 KB
LDS,而每 CU 只有 64 KB,故僅能常駐 2 個 workgroup、4 個 wave(上限 32),**佔用率
12.5%**。LDS 比次要限制嚴格約四倍。佔用率是由啟動幾何與硬體上限推算,非以硬體計數器
量測,故只能說「與觀察一致」,尚不足以定案。

## Closing the loop: the cap that cost a factor of two / 讓迴圈閉合:代價兩倍的上限

The occupancy receipt made a falsifiable prediction, so the next step is to act
on it and see whether the prediction survives. It did, and the cause turned out
to be four lines of declaration.

`integrated_peps_wmma` sizes its four `__shared__` tiles from `MAX_INPUT_DIM`
and `MAX_HIDDEN_DIM`, which are worst-case caps of 512 and 128:

```
feature_tile  16 * 512 * 2 = 16384
hidden_a      16 * 128 * 2 =  4096
hidden_b      16 * 128 * 2 =  4096
accumulator   16 * 128 * 4 =  8192
                             -----
                             32768   exactly the 32 KB the profiler reported
```

But `aggregate_dim` gives **16** channels for `bi-grid`, **112** for
`grid-peps-3f`, and **44** and **46** for the two Pink methods, and every
benchmarked configuration uses a hidden width of 64. The largest tile is sized
for a configuration none of the four methods ever asks for. Roughly 20 KB of the
32 KB is reserved and never touched — and LDS is reserved per workgroup whether
it is read or not, so the unused part is paid for in occupancy.

Narrowing the caps to 128 and 64 drops `group_segment_fixed_size` to **12288**
bytes, which lifts derived occupancy from 12.5% to **31.25%**. The caps became
guarded macros, so both binaries come from one identical source file and differ
only by `-D` flags:

```bash
bash hip/build_kernel.sh gfx1201                     # stock, unchanged default
PEPS_EXTRA_FLAGS="-DPEPS_MAX_INPUT_DIM=128 -DPEPS_MAX_HIDDEN_DIM=64" \
  bash hip/build_kernel.sh gfx1201 lds12k            # narrowed
```

Under the settled protocol, on both parts:

| method | gfx1201 stock | gfx1201 narrowed | gfx1151 stock | gfx1151 narrowed |
| --- | ---: | ---: | ---: | ---: |
| `bi-grid` | 7.38 | **3.67** | 12.63 | **6.50** |
| `grid-peps-3f` | 15.99 | **8.53** | 27.82 | **16.37** |
| `grid-pink-peps-3f` | 18.27 | **8.69** | 28.45 | **14.52** |
| `grid-pink-peps-4f` | 21.20 | **9.95** | 33.19 | **16.60** |

**1.70x to 2.13x, with every output checksum byte-identical** on the WMMA path
on both parts and on the scalar path as well. The gap to the paper's reference
numbers narrows from 1.7x–4.2x to 0.85x–2.0x, and `bi-grid` at 3.67 ms now sits
below the paper's 4.32 ms.

Two things in that table deserve more attention than the headline.

**The return is sublinear.** Occupancy rose 2.5x and latency improved about 2x.
Reporting the 2.5x as the speedup would be overstating it. Occupancy was the
binding constraint, not the only one.

**The Pink ordering was an artefact.** The paper puts Grid-PinkPEPS ahead of
Grid-PEPS at three frequencies, 4.86 against 5.47, and the stock build reversed
that on both parts — which is what previously looked like a reproduction
failure. The narrowed build restores the paper's ordering on gfx1151 (14.52
against 16.37) and shrinks the gfx1201 inversion from 2.28 ms to 0.16 ms. The
reason is in the arithmetic above: Pink aggregates 44 channels against concat's
112, but a shared worst-case cap forces both to reserve the same LDS, so both
pay the same occupancy penalty and Pink's smaller feature vector buys nothing.
**A measurement artefact was masquerading as a disagreement with the paper.**

What this cost is generality, and that has to be said plainly. The stock build
accepts an aggregated input up to 512; the narrowed build refuses anything above
128 and fails closed with a diagnostic, because `check_config` already validates
against the cap. This is a specialisation to the deployed configuration, not a
free win. The honest statement of the finding is that **a generous compile-time
cap on an LDS tile cost about a factor of two in latency**, on both RDNA
generations tested. The default build is unchanged; the narrow build has to be
asked for.

`results/hip_lds_ab.json` records the footprints, both measurement protocols,
every checksum, and what remains unmeasured — chiefly that the occupancy figure
explaining the speedup is still derived from launch geometry rather than read
from a counter.

佔用率 receipt 提出了一個可證偽的預測,於是下一步就是去驗它。結果站得住,而原因只是
四行宣告:`__shared__` 分頁由 `MAX_INPUT_DIM=512`、`MAX_HIDDEN_DIM=128` 這兩個**最壞
情況上限**決定,合計正好 32 KB;但四個方法實際只用到 16/112/44/46 與 hidden 64。約
20 KB 被保留卻從未觸碰,而 LDS 是**每個 workgroup 整份保留**,用不到的部分照樣以佔用
率支付。把上限收到 128/64,`group_segment_fixed_size` 降到 12288,推算佔用率由 12.5%
升至 **31.25%**,兩張卡、四個方法一致獲得 **1.70–2.13 倍**加速,且**所有 checksum 完全
相同**。兩個值得注意的細節:一是**回報次線性**(佔用率 2.5 倍只換到約 2 倍),說明
佔用率是綁定約束但非唯一約束;二是**先前的 Pink 排序異常是量測假象**——論文說 Pink 快
於 PEPS,stock 版在兩張卡上都相反,收窄上限後 gfx1151 恢復論文排序、gfx1201 的反轉從
2.28 ms 縮到 0.16 ms,因為共用的最壞情況上限讓 Pink 較小的特徵向量完全得不到好處。
代價是通用性:窄版拒絕超過 128 的輸入(`check_config` 既有檢查會擋下並報錯),因此這
是針對部署組態的特化而非免費的勝利。預設建置未更動。

## The kernel meets the reproduction, halfway / kernel 與重現的交會,只有一半

The repository's headline result is the texture reproduction, and it had never
used the repository's own fused kernel. Closing that gap turned out to be less
about wiring and more about finding out what the kernel can and cannot say.

Two things had to be established before any number was worth taking.

**There was no PyTorch latency to compare against.** The 3F/4F sweep has never
run: `completed_jobs: 0`, `latency_observation_count: 0`, every latency column
empty. The limitation attached to it said the sweep "records real local latency
but is not comparable" to the paper, which credited it with a measurement it
does not have. That text has been corrected in place.

**The kernel can express half the method set.** The four Grid methods match the
fused kernel exactly in decoder shape — hidden width 64, four Linear layers,
three output channels — and differ only in feature width, 17 against the
paper's 16. Running the kernel at 17 channels reproduces their decoder input
dimensions exactly:

| texture method | `decoder_input_dim` | kernel at C=17 |
| --- | ---: | ---: |
| `Grid-PEPS3F` | 119 | 119 |
| `Grid-PEPS4F` | 153 | 153 |
| `Grid-PinkPEPS3F` | 45 | 45 |
| `Grid-PinkPEPS4F` | 47 | 47 |

The four NTC methods cannot be run at all. They aggregate a 68-channel grid
(`4 * 12 + 20`) against the kernel's `MAX_CHANNELS` of 32, and they add a
12-dimension tiled encoding that `aggregate_dim` does not model. Both refusals
are fail-closed and were confirmed on hardware rather than inferred:
`geometry concat 68 4` prints `invalid integrated workload dimensions`. Raising
the cap is not a fix — `NTC_PEPS4F` aggregates 624, and sizing the tiles for it
returns occupancy to 12.5%, handing back the whole gain from the previous
section.

A `geometry <mode> <channels> <frequencies>` subcommand was added rather than
changing the four named methods, so the paper's configuration keeps reporting
exactly what it reported before. Adding it exposed a latent defect: the receipt
emitted `feature_dim` as the literal `16` and computed `selected_feature_dim`
from a hardcoded `16`. Harmless while every path used 16 channels, wrong the
moment one did not. Both now derive from the running configuration.

At the Grid family's own geometry, under the settled protocol:

| method | stock (32 KB) | 160-cap (13 KB) | speedup |
| --- | ---: | ---: | ---: |
| `Grid-PEPS3F` | 18.33 | **10.04** | 1.83x |
| `Grid-PEPS4F` | 20.28 | **11.33** | 1.79x |
| `Grid-PinkPEPS3F` | 18.67 | **9.59** | 1.95x |
| `Grid-PinkPEPS4F` | 20.76 | **10.54** | 1.97x |

The cap is 160 rather than 128 because `Grid-PEPS4F` aggregates 153, which the
128-cap build refuses with `aggregated input 153 exceeds cap 128`. All four
checksums are identical between the two builds.

And here the Pink ordering resolves completely. On the stock build Pink is
slower than concat at **both** frequencies (18.67 against 18.33, 20.76 against
20.28), contradicting the paper. On the narrowed build Pink is faster at
**both** (9.59 against 10.04, 10.54 against 11.33), agreeing with it. The
earlier sections saw this reversal partially; at the reproduction's own
geometry it is unambiguous.

`results/hip_texture_geometry.json` records all of it, including the part that
matters most: this runs random weights, not the trained checkpoints. It gives
the cost of the decoder's shape, not the quality of the reproduction's output,
and it is not a substitute for the sweep.

這個 repo 的招牌成果是 texture 重現,而它從未用過自己的 fused kernel。把兩者接起來
之後發現的重點不在接線:**3F/4F sweep 根本沒跑過**(`completed_jobs: 0`,所有延遲欄
位為空),原本的限制文字卻寫它「records real local latency」,等於承認了一份不存在的
量測,已就地訂正。而 **fused kernel 只能表達八個方法中的四個**:Grid 家族與 kernel 的
decoder 形狀完全相同(hidden 64、4 層 Linear、輸出 3),僅特徵寬度 17 對 16,以 C=17
執行即精確重現 119/153/45/47;NTC 家族的 68 通道超過 kernel 的 `MAX_CHANNELS = 32`,
且多出 12 維 tiled encoding,兩者皆 fail closed 並已在硬體上確認。把上限提高不是解法
——`NTC_PEPS4F` 需要 624,為它配置分頁會讓佔用率退回 12.5%,前一節的增益全數吐回。
在 Grid 家族自己的幾何上,收窄上限帶來 **1.79–1.97 倍**加速且 checksum 全同;更關鍵的
是 **Pink 排序在此完全釐清**:stock 版在兩個頻率上 Pink 都比 concat 慢(與論文相反),
收窄後兩個頻率上 Pink 都快(與論文一致)。此處使用隨機權重而非訓練檢查點,量到的是
decoder 形狀的成本,不是重現的品質,也不能取代 sweep。

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
