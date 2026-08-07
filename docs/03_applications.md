# Part III — Three applications (W07–W09) / 三個應用

> **Artifact status:** `table1_image.csv`, `table2_texture.csv`, and
> `table3_sdf.csv` are `legacy-unverified`. Values below describe historical
> teaching artifacts only; none is current paper-reproduction evidence. The
> common runner writes `results/runs/<run-id>/`; specialized calibration runners
> write separate versioned receipts. Their exact release statuses are indexed
> by `results/course_release/receipt.json`.

## Released teaching evidence / 已發布教學證據

The release validates executable paths and provenance, not paper numbers:

- image, texture, and SDF synthetic `course_fast` runs each retain a run
  manifest, raw rows, and summary and are
  `validated-course-smoke-not-paper-comparable`;
- the Kodak three-image/two-seed convergence pilot is complete but
  **inconclusive**: no optimizer-step budget is recommended;
- the two-set texture convergence pilot is complete but **inconclusive**:
  method ordering is unstable and the 71,280,000-step run is not authorized;
- Lucy, Thai Statue, and Armadillo have validated public 512³ provenance
  receipts. This is input evidence, not SDF model-quality evidence.

The release has zero paper-comparable results. Figure 5 dataset/training-budget
omissions, the Table 1 training-step omission and loss/recipe conflict,
optimizer/seed assumptions, and the unreleased SDF converter remain visible.
Pitted Stonefish remains `deferred_auth_required`; no substitute is used.

本次 course release 只驗證執行路徑與來源:三個 synthetic smoke、兩個**無結論**
pilot、以及 Lucy/Thai Statue/Armadillo 的 512³ provenance。論文可比結果仍為零，
Stonefish 仍需授權，且不以替代資料冒充。

## W07 · Implicit image representation (full Table 1) / 隱式影像(完整 Table 1)

`paper_exact` verifies all 24 original-orientation Kodak images and runs the
published PE, LPE, NTC_N, Grid, G-PEPS, G-P-PEPS, NTC_PEPS, NTC_PinkPEPS, and
G-P-PEPS-25 configurations. Portrait images rotate the reported grid dimensions
instead of changing sampling density. Metrics are PSNR, official LDR-FLIP,
AlexNet LPIPS, a versioned LSD oracle, and windowed torchmetrics SSIM.

The paper text is internally inconsistent: it describes L1/fixed-LR/leaky-ReLU,
but the published Table 1 PSNR values equal the appendix L2/GELU/dual-LR/cosine
row. The executable Table 1 path uses the latter and marks the undisclosed
training-step count as a protocol assumption.

`paper_exact` 會驗證 24 張原始方向 Kodak，portrait 影像會同步旋轉 grid 尺寸。
完整九種方法與五項指標都寫入逐影像 rows。Table 1 採與已出版數字一致的 appendix
L2/GELU/dual-LR/cosine 路徑，manifest 同時保留主文 L1 衝突與步數未公開限制。

## W08 · Neural texture compression (main track) / 神經材質壓縮(主線)

`paper_exact` uses all 18 named native-4K sets. Each available map is decoded as
one RGB target; absent maps are errors, not synthetic constants. NTC_N is the
paper two-grid architecture: four G0 corner vectors (12 channels each), one
bilinearly filtered G1 vector (20 channels), and a 12-value, three-octave tiled
triangular encoding. The 11 Table 2 methods train with GELU, L1, grid LR 0.1,
MLP LR 0.001, cosine decay, and 3,000 × 40 batches of 60,000 pixel locations.

PSNR and SSIM are evaluated per RGB map. Global means weight every map equally;
category rows separately aggregate AO, ARM, DIFF, Displacement, metal, normal,
rough, and specular. This avoids the old fixed-nine-channel/whole-bundle metric.
The manifest also labels the frozen Adam/three-seed choices as assumptions
because the paper does not report them.

> **On the RTXNTC comparison (honest note).** The official NVIDIA RTXNTC SDK
> cannot be built on our AMD hardware: its compressor hard-requires the CUDA
> Toolkit (`find_package(CUDAToolkit REQUIRED)` in `tools/cli/CMakeLists.txt`)
> and its inference path needs Vulkan/DX12 Cooperative Vector with an NVIDIA
> preview driver. Box B has no NVIDIA GPU, CUDA, or nvcc. We therefore compare
> against a local **PyTorch proxy** (`apps/texture/rtxntc.py`) for architecture
> exploration. Similar components do not establish output, rate, or runtime
> equivalence.

`paper_exact` 使用 18 組原生 4K set；每張存在的 map 都是獨立 RGB target。
NTC_N 輸入為 G0 四角 concat、G1 bilinear 與三 octave tiled triangular PE。
PSNR/SSIM 先逐 map 計算，再依八種 texture type 與全域彙總，不再把固定 9-channel
bundle 當成一張影像計分。

> **關於 RTXNTC 對照(誠實註記).** 官方 NVIDIA RTXNTC SDK 在本 AMD 硬體**無法建置**:
> 壓縮器硬相依 CUDA Toolkit(`tools/cli/CMakeLists.txt` 的
> `find_package(CUDAToolkit REQUIRED)`),推論路徑需 Vulkan/DX12 Cooperative Vector
> 加 NVIDIA 預覽驅動。Box B 無 NVIDIA GPU、無 CUDA、無 nvcc。故僅以未驗證的
> **PyTorch proxy**(`apps/texture/rtxntc.py`)作架構探索;元件相似不等於結果等價。

### What W08 found / W08 的結果

Table 2 completed at 594 of 594 jobs with no errors, and all eleven methods
landed below their published values, by a mean of 1.154 dB. The ordering also
disagrees: `BI-Grid` finishes above `NTC_PEPS` where the paper has the reverse.

The aggregation described above is what explains both. `table2.json` records
it as `map_weighted`: because global means weight every map equally, the score
is an average over individual maps, and the eight categories are not comparable
with each other. They span 19.4 dB, from `normal` at 32.6 to `Displacement` at
52.1, and this selection puts 47% of its maps in the two lowest. `NTC_PEPS`
minus `BI-Grid` is itself category-dependent, running from -1.17 dB on
`Displacement` to +2.06 dB on `metal`, so reweighting the same measured jobs to
equal categories swaps six method pairs, moves `NTC_PEPS` from third place to
first, and cuts the out-of-sample error against the published values by 3.1x.

Two alternatives were tested and neither survives. Training on to 240k and
480k steps makes the PEPS advantage *shrink* rather than grow, so compute is
not the cause, and `budget_probe/` holds those curves. The paper does report L1
and this track uses it, so the loss family is not the cause either. How that L1
is reduced across a set's maps is genuinely unreported, and changing that
per-map reduction moves the advantage roughly sixfold on `paving-stones-070`;
but it does nothing on `metal-plates-013`, and its effect on the ordering
reverses sign between the two, so no single choice repairs the table.
`ordering_probe/` records both ladders, including the one that is not even
monotone.

The paper names eighteen sets and eight categories but publishes no file list,
so all of this bounds the discrepancy rather than measuring it.
`results/texture_repro/shortfall_analysis/` carries the numbers and the
limitations.

Table 2 完整跑完 594/594 且零錯誤,但十一個方法全部低於論文值,平均 1.154 dB,
排序也相反。原因就在上述彙總方式(`table2.json` 記為 `map_weighted`):分數是對個別
map 平均,而八個類別相差 19.4 dB,本選集有 47% 的 map 落在最低的兩類;而 `NTC_PEPS`
減 `BI-Grid` 本身也隨類別改變(`Displacement` -1.17 dB,`metal` +2.06 dB)。重新
加權成類別均衡後,六組方法對互換,`NTC_PEPS` 由第三升至第一,樣本外誤差降低 3.1 倍。

兩個替代解釋都不成立:延長訓練到 240k/480k 反而讓 PEPS 優勢縮小,故非算力問題;論文
載明 L1 且本軌道確實使用 L1,故非 loss family。真正未被指定的是「L1 如何在各 map 之間
reduce」,它在 `paving-stones-070` 上可讓優勢變動約六倍,但在 `metal-plates-013` 上
毫無作用,且對排序的影響在兩者間換符號,無法單獨修正整張表。論文未公布檔案清單,故以上
只能界定範圍。

## W09 · Signed distance functions / 有號距離函數

`paper_exact` trains on fresh uniform coordinates sampled from each named 512³
volume and obtains targets by trilinear interpolation with inclusive boundaries.
Table 3 runs PE, LPE, TI-grid, Grid-PEPS, Hash, Hash-PEPS, M-Grid, M-PEPS,
M-Hash, and M-HashPEPS under MAPE; the appendix command repeats all methods with
L1. Evaluation streams every one of the 512³ voxels into an exact integer IoU
accumulator.
Adam and seed 0 are explicit reproduction assumptions, as is the documented
`mesh-to-sdf` replacement for the authors' unreleased C++/HIP converter.

Table 4 uses Pitted Stonefish and L1. Its second row doubles every 3D grid
resolution and raises every hash cap by three bits, yielding exactly 8× learned
encoder parameters for all applicable methods. It is not represented by a small
torus analogy.

Processed distances use centered `[-1,1]³` units while model coordinates use
`[0,1]³`, so a true field has input-gradient norm 2. The `course_fast` eikonal
path now uses that norm and samples finite-difference centers strictly inside
`[h,1-h]³`; paper runs do not add eikonal regularization.

`paper_exact` 會在四個 512³ volume 上重新均勻抽樣座標並做 trilinear target
sampling，Table 3 執行 MAPE 與完整 L1 appendix。Table 4 真正比較 Pitted
Stonefish 的 1×/8× encoder；不是用小 torus 類比。由於距離單位是 `[-1,1]³`、
模型輸入是 `[0,1]³`，course eikonal 的正確目標梯度 norm 為 2，且有限差分不再
於邊界 clamp。

## Commands / 指令

```bash
python -m experiments.reproduce check --profile paper_exact
# image-table1 is code-disabled pending the separate full-reproduction gate.
python -m experiments.reproduce run --artifact texture-table2
python -m experiments.reproduce run --artifact sdf-table3-mape
python -m experiments.reproduce run --artifact sdf-table3-l1
python -m experiments.reproduce run --artifact sdf-table4
```

Table 1 currently accepts no receipt and exits before loading data or creating
workers. A future full-reproduction gate must deliberately change the
code-level interlock; its receipt contract is frozen in
`results/schemas/full_run_authorization.schema.json`. Convergence-pilot
evidence is non-authorizing.

Figure 5 additionally requires a user-supplied checksum receipt because the
paper does not name its 4K image suite:

```bash
python -m experiments.reproduce run --artifact image-fig5 \
  --fig5-manifest /path/to/fig5-dataset.json \
  --assumed-steps <N> --allow-protocol-assumptions
```
