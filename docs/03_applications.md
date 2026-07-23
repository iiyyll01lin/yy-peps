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
