# Part II — PEPS core (W04–W06) / PEPS 核心

> **Artifact status:** all numerical CSVs cited below are
> `legacy-unverified`. Equation/layout claims are covered by static tests;
> historical quality values are not verified findings.

## W04 · Building the PEPS wrapper / 建立 PEPS wrapper

**English.** The wrapper implements `M(A(E(P_1)..E(P_{2L+1})), delta)` (paper
Eq. 8): a projector makes `2L+1` points, a **shared** encoder samples each, an
aggregator concatenates/allocates, and an MLP decodes. Making the encoder the
identity and the aggregator concat produces features that are **affinely
equivalent** to APE features (verified directly in lab W04). This is the precise
sense in which the identity path has the same representational information.

The optional **`delta`** term (Eq. 8's input skip) concatenates the raw coords to
the aggregated vector before the decoder; it is exposed as a `delta=False` flag on
all three builders (`build_grid_peps`, `build_grid_peps_texture`, `build_sdf_peps`).
The legacy `course_fast` CSV records **41.023 → 41.383 dB** on average
(**+0.360 dB**) for **+128 parameters**, with per-seed effects from **−1.74 to
+3.04 dB**. Those values require a verified rerun before interpretation;
`delta` remains **off by default**. See `results/delta_ablation.csv`.

**繁體中文.** wrapper 實作論文式(8):投影器產生 `2L+1` 個點,**共享**編碼器逐點
取樣,聚合器串接/分配,MLP 解碼。把編碼器設為 identity、聚合器設為 concat 時,
所得特徵與 APE 特徵**仿射等價**;W04 直接驗證此關係。

可選的 **`delta`** 項(式 8 的輸入 skip)把原始座標接到聚合向量後再進解碼器;三個
builder(`build_grid_peps`、`build_grid_peps_texture`、`build_sdf_peps`)都以
`delta=False` 旗標開放。legacy CSV 記錄平均 **41.023 → 41.383 dB**
(**+0.360 dB**)、多 **128 個參數**,但驗證重跑前不可解讀;預設仍為**關閉**。

## W05 · Grid-PEPS on images / 影像上的 Grid-PEPS

**English.** Swapping the identity encoder for a learned grid gives Grid-PEPS.
The legacy W05 CSV records **42.2 dB vs 37.7 dB** for one teaching comparison,
but it is not a verified Table 1 reproduction. The proposed mechanism is that
many Lissajous samples expose more high-frequency signal; a matched verified
rerun must test that observation.

**繁體中文.** 把 identity 換成可學習 grid 即 Grid-PEPS。在 Kodak、相同參數預算下達到
legacy W05 CSV 記錄 **42.2 dB vs 37.7 dB**,但這不是已驗證的 Table 1 重現。

## W06 · Pink-PEPS & the 1/f story / Pink-PEPS 與 1/f

**English.** Pink-PEPS now follows Algorithm 1 exactly. For latent width `d`,
frequency `i` gets `a_i=max(1,floor(d/2^i))`; cumulative
`G_i=sum_{j=0}^i a_j` selects `S_i[-G_i:-G_{i-1}]` and
`C_i[G_{i-1}:G_i]`, while inputs retain the grouped point order
`(x,S_1..S_L,C_1..C_L)`. For `d=8,L=6`, concat has dimension 104 and Pink has
dimension 28. The legacy `course_fast` W06 CSV records:

- same 128×128×8 grid: concat **34.456 dB / 146,307 params** versus Pink
  **35.500 dB / 141,443 params** (**+1.044 dB**);
- aggregation / decoder input **104 → 28 (−73.1%)**, but total parameters only
  **−3.3%**, because the shared grid alone is 131,072 parameters (~90% of concat);
- shrinking Pink to 120 and 112 resolution saves **14.2% / 24.3% total** but costs
  **0.600 / 0.761 dB** relative to concat.

The dimension change is a tested structural fact. The quality/parameter rows
remain unverified and do **not** establish −25% *total* parameters at matched
PSNR. See `results/pink_param_savings.csv`.

**繁體中文.** Pink-PEPS 現在精確遵循 Algorithm 1。latent 寬度為 `d` 時,頻率 `i`
分得 `a_i=max(1,floor(d/2^i))`;累積 `G_i=sum_{j=0}^i a_j` 後分別取
`S_i[-G_i:-G_{i-1}]` 與 `C_i[G_{i-1}:G_i]`,輸入點順序仍是
`(x,S_1..S_L,C_1..C_L)`。`d=8,L=6` 時 concat 維度為 104、Pink 為 28。這是結構
縮減。legacy W06 CSV 在同一個 128×128×8 grid 上記錄:concat
**34.456 dB / 146,307 參數**,Pink **35.500 dB / 141,443 參數**
(**+1.044 dB**);聚合/解碼器輸入 **104 → 28(−73.1%)**,但總參數只少
**3.3%**,因共享 grid 本身已有 131,072 參數(約佔 concat 90%)。把 Pink resolution
縮至 120 / 112 可省總參數 **14.2% / 24.3%**,但相對 concat 分別掉
**0.600 / 0.761 dB**。維度縮減是已測的結構事實,品質列仍未驗證,
且**不能**宣稱在 matched PSNR 下總參數 −25%;舊前緣見
`results/pink_param_savings.csv`。
