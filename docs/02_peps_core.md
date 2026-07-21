# Part II — PEPS core (W04–W06) / PEPS 核心

## W04 · Building the PEPS wrapper / 建立 PEPS wrapper

**English.** The wrapper implements `M(A(E(P_1)..E(P_{2L+1})), delta)` (paper
Eq. 8): a projector makes `2L+1` points, a **shared** encoder samples each, an
aggregator concatenates/allocates, and an MLP decodes. Making the encoder the
identity and the aggregator concat recovers APE exactly (verified to <1e-4
residual in lab W04) — proving PEPS generalizes positional encoding.

**繁體中文.** wrapper 實作論文式(8):投影器產生 `2L+1` 個點,**共享**編碼器逐點
取樣,聚合器串接/分配,MLP 解碼。把編碼器設為 identity、聚合器設為 concat,即精確
退化回 APE(W04 實作驗證殘差 <1e-4)——證明 PEPS 泛化了位置編碼。

## W05 · Grid-PEPS on images / 影像上的 Grid-PEPS

**English.** Swapping the identity encoder for a learned grid gives Grid-PEPS. On
a Kodak image at matched parameter budgets it reaches **42.2 dB vs the grid
baseline's 37.7 dB** (lab W05, our run) — reproducing the G-PEPS rows of Table 1.
The gain comes from sampling one shared grid at many Lissajous points, extracting
more high-frequency signal per parameter.

**繁體中文.** 把 identity 換成可學習 grid 即 Grid-PEPS。在 Kodak、相同參數預算下達到
**42.2 dB,而 grid 基線 37.7 dB**(W05 實測)——重現 Table 1 的 G-PEPS 列。增益來自
在多個 Lissajous 點取樣同一個共享 grid,每個參數榨出更多高頻訊號。

## W06 · Pink-PEPS & the 1/f story / Pink-PEPS 與 1/f

**English.** Natural images have `1/f^alpha` power spectra (lab W06 measures
alpha≈2.4 on Kodak). The Pink aggregator allocates latent width **inversely to
frequency**, matching quality with fewer parameters and computing only sub-vectors
(so it is also faster). This reproduces the paper's parameter-savings result; our
current allocation is conservative (tune toward the paper's −25% as an exercise).

**繁體中文.** 自然影像有 `1/f^alpha` 功率譜(W06 實測 Kodak alpha≈2.4)。Pink 聚合器
把 latent 寬度**與頻率成反比**分配,以更少參數達到同等品質,且只算子向量(故也較快)。
這重現論文的參數節省結果;目前分配偏保守(把它調到論文的 −25% 可作為練習)。
