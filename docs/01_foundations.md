# Part I — Foundations (W01–W03) / 基礎

This chapter accompanies the first three lab notebooks. It builds the intuition
PEPS rests on: why coordinate networks blur, how positional encoding fixes it,
and why learned grids hit a bottleneck that PEPS breaks.

本章對應前三個實作 notebook,建立 PEPS 賴以成立的直覺:座標網路為何糊、位置編碼
如何修正、以及可學習 grid 為何遇到瓶頸而 PEPS 如何打破它。

> Numerical notebook outputs cited here are historical teaching examples and
> remain `legacy-unverified`. The affine-equivalence statement is covered by a
> deterministic equation test.

---

## W01 · Implicit Neural Representations & spectral bias / INR 與頻譜偏差

**English.** An implicit neural representation stores a signal as a function
`f_theta(coord) -> value`. For an image, `f(x, y) -> (r, g, b)`. Training fits a
network to reproduce known pixels; the network then represents the image
continuously. A plain MLP exhibits **spectral bias** — a documented tendency to
fit low-frequency components first and high frequencies much later. On a Kodak
image this shows up as a blurry reconstruction and a missing high-frequency ring
in the FFT (lab W01, cells 2–3). A legacy notebook output shows roughly
18–20 dB without positional encoding and roughly 40 dB with encoding; rerun it
before treating those values as evidence.

**繁體中文.** 隱式神經表示把訊號存成函式 `f_theta(座標)->數值`。影像即
`f(x,y)->(r,g,b)`。訓練讓網路重現已知像素,網路便連續地表示該影像。純 MLP 有
**頻譜偏差**——先擬合低頻、很晚才擬合高頻的已知傾向。在 Kodak 影像上表現為模糊的
重建與 FFT 中缺失的高頻環(W01 實作 cell 2–3)。legacy notebook 記錄約
18–20 dB vs 約 40 dB;重跑驗證前只作教學示例。

---

## W02 · Positional encoding & the Lissajous view / 位置編碼與 Lissajous 視角

**English.** Absolute positional encoding (APE) lifts a coordinate into a
higher-dimensional Fourier feature vector,
`enc(x) = [x, sin(2^1 pi x), cos(2^1 pi x), ..., sin(2^L pi x), cos(...)]`,
letting the MLP represent high frequencies. PEPS reinterprets this geometrically:
each frequency pair `(sin, cos)` traces a **Lissajous curve** as `x` sweeps. The
projector produces `2L+1` "points of interest"
`S_i=(1+sin(x phi_i))/2, C_i=(1+cos(x phi_i))/2` (paper Eq. 6–7). Crucially, if the
shared encoder is the identity, the projected+concatenated features are an affine
transform of APE features. Thus **identity PEPS is affinely equivalent to APE**
(verified directly in lab W02); learned encoders generalize this identity path.

**繁體中文.** 絕對位置編碼(APE)把座標升維成傅立葉特徵向量,讓 MLP 能表示高頻。
PEPS 用幾何重新詮釋:當 `x` 掃過,每個 `(sin, cos)` 對描出一條 **Lissajous 曲線**。
投影器產生 `2L+1` 個「興趣點」(論文式 6–7)。關鍵是:若共享編碼器為 identity,
投影+串接的特徵是 APE 特徵的仿射變換,因此 **identity PEPS 與 APE 仿射等價**
(W02 直接驗證);可學習編碼器則泛化這條 identity 路徑。

---

## W03 · Grid encoders & the bottleneck / grid 編碼器與瓶頸

**English.** A learned feature grid, sampled by bilinear interpolation, is a
powerful learned positional encoder (used by Instant-NGP and NTC). But growing the
grid runs into a **bottleneck**: past a point, adding resolution/parameters yields
diminishing PSNR (paper Fig. 5). PEPS's fix is to sample **one shared grid at many
Lissajous points** and aggregate — extracting more signal from the same
parameters. Lab W03 asks students to test the params-vs-PSNR trend under a named
profile; its current output is not a verified Fig. 5 reproduction.

**繁體中文.** 用雙線性內插取樣的可學習特徵 grid 是強大的可學習位置編碼(Instant-NGP、
NTC 都用)。但把 grid 變大會遇到**瓶頸**:超過某點後,增加解析度/參數的 PSNR 收益
遞減(論文 Fig.5)。PEPS 的解法是**在多個 Lissajous 點取樣同一個共享 grid** 再聚合
——用相同參數榨出更多訊號。W03 要求在明確 profile 下檢驗「參數 vs PSNR」趨勢;
現有輸出不是已驗證的 Fig.5 重現。
