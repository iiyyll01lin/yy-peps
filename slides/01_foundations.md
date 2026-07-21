---
marp: true
theme: default
paginate: true
title: PEPS on AMD — Part I: Foundations
---

<!--
繁體中文:本檔為 Part I(基礎,W01–03)的 Marp 投影片。輸出 PDF:
  npx @marp-team/marp-cli slides/01_foundations.md -o slides/01_foundations.pdf
或於 slides/ 執行 `make 01_foundations.pdf`。每張投影片:英文標題 + 繁中要點。
來源:docs/01_foundations.md。
-->

# Part I — Foundations
## INR · Positional encoding · The grid bottleneck
### W01–W03 · 為 PEPS 打地基

Why coordinate networks blur, how encoding fixes it,
and why learned grids need PEPS.
座標網路為何糊、編碼如何修正、grid 為何需要 PEPS。

> Numerical examples: **legacy-unverified** · 數值示例尚未驗證

---

# What this Part builds / 本部分建立

- **W01** Implicit neural representations & spectral bias
- **W02** Positional encoding & the Lissajous view
- **W03** Learned grids & the bottleneck PEPS breaks

<br>

- **W01** 隱式神經表示與頻譜偏差
- **W02** 位置編碼與 Lissajous 視角
- **W03** 可學習 grid 與 PEPS 打破的瓶頸

---

# W01 · What is an INR? / 何謂 INR

An implicit neural representation stores a signal as a function
`f_theta(coord) -> value`. For an image, `f(x, y) -> (r, g, b)`.

隱式神經表示把訊號存成函式 `f_theta(座標) -> 數值`;影像即 `f(x,y) -> (r,g,b)`。

> Train the network to reproduce known pixels ⇒ it then represents
> the image **continuously**.
> 訓練網路重現已知像素 ⇒ 它便**連續地**表示該影像。

---

# W01 · Spectral bias / 頻譜偏差

- A plain MLP fits **low frequencies first**, high frequencies much later
- Shows up as a blurry image + a missing high-frequency ring in the FFT
- **Legacy notebook example:** plain MLP ~18–20 dB → ~40 dB with encoding

<br>

- 純 MLP **先擬合低頻**,很晚才擬合高頻
- 表現為模糊影像 + FFT 缺失的高頻環
- **舊 notebook 示例:**純 MLP 約 18–20 dB → 加編碼後約 40 dB

---

# W02 · Positional encoding / 位置編碼

Absolute positional encoding (APE) lifts a coordinate into Fourier features:

`enc(x) = [x, sin(2^1 pi x), cos(2^1 pi x), ..., sin(2^L pi x), cos(...)]`

so the MLP can represent high frequencies.

APE 把座標升維成傅立葉特徵,讓 MLP 能表示高頻。

---

# W02 · The Lissajous view / Lissajous 視角

- Each frequency pair `(sin, cos)` traces a **Lissajous curve** as `x` sweeps
- The projector emits `2L+1` "points of interest"
  `S_i=(1+sin(x phi_i))/2, C_i=(1+cos(x phi_i))/2` (Eq. 6–7)

<br>

- 當 `x` 掃過,每個 `(sin, cos)` 對描出一條 **Lissajous 曲線**
- 投影器產生 `2L+1` 個「興趣點」(論文式 6–7)

---

# W02 · The key result / 關鍵結果

> With an **identity** shared encoder, projected+concatenated features are an
> **affine transform of APE features**.
> W02 verifies this affine equivalence directly.

> 共享編碼器為 **identity** 時,投影+串接特徵是 APE 特徵的**仿射變換**,
> 因此兩者**仿射等價**;W02 直接驗證此關係。

Learned encoders generalize this affine-equivalent identity path.
可學習編碼器則把這條仿射等價的 identity 路徑加以泛化。

---

# W03 · Learned grid encoders / 可學習 grid 編碼器

- A feature grid sampled by **bilinear interpolation** is a powerful learned
  positional encoder (Instant-NGP, NTC)
- Swap APE's fixed Fourier features for a **trainable** grid of latents

<br>

- 以**雙線性內插**取樣的特徵 grid 是強大的可學習位置編碼(Instant-NGP、NTC)
- 把 APE 的固定傅立葉特徵換成**可訓練**的 latent grid

---

# W03 · The bottleneck / 瓶頸

- Past a point, adding resolution/params yields **diminishing PSNR** (Fig. 5)
- **PEPS's fix:** sample **one shared grid at many Lissajous points**, then
  aggregate — more signal from the *same* parameters

<br>

- 超過某點後,增加解析度/參數的 PSNR 收益**遞減**(Fig. 5)
- **PEPS 解法:**在多個 Lissajous 點取樣**同一個共享 grid**再聚合 —— 同參數榨更多訊號

> This params-vs-PSNR curve is the motivational centerpiece of the course.
> 「參數 vs PSNR」曲線是整門課的動機核心。

---

# Honest limitations / 誠實的限制

1. Part I numbers are **pedagogical** (single image, short training)
2. Spectral bias is known; current lab outputs are not verified evidence
3. The grid bottleneck sets up Part II; the payoff arrives with Grid-PEPS

<br>

1. Part I 數字是**教學用**(單張影像、短訓練)
2. 頻譜偏差是*已知*結果 —— 我們重現而非發現
3. grid 瓶頸為 Part II 鋪路;真正的回報在 Grid-PEPS

---

# <!-- fit --> Part II →

### From foundations to the PEPS wrapper
### 從基礎到 PEPS wrapper — `docs/02` · `notebooks/W04–W06`
