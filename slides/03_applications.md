---
marp: true
theme: default
paginate: true
title: PEPS on AMD — Part III: Applications
---

<!--
繁體中文:本檔為 Part III(三個應用,W07–09)的 Marp 投影片。輸出 PDF:
  npx @marp-team/marp-cli slides/03_applications.md -o slides/03_applications.pdf
或於 slides/ 執行 `make 03_applications.pdf`。每張投影片:英文標題 + 繁中要點。
來源:docs/03_applications.md、results/table1_image.csv、table2_texture.csv、table3_sdf.csv。
-->

# Part III — Three applications
## Image · Texture (main) · SDF
### W07–W09 · 把 PEPS 用在真實訊號上

One wrapper, three signals — three protocols to verify.
一個 wrapper、三種訊號 —— 三組待驗證協定。

> All displayed CSV rows: **legacy-unverified** · 所有數值列尚未驗證

---

# What this Part covers / 本部分涵蓋

- **W07** Implicit image representation — the full Table 1 (PSNR/SSIM/**LSD**)
- **W08** Neural texture compression — local proxy, not RTXNTC parity
- **W09** Signed distance functions — matched-comparison protocol

<br>

- **W07** 隱式影像 —— 完整 Table 1(PSNR/SSIM/**LSD**)
- **W08** 神經材質壓縮 —— 本地 proxy,非 RTXNTC parity
- **W09** 有號距離函數 —— 公平對照協定

---

# W07 · Image & why LSD / 影像與 LSD

- Evaluate grid / Grid-PEPS / Pink-PEPS on PSNR, SSIM, **LSD**
- **LSD** (log-spectral distance) measures high-frequency fidelity not summarized by PSNR
- Legacy rows record **LSD 0.51 vs 0.82**; no ranking is verified

<br>

- 以 PSNR、SSIM、**LSD** 評估三方法;LSD 補充高頻保真度
- 舊列記錄 **LSD 0.51 vs 0.82**;排名尚未驗證

---

# W07 · Legacy Table 1 rows / 舊 Table 1 列

| method | params | PSNR | SSIM |
|---|---|---|---|
| grid | 136,003 | 37.74 dB | 0.9966 |
| **Grid-PEPS** | 142,147 | **42.20 dB** | **0.9988** |

The recorded **+4.5 dB** requires a matched verified rerun.
舊 **+4.5 dB** 需完成公平重跑後才能主張。

---

# W08 · Neural texture compression / 神經材質壓縮

- A PBR material is a **9-channel** signal (albedo, normal, roughness, metal, AO)
- The course has a single-grid baseline, local multi-resolution **proxy**,
  **Grid-PEPS**, and **NTC-PEPS**

<br>

- PBR 材質是 **9 通道**訊號;在涵蓋頻率光譜的 AmbientCG 材質上擬合多種模型
- 單解析度基線、本地多解析度 **proxy**、**Grid-PEPS**、**NTC-PEPS**

---

# W08 · Legacy reduced-data rows / 舊縮小資料列

| material | ntc | multires proxy | grid_peps | ntc_peps |
|---|---|---|---|---|
| **MetalPlates013** (high-freq) | 37.29 | **45.56** | 37.87 | 38.20 |
| Metal032 | 51.94 | **59.60** | 54.94 | 52.75 |
| Planks020 | 37.14 | **44.78** | 35.89 | 36.28 |
| Rock023 | 35.13 | **47.29** | 35.11 | 35.37 |

These proxy/course-fast values are unverified and are not the paper's 18-set 4K result.
這些 proxy/course-fast 值尚未驗證,也不是論文 18 組 4K 結果。

---

# W08 · The RTXNTC honest note / RTXNTC 誠實註記

> Official NVIDIA RTXNTC **cannot build on our AMD hardware**:
> its compressor hard-requires the CUDA Toolkit, and inference needs
> Vulkan/DX12 Cooperative Vector on an NVIDIA preview driver.

> 官方 RTXNTC 在本 AMD 硬體**無法建置**(硬相依 CUDA、需 NVIDIA 驅動)。

The local **PyTorch proxy** (`apps/texture/rtxntc.py`) explores similar
components; it has no official output, rate, or runtime parity claim.
本地 **PyTorch proxy** 僅探索相似元件,不宣稱官方 parity。

---

# W09 · Signed distance functions / 有號距離函數

- An SDF stores a shape as `f(x, y, z) -> signed distance`
- Compare dense **grid**, **multires**, **hash** encoders + their PEPS versions
  on **IoU** (Table 3); render with marching cubes

<br>

- SDF 把形狀存成 `f(x,y,z) -> 有號距離`
- 比較 grid / multires / hash 及其 PEPS 版本的 **IoU**(Table 3),以 marching cubes 渲染

---

# W09 · Legacy torus IoU rows / 舊 torus IoU 列

| encoder | params | IoU |
|---|---|---|
| grid | 446,913 | 0.269 |
| **grid_peps** | 449,985 | **0.345** |
| multires | 4,797,121 | 0.303 |
| **multires_peps** | 4,803,265 | **0.343** |
| hash | 2,102,465 | 0.325 |

Training used near-surface samples and full-volume evaluation, but these rows
remain unverified and do not establish the four-asset paper result.
這些列採近表面訓練/全體積評估,但仍未驗證,不能代表論文四資產結果。

---

# Honest limitations / 誠實的限制

1. Texture comparison uses a local proxy, not RTXNTC parity
2. Absolute full-volume SDF IoU is **modest** because near-surface training
   prioritizes the zero level set and starves the bulk interior
3. These are fitted-torus teaching runs, not the paper's Pitted Stonefish result

<br>

1. 材質對照使用本地 proxy,不是 RTXNTC parity
2. 近表面訓練優先照顧零等值面、犧牲內部,故全體積 SDF IoU **不高**
3. 這是 torus 教學實驗,不是論文 Pitted Stonefish 的重現

---

# <!-- fit --> Part IV →

### Does the PEPS edge survive quantization?
### PEPS 優勢在量化後還在嗎? — `docs/04` · `notebooks/W10`
