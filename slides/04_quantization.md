---
marp: true
theme: default
paginate: true
title: PEPS on AMD — Part IV: Quantization
---

<!--
繁體中文:本檔為 Part IV(量化,W10 原創貢獻)的 Marp 投影片。輸出 PDF:
  npx @marp-team/marp-cli slides/04_quantization.md -o slides/04_quantization.pdf
或於 slides/ 執行 `make 04_quantization.pdf`。每張投影片:英文標題 + 繁中要點。
來源:docs/04_quantization.md、results/w10_rate_distortion.csv。
-->

# Part IV — Quantization
## A matched-size extension
### W10 · 配對大小的延伸研究

Full precision is a lab result. **Shipping codecs quantize.**
全精度是實驗室結果,**量產 codec 都量化**。

> Result CSV status: **legacy-unverified** · 結果 CSV 尚未驗證

---

# Motivation / 動機

- The paper reports PEPS's advantage **only in full precision** — never quantizes
- Yet every shipping texture codec uses **quantized weights**:
  RTXNTC's cooperative-vector **int8** path, BCn block compression

<br>

- 論文只在**全精度**報告 PEPS 優勢 —— 從未量化
- 但每個量產材質 codec 都用**量化權重**:RTXNTC 的 int8 路徑、BCn 區塊壓縮

---

# What we do / 我們做什麼

- Apply **post-training quantization (PTQ)** to grid latents + MLP weights
- Count payload + scales + tensor/model metadata; plot PSNR vs **bpp/bpt**
- Compare per-tensor, per-channel, and mixed precision over three seeds
- `peps/quant/ptq.py` · exercised by `notebooks/W10`

<br>

- 對 grid latent + MLP 權重做 **PTQ**
- 計入 payload、scale、tensor/model metadata;以 **bpp/bpt** 畫率失真曲線
- 三個 seed 比較 per-tensor、per-channel 與 mixed precision

---

# Evidence status / 證據狀態

- The checked-in CSV is **legacy-unverified**
- Its schema records three seeds and packed rates, but schema validity alone is insufficient
- A fresh accepted run manifest and raw evidence are still required

<br>

- checked-in CSV 是 **legacy-unverified**
- schema 記錄三個 seed 與 packed rate,但 schema 正確不等於證據已驗證

---

# Matched-size protocol / 配對大小 protocol

1. Match total model parameter counts before training
2. Quantize paired models with the same ablation
3. Require packed `total_encoded_bits` within **2.5%**
4. Repeat on **three fixed seeds**

只有通過完整 encoded-size 配對且跨 seed 穩定,才可形成比較主張。

---

# Legacy teaching rows / 舊教學列(未驗證)

| plan | bpp grid / PEPS | grid | PEPS | gap |
|---|---:|---:|---:|---:|
| fp32 | 45.661 / 45.688 | 39.380 | 42.022 | +2.642 |
| int8 tensor | 11.491 / 11.500 | 36.506 | 41.100 | **+4.594** |
| int8 channel | 11.556 / 11.564 | 38.702 | 41.623 | +2.921 |
| latent-6 / weight-8 | 8.887 / 9.019 | 33.678 | 37.825 | +4.147 |

These are the CSV's recorded means, not a current repository finding.
這些是 CSV 所記錄的平均值,不是目前 repo 的已驗證發現。

---

# Mechanism: hypothesis only / 機制僅為假說

- Multiple projected samples may alter quantization-error propagation
- “Errors average out” has **not** been isolated experimentally
- A causal test must hold decoder, training budget, and encoded size fixed

<br>

- 多點 projected sampling 可能改變量化誤差傳播
- 「誤差被平均」尚未由控制實驗隔離
- 因果測試須固定 decoder、訓練 budget 與 encoded size

---

# The rate-distortion figure / 率失真圖

- PSNR (y) vs **total encoded bits per pixel/texel** (x)
- Show every seed and verify every pair under the new run manifest
- Positive paired gaps remain descriptive; report their seed-to-seed spread

<br>

- 率失真圖:PSNR(y)vs 完整 encoded bpp/bpt(x)
- 顯示每個 seed,並在新 run manifest 下重新驗證每個 pair
- 正差距是描述性結果,仍須回報 seed 間變異

---

# Narrative hook → hardware / 敘事掛勾

> Our **software int8 PTQ** is the analogue of RTXNTC's **hardware
> cooperative-vector int8** matmul.

> 我們的**軟體 int8 PTQ** 是 RTXNTC **硬體 cooperative-vector int8** 矩陣乘的對應。

Part V (W11–W12) takes this to **actual HIP/WMMA kernels** on RDNA hardware.
Part V 把它帶到 RDNA 硬體上真正的 **HIP/WMMA kernel**。

---

# Honest limitations / 誠實的限制

1. This is **post-training** quantization — no fine-tuning to recover accuracy
2. bpp/bpt covers the packed model, not the complete entropy-coded codec
3. The checked-in repeats remain unverified; one image also cannot establish a general cause

<br>

1. 這是**事後**量化 —— 未微調回補精度
2. bpp/bpt 只涵蓋 packed model,不是完整 entropy-coded codec
3. matched 重複已完成,但單張影像不能建立一般性因果

---

# <!-- fit --> Part V →

### PyTorch → HIP → RDNA4 WMMA, on real silicon
### 在真實晶片上 — `docs/05` · `notebooks/W11–W12`
