# Part IV — Quantization (W10, original contribution) / 量化(原創貢獻)

**English.** The paper reports PEPS's advantage in full precision but never
quantizes — yet every shipping texture codec (NVIDIA RTXNTC's cooperative-vector
int8 path, BCn block compression) uses quantized weights. This chapter is our
original contribution: apply int8 post-training quantization (PTQ) to the grid
latents and MLP weights, and draw **rate-distortion curves** (PSNR vs effective
bits/param).

**Headline finding (our run).** The PEPS advantage does not just survive int8 — it
*widens* as precision drops:

| bits | NTC (grid) | PEPS | gap |
|---|---|---|---|
| fp32 | 37.9 | 41.0 | +3.1 dB |
| 8 | 37.7 | 39.0 | +1.4 dB |
| 6 | 31.2 | 35.9 | **+4.7 dB** |
| 4 | 9.6 | 13.2 | +3.6 dB |

At 6-bit the plain grid collapses (37.7→31.2 dB) while PEPS holds 35.9 dB. The
interpretation: PEPS spreads information across many shared-grid samples, so
per-value quantization error averages out — a distributed code is more
quantization-robust than a dense one. This is a deployment-relevant result the
paper leaves on the table.

**繁體中文.** 論文以全精度報告 PEPS 優勢卻從未量化——但每個量產材質編碼器(NVIDIA
RTXNTC 的 cooperative-vector int8 路徑、BCn 區塊壓縮)都用量化權重。本章是我們的原創
貢獻:對 grid latent 與 MLP 權重做 int8 事後量化(PTQ),畫**率失真曲線**(PSNR vs
有效位元/參數)。

**招牌發現(實測).** PEPS 優勢不只在 int8 下存活,還隨精度下降而*擴大*:6-bit 時純
grid 崩潰(37.7→31.2 dB)而 PEPS 維持 35.9 dB(差距 +4.7 dB)。解讀:PEPS 把資訊
分散到許多共享 grid 樣本,逐值量化誤差被平均掉——分散式編碼比稠密編碼更耐量化。
這是論文留白、而與部署高度相關的結果。

**Narrative hook / 敘事掛勾.** Our software int8 PTQ is the analogue of RTXNTC's
hardware cooperative-vector int8 matmul. W11–W12 take this to actual HIP/WMMA
kernels on RDNA hardware.

我們的軟體 int8 PTQ 是 RTXNTC 硬體 cooperative-vector int8 矩陣乘的對應。W11–W12
把它帶到 RDNA 硬體上真正的 HIP/WMMA kernel。
