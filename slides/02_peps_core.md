---
marp: true
theme: default
paginate: true
title: PEPS on AMD — Part II: PEPS core
---

<!--
繁體中文:本檔為 Part II(PEPS 核心,W04–06)的 Marp 投影片。輸出 PDF:
  npx @marp-team/marp-cli slides/02_peps_core.md -o slides/02_peps_core.pdf
或於 slides/ 執行 `make 02_peps_core.pdf`。每張投影片:英文標題 + 繁中要點。
來源:docs/02_peps_core.md、results/table1_image.csv、
results/pink_param_savings.csv、results/delta_ablation.csv。
-->

# Part II — PEPS core
## The wrapper · Grid-PEPS · the 1/f (Pink) story
### W04–W06 · 把 PEPS 組起來

Assemble `Project → Encode → Aggregate → MLP`, then test the protocol.
組裝 `投影 → 編碼 → 聚合 → MLP`,再檢驗協定。

> Numerical CSVs: **legacy-unverified** · 數值 CSV 尚未驗證

---

# What this Part builds / 本部分建立

- **W04** The PEPS wrapper (paper Eq. 8) + identity/APE affine equivalence
- **W05** Grid-PEPS on images — test the Table 1 protocol
- **W06** Pink-PEPS & the 1/f power-spectrum story

<br>

- **W04** PEPS wrapper(論文式 8)+ identity/APE 仿射等價
- **W05** 影像上的 Grid-PEPS —— 檢驗 Table 1 協定
- **W06** Pink-PEPS 與 1/f 功率譜

---

# W04 · The wrapper / wrapper

The wrapper implements `M(A(E(P_1)..E(P_{2L+1})), delta)` (Eq. 8):

- a **projector** makes `2L+1` points
- a **shared** encoder samples each point
- an **aggregator** concatenates/allocates
- an **MLP** decodes

投影器產生 `2L+1` 點 → **共享**編碼器逐點取樣 → 聚合器串接/分配 → MLP 解碼。

---

# W04 · Generalization check / 泛化檢驗

> Identity encoder + concat aggregator produces features
> **affinely equivalent to APE features** (verified directly in W04).

> identity 編碼器 + concat 聚合器的特徵與 APE 特徵
> **仿射等價**(W04 直接驗證)。

The wrapper is the reusable object **every application** uses.
wrapper 是**每個應用**都會用到的可重用物件。

---

# W04 · Eq. (8) delta / 式(8) delta

- `delta=True` appends raw coordinates before the decoder
- Legacy CSV mean: **41.023 → 41.383 dB (+0.360 dB)**
- Costs only **+128 parameters**, but per-seed effects span **−1.74 to +3.04 dB**

<br>

- `delta=True` 在解碼器前接上原始座標
- legacy CSV 平均 **41.023 → 41.383 dB(+0.360 dB)**
- 只多 **128 參數**,但各 seed 效果介於 **−1.74 至 +3.04 dB**

---

# W05 · Grid-PEPS on images / 影像上的 Grid-PEPS

Swap the identity encoder for a **learned grid** ⇒ Grid-PEPS.
Legacy teaching rows at nominally matched budgets:

| method | params | PSNR | SSIM |
|---|---|---|---|
| grid | 136,003 | 37.74 dB | 0.9966 |
| **Grid-PEPS** | 142,147 | **42.20 dB** | **0.9988** |

把 identity 換成**可學習 grid** 即 Grid-PEPS;表中為未驗證的舊教學列。

---

# W05 · Why it works / 為何有效

- One shared grid, sampled at **many Lissajous points**
- Proposed mechanism: more **high-frequency signal per parameter**
- The legacy +4.5 dB row requires a verified rerun

<br>

- 同一個共享 grid,在**多個 Lissajous 點**取樣
- 每個參數榨出更多**高頻訊號**是待驗證機制
- 舊 +4.5 dB 列需重跑驗證

---

# W06 · The 1/f story / 1/f 故事

- Natural images have `1/f^alpha` power spectra
- W06 asks students to estimate alpha; the current output is unverified
- Low frequencies carry most of the energy

<br>

- 自然影像有 `1/f^alpha` 功率譜
- W06 要求估計 alpha;現有輸出尚未驗證
- 低頻承載大部分能量

---

# W06 · The Pink aggregator / Pink 聚合器

- Exact allocation: `a_i = max(1, floor(d / 2^i))`
- `G_i = sum_{j=0}^i a_j` drives opposite circular sin/cos slices
- Point order stays `(x, S_1..S_L, C_1..C_L)`
- Structural example `d=8,L=6`: aggregation **104 → 28**

<br>

- 精確配置:`a_i = max(1, floor(d / 2^i))`
- 累積 `G_i` 決定 sin/cos 反向 circular slices
- 點順序固定為 `(x, S_1..S_L, C_1..C_L)`
- 結構範例 `d=8,L=6`:聚合維度 **104 → 28**

---

# W06 · Legacy `course_fast` rows / 舊教學列

| aggregator | params | agg input | PSNR |
|---|---:|---:|---:|
| concat | 146,307 | 104 | 34.456 dB |
| **Pink** | **141,443** | **28** | **35.500 dB** |

- Decoder input **−73.1%**, total parameters only **−3.3%**
- The recorded **+1.044 dB** is not a verified no-quality-loss claim

聚合/decoder input **−73.1%**,但總參數只少 **3.3%**;舊 **+1.044 dB**
不是已驗證的無損結論。

---

# The aggregator family / 聚合器家族

| kind | alpha | idea |
|---|---|---|
| **concat** | 0 | plain PEPS: concatenate all point latents |
| **pink** | 1 | width ~ 1/f — the paper's param-saving story |
| **brownian** | 2 | width ~ 1/f² — steeper, stress-test generalization |

`peps/aggregate.py` · `make_aggregator("concat"|"pink"|"brownian", ...)`

三種策略對應論文 alpha 參數;皆讓梯度到達整個 grid。

---

# Honest limitations / 誠實的限制

1. Legacy numerical rows remain `legacy-unverified`
2. The shared grid is ~90% of concat parameters, so **−73.1% decoder input**
   becomes only **−3.3% total**
3. A legacy **−24.3% total** Pink row loses **0.761 dB** — not
   matched quality and therefore not a −25% matched-PSNR claim

<br>

1. 舊數值列仍是 `legacy-unverified`
2. 共享 grid 佔 concat 約 90% 參數,故 decoder input **−73.1%**只換得總量
   **−3.3%**
3. 舊總參數 **−24.3%** 列損失 **0.761 dB**,不是 matched quality,不能宣稱
   matched-PSNR 下 −25%

---

# <!-- fit --> Part III →

### Three applications: image · texture · SDF
### 三個應用 — `docs/03` · `notebooks/W07–W09`
