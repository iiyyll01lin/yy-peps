---
marp: true
theme: default
paginate: true
title: PEPS on AMD — Course Overview
---

<!--
繁體中文:本檔為 Marp 投影片(markdown)。輸出 PDF/PPTX:
  npx @marp-team/marp-cli slides/00_overview.md -o slides/00_overview.pdf
每張投影片:英文標題 + 繁中要點。
-->

# PEPS on AMD
## Positional Encoding Projected Sampling
### 重實作 · 協定 · 量化 · 選配 AMD kernel

A one-semester educational reimplementation
一門 PEPS 教學型重實作課程

---

# Why this course / 為何開這門課

- **Study** PEPS through a testable independent implementation
- **Separate** `course_fast` teaching runs from the `paper_exact` protocol
- **Extend** with a quantization study the paper omits
- **Exercise** optional HIP/WMMA paths on explicitly gated AMD hardware

<br>

- **研讀** PEPS 並建立可測試的獨立實作
- **區分** `course_fast` 與 `paper_exact`
- **延伸** 論文未做的量化研究
- **練習** 選配且明確 gated 的 AMD HIP/WMMA

---

# The core idea / 核心想法

PEPS = **Project** coordinates onto Lissajous curves →
**sample one shared grid** at those points → **aggregate** → tiny MLP.

PEPS = 把座標**投影**到 Lissajous 曲線 → 在這些點**取樣同一個共享 grid**
→ **聚合** → 小 MLP。

> Identity encoder + concat ⇒ features affinely equivalent to positional encoding.
> Identity 編碼器 + concat ⇒ 特徵與經典位置編碼仿射等價。

---

# Course spine / 課程主軸

| Part | Weeks | Content |
|---|---|---|
| I Foundations | 1–3 | INR, positional encoding, grid bottleneck |
| II PEPS core | 4–6 | wrapper, Grid-PEPS, Pink 1/f |
| III Applications | 7–9 | image, texture (main), SDF |
| IV Quantization | 10 | **original** rate-distortion study |
| V AMD hardware | 11–12 | HIP, RDNA4 WMMA |

---

# Two AMD boxes, one repo / 兩台 AMD、一個 repo

- **Box B** — 4× Navi 48, **RDNA 4** (`gfx1201`): the paper's target
- **Box A** — Radeon 8060S, **RDNA 3.5** (`gfx1151`): comparison point

The optional self-hosted workflow targets both ISAs; CPU CI makes no GPU claim.
選配 self-hosted workflow 針對兩種 ISA;CPU CI 不宣稱 GPU 證據。

---

# Evidence status / 證據狀態

- Every top-level result CSV remains **legacy-unverified**
- Released: 3 manifest-backed synthetic smokes, 2 **inconclusive** pilots,
  and 3 public 512³ SDF provenance receipts
- All released evidence is explicitly **not paper-comparable**
- Pitted Stonefish remains authorization-blocked; paper result count = **0**
- `results/course_release/receipt.json` is authoritative

現有 CSV 皆為 **legacy-unverified**;course release 只含執行 smoke、無結論 pilot
與輸入 provenance,不能當論文數值重現。

---

# Honest limitations / 誠實的限制

1. This is an **educational independent implementation**, not yet a full reproduction
2. Local texture baselines are proxies, not official RTXNTC parity
3. Quantization mechanisms and GPU latency remain hypotheses/evidence tasks until verified

<br>

1. 這是**教學型獨立實作**,尚非完整重現
2. 本地材質基線是 proxy,不是官方 RTXNTC parity
3. 量化機制與 GPU 延遲需完成驗證後才能主張

---

# <!-- fit --> 開始 / Let's build
### notebooks/ · peps/ · apps/ · hip/
