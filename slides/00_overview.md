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
### 重現 · 對照 · 量化 · RDNA4 kernel

A one-semester course on real AMD silicon
一門在真實 AMD 硬體上的一學期課程

---

# Why this course / 為何開這門課

- **Reproduce** AMD's PEPS paper end-to-end (no official code exists)
- **Benchmark** vs an RTXNTC-equivalent (official RTXNTC is CUDA-only)
- **Extend** with a quantization study the paper omits
- **Land** on RDNA4 HIP/WMMA kernels — real hardware

<br>

- **重現** AMD PEPS 論文(無官方碼)
- **對照** RTXNTC 等價 baseline(官方為 CUDA-only)
- **延伸** 論文未做的量化研究
- **落地** RDNA4 HIP/WMMA kernel — 真實硬體

---

# The core idea / 核心想法

PEPS = **Project** coordinates onto Lissajous curves →
**sample one shared grid** at those points → **aggregate** → tiny MLP.

PEPS = 把座標**投影**到 Lissajous 曲線 → 在這些點**取樣同一個共享 grid**
→ **聚合** → 小 MLP。

> Identity encoder ⇒ PEPS degenerates to classic positional encoding.
> Identity 編碼器 ⇒ PEPS 退化回經典位置編碼。

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

Kernels compile for both; results merged into one git history.

kernel 兩者皆可編譯;結果融合到同一份 git 歷史。

---

# Key results reproduced / 已重現的關鍵結果

- Grid-PEPS **42.2 dB** vs grid 37.7 dB at matched params (Table 1)
- Kodak PSD slope ≈ **1/f²** (Fig. 3) → motivates Pink
- NTC_PEPS best on textures, largest gain on high-freq metal (Table 2)
- **W10 original**: PEPS gap *widens* under int8 (6-bit: +4.7 dB)
- WMMA MLP kernel verified on real RDNA hardware

---

# Honest limitations / 誠實的限制

1. PEPS is a **faithful reimplementation** — a reproducibility lesson
2. Inference overhead is real (~12–26% slower) — measured, not hidden
3. The paper excludes quantization — **Week 10 tests if the edge holds**

<br>

1. PEPS 是**忠實重現** — 本身即一堂可重現性課
2. 推論開銷真實(慢約 12–26%)— 量出來、不藏
3. 論文未含量化 — **第 10 週檢驗優勢是否成立**

---

# <!-- fit --> 開始 / Let's build
### notebooks/ · peps/ · apps/ · hip/
