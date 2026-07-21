# PEPS on AMD — Course & Reproduction Repo
# AMD 上的 PEPS — 課程與重現專案

A university-semester course that reproduces AMD's **PEPS** (Positional Encoding
Projected Sampling), benchmarks it against NVIDIA's **RTXNTC**, adds a
quantization study the original paper omits, and culminates in an RDNA4
HIP/WMMA kernel chapter.

一門大學一學期的課程,完整重現 AMD 的 **PEPS** 論文、與 NVIDIA **RTXNTC** 對照、
補上原論文刻意留白的量化研究,並以 RDNA4 HIP/WMMA kernel 章節收尾。

---

## Hardware targets / 硬體目標

This repo is developed and tested on **two AMD machines**, and merged into one
shared history. Part V (HIP/WMMA) is written for **both** RDNA generations.

本專案在**兩台 AMD 機器**上開發與測試,最後融合到同一份 git 歷史。Part V
(HIP/WMMA)同時針對**兩個** RDNA 世代撰寫。

| Box | GPU | ISA | Role / 角色 |
|---|---|---|---|
| **A** | Radeon 8060S (Strix Halo iGPU) | `gfx1151` / RDNA 3.5 | RDNA3.5 kernel testing |
| **B** | 4× Navi 48 (RX 9070-class) | `gfx1201` / RDNA 4 | Main dev; RDNA4 kernel testing |

Parts I–IV (PyTorch/ROCm training + quantization) run on either box. Part V
kernels have per-ISA variants.

Parts I–IV(PyTorch/ROCm 訓練 + 量化)兩台皆可跑。Part V kernel 有各世代版本。

---

## Setup / 安裝

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# ROCm 7.0 build of PyTorch (matches ROCm 7.x on both boxes):
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.0
pip install -r env/requirements.txt
```

See `env/rocm_setup.md` for AMD/ROCm-specific notes.
詳見 `env/rocm_setup.md`。

---

## Repo layout / 專案結構

```
peps/         reusable library (encoders, projector, aggregators, wrapper, metrics, train)
apps/         image / texture / sdf applications
notebooks/    weekly teaching spine (W01..W12)
hip/          RDNA3.5 + RDNA4 HIP/WMMA kernels
docs/         bilingual textbook (markdown)
slides/       bilingual Marp decks
data/         download.py for Kodak / AmbientCG / Stanford meshes
results/      generated tables & figures
```

The four deliverables (notebook / slides / repo / written docs) are **four views
of this one repo**, so they never drift apart.

四種交付(notebook / 投影片 / repo / 書面文件)是**同一個 repo 的四種視圖**,永不脫節。

---

## License / 授權

MIT (course code). PEPS itself has **no official code** — this is a faithful
reimplementation of the paper's Algorithm 1.

MIT(課程程式碼)。PEPS 本身**無官方程式碼**,本專案為論文 Algorithm 1 的忠實重現。
