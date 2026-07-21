"""Generate the teaching notebooks W01-W03 as .ipynb files.

繁體中文:用程式生成 W01-W03 教學 notebook,避免手刻 JSON 出錯。每個 notebook
交替 markdown(雙語講解)與可執行的 code cell。之後各週可依同一模式擴充。
執行:python notebooks/_gen_notebooks.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _src(lines),
    }


def _src(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def notebook(cells) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(name: str, nb: dict) -> None:
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("wrote", path)


BOOT = code(
    "# Repo bootstrap: make `peps` and `apps` importable from the notebook.",
    "import sys, os",
    "sys.path.insert(0, os.path.abspath('..'))",
    "import torch",
    "from peps.train import auto_device",
    "device = auto_device()",
    "print('torch', torch.__version__, '| device', device)",
)


# ------------------------------------------------------------------ W01
W01 = notebook([
    md(
        "# W01 · Implicit Neural Representations & spectral bias",
        "# W01 · 隱式神經表示與頻譜偏差",
        "",
        "**English.** An *implicit neural representation* (INR) stores a signal as a",
        "function `f(coordinate) -> value` realized by a neural network, instead of an",
        "array of pixels. A plain coordinate MLP suffers from **spectral bias**: it",
        "learns low frequencies fast and high frequencies slowly (or never), so the",
        "reconstruction looks blurry. This motivates positional encoding (W02) and",
        "learned grids (W03), which PEPS unifies.",
        "",
        "**繁體中文.** 隱式神經表示(INR)把訊號存成一個由神經網路實現的函式",
        "`f(座標)->數值`,而非像素陣列。純座標 MLP 有**頻譜偏差**:低頻學得快、",
        "高頻學得慢(甚至學不到),重建結果糊糊的。這正是位置編碼(W02)與可學習",
        "grid(W03)的動機,而 PEPS 把兩者統一起來。",
    ),
    BOOT,
    md("## 1. Load one Kodak image / 載入一張 Kodak 影像"),
    code(
        "from apps.image.data import load_image, image_to_coords_targets, find_kodak",
        "img = load_image(find_kodak(1), max_size=256)   # (H, W, 3) in [0,1]",
        "coords, targets, (H, W) = image_to_coords_targets(img)",
        "print('image', H, W, '| coords', coords.shape, '| targets', targets.shape)",
        "import matplotlib.pyplot as plt",
        "plt.imshow(img); plt.title('target Kodak image'); plt.axis('off'); plt.show()",
    ),
    md(
        "## 2. Fit a plain MLP (no positional encoding) / 純 MLP 擬合(無位置編碼)",
        "Watch it converge to a blurry, low-frequency version. 觀察它收斂成糊掉的低頻版本。",
    ),
    code(
        "from apps.image.build import build_plain_mlp",
        "from peps.train import fit, TrainConfig, render_full",
        "from peps.metrics import psnr",
        "",
        "model, pc = build_plain_mlp(num_frequencies=0)   # raw (x,y) input",
        "losses = []",
        "fit(model, coords, targets,",
        "    TrainConfig(steps=1500, batch_size=16384, lr=1e-2, device=device),",
        "    on_log=lambda s, l: losses.append((s, l)))",
        "pred = render_full(model, coords, device=device).reshape(H, W, 3).clamp(0, 1)",
        "print(f'plain MLP: params={pc}  PSNR={psnr(pred, img):.2f} dB')",
        "",
        "fig, ax = plt.subplots(1, 2, figsize=(9, 4))",
        "ax[0].imshow(img); ax[0].set_title('target'); ax[0].axis('off')",
        "ax[1].imshow(pred); ax[1].set_title(f'plain MLP ({psnr(pred, img):.1f} dB)'); ax[1].axis('off')",
        "plt.show()",
    ),
    md(
        "## 3. The spectral-bias diagnostic / 頻譜偏差診斷",
        "Compare the FFT magnitude of target vs reconstruction; the high-frequency",
        "ring is missing. 比較目標與重建的 FFT 幅值,高頻環會缺失。",
    ),
    code(
        "import torch",
        "def logmag(x):",
        "    g = x.mean(-1)  # grayscale",
        "    F = torch.fft.fftshift(torch.fft.fft2(g))",
        "    return torch.log(F.abs() + 1e-6)",
        "fig, ax = plt.subplots(1, 2, figsize=(9, 4))",
        "ax[0].imshow(logmag(img), cmap='magma'); ax[0].set_title('target FFT'); ax[0].axis('off')",
        "ax[1].imshow(logmag(pred), cmap='magma'); ax[1].set_title('plain MLP FFT'); ax[1].axis('off')",
        "plt.show()",
    ),
    md(
        "## 4. Takeaway / 小結",
        "Plain coordinate MLPs cannot represent fine detail. Next week we add",
        "**positional encoding** and view it through the Lissajous lens that PEPS uses.",
        "",
        "純座標 MLP 無法表示細節。下週加入**位置編碼**,並用 PEPS 的 Lissajous 視角來理解它。",
    ),
])


# ------------------------------------------------------------------ W02
W02 = notebook([
    md(
        "# W02 · Positional encoding & the Lissajous view",
        "# W02 · 位置編碼與 Lissajous 視角",
        "",
        "**English.** Absolute positional encoding (APE) maps `x` to",
        "`[x, sin(2^i pi x), cos(2^i pi x), ...]` for `i=1,...,L`. PEPS",
        "reinterprets each frequency as",
        "a point moving on a **Lissajous curve**: `S_i=(1+sin(x phi_i))/2`,",
        "`C_i=(1+cos(x phi_i))/2`. This notebook reproduces the point-motion picture",
        "(paper Fig. 2) and shows the Identity-encoder affine equivalence to APE.",
        "",
        "**繁體中文.** 絕對位置編碼(APE)把 `x` 映成",
        "`[x, sin(2^i pi x), cos(2^i pi x), ...]`(`i=1,...,L`)。PEPS 把每個頻率",
        "重新詮釋為在",
        "**Lissajous 曲線**上移動的點:`S_i=(1+sin)/2`、`C_i=(1+cos)/2`。本 notebook",
        "重現點運動圖(論文 Fig.2),並展示 Identity encoder 與 APE 的等價。",
    ),
    BOOT,
    md("## 1. Lissajous point motion (reproduce Fig. 2) / Lissajous 點運動(重現 Fig.2)"),
    code(
        "import torch, matplotlib.pyplot as plt",
        "from peps import Projector",
        "L = 4",
        "proj = Projector(num_frequencies=L, include_input=True)",
        "xs = torch.linspace(0, 1, 400).unsqueeze(1).repeat(1, 2)  # 2D coord sweep",
        "pts = proj(xs)   # (400, 2L+1, 2)",
        "print('num points per coord =', proj.num_points)",
        "",
        "fig, ax = plt.subplots(figsize=(5, 5))",
        "for p in range(pts.shape[1]):",
        "    ax.plot(pts[:, p, 0], pts[:, p, 1], lw=1)",
        "ax.set_title('Lissajous trajectories of projected points')",
        "ax.set_xlabel('x-channel'); ax.set_ylabel('y-channel'); ax.set_aspect('equal')",
        "plt.show()",
    ),
    md(
        "## 2. Identity PEPS is affinely equivalent to APE / Identity PEPS 與 APE 仿射等價",
        "If the shared encoder is the identity, PEPS's projected+concatenated features",
        "are an affine transform of APE features — same expressive power. 若共享編碼器",
        "為 identity,PEPS 的投影+串接特徵是 APE 特徵的仿射變換,表達力相同。",
    ),
    code(
        "from peps import IdentityEncoder, AbsolutePositionalEncoding",
        "x = torch.rand(500, 2)",
        "pts = proj(x).reshape(x.shape[0], -1)                 # PEPS+Identity features",
        "ape = AbsolutePositionalEncoding(2, L, include_input=True)(x)",
        "print('PEPS feat dim', pts.shape[1], '| APE feat dim', ape.shape[1])",
        "# least-squares fit APE -> PEPS: near-zero residual proves affine equivalence",
        "A = torch.cat([ape, torch.ones(x.shape[0], 1)], 1)",
        "sol = torch.linalg.lstsq(A, pts).solution",
        "resid = (A @ sol - pts).abs().max().item()",
        "print(f'max residual = {resid:.2e}  -> affine-equivalent' )",
    ),
    md(
        "## 3. Takeaway / 小結",
        "PEPS *generalizes* APE: swap the identity encoder for a learned grid and the",
        "same projection machinery becomes a powerful learned encoder (W03-W05).",
        "",
        "PEPS 是 APE 的**泛化**:把 identity 換成可學習 grid,同一套投影機制就變成強大的",
        "可學習編碼器(W03-W05)。",
    ),
])


# ------------------------------------------------------------------ W03
W03 = notebook([
    md("# W03 · Grid bottleneck and paper Figure 5",
       "# W03 · Grid 瓶頸與論文 Figure 5",
       "",
       "The paper sweeps **all 4×4 combinations** of grid resolutions",
       "`[16,32,64,128]` and feature widths `[8,16,32,64]` for BI-grid, LPE,",
       "and three-frequency Grid-PEPS on native-4K images with L1. The paper does",
       "not identify those images, optimizer, batch size, or training steps, so an",
       "exact claim is blocked until a checksum manifest and explicit assumptions",
       "are supplied. `course_fast` remains a separate, runnable smoke path.",
       "",
       "論文對 BI-grid、LPE、三頻 Grid-PEPS 執行 4×4 解析度/特徵寬度 sweep。",
       "由於論文未公開影像清單與完整訓練預算，本 notebook 不會拿 Kodak 或論文數字",
       "冒充本機 Figure 5 結果。"),
    BOOT,
    md("## 1. Choose a profile / 選擇執行軌"),
    code(
        "import subprocess, json",
        "PROFILE = os.environ.get('PEPS_PROFILE', 'course_fast')",
        "if PROFILE not in {'course_fast', 'paper_exact'}:",
        "    raise ValueError('PEPS_PROFILE must be course_fast or paper_exact')",
        "print('profile:', PROFILE)",
    ),
    md(
        "## 2. Structural sweep oracle / Sweep 結構 oracle",
        "This builds the exact matrix and checks dimensions only; it does not emit",
        "quality numbers.",
    ),
    code(
        "from apps.image.build import build_paper_fig5",
        "matrix = []",
        "for method in ('bi_grid', 'lpe', 'grid_peps'):",
        "    for resolution in (16, 32, 64, 128):",
        "        for feature_dim in (8, 16, 32, 64):",
        "            model, params = build_paper_fig5(method, resolution=resolution, feature_dim=feature_dim)",
        "            matrix.append((method, resolution, feature_dim, params))",
        "print('matrix entries:', len(matrix))",
        "assert len(matrix) == 3 * 4 * 4",
    ),
    md("## 3. Machine-readable readiness / 機器可讀 readiness"),
    code(
        "check_cmd = [sys.executable, '-m', 'experiments.reproduce', 'check',",
        "             '--profile', PROFILE, '--artifact', 'image-fig5']",
        "if os.environ.get('FIG5_MANIFEST'):",
        "    check_cmd += ['--fig5-manifest', os.environ['FIG5_MANIFEST']]",
        "checked = subprocess.run(check_cmd, text=True, capture_output=True)",
        "print(checked.stdout)",
    ),
    md(
        "## 4. Run with provenance / 以 provenance 執行",
        "`course_fast` executes a real two-step image optimization. `paper_exact`",
        "requires `FIG5_MANIFEST`, `FIG5_STEPS`, and explicit opt-in; its manifest",
        "will remain labelled `protocol_assumption`.",
    ),
    code(
        "if PROFILE == 'course_fast':",
        "    run_cmd = [sys.executable, '-m', 'experiments.reproduce', 'smoke', '--task', 'image']",
        "elif os.environ.get('RUN_PAPER_EXACT') == '1':",
        "    run_cmd = [sys.executable, '-m', 'experiments.reproduce', 'run',",
        "               '--artifact', 'image-fig5', '--fig5-manifest', os.environ['FIG5_MANIFEST'],",
        "               '--assumed-steps', os.environ['FIG5_STEPS'], '--allow-protocol-assumptions']",
        "else:",
        "    run_cmd = None",
        "    print('Paper run not started; exact dataset/training details are unavailable.')",
        "if run_cmd:",
        "    completed = subprocess.run(run_cmd, check=True, text=True, capture_output=True)",
        "    print(completed.stdout)",
    ),
    md("## 5. Result contract / 結果契約",
       "Use only `summary.csv` beside the printed `manifest.json`; never import a",
       "legacy CSV or copy the curve from the publication."),
])
for index, cell in enumerate(W03["cells"]):
    cell.setdefault("id", f"w03c{index:03d}")


if __name__ == "__main__":
    write("W01_intro_inr.ipynb", W01)
    write("W02_positional_encoding.ipynb", W02)
    write("W03_grid_bottleneck.ipynb", W03)
