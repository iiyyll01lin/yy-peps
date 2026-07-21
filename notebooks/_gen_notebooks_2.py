"""Generate teaching notebooks W04-W07 (PEPS core + image Table 1).

繁體中文:生成 W04-W07 notebook。W04 建 PEPS wrapper 並驗證 Identity==APE;
W05 在 Kodak 上訓 Grid-PEPS 重現 Table 1 列;W06 驗證 1/f PSD 並實作 Pink;
W07 完整 Table 1 全指標。執行:python notebooks/_gen_notebooks_2.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_id = [0]


def _next_id():
    _id[0] += 1
    return f"cell{_id[0]:03d}"


def md(*lines):
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "id": _next_id(), "metadata": {},
            "execution_count": None, "outputs": [], "source": _src(lines)}


def _src(lines):
    text = "\n".join(lines).split("\n")
    return [p + "\n" for p in text[:-1]] + [text[-1]]


def notebook(cells):
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"}},
            "nbformat": 4, "nbformat_minor": 5}


def write(name, nb):
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("wrote", name)


BOOT = code(
    "import sys, os; sys.path.insert(0, os.path.abspath('..'))",
    "import torch, matplotlib.pyplot as plt",
    "from peps.train import auto_device",
    "device = auto_device(); print('device', device)",
)


# ---------------------------------------------------------------- W04
W04 = notebook([
    md("# W04 · Building the PEPS wrapper / 建立 PEPS wrapper",
       "",
       "**English.** We assemble the full pipeline `Project -> Encode -> Aggregate ->",
       "Model` (paper Eq. 8) and confirm the identity-encoder recovers APE, so PEPS is",
       "a proper generalization. We then swap in a learned grid to get **Grid-PEPS**.",
       "",
       "**繁體中文.** 組裝完整流程 `投影->編碼->聚合->模型`(論文式 8),確認 identity",
       "編碼器退化回 APE,故 PEPS 為正確的泛化;再換入可學習 grid 得到 **Grid-PEPS**。"),
    BOOT,
    md("## 1. Assemble PEPS by hand / 手動組裝 PEPS"),
    code("from peps import Projector, GridEncoder, MLP, PEPS, make_aggregator",
         "L = 6",
         "proj = Projector(num_frequencies=L)",
         "enc  = GridEncoder(dim=2, resolution=128, feature_dim=4)",
         "agg  = make_aggregator('concat', proj.num_points, enc.feature_dim)",
         "mlp  = MLP(agg.out_dim, out_dim=3, hidden_dim=64, num_layers=3)",
         "model = PEPS(proj, enc, agg, mlp)",
         "print('points', proj.num_points, '| agg out', agg.out_dim,",
         "      '| params', sum(p.numel() for p in model.parameters()))",
         "print(model(torch.rand(8, 2)).shape)"),
    md("## 2. Identity-encoder PEPS == APE (numerical proof) / Identity 等價 APE(數值證明)"),
    code("from peps import AbsolutePositionalEncoding",
         "x = torch.rand(400, 2)",
         "peps_feat = proj(x).reshape(x.shape[0], -1)",
         "ape_feat  = AbsolutePositionalEncoding(2, L, include_input=True)(x)",
         "A = torch.cat([ape_feat, torch.ones(x.shape[0], 1)], 1)",
         "resid = (A @ torch.linalg.lstsq(A, peps_feat).solution - peps_feat).abs().max()",
         "print(f'affine residual APE->PEPS(identity): {resid.item():.2e}')"),
    md("## 3. Takeaway / 小結",
       "The wrapper is the reusable object every application uses. Next week we train",
       "Grid-PEPS on Kodak and reproduce Table 1. 下週在 Kodak 上訓 Grid-PEPS 重現 Table 1。"),
])


# ---------------------------------------------------------------- W05
W05 = notebook([
    md("# W05 · Grid-PEPS on images (Table 1) / 影像上的 Grid-PEPS(Table 1)",
       "",
       "**English.** Train a grid baseline and Grid-PEPS on a Kodak image and compare",
       "PSNR at matched parameter budgets — the G-PEPS rows of paper Table 1. We also",
       "draw the dual-scatter comparison (paper Fig. 7 style).",
       "",
       "**繁體中文.** 在 Kodak 影像上訓練 grid baseline 與 Grid-PEPS,在相同參數預算下",
       "比較 PSNR —— 即論文 Table 1 的 G-PEPS 列;並畫雙散點對比(Fig.7 風格)。"),
    BOOT,
    code("from apps.image.data import load_image, image_to_coords_targets, find_kodak",
         "from apps.image.build import build_grid, build_grid_peps",
         "from peps.train import fit, TrainConfig, render_full",
         "from peps.metrics import psnr, ssim",
         "img = load_image(find_kodak(1), max_size=384)",
         "coords, targets, (H, W) = image_to_coords_targets(img)",
         "print('image', H, W)"),
    md("## 1. Train grid vs Grid-PEPS at matched budget / 相同預算下對比"),
    code("def train_eval(builder_out, steps=2500):",
         "    model, pc = builder_out",
         "    fit(model, coords, targets,",
         "        TrainConfig(steps=steps, batch_size=32768, lr=1e-2, device=device))",
         "    pred = render_full(model, coords, device=device).reshape(H, W, 3).clamp(0, 1)",
         "    return pc, psnr(pred, img), ssim(pred, img), pred",
         "",
         "rows = {}",
         "rows['grid']      = train_eval(build_grid(resolution=128, feature_dim=8))",
         "rows['grid_peps'] = train_eval(build_grid_peps(resolution=128, feature_dim=8, num_frequencies=6))",
         "for k, (pc, ps, ss, _) in rows.items():",
         "    print(f'{k:10s} params={pc:8d}  PSNR={ps:6.2f} dB  SSIM={ss:.4f}')"),
    md("## 2. Visual + dual-scatter (Fig. 7 style) / 視覺與雙散點"),
    code("fig, ax = plt.subplots(1, 3, figsize=(12, 4))",
         "ax[0].imshow(img); ax[0].set_title('target'); ax[0].axis('off')",
         "ax[1].imshow(rows['grid'][3]);      ax[1].set_title(f\"grid {rows['grid'][1]:.1f} dB\"); ax[1].axis('off')",
         "ax[2].imshow(rows['grid_peps'][3]); ax[2].set_title(f\"Grid-PEPS {rows['grid_peps'][1]:.1f} dB\"); ax[2].axis('off')",
         "plt.show()"),
    md("## 3. Save Table 1 row / 存 Table 1 列",
       "Results are appended to `results/table1_image.csv` for the W07 wrap-up."),
    code("import csv, os",
         "os.makedirs('../results', exist_ok=True)",
         "path = '../results/table1_image.csv'",
         "new = not os.path.exists(path)",
         "with open(path, 'a', newline='') as f:",
         "    w = csv.writer(f)",
         "    if new: w.writerow(['method', 'params', 'psnr', 'ssim'])",
         "    for k, (pc, ps, ss, _) in rows.items():",
         "        w.writerow([k, pc, round(ps, 3), round(ss, 4)])",
         "print('saved', path)"),
])


# ---------------------------------------------------------------- W06
W06 = notebook([
    md("# W06 · Pink-PEPS & the 1/f story / Pink-PEPS 與 1/f",
       "",
       "**English.** Natural images have power spectra that fall off roughly as",
       "`1/f^alpha`. Pink-PEPS exploits this by allocating latent capacity **inversely",
       "to frequency**, so it matches Grid-PEPS quality with fewer parameters (the",
       "paper's ~-25% result). We first verify the 1/f slope (Fig. 3), then train Pink.",
       "",
       "**繁體中文.** 自然影像的功率譜大致以 `1/f^alpha` 衰減。Pink-PEPS 依此把 latent",
       "容量**與頻率成反比**分配,以更少參數達到 Grid-PEPS 品質(論文約 -25%)。先驗證",
       "1/f 斜率(Fig.3),再訓練 Pink。"),
    BOOT,
    md("## 1. Verify 1/f PSD on Kodak (reproduce Fig. 3) / 驗證 1/f(重現 Fig.3)"),
    code("from apps.image.data import load_image, find_kodak",
         "from peps.spectral import radial_psd, fit_one_over_f",
         "import numpy as np",
         "img = load_image(find_kodak(1), max_size=512)",
         "freqs, psd = radial_psd(img, nbins=80)",
         "alpha = fit_one_over_f(freqs, psd)",
         "print(f'estimated 1/f slope alpha = {alpha:.2f}')",
         "plt.loglog(freqs, psd, '.'); plt.xlabel('radial freq'); plt.ylabel('power')",
         "plt.title(f'Radial PSD (slope ~ {alpha:.2f})'); plt.grid(True, which='both', alpha=0.3); plt.show()"),
    md("## 2. Train Pink-PEPS vs Grid-PEPS / 訓練 Pink 對比 Grid"),
    code("from apps.image.data import image_to_coords_targets",
         "from apps.image.build import build_grid_peps",
         "from peps.train import fit, TrainConfig, render_full",
         "from peps.metrics import psnr",
         "coords, targets, (H, W) = image_to_coords_targets(img)",
         "def run(agg):",
         "    model, pc = build_grid_peps(resolution=128, feature_dim=8, num_frequencies=6, aggregator=agg)",
         "    fit(model, coords, targets, TrainConfig(steps=2500, batch_size=32768, lr=1e-2, device=device))",
         "    pred = render_full(model, coords, device=device).reshape(H, W, 3).clamp(0, 1)",
         "    return pc, psnr(pred, img)",
         "c_pc, c_ps = run('concat')",
         "p_pc, p_ps = run('pink')",
         "print(f'concat: params={c_pc}  PSNR={c_ps:.2f}')",
         "print(f'pink  : params={p_pc}  PSNR={p_ps:.2f}  ({100*(1-p_pc/c_pc):.0f}% fewer params)')"),
    md("## 3. Takeaway / 小結",
       "Pink matches concat quality with fewer parameters — capacity follows the",
       "signal's spectrum. Pink 以更少參數達到 concat 品質,容量跟著訊號頻譜走。"),
])


# ---------------------------------------------------------------- W07
W07 = notebook([
    md("# W07 · Application 1 wrap-up: full Table 1 / 應用一收尾:完整 Table 1",
       "",
       "**English.** Aggregate all methods across several Kodak images with all metrics",
       "(PSNR, SSIM, LSD). We emphasize **LSD** (log-spectral distance): it measures",
       "high-frequency fidelity, exactly where PEPS wins and plain PSNR under-reports.",
       "",
       "**繁體中文.** 匯總多張 Kodak 上所有方法與所有指標(PSNR、SSIM、LSD)。特別強調",
       "**LSD**(對數頻譜距離):量測高頻保真度,正是 PEPS 勝出、而純 PSNR 低估之處。"),
    BOOT,
    code("from apps.image.data import load_image, image_to_coords_targets, find_kodak",
         "from apps.image.build import build_grid, build_grid_peps",
         "from peps.train import fit, TrainConfig, render_full",
         "from peps.metrics import psnr, ssim, lsd",
         "import numpy as np",
         "methods = {",
         "  'grid':       lambda: build_grid(resolution=128, feature_dim=8),",
         "  'grid_peps':  lambda: build_grid_peps(128, 8, 6, 'concat'),",
         "  'pink_peps':  lambda: build_grid_peps(128, 8, 6, 'pink'),",
         "}",
         "images = [1, 5, 19]   # a few Kodak images (subset for speed)"),
    md("## 1. Sweep methods x images / 掃方法 x 影像"),
    code("results = {m: {'psnr': [], 'ssim': [], 'lsd': []} for m in methods}",
         "for idx in images:",
         "    img = load_image(find_kodak(idx), max_size=384)",
         "    coords, targets, (H, W) = image_to_coords_targets(img)",
         "    for m, builder in methods.items():",
         "        model, pc = builder()",
         "        fit(model, coords, targets, TrainConfig(steps=2000, batch_size=32768, lr=1e-2, device=device))",
         "        pred = render_full(model, coords, device=device).reshape(H, W, 3).clamp(0, 1)",
         "        results[m]['psnr'].append(psnr(pred, img))",
         "        results[m]['ssim'].append(ssim(pred, img))",
         "        results[m]['lsd'].append(lsd(pred, img))",
         "    print('done image', idx)"),
    md("## 2. Table 1 summary / Table 1 匯總"),
    code("print(f\"{'method':12s} {'PSNR':>7s} {'SSIM':>7s} {'LSD':>7s}\")",
         "for m in methods:",
         "    ps = np.mean(results[m]['psnr']); ss = np.mean(results[m]['ssim']); ls = np.mean(results[m]['lsd'])",
         "    print(f'{m:12s} {ps:7.2f} {ss:7.4f} {ls:7.3f}')",
         "print('\\nLower LSD = better high-frequency fidelity (PEPS advantage).')"),
    md("## 3. Takeaway / 小結",
       "PEPS variants lead on PSNR and especially LSD; Pink does it with fewer params.",
       "This closes Application 1. Next: neural texture compression (W08).",
       "",
       "PEPS 變體在 PSNR、尤其 LSD 領先;Pink 用更少參數達成。應用一到此結束,",
       "接著:神經材質壓縮(W08)。"),
])


if __name__ == "__main__":
    write("W04_peps_wrapper.ipynb", W04)
    write("W05_grid_peps_image.ipynb", W05)
    write("W06_pink_peps.ipynb", W06)
    write("W07_image_table1.ipynb", W07)
