"""Generate W08 — neural texture compression notebook.

繁體中文:生成 W08 材質壓縮 notebook。在 4 組 AmbientCG 材質上比較 NTC 基線 /
Grid-PEPS / NTC_PEPS(pink),重現 Table 2 的逐材質 PSNR,並與 NVIDIA RTXNTC
的 MetalPlates013 並排(視覺)。執行:python notebooks/_gen_w08.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_id = [0]


def _nid():
    _id[0] += 1
    return f"w08c{_id[0]:03d}"


def md(*l):
    return {"cell_type": "markdown", "id": _nid(), "metadata": {}, "source": _s(l)}


def code(*l):
    return {"cell_type": "code", "id": _nid(), "metadata": {},
            "execution_count": None, "outputs": [], "source": _s(l)}


def _s(l):
    t = "\n".join(l).split("\n")
    return [p + "\n" for p in t[:-1]] + [t[-1]]


NB = {
    "cells": [
        md("# W08 · Application 2 — Neural Texture Compression / 神經材質壓縮",
           "",
           "**English.** A PBR material is a 9-channel signal (albedo RGB, normal XY,",
           "roughness, metalness, AO). We fit it with an NTC-style grid baseline and with",
           "PEPS variants, then compare per-material PSNR — reproducing the structure of",
           "paper Table 2. MetalPlates013 is exactly NVIDIA RTXNTC's demo set, giving a",
           "direct point of comparison. PEPS's advantage is largest on high-frequency",
           "metals and smallest on low-frequency wood — we show this honestly.",
           "",
           "**繁體中文.** PBR 材質是 9 通道訊號(albedo RGB、normal XY、roughness、",
           "metalness、AO)。用 NTC 風格 grid 基線與 PEPS 變體擬合,比較逐材質 PSNR,",
           "重現論文 Table 2 結構。MetalPlates013 正是 NVIDIA RTXNTC 的示範材質,可直接",
           "對照。PEPS 優勢在高頻金屬最大、低頻木頭最小 —— 誠實呈現。"),
        code("import sys, os; sys.path.insert(0, os.path.abspath('..'))",
             "import torch, numpy as np, matplotlib.pyplot as plt",
             "from peps.train import auto_device, fit, TrainConfig, render_full",
             "from peps.metrics import psnr",
             "from apps.texture.data import load_pbr_bundle, bundle_to_coords_targets, find_bundle, CHANNEL_LAYOUT",
             "from apps.texture.build import build_ntc_baseline, build_grid_peps_texture",
             "device = auto_device(); print('device', device, '| bundle channels', sum(c for _,c in CHANNEL_LAYOUT))"),
        md("## 1. Materials spanning the frequency spectrum / 涵蓋頻率光譜的材質"),
        code("sets = ['MetalPlates013', 'Metal032', 'Planks020', 'Rock023']",
             "notes = {'MetalPlates013':'high-freq metal (NVIDIA demo)', 'Metal032':'metal',",
             "         'Planks020':'low-freq wood', 'Rock023':'mid-freq noisy'}",
             "bundles = {}",
             "for s in sets:",
             "    b = load_pbr_bundle(find_bundle(s), size=512)",
             "    bundles[s] = b",
             "    print(f'{s:16s} {b.shape}  ({notes[s]})')"),
        md("## 2. Train NTC vs Grid-PEPS vs NTC_PEPS per material / 逐材質訓練三方法",
           "Matched grid resolution/feature dim; PEPS samples the shared grid at 2L+1",
           "Lissajous points. 相同 grid 解析度/特徵維度;PEPS 在 2L+1 個點取樣共享 grid。"),
        code("methods = {",
             "  'ntc':       lambda: build_ntc_baseline(resolution=256, feature_dim=8),",
             "  'grid_peps': lambda: build_grid_peps_texture(256, 8, 6, 'concat'),",
             "  'ntc_peps':  lambda: build_grid_peps_texture(256, 8, 6, 'pink'),",
             "}",
             "table = {m: {} for m in methods}",
             "recon = {}  # keep MetalPlates013 recons for the side-by-side",
             "for s in sets:",
             "    b = bundles[s]",
             "    coords, targets, (H, W) = bundle_to_coords_targets(b)",
             "    for m, builder in methods.items():",
             "        model, pc = builder()",
             "        fit(model, coords, targets, TrainConfig(steps=2000, batch_size=32768, lr=1e-2, device=device))",
             "        pred = render_full(model, coords, device=device).reshape(H, W, -1).clamp(0, 1)",
             "        table[m][s] = psnr(pred, b)",
             "        if s == 'MetalPlates013': recon[m] = pred",
             "    print('done', s)"),
        md("## 3. Table 2 — per-material PSNR / 逐材質 PSNR"),
        code("print(f\"{'material':16s} \" + ' '.join(f'{m:>10s}' for m in methods))",
             "for s in sets:",
             "    print(f'{s:16s} ' + ' '.join(f'{table[m][s]:10.2f}' for m in methods))",
             "print()",
             "for m in methods:",
             "    print(f'{m:16s} mean PSNR = {np.mean(list(table[m].values())):.2f} dB')"),
        md("## 4. RTXNTC side-by-side on MetalPlates013 (albedo) / 與 RTXNTC 並排(albedo)"),
        code("fig, ax = plt.subplots(1, 4, figsize=(14, 4))",
             "ax[0].imshow(bundles['MetalPlates013'][..., :3]); ax[0].set_title('target albedo'); ax[0].axis('off')",
             "for i, m in enumerate(methods):",
             "    ax[i+1].imshow(recon[m][..., :3])",
             "    ax[i+1].set_title(f'{m}\\n{table[m][\"MetalPlates013\"]:.1f} dB'); ax[i+1].axis('off')",
             "plt.suptitle('MetalPlates013 — same set NVIDIA RTXNTC demos'); plt.show()"),
        md("## 5. Save + takeaway / 存檔與小結",
           "PEPS variants lead on metals; the gap narrows on low-frequency wood — the",
           "honest picture the paper's Table 2 also shows.",
           "",
           "PEPS 變體在金屬領先;低頻木頭差距縮小 —— 這正是論文 Table 2 的誠實圖像。"),
        code("import csv",
             "os.makedirs('../results', exist_ok=True)",
             "with open('../results/table2_texture.csv', 'w', newline='') as f:",
             "    w = csv.writer(f); w.writerow(['material'] + list(methods))",
             "    for s in sets: w.writerow([s] + [round(table[m][s], 3) for m in methods])",
             "print('saved ../results/table2_texture.csv')"),
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5,
}

if __name__ == "__main__":
    with open(os.path.join(HERE, "W08_texture_ntc.ipynb"), "w", encoding="utf-8") as f:
        json.dump(NB, f, ensure_ascii=False, indent=1)
    print("wrote W08_texture_ntc.ipynb")
