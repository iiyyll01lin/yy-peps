"""Generate W10 — quantization study (original contribution).

繁體中文:生成 W10 量化研究 notebook(原創貢獻)。對 NTC 基線與 PEPS 做 int8 PTQ,
畫招牌圖:PSNR-vs-bitrate 率失真曲線(四條線:NTC / NTC+quant / PEPS / PEPS+quant),
檢驗 PEPS 的 -25% 參數優勢在量化後是否保持。執行:python notebooks/_gen_w10.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_id = [0]


def _nid():
    _id[0] += 1
    return f"w10c{_id[0]:03d}"


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
        md("# W10 · Does PEPS's edge survive quantization? / PEPS 的優勢在量化後還在嗎?",
           "",
           "**English.** The paper reports PEPS's parameter/quality advantage in full",
           "precision but never quantizes. Real texture codecs (NVIDIA RTXNTC, BCn) ship",
           "quantized weights. This is our original chapter: apply int8 post-training",
           "quantization to latents + MLP weights and draw **rate-distortion curves**",
           "(PSNR vs effective bitrate). If the PEPS gap holds after quantization, the",
           "advantage is real for deployment; if it collapses, that's an honest finding.",
           "",
           "**繁體中文.** 論文以全精度報告 PEPS 的參數/品質優勢,卻從未量化。真實材質編碼器",
           "(NVIDIA RTXNTC、BCn)都出貨量化權重。這是我們的原創章節:對 latent + MLP 權重",
           "做 int8 事後量化,畫**率失真曲線**(PSNR vs 有效位元率)。若 PEPS 差距在量化後",
           "維持,優勢對部署為真;若崩潰,也是誠實的發現。"),
        code("import sys, os; sys.path.insert(0, os.path.abspath('..'))",
             "import copy, torch, numpy as np, matplotlib.pyplot as plt",
             "from apps.image.data import load_image, image_to_coords_targets, find_kodak",
             "from apps.image.build import build_grid, build_grid_peps",
             "from peps.train import fit, TrainConfig, render_full, auto_device",
             "from peps.metrics import psnr",
             "from peps.quant.ptq import quantize_model, model_bitrate",
             "device = auto_device(); print('device', device)"),
        md("## 1. Train full-precision NTC baseline and PEPS / 訓練全精度 NTC 基線與 PEPS"),
        code("img = load_image(find_kodak(1), max_size=384)",
             "coords, targets, (H, W) = image_to_coords_targets(img)",
             "def train(builder_out, steps=2500):",
             "    model, pc = builder_out",
             "    fit(model, coords, targets, TrainConfig(steps=steps, batch_size=32768, lr=1e-2, device=device))",
             "    return model, pc",
             "# 'NTC' baseline here = plain grid+MLP; 'PEPS' = Grid-PEPS (concat)",
             "ntc_model,  ntc_pc  = train(build_grid(resolution=128, feature_dim=8))",
             "peps_model, peps_pc = train(build_grid_peps(128, 8, 6, 'concat'))",
             "def eval_psnr(m):",
             "    pred = render_full(m, coords, device=device).reshape(H, W, 3).clamp(0, 1)",
             "    return psnr(pred, img)",
             "print(f'NTC  fp: params={ntc_pc}  PSNR={eval_psnr(ntc_model):.2f}')",
             "print(f'PEPS fp: params={peps_pc}  PSNR={eval_psnr(peps_model):.2f}')"),
        md("## 2. Quantize at several bit widths -> rate-distortion points / 多位元寬量化 -> 率失真點",
           "For each of {8,6,4} bits we clone the trained model, fake-quantize, and record",
           "(effective bits/param, PSNR). Effective bitrate x param-count = storage.",
           "對 {8,6,4} 位元各複製模型、模擬量化,記錄(有效位元/參數, PSNR)。"),
        code("def rd_curve(model, pc, bit_widths=(8, 6, 4)):",
             "    pts = []",
             "    for b in bit_widths:",
             "        m = copy.deepcopy(model)",
             "        quantize_model(m, weight_bits=b, latent_bits=b)",
             "        eff = model_bitrate(m, latent_bits=b, weight_bits=b)",
             "        # storage proxy = effective bits/param * params (lower-left is better)",
             "        pts.append((eff, eval_psnr(m), b))",
             "    return pts",
             "",
             "ntc_fp  = (32.0, eval_psnr(ntc_model))",
             "peps_fp = (32.0, eval_psnr(peps_model))",
             "ntc_q   = rd_curve(ntc_model, ntc_pc)",
             "peps_q  = rd_curve(peps_model, peps_pc)",
             "print('NTC  quant:', [(b, round(p,2)) for _,p,b in ntc_q])",
             "print('PEPS quant:', [(b, round(p,2)) for _,p,b in peps_q])"),
        md("## 3. The money figure — rate-distortion / 招牌圖:率失真"),
        code("plt.figure(figsize=(7, 5))",
             "def line(fp, q, label, style):",
             "    xs = [fp[0]] + [e for e, _, _ in q]",
             "    ys = [fp[1]] + [p for _, p, _ in q]",
             "    plt.plot(xs, ys, style, label=label, ms=8)",
             "    for e, p, b in q: plt.annotate(f'{b}b', (e, p), textcoords='offset points', xytext=(4, 4))",
             "line(ntc_fp,  ntc_q,  'NTC (grid)',   'o--')",
             "line(peps_fp, peps_q, 'PEPS',         's-')",
             "plt.xlabel('effective bits / param  (lower = smaller)')",
             "plt.ylabel('PSNR (dB)')",
             "plt.title('Rate-distortion: does the PEPS gap survive int8?')",
             "plt.gca().invert_xaxis(); plt.legend(); plt.grid(True, alpha=0.3); plt.show()"),
        md("## 4. Gap analysis / 差距分析"),
        code("for b_idx, b in enumerate((8, 6, 4)):",
             "    gap = peps_q[b_idx][1] - ntc_q[b_idx][1]",
             "    print(f'{b}-bit: PEPS - NTC = {gap:+.2f} dB')",
             "fp_gap = peps_fp[1] - ntc_fp[1]",
             "print(f'fp32 : PEPS - NTC = {fp_gap:+.2f} dB')",
             "print('\\nIf the gap stays positive as bits drop, PEPS advantage survives quantization.')",
             "import csv",
             "os.makedirs('../results', exist_ok=True)",
             "with open('../results/w10_rate_distortion.csv', 'w', newline='') as f:",
             "    w = csv.writer(f); w.writerow(['method', 'bits', 'eff_bits', 'psnr'])",
             "    w.writerow(['ntc', 32, 32.0, round(ntc_fp[1],3)])",
             "    w.writerow(['peps', 32, 32.0, round(peps_fp[1],3)])",
             "    for e,p,b in ntc_q:  w.writerow(['ntc', b, round(e,2), round(p,3)])",
             "    for e,p,b in peps_q: w.writerow(['peps', b, round(e,2), round(p,3)])",
             "print('saved ../results/w10_rate_distortion.csv')"),
        md("## 5. Narrative hook -> BCn / cooperative vectors / 敘事掛勾",
           "RTXNTC decodes with cooperative-vector int8 matrix ops; BCn is fixed-rate",
           "block compression. Our int8 PTQ is the software analogue — the same question",
           "(quality at low bitrate) the hardware path optimizes. W11-W12 take this to",
           "actual HIP/WMMA int8 kernels.",
           "",
           "RTXNTC 用 cooperative-vector int8 矩陣運算解碼;BCn 是定率區塊壓縮。我們的",
           "int8 PTQ 是其軟體對應 —— 同一個問題(低位元率下的品質)。W11-W12 帶到真正的",
           "HIP/WMMA int8 kernel。"),
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5,
}

if __name__ == "__main__":
    with open(os.path.join(HERE, "W10_quantization.ipynb"), "w", encoding="utf-8") as f:
        json.dump(NB, f, ensure_ascii=False, indent=1)
    print("wrote W10_quantization.ipynb")
