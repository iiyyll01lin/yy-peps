"""Generate teaching notebooks W04-W07 (PEPS core + image Table 1).

繁體中文:生成 W04-W07 notebook。W04 建 PEPS wrapper 並驗證 Identity 與 APE 仿射等價;
W05 在 Kodak 上訓 Grid-PEPS 重現 Table 1 列;W06 驗證 1/f PSD 並實作 Pink;
W07 完整 Table 1 全指標。執行:python notebooks/_gen_notebooks_2.py
"""

from __future__ import annotations

import os

import nbformat

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
    path = os.path.join(HERE, name)
    document = nbformat.from_dict(nb)
    nbformat.validate(document)
    nbformat.write(document, path)
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
       "Model` (paper Eq. 8) and confirm that identity-encoder PEPS features are",
       "affinely equivalent to APE features. We then swap in a learned grid to get",
       "**Grid-PEPS**.",
       "",
       "**繁體中文.** 組裝完整流程 `投影->編碼->聚合->模型`(論文式 8),確認 identity",
       "編碼器的 PEPS 特徵與 APE 特徵仿射等價;再換入可學習 grid 得到 **Grid-PEPS**。"),
    BOOT,
    md("## 1. Assemble PEPS by hand / 手動組裝 PEPS"),
    code("from peps import Projector, GridEncoder, MLP, PEPS, make_aggregator",
         "L = 6",
         "proj = Projector(num_frequencies=L)",
         "enc  = GridEncoder(dim=2, resolution=128, feature_dim=4)",
         "agg  = make_aggregator('concat', proj.num_points, enc.feature_dim)",
         "mlp  = MLP(agg.out_dim, out_dim=3, hidden_dim=64, num_layers=4)",
         "model = PEPS(proj, enc, agg, mlp)",
         "print('points', proj.num_points, '| agg out', agg.out_dim,",
         "      '| params', sum(p.numel() for p in model.parameters()))",
         "print(model(torch.rand(8, 2)).shape)"),
    md("## 2. Identity PEPS is affinely equivalent to APE / Identity PEPS 與 APE 仿射等價"),
    code("from peps import AbsolutePositionalEncoding",
         "x = torch.rand(400, 2)",
         "peps_feat = proj(x).reshape(x.shape[0], -1)",
         "ape_feat  = AbsolutePositionalEncoding(2, L, include_input=True)(x)",
         "A = torch.cat([ape_feat, torch.ones(x.shape[0], 1)], 1)",
         "resid = (A @ torch.linalg.lstsq(A, peps_feat).solution - peps_feat).abs().max()",
         "print(f'affine residual APE->PEPS(identity): {resid.item():.2e}')"),
    md("## 3. Eq. (8) input-delta ablation / 式(8) 輸入 delta 消融",
       "The full Eq. (8) is `M(A(E(P_1)..E(P_{2L+1})), delta)`. Setting `delta=True`",
       "concatenates the raw coords to the aggregated vector before the decoder — a",
       "few extra params that let the MLP see exact position, not only sampled latents.",
       "With a **learned grid** the latents already encode position, so we expect the",
       "effect to be small/noisy; we average over seeds and report honestly. It is off",
       "by default.",
       "",
       "式(8) 完整為 `M(A(E(P_1)..E(P_{2L+1})), delta)`。`delta=True` 把原始座標接到聚合",
       "向量後再進解碼器——只多幾個參數,讓 MLP 看到精確位置而非僅取樣的 latent。由於",
       "**可學習 grid** 的 latent 已編碼位置,預期效果小且有雜訊;故對多個 seed 取平均、",
       "誠實呈現。預設關閉。"),
    code("from apps.image.data import load_image, image_to_coords_targets, find_kodak",
         "from apps.image.build import build_grid_peps",
         "from peps.train import fit, TrainConfig, render_full",
         "from peps.metrics import psnr",
         "import numpy as np",
         "img = load_image(find_kodak(1), max_size=256)",
         "coords, targets, (H, W) = image_to_coords_targets(img)",
         "def run_delta(delta, seed, steps=1500):",
         "    torch.manual_seed(seed)",
         "    model, pc = build_grid_peps(resolution=96, feature_dim=6, num_frequencies=6, aggregator='concat', delta=delta)",
         "    fit(model, coords, targets, TrainConfig(steps=steps, batch_size=32768, lr=1e-2, device=device))",
         "    pred = render_full(model, coords, device=device).reshape(H, W, 3).clamp(0, 1)",
         "    return pc, psnr(pred, img)",
         "# average over seeds: the grid_sample backward is non-deterministic on GPU,",
         "# and with a learned grid the delta effect is small, so a single run is noisy.",
         "seeds = [0, 1, 2]",
         "off = [run_delta(False, s) for s in seeds]",
         "on  = [run_delta(True,  s) for s in seeds]",
         "off_pc, on_pc = off[0][0], on[0][0]",
         "off_ps, on_ps = float(np.mean([p for _, p in off])), float(np.mean([p for _, p in on]))",
         "for i, s in enumerate(seeds):",
         "    print(f'seed {s}: delta_off={off[i][1]:.2f}  delta_on={on[i][1]:.2f}  ({on[i][1]-off[i][1]:+.2f} dB)')",
         "print(f'mean/{len(seeds)} seeds: off={off_ps:.2f}  on={on_ps:.2f}  effect={on_ps-off_ps:+.2f} dB  (+{on_pc-off_pc} params)')",
         "import csv, os",
         "os.makedirs('../results', exist_ok=True)",
         "with open('../results/delta_ablation.csv', 'w', newline='') as f:",
         "    w = csv.writer(f, lineterminator='\\n'); w.writerow(['delta', 'params', 'psnr_mean', 'n_seeds'])",
         "    w.writerow(['off', off_pc, round(off_ps, 3), len(seeds)])",
         "    w.writerow(['on', on_pc, round(on_ps, 3), len(seeds)])",
         "print('saved ../results/delta_ablation.csv')"),
    md("## 4. Takeaway / 小結",
       "The wrapper is the reusable object every application uses, and `delta` is the",
       "optional Eq. (8) skip. Next week we train Grid-PEPS on Kodak and reproduce",
       "Table 1. 下週在 Kodak 上訓 Grid-PEPS 重現 Table 1。"),
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
       "`1/f^alpha`. Pink-PEPS follows Algorithm 1 exactly: for latent width `d`,",
       "`a_i=max(1,floor(d/2^i))`, `G_i=sum_{j=0}^i a_j`, and the grouped point order",
       "`(x,S_1..S_L,C_1..C_L)` uses opposite circular slices for sine and cosine.",
       "For `d=8,L=6`, this deterministically changes the aggregation dimension from",
       "104 to 28. That is a structural invariant, not a matched-PSNR or total-parameter",
       "claim; rerun the cells below before reporting empirical savings.",
       "",
       "**繁體中文.** 自然影像的功率譜大致以 `1/f^alpha` 衰減。Pink-PEPS 依此把 latent",
       "容量依論文 Algorithm 1 分配:`a_i=max(1,floor(d/2^i))`,以累積 `G_i` 決定",
       "sin/cos 反向的 circular slices,並嚴守 `(x,S_1..S_L,C_1..C_L)` 點順序。",
       "`d=8,L=6` 時聚合維度必然由 104 變為 28;這是結構不變量,不是等 PSNR 或總參數",
       "結論。下列實驗須重跑後才能報告節省幅度。"),
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
    md("## 2. Re-run Pink vs Concat on the same grid / 同一 grid 重跑 Pink vs Concat",
       "Both share the 128x128x8 grid; only the aggregator differs. Measure quality",
       "again after the corrected allocation instead of carrying forward old results.",
       "兩者共用 128x128x8 grid,只差聚合器。修正配置後重新量測,不沿用舊結果。"),
    code("from apps.image.data import image_to_coords_targets",
         "from apps.image.build import build_grid_peps",
         "from peps.train import fit, TrainConfig, render_full",
         "from peps.metrics import psnr",
         "coords, targets, (H, W) = image_to_coords_targets(img)",
         "def run(agg, res=128, fd=8, steps=2500):",
         "    model, pc = build_grid_peps(resolution=res, feature_dim=fd, num_frequencies=6, aggregator=agg)",
         "    fit(model, coords, targets, TrainConfig(steps=steps, batch_size=32768, lr=1e-2, device=device))",
         "    pred = render_full(model, coords, device=device).reshape(H, W, 3).clamp(0, 1)",
         "    return pc, psnr(pred, img), model.aggregator.out_dim",
         "c_pc, c_ps, c_od = run('concat')",
         "p_pc, p_ps, p_od = run('pink')",
         "print(f'concat: params={c_pc}  agg_out_dim={c_od}  PSNR={c_ps:.2f} dB')",
         "print(f'pink  : params={p_pc}  agg_out_dim={p_od}  PSNR={p_ps:.2f} dB')",
         "print(f'  measured dPSNR={p_ps-c_ps:+.2f} dB; '",
         "      f'aggregation {100*(1-p_od/c_od):.0f}% smaller; total {100*(1-p_pc/c_pc):.0f}% smaller')"),
    md("## 3. Recompute the total-parameter frontier / 重算總參數前緣",
       "The shared grid and corrected decoder depth both affect the total budget.",
       "Record the complete rerun rather than inferring total savings from out_dim.",
       "共享 grid 與修正後的 decoder 深度都影響總預算;應記錄完整重跑結果,不可只由",
       "out_dim 推論總參數節省。"),
    code("import csv, os",
         "rows = [('concat', 128, 8, c_pc, c_od, c_ps), ('pink', 128, 8, p_pc, p_od, p_ps)]",
         "for res in (120, 112):",
         "    pc, ps, od = run('pink', res=res)",
         "    rows.append(('pink', res, 8, pc, od, ps))",
         "print(f\"{'agg':7s}{'res':>5s}{'params':>9s}{'agg_od':>8s}{'PSNR':>8s}{'total-':>9s}{'dPSNR':>8s}\")",
         "for agg, res, fd, pc, od, ps in rows:",
         "    print(f'{agg:7s}{res:5d}{pc:9d}{od:8d}{ps:8.2f}{100*(1-pc/c_pc):8.0f}%{ps-c_ps:+8.2f}')",
         "os.makedirs('../results', exist_ok=True)",
         "with open('../results/pink_param_savings.csv', 'w', newline='') as f:",
         "    w = csv.writer(f, lineterminator='\\n')",
         "    w.writerow(['aggregator','resolution','feature_dim','params','pct_fewer_total','agg_out_dim','pct_fewer_agg','psnr','dpsnr_vs_concat'])",
         "    for agg, res, fd, pc, od, ps in rows:",
         "        w.writerow([agg, res, fd, pc, round(100*(1-pc/c_pc),1), od, round(100*(1-od/c_od),1), round(ps,3), round(ps-c_ps,3)])",
         "print('saved ../results/pink_param_savings.csv')"),
    md("## 4. Takeaway / 小結",
       "The exact allocation and circular slices are now testable invariants. Quality",
       "and total-parameter savings remain empirical outputs of this rerun.",
       "",
       "精確配置與 circular slices 現在是可測試的不變量;品質與總參數節省仍須由本次",
       "重跑產生。"),
])


# ---------------------------------------------------------------- W07
W07 = notebook([
    md("# W07 · Full Kodak Table 1 / 完整 Kodak Table 1",
       "",
       "`paper_exact` covers all 24 checksum-verified original-orientation Kodak",
       "images and all nine published methods: PE, LPE, NTC_N, Grid, G-PEPS,",
       "G-P-PEPS, NTC_PEPS, NTC_PinkPEPS, and G-P-PEPS-25. Every image records",
       "PSNR, official LDR-FLIP, AlexNet LPIPS, the frozen LSD oracle, and windowed",
       "torchmetrics SSIM.",
       "",
       "論文主文稱 L1/fixed-LR/leaky-ReLU，但 Table 1 數字與 appendix 的",
       "L2/GELU/dual-LR/cosine 列相同。本 runner 採後者以對應已出版表格，並在",
       "manifest 標記此衝突與未公開的訓練步數假設。"),
    BOOT,
    code("import subprocess, json",
         "PROFILE = os.environ.get('PEPS_PROFILE', 'course_fast')",
         "if PROFILE not in {'course_fast', 'paper_exact'}:",
         "    raise ValueError('PEPS_PROFILE must be course_fast or paper_exact')",
         "print('profile:', PROFILE)"),
    md("## 1. Metric oracle / 指標 oracle",
       "The exact implementation/version receipt is embedded in every run manifest."),
    code("from peps.metrics import metric_oracles, metric_versions",
         "print(json.dumps(metric_oracles(), indent=2))",
         "print(json.dumps(metric_versions(), indent=2))"),
    md("## 2. Readiness / 執行條件"),
    code("check_cmd = [sys.executable, '-m', 'experiments.reproduce', 'check',",
         "             '--profile', PROFILE, '--artifact', 'image-table1']",
         "checked = subprocess.run(check_cmd, text=True, capture_output=True)",
         "print(checked.stdout)",
         "if checked.stderr: print(checked.stderr)"),
    md("## 3. Execute with a manifest / 以 manifest 執行",
       "`course_fast` runs a deterministic two-step image smoke test. The complete",
       "24×9 paper workload is opt-in and never launched from a default notebook run."),
    code("if PROFILE == 'course_fast':",
         "    run_cmd = [sys.executable, '-m', 'experiments.reproduce', 'smoke', '--task', 'image']",
         "elif os.environ.get('RUN_PAPER_EXACT') == '1':",
         "    run_cmd = [sys.executable, '-m', 'experiments.reproduce', 'run',",
         "               '--artifact', 'image-table1', '--allow-protocol-assumptions']",
         "else:",
         "    run_cmd = None",
         "    print('Paper run not started. Set RUN_PAPER_EXACT=1 after readiness passes.')",
         "if run_cmd:",
         "    completed = subprocess.run(run_cmd, check=True, text=True, capture_output=True)",
         "    print(completed.stdout)"),
    md("## 4. Result contract / 結果契約",
       "Only a printed run directory containing `manifest.json`, `instances.csv`,",
       "and `summary.csv` can support a numeric claim. The legacy",
       "`results/table1_image.csv` is retained as unverified course history and is",
       "never imported by this notebook."),
])


if __name__ == "__main__":
    write("W04_peps_wrapper.ipynb", W04)
    write("W05_grid_peps_image.ipynb", W05)
    write("W06_pink_peps.ipynb", W06)
    write("W07_image_table1.ipynb", W07)
