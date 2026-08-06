"""Generate W08 with separate course-fast and paper-exact texture tracks."""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_id = [0]


def _nid():
    _id[0] += 1
    return f"w08c{_id[0]:03d}"


def _source(lines):
    text = "\n".join(lines).split("\n")
    return [part + "\n" for part in text[:-1]] + [text[-1]]


def md(*lines):
    return {
        "cell_type": "markdown",
        "id": _nid(),
        "metadata": {},
        "source": _source(lines),
    }


def code(*lines):
    return {
        "cell_type": "code",
        "id": _nid(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _source(lines),
    }


NB = {
    "cells": [
        md(
            "# W08 · Paper texture compression / 論文材質壓縮",
            "",
            "`course_fast` is a deterministic synthetic smoke run. `paper_exact` is the",
            "18-set native-4K Table 2 protocol: every available RGB map is a target,",
            "NTC_N uses G0 corner concatenation + G1 bilinear sampling + a 3-octave",
            "8-texel tiled triangular encoding, and all methods train with GELU,",
            "dual learning rates (grid 0.1 / MLP 0.001), L1, and cosine decay for",
            "3,000 × 40 batches of 60,000 pixel coordinates.",
            "",
            "`course_fast` 是可重現的合成 smoke run；`paper_exact` 才是 18 組原生 4K",
            "Table 2。每張可用 RGB map 都獨立計分，最後依 AO/ARM/DIFF/Displacement/",
            "metal/normal/rough/specular 類型與全域彙總。",
        ),
        code(
            "import os, sys, json, subprocess",
            "sys.path.insert(0, os.path.abspath('..'))",
            "PROFILE = os.environ.get('PEPS_PROFILE', 'course_fast')",
            "if PROFILE not in {'course_fast', 'paper_exact'}:",
            "    raise ValueError('PEPS_PROFILE must be course_fast or paper_exact')",
            "print('profile:', PROFILE)",
        ),
        md(
            "## 1. Inspect the exact NTC_N input / 檢查精確 NTC_N 輸入",
            "The paper configuration supplies 48 G0 values, 20 G1 values, and 12",
            "tiled-encoding values: 80 decoder inputs. This structural check does not",
            "claim a quality result.",
        ),
        code(
            "from apps.texture.build import build_paper_texture",
            "model, params = build_paper_texture('ntc_n', num_textures=5)",
            "encoder = model[0]",
            "print({'decoder_inputs': encoder.feature_dim,",
            "       'g0_values': encoder.g0.feature_dim,",
            "       'g1_values': encoder.g1.feature_dim,",
            "       'tiled_values': encoder.tiled_encoding.feature_dim,",
            "       'parameters': params})",
            "assert encoder.feature_dim == 80",
        ),
        md(
            "## 2. Check data and hardware / 檢查資料與硬體",
            "The readiness report is machine-readable. Missing maps are errors; the",
            "loader never invents texture channels.",
        ),
        code(
            "cmd = [sys.executable, '-m', 'experiments.reproduce', 'check',",
            "       '--profile', PROFILE, '--artifact', 'texture-table2']",
            "check = subprocess.run(cmd, text=True, capture_output=True)",
            "print(check.stdout)",
            "if check.stderr: print(check.stderr)",
        ),
        md(
            "## 3. Execute the selected track / 執行所選軌",
            "`course_fast` performs a two-step real optimization and writes a run",
            "manifest. The paper run is intentionally opt-in because it is 18 × 11",
            "models × 120,000 optimizer steps at 4K.",
        ),
        code(
            "if PROFILE == 'course_fast':",
            "    run_cmd = [sys.executable, '-m', 'experiments.reproduce', 'smoke',",
            "               '--task', 'texture']",
            "elif os.environ.get('RUN_PAPER_EXACT') == '1':",
            "    run_cmd = [sys.executable, '-m', 'experiments.reproduce', 'run',",
            "               '--artifact', 'texture-table2']",
            "else:",
            "    run_cmd = None",
            "    print('Paper run not started. Set RUN_PAPER_EXACT=1 after prerequisites pass.')",
            "if run_cmd:",
            "    completed = subprocess.run(run_cmd, check=True, text=True, capture_output=True)",
            "    receipt = json.loads(completed.stdout)",
            "    print(json.dumps(receipt, indent=2))",
        ),
        md(
            "## 4. RTXNTC proxy policy / RTXNTC proxy 規則",
            "The local multi-grid module is an **unverified RTXNTC-inspired proxy**,",
            "not an equivalent implementation. It is available only for course",
            "discussion and is excluded from paper Table 2.",
        ),
        code(
            "from apps.texture.rtxntc import build_rtxntc_proxy",
            "proxy, proxy_params = build_rtxntc_proxy()",
            "print({'label': 'rtxntc_proxy_unverified', 'params': proxy_params,",
            "       'paper_table2_member': False})",
        ),
        md(
            "## 5. Budget versus loss sensitivity / 預算與損失函數的敏感度",
            "Table 2 reproduces the paper's qualitative pattern but falls about",
            "0.44 dB short on the `NTC_PEPS` minus `NTC_N` gain, which is enough to",
            "stop `NTC_PEPS` overtaking `BI-Grid`. Two bounded probes in",
            "`results/texture_repro/budget_probe/` test the two obvious causes.",
            "",
            "Retraining at 240k and 480k steps makes the gain *shrink*, so compute is",
            "not the cause. Changing how that L1 is reduced across a set's maps",
            "multiplies the gain sixfold on one material and does nothing on another.",
            "Read the committed CSV rather than re-running anything.",
        ),
        code(
            "import csv, pathlib",
            "",
            "root = pathlib.Path('..').resolve()",
            "curves = root / 'results/texture_repro/budget_probe/curves.csv'",
            "rows = list(csv.DictReader(curves.open(newline='', encoding='utf-8')))",
            "",
            "def gap(loss, instance, seed, steps):",
            "    for row in rows:",
            "        if (row['loss'], row['instance'], row['seed'],",
            "                row['optimizer_steps']) == (loss, instance, seed, steps):",
            "            return float(row['peps_advantage_db'])",
            "    return None",
            "",
            "print('budget effect, global L1 (advantage shrinks):')",
            "for inst, seed in (('paving-stones-070', '0'), ('metal-plates-013', '0')):",
            "    track = [(s, gap('global_l1', inst, seed, s))",
            "             for s in ('120000', '240000', '480000')]",
            "    shown = ' -> '.join(f'{int(s)//1000}k {v:+.4f}'",
            "                        for s, v in track if v is not None)",
            "    print(f'  {inst} seed{seed}: {shown}')",
            "",
            "print('loss effect at matched budget (advantage grows):')",
            "for steps in ('240000', '480000'):",
            "    a = gap('global_l1', 'paving-stones-070', '0', steps)",
            "    b = gap('per_map_normalised_l1', 'paving-stones-070', '0', steps)",
            "    if a and b:",
            "        print(f'  {int(steps)//1000}k: global_l1 {a:+.4f} -> '",
            "              f'per_map {b:+.4f}  ({b / a:.2f}x)')",
        ),
        md(
            "The paper does report the Table 2 recipe, L1 included, and this run",
            "uses L1, so the loss family is not a deviation. What `L1` leaves open",
            "is the reduction across maps. Every map is decoded as three output",
            "channels, and the frozen recipe reduces one L1 globally over all of",
            "them, weighting each map's mean absolute error equally. Normalising",
            "per map instead divides each term by its own magnitude, handing more",
            "gradient to maps that are already accurate. PEPS is strongest on the",
            "smooth maps, so that reweighting is what makes its advantage visible.",
            "",
            "A second material overturns the general reading. On `metal-plates-013`",
            "the same swap changes almost nothing, +2.3199 to +2.2105 dB. The",
            "sixfold swing belongs to `paving-stones-070`, not to the reduction:",
            "per-map normalisation only unlocks advantage a global reduction was",
            "hiding, and on `metal-plates-013` PEPS already leads by a wide margin.",
            "",
            "So the lesson is not that the reduction explains the shortfall. It is",
            "that an unreported detail one level below the published recipe can move",
            "a per-set result several fold while leaving another material untouched.",
            "Rank causes by measured sensitivity, and check a mechanism replicates",
            "before believing it.",
            "",
            "Both probes stay labelled `bounded_budget_probe_not_paper_comparable`:",
            "two sets, at most two seeds, and losses of our own construction.",
        ),
        md(
            "## 6. Where the shortfall actually comes from",
            "All eleven methods land about 1.15 dB below the paper. Before blaming",
            "the implementation, read what the number averages: `table2.json`",
            "declares `map_weighted`, so it is a plain mean over the 76 individual",
            "maps. Map categories span 19 dB, and the paper never published its file",
            "list, so composition is a free parameter.",
        ),
        code(
            "import csv, pathlib",
            "",
            "root = pathlib.Path('..').resolve()",
            "comp = root / 'results/texture_repro/shortfall_analysis/composition.csv'",
            "cat = list(csv.DictReader(comp.open(newline='', encoding='utf-8')))",
            "",
            "total = sum(int(r['map_count']) for r in cat)",
            "lo, hi = cat[0], cat[-1]",
            "spread = float(hi['mean_psnr_db']) - float(lo['mean_psnr_db'])",
            "per_swap = spread / total",
            "shortfall = 1.1538",
            "",
            "for r in cat:",
            "    print(f\"  {r['map_category']:<14}{r['map_count']:>3} maps \"",
            "          f\"{r['share_percent']:>6}%  {float(r['mean_psnr_db']):8.3f} dB\")",
            "print(f'category spread {lo[\'map_category\']} -> {hi[\'map_category\']}'",
            "      f' = {spread:.3f} dB over {total} maps')",
            "print(f'one swap moves the mean {per_swap:.4f} dB; closing {shortfall} dB'",
            "      f' needs {shortfall / per_swap:.1f} swaps'",
            "      f' = {100 * shortfall / per_swap / total:.1f}% of the selection')",
        ),
        md(
            "Under 6% of the selection accounts for the whole shortfall, and a",
            "category-balanced selection would have 1.83x the headroom needed. The",
            "near-uniform offset across all eleven methods and the close SSIM match",
            "both fit a content difference rather than an algorithmic error.",
            "",
            "It does not explain the *ordering* though: composition moves every",
            "method almost equally, yet we still rank the Grid family above the NTC",
            "family where the paper does the reverse. That is the open question.",
            "",
            "Note the process lesson too. This costs no GPU time and uses data that",
            "already existed, but it was run only after several GPU-hours of loss",
            "probes. Cheap variance decomposition belongs first.",
        ),
        md(
            "## 7. Result contract / 結果契約",
            "Read `summary.csv` only from the `run_dir` printed above. Its values are",
            "means of per-map PSNR/SSIM rows; each run directory also contains",
            "`manifest.json` and `instances.csv`. Legacy `results/table2_texture.csv`",
            "remains explicitly unverified and is never imported by this notebook.",
        ),
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


if __name__ == "__main__":
    path = os.path.join(HERE, "W08_texture_ntc.ipynb")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(NB, handle, ensure_ascii=False, indent=1)
    print("wrote W08_texture_ntc.ipynb")
