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
            "## 6. Budget versus loss sensitivity / 預算與損失函數的敏感度",
            "Table 2 reproduces the paper's qualitative pattern but falls about",
            "0.44 dB short on the `NTC_PEPS` minus `NTC_N` gain, which is enough to",
            "stop `NTC_PEPS` overtaking `BI-Grid`. Two bounded probes in",
            "`results/texture_repro/budget_probe/` test the two obvious causes.",
            "",
            "Retraining at 240k and 480k steps makes the gain *shrink*, so compute is",
            "not the cause. Changing only the training loss multiplies the gain by",
            "roughly eight. Read the committed CSV rather than re-running anything.",
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
            "Every map occupies exactly three output channels, so the frozen global",
            "L1 already weights each map's mean absolute error equally. Normalising",
            "per map instead divides each term by its own magnitude, which hands more",
            "gradient to the maps that are already accurate. PEPS is strongest on the",
            "smooth maps, so that reweighting is what makes its advantage visible.",
            "",
            "This is the reproduction lesson worth keeping: Table 2 scores a per-map",
            "average of PSNR, a relative log-domain quantity, while the recipe",
            "optimises one absolute global L1. The paper never reports its texture",
            "loss, and that single unreported choice moves the headline effect far",
            "more than any budget change we measured. Rank candidate causes by",
            "measured sensitivity before spending GPU time.",
            "",
            "Both probes stay labelled `bounded_budget_probe_not_paper_comparable`:",
            "two sets, at most two seeds, and a loss of our own construction.",
        ),
        md(
            "## 5. Result contract / 結果契約",
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
