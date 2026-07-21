"""Generate W09 with course-fast and full paper SDF protocols."""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_id = [0]


def _nid():
    _id[0] += 1
    return f"w09c{_id[0]:03d}"


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
            "# W09 · Paper signed-distance functions / 論文 SDF",
            "",
            "`course_fast` is a tiny analytic SDF smoke run. `paper_exact` uses Lucy,",
            "Pitted Stonefish, Thai Statue, and Armadillo as provenance-tracked 512³",
            "volumes. Table 3 runs PE/LPE/grid/hash/multi-grid/multi-hash and every",
            "PEPS counterpart with MAPE; the appendix repeats all methods with L1.",
            "Table 4 is the actual Pitted Stonefish 1×/8× comparison, not a small-torus",
            "analogy.",
            "",
            "`course_fast` 是小型解析 SDF smoke run；`paper_exact` 使用四個具 provenance",
            "的 512³ 實例。Table 4 會真正跑 Pitted Stonefish 的全部 1×/8× 方法。",
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
            "## 1. Coordinate and boundary oracle / 座標與邊界 oracle",
            "Stored distances are measured in centered `[-1,1]³` units while the model",
            "receives `[0,1]³`. Therefore an exact SDF has input-gradient norm 2.",
            "Course eikonal samples stay in `[h,1-h]³`; clamping perturbed points at",
            "the boundary while dividing by `2h` is mathematically wrong. Paper runs",
            "use no eikonal term.",
        ),
        code(
            "import torch",
            "from apps.sdf.data import SDF_COORDINATE_SCALE, sample_sdf_tensor",
            "line = torch.arange(2, dtype=torch.float32)",
            "z, y, x = torch.meshgrid(line, line, line, indexing='ij')",
            "volume = x + 10*y + 100*z",
            "probe = sample_sdf_tensor(volume, torch.tensor([[0.,0.,0.],[1.,1.,1.],[.5,.5,.5]]))",
            "print({'coordinate_scale': SDF_COORDINATE_SCALE, 'boundary_probe': probe[:,0].tolist()})",
            "assert torch.allclose(probe[:,0], torch.tensor([0.,111.,55.5]))",
        ),
        md(
            "## 2. Verify the real paper inputs / 驗證真實論文輸入",
            "The report checks all four volume/provenance pairs and GPU visibility.",
            "Pitted Stonefish requires the authorized Academic-only CT asset.",
        ),
        code(
            "artifact = 'sdf-table3-mape' if PROFILE == 'paper_exact' else 'sdf-table3-mape'",
            "cmd = [sys.executable, '-m', 'experiments.reproduce', 'check',",
            "       '--profile', PROFILE, '--artifact', artifact]",
            "checked = subprocess.run(cmd, text=True, capture_output=True)",
            "print(checked.stdout)",
            "if checked.stderr: print(checked.stderr)",
        ),
        md(
            "## 3. Execute course-fast / 執行 course-fast",
            "This is a real two-step optimization with a generated input hash, raw",
            "per-instance rows, and a run manifest. It is not paper evidence.",
        ),
        code(
            "receipt = None",
            "if PROFILE == 'course_fast':",
            "    completed = subprocess.run(",
            "        [sys.executable, '-m', 'experiments.reproduce', 'smoke', '--task', 'sdf'],",
            "        check=True, text=True, capture_output=True)",
            "    receipt = json.loads(completed.stdout)",
            "    print(json.dumps(receipt, indent=2))",
        ),
        md(
            "## 4. Exact commands / 精確指令",
            "Full runs are opt-in. Each command streams 512³ IoU evaluation, checkpoints",
            "training, and writes its own manifest. The SDF converter limitation is",
            "retained in `verification_status`; results are never copied from the paper.",
        ),
        code(
            "commands = [",
            "  [sys.executable, '-m', 'experiments.reproduce', 'run', '--artifact', 'sdf-table3-mape'],",
            "  [sys.executable, '-m', 'experiments.reproduce', 'run', '--artifact', 'sdf-table3-l1'],",
            "  [sys.executable, '-m', 'experiments.reproduce', 'run', '--artifact', 'sdf-table4'],",
            "]",
            "for command in commands: print(' '.join(command))",
            "if PROFILE == 'paper_exact' and os.environ.get('RUN_PAPER_EXACT') == '1':",
            "    for command in commands:",
            "        completed = subprocess.run(command, check=True, text=True, capture_output=True)",
            "        print(completed.stdout)",
            "elif PROFILE == 'paper_exact':",
            "    print('Not started. Set RUN_PAPER_EXACT=1 only after the readiness report passes.')",
        ),
        md(
            "## 5. Table 4 budget invariant / Table 4 預算不變量",
            "Doubling every 3D resolution and increasing hash caps by three bits gives",
            "exactly 8× learned encoder parameters. PE has no learned encoder and is",
            "shared between the two rows.",
        ),
        code(
            "from apps.sdf.build import build_paper_sdf",
            "from peps.train import split_encoder_decoder_parameters",
            "for method in ('grid','hash','lpe','grid_peps','m_peps','m_grid','m_hash','m_hashpeps'):",
            "    small, _ = build_paper_sdf(method, encoder_parameter_multiplier=1)",
            "    large, _ = build_paper_sdf(method, encoder_parameter_multiplier=8)",
            "    small_e, _ = split_encoder_decoder_parameters(small)",
            "    large_e, _ = split_encoder_decoder_parameters(large)",
            "    counts = (sum(p.numel() for p in small_e), sum(p.numel() for p in large_e))",
            "    print(method, counts)",
            "    assert counts[1] == 8 * counts[0]",
        ),
        md(
            "## 6. Result contract / 結果契約",
            "Use only the `summary.csv` inside a printed run directory. Legacy",
            "`results/table3_sdf.csv` is an unverified torus teaching artifact and is",
            "never imported here.",
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
    path = os.path.join(HERE, "W09_sdf.ipynb")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(NB, handle, ensure_ascii=False, indent=1)
    print("wrote W09_sdf.ipynb")
