"""Generate W13 (project kickoff) and W14 (project showcase) capstone notebooks.

繁體中文:生成 W13(專題起跑)與 W14(專題成果展)notebook。這是 12 週教學結束後的
兩週專題。學生從四條軌道擇一(3D 短片 / 量化校準 / 新聚合器 / 端到端效能),
W13 選題並重跑基線,W14 跑出延伸、產出可驗證 artifacts 並自評。這兩份是 scaffold
(骨架),程式格皆為安全佔位,學生填入 `TODO(student)` 處。完整說明見 docs/06_capstone.md。
執行:python notebooks/_gen_w13_w14.py
"""

from __future__ import annotations

import os

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
_id = [0]


def _nid():
    _id[0] += 1
    return f"capc{_id[0]:03d}"


def md(*l):
    return {"cell_type": "markdown", "id": _nid(), "metadata": {}, "source": _s(l)}


def code(*l):
    return {"cell_type": "code", "id": _nid(), "metadata": {},
            "execution_count": None, "outputs": [], "source": _s(l)}


def _s(l):
    t = "\n".join(l).split("\n")
    return [p + "\n" for p in t[:-1]] + [t[-1]]


def nb(cells):
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"}},
            "nbformat": 4, "nbformat_minor": 5}


BOOT = code(
    "import sys, os; sys.path.insert(0, os.path.abspath('..'))",
    "import math, torch, matplotlib.pyplot as plt",
    "from peps.train import auto_device",
    "def _required_text(value, name):",
    "    if not isinstance(value, str) or not value.strip() or 'TODO' in value:",
    "        raise ValueError(f'{name} must be nonblank and contain no TODO')",
    "    return value.strip()",
    "device = auto_device(); print('device', device)",
)

# The four capstone tracks, kept in one place so both notebooks agree with
# docs/06_capstone.md. Each: goal, starting files, and the concrete deliverable.
TRACKS = code(
    "# The four capstone tracks (see docs/06_capstone.md for the full brief + rubric).",
    "TRACKS = {",
    "  'a': {'name': 'Short 3D video volume (x, y, t)',",
    "        'start': ['peps/wrapper.py', 'peps/projector.py', 'peps/encoders/grid.py',",
    "                  'apps/image/ (as a template)', 'peps/train.py'],",
    "        'deliverable': '3D grid vs 3D Grid-PEPS on one small licensed clip',",
    "        'csv': 'results/capstone_video3d.csv'},",
    "  'b': {'name': 'Quantization calibration or short QAT recovery',",
    "        'start': ['peps/quant/ptq.py', 'notebooks/W10_quantization.ipynb',",
    "                  'tests/test_quantization.py'],",
    "        'deliverable': 'clipping/calibration or QAT vs rerun per-channel PTQ',",
    "        'csv': 'results/capstone_quant_calibration.csv'},",
    "  'c': {'name': 'Design + evaluate a new aggregator (beyond concat/pink/brownian)',",
    "        'start': ['peps/aggregate.py', 'apps/image/build.py',",
    "                  'notebooks/W06_pink_peps.ipynb'],",
    "        'deliverable': 'new aggregator kind + params-vs-PSNR vs the existing three',",
    "        'csv': 'results/capstone_aggregator.csv'},",
    "  'd': {'name': 'End-to-end PEPS runtime optimization and receipt',",
    "        'start': ['hip/wmma_mlp.hip', 'hip/fused_peps_kernel.hip',",
    "                  'hip/bench_latency.sh', 'tests/test_hip_parity.py'],",
    "        'deliverable': 'one optimization vs rerun full-pipeline baseline; parity + latency',",
    "        'csv': 'results/capstone_runtime.csv'},",
    "}",
    "",
    "# >>> Pick your track here <<<",
    "TRACK = 'a'  # TODO(student): one of 'a' | 'b' | 'c' | 'd'",
    "t = TRACKS[TRACK]",
    "print('Track', TRACK, '-', t['name'])",
    "print('Starting files:'); [print('  -', s) for s in t['start']]",
    "print('Deliverable   :', t['deliverable'])",
    "print('Results CSV   :', t['csv'])",
)


# ----------------------------------------------------------------- W13
W13 = nb([
    md("# W13 · Project kickoff / 專題起跑",
       "",
       "**English.** The 12-week spine is done — you can now build the PEPS wrapper,",
       "run the course protocols, quantize, and touch HIP. The capstone (W13–W14) turns that",
       "into a small research contribution. **This week (W13):** pick one of four tracks,",
       "rerun the *baseline you intend to compare*, and lock a success metric.",
       "**Next week (W14):** run the extension and produce a notebook, nonblank CSV,",
       "run manifest, submission manifest, and slide. Full brief: `docs/06_capstone.md`.",
       "",
       "This notebook is a **scaffold**: the code cells are safe placeholders — you fill",
       "in the `TODO(student)` parts for your chosen track.",
       "",
       "**繁體中文.** 12 週主軸已完成 —— 你已能組 PEPS wrapper、執行協定、量化、碰 HIP。",
       "專題(W13–W14)把這些變成一個小型研究貢獻。**本週(W13):**四選一,重跑要比較的",
       "*基線*並鎖定成功指標。**下週(W14):**跑出延伸並產出完整可驗證 artifacts。",
       "完整說明見 `docs/06_capstone.md`。此為**骨架**",
       "notebook,程式格為安全佔位,請在 `TODO(student)` 處填入你選定軌道的內容。"),
    BOOT,
    md("## 1. Pick your track / 選擇軌道",
       "Set `TRACK` below. Each track lists its starting files and the deliverable.",
       "設定 `TRACK`;每條軌道列出起始檔案與交付物。"),
    TRACKS,
    md("## 2. Rerun the matched baseline / 重跑公平基線",
       "Tracked result CSVs are legacy-unverified and may be used only for orientation.",
       "Rerun the baseline under your selected profile; copied numbers do not pass.",
       "Fill the branch for your track; the others are guidance.",
       "",
       "既有結果 CSV 皆為 legacy-unverified,只能作為方向。請在所選 profile 下親自重跑",
       "基線;抄數字不能通過。"),
    code("# TODO(student): rerun ONE baseline for your track. Sketches below.",
         "if TRACK == 'a':",
         "    # Use a self-created/CC0 clip, e.g. 8-16 frames at <=64x64.",
         "    # GridEncoder already supports dim=3; coordinates are (x, y, t).",
         "    print('a) baseline = plain 3D grid on the exact clip used by the extension')",
         "elif TRACK == 'b':",
         "    # int4/int8, mixed precision, per-channel PTQ, and metadata accounting exist.",
         "    # Orientation only: tracked per-channel means are 38.702/41.623 dB at",
         "    # 11.556/11.564 bpp; rerun them, do not copy these legacy-unverified rows.",
         "    print('b) baseline = rerun per-channel PTQ at the chosen total encoded bits')",
         "elif TRACK == 'c':",
         "    # Rerun corrected concat/pink under one matched image profile.",
         "    print('c) baseline = rerun concat/pink under the same data, seeds, and budget')",
         "elif TRACK == 'd':",
         "    # Hardware-gated: rerun the complete pipeline, not a standalone GEMM CSV row.",
         "    print('d) baseline = rerun full-pipeline parity + latency with a hardware receipt')",
         "else:",
         "    raise ValueError(f'unknown TRACK {TRACK!r} — pick a/b/c/d')"),
    md("## 3. Lock a success metric + record the baseline / 鎖定指標並記錄基線",
       "State the single number that defines success and save the verified rerun.",
       "The helper refuses blank, non-finite, unknown-profile, or legacy values.",
       "",
       "寫下定義成功的單一數字並儲存已驗證的重跑。helper 會拒絕空白、非有限值、",
       "未知 profile 或 legacy 狀態。"),
    code("import csv",
         "os.makedirs('../results', exist_ok=True)",
         "PROFILE = 'course_fast'  # TODO(student): course_fast or paper_exact",
         "SEED = 0",
         "def record_baseline(metric_name, value, units, path=None):",
         "    metric_name = _required_text(metric_name, 'metric_name')",
         "    units = _required_text(units, 'units')",
         "    if PROFILE not in {'course_fast', 'paper_exact'}:",
         "        raise ValueError('PROFILE must be course_fast or paper_exact')",
         "    value = float(value)",
         "    if not math.isfinite(value): raise ValueError('baseline value must be finite')",
         "    if isinstance(SEED, bool) or not isinstance(SEED, int): raise ValueError('SEED must be int')",
         "    path = path or f'../results/capstone_{TRACK}_baseline.csv'",
         "    with open(path, 'w', newline='') as f:",
         "        w = csv.writer(f, lineterminator='\\n')",
         "        w.writerow(['stage', 'metric', 'value', 'units', 'profile', 'seed', 'status'])",
         "        w.writerow(['baseline', metric_name, value, units, PROFILE, SEED, 'verified'])",
         "    print('wrote', path)",
         "    return path",
         "",
         "# No example result is supplied: copying a legacy number is not a rerun.",
         "# record_baseline('YOUR_METRIC', YOUR_FINITE_VALUE, 'YOUR_UNITS')",
         "print('After your rerun, call record_baseline(...) to save verified evidence.')"),
    md("## 4. Deliverables & timeline / 交付物與時程",
       "Every track ships the same evidence bundle:",
       "",
       "- an executed **notebook** and nonblank numeric **results CSV**,",
       "- a **run manifest** and completed **capstone submission JSON**,",
       "- one bilingual **Marp slide** that passes the course build.",
       "",
       "**W13 exit check:** track chosen, baseline rerun, metric + verified CSV in place.",
       "",
       "每條軌道都交付 notebook、非空數值 CSV、run/submission manifest 與一張可建置的",
       "雙語 Marp 投影片。**W13 出關檢查:**已選題、已重跑基線、指標與 verified CSV 就緒。"),
    md("## 5. Kickoff takeaway / 起跑小結",
       "A good capstone is **narrow and honest**: one baseline, one change, one number,",
       "reported truthfully — win or lose. Next week you execute and present.",
       "",
       "好的專題**窄而誠實**:一個基線、一個改動、一個數字,如實回報 —— 無論勝負。",
       "下週執行並發表。"),
])


# ----------------------------------------------------------------- W14
W14 = nb([
    md("# W14 · Project showcase / 專題成果展",
       "",
       "**English.** Execute the extension you scoped in W13, produce the artifacts, and",
       "self-assess against the rubric. Keep it **honest** — a negative result, reported",
       "cleanly with a matched baseline, scores well. Historical course CSVs remain",
       "legacy-unverified and are not evidence for your submission; Table 3's tracked",
       "0.269–0.345 SDF IoUs are an honesty example, not a baseline to copy. Full rubric:",
       "`docs/06_capstone.md`.",
       "",
       "**繁體中文.** 執行 W13 界定的延伸,產出交付物,並對照評分表自評。保持**誠實**——",
       "負結果只要有相同條件的基線且乾淨呈現,一樣拿分。既有課程 CSV 仍是",
       "legacy-unverified,不能當成本次提交證據;Table 3 的 0.269–0.345 SDF IoU",
       "只是誠實呈現範例,不能直接抄成基線。",
       "完整評分表見 `docs/06_capstone.md`。"),
    BOOT,
    TRACKS,
    md("## 1. Load your W13 baseline / 載入 W13 基線",
       "Pull back the baseline number you recorded last week so the comparison is",
       "matched. 讀回上週記錄的基線,確保對照條件一致。"),
    code("import csv",
         "base_path = f'../results/capstone_{TRACK}_baseline.csv'",
         "if not os.path.exists(base_path):",
         "    raise FileNotFoundError('Run W13 and record a verified baseline first')",
         "with open(base_path) as f: baseline_rows = list(csv.DictReader(f))",
         "if len(baseline_rows) != 1 or baseline_rows[0].get('status') != 'verified':",
         "    raise ValueError('W13 baseline must contain exactly one verified row')",
         "baseline_row = baseline_rows[0]",
         "baseline_value = float(baseline_row['value'])",
         "if not math.isfinite(baseline_value): raise ValueError('baseline must be finite')",
         "print('baseline:', baseline_row)"),
    md("## 2. Run your extension / 跑出延伸",
       "The single change your project is about. Sketches per track below — fill the one",
       "for your `TRACK`.",
       "",
       "你的專題所在的那**一個**改動。以下為各軌道草圖,填入你 `TRACK` 對應者。"),
    code("# TODO(student): implement your extension for the chosen track.",
         "if TRACK == 'a':",
         "    # Supported scope: a short (x,y,t) volume through GridEncoder(dim=3).",
         "    print('a) fit 3D Grid-PEPS on the same small clip and matched budget')",
         "elif TRACK == 'b':",
         "    # int4/int8 and mixed/per-channel PTQ already exist; extend calibration or QAT.",
         "    print('b) test clipping/calibration or short QAT at matched total encoded bits')",
         "elif TRACK == 'c':",
         "    # Add your aggregator to peps/aggregate.py + make_aggregator, then compare.",
         "    print('c) compare your aggregator vs concat/pink/brownian at matched budget')",
         "elif TRACK == 'd':",
         "    # Optimize one full-pipeline component and preserve parity.",
         "    print('d) compare one optimization against a rerun full-pipeline baseline')",
         "else:",
         "    raise ValueError(f'unknown TRACK {TRACK!r}')"),
    md("## 3. Results table + figure -> CSV / 結果表 + 圖 -> CSV",
       "Put baseline and your result side by side, write the CSV, and draw one figure.",
       "把基線與你的結果並排,寫出 CSV,畫一張圖。"),
    code("import csv",
         "os.makedirs('../results', exist_ok=True)",
         "# TODO(student): set only after the extension run finishes.",
         "EXTENSION_VALUE = None",
         "EXTENSION_LABEL = None",
         "if EXTENSION_VALUE is None or EXTENSION_LABEL is None:",
         "    raise ValueError('Refusing to write a blank submission: fill extension value/label')",
         "extension_value = float(EXTENSION_VALUE)",
         "if not math.isfinite(extension_value): raise ValueError('extension value must be finite')",
         "extension_label = _required_text(EXTENSION_LABEL, 'EXTENSION_LABEL')",
         "metric = _required_text(baseline_row['metric'], 'baseline metric')",
         "units = _required_text(baseline_row['units'], 'baseline units')",
         "profile = baseline_row['profile']",
         "seed = int(baseline_row['seed'])",
         "rows = [",
         "    ('baseline', metric, baseline_value, units, profile, seed, 'verified'),",
         "    ('extension', metric, extension_value, units, profile, seed, 'verified'),",
         "]",
         "out = f'../results/capstone_{TRACK}_result.csv'",
         "with open(out, 'w', newline='') as f:",
         "    w = csv.writer(f, lineterminator='\\n')",
         "    w.writerow(['stage', 'metric', 'value', 'units', 'profile', 'seed', 'status'])",
         "    w.writerows(rows)",
         "print('wrote', out)",
         "plt.bar([row[0] for row in rows], [row[2] for row in rows])",
         "plt.ylabel(f'{metric} ({units})')",
         "plt.title(f'Capstone {TRACK}: matched baseline vs extension'); plt.show()"),
    md("## 4. Make your slide / 做你的投影片",
       "Drop a deck into `slides/` in the course's bilingual style (English title + 繁中",
       "要點, `---` separators) and build it. Skeleton to copy:",
       "",
       "以課程雙語風格(英文標題 + 繁中要點、`---` 分隔)在 `slides/` 放一份投影片並建置。",
       "可複製的骨架:"),
    code("slide = '''---",
         "marp: true",
         "theme: default",
         "paginate: true",
         "title: Capstone — <your title>",
         "---",
         "",
         "# <Your project> / <你的專題>",
         "",
         "- Baseline / 基線: <number>",
         "- Change / 改動: <one sentence>",
         "- Result / 結果: <number> (win or honest loss)",
         "'''",
         "# TODO(student): save as slides/06_capstone_<name>.md then build:",
         "#   cd slides && make 06_capstone_<name>.pdf",
         "print(slide)"),
    md("## 5. Self-assessment vs rubric / 對照評分表自評",
       "Score yourself against `docs/06_capstone.md`. Be honest — the rubric rewards a",
       "matched baseline and clean reporting over a big-but-unfair number.",
       "",
       "對照 `docs/06_capstone.md` 自評。誠實為上 —— 評分表看重相同條件的基線與乾淨呈現,",
       "勝過又大又不公平的數字。"),
    code("rubric = {",
         "  'baseline_correctness (30%)':     'baseline rerun; extension is correct',",
         "  'extension_depth (25%)':          'the change is non-trivial and well-motivated',",
         "  'empirical_rigor (25%)':          'matched params/bitrate/tolerance; fair compare',",
         "  'honest_reporting (10%)':         'limitations + negative results stated plainly',",
         "  'communication (10%)':            'notebook + CSV + manifests + bilingual slide',",
         "}",
         "# TODO(student): rate each 0-1 and justify in one line.",
         "for k, v in rubric.items():",
         "    print(f'[ ] {k}: {v}')"),
    md("## 6. Reflection / honest limitations / 反思與誠實限制",
       "Close like the course does: what worked, what didn't, and the one experiment you'd",
       "run next. A clean negative result is a real contribution.",
       "",
       "如課程般收尾:什麼有效、什麼無效、以及下一個你會做的實驗。一個乾淨的負結果也是",
       "真正的貢獻。"),
    md("## 7. Submission gate / 提交門檻",
       "Copy `course/templates/capstone_submission.json`, fill every field, then run:",
       "",
       "`python3 scripts/validate_submission.py <submission.json> --kind capstone`",
       "",
       "The template intentionally fails while any placeholder, blank value, legacy",
       "baseline, missing artifact, or non-finite CSV value remains.",
       "",
       "複製 capstone submission 模板、填完全部欄位後執行 validator。任何 placeholder、",
       "空值、legacy 基線、缺失 artifact 或非有限 CSV 值都會被拒絕。"),
])


def write(name, notebook):
    path = os.path.join(HERE, name)
    document = nbformat.from_dict(notebook)
    nbformat.validate(document)
    nbformat.write(document, path)
    print("wrote", name)


if __name__ == "__main__":
    write("W13_project_kickoff.ipynb", W13)
    write("W14_project_showcase.ipynb", W14)
