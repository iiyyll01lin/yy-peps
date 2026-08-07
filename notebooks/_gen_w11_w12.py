"""Generate W11/W12 integrated HIP workload and diagnostic notebooks.

Local measurements are written only after a successful build/run. Integrated
baseline/PEPS/Pink rows are separated from supplementary component
microbenchmarks and are explicitly not labeled paper reproductions.
"""

from __future__ import annotations

import os

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
_id = [0]


def _nid():
    _id[0] += 1
    return f"hipc{_id[0]:03d}"


def _s(src):
    # Each arg may itself contain newlines; join then re-split into nbformat lines.
    t = "\n".join(src).split("\n")
    return [p + "\n" for p in t[:-1]] + [t[-1]]


def md(*l):
    return {"cell_type": "markdown", "id": _nid(), "metadata": {}, "source": _s(l)}


def code(*l):
    return {"cell_type": "code", "id": _nid(), "metadata": {},
            "execution_count": None, "outputs": [], "source": _s(l)}


def nb(cells):
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"}},
            "nbformat": 4, "nbformat_minor": 5}


# ----------------------------------------------------------------- shared setup
# Robust environment detection: `offload-arch` queries the device directly and
# works even where `rocminfo` prints no gfx line (observed on the RDNA4 box).
SETUP = code(
    "import os, re, shutil, subprocess, sys",
    "if os.path.basename(os.getcwd()) == 'notebooks':",
    "    os.chdir('..')  # run from repo root",
    "",
    "def _hipcc():",
    "    candidates = ('/opt/rocm/bin/hipcc', '/opt/rocm/bin/amdclang++',",
    "                  shutil.which('hipcc'))",
    "    return next((item for item in candidates if item and os.path.exists(item)), None)",
    "",
    "def compiler_command(src, out):",
    "    compiler = _hipcc()",
    "    command = [compiler]",
    "    if os.path.basename(compiler).startswith('amdclang'):",
    "        command.extend(['-x', 'hip'])",
    "    command.extend([f'--offload-arch={arch}', src])",
    "    if os.path.basename(compiler).startswith('amdclang'):",
    "        command.extend(['-L/opt/rocm/lib', '-lamdhip64'])",
    "    command.extend(['-o', out])",
    "    return command",
    "",
    "def detect_arch():",
    "    for tool in ('offload-arch', '/opt/rocm/bin/offload-arch'):",
    "        exe = tool if (os.path.isabs(tool) and os.path.exists(tool)) else shutil.which(tool)",
    "        if not exe:",
    "            continue",
    "        out = subprocess.run([exe], capture_output=True, text=True).stdout",
    "        toks = [t for t in out.split() if t.startswith('gfx')]",
    "        if toks:",
    "            return toks[0].strip()",
    "    if shutil.which('rocminfo'):",
    "        out = subprocess.run(['rocminfo'], capture_output=True, text=True).stdout",
    "        m = re.search(r'gfx[0-9a-f]+', out)",
    "        if m:",
    "            return m.group(0)",
    "    return 'unknown'",
    "",
    "def box_of(arch):",
    "    # Box B = RDNA4 (gfx1201); Box A = RDNA3.5 (gfx1151); else hostname.",
    "    return {'gfx1201': 'B', 'gfx1151': 'A'}.get(arch, os.uname().nodename)",
    "",
    "have_hipcc = _hipcc() is not None",
    "arch = detect_arch()",
    "box = box_of(arch)",
    "have_gpu = arch != 'unknown'",
    "rocm_version = 'unknown'",
    "rocm_version_file = '/opt/rocm/.info/version'",
    "if os.path.exists(rocm_version_file):",
    "    with open(rocm_version_file) as handle: rocm_version = handle.read().strip()",
    "else:",
    "    hipconfig = shutil.which('hipconfig') or '/opt/rocm/bin/hipconfig'",
    "    if os.path.exists(hipconfig):",
    "        rv = subprocess.run([hipconfig, '--version'], capture_output=True, text=True).stdout.strip()",
    "        if rv: rocm_version = rv.splitlines()[0]",
    "print('HIP compiler:', _hipcc(), '| gpu:', have_gpu, '| gfx arch:', arch,",
    "      '| box:', box, '| ROCm:', rocm_version)",
    "print('RDNA4' if arch == 'gfx1201' else 'RDNA3.5' if arch == 'gfx1151' else '(other)')",
)

# Build / run / parse / CSV-upsert helpers used by the benchmark cells below.
BENCH = code(
    "import csv",
    "",
    "CSV_PATH = 'results/hip_latency.csv'",
    "CSV_COLS = [",
    " 'schema_version','benchmark_kind','kernel','mode','implementation','isa','box',",
    " 'rocm_version','dtype','output_width','output_height','grid_width','grid_height',",
    " 'feature_dim','num_frequencies','hidden_dim','hidden_layers','out_dim','activation',",
    " 'workload','iters','ms_per_iter','parity_status','provenance','comparable_to_paper']",
    "",
    "def build_kernel(src, out):",
    "    env = dict(os.environ); env['PATH'] = '/opt/rocm/bin:' + env.get('PATH', '')",
    "    include_path = env.get('CPLUS_INCLUDE_PATH')",
    "    env['CPLUS_INCLUDE_PATH'] = ('/opt/rocm/include' if not include_path else",
    "                                 '/opt/rocm/include' + os.pathsep + include_path)",
    "    r = subprocess.run(compiler_command(src, out),",
    "                       capture_output=True, text=True, timeout=600, env=env)",
    "    return r",
    "",
    "def run_kernel(binary, args):",
    "    env = dict(os.environ); env.setdefault('HIP_VISIBLE_DEVICES', '0')",
    "    return subprocess.run([binary, *map(str, args)], capture_output=True,",
    "                          text=True, timeout=1800, env=env)",
    "",
    "def parse_ms(stdout):",
    "    m = re.search(r'([0-9.]+)\\s*ms/iter', stdout)",
    "    return float(m.group(1)) if m else None",
    "",
    "def result_row(**values):",
    "    row = {column: '' for column in CSV_COLS}",
    "    row.update(schema_version=2, isa=arch, box=box, rocm_version=rocm_version,",
    "               comparable_to_paper='false', **values)",
    "    return {column: str(row[column]) for column in CSV_COLS}",
    "",
    "def upsert_latency(rows):",
    "    \"\"\"Upsert local measurements without changing external paper values.\"\"\"",
    "    existing = []",
    "    if os.path.exists(CSV_PATH):",
    "        with open(CSV_PATH) as f:",
    "            existing = list(csv.DictReader(f))",
    "    if existing and set(existing[0]) != set(CSV_COLS):",
    "        legacy_cols = {'kernel','isa','box','dtype','workload','iters','ms_per_iter'}",
    "        if set(existing[0]) != legacy_cols:",
    "            raise ValueError('hip_latency.csv has an unknown incompatible schema')",
    "        migrated = []",
    "        for old in existing:",
    "            row = {column: '' for column in CSV_COLS}",
    "            is_wmma = old['kernel'] == 'wmma_mlp'",
    "            row.update(schema_version='1',",
    "                benchmark_kind='legacy_supplementary_microbenchmark',",
    "                kernel=old['kernel'], mode='layer_only' if is_wmma else 'first_layer',",
    "                implementation='legacy_rocwmma_tile' if is_wmma else 'legacy_scalar_fp32',",
    "                isa=old['isa'], box=old['box'], dtype=old['dtype'],",
    "                workload=old['workload'], iters=old['iters'],",
    "                ms_per_iter=old['ms_per_iter'], parity_status='legacy_recorded',",
    "                provenance='legacy_pre_schema2_csv', comparable_to_paper='false')",
    "            migrated.append(row)",
    "        existing = migrated",
    "    key = lambda r: (r['benchmark_kind'],r['kernel'],r['mode'],r['isa'],",
    "                     r['dtype'],r['workload'])",
    "    merged = {key(r): {k: r.get(k, '') for k in CSV_COLS} for r in existing}",
    "    for r in rows:",
    "        merged[key(r)] = {k: r.get(k, '') for k in CSV_COLS}",
    "    ordered = sorted(merged.values(),",
    "        key=lambda r: (r['benchmark_kind'],r['box'],r['kernel'],r['mode'],r['dtype'],r['workload']))",
    "    os.makedirs('results', exist_ok=True)",
    "    with open(CSV_PATH, 'w', newline='') as f:",
    "        w = csv.DictWriter(f, fieldnames=CSV_COLS, lineterminator='\\n'); w.writeheader()",
    "        w.writerows(ordered)",
    "    return ordered",
    "",
    "def show_latency():",
    "    if not os.path.exists(CSV_PATH):",
    "        print('(no results/hip_latency.csv yet)'); return",
    "    with open(CSV_PATH) as f:",
    "        print(f.read())",
)


# ----------------------------------------------------------------- W11
W11 = nb([
    md("# W11 · Integrated PEPS in HIP / HIP 整合 PEPS workload\n"
       "\n"
       "`hip/fused_peps_kernel.hip` now runs the complete path: projection, every\n"
       "shared-grid sample, baseline/concat/paper-Pink aggregation, and four Linear\n"
       "layers (three hidden + output). The default geometry is the paper runtime\n"
       "workload: 1024² RGB, a 1024² grid with C=16, L=3, and hidden width 64.\n"
       "\n"
       "整合 kernel 包含 projection、所有 shared-grid samples、baseline/concat/\n"
       "paper-exact Pink,以及完整三 hidden layer MLP。"),
    SETUP,
    BENCH,
    md("## 1. Build only for a detected local GPU / 僅為已偵測 GPU 編譯"),
    code("build_ok = False",
         "if have_hipcc and have_gpu:",
         "    build = build_kernel('hip/fused_peps_kernel.hip', 'hip/fused_peps')",
         "    build_ok = build.returncode == 0",
         "    print('build ok' if build_ok else build.stderr[-1200:])",
         "else:",
         "    print('skipped: hipcc and a real AMD architecture are both required')"),
    md("## 2. End-to-end parity fixtures / 端到端對拍\n"
       "The three fixture modes are compared with `Projector`, `GridEncoder`, the\n"
       "corresponding aggregator, and a GELU MLP with exactly three hidden layers.\n"
       "Fixtures cover both scalar fp32 and fused fp16/rocWMMA implementations."),
    code("parity = subprocess.run([sys.executable, '-m', 'pytest',",
         "    'tests/test_hip_parity.py', '-q', '-k', 'integrated'],",
         "    capture_output=True, text=True)",
         "parity_ok = parity.returncode == 0",
         "print((parity.stdout or parity.stderr)[-2000:])"),
    md("## 3. Measure the integrated geometry / 量測整合 workload\n"
       "Rows are written only after build, parity, execution, and output parsing all\n"
       "succeed. This scalar fp32 reference is **not** the paper's optimized WMMA kernel."),
    code("INTEGRATED_ITERS = int(os.getenv('PEPS_HIP_INTEGRATED_ITERS', '20'))",
         "rows = []",
         "if build_ok and parity_ok:",
         "    for mode in ('baseline', 'peps', 'pink'):",
         "        run = run_kernel('hip/fused_peps',",
         "                         ['workload', mode, 1024, INTEGRATED_ITERS])",
         "        print((run.stdout + run.stderr).strip())",
         "        ms = parse_ms(run.stdout) if run.returncode == 0 else None",
         "        if ms is not None:",
         "            rows.append(result_row(benchmark_kind='integrated_paper_workload',",
         "                kernel='fused_peps', mode=mode, implementation='scalar_fp32',",
         "                dtype='fp32', output_width=1024, output_height=1024,",
         "                grid_width=1024, grid_height=1024, feature_dim=16,",
         "                num_frequencies=0 if mode == 'baseline' else 3,",
         "                hidden_dim=64, hidden_layers=3, out_dim=3, activation='gelu',",
         "                workload='1024x1024_rgb_grid1024_c16_l3_h64x3',",
         "                iters=INTEGRATED_ITERS, ms_per_iter=f'{ms:.6f}',",
         "                parity_status='passed', provenance='W11_local_notebook'))",
         "if rows:",
         "    upsert_latency(rows); print('wrote', len(rows), 'integrated row(s)')",
         "else:",
         "    print('no integrated rows written')",
         "show_latency()"),
    md("## 4. Comparison boundary / 比較界線\n"
       "The paper reports 4.32 ms (BI-grid), 5.47 ms (Grid-PEPS), and 4.86 ms\n"
       "(Grid-PinkPEPS) on RX 9070 XT for this geometry. Those are external paper\n"
       "values, not rows generated here. A clean-build 30-warmup/100-iteration 1024²\n"
       "fused-fp16 receipt now passes all-mode parity and timing, but explicitly sets\n"
       "`directly_comparable=false`: the paper does not disclose matching precision,\n"
       "timing/synchronization boundaries, or kernel source."),
])


# ----------------------------------------------------------------- W12
W12 = nb([
    md("# W12 · WMMA diagnostics and occupancy tuning / WMMA 診斷與佔用率調校\n"
       "\n"
       "`hip/wmma_mlp.hip` verifies isolated fp16 and int8 rocWMMA GEMMs. They are useful\n"
       "component diagnostics, but they do not include projection, sampling, aggregation,\n"
       "biases/activations, or all decoder layers. W11 is the primary workload path.\n"
       "\n"
       "Section 3 then does the part that makes the diagnostics worth collecting: it\n"
       "derives which resource caps occupancy, changes that resource, and re-measures.\n"
       "A profile you do not act on is a screenshot.\n"
       "\n"
       "本章保留 fp16/int8 rocWMMA component diagnostics;它們不含完整 PEPS pipeline,\n"
       "因此不能當作 paper-workload latency。第 3 節則完成量測的目的:找出限制佔用率的\n"
       "資源、改動它、重新量測——沒有後續動作的 profile 只是一張截圖。"),
    SETUP,
    BENCH,
    md("## 1. Build and run fixture parity / 編譯並對拍"),
    code("build_ok = False",
         "if have_hipcc and have_gpu:",
         "    build = build_kernel('hip/wmma_mlp.hip', 'hip/wmma_mlp')",
         "    build_ok = build.returncode == 0",
         "    print('build ok' if build_ok else build.stderr[-1200:])",
         "else:",
         "    print('skipped: hipcc and a real AMD architecture are both required')",
         "parity = subprocess.run([sys.executable, '-m', 'pytest',",
         "    'tests/test_hip_parity.py', '-q', '-k', 'wmma'],",
         "    capture_output=True, text=True)",
         "parity_ok = parity.returncode == 0",
         "print((parity.stdout or parity.stderr)[-2000:])"),
    md("## 2. Supplementary layer microbenchmarks / 補充 layer microbenchmark\n"
       "The large GEMM is opt-in to avoid unsafe allocation/runtime on small machines."),
    code("SIZES = [('4096x64x64', 4096, 64, 64,",
         "          int(os.getenv('PEPS_HIP_MICRO_ITERS', '1000')))]",
         "if os.getenv('RUN_LARGE_WMMA', '0') == '1':",
         "    SIZES.append(('2048x2048x2048', 2048, 2048, 2048,",
         "                  int(os.getenv('PEPS_HIP_LARGE_ITERS', '200'))))",
         "rows = []",
         "if build_ok and parity_ok:",
         "    for label, M, K, N, it in SIZES:",
         "        for dt in ('fp16', 'int8'):",
         "            run = run_kernel('hip/wmma_mlp', ['bench', dt, M, K, N, it])",
         "            print((run.stdout + run.stderr).strip()); print()",
         "            ms = parse_ms(run.stdout) if run.returncode == 0 else None",
         "            if ms is not None:",
         "                rows.append(result_row(benchmark_kind='supplementary_microbenchmark',",
         "                    kernel='wmma_mlp', mode='layer_only', implementation='rocwmma_tile',",
         "                    dtype=dt, feature_dim=K, hidden_dim=N, workload=label,",
         "                    iters=it, ms_per_iter=f'{ms:.6f}', parity_status='passed',",
         "                    provenance='W12_local_notebook'))",
         "if rows:",
         "    upsert_latency(rows); print('wrote', len(rows), 'diagnostic row(s)')",
         "else:",
         "    print('no diagnostic rows written')",
         "show_latency()"),
    md("## 3. Find the limiter, then close the loop / 找出瓶頸,讓迴圈閉合\n"
       "A latency number on its own does not tell you what to change. Occupancy does,\n"
       "and for this kernel you can derive it on paper before touching the GPU: the\n"
       "four `__shared__` tiles in `integrated_peps_wmma` are sized from compile-time\n"
       "caps, so the launch reserves the same LDS whatever dimensions a run uses.\n"
       "Work out workgroups-per-CU three ways — LDS, wave slots, registers — and the\n"
       "smallest one is what you are actually fighting. 先在紙上算出瓶頸,再上機驗證。"),
    code("# Occupancy is arithmetic. This cell needs no GPU.",
         "WMMA_TILE, LDS_PER_CU, WAVES_PER_WG, MAX_WAVES = 16, 64 * 1024, 2, 32",
         "",
         "def footprint(input_cap, hidden_cap):",
         "    # feature_tile + hidden_a + hidden_b are fp16; accumulator is fp32.",
         "    return WMMA_TILE * (input_cap * 2 + hidden_cap * 2",
         "                        + hidden_cap * 2 + hidden_cap * 4)",
         "",
         "def occupancy(bytes_per_wg):",
         "    wgs = LDS_PER_CU // bytes_per_wg",
         "    return wgs, wgs * WAVES_PER_WG / MAX_WAVES",
         "",
         "STOCK, TUNED = footprint(512, 128), footprint(128, 64)",
         "for label, fp in (('stock  (512/128)', STOCK), ('narrowed (128/64)', TUNED)):",
         "    wgs, occ = occupancy(fp)",
         "    print(f'{label}: {fp:6d} B/workgroup -> {wgs} workgroups/CU"
         " -> {occ:6.2%} occupancy')",
         "print()",
         "print('predicted occupancy gain:', f'{occupancy(TUNED)[1] / occupancy(STOCK)[1]:.2f}x')",
         "print('but wave slots would allow 16 workgroups and registers 9,')",
         "print('so LDS is the binding constraint by roughly 4x.')"),
    md("The caps are 512 and 128. `aggregate_dim` needs **16 / 112 / 44 / 46** for the\n"
       "four methods and the hidden width is always 64, so most of that reservation is\n"
       "never touched — and LDS is reserved whether it is read or not. Narrowing the\n"
       "caps is therefore a change you can predict the effect of *before* measuring,\n"
       "which is what makes it an experiment rather than a guess.\n\n"
       "```bash\n"
       "bash hip/build_kernel.sh gfx1201                       # stock\n"
       "PEPS_EXTRA_FLAGS=\"-DPEPS_MAX_INPUT_DIM=128 -DPEPS_MAX_HIDDEN_DIM=64\" \\\n"
       "  bash hip/build_kernel.sh gfx1201 lds12k              # narrowed\n"
       "amdclang++ ... -S -o - | grep group_segment_fixed_size # confirm the footprint\n"
       "```\n\n"
       "Then re-measure **with the settled protocol**, not a back-to-back sweep:\n"
       "`python hip/stable_latency.py --binary <build> --out <receipt>`."),
    code("# What the intervention actually bought, from the committed receipt.",
         "import json, pathlib",
         "receipt = pathlib.Path('..') / 'results' / 'hip_lds_ab.json'",
         "if not receipt.exists():",
         "    receipt = pathlib.Path('results') / 'hip_lds_ab.json'",
         "if receipt.exists():",
         "    ab = json.loads(receipt.read_text())",
         "    print(f\"{'method':<20}{'stock':>8}{'tuned':>8}{'speedup':>9}{'paper':>7}\")",
         "    for part, block in ab['latency'].items():",
         "        print(f'-- {part} ({block[\"part\"]})')",
         "        for name, row in block['methods'].items():",
         "            print(f\"{name:<20}{row['stock']:>8.2f}{row['tuned']:>8.2f}\"",
         "                  f\"{row['speedup']:>8.2f}x{row['paper']:>7.2f}\")",
         "    print()",
         "    print('checksums identical:', ab['numerical_equivalence']['verdict'])",
         "    print('predicted occupancy gain:', ab['occupancy_change']['predicted_ratio'], 'x')",
         "    print()",
         "    print('READ THIS:', ab['analysis']['return_is_sublinear'])",
         "else:",
         "    print('receipt not found; run the A/B yourself and record one')"),
    md("Three things in that output are the actual lesson, and none of them is the\n"
       "speedup.\n\n"
       "1. **The checksums are identical.** A performance change that alters the output\n"
       "   is not a performance change. Report the checksum comparison or the number\n"
       "   means nothing. 沒有對過 checksum 的加速不算加速。\n"
       "2. **The return is sublinear.** Occupancy rose 2.5x; latency improved about 2x.\n"
       "   Quoting the 2.5x would be overstating the result. Occupancy was the binding\n"
       "   constraint, not the only one.\n"
       "3. **It cost generality.** The narrowed build refuses an aggregated input above\n"
       "   128 and fails closed through the existing `check_config` guard. This is a\n"
       "   specialisation to the deployed configuration, not a free win, and saying so\n"
       "   is part of reporting it honestly.\n\n"
       "There is also a warning here about your own earlier conclusions. The stock\n"
       "build made Grid-PinkPEPS look *slower* than Grid-PEPS, contradicting the paper.\n"
       "That was not a reproduction failure — both methods were forced to reserve the\n"
       "same worst-case LDS, so Pink's smaller 44-channel aggregate bought nothing. Fix\n"
       "the footprint and the paper's ordering comes back. **A measurement artefact had\n"
       "been sitting there looking like a disagreement with the paper.** 先懷疑量測,\n"
       "再懷疑論文。"),
    md("## 4. What remains before a paper comparison / 論文比較前仍缺什麼\n"
       "The W11 kernel now integrates rocWMMA across all four Linear layers and passes\n"
       "baseline/PEPS/Pink fp16 parity and repeated 1024² timing. The remaining gate is\n"
       "comparison fidelity: match disclosed precision/timing boundaries and target-GPU\n"
       "identity, retain compiler and warmup provenance, and never promote local timings\n"
       "to paper reproduction while the receipt says `directly_comparable=false`.\n\n"
       "That gate is unchanged by the optimisation above. Getting `bi-grid` under the\n"
       "paper's 4.32 ms is **not** a reproduction of the paper's number; it is a local\n"
       "measurement of a workload that has not been shown to match."),
])


def write(name, notebook):
    path = os.path.join(HERE, name)
    document = nbformat.from_dict(notebook)
    nbformat.validate(document)
    nbformat.write(document, path)
    print("wrote", name)


if __name__ == "__main__":
    write("W11_hip.ipynb", W11)
    write("W12_hip_wmma.ipynb", W12)
