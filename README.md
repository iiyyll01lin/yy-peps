# PEPS on AMD — Course & Educational Reimplementation
# AMD 上的 PEPS — 課程與教學型重實作

A semester course and independent implementation of **PEPS** (Positional
Encoding Projected Sampling). It separates a reduced `course_fast` path from a
declared `paper_exact` protocol, studies quantization, and includes optional
HIP/WMMA exercises. Local texture baselines are proxies; they are not official
RTXNTC parity implementations.

> **Current evidence status:** every tracked `results/*.csv` is
> `legacy-unverified` in `results/manifest.json`. The repository is not yet a
> verified end-to-end reproduction of the paper. Numerical claims require a
> fresh run manifest and raw per-instance evidence.

本專案是一學期的 PEPS 課程與獨立實作,明確區分縮小的 `course_fast` 與論文協定
`paper_exact`,並包含量化研究與選配 HIP/WMMA 練習。現有 CSV 皆為
`legacy-unverified`;完成附 manifest 的重跑前,不得宣稱完整重現。

---

## Hardware targets / 硬體目標

The optional hardware material targets two self-hosted AMD systems. CPU CI does
not assert GPU availability; AMD jobs run only through the explicitly gated
self-hosted workflow.

| Box | GPU | ISA | Role / 角色 |
|---|---|---|---|
| **A** | Radeon 8060S (Strix Halo iGPU) | `gfx1151` / RDNA 3.5 | RDNA3.5 kernel testing |
| **B** | 4× Navi 48 (RX 9070-class) | `gfx1201` / RDNA 4 | Main dev; RDNA4 kernel testing |

Parts I–IV can run on CPU or PyTorch/ROCm. Part V has per-ISA variants and must
record the actual GPU/toolchain used.

---

## Setup / 安裝

Python 3.12 is the constrained reference environment. Select exactly one Torch
wheel family before installing the shared dependencies.

### CPU / CPU CI

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r env/torch-cpu.txt \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r env/requirements.txt -c env/constraints.txt
python -m pip install -e . --no-deps
```

### AMD ROCm 7.0 wheels

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r env/torch-rocm70.txt \
  --index-url https://download.pytorch.org/whl/rocm7.0
python -m pip install -r env/requirements.txt -c env/constraints.txt
python -m pip install -e . --no-deps
```

`env/constraints.txt` pins the direct cross-platform dependencies. It is a
constraints approach, not a universal transitive lock; every result-producing
run must also retain `python -m pip freeze --all`. See `env/rocm_setup.md`.

## Focused validation / 聚焦驗證

```bash
# Release metadata, weekly map, notebook parse/compile smoke, data/result
# manifests, and Marp source structure
python scripts/validate_course.py

# CPU algorithm/profile/accounting checks
pytest -q tests/test_paper_data.py tests/test_paper_equations.py tests/test_profiles.py \
  tests/test_quantization.py tests/test_report.py

# Requires Node.js >=18; pinned Marp 4.5.0, HTML build (no browser/PDF needed)
make -C slides validate
```

The notebook smoke parses and compiles every Python cell; it deliberately does
not execute full training notebooks in CPU CI.

---

## Repo layout / 專案結構

```
peps/         reusable library (encoders, projector, aggregators, wrapper, metrics, train)
apps/         image / texture / sdf applications
notebooks/    weekly teaching spine (W01..W14)
hip/          RDNA3.5 + RDNA4 HIP/WMMA kernels
docs/         bilingual textbook (markdown)
slides/       bilingual Marp decks
course/       machine-readable labs and submission templates
data/         licensed/provenance-aware manifests and downloader
results/      artifacts plus explicit verification status
scripts/      static course and submission validators
```

The validators reduce drift between notebooks, slides, docs, and artifacts; they
do not make experimental results verified by themselves.

---

## Reproduction profiles / 重現設定

`peps.profiles` separates immutable `paper_exact` declarations from
`course_fast` teaching workloads. A result is evidence only when its manifest
records the resolved config, seed, Git revision, data hashes, software/hardware,
runtime, and raw per-instance rows. See `results/manifest.json` and
`docs/07_readings_and_labs.md`.

The dedicated application runner keeps the two paths separate:

```bash
# Cheap real optimizations with manifests; not paper-comparable
python -m experiments.reproduce smoke --task all

# Machine-readable exact-data/compute blockers
python -m experiments.reproduce check --profile paper_exact \
  --output results/paper_exact_prerequisites.json

# Full application artifacts (only after the check passes)
python -m experiments.reproduce run --artifact texture-table2
python -m experiments.reproduce run --artifact sdf-table3-mape
```

Image Figure 5 and Table 1 remain marked with protocol assumptions because the
paper omits the Figure 5 image identities and image training-step count. See
`docs/03_applications.md` for the exact commands and Table 1 loss inconsistency.

## Data / 資料

```bash
python data/download.py list all
python data/download.py fetch kodak
python data/download.py verify all
```

The checked-in manifests pin available checksums and record provenance/license
constraints. Not all data is CC0: Kodak uses source-specific PhotoCD sampler
grant/credit terms, Stanford meshes are research-only with attribution
requirements, and Pitted Stonefish is account-gated/non-commercial. Raw data
stays git-ignored.

---

## Citation and license / 引用與授權

- Repository code and original course text: MIT, see `LICENSE`.
- Third-party datasets keep their own terms; see `data/manifests/` and downloader
  output. The MIT license does not relicense them.
- Cite the repository with `CITATION.cff` and the PEPS paper with
  `references.bib` (publisher DOI
  [`10.1145/3806062`](https://doi.org/10.1145/3806062), arXiv
  [`2604.24167`](https://arxiv.org/abs/2604.24167)).
