# PEPS on AMD — Course & Educational Reimplementation
# AMD 上的 PEPS — 課程與教學型重實作

A semester course and independent implementation of **PEPS** (Positional
Encoding Projected Sampling). It separates a reduced `course_fast` path from a
declared `paper_exact` protocol, studies quantization, and includes optional
HIP/WMMA exercises. Local texture baselines are proxies; they are not official
RTXNTC parity implementations.

> **Current evidence status:** every tracked `results/*.csv` is
> `legacy-unverified` in `results/manifest.json`. The course release separately
> indexes three manifest-backed synthetic smokes, two complete but inconclusive
> pilots, and three public 512³ SDF provenance receipts. It contains zero
> paper-comparable results; see `results/course_release/receipt.json`.

本專案是一學期的 PEPS 課程與獨立實作,明確區分縮小的 `course_fast` 與論文協定
`paper_exact`,並包含量化研究與選配 HIP/WMMA 練習。現有 CSV 皆為
`legacy-unverified`;course release 只發布 synthetic smoke、無結論 pilot 與三個
公開 512³ SDF provenance,不包含可與論文比較的數值。

---

## What the reproduction found / 重現結果

Table 2 is reproduced in full: 594 of 594 jobs across eleven methods, three
seeds and eighteen materials, with zero errors. Every method lands below its
published value, by a mean of **1.154 dB**, and the reproduced ordering puts the
Grid family above the NTC family where the paper does the reverse.

Both symptoms trace to the same unpublished choice, the map-file selection. The
reported score is `map_weighted`, an average over individual maps, the eight map
categories span **19.4 dB**, and this selection puts 47% of its maps in the two
lowest-scoring categories. Reweighting the same measured jobs to equal
categories cuts the out-of-sample error against the published values by
**3.1x**, moves `NTC_PEPS` from third place to first, and swaps six method
pairs.

Two candidate explanations are ruled out rather than left hanging. Training
longer makes the PEPS advantage *shrink*, so compute is not the cause. The paper
does report its L1 loss and this reproduction uses it, so the loss family is not
the cause either. What the paper leaves open one level below that, the reduction
applied to the per-map L1 terms, does move a result several fold, but its effect
reverses sign between materials and so cannot repair the table on its own.

Everything stops at sufficiency. Showing that a choice *can* produce an effect
is not showing the authors made it, and the paper's file list is unpublished, so
the difference is bounded here rather than measured. Evidence, with its own
limitations, lives in `results/texture_repro/shortfall_analysis/`,
`ordering_probe/` and `budget_probe/`; the methodology is in
`docs/reproducibility.md`.

Table 2 已完整重現(594/594,十一個方法、三顆種子、十八個材質、零錯誤),但所有方法
都低於論文值,平均 **1.154 dB**,且方法排序與論文相反。兩個症狀都指向同一個未公布的
選擇:map 檔案選集。表頭數字是對個別 map 取平均,而八個 map 類別相差 **19.4 dB**,
本選集有 47% 的 map 落在分數最低的兩類。把同一批測量重新加權成類別均衡後,對論文值的
樣本外誤差降低 **3.1 倍**,`NTC_PEPS` 由第三升至第一,六組方法對互換位置。

算力與 loss family 已被排除:延長訓練反而讓 PEPS 優勢縮小,而論文確實載明 L1、本重現
也使用 L1。論文未指定的下一層細節(L1 如何在一組材質的各 map 之間 reduce)影響可達數
倍,但其效果在不同材質間會換符號,無法單獨修正整張表。以上全部止於充分性,不宣稱必然
性;論文未公布檔案清單,因此此處只能界定範圍而非量測差異。

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

The release checklist is `course/RELEASE_CHECKLIST.md`. It forbids paper/full
training during release validation and requires the machine-readable receipt to
retain all protocol and authorization blockers.

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

## Multi-GPU modes / 多 GPU 模式

The repository has two deliberately separate modes:

- `experiments.run` assigns **different jobs** (instance/method/seed tuples) to
  ranks. Its `world_size` is job-shard metadata, not DDP.
- `experiments.ddp` trains **one selected job** with PyTorch DDP. On AMD wheels
  devices are still named `cuda:N`; PyTorch backend `nccl` dispatches to RCCL.
  `training.batch_size` remains global, so the paper value 60,000 becomes
  15,000 samples per rank with four GPUs.

```bash
# Existing independent-job sharding (aggregate throughput, not one-job speedup)
.venv/bin/torchrun --standalone --nproc-per-node=4 -m experiments.run \
  --config configs/paper/image_full.toml --input /path/to/instances.pt \
  --output results/paper/image-job-shards

# New: one G-PEPS training job across all four GPUs
.venv/bin/torchrun --standalone --nproc-per-node=4 -m experiments.ddp \
  --config configs/paper/image_full.toml --input /path/to/instances.pt \
  --output results/paper/image-g-peps-ddp \
  --instance kodim01 --method G-PEPS --seed 0

# Topology, directed P2P copies, RCCL all-reduce, and fixed-global-batch
# 1-GPU versus 4-GPU training throughput
.venv/bin/python -m experiments.multigpu suite \
  --output results/multigpu/benchmark.json
```

The DDP output/checkpoint is rank-0-only and resumes automatically. Checkpoints
store the underlying model state without a `module.` prefix, so they can resume
on one or multiple GPUs. If RCCL peer IPC fails on a PCIe host booted without
IOMMU passthrough, the benchmark's default `--rccl-p2p auto` mode records the
failure and retries with host transport. For a training run, the equivalent
explicit diagnostic fallback is `--disable-rccl-p2p`; it is slower and must not
be reported as direct P2P collective performance. See
`docs/reproducibility.md` for the tensor input schema and timing methodology.

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
