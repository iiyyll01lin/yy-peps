# PEPS on AMD — Course & Educational Reimplementation
# AMD 上的 PEPS — 課程與教學型重實作

A semester course and independent implementation of **PEPS** (Positional
Encoding Projected Sampling). It separates a reduced `course_fast` path from a
declared `paper_exact` protocol, studies quantization, and includes optional
HIP/WMMA exercises. Local texture baselines are proxies; they are not official
RTXNTC parity implementations.

> **Current evidence status:** every tracked `results/*.csv` is
> `legacy-unverified` in `results/manifest.json`; the HIP receipts carry
> `measured-not-verifiable-by-this-policy`, which says what this policy can
> certify rather than how good the measurement is. The course release separately
> indexes the complete Table 2 run and its shortfall diagnosis, three
> manifest-backed synthetic smokes, two complete but inconclusive pilots, and
> three public 512³ SDF provenance receipts. It contains zero paper-comparable
> results; see `results/course_release/receipt.json`.

本專案是一學期的 PEPS 課程與獨立實作,明確區分縮小的 `course_fast` 與論文協定
`paper_exact`,並包含量化研究與選配 HIP/WMMA 練習。現有 CSV 皆為
`legacy-unverified`,HIP receipt 則為 `measured-not-verifiable-by-this-policy`;
course release 索引完整的 Table 2 執行與其缺口診斷、synthetic smoke、無結論 pilot
與三個公開 512³ SDF provenance,但不包含可與論文比較的數值。

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

## What the AMD track found / AMD 軌道的結果

Three findings, and the same shape each time: what looked like a result about
the paper or the hardware was a defect in this repository's own measurement or
model.

**The first latency table was an artefact.** It timed each method to completion
in turn from an idle card, so whichever went first absorbed the clock ramp and
was inflated **5.7x**, and the ordering it reported was the measurement order.
The overstatement decays monotonically down the table, 5.72x then 3.00x then
2.78x then 2.53x, which is the fingerprint rather than a coincidence.

**A compile-time cap cost about half the speed.** The kernel's `__shared__`
tiles are sized for a worst case none of the four methods reaches, so every
workgroup reserved 32 KB to use at most 12 KB, and LDS is reserved whether it is
read or not. Narrowing the caps roughly halves latency on both parts with
byte-identical output; per-method caps go further, and `bi-grid` runs at
**2.95 ms** against the paper's 4.32 ms reference. The Pink ordering that had
looked like a disagreement with the paper came back into line at the same time,
having been an artefact of the shared cap rather than a property of the method.

**The occupancy model was wrong twice**, caught both times by a hardware
counter. It survived because the first version matched five of seven measured
footprints and its replacement three of seven: *a model that is right most of
the time looks confirmed every time it is checked on an easy case.*

That optimisation is now finished rather than merely exhausted, and the
arithmetic says so. The three fixed tiles cost `16 x 64 x (2+2+4) = 8192` bytes,
while sixteen workgroups per WGP would need the entire allocation to fit inside
8192, so no cap can reach it. **Fourteen waves per compute unit, 43.75%, is the
ceiling, and `bi-grid` already measures 13.94.** Going further means shrinking
the hidden tiles, which means changing precision, which would cost the
byte-identical property that makes these comparisons worth making.

None of these numbers are comparable to the paper's: the workload, output size,
precision and timing boundary have not been shown to match. `docs/05_amd_hardware.md`
carries the sequence and `results/hip_*.json` the measurements with their limitations.

AMD 軌道有三個發現,而且形狀相同:**看起來像「關於論文」或「關於硬體」的結果,其實是
本 repo 自己量測或模型的缺陷。**

第一份延遲表是假象——它從閒置卡上逐方法連續量測,先跑的吸收了時脈爬升而被膨脹 **5.7
倍**,它報告的排序就是量測順序;高估倍率沿表單調遞減(5.72→3.00→2.78→2.53),這是指紋
而非巧合。第二,**一個編譯期上限值掉一半速度**:`__shared__` 分頁按四個方法都達不到的
最壞情況配置,每個 workgroup 保留 32 KB 卻最多用 12 KB,而 LDS 不論讀不讀都整份保留;
收窄後兩張卡的延遲各減半且輸出逐位元相同,每方法特化更進一步,`bi-grid` 達 **2.95 ms**
(論文參考值 4.32 ms),同時先前看似「與論文不符」的 Pink 排序也回到論文方向——它是共用
上限的假象。第三,**佔用率模型錯了兩次**,兩次都是硬體計數器抓到的;它們能存活是因為在
七個 footprint 中分別對了五個與三個:*一個大多數時候正確的模型,在每次用簡單案例檢查時
都像是被驗證了。*

這條最佳化現在是**可證明的終點**而非「大概沒了」:三個固定分頁佔 `16×64×(2+2+4) = 8192`
bytes,而每 WGP 十六個 workgroup 需要整份配置塞進 8192,任何上限都不可能達到。**每 CU
十四個 wave(43.75%)是天花板,而 `bi-grid` 已量到 13.94。** 再往下必須縮小 hidden 分頁,
即動精度,那會失去「逐位元相同」這個讓比較有意義的性質。以上數字皆不可與論文直接比較。

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
