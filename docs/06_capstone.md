# Part VI — Capstone project (W13–W14) / 專題

**English.** The 12 teaching weeks give you a working PEPS wrapper, application
protocols, quantization tools, and HIP/WMMA exercises. The capstone turns that
into a small **research contribution of your own**: pick one of four tracks,
rerun a matched baseline, make **one** well-motivated change, and report the
result honestly — win or lose. Two labs scaffold the work:
`W13_project_kickoff.ipynb` and `W14_project_showcase.ipynb`.

**繁體中文.** 12 週教學讓你擁有可運作的 PEPS wrapper、應用協定、量化工具與
HIP/WMMA 練習。專題把這些變成你**自己的小型研究貢獻**:四選一,重跑公平基線,
做**一個**有充分動機的改動,並誠實回報結果 —— 無論勝負。兩份 scaffold notebook
帶你完成:`W13_project_kickoff.ipynb` 與 `W14_project_showcase.ipynb`。

> **Evidence status.** Every currently tracked result CSV is
> `legacy-unverified` in `results/manifest.json`. It may orient a proposal but
> cannot serve as the capstone baseline. The baseline must be rerun.

---

## How the two weeks work / 兩週怎麼運作

**English.**
- **W13 · kickoff.** Choose a track, declare `course_fast` or `paper_exact`,
  rerun the matched baseline, lock one primary metric, and save a verified row
  to `results/capstone_<track>_baseline.csv`.
- **W14 · showcase.** Execute one change, produce the complete evidence bundle,
  validate it, and self-assess against the rubric.

**繁體中文.**
- **W13 · 起跑.** 選定軌道與 profile,親自重跑公平基線,鎖定單一主要指標,再寫入
  `results/capstone_<track>_baseline.csv`。
- **W14 · 成果展.** 執行一個改動,產出完整證據包,通過 validator 並依評分表自評。

---

## Deliverables — one validated evidence bundle / 交付物

Each project ships:

1. an executed **notebook**;
2. a nonblank numeric **results CSV** with baseline and extension rows;
3. a **run manifest** with profile, config, seeds, Git revision, data hashes,
   software/hardware, and runtime;
4. a completed **capstone submission JSON**, copied from
   `course/templates/capstone_submission.json`;
5. one bilingual **Marp slide** that passes `make -C slides validate`.

提交前執行:

```bash
mkdir -p course/submissions
cp course/templates/capstone_submission.json \
  course/submissions/capstone_<student>.json
# Fill every field, then:
python3 scripts/validate_submission.py <submission.json> --kind capstone
```

任何 placeholder、空值、非有限數值、legacy 基線、遺失 artifact 或 profile
不一致都會被拒絕。
---

## The four tracks / 四條軌道

### Track A — Short 3D video volume / 3D 短片

**English.**
- **Goal.** Fit one self-created or CC0 short clip as a 3D signal
  `f(x,y,t) -> RGB` and compare a dense 3D grid with 3D Grid-PEPS.
- **Bounded scope.** Use 8–16 frames at no more than 64×64 for `course_fast`.
  `GridEncoder(dim=3)` already supports this coordinate space. Audio (1D) and
  light fields (4D) are outside this track because the checked-in grid encoder
  supports only 2D/3D.
- **Starting files.** `peps/wrapper.py`, `peps/projector.py`,
  `peps/encoders/grid.py`, `peps/train.py`, and `apps/image/` as a template.
- **Deliverable.** Matched 3D-grid vs 3D-Grid-PEPS rows in
  `results/capstone_video3d.csv`, plus a reconstruction slice/video and data
  license/hash receipt.
- **What "good" looks like.** Finite reconstruction metrics, a declared matching
  tolerance, and a result reported honestly regardless of winner.

**繁體中文.** 以 `(x,y,t)->RGB` 表示 8–16 張、最高 64×64 的自製或 CC0 短片,
使用既有 `GridEncoder(dim=3)` 比較 3D grid 與 3D Grid-PEPS。現有 encoder 僅支援
2D/3D,故本軌不收 1D 音訊或 4D 光場。交付 `capstone_video3d.csv`、可視化與
資料授權/hash receipt;不要求 PEPS 勝出。

### Track B — Calibration or QAT recovery / 校準或 QAT 回復

**English.**
- **Already completed.** The library already supports int4/int8 and mixed
  component widths; W10 covers per-channel quantization, a mixed 6/8-bit plan,
  and metadata-aware encoded bits.
  Merely adding a bit width or mixed configuration is not a capstone.
- **Goal.** Add one recovery method: calibrated percentile/MSE clipping,
  representative-set calibration, or a short quantization-aware fine-tune.
  Compare it to a freshly rerun per-channel PTQ baseline at matched total
  encoded bits and at least three seeds.
- **Starting files.** `peps/quant/ptq.py`, `tests/test_quantization.py`, and
  `notebooks/W10_quantization.ipynb`.
- **Deliverable.** `results/capstone_quant_calibration.csv` with raw seed rows,
  total encoded bits, calibration method/config, and quality metric.
- **What "good" looks like.** Storage accounting still passes; calibration data
  is separated from evaluation data; any recovery is stated as an observation,
  not a causal explanation.

For orientation only, the current tracked W10 per-channel int8 means are
grid/PEPS **38.702/41.623 dB** at **11.556/11.564 bpp**. Because the artifact is
still `legacy-unverified` under the repository policy, the project must rerun
this baseline rather than copy it.

**繁體中文.** 函式庫已支援 int4/int8 與混合位寬;W10 已涵蓋 per-channel、mixed
6/8-bit plan 與 metadata-aware bits。只新增位寬不算專題。請新增 percentile/MSE
clipping、代表集校準或短 QAT 其中一項,在相同總 encoded bits、至少三個 seed 下
對照親自重跑的 per-channel PTQ。

### Track C — Design a new aggregator / 設計新聚合器

**English.**
- **Goal.** Add a new aggregator to `peps/aggregate.py` (beyond `concat` / `pink` /
  `brownian`) — e.g. a learned/gated allocation, attention over the `2L+1` points, or
  a different `1/f^alpha` schedule — and evaluate params-vs-PSNR against the existing
  three at a matched budget.
- **Starting files.** `peps/aggregate.py` (`_FrequencyAllocAggregator`,
  `make_aggregator`), `apps/image/build.py` (`build_grid_peps`),
  `notebooks/W06_pink_peps.ipynb`.
- **Deliverable.** A new aggregator class wired into `make_aggregator`, a notebook
  comparing it to concat/pink/brownian, `results/capstone_aggregator.csv`, a slide.
- **What "good" looks like.** The aggregator is well-motivated and differentiable,
  gradients still reach the whole `k`-dim grid feature, and the comparison is fair
  (matched params); an honest ablation even if it doesn't win.

**繁體中文.** **目標:**在 `peps/aggregate.py` 新增一個聚合器(超越 `concat`/`pink`/
`brownian`)—— 如可學習/門控分配、對 `2L+1` 點做 attention、或不同的 `1/f^alpha`
排程 —— 並在相同預算下與現有三者比較 params-vs-PSNR。**起始檔案:**`peps/aggregate.py`
(`_FrequencyAllocAggregator`、`make_aggregator`)、`apps/image/build.py`
(`build_grid_peps`)、`notebooks/W06_pink_peps.ipynb`。**交付物:**接入
`make_aggregator` 的新聚合器類別、與三者比較的 notebook、`results/capstone_aggregator.csv`、
一張投影片。**「好」的樣子:**聚合器有動機且可微,梯度仍能到達整個 `k` 維 grid 特徵,
對照公平(相同參數);即使沒贏也有誠實的 ablation。

### Track D — End-to-end PEPS runtime optimization / 端到端效能優化

**English.**
- **Already completed.** W11/W12 provide a first-layer fused teaching kernel and
  standalone fp16/int8 WMMA GEMMs with parity checks.
- **Goal.** Optimize one still-unfused full-pipeline component—for example,
  multi-layer decoder fusion, projection/sample launch reduction, or end-to-end
  quantized activation handling—while preserving numerical parity.
- **Starting files.** `hip/wmma_mlp.hip`, `hip/fused_peps_kernel.hip`,
  `hip/bench_latency.sh`, `hip/README.md`, `tests/test_hip_parity.py`,
  `peps/quant/ptq.py`, `notebooks/W12_hip_wmma.ipynb`.
- **Deliverable.** A rerun full-pipeline baseline and one optimized path,
  documented tolerance, raw latency rows in `results/capstone_runtime.csv`, and
  a hardware/software receipt.
- **What "good" looks like.** End-to-end parity, genuinely measured latency on
  ≥1 box, and an honest comparison that separates launch-bound from
  compute-bound effects.

**繁體中文.** W11/W12 已完成 first-layer fused 教學 kernel 與獨立 fp16/int8 WMMA。
本軌請優化一個尚未融合的端到端元件,例如多層 decoder fusion、減少投影/取樣 launch,
或完整量化 activation handling,並維持數值 parity。
**起始檔案:**`hip/wmma_mlp.hip`、
`hip/fused_peps_kernel.hip`、`hip/bench_latency.sh`、`hip/README.md`、
`tests/test_hip_parity.py`、`peps/quant/ptq.py`、`notebooks/W12_hip_wmma.ipynb`。
**交付物:**親自重跑的 full-pipeline baseline、一個優化路徑、數值容忍、
`results/capstone_runtime.csv` 原始延遲列與硬軟體 receipt。**「好」的樣子:**
端到端 parity、至少一台實測,並誠實區分 launch-bound 與 compute-bound 效果。

---

## Grading rubric / 評分表

**English.** All tracks share one rubric (100 points). The largest weights reward a
**matched baseline** and **honest reporting** over a big-but-unfair number.

| Criterion | Weight | Full marks means |
|---|---|---|
| Baseline & correctness | 30 | baseline rerun in-repo; the extension is correct |
| Extension depth | 25 | the one change is non-trivial and well-motivated |
| Empirical rigor | 25 | matched params/bitrate/tolerance; fair, checkable comparison |
| Honest reporting | 10 | limitations + any negative result stated plainly |
| Communication | 10 | notebook, CSV, manifests, and bilingual slide pass validation |

**繁體中文.** 各軌道共用一份評分表(100 分)。最大權重獎勵**相同條件的基線**與**誠實
回報**,勝過又大又不公平的數字。

| 評分項 | 權重 | 滿分標準 |
|---|---|---|
| 基線與正確性 | 30 | repo 內親自重跑基線;延伸正確 |
| 延伸深度 | 25 | 那一個改動非瑣碎且有動機 |
| 實證嚴謹 | 25 | 相同參數/位元率/容差;公平且可查證的對照 |
| 誠實回報 | 10 | 明確陳述限制與任何負結果 |
| 溝通表達 | 10 | notebook、CSV、manifests 與雙語投影片通過驗證 |

---

## Honesty clause / 誠實條款

**English.** A clean **negative result** — "my aggregator does not beat pink at
matched params, and here is the ablation" — earns **full** empirical-rigor and
honest-reporting marks. Report what you measured, not what you hoped. Do not use
the historical course CSVs as evidence until they are rerun and verified. They
still model honest reporting: Table 3 retains modest full-volume SDF IoUs
(**0.269–0.345**) even though PEPS improves its matched grid and multires bases.

**繁體中文.** 一個乾淨的**負結果** ——「我的聚合器在相同參數下沒贏 pink,這是 ablation」
—— 可拿**滿分**的實證嚴謹與誠實回報分。回報你量到的,而非你期望的;舊課程 CSV
在完成重跑與驗證前不能當證據。它們仍示範誠實呈現:Table 3 保留偏低的全體積
SDF IoU(**0.269–0.345**),即使 PEPS 提升同參數 grid 與 multires 基線。
