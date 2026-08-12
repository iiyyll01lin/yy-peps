# Weekly readings and lab gates / 每週閱讀與實作門檻

`course/labs.json` is the machine-readable source of truth for this page. A lab
passes on reproducible evidence and protocol invariants; it never requires a
particular method to win. Run the source smoke for one week with:

```bash
python3 scripts/validate_course.py --notebook W04
```

`course/labs.json` 是本頁的機器可讀真相來源。實作是否通過取決於可重現證據與協定
不變量，不要求某方法一定勝出。既有 `results/*.csv` 仍是
`legacy-unverified`，不可拿來代替本人的重跑。

## Runtime classes / 執行時間分級

- **quick** — static checks or small synthetic work; CPU target under 10 minutes.
- **course-fast** — reduced data/steps for one class period; hardware-dependent,
  normally 10–90 minutes. It is not paper evidence.
- **paper-exact** — full paper data/protocol; expect hours to days and record the
  actual runtime. It is never required in ordinary CPU CI.
- **amd-optional** — requires an explicitly enabled, self-hosted ROCm runner.
- **project** — student-scoped W13–W14 work with its own declared budget.

The times are planning targets, not performance claims.

## Weekly map / 逐週地圖

| Week | Required reading | Minimum success evidence |
|---|---|---|
| W01 | [On the Spectral Bias of Neural Networks](https://proceedings.mlr.press/v97/rahaman19a.html) | One seeded run has finite loss and finishes below its initial loss; all code cells compile. |
| W02 | [Fourier Features](https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html) | Identity-PEPS/APE affine residual is finite and `<=1e-5`; the equation test passes. |
| W03 | [PEPS paper](https://doi.org/10.1145/3806062), image experiment | Output names `course_fast` or `paper_exact` and records resolved config, seed, data hash, and raw rows. |
| W04 | [PEPS paper](https://doi.org/10.1145/3806062), method | Layout, shared-gradient, delta, and affine-equivalence tests pass. |
| W05 | [PEPS paper](https://doi.org/10.1145/3806062), Kodak protocol | Grid and Grid-PEPS use matched data, seeds, optimizer/steps, metrics, and documented parameter tolerance; both results are finite. |
| W06 | [PEPS paper](https://doi.org/10.1145/3806062), Algorithm 1 | Worked allocation, circular wrap, and whole-grid gradient tests pass; empirical numbers remain unverified until rerun. |
| W07 | [FLIP](https://doi.org/10.1145/3406183) | `paper_exact` verifies all 24 Kodak checksums; subsets say `course_fast`; metric versions and per-image rows are retained. |
| W08 | [Random-Access Neural Compression of Material Textures](https://doi.org/10.1145/3592407) and [PEPS](https://doi.org/10.1145/3806062) | Every map has source, license, dimensions, color/normal convention, and checksum evidence; proxies are called proxies; Table 2 is recomputed under a different map-category mix before any comparison with the paper, and the method pairs that swap are reported. Compute and loss family are ruled out before any mechanism is proposed; an inferred composition is validated on a held-out method and the held-out error is reported against the starting error, not the fit; and the paragraph making the claim also says where the mechanism does not hold. |
| W09 | [Local Positional Encoding](https://doi.org/10.2312/pg.20231273) and [PEPS](https://doi.org/10.1145/3806062) | Lucy, Thai Statue, and Armadillo have validated 512³ receipts; Pitted Stonefish stays authorization-blocked with no substitute. Sign, axis order, loss, seeds, and any per-instance IoU are explicit. |
| W10 | [Integer-only quantization](https://doi.org/10.1109/CVPR.2018.00068) | Storage accounting tests include all parameters and metadata; multi-seed observations are separated from causal hypotheses. |
| W11 | [HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/) | CPU fixture parity passes; GPU evidence records ISA, ROCm, exact workload, repetitions, tolerance, or an explicit skip. Latency comes from a settled card measured in interleaved rounds with a rotating start, with the worst round spread reported and under `1.10x`; the toolchain the binary was actually built against is stated, not the one the machine advertises. |
| W12 | [RDNA 4 ISA guide](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf) | CPU fp16/int8 references pass; paper latency is compared only under a matched full workload. Occupancy is derived three ways (LDS, wave slots, registers) and the binding one is named; one measure-diagnose-change-remeasure cycle is closed with the code object footprint before and after, checksums shown unchanged, the predicted occupancy gain stated before measuring, and the generality the change cost written down. |
| W13 | [ACM Artifact Review and Badging](https://www.acm.org/publications/policies/artifact-review-and-badging-current) | Nonblank proposal, finite rerun baseline, profile, seeds, data hashes, primary metric, and evidence paths; variant runs are isolated in a clone and long runs declare a progress-based liveness check. |
| W14 | [Good Enough Practices in Scientific Computing](https://doi.org/10.1371/journal.pcbi.1005510) | Validator accepts the notebook, nonblank numeric CSV, run manifest, Marp slide, conclusion, and limitations; any claim overturned by later evidence is retracted in place rather than deleted. |

## Focused grading commands / 聚焦評分指令

```bash
# Paper equations and wrapper invariants
pytest -q tests/test_paper_equations.py

# Quantization accounting (not a full experiment rerun)
pytest -q tests/test_quantization.py

# CPU reference portions of HIP parity
pytest -q \
  tests/test_hip_parity.py::test_integrated_features_match_projector_grid_and_aggregator \
  tests/test_hip_parity.py::test_complete_three_hidden_layer_reference_matches_mlp_module \
  tests/test_hip_parity.py::test_wmma_fp16_ref_matches_torch_half_matmul \
  tests/test_hip_parity.py::test_wmma_int8_ref_is_exact_and_bounded

# A completed submission (templates intentionally fail until filled)
python3 scripts/validate_submission.py path/to/submission.json --kind capstone
```

The complete wording, optional readings, and evidence commands remain in
`course/labs.json`; update that file first when the curriculum changes.
