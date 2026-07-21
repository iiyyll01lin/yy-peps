# results/ — generated tables & figures / 生成的表格與圖

> **Status:** every top-level legacy experiment CSV is `legacy-unverified`. The
> authoritative per-file status and verification requirements are in
> `manifest.json`. A notebook rerun or schema check alone does not make an
> artifact verified. Manifest-backed `runs/<run-id>/*.csv` are separate.

Quantitative claims require a named profile, resolved configuration, seeds, Git
revision, input checksums, software/hardware metadata, runtime, and raw
per-instance rows. Until those are attached, values may be shown only as
historical teaching output.

目前所有 CSV 皆為 `legacy-unverified`。只重跑 notebook 或通過 schema 不足以成為
已驗證證據;仍需 profile、完整 config/seed/git/data hash、軟硬體、時間與逐 instance 原始列。

- `*.csv` — tracked, diffable artifacts with notebook-level provenance.
- `*.png` — visualizations regenerated from the CSVs (git-ignored).

Written via `peps.report` (`write_table` / `plot_xy`). Regenerate everything by
re-running the weekly notebooks, or the per-app build scripts.

透過 `peps.report` 產生。重跑各週 notebook 或各 app 的 build 腳本即可重生。

## Manifest-backed paper artifacts

`python -m experiments.reproduce` is the only paper-artifact producer. It never
reads the legacy tables. A successful execution creates:

```text
results/runs/<run-id>/
  manifest.json       peps.run_manifest v1
  instances.csv       tidy raw per-instance/per-map measurements
  summary.csv         peps.paper_artifact_summary v1
  summary.json        the same aggregate in machine-readable JSON
```

`summary.csv` is derived only after validating the colocated manifest.
Texture summaries average individual RGB-map rows globally and by the eight
paper semantics. SDF summaries include streamed 512³ IoU observations.

Checked-in contracts:

- `schemas/paper_artifact_summary.schema.json`
- `schemas/reproduction_prerequisites.schema.json`
- `schemas/fig5_dataset.schema.json`

Use `python -m experiments.reproduce check --profile paper_exact` for
machine-readable data/hardware blockers. `course_fast` smoke runs use the same
manifest contract but are marked `course_fast_smoke_not_paper_comparable`.

| File | Produced by | Current status |
|---|---|---|
| `table1_image.csv` | W05 / W07 | legacy-unverified |
| `table2_texture.csv` | W08 | legacy-unverified |
| `table3_sdf.csv` | W09 | legacy-unverified |
| `w10_rate_distortion.csv` | W10 | current 3-seed matched-size rerun; unverified |
| `w10_rate_distortion.schema.json` | W10 | packed-bit/bpp/bpt row contract |
| `delta_ablation.csv` | W04 | current 3-seed course rerun; unverified |
| `pink_param_savings.csv` | W06 | current course rerun; unverified |
| `hip_latency.csv` | W11 / W12 | current Box B integrated rows + legacy diagnostics; unverified |
| `hip_latency.schema.json` | W11 / W12 | workload/provenance row contract |
| `hip_benchmark_gfx1201.json` | HIP runner | blocked-performance receipt; parity passed |
| `hip_benchmark.schema.json` | HIP runner | passed/blocked benchmark bundle contract |
| `hip_benchmark_receipt.schema.json` | HIP CLI | integrated fp16 timing receipt contract |

Current rerun summaries (historical teaching output, not verified paper evidence):

- W04 delta: **41.023 → 41.383 dB** over three seeds (**+0.360 dB**, +128 params).
- W06 Pink: decoder input **104 → 28 (−73.1%)**, total params
  **146,307 → 141,443 (−3.3%)**, PSNR **34.456 → 35.500 dB**. The
  112-resolution point saves 24.3% total but is 0.761 dB below concat.
- W10 matched int8 per-tensor means: **36.506 / 41.100 dB** at
  **11.491 / 11.500 bpp** (grid / PEPS, +4.594 dB).
- W11 integrated scalar-fp32 1024² baseline / concat / Pink on Box B:
  **246.3032 / 411.7999 / 295.3217 ms/iter**, parity passed and
  `comparable_to_paper=false`.
- W12 isolated 4096×64×64 Box B WMMA: **15.0896 ms fp16 / 15.4592 ms int8**.
- Fused fp16 all-mode parity passed, but its safety preflight projected
  **280.6 s** for only one warmup plus two iterations across four methods; the
  bounded runner refused the paper-scale timing and wrote no latency row.

Component microbenchmarks are never treated as paper-workload comparisons.
