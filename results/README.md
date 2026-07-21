# results/ — generated tables & figures / 生成的表格與圖

> **Legacy / unverified:** the checked-in CSVs were generated before the
> paper-exact core equation fix. They are retained for provenance only and must
> not be treated as reproduced paper results. Later phases will regenerate and
> validate them from immutable paper configs.

Every future quantitative claim in `docs/` and the notebooks must be backed by
a validated file here.

目前 CSV 均為核心公式修正前的 legacy／未驗證產物，只供追溯，不代表已重現論文；
後續階段會以固定設定重新產生並驗證。

- `*.csv` — tracked, diffable artifacts; current files are legacy/unverified.
- `*.png` — visualizations regenerated from the CSVs (git-ignored).

Written via `peps.report` (`write_table` / `plot_xy`). Regenerate everything by
re-running the weekly notebooks, or the per-app build scripts.

透過 `peps.report` 產生。重跑各週 notebook 或各 app 的 build 腳本即可重生。

| File | Produced by | Backs |
|---|---|---|
| `fig5_bottleneck.csv` | W03 / W05 | params-vs-PSNR 瓶頸曲線 (Fig.5) |
| `table1_image.csv` | W07 | 影像 PSNR/SSIM/LSD (Table 1) |
| `table2_texture.csv` | W08 | 材質 NTC 對照 (Table 2) |
| `table3_sdf.csv` | W09 | SDF IoU (Table 3) |
| `w10_rate_distortion.csv` | W10 | 量化率失真曲線 (原創) |
| `hip_latency.csv` | W11 / W12 | RDNA3.5 / RDNA4 kernel latency |
