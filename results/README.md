# results/ — generated tables & figures / 生成的表格與圖

Every quantitative claim in `docs/` and the notebooks is backed by a file here.

`docs/` 與 notebook 中的每個量化宣稱都由此處的檔案背書。

- `*.csv` — the **source of truth** (tracked in git, diffable).
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
