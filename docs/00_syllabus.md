# Syllabus / 教學大綱

**PEPS on AMD — reproducing Positional Encoding Projected Sampling**
**AMD 上的 PEPS — 重現位置編碼投影取樣**

12 teaching weeks + 2 project weeks. Each week: one concept lecture + one runnable
lab notebook + reading. Everything builds one master repo.

12 教學週 + 2 專題週。每週:一堂概念講授 + 一堂可跑實作 notebook + 指定閱讀。
全部堆進同一個母 repo。

## Hardware / 硬體
Two AMD boxes, merged into one git history:
- **Box B** — 4× Navi 48, `gfx1201` / RDNA 4 (main dev; the paper's target class).
- **Box A** — Radeon 8060S, `gfx1151` / RDNA 3.5 (RDNA3.5 comparison point).

## Weekly map / 逐週地圖
| Week | Topic | Lab notebook |
|---|---|---|
| W01 | INR & spectral bias / INR 與頻譜偏差 | `W01_intro_inr.ipynb` |
| W02 | Positional encoding & Lissajous / 位置編碼與 Lissajous | `W02_positional_encoding.ipynb` |
| W03 | Grid encoders & the bottleneck / grid 與瓶頸 | `W03_grid_bottleneck.ipynb` |
| W04 | Building the PEPS wrapper / 建立 PEPS wrapper | `W04_peps_wrapper.ipynb` |
| W05 | Grid-PEPS on images / 影像上的 Grid-PEPS | `W05_grid_peps_image.ipynb` |
| W06 | Pink-PEPS & the 1/f story / Pink-PEPS 與 1/f | `W06_pink_peps.ipynb` |
| W07 | App 1 wrap-up: image / 應用一收尾 | `W07_image_table1.ipynb` |
| W08 | App 2: neural texture compression / 材質壓縮 | `W08_texture_ntc.ipynb` |
| W09 | App 3: signed distance functions / SDF | `W09_sdf.ipynb` |
| W10 | Quantization study (original) / 量化研究(原創) | `W10_quantization.ipynb` |
| W11 | PyTorch -> HIP / HIP 移植 | `W11_hip.ipynb` |
| W12 | RDNA4 WMMA / RDNA4 WMMA | `W12_hip_wmma.ipynb` |
| W13-14 | Project weeks / 專題週 | student choice |

## Assessment / 評量
Weekly labs; a mid-term reproduction milestone (Fig. 5 + Table 1); a final project
extending PEPS or the quantization study.

每週實作;期中重現里程碑(Fig.5 + Table 1);期末專題(延伸 PEPS 或量化研究)。

## Reproduction targets / 重現目標
See `../results/` and the plan's §8 table. Each figure/table has a success
criterion checked by the corresponding week's lab.

見 `../results/` 與計畫 §8 表。每個圖表都有對應週次實作檢驗的成功標準。
