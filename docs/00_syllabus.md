# Syllabus / 教學大綱

**PEPS on AMD — studying Positional Encoding Projected Sampling**
**AMD 上的 PEPS — 研讀位置編碼投影取樣**

12 teaching weeks + 2 project weeks. Each week: one concept lecture + one runnable
lab notebook + reading. Everything builds one master repo.

12 教學週 + 2 專題週。每週:一堂概念講授 + 一堂可跑實作 notebook + 指定閱讀。
全部堆進同一個母 repo。

This is an educational reimplementation with separate `course_fast` and
`paper_exact` protocols. Existing result CSVs remain `legacy-unverified`.
The teaching release indexes manifest-backed synthetic smokes, two explicitly
inconclusive pilots, and three public 512³ SDF provenance receipts; none is
paper-comparable.

本 repo 是教學型重實作,明確區分 `course_fast` 與 `paper_exact`;既有結果 CSV 在附完整
manifest 重跑前皆為 `legacy-unverified`。已發布證據不含論文可比數值。

## What this reproduction found / 本重現的結果

The texture track reproduced Table 2 in full, 594 of 594 jobs, and every method
landed about 1.154 dB below its published value with the method ordering
reversed. Both trace to the unpublished map-file selection rather than to
compute or to the loss: the reported score averages over individual maps whose
eight categories span 19.4 dB, and reweighting the same measured jobs to equal
categories reverses the ordering and cuts the out-of-sample error 3.1x. The
claim stops at sufficiency, since the paper's file list is not published.

Read `docs/reproducibility.md` for how that was established and
`results/texture_repro/` for the evidence and its limitations. W08 and W13/W14
grade students on exactly these habits.

The AMD track found something different, three times over, and it was the same
thing each time: what looked like a result about the paper or the hardware was
a defect in our own measurement or model. The kernel's first latency table came
from running each method to completion in turn on an idle card, which inflated
whichever method went first by 5.7x and invented an ordering. The kernel's
`__shared__` tiles were sized from compile-time caps set for the worst case, so
every workgroup reserved 32 KB to use at most 12 KB; narrowing them cut latency
roughly in half on both parts with byte-identical output, and the Pink ordering
that had looked like a disagreement with the paper came back into line, having
been an artefact of the shared cap rather than a property of the method. Then
the occupancy arithmetic explaining all of it turned out to be wrong twice,
caught both times by a hardware counter. It survived because it was right on
five of seven footprints, and its replacement on three of seven: **a model that
is right most of the time looks confirmed every time it is checked on an easy
case.**

Read `docs/05_amd_hardware.md` for the sequence and `results/hip_*.json` for
the measurements and what each still does not establish. W11 and W12 grade the
protocol, not only the parity.

材質軌道完整重現了 Table 2(594/594),但所有方法約低於論文值 1.154 dB 且排序相反。
兩者都指向未公布的 map 檔案選集,而非算力或 loss:表頭分數是對個別 map 平均,八個
類別相差 19.4 dB,重新加權成均衡後排序即反轉、樣本外誤差降低 3.1 倍。因論文未公布
檔案清單,此結論止於充分性。方法見 `docs/reproducibility.md`,證據見
`results/texture_repro/`。

AMD 軌道的發現屬於另一類,而且同一件事發生了三次:**看起來像「關於論文」或「關於
硬體」的結果,其實是我們自己量測或模型的缺陷**。第一份延遲表是從閒置卡上逐方法連續
量測得到的,先跑的方法被膨脹 5.7 倍,並憑空造出一個排序。kernel 的 `__shared__`
分頁由最壞情況的編譯期上限決定,每個 workgroup 保留 32 KB 卻最多只用 12 KB;收窄
上限後兩張卡的延遲各減半,且輸出逐位元相同,而先前看似「與論文不符」的 Pink 排序
也回到論文方向——它是共用上限的假象,不是方法的性質。接著,解釋這一切的佔用率算術
被發現錯了兩次,兩次都是硬體計數器抓到的。它們之所以能存活,是因為分別在七個
footprint 中對了五個與三個:**一個大多數時候正確的模型,在每次用簡單案例檢查時都
像是被驗證了。** 過程見 `docs/05_amd_hardware.md`,量測見 `results/hip_*.json`。

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
| W13 | Project kickoff / 專題起跑 | `W13_project_kickoff.ipynb` |
| W14 | Project showcase / 專題成果展 | `W14_project_showcase.ipynb` |

## Assessment / 評量
Weekly labs; a matched-evidence midterm (Fig. 5 + Table 1); and a final
**capstone** (W13–W14). The capstone has
**four concrete tracks** (short 3D video / calibration or QAT / new aggregator /
end-to-end runtime optimization), each shipping a validated notebook, results,
run/submission manifests, and one slide, graded by a shared rubric. Full brief,
tracks, and rubric: `06_capstone.md`. Midterm details: `08_midterm.md`.

每週實作;期中交付 Fig.5 + Table 1 的公平證據;期末**專題**(W13–W14)有
**四條具體軌道**(3D 短片 / 校準或 QAT / 新聚合器 /
端到端效能優化),每條交付通過驗證的 notebook、results、run/submission manifest
與一張投影片,依共用評分表評分。完整說明見 `06_capstone.md` 與 `08_midterm.md`。

## Readings and success criteria / 閱讀與成功門檻

The required/optional reading, runtime class, and evidence-based success
criteria for every week are in `07_readings_and_labs.md`; the machine-readable
source is `../course/labs.json`.

每週必讀/選讀、時間分級與證據式成功門檻見 `07_readings_and_labs.md`;機器可讀來源為
`../course/labs.json`。
