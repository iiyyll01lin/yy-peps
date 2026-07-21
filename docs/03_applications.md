# Part III — Three applications (W07–W09) / 三個應用

## W07 · Implicit image representation (full Table 1) / 隱式影像(完整 Table 1)

**English.** We evaluate grid, Grid-PEPS, and Pink-PEPS across several Kodak images
on PSNR, SSIM, and **LSD** (log-spectral distance). LSD matters because it measures
high-frequency fidelity — where PEPS wins and plain PSNR under-reports. Our run:
pink_peps leads on all three (LSD 0.51 vs grid 0.82).

**繁體中文.** 在多張 Kodak 上以 PSNR、SSIM、**LSD**(對數頻譜距離)評估三方法。LSD
重要在於量測高頻保真度——PEPS 勝出而純 PSNR 低估之處。實測:pink_peps 三項皆領先
(LSD 0.51 vs grid 0.82)。

## W08 · Neural texture compression (main track) / 神經材質壓縮(主線)

**English.** A PBR material is a 9-channel signal (albedo, normal, roughness,
metalness, AO). We fit an NTC-style grid baseline vs Grid-PEPS vs NTC_PEPS on four
AmbientCG sets spanning the frequency spectrum. NTC_PEPS has the best mean PSNR;
the gain is largest on high-frequency **MetalPlates013** (the exact set NVIDIA
RTXNTC demos) and smallest on low-frequency wood — the honest picture of Table 2.

**繁體中文.** PBR 材質是 9 通道訊號。用 NTC 風格基線 vs Grid-PEPS vs NTC_PEPS 在四組
涵蓋頻率光譜的 AmbientCG 材質上擬合。NTC_PEPS 平均 PSNR 最佳;增益在高頻
**MetalPlates013**(NVIDIA RTXNTC 示範的同一組)最大、在低頻木頭最小——Table 2 的
誠實圖像。

## W09 · Signed distance functions / 有號距離函數

**English.** An SDF stores a shape as `f(x,y,z) -> signed distance`. We compare
dense TI-grid, multi-resolution, and hash encoders plus their PEPS versions on IoU
(Table 3), and render with marching cubes. Key nuance: PEPS's advantage appears on
**high-frequency / hard** shapes (the paper's Pitted Stonefish), not on smooth ones
— on a smooth torus a dense grid can already win, which the lab shows honestly.

**繁體中文.** SDF 把形狀存成 `f(x,y,z)->有號距離`。比較 dense TI-grid、multi-res、hash
及其 PEPS 版本的 IoU(Table 3),並用 marching cubes 渲染。關鍵細節:PEPS 的優勢出現在
**高頻/困難**形狀(論文的 Pitted Stonefish),而非光滑形狀——在光滑 torus 上 dense grid
已可勝出,實作誠實呈現這點。
