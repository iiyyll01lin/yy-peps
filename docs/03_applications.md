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
metalness, AO). We fit four models on AmbientCG sets spanning the frequency
spectrum: a single-grid NTC baseline, an **RTXNTC-equivalent** (multi-resolution
latent grid + small MLP, mirroring RTXNTC's inference structure), Grid-PEPS, and
NTC_PEPS. NTC_PEPS has the best mean PSNR; the gain is largest on high-frequency
**MetalPlates013** (the exact set NVIDIA RTXNTC demos) and smallest on
low-frequency wood — the honest picture of Table 2.

> **On the RTXNTC comparison (honest note).** The official NVIDIA RTXNTC SDK
> cannot be built on our AMD hardware: its compressor hard-requires the CUDA
> Toolkit (`find_package(CUDAToolkit REQUIRED)` in `tools/cli/CMakeLists.txt`)
> and its inference path needs Vulkan/DX12 Cooperative Vector with an NVIDIA
> preview driver. Box B has no NVIDIA GPU, CUDA, or nvcc. We therefore compare
> against a faithful **PyTorch RTXNTC-equivalent** (`apps/texture/rtxntc.py`)
> that reproduces the same architecture — multi-res latents, small per-texel MLP,
> and an int8 path (W10) standing in for RTXNTC's cooperative-vector inference.

**繁體中文.** PBR 材質是 9 通道訊號。在涵蓋頻率光譜的 AmbientCG 材質上擬合四種模型:
單解析度 NTC 基線、**RTXNTC 等價**(多解析度 latent grid + 小 MLP,對應 RTXNTC 推論
結構)、Grid-PEPS、NTC_PEPS。NTC_PEPS 平均 PSNR 最佳;增益在高頻 **MetalPlates013**
(NVIDIA RTXNTC 示範的同一組)最大、在低頻木頭最小——Table 2 的誠實圖像。

> **關於 RTXNTC 對照(誠實註記).** 官方 NVIDIA RTXNTC SDK 在本 AMD 硬體**無法建置**:
> 壓縮器硬相依 CUDA Toolkit(`tools/cli/CMakeLists.txt` 的
> `find_package(CUDAToolkit REQUIRED)`),推論路徑需 Vulkan/DX12 Cooperative Vector
> 加 NVIDIA 預覽驅動。Box B 無 NVIDIA GPU、無 CUDA、無 nvcc。故改與忠實的
> **PyTorch RTXNTC 等價實作**(`apps/texture/rtxntc.py`)對照,重現相同架構——多解析度
> latent、小型逐 texel MLP、以 int8 路徑(W10)對應 RTXNTC 的 cooperative-vector 推論。

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
