"""RTXNTC-equivalent baseline (PyTorch) — the honest AMD substitute.

繁體中文:RTXNTC 對照 baseline。README 主打「對照 NVIDIA RTXNTC」,但官方
RTXNTC 在本 AMD 硬體上**無法建置**(spike 結論,見下),故本檔提供論文架構等價的
PyTorch 重實作,讓 W08 仍能做三方對照(grid / Grid-PEPS / RTXNTC-equivalent)。

========================================================================
RTXNTC build spike (Box B, RDNA4 gfx1201, ROCm 7.2.3) — 結論:不可建置
------------------------------------------------------------------------
官方 https://github.com/NVIDIA-RTX/RTXNTC 的硬相依:
  * tools/cli/CMakeLists.txt:17  find_package(CUDAToolkit REQUIRED)
    + target_link_libraries(... CUDA::cudart_static ...)   ← 壓縮器需 CUDA
  * 推論路徑需 Vulkan/DX12 Cooperative Vector + NVIDIA 預覽驅動 590.26。
Box B 現況:無 NVIDIA GPU、無 /usr/local/cuda、無 nvcc(lspci 僅見 AMD RDNA4)。
→ 連離線壓縮器 ntc-cli 都因 CUDAToolkit REQUIRED 無法 configure。
→ 依計畫 fallback 到本等價實作,並在 docs 誠實標註。
========================================================================

RTXNTC 的推論架構(對照本檔重實作的對應):
  1. 多解析度 latent 網格(NTC 的壓縮表示)  -> MultiResGridEncoder
  2. 小 MLP 解碼器(逐 texel 推論)          -> peps.MLP
  3. int8 cooperative-vector 硬體推論路徑    -> quant/ptq.py 的 int8 模擬
本檔把這三者組成一個「NTC 風格」模型,與 Grid-PEPS 在相同參數預算下對照。
與 apps/texture/build.py 的 build_ntc_baseline(單解析度 grid)不同:此處用
多解析度 latent，更貼近 RTXNTC 的實際壓縮表示。
"""

from __future__ import annotations

import torch.nn as nn

from peps import MLP
from peps.encoders.multires import MultiResGridEncoder

OUT_CHANNELS = 9  # PBR bundle: albedo(3)+normal(3)+rough+metal+ao


def build_rtxntc_equiv(
    base_resolution: int = 64,
    n_levels: int = 4,
    per_level_feature: int = 4,
    per_level_scale: float = 2.0,
    hidden_dim: int = 64,
    num_layers: int = 3,
):
    """RTXNTC-equivalent: multi-resolution latent grid -> small MLP -> 9ch.

    Mirrors RTXNTC's inference structure (compressed multi-res latents + a small
    per-texel MLP). Quantize with ``peps.quant.ptq`` to emulate the int8
    cooperative-vector path. Returns ``(model, param_count)``.
    """
    enc = MultiResGridEncoder(
        dim=2,
        base_resolution=base_resolution,
        n_levels=n_levels,
        per_level_scale=per_level_scale,
        feature_dim=per_level_feature,
    )
    mlp = MLP(enc.feature_dim, OUT_CHANNELS, hidden_dim, num_layers)
    model = nn.Sequential(enc, mlp)
    return model, sum(p.numel() for p in model.parameters())
