"""Unverified RTXNTC-inspired proxy retained for the course track.

This module is *not* an RTXNTC-equivalent implementation. It only shares the
broad idea of multi-resolution latent grids followed by a small MLP; it does
not reproduce the official codec, latent quantization, network inputs,
training, mip handling, or inference math. It must be labelled ``proxy`` in
artifacts and is excluded from paper-exact Table 2.

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

The proxy components are:
  1. 多解析度 latent 網格(NTC 的壓縮表示)  -> MultiResGridEncoder
  2. 小 MLP 解碼器(逐 texel 推論)          -> peps.MLP
  3. int8 cooperative-vector 硬體推論路徑    -> quant/ptq.py 的 int8 模擬
These similarities are insufficient to establish parity with RTXNTC.
"""

from __future__ import annotations

import torch.nn as nn

from peps import MLP
from peps.encoders.multires import MultiResGridEncoder

OUT_CHANNELS = 9  # PBR bundle: albedo(3)+normal(3)+rough+metal+ao


def build_rtxntc_proxy(
    base_resolution: int = 64,
    n_levels: int = 4,
    per_level_feature: int = 4,
    per_level_scale: float = 2.0,
    hidden_dim: int = 64,
    num_layers: int = 4,
):
    """Build the unverified multi-grid course proxy."""
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


# Backward-compatible import for old notebooks. New code and all displayed
# labels must use ``build_rtxntc_proxy`` / ``rtxntc_proxy``.
build_rtxntc_equiv = build_rtxntc_proxy
