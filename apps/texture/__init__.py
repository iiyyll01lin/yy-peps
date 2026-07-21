"""Application 2 — neural texture compression (main track).

繁體中文:應用二,神經材質壓縮(主線)。把一組 PBR 材質(albedo+normal+roughness+
metalness+AO ≈ 9 通道)當成一個多通道訊號 f(u,v)->9ch 來擬合,對照 NTC 基線、
Grid-PEPS 與 NTC_PEPS。對應論文 Table 2,並與 NVIDIA RTXNTC 的 MetalPlates013 並排。
"""

from .data import load_pbr_bundle, bundle_to_coords_targets, CHANNEL_LAYOUT
from .build import build_ntc_baseline, build_grid_peps_texture

__all__ = [
    "load_pbr_bundle",
    "bundle_to_coords_targets",
    "CHANNEL_LAYOUT",
    "build_ntc_baseline",
    "build_grid_peps_texture",
]
