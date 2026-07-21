"""Model builders for neural texture compression.

繁體中文:材質壓縮的模型工廠。
- build_ntc_baseline:NTC 風格基線 = 單一 grid encoder + MLP,輸出 9 通道。
- build_grid_peps_texture:Grid-PEPS / NTC_PEPS,同一共享 grid 在 Lissajous 點取樣。
兩者輸出皆為 9 通道 bundle,方便逐通道與逐材質對照(Table 2)。
"""

from __future__ import annotations

import torch.nn as nn

from peps import Projector, GridEncoder, MLP, PEPS, make_aggregator

OUT_CHANNELS = 9


def build_ntc_baseline(resolution: int = 256, feature_dim: int = 8,
                       hidden_dim: int = 64, num_layers: int = 3):
    enc = GridEncoder(dim=2, resolution=resolution, feature_dim=feature_dim)
    mlp = MLP(feature_dim, OUT_CHANNELS, hidden_dim, num_layers)
    model = nn.Sequential(enc, mlp)
    return model, sum(p.numel() for p in model.parameters())


def build_grid_peps_texture(resolution: int = 256, feature_dim: int = 8,
                            num_frequencies: int = 6, aggregator: str = "concat",
                            hidden_dim: int = 64, num_layers: int = 3):
    proj = Projector(num_frequencies)
    enc = GridEncoder(dim=2, resolution=resolution, feature_dim=feature_dim)
    agg = make_aggregator(aggregator, proj.num_points, feature_dim)
    mlp = MLP(agg.out_dim, OUT_CHANNELS, hidden_dim, num_layers)
    model = PEPS(proj, enc, agg, mlp)
    return model, sum(p.numel() for p in model.parameters())
