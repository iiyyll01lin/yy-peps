"""Model builders for the image application.

繁體中文:影像應用的模型工廠。提供三種 baseline 供逐週對照:
- build_plain_mlp:純 MLP(+可選 APE),示範頻譜偏差(W01)。
- build_grid:單純 grid encoder + MLP(W03 的 grid baseline)。
- build_grid_peps:Grid-PEPS(W05 主角)。
所有 builder 回傳 (model, param_count),方便畫「參數 vs PSNR」曲線(Fig.5)。
"""

from __future__ import annotations

import torch.nn as nn

from peps import (
    Projector, GridEncoder, MLP, PEPS, make_aggregator,
    AbsolutePositionalEncoding,
)


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def build_plain_mlp(
    num_frequencies: int = 0,
    hidden_dim: int = 64,
    num_layers: int = 4,
    out_dim: int = 3,
):
    """Plain coordinate MLP, optionally fronted by APE.

    ``num_frequencies=0`` -> raw (x,y) input: strong spectral bias (W01).
    ``num_frequencies>0`` -> APE input: recovers high frequencies.
    """
    if num_frequencies > 0:
        ape = AbsolutePositionalEncoding(2, num_frequencies, include_input=True)
        mlp = MLP(ape.feature_dim, out_dim, hidden_dim, num_layers)
        model = nn.Sequential(ape, mlp)
    else:
        model = MLP(2, out_dim, hidden_dim, num_layers)
    return model, _count(model)


def build_grid(
    resolution: int = 128,
    feature_dim: int = 4,
    hidden_dim: int = 64,
    num_layers: int = 3,
    out_dim: int = 3,
):
    """Single grid encoder -> MLP (the grid baseline that stalls in Fig.5)."""
    enc = GridEncoder(dim=2, resolution=resolution, feature_dim=feature_dim)
    mlp = MLP(feature_dim, out_dim, hidden_dim, num_layers)
    model = nn.Sequential(enc, mlp)
    return model, _count(model)


def build_grid_peps(
    resolution: int = 128,
    feature_dim: int = 4,
    num_frequencies: int = 6,
    aggregator: str = "concat",
    hidden_dim: int = 64,
    num_layers: int = 3,
    out_dim: int = 3,
):
    """Grid-PEPS: shared grid sampled at Lissajous points, then aggregated."""
    proj = Projector(num_frequencies)
    enc = GridEncoder(dim=2, resolution=resolution, feature_dim=feature_dim)
    agg = make_aggregator(aggregator, proj.num_points, feature_dim)
    mlp = MLP(agg.out_dim, out_dim, hidden_dim, num_layers)
    model = PEPS(proj, enc, agg, mlp)
    return model, _count(model)
