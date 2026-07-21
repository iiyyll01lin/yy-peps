"""Model builders for SDF fitting (3D).

繁體中文:SDF 擬合的模型工廠(3D)。四種 encoder + 其 PEPS 版本,對應論文
Table 3 的列:TI-grid(三線性 dense grid)、multi-res、hash,以及各自的 PEPS。
輸出為 1 維有號距離。
"""

from __future__ import annotations

import torch.nn as nn

from peps import Projector, GridEncoder, MLP, PEPS, make_aggregator
from peps.encoders.multires import MultiResGridEncoder, HashGridEncoder


def build_sdf_grid(resolution: int = 64, feature_dim: int = 4,
                   hidden_dim: int = 64, num_layers: int = 3):
    enc = GridEncoder(dim=3, resolution=resolution, feature_dim=feature_dim)
    mlp = MLP(feature_dim, 1, hidden_dim, num_layers)
    model = nn.Sequential(enc, mlp)
    return model, sum(p.numel() for p in model.parameters())


def build_sdf_multires(base_resolution: int = 16, n_levels: int = 4,
                       feature_dim: int = 2, hidden_dim: int = 64, num_layers: int = 3):
    enc = MultiResGridEncoder(dim=3, base_resolution=base_resolution,
                              n_levels=n_levels, feature_dim=feature_dim)
    mlp = MLP(enc.feature_dim, 1, hidden_dim, num_layers)
    model = nn.Sequential(enc, mlp)
    return model, sum(p.numel() for p in model.parameters())


def build_sdf_hash(n_levels: int = 8, feature_dim: int = 2,
                   log2_hashmap_size: int = 18, hidden_dim: int = 64, num_layers: int = 3):
    enc = HashGridEncoder(dim=3, n_levels=n_levels, feature_dim=feature_dim,
                          log2_hashmap_size=log2_hashmap_size)
    mlp = MLP(enc.feature_dim, 1, hidden_dim, num_layers)
    model = nn.Sequential(enc, mlp)
    return model, sum(p.numel() for p in model.parameters())


def build_sdf_peps(encoder: str = "grid", num_frequencies: int = 6,
                   aggregator: str = "concat", hidden_dim: int = 64, num_layers: int = 3,
                   **enc_kwargs):
    """PEPS-wrapped SDF encoder. ``encoder`` in {grid, multires, hash}."""
    proj = Projector(num_frequencies)
    if encoder == "grid":
        enc = GridEncoder(dim=3, resolution=enc_kwargs.get("resolution", 64),
                          feature_dim=enc_kwargs.get("feature_dim", 4))
        k = enc.feature_dim
    elif encoder == "multires":
        enc = MultiResGridEncoder(dim=3, base_resolution=enc_kwargs.get("base_resolution", 16),
                                  n_levels=enc_kwargs.get("n_levels", 4),
                                  feature_dim=enc_kwargs.get("feature_dim", 2))
        k = enc.feature_dim
    elif encoder == "hash":
        enc = HashGridEncoder(dim=3, n_levels=enc_kwargs.get("n_levels", 8),
                              feature_dim=enc_kwargs.get("feature_dim", 2),
                              log2_hashmap_size=enc_kwargs.get("log2_hashmap_size", 18))
        k = enc.feature_dim
    else:
        raise ValueError(f"unknown encoder {encoder!r}")
    agg = make_aggregator(aggregator, proj.num_points, k)
    mlp = MLP(agg.out_dim, 1, hidden_dim, num_layers)
    model = PEPS(proj, enc, agg, mlp)
    return model, sum(p.numel() for p in model.parameters())
