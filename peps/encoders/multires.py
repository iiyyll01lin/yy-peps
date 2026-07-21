"""Multi-resolution and hash grid encoders (for the SDF table, W09).

繁體中文:多解析度與 hash grid 編碼器,供 SDF 章節(W09)重現論文表格。
- MultiResGridEncoder:多個不同解析度的 dense grid,latent 串接(Instant-NGP 風格,
  但用 dense 而非 hash,教學上更好理解)。
- HashGridEncoder:多解析度 + 空間雜湊表(記憶體省,對應論文 hash-grid 列)。
兩者介面與 GridEncoder 相同(coords (N,dim) in [0,1] -> (N, feature_dim*levels)),
因此可直接放進 PEPS wrapper 當共享 encoder。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResGridEncoder(nn.Module):
    """Concatenated dense grids at geometrically-spaced resolutions.

    Args:
        dim: 2 or 3.
        base_resolution: coarsest grid side length.
        n_levels: number of resolution levels.
        per_level_scale: geometric growth factor between levels.
        feature_dim: channels per level.
    Output dim = ``feature_dim * n_levels``.
    """

    def __init__(self, dim: int, base_resolution: int = 16, n_levels: int = 4,
                 per_level_scale: float = 2.0, feature_dim: int = 2,
                 init_std: float = 1e-2) -> None:
        super().__init__()
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        self.dim = dim
        self.n_levels = n_levels
        self.per_level_feature = feature_dim
        self.feature_dim = feature_dim * n_levels

        grids = nn.ParameterList()
        self.resolutions = []
        for lvl in range(n_levels):
            res = max(2, int(round(base_resolution * (per_level_scale ** lvl))))
            self.resolutions.append(res)
            shape = (1, feature_dim) + (res,) * dim
            grids.append(nn.Parameter(torch.randn(*shape) * init_std))
        self.grids = grids

    def _sample(self, grid: torch.Tensor, g: torch.Tensor, n: int) -> torch.Tensor:
        if self.dim == 2:
            samp = g.view(1, n, 1, 2)
            out = F.grid_sample(grid, samp, mode="bilinear",
                                align_corners=True, padding_mode="border")
            return out.view(self.per_level_feature, n).t()
        samp = g.view(1, n, 1, 1, 3)
        out = F.grid_sample(grid, samp, mode="bilinear",
                            align_corners=True, padding_mode="border")
        return out.view(self.per_level_feature, n).t()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        n = coords.shape[0]
        g = coords * 2.0 - 1.0
        feats = [self._sample(grid, g, n) for grid in self.grids]
        return torch.cat(feats, dim=1).contiguous()

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.grids)


_PRIMES = (1, 2654435761, 805459861)


class HashGridEncoder(nn.Module):
    """Multi-resolution hash grid (Instant-NGP style) for 2D/3D.

    Each level hashes integer vertex coordinates into a fixed-size table, then
    (bi/tri)linearly interpolates the 2^dim corner entries. Memory is bounded by
    ``2**log2_hashmap_size`` per level regardless of resolution.
    """

    def __init__(self, dim: int, n_levels: int = 8, feature_dim: int = 2,
                 base_resolution: int = 16, per_level_scale: float = 1.5,
                 log2_hashmap_size: int = 19, init_std: float = 1e-4) -> None:
        super().__init__()
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        self.dim = dim
        self.n_levels = n_levels
        self.per_level_feature = feature_dim
        self.feature_dim = feature_dim * n_levels
        self.table_size = 2 ** log2_hashmap_size

        self.resolutions = [
            max(2, int(round(base_resolution * (per_level_scale ** l))))
            for l in range(n_levels)
        ]
        self.tables = nn.Parameter(
            torch.randn(n_levels, self.table_size, feature_dim) * init_std
        )
        # corner offsets: (2^dim, dim)
        offs = torch.tensor(
            [[(i >> d) & 1 for d in range(dim)] for i in range(2 ** dim)],
            dtype=torch.long,
        )
        self.register_buffer("corner_offsets", offs, persistent=False)

    def _hash(self, ipos: torch.Tensor, level: int) -> torch.Tensor:
        # ipos: (N, dim) long -> (N,) index into table
        h = torch.zeros(ipos.shape[0], dtype=torch.long, device=ipos.device)
        for d in range(self.dim):
            h = h ^ (ipos[:, d] * _PRIMES[d])
        return h % self.table_size

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        n = coords.shape[0]
        out = []
        for lvl in range(self.n_levels):
            res = self.resolutions[lvl]
            p = coords * (res - 1)             # (N, dim) in [0, res-1]
            p0 = torch.floor(p).long()
            frac = p - p0.float()
            acc = torch.zeros(n, self.per_level_feature, device=coords.device)
            for c in range(self.corner_offsets.shape[0]):
                off = self.corner_offsets[c]
                ipos = p0 + off
                idx = self._hash(ipos, lvl)
                feat = self.tables[lvl][idx]  # (N, feature_dim)
                # trilinear/bilinear weight for this corner
                w = torch.ones(n, device=coords.device)
                for d in range(self.dim):
                    w = w * (frac[:, d] if off[d] == 1 else (1 - frac[:, d]))
                acc = acc + feat * w.unsqueeze(1)
            out.append(acc)
        return torch.cat(out, dim=1).contiguous()

    @property
    def num_params(self) -> int:
        return self.tables.numel()
