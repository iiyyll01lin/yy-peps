"""Dense and hashed multi-grid encoders used by the paper baselines."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_resolutions(
    base_resolution: int,
    n_levels: int,
    per_level_scale: float,
    resolutions,
) -> tuple[int, ...]:
    if resolutions is None:
        if isinstance(n_levels, bool) or not isinstance(n_levels, int):
            raise TypeError("n_levels must be an integer")
        if n_levels < 1:
            raise ValueError("n_levels must be positive")
        values = tuple(
            max(2, int(round(base_resolution * (per_level_scale**level))))
            for level in range(n_levels)
        )
    else:
        if not isinstance(resolutions, Sequence):
            raise TypeError("resolutions must be a sequence")
        values = tuple(resolutions)
        if not values:
            raise ValueError("resolutions cannot be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("resolution entries must be integers")
    if any(value < 2 for value in values):
        raise ValueError("resolution entries must be at least 2")
    return values


def _validate_coords(coords: torch.Tensor, dim: int) -> None:
    if coords.ndim != 2 or coords.shape[1] != dim:
        raise ValueError(
            f"coords must have shape (N, {dim}), got {tuple(coords.shape)}"
        )
    if not coords.is_floating_point():
        raise TypeError("coords must be a floating-point tensor")


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

    def __init__(
        self,
        dim: int,
        base_resolution: int = 16,
        n_levels: int = 4,
        per_level_scale: float = 2.0,
        feature_dim: int = 2,
        init_std: float = 1e-2,
        *,
        resolutions=None,
    ) -> None:
        super().__init__()
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
            raise TypeError("feature_dim must be an integer")
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        self.dim = dim
        resolved = _resolve_resolutions(
            base_resolution, n_levels, per_level_scale, resolutions
        )
        self.n_levels = len(resolved)
        self.per_level_feature = feature_dim
        self.feature_dim = feature_dim * self.n_levels

        grids = nn.ParameterList()
        self.resolutions = resolved
        for res in self.resolutions:
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
        _validate_coords(coords, self.dim)
        n = coords.shape[0]
        g = coords * 2.0 - 1.0
        feats = [self._sample(grid, g, n) for grid in self.grids]
        return torch.cat(feats, dim=1).contiguous()

    def sample_channels(self, coords: torch.Tensor, channel_indices) -> torch.Tensor:
        indices = torch.as_tensor(
            channel_indices, device=coords.device, dtype=torch.long
        )
        if indices.ndim != 1:
            raise ValueError("channel_indices must be one-dimensional")
        if indices.numel() == 0:
            return coords.new_empty((coords.shape[0], 0))
        if (indices < 0).any() or (indices >= self.feature_dim).any():
            raise IndexError("channel index is out of range")
        return self(coords).index_select(1, indices)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.grids)


_PRIMES = (1, 2654435761, 805459861)


class HashGridEncoder(nn.Module):
    """Single- or multi-resolution hash grid.

    Each level allocates ``min(resolution**dim, 2**log2_hashmap_size)`` entries.
    This detail is required for the paper's ``[16, 32, 64, 128]`` multi-hash
    baseline: its first two levels remain collision-free while only the larger
    levels are capped, yielding the reported roughly 590k encoder parameters.
    """

    def __init__(
        self,
        dim: int,
        n_levels: int = 8,
        feature_dim: int = 2,
        base_resolution: int = 16,
        per_level_scale: float = 1.5,
        log2_hashmap_size: int = 19,
        init_std: float = 1e-4,
        *,
        resolutions=None,
    ) -> None:
        super().__init__()
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
            raise TypeError("feature_dim must be an integer")
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if (
            isinstance(log2_hashmap_size, bool)
            or not isinstance(log2_hashmap_size, int)
        ):
            raise TypeError("log2_hashmap_size must be an integer")
        if log2_hashmap_size < 1:
            raise ValueError("log2_hashmap_size must be positive")
        self.dim = dim
        self.resolutions = _resolve_resolutions(
            base_resolution, n_levels, per_level_scale, resolutions
        )
        self.n_levels = len(self.resolutions)
        self.per_level_feature = feature_dim
        self.feature_dim = feature_dim * self.n_levels
        self.max_table_size = 2**log2_hashmap_size
        self.table_sizes = tuple(
            min(resolution**dim, self.max_table_size)
            for resolution in self.resolutions
        )
        self.tables = nn.ParameterList(
            [
                nn.Parameter(torch.randn(size, feature_dim) * init_std)
                for size in self.table_sizes
            ]
        )
        offs = torch.tensor(
            [[(index >> axis) & 1 for axis in range(dim)] for index in range(2**dim)],
            dtype=torch.long,
        )
        self.register_buffer("corner_offsets", offs, persistent=False)

    def _dense_index(self, positions: torch.Tensor, resolution: int) -> torch.Tensor:
        index = positions[:, 0]
        stride = resolution
        for axis in range(1, self.dim):
            index = index + positions[:, axis] * stride
            stride *= resolution
        return index

    def _hash(self, positions: torch.Tensor, table_size: int) -> torch.Tensor:
        h = torch.zeros(
            positions.shape[0], dtype=torch.long, device=positions.device
        )
        for d in range(self.dim):
            h = h ^ (positions[:, d] * _PRIMES[d])
        return torch.remainder(h, table_size)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        _validate_coords(coords, self.dim)
        n = coords.shape[0]
        out = []
        for lvl in range(self.n_levels):
            res = self.resolutions[lvl]
            table_size = self.table_sizes[lvl]
            p = coords * (res - 1)
            p0 = torch.floor(p).long()
            frac = p - p0.float()
            maximum = torch.full(
                (self.dim,), res - 1, device=coords.device, dtype=torch.long
            )
            acc = coords.new_zeros((n, self.per_level_feature))
            for c in range(self.corner_offsets.shape[0]):
                off = self.corner_offsets[c]
                positions = p0 + off
                positions = torch.minimum(
                    torch.maximum(positions, torch.zeros_like(positions)),
                    maximum,
                )
                if table_size == res**self.dim:
                    idx = self._dense_index(positions, res)
                else:
                    idx = self._hash(positions, table_size)
                feat = self.tables[lvl][idx]
                w = coords.new_ones(n)
                for d in range(self.dim):
                    w = w * (frac[:, d] if off[d] == 1 else (1 - frac[:, d]))
                acc = acc + feat * w.unsqueeze(1)
            out.append(acc)
        return torch.cat(out, dim=1).contiguous()

    def sample_channels(self, coords: torch.Tensor, channel_indices) -> torch.Tensor:
        indices = torch.as_tensor(
            channel_indices, device=coords.device, dtype=torch.long
        )
        if indices.ndim != 1:
            raise ValueError("channel_indices must be one-dimensional")
        if indices.numel() == 0:
            return coords.new_empty((coords.shape[0], 0))
        if (indices < 0).any() or (indices >= self.feature_dim).any():
            raise IndexError("channel index is out of range")
        return self(coords).index_select(1, indices)

    @property
    def num_params(self) -> int:
        return sum(table.numel() for table in self.tables)
