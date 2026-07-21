"""Paper NTC_N encoder components.

NTC_N uses four-neighbour concatenation from a high-resolution grid, bilinear
interpolation from a lower-resolution grid, and a 3-octave tiled triangular
positional encoding.  For the paper's 12/20 channel grids this produces
``4 * 12 + 20 + 12 = 80`` decoder inputs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..aggregate import make_aggregator
from ..projector import Projector
from .grid import GridEncoder
from .lpe import _coordinate_resolution


def _grid_encoder_resolution(coordinate_resolution: tuple[int, ...]):
    # GridEncoder stores tensors in PyTorch spatial order (..., H, W) while
    # public NTC resolutions follow coordinate order (x, y).
    return tuple(reversed(coordinate_resolution))


def _triangle_cos(phase: torch.Tensor) -> torch.Tensor:
    """Unit triangle-wave approximation of cosine for phase in cycles."""

    wrapped = torch.remainder(phase, 1.0)
    return torch.abs(wrapped - 0.5) * 4.0 - 1.0


class FourNeighborGridEncoder(nn.Module):
    """Return the four closest 2D grid vectors without interpolation.

    Corners are concatenated in ``(00, 10, 01, 11)`` order, where each pair is
    ``(x_offset, y_offset)``.  This is the learned interpolation input ``G0``
    described in the NTC paper.
    """

    def __init__(
        self,
        resolution,
        feature_dim: int,
        *,
        init_std: float = 1e-2,
    ) -> None:
        super().__init__()
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
            raise TypeError("feature_dim must be an integer")
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if not math.isfinite(init_std) or init_std < 0:
            raise ValueError("init_std must be finite and non-negative")

        self.dim = 2
        self.resolution = _coordinate_resolution(resolution, self.dim)
        self.per_corner_feature_dim = feature_dim
        self.feature_dim = 4 * feature_dim
        entries = math.prod(self.resolution)
        self.grid = nn.Parameter(torch.randn(entries, feature_dim) * init_std)
        self.corner_order = ((0, 0), (1, 0), (0, 1), (1, 1))

    @property
    def num_params(self) -> int:
        return self.grid.numel()

    def _linear_indices(self, positions: torch.Tensor) -> torch.Tensor:
        return positions[:, 0] + positions[:, 1] * self.resolution[0]

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"coords must have shape (N, 2), got {tuple(coords.shape)}"
            )
        if not coords.is_floating_point():
            raise TypeError("coords must be a floating-point tensor")

        scale = coords.new_tensor(
            (self.resolution[0] - 1, self.resolution[1] - 1)
        )
        base = torch.floor(coords * scale).long()
        maximum = torch.tensor(
            self.resolution, device=coords.device, dtype=torch.long
        ) - 1
        corners = []
        for offset in self.corner_order:
            position = base + torch.tensor(
                offset, device=coords.device, dtype=torch.long
            )
            position = torch.minimum(
                torch.maximum(position, torch.zeros_like(position)), maximum
            )
            corners.append(self.grid[self._linear_indices(position)])
        return torch.cat(corners, dim=1)

    def extra_repr(self) -> str:
        return (
            f"resolution={self.resolution}, "
            f"per_corner_feature_dim={self.per_corner_feature_dim}, "
            f"feature_dim={self.feature_dim}"
        )


class TiledTriangularEncoding(nn.Module):
    """NTC's 3-octave, 8-texel tiled triangular positional encoding.

    Each octave emits cosine- and sine-phase triangle waves for every axis.
    For 2D and three octaves the output width is 12.  ``resolution`` is the
    target signal resolution in coordinate order; normalized coordinates are
    mapped to integer texel positions using ``resolution - 1``.
    """

    def __init__(
        self,
        resolution,
        *,
        dim: int = 2,
        num_octaves: int = 3,
        tile_size: int = 8,
    ) -> None:
        super().__init__()
        if isinstance(num_octaves, bool) or not isinstance(num_octaves, int):
            raise TypeError("num_octaves must be an integer")
        if num_octaves < 1:
            raise ValueError("num_octaves must be positive")
        if isinstance(tile_size, bool) or not isinstance(tile_size, int):
            raise TypeError("tile_size must be an integer")
        if tile_size < 1:
            raise ValueError("tile_size must be positive")
        self.dim = dim
        self.resolution = _coordinate_resolution(resolution, dim)
        self.num_octaves = num_octaves
        self.tile_size = tile_size
        self.feature_dim = dim * num_octaves * 2

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != self.dim:
            raise ValueError(
                f"coords must have shape (N, {self.dim}), got "
                f"{tuple(coords.shape)}"
            )
        if not coords.is_floating_point():
            raise TypeError("coords must be a floating-point tensor")

        texel_scale = coords.new_tensor(
            tuple(value - 1 for value in self.resolution)
        )
        texel = coords * texel_scale
        features = []
        for axis in range(self.dim):
            for octave in range(self.num_octaves):
                phase = texel[:, axis] * (2**octave) / self.tile_size
                features.append(_triangle_cos(phase))
                features.append(_triangle_cos(phase - 0.25))
        return torch.stack(features, dim=1)

    def extra_repr(self) -> str:
        return (
            f"resolution={self.resolution}, num_octaves={self.num_octaves}, "
            f"tile_size={self.tile_size}, feature_dim={self.feature_dim}"
        )


class NTCNEncoder(nn.Module):
    """Unquantized NTC_N encoder used by the PEPS comparisons."""

    def __init__(
        self,
        signal_resolution,
        *,
        g0_resolution,
        g0_feature_dim: int = 12,
        g1_resolution,
        g1_feature_dim: int = 20,
        num_octaves: int = 3,
        tile_size: int = 8,
        init_std: float = 1e-2,
    ) -> None:
        super().__init__()
        self.dim = 2
        self.signal_resolution = _coordinate_resolution(
            signal_resolution, self.dim
        )
        g0_resolution = _coordinate_resolution(g0_resolution, self.dim)
        g1_resolution = _coordinate_resolution(g1_resolution, self.dim)
        self.g0 = FourNeighborGridEncoder(
            g0_resolution, g0_feature_dim, init_std=init_std
        )
        self.g1 = GridEncoder(
            dim=2,
            resolution=_grid_encoder_resolution(g1_resolution),
            feature_dim=g1_feature_dim,
            init_std=init_std,
        )
        self.tiled_encoding = TiledTriangularEncoding(
            self.signal_resolution,
            num_octaves=num_octaves,
            tile_size=tile_size,
        )
        self.feature_dim = (
            self.g0.feature_dim
            + self.g1.feature_dim
            + self.tiled_encoding.feature_dim
        )

    @property
    def num_params(self) -> int:
        return self.g0.num_params + self.g1.num_params

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                self.g0(coords),
                self.g1(coords),
                self.tiled_encoding(coords),
            ),
            dim=1,
        )

    def extra_repr(self) -> str:
        return (
            f"signal_resolution={self.signal_resolution}, "
            f"feature_dim={self.feature_dim}, num_params={self.num_params}"
        )


class NTCPEPSEncoder(nn.Module):
    """NTC_N grids replaced by their shared PEPS counterpart.

    Tiled positional encoding remains evaluated once at the original texel,
    while the two learned grids are sampled at all PEPS points.  This follows
    the paper's description of replacing the NTC grids, rather than repeatedly
    encoding the analytic tiled features.
    """

    def __init__(
        self,
        signal_resolution,
        *,
        g0_resolution,
        g0_feature_dim: int = 12,
        g1_resolution,
        g1_feature_dim: int = 20,
        num_frequencies: int = 4,
        aggregator: str = "concat",
        num_octaves: int = 3,
        tile_size: int = 8,
        init_std: float = 1e-2,
        append_input: bool = False,
    ) -> None:
        super().__init__()
        signal_resolution = _coordinate_resolution(signal_resolution, 2)
        g0_resolution = _coordinate_resolution(g0_resolution, 2)
        g1_resolution = _coordinate_resolution(g1_resolution, 2)
        self.projector = Projector(num_frequencies)
        self.g0 = FourNeighborGridEncoder(
            g0_resolution, g0_feature_dim, init_std=init_std
        )
        self.g1 = GridEncoder(
            dim=2,
            resolution=_grid_encoder_resolution(g1_resolution),
            feature_dim=g1_feature_dim,
            init_std=init_std,
        )
        self.tiled_encoding = TiledTriangularEncoding(
            signal_resolution,
            num_octaves=num_octaves,
            tile_size=tile_size,
        )
        self.grid_feature_dim = self.g0.feature_dim + self.g1.feature_dim
        frequency_allocated = aggregator.lower() in {"pink", "brownian"}
        aggregate_kwargs = (
            {
                "num_frequencies": self.projector.num_frequencies,
                "include_input": self.projector.include_input,
                "frequency_scales": self.projector.frequency_scales,
            }
            if frequency_allocated
            else {}
        )
        self.aggregator = make_aggregator(
            aggregator,
            self.projector.num_points,
            self.grid_feature_dim,
            **aggregate_kwargs,
        )
        self.append_input = bool(append_input)
        self.feature_dim = (
            self.aggregator.out_dim
            + self.tiled_encoding.feature_dim
            + (2 if self.append_input else 0)
        )

    @property
    def num_params(self) -> int:
        return self.g0.num_params + self.g1.num_params

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"coords must have shape (N, 2), got {tuple(coords.shape)}"
            )
        points = self.projector(coords)
        batch, count, dim = points.shape
        flattened = points.reshape(batch * count, dim)
        grid_features = torch.cat(
            (self.g0(flattened), self.g1(flattened)),
            dim=1,
        ).reshape(batch, count, self.grid_feature_dim)
        features = [
            self.aggregator(grid_features),
            self.tiled_encoding(coords),
        ]
        if self.append_input:
            features.append(coords)
        return torch.cat(features, dim=1)
