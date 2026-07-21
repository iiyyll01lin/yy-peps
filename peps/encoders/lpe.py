"""Local positional encoding from Fujieda et al. (paper Eq. 6--7).

The encoder stores one independently learned coefficient for every element of
the local Fourier basis at each grid vertex.  Coefficients are multilinearly
interpolated before they modulate the local basis, which keeps the encoding
continuous across cell boundaries.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import torch
import torch.nn as nn


def _coordinate_resolution(resolution, dim: int) -> tuple[int, ...]:
    if isinstance(resolution, int):
        values = (resolution,) * dim
    elif isinstance(resolution, Sequence):
        values = tuple(resolution)
    else:
        raise TypeError("resolution must be an integer or a sequence")
    if len(values) != dim:
        raise ValueError(f"resolution must contain {dim} entries")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("resolution entries must be integers")
    if any(value < 2 for value in values):
        raise ValueError("resolution entries must be at least 2")
    return values


class LocalPositionalEncoding(nn.Module):
    """Paper-faithful local positional encoding (LPE).

    ``resolution`` is expressed in coordinate-axis order, e.g. ``(W, H)`` for
    coordinates ordered as ``(x, y)``.  Following Eq. 6--7, each coordinate
    axis has ``2 * num_frequencies`` coefficients.  The first two are unmodulated
    grid features ``A_G``; the remaining pairs modulate local cosine/sine bases
    with angular frequencies ``2**i * pi`` for ``i=1..n-1``.

    The parameter count is therefore
    ``prod(resolution) * dim * 2 * num_frequencies``, matching the paper's
    32^3 x 18 SDF configuration.
    """

    def __init__(
        self,
        dim: int,
        resolution,
        num_frequencies: int,
        *,
        init_bound: float = 1e-4,
        boundary: str = "border",
    ) -> None:
        super().__init__()
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError("dim must be an integer")
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if (
            isinstance(num_frequencies, bool)
            or not isinstance(num_frequencies, int)
        ):
            raise TypeError("num_frequencies must be an integer")
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be positive")
        if not math.isfinite(init_bound) or init_bound < 0:
            raise ValueError("init_bound must be finite and non-negative")
        if boundary not in {"border", "periodic"}:
            raise ValueError("boundary must be 'border' or 'periodic'")

        self.dim = dim
        self.resolution = _coordinate_resolution(resolution, dim)
        self.num_frequencies = num_frequencies
        self.boundary = boundary
        self.per_axis_feature_dim = 2 * num_frequencies
        self.feature_dim = dim * self.per_axis_feature_dim

        entries = math.prod(self.resolution)
        coefficients = torch.empty(entries, self.feature_dim)
        nn.init.uniform_(coefficients, -init_bound, init_bound)
        self.coefficients = nn.Parameter(coefficients)
        offsets = torch.tensor(
            tuple(itertools.product((0, 1), repeat=dim)),
            dtype=torch.long,
        )
        self.register_buffer("corner_offsets", offsets, persistent=False)

    @property
    def num_params(self) -> int:
        return self.coefficients.numel()

    def _linear_indices(self, positions: torch.Tensor) -> torch.Tensor:
        index = positions[:, 0]
        stride = self.resolution[0]
        for axis in range(1, self.dim):
            index = index + positions[:, axis] * stride
            stride *= self.resolution[axis]
        return index

    def interpolate_coefficients(self, coords: torch.Tensor) -> torch.Tensor:
        """Multilinearly interpolate Eq. 7 coefficients at ``coords``."""

        if coords.ndim != 2 or coords.shape[1] != self.dim:
            raise ValueError(
                f"coords must have shape (N, {self.dim}), got "
                f"{tuple(coords.shape)}"
            )
        if not coords.is_floating_point():
            raise TypeError("coords must be a floating-point tensor")

        resolution = coords.new_tensor(self.resolution)
        scaled = coords * resolution
        base_unbounded = torch.floor(scaled).long()
        local = scaled - torch.floor(scaled)
        max_index = torch.tensor(
            self.resolution, device=coords.device, dtype=torch.long
        ) - 1

        result = coords.new_zeros((coords.shape[0], self.feature_dim))
        for offset in self.corner_offsets:
            position = base_unbounded + offset
            if self.boundary == "periodic":
                position = torch.remainder(position, max_index + 1)
            else:
                position = torch.minimum(
                    torch.maximum(position, torch.zeros_like(position)),
                    max_index,
                )
            weight = coords.new_ones(coords.shape[0])
            for axis in range(self.dim):
                weight = weight * (
                    local[:, axis]
                    if int(offset[axis]) == 1
                    else 1.0 - local[:, axis]
                )
            values = self.coefficients[self._linear_indices(position)]
            result = result + values * weight.unsqueeze(1)
        return result

    def local_basis(self, coords: torch.Tensor) -> torch.Tensor:
        """Return the unweighted Eq. 6 basis in axis-major order."""

        resolution = coords.new_tensor(self.resolution)
        local = torch.remainder(coords * resolution, 1.0)
        basis = coords.new_ones(
            (coords.shape[0], self.dim, self.per_axis_feature_dim)
        )
        for frequency_index in range(1, self.num_frequencies):
            angle = local * (2**frequency_index * math.pi)
            output_index = 2 * frequency_index
            basis[:, :, output_index] = torch.cos(angle)
            basis[:, :, output_index + 1] = torch.sin(angle)
        return basis.reshape(coords.shape[0], self.feature_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coefficients = self.interpolate_coefficients(coords)
        return coefficients * self.local_basis(coords)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, resolution={self.resolution}, "
            f"num_frequencies={self.num_frequencies}, "
            f"feature_dim={self.feature_dim}, boundary={self.boundary!r}"
        )
