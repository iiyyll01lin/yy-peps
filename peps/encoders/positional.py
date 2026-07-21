"""Analytic positional encodings — APE (paper Eq. 1-5) and Identity."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..projector import resolve_frequency_schedule


class IdentityEncoder(nn.Module):
    """Return coordinates unchanged. Encodes to ``feature_dim == dim``.

    Identity-encoder PEPS features are affinely equivalent to APE features.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.feature_dim = dim

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return coords


class AbsolutePositionalEncoding(nn.Module):
    """Classic Fourier / positional encoding (paper Eq. 1-5).

    The feature order mirrors the PEPS point contract:
    ``[x, sin(phi_1*x), ..., sin(phi_L*x), cos(phi_1*x), ...,
    cos(phi_L*x)]``, with each entry containing all coordinate dimensions.
    The paper schedule is ``phi_i = 2**i*pi`` for ``i=1,...,L``.

    Args:
        dim: input coordinate dimensionality.
        num_frequencies: ``L``.
        include_input: prepend raw ``x`` to the encoding.
        base: frequency ladder base (2 in the paper).
        frequency_exponents: optional exponents for ``base**exponent*pi``.
        frequencies: optional angular coefficients ``phi_i`` supplied directly.

    Shape:
        input  ``coords``: ``(N, dim)``.
        output ``feats``:  ``(N, feature_dim)`` where
            ``feature_dim = dim * (include_input + 2 * L)``.
    """

    def __init__(
        self,
        dim: int,
        num_frequencies: int | None = None,
        include_input: bool = True,
        base: float = 2.0,
        *,
        frequency_exponents=None,
        frequencies=None,
    ) -> None:
        super().__init__()
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError("dim must be an integer")
        if dim <= 0:
            raise ValueError("dim must be positive")
        num_frequencies, freqs, exponents = resolve_frequency_schedule(
            num_frequencies,
            base=base,
            frequency_exponents=frequency_exponents,
            frequencies=frequencies,
        )
        self.dim = dim
        self.num_frequencies = num_frequencies
        self.include_input = bool(include_input)
        self.base = float(base)
        self.register_buffer("freqs", freqs, persistent=False)
        self.register_buffer("frequency_exponents", exponents, persistent=False)
        self.feature_dim = dim * (
            int(self.include_input) + 2 * num_frequencies
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim < 1 or coords.shape[-1] != self.dim:
            raise ValueError(
                f"coords must have shape (..., {self.dim}), got "
                f"{tuple(coords.shape)}"
            )
        if not coords.is_floating_point():
            raise TypeError("coords must be a floating-point tensor")

        outs = []
        if self.include_input:
            outs.append(coords)
        if self.num_frequencies > 0:
            freqs = self.freqs.to(dtype=coords.dtype)
            ang = coords.unsqueeze(-2) * freqs.view(-1, 1)
            leading_shape = coords.shape[:-1]
            outs.append(torch.sin(ang).reshape(*leading_shape, -1))
            outs.append(torch.cos(ang).reshape(*leading_shape, -1))
        if not outs:
            return coords.new_empty((*coords.shape[:-1], 0))
        return torch.cat(outs, dim=-1)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_frequencies={self.num_frequencies}, "
            f"include_input={self.include_input}, feature_dim={self.feature_dim}"
        )
