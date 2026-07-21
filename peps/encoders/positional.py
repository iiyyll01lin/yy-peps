"""Analytic positional encodings — APE (paper Eq. 1-5) and Identity.

繁體中文:解析式(不可學習)編碼器。
- AbsolutePositionalEncoding (APE):經典 sin/cos 頻率編碼。
- IdentityEncoder:原封不動回傳座標。把 IdentityEncoder 放進 PEPS wrapper 時,
  整個 PEPS 會退化回純 APE —— 這正是論文「PEPS 是 APE 的泛化」宣稱的驗證點,
  也是我們單元測試的 sanity check。
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class IdentityEncoder(nn.Module):
    """Return coordinates unchanged. Encodes to ``feature_dim == dim``.

    Used to prove the ``Identity-encoder PEPS == APE`` equivalence.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.feature_dim = dim

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return coords


class AbsolutePositionalEncoding(nn.Module):
    """Classic Fourier / positional encoding (paper Eq. 1-5).

    ``enc(x) = [x, sin(2^0 pi x), cos(2^0 pi x), ..., sin(2^{L-1} pi x), cos(...)]``

    Args:
        dim: input coordinate dimensionality.
        num_frequencies: ``L``.
        include_input: prepend raw ``x`` to the encoding.
        base: frequency ladder base (2 in the paper).

    Shape:
        input  ``coords``: ``(N, dim)``.
        output ``feats``:  ``(N, feature_dim)`` where
            ``feature_dim = dim * (include_input + 2 * L)``.
    """

    def __init__(
        self,
        dim: int,
        num_frequencies: int,
        include_input: bool = True,
        base: float = 2.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        freqs = base ** torch.arange(num_frequencies, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)
        self.feature_dim = dim * (int(include_input) + 2 * num_frequencies)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        outs = []
        if self.include_input:
            outs.append(coords)
        if self.num_frequencies > 0:
            # (N, dim, L)
            ang = coords.unsqueeze(-1) * self.freqs
            n = coords.shape[0]
            outs.append(torch.sin(ang).reshape(n, -1))
            outs.append(torch.cos(ang).reshape(n, -1))
        return torch.cat(outs, dim=-1)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_frequencies={self.num_frequencies}, "
            f"include_input={self.include_input}, feature_dim={self.feature_dim}"
        )
