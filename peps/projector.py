"""Lissajous projector — paper Eq. (6)-(7).

繁體中文:PEPS 的第一步。對輸入座標 x,用 L 個倍頻 phi_i = 2^i * pi 產生
一組「興趣點」P_x = (x, S_1..S_L, C_1..C_L),其中
    S_i = (1 + sin(x * phi_i)) / 2
    C_i = (1 + cos(x * phi_i)) / 2
每個座標維度各自旋轉,產生 Lissajous 軌跡。輸出點數 = 2L + 1。
這些點之後會拿去對「共享的」grid encoder 取樣。
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class Projector(nn.Module):
    """Project coordinates into ``2L + 1`` points of interest via Lissajous curves.

    Args:
        num_frequencies: ``L`` in the paper. Number of octave frequencies.
        include_input: if True, the original ``x`` is kept as the first point
            (matches the paper's ``P_x`` which begins with ``x``). Size becomes
            ``2L + 1``; if False, size is ``2L``.
        base: base of the geometric frequency ladder (paper uses 2).

    Shape:
        input  ``x``: ``(..., d)`` with coordinates expected in ``[0, 1]``.
        output   ``P``: ``(..., num_points, d)`` where
                 ``num_points = 2L + 1`` (or ``2L`` if not ``include_input``).
        Output values lie in ``[0, 1]`` (the ``(1 + sin)/2`` mapping), so they can
        be fed straight into a grid encoder that expects normalized coords.
    """

    def __init__(
        self,
        num_frequencies: int,
        include_input: bool = True,
        base: float = 2.0,
    ) -> None:
        super().__init__()
        if num_frequencies < 0:
            raise ValueError("num_frequencies (L) must be >= 0")
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        # phi_i = base^i * pi, for i = 0 .. L-1
        freqs = base ** torch.arange(num_frequencies, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def num_points(self) -> int:
        return 2 * self.num_frequencies + (1 if self.include_input else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d)
        points = []
        if self.include_input:
            points.append(x.unsqueeze(-2))  # (..., 1, d)
        if self.num_frequencies > 0:
            # angles: (..., L, d) = x[..., None, :] * freqs[:, None]
            angles = x.unsqueeze(-2) * self.freqs.view(-1, 1)
            s = (1.0 + torch.sin(angles)) * 0.5  # (..., L, d)
            c = (1.0 + torch.cos(angles)) * 0.5  # (..., L, d)
            points.append(s)
            points.append(c)
        # concat along the "points" axis -> (..., num_points, d)
        return torch.cat(points, dim=-2)

    def extra_repr(self) -> str:
        return (
            f"num_frequencies={self.num_frequencies}, "
            f"include_input={self.include_input}, num_points={self.num_points}"
        )
