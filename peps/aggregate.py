"""Aggregators — combine per-point latents into one vector (paper Sec. 4).

繁體中文:聚合器。Projector 產生 2L+1 個興趣點,每個點對共享 grid 取樣得到一個
latent l_i(維度 k)。聚合器把這 2L+1 個 latent 併成一個向量餵給 MLP。

三種聚合策略(對應論文 alpha 參數):
- Concat  (alpha=0, 原始 PEPS):直接串接 -> 維度 (2L+1)*k。
- Pink    (alpha=1):依「與頻率成反比」分配 latent 維度 a_n = max(1, d/f_n),
          取循環位移子向量,讓整個 grid 都收到梯度,且只算子部分 -> 較快。
- Brownian(alpha=2):更陡的 1/f^2 分配,用來壓力測試泛化性。

Pink/Brownian 的動機:自然影像的功率譜密度 (PSD) 呈 1/f^alpha,低頻能量大。
把更多 latent 容量分配給低頻點,參數用得更省(論文的 -25% 參數結果)。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConcatAggregator(nn.Module):
    """Plain PEPS (alpha=0): concatenate all point latents.

    input  ``latents``: ``(N, num_points, k)``
    output ``vec``:      ``(N, num_points * k)``
    """

    def __init__(self, num_points: int, feature_dim: int) -> None:
        super().__init__()
        self.num_points = num_points
        self.feature_dim = feature_dim
        self.out_dim = num_points * feature_dim

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        n = latents.shape[0]
        return latents.reshape(n, -1)


class _FrequencyAllocAggregator(nn.Module):
    """Shared logic for Pink/Brownian: allocate dims inversely to frequency.

    For point index ``n`` (0 = input/DC, then frequency octaves), the allocated
    width is ``a_n = max(1, round(feature_dim / f_n**alpha))`` where ``f_n`` is a
    per-point frequency weight. We then take a circular-shifted sub-vector of the
    point's latent so gradients still reach the whole ``k``-dim grid feature,
    while the concatenated output is smaller than plain concat.
    """

    def __init__(self, num_points: int, feature_dim: int, alpha: float) -> None:
        super().__init__()
        self.num_points = num_points
        self.feature_dim = feature_dim
        self.alpha = alpha

        # Point 0 is the raw input (DC). Points 1..2L come in (sin_i, cos_i) pairs
        # sharing octave i. Frequency weight f_n grows with octave.
        widths = []
        shifts = []
        for n in range(num_points):
            if n == 0:
                octave = 0
            else:
                octave = (n - 1) // 2  # 0,0,1,1,2,2,...
            f_n = float(octave + 1)  # 1,1,1,2,2,3,3,...
            a_n = max(1, round(feature_dim / (f_n ** alpha)))
            a_n = min(a_n, feature_dim)
            widths.append(a_n)
            # circular start offset so different points read different slices
            shifts.append((n * 1) % feature_dim)
        self.widths = widths
        self.shifts = shifts
        self.out_dim = sum(widths)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # latents: (N, num_points, k)
        n = latents.shape[0]
        k = self.feature_dim
        parts = []
        for p in range(self.num_points):
            w = self.widths[p]
            s = self.shifts[p]
            idx = (torch.arange(w, device=latents.device) + s) % k
            parts.append(latents[:, p, :].index_select(1, idx))
        return torch.cat(parts, dim=1)

    def extra_repr(self) -> str:
        return (
            f"num_points={self.num_points}, feature_dim={self.feature_dim}, "
            f"alpha={self.alpha}, out_dim={self.out_dim}"
        )


class PinkAggregator(_FrequencyAllocAggregator):
    """Pink aggregator (alpha=1): dims ~ 1/f. Reproduces the -25% param story."""

    def __init__(self, num_points: int, feature_dim: int) -> None:
        super().__init__(num_points, feature_dim, alpha=1.0)


class BrownianAggregator(_FrequencyAllocAggregator):
    """Brownian aggregator (alpha=2): dims ~ 1/f^2. Stress-test generalization."""

    def __init__(self, num_points: int, feature_dim: int) -> None:
        super().__init__(num_points, feature_dim, alpha=2.0)


def make_aggregator(kind: str, num_points: int, feature_dim: int) -> nn.Module:
    """Factory: ``kind`` in {"concat", "pink", "brownian"}."""
    kind = kind.lower()
    if kind == "concat":
        return ConcatAggregator(num_points, feature_dim)
    if kind == "pink":
        return PinkAggregator(num_points, feature_dim)
    if kind == "brownian":
        return BrownianAggregator(num_points, feature_dim)
    raise ValueError(f"unknown aggregator kind: {kind!r}")
