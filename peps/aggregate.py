"""PEPS concatenation and frequency-allocation aggregators.

The point axis follows the paper's exact contract
``(x, S_1, ..., S_L, C_1, ..., C_L)``.  With the paper schedule, frequency
``i`` receives ``a_i = max(1, floor(d / 2**i))`` channels.  Cumulative
allocations ``G_i`` drive forward cosine slices and reverse sine slices.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _validate_size(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {relation}")
    return value


def _as_schedule(values, expected: int, name: str, *, positive: bool) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        result = values.detach().to(device="cpu", dtype=torch.float64)
    else:
        result = torch.as_tensor(list(values), dtype=torch.float64)
    if result.ndim != 1 or result.numel() != expected:
        raise ValueError(
            f"{name} must have {expected} entries, got shape "
            f"{tuple(result.shape)}"
        )
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    if positive and (result <= 0).any():
        raise ValueError(f"{name} must be strictly positive")
    return result


def _validate_selected(latents, widths: list[int]) -> list[torch.Tensor]:
    if len(latents) != len(widths):
        raise ValueError(
            f"expected {len(widths)} point latents, got {len(latents)}"
        )
    checked = []
    batch_size = None
    for point, (latent, width) in enumerate(zip(latents, widths)):
        if not isinstance(latent, torch.Tensor) or latent.ndim != 2:
            raise ValueError(
                f"latent for point {point} must have shape (N, {width})"
            )
        if latent.shape[1] != width:
            raise ValueError(
                f"latent for point {point} has {latent.shape[1]} channels, "
                f"expected {width}"
            )
        if batch_size is None:
            batch_size = latent.shape[0]
        elif latent.shape[0] != batch_size:
            raise ValueError("all point latents must have the same batch size")
        checked.append(latent)
    return checked


class ConcatAggregator(nn.Module):
    """Plain PEPS (alpha=0): concatenate all point latents.

    input  ``latents``: ``(N, num_points, k)``
    output ``vec``:      ``(N, num_points * k)``
    """

    def __init__(self, num_points: int, feature_dim: int) -> None:
        super().__init__()
        self.num_points = _validate_size(num_points, "num_points", allow_zero=True)
        self.feature_dim = _validate_size(feature_dim, "feature_dim")
        all_channels = tuple(range(self.feature_dim))
        self.point_channel_indices = tuple(
            all_channels for _ in range(self.num_points)
        )
        self.widths = [self.feature_dim] * self.num_points
        self.out_dim = self.num_points * self.feature_dim

    def channel_indices_for_point(
        self, point: int, *, device=None
    ) -> torch.Tensor:
        if point < 0 or point >= self.num_points:
            raise IndexError(f"point index {point} is out of range")
        return torch.tensor(
            self.point_channel_indices[point],
            dtype=torch.long,
            device=device,
        )

    def aggregate_selected(self, latents) -> torch.Tensor:
        checked = _validate_selected(latents, self.widths)
        if not checked:
            raise ValueError("cannot infer batch size from an empty latent list")
        return torch.cat(checked, dim=1)

    def forward_points(self, latents) -> torch.Tensor:
        return self.aggregate_selected(latents)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3:
            raise ValueError("latents must have shape (N, num_points, feature_dim)")
        if latents.shape[1:] != (self.num_points, self.feature_dim):
            raise ValueError(
                "latents have point/channel shape "
                f"{tuple(latents.shape[1:])}, expected "
                f"({self.num_points}, {self.feature_dim})"
            )
        return latents.reshape(latents.shape[0], self.out_dim)


class _FrequencyAllocAggregator(nn.Module):
    """Paper Algorithm 1 with configurable ``1/f**alpha`` allocation."""

    def __init__(
        self,
        num_points: int,
        feature_dim: int,
        alpha: float,
        min_width: int = 1,
        *,
        num_frequencies: int | None = None,
        include_input: bool | None = None,
        base: float = 2.0,
        frequency_exponents=None,
        frequencies=None,
        frequency_scales=None,
    ) -> None:
        super().__init__()
        self.num_points = _validate_size(num_points, "num_points", allow_zero=True)
        self.feature_dim = _validate_size(feature_dim, "feature_dim")
        self.min_width = _validate_size(min_width, "min_width")
        if self.min_width > self.feature_dim:
            raise ValueError("min_width cannot exceed feature_dim")
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be a finite non-negative number")
        if not math.isfinite(base) or base <= 0:
            raise ValueError("base must be a finite positive number")
        self.alpha = float(alpha)
        self.base = float(base)

        if include_input is None:
            include_input = bool(self.num_points % 2)
        self.include_input = bool(include_input)
        remaining = self.num_points - int(self.include_input)
        if remaining < 0 or remaining % 2:
            raise ValueError(
                "num_points is incompatible with layout "
                "(input?, S_1..S_L, C_1..C_L)"
            )
        inferred_frequencies = remaining // 2
        if num_frequencies is None:
            num_frequencies = inferred_frequencies
        else:
            _validate_size(
                num_frequencies, "num_frequencies", allow_zero=True
            )
            if num_frequencies != inferred_frequencies:
                raise ValueError(
                    f"num_points={self.num_points} and include_input="
                    f"{self.include_input} imply L={inferred_frequencies}, "
                    f"not {num_frequencies}"
                )
        self.num_frequencies = num_frequencies

        explicit_schedules = sum(
            value is not None
            for value in (frequency_exponents, frequencies, frequency_scales)
        )
        if explicit_schedules > 1:
            raise ValueError(
                "frequency_exponents, frequencies, and frequency_scales are "
                "mutually exclusive"
            )
        if frequency_scales is not None:
            scales = _as_schedule(
                frequency_scales,
                self.num_frequencies,
                "frequency_scales",
                positive=True,
            )
        elif frequencies is not None:
            angular = _as_schedule(
                frequencies,
                self.num_frequencies,
                "frequencies",
                positive=True,
            )
            # The paper's worked example maps phi_i=2**i*pi to allocation
            # scale 2**i, yielding widths d/2**i.
            scales = angular / math.pi
        else:
            exponents = (
                torch.arange(
                    1, self.num_frequencies + 1, dtype=torch.float64
                )
                if frequency_exponents is None
                else _as_schedule(
                    frequency_exponents,
                    self.num_frequencies,
                    "frequency_exponents",
                    positive=False,
                )
            )
            scales = torch.pow(
                torch.tensor(self.base, dtype=torch.float64), exponents
            )
        if not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("resolved frequency scales must be finite and positive")
        self.register_buffer("frequency_scales", scales, persistent=False)

        self.frequency_widths = [
            min(
                self.feature_dim,
                max(
                    self.min_width,
                    math.floor(
                        self.feature_dim / (float(scale) ** self.alpha)
                    ),
                ),
            )
            for scale in scales
        ]
        cumulative = [0]
        for width in self.frequency_widths:
            cumulative.append(cumulative[-1] + width)
        self.cumulative_allocations = tuple(cumulative)

        # Paper notation:
        #   l^{S_i}_{-G_i:-G_{i-1}} and l^{C_i}_{G_{i-1}:G_i}.
        # The point list itself remains grouped as x, all S_i, then all C_i.
        sin_indices = [
            tuple(
                index % self.feature_dim
                for index in range(-cumulative[i], -cumulative[i - 1])
            )
            for i in range(1, self.num_frequencies + 1)
        ]
        cos_indices = [
            tuple(
                index % self.feature_dim
                for index in range(cumulative[i - 1], cumulative[i])
            )
            for i in range(1, self.num_frequencies + 1)
        ]
        point_indices = []
        if self.include_input:
            point_indices.append(tuple(range(self.feature_dim)))
        point_indices.extend(sin_indices)
        point_indices.extend(cos_indices)
        self.point_channel_indices = tuple(point_indices)
        self.widths = [len(indices) for indices in self.point_channel_indices]
        self.shifts = [
            indices[0] if indices else 0 for indices in self.point_channel_indices
        ]
        self.out_dim = sum(self.widths)

    @property
    def point_layout(self) -> tuple[str, ...]:
        layout = ["input"] if self.include_input else []
        layout.extend(f"sin_{i}" for i in range(1, self.num_frequencies + 1))
        layout.extend(f"cos_{i}" for i in range(1, self.num_frequencies + 1))
        return tuple(layout)

    def channel_indices_for_point(
        self, point: int, *, device=None
    ) -> torch.Tensor:
        if point < 0 or point >= self.num_points:
            raise IndexError(f"point index {point} is out of range")
        return torch.tensor(
            self.point_channel_indices[point],
            dtype=torch.long,
            device=device,
        )

    def aggregate_selected(self, latents) -> torch.Tensor:
        checked = _validate_selected(latents, self.widths)
        if not checked:
            raise ValueError("cannot infer batch size from an empty latent list")
        return torch.cat(checked, dim=1)

    def forward_points(self, latents) -> torch.Tensor:
        if len(latents) != self.num_points:
            raise ValueError(
                f"expected {self.num_points} point latents, got {len(latents)}"
            )
        selected = []
        for point, latent in enumerate(latents):
            if latent.ndim != 2 or latent.shape[1] != self.feature_dim:
                raise ValueError(
                    f"latent for point {point} must have shape "
                    f"(N, {self.feature_dim})"
                )
            indices = self.channel_indices_for_point(
                point, device=latent.device
            )
            selected.append(latent.index_select(1, indices))
        return self.aggregate_selected(selected)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3:
            raise ValueError("latents must have shape (N, num_points, feature_dim)")
        if latents.shape[1:] != (self.num_points, self.feature_dim):
            raise ValueError(
                "latents have point/channel shape "
                f"{tuple(latents.shape[1:])}, expected "
                f"({self.num_points}, {self.feature_dim})"
            )
        parts = []
        for point in range(self.num_points):
            indices = self.channel_indices_for_point(
                point, device=latents.device
            )
            parts.append(latents[:, point, :].index_select(1, indices))
        if not parts:
            return latents.new_empty((latents.shape[0], 0))
        return torch.cat(parts, dim=1)

    def extra_repr(self) -> str:
        return (
            f"num_points={self.num_points}, feature_dim={self.feature_dim}, "
            f"alpha={self.alpha}, out_dim={self.out_dim}, "
            f"point_layout={self.point_layout}"
        )


class PinkAggregator(_FrequencyAllocAggregator):
    """Pink aggregator (``alpha=1``)."""

    def __init__(
        self, num_points: int, feature_dim: int, min_width: int = 1, **kwargs
    ) -> None:
        super().__init__(
            num_points,
            feature_dim,
            alpha=1.0,
            min_width=min_width,
            **kwargs,
        )


class BrownianAggregator(_FrequencyAllocAggregator):
    """Brownian aggregator (``alpha=2``)."""

    def __init__(
        self, num_points: int, feature_dim: int, min_width: int = 1, **kwargs
    ) -> None:
        super().__init__(
            num_points,
            feature_dim,
            alpha=2.0,
            min_width=min_width,
            **kwargs,
        )


def make_aggregator(
    kind: str, num_points: int, feature_dim: int, **kwargs
) -> nn.Module:
    """Factory: ``kind`` in {"concat", "pink", "brownian"}."""
    kind = kind.lower()
    if kind == "concat":
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                f"concat aggregator does not use schedule arguments: {unexpected}"
            )
        return ConcatAggregator(num_points, feature_dim)
    if kind == "pink":
        return PinkAggregator(num_points, feature_dim, **kwargs)
    if kind == "brownian":
        return BrownianAggregator(num_points, feature_dim, **kwargs)
    raise ValueError(f"unknown aggregator kind: {kind!r}")
