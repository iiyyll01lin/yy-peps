"""Lissajous projector implementing PEPS Eq. (6)-(7).

The paper indexes frequencies from one: ``phi_i = 2**i * pi`` for
``i = 1, ..., L``.  The point axis has the explicit, stable layout
``(x, S_1, ..., S_L, C_1, ..., C_L)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _as_1d_float_tensor(values, name: str) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        result = values.detach().to(device="cpu", dtype=torch.float64)
    else:
        result = torch.as_tensor(list(values), dtype=torch.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence")
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def resolve_frequency_schedule(
    num_frequencies: int | None,
    *,
    base: float = 2.0,
    frequency_exponents=None,
    frequencies=None,
) -> tuple[int, torch.Tensor, torch.Tensor | None]:
    """Resolve a PEPS frequency schedule.

    ``frequency_exponents`` defines ``phi_i = base**exponent_i * pi``.
    ``frequencies`` supplies the angular coefficients ``phi_i`` directly.
    When neither is supplied, the paper schedule uses exponents ``1..L``.
    """

    if frequency_exponents is not None and frequencies is not None:
        raise ValueError(
            "frequency_exponents and frequencies are mutually exclusive"
        )
    if not math.isfinite(base) or base <= 0:
        raise ValueError("base must be a finite positive number")

    explicit = frequency_exponents if frequency_exponents is not None else frequencies
    explicit_name = (
        "frequency_exponents"
        if frequency_exponents is not None
        else "frequencies"
    )
    values = (
        _as_1d_float_tensor(explicit, explicit_name)
        if explicit is not None
        else None
    )

    if num_frequencies is None:
        if values is None:
            raise ValueError(
                "num_frequencies is required when no explicit schedule is supplied"
            )
        num_frequencies = int(values.numel())
    if isinstance(num_frequencies, bool) or not isinstance(num_frequencies, int):
        raise TypeError("num_frequencies (L) must be an integer")
    if num_frequencies < 0:
        raise ValueError("num_frequencies (L) must be >= 0")
    if values is not None and values.numel() != num_frequencies:
        raise ValueError(
            f"{explicit_name} has {values.numel()} entries, expected "
            f"{num_frequencies}"
        )

    if frequencies is not None:
        freqs = values
        assert freqs is not None
        if (freqs <= 0).any():
            raise ValueError("frequencies must be strictly positive")
        exponents = None
    else:
        exponents = (
            values
            if values is not None
            else torch.arange(1, num_frequencies + 1, dtype=torch.float64)
        )
        freqs = torch.pow(torch.tensor(base, dtype=torch.float64), exponents)
        freqs = freqs * math.pi
        if not torch.isfinite(freqs).all() or (freqs <= 0).any():
            raise ValueError("resolved frequencies must be finite and positive")

    return num_frequencies, freqs, exponents


class Projector(nn.Module):
    """Project coordinates into ``2L + 1`` points of interest via Lissajous curves.

    Args:
        num_frequencies: ``L`` in the paper. Number of octave frequencies.
        include_input: if True, the original ``x`` is kept as the first point
            (matches the paper's ``P_x`` which begins with ``x``). Size becomes
            ``2L + 1``; if False, size is ``2L``.
        base: base of the geometric frequency ladder (paper uses 2).
        frequency_exponents: optional exponents for ``base**exponent * pi``.
            The paper default is ``(1, ..., L)``.
        frequencies: optional angular coefficients ``phi_i`` supplied directly.
            This is mutually exclusive with ``frequency_exponents``.

    Shape:
        input  ``x``: ``(..., d)`` with coordinates expected in ``[0, 1]``.
        output   ``P``: ``(..., num_points, d)`` where
                 ``num_points = 2L + 1`` (or ``2L`` if not ``include_input``).
        Output values lie in ``[0, 1]`` (the ``(1 + sin)/2`` mapping), so they can
        be fed straight into a grid encoder that expects normalized coords.
    """

    def __init__(
        self,
        num_frequencies: int | None = None,
        include_input: bool = True,
        base: float = 2.0,
        *,
        frequency_exponents=None,
        frequencies=None,
    ) -> None:
        super().__init__()
        num_frequencies, freqs, exponents = resolve_frequency_schedule(
            num_frequencies,
            base=base,
            frequency_exponents=frequency_exponents,
            frequencies=frequencies,
        )
        self.num_frequencies = num_frequencies
        self.include_input = bool(include_input)
        self.base = float(base)
        self.register_buffer("freqs", freqs, persistent=False)
        self.register_buffer("frequency_exponents", exponents, persistent=False)

    @property
    def num_points(self) -> int:
        return 2 * self.num_frequencies + (1 if self.include_input else 0)

    @property
    def point_layout(self) -> tuple[str, ...]:
        """Names for the point axis, in the exact order returned by ``forward``."""

        layout = ["input"] if self.include_input else []
        layout.extend(f"sin_{i}" for i in range(1, self.num_frequencies + 1))
        layout.extend(f"cos_{i}" for i in range(1, self.num_frequencies + 1))
        return tuple(layout)

    @property
    def frequency_scales(self) -> torch.Tensor:
        """Allocation scales used by the paper's worked Pink example.

        The example maps ``phi_i=2**i*pi`` to scale ``2**i`` (and therefore
        widths ``d/2**i``), so the normalization is relative to ``pi``.
        """

        return self.freqs / math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1:
            raise ValueError("x must have shape (..., d)")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor")

        points = []
        if self.include_input:
            points.append(x.unsqueeze(-2))  # (..., 1, d)
        if self.num_frequencies > 0:
            # angles: (..., L, d) = x[..., None, :] * freqs[:, None]
            freqs = self.freqs.to(dtype=x.dtype)
            angles = x.unsqueeze(-2) * freqs.view(-1, 1)
            s = (1.0 + torch.sin(angles)) * 0.5  # (..., L, d)
            c = (1.0 + torch.cos(angles)) * 0.5  # (..., L, d)
            points.append(s)
            points.append(c)
        if not points:
            return x.new_empty((*x.shape[:-1], 0, x.shape[-1]))
        return torch.cat(points, dim=-2)

    def extra_repr(self) -> str:
        return (
            f"num_frequencies={self.num_frequencies}, "
            f"include_input={self.include_input}, num_points={self.num_points}, "
            f"point_layout={self.point_layout}"
        )
