"""Image-only model variants used by the PEPS appendix reproduction.

These builders deliberately live outside the core PEPS modules.  The paper
describes the ablations qualitatively but does not publish their exact
parameter split or WIRE hyperparameters, so every non-canonical choice is
exposed as an argument and recorded by the experiment config.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from apps.image.data import orient_resolution_xy
from peps import GridEncoder, MLP, PEPS, Projector, make_aggregator


def _grid_storage_resolution(resolution_xy) -> tuple[int, int]:
    width, height = (int(value) for value in resolution_xy)
    return (height, width)


class FullSumAggregator(nn.Module):
    """Sum every projected-point latent into one feature vector."""

    def __init__(self, num_points: int, feature_dim: int) -> None:
        super().__init__()
        if num_points < 1 or feature_dim < 1:
            raise ValueError("num_points and feature_dim must be positive")
        self.num_points = int(num_points)
        self.feature_dim = int(feature_dim)
        self.out_dim = self.feature_dim

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3 or latents.shape[1:] != (
            self.num_points,
            self.feature_dim,
        ):
            raise ValueError(
                "latents must have shape "
                f"(N, {self.num_points}, {self.feature_dim})"
            )
        return latents.sum(dim=1)

    def forward_points(self, latents) -> torch.Tensor:
        if len(latents) != self.num_points:
            raise ValueError(f"expected {self.num_points} point latents")
        return self.forward(torch.stack(tuple(latents), dim=1))


class FrequencyPairSumAggregator(nn.Module):
    """Concatenate ``x`` and each ``S_i + C_i`` frequency pair.

    The paper only says that it summed ``l_{S_i}+l_{C_i}``.  Retaining the
    original-point latent and concatenating the frequency sums is the explicit
    interpretation frozen for this sensitivity experiment.
    """

    def __init__(self, num_frequencies: int, feature_dim: int) -> None:
        super().__init__()
        if num_frequencies < 1 or feature_dim < 1:
            raise ValueError("num_frequencies and feature_dim must be positive")
        self.num_frequencies = int(num_frequencies)
        self.feature_dim = int(feature_dim)
        self.num_points = 2 * self.num_frequencies + 1
        self.out_dim = (self.num_frequencies + 1) * self.feature_dim

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3 or latents.shape[1:] != (
            self.num_points,
            self.feature_dim,
        ):
            raise ValueError(
                "latents must have shape "
                f"(N, {self.num_points}, {self.feature_dim})"
            )
        parts = [latents[:, 0, :]]
        for frequency in range(self.num_frequencies):
            sine = latents[:, 1 + frequency, :]
            cosine = latents[:, 1 + self.num_frequencies + frequency, :]
            parts.append(sine + cosine)
        return torch.cat(parts, dim=1)

    def forward_points(self, latents) -> torch.Tensor:
        if len(latents) != self.num_points:
            raise ValueError(f"expected {self.num_points} point latents")
        return self.forward(torch.stack(tuple(latents), dim=1))


class RealGaborActivation(nn.Module):
    """Real-valued Gabor sensitivity used when exact WIRE settings are absent."""

    def __init__(self, omega: float, scale: float) -> None:
        super().__init__()
        if not math.isfinite(omega) or omega <= 0:
            raise ValueError("omega must be finite and positive")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        self.omega = float(omega)
        self.scale = float(scale)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * value) * torch.exp(
            -(self.scale * value).square()
        )


def _decoder(
    input_dim: int,
    out_dim: int,
    *,
    hidden_dim: int,
    num_layers: int,
    activation: str,
    output_activation,
    wire_omega: float | None = None,
    wire_scale: float | None = None,
) -> nn.Module:
    if activation != "wire":
        return MLP(
            input_dim,
            out_dim,
            hidden_dim,
            num_layers,
            activation=activation,
            output_activation=output_activation,
        )
    if wire_omega is None or wire_scale is None:
        raise ValueError("WIRE sensitivity requires omega and scale")
    return MLP(
        input_dim,
        out_dim,
        hidden_dim,
        num_layers,
        activation=RealGaborActivation(wire_omega, wire_scale),
        output_activation=output_activation,
    )


def build_paper_image_ablation(
    variant: str,
    *,
    signal_resolution=(768, 512),
    resolution=(196, 128),
    no_sharing_resolution=(74, 48),
    feature_dim: int = 17,
    num_frequencies: int = 3,
    hidden_dim: int = 64,
    num_layers: int = 4,
    out_dim: int = 3,
    activation: str = "gelu",
    output_activation=None,
    wire_omega: float = 20.0,
    wire_scale: float = 10.0,
):
    """Build a frozen interpretation of one image/core appendix ablation."""

    name = variant.lower().replace("-", "_")
    aliases = {
        "no_sharing_grid": "no_sharing",
        "remove_original_point": "no_original_point",
        "frequency_sum": "frequency_pair_sum",
        "sum": "full_sum",
    }
    name = aliases.get(name, name)
    signal_resolution = tuple(int(value) for value in signal_resolution)
    grid_xy = orient_resolution_xy(resolution, signal_resolution)
    storage_resolution = _grid_storage_resolution(grid_xy)

    include_input = name != "no_original_point"
    projector = Projector(
        num_frequencies=num_frequencies,
        include_input=include_input,
    )
    selective_sampling = False

    if name == "no_sharing":
        split_xy = orient_resolution_xy(
            no_sharing_resolution,
            signal_resolution,
        )
        split_storage = _grid_storage_resolution(split_xy)
        encoder = [
            GridEncoder(
                dim=2,
                resolution=split_storage,
                feature_dim=feature_dim,
            )
            for _ in range(projector.num_points)
        ]
        aggregator = make_aggregator(
            "concat",
            projector.num_points,
            feature_dim,
        )
    else:
        encoder = GridEncoder(
            dim=2,
            resolution=storage_resolution,
            feature_dim=feature_dim,
        )
        if name in {"concat", "no_original_point", "wire"}:
            aggregator = make_aggregator(
                "concat",
                projector.num_points,
                feature_dim,
            )
        elif name in {"pink", "brownian"}:
            aggregator = make_aggregator(
                name,
                projector.num_points,
                feature_dim,
                num_frequencies=projector.num_frequencies,
                include_input=projector.include_input,
                frequency_scales=projector.frequency_scales,
            )
            selective_sampling = True
        elif name == "full_sum":
            aggregator = FullSumAggregator(
                projector.num_points,
                feature_dim,
            )
        elif name == "frequency_pair_sum":
            if not projector.include_input:
                raise ValueError("frequency-pair sum requires the original point")
            aggregator = FrequencyPairSumAggregator(
                num_frequencies,
                feature_dim,
            )
        else:
            raise ValueError(f"unknown image ablation variant: {variant!r}")

    decoder_activation = "wire" if name == "wire" else activation
    decoder = _decoder(
        aggregator.out_dim,
        out_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=decoder_activation,
        output_activation=output_activation,
        wire_omega=wire_omega,
        wire_scale=wire_scale,
    )
    model = PEPS(
        projector,
        encoder,
        aggregator,
        decoder,
        selective_sampling=selective_sampling,
    )
    return model, sum(parameter.numel() for parameter in model.parameters())
