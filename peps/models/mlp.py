"""Small MLP decoder with explicit hidden-layer semantics."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


def _make_activation(activation, negative_slope: float) -> nn.Module:
    if isinstance(activation, nn.Module):
        return copy.deepcopy(activation)
    if not isinstance(activation, str):
        raise TypeError("activation must be a string or nn.Module")

    name = activation.lower().replace("-", "_")
    if name == "relu":
        return nn.ReLU()
    if name in {"leaky_relu", "leakyrelu"}:
        return nn.LeakyReLU(negative_slope=negative_slope)
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name in {"identity", "linear", "none"}:
        return nn.Identity()
    raise ValueError(f"unknown activation: {activation!r}")


def _validate_dimension(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {relation}")
    return value


class MLP(nn.Module):
    """A decoder whose ``num_layers`` is the number of hidden layers.

    Thus the paper's ``num_layers=3`` creates three 64-neuron hidden layers
    followed by a separate output layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        activation: str | nn.Module = "relu",
        output_activation=None,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.in_dim = _validate_dimension(in_dim, "in_dim")
        self.out_dim = _validate_dimension(out_dim, "out_dim")
        self.hidden_dim = _validate_dimension(hidden_dim, "hidden_dim")
        self.num_hidden_layers = _validate_dimension(
            num_layers, "num_layers", allow_zero=True
        )
        # Compatibility for callers that inspect this attribute. Its meaning is
        # now explicit: hidden layers, not total Linear layers.
        self.num_layers = self.num_hidden_layers
        if not math.isfinite(negative_slope) or negative_slope < 0:
            raise ValueError("negative_slope must be finite and non-negative")

        layers = []
        current_dim = self.in_dim
        for _ in range(self.num_hidden_layers):
            layers.append(nn.Linear(current_dim, self.hidden_dim))
            layers.append(_make_activation(activation, negative_slope))
            current_dim = self.hidden_dim
        layers.append(nn.Linear(current_dim, self.out_dim))
        if output_activation is not None:
            layers.append(_make_activation(output_activation, negative_slope))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
