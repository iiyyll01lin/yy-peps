"""Model builders for the image application.

繁體中文:影像應用的模型工廠。提供三種 baseline 供逐週對照:
- build_plain_mlp:純 MLP(+可選 APE),示範頻譜偏差(W01)。
- build_grid:單純 grid encoder + MLP(W03 的 grid baseline)。
- build_grid_peps:Grid-PEPS(W05 主角)。
所有 builder 回傳 (model, param_count),方便畫「參數 vs PSNR」曲線(Fig.5)。
"""

from __future__ import annotations

import torch.nn as nn

from peps import (
    AbsolutePositionalEncoding,
    GridEncoder,
    LocalPositionalEncoding,
    MLP,
    NTCNEncoder,
    NTCPEPSEncoder,
    PEPS,
    Projector,
    make_aggregator,
)


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _paper_grid_resolution(resolution):
    if isinstance(resolution, int):
        return resolution
    # Paper dimensions are in coordinate order (W, H); GridEncoder stores H, W.
    return tuple(reversed(tuple(resolution)))


def _peps_model(
    encoder: nn.Module,
    *,
    num_frequencies: int,
    aggregator: str,
    out_dim: int,
    hidden_dim: int,
    num_layers: int,
    activation: str,
    output_activation,
) -> nn.Module:
    projector = Projector(num_frequencies)
    frequency_allocated = aggregator.lower() in {"pink", "brownian"}
    aggregate_kwargs = (
        {
            "num_frequencies": projector.num_frequencies,
            "include_input": projector.include_input,
            "frequency_scales": projector.frequency_scales,
        }
        if frequency_allocated
        else {}
    )
    aggregate = make_aggregator(
        aggregator,
        projector.num_points,
        encoder.feature_dim,
        **aggregate_kwargs,
    )
    decoder = MLP(
        aggregate.out_dim,
        out_dim,
        hidden_dim,
        num_layers,
        activation=activation,
        output_activation=output_activation,
    )
    return PEPS(
        projector,
        encoder,
        aggregate,
        decoder,
        selective_sampling=frequency_allocated,
    )


def build_plain_mlp(
    num_frequencies: int = 0,
    hidden_dim: int = 64,
    num_layers: int = 4,
    out_dim: int = 3,
    activation: str = "relu",
    output_activation=None,
):
    """Plain coordinate MLP, optionally fronted by APE.

    ``num_frequencies=0`` -> raw (x,y) input: strong spectral bias (W01).
    ``num_frequencies>0`` -> APE input: recovers high frequencies.
    """
    if num_frequencies > 0:
        ape = AbsolutePositionalEncoding(2, num_frequencies, include_input=True)
        mlp = MLP(
            ape.feature_dim,
            out_dim,
            hidden_dim,
            num_layers,
            activation=activation,
            output_activation=output_activation,
        )
        model = nn.Sequential(ape, mlp)
    else:
        model = MLP(
            2,
            out_dim,
            hidden_dim,
            num_layers,
            activation=activation,
            output_activation=output_activation,
        )
    return model, _count(model)


def build_grid(
    resolution: int = 128,
    feature_dim: int = 4,
    hidden_dim: int = 64,
    num_layers: int = 3,
    out_dim: int = 3,
    activation: str = "relu",
    output_activation=None,
):
    """Single grid encoder -> MLP (the grid baseline that stalls in Fig.5)."""
    enc = GridEncoder(dim=2, resolution=resolution, feature_dim=feature_dim)
    mlp = MLP(
        feature_dim,
        out_dim,
        hidden_dim,
        num_layers,
        activation=activation,
        output_activation=output_activation,
    )
    model = nn.Sequential(enc, mlp)
    return model, _count(model)


def build_grid_peps(
    resolution: int = 128,
    feature_dim: int = 4,
    num_frequencies: int = 6,
    aggregator: str = "concat",
    hidden_dim: int = 64,
    num_layers: int = 3,
    out_dim: int = 3,
    activation: str = "relu",
    output_activation=None,
):
    """Grid-PEPS: shared grid sampled at Lissajous points, then aggregated."""
    enc = GridEncoder(dim=2, resolution=resolution, feature_dim=feature_dim)
    model = _peps_model(
        enc,
        num_frequencies=num_frequencies,
        aggregator=aggregator,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        output_activation=output_activation,
    )
    return model, _count(model)


def build_lpe(
    resolution=(196, 128),
    num_frequencies: int = 4,
    hidden_dim: int = 64,
    num_layers: int = 3,
    out_dim: int = 3,
    activation: str = "leaky_relu",
    output_activation="sigmoid",
):
    """Local positional encoding baseline with Eq. 6--7 coefficients."""

    encoder = LocalPositionalEncoding(
        dim=2,
        resolution=resolution,
        num_frequencies=num_frequencies,
    )
    decoder = MLP(
        encoder.feature_dim,
        out_dim,
        hidden_dim,
        num_layers,
        activation=activation,
        output_activation=output_activation,
    )
    model = nn.Sequential(encoder, decoder)
    return model, _count(model)


def build_ntc_n(
    signal_resolution=(768, 512),
    g0_resolution=(192, 128),
    g0_feature_dim: int = 12,
    g1_resolution=(96, 64),
    g1_feature_dim: int = 20,
    hidden_dim: int = 64,
    num_layers: int = 3,
    out_dim: int = 3,
    activation: str = "leaky_relu",
    output_activation="sigmoid",
):
    """Paper NTC_N baseline (unquantized)."""

    encoder = NTCNEncoder(
        signal_resolution,
        g0_resolution=g0_resolution,
        g0_feature_dim=g0_feature_dim,
        g1_resolution=g1_resolution,
        g1_feature_dim=g1_feature_dim,
    )
    decoder = MLP(
        encoder.feature_dim,
        out_dim,
        hidden_dim,
        num_layers,
        activation=activation,
        output_activation=output_activation,
    )
    model = nn.Sequential(encoder, decoder)
    return model, _count(model)


def build_ntc_peps(
    signal_resolution=(768, 512),
    g0_resolution=(192, 128),
    g0_feature_dim: int = 12,
    g1_resolution=(96, 64),
    g1_feature_dim: int = 20,
    num_frequencies: int = 3,
    aggregator: str = "concat",
    hidden_dim: int = 64,
    num_layers: int = 3,
    out_dim: int = 3,
    activation: str = "leaky_relu",
    output_activation="sigmoid",
):
    """Replace NTC_N's learned grids with shared PEPS sampling."""

    encoder = NTCPEPSEncoder(
        signal_resolution,
        g0_resolution=g0_resolution,
        g0_feature_dim=g0_feature_dim,
        g1_resolution=g1_resolution,
        g1_feature_dim=g1_feature_dim,
        num_frequencies=num_frequencies,
        aggregator=aggregator,
    )
    decoder = MLP(
        encoder.feature_dim,
        out_dim,
        hidden_dim,
        num_layers,
        activation=activation,
        output_activation=output_activation,
    )
    model = nn.Sequential(encoder, decoder)
    return model, _count(model)


def build_paper_image(method: str, *, out_dim: int = 3, **overrides):
    """Build a Table 1 method from immutable paper defaults."""

    common = {
        "hidden_dim": 64,
        "num_layers": 3,
        "out_dim": out_dim,
        "activation": "leaky_relu",
        "output_activation": "sigmoid",
    }
    name = method.lower().replace("-", "_")
    if name == "pe":
        kwargs = {**common, "num_frequencies": 10, "hidden_dim": 300}
        kwargs.update(overrides)
        return build_plain_mlp(**kwargs)
    if name == "lpe":
        kwargs = {**common, "resolution": (196, 128), "num_frequencies": 4}
        kwargs.update(overrides)
        return build_lpe(**kwargs)
    if name == "ntc_n":
        kwargs = dict(common)
        kwargs.update(overrides)
        return build_ntc_n(**kwargs)
    if name == "grid":
        kwargs = {
            **common,
            "resolution": _paper_grid_resolution((196, 128)),
            "feature_dim": 17,
        }
        kwargs.update(overrides)
        return build_grid(**kwargs)
    if name in {"g_peps", "g_p_peps", "g_p_peps_25"}:
        kwargs = {
            **common,
            "resolution": _paper_grid_resolution((196, 128)),
            "feature_dim": 13 if name.endswith("_25") else 17,
            "num_frequencies": 3,
            "aggregator": "pink" if "_p_" in name else "concat",
        }
        kwargs.update(overrides)
        return build_grid_peps(**kwargs)
    if name in {"ntc_peps", "ntc_pinkpeps"}:
        kwargs = {
            **common,
            "num_frequencies": 3,
            "aggregator": "pink" if "pink" in name else "concat",
        }
        kwargs.update(overrides)
        return build_ntc_peps(**kwargs)
    raise ValueError(f"unknown paper image method: {method!r}")
