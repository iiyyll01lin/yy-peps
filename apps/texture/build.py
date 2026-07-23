"""Model builders for paper neural texture compression methods."""

from __future__ import annotations

import torch.nn as nn

from peps import (
    GridEncoder,
    LocalPositionalEncoding,
    MLP,
    NTCNEncoder,
    NTCPEPSEncoder,
    PEPS,
    Projector,
    make_aggregator,
)

OUT_CHANNELS = 9


def _count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _resolve_out_dim(
    out_dim: int | None,
    num_textures: int | None,
) -> int:
    derived = None if num_textures is None else 3 * num_textures
    if out_dim is not None and derived is not None and out_dim != derived:
        raise ValueError("out_dim must equal 3 * num_textures")
    value = out_dim if out_dim is not None else derived
    if value is None:
        value = OUT_CHANNELS
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("output dimension must be an integer")
    if value < 1:
        raise ValueError("output dimension must be positive")
    return value


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
    delta: bool,
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
        aggregate.out_dim + (2 if delta else 0),
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
        append_input_delta=delta,
        selective_sampling=frequency_allocated,
    )


def build_ntc_baseline(
    resolution: int = 256,
    feature_dim: int = 8,
    hidden_dim: int = 64,
    num_layers: int = 4,
    out_dim: int | None = None,
    num_textures: int | None = None,
    activation: str = "relu",
    output_activation=None,
):
    """Legacy single-grid teaching baseline (not the paper's NTC_N)."""

    out_dim = _resolve_out_dim(out_dim, num_textures)
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


def build_bi_grid_texture(**kwargs):
    """Paper BI-grid baseline; legacy ``build_ntc_baseline`` remains an alias."""

    return build_ntc_baseline(**kwargs)


def build_grid_peps_texture(
    resolution: int = 256,
    feature_dim: int = 8,
    num_frequencies: int = 6,
    aggregator: str = "concat",
    hidden_dim: int = 64,
    num_layers: int = 4,
    delta: bool = False,
    out_dim: int | None = None,
    num_textures: int | None = None,
    activation: str = "relu",
    output_activation=None,
):
    """``delta`` toggles the Eq. (8) input-delta (raw (u,v) concatenated to the
    aggregated vector before the decoder). It is off by default."""
    out_dim = _resolve_out_dim(out_dim, num_textures)
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
        delta=delta,
    )
    return model, _count(model)


def build_lpe_texture(
    resolution: int = 1024,
    num_frequencies: int = 4,
    hidden_dim: int = 64,
    num_layers: int = 4,
    out_dim: int | None = None,
    num_textures: int | None = None,
    activation: str = "gelu",
    output_activation=None,
):
    out_dim = _resolve_out_dim(out_dim, num_textures)
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
    signal_resolution=4096,
    g0_resolution=1024,
    g0_feature_dim: int = 12,
    g1_resolution=512,
    g1_feature_dim: int = 20,
    hidden_dim: int = 64,
    num_layers: int = 4,
    out_dim: int | None = None,
    num_textures: int | None = None,
    activation: str = "gelu",
    output_activation=None,
):
    """Paper NTC_N: G0 corner concat + G1 bilinear + tiled PE."""

    out_dim = _resolve_out_dim(out_dim, num_textures)
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


def build_ntc_peps_texture(
    signal_resolution=4096,
    g0_resolution=1024,
    g0_feature_dim: int = 12,
    g1_resolution=512,
    g1_feature_dim: int = 20,
    num_frequencies: int = 4,
    aggregator: str = "concat",
    hidden_dim: int = 64,
    num_layers: int = 4,
    out_dim: int | None = None,
    num_textures: int | None = None,
    activation: str = "gelu",
    output_activation=None,
    delta: bool = False,
):
    out_dim = _resolve_out_dim(out_dim, num_textures)
    encoder = NTCPEPSEncoder(
        signal_resolution,
        g0_resolution=g0_resolution,
        g0_feature_dim=g0_feature_dim,
        g1_resolution=g1_resolution,
        g1_feature_dim=g1_feature_dim,
        num_frequencies=num_frequencies,
        aggregator=aggregator,
        append_input=delta,
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


def build_paper_texture(
    method: str,
    *,
    out_dim: int | None = None,
    num_textures: int | None = None,
    **overrides,
):
    """Build a Table 2 method from the paper's 4K configuration."""

    signal_resolution = overrides.pop("signal_resolution", 4096)
    common = {
        "hidden_dim": 64,
        "num_layers": 4,
        "out_dim": out_dim,
        "num_textures": num_textures,
        "activation": "gelu",
        "output_activation": None,
    }
    name = method.lower().replace("-", "_")
    aliases = {
        "grid_peps": "grid_peps4f",
        "grid_pink_peps": "grid_pinkpeps4f",
        "grid_peps_25": "grid_peps4f_25",
        "grid_pink_peps_25": "grid_pinkpeps4f_25",
        "grid_pink_peps3f": "grid_pinkpeps3f",
        "grid_pink_peps4f": "grid_pinkpeps4f",
        "ntc_pink_peps": "ntc_pinkpeps",
        "ntc_pink_peps_25": "ntc_pinkpeps_25",
        "ntc_peps4f": "ntc_peps",
        "ntc_pinkpeps4f": "ntc_pinkpeps",
        "ntc_pink_peps3f": "ntc_pinkpeps3f",
        "ntc_pink_peps4f": "ntc_pinkpeps",
        "grid_pinkpeps": "grid_pinkpeps4f",
    }
    name = aliases.get(name, name)
    if name == "lpe":
        kwargs = {**common, "resolution": 1024, "num_frequencies": 4}
        kwargs.update(overrides)
        return build_lpe_texture(**kwargs)
    if name == "ntc_n":
        kwargs = {**common, "signal_resolution": signal_resolution}
        kwargs.update(overrides)
        return build_ntc_n(**kwargs)
    if name == "bi_grid":
        kwargs = {**common, "resolution": 1024, "feature_dim": 17}
        kwargs.update(overrides)
        return build_ntc_baseline(**kwargs)
    if name in {
        "grid_peps3f",
        "grid_pinkpeps3f",
        "grid_peps4f",
        "grid_pinkpeps4f",
        "grid_peps3f_25",
        "grid_pinkpeps3f_25",
        "grid_peps4f_25",
        "grid_pinkpeps4f_25",
    }:
        reduced = name.endswith("_25")
        kwargs = {
            **common,
            "resolution": 1024,
            "feature_dim": 13 if reduced else 17,
            "num_frequencies": 3 if "3f" in name else 4,
            "aggregator": "pink" if "pink" in name else "concat",
        }
        kwargs.update(overrides)
        return build_grid_peps_texture(**kwargs)
    if name in {
        "ntc_peps3f",
        "ntc_pinkpeps3f",
        "ntc_peps",
        "ntc_pinkpeps",
        "ntc_peps3f_25",
        "ntc_pinkpeps3f_25",
        "ntc_peps_25",
        "ntc_pinkpeps_25",
    }:
        reduced = name.endswith("_25")
        kwargs = {
            **common,
            "signal_resolution": signal_resolution,
            "g0_feature_dim": 9 if reduced else 12,
            "g1_feature_dim": 15 if reduced else 20,
            "num_frequencies": 3 if "3f" in name else 4,
            "aggregator": "pink" if "pink" in name else "concat",
        }
        kwargs.update(overrides)
        return build_ntc_peps_texture(**kwargs)
    raise ValueError(f"unknown paper texture method: {method!r}")
