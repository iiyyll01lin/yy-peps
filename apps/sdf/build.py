"""Model builders for SDF fitting (3D).

繁體中文:SDF 擬合的模型工廠(3D)。四種 encoder + 其 PEPS 版本,對應論文
Table 3 的列:TI-grid(三線性 dense grid)、multi-res、hash,以及各自的 PEPS。
輸出為 1 維有號距離。
"""

from __future__ import annotations

import torch.nn as nn

from peps import (
    AbsolutePositionalEncoding,
    GridEncoder,
    HashGridEncoder,
    LocalPositionalEncoding,
    MLP,
    MultiResGridEncoder,
    PEPS,
    Projector,
    make_aggregator,
)


def _count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _decoder(
    in_dim: int,
    *,
    hidden_dim: int,
    num_layers: int,
    activation: str,
    output_activation,
) -> MLP:
    return MLP(
        in_dim,
        1,
        hidden_dim,
        num_layers,
        activation=activation,
        output_activation=output_activation,
    )


def build_sdf_grid(
    resolution: int = 64,
    feature_dim: int = 4,
    hidden_dim: int = 64,
    num_layers: int = 4,
    activation: str = "relu",
    output_activation=None,
):
    enc = GridEncoder(dim=3, resolution=resolution, feature_dim=feature_dim)
    mlp = _decoder(
        feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        output_activation=output_activation,
    )
    model = nn.Sequential(enc, mlp)
    return model, _count(model)


def build_sdf_multires(
    base_resolution: int = 16,
    n_levels: int = 4,
    feature_dim: int = 2,
    hidden_dim: int = 64,
    num_layers: int = 4,
    resolutions=None,
    activation: str = "relu",
    output_activation=None,
):
    enc = MultiResGridEncoder(
        dim=3,
        base_resolution=base_resolution,
        n_levels=n_levels,
        feature_dim=feature_dim,
        resolutions=resolutions,
    )
    mlp = _decoder(
        enc.feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        output_activation=output_activation,
    )
    model = nn.Sequential(enc, mlp)
    return model, _count(model)


def build_sdf_hash(
    n_levels: int = 8,
    feature_dim: int = 2,
    log2_hashmap_size: int = 18,
    hidden_dim: int = 64,
    num_layers: int = 4,
    base_resolution: int = 16,
    per_level_scale: float = 1.5,
    resolutions=None,
    activation: str = "relu",
    output_activation=None,
):
    enc = HashGridEncoder(
        dim=3,
        n_levels=n_levels,
        feature_dim=feature_dim,
        base_resolution=base_resolution,
        per_level_scale=per_level_scale,
        log2_hashmap_size=log2_hashmap_size,
        resolutions=resolutions,
    )
    mlp = _decoder(
        enc.feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        output_activation=output_activation,
    )
    model = nn.Sequential(enc, mlp)
    return model, _count(model)


def build_sdf_lpe(
    resolution: int = 32,
    num_frequencies: int = 3,
    hidden_dim: int = 64,
    num_layers: int = 4,
    activation: str = "silu",
    output_activation=None,
):
    encoder = LocalPositionalEncoding(
        dim=3,
        resolution=resolution,
        num_frequencies=num_frequencies,
    )
    decoder = _decoder(
        encoder.feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        output_activation=output_activation,
    )
    model = nn.Sequential(encoder, decoder)
    return model, _count(model)


def _make_encoder(kind: str, kwargs) -> nn.Module:
    if kind == "grid":
        return GridEncoder(
            dim=3,
            resolution=kwargs.get("resolution", 64),
            feature_dim=kwargs.get("feature_dim", 4),
        )
    if kind in {"multires", "multigrid"}:
        return MultiResGridEncoder(
            dim=3,
            base_resolution=kwargs.get("base_resolution", 16),
            n_levels=kwargs.get("n_levels", 4),
            feature_dim=kwargs.get("feature_dim", 2),
            resolutions=kwargs.get("resolutions"),
        )
    if kind in {"hash", "single_hash", "multihash"}:
        return HashGridEncoder(
            dim=3,
            n_levels=kwargs.get("n_levels", 8),
            feature_dim=kwargs.get("feature_dim", 2),
            base_resolution=kwargs.get("base_resolution", 16),
            per_level_scale=kwargs.get("per_level_scale", 1.5),
            log2_hashmap_size=kwargs.get("log2_hashmap_size", 18),
            resolutions=kwargs.get("resolutions"),
        )
    raise ValueError(f"unknown encoder {kind!r}")


def build_sdf_peps(
    encoder: str = "grid",
    num_frequencies: int = 6,
    aggregator: str = "concat",
    hidden_dim: int = 64,
    num_layers: int = 4,
    delta: bool = False,
    activation: str = "relu",
    output_activation=None,
    **enc_kwargs,
):
    """PEPS-wrapped SDF encoder.

    ``delta`` toggles the Eq. (8) input-delta: the raw ``(x, y, z)`` query coords
    are concatenated to the aggregated vector before the decoder. It is off by
    default.
    """
    proj = Projector(num_frequencies)
    enc = _make_encoder(encoder, enc_kwargs)
    k = enc.feature_dim
    frequency_allocated = aggregator.lower() in {"pink", "brownian"}
    agg_kwargs = (
        {
            "num_frequencies": proj.num_frequencies,
            "include_input": proj.include_input,
            "frequency_scales": proj.frequency_scales,
        }
        if frequency_allocated
        else {}
    )
    agg = make_aggregator(aggregator, proj.num_points, k, **agg_kwargs)
    delta_dim = 3 if delta else 0
    mlp = _decoder(
        agg.out_dim + delta_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        output_activation=output_activation,
    )
    model = PEPS(
        proj,
        enc,
        agg,
        mlp,
        append_input_delta=delta,
        selective_sampling=frequency_allocated,
    )
    return model, _count(model)


def build_paper_sdf(
    method: str,
    *,
    encoder_parameter_multiplier: int = 1,
    **overrides,
):
    """Build a paper Table 3 or Table 4 SDF method.

    ``encoder_parameter_multiplier=8`` implements the actual second row of
    Table 4: every spatial resolution is doubled and hash-table caps grow by
    three bits. In 3D this multiplies every encoder budget by exactly eight.
    PE has no learned encoder and is intentionally unchanged between rows.
    """

    if encoder_parameter_multiplier not in {1, 8}:
        raise ValueError("encoder_parameter_multiplier must be 1 or 8")
    spatial_scale = 2 if encoder_parameter_multiplier == 8 else 1
    hash_log_growth = 3 if encoder_parameter_multiplier == 8 else 0

    common = {
        "hidden_dim": 64,
        "num_layers": 4,
        "activation": "silu",
        "output_activation": None,
    }
    name = method.lower().replace("-", "_")
    aliases = {
        "ti_grid": "grid",
        "multi_grid": "m_grid",
        "multi_grid_peps": "m_peps",
        "multi_hash": "m_hash",
        "multi_hash_peps": "m_hashpeps",
        "hashpeps": "hash_peps",
        "gridpeps": "grid_peps",
        "mpeps": "m_peps",
        "m_hash_peps": "m_hashpeps",
        "mhashpeps": "m_hashpeps",
    }
    name = aliases.get(name, name)
    if name == "pe":
        kwargs = {**common, "num_frequencies": 10}
        kwargs.update(overrides)
        encoding = AbsolutePositionalEncoding(3, kwargs.pop("num_frequencies"))
        decoder = _decoder(encoding.feature_dim, **kwargs)
        model = nn.Sequential(encoding, decoder)
        return model, _count(model)
    if name == "lpe":
        kwargs = {
            **common,
            "resolution": 32 * spatial_scale,
            "num_frequencies": 3,
        }
        kwargs.update(overrides)
        return build_sdf_lpe(**kwargs)
    if name == "grid":
        kwargs = {
            **common,
            "resolution": 32 * spatial_scale,
            "feature_dim": 18,
        }
        kwargs.update(overrides)
        return build_sdf_grid(**kwargs)
    if name == "hash":
        kwargs = {
            **common,
            "n_levels": 1,
            "resolutions": (64 * spatial_scale,),
            "feature_dim": 18,
            "log2_hashmap_size": 15 + hash_log_growth,
        }
        kwargs.update(overrides)
        return build_sdf_hash(**kwargs)
    if name == "m_grid":
        kwargs = {
            **common,
            "resolutions": tuple(
                resolution * spatial_scale for resolution in (16, 32, 64)
            ),
            "feature_dim": 2,
        }
        kwargs.update(overrides)
        return build_sdf_multires(**kwargs)
    if name == "m_hash":
        kwargs = {
            **common,
            "resolutions": tuple(
                resolution * spatial_scale
                for resolution in (16, 32, 64, 128)
            ),
            "feature_dim": 2,
            "log2_hashmap_size": 17 + hash_log_growth,
        }
        kwargs.update(overrides)
        return build_sdf_hash(**kwargs)
    peps_encoders = {
        "grid_peps": (
            "grid",
            {"resolution": 32 * spatial_scale, "feature_dim": 18},
        ),
        "hash_peps": (
            "single_hash",
            {
                "n_levels": 1,
                "resolutions": (64 * spatial_scale,),
                "feature_dim": 18,
                "log2_hashmap_size": 15 + hash_log_growth,
            },
        ),
        "m_peps": (
            "multigrid",
            {
                "resolutions": tuple(
                    resolution * spatial_scale
                    for resolution in (16, 32, 64)
                ),
                "feature_dim": 2,
            },
        ),
        "m_hashpeps": (
            "multihash",
            {
                "resolutions": tuple(
                    resolution * spatial_scale
                    for resolution in (16, 32, 64, 128)
                ),
                "feature_dim": 2,
                "log2_hashmap_size": 17 + hash_log_growth,
            },
        ),
    }
    if name in peps_encoders:
        encoder, encoder_kwargs = peps_encoders[name]
        kwargs = {
            **common,
            "encoder": encoder,
            "num_frequencies": 3,
            "aggregator": "concat",
            **encoder_kwargs,
        }
        kwargs.update(overrides)
        return build_sdf_peps(**kwargs)
    raise ValueError(f"unknown paper SDF method: {method!r}")
