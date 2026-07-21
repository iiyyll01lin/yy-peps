"""Core unit tests — run on CPU (no GPU needed).

繁體中文:核心單元測試,純 CPU 可跑。驗證 projector 形狀、grid 取樣、聚合器維度,
以及最重要的 sanity check:Identity-encoder + concat 的 PEPS 能學到與純 APE+MLP
相同的表達能力(論文「PEPS 是 APE 泛化」宣稱)。
"""

import math
import torch

from peps import (
    Projector, GridEncoder, MLP, PEPS,
    make_aggregator, IdentityEncoder, AbsolutePositionalEncoding,
)
from peps.aggregate import (
    ConcatAggregator, PinkAggregator, BrownianAggregator, make_aggregator,
)


def test_projector_shape_and_range():
    L = 4
    proj = Projector(num_frequencies=L, include_input=True)
    assert proj.num_points == 2 * L + 1
    x = torch.rand(100, 2)
    p = proj(x)
    assert p.shape == (100, 2 * L + 1, 2)
    # sin/cos points mapped to [0,1]
    assert p.min() >= 0.0 - 1e-6
    assert p.max() <= 1.0 + 1e-6
    # first point is the raw input
    assert torch.allclose(p[:, 0, :], x)


def test_projector_no_input():
    proj = Projector(num_frequencies=3, include_input=False)
    assert proj.num_points == 6
    assert proj(torch.rand(5, 3)).shape == (5, 6, 3)


def test_grid_encoder_2d_shape():
    enc = GridEncoder(dim=2, resolution=32, feature_dim=4)
    out = enc(torch.rand(50, 2))
    assert out.shape == (50, 4)


def test_grid_encoder_3d_shape():
    enc = GridEncoder(dim=3, resolution=16, feature_dim=2)
    out = enc(torch.rand(20, 3))
    assert out.shape == (20, 2)


def test_grid_gradient_flows_to_whole_grid():
    # A shared grid sampled at many projected points should receive gradients.
    enc = GridEncoder(dim=2, resolution=16, feature_dim=3)
    coords = torch.rand(200, 2)
    out = enc(coords).sum()
    out.backward()
    assert enc.grid.grad is not None
    assert enc.grid.grad.abs().sum() > 0


def test_concat_aggregator_dim():
    agg = ConcatAggregator(num_points=9, feature_dim=4)
    assert agg.out_dim == 36
    lat = torch.randn(10, 9, 4)
    assert agg(lat).shape == (10, 36)


def test_pink_aggregator_is_smaller_than_concat():
    num_points, k = 9, 8
    concat = ConcatAggregator(num_points, k)
    pink = PinkAggregator(num_points, k)
    # Pink allocates fewer dims to high-frequency points -> smaller output.
    assert pink.out_dim < concat.out_dim
    lat = torch.randn(4, num_points, k)
    assert pink(lat).shape == (4, pink.out_dim)


def test_brownian_smaller_than_pink():
    """Brownian (alpha=2) allocates even fewer dims than Pink (alpha=1)."""
    num_points, k = 13, 8
    pink = PinkAggregator(num_points, k)
    brown = BrownianAggregator(num_points, k)
    assert brown.out_dim <= pink.out_dim < ConcatAggregator(num_points, k).out_dim
    lat = torch.randn(3, num_points, k)
    assert brown(lat).shape == (3, brown.out_dim)


def test_aggregator_factory_kinds():
    for kind, cls in [("concat", ConcatAggregator),
                      ("pink", PinkAggregator),
                      ("brownian", BrownianAggregator)]:
        agg = make_aggregator(kind, num_points=9, feature_dim=4)
        assert isinstance(agg, cls)
    import pytest
    with pytest.raises(ValueError):
        make_aggregator("nope", 9, 4)


def test_pink_gradient_reaches_whole_feature_dim():
    """Circular-shifted sub-vectors must let gradients reach every one of the k
    grid-feature channels across the point set (the whole shared grid learns)."""
    num_points, k = 9, 8
    pink = PinkAggregator(num_points, k)
    lat = torch.randn(5, num_points, k, requires_grad=True)
    pink(lat).sum().backward()
    # every feature channel receives gradient from at least one point
    per_channel = lat.grad.abs().sum(dim=(0, 1))  # (k,)
    assert (per_channel > 0).all(), per_channel


def test_peps_forward_shape():
    L = 4
    proj = Projector(L)
    enc = GridEncoder(dim=2, resolution=32, feature_dim=4)
    agg = make_aggregator("concat", proj.num_points, enc.feature_dim)
    mlp = MLP(in_dim=agg.out_dim, out_dim=3)
    model = PEPS(proj, enc, agg, mlp)
    y = model(torch.rand(64, 2))
    assert y.shape == (64, 3)


def test_identity_peps_matches_ape_capacity():
    """Sanity check: PEPS with IdentityEncoder + concat produces a feature
    vector that is an affine-equivalent of the APE features, so a linear layer
    on either reaches the same fit. We check the projected+identity features
    span the same info as APE's sin/cos by fitting a linear map between them.
    """
    torch.manual_seed(0)
    L, dim = 3, 2
    x = torch.rand(500, dim)

    # PEPS path: projector points -> identity encode -> concat
    proj = Projector(L, include_input=True)
    ident = IdentityEncoder(dim)
    pts = proj(x)                                  # (N, 2L+1, dim)
    peps_feat = pts.reshape(x.shape[0], -1)        # (N, (2L+1)*dim)

    # APE path
    ape = AbsolutePositionalEncoding(dim, L, include_input=True)
    ape_feat = ape(x)                              # (N, dim*(1+2L))

    # Both have the same dimensionality and carry the same sin/cos content
    # (PEPS uses (1+sin)/2 and (1+cos)/2 — an affine transform of APE's sin/cos).
    assert peps_feat.shape[1] == ape_feat.shape[1]

    # Fit a least-squares linear map APE_feat -> peps_feat; residual ~ 0 proves
    # they are affine-equivalent (hence equal MLP expressive power).
    A = torch.cat([ape_feat, torch.ones(x.shape[0], 1)], dim=1)
    sol, *_ = torch.linalg.lstsq(A, peps_feat)
    resid = (A @ sol - peps_feat).abs().max().item()
    assert resid < 1e-4, f"residual too large: {resid}"


def test_peps_trains_one_step():
    from peps.train import fit, TrainConfig
    L = 4
    proj = Projector(L)
    enc = GridEncoder(dim=2, resolution=16, feature_dim=2)
    agg = make_aggregator("concat", proj.num_points, enc.feature_dim)
    mlp = MLP(in_dim=agg.out_dim, out_dim=1)
    model = PEPS(proj, enc, agg, mlp)
    coords = torch.rand(1000, 2)
    targets = torch.sin(coords[:, :1] * 6.28)
    fit(model, coords, targets,
        TrainConfig(steps=5, batch_size=256, device=torch.device("cpu")))
