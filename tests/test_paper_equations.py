"""Independent semantic oracles for the PEPS paper equations."""

import math

import pytest
import torch
import torch.nn as nn

from peps import (
    AbsolutePositionalEncoding,
    BrownianAggregator,
    ConcatAggregator,
    GridEncoder,
    IdentityEncoder,
    MLP,
    PEPS,
    PinkAggregator,
    Projector,
)


def _paper_frequencies(count: int, *, dtype=torch.float64) -> torch.Tensor:
    return torch.tensor(
        [2**index * math.pi for index in range(1, count + 1)],
        dtype=dtype,
    )


def _manual_points(x: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    sine = [
        (1.0 + torch.sin(x * frequency)) / 2.0
        for frequency in frequencies
    ]
    cosine = [
        (1.0 + torch.cos(x * frequency)) / 2.0
        for frequency in frequencies
    ]
    return torch.stack([x, *sine, *cosine], dim=1)


def _manual_ape(x: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    parts = [x]
    parts.extend(torch.sin(x * frequency) for frequency in frequencies)
    parts.extend(torch.cos(x * frequency) for frequency in frequencies)
    return torch.cat(parts, dim=-1)


def test_eq_1_2_and_6_7_use_paper_frequency_indexing_and_layout():
    x = torch.tensor([[0.125, 0.375], [0.2, 0.7]], dtype=torch.float64)
    frequencies = _paper_frequencies(3)

    projector = Projector(3).double()
    assert projector.point_layout == (
        "input",
        "sin_1",
        "sin_2",
        "sin_3",
        "cos_1",
        "cos_2",
        "cos_3",
    )
    assert torch.allclose(projector.freqs, frequencies)
    assert torch.equal(
        projector.frequency_scales,
        torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64),
    )
    assert torch.allclose(projector(x), _manual_points(x, frequencies))

    ape = AbsolutePositionalEncoding(2, 3).double()
    assert torch.allclose(ape.freqs, frequencies)
    assert torch.allclose(ape(x), _manual_ape(x, frequencies))


def test_eq_3_4_translation_and_eq_5_rotation_identity():
    x = torch.tensor([[0.173]], dtype=torch.float64)
    offset = torch.tensor([[0.217]], dtype=torch.float64)
    phi = 6.0 * math.pi
    encoding = AbsolutePositionalEncoding(
        1, frequencies=[phi], include_input=False
    ).double()

    sine, cosine = encoding(x).unbind(dim=1)
    moved_sine, moved_cosine = encoding(x + offset).unbind(dim=1)
    angle = offset.squeeze(1) * phi

    expected_sine = sine * torch.cos(angle) + cosine * torch.sin(angle)
    expected_cosine = cosine * torch.cos(angle) - sine * torch.sin(angle)
    assert torch.allclose(moved_sine, expected_sine, atol=1e-12)
    assert torch.allclose(moved_cosine, expected_cosine, atol=1e-12)

    rotation = torch.stack(
        [
            torch.stack([torch.cos(angle), -torch.sin(angle)], dim=-1),
            torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1),
        ],
        dim=1,
    )
    original_cos_sin = torch.stack([cosine, sine], dim=-1).unsqueeze(-1)
    rotated = torch.bmm(rotation, original_cos_sin).squeeze(-1)
    assert torch.allclose(
        rotated,
        torch.stack([moved_cosine, moved_sine], dim=-1),
        atol=1e-12,
    )


def test_explicit_frequency_coefficients_and_exponents():
    x = torch.tensor([[0.25]], dtype=torch.float64)
    direct = Projector(frequencies=[math.pi, 3 * math.pi]).double()
    exponent = Projector(
        2, base=3.0, frequency_exponents=[0.0, 2.0]
    ).double()

    assert direct.num_frequencies == 2
    assert torch.allclose(
        direct.freqs, torch.tensor([math.pi, 3 * math.pi], dtype=torch.float64)
    )
    assert torch.allclose(
        exponent.freqs,
        torch.tensor([math.pi, 9 * math.pi], dtype=torch.float64),
    )
    assert torch.allclose(
        direct(x),
        _manual_points(x, direct.freqs),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        Projector(
            1,
            frequency_exponents=[1],
            frequencies=[2 * math.pi],
        )
    with pytest.raises(ValueError, match="expected 2"):
        Projector(2, frequencies=[2 * math.pi])


def test_projector_zero_frequency_border_and_invalid_shape_contracts():
    border = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    projected = Projector(2).double()(border)
    assert torch.allclose(projected[:, 0], border)
    assert torch.allclose(
        projected[:, 1:3], torch.full((1, 2, 2), 0.5, dtype=torch.float64)
    )
    assert torch.allclose(
        projected[:, 3:5], torch.ones((1, 2, 2), dtype=torch.float64)
    )

    x = torch.rand(4, 2)
    assert torch.equal(Projector(0)(x), x.unsqueeze(1))
    assert Projector(0, include_input=False)(x).shape == (4, 0, 2)
    assert AbsolutePositionalEncoding(
        2, 0, include_input=False
    )(x).shape == (4, 0)

    with pytest.raises(TypeError, match="floating-point"):
        Projector(1)(torch.ones(2, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="num_frequencies is required"):
        Projector()


def test_pink_paper_example_has_22_values_and_exact_circular_slices():
    aggregator = PinkAggregator(num_points=7, feature_dim=8)
    assert aggregator.point_layout == (
        "input",
        "sin_1",
        "sin_2",
        "sin_3",
        "cos_1",
        "cos_2",
        "cos_3",
    )
    assert aggregator.frequency_widths == [4, 2, 1]
    assert aggregator.widths == [8, 4, 2, 1, 4, 2, 1]
    assert aggregator.out_dim == 22
    assert aggregator.point_channel_indices == (
        (0, 1, 2, 3, 4, 5, 6, 7),
        (4, 5, 6, 7),
        (2, 3),
        (1,),
        (0, 1, 2, 3),
        (4, 5),
        (6,),
    )

    latents = torch.arange(7 * 8, dtype=torch.float32).reshape(1, 7, 8)
    expected = torch.cat(
        [
            latents[:, point, list(indices)]
            for point, indices in enumerate(
                aggregator.point_channel_indices
            )
        ],
        dim=1,
    )
    assert torch.equal(aggregator(latents), expected)


def test_brownian_uses_floor_and_squared_frequency():
    aggregator = BrownianAggregator(num_points=7, feature_dim=8)
    assert aggregator.frequency_widths == [2, 1, 1]
    assert aggregator.widths == [8, 2, 1, 1, 2, 1, 1]
    assert aggregator.out_dim == 16

    custom = PinkAggregator(
        num_points=5,
        feature_dim=10,
        frequency_scales=[1.0, 3.0],
    )
    assert custom.frequency_widths == [10, 3]
    assert custom.out_dim == 36

    from_angular_coefficients = PinkAggregator(
        num_points=5,
        feature_dim=8,
        frequencies=[2 * math.pi, 4 * math.pi],
    )
    assert from_angular_coefficients.frequency_widths == [4, 2]


def test_circular_slices_wrap_after_cumulative_allocation_exceeds_width():
    aggregator = PinkAggregator(num_points=9, feature_dim=3)
    assert aggregator.frequency_widths == [1, 1, 1, 1]
    assert aggregator.point_channel_indices == (
        (0, 1, 2),
        (2,),
        (1,),
        (0,),
        (2,),
        (0,),
        (1,),
        (2,),
        (0,),
    )


def test_identity_encoder_peps_is_exact_affine_ape_equivalent():
    x = torch.tensor(
        [[0.125, 0.375], [0.2, 0.7], [0.9, 0.1]],
        dtype=torch.float64,
    )
    projector = Projector(3).double()
    aggregator = ConcatAggregator(projector.num_points, feature_dim=2)
    peps = PEPS(
        projector,
        IdentityEncoder(2),
        aggregator,
        nn.Identity(),
    ).double()

    normalized = peps(x)
    affine_ape = normalized.clone()
    affine_ape[:, 2:] = affine_ape[:, 2:] * 2.0 - 1.0
    expected = _manual_ape(x, _paper_frequencies(3))
    assert torch.allclose(affine_ape, expected, atol=1e-12)
    assert torch.allclose(
        AbsolutePositionalEncoding(2, 3).double()(x),
        expected,
        atol=1e-12,
    )


def test_shared_and_per_point_encoders_all_receive_gradients():
    torch.manual_seed(4)
    projector = Projector(1)
    aggregator = ConcatAggregator(projector.num_points, feature_dim=3)

    shared = nn.Linear(2, 3, bias=False)
    shared_decoder = nn.Linear(aggregator.out_dim, 1, bias=False)
    nn.init.ones_(shared_decoder.weight)
    shared_model = PEPS(projector, shared, aggregator, shared_decoder)
    shared_model(torch.rand(8, 2)).sum().backward()
    assert shared.weight.grad is not None
    assert shared.weight.grad.abs().sum() > 0

    encoders = [nn.Linear(2, 3, bias=False) for _ in range(3)]
    separate_decoder = nn.Linear(aggregator.out_dim, 1, bias=False)
    nn.init.ones_(separate_decoder.weight)
    separate_model = PEPS(
        Projector(1),
        encoders,
        ConcatAggregator(3, 3),
        separate_decoder,
    )
    separate_model(torch.rand(8, 2)).sum().backward()
    assert all(
        encoder.weight.grad is not None
        and encoder.weight.grad.abs().sum() > 0
        for encoder in encoders
    )

    with pytest.raises(ValueError, match="per-point encoders"):
        PEPS(Projector(1), encoders[:2], aggregator, nn.Identity())


def test_eq_8_wrapper_appends_input_and_explicit_delta():
    x = torch.tensor([[0.2, 0.7], [0.4, 0.1]])
    delta = torch.tensor([[3.0], [5.0]])
    model = PEPS(
        Projector(0),
        IdentityEncoder(2),
        ConcatAggregator(1, 2),
        nn.Identity(),
        append_input_delta=True,
    )
    assert torch.equal(model(x, delta=delta), torch.cat([x, x, delta], dim=1))

    with pytest.raises(ValueError, match="delta must have shape"):
        model(x, delta=torch.ones(3, 1))
    with pytest.raises(ValueError, match="x must have shape"):
        model(x.unsqueeze(0))


class _SelectiveProbe(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.requests = []

    def forward(self, coords):
        raise AssertionError("full encoder path should not run")

    def sample_channels(self, coords, channel_indices):
        request = tuple(int(index) for index in channel_indices.cpu())
        self.requests.append(request)
        values = channel_indices.to(dtype=coords.dtype)
        return values.unsqueeze(0).expand(coords.shape[0], -1)


def test_selective_wrapper_requests_only_paper_circular_slices():
    projector = Projector(3)
    aggregator = PinkAggregator(projector.num_points, feature_dim=8)
    encoder = _SelectiveProbe(feature_dim=8)
    model = PEPS(
        projector,
        encoder,
        aggregator,
        nn.Identity(),
        selective_sampling=True,
    )
    output = model(torch.rand(2, 2))

    assert tuple(encoder.requests) == aggregator.point_channel_indices
    expected_row = torch.tensor(
        [
            channel
            for indices in aggregator.point_channel_indices
            for channel in indices
        ],
        dtype=output.dtype,
    )
    assert torch.equal(output, expected_row.unsqueeze(0).expand(2, -1))


def test_grid_channel_selective_sampling_matches_full_aggregation_at_borders():
    torch.manual_seed(2)
    projector = Projector(2)
    aggregator = PinkAggregator(projector.num_points, feature_dim=8)
    encoder = GridEncoder(dim=2, resolution=4, feature_dim=8)
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.25, 0.75]])

    points = projector(x)
    full_latents = torch.stack(
        [encoder(points[:, point]) for point in range(projector.num_points)],
        dim=1,
    )
    expected = aggregator(full_latents)
    selective = PEPS(
        projector,
        encoder,
        aggregator,
        nn.Identity(),
        selective_sampling=True,
    )
    assert torch.allclose(selective(x), expected)


def test_mlp_num_layers_counts_hidden_layers_and_output_activation():
    model = MLP(
        in_dim=5,
        out_dim=2,
        hidden_dim=7,
        num_layers=3,
        activation="leaky_relu",
        output_activation="sigmoid",
    )
    linear_layers = [
        module for module in model.net if isinstance(module, nn.Linear)
    ]
    hidden_activations = [
        module for module in model.net if isinstance(module, nn.LeakyReLU)
    ]
    assert model.num_hidden_layers == 3
    assert len(linear_layers) == 4
    assert [layer.in_features for layer in linear_layers] == [5, 7, 7, 7]
    assert [layer.out_features for layer in linear_layers] == [7, 7, 7, 2]
    assert len(hidden_activations) == 3
    output = model(torch.randn(6, 5))
    assert ((0.0 <= output) & (output <= 1.0)).all()

    assert len(
        [
            module
            for module in MLP(2, 1, num_layers=0).net
            if isinstance(module, nn.Linear)
        ]
    ) == 1
    assert any(
        isinstance(module, nn.GELU)
        for module in MLP(2, 1, num_layers=1, activation="gelu").net
    )
    assert any(
        isinstance(module, nn.SiLU)
        for module in MLP(2, 1, num_layers=1, activation="silu").net
    )

    with pytest.raises(ValueError, match="unknown activation"):
        MLP(2, 1, activation="not-an-activation")
