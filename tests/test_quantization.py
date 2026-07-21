"""Focused tests for quantization fidelity and encoded-rate accounting."""

import copy

import torch
import torch.nn as nn

from peps.quant import (
    EncodingMetadata,
    QuantizationConfig,
    QuantizationSpec,
    estimate_model_bits,
    make_config,
    model_bitrate,
    quantize_model,
    quantize_tensor,
)


class _TinyCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.grid = nn.Parameter(
            torch.linspace(-1.0, 1.0, 8).reshape(1, 2, 2, 2)
        )
        self.linear = nn.Linear(2, 1)


def test_per_channel_quantization_uses_broadcastable_scales_and_reduces_error():
    values = torch.tensor(
        [[-0.01, 0.01, -0.005], [-100.0, 50.0, 100.0]],
        dtype=torch.float32,
    )
    per_tensor, tensor_scale = quantize_tensor(values, 4)
    per_channel, channel_scales = quantize_tensor(
        values, 4, granularity="per_channel", channel_axis=0
    )

    assert tensor_scale.ndim == 0
    assert channel_scales.shape == (2, 1)
    tensor_error = (per_tensor.float() * tensor_scale - values).abs().mean()
    channel_error = (per_channel.float() * channel_scales - values).abs().mean()
    assert channel_error < tensor_error


def test_encoded_bits_include_every_parameter_scales_and_metadata():
    model = _TinyCodec()
    config = QuantizationConfig(
        latent=QuantizationSpec(
            bits=4,
            granularity="per_channel",
            scale_bits=16,
        ),
        weight=QuantizationSpec(
            bits=8,
            granularity="per_channel",
            scale_bits=32,
        ),
        bias=QuantizationSpec(bits=32),
        metadata=EncodingMetadata(),
    )
    report = estimate_model_bits(
        model, config, num_pixels=4, num_texels=8, num_tokens=2
    )

    assert report.total_parameters == sum(p.numel() for p in model.parameters())
    assert {tensor.name for tensor in report.tensors} == {
        "grid",
        "linear.weight",
        "linear.bias",
    }
    by_name = {tensor.name: tensor for tensor in report.tensors}
    assert by_name["grid"].channel_axis == 1
    assert by_name["grid"].scale_count == 2
    assert by_name["linear.weight"].channel_axis == 0
    assert by_name["linear.weight"].scale_count == 1
    assert by_name["linear.bias"].scale_count == 0
    assert report.scale_bits == 2 * 16 + 32
    assert report.metadata_bits > 0
    assert report.total_encoded_bits == (
        report.payload_bits + report.scale_bits + report.metadata_bits
    )
    assert report.bpp == report.total_encoded_bits / 4
    assert report.bpt == report.total_encoded_bits / 8
    assert report.bits_per_token == report.total_encoded_bits / 2


def test_mixed_precision_glob_override_controls_storage_and_fake_quantization():
    torch.manual_seed(4)
    model = _TinyCodec()
    before = copy.deepcopy(model)
    config = make_config(
        latent_bits=6,
        weight_bits=8,
        latent_granularity="per_tensor",
        weight_granularity="per_channel",
        overrides={
            "linear.weight": QuantizationSpec(
                bits=4, granularity="per_channel", channel_axis=0
            ),
            "*.bias": QuantizationSpec(bits=32),
        },
    )
    expected = estimate_model_bits(model, config, num_pixels=4)
    metrics = quantize_model(model, config=config, num_pixels=4)

    encoded = {tensor.name: tensor for tensor in expected.tensors}
    assert encoded["grid"].bits == 6
    assert encoded["linear.weight"].bits == 4
    assert encoded["linear.bias"].bits == 32
    assert metrics["total_encoded_bits"] == expected.total_encoded_bits
    assert metrics["bpp"] == expected.bpp
    assert torch.equal(model.linear.bias, before.linear.bias)
    assert not torch.equal(model.grid, before.grid)


def test_model_bitrate_is_total_encoded_bits_per_parameter_not_payload_proxy():
    model = _TinyCodec()
    report = estimate_model_bits(
        model,
        make_config(latent_bits=8, weight_bits=8, bias_bits=8),
    )
    measured = model_bitrate(model, latent_bits=8, weight_bits=8, bias_bits=8)
    assert measured == report.bits_per_parameter
    assert measured > 8.0  # scale and model/tensor metadata are intentionally counted
