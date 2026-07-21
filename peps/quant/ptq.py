"""Post-training quantization and encoded-size accounting for W10.

The fake-quantization functions measure reconstruction sensitivity; they do not
claim a speed-up.  The storage report describes a small, explicit packed model
format and counts *all* named parameters, scale payloads, tensor headers, shape
records, names, and the model header.  Consequently ``total_encoded_bits`` is a
model-size estimate, not the old ``bit_width * selected_parameter_count`` proxy.

Both per-tensor and per-channel symmetric quantization are supported.  A
``QuantizationConfig`` can mix precisions by parameter role and can override
individual parameters with glob rules, which is sufficient for reproducible
per-channel and mixed-precision ablations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatchcase
import math
from typing import Mapping

import torch
import torch.nn as nn


_VALID_GRANULARITIES = {"per_tensor", "per_channel"}


def _validate_bits(bits: int) -> int:
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise TypeError("bits must be an integer")
    if bits != 32 and not 2 <= bits <= 16:
        raise ValueError("bits must be in [2, 16], or 32 for unquantized fp32")
    return bits


def _validate_granularity(granularity: str) -> str:
    if granularity not in _VALID_GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {sorted(_VALID_GRANULARITIES)}"
        )
    return granularity


def _normalize_axis(axis: int, ndim: int) -> int:
    if ndim == 0:
        if axis not in (0, -1):
            raise IndexError("a scalar tensor only accepts channel_axis=0")
        return 0
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise IndexError(f"channel_axis={axis} is out of range for ndim={ndim}")
    return axis


@dataclass(frozen=True)
class QuantizationSpec:
    """Encoding choice for one parameter class.

    ``bits=32`` means an unquantized fp32 payload and therefore has no scales.
    For per-channel quantization, ``channel_axis=None`` asks model accounting to
    infer the conventional axis (output axis for weights, feature axis for grids).
    """

    bits: int = 8
    granularity: str = "per_tensor"
    channel_axis: int | None = None
    scale_bits: int = 32

    def __post_init__(self) -> None:
        _validate_bits(self.bits)
        _validate_granularity(self.granularity)
        if (
            isinstance(self.scale_bits, bool)
            or not isinstance(self.scale_bits, int)
            or self.scale_bits <= 0
        ):
            raise ValueError("scale_bits must be a positive integer")

    @property
    def is_quantized(self) -> bool:
        return self.bits < 32


@dataclass(frozen=True)
class EncodingMetadata:
    """Bit costs of the documented packed-model container.

    Model header (128 bits): magic, version, flags, tensor count, reserved.
    Per-tensor fixed header (160 bits): name length, rank, bit width,
    granularity, scale dtype, channel axis, scale count, and payload length.
    Tensor names are UTF-8 and each shape dimension is stored as uint64.
    """

    model_header_bits: int = 128
    tensor_fixed_bits: int = 160
    shape_dimension_bits: int = 64

    def __post_init__(self) -> None:
        for name in (
            "model_header_bits",
            "tensor_fixed_bits",
            "shape_dimension_bits",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value % 8
            ):
                raise ValueError(f"{name} must be a non-negative whole-byte count")

    def tensor_bits(self, name: str, ndim: int) -> int:
        return (
            self.tensor_fixed_bits
            + 8 * len(name.encode("utf-8"))
            + self.shape_dimension_bits * ndim
        )


@dataclass(frozen=True)
class QuantizationConfig:
    """Role defaults plus ordered glob overrides for mixed precision."""

    latent: QuantizationSpec = field(default_factory=QuantizationSpec)
    weight: QuantizationSpec = field(default_factory=QuantizationSpec)
    bias: QuantizationSpec = field(
        default_factory=lambda: QuantizationSpec(bits=32)
    )
    other: QuantizationSpec = field(
        default_factory=lambda: QuantizationSpec(bits=32)
    )
    overrides: Mapping[str, QuantizationSpec] = field(default_factory=dict)
    metadata: EncodingMetadata = field(default_factory=EncodingMetadata)

    def spec_for(self, name: str, role: str) -> QuantizationSpec:
        spec = getattr(self, role)
        # Ordered mappings make the last matching rule the most specific rule.
        for pattern, override in self.overrides.items():
            if fnmatchcase(name, pattern):
                spec = override
        return spec


@dataclass(frozen=True)
class TensorEncoding:
    name: str
    role: str
    shape: tuple[int, ...]
    num_parameters: int
    bits: int
    granularity: str
    channel_axis: int | None
    scale_count: int
    payload_bits: int
    scale_bits: int
    metadata_bits: int

    @property
    def total_bits(self) -> int:
        return self.payload_bits + self.scale_bits + self.metadata_bits


@dataclass(frozen=True)
class ModelEncoding:
    """Complete encoded-size report with deployment-oriented rates."""

    tensors: tuple[TensorEncoding, ...]
    model_metadata_bits: int
    num_pixels: int | None = None
    num_texels: int | None = None
    num_tokens: int | None = None

    @property
    def total_parameters(self) -> int:
        return sum(tensor.num_parameters for tensor in self.tensors)

    @property
    def payload_bits(self) -> int:
        return sum(tensor.payload_bits for tensor in self.tensors)

    @property
    def scale_bits(self) -> int:
        return sum(tensor.scale_bits for tensor in self.tensors)

    @property
    def metadata_bits(self) -> int:
        return self.model_metadata_bits + sum(
            tensor.metadata_bits for tensor in self.tensors
        )

    @property
    def total_encoded_bits(self) -> int:
        return self.payload_bits + self.scale_bits + self.metadata_bits

    @property
    def total_encoded_bytes(self) -> int:
        return math.ceil(self.total_encoded_bits / 8)

    @property
    def bits_per_parameter(self) -> float:
        return self.total_encoded_bits / max(self.total_parameters, 1)

    @staticmethod
    def _rate(bits: int, count: int | None) -> float | None:
        if count is None:
            return None
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("rate denominators must be positive integers")
        return bits / count

    @property
    def bits_per_pixel(self) -> float | None:
        return self._rate(self.total_encoded_bits, self.num_pixels)

    @property
    def bits_per_texel(self) -> float | None:
        return self._rate(self.total_encoded_bits, self.num_texels)

    @property
    def bits_per_token(self) -> float | None:
        return self._rate(self.total_encoded_bits, self.num_tokens)

    @property
    def bpp(self) -> float | None:
        return self.bits_per_pixel

    @property
    def bpt(self) -> float | None:
        """Bits per texel (not the ambiguous bits-per-token abbreviation)."""

        return self.bits_per_texel

    def as_dict(self) -> dict[str, object]:
        role_params = {
            role: sum(
                tensor.num_parameters
                for tensor in self.tensors
                if tensor.role == role
            )
            for role in ("latent", "weight", "bias", "other")
        }
        return {
            "total_params": self.total_parameters,
            "latent_params": role_params["latent"],
            "weight_params": role_params["weight"],
            "bias_params": role_params["bias"],
            "other_params": role_params["other"],
            "payload_bits": self.payload_bits,
            "scale_bits": self.scale_bits,
            "metadata_bits": self.metadata_bits,
            "total_encoded_bits": self.total_encoded_bits,
            "total_encoded_bytes": self.total_encoded_bytes,
            "bits_per_parameter": self.bits_per_parameter,
            "effective_bits": self.bits_per_parameter,
            "bits_per_pixel": self.bits_per_pixel,
            "bits_per_texel": self.bits_per_texel,
            "bits_per_token": self.bits_per_token,
            "bpp": self.bpp,
            "bpt": self.bpt,
            "tensors": self.tensors,
        }


@dataclass
class QuantResult:
    bits: int
    n_params: int
    scale_count: int = 0


def quantize_tensor(
    x: torch.Tensor,
    bits: int = 8,
    *,
    granularity: str = "per_tensor",
    channel_axis: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric linear quantization returning integer codes and scales.

    Per-channel scales retain singleton dimensions and are directly
    broadcastable back over ``x``.  The negative endpoint is retained
    (for example ``[-128, 127]`` for int8), matching the HIP int8 fixture.
    """

    _validate_bits(bits)
    _validate_granularity(granularity)
    if bits == 32:
        raise ValueError("bits=32 is an fp32 storage choice, not integer quantization")
    if not x.is_floating_point():
        raise TypeError("x must be a floating-point tensor")
    if x.numel() == 0:
        raise ValueError("cannot quantize an empty tensor")

    qmax = 2 ** (bits - 1) - 1
    if granularity == "per_tensor":
        if channel_axis is not None:
            raise ValueError("channel_axis is only valid for per_channel")
        maximum = x.abs().amax()
    else:
        if channel_axis is None:
            raise ValueError("per_channel quantization requires channel_axis")
        axis = _normalize_axis(channel_axis, x.ndim)
        if x.ndim == 0:
            maximum = x.abs()
        else:
            reduce_dims = tuple(dim for dim in range(x.ndim) if dim != axis)
            maximum = x.abs().amax(dim=reduce_dims, keepdim=True)

    minimum_scale = torch.finfo(x.dtype).eps
    scale = maximum.clamp(min=minimum_scale) / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    dtype = torch.int8 if bits <= 8 else torch.int16
    return q.to(dtype), scale


def dequantize_tensor(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize codes; scalar and broadcastable per-channel scales work."""

    return q.float() * scale


def fake_quantize(
    x: torch.Tensor,
    bits: int = 8,
    *,
    granularity: str = "per_tensor",
    channel_axis: int | None = None,
) -> torch.Tensor:
    """Quantize then dequantize for quality evaluation."""

    q, scale = quantize_tensor(
        x,
        bits,
        granularity=granularity,
        channel_axis=channel_axis,
    )
    return dequantize_tensor(q, scale)


def _parameter_role(name: str) -> str:
    tokens = name.lower().split(".")
    if any(token.startswith("grid") or token.startswith("table") for token in tokens):
        return "latent"
    if tokens and tokens[-1] == "bias":
        return "bias"
    if tokens and tokens[-1] == "weight":
        return "weight"
    return "other"


def _inferred_channel_axis(
    name: str,
    parameter: torch.Tensor,
    role: str,
    requested: int | None,
) -> int | None:
    if requested is not None:
        return _normalize_axis(requested, parameter.ndim)
    if parameter.ndim == 0:
        return 0
    if role == "latent":
        # GridEncoder: (1, C, H, W[/D]); hash tables: (entries, C).
        return 1 if parameter.ndim >= 2 else 0
    # Linear/conv weights conventionally quantize each output channel.
    if role in {"weight", "bias"}:
        return 0
    raise ValueError(
        f"cannot infer per-channel axis for {name!r}; set channel_axis explicitly"
    )


def _packed_payload_bits(numel: int, bits: int) -> int:
    """Packed payloads are byte-aligned, including their final partial byte."""

    return math.ceil(numel * bits / 8) * 8


def estimate_model_bits(
    model: nn.Module,
    config: QuantizationConfig | None = None,
    *,
    num_pixels: int | None = None,
    num_texels: int | None = None,
    num_tokens: int | None = None,
) -> ModelEncoding:
    """Account for every named parameter in the packed model representation."""

    config = config or QuantizationConfig()
    encoded = []
    for name, parameter in model.named_parameters():
        role = _parameter_role(name)
        spec = config.spec_for(name, role)
        axis = None
        scale_count = 0
        if spec.is_quantized:
            if spec.granularity == "per_channel":
                axis = _inferred_channel_axis(
                    name, parameter, role, spec.channel_axis
                )
                scale_count = (
                    1 if parameter.ndim == 0 else int(parameter.shape[axis])
                )
            else:
                scale_count = 1
        encoded.append(
            TensorEncoding(
                name=name,
                role=role,
                shape=tuple(parameter.shape),
                num_parameters=parameter.numel(),
                bits=spec.bits,
                granularity=spec.granularity,
                channel_axis=axis,
                scale_count=scale_count,
                payload_bits=_packed_payload_bits(parameter.numel(), spec.bits),
                scale_bits=scale_count * spec.scale_bits,
                metadata_bits=config.metadata.tensor_bits(name, parameter.ndim),
            )
        )
    return ModelEncoding(
        tensors=tuple(encoded),
        model_metadata_bits=config.metadata.model_header_bits,
        num_pixels=num_pixels,
        num_texels=num_texels,
        num_tokens=num_tokens,
    )


def make_config(
    *,
    latent_bits: int = 8,
    weight_bits: int = 8,
    bias_bits: int = 32,
    latent_granularity: str = "per_tensor",
    weight_granularity: str = "per_tensor",
    latent_channel_axis: int | None = None,
    weight_channel_axis: int | None = None,
    overrides: Mapping[str, QuantizationSpec] | None = None,
    metadata: EncodingMetadata | None = None,
) -> QuantizationConfig:
    """Convenience constructor used by notebooks and legacy call sites."""

    return QuantizationConfig(
        latent=QuantizationSpec(
            latent_bits, latent_granularity, latent_channel_axis
        ),
        weight=QuantizationSpec(
            weight_bits, weight_granularity, weight_channel_axis
        ),
        bias=QuantizationSpec(bias_bits),
        overrides=overrides or {},
        metadata=metadata or EncodingMetadata(),
    )


@torch.no_grad()
def quantize_module_params(
    module: nn.Module,
    bits: int = 8,
    param_filter=lambda name: True,
    *,
    granularity: str = "per_tensor",
    channel_axis: int | None = None,
) -> QuantResult:
    """In-place fake-quantize matching module parameters."""

    n_params = 0
    scale_count = 0
    for name, parameter in module.named_parameters():
        if not param_filter(name):
            continue
        axis = channel_axis
        if granularity == "per_channel" and axis is None:
            axis = 0
        parameter.copy_(
            fake_quantize(
                parameter,
                bits,
                granularity=granularity,
                channel_axis=axis,
            )
        )
        n_params += parameter.numel()
        scale_count += (
            1
            if granularity == "per_tensor" or parameter.ndim == 0
            else parameter.shape[_normalize_axis(axis, parameter.ndim)]
        )
    return QuantResult(bits=bits, n_params=n_params, scale_count=scale_count)


@torch.no_grad()
def quantize_model(
    model: nn.Module,
    weight_bits: int = 8,
    latent_bits: int = 8,
    *,
    bias_bits: int = 32,
    weight_granularity: str = "per_tensor",
    latent_granularity: str = "per_tensor",
    config: QuantizationConfig | None = None,
    overrides: Mapping[str, QuantizationSpec] | None = None,
    num_pixels: int | None = None,
    num_texels: int | None = None,
    num_tokens: int | None = None,
) -> dict[str, object]:
    """Fake-quantize according to ``config`` and return complete size metrics.

    Biases remain fp32 by default, as in common weight-only PTQ.  Pass
    ``bias_bits`` or a glob override to include them in a mixed-precision study.
    """

    config = config or make_config(
        latent_bits=latent_bits,
        weight_bits=weight_bits,
        bias_bits=bias_bits,
        latent_granularity=latent_granularity,
        weight_granularity=weight_granularity,
        overrides=overrides,
    )
    report = estimate_model_bits(
        model,
        config,
        num_pixels=num_pixels,
        num_texels=num_texels,
        num_tokens=num_tokens,
    )
    report_by_name = {tensor.name: tensor for tensor in report.tensors}
    for name, parameter in model.named_parameters():
        tensor = report_by_name[name]
        if tensor.bits == 32:
            continue
        parameter.copy_(
            fake_quantize(
                parameter,
                tensor.bits,
                granularity=tensor.granularity,
                channel_axis=tensor.channel_axis,
            )
        )
    return report.as_dict()


def model_bitrate(
    model: nn.Module,
    latent_bits: int,
    weight_bits: int,
    *,
    bias_bits: int = 32,
    latent_granularity: str = "per_tensor",
    weight_granularity: str = "per_tensor",
) -> float:
    """Encoded bits per parameter, including scales and metadata."""

    config = make_config(
        latent_bits=latent_bits,
        weight_bits=weight_bits,
        bias_bits=bias_bits,
        latent_granularity=latent_granularity,
        weight_granularity=weight_granularity,
    )
    return estimate_model_bits(model, config).bits_per_parameter
