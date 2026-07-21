"""Export PyTorch PEPS weights to the fused HIP fixture format.

The binary schema is intentionally small and stable so the HIP executable can
be tested without loading Python or a framework at runtime. Matrices are stored
input-major (``[in, out]``), while ``torch.nn.Linear`` stores ``[out, in]``;
the exporter performs and validates that transpose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

WORKLOAD_MAGIC = 0x50505332  # PPS2
WORKLOAD_SCHEMA = 2

MODE_BASELINE = 0
MODE_CONCAT = 1
MODE_PINK = 2

ACT_RELU = 0
ACT_GELU = 1
ACT_LEAKY_RELU = 2


@dataclass(frozen=True)
class MethodSpec:
    """One of the four paper runtime configurations."""

    name: str
    mode: int
    frequencies: int
    paper_ms: float

    @property
    def is_selective(self) -> bool:
        return self.mode == MODE_PINK


METHOD_SPECS = {
    spec.name: spec
    for spec in (
        MethodSpec("bi-grid", MODE_BASELINE, 0, 4.32),
        MethodSpec("grid-peps-3f", MODE_CONCAT, 3, 5.47),
        MethodSpec("grid-pink-peps-3f", MODE_PINK, 3, 4.86),
        MethodSpec("grid-pink-peps-4f", MODE_PINK, 4, 4.99),
    )
}

METHOD_ALIASES = {
    "baseline": "bi-grid",
    "grid": "bi-grid",
    "peps": "grid-peps-3f",
    "pink": "grid-pink-peps-3f",
    "pink3": "grid-pink-peps-3f",
    "pink4": "grid-pink-peps-4f",
}


def resolve_method(method: str | MethodSpec) -> MethodSpec:
    if isinstance(method, MethodSpec):
        return method
    name = METHOD_ALIASES.get(method.lower(), method.lower())
    try:
        return METHOD_SPECS[name]
    except KeyError as exc:
        choices = ", ".join(METHOD_SPECS)
        raise ValueError(f"unknown HIP method {method!r}; choose {choices}") from exc


def pink_widths(channels: int, frequencies: int) -> tuple[int, ...]:
    return tuple(max(1, channels // (2**index)) for index in range(1, frequencies + 1))


def aggregate_dim(method: str | MethodSpec, channels: int) -> int:
    spec = resolve_method(method)
    if spec.mode == MODE_BASELINE:
        return channels
    if spec.mode == MODE_CONCAT:
        return (2 * spec.frequencies + 1) * channels
    return channels + 2 * sum(pink_widths(channels, spec.frequencies))


def pink_channel_indices(
    channels: int, frequencies: int
) -> tuple[tuple[int, ...], ...]:
    widths = pink_widths(channels, frequencies)
    cumulative = [0]
    for width in widths:
        cumulative.append(cumulative[-1] + width)
    sine = [
        tuple(index % channels for index in range(-cumulative[i], -cumulative[i - 1]))
        for i in range(1, frequencies + 1)
    ]
    cosine = [
        tuple(index % channels for index in range(cumulative[i - 1], cumulative[i]))
        for i in range(1, frequencies + 1)
    ]
    return (tuple(range(channels)), *sine, *cosine)


def projected_points(coords: np.ndarray, frequencies: int) -> np.ndarray:
    """Return ``(x, S_1..S_L, C_1..C_L)`` using fp32 device arithmetic."""

    coords = np.asarray(coords, dtype=np.float32)
    points = [coords]
    for function in (np.sin, np.cos):
        for index in range(1, frequencies + 1):
            phi = np.float32((2**index) * np.pi)
            angle = np.asarray(coords * phi, dtype=np.float32)
            points.append(
                np.asarray(
                    np.float32(0.5)
                    * (np.float32(1.0) + function(angle)),
                    dtype=np.float32,
                )
            )
    return np.stack(points, axis=1)


def bilinear_sample(grid_chw: np.ndarray, u: float, v: float) -> np.ndarray:
    """Channel-first, align-corners bilinear sampling with border clamp."""

    grid = np.asarray(grid_chw)
    channels, height, width = grid.shape
    u32 = np.clip(np.float32(u), np.float32(0), np.float32(1))
    v32 = np.clip(np.float32(v), np.float32(0), np.float32(1))
    fx = np.float32(u32 * np.float32(width - 1))
    fy = np.float32(v32 * np.float32(height - 1))
    x0 = int(np.floor(fx))
    y0 = int(np.floor(fy))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    wx = np.float32(fx - np.float32(x0))
    wy = np.float32(fy - np.float32(y0))
    v00 = grid[:, y0, x0].astype(np.float32, copy=False)
    v01 = grid[:, y0, x1].astype(np.float32, copy=False)
    v10 = grid[:, y1, x0].astype(np.float32, copy=False)
    v11 = grid[:, y1, x1].astype(np.float32, copy=False)
    top = v00 * (np.float32(1) - wx) + v01 * wx
    bottom = v10 * (np.float32(1) - wx) + v11 * wx
    return np.asarray(
        top * (np.float32(1) - wy) + bottom * wy,
        dtype=np.float32,
    ).reshape(channels)


def integrated_features(
    grid_chw: np.ndarray,
    coords: np.ndarray,
    method: str | MethodSpec,
) -> np.ndarray:
    spec = resolve_method(method)
    grid = np.asarray(grid_chw)
    coordinates = np.asarray(coords, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if spec.mode == MODE_BASELINE:
        return np.stack(
            [bilinear_sample(grid, u, v) for u, v in coordinates]
        ).astype(np.float32)

    points = projected_points(coordinates, spec.frequencies)
    if spec.mode == MODE_CONCAT:
        sampled = [
            np.stack([bilinear_sample(grid, u, v) for u, v in points[:, point]])
            for point in range(points.shape[1])
        ]
        return np.concatenate(sampled, axis=1).astype(np.float32)

    selected = []
    indices_by_point = pink_channel_indices(grid.shape[0], spec.frequencies)
    for point, indices in enumerate(indices_by_point):
        sampled = np.stack(
            [bilinear_sample(grid, u, v) for u, v in points[:, point]]
        )
        selected.append(sampled[:, list(indices)])
    return np.concatenate(selected, axis=1).astype(np.float32)


def _activate(value: torch.Tensor, activation: int) -> torch.Tensor:
    if activation == ACT_GELU:
        return F.gelu(value, approximate="none")
    if activation == ACT_LEAKY_RELU:
        return F.leaky_relu(value, negative_slope=0.01)
    if activation == ACT_RELU:
        return F.relu(value)
    raise ValueError(f"unsupported activation code {activation}")


@dataclass
class Fixture:
    """Framework-neutral contents of one end-to-end inference fixture."""

    method: MethodSpec
    grid: np.ndarray
    coords: np.ndarray
    weights: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    biases: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    activation: int = ACT_GELU

    def validate(self) -> None:
        self.method = resolve_method(self.method)
        self.grid = np.ascontiguousarray(self.grid, dtype=np.float32)
        self.coords = np.ascontiguousarray(self.coords, dtype=np.float32)
        if self.grid.ndim != 3:
            raise ValueError("grid must have shape (C, H, W)")
        channels, height, width = self.grid.shape
        if not (1 <= channels <= 32 and height >= 2 and width >= 2):
            raise ValueError("grid must have C<=32 and spatial dimensions >=2")
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError("coords must have shape (N, 2)")
        if self.coords.shape[0] < 1:
            raise ValueError("fixture needs at least one coordinate")
        if not np.isfinite(self.grid).all() or not np.isfinite(self.coords).all():
            raise ValueError("grid and coords must be finite")

        self.weights = tuple(
            np.ascontiguousarray(weight, dtype=np.float32)
            for weight in self.weights
        )
        self.biases = tuple(
            np.ascontiguousarray(bias, dtype=np.float32).reshape(-1)
            for bias in self.biases
        )
        if len(self.weights) != 4 or len(self.biases) != 4:
            raise ValueError("decoder must contain exactly four Linear layers")
        input_dim = aggregate_dim(self.method, channels)
        hidden = self.biases[0].shape[0]
        output = self.biases[3].shape[0]
        expected = (
            (input_dim, hidden),
            (hidden, hidden),
            (hidden, hidden),
            (hidden, output),
        )
        if tuple(weight.shape for weight in self.weights) != expected:
            raise ValueError(
                f"weight shapes {tuple(w.shape for w in self.weights)} "
                f"do not match {expected}"
            )
        expected_biases = ((hidden,), (hidden,), (hidden,), (output,))
        if tuple(bias.shape for bias in self.biases) != expected_biases:
            raise ValueError("bias shapes do not match the four Linear layers")
        if not (1 <= hidden <= 128 and 1 <= output <= 16):
            raise ValueError("HIP limits are hidden<=128 and output<=16")
        if input_dim > 512:
            raise ValueError("aggregated input exceeds the HIP limit of 512")
        if self.activation not in (ACT_RELU, ACT_GELU, ACT_LEAKY_RELU):
            raise ValueError("unsupported hidden activation")
        for value in (*self.weights, *self.biases):
            if not np.isfinite(value).all():
                raise ValueError("weights and biases must be finite")

    @property
    def hidden_dim(self) -> int:
        return int(self.biases[0].shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.biases[3].shape[0])

    def reference(self, precision: str = "fp32") -> np.ndarray:
        """Compute the complete CPU output for fp32 or fused fp16/WMMA."""

        self.validate()
        if precision not in {"fp32", "fp16"}:
            raise ValueError("precision must be fp32 or fp16")
        grid = self.grid
        if precision == "fp16":
            grid = grid.astype(np.float16).astype(np.float32)
        features = integrated_features(grid, self.coords, self.method)
        value = torch.from_numpy(features)
        for layer in range(3):
            weight = self.weights[layer]
            if precision == "fp16":
                value = value.half().float()
                weight = weight.astype(np.float16).astype(np.float32)
            value = _activate(
                value @ torch.from_numpy(weight)
                + torch.from_numpy(self.biases[layer]),
                self.activation,
            )
        weight = self.weights[3]
        if precision == "fp16":
            value = value.half().float()
            weight = weight.astype(np.float16).astype(np.float32)
        return (
            value @ torch.from_numpy(weight)
            + torch.from_numpy(self.biases[3])
        ).numpy()


def make_random_fixture(
    method: str | MethodSpec,
    *,
    channels: int = 16,
    grid_height: int = 11,
    grid_width: int = 9,
    points: int = 37,
    hidden: int = 64,
    output: int = 3,
    activation: int = ACT_GELU,
    seed: int = 20260427,
    coords: np.ndarray | None = None,
) -> Fixture:
    spec = resolve_method(method)
    rng = np.random.default_rng(seed)
    grid = (rng.standard_normal((channels, grid_height, grid_width)) * 0.2).astype(
        np.float32
    )
    if coords is None:
        coords = rng.uniform(0.0, 1.0, size=(points, 2)).astype(np.float32)
    else:
        coords = np.asarray(coords, dtype=np.float32)
        points = coords.shape[0]
    input_dim = aggregate_dim(spec, channels)
    shapes = (
        (input_dim, hidden),
        (hidden, hidden),
        (hidden, hidden),
        (hidden, output),
    )
    weights = tuple(
        (rng.standard_normal(shape) * 0.04).astype(np.float32)
        for shape in shapes
    )
    biases = tuple(
        (rng.standard_normal(shape[1]) * 0.01).astype(np.float32)
        for shape in shapes
    )
    fixture = Fixture(spec, grid, coords, weights, biases, activation)
    fixture.validate()
    return fixture


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def fixture_bytes(fixture: Fixture) -> bytes:
    fixture.validate()
    channels, height, width = fixture.grid.shape
    header = np.array(
        [
            WORKLOAD_MAGIC,
            WORKLOAD_SCHEMA,
            fixture.method.mode,
            channels,
            height,
            width,
            fixture.coords.shape[0],
            fixture.method.frequencies,
            fixture.hidden_dim,
            fixture.output_dim,
            fixture.activation,
        ],
        dtype="<i4",
    )
    chunks = [header.tobytes(), fixture.grid.astype("<f4", copy=False).tobytes()]
    chunks.append(fixture.coords.astype("<f4", copy=False).tobytes())
    for weight, bias in zip(fixture.weights, fixture.biases):
        chunks.append(weight.astype("<f4", copy=False).tobytes())
        chunks.append(bias.astype("<f4", copy=False).tobytes())
    return b"".join(chunks)


def write_fixture(
    path: str | Path,
    fixture: Fixture,
    *,
    manifest_path: str | Path | None = None,
) -> dict:
    path = Path(path)
    payload = fixture_bytes(fixture)
    _atomic_write(path, payload)
    fixture.validate()
    metadata = {
        "schema_version": WORKLOAD_SCHEMA,
        "format": "peps_hip_fixture",
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "method": fixture.method.name,
        "mode": fixture.method.mode,
        "frequencies": fixture.method.frequencies,
        "selective_channel_sampling": fixture.method.is_selective,
        "grid_shape_chw": list(fixture.grid.shape),
        "coords_shape": list(fixture.coords.shape),
        "weight_layout": "input_major",
        "weight_shapes": [list(weight.shape) for weight in fixture.weights],
        "bias_shapes": [list(bias.shape) for bias in fixture.biases],
        "hidden_layers": 3,
        "output_dim": fixture.output_dim,
        "activation": fixture.activation,
    }
    if manifest_path is not None:
        manifest = Path(manifest_path)
        _atomic_write(
            manifest,
            (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(),
        )
    return metadata


def write_weight_archive(path: str | Path, fixture: Fixture) -> dict:
    """Write a portable NPZ archive with grid and input-major decoder weights."""

    fixture.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            grid=fixture.grid,
            **{f"weight_{i + 1}": value for i, value in enumerate(fixture.weights)},
            **{f"bias_{i + 1}": value for i, value in enumerate(fixture.biases)},
            method=np.array(fixture.method.name),
            frequencies=np.array(fixture.method.frequencies, dtype=np.int32),
            weight_layout=np.array("input_major"),
        )
    os.replace(temporary, path)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "weight_layout": "input_major",
    }


def read_output(path: str | Path) -> tuple[int, np.ndarray]:
    payload = Path(path).read_bytes()
    if len(payload) < 20:
        raise ValueError("short HIP output")
    magic, schema, mode, points, output = struct.unpack_from("<5i", payload)
    if magic != WORKLOAD_MAGIC or schema != WORKLOAD_SCHEMA:
        raise ValueError("bad HIP output header")
    expected = 20 + points * output * 4
    if len(payload) != expected:
        raise ValueError(f"HIP output has {len(payload)} bytes, expected {expected}")
    values = np.frombuffer(payload, dtype="<f4", offset=20).copy()
    return mode, values.reshape(points, output)


def _find_grid(model: torch.nn.Module) -> torch.Tensor:
    candidates = [
        module.grid
        for module in model.modules()
        if isinstance(getattr(module, "grid", None), torch.Tensor)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one module with a grid tensor, found {len(candidates)}"
        )
    return candidates[0]


def _decoder_layers(model: torch.nn.Module) -> list[torch.nn.Linear]:
    layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
    if len(layers) != 4:
        raise ValueError(f"HIP exporter requires four Linear layers, found {len(layers)}")
    return layers


def _activation_code(model: torch.nn.Module) -> int:
    supported = []
    for module in model.modules():
        if isinstance(module, torch.nn.GELU):
            if module.approximate != "none":
                raise ValueError("HIP GELU requires approximate='none'")
            supported.append(ACT_GELU)
        elif isinstance(module, torch.nn.LeakyReLU):
            if not math.isclose(float(module.negative_slope), 0.01):
                raise ValueError("HIP LeakyReLU requires negative_slope=0.01")
            supported.append(ACT_LEAKY_RELU)
        elif isinstance(module, torch.nn.ReLU):
            supported.append(ACT_RELU)
        elif isinstance(module, (torch.nn.SiLU, torch.nn.Sigmoid, torch.nn.Tanh)):
            raise ValueError(
                f"{type(module).__name__} is not supported by this RGB HIP path"
            )
    if len(supported) != 3 or len(set(supported)) != 1:
        raise ValueError("decoder must have three identical supported activations")
    return supported[0]


def export_pytorch_fixture(
    path: str | Path,
    *,
    model: torch.nn.Module,
    coords: torch.Tensor | np.ndarray,
    method: str | MethodSpec,
    grid: torch.Tensor | np.ndarray | None = None,
    decoder: torch.nn.Module | None = None,
    weights_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> Fixture:
    """Export an actual trained Grid/PEPS model and coordinates.

    ``model`` is used to discover a single grid and the decoder. Callers may
    provide ``grid`` or ``decoder`` explicitly for custom wrappers.
    """

    if grid is None:
        grid = _find_grid(model)
    if isinstance(grid, torch.Tensor):
        grid_tensor = grid.detach().cpu()
        if grid_tensor.ndim == 4 and grid_tensor.shape[0] == 1:
            grid_tensor = grid_tensor[0]
        grid_array = grid_tensor.float().numpy()
    else:
        grid_array = np.asarray(grid, dtype=np.float32)

    decoder = model if decoder is None else decoder
    layers = _decoder_layers(decoder)
    activation = _activation_code(decoder)
    weights = tuple(
        layer.weight.detach().cpu().float().t().contiguous().numpy()
        for layer in layers
    )
    biases = tuple(
        layer.bias.detach().cpu().float().contiguous().numpy()
        if layer.bias is not None
        else np.zeros(layer.out_features, dtype=np.float32)
        for layer in layers
    )
    coordinate_array = (
        coords.detach().cpu().float().contiguous().numpy()
        if isinstance(coords, torch.Tensor)
        else np.asarray(coords, dtype=np.float32)
    )
    fixture = Fixture(
        resolve_method(method),
        grid_array,
        coordinate_array,
        weights,
        biases,
        activation,
    )
    write_fixture(path, fixture, manifest_path=manifest_path)
    if weights_path is not None:
        write_weight_archive(weights_path, fixture)
    return fixture


def _paper_coordinates(side: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, side, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((xx, yy), axis=-1).reshape(-1, 2)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--method", choices=tuple(METHOD_SPECS), required=True)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--grid-height", type=int, default=11)
    parser.add_argument("--grid-width", type=int, default=9)
    parser.add_argument("--points", type=int, default=37)
    parser.add_argument("--side", type=int)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260427)
    args = parser.parse_args(argv)
    coords = None
    points = args.points
    if args.side is not None:
        if args.side < 2:
            parser.error("--side must be >=2")
        coords = _paper_coordinates(args.side)
        points = coords.shape[0]
    fixture = make_random_fixture(
        args.method,
        channels=args.channels,
        grid_height=args.grid_height,
        grid_width=args.grid_width,
        points=points,
        hidden=args.hidden,
        output=3,
        seed=args.seed,
        coords=coords,
    )
    metadata = write_fixture(args.output, fixture, manifest_path=args.manifest)
    if args.weights is not None:
        metadata["weights"] = write_weight_archive(args.weights, fixture)
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
