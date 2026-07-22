"""Deterministic Armadillo surface rendering for SDF validation.

The paper does not publish camera or lighting parameters for its Armadillo
sample.  This module therefore implements an explicit, fixed orthographic
protocol and labels its output as a render-protocol assumption.  It renders
streamed SDF slabs, so a 512^3 prediction never has to be retained in memory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
import io
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np
from PIL import Image
import torch


@dataclass(frozen=True)
class OrthographicRenderProtocol:
    """Fixed camera and Lambertian lighting metadata."""

    camera: str = "orthographic_z_min_to_z_max"
    image_axes: tuple[str, str] = ("x", "y")
    surface_rule: str = "first_negative_voxel"
    normal_estimator: str = "screen_space_depth_central_difference"
    light_direction: tuple[float, float, float] = (0.4, -0.5, 0.7681145748)
    ambient: float = 0.25
    diffuse: float = 0.75
    albedo: tuple[float, float, float] = (0.64, 0.69, 0.76)
    background: tuple[float, float, float] = (1.0, 1.0, 1.0)
    flip_mode: str = "LDR"
    flip_ppd: float = 67.0
    paper_camera_available: bool = False
    verification_status: str = "render_protocol_assumption"

    def __post_init__(self) -> None:
        if self.camera != "orthographic_z_min_to_z_max":
            raise ValueError("only the frozen z-min orthographic camera is supported")
        if self.image_axes != ("x", "y"):
            raise ValueError("image axes must be ('x', 'y')")
        if self.surface_rule != "first_negative_voxel":
            raise ValueError("surface_rule must be first_negative_voxel")
        if self.normal_estimator != "screen_space_depth_central_difference":
            raise ValueError("unsupported normal estimator")
        if self.flip_mode != "LDR":
            raise ValueError("only LDR FLIP is supported")
        if not math.isfinite(self.flip_ppd) or self.flip_ppd <= 0:
            raise ValueError("flip_ppd must be finite and positive")
        for name in ("ambient", "diffuse"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("light_direction", "albedo", "background"):
            values = getattr(self, name)
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain three finite values")
        if np.linalg.norm(np.asarray(self.light_direction)) == 0:
            raise ValueError("light_direction cannot be zero")
        if any(not 0 <= value <= 1 for value in (*self.albedo, *self.background)):
            raise ValueError("albedo and background values must be in [0, 1]")
        if self.paper_camera_available:
            raise ValueError("the paper does not publish its Armadillo camera")
        if self.verification_status != "render_protocol_assumption":
            raise ValueError("render output must retain the protocol-assumption label")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "OrthographicRenderProtocol":
        expected = set(cls.__dataclass_fields__)
        supplied = set(values)
        extra = supplied - expected - {"enabled", "asset", "resolution"}
        if extra:
            raise ValueError(f"unknown render fields: {sorted(extra)}")
        kwargs = {
            key: values[key]
            for key in expected
            if key in values
        }
        for name in (
            "image_axes",
            "light_direction",
            "albedo",
            "background",
        ):
            if name in kwargs:
                kwargs[name] = tuple(kwargs[name])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["image_axes"] = list(self.image_axes)
        payload["light_direction"] = list(self.light_direction)
        payload["albedo"] = list(self.albedo)
        payload["background"] = list(self.background)
        return payload


class FirstSurfaceAccumulator:
    """Collect the first inside voxel along +z for two streamed SDFs."""

    def __init__(self, resolution: int) -> None:
        if isinstance(resolution, bool) or not isinstance(resolution, int):
            raise TypeError("resolution must be an integer")
        if resolution < 2:
            raise ValueError("resolution must be at least two")
        self.resolution = resolution
        shape = (resolution, resolution)
        self.prediction_depth = np.full(shape, np.nan, dtype=np.float32)
        self.reference_depth = np.full(shape, np.nan, dtype=np.float32)

    def _update_one(
        self,
        destination: np.ndarray,
        values: torch.Tensor,
        start: int,
    ) -> None:
        if values.ndim != 3 or tuple(values.shape[1:]) != (
            self.resolution,
            self.resolution,
        ):
            raise ValueError("SDF slab must have shape (depth, resolution, resolution)")
        inside = values < 0
        any_inside = inside.any(dim=0)
        unseen = torch.from_numpy(np.isnan(destination))
        selected = any_inside.cpu() & unseen
        if not selected.any():
            return
        first_local = inside.to(dtype=torch.uint8).argmax(dim=0).cpu()
        depth = (first_local.to(torch.float32) + start) / (self.resolution - 1)
        destination[selected.numpy()] = depth[selected].numpy()

    def update(
        self,
        z_slice: slice,
        prediction: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        start = 0 if z_slice.start is None else int(z_slice.start)
        stop = self.resolution if z_slice.stop is None else int(z_slice.stop)
        if z_slice.step not in {None, 1} or not 0 <= start < stop <= self.resolution:
            raise ValueError("z_slice must be a non-empty contiguous slice")
        expected = (stop - start, self.resolution, self.resolution)
        if tuple(prediction.shape) != expected or tuple(reference.shape) != expected:
            raise ValueError(f"slabs must have shape {expected}")
        self._update_one(self.prediction_depth, prediction.detach().cpu(), start)
        self._update_one(self.reference_depth, reference.detach().cpu(), start)

    def render(
        self,
        protocol: OrthographicRenderProtocol,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            shade_depth(self.prediction_depth, protocol),
            shade_depth(self.reference_depth, protocol),
        )


def shade_depth(
    depth: np.ndarray,
    protocol: OrthographicRenderProtocol,
) -> np.ndarray:
    """Convert a first-surface depth map to an sRGB-like validation image."""

    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("depth must be a square rank-2 array")
    mask = np.isfinite(values)
    filled = np.where(mask, values, 1.0)
    grad_y, grad_x = np.gradient(filled)
    scale = max(values.shape[0] - 1, 1)
    normals = np.stack(
        (-grad_x * scale, -grad_y * scale, np.ones_like(filled)),
        axis=-1,
    )
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True).clip(1e-12)
    light = np.asarray(protocol.light_direction, dtype=np.float32)
    light /= np.linalg.norm(light)
    lambert = np.clip(np.sum(normals * light, axis=-1), 0.0, 1.0)
    intensity = protocol.ambient + protocol.diffuse * lambert
    surface = intensity[..., None] * np.asarray(protocol.albedo, dtype=np.float32)
    background = np.asarray(protocol.background, dtype=np.float32)
    image = np.where(mask[..., None], surface, background)
    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


def evaluate_flip(
    reference: np.ndarray,
    prediction: np.ndarray,
    protocol: OrthographicRenderProtocol,
) -> tuple[np.ndarray, float, dict[str, object]]:
    """Return the official package's magma FLIP map and mean error."""

    try:
        import flip_evaluator
    except ImportError as exc:
        raise ImportError(
            "Armadillo render validation requires the flip-evaluator package"
        ) from exc
    error_map, mean_error, parameters = flip_evaluator.evaluate(
        np.asarray(reference, dtype=np.float32),
        np.asarray(prediction, dtype=np.float32),
        protocol.flip_mode,
        inputsRGB=True,
        applyMagma=True,
        computeMeanError=True,
        parameters={"ppd": protocol.flip_ppd},
    )
    mapped = np.asarray(error_map, dtype=np.float32)
    if mapped.ndim == 2:
        mapped = np.repeat(mapped[..., None], 3, axis=-1)
    elif mapped.ndim == 3 and mapped.shape[0] == 3 and mapped.shape[-1] != 3:
        mapped = np.moveaxis(mapped, 0, -1)
    if mapped.ndim != 3 or mapped.shape[-1] != 3:
        raise ValueError(f"unexpected FLIP map shape {mapped.shape}")
    plain_parameters = {
        str(key): (
            value.item()
            if isinstance(value, np.generic)
            else list(value)
            if isinstance(value, np.ndarray)
            else value
        )
        for key, value in dict(parameters).items()
    }
    try:
        plain_parameters["package_version"] = importlib_metadata.version(
            "flip-evaluator"
        )
    except importlib_metadata.PackageNotFoundError:
        plain_parameters["package_version"] = None
    return np.clip(mapped, 0.0, 1.0), float(mean_error), plain_parameters


def save_png_atomic(path: str | Path, image: np.ndarray) -> Path:
    """Write an RGB float image atomically as an 8-bit PNG."""

    destination = Path(path)
    values = np.asarray(image)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("PNG image must have shape (H, W, 3)")
    if not np.isfinite(values).all():
        raise ValueError("PNG image contains non-finite values")
    encoded = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(encoded, mode="RGB").save(buffer, format="PNG")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(buffer.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination
