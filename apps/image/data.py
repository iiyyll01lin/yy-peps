"""Image <-> coordinate/target tensors.

繁體中文:影像與(座標,目標)張量的互轉工具。座標正規化到 [0,1],目標為 RGB in [0,1]。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from data.manifest import (
    DataIntegrityError,
    ManifestError,
    hash_file,
    load_manifest,
    resolve_local_path,
    verify_file,
)


@dataclass(frozen=True)
class LoadedImage:
    """Image tensor plus the source receipt used by a reproduction run."""

    image_id: str
    tensor: torch.Tensor
    source_path: Path
    width: int
    height: int
    color_space: str = "sRGB"

    @property
    def resolution_xy(self) -> tuple[int, int]:
        return (self.width, self.height)


def load_image(path: str, max_size: int | None = None) -> torch.Tensor:
    """Load an image as a float tensor ``(H, W, 3)`` in ``[0, 1]``."""
    img = Image.open(path).convert("RGB")
    if max_size is not None and max(img.size) > max_size:
        s = max_size / max(img.size)
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def coords_grid(h: int, w: int) -> torch.Tensor:
    """Return ``(H*W, 2)`` coordinates in ``[0, 1]``, order (x=col, y=row).

    Row-major so that ``.reshape(H, W, C)`` reassembles the image correctly.
    """
    ys = torch.linspace(0, 1, h)
    xs = torch.linspace(0, 1, w)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (H*W, 2), (x, y)


def image_to_coords_targets(img: torch.Tensor):
    """Return ``(coords (H*W,2), targets (H*W,3), (H, W))`` for fitting."""
    h, w, _ = img.shape
    coords = coords_grid(h, w)
    targets = img.reshape(h * w, 3)
    return coords, targets, (h, w)


def find_kodak(idx: int = 1, root: str | None = None) -> str:
    """Path to ``kodim{idx:02d}.png`` under ``data/raw/kodak``."""
    if root is None:
        here = os.path.dirname(__file__)
        root = os.path.abspath(os.path.join(here, "..", "..", "data", "raw", "kodak"))
    return os.path.join(root, f"kodim{idx:02d}.png")


def load_paper_kodak(
    *,
    raw_root: str | Path | None = None,
    instance_ids: Sequence[str] | None = None,
    verify_checksums: bool = True,
) -> tuple[LoadedImage, ...]:
    """Load original-orientation Kodak images from the pinned 24-file manifest."""

    manifest = load_manifest("kodak")
    requested = (
        tuple(f"kodim{index:02d}" for index in range(1, 25))
        if instance_ids is None
        else tuple(instance_ids)
    )
    records = {item["id"]: item for item in manifest["images"]}
    unknown = sorted(set(requested) - set(records))
    if unknown:
        raise ManifestError(f"unknown Kodak image IDs: {unknown}")
    root = (
        Path(__file__).resolve().parents[2] / "data" / "raw"
        if raw_root is None
        else Path(raw_root)
    )
    loaded = []
    for image_id in requested:
        spec = records[image_id]
        path = resolve_local_path(root, spec)
        if verify_checksums:
            verify_file(path, spec)
        elif not path.is_file():
            raise FileNotFoundError(path)
        tensor = load_image(str(path))
        height, width = tensor.shape[:2]
        if (width, height) != (spec["width"], spec["height"]):
            raise DataIntegrityError(
                f"{path}: manifest dimensions {(spec['width'], spec['height'])}, "
                f"decoded {(width, height)}"
            )
        loaded.append(
            LoadedImage(
                image_id=image_id,
                tensor=tensor,
                source_path=path,
                width=width,
                height=height,
                color_space=spec["color_space"],
            )
        )
    return tuple(loaded)


def orient_resolution_xy(
    landscape_resolution_xy: Sequence[int],
    signal_resolution_xy: Sequence[int],
) -> tuple[int, int]:
    """Rotate paper grid dimensions with portrait Kodak images.

    The paper reports landscape dimensions. Seven Kodak files are stored in
    original portrait orientation; retaining a 196x128 grid without rotating it
    changes texel density by axis and is not the same architecture.
    """

    if len(landscape_resolution_xy) != 2 or len(signal_resolution_xy) != 2:
        raise ValueError("resolutions must be (width, height)")
    grid = (int(landscape_resolution_xy[0]), int(landscape_resolution_xy[1]))
    signal = (int(signal_resolution_xy[0]), int(signal_resolution_xy[1]))
    if min(*grid, *signal) < 2:
        raise ValueError("resolution dimensions must be at least 2")
    grid_landscape = grid[0] >= grid[1]
    signal_landscape = signal[0] >= signal[1]
    return grid if grid_landscape == signal_landscape else (grid[1], grid[0])


def load_fig5_image_manifest(
    manifest_path: str | Path,
    *,
    verify_checksums: bool = True,
) -> tuple[LoadedImage, ...]:
    """Load a user-supplied checksum manifest for the unnamed Fig. 5 image suite.

    The paper never identifies the 4K images used for Fig. 5. A caller must
    therefore provide a dataset receipt rather than silently substituting Kodak
    or copying paper values. The accepted schema is documented by
    ``results/schemas/fig5_dataset.schema.json``.
    """

    path = Path(manifest_path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid Fig. 5 manifest {path}: {exc}") from exc
    if payload.get("schema") != "peps.fig5_dataset" or payload.get(
        "schema_version"
    ) != 1:
        raise ManifestError("Fig. 5 manifest must use peps.fig5_dataset v1")
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise ManifestError("Fig. 5 manifest images must be a non-empty list")
    seen: set[str] = set()
    result = []
    for item in images:
        if not isinstance(item, Mapping):
            raise ManifestError("Fig. 5 image entries must be objects")
        image_id = item.get("id")
        relative = item.get("path")
        digest = item.get("sha256")
        if not isinstance(image_id, str) or not image_id or image_id in seen:
            raise ManifestError("Fig. 5 image IDs must be non-empty and unique")
        if not isinstance(relative, str) or not relative:
            raise ManifestError(f"{image_id}: path is required")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ManifestError(f"{image_id}: lowercase SHA-256 is required")
        source = (path.parent / relative).resolve()
        try:
            source.relative_to(path.parent)
        except ValueError as exc:
            raise ManifestError(f"{image_id}: path escapes manifest directory") from exc
        if not source.is_file():
            raise FileNotFoundError(source)
        if verify_checksums and hash_file(source, "sha256") != digest:
            raise DataIntegrityError(f"{source}: Fig. 5 SHA-256 mismatch")
        tensor = load_image(str(source))
        height, width = tensor.shape[:2]
        if max(width, height) < 3840 or min(width, height) < 2048:
            raise DataIntegrityError(
                f"{source}: expected native 4K dimensions, found {width}x{height}"
            )
        declared = item.get("resolution_xy")
        if declared is not None and tuple(declared) != (width, height):
            raise DataIntegrityError(
                f"{source}: declared resolution {declared}, found {(width, height)}"
            )
        seen.add(image_id)
        result.append(
            LoadedImage(
                image_id=image_id,
                tensor=tensor,
                source_path=source,
                width=width,
                height=height,
                color_space=str(item.get("color_space", "sRGB")),
            )
        )
    return tuple(result)
