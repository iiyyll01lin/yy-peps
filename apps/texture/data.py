"""Texture maps <-> coordinate/target tensors.

``load_paper_texture_set`` is the manifest-driven paper path and preserves the
actual number of RGB maps in each set. ``load_pbr_bundle`` remains for old
course notebooks, but is strict: it no longer invents missing channels.
"""

from __future__ import annotations

import glob
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data.manifest import LoadedTextureSet, load_texture_set

# Legacy fixed bundle used only by existing teaching notebooks.
CHANNEL_LAYOUT = [
    ("Color", 3),          # albedo RGB
    ("NormalGL", 3),       # OpenGL normal XYZ
    ("Roughness", 1),
    ("Metalness", 1),
    ("AmbientOcclusion", 1),
]
TOTAL_CHANNELS = sum(c for _, c in CHANNEL_LAYOUT)  # = 9


def _find_map(bundle_dir: str, suffix: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(bundle_dir, f"*_{suffix}.png")))
    if len(hits) > 1:
        raise ValueError(f"ambiguous {suffix} map in {bundle_dir}: {hits}")
    return hits[0] if hits else None


def load_paper_texture_set(
    set_id: str,
    *,
    root: str | None = None,
    size: int | tuple[int, int] | None = None,
    verify_checksums: bool = True,
) -> LoadedTextureSet:
    """Load one of the 18 paper sets as dynamic ``H x W x (3k)`` data."""

    return load_texture_set(
        set_id,
        raw_root=root,
        output_size=size,
        verify_checksums=verify_checksums,
    )


def load_pbr_bundle(bundle_dir: str, size: int = 512) -> torch.Tensor:
    """Load the legacy fixed 9-channel bundle without synthetic defaults.

    All five maps are required. OpenGL normals are filtered as vectors and
    renormalized after resizing.
    """

    planes: list[torch.Tensor] = []
    for suffix, nch in CHANNEL_LAYOUT:
        path = _find_map(bundle_dir, suffix)
        if path is None:
            raise FileNotFoundError(
                f"required {suffix} map is missing from {bundle_dir}; "
                "the loader does not synthesize paper targets"
            )
        with Image.open(path) as image:
            array = np.asarray(image)
        if array.dtype == np.uint8:
            array = array.astype(np.float32) / 255.0
        elif array.dtype == np.uint16:
            array = array.astype(np.float32) / 65535.0
        else:
            raise ValueError(f"unsupported texture dtype {array.dtype}: {path}")
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.ndim != 3 or array.shape[-1] < 3:
            raise ValueError(f"unsupported texture shape {array.shape}: {path}")
        tensor = torch.from_numpy(np.ascontiguousarray(array[..., :3]))
        if suffix == "NormalGL":
            tensor = tensor.mul(2.0).sub(1.0)
        chw = tensor.permute(2, 0, 1).unsqueeze(0)
        chw = F.interpolate(
            chw,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        if suffix == "NormalGL":
            length = torch.linalg.vector_norm(chw, dim=1, keepdim=True)
            fallback = torch.zeros_like(chw)
            fallback[:, 2:3] = 1.0
            chw = torch.where(length > 1e-8, chw / length.clamp_min(1e-8), fallback)
            chw = chw.add(1.0).mul(0.5).clamp_(0.0, 1.0)
        plane = chw.squeeze(0).permute(1, 2, 0).contiguous()
        planes.append(plane[..., :nch])
    return torch.cat(planes, dim=-1)  # (H, W, 9)


def bundle_to_coords_targets(bundle: torch.Tensor):
    """Return ``(coords (H*W,2), targets (H*W,C), (H, W))``."""
    h, w, c = bundle.shape
    ys = torch.linspace(0, 1, h)
    xs = torch.linspace(0, 1, w)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    targets = bundle.reshape(h * w, c)
    return coords, targets, (h, w)


def sample_random_pixels(
    texture: torch.Tensor,
    batch_size: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample paper texture batches from discrete pixel locations.

    Coordinates use the same inclusive ``[0,1]`` convention as grid encoders,
    so the first and last texels map exactly to the domain boundaries.
    """

    if texture.ndim != 3:
        raise ValueError("texture must have shape (H, W, C)")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    height, width, channels = texture.shape
    flat = torch.randint(
        height * width,
        (batch_size,),
        generator=generator,
    )
    y = torch.div(flat, width, rounding_mode="floor")
    x = torch.remainder(flat, width)
    coords = torch.stack(
        (
            x.to(torch.float32) / (width - 1),
            y.to(torch.float32) / (height - 1),
        ),
        dim=1,
    )
    targets = texture.reshape(-1, channels).index_select(0, flat)
    return coords, targets


def sample_texture_tensor(
    texture: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    """Bilinearly filter an ``HWC`` texture at arbitrary ``(u,v)`` coordinates."""

    if texture.ndim != 3:
        raise ValueError("texture must have shape (H, W, C)")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if coords.numel() and ((coords < 0).any() or (coords > 1).any()):
        raise ValueError("coords must lie in [0, 1]")
    source = texture.to(device=coords.device, dtype=coords.dtype)
    query = (coords * 2.0 - 1.0).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(
        source.permute(2, 0, 1).unsqueeze(0),
        query,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()


def texture_map_metric_rows(
    prediction: torch.Tensor,
    loaded: LoadedTextureSet,
    *,
    metrics: Iterable[str] = ("psnr", "ssim"),
    clamp_prediction: bool = True,
) -> list[dict[str, object]]:
    """Evaluate every RGB map independently, as required by paper Table 2."""

    if prediction.shape != loaded.tensor.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} does not match "
            f"target {tuple(loaded.tensor.shape)}"
        )
    from peps.metrics import psnr, ssim

    functions = {"psnr": psnr, "ssim": ssim}
    names = tuple(metrics)
    unknown = sorted(set(names) - set(functions))
    if unknown:
        raise ValueError(f"unsupported texture metrics: {unknown}")
    evaluated = prediction.clamp(0.0, 1.0) if clamp_prediction else prediction
    rows: list[dict[str, object]] = []
    for texture_map in loaded.maps:
        predicted_map = evaluated[..., texture_map.channel_slice]
        target_map = loaded.tensor[..., texture_map.channel_slice]
        for metric_name in names:
            rows.append(
                {
                    "texture_set": loaded.set_id,
                    "map_id": texture_map.map_id,
                    "semantic": texture_map.semantic,
                    "metric": metric_name,
                    "value": functions[metric_name](predicted_map, target_map),
                }
            )
    return rows


def aggregate_texture_map_metrics(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Mean per-map values globally and by the eight paper texture types."""

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        metric_name = str(row["metric"])
        semantic = str(row["semantic"])
        value = float(row["value"])
        grouped[(metric_name, semantic)].append(value)
        grouped[(metric_name, "global")].append(value)
    return [
        {
            "metric": metric_name,
            "semantic": semantic,
            "count": len(values),
            "mean": sum(values) / len(values),
        }
        for (metric_name, semantic), values in sorted(grouped.items())
    ]


def find_bundle(name: str, root: str | None = None) -> str:
    """Directory for a downloaded AmbientCG set under data/raw/ambientcg/<name>."""
    if root is None:
        here = os.path.dirname(__file__)
        root = os.path.abspath(os.path.join(here, "..", "..", "data", "raw", "ambientcg"))
    return os.path.join(root, name)
