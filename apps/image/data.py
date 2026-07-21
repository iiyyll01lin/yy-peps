"""Image <-> coordinate/target tensors.

繁體中文:影像與(座標,目標)張量的互轉工具。座標正規化到 [0,1],目標為 RGB in [0,1]。
"""

from __future__ import annotations

import os
import numpy as np
import torch
from PIL import Image


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
