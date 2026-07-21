"""PBR material bundle <-> coordinate/target tensors.

繁體中文:把一組 AmbientCG PBR 貼圖疊成一個多通道 bundle。NTC 定義的 bundle:
albedo(RGB) + normal(XY) + roughness + metalness + AO ≈ 9 通道。座標正規化 [0,1]。
下載內容位於 data/raw/ambientcg/<Name>/,檔名如 <Name>_2K-PNG_Color.png 等。
"""

from __future__ import annotations

import glob
import os
import numpy as np
import torch
from PIL import Image

# Channel layout of the 9-channel NTC bundle. Each entry: (suffix, channels).
# Normal uses only X,Y (Z reconstructed as sqrt(1-x^2-y^2) at decode).
CHANNEL_LAYOUT = [
    ("Color", 3),          # albedo RGB
    ("NormalGL", 3),       # normal XYZ (GL convention, full RGB)
    ("Roughness", 1),
    ("Metalness", 1),
    ("AmbientOcclusion", 1),
]
TOTAL_CHANNELS = sum(c for _, c in CHANNEL_LAYOUT)  # = 9


def _find_map(bundle_dir: str, suffix: str) -> str | None:
    hits = glob.glob(os.path.join(bundle_dir, f"*_{suffix}.png"))
    return hits[0] if hits else None


def load_pbr_bundle(bundle_dir: str, size: int = 512) -> torch.Tensor:
    """Load a PBR set as ``(H, W, 9)`` float tensor in ``[0, 1]``.

    Missing maps (e.g. no Metalness for a dielectric) are filled with a constant
    (0 for metalness, 1 for AO) so the bundle always has 9 channels.
    """
    planes = []
    defaults = {"Metalness": 0.0, "AmbientOcclusion": 1.0, "Roughness": 0.5}
    for suffix, nch in CHANNEL_LAYOUT:
        path = _find_map(bundle_dir, suffix)
        if path is None:
            fill = defaults.get(suffix, 0.0)
            planes.append(torch.full((size, size, nch), fill))
            continue
        img = Image.open(path).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)  # (H,W,3)
        planes.append(arr[..., :nch])
    return torch.cat(planes, dim=-1)  # (H, W, 9)


def bundle_to_coords_targets(bundle: torch.Tensor):
    """Return ``(coords (H*W,2), targets (H*W,9), (H, W))``."""
    h, w, c = bundle.shape
    ys = torch.linspace(0, 1, h)
    xs = torch.linspace(0, 1, w)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    targets = bundle.reshape(h * w, c)
    return coords, targets, (h, w)


def find_bundle(name: str, root: str | None = None) -> str:
    """Directory for a downloaded AmbientCG set under data/raw/ambientcg/<name>."""
    if root is None:
        here = os.path.dirname(__file__)
        root = os.path.abspath(os.path.join(here, "..", "..", "data", "raw", "ambientcg"))
    return os.path.join(root, name)
