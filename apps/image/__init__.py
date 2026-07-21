"""Application 1 — implicit image representation.

繁體中文:應用一,隱式影像表示。把一張影像視為函式 f(x,y)->RGB,用座標網路擬合。
提供影像載入、座標/目標張量建構、以及建立各種 encoder 的工廠函式,供 W01-W07 使用。
"""

from .data import load_image, image_to_coords_targets, coords_grid
from .build import build_plain_mlp, build_grid, build_grid_peps

__all__ = [
    "load_image",
    "image_to_coords_targets",
    "coords_grid",
    "build_plain_mlp",
    "build_grid",
    "build_grid_peps",
]
