"""Learned grid encoder with bilinear (2D) / trilinear (3D) interpolation.

繁體中文:可學習的 grid encoder。在一個 (H x W [x D]) 的特徵網格上,對每個
查詢座標做雙線性(2D)或三線性(3D)內插,取出長度為 feature_dim 的 latent。
座標輸入預期為 [0, 1]。這是 PEPS 取樣的「共享」編碼器,所有興趣點都取樣同一個
grid,梯度因此回流到整個 grid。

單解析度版本 (GridEncoder) 已足夠重現論文的 BI/TI-grid 與 Grid-PEPS。
多解析度 / hash 版本留待 SDF 章節 (W09) 擴充。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GridEncoder(nn.Module):
    """Dense learnable feature grid, sampled by (bi/tri)linear interpolation.

    Args:
        dim: spatial dimensionality (2 for images/textures, 3 for SDF).
        resolution: int or tuple. Grid side length(s). For dim=2 a single int
            gives a square grid; a tuple ``(H, W)`` sets each axis.
        feature_dim: latent channels ``k`` stored per grid vertex.
        align_corners: passed to grid_sample; True maps [0,1] to the vertex
            centers inclusive of the corners (standard for INR grids).
        init_std: std of the Gaussian used to initialize grid features.

    Shape:
        input  ``coords``: ``(N, dim)`` in ``[0, 1]``.
        output ``latent``: ``(N, feature_dim)``.
    """

    def __init__(
        self,
        dim: int,
        resolution,
        feature_dim: int = 2,
        align_corners: bool = True,
        init_std: float = 1e-2,
    ) -> None:
        super().__init__()
        if dim not in (2, 3):
            raise ValueError("GridEncoder supports dim in {2, 3}")
        self.dim = dim
        self.feature_dim = feature_dim
        self.align_corners = align_corners

        if isinstance(resolution, int):
            resolution = (resolution,) * dim
        if len(resolution) != dim:
            raise ValueError(f"resolution must have {dim} entries, got {resolution}")
        self.resolution = tuple(int(r) for r in resolution)

        # Grid stored as (1, feature_dim, [D,] H, W) for grid_sample.
        if dim == 2:
            H, W = self.resolution
            grid = torch.randn(1, feature_dim, H, W) * init_std
        else:
            D, H, W = self.resolution
            grid = torch.randn(1, feature_dim, D, H, W) * init_std
        self.grid = nn.Parameter(grid)

    @property
    def num_params(self) -> int:
        return self.grid.numel()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (N, dim) in [0,1] -> grid_sample expects [-1, 1]
        n = coords.shape[0]
        g = coords * 2.0 - 1.0  # to [-1, 1]

        if self.dim == 2:
            # grid_sample 2D wants grid of shape (1, N, 1, 2), coords order (x, y)
            # our coords are (row=y? col=x?). We treat coords[...,0] as x-axis (W)
            # and coords[...,1] as y-axis (H) to match (x, y) convention.
            samp = g.view(1, n, 1, 2)
            out = F.grid_sample(
                self.grid, samp, mode="bilinear",
                align_corners=self.align_corners, padding_mode="border",
            )  # (1, feature_dim, N, 1)
            return out.view(self.feature_dim, n).t().contiguous()
        else:
            # 3D: grid of shape (1, N, 1, 1, 3), coords order (x, y, z)
            samp = g.view(1, n, 1, 1, 3)
            out = F.grid_sample(
                self.grid, samp, mode="bilinear",
                align_corners=self.align_corners, padding_mode="border",
            )  # (1, feature_dim, N, 1, 1)
            return out.view(self.feature_dim, n).t().contiguous()

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, resolution={self.resolution}, "
            f"feature_dim={self.feature_dim}, num_params={self.num_params}"
        )
