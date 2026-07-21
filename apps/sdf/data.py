"""SDF sampling — procedural shapes and mesh-to-SDF.

繁體中文:SDF 取樣。座標定義在 [0,1]^3(內部轉到 [-1,1] 算距離)。
提供球、torus 的解析 SDF(免下載即可跑),以及用 mesh-to-sdf 從真實網格取樣。
另提供 make_query_grid 產生 R^3 密集查詢網格,供 IoU 與 marching cubes 使用。
"""

from __future__ import annotations

import numpy as np
import torch


def _to_centered(coords: torch.Tensor) -> torch.Tensor:
    """[0,1]^3 -> [-1,1]^3."""
    return coords * 2.0 - 1.0


def sample_sphere_sdf(n: int, radius: float = 0.6, seed: int = 0):
    """Random points in [0,1]^3 with signed distance to a centered sphere."""
    g = torch.Generator().manual_seed(seed)
    coords = torch.rand(n, 3, generator=g)
    p = _to_centered(coords)
    sdf = p.norm(dim=1, keepdim=True) - radius
    return coords, sdf


def sample_torus_sdf(n: int, R: float = 0.5, r: float = 0.2, seed: int = 0):
    """Signed distance to a torus (major R, minor r) centered in [-1,1]^3."""
    g = torch.Generator().manual_seed(seed)
    coords = torch.rand(n, 3, generator=g)
    p = _to_centered(coords)
    qx = torch.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) - R
    sdf = (torch.sqrt(qx ** 2 + p[:, 1] ** 2) - r).unsqueeze(1)
    return coords, sdf


def sample_mesh_sdf(mesh_path: str, n: int = 200000, seed: int = 0):
    """Sample signed distances from a real mesh via the ``mesh-to-sdf`` package.

    Returns coords in [0,1]^3 (mesh normalized to the unit cube) and SDF values.
    Requires ``trimesh`` and ``mesh-to-sdf``.
    """
    import trimesh
    from mesh_to_sdf import mesh_to_sdf as _m2s

    mesh = trimesh.load(mesh_path, force="mesh")
    # normalize mesh to fit in [-1,1]^3
    mesh.vertices -= mesh.vertices.mean(0)
    scale = np.abs(mesh.vertices).max()
    mesh.vertices /= scale

    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    sdf = _m2s(mesh, pts)
    coords = torch.from_numpy((pts + 1.0) / 2.0)          # -> [0,1]
    return coords, torch.from_numpy(sdf).unsqueeze(1).float()


def make_query_grid(res: int = 64) -> tuple[torch.Tensor, tuple]:
    """Dense [0,1]^3 query grid, ``(res^3, 3)`` coords + the ``(res,res,res)`` shape."""
    lin = torch.linspace(0, 1, res)
    gz, gy, gx = torch.meshgrid(lin, lin, lin, indexing="ij")
    coords = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)
    return coords, (res, res, res)


def occupancy(sdf_values: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Boolean inside-surface occupancy grid from SDF values (sdf < 0)."""
    return (sdf_values.reshape(shape) < 0)
