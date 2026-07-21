"""SDF sampling — procedural shapes and mesh-to-SDF.

繁體中文:SDF 取樣。座標定義在 [0,1]^3(內部轉到 [-1,1] 算距離)。
提供球、torus 的解析 SDF(免下載即可跑),以及用 mesh-to-sdf 從真實網格取樣。
另提供 make_query_grid 產生 R^3 密集查詢網格,供 IoU 與 marching cubes 使用。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterator, Mapping
import warnings

import numpy as np
import torch
import torch.nn.functional as F


MODEL_DOMAIN = (0.0, 1.0)
CENTERED_DOMAIN = (-1.0, 1.0)
# Distances in the paper volumes and analytic teaching shapes are measured
# after mapping [0,1]^3 to [-1,1]^3. Therefore df/d(model_coord) has norm 2.
SDF_COORDINATE_SCALE = 2.0


@dataclass(frozen=True)
class PaperSDFVolume:
    """One provenance-validated paper SDF volume."""

    asset_id: str
    values: np.ndarray
    volume_path: Path
    provenance_path: Path
    provenance: Mapping[str, Any]


def _to_centered(coords: torch.Tensor) -> torch.Tensor:
    """[0,1]^3 -> [-1,1]^3."""
    return coords * SDF_COORDINATE_SCALE - 1.0


def _to_model_coords(centered: torch.Tensor) -> torch.Tensor:
    """[-1,1]^3 -> [0,1]^3."""

    return (centered + 1.0) / SDF_COORDINATE_SCALE


def sample_sphere_sdf(n: int, radius: float = 0.6, seed: int = 0):
    """Random points in [0,1]^3 with signed distance to a centered sphere."""
    g = torch.Generator().manual_seed(seed)
    coords = torch.rand(n, 3, generator=g)
    p = _to_centered(coords)
    sdf = p.norm(dim=1, keepdim=True) - radius
    return coords, sdf


def _torus_sdf(coords: torch.Tensor, R: float, r: float) -> torch.Tensor:
    p = _to_centered(coords)
    qx = torch.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) - R
    return (torch.sqrt(qx ** 2 + p[:, 1] ** 2) - r).unsqueeze(1)


def sample_torus_sdf(n: int, R: float = 0.5, r: float = 0.2, seed: int = 0):
    """Signed distance to a torus (major R, minor r) centered in [-1,1]^3."""
    g = torch.Generator().manual_seed(seed)
    coords = torch.rand(n, 3, generator=g)
    return coords, _torus_sdf(coords, R, r)


def sample_torus_sdf_near_surface(
    n: int, R: float = 0.5, r: float = 0.2, seed: int = 0,
    near_frac: float = 0.7, near_sigma: float = 0.05,
):
    """Torus SDF with **near-surface importance sampling**.

    An SDF trained on uniform points wastes capacity on empty space and blurs the
    zero-level set (the only part marching cubes renders). Following DeepSDF/IGR,
    we draw ``near_frac`` of points close to the surface and the rest uniformly.

    Near-surface points are made by jittering surface points (found by projecting
    uniform samples onto the torus analytically) with Gaussian noise ``near_sigma``.

    Returns ``(coords (n,3) in [0,1], sdf (n,1))``.
    """
    g = torch.Generator().manual_seed(seed)
    n_near = int(round(n * near_frac))
    n_far = n - n_near

    # --- surface points on the torus, then jitter ---
    # sample two angles, place points exactly on the torus in centered [-1,1]^3
    theta = torch.rand(n_near, generator=g) * 2 * math.pi   # around tube
    phi = torch.rand(n_near, generator=g) * 2 * math.pi     # around center
    cx = (R + r * torch.cos(theta)) * torch.cos(phi)
    cz = (R + r * torch.cos(theta)) * torch.sin(phi)
    cy = r * torch.sin(theta)
    surf = torch.stack([cx, cy, cz], dim=1)                 # (n_near,3) in [-1,1]
    surf = surf + near_sigma * torch.randn(n_near, 3, generator=g)
    near_coords = (surf + 1.0) / 2.0                        # -> [0,1]

    far_coords = torch.rand(n_far, 3, generator=g)
    coords = torch.cat([near_coords, far_coords], dim=0).clamp(0.0, 1.0)
    # analytic SDF for every point (exact labels)
    sdf = _torus_sdf(coords, R, r)
    return coords, sdf


def sample_mesh_sdf(mesh_path: str, n: int = 200000, seed: int = 0):
    """Sample signed distances from a real mesh via the ``mesh-to-sdf`` package.

    Returns coords in [0,1]^3 (mesh normalized to the unit cube) and SDF values.
    Requires ``trimesh`` and ``mesh-to-sdf``.
    """
    import trimesh
    from mesh_to_sdf import mesh_to_sdf as _m2s

    mesh = trimesh.load(mesh_path, force="mesh")
    # Match data/preprocess_sdf.py: center the AABB and make the longest extent
    # span exactly [-1,1]. Vertex-mean centering changes the paper query domain
    # for asymmetric meshes and was the old course-path discrepancy.
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    longest = float(extents.max())
    if not np.isfinite(bounds).all() or longest <= 0:
        raise ValueError("mesh has invalid or degenerate bounds")
    center = bounds.mean(axis=0)
    mesh.vertices = (np.asarray(mesh.vertices) - center) * (2.0 / longest)

    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    sdf = _m2s(mesh, pts)
    coords = torch.from_numpy((pts + 1.0) / SDF_COORDINATE_SCALE)
    return coords, torch.from_numpy(sdf).unsqueeze(1).float()


def sample_sdf_tensor(
    volume: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    """Trilinearly sample a ``(D,H,W)`` SDF at ``(x,y,z)`` model coordinates.

    The processed paper volume uses inclusive samples and ``zyx`` storage, so
    ``align_corners=True`` is required for exact voxel-center agreement.
    Boundary values are sampled directly; no half-texel shift or wrapping is
    applied.
    """

    if volume.ndim != 3:
        raise ValueError("volume must have shape (D, H, W)")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    if not coords.is_floating_point():
        raise TypeError("coords must be floating point")
    if coords.numel() and ((coords < 0).any() or (coords > 1).any()):
        raise ValueError("coords must lie in [0, 1]")
    source = volume.to(device=coords.device, dtype=coords.dtype)
    query = (coords * 2.0 - 1.0).reshape(1, -1, 1, 1, 3)
    sampled = F.grid_sample(
        source.reshape(1, 1, *source.shape),
        query,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(-1, 1)


def sample_sdf_volume(
    volume: np.ndarray,
    coords: torch.Tensor,
) -> torch.Tensor:
    """NumPy/memmap convenience wrapper for :func:`sample_sdf_tensor`."""

    array = np.asarray(volume)
    if array.dtype != np.float32:
        raise ValueError("paper SDF volumes must be float32")
    if array.flags.writeable:
        tensor = torch.from_numpy(array)
    else:
        # PyTorch warns because writes through this view would be undefined.
        # This function only reads the volume; preserving the memmap avoids a
        # hidden 512^3 (512 MiB) copy per sampling call.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given NumPy array is not writable",
            )
            tensor = torch.from_numpy(array)
    return sample_sdf_tensor(tensor, coords)


def load_paper_sdf_volume(
    asset_id: str,
    *,
    processed_root: str | Path | None = None,
    verify_checksum: bool = True,
) -> PaperSDFVolume:
    """Load one of the four named 512^3 volumes with its provenance receipt."""

    from data.preprocess_sdf import DEFAULT_OUTPUT_ROOT, load_sdf_volume

    allowed = {"lucy", "pitted-stonefish", "thai-statue", "armadillo"}
    if asset_id not in allowed:
        raise ValueError(f"unknown paper SDF asset {asset_id!r}")
    root = DEFAULT_OUTPUT_ROOT if processed_root is None else Path(processed_root)
    volume_path = root / asset_id / "sdf_512.npy"
    provenance_path = volume_path.with_suffix(".provenance.json")
    values = load_sdf_volume(
        volume_path,
        provenance_path,
        require_paper_protocol=True,
        verify_checksum=verify_checksum,
    )
    import json

    with provenance_path.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    return PaperSDFVolume(
        asset_id=asset_id,
        values=values,
        volume_path=volume_path,
        provenance_path=provenance_path,
        provenance=provenance,
    )


def iter_query_slabs(
    resolution: int = 512,
    *,
    slab_depth: int = 1,
) -> Iterator[tuple[slice, torch.Tensor]]:
    """Yield dense ``(x,y,z)`` coordinates without allocating a full 512^3 grid."""

    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    if slab_depth < 1:
        raise ValueError("slab_depth must be positive")
    line = torch.linspace(0.0, 1.0, resolution)
    for start in range(0, resolution, slab_depth):
        stop = min(resolution, start + slab_depth)
        z = line[start:stop]
        gz, gy, gx = torch.meshgrid(z, line, line, indexing="ij")
        coords = torch.stack(
            (gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)),
            dim=1,
        )
        yield slice(start, stop), coords


def make_query_grid(res: int = 64) -> tuple[torch.Tensor, tuple]:
    """Dense [0,1]^3 query grid, ``(res^3, 3)`` coords + the ``(res,res,res)`` shape."""
    lin = torch.linspace(0, 1, res)
    gz, gy, gx = torch.meshgrid(lin, lin, lin, indexing="ij")
    coords = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)
    return coords, (res, res, res)


def occupancy(sdf_values: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Boolean inside-surface occupancy grid from SDF values (sdf < 0)."""
    return (sdf_values.reshape(shape) < 0)
