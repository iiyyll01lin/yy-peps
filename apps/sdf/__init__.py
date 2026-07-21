"""Application 3 — course analytic SDFs and paper 512^3 volumes.

The paper path exposes all Table 3 encoders and exact 1x/8x Table 4 builders;
the procedural sphere/torus helpers remain the explicitly separate course path.
"""

from .data import (
    SDF_COORDINATE_SCALE,
    PaperSDFVolume,
    iter_query_slabs,
    load_paper_sdf_volume,
    make_query_grid,
    sample_mesh_sdf,
    sample_sdf_tensor,
    sample_sdf_volume,
    sample_sphere_sdf,
    sample_torus_sdf,
)
from .build import (
    build_paper_sdf,
    build_sdf_grid,
    build_sdf_hash,
    build_sdf_multires,
    build_sdf_peps,
)

__all__ = [
    "sample_sphere_sdf",
    "sample_torus_sdf",
    "sample_mesh_sdf",
    "sample_sdf_tensor",
    "sample_sdf_volume",
    "load_paper_sdf_volume",
    "iter_query_slabs",
    "PaperSDFVolume",
    "SDF_COORDINATE_SCALE",
    "make_query_grid",
    "build_sdf_grid",
    "build_sdf_multires",
    "build_sdf_hash",
    "build_sdf_peps",
    "build_paper_sdf",
]
