"""Application 3 — signed distance functions (extension track).

繁體中文:應用三,有號距離函數(延伸)。把形狀表示為 f(x,y,z)->有號距離,
用座標網路擬合。提供程序生成形狀(球/torus,免下載即可跑)與真實網格→SDF 取樣
(mesh-to-sdf,給 Stanford Armadillo/Thai Statue)。對照 TI-grid / hash / multi-res
及其 PEPS 版本,重現 Table 3(IoU)與 Table 4 困難實例。
"""

from .data import sample_sphere_sdf, sample_torus_sdf, sample_mesh_sdf, make_query_grid
from .build import build_sdf_grid, build_sdf_multires, build_sdf_hash, build_sdf_peps

__all__ = [
    "sample_sphere_sdf",
    "sample_torus_sdf",
    "sample_mesh_sdf",
    "make_query_grid",
    "build_sdf_grid",
    "build_sdf_multires",
    "build_sdf_hash",
    "build_sdf_peps",
]
