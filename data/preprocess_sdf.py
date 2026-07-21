"""Convert paper meshes into chunked, provenance-tracked 512^3 SDF volumes.

The checked-in manifest fixes normalization, grid layout, sign convention, and
the default ``mesh-to-sdf`` settings. No fallback method is selected silently.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.manifest import (  # noqa: E402
    DATA_DIR,
    DEFAULT_RAW_ROOT,
    DataIntegrityError,
    ManifestError,
    MissingDataError,
    hash_file,
    load_manifest,
    resolve_local_path,
    verify_file,
)


DEFAULT_OUTPUT_ROOT = DATA_DIR / "processed" / "sdf"


@dataclass(frozen=True)
class SDFConfig:
    resolution: int
    surface_point_method: str
    sign_method: str
    scan_count: int
    scan_resolution: int
    sample_point_count: int
    normal_sample_count: int
    seed: int
    max_query_points: int
    query_workers: int
    pyopengl_platform: str


class _SampledSurfacePointCloud:
    """Parallel cKDTree query with mesh-to-sdf's normal-majority sign rule."""

    def __init__(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        *,
        workers: int,
    ) -> None:
        from scipy.spatial import cKDTree

        self.points = np.ascontiguousarray(points)
        self.normals = np.ascontiguousarray(normals)
        self.workers = workers
        self.tree = cKDTree(self.points, compact_nodes=True, balanced_tree=True)

    def get_sdf_in_batches(
        self,
        query_points: np.ndarray,
        *,
        use_depth_buffer: bool,
        sample_count: int,
        batch_size: int,
    ) -> np.ndarray:
        if use_depth_buffer:
            raise ManifestError("sample surface points do not support depth signs")
        batches = []
        for start in range(0, len(query_points), batch_size):
            query = query_points[start : start + batch_size]
            distances, indices = self.tree.query(
                query,
                k=sample_count,
                workers=self.workers,
            )
            if sample_count == 1:
                distances = distances[:, None]
                indices = indices[:, None]
            closest = self.points[indices]
            direction = query[:, None, :] - closest
            inside_votes = (
                np.einsum("ijk,ijk->ij", direction, self.normals[indices]) < 0
            )
            inside = inside_votes.sum(axis=1) > sample_count * 0.5
            signed = distances[:, 0].astype(np.float32, copy=False)
            signed[inside] *= -1
            batches.append(signed)
        return np.concatenate(batches)


def sdf_asset_spec(
    asset_id: str,
    manifest: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    source = load_manifest("sdf") if manifest is None else manifest
    for item in source["assets"]:
        if item["id"] == asset_id:
            return item
    raise ManifestError(f"unknown SDF asset {asset_id!r}")


def normalize_mesh(mesh: Any) -> tuple[Any, dict[str, Any]]:
    """Center a mesh bounding box and scale its largest extent to ``[-1, 1]``."""

    import trimesh

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise DataIntegrityError("mesh has invalid bounds")
    extents = bounds[1] - bounds[0]
    max_extent = float(extents.max())
    if max_extent <= 0:
        raise DataIntegrityError("mesh has zero spatial extent")
    center = bounds.mean(axis=0)
    scale = 2.0 / max_extent
    vertices = (np.asarray(mesh.vertices, dtype=np.float64) - center) * scale
    normalized = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(mesh.faces),
        process=False,
        validate=False,
    )
    transform = {
        "method": "axis_aligned_bounding_box",
        "original_bounds": bounds.tolist(),
        "original_extents": extents.tolist(),
        "center_subtracted": center.tolist(),
        "isotropic_scale": scale,
        "normalized_domain": [-1.0, 1.0],
        "formula": "p_normalized = (p - bbox_center) * isotropic_scale",
    }
    return normalized, transform


def preprocess_sdf(
    asset_id: str,
    *,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    output_path: str | Path,
    provenance_path: str | Path,
    config: SDFConfig,
) -> dict[str, Any]:
    """Build one SDF volume atomically and return its provenance record."""

    manifest = load_manifest("sdf")
    asset = sdf_asset_spec(asset_id, manifest)
    mesh_path = resolve_local_path(raw_root, asset["mesh"])
    input_file = _verify_input_mesh(asset, mesh_path)
    started = datetime.now(timezone.utc)

    os.environ.setdefault("PYOPENGL_PLATFORM", config.pyopengl_platform)
    import trimesh

    loaded = trimesh.load(mesh_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_geometry()
    if not isinstance(loaded, trimesh.Trimesh):
        raise DataIntegrityError(f"{mesh_path}: did not contain triangle geometry")
    if loaded.faces.ndim != 2 or loaded.faces.shape[1] != 3:
        raise DataIntegrityError(f"{mesh_path}: mesh is not triangulated")

    expected_vertices = asset["mesh"].get("vertex_count")
    expected_faces = asset["mesh"].get("face_count")
    if expected_vertices is not None and len(loaded.vertices) != expected_vertices:
        raise DataIntegrityError(
            f"{mesh_path}: expected {expected_vertices} vertices, "
            f"found {len(loaded.vertices)}"
        )
    if expected_faces is not None and len(loaded.faces) != expected_faces:
        raise DataIntegrityError(
            f"{mesh_path}: expected {expected_faces} faces, found {len(loaded.faces)}"
        )

    mesh_metadata: dict[str, Any] = {
        "vertices": int(len(loaded.vertices)),
        "faces": int(len(loaded.faces)),
    }
    if len(loaded.faces) <= 2_000_000:
        mesh_metadata.update(
            {
                "watertight": bool(loaded.is_watertight),
                "winding_consistent": bool(loaded.is_winding_consistent),
                "body_count": int(loaded.body_count),
                "topology_check": "completed",
            }
        )
    else:
        mesh_metadata.update(
            {
                "watertight": None,
                "winding_consistent": None,
                "body_count": None,
                "topology_check": (
                    "skipped above 2M faces to avoid a large temporary edge graph"
                ),
            }
        )
    normalized, transform = normalize_mesh(loaded)
    point_cloud = _build_surface_point_cloud(normalized, config)

    output = Path(output_path)
    provenance = Path(provenance_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.part-{os.getpid()}")
    temporary.unlink(missing_ok=True)

    resolution = config.resolution
    volume = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(resolution, resolution, resolution),
    )
    values = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    slab_depth = max(
        1,
        min(resolution, config.max_query_points // (resolution * resolution)),
    )
    minimum = float("inf")
    maximum = float("-inf")
    negative_count = 0
    positive_count = 0
    max_neighbor_delta = 0.0
    previous_z_plane: np.ndarray | None = None
    try:
        for z_start in range(0, resolution, slab_depth):
            z_stop = min(resolution, z_start + slab_depth)
            zz, yy, xx = np.meshgrid(
                values[z_start:z_stop],
                values,
                values,
                indexing="ij",
            )
            query = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
            sdf = point_cloud.get_sdf_in_batches(
                query,
                use_depth_buffer=config.sign_method == "depth",
                sample_count=config.normal_sample_count,
                batch_size=config.max_query_points,
            )
            sdf = np.asarray(sdf, dtype=np.float32)
            if sdf.shape != (query.shape[0],) or not np.isfinite(sdf).all():
                raise DataIntegrityError("mesh-to-sdf returned invalid values")
            slab = sdf.reshape(z_stop - z_start, resolution, resolution)
            volume[z_start:z_stop] = slab
            if slab.shape[0] > 1:
                max_neighbor_delta = max(
                    max_neighbor_delta,
                    float(np.abs(np.diff(slab, axis=0)).max()),
                )
            max_neighbor_delta = max(
                max_neighbor_delta,
                float(np.abs(np.diff(slab, axis=1)).max()),
                float(np.abs(np.diff(slab, axis=2)).max()),
            )
            if previous_z_plane is not None:
                max_neighbor_delta = max(
                    max_neighbor_delta,
                    float(np.abs(slab[0] - previous_z_plane).max()),
                )
            previous_z_plane = slab[-1].copy()
            minimum = min(minimum, float(sdf.min()))
            maximum = max(maximum, float(sdf.max()))
            negative_count += int(np.count_nonzero(sdf < 0))
            positive_count += int(np.count_nonzero(sdf > 0))
            print(f"[sdf] {asset_id}: z={z_stop}/{resolution}", flush=True)
        volume.flush()
    except Exception:
        del volume
        temporary.unlink(missing_ok=True)
        raise
    del volume

    if negative_count == 0 or positive_count == 0:
        temporary.unlink(missing_ok=True)
        raise DataIntegrityError(
            "generated volume does not contain both interior and exterior values; "
            "check mesh winding and sign method"
        )
    grid_spacing = 2.0 / (resolution - 1)
    continuity_limit = 1.5 * grid_spacing
    if max_neighbor_delta > continuity_limit:
        temporary.unlink(missing_ok=True)
        raise DataIntegrityError(
            "generated volume violates the SDF continuity check: maximum adjacent "
            f"delta {max_neighbor_delta:.6g} exceeds {continuity_limit:.6g}"
        )
    os.replace(temporary, output)

    output_spec = {
        "bytes": output.stat().st_size,
        "checksum": {
            "algorithm": "sha256",
            "value": hash_file(output, "sha256"),
        },
    }
    finished = datetime.now(timezone.utc)
    paper_protocol = manifest["preprocessing"]
    canonical = _is_canonical_config(config, paper_protocol)
    record = {
        "schema_version": 1,
        "asset_id": asset_id,
        "paper_name": asset["paper_name"],
        "canonical_paper_protocol": canonical,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "input": {
            "path": _portable_path(mesh_path),
            **input_file,
            "mesh": mesh_metadata,
        },
        "normalization": transform,
        "grid": {
            "shape": [resolution, resolution, resolution],
            "axis_order": "zyx",
            "query_component_order": "xyz",
            "sample_locations": "inclusive linspace(-1, 1, resolution)",
            "coordinate_mapping": "paper coordinates [0,1]^3 = (xyz + 1) / 2",
            "distance_units": "normalized [-1,1] coordinate units",
            "dtype": "float32",
            "sign_convention": "negative_inside",
        },
        "algorithm": {
            "implementation": "mesh-to-sdf",
            **asdict(config),
            "automatic_fallback": False,
        },
        "output": {
            "path": _portable_path(output),
            **output_spec,
            "minimum": minimum,
            "maximum": maximum,
            "negative_fraction": negative_count / (resolution**3),
            "positive_fraction": positive_count / (resolution**3),
            "grid_spacing": grid_spacing,
            "max_neighbor_delta": max_neighbor_delta,
            "max_neighbor_delta_over_spacing": max_neighbor_delta / grid_spacing,
            "continuity_limit_over_spacing": 1.5,
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": importlib_metadata.version("scipy"),
            "trimesh": importlib_metadata.version("trimesh"),
            "mesh-to-sdf": importlib_metadata.version("mesh-to-sdf"),
        },
        "paper_protocol": paper_protocol,
        "known_limit": (
            "The paper states that an unreleased C++/HIP converter was used; "
            "mesh-to-sdf is a documented reproduction protocol, not a bit-exact "
            "implementation of the authors' converter."
        ),
    }
    _atomic_json(provenance, record)
    return record


def load_sdf_volume(
    volume_path: str | Path,
    provenance_path: str | Path,
    *,
    require_paper_protocol: bool = True,
    verify_checksum: bool = True,
) -> np.ndarray:
    """Strictly open an SDF volume only when its provenance validates."""

    volume_file = Path(volume_path)
    provenance_file = Path(provenance_path)
    if not provenance_file.is_file():
        raise MissingDataError(f"missing SDF provenance: {provenance_file}")
    with provenance_file.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    if record.get("schema_version") != 1:
        raise DataIntegrityError("unsupported SDF provenance schema")
    if require_paper_protocol and not record.get("canonical_paper_protocol"):
        raise DataIntegrityError("SDF volume was not built with the paper protocol")
    grid = record.get("grid", {})
    if require_paper_protocol:
        protocol = load_manifest("sdf")["preprocessing"]
        expected_shape = [protocol["resolution"]] * 3
        if grid.get("shape") != expected_shape:
            raise DataIntegrityError(
                f"paper SDF volume must have shape {expected_shape}"
            )
        algorithm = record.get("algorithm", {})
        for key in (
            "resolution",
            "surface_point_method",
            "sign_method",
            "scan_count",
            "scan_resolution",
            "sample_point_count",
            "normal_sample_count",
            "seed",
        ):
            if algorithm.get(key) != protocol[key]:
                raise DataIntegrityError(
                    f"SDF provenance differs from paper protocol at {key}"
                )
    if grid.get("axis_order") != "zyx":
        raise DataIntegrityError("SDF provenance has an unexpected axis order")
    if grid.get("sign_convention") != "negative_inside":
        raise DataIntegrityError("SDF provenance has an unexpected sign convention")
    if grid.get("dtype") != "float32":
        raise DataIntegrityError("SDF provenance has an unexpected dtype")

    output = record.get("output", {})
    if verify_checksum:
        verify_file(volume_file, output)
    elif not volume_file.is_file():
        raise MissingDataError(f"missing SDF volume: {volume_file}")
    array = np.load(volume_file, mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(int(v) for v in grid.get("shape", []))
    if array.shape != expected_shape or array.dtype != np.float32:
        raise DataIntegrityError(
            f"{volume_file}: expected {expected_shape} float32, "
            f"found {array.shape} {array.dtype}"
        )
    return array


def _build_surface_point_cloud(mesh: Any, config: SDFConfig) -> Any:
    if config.surface_point_method == "scan":
        from mesh_to_sdf import get_surface_point_cloud

        return get_surface_point_cloud(
            mesh,
            surface_point_method="scan",
            bounding_radius=3**0.5,
            scan_count=config.scan_count,
            scan_resolution=config.scan_resolution,
            calculate_normals=config.sign_method == "normal",
        )
    if config.surface_point_method == "sample":
        if config.sign_method != "normal":
            raise ManifestError("sample surface points require sign_method='normal'")
        from trimesh.sample import sample_surface

        points, face_indices = sample_surface(
            mesh,
            count=config.sample_point_count,
            seed=config.seed,
        )
        normals = mesh.face_normals[face_indices]
        return _SampledSurfacePointCloud(
            points,
            normals,
            workers=config.query_workers,
        )
    raise ManifestError(
        f"unknown surface-point method {config.surface_point_method!r}"
    )


def _verify_input_mesh(
    asset: Mapping[str, Any],
    mesh_path: Path,
) -> dict[str, Any]:
    mesh_spec = asset["mesh"]
    if mesh_spec.get("checksum") is not None:
        verify_file(mesh_path, mesh_spec)
        return {
            "bytes": mesh_path.stat().st_size,
            "checksum": mesh_spec["checksum"],
            "checksum_source": "checked-in manifest",
        }

    receipt_path = mesh_path.with_suffix(mesh_path.suffix + ".acquisition.json")
    if not receipt_path.is_file():
        raise MissingDataError(
            f"{mesh_path}: restricted mesh needs local acquisition receipt; run "
            "data/download.py fetch sdf --asset pitted-stonefish after placing "
            "the authorized file"
        )
    with receipt_path.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    if receipt.get("asset_id") != asset["id"]:
        raise DataIntegrityError("restricted mesh receipt has wrong asset ID")
    file_spec = receipt.get("file", {})
    verify_file(mesh_path, file_spec)
    return {
        "bytes": file_spec["bytes"],
        "checksum": file_spec["checksum"],
        "checksum_source": "git-ignored local acquisition receipt",
        "source_uid": receipt.get("source_uid"),
    }


def _is_canonical_config(
    config: SDFConfig,
    protocol: Mapping[str, Any],
) -> bool:
    keys = (
        "resolution",
        "surface_point_method",
        "sign_method",
        "scan_count",
        "scan_resolution",
        "sample_point_count",
        "normal_sample_count",
        "seed",
    )
    values = asdict(config)
    return all(values[key] == protocol[key] for key in keys)


def _config_from_args(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
) -> SDFConfig:
    def chosen(name: str) -> Any:
        value = getattr(args, name)
        return protocol[name] if value is None else value

    config = SDFConfig(
        resolution=chosen("resolution"),
        surface_point_method=chosen("surface_point_method"),
        sign_method=chosen("sign_method"),
        scan_count=chosen("scan_count"),
        scan_resolution=chosen("scan_resolution"),
        sample_point_count=chosen("sample_point_count"),
        normal_sample_count=chosen("normal_sample_count"),
        seed=chosen("seed"),
        max_query_points=args.max_query_points,
        query_workers=args.query_workers,
        pyopengl_platform=args.pyopengl_platform,
    )
    if config.resolution < 4:
        raise ManifestError("resolution must be at least 4")
    if config.max_query_points < config.resolution**2:
        raise ManifestError("max-query-points must fit at least one z slice")
    if config.query_workers == 0 or config.query_workers < -1:
        raise ManifestError("query-workers must be -1 or a positive integer")
    if not _is_canonical_config(config, protocol) and not args.allow_protocol_override:
        raise ManifestError(
            "requested settings differ from the checked-in paper protocol; "
            "pass --allow-protocol-override for an explicitly non-canonical smoke run"
        )
    return config


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    repository = DATA_DIR.parent.resolve()
    try:
        return resolved.relative_to(repository).as_posix()
    except ValueError:
        return path.name


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.part-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=("lucy", "pitted-stonefish", "thai-statue", "armadillo"))
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--resolution", type=int)
    parser.add_argument(
        "--surface-point-method",
        choices=("scan", "sample"),
        dest="surface_point_method",
    )
    parser.add_argument("--sign-method", choices=("normal", "depth"), dest="sign_method")
    parser.add_argument("--scan-count", type=int, dest="scan_count")
    parser.add_argument("--scan-resolution", type=int, dest="scan_resolution")
    parser.add_argument("--sample-point-count", type=int, dest="sample_point_count")
    parser.add_argument("--normal-sample-count", type=int, dest="normal_sample_count")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-query-points", type=int, default=1_048_576)
    parser.add_argument(
        "--query-workers",
        type=int,
        default=-1,
        help="parallel cKDTree workers for sampled surfaces (-1 uses all cores)",
    )
    parser.add_argument("--pyopengl-platform", default="egl")
    parser.add_argument("--allow-protocol-override", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = load_manifest("sdf")
    protocol = manifest["preprocessing"]
    try:
        config = _config_from_args(args, protocol)
        suffix = config.resolution
        output = args.output or DEFAULT_OUTPUT_ROOT / args.asset / f"sdf_{suffix}.npy"
        provenance = args.provenance or output.with_suffix(".provenance.json")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "asset": args.asset,
                        "output": _portable_path(output),
                        "provenance": _portable_path(provenance),
                        "config": asdict(config),
                        "canonical": _is_canonical_config(config, protocol),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        record = preprocess_sdf(
            args.asset,
            raw_root=args.raw_root,
            output_path=output,
            provenance_path=provenance,
            config=config,
        )
        print(f"[done] {output}")
        print(f"[hash] {record['output']['checksum']['value']}")
        print(f"[provenance] {provenance}")
        return 0
    except (DataIntegrityError, ManifestError, MissingDataError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
