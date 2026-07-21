"""Strict loaders and integrity checks for paper datasets.

The manifests are the source of truth.  Missing maps, unexpected dimensions,
and checksum mismatches are errors; no synthetic texture channels are inserted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


DATA_DIR = Path(__file__).resolve().parent
MANIFEST_DIR = DATA_DIR / "manifests"
DEFAULT_RAW_ROOT = DATA_DIR / "raw"

_MANIFEST_FILES = {
    "kodak": "kodak.json",
    "textures": "textures.json",
    "sdf": "sdf.json",
}
_TEXTURE_SEMANTICS = {
    "AO",
    "ARM",
    "DIFF",
    "Displacement",
    "metal",
    "normal",
    "rough",
    "specular",
}


class ManifestError(ValueError):
    """The checked-in manifest is malformed or internally inconsistent."""


class MissingDataError(FileNotFoundError):
    """A required raw or processed artifact is absent."""


class DataIntegrityError(RuntimeError):
    """A local artifact does not match its manifest or provenance."""


@dataclass(frozen=True)
class LoadedMap:
    """Channel location and provenance for one texture map."""

    map_id: str
    semantic: str
    channel_slice: slice
    source_path: Path


@dataclass(frozen=True)
class LoadedTextureSet:
    """A dynamically sized ``H x W x (3k)`` paper texture target."""

    set_id: str
    tensor: torch.Tensor
    maps: tuple[LoadedMap, ...]
    source_size: tuple[int, int]
    output_size: tuple[int, int]

    @property
    def channel_count(self) -> int:
        return int(self.tensor.shape[-1])


def load_manifest(name_or_path: str | Path) -> dict[str, Any]:
    """Read and validate one checked-in JSON manifest."""

    value = Path(name_or_path)
    if value.name in _MANIFEST_FILES:
        value = Path(_MANIFEST_FILES[value.name])
    if not value.suffix:
        try:
            value = Path(_MANIFEST_FILES[str(value)])
        except KeyError as exc:
            raise ManifestError(f"unknown manifest {name_or_path!r}") from exc
    path = value if value.is_absolute() else MANIFEST_DIR / value
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON manifest {path}: {exc}") from exc

    if manifest.get("schema_version") != 1:
        raise ManifestError(f"{path}: expected schema_version 1")
    kind = manifest.get("kind")
    if kind == "image_dataset":
        _validate_kodak_manifest(manifest, path)
    elif kind == "texture_set_dataset":
        _validate_texture_manifest(manifest, path)
    elif kind == "sdf_dataset":
        _validate_sdf_manifest(manifest, path)
    else:
        raise ManifestError(f"{path}: unknown manifest kind {kind!r}")
    return manifest


def hash_file(path: str | Path, algorithm: str = "sha256") -> str:
    """Hash a file without loading it into memory."""

    if algorithm not in {"md5", "sha256"}:
        raise ManifestError(f"unsupported checksum algorithm {algorithm!r}")
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def verify_file(
    path: str | Path,
    spec: Mapping[str, Any],
    *,
    require_checksum: bool = True,
) -> Path:
    """Verify existence, byte size, and checksum against a file spec."""

    candidate = Path(path)
    if not candidate.is_file():
        raise MissingDataError(f"required data file is missing: {candidate}")

    expected_size = spec.get("bytes")
    if expected_size is not None and candidate.stat().st_size != int(expected_size):
        raise DataIntegrityError(
            f"{candidate}: expected {expected_size} bytes, "
            f"found {candidate.stat().st_size}"
        )

    checksum = spec.get("checksum")
    if checksum is None:
        if require_checksum:
            raise DataIntegrityError(f"{candidate}: manifest has no pinned checksum")
        return candidate
    algorithm = checksum.get("algorithm")
    expected = checksum.get("value")
    if algorithm not in {"md5", "sha256"} or not isinstance(expected, str):
        raise ManifestError(f"{candidate}: malformed checksum specification")
    actual = hash_file(candidate, algorithm)
    if actual.lower() != expected.lower():
        raise DataIntegrityError(
            f"{candidate}: {algorithm} mismatch; expected {expected}, found {actual}"
        )
    return candidate


def resolve_local_path(raw_root: str | Path, spec: Mapping[str, Any]) -> Path:
    """Resolve a manifest path while rejecting traversal outside ``raw_root``."""

    relative = spec.get("local_path")
    if not isinstance(relative, str) or not relative:
        raise ManifestError("file spec has no local_path")
    root = Path(raw_root).resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"local_path escapes raw root: {relative!r}") from exc
    return candidate


def texture_set_spec(
    set_id: str,
    manifest: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Return one texture-set record by stable ID."""

    source = load_manifest("textures") if manifest is None else manifest
    for item in source["sets"]:
        if item["id"] == set_id:
            return item
    raise ManifestError(f"texture set {set_id!r} is not in the paper manifest")


def load_texture_set(
    set_id: str,
    *,
    raw_root: str | Path | None = None,
    output_size: int | Sequence[int] | None = None,
    verify_checksums: bool = True,
    manifest_path: str | Path = "textures",
) -> LoadedTextureSet:
    """Load every required map and concatenate it as three channels per map.

    Grayscale maps are replicated to RGB and alpha is discarded.  Normal maps
    are decoded from OpenGL UNORM RGB, filtered as vectors, normalized, and
    encoded back to ``[0, 1]``.
    """

    manifest = load_manifest(manifest_path)
    item = texture_set_spec(set_id, manifest)
    root = DEFAULT_RAW_ROOT if raw_root is None else Path(raw_root)
    requested_size = _coerce_output_size(output_size)

    tensors: list[torch.Tensor] = []
    loaded_maps: list[LoadedMap] = []
    source_size: tuple[int, int] | None = None
    channel_start = 0

    for map_spec in item["maps"]:
        path = resolve_local_path(root, map_spec)
        if verify_checksums:
            verify_file(path, map_spec)
        elif not path.is_file():
            raise MissingDataError(f"required texture map is missing: {path}")

        image, native_size = _decode_image(path, map_spec)
        if source_size is None:
            source_size = native_size
        elif source_size != native_size:
            raise DataIntegrityError(
                f"{set_id}: maps do not share one native size: "
                f"{source_size} vs {native_size} ({path.name})"
            )

        target_size = requested_size or native_size
        image = _filter_map(
            image,
            target_size,
            is_normal=map_spec["semantic"] == "normal",
            normal_convention=map_spec.get("normal_convention"),
        )
        tensors.append(image)
        loaded_maps.append(
            LoadedMap(
                map_id=map_spec["id"],
                semantic=map_spec["semantic"],
                channel_slice=slice(channel_start, channel_start + 3),
                source_path=path,
            )
        )
        channel_start += 3

    if not tensors or source_size is None:
        raise ManifestError(f"{set_id}: texture set has no maps")
    tensor = torch.cat(tensors, dim=-1).contiguous()
    expected_channels = 3 * len(item["maps"])
    if tensor.shape[-1] != expected_channels:
        raise DataIntegrityError(
            f"{set_id}: expected {expected_channels} channels, got {tensor.shape[-1]}"
        )
    return LoadedTextureSet(
        set_id=set_id,
        tensor=tensor,
        maps=tuple(loaded_maps),
        source_size=source_size,
        output_size=(int(tensor.shape[0]), int(tensor.shape[1])),
    )


def verify_kodak(
    *,
    raw_root: str | Path | None = None,
    manifest_path: str | Path = "kodak",
) -> tuple[Path, ...]:
    """Strictly verify all 24 original-orientation Kodak PNGs."""

    manifest = load_manifest(manifest_path)
    root = DEFAULT_RAW_ROOT if raw_root is None else Path(raw_root)
    verified: list[Path] = []
    for image in manifest["images"]:
        path = resolve_local_path(root, image)
        verify_file(path, image)
        with Image.open(path) as handle:
            actual = (handle.width, handle.height, handle.mode)
        expected = (image["width"], image["height"], image["mode"])
        if actual != expected:
            raise DataIntegrityError(
                f"{path}: expected image metadata {expected}, found {actual}"
            )
        verified.append(path)
    return tuple(verified)


def iter_manifest_files(
    manifest: Mapping[str, Any],
) -> Iterable[Mapping[str, Any]]:
    """Yield pinned local artifacts from any supported manifest."""

    if manifest["kind"] == "image_dataset":
        yield from manifest["images"]
    elif manifest["kind"] == "texture_set_dataset":
        for item in manifest["sets"]:
            archive = item.get("archive")
            if archive is not None:
                yield archive
            yield from item["maps"]
    elif manifest["kind"] == "sdf_dataset":
        for item in manifest["assets"]:
            archive = item.get("archive")
            if archive is not None:
                yield archive
            mesh = item.get("mesh")
            if mesh is not None:
                yield mesh


def _decode_image(
    path: Path,
    spec: Mapping[str, Any],
) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(path) as image:
        image.load()
        expected_format = spec.get("format")
        if expected_format and image.format != expected_format:
            raise DataIntegrityError(
                f"{path}: expected {expected_format}, found {image.format}"
            )
        expected_mode = spec.get("storage_mode")
        if expected_mode and image.mode != expected_mode:
            raise DataIntegrityError(
                f"{path}: expected storage mode {expected_mode}, found {image.mode}"
            )
        expected_size = (int(spec["width"]), int(spec["height"]))
        if image.size != expected_size:
            raise DataIntegrityError(
                f"{path}: expected dimensions {expected_size}, found {image.size}"
            )
        array = np.array(image)

    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    elif array.dtype == np.uint16:
        array = array.astype(np.float32) / 65535.0
    elif np.issubdtype(array.dtype, np.integer):
        bit_depth = spec.get("encoded_bit_depth")
        if bit_depth not in {8, 16}:
            raise DataIntegrityError(
                f"{path}: integer mode {array.dtype} needs encoded_bit_depth 8 or 16"
            )
        array = array.astype(np.float32) / float((1 << bit_depth) - 1)
    elif np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    else:
        raise DataIntegrityError(f"{path}: unsupported pixel dtype {array.dtype}")

    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] >= 3:
        array = array[..., :3]
    else:
        raise DataIntegrityError(f"{path}: unsupported image shape {array.shape}")
    if not np.isfinite(array).all():
        raise DataIntegrityError(f"{path}: image contains non-finite values")
    if float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise DataIntegrityError(f"{path}: decoded values are outside [0, 1]")
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    return tensor, (int(tensor.shape[0]), int(tensor.shape[1]))


def _filter_map(
    image: torch.Tensor,
    output_size: tuple[int, int],
    *,
    is_normal: bool,
    normal_convention: str | None,
) -> torch.Tensor:
    if is_normal:
        if normal_convention != "OpenGL":
            raise ManifestError(
                f"normal map must declare OpenGL convention, got {normal_convention!r}"
            )
        image = image.mul(2.0).sub(1.0)

    chw = image.permute(2, 0, 1).unsqueeze(0)
    if tuple(image.shape[:2]) != output_size:
        chw = F.interpolate(
            chw,
            size=output_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    if is_normal:
        length = torch.linalg.vector_norm(chw, dim=1, keepdim=True)
        fallback = torch.zeros_like(chw)
        fallback[:, 2:3] = 1.0
        chw = torch.where(length > 1e-8, chw / length.clamp_min(1e-8), fallback)
        chw = chw.add(1.0).mul(0.5).clamp_(0.0, 1.0)
    return chw.squeeze(0).permute(1, 2, 0).contiguous()


def _coerce_output_size(
    output_size: int | Sequence[int] | None,
) -> tuple[int, int] | None:
    if output_size is None:
        return None
    if isinstance(output_size, int):
        result = (output_size, output_size)
    else:
        if len(output_size) != 2:
            raise ValueError("output_size must be an int or (height, width)")
        result = (int(output_size[0]), int(output_size[1]))
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError("output dimensions must be positive")
    return result


def _validate_checksum(spec: Mapping[str, Any], context: str, optional: bool = False) -> None:
    checksum = spec.get("checksum")
    if checksum is None:
        if optional:
            return
        raise ManifestError(f"{context}: checksum is required")
    algorithm = checksum.get("algorithm")
    value = checksum.get("value")
    lengths = {"md5": 32, "sha256": 64}
    if algorithm not in lengths or not isinstance(value, str):
        raise ManifestError(f"{context}: malformed checksum")
    if len(value) != lengths[algorithm] or any(c not in "0123456789abcdef" for c in value.lower()):
        raise ManifestError(f"{context}: malformed {algorithm} value")


def _validate_file_spec(
    spec: Mapping[str, Any],
    context: str,
    *,
    checksum_optional: bool = False,
) -> None:
    if not isinstance(spec.get("local_path"), str):
        raise ManifestError(f"{context}: local_path is required")
    if "bytes" in spec and (not isinstance(spec["bytes"], int) or spec["bytes"] <= 0):
        raise ManifestError(f"{context}: bytes must be a positive integer")
    _validate_checksum(spec, context, checksum_optional)


def _validate_kodak_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    images = manifest.get("images")
    if not isinstance(images, list) or len(images) != 24:
        raise ManifestError(f"{path}: Kodak manifest must contain 24 images")
    expected_ids = [f"kodim{i:02d}" for i in range(1, 25)]
    if [item.get("id") for item in images] != expected_ids:
        raise ManifestError(f"{path}: Kodak IDs/order must be kodim01..kodim24")
    for image in images:
        context = f"{path}:{image.get('id')}"
        _validate_file_spec(image, context)
        if image.get("mode") != "RGB" or image.get("color_space") != "sRGB":
            raise ManifestError(f"{context}: expected RGB sRGB metadata")
        if {image.get("width"), image.get("height")} != {512, 768}:
            raise ManifestError(f"{context}: expected a 768x512 orientation")
        if not image.get("credit", {}).get("photographer"):
            raise ManifestError(f"{context}: photographer credit is required")


def _validate_texture_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    sets = manifest.get("sets")
    if not isinstance(sets, list) or not sets:
        raise ManifestError(f"{path}: texture manifest has no sets")
    paper_manifest = manifest.get("dataset_id") == "peps-paper-4k-texture-sets"
    if paper_manifest and len(sets) != 18:
        raise ManifestError(f"{path}: paper texture manifest must contain 18 sets")
    semantic_order = manifest.get("semantic_order", [])
    semantic_rank = {name: index for index, name in enumerate(semantic_order)}
    ids = [item.get("id") for item in sets]
    if len(set(ids)) != len(ids):
        raise ManifestError(f"{path}: duplicate texture-set ID")
    for item in sets:
        context = f"{path}:{item.get('id')}"
        maps = item.get("maps")
        if not isinstance(maps, list) or not maps:
            raise ManifestError(f"{context}: maps must be a non-empty list")
        if paper_manifest and item.get("license") not in manifest.get("licenses", {}):
            raise ManifestError(f"{context}: unknown or missing license record")
        map_ids = [entry.get("id") for entry in maps]
        if len(set(map_ids)) != len(map_ids):
            raise ManifestError(f"{context}: duplicate map ID")
        if semantic_rank:
            ranks = [semantic_rank.get(entry.get("semantic"), -1) for entry in maps]
            if ranks != sorted(ranks):
                raise ManifestError(f"{context}: maps violate semantic_order")
        for entry in maps:
            map_context = f"{context}:{entry.get('id')}"
            _validate_file_spec(entry, map_context)
            if entry.get("semantic") not in _TEXTURE_SEMANTICS:
                raise ManifestError(
                    f"{map_context}: unknown semantic {entry.get('semantic')!r}"
                )
            if entry.get("channels") != 3:
                raise ManifestError(f"{map_context}: paper maps must decode to RGB")
            if entry.get("color_space") not in {"sRGB", "linear"}:
                raise ManifestError(f"{map_context}: color_space is required")
            if paper_manifest:
                expected_color_space = (
                    "sRGB" if entry.get("semantic") == "DIFF" else "linear"
                )
                if entry.get("color_space") != expected_color_space:
                    raise ManifestError(
                        f"{map_context}: expected {expected_color_space} color space"
                    )
            dimensions = (entry.get("width"), entry.get("height"))
            if paper_manifest and dimensions != (4096, 4096):
                raise ManifestError(f"{map_context}: paper textures must be 4K")
            if not paper_manifest and (
                not all(isinstance(value, int) for value in dimensions)
                or min(dimensions) <= 0
            ):
                raise ManifestError(f"{map_context}: invalid dimensions")
            if entry.get("semantic") == "normal":
                if entry.get("normal_convention") != "OpenGL":
                    raise ManifestError(f"{map_context}: normal must be OpenGL")
                if entry.get("color_space") != "linear":
                    raise ManifestError(f"{map_context}: normal must be linear")
            storage_mode = entry.get("storage_mode")
            if paper_manifest and storage_mode not in {"L", "RGB", "RGBA", "I;16"}:
                raise ManifestError(f"{map_context}: unsupported storage_mode")
            if storage_mode is not None:
                expected_bits = 16 if storage_mode == "I;16" else 8
                if entry.get("encoded_bit_depth") != expected_bits:
                    raise ManifestError(f"{map_context}: encoded bit depth mismatch")
        archive = item.get("archive")
        if archive is not None:
            _validate_file_spec(archive, f"{context}:archive")


def _validate_sdf_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    assets = manifest.get("assets")
    if not isinstance(assets, list) or len(assets) != 4:
        raise ManifestError(f"{path}: SDF manifest must contain four assets")
    expected = {"lucy", "pitted-stonefish", "thai-statue", "armadillo"}
    if {item.get("id") for item in assets} != expected:
        raise ManifestError(f"{path}: unexpected SDF asset IDs")
    protocol = manifest.get("preprocessing")
    if not isinstance(protocol, dict):
        raise ManifestError(f"{path}: preprocessing protocol is required")
    if protocol.get("resolution") != 512:
        raise ManifestError(f"{path}: paper SDF resolution must be 512")
    if protocol.get("sign_convention") != "negative_inside":
        raise ManifestError(f"{path}: SDF sign must be negative inside")
    if protocol.get("axis_order") != "zyx":
        raise ManifestError(f"{path}: SDF axis order must be zyx")
    if protocol.get("surface_point_method") not in {"open3d", "sample", "scan"}:
        raise ManifestError(f"{path}: unsupported SDF surface point method")
    if protocol.get("sign_method") not in {"ray_parity", "normal", "depth"}:
        raise ManifestError(f"{path}: unsupported SDF sign method")
    if protocol.get("surface_point_method") == "open3d":
        if protocol.get("sign_method") != "ray_parity":
            raise ManifestError(f"{path}: Open3D requires ray-parity signs")
        ray_nsamples = protocol.get("ray_nsamples")
        if not isinstance(ray_nsamples, int) or ray_nsamples < 1 or ray_nsamples % 2 != 1:
            raise ManifestError(f"{path}: ray_nsamples must be a positive odd integer")
    for key in (
        "scan_count",
        "scan_resolution",
        "sample_point_count",
        "normal_sample_count",
    ):
        if not isinstance(protocol.get(key), int) or protocol[key] <= 0:
            raise ManifestError(f"{path}: invalid SDF preprocessing value {key}")
    for item in assets:
        context = f"{path}:{item.get('id')}"
        archive = item.get("archive")
        if archive is not None:
            _validate_file_spec(
                archive,
                f"{context}:archive",
                checksum_optional=item.get("access") != "public",
            )
        mesh = item.get("mesh")
        if not isinstance(mesh, dict):
            raise ManifestError(f"{context}: mesh spec is required")
        _validate_file_spec(
            mesh,
            f"{context}:mesh",
            checksum_optional=item.get("access") != "public",
        )
