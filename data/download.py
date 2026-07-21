"""Fetch and verify the repository's frozen PEPS dataset manifests.

The paper names datasets/assets but does not publish every texture-map choice or
its bit-exact SDF converter. The manifests make those reproduction assumptions
explicit; checksum verification proves local bytes match the manifests, not that
the resulting experiment has reproduced the paper.

Examples:
    python data/download.py list textures
    python data/download.py fetch kodak
    python data/download.py fetch textures
    python data/download.py fetch sdf
    python data/download.py verify all

Raw assets are written below ``data/raw/`` and are git-ignored. The Pitted
Stonefish download is never attempted unless explicitly requested with
``--include-restricted`` or ``--asset pitted-stonefish``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
from typing import Any, Iterable, Mapping, Sequence
import urllib.error
import urllib.request
import zipfile

from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.manifest import (  # noqa: E402
    DEFAULT_RAW_ROOT,
    DataIntegrityError,
    ManifestError,
    MissingDataError,
    hash_file,
    load_manifest,
    resolve_local_path,
    verify_file,
    verify_kodak,
)


USER_AGENT = "PEPS-paper-reproduction/0.1 (+https://arxiv.org/abs/2604.24167)"
CHUNK_SIZE = 1024 * 1024


class AccessRequiredError(RuntimeError):
    """A dataset requires user-granted credentials or a manual file."""


def _request_json(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    private_url: bool = False,
) -> dict[str, Any]:
    request_headers = {"User-Agent": USER_AGENT}
    request_headers.update(headers or {})
    request = urllib.request.Request(url, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        location = "authenticated endpoint" if private_url else url
        raise RuntimeError(f"{location}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        location = "authenticated endpoint" if private_url else url
        raise RuntimeError(f"{location}: network error: {exc.reason}") from exc


def _atomic_download(
    url: str,
    destination: Path,
    spec: Mapping[str, Any],
    *,
    headers: Mapping[str, str] | None = None,
    private_url: bool = False,
) -> str:
    if destination.is_file():
        verify_file(destination, spec)
        _verify_image_metadata(destination, spec)
        print(f"[ok]   {destination}")
        return "existing"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part-{os.getpid()}")
    request_headers = {"User-Agent": USER_AGENT}
    request_headers.update(headers or {})
    request = urllib.request.Request(url, headers=request_headers)
    label = destination.name if private_url else url
    print(f"[get]  {label}")
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=CHUNK_SIZE)
        verify_file(temporary, spec)
        _verify_image_metadata(temporary, spec)
        os.replace(temporary, destination)
    except urllib.error.HTTPError as exc:
        temporary.unlink(missing_ok=True)
        location = "authenticated download" if private_url else url
        raise RuntimeError(f"{location}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        temporary.unlink(missing_ok=True)
        location = "authenticated download" if private_url else url
        raise RuntimeError(f"{location}: network error: {exc.reason}") from exc
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"[hash] {destination}")
    return "downloaded"


def _atomic_download_unpinned(
    url: str,
    destination: Path,
    *,
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Download a short-lived authenticated URL and produce a local receipt."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part-{os.getpid()}")
    request_headers = {"User-Agent": USER_AGENT}
    request_headers.update(headers or {})
    request = urllib.request.Request(url, headers=request_headers)
    print(f"[get]  authenticated asset -> {destination.name}")
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=CHUNK_SIZE)
        if temporary.stat().st_size == 0:
            raise DataIntegrityError("authenticated asset download is empty")
        os.replace(temporary, destination)
    except urllib.error.HTTPError as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"authenticated download: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"authenticated download: network error: {exc.reason}") from exc
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "bytes": destination.stat().st_size,
        "checksum": {
            "algorithm": "sha256",
            "value": hash_file(destination, "sha256"),
        },
    }


def fetch_kodak(raw_root: Path) -> None:
    manifest = load_manifest("kodak")
    print("Kodak PCD0992: 24 original-orientation PNGs")
    for image in manifest["images"]:
        destination = resolve_local_path(raw_root, image)
        _atomic_download(image["url"], destination, image)
    verify_kodak(raw_root=raw_root)


def fetch_textures(
    raw_root: Path,
    selected: Sequence[str] | None = None,
) -> None:
    manifest = load_manifest("textures")
    items = _select(manifest["sets"], selected, "texture")
    for item in items:
        print(f"{item['id']}: {item['paper_name']} ({item['source']['provider']})")
        archive = item.get("archive")
        if archive is not None:
            archive_path = resolve_local_path(raw_root, archive)
            _atomic_download(archive["url"], archive_path, archive)
            _extract_zip_maps(archive_path, item["maps"], raw_root)
        else:
            for map_spec in item["maps"]:
                destination = resolve_local_path(raw_root, map_spec)
                _atomic_download(map_spec["url"], destination, map_spec)


def fetch_sdf(
    raw_root: Path,
    selected: Sequence[str] | None = None,
    *,
    include_restricted: bool = False,
) -> None:
    manifest = load_manifest("sdf")
    explicit = set(selected or ())
    items = _select(manifest["assets"], selected, "SDF")
    for item in items:
        if item["access"] != "public":
            if include_restricted or item["id"] in explicit:
                _fetch_stonefish(item, raw_root)
            else:
                print(
                    "[skip] pitted-stonefish requires a Sketchfab token or manual file; "
                    "use --include-restricted to opt in"
                )
            continue
        print(f"{item['id']}: {item['paper_name']}")
        archive = item["archive"]
        archive_path = resolve_local_path(raw_root, archive)
        _atomic_download(archive["url"], archive_path, archive)
        mesh_path = resolve_local_path(raw_root, item["mesh"])
        _extract_mesh(archive_path, mesh_path, item)


def verify_dataset(
    target: str,
    raw_root: Path,
    selected: Sequence[str] | None = None,
) -> bool:
    failures: list[str] = []
    manifests = _target_manifests(target)
    for name in manifests:
        manifest = load_manifest(name)
        if name == "kodak":
            specs: Iterable[Mapping[str, Any]] = manifest["images"]
        elif name == "textures":
            items = _select(manifest["sets"], selected, "texture")
            specs = (
                spec
                for item in items
                for spec in (
                    ([item["archive"]] if item.get("archive") else []) + item["maps"]
                )
            )
        else:
            items = _select(manifest["assets"], selected, "SDF")
            specs = (
                spec
                for item in items
                for spec in [item.get("archive"), item["mesh"]]
                if spec is not None
            )
        for spec in specs:
            path = resolve_local_path(raw_root, spec)
            try:
                if spec.get("checksum") is None:
                    _verify_local_receipt(path)
                else:
                    verify_file(path, spec)
                _verify_image_metadata(path, spec)
                print(f"[ok]   {path}")
            except (MissingDataError, DataIntegrityError) as exc:
                print(f"[fail] {exc}")
                failures.append(str(exc))
    print(f"verification: {len(failures)} failure(s)")
    return not failures


def list_dataset(target: str) -> None:
    for name in _target_manifests(target):
        manifest = load_manifest(name)
        print(f"{name}: {manifest['dataset_id']}")
        if name == "kodak":
            print(f"  {len(manifest['images'])} images")
        elif name == "textures":
            for item in manifest["sets"]:
                semantics = ",".join(entry["semantic"] for entry in item["maps"])
                print(
                    f"  {item['id']}: {len(item['maps'])} maps "
                    f"[{semantics}] ({item['source']['provider']})"
                )
        else:
            for item in manifest["assets"]:
                print(
                    f"  {item['id']}: {item['access']}, "
                    f"{item['license']['name']}"
                )


def _extract_zip_maps(
    archive_path: Path,
    maps: Sequence[Mapping[str, Any]],
    raw_root: Path,
) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        members = {item.filename: item for item in archive.infolist()}
        for map_spec in maps:
            member_name = map_spec.get("archive_member")
            if not isinstance(member_name, str):
                raise ManifestError(f"{archive_path}: map has no archive_member")
            _validate_archive_member(member_name)
            member = members.get(member_name)
            if member is None or member.is_dir():
                raise DataIntegrityError(
                    f"{archive_path}: required member is missing: {member_name}"
                )
            destination = resolve_local_path(raw_root, map_spec)
            if destination.is_file():
                verify_file(destination, map_spec)
                _verify_image_metadata(destination, map_spec)
                print(f"[ok]   {destination}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(
                f".{destination.name}.part-{os.getpid()}"
            )
            try:
                with archive.open(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=CHUNK_SIZE)
                verify_file(temporary, map_spec)
                _verify_image_metadata(temporary, map_spec)
                os.replace(temporary, destination)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            print(f"[unzip] {destination}")


def _extract_mesh(
    archive_path: Path,
    mesh_path: Path,
    item: Mapping[str, Any],
) -> None:
    mesh_spec = item["mesh"]
    if mesh_path.is_file():
        verify_file(mesh_path, mesh_spec)
        print(f"[ok]   {mesh_path}")
        return
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = mesh_path.with_name(f".{mesh_path.name}.part-{os.getpid()}")
    compression = item["archive"]["compression"]
    member_name = mesh_spec.get("archive_member")
    try:
        if compression == "gzip":
            with gzip.open(archive_path, "rb") as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=CHUNK_SIZE)
        elif compression == "tar.gz":
            if not isinstance(member_name, str):
                raise ManifestError(f"{item['id']}: tar archive member is required")
            _validate_archive_member(member_name)
            with tarfile.open(archive_path, "r:gz") as archive:
                member = archive.getmember(member_name)
                if not member.isfile():
                    raise DataIntegrityError(
                        f"{archive_path}: mesh member is not a file"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise DataIntegrityError(
                        f"{archive_path}: cannot read {member_name}"
                    )
                with source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=CHUNK_SIZE)
        else:
            raise ManifestError(f"unsupported mesh compression {compression!r}")
        verify_file(temporary, mesh_spec)
        os.replace(temporary, mesh_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"[extract] {mesh_path}")


def _fetch_stonefish(item: Mapping[str, Any], raw_root: Path) -> None:
    mesh_path = resolve_local_path(raw_root, item["mesh"])
    receipt_path = mesh_path.with_suffix(mesh_path.suffix + ".acquisition.json")
    if mesh_path.is_file():
        receipt = _write_local_receipt(mesh_path, receipt_path, item)
        print(
            f"[local] {mesh_path} ({receipt['file']['checksum']['value'][:12]}…)"
        )
        return

    token = os.environ.get("SKETCHFAB_OAUTH_TOKEN")
    authorization = f"Bearer {token}" if token else None
    if authorization is None:
        token = os.environ.get("SKETCHFAB_API_TOKEN")
        authorization = f"Token {token}" if token else None
    if authorization is None:
        raise AccessRequiredError(
            "pitted-stonefish is Academic-only and requires Sketchfab account "
            "authorization. Set SKETCHFAB_OAUTH_TOKEN or SKETCHFAB_API_TOKEN, "
            f"or manually place the authorized GLB at {mesh_path}. Tokens and "
            "raw assets must not be committed."
        )

    source = item["source"]
    metadata = _request_json(source["metadata_url"])
    if metadata.get("uid") != source["uid"]:
        raise DataIntegrityError("Sketchfab returned an unexpected model UID")
    if metadata.get("faceCount") != source["face_count"]:
        raise DataIntegrityError("Sketchfab model face count changed")
    if metadata.get("license", {}).get("slug") != source["license_slug"]:
        raise DataIntegrityError("Sketchfab model license changed")

    download = _request_json(
        source["download_api_url"],
        headers={"Authorization": authorization},
        private_url=True,
    )
    selected = download.get("glb")
    if not isinstance(selected, dict) or not isinstance(selected.get("url"), str):
        raise AccessRequiredError(
            "Sketchfab did not offer the expected GLB for this account; download "
            f"the model manually and place it at {mesh_path}"
        )
    file_spec = _atomic_download_unpinned(selected["url"], mesh_path)
    receipt = _write_local_receipt(
        mesh_path,
        receipt_path,
        item,
        file_spec=file_spec,
        metadata=metadata,
    )
    print(f"[receipt] {receipt_path}")
    print(f"[hash] {receipt['file']['checksum']['value']}")


def _write_local_receipt(
    mesh_path: Path,
    receipt_path: Path,
    item: Mapping[str, Any],
    *,
    file_spec: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if file_spec is None:
        file_spec = {
            "bytes": mesh_path.stat().st_size,
            "checksum": {
                "algorithm": "sha256",
                "value": hash_file(mesh_path, "sha256"),
            },
        }
    receipt = {
        "schema_version": 1,
        "asset_id": item["id"],
        "canonical": True,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_uid": item["source"]["uid"],
        "file": {
            "name": mesh_path.name,
            "bytes": file_spec["bytes"],
            "checksum": file_spec["checksum"],
        },
        "remote_metadata": {
            "face_count": (metadata or {}).get(
                "faceCount", item["source"]["face_count"]
            ),
            "license_slug": (metadata or {}).get("license", {}).get(
                "slug", item["source"]["license_slug"]
            ),
            "updated_at": (metadata or {}).get("updatedAt"),
        },
        "note": "Local receipt only; never commit the Academic-only raw asset.",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_name(f".{receipt_path.name}.part-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, receipt_path)
    return receipt


def _validate_archive_member(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise DataIntegrityError(f"unsafe archive member path: {name!r}")


def _verify_image_metadata(path: Path, spec: Mapping[str, Any]) -> None:
    if "width" not in spec or "height" not in spec:
        return
    with Image.open(path) as image:
        actual = (image.width, image.height, image.format, image.mode)
    expected = (
        spec["width"],
        spec["height"],
        spec.get("format"),
        spec.get("storage_mode", spec.get("mode")),
    )
    if actual[:3] != expected[:3]:
        raise DataIntegrityError(
            f"{path}: expected image metadata {expected[:3]}, found {actual[:3]}"
        )
    if expected[3] is not None and actual[3] != expected[3]:
        raise DataIntegrityError(
            f"{path}: expected storage mode {expected[3]}, found {actual[3]}"
        )


def _verify_local_receipt(path: Path) -> None:
    receipt_path = path.with_suffix(path.suffix + ".acquisition.json")
    if not receipt_path.is_file():
        raise MissingDataError(f"missing local acquisition receipt: {receipt_path}")
    with receipt_path.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    file_spec = receipt.get("file")
    if not isinstance(file_spec, dict):
        raise DataIntegrityError(f"{receipt_path}: malformed local receipt")
    verify_file(path, file_spec)


def _select(
    items: Sequence[Mapping[str, Any]],
    selected: Sequence[str] | None,
    label: str,
) -> list[Mapping[str, Any]]:
    if not selected:
        return list(items)
    by_id = {item["id"]: item for item in items}
    unknown = sorted(set(selected) - set(by_id))
    if unknown:
        raise ManifestError(f"unknown {label} asset(s): {', '.join(unknown)}")
    return [by_id[item_id] for item_id in selected]


def _target_manifests(target: str) -> tuple[str, ...]:
    if target == "all":
        return ("kodak", "textures", "sdf")
    if target not in {"kodak", "textures", "sdf"}:
        raise ManifestError(f"unknown dataset target {target!r}")
    return (target,)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("list", "fetch", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("target", choices=("kodak", "textures", "sdf", "all"))
        child.add_argument(
            "--asset",
            action="append",
            default=None,
            help="stable manifest asset ID; repeat to select several",
        )
        child.add_argument(
            "--raw-root",
            type=Path,
            default=DEFAULT_RAW_ROOT,
            help="raw dataset root (default: data/raw)",
        )
        if command == "fetch":
            child.add_argument(
                "--include-restricted",
                action="store_true",
                help="opt in to the credential-gated Pitted Stonefish download",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.target == "all" and args.asset:
            raise ManifestError("--asset cannot be combined with target 'all'")
        if args.command == "list":
            list_dataset(args.target)
            return 0
        if args.command == "verify":
            return 0 if verify_dataset(args.target, args.raw_root, args.asset) else 1

        for target in _target_manifests(args.target):
            if target == "kodak":
                if args.asset:
                    raise ManifestError("--asset is not supported for Kodak")
                fetch_kodak(args.raw_root)
            elif target == "textures":
                fetch_textures(args.raw_root, args.asset)
            else:
                fetch_sdf(
                    args.raw_root,
                    args.asset,
                    include_restricted=args.include_restricted,
                )
        return 0
    except (
        AccessRequiredError,
        DataIntegrityError,
        ManifestError,
        MissingDataError,
        RuntimeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
