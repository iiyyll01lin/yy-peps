"""Reproducible reporting and run provenance under ``results/``.

繁體中文:可重現報告工具。所有 notebook 透過本模組把 Table/Fig 寫成
``results/<name>.csv``(數字)與 ``results/<name>.png``(圖),讓 docs 引用的每個
dB/IoU 數字都有對應產出檔背書,而非寫死在文字裡。

設計原則:
- CSV 是「真相來源」:純文字、進 git、可 diff。
- PNG 為視覺化:被 .gitignore 忽略(可由 CSV 重新生成)。
- 每個 run 可另寫 immutable config/provenance JSON 及 tidy per-instance CSV。
- 不依賴 pandas(遠端 venv 未必有);只用 csv 標準庫 + 可選 matplotlib。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
from importlib import metadata as importlib_metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

# results/ lives at repo root, one level up from this file's package.
RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results")
)

RUN_SCHEMA = "peps.run_manifest"
RUN_SCHEMA_VERSION = 1
INSTANCE_SCHEMA = "peps.instance_metric"
INSTANCE_SCHEMA_VERSION = 1
INSTANCE_STATUSES = ("ok", "failed", "skipped")
INSTANCE_COLUMNS = (
    "schema",
    "schema_version",
    "run_id",
    "experiment",
    "profile",
    "seed",
    "instance_id",
    "split",
    "method",
    "metric",
    "value",
    "unit",
    "status",
    "duration_seconds",
    "metadata_json",
)
DATASET_HASH_FIELDS = ("id", "path", "algorithm", "digest", "bytes")
GIT_STATE_FIELDS = (
    "available",
    "sha",
    "branch",
    "dirty",
    "tracked_changes",
    "untracked_files",
    "error",
)
ENVIRONMENT_FIELDS = (
    "platform",
    "packages",
    "pytorch",
    "rocm",
    "gpu",
    "collection_errors",
)
DEFAULT_PACKAGE_NAMES = (
    "peps",
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "Pillow",
    "imageio",
    "matplotlib",
    "tqdm",
    "scikit-image",
    "torchmetrics",
    "lpips",
    "flip-evaluator",
    "trimesh",
    "mesh-to-sdf",
)
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_SEED = 2**63 - 1


@dataclass(frozen=True, slots=True)
class InstanceRow:
    """One tidy per-instance metric observation."""

    instance_id: str
    method: str
    metric: str
    value: float | int | None
    split: str = "test"
    unit: str = ""
    seed: int | None = None
    status: str = "ok"
    duration_seconds: float | None = None
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Paths written by :func:`write_run`."""

    run_dir: str
    manifest_path: str
    instances_path: str


def results_path(name: str) -> str:
    """Absolute path under ``results/`` for a given basename (dirs auto-created)."""
    p = os.path.join(RESULTS_DIR, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _jsonable(value: Any, *, location: str = "$") -> Any:
    """Convert supported values to strict, deterministic JSON data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location}: non-finite floats are not valid provenance")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _jsonable(value.value, location=location)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict(), location=location)
    if is_dataclass(value):
        return _jsonable(
            {field.name: getattr(value, field.name) for field in fields(value)},
            location=location,
        )
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location}: JSON object keys must be strings")
            result[key] = _jsonable(item, location=f"{location}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{location}: unsupported provenance value {type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _config_digest(config: object) -> str:
    return hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()


def _normalise_timestamp(value: datetime | str | None = None) -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be an ISO-8601 datetime") from exc
    else:
        raise TypeError("timestamp must be datetime, string, or None")
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _make_run_id(experiment: str, profile: str, seed: int, timestamp: str) -> str:
    compact_time = timestamp.replace("-", "").replace(":", "").replace(".", "")
    compact_time = compact_time.replace("+0000", "Z")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{experiment}-{profile}").strip("-")
    suffix = f"-s{seed}-{uuid4().hex[:8]}"
    max_slug_length = 128 - len(compact_time) - len(suffix) - 1
    return f"{compact_time}-{slug[:max_slug_length]}{suffix}"


def hash_dataset_files(
    files: Mapping[str, str | os.PathLike[str]],
) -> tuple[dict[str, object], ...]:
    """SHA-256 hash dataset artifacts by stable logical ID."""

    hashed: list[dict[str, object]] = []
    logical_ids = list(files)
    if any(not isinstance(item, str) or not item for item in logical_ids):
        raise ValueError("dataset logical IDs must be non-empty strings")
    for logical_id in sorted(logical_ids):
        source = os.fspath(files[logical_id])
        path = Path(source)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        hashed.append(
            {
                "id": logical_id,
                "path": source,
                "algorithm": "sha256",
                "digest": digest.hexdigest(),
                "bytes": size,
            }
        )
    return tuple(hashed)


def _git_command(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def collect_git_state(repo_root: str | os.PathLike[str] | None = None) -> dict[str, object]:
    """Collect commit, branch, and dirty-state metadata without mutating git."""

    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    try:
        sha_result = _git_command(root, "rev-parse", "HEAD")
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "sha": None,
            "branch": None,
            "dirty": None,
            "tracked_changes": None,
            "untracked_files": None,
            "error": type(exc).__name__,
        }
    if sha_result.returncode != 0:
        return {
            "available": False,
            "sha": None,
            "branch": None,
            "dirty": None,
            "tracked_changes": None,
            "untracked_files": None,
            "error": sha_result.stderr.strip() or "not_a_git_repository",
        }

    status_result = _git_command(root, "status", "--porcelain=v1", "--untracked-files=normal")
    branch_result = _git_command(root, "symbolic-ref", "--short", "-q", "HEAD")
    status_lines = (
        [line for line in status_result.stdout.splitlines() if line]
        if status_result.returncode == 0
        else []
    )
    untracked = sum(line.startswith("??") for line in status_lines)
    tracked = len(status_lines) - untracked
    return {
        "available": True,
        "sha": sha_result.stdout.strip(),
        "branch": branch_result.stdout.strip() or None,
        "dirty": bool(status_lines) if status_result.returncode == 0 else None,
        "tracked_changes": tracked if status_result.returncode == 0 else None,
        "untracked_files": untracked if status_result.returncode == 0 else None,
        "error": None if status_result.returncode == 0 else status_result.stderr.strip(),
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _rocm_runtime_version() -> str | None:
    roots = [Path(os.environ.get("ROCM_HOME", "/opt/rocm")), Path("/opt/rocm")]
    for root in roots:
        for relative in (".info/version", ".info/version-dev"):
            candidate = root / relative
            try:
                value = candidate.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if value:
                return value
    return None


def collect_environment(
    package_names: Sequence[str] = DEFAULT_PACKAGE_NAMES,
) -> dict[str, object]:
    """Collect Python/package, PyTorch/ROCm, and visible GPU metadata."""

    packages = {
        name: _distribution_version(name)
        for name in sorted(dict.fromkeys(package_names))
    }
    pytorch: dict[str, object] = {
        "available": False,
        "version": packages.get("torch"),
        "cuda_version": None,
        "hip_version": None,
    }
    gpu: dict[str, object] = {"available": False, "count": 0, "devices": []}
    errors: list[str] = []

    try:
        import torch

        pytorch.update(
            {
                "available": True,
                "version": torch.__version__,
                "cuda_version": getattr(torch.version, "cuda", None),
                "hip_version": getattr(torch.version, "hip", None),
            }
        )
        try:
            cuda_available = bool(torch.cuda.is_available())
            count = int(torch.cuda.device_count()) if cuda_available else 0
            devices = []
            for index in range(count):
                properties = torch.cuda.get_device_properties(index)
                try:
                    capability = list(torch.cuda.get_device_capability(index))
                except Exception:
                    capability = None
                devices.append(
                    {
                        "index": index,
                        "name": str(properties.name),
                        "architecture": (
                            str(properties.gcnArchName)
                            if getattr(properties, "gcnArchName", None) is not None
                            else None
                        ),
                        "capability": capability,
                        "total_memory_bytes": (
                            int(properties.total_memory)
                            if getattr(properties, "total_memory", None) is not None
                            else None
                        ),
                        "multiprocessor_count": (
                            int(properties.multi_processor_count)
                            if getattr(properties, "multi_processor_count", None)
                            is not None
                            else None
                        ),
                    }
                )
            gpu = {"available": cuda_available, "count": count, "devices": devices}
        except Exception as exc:
            errors.append(f"gpu:{type(exc).__name__}:{exc}")
    except Exception as exc:
        errors.append(f"torch:{type(exc).__name__}:{exc}")

    return {
        "platform": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "pytorch": pytorch,
        "rocm": {
            "runtime_version": _rocm_runtime_version(),
            "torch_hip_version": pytorch["hip_version"],
            "rocm_home": os.environ.get("ROCM_HOME"),
        },
        "gpu": gpu,
        "collection_errors": errors,
    }


def build_run_manifest(
    *,
    experiment: str,
    profile: str | object,
    config: object | None,
    seed: int,
    git_state: Mapping[str, object],
    dataset_hashes: Sequence[Mapping[str, object]],
    environment: Mapping[str, object],
    timestamp: datetime | str | None = None,
    run_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a strict versioned manifest from already collected provenance."""

    if not isinstance(experiment, str) or not experiment:
        raise ValueError("experiment must be a non-empty string")
    profile_name = profile if isinstance(profile, str) else getattr(profile, "name", None)
    if not isinstance(profile_name, str) or not profile_name:
        raise ValueError("profile must be a name or an object with a non-empty name")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0 or seed > _MAX_SEED:
        raise ValueError(f"seed must be between 0 and {_MAX_SEED}")
    if config is None:
        resolved_profile = profile
        if isinstance(profile, str):
            from .profiles import get_profile

            resolved_profile = get_profile(profile)
        to_dict = getattr(resolved_profile, "to_dict", None)
        if not callable(to_dict):
            raise ValueError("config is required for profiles without to_dict()")
        config = to_dict()

    created_at = _normalise_timestamp(timestamp)
    identifier = run_id or _make_run_id(experiment, profile_name, seed, created_at)
    if not _RUN_ID.fullmatch(identifier):
        raise ValueError("run_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    config_value = _jsonable(config, location="$.config")
    if not isinstance(config_value, dict):
        raise ValueError("config must be a JSON object")
    manifest: dict[str, object] = {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": identifier,
        "experiment": experiment,
        "profile": profile_name,
        "created_at_utc": created_at,
        "seed": seed,
        "config": config_value,
        "config_sha256": _config_digest(config_value),
        "provenance": {
            "git": _jsonable(git_state, location="$.provenance.git"),
            "datasets": _jsonable(
                list(dataset_hashes), location="$.provenance.datasets"
            ),
            "environment": _jsonable(
                environment, location="$.provenance.environment"
            ),
        },
        "instances": {
            "schema": INSTANCE_SCHEMA,
            "schema_version": INSTANCE_SCHEMA_VERSION,
            "format": "csv",
            "path": "instances.csv",
            "columns": list(INSTANCE_COLUMNS),
            "row_count": 0,
        },
        "metadata": _jsonable(metadata or {}, location="$.metadata"),
    }
    validate_run_manifest(manifest)
    return manifest


def collect_run_manifest(
    *,
    experiment: str,
    profile: str | object,
    config: object | None,
    seed: int,
    dataset_files: Mapping[str, str | os.PathLike[str]],
    repo_root: str | os.PathLike[str] | None = None,
    package_names: Sequence[str] = DEFAULT_PACKAGE_NAMES,
    timestamp: datetime | str | None = None,
    run_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Collect local provenance and build a run manifest."""

    return build_run_manifest(
        experiment=experiment,
        profile=profile,
        config=config,
        seed=seed,
        git_state=collect_git_state(repo_root),
        dataset_hashes=hash_dataset_files(dataset_files),
        environment=collect_environment(package_names),
        timestamp=timestamp,
        run_id=run_id,
        metadata=metadata,
    )


def validate_run_manifest(manifest: Mapping[str, object]) -> None:
    """Validate the stable v1 run-manifest contract."""

    required = {
        "schema",
        "schema_version",
        "run_id",
        "experiment",
        "profile",
        "created_at_utc",
        "seed",
        "config",
        "config_sha256",
        "provenance",
        "instances",
        "metadata",
    }
    missing = required.difference(manifest)
    extra = set(manifest).difference(required)
    if missing or extra:
        raise ValueError(
            f"run manifest keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if manifest["schema"] != RUN_SCHEMA or manifest["schema_version"] != RUN_SCHEMA_VERSION:
        raise ValueError("unsupported run manifest schema")
    run_id = manifest["run_id"]
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("invalid run_id")
    if not isinstance(manifest["experiment"], str) or not manifest["experiment"]:
        raise ValueError("invalid experiment")
    if not isinstance(manifest["profile"], str) or not manifest["profile"]:
        raise ValueError("invalid profile")
    if (
        isinstance(manifest["seed"], bool)
        or not isinstance(manifest["seed"], int)
        or not 0 <= manifest["seed"] <= _MAX_SEED
    ):
        raise ValueError("invalid seed")
    timestamp = manifest["created_at_utc"]
    if not isinstance(timestamp, str) or _normalise_timestamp(timestamp) != timestamp:
        raise ValueError("created_at_utc must be a canonical UTC ISO-8601 timestamp")
    config = _jsonable(manifest["config"], location="$.config")
    if manifest["config_sha256"] != _config_digest(config):
        raise ValueError("config_sha256 does not match config")

    provenance = manifest["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be an object")
    if set(provenance) != {"git", "datasets", "environment"}:
        raise ValueError("provenance must contain git, datasets, and environment")
    if not isinstance(provenance["git"], Mapping):
        raise ValueError("provenance.git must be an object")
    if set(provenance["git"]) != set(GIT_STATE_FIELDS):
        raise ValueError("provenance.git does not match the stable schema")
    git_state = provenance["git"]
    if not isinstance(git_state["available"], bool):
        raise ValueError("provenance.git.available must be boolean")
    sha = git_state["sha"]
    if sha is not None and (
        not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", sha)
    ):
        raise ValueError("provenance.git.sha must be a hexadecimal object ID")
    if git_state["available"] and sha is None:
        raise ValueError("available git provenance requires a SHA")
    for field_name in ("branch", "error"):
        if git_state[field_name] is not None and not isinstance(
            git_state[field_name], str
        ):
            raise ValueError(f"provenance.git.{field_name} must be a string or null")
    if git_state["dirty"] is not None and not isinstance(git_state["dirty"], bool):
        raise ValueError("provenance.git.dirty must be boolean or null")
    for field_name in ("tracked_changes", "untracked_files"):
        value = git_state[field_name]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(
                f"provenance.git.{field_name} must be non-negative or null"
            )
    if not isinstance(provenance["datasets"], list):
        raise ValueError("provenance.datasets must be an array")
    if not isinstance(provenance["environment"], Mapping):
        raise ValueError("provenance.environment must be an object")
    if set(provenance["environment"]) != set(ENVIRONMENT_FIELDS):
        raise ValueError("provenance.environment does not match the stable schema")
    environment = provenance["environment"]
    for field_name in ("platform", "packages", "pytorch", "rocm", "gpu"):
        if not isinstance(environment[field_name], Mapping):
            raise ValueError(f"provenance.environment.{field_name} must be an object")
    if not isinstance(environment["collection_errors"], list):
        raise ValueError("provenance.environment.collection_errors must be an array")
    dataset_ids: set[str] = set()
    for index, dataset in enumerate(provenance["datasets"]):
        if not isinstance(dataset, Mapping):
            raise ValueError(f"provenance.datasets[{index}] must be an object")
        if set(dataset) != set(DATASET_HASH_FIELDS):
            raise ValueError(f"provenance.datasets[{index}] has invalid fields")
        logical_id = dataset["id"]
        if not isinstance(logical_id, str) or not logical_id or logical_id in dataset_ids:
            raise ValueError("dataset IDs must be non-empty and unique")
        dataset_ids.add(logical_id)
        if not isinstance(dataset["path"], str) or not dataset["path"]:
            raise ValueError("dataset paths must be non-empty strings")
        if dataset["algorithm"] != "sha256":
            raise ValueError("dataset hashes must use sha256")
        digest = dataset["digest"]
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError("dataset SHA-256 digests must be lowercase hex")
        byte_count = dataset["bytes"]
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise ValueError("dataset byte counts must be non-negative integers")

    descriptor = manifest["instances"]
    if not isinstance(descriptor, Mapping):
        raise ValueError("instances must be an object")
    if set(descriptor) != {
        "schema",
        "schema_version",
        "format",
        "path",
        "columns",
        "row_count",
    }:
        raise ValueError("instances does not match the stable schema")
    if descriptor.get("schema") != INSTANCE_SCHEMA:
        raise ValueError("unsupported instance schema")
    if descriptor.get("schema_version") != INSTANCE_SCHEMA_VERSION:
        raise ValueError("unsupported instance schema version")
    if descriptor.get("format") != "csv" or descriptor.get("path") != "instances.csv":
        raise ValueError("instances must use instances.csv")
    if descriptor.get("columns") != list(INSTANCE_COLUMNS):
        raise ValueError("instance columns do not match the stable schema")
    row_count = descriptor.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        raise ValueError("instances.row_count must be a non-negative integer")
    if not isinstance(manifest["metadata"], Mapping):
        raise ValueError("metadata must be an object")
    _jsonable(manifest["metadata"], location="$.metadata")


def _row_mapping(row: InstanceRow | Mapping[str, object]) -> dict[str, object]:
    if isinstance(row, InstanceRow):
        return {field.name: getattr(row, field.name) for field in fields(row)}
    if not isinstance(row, Mapping):
        raise TypeError("instance rows must be InstanceRow or mappings")
    allowed = {
        "instance_id",
        "method",
        "metric",
        "value",
        "split",
        "unit",
        "seed",
        "status",
        "duration_seconds",
        "metadata",
    }
    extra = set(row).difference(allowed)
    if extra:
        raise ValueError(f"unknown instance row fields: {sorted(extra)}")
    return dict(row)


def _normalise_instance_row(
    row: InstanceRow | Mapping[str, object],
    manifest: Mapping[str, object],
) -> dict[str, object]:
    raw = _row_mapping(row)
    for field_name in ("instance_id", "method", "metric"):
        if not isinstance(raw.get(field_name), str) or not raw[field_name]:
            raise ValueError(f"instance row {field_name} must be a non-empty string")
    for field_name, default in (("split", "test"), ("unit", ""), ("status", "ok")):
        raw.setdefault(field_name, default)
        if not isinstance(raw[field_name], str):
            raise ValueError(f"instance row {field_name} must be a string")
    if raw["status"] not in INSTANCE_STATUSES:
        raise ValueError(f"instance row status must be one of {INSTANCE_STATUSES}")

    value = raw.get("value")
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("instance row value must be numeric or None")
        if not math.isfinite(float(value)):
            raise ValueError("instance row value must be finite")
    if raw["status"] == "ok" and value is None:
        raise ValueError("successful instance rows require a value")

    seed = raw.get("seed")
    if seed is None:
        seed = manifest["seed"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= _MAX_SEED
    ):
        raise ValueError("instance row seed must be an integer")

    duration = raw.get("duration_seconds")
    if duration is not None:
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError("duration_seconds must be numeric or None")
        if not math.isfinite(float(duration)) or duration < 0:
            raise ValueError("duration_seconds must be finite and non-negative")

    metadata_value = raw.get("metadata") or {}
    if not isinstance(metadata_value, Mapping):
        raise ValueError("instance row metadata must be an object")
    metadata = _jsonable(metadata_value, location="$.instance.metadata")
    return {
        "schema": INSTANCE_SCHEMA,
        "schema_version": INSTANCE_SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "experiment": manifest["experiment"],
        "profile": manifest["profile"],
        "seed": seed,
        "instance_id": raw["instance_id"],
        "split": raw["split"],
        "method": raw["method"],
        "metric": raw["metric"],
        "value": "" if value is None else value,
        "unit": raw["unit"],
        "status": raw["status"],
        "duration_seconds": "" if duration is None else duration,
        "metadata_json": _canonical_json(metadata),
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_run(
    manifest: Mapping[str, object],
    rows: Sequence[InstanceRow | Mapping[str, object]],
    *,
    output_dir: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
) -> RunArtifacts:
    """Write ``manifest.json`` and stable tidy ``instances.csv`` atomically."""

    validate_run_manifest(manifest)
    payload = _jsonable(manifest, location="$")
    normalised_rows = [_normalise_instance_row(row, payload) for row in rows]
    payload["instances"]["row_count"] = len(normalised_rows)
    validate_run_manifest(payload)

    root = Path(output_dir) if output_dir is not None else Path(RESULTS_DIR) / "runs"
    run_dir = root / str(payload["run_id"])
    manifest_path = run_dir / "manifest.json"
    instances_path = run_dir / "instances.csv"
    if not overwrite and (manifest_path.exists() or instances_path.exists()):
        raise FileExistsError(f"run already exists: {run_dir}")

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=list(INSTANCE_COLUMNS),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(normalised_rows)
    manifest_text = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    _atomic_write(instances_path, csv_buffer.getvalue())
    _atomic_write(manifest_path, manifest_text)
    return RunArtifacts(
        run_dir=str(run_dir),
        manifest_path=str(manifest_path),
        instances_path=str(instances_path),
    )


def write_table(name: str, rows: Sequence[Mapping[str, object]],
                columns: Sequence[str] | None = None) -> str:
    """Write a list-of-dicts as a CSV under ``results/``.

    Args:
        name: filename, e.g. ``"table1_image.csv"``.
        rows: sequence of dict rows.
        columns: explicit column order; inferred from the first row if omitted.
    Returns the written path.
    """
    if not rows:
        raise ValueError("write_table: rows is empty")
    if columns is None:
        columns = list(rows[0].keys())
    path = results_path(name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
    return path


def read_table(name: str) -> list[dict[str, str]]:
    """Read back a CSV written by :func:`write_table` (values as strings)."""
    path = results_path(name)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def markdown_table(rows: Sequence[Mapping[str, object]],
                   columns: Sequence[str] | None = None) -> str:
    """Render rows as a GitHub-flavored markdown table (for docs/notebooks)."""
    if not rows:
        return ""
    if columns is None:
        columns = list(rows[0].keys())
    cols = list(columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
        for r in rows
    ]
    return "\n".join([head, sep, *body])


def save_figure(name: str, fig=None) -> str:
    """Save a matplotlib figure under ``results/`` (PNG). No-op-safe if headless.

    Uses the Agg backend implicitly when called on a remote box without a display.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    path = results_path(name)
    (fig or plt.gcf()).savefig(path, dpi=120, bbox_inches="tight")
    return path


def plot_xy(name: str, series: Mapping[str, tuple[Iterable[float], Iterable[float]]],
            xlabel: str = "", ylabel: str = "", title: str = "",
            logx: bool = False, logy: bool = False) -> str:
    """Convenience: line plot of multiple named ``(xs, ys)`` series -> PNG.

    Used for params-vs-PSNR (Fig.5) and rate-distortion (W10) curves.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, (xs, ys) in series.items():
        ax.plot(list(xs), list(ys), marker="o", label=label)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = results_path(name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
