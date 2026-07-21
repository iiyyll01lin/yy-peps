"""End-to-end runner for PEPS paper artifacts and honest blocker reports.

The legacy weekly CSVs are never used as inputs. Every numeric output produced
here lives beside a ``peps.run_manifest`` and tidy ``instances.csv`` receipt.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from apps.image.data import (
    image_to_coords_targets,
    load_fig5_image_manifest,
    load_paper_kodak,
)
from apps.sdf.build import build_paper_sdf
from apps.sdf.data import (
    iter_query_slabs,
    load_paper_sdf_volume,
    sample_sdf_tensor,
)
from apps.texture.data import (
    bundle_to_coords_targets,
    load_paper_texture_set,
)
from data.manifest import (
    DEFAULT_RAW_ROOT,
    DataIntegrityError,
    ManifestError,
    MissingDataError,
    load_manifest,
    resolve_local_path,
    verify_file,
)
from experiments.config import ExperimentConfig, MethodConfig, load_experiment_config
from experiments.runner import (
    ExperimentRunner,
    TensorInstance,
    atomic_torch_save,
    atomic_write_json,
)
from peps.metrics import (
    IoUAccumulator,
    metric_oracles,
    metric_versions,
)
from peps.profiles import get_profile
from peps.report import InstanceRow, collect_run_manifest, validate_run_manifest, write_run
from peps.train import (
    l1_loss,
    make_paper_optimizer,
    mape_loss,
    paper_sdf_recipe,
    split_encoder_decoder_parameters,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results"
ARTIFACTS = (
    "image-fig5",
    "image-table1",
    "texture-table2",
    "sdf-table3-mape",
    "sdf-table3-l1",
    "sdf-table4",
)
PAPER_SDF_METHODS = (
    ("PE", "pe"),
    ("LPE", "lpe"),
    ("TI-Grid", "grid"),
    ("Grid-PEPS", "grid_peps"),
    ("Hash", "hash"),
    ("Hash-PEPS", "hash_peps"),
    ("M-Grid", "m_grid"),
    ("M-PEPS", "m_peps"),
    ("M-Hash", "m_hash"),
    ("M-HashPEPS", "m_hashpeps"),
)
SUMMARY_COLUMNS = (
    "schema",
    "schema_version",
    "run_id",
    "artifact",
    "profile",
    "scope",
    "method",
    "metric",
    "count",
    "mean",
    "unit",
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _config_payload(config: ExperimentConfig) -> dict[str, object]:
    return {
        "schema_version": config.schema_version,
        "name": config.name,
        "paper": config.paper,
        "task": config.task,
        "profile": config.profile,
        "dataset": config.dataset,
        "canonical": config.canonical,
        "seeds": list(config.seeds),
        "training": _plain(config.training),
        "runner": _plain(config.runner),
        "methods": [
            {
                "name": method.name,
                "factory": method.factory,
                "kwargs": _plain(method.kwargs),
                "seeds": None if method.seeds is None else list(method.seeds),
                "role": method.role,
                "training": _plain(method.training),
                "expected_encoder_params": method.expected_encoder_params,
                "expected_total_params": method.expected_total_params,
            }
            for method in config.methods
        ],
        "source": str(config.source),
    }


def _execution_key(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        _plain(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _optional_dependency(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _gpu_receipt() -> dict[str, object]:
    available = bool(torch.cuda.is_available())
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": str(properties.name),
                    "architecture": getattr(properties, "gcnArchName", None),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    return {"available": available, "count": len(devices), "devices": devices}


def _blocker(
    artifact: str,
    code: str,
    requirement: str,
    detail: str,
    remediation: str,
) -> dict[str, str]:
    return {
        "artifact": artifact,
        "code": code,
        "requirement": requirement,
        "detail": detail,
        "remediation": remediation,
    }


def _check_kodak(artifact: str) -> tuple[list[dict], dict[str, object]]:
    try:
        images = load_paper_kodak()
    except (DataIntegrityError, FileNotFoundError, ManifestError) as exc:
        return [
            _blocker(
                artifact,
                "kodak_unavailable",
                "24 checksum-verified original-orientation Kodak PNGs",
                str(exc),
                "python data/download.py fetch kodak",
            )
        ], {"verified_images": 0}
    return [], {"verified_images": len(images)}


def _check_textures(artifact: str) -> tuple[list[dict], dict[str, object]]:
    manifest = load_manifest("textures")
    checked = 0
    try:
        for texture_set in manifest["sets"]:
            for map_spec in texture_set["maps"]:
                verify_file(resolve_local_path(DEFAULT_RAW_ROOT, map_spec), map_spec)
                checked += 1
    except (DataIntegrityError, MissingDataError, ManifestError) as exc:
        return [
            _blocker(
                artifact,
                "paper_textures_unavailable",
                "all maps for the pinned 18-set 4K texture dataset",
                str(exc),
                "python data/download.py fetch textures",
            )
        ], {"verified_sets": 0, "verified_maps": checked}
    return [], {
        "verified_sets": len(manifest["sets"]),
        "verified_maps": checked,
    }


def _check_sdf(
    artifact: str,
    *,
    verify_checksums: bool,
) -> tuple[list[dict], dict[str, object]]:
    checked = []
    required_assets = (
        ("pitted-stonefish",)
        if artifact == "sdf-table4"
        else ("lucy", "pitted-stonefish", "thai-statue", "armadillo")
    )
    for asset_id in required_assets:
        try:
            volume = load_paper_sdf_volume(
                asset_id,
                verify_checksum=verify_checksums,
            )
            if tuple(volume.values.shape) != (512, 512, 512):
                raise DataIntegrityError(
                    f"{asset_id}: expected 512^3, found {volume.values.shape}"
                )
            checked.append(asset_id)
        except (DataIntegrityError, MissingDataError, FileNotFoundError, ManifestError) as exc:
            return [
                _blocker(
                    artifact,
                    "paper_sdf_unavailable",
                    (
                        "provenance-validated Pitted Stonefish 512^3 volume"
                        if artifact == "sdf-table4"
                        else "four provenance-validated 512^3 SDF volumes"
                    ),
                    str(exc),
                    "python data/download.py fetch sdf && "
                    f"python data/preprocess_sdf.py {asset_id}",
                )
            ], {"verified_volumes": checked}
    return [], {"verified_volumes": checked}


def check_prerequisites(
    *,
    profile: str,
    artifacts: Sequence[str] = ARTIFACTS,
    fig5_manifest: str | Path | None = None,
    verify_sdf_checksums: bool = False,
) -> dict[str, object]:
    """Return a machine-readable readiness report without starting training."""

    get_profile(profile)
    unknown = sorted(set(artifacts) - set(ARTIFACTS))
    if unknown:
        raise ValueError(f"unknown artifacts: {unknown}")
    gpu = _gpu_receipt()
    reports = []
    for artifact in artifacts:
        blockers: list[dict] = []
        warnings: list[dict] = []
        checks: dict[str, object] = {}
        if profile == "course_fast":
            checks["synthetic_smoke_available"] = True
        elif artifact == "image-fig5":
            if fig5_manifest is None:
                blockers.append(
                    _blocker(
                        artifact,
                        "fig5_dataset_not_reported",
                        "checksum manifest for the unnamed native-4K image suite",
                        "The paper does not identify the Fig. 5 images or image count.",
                        "Pass --fig5-manifest using results/schemas/fig5_dataset.schema.json.",
                    )
                )
            else:
                try:
                    images = load_fig5_image_manifest(fig5_manifest)
                    checks["verified_images"] = len(images)
                except (DataIntegrityError, FileNotFoundError, ManifestError) as exc:
                    blockers.append(
                        _blocker(
                            artifact,
                            "fig5_dataset_invalid",
                            "checksum-verified native-4K image suite",
                            str(exc),
                            "Fix the supplied --fig5-manifest and local files.",
                        )
                    )
            blockers.append(
                _blocker(
                    artifact,
                    "fig5_training_budget_not_reported",
                    "reported optimizer, batch size, and number of training steps",
                    "The paper states only equal steps across methods.",
                    "Run with explicit --assumed-steps/--assumed-batch-size and report "
                    "the output as a protocol-assumption sensitivity run.",
                )
            )
        elif artifact == "image-table1":
            found, checks = _check_kodak(artifact)
            blockers.extend(found)
            checks.update(
                {
                    "methods": 9,
                    "seeds": 3,
                    "jobs": 24 * 9 * 3,
                    "optimizer_steps_per_job": 120_000,
                    "optimizer_steps_total": 24 * 9 * 3 * 120_000,
                }
            )
            warnings.append(
                {
                    "code": "table1_loss_text_conflict",
                    "detail": "Published Table 1 values equal the appendix L2/GELU/"
                    "dual-LR row, while the main narrative says L1/fixed LR.",
                }
            )
            blockers.append(
                _blocker(
                    artifact,
                    "image_training_steps_not_reported",
                    "paper-reported image optimizer-step count",
                    "The frozen 120000-step value is a local sensitivity assumption.",
                    "Use --allow-protocol-assumptions and retain "
                    "verification_status=protocol_assumption.",
                )
            )
            warnings.append(
                {
                    "code": "large_compute_budget",
                    "detail": "The configured three-seed run contains 648 jobs and "
                    "77,760,000 optimizer steps.",
                }
            )
            for module, package in (
                ("torchmetrics", "torchmetrics"),
                ("lpips", "lpips"),
                ("flip_evaluator", "flip-evaluator"),
            ):
                if not _optional_dependency(module):
                    blockers.append(
                        _blocker(
                            artifact,
                            f"missing_{module}",
                            package,
                            f"Python module {module!r} is unavailable.",
                            "pip install -e '.[paper]'",
                        )
                    )
        elif artifact == "texture-table2":
            found, checks = _check_textures(artifact)
            blockers.extend(found)
            checks.update(
                {
                    "methods": 11,
                    "seeds": 3,
                    "jobs": 18 * 11 * 3,
                    "optimizer_steps_per_job": 120_000,
                    "optimizer_steps_total": 18 * 11 * 3 * 120_000,
                }
            )
            if not _optional_dependency("torchmetrics"):
                blockers.append(
                    _blocker(
                        artifact,
                        "missing_torchmetrics",
                        "torchmetrics windowed SSIM",
                        "Python module 'torchmetrics' is unavailable.",
                        "pip install -e '.[paper]'",
                    )
                )
            warnings.append(
                {
                    "code": "texture_file_selection_assumption",
                    "detail": "The paper names sets/categories but not exact source files; "
                    "the checked-in manifest freezes a non-duplicating selection.",
                }
            )
            warnings.append(
                {
                    "code": "optimizer_and_seed_not_reported",
                    "detail": "The runner freezes Adam and three seeds; the paper "
                    "does not report either choice.",
                }
            )
            warnings.append(
                {
                    "code": "large_compute_budget",
                    "detail": "The configured three-seed run contains 594 jobs and "
                    "71,280,000 optimizer steps at native 4K.",
                }
            )
        else:
            found, checks = _check_sdf(
                artifact,
                verify_checksums=verify_sdf_checksums,
            )
            blockers.extend(found)
            if artifact == "sdf-table4":
                checks.update(
                    {
                        "methods": 9,
                        "budget_rows": 2,
                        "jobs": 17,
                        "optimizer_steps_per_job": 120_000,
                        "optimizer_steps_total": 17 * 120_000,
                    }
                )
            else:
                checks.update(
                    {
                        "instances": 4,
                        "methods": 10,
                        "jobs": 40,
                        "optimizer_steps_per_job": 120_000,
                        "optimizer_steps_total": 40 * 120_000,
                    }
                )
            warnings.append(
                {
                    "code": "unreleased_sdf_converter",
                    "detail": "The authors' C++/HIP mesh converter is unavailable; "
                    "the checksum-bearing mesh-to-sdf protocol is an approximation.",
                }
            )
            warnings.append(
                {
                    "code": "optimizer_and_seed_not_reported",
                    "detail": "The runner freezes Adam and seed 0; the paper does "
                    "not report either choice.",
                }
            )
        if profile == "paper_exact" and not gpu["available"]:
            blockers.append(
                _blocker(
                    artifact,
                    "gpu_unavailable",
                    "GPU execution for the full paper workload",
                    "PyTorch reports no CUDA/ROCm device.",
                    "Run on a machine with a visible supported GPU; do not start the "
                    "120000-step paper workload on CPU.",
                )
            )
        reports.append(
            {
                "artifact": artifact,
                "ready": not blockers,
                "checks": checks,
                "blockers": blockers,
                "warnings": warnings,
            }
        )
    return {
        "schema": "peps.reproduction_prerequisites",
        "schema_version": 1,
        "profile": profile,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "ready": all(report["ready"] for report in reports),
        "environment": {"gpu": gpu, "torch": torch.__version__},
        "artifacts": reports,
    }


def _filter_methods(
    config: ExperimentConfig,
    names: Sequence[str] | None,
) -> ExperimentConfig:
    if not names:
        return config
    requested = set(names)
    selected = tuple(method for method in config.methods if method.name in requested)
    missing = sorted(requested - {method.name for method in selected})
    if missing:
        raise ValueError(f"unknown methods for {config.name}: {missing}")
    return replace(config, methods=selected)


def _record_rows(records: Sequence[Mapping[str, object]]) -> list[InstanceRow]:
    rows = []
    for record in records:
        metrics = record["metrics"]
        if not isinstance(metrics, Mapping):
            raise ValueError("record metrics must be a mapping")
        for metric_name, value in metrics.items():
            metadata: dict[str, object] = {
                "parameters": record.get("parameters"),
                "compression_factor": record.get("compression_factor"),
                "role": record.get("role", "canonical"),
                "metric_versions": record.get("metric_versions", {}),
            }
            name = str(metric_name)
            if "/map/" in name:
                metric, suffix = name.split("/map/", 1)
                map_id, semantic = suffix.rsplit("/", 1)
                metadata.update(
                    {
                        "scope": "texture_map",
                        "map_id": map_id,
                        "semantic": semantic,
                    }
                )
                name = metric
            elif "/semantic/" in name:
                metric, semantic = name.split("/semantic/", 1)
                metadata.update(
                    {"scope": "texture_semantic", "semantic": semantic}
                )
                name = metric
            else:
                metadata["scope"] = "instance"
            rows.append(
                InstanceRow(
                    instance_id=str(record["instance"]),
                    method=str(record["method"]),
                    metric=name,
                    value=float(value),
                    unit="dB" if name == "psnr" else "",
                    seed=int(record["seed"]),
                    duration_seconds=float(record.get("elapsed_seconds", 0.0)),
                    metadata=metadata,
                )
            )
    return rows


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def aggregate_run(run_dir: str | Path, *, artifact: str | None = None) -> Path:
    """Create a schema-stable aggregate only from a validated run receipt."""

    directory = Path(run_dir)
    with (directory / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_run_manifest(manifest)
    resolved_artifact = artifact or manifest["metadata"].get("artifact")
    if resolved_artifact not in ARTIFACTS and not str(resolved_artifact).startswith(
        "smoke-"
    ):
        raise ValueError("run manifest does not name a supported artifact")
    with (directory / "instances.csv").open(newline="", encoding="utf-8") as handle:
        observations = list(csv.DictReader(handle))

    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for row in observations:
        if row["status"] != "ok" or row["value"] == "":
            continue
        metadata = json.loads(row["metadata_json"])
        scope = str(metadata.get("scope", "instance"))
        semantic = metadata.get("semantic")
        if resolved_artifact in {"texture-table2", "smoke-texture"}:
            if scope != "texture_map":
                continue
            scopes = (f"semantic:{semantic}", "global")
        else:
            scopes = ("global",)
        for summary_scope in scopes:
            key = (
                summary_scope,
                row["method"],
                row["metric"],
                row["unit"],
            )
            grouped.setdefault(key, []).append(float(row["value"]))

    output_rows = []
    for (scope, method, metric, unit), values in sorted(grouped.items()):
        output_rows.append(
            {
                "schema": "peps.paper_artifact_summary",
                "schema_version": 1,
                "run_id": manifest["run_id"],
                "artifact": resolved_artifact,
                "profile": manifest["profile"],
                "scope": scope,
                "method": method,
                "metric": metric,
                "count": len(values),
                "mean": sum(values) / len(values),
                "unit": unit,
            }
        )
    path = directory / "summary.csv"
    _atomic_csv(path, output_rows)
    atomic_write_json(
        directory / "summary.json",
        {
            "schema": "peps.paper_artifact_summary",
            "schema_version": 1,
            "run_id": manifest["run_id"],
            "artifact": resolved_artifact,
            "rows": output_rows,
        },
    )
    return path


def _write_receipt(
    *,
    artifact: str,
    profile_name: str,
    config: Mapping[str, object],
    seed: int,
    dataset_files: Mapping[str, Path],
    records: Sequence[Mapping[str, object]],
    output_root: Path,
    raw_output: Path,
    verification_status: str,
) -> dict[str, object]:
    profile = get_profile(profile_name)
    manifest = collect_run_manifest(
        experiment=artifact,
        profile=profile,
        config={
            "profile": profile.to_dict(),
            "resolved_experiment": _plain(config),
            "metric_oracles": metric_oracles(),
        },
        seed=seed,
        dataset_files=dataset_files,
        repo_root=ROOT,
        metadata={
            "artifact": artifact,
            "verification_status": verification_status,
            "raw_output": str(raw_output),
            "summary": "summary.csv",
        },
    )
    written = write_run(
        manifest,
        _record_rows(records),
        output_dir=output_root / "runs",
    )
    summary = aggregate_run(written.run_dir, artifact=artifact)
    return {
        "run_id": manifest["run_id"],
        "run_dir": written.run_dir,
        "manifest": written.manifest_path,
        "instances": written.instances_path,
        "summary": str(summary),
    }


def _tensor_run(
    *,
    artifact: str,
    profile_name: str,
    config: ExperimentConfig,
    instances: Sequence[TensorInstance],
    dataset_files: Mapping[str, Path],
    output_root: Path,
    device: torch.device,
    force: bool,
    verification_status: str,
) -> dict[str, object]:
    payload = {
        "artifact": artifact,
        "config": _config_payload(config),
        "instances": [instance.name for instance in instances],
    }
    raw_output = (
        output_root
        / "work"
        / artifact
        / _execution_key(payload)
    )
    runner = ExperimentRunner(
        config,
        raw_output,
        device=device,
        force=force,
    )
    records = runner.run(instances)
    return _write_receipt(
        artifact=artifact,
        profile_name=profile_name,
        config=_config_payload(config),
        seed=config.seeds[0],
        dataset_files=dataset_files,
        records=records,
        output_root=output_root,
        raw_output=raw_output,
        verification_status=verification_status,
    )


def run_image_table1(
    *,
    output_root: Path,
    device: torch.device,
    instance_ids: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, object]:
    config = load_experiment_config(ROOT / "configs/paper/image_full.toml")
    config = _filter_methods(config, methods)
    loaded = load_paper_kodak(instance_ids=instance_ids)
    instances = []
    files = {}
    for image in loaded:
        coords, targets, (height, width) = image_to_coords_targets(image.tensor)
        instances.append(
            TensorInstance(
                image.image_id,
                coords,
                targets,
                shape=(height, width, 3),
                metadata={
                    "num_signal_values": targets.numel(),
                    "resolution_xy": [width, height],
                    "color_space": image.color_space,
                },
            )
        )
        files[image.image_id] = image.source_path
    return _tensor_run(
        artifact="image-table1",
        profile_name="paper_exact",
        config=config,
        instances=instances,
        dataset_files=files,
        output_root=output_root,
        device=device,
        force=force,
        verification_status="protocol_assumption",
    )


def _fig5_config(
    manifest_path: Path,
    *,
    steps: int,
    batch_size: int,
) -> ExperimentConfig:
    methods = []
    for method_name, method_key in (
        ("BI-Grid", "bi_grid"),
        ("LPE", "lpe"),
        ("Grid-PEPS", "grid_peps"),
    ):
        for resolution in (16, 32, 64, 128):
            for feature_dim in (8, 16, 32, 64):
                methods.append(
                    MethodConfig(
                        name=f"{method_name}-r{resolution}-d{feature_dim}",
                        factory="apps.image.build:build_paper_fig5",
                        kwargs={
                            "method": method_key,
                            "resolution": resolution,
                            "feature_dim": feature_dim,
                        },
                    )
                )
    return ExperimentConfig(
        schema_version=1,
        name="image-fig5-full-factorial",
        paper="PEPS Extended arXiv:2604.24167v1",
        task="image",
        profile="full",
        dataset="user-receipted-paper-4k-image-suite",
        canonical=False,
        seeds=(0,),
        training={
            "task": "image",
            "loss": "l1",
            "steps": steps,
            "batch_size": batch_size,
            "model_lr": 0.01,
            "encoder_lr": 0.01,
            "cosine": False,
            "log_every": 200,
            "checkpoint_every": 1000,
        },
        runner={
            "metrics": ("psnr",),
            "render_chunk": 262_144,
            "protocol_note": "Dataset, optimizer, batch size, and steps are "
            "not reported by the paper; this is a sensitivity run.",
        },
        methods=tuple(methods),
        source=manifest_path,
    )


def run_image_fig5(
    *,
    manifest_path: Path,
    steps: int,
    batch_size: int,
    output_root: Path,
    device: torch.device,
    instance_ids: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, object]:
    loaded = load_fig5_image_manifest(manifest_path)
    if instance_ids:
        selected = set(instance_ids)
        loaded = tuple(image for image in loaded if image.image_id in selected)
        missing = sorted(selected - {image.image_id for image in loaded})
        if missing:
            raise ValueError(f"unknown Fig. 5 images: {missing}")
    config = _filter_methods(
        _fig5_config(manifest_path, steps=steps, batch_size=batch_size),
        methods,
    )
    instances = []
    files = {"fig5-dataset-manifest": manifest_path}
    for image in loaded:
        coords, targets, (height, width) = image_to_coords_targets(image.tensor)
        instances.append(
            TensorInstance(
                image.image_id,
                coords,
                targets,
                shape=(height, width, 3),
                metadata={
                    "num_signal_values": targets.numel(),
                    "resolution_xy": [width, height],
                },
            )
        )
        files[image.image_id] = image.source_path
    return _tensor_run(
        artifact="image-fig5",
        profile_name="paper_exact",
        config=config,
        instances=instances,
        dataset_files=files,
        output_root=output_root,
        device=device,
        force=force,
        verification_status="protocol_assumption",
    )


def run_texture_table2(
    *,
    output_root: Path,
    device: torch.device,
    instance_ids: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, object]:
    config = _filter_methods(
        load_experiment_config(ROOT / "configs/paper/texture_full.toml"),
        methods,
    )
    manifest = load_manifest("textures")
    available_ids = tuple(item["id"] for item in manifest["sets"])
    selected_ids = available_ids if instance_ids is None else tuple(instance_ids)
    unknown = sorted(set(selected_ids) - set(available_ids))
    if unknown:
        raise ValueError(f"unknown paper texture sets: {unknown}")
    files: dict[str, Path] = {}
    records: list[dict] = []
    payload = {
        "artifact": "texture-table2",
        "config": _config_payload(config),
        "instances": list(selected_ids),
    }
    raw_output = (
        output_root
        / "work"
        / "texture-table2"
        / _execution_key(payload)
    )
    runner = ExperimentRunner(
        config,
        raw_output,
        device=device,
        force=force,
    )
    for set_id in selected_ids:
        loaded = load_paper_texture_set(set_id)
        coords, targets, (height, width) = bundle_to_coords_targets(loaded.tensor)
        map_specs = []
        for texture_map in loaded.maps:
            map_specs.append(
                {
                    "map_id": texture_map.map_id,
                    "semantic": texture_map.semantic,
                    "channel_start": texture_map.channel_slice.start,
                    "channel_stop": texture_map.channel_slice.stop,
                }
            )
            files[f"{set_id}:{texture_map.map_id}"] = texture_map.source_path
        instance = TensorInstance(
            set_id,
            coords,
            targets,
            shape=(height, width, targets.shape[1]),
            metadata={
                "num_signal_values": targets.numel(),
                "texture_maps": map_specs,
                "resolution_xy": [width, height],
            },
        )
        records.extend(runner.run((instance,)))
        del instance, coords, targets, loaded
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return _write_receipt(
        artifact="texture-table2",
        profile_name="paper_exact",
        config=_config_payload(config),
        seed=config.seeds[0],
        dataset_files=files,
        records=records,
        output_root=output_root,
        raw_output=raw_output,
        verification_status="paper_protocol_with_explicit_dataset_optimizer_and_seed_assumptions",
    )


def _sdf_parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    encoder, decoder = split_encoder_decoder_parameters(model)
    encoder_count = sum(parameter.numel() for parameter in encoder)
    decoder_count = sum(parameter.numel() for parameter in decoder)
    return {
        "encoder": encoder_count,
        "decoder": decoder_count,
        "total": encoder_count + decoder_count,
    }


@torch.no_grad()
def _evaluate_sdf(
    model: torch.nn.Module,
    volume: np.ndarray,
    *,
    device: torch.device,
    render_chunk: int,
    mape_epsilon: float,
) -> dict[str, float]:
    resolution = int(volume.shape[0])
    slab_depth = max(1, render_chunk // (resolution * resolution))
    accumulator = IoUAccumulator()
    absolute_sum = 0.0
    percentage_sum = 0.0
    value_count = 0
    model.eval()
    for z_slice, coords in iter_query_slabs(
        resolution,
        slab_depth=slab_depth,
    ):
        prediction_parts = []
        for start in range(0, coords.shape[0], render_chunk):
            prediction_parts.append(
                model(coords[start : start + render_chunk].to(device)).cpu()
            )
        prediction = torch.cat(prediction_parts, dim=0)
        target = torch.from_numpy(
            np.array(volume[z_slice], dtype=np.float32, copy=True)
        ).reshape(-1, 1)
        accumulator.update(prediction < 0, target < 0)
        absolute_sum += float((prediction - target).abs().double().sum().item())
        percentage_sum += float(
            (
                (prediction - target).abs()
                / target.abs().clamp_min(mape_epsilon)
            )
            .double()
            .sum()
            .item()
        )
        value_count += target.numel()
    return {
        "iou": accumulator.compute(),
        "l1": absolute_sum / value_count,
        "mape": 100.0 * percentage_sum / value_count,
    }


def _train_sdf_job(
    *,
    instance_id: str,
    volume: np.ndarray,
    volume_on_device: torch.Tensor,
    display_name: str,
    method_key: str,
    loss_name: str,
    budget: int,
    seed: int,
    raw_output: Path,
    device: torch.device,
    force: bool,
) -> dict[str, object]:
    budget_name = f"{budget}x"
    result_path = (
        raw_output
        / "raw"
        / instance_id
        / display_name
        / loss_name
        / budget_name
        / f"seed-{seed}.json"
    )
    checkpoint_path = (
        raw_output
        / "checkpoints"
        / instance_id
        / display_name
        / loss_name
        / budget_name
        / f"seed-{seed}.pt"
    )
    if result_path.is_file() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, reported = build_paper_sdf(
        method_key,
        encoder_parameter_multiplier=budget,
    )
    counts = _sdf_parameter_counts(model)
    if counts["total"] != reported:
        raise AssertionError("SDF builder parameter count mismatch")
    model = model.to(device)
    recipe = replace(
        paper_sdf_recipe(loss=loss_name, seed=seed),
        device=device,
        checkpoint_every=1000,
    )
    optimizer = make_paper_optimizer(model, recipe)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    start_step = 0
    if checkpoint_path.is_file() and not force:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("schema_version") != 1:
            raise ValueError("unsupported SDF checkpoint schema")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.set_state(checkpoint["coordinate_generator"])
        start_step = int(checkpoint["step"])
    loss_function = (
        l1_loss
        if loss_name == "l1"
        else lambda prediction, target: mape_loss(
            prediction,
            target,
            epsilon=recipe.mape_epsilon,
        )
    )
    started = time.perf_counter()
    for step_index in range(start_step, recipe.total_steps):
        coords = torch.rand(
            recipe.batch_size,
            3,
            generator=generator,
        ).to(device)
        target = sample_sdf_tensor(volume_on_device, coords)
        prediction = model(coords)
        loss = loss_function(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        completed = step_index + 1
        if (
            completed % recipe.checkpoint_every == 0
            or completed == recipe.total_steps
        ):
            atomic_torch_save(
                checkpoint_path,
                {
                    "schema_version": 1,
                    "step": completed,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "coordinate_generator": generator.get_state(),
                },
            )
    metrics = _evaluate_sdf(
        model,
        volume,
        device=device,
        render_chunk=262_144,
        mape_epsilon=recipe.mape_epsilon,
    )
    record = {
        "schema_version": 1,
        "experiment": "sdf-paper",
        "profile": "paper_exact",
        "paper": "PEPS Extended arXiv:2604.24167v1",
        "task": "sdf",
        "dataset": "paper-sdf-4-512cubed",
        "instance": instance_id,
        "method": display_name,
        "role": "canonical",
        "seed": seed,
        "parameters": counts,
        "compression_factor": volume.size / counts["total"],
        "training": {
            **asdict(recipe),
            "device": str(device),
            "coordinate_sampling": "fresh_uniform_[0,1]^3_each_step",
            "target_sampling": "trilinear_512cubed_align_corners_true",
            "eikonal": False,
            "encoder_parameter_multiplier": budget,
        },
        "metrics": metrics,
        "metric_versions": metric_versions(),
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(result_path, record)
    return record


def run_sdf_artifact(
    artifact: str,
    *,
    output_root: Path,
    device: torch.device,
    instance_ids: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, object]:
    if artifact not in {"sdf-table3-mape", "sdf-table3-l1", "sdf-table4"}:
        raise ValueError(f"unsupported SDF artifact {artifact}")
    selected_instances = (
        ("pitted-stonefish",)
        if artifact == "sdf-table4"
        else ("lucy", "pitted-stonefish", "thai-statue", "armadillo")
    )
    if instance_ids is not None:
        requested = set(instance_ids)
        selected_instances = tuple(
            instance for instance in selected_instances if instance in requested
        )
        missing = sorted(requested - set(selected_instances))
        if missing:
            raise ValueError(f"invalid instances for {artifact}: {missing}")
    selected_methods = PAPER_SDF_METHODS
    if artifact == "sdf-table4":
        # Match the published Table 4 columns exactly. Hash-PEPS appears in
        # Table 3 but is not one of the nine methods in the 1x/8x table.
        selected_methods = tuple(
            item for item in selected_methods if item[0] != "Hash-PEPS"
        )
    if methods:
        requested_methods = set(methods)
        selected_methods = tuple(
            item for item in selected_methods if item[0] in requested_methods
        )
        missing_methods = sorted(
            requested_methods - {item[0] for item in selected_methods}
        )
        if missing_methods:
            raise ValueError(f"unknown SDF methods: {missing_methods}")
    loss_name = "l1" if artifact in {"sdf-table3-l1", "sdf-table4"} else "mape"
    budgets = (1, 8) if artifact == "sdf-table4" else (1,)
    payload = {
        "artifact": artifact,
        "instances": list(selected_instances),
        "methods": [item[0] for item in selected_methods],
        "loss": loss_name,
        "budgets": list(budgets),
        "profile": get_profile("paper_exact").sdf,
    }
    raw_output = output_root / "work" / artifact / _execution_key(payload)
    files: dict[str, Path] = {}
    records = []
    for instance_id in selected_instances:
        loaded = load_paper_sdf_volume(instance_id)
        files[f"{instance_id}:volume"] = loaded.volume_path
        files[f"{instance_id}:provenance"] = loaded.provenance_path
        # Materialize once per instance. Reusing this tensor across methods avoids
        # a 512^3 host-to-device copy on every job.
        volume_tensor = torch.from_numpy(
            np.array(loaded.values, dtype=np.float32, copy=True)
        ).to(device)
        pe_cache: dict[int, dict[str, object]] = {}
        for display_name, method_key in selected_methods:
            for budget in budgets:
                if method_key == "pe" and budget == 8 and 1 in pe_cache:
                    reused = dict(pe_cache[1])
                    reused["method"] = f"{display_name} [8x]"
                    reused["training"] = {
                        **dict(reused["training"]),
                        "encoder_parameter_multiplier": 8,
                        "shared_no_encoder_measurement": True,
                    }
                    records.append(reused)
                    continue
                record = _train_sdf_job(
                    instance_id=instance_id,
                    volume=loaded.values,
                    volume_on_device=volume_tensor,
                    display_name=(
                        f"{display_name} [{budget}x]"
                        if artifact == "sdf-table4"
                        else display_name
                    ),
                    method_key=method_key,
                    loss_name=loss_name,
                    budget=budget,
                    seed=0,
                    raw_output=raw_output,
                    device=device,
                    force=force,
                )
                records.append(record)
                if method_key == "pe":
                    pe_cache[budget] = record
        del volume_tensor, loaded
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return _write_receipt(
        artifact=artifact,
        profile_name="paper_exact",
        config=_plain(payload),
        seed=0,
        dataset_files=files,
        records=records,
        output_root=output_root,
        raw_output=raw_output,
        verification_status="paper_protocol_with_explicit_optimizer_seed_and_sdf_preprocessor_assumptions",
    )


def _smoke_instance(task: str, output_root: Path) -> tuple[TensorInstance, Path]:
    generator = torch.Generator().manual_seed(17)
    if task == "image":
        line = torch.linspace(0.0, 1.0, 8)
        y, x = torch.meshgrid(line, line, indexing="ij")
        coords = torch.stack((x.reshape(-1), y.reshape(-1)), dim=1)
        targets = torch.stack(
            (coords[:, 0], coords[:, 1], (coords[:, 0] + coords[:, 1]) / 2),
            dim=1,
        )
        instance = TensorInstance(
            "synthetic-image",
            coords,
            targets,
            shape=(8, 8, 3),
        )
    elif task == "texture":
        line = torch.linspace(0.0, 1.0, 16)
        y, x = torch.meshgrid(line, line, indexing="ij")
        coords = torch.stack((x.reshape(-1), y.reshape(-1)), dim=1)
        first = torch.stack((coords[:, 0], coords[:, 1], coords[:, 0] * coords[:, 1]), dim=1)
        second = torch.stack(
            (
                0.5 + 0.5 * torch.sin(coords[:, 0] * 2 * math.pi),
                coords[:, 1].square(),
                1.0 - coords[:, 0],
            ),
            dim=1,
        )
        targets = torch.cat((first, second), dim=1)
        instance = TensorInstance(
            "synthetic-texture",
            coords,
            targets,
            shape=(16, 16, 6),
            metadata={
                "texture_maps": [
                    {
                        "map_id": "diffuse",
                        "semantic": "DIFF",
                        "channel_start": 0,
                        "channel_stop": 3,
                    },
                    {
                        "map_id": "rough",
                        "semantic": "rough",
                        "channel_start": 3,
                        "channel_stop": 6,
                    },
                ]
            },
        )
    elif task == "sdf":
        coords = torch.rand(256, 3, generator=generator)
        centered = coords * 2.0 - 1.0
        targets = centered.norm(dim=1, keepdim=True) - 0.6
        instance = TensorInstance("synthetic-sdf", coords, targets)
    else:
        raise ValueError(f"unknown smoke task {task!r}")
    input_dir = output_root / "smoke-inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / f"{task}.pt"
    torch.save(
        {
            "schema": "peps.synthetic_smoke_input",
            "schema_version": 1,
            "task": task,
            "coords": instance.coords,
            "targets": instance.targets,
            "shape": instance.shape,
            "metadata": dict(instance.metadata),
        },
        path,
    )
    return instance, path


def run_course_smoke(
    task: str,
    *,
    output_root: Path,
    device: torch.device,
    force: bool = False,
) -> dict[str, object]:
    config = load_experiment_config(
        ROOT / f"configs/paper/{task}_smoke.toml"
    )
    instance, input_path = _smoke_instance(task, output_root)
    return _tensor_run(
        artifact=f"smoke-{task}",
        profile_name="course_fast",
        config=config,
        instances=(instance,),
        dataset_files={instance.name: input_path},
        output_root=output_root,
        device=device,
        force=force,
        verification_status="course_fast_smoke_not_paper_comparable",
    )


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check")
    check.add_argument("--profile", choices=("paper_exact", "course_fast"), default="paper_exact")
    check.add_argument("--artifact", choices=ARTIFACTS, action="append")
    check.add_argument("--fig5-manifest", type=Path)
    check.add_argument("--verify-sdf-checksums", action="store_true")
    check.add_argument("--output", type=Path)

    run = subparsers.add_parser("run")
    run.add_argument("--artifact", choices=ARTIFACTS, required=True)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--device", default="auto")
    run.add_argument("--instance", action="append")
    run.add_argument("--method", action="append")
    run.add_argument("--force", action="store_true")
    run.add_argument("--fig5-manifest", type=Path)
    run.add_argument("--assumed-steps", type=int)
    run.add_argument("--assumed-batch-size", type=int, default=60_000)
    run.add_argument("--allow-protocol-assumptions", action="store_true")

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--task", choices=("image", "texture", "sdf", "all"), default="all")
    smoke.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    smoke.add_argument("--device", default="auto")
    smoke.add_argument("--force", action="store_true")

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--run-dir", type=Path, required=True)
    aggregate.add_argument("--artifact")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "check":
        payload = check_prerequisites(
            profile=arguments.profile,
            artifacts=tuple(arguments.artifact or ARTIFACTS),
            fig5_manifest=arguments.fig5_manifest,
            verify_sdf_checksums=arguments.verify_sdf_checksums,
        )
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if arguments.output is not None:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(text, encoding="utf-8")
        print(text, end="")
        return 0 if payload["ready"] else 2
    if arguments.command == "aggregate":
        print(aggregate_run(arguments.run_dir, artifact=arguments.artifact))
        return 0
    if arguments.command == "smoke":
        tasks = ("image", "texture", "sdf") if arguments.task == "all" else (arguments.task,)
        outputs = [
            run_course_smoke(
                task,
                output_root=arguments.output_root,
                device=_device(arguments.device),
                force=arguments.force,
            )
            for task in tasks
        ]
        print(json.dumps(outputs, indent=2, sort_keys=True))
        return 0

    artifact = arguments.artifact
    if artifact in {"image-fig5", "image-table1"} and not arguments.allow_protocol_assumptions:
        raise SystemExit(
            "image paper runs require --allow-protocol-assumptions because the "
            "paper does not report all training/dataset details"
        )
    common = {
        "output_root": arguments.output_root,
        "device": _device(arguments.device),
        "instance_ids": arguments.instance,
        "methods": arguments.method,
        "force": arguments.force,
    }
    if artifact == "image-table1":
        output = run_image_table1(**common)
    elif artifact == "image-fig5":
        if arguments.fig5_manifest is None or arguments.assumed_steps is None:
            raise SystemExit(
                "image-fig5 requires --fig5-manifest and --assumed-steps"
            )
        output = run_image_fig5(
            manifest_path=arguments.fig5_manifest,
            steps=arguments.assumed_steps,
            batch_size=arguments.assumed_batch_size,
            **common,
        )
    elif artifact == "texture-table2":
        output = run_texture_table2(**common)
    else:
        output = run_sdf_artifact(artifact, **common)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
