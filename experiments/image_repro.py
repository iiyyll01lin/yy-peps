"""Auditable orchestration for the PEPS image/core reproduction matrix.

The command keeps canonical, smoke, and appendix outputs in disjoint
namespaces.  Full jobs refuse CPU execution, are deterministically sharded by
instance/method/seed, and reuse the checkpoint contract from
``experiments.runner``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from apps.image.data import image_to_coords_targets, load_paper_kodak
from data.manifest import hash_file, load_manifest
from experiments.config import ExperimentConfig, load_experiment_config
from experiments.full_run_authorization import (
    validate_image_table1_authorization,
)
from experiments.runner import (
    ExperimentRunner,
    TensorInstance,
    atomic_write_json,
    enumerate_jobs,
)
from peps.metrics import metric_versions
from peps.report import collect_environment, collect_git_state
from peps.train import paper_recipe_from_mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results"
SCHEMA_VERSION = 1
ARTIFACT_CONFIGS = {
    "table1": ROOT / "configs/paper/image_full.toml",
    "table5": ROOT / "configs/paper/image_table5_full.toml",
    "core-ablations": ROOT / "configs/paper/image_core_ablations_full.toml",
    "recipe-ablations": ROOT
    / "configs/paper/image_recipe_ablations_full.toml",
    "smoke": ROOT / "configs/paper/image_repro_smoke.toml",
    "appendix-smoke": ROOT / "configs/paper/image_appendix_smoke.toml",
}
CODE_RECEIPT_PATHS = (
    ROOT / "apps/image/build.py",
    ROOT / "experiments/full_run_authorization.py",
    ROOT / "experiments/image_models.py",
    ROOT / "experiments/runner.py",
    ROOT / "peps/train.py",
)
PAPER_TABLE1 = {
    "PE": {"psnr": 39.91, "flip": 3.71e-2, "lpips": 1.47e-2, "lsd": 4.46e-2, "ssim": 0.960},
    "LPE": {"psnr": 45.06, "flip": 1.84e-2, "lpips": 1.62e-3, "lsd": 7.49e-3, "ssim": 0.992},
    "NTC_N": {"psnr": 44.87, "flip": 1.96e-2, "lpips": 2.23e-3, "lsd": 9.74e-3, "ssim": 0.992},
    "Grid": {"psnr": 45.30, "flip": 1.83e-2, "lpips": 1.37e-3, "lsd": 1.24e-2, "ssim": 0.992},
    "G-PEPS": {"psnr": 47.72, "flip": 2.08e-2, "lpips": 1.30e-3, "lsd": 4.19e-3, "ssim": 0.993},
    "G-P-PEPS": {"psnr": 47.83, "flip": 2.24e-2, "lpips": 1.05e-3, "lsd": 4.41e-3, "ssim": 0.993},
    "NTC_PEPS": {"psnr": 48.02, "flip": 2.07e-2, "lpips": 1.43e-3, "lsd": 4.07e-3, "ssim": 0.994},
    "NTC_PinkPEPS": {"psnr": 48.07, "flip": 2.14e-2, "lpips": 1.20e-3, "lsd": 4.22e-3, "ssim": 0.994},
    "G-P-PEPS-25": {"psnr": 44.89, "flip": 2.81e-2, "lpips": 3.12e-3, "lsd": 8.45e-3, "ssim": 0.987},
}
PAPER_TABLE5 = {
    "Grid": {"psnr_l1": 40.871, "psnr_l2": 45.30, "ssim_l1": 0.973, "ssim_l2": 0.992},
    "G-P-PEPS": {"psnr_l1": 44.237, "psnr_l2": 47.83, "ssim_l1": 0.975, "ssim_l2": 0.993},
    "NTC_N": {"psnr_l1": 41.229, "psnr_l2": 44.87, "ssim_l1": 0.968, "ssim_l2": 0.992},
}
PROTOCOL_LIMITATIONS = (
    {
        "code": "image_training_budget_not_reported",
        "detail": "The paper does not disclose Kodak optimizer steps or batch size; 120000 steps and batch 60000 are sensitivity assumptions.",
    },
    {
        "code": "optimizer_seed_output_not_reported",
        "detail": "Adam, seeds 0/1/2, and a linear RGB output are explicit local assumptions because the paper does not report them.",
    },
    {
        "code": "table1_recipe_text_conflict",
        "detail": "Table 1 values equal the Appendix L2/GELU/dual-cosine row, while the main image text describes L1/LeakyReLU/fixed LR.",
    },
    {
        "code": "fig5_dataset_and_steps_not_reported",
        "detail": "The paper identifies neither the native-4K Figure 5 images nor the training budget, so no exact Figure 5 run can be launched.",
    },
    {
        "code": "lsd_normalization_not_reported",
        "detail": "The paper names LSD but does not publish normalization; peps.metrics freezes an explicit log1p orthonormal-FFT oracle.",
    },
    {
        "code": "appendix_numeric_protocol_not_reported",
        "detail": "No-sharing, remove-x, sum, WIRE, and related Appendix text has no exact settings or numeric rows; these configs are labelled sensitivity runs.",
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_identity(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> dict[str, object] | None:
    """Return boot-scoped process identity evidence, not just a reusable PID."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        return None
    process_root = proc_root / str(pid)
    try:
        stat_text = (process_root / "stat").read_text(encoding="utf-8")
        command = (process_root / "cmdline").read_bytes()
        boot_id = (
            proc_root / "sys/kernel/random/boot_id"
        ).read_text(encoding="utf-8").strip()
        closing_parenthesis = stat_text.rfind(")")
        if closing_parenthesis < 0:
            return None
        # The suffix starts at proc(5) field 3 (state); starttime is field 22.
        stat_fields = stat_text[closing_parenthesis + 1 :].split()
        start_time_ticks = int(stat_fields[19])
        if not command or not boot_id:
            return None
    except (IndexError, OSError, ValueError):
        return None
    return {
        "boot_id": boot_id,
        "start_time_ticks": start_time_ticks,
        "command_sha256": hashlib.sha256(command).hexdigest(),
    }


def _worker_liveness(
    payload: Mapping[str, object],
) -> tuple[bool, dict[str, object]]:
    pid = payload.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        return False, {"status": "not_alive", "reason": "invalid_pid"}
    observed = _process_identity(pid)
    if observed is None:
        return False, {
            "status": "not_alive",
            "reason": "pid_not_present_or_unreadable",
        }
    expected = payload.get("process_identity")
    if not isinstance(expected, Mapping):
        return False, {
            "status": "unverified",
            "reason": "worker_record_has_no_process_identity",
        }
    identity_fields = ("boot_id", "start_time_ticks", "command_sha256")
    if any(expected.get(name) != observed[name] for name in identity_fields):
        return False, {
            "status": "not_alive",
            "reason": "pid_identity_mismatch",
        }
    return True, {
        "status": "verified_alive",
        "reason": "boot_id_start_time_and_command_match",
    }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
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


def _sha256(path: Path) -> str:
    return hash_file(path, "sha256")


def _config_digest(config_path: Path) -> str:
    return _sha256(config_path)[:16]


def _code_receipts() -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in CODE_RECEIPT_PATHS
    ]


def _artifact_output(output_root: Path, artifact: str) -> Path:
    config_path = ARTIFACT_CONFIGS[artifact]
    return (
        output_root
        / "work"
        / "image-repro"
        / artifact
        / _config_digest(config_path)
    )


def _filter_config(
    config: ExperimentConfig,
    methods: Sequence[str] | None,
) -> ExperimentConfig:
    if not methods:
        return config
    requested = set(methods)
    selected = tuple(
        method for method in config.methods if method.name in requested
    )
    missing = sorted(requested - {method.name for method in selected})
    if missing:
        raise ValueError(f"unknown methods for {config.name}: {missing}")
    return replace(config, methods=selected)


def _load_instances(
    instance_ids: Sequence[str] | None = None,
) -> tuple[tuple[TensorInstance, ...], tuple[dict[str, object], ...]]:
    loaded = load_paper_kodak(instance_ids=instance_ids)
    instances = []
    receipts = []
    specs = {
        receipt["id"]: receipt for receipt in _manifest_receipts()
    }
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
                    "source_path": str(image.source_path),
                },
            )
        )
        spec = specs[image.image_id]
        receipts.append(
            {
                "id": image.image_id,
                "path": str(image.source_path),
                "sha256": spec["sha256"],
                "bytes": image.source_path.stat().st_size,
                "resolution_xy": [width, height],
                "color_space": image.color_space,
            }
        )
    return tuple(instances), tuple(receipts)


def _manifest_receipts() -> tuple[dict[str, object], ...]:
    manifest = load_manifest("kodak")
    raw_root = ROOT / "data/raw"
    return tuple(
        {
            "id": item["id"],
            "path": str(raw_root / item["local_path"]),
            "sha256": item["checksum"]["value"],
            "bytes": item["bytes"],
            "resolution_xy": [item["width"], item["height"]],
            "color_space": item["color_space"],
        }
        for item in manifest["images"]
    )


def _dummy_instances(
    config: ExperimentConfig | None = None,
) -> tuple[TensorInstance, ...]:
    manifest = load_manifest("kodak")
    requested = (
        None
        if config is None
        else config.runner.get("instance_ids")
    )
    requested_ids = None if requested is None else set(requested)
    return tuple(
        TensorInstance(
            item["id"],
            torch.zeros(1, 2),
            torch.zeros(1, 3),
        )
        for item in manifest["images"]
        if requested_ids is None or item["id"] in requested_ids
    )


def _safe_component(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )


def _job_paths(output_dir: Path, instance: str, method: str, seed: int):
    stem = (
        Path("raw")
        / _safe_component(instance)
        / _safe_component(method)
        / f"seed-{seed}"
    )
    return (
        output_dir / stem.with_suffix(".json"),
        output_dir / "checkpoints" / stem.with_suffix(".pt"),
    )


def _job_total_steps(config: ExperimentConfig, method) -> int:
    values = dict(config.training)
    values.update(method.training)
    return paper_recipe_from_mapping(values).total_steps


def _write_job_plan(
    *,
    artifact: str,
    config: ExperimentConfig,
    output_dir: Path,
    receipts: Sequence[Mapping[str, object]],
    world_size: int,
    full_run_authorization: Mapping[str, object] | None,
) -> None:
    jobs = enumerate_jobs(config, _dummy_instances(config))
    total_steps = sum(_job_total_steps(config, job.method) for job in jobs)
    payload = {
        "schema": "peps.image_job_plan",
        "schema_version": SCHEMA_VERSION,
        "artifact": artifact,
        "profile": config.profile,
        "canonical": config.canonical,
        "paper": config.paper,
        "created_at_utc": _utc_now(),
        "config": _config_payload(config),
        "config_sha256": _sha256(config.source),
        "code_receipts": _code_receipts(),
        "git": collect_git_state(ROOT),
        "dataset": {
            "id": "kodak-pcd0992",
            "instances": list(receipts),
            "instance_count": len(receipts),
            "original_resolution": True,
        },
        "parallelism": {
            "mode": "job_shard",
            "world_size": world_size,
            "assignment": "global_job_index_mod_world_size",
            "same_model_distributed": False,
        },
        "full_run_authorization": (
            None
            if full_run_authorization is None
            else dict(full_run_authorization)
        ),
        "expected_jobs": len(jobs),
        "expected_optimizer_steps": total_steps,
        "checkpoint_every": int(config.training.get("checkpoint_every", 0)),
        "output_dir": str(output_dir),
        "limitations": list(PROTOCOL_LIMITATIONS),
    }
    atomic_write_json(output_dir / "job-plan.json", payload)


def run_artifact(
    artifact: str,
    *,
    output_root: Path,
    rank: int,
    world_size: int,
    device: torch.device,
    instance_ids: Sequence[str] | None,
    methods: Sequence[str] | None,
    force: bool,
    allow_protocol_assumptions: bool,
    authorization_receipt: Path | None = None,
) -> dict[str, object]:
    config_path = ARTIFACT_CONFIGS[artifact]
    complete_config = load_experiment_config(config_path)
    full_run_authorization = None
    if artifact == "table1":
        authorization = validate_image_table1_authorization(
            authorization_receipt,
            config_path=config_path,
        )
        assert authorization_receipt is not None
        authorization_path = authorization_receipt.expanduser().resolve()
        full_run_authorization = {
            "schema": authorization["schema"],
            "schema_version": authorization["schema_version"],
            "approval_id": authorization["approval_id"],
            "approved_by": authorization["approved_by"],
            "issued_at_utc": authorization["issued_at_utc"],
            "expires_at_utc": authorization["expires_at_utc"],
            "boot_id": authorization["boot_id"],
            "receipt": str(authorization_path),
            "receipt_sha256": _sha256(authorization_path),
        }
    if complete_config.profile == "full":
        if not allow_protocol_assumptions:
            raise ValueError(
                "full image runs require --allow-protocol-assumptions"
            )
        if device.type != "cuda":
            raise ValueError("full image runs refuse CPU execution")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    config = _filter_config(complete_config, methods)
    configured_ids = complete_config.runner.get("instance_ids")
    effective_instance_ids = (
        instance_ids
        if instance_ids is not None
        else configured_ids
    )
    instances, selected_receipts = _load_instances(effective_instance_ids)
    planned_ids = {item.name for item in _dummy_instances(complete_config)}
    receipts = tuple(
        receipt
        for receipt in _manifest_receipts()
        if receipt["id"] in planned_ids
    )
    output_dir = _artifact_output(output_root, artifact)
    _write_job_plan(
        artifact=artifact,
        config=complete_config,
        output_dir=output_dir,
        receipts=receipts,
        world_size=world_size,
        full_run_authorization=full_run_authorization,
    )

    complete_jobs = enumerate_jobs(
        complete_config,
        _dummy_instances(complete_config),
    )
    selected_names = {instance.name for instance in instances}
    selected_methods = {method.name for method in config.methods}
    selected_jobs = [
        job
        for job in complete_jobs
        if job.instance.name in selected_names
        and job.method.name in selected_methods
        and job.index % world_size == rank
    ]
    worker_path = output_dir / f"worker-rank-{rank}.json"
    worker = {
        "schema": "peps.image_worker_status",
        "schema_version": SCHEMA_VERSION,
        "artifact": artifact,
        "rank": rank,
        "world_size": world_size,
        "pid": os.getpid(),
        "process_identity": _process_identity(os.getpid()),
        "device": str(device),
        "state": "running",
        "started_at_utc": _utc_now(),
        "selected_instances": sorted(selected_names),
        "selected_methods": sorted(selected_methods),
        "selected_dataset_receipts": list(selected_receipts),
        "expected_selected_shard_jobs": len(selected_jobs),
        "output_dir": str(output_dir),
        "full_run_authorization": full_run_authorization,
    }
    atomic_write_json(worker_path, worker)
    runner = ExperimentRunner(
        config,
        output_dir,
        rank=rank,
        world_size=world_size,
        local_rank=0,
        device=device,
        force=force,
    )
    try:
        records = runner.run(instances)
    except BaseException as exc:
        worker.update(
            {
                "state": (
                    "interrupted"
                    if isinstance(exc, KeyboardInterrupt)
                    else "failed"
                ),
                "finished_at_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        atomic_write_json(worker_path, worker)
        raise
    worker.update(
        {
            "state": "complete",
            "finished_at_utc": _utc_now(),
            "records": len(records),
        }
    )
    atomic_write_json(worker_path, worker)
    return {
        "artifact": artifact,
        "rank": rank,
        "world_size": world_size,
        "records": len(records),
        "output_dir": str(output_dir),
    }


def _checkpoint_job_identity(
    config: ExperimentConfig,
    job,
    total_steps: int,
) -> dict[str, object]:
    return {
        "experiment": config.name,
        "profile": config.profile,
        "instance": job.instance.name,
        "method": job.method.name,
        "seed": job.seed,
        "total_steps": total_steps,
        "config_sha256": _sha256(config.source),
    }


def _read_checkpoint_step(
    path: Path,
    total_steps: int,
    *,
    expected_job: Mapping[str, object],
    cosine: bool,
) -> tuple[int, str | None, str | None]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint is not a mapping")
        if int(state.get("schema_version", 0)) != 1:
            raise ValueError("unsupported training checkpoint schema")
        raw_step = state["step"]
        if isinstance(raw_step, bool) or not isinstance(raw_step, int):
            raise TypeError("checkpoint step is not an integer")
        step = raw_step
        model = state["model"]
        optimizer = state["optimizer"]
        stream = state["minibatch_stream"]
        if not isinstance(model, Mapping) or not model:
            raise ValueError("checkpoint model state is empty or invalid")
        if not isinstance(optimizer, Mapping):
            raise TypeError("checkpoint optimizer state is invalid")
        if not isinstance(stream, Mapping):
            raise TypeError("checkpoint minibatch stream is invalid")
        for name in ("size", "batch_size", "seed", "draws", "generator_state"):
            if name not in stream:
                raise ValueError(f"checkpoint minibatch stream is missing {name}")
        if int(stream["draws"]) != step:
            raise ValueError("checkpoint draw count does not match step")
        generator_state = stream["generator_state"]
        if not isinstance(generator_state, torch.Tensor):
            raise TypeError("checkpoint generator state is not a tensor")
        scheduler = state.get("scheduler")
        if cosine and not isinstance(scheduler, Mapping):
            raise ValueError("checkpoint is missing cosine scheduler state")
        checkpoint_job = state.get("job")
        warning = None
        if checkpoint_job is None:
            warning = (
                "checkpoint predates embedded job identity; path and training "
                "state are valid but identity is legacy-unverified"
            )
        elif not isinstance(checkpoint_job, Mapping):
            raise TypeError("checkpoint job identity is invalid")
        else:
            identity_fields = (
                "experiment",
                "profile",
                "instance",
                "method",
                "seed",
                "total_steps",
            )
            if any(
                checkpoint_job.get(name) != expected_job[name]
                for name in identity_fields
            ):
                raise ValueError("checkpoint belongs to a different job")
            saved_config_sha256 = checkpoint_job.get("config_sha256")
            if (
                saved_config_sha256 is not None
                and saved_config_sha256 != expected_job["config_sha256"]
            ):
                raise ValueError(
                    "checkpoint belongs to a different config revision"
                )
            if saved_config_sha256 is None:
                warning = (
                    "checkpoint job identity predates config hashing; training "
                    "state is valid but config identity is legacy-unverified"
                )
    except Exception as exc:
        return 0, f"{type(exc).__name__}: {exc}", None
    if not 0 <= step <= total_steps:
        return (
            0,
            f"checkpoint step {step} outside [0, {total_steps}]",
            None,
        )
    return step, None, warning


def _load_result_record(
    path: Path,
    *,
    config: ExperimentConfig,
    job,
    total_steps: int,
) -> tuple[dict[str, object] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("result is not an object")
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported result schema")
        if payload["experiment"] != config.name:
            raise ValueError("experiment mismatch")
        if payload["task"] != config.task:
            raise ValueError("task mismatch")
        if payload["dataset"] != config.dataset:
            raise ValueError("dataset mismatch")
        if payload["instance"] != job.instance.name:
            raise ValueError("instance mismatch")
        if payload["method"] != job.method.name:
            raise ValueError("method mismatch")
        if int(payload["seed"]) != job.seed:
            raise ValueError("seed mismatch")
        if int(payload["job_index"]) != job.index:
            raise ValueError("job-index mismatch")
        if payload["profile"] != config.profile:
            raise ValueError("profile mismatch")
        if int(payload["training"]["total_steps"]) != total_steps:
            raise ValueError("optimizer-step budget mismatch")
        if payload.get("parallelism", {}).get("mode") != "job_shard":
            raise ValueError("missing job-shard provenance")
        measured = payload["metrics"]
        expected_metrics = set(config.runner.get("metrics", ()))
        if set(measured) != expected_metrics:
            raise ValueError(
                f"metric keys {sorted(measured)} != {sorted(expected_metrics)}"
            )
        if not all(math.isfinite(float(value)) for value in measured.values()):
            raise ValueError("non-finite metric")
        parameters = payload["parameters"]
        if (
            job.method.expected_encoder_params is not None
            and int(parameters["encoder"])
            != job.method.expected_encoder_params
        ):
            raise ValueError("encoder parameter budget mismatch")
        if (
            job.method.expected_total_params is not None
            and int(parameters["total"])
            != job.method.expected_total_params
        ):
            raise ValueError("total parameter budget mismatch")
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return payload, None


def _validated_artifact_records(
    artifact: str,
    *,
    output_root: Path,
) -> list[dict[str, object]]:
    config = load_experiment_config(ARTIFACT_CONFIGS[artifact])
    output_dir = _artifact_output(output_root, artifact)
    records = []
    for job in enumerate_jobs(config, _dummy_instances(config)):
        total_steps = _job_total_steps(config, job.method)
        result_path, _ = _job_paths(
            output_dir,
            job.instance.name,
            job.method.name,
            job.seed,
        )
        if not result_path.is_file():
            continue
        payload, error = _load_result_record(
            result_path,
            config=config,
            job=job,
            total_steps=total_steps,
        )
        if error is None:
            assert payload is not None
            records.append(payload)
    return records


def _worker_statuses(output_dir: Path) -> list[dict[str, object]]:
    statuses = []
    for path in sorted(output_dir.glob("worker-rank-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            statuses.append(
                {
                    "path": str(path),
                    "state": "unreadable",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if not isinstance(payload, dict):
            statuses.append(
                {
                    "path": str(path),
                    "state": "unreadable",
                    "error": "TypeError: worker status is not an object",
                }
            )
            continue
        selected_receipts = payload.pop("selected_dataset_receipts", ())
        payload["selected_dataset_receipt_count"] = (
            len(selected_receipts)
            if isinstance(selected_receipts, (list, tuple))
            else 0
        )
        process_alive, liveness = _worker_liveness(payload)
        payload["process_alive"] = process_alive
        payload["liveness_evidence"] = liveness
        statuses.append(payload)
    return statuses


def artifact_progress(
    artifact: str,
    *,
    output_root: Path,
) -> dict[str, object]:
    config = load_experiment_config(ARTIFACT_CONFIGS[artifact])
    output_dir = _artifact_output(output_root, artifact)
    jobs = enumerate_jobs(config, _dummy_instances(config))
    complete = 0
    checkpointed = 0
    optimizer_steps = 0
    expected_steps = 0
    checkpoint_errors = []
    checkpoint_warnings = []
    result_errors = []
    progress_times: list[datetime] = []
    expected_result_paths: set[Path] = set()
    expected_checkpoint_paths: set[Path] = set()
    per_rank: dict[int, dict[str, int]] = {}
    world_size = int(config.runner.get("world_size", 4))
    for job in jobs:
        total = _job_total_steps(config, job.method)
        expected_steps += total
        rank = job.index % world_size
        shard = per_rank.setdefault(
            rank,
            {
                "expected_jobs": 0,
                "completed_jobs": 0,
                "checkpointed_incomplete_jobs": 0,
                "optimizer_steps": 0,
            },
        )
        shard["expected_jobs"] += 1
        result_path, checkpoint_path = _job_paths(
            output_dir,
            job.instance.name,
            job.method.name,
            job.seed,
        )
        expected_result_paths.add(result_path)
        expected_checkpoint_paths.add(checkpoint_path)
        if result_path.is_file():
            _, error = _load_result_record(
                result_path,
                config=config,
                job=job,
                total_steps=total,
            )
            if error is None:
                complete += 1
                optimizer_steps += total
                shard["completed_jobs"] += 1
                shard["optimizer_steps"] += total
                progress_times.append(
                    datetime.fromtimestamp(
                        result_path.stat().st_mtime,
                        timezone.utc,
                    )
                )
                continue
            result_errors.append(
                {
                    "path": str(result_path),
                    "instance": job.instance.name,
                    "method": job.method.name,
                    "seed": job.seed,
                    "error": error,
                }
            )
        if checkpoint_path.is_file():
            recipe = paper_recipe_from_mapping(
                {
                    **dict(config.training),
                    **dict(job.method.training),
                }
            )
            step, error, warning = _read_checkpoint_step(
                checkpoint_path,
                total,
                expected_job=_checkpoint_job_identity(config, job, total),
                cosine=recipe.cosine,
            )
            if error is not None:
                checkpoint_errors.append(
                    {
                        "path": str(checkpoint_path),
                        "instance": job.instance.name,
                        "method": job.method.name,
                        "seed": job.seed,
                        "error": error,
                    }
                )
                continue
            optimizer_steps += step
            shard["optimizer_steps"] += step
            progress_times.append(
                datetime.fromtimestamp(
                    checkpoint_path.stat().st_mtime,
                    timezone.utc,
                )
            )
            if step:
                checkpointed += 1
                shard["checkpointed_incomplete_jobs"] += 1
            if warning is not None:
                checkpoint_warnings.append(
                    {
                        "path": str(checkpoint_path),
                        "instance": job.instance.name,
                        "method": job.method.name,
                        "seed": job.seed,
                        "warning": warning,
                    }
                )
    workers = _worker_statuses(output_dir)
    for worker in workers:
        rank = worker.get("rank")
        evidence = per_rank.get(rank) if isinstance(rank, int) else None
        if evidence is not None:
            worker["progress_evidence"] = dict(evidence)
        recorded_state = worker.get("state")
        worker["recorded_state"] = recorded_state
        if recorded_state == "running":
            if worker.get("process_alive"):
                worker["effective_state"] = "running"
            elif evidence and (
                evidence["completed_jobs"]
                or evidence["checkpointed_incomplete_jobs"]
            ):
                worker["effective_state"] = "stopped_checkpointed"
            else:
                worker["effective_state"] = "stopped_incomplete"
        else:
            worker["effective_state"] = recorded_state
        # ``state`` in the derived status is current evidence; retain the raw
        # receipt value separately so stale "running" records stay auditable.
        worker["state"] = worker["effective_state"]
    active_workers = sum(
        bool(worker.get("process_alive"))
        and worker.get("recorded_state") == "running"
        for worker in workers
    )
    start_times = []
    for worker in workers:
        value = worker.get("started_at_utc")
        if not isinstance(value, str):
            continue
        try:
            start_times.append(datetime.fromisoformat(value))
        except ValueError:
            continue
    throughput_observation = None
    if start_times and optimizer_steps:
        observation_end = (
            datetime.now(timezone.utc)
            if active_workers
            else max(progress_times, default=min(start_times))
        )
        elapsed = (observation_end - min(start_times)).total_seconds()
        if elapsed > 0:
            rate = optimizer_steps / elapsed
            throughput_observation = {
                "launch_elapsed_seconds": elapsed,
                "observation_ended_at_utc": observation_end.isoformat(),
                "observation_end_evidence": (
                    "current_time_while_workers_verified_alive"
                    if active_workers
                    else "latest_valid_result_or_checkpoint_mtime"
                ),
                "aggregate_checkpointed_steps_per_second": rate,
                "naive_remaining_seconds_at_current_job_mix": (
                    (expected_steps - optimizer_steps) / rate
                ),
                "warning": "Not an ETA: method costs differ and progress is visible only at checkpoint boundaries.",
            }
    unexpected_result_files = sorted(
        str(path)
        for path in output_dir.glob("raw/**/*.json")
        if path not in expected_result_paths
    )
    unexpected_checkpoint_files = sorted(
        str(path)
        for path in output_dir.glob("checkpoints/raw/**/*.pt")
        if path not in expected_checkpoint_paths
    )
    incomplete_temporary_outputs = sorted(
        str(path) for path in output_dir.glob("**/*.tmp")
    )
    worker_status_errors = [
        {
            "path": worker.get("path"),
            "error": worker.get("error"),
        }
        for worker in workers
        if worker.get("state") == "unreadable"
    ]
    return {
        "artifact": artifact,
        "profile": config.profile,
        "canonical": config.canonical,
        "config": str(config.source.relative_to(ROOT)),
        "config_sha256": _sha256(config.source),
        "output_dir": str(output_dir),
        "expected_jobs": len(jobs),
        "completed_jobs": complete,
        "checkpointed_incomplete_jobs": checkpointed,
        "job_completion_fraction": complete / len(jobs),
        "expected_optimizer_steps": expected_steps,
        "accounted_optimizer_steps": optimizer_steps,
        "optimizer_step_completion_fraction": (
            optimizer_steps / expected_steps if expected_steps else 0.0
        ),
        "per_rank": {str(rank): values for rank, values in sorted(per_rank.items())},
        "workers": workers,
        "active_workers": active_workers,
        "throughput_observation": throughput_observation,
        "checkpoint_errors": checkpoint_errors,
        "checkpoint_warnings": checkpoint_warnings,
        "result_errors": result_errors,
        "worker_status_errors": worker_status_errors,
        "unexpected_result_files": unexpected_result_files,
        "unexpected_checkpoint_files": unexpected_checkpoint_files,
        "incomplete_temporary_outputs": incomplete_temporary_outputs,
        "output_integrity_ok": not any(
            (
                checkpoint_errors,
                result_errors,
                worker_status_errors,
                unexpected_result_files,
                unexpected_checkpoint_files,
                incomplete_temporary_outputs,
            )
        ),
        "complete": complete == len(jobs),
    }


def _paired_statistics(
    baseline_records: Sequence[Mapping[str, object]],
    candidate_records: Sequence[Mapping[str, object]],
    *,
    baseline: str,
    candidate: str,
    metric: str,
    expected_pairs: int,
) -> dict[str, object]:
    left = {
        (str(record["instance"]), int(record["seed"])): float(
            record["metrics"][metric]
        )
        for record in baseline_records
        if record["method"] == baseline and metric in record["metrics"]
    }
    right = {
        (str(record["instance"]), int(record["seed"])): float(
            record["metrics"][metric]
        )
        for record in candidate_records
        if record["method"] == candidate and metric in record["metrics"]
    }
    keys = sorted(set(left) & set(right))
    differences = [right[key] - left[key] for key in keys]
    if not differences:
        return {
            "baseline": baseline,
            "candidate": candidate,
            "metric": metric,
            "count": 0,
            "expected_pairs": expected_pairs,
            "complete": False,
            "mean_delta": None,
            "ci95_low": None,
            "ci95_high": None,
            "wins": 0,
            "ties": 0,
            "losses": 0,
        }
    mean = sum(differences) / len(differences)
    if len(differences) == 1:
        low = high = None
    else:
        variance = sum((value - mean) ** 2 for value in differences) / (
            len(differences) - 1
        )
        try:
            from scipy.stats import t as student_t

            critical = float(student_t.ppf(0.975, len(differences) - 1))
        except ImportError:
            critical = 1.96
        half_width = critical * math.sqrt(variance / len(differences))
        low, high = mean - half_width, mean + half_width
    return {
        "baseline": baseline,
        "candidate": candidate,
        "metric": metric,
        "count": len(differences),
        "expected_pairs": expected_pairs,
        "complete": len(differences) == expected_pairs,
        "mean_baseline": sum(left[key] for key in keys) / len(keys),
        "mean_candidate": sum(right[key] for key in keys) / len(keys),
        "mean_delta": mean,
        "ci_method": "paired Student-t over instance-seed rows",
        "ci95_low": low,
        "ci95_high": high,
        "wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
        "pairs": [
            {
                "instance": instance,
                "seed": seed,
                "baseline": left[(instance, seed)],
                "candidate": right[(instance, seed)],
                "delta": right[(instance, seed)] - left[(instance, seed)],
            }
            for instance, seed in keys
        ],
    }


def write_paired_report(
    *,
    output_root: Path,
    destination: Path,
) -> dict[str, object]:
    records = {
        artifact: _validated_artifact_records(
            artifact,
            output_root=output_root,
        )
        for artifact in ARTIFACT_CONFIGS
    }
    comparisons = [
        ("table1", "table1", "Grid", "G-PEPS", "psnr", 72),
        ("table1", "table1", "Grid", "G-P-PEPS", "psnr", 72),
        ("table1", "table1", "NTC_N", "NTC_PinkPEPS", "psnr", 72),
        ("table1", "table1", "LPE", "Grid", "psnr", 72),
        ("table1", "table1", "Grid", "G-P-PEPS-25", "psnr", 72),
        ("table1", "table1", "G-PEPS", "G-P-PEPS", "psnr", 72),
        ("table5", "table5", "Grid-L1", "G-P-PEPS-L1", "psnr", 72),
        ("table5", "table5", "NTC_N-L1", "G-P-PEPS-L1", "psnr", 72),
        ("table1", "core-ablations", "G-PEPS", "G-PEPS-no-sharing", "psnr", 72),
        ("table1", "core-ablations", "G-PEPS", "G-PEPS-no-original-point", "psnr", 72),
        ("table1", "core-ablations", "G-PEPS", "G-PEPS-full-sum", "psnr", 72),
        ("table1", "core-ablations", "G-PEPS", "G-PEPS-frequency-pair-sum", "psnr", 72),
        ("table1", "core-ablations", "G-P-PEPS", "G-Brownian-PEPS", "psnr", 72),
        ("smoke", "smoke", "Grid", "G-PEPS", "psnr", 24),
        ("smoke", "smoke", "Grid", "G-P-PEPS", "psnr", 24),
        ("smoke", "smoke", "NTC_N", "NTC_PinkPEPS", "psnr", 24),
        ("appendix-smoke", "appendix-smoke", "Grid-L1", "G-P-PEPS-L1", "psnr", 1),
        ("appendix-smoke", "appendix-smoke", "NTC_N-L1", "G-P-PEPS-L1", "psnr", 1),
        ("smoke", "appendix-smoke", "G-PEPS", "G-PEPS-no-sharing", "psnr", 1),
        ("smoke", "appendix-smoke", "G-PEPS", "G-PEPS-no-original-point", "psnr", 1),
        ("smoke", "appendix-smoke", "G-PEPS", "G-PEPS-full-sum", "psnr", 1),
        ("smoke", "appendix-smoke", "G-PEPS", "G-PEPS-frequency-pair-sum", "psnr", 1),
        ("smoke", "appendix-smoke", "G-P-PEPS", "G-Brownian-PEPS", "psnr", 1),
    ]
    rows = []
    for (
        baseline_artifact,
        candidate_artifact,
        baseline,
        candidate,
        metric,
        expected,
    ) in comparisons:
        row = _paired_statistics(
            records[baseline_artifact],
            records[candidate_artifact],
            baseline=baseline,
            candidate=candidate,
            metric=metric,
            expected_pairs=expected,
        )
        row["baseline_artifact"] = baseline_artifact
        row["candidate_artifact"] = candidate_artifact
        row["verification_status"] = (
            "smoke_not_paper_comparable"
            if baseline_artifact.endswith("smoke")
            or candidate_artifact.endswith("smoke")
            else (
                "complete_protocol_assumption"
                if row["complete"]
                else "partial_do_not_interpret"
            )
        )
        rows.append(row)
    payload = {
        "schema": "peps.image_paired_report",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "paper": "PEPS Extended arXiv:2604.24167v1",
        "pairing_unit": ["instance", "seed"],
        "multiple_comparison_adjustment": None,
        "comparisons": rows,
    }
    atomic_write_json(destination, payload)
    return payload


def write_summary_report(
    *,
    output_root: Path,
    destination: Path,
) -> list[dict[str, object]]:
    rows = []
    for artifact in ARTIFACT_CONFIGS:
        config = load_experiment_config(ARTIFACT_CONFIGS[artifact])
        records = _validated_artifact_records(
            artifact,
            output_root=output_root,
        )
        planned_instances = len(_dummy_instances(config))
        expected_by_method = {
            method.name: planned_instances * len(method.seeds or config.seeds)
            for method in config.methods
        }
        grouped: dict[tuple[str, str], list[float]] = {}
        for record in records:
            for metric, value in record["metrics"].items():
                grouped.setdefault(
                    (str(record["method"]), str(metric)),
                    [],
                ).append(float(value))
        for (method, metric), values in sorted(grouped.items()):
            count = len(values)
            expected = expected_by_method[method]
            mean = sum(values) / count
            if count > 1:
                variance = sum((value - mean) ** 2 for value in values) / (
                    count - 1
                )
                standard_deviation = math.sqrt(variance)
                try:
                    from scipy.stats import t as student_t

                    critical = float(student_t.ppf(0.975, count - 1))
                except ImportError:
                    critical = 1.96
                half_width = critical * standard_deviation / math.sqrt(count)
                ci_low = mean - half_width
                ci_high = mean + half_width
            else:
                standard_deviation = None
                ci_low = None
                ci_high = None
            paper_value = (
                PAPER_TABLE1.get(method, {}).get(metric)
                if artifact == "table1"
                else None
            )
            rows.append(
                {
                    "artifact": artifact,
                    "profile": config.profile,
                    "method": method,
                    "metric": metric,
                    "count": count,
                    "expected_count": expected,
                    "complete": count == expected,
                    "mean": mean,
                    "std": standard_deviation,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "paper_value": paper_value,
                    "delta_from_paper": (
                        None if paper_value is None else mean - paper_value
                    ),
                    "verification_status": (
                        "smoke_not_paper_comparable"
                        if config.profile == "smoke"
                        else (
                            "complete_protocol_assumption"
                            if count == expected
                            else "partial_do_not_interpret"
                        )
                    ),
                }
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "artifact",
        "profile",
        "method",
        "metric",
        "count",
        "expected_count",
        "complete",
        "mean",
        "std",
        "ci95_low",
        "ci95_high",
        "paper_value",
        "delta_from_paper",
        "verification_status",
    )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return rows


def write_instance_report(
    *,
    output_root: Path,
    destination: Path,
) -> int:
    resolution_by_instance = {
        receipt["id"]: receipt["resolution_xy"]
        for receipt in _manifest_receipts()
    }
    rows = []
    for artifact in ARTIFACT_CONFIGS:
        config = load_experiment_config(ARTIFACT_CONFIGS[artifact])
        records = _validated_artifact_records(
            artifact,
            output_root=output_root,
        )
        expected_jobs = len(enumerate_jobs(config, _dummy_instances(config)))
        artifact_complete = len(records) == expected_jobs
        for record in records:
            width, height = resolution_by_instance[str(record["instance"])]
            for metric, value in sorted(record["metrics"].items()):
                rows.append(
                    {
                        "artifact": artifact,
                        "profile": config.profile,
                        "instance": record["instance"],
                        "width": width,
                        "height": height,
                        "method": record["method"],
                        "seed": record["seed"],
                        "metric": metric,
                        "value": value,
                        "encoder_params": record["parameters"]["encoder"],
                        "decoder_params": record["parameters"]["decoder"],
                        "total_params": record["parameters"]["total"],
                        "compression_factor": record["compression_factor"],
                        "elapsed_seconds": record["elapsed_seconds"],
                        "job_shard_rank": record["rank"],
                        "job_shard_world_size": record["world_size"],
                        "git_commit": record.get("git_commit"),
                        "git_dirty": record.get("git_dirty"),
                        "config_source": record["config_source"],
                        "training_json": json.dumps(
                            record["training"],
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "metric_versions_json": json.dumps(
                            record["metric_versions"],
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "verification_status": (
                            "smoke_not_paper_comparable"
                            if config.profile == "smoke"
                            else (
                                "complete_protocol_assumption"
                                if artifact_complete
                                else "partial_do_not_interpret"
                            )
                        ),
                    }
                )
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "artifact",
        "profile",
        "instance",
        "width",
        "height",
        "method",
        "seed",
        "metric",
        "value",
        "encoder_params",
        "decoder_params",
        "total_params",
        "compression_factor",
        "elapsed_seconds",
        "job_shard_rank",
        "job_shard_world_size",
        "git_commit",
        "git_dirty",
        "config_source",
        "training_json",
        "metric_versions_json",
        "verification_status",
    )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return len(rows)


def protocol_report(output_root: Path) -> dict[str, object]:
    _, receipts = _load_instances()
    configs = {}
    for artifact, path in ARTIFACT_CONFIGS.items():
        config = load_experiment_config(path)
        jobs = enumerate_jobs(config, _dummy_instances(config))
        configs[artifact] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "profile": config.profile,
            "canonical": config.canonical,
            "methods": [
                {
                    "name": method.name,
                    "role": method.role,
                    "expected_encoder_params": method.expected_encoder_params,
                    "expected_total_params": method.expected_total_params,
                }
                for method in config.methods
            ],
            "seeds": list(config.seeds),
            "jobs": len(jobs),
            "optimizer_steps": sum(
                _job_total_steps(config, job.method) for job in jobs
            ),
            "output_dir": str(_artifact_output(output_root, artifact)),
        }
    return {
        "schema": "peps.image_reproduction_protocol",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "paper": {
            "reference": "PEPS Extended arXiv:2604.24167v1",
            "source_retrieved": "https://arxiv.org/src/2604.24167",
            "table1": PAPER_TABLE1,
            "table5": PAPER_TABLE5,
        },
        "dataset": {
            "id": "kodak-pcd0992",
            "instance_count": len(receipts),
            "all_original_resolution": True,
            "landscape_count": sum(
                receipt["resolution_xy"] == [768, 512]
                for receipt in receipts
            ),
            "portrait_count": sum(
                receipt["resolution_xy"] == [512, 768]
                for receipt in receipts
            ),
            "total_pixels": sum(
                math.prod(receipt["resolution_xy"]) for receipt in receipts
            ),
            "instances": list(receipts),
        },
        "configs": configs,
        "figures": {
            "1": {"kind": "analytic schematic", "requires_training": False},
            "2": {"kind": "analytic rotation", "requires_training": False},
            "3": {"kind": "Kodak and texture PSD", "requires_training": False, "dataset_assumption": "first ten paper-listed texture sets"},
            "4": {"kind": "Pink d=8 allocation", "requires_training": False},
            "5": {"kind": "native-4K parameter sweep", "blocked": True, "reason": "dataset identity/count and optimizer steps are not reported"},
            "6": {"kind": "Kodak qualitative and FLIP", "depends_on": "complete table1 checkpoints"},
            "7": {"kind": "24-image dual scatter", "depends_on": "complete table1 raw metrics"},
            "10": {"kind": "analytic Lissajous curves", "requires_training": False},
            "11": {"kind": "analytic frequency-parameterized curves", "requires_training": False, "paper_ambiguity": "the source does not define the third plotted coordinate"},
        },
        "limitations": list(PROTOCOL_LIMITATIONS),
        "code_receipts": _code_receipts(),
        "git": collect_git_state(ROOT),
        "environment": collect_environment(),
        "metric_versions": metric_versions(),
    }


def write_status(
    *,
    output_root: Path,
    destination: Path,
    paired_destination: Path,
) -> dict[str, object]:
    protocol_path = destination.with_name("image_repro_protocol.json")
    atomic_write_json(protocol_path, protocol_report(output_root))
    progress = {
        artifact: artifact_progress(artifact, output_root=output_root)
        for artifact in ARTIFACT_CONFIGS
    }
    paired = write_paired_report(
        output_root=output_root,
        destination=paired_destination,
    )
    summary_destination = destination.with_name("image_repro_summary.csv")
    summary_rows = write_summary_report(
        output_root=output_root,
        destination=summary_destination,
    )
    instance_destination = destination.with_name("image_repro_instances.csv")
    instance_rows = write_instance_report(
        output_root=output_root,
        destination=instance_destination,
    )
    payload = {
        "schema": "peps.image_reproduction_status",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "paper": "PEPS Extended arXiv:2604.24167v1",
        "dataset": {
            "verified_original_kodak_images": 24,
            "resolution_counts": {"768x512": 18, "512x768": 6},
        },
        "artifacts": progress,
        "paired_report": str(paired_destination),
        "paired_complete_comparisons": sum(
            comparison["complete"]
            for comparison in paired["comparisons"]
        ),
        "summary": str(summary_destination),
        "summary_rows": len(summary_rows),
        "instances": str(instance_destination),
        "instance_metric_rows": instance_rows,
        "protocol": str(protocol_path),
        "limitations": list(PROTOCOL_LIMITATIONS),
    }
    atomic_write_json(destination, payload)
    return payload


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    protocol = subparsers.add_parser("protocol")
    protocol.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "image_repro_protocol.json",
    )
    protocol.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    run = subparsers.add_parser("run")
    run.add_argument("--artifact", choices=ARTIFACT_CONFIGS, required=True)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--rank", type=int, default=0)
    run.add_argument("--world-size", type=int, default=1)
    run.add_argument("--device", default="auto")
    run.add_argument("--instance", action="append")
    run.add_argument("--method", action="append")
    run.add_argument("--force", action="store_true")
    run.add_argument("--allow-protocol-assumptions", action="store_true")
    run.add_argument("--authorization-receipt", type=Path)

    authorization = subparsers.add_parser("authorization-check")
    authorization.add_argument(
        "--artifact",
        choices=("table1",),
        default="table1",
    )
    authorization.add_argument(
        "--authorization-receipt",
        type=Path,
        required=True,
    )

    status = subparsers.add_parser("status")
    status.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    status.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "image_repro_status.json",
    )
    status.add_argument(
        "--paired-output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "image_repro_paired.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "authorization-check":
        receipt = validate_image_table1_authorization(
            arguments.authorization_receipt,
            config_path=ARTIFACT_CONFIGS[arguments.artifact],
        )
        print(
            json.dumps(
                {
                    "valid": True,
                    "artifact": arguments.artifact,
                    "approval_id": receipt["approval_id"],
                    "approved_by": receipt["approved_by"],
                    "expires_at_utc": receipt["expires_at_utc"],
                    "boot_id": receipt["boot_id"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "protocol":
        payload = protocol_report(arguments.output_root)
        atomic_write_json(arguments.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if arguments.command == "status":
        payload = write_status(
            output_root=arguments.output_root,
            destination=arguments.output,
            paired_destination=arguments.paired_output,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    payload = run_artifact(
        arguments.artifact,
        output_root=arguments.output_root,
        rank=arguments.rank,
        world_size=arguments.world_size,
        device=_device(arguments.device),
        instance_ids=arguments.instance,
        methods=arguments.method,
        force=arguments.force,
        allow_protocol_assumptions=arguments.allow_protocol_assumptions,
        authorization_receipt=arguments.authorization_receipt,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
