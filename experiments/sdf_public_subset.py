"""Manifest-backed execution for the SDF ``3-of-4 public subset``.

This launcher is intentionally narrower than the general SDF reproduction
entry point.  It runs the frozen MAPE and L1 public-shape configs on physical
GPUs 2 and 3, indexes every checkpoint, and never loads the Table 4 config or
attempts to access Pitted Stonefish.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch

from data.manifest import hash_file
from experiments.runner import atomic_write_json
from experiments.sdf_repro import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_WORK_ROOT,
    PAPER,
    PUBLIC_ASSETS,
    PUBLIC_SUBSET_LABEL,
    STONEFISH_ASSET,
    _job_paths,
    aggregate_config,
    assert_config_parameter_budgets,
    enumerate_sdf_jobs,
    estimate_cost,
    load_sdf_repro_config,
    run_shard,
    validate_public_volumes,
)
from peps.report import collect_git_state


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = (
    ROOT / "configs/paper/sdf/table3_mape.toml",
    ROOT / "configs/paper/sdf/table6_l1.toml",
)
PHYSICAL_GPU_IDS = (2, 3)
WORLD_SIZE = len(PHYSICAL_GPU_IDS)
RECEIPT_SCHEMA = ROOT / "results/schemas/sdf_public_subset_receipt.schema.json"
DEFAULT_RECEIPT = DEFAULT_OUTPUT_ROOT / "public_subset_receipt.json"
DEFAULT_BLOCKED_RECEIPT = DEFAULT_OUTPUT_ROOT / "public_subset_blocked.json"
DEFAULT_ROWS = DEFAULT_OUTPUT_ROOT / "public_subset_per_instance_method_loss.csv"
CODE_PATHS = (
    ROOT / "experiments/sdf_public_subset.py",
    ROOT / "experiments/sdf_repro.py",
    ROOT / "apps/sdf/build.py",
    ROOT / "apps/sdf/data.py",
    ROOT / "apps/sdf/render.py",
    ROOT / "peps/__init__.py",
    ROOT / "peps/aggregate.py",
    ROOT / "peps/projector.py",
    ROOT / "peps/wrapper.py",
    ROOT / "peps/train.py",
    ROOT / "peps/metrics.py",
    ROOT / "peps/models/mlp.py",
    ROOT / "peps/encoders/grid.py",
    ROOT / "peps/encoders/lpe.py",
    ROOT / "peps/encoders/multires.py",
    ROOT / "peps/encoders/positional.py",
)
CSV_FIELDS = (
    "scope",
    "paper_table",
    "artifact",
    "loss",
    "instance",
    "method",
    "method_key",
    "seed",
    "iou",
    "paper_iou",
    "delta_iou",
    "l1",
    "mape",
    "flip",
    "encoder_params",
    "decoder_params",
    "total_params",
    "evaluated_voxels",
    "training_seconds",
    "streamed_evaluation_seconds",
    "render_and_flip_seconds",
    "total_gpu_occupied_seconds",
    "physical_gpu_id",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _plain(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _configs():
    configs = tuple(load_sdf_repro_config(path) for path in CONFIG_PATHS)
    _check(len(configs) == 2, "public subset requires exactly MAPE and L1 configs")
    _check(
        {str(config.training["loss"]) for config in configs} == {"mape", "l1"},
        "public subset must contain exactly MAPE and L1",
    )
    for config in configs:
        _check(config.scope == PUBLIC_SUBSET_LABEL, "public subset label drifted")
        _check(config.assets == PUBLIC_ASSETS, "public asset list drifted")
        _check(config.status == "runnable", "public config is not runnable")
        _check(
            int(config.sharding["world_size"]) == WORLD_SIZE,
            "public config must use two worker shards",
        )
    return configs


def _code_receipts() -> list[dict[str, object]]:
    return [
        {
            "path": _portable(path),
            "bytes": path.stat().st_size,
            "sha256": hash_file(path),
        }
        for path in CODE_PATHS
    ]


def _run_identity(
    configs,
    volume_validation: Mapping[str, Any],
    code_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "scope": PUBLIC_SUBSET_LABEL,
        "configs": [
            {
                "artifact": config.artifact,
                "source": _portable(config.source),
                "sha256": config.digest,
            }
            for config in configs
        ],
        "volumes": [
            {
                "asset_id": row["asset_id"],
                "volume_sha256": row["volume_sha256"],
                "tracked_provenance_sha256": row[
                    "tracked_provenance_sha256"
                ],
            }
            for row in volume_validation["volumes"]
        ],
        "code_receipts": list(code_receipts),
        "parallelism": {
            "mode": "independent_job_modulo",
            "world_size": WORLD_SIZE,
            "physical_gpu_ids": list(PHYSICAL_GPU_IDS),
            "same_model_distributed": False,
        },
    }


def _rocm_snapshot() -> dict[str, dict[str, int]]:
    completed = subprocess.run(
        ["rocm-smi", "--showmeminfo", "vram", "--showuse", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "rocm-smi preflight failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("rocm-smi did not return JSON") from exc
    result: dict[str, dict[str, int]] = {}
    for physical_id in PHYSICAL_GPU_IDS:
        key = f"card{physical_id}"
        row = payload.get(key)
        if not isinstance(row, Mapping):
            raise RuntimeError(f"rocm-smi omitted physical GPU {physical_id}")
        result[str(physical_id)] = {
            "gpu_use_percent": int(row["GPU use (%)"]),
            "vram_total_bytes": int(row["VRAM Total Memory (B)"]),
            "vram_used_bytes": int(row["VRAM Total Used Memory (B)"]),
        }
    return result


def _require_target_gpus_idle(snapshot: Mapping[str, Mapping[str, int]]) -> None:
    baseline_limit = 512 * 1024 * 1024
    busy = {
        gpu_id: dict(row)
        for gpu_id, row in snapshot.items()
        if int(row["gpu_use_percent"]) != 0
        or int(row["vram_used_bytes"]) > baseline_limit
    }
    if busy:
        raise RuntimeError(
            "physical GPUs 2 and 3 are not idle enough for isolated execution: "
            + json.dumps(busy, sort_keys=True)
        )


def prepare_run_manifest(
    *,
    work_root: Path = DEFAULT_WORK_ROOT,
    processed_root: Path | None = None,
    require_idle: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Validate exact inputs and create/reuse a deterministic run manifest."""

    configs = _configs()
    volume_validation = validate_public_volumes(
        processed_root=processed_root,
        verify_checksums=True,
    )
    _check(volume_validation["status"] == "passed", "volume validation failed")
    _check(
        [row["asset_id"] for row in volume_validation["volumes"]]
        == list(PUBLIC_ASSETS),
        "volume validation did not cover the frozen public assets",
    )
    parameter_assertions = {
        config.artifact: list(assert_config_parameter_budgets(config))
        for config in configs
    }
    code_receipts = _code_receipts()
    identity = _run_identity(configs, volume_validation, code_receipts)
    run_id = _canonical_sha256(identity)[:24]
    run_dir = work_root.resolve() / run_id
    manifest_path = run_dir / "run-manifest.json"
    snapshot = _rocm_snapshot()
    if require_idle:
        _require_target_gpus_idle(snapshot)
    jobs = enumerate_sdf_jobs(configs)
    cost = estimate_cost(configs)
    payload: dict[str, Any] = {
        "schema": "peps.sdf_public_subset_run_manifest",
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": _now(),
        "scope": PUBLIC_SUBSET_LABEL,
        "paper": PAPER,
        "claim": {
            "full_table3": False,
            "paper_global_comparable": False,
            "canonical_four_shape": False,
        },
        "run_identity": identity,
        "configs": [
            {
                "artifact": config.artifact,
                "paper_table": config.paper_table,
                "loss": config.training["loss"],
                "source": _portable(config.source),
                "sha256": config.digest,
                "total_steps_per_job": config.total_steps,
                "batch_size": config.training["batch_size"],
                "evaluation_resolution": config.evaluation["resolution"],
                "methods": [method.name for method in config.methods],
            }
            for config in configs
        ],
        "jobs": [
            {
                **job.identity,
                "job_index": job.index,
                "assigned_rank": job.index % WORLD_SIZE,
                "physical_gpu_id": PHYSICAL_GPU_IDS[job.index % WORLD_SIZE],
            }
            for job in jobs
        ],
        "coverage": {
            "expected_jobs": len(jobs),
            "expected_optimizer_steps": cost["optimizer_steps"],
            "expected_sampled_training_points": cost[
                "sampled_training_points"
            ],
            "expected_streamed_evaluation_queries": cost[
                "full_volume_queries"
            ],
        },
        "parallelism": {
            "mode": "independent_job_modulo",
            "world_size": WORLD_SIZE,
            "physical_gpu_ids": list(PHYSICAL_GPU_IDS),
            "visibility_contract": {
                "ROCR_VISIBLE_DEVICES": "<physical_gpu_id>",
                "HIP_VISIBLE_DEVICES": "0",
                "CUDA_VISIBLE_DEVICES": "0",
                "worker_device": "cuda:0",
            },
            "same_model_distributed": False,
        },
        "target_gpu_preflight": snapshot,
        "volume_validation": volume_validation,
        "parameter_assertions": parameter_assertions,
        "evaluation_memory_contract": {
            "streamed_512_cubed": True,
            "full_coordinate_grid_materialized": False,
            "chunk_size": 262144,
            "volume_payload_bytes_per_worker": 512**3 * 4,
        },
        "stonefish": {
            "asset_id": STONEFISH_ASSET,
            "status": "deferred_auth_required",
            "authorization_checked": False,
            "data_access_attempted": False,
            "substitution_allowed": False,
            "substitution_used": False,
        },
        "table4": {
            "executed": False,
            "consolidated": False,
        },
        "git": collect_git_state(ROOT),
        "output_dir": _portable(run_dir),
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        _check(existing.get("run_id") == run_id, "existing run ID drifted")
        _check(
            existing.get("run_identity") == identity,
            "existing run manifest identity drifted",
        )
        return existing, run_dir
    atomic_write_json(manifest_path, payload)
    return payload, run_dir


def _load_run_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _check(
        payload.get("schema") == "peps.sdf_public_subset_run_manifest",
        "bad run manifest schema",
    )
    _check(payload.get("schema_version") == 1, "bad run manifest version")
    _check(payload.get("scope") == PUBLIC_SUBSET_LABEL, "bad subset label")
    _check(
        payload.get("parallelism", {}).get("physical_gpu_ids")
        == list(PHYSICAL_GPU_IDS),
        "run manifest GPU assignment drifted",
    )
    _check(
        payload.get("stonefish", {}).get("status")
        == "deferred_auth_required"
        and payload["stonefish"].get("substitution_used") is False,
        "Stonefish blocker drifted",
    )
    return payload


def _worker_hardware(physical_gpu_id: int) -> dict[str, object]:
    _check(
        os.environ.get("ROCR_VISIBLE_DEVICES") == str(physical_gpu_id),
        "ROCR_VISIBLE_DEVICES must name the assigned physical GPU",
    )
    _check(
        os.environ.get("HIP_VISIBLE_DEVICES") == "0"
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        "HIP/CUDA aliases must expose only logical device zero",
    )
    _check(torch.cuda.is_available(), "ROCm device is unavailable")
    _check(torch.cuda.device_count() == 1, "worker must see exactly one GPU")
    properties = torch.cuda.get_device_properties(0)
    architecture = str(getattr(properties, "gcnArchName", "unknown"))
    _check(architecture == "gfx1201", f"unexpected GPU architecture {architecture}")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return {
        "physical_gpu_id": physical_gpu_id,
        "logical_torch_index": 0,
        "name": str(properties.name),
        "architecture": architecture,
        "total_memory_bytes": int(total_bytes),
        "free_memory_bytes_at_start": int(free_bytes),
        "torch_version": torch.__version__,
        "rocm_version": torch.version.hip,
        "visibility": {
            "ROCR_VISIBLE_DEVICES": os.environ["ROCR_VISIBLE_DEVICES"],
            "HIP_VISIBLE_DEVICES": os.environ["HIP_VISIBLE_DEVICES"],
            "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        },
    }


def run_worker(
    *,
    run_manifest_path: Path,
    rank: int,
    physical_gpu_id: int,
    processed_root: Path | None = None,
) -> int:
    manifest = _load_run_manifest(run_manifest_path)
    _check(0 <= rank < WORLD_SIZE, "rank is outside the two-worker run")
    _check(
        PHYSICAL_GPU_IDS[rank] == physical_gpu_id,
        "rank/physical GPU assignment mismatch",
    )
    hardware = _worker_hardware(physical_gpu_id)
    run_dir = run_manifest_path.parent
    status_path = run_dir / f"worker-rank-{rank}.json"
    status: dict[str, Any] = {
        "schema": "peps.sdf_public_subset_worker",
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "rank": rank,
        "world_size": WORLD_SIZE,
        "physical_gpu_id": physical_gpu_id,
        "pid": os.getpid(),
        "state": "running",
        "started_at_utc": _now(),
        "hardware": hardware,
        "expected_job_indices": [
            row["job_index"]
            for row in manifest["jobs"]
            if row["assigned_rank"] == rank
        ],
    }
    atomic_write_json(status_path, status)
    try:
        summary = run_shard(
            _configs(),
            rank=rank,
            world_size=WORLD_SIZE,
            device=torch.device("cuda:0"),
            work_root=run_dir,
            render_root=DEFAULT_OUTPUT_ROOT / "renders",
            processed_root=processed_root,
            verify_checksums=False,
            run_id=str(manifest["run_id"]),
            physical_gpu_id=physical_gpu_id,
        )
    except BaseException as exc:
        status.update(
            {
                "state": "interrupted"
                if isinstance(exc, KeyboardInterrupt)
                else "failed",
                "finished_at_utc": _now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        atomic_write_json(status_path, status)
        raise
    status.update(
        {
            "state": "complete",
            "finished_at_utc": _now(),
            "summary": summary,
        }
    )
    atomic_write_json(status_path, status)
    return 0


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _collect_evidence(
    manifest: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    configs = _configs()
    jobs = enumerate_sdf_jobs(configs)
    job_lookup = {job.index: job for job in jobs}
    manifest_jobs = {int(row["job_index"]): row for row in manifest["jobs"]}
    _check(set(job_lookup) == set(manifest_jobs), "manifest job set drifted")
    rows: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    completed_steps = 0
    for index, job in job_lookup.items():
        declared = manifest_jobs[index]
        _check(
            {key: declared[key] for key in job.identity} == job.identity,
            f"manifest identity drift for job {index}",
        )
        result_path, checkpoint_path = _job_paths(job, run_dir)
        checkpoint_step = 0
        if checkpoint_path.is_file():
            try:
                state = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                _check(
                    state.get("schema") == "peps.sdf_repro_checkpoint"
                    and state.get("schema_version") == 1,
                    "bad checkpoint schema",
                )
                _check(
                    state.get("run_id") == manifest["run_id"]
                    and state.get("job") == job.identity,
                    "checkpoint identity mismatch",
                )
                checkpoint_step = int(state["step"])
                _check(
                    0 <= checkpoint_step <= job.config.total_steps,
                    "checkpoint step outside config",
                )
                checkpoints.append(
                    {
                        "job_index": index,
                        **job.identity,
                        "step": checkpoint_step,
                        "total_steps": job.config.total_steps,
                        "path": _portable(checkpoint_path),
                        "bytes": checkpoint_path.stat().st_size,
                        "sha256": hash_file(checkpoint_path),
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "job_index": index,
                        "path": _portable(checkpoint_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        completed_steps += checkpoint_step
        if not result_path.is_file():
            continue
        try:
            record = json.loads(result_path.read_text(encoding="utf-8"))
            _check(record.get("status") == "complete", "result is incomplete")
            _check(record.get("run_id") == manifest["run_id"], "run ID mismatch")
            _check(record.get("scope") == PUBLIC_SUBSET_LABEL, "scope drifted")
            _check(record.get("instance") in PUBLIC_ASSETS, "non-public asset")
            _check(
                record.get("canonical_four_shape") is False
                and record.get("paper_global_comparable") is False,
                "result overclaims full-table comparability",
            )
            metrics = record["evaluation"]["metrics"]
            _check(
                int(metrics["evaluated_voxels"]) == 512**3,
                "evaluation did not stream all 512^3 voxels",
            )
            _check(
                record["evaluation"]["chunked_full_volume"] is True
                and record["evaluation"]["full_coordinate_grid_materialized"]
                is False,
                "evaluation streaming contract drifted",
            )
            for metric in ("iou", "l1", "mape"):
                _check(
                    math.isfinite(float(metrics[metric])),
                    f"non-finite {metric}",
                )
            paper_iou = job.method.paper_iou[job.asset]
            runtime = record["runtime"]
            rows.append(
                {
                    "scope": PUBLIC_SUBSET_LABEL,
                    "paper_table": record["paper_table"],
                    "artifact": record["artifact"],
                    "loss": record["training"]["loss"],
                    "instance": record["instance"],
                    "method": record["method"],
                    "method_key": record["method_key"],
                    "seed": record["seed"],
                    "iou": metrics["iou"],
                    "paper_iou": paper_iou,
                    "delta_iou": float(metrics["iou"]) - float(paper_iou),
                    "l1": metrics["l1"],
                    "mape": metrics["mape"],
                    "flip": metrics.get("flip", ""),
                    "encoder_params": record["parameters"]["encoder"],
                    "decoder_params": record["parameters"]["decoder"],
                    "total_params": record["parameters"]["total"],
                    "evaluated_voxels": metrics["evaluated_voxels"],
                    "training_seconds": runtime["training_seconds"],
                    "streamed_evaluation_seconds": runtime[
                        "streamed_evaluation_seconds"
                    ],
                    "render_and_flip_seconds": runtime[
                        "render_and_flip_seconds"
                    ],
                    "total_gpu_occupied_seconds": runtime[
                        "total_gpu_occupied_seconds"
                    ],
                    "physical_gpu_id": record["parallelism"][
                        "physical_gpu_id"
                    ],
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "job_index": index,
                    "path": _portable(result_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    rows.sort(key=lambda row: (str(row["loss"]), str(row["instance"]), str(row["method"])))
    checkpoints.sort(key=lambda row: int(row["job_index"]))
    return {
        "rows": rows,
        "checkpoints": checkpoints,
        "errors": errors,
        "completed_optimizer_steps": completed_steps,
        "expected_jobs": len(jobs),
        "expected_optimizer_steps": sum(job.config.total_steps for job in jobs),
    }


def _worker_statuses(run_dir: Path) -> list[dict[str, Any]]:
    statuses = []
    for rank in range(WORLD_SIZE):
        path = run_dir / f"worker-rank-{rank}.json"
        if path.is_file():
            statuses.append(json.loads(path.read_text(encoding="utf-8")))
    return statuses


def _gpu_hours(
    rows: Sequence[Mapping[str, object]],
    workers: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    by_physical: dict[str, float] = {}
    for row in rows:
        key = str(row["physical_gpu_id"])
        by_physical[key] = by_physical.get(key, 0.0) + float(
            row["total_gpu_occupied_seconds"]
        )
    starts = [
        datetime.fromisoformat(str(worker["started_at_utc"]))
        for worker in workers
        if worker.get("started_at_utc")
    ]
    finishes = [
        datetime.fromisoformat(str(worker["finished_at_utc"]))
        for worker in workers
        if worker.get("finished_at_utc")
    ]
    launch_wall = (
        (max(finishes) - min(starts)).total_seconds()
        if starts and len(finishes) == WORLD_SIZE
        else None
    )
    cumulative_seconds = sum(by_physical.values())
    return {
        "cumulative_job_gpu_seconds": cumulative_seconds,
        "cumulative_job_gpu_hours": cumulative_seconds / 3600.0,
        "job_gpu_hours_by_physical_gpu": {
            key: value / 3600.0 for key, value in sorted(by_physical.items())
        },
        "launch_wall_seconds": launch_wall,
        "allocated_two_gpu_hours": (
            None if launch_wall is None else launch_wall * WORLD_SIZE / 3600.0
        ),
    }


def build_receipt(
    *,
    run_manifest_path: Path,
    output_path: Path = DEFAULT_RECEIPT,
    rows_path: Path = DEFAULT_ROWS,
    blocked: bool = False,
    blocker: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    manifest = _load_run_manifest(run_manifest_path)
    run_dir = run_manifest_path.parent
    evidence = _collect_evidence(manifest, run_dir)
    workers = _worker_statuses(run_dir)
    rows = evidence["rows"]
    checkpoints = evidence["checkpoints"]
    complete = (
        not evidence["errors"]
        and len(rows) == evidence["expected_jobs"]
        and len(checkpoints) == evidence["expected_jobs"]
        and all(
            int(row["step"]) == int(row["total_steps"])
            for row in checkpoints
        )
    )
    if not blocked and not complete:
        raise RuntimeError("cannot issue a complete receipt for partial evidence")
    if complete:
        _atomic_csv(rows_path, rows)
    aggregates: list[dict[str, object]] = []
    if complete and not blocked:
        for config in _configs():
            aggregate = aggregate_config(
                config,
                work_root=run_dir,
                output_root=DEFAULT_OUTPUT_ROOT,
            )
            path = DEFAULT_OUTPUT_ROOT / config.artifact / "manifest.json"
            aggregates.append(
                {
                    "artifact": config.artifact,
                    "path": _portable(path),
                    "sha256": hash_file(path),
                    "scope": aggregate["scope"],
                    "aggregate_label": aggregate["aggregate_label"],
                }
            )
    status = "blocked" if blocked else "complete"
    payload: dict[str, Any] = {
        "schema": "peps.sdf_public_subset_receipt",
        "schema_version": 1,
        "generated_at_utc": _now(),
        "status": status,
        "scope": PUBLIC_SUBSET_LABEL,
        "paper": PAPER,
        "run_id": manifest["run_id"],
        "run_manifest": _portable(run_manifest_path),
        "claims": {
            "full_table3": False,
            "paper_global_comparable": False,
            "canonical_four_shape": False,
            "public_subset_complete": bool(complete and not blocked),
        },
        "coverage": {
            "expected_jobs": evidence["expected_jobs"],
            "completed_jobs": len(rows),
            "expected_optimizer_steps": evidence["expected_optimizer_steps"],
            "checkpointed_optimizer_steps": evidence[
                "completed_optimizer_steps"
            ],
            "complete": bool(complete and not blocked),
            "losses": ["mape", "l1"],
            "assets": list(PUBLIC_ASSETS),
        },
        "results": {
            "path": _portable(rows_path) if complete else None,
            "sha256": hash_file(rows_path) if complete else None,
            "rows": len(rows),
            "per_instance_method_loss": rows,
            "aggregate_manifests": aggregates,
        },
        "gpu_runtime": _gpu_hours(rows, workers),
        "workers": workers,
        "checkpoints": checkpoints,
        "validation": {
            "valid": not evidence["errors"],
            "errors": evidence["errors"],
            "volume_checksums_verified": manifest["volume_validation"][
                "checksums_verified"
            ],
            "provenance_validated_assets": [
                row["asset_id"]
                for row in manifest["volume_validation"]["volumes"]
            ],
            "parameter_assertions_passed": all(
                row["assertion"] == "passed"
                for rows_for_config in manifest["parameter_assertions"].values()
                for row in rows_for_config
            ),
            "all_complete_results_streamed_512_cubed": all(
                int(row["evaluated_voxels"]) == 512**3 for row in rows
            ),
            "schema": _portable(RECEIPT_SCHEMA),
        },
        "stonefish": {
            "asset_id": STONEFISH_ASSET,
            "status": "deferred_auth_required",
            "authorization_checked": False,
            "data_access_attempted": False,
            "substitution_allowed": False,
            "substitution_used": False,
            "numeric_results_generated": False,
        },
        "table4": {
            "executed": False,
            "consolidated": False,
        },
        "limitations": [
            {
                "code": "public_subset_only",
                "detail": (
                    "Lucy, Thai Statue, and Armadillo are 3 of the 4 paper "
                    "shapes; this is never a full Table 3 or Global result."
                ),
            },
            {
                "code": "stonefish_authorization_blocked",
                "detail": (
                    "Canonical Pitted Stonefish remains authorization-blocked; "
                    "no substitute or numeric result was used."
                ),
            },
            {
                "code": "unreleased_sdf_converter",
                "detail": (
                    "The provenance-validated volumes use the documented "
                    "Open3D reproduction protocol, not the authors' unreleased "
                    "bit-exact C++/HIP converter."
                ),
            },
            {
                "code": "optimizer_seed_mape_zero_assumptions",
                "detail": (
                    "Adam, seed 0, and MAPE epsilon 1e-6 are explicit frozen "
                    "reproduction assumptions where the paper is silent."
                ),
            },
            {
                "code": "l1_appendix_method_scope",
                "detail": (
                    "The L1 config follows the published nine-method appendix "
                    "subset and does not invent a Hash-PEPS row."
                ),
            },
            {
                "code": "armadillo_render_protocol_assumption",
                "detail": (
                    "L1 Armadillo FLIP renders use the checked-in fixed camera "
                    "because the paper camera is unavailable."
                ),
            },
        ],
        "blocked": None,
    }
    if blocked:
        payload["blocked"] = {
            "reason": _plain(blocker or {"error": "execution did not complete"}),
            "exact_settings_preserved": True,
            "settings_reduced": False,
            "completed_checkpoints_preserved": True,
        }
    atomic_write_json(output_path, payload)
    validate_receipt(output_path, require_work=True)
    return payload


def validate_receipt(
    receipt_path: Path = DEFAULT_RECEIPT,
    *,
    require_work: bool = True,
) -> dict[str, object]:
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    _check(payload.get("schema") == "peps.sdf_public_subset_receipt", "bad schema")
    _check(payload.get("schema_version") == 1, "bad schema version")
    _check(payload.get("scope") == PUBLIC_SUBSET_LABEL, "bad subset label")
    _check(payload.get("status") in {"complete", "blocked"}, "bad status")
    claims = payload.get("claims", {})
    _check(
        claims.get("full_table3") is False
        and claims.get("paper_global_comparable") is False
        and claims.get("canonical_four_shape") is False,
        "receipt overclaims full Table 3",
    )
    stonefish = payload.get("stonefish", {})
    _check(
        stonefish.get("status") == "deferred_auth_required"
        and stonefish.get("authorization_checked") is False
        and stonefish.get("data_access_attempted") is False
        and stonefish.get("substitution_used") is False
        and stonefish.get("numeric_results_generated") is False,
        "Stonefish blocker contract drifted",
    )
    _check(
        payload.get("table4") == {"executed": False, "consolidated": False},
        "Table 4 must remain untouched",
    )
    if payload["status"] == "complete":
        _check(payload["coverage"]["complete"] is True, "coverage incomplete")
        _check(payload["coverage"]["completed_jobs"] == 57, "wrong job count")
        _check(payload["results"]["rows"] == 57, "wrong result row count")
        rows_path = ROOT / str(payload["results"]["path"])
        _check(hash_file(rows_path) == payload["results"]["sha256"], "CSV drift")
        with rows_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        _check(len(rows) == 57, "CSV row count drift")
        _check(
            {row["scope"] for row in rows} == {PUBLIC_SUBSET_LABEL},
            "CSV scope drift",
        )
        _check(
            {row["loss"] for row in rows} == {"mape", "l1"},
            "CSV loss coverage drift",
        )
    else:
        blocked = payload.get("blocked")
        _check(isinstance(blocked, Mapping), "blocked receipt lacks reason")
        _check(
            blocked.get("exact_settings_preserved") is True
            and blocked.get("settings_reduced") is False
            and blocked.get("completed_checkpoints_preserved") is True,
            "blocked receipt reduced or discarded exact work",
        )
    if require_work:
        manifest_path = ROOT / str(payload["run_manifest"])
        manifest = _load_run_manifest(manifest_path)
        _check(manifest["run_id"] == payload["run_id"], "run manifest drift")
        evidence = _collect_evidence(manifest, manifest_path.parent)
        _check(not evidence["errors"], "work evidence failed validation")
        _check(
            len(evidence["rows"]) == payload["coverage"]["completed_jobs"],
            "work/result coverage drift",
        )
    try:
        import jsonschema
    except ImportError:
        schema_validated = False
    else:
        schema = json.loads(RECEIPT_SCHEMA.read_text(encoding="utf-8"))
        jsonschema.validate(payload, schema)
        schema_validated = True
    return {
        "valid": True,
        "status": payload["status"],
        "scope": PUBLIC_SUBSET_LABEL,
        "receipt": _portable(receipt_path),
        "work_validated": require_work,
        "json_schema_validated": schema_validated,
        "completed_jobs": payload["coverage"]["completed_jobs"],
    }


def _progress(run_dir: Path) -> tuple[int, int]:
    complete = 0
    checkpointed = 0
    for path in run_dir.glob("**/raw/**/*.json"):
        try:
            complete += (
                json.loads(path.read_text(encoding="utf-8")).get("status")
                == "complete"
            )
        except (OSError, json.JSONDecodeError):
            continue
    checkpointed = sum(1 for _ in run_dir.glob("**/checkpoints/**/*.pt"))
    return complete, checkpointed


def launch(
    *,
    work_root: Path = DEFAULT_WORK_ROOT,
    processed_root: Path | None = None,
    receipt_path: Path = DEFAULT_RECEIPT,
    blocked_receipt_path: Path = DEFAULT_BLOCKED_RECEIPT,
) -> int:
    processes: list[subprocess.Popen] = []
    handles = []
    manifest: dict[str, Any] | None = None
    run_dir: Path | None = None
    failure: dict[str, object] | None = None
    try:
        manifest, run_dir = prepare_run_manifest(
            work_root=work_root,
            processed_root=processed_root,
            require_idle=True,
        )
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        for rank, physical_gpu_id in enumerate(PHYSICAL_GPU_IDS):
            log_path = logs_dir / f"rank-{rank}-physical-{physical_gpu_id}.log"
            handle = log_path.open("a", encoding="utf-8")
            handles.append(handle)
            environment = os.environ.copy()
            environment.update(
                {
                    "ROCR_VISIBLE_DEVICES": str(physical_gpu_id),
                    "HIP_VISIBLE_DEVICES": "0",
                    "CUDA_VISIBLE_DEVICES": "0",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            command = [
                sys.executable,
                "-m",
                "experiments.sdf_public_subset",
                "worker",
                "--run-manifest",
                str(run_dir / "run-manifest.json"),
                "--rank",
                str(rank),
                "--physical-gpu-id",
                str(physical_gpu_id),
            ]
            if processed_root is not None:
                command.extend(["--processed-root", str(processed_root)])
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            )
        next_update = 0.0
        while any(process.poll() is None for process in processes):
            now = time.time()
            return_codes = [process.poll() for process in processes]
            failed = [
                {"rank": rank, "return_code": code}
                for rank, code in enumerate(return_codes)
                if code not in (None, 0)
            ]
            if failed:
                failure = {
                    "type": "worker_failure",
                    "workers": failed,
                    "observed_at_utc": _now(),
                }
                for process in processes:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                break
            if now >= next_update:
                complete, checkpointed = _progress(run_dir)
                print(
                    f"SDF {PUBLIC_SUBSET_LABEL}: {complete}/57 complete, "
                    f"{checkpointed}/57 checkpoint files",
                    flush=True,
                )
                next_update = now + 60
            time.sleep(2)
        for process in processes:
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.terminate()
        return_codes = [process.wait() for process in processes]
        if failure is None and any(code != 0 for code in return_codes):
            failure = {
                "type": "worker_failure",
                "return_codes": return_codes,
                "observed_at_utc": _now(),
            }
    except KeyboardInterrupt:
        failure = {
            "type": "launcher_interrupted",
            "observed_at_utc": _now(),
        }
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.terminate()
    except BaseException as exc:
        failure = {
            "type": type(exc).__name__,
            "error": str(exc),
            "observed_at_utc": _now(),
        }
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
    finally:
        for handle in handles:
            handle.close()
    if manifest is None or run_dir is None:
        fallback = {
            "schema": "peps.sdf_public_subset_receipt",
            "schema_version": 1,
            "generated_at_utc": _now(),
            "status": "blocked",
            "scope": PUBLIC_SUBSET_LABEL,
            "paper": PAPER,
            "run_id": None,
            "run_manifest": None,
            "claims": {
                "full_table3": False,
                "paper_global_comparable": False,
                "canonical_four_shape": False,
                "public_subset_complete": False,
            },
            "coverage": {
                "expected_jobs": 57,
                "completed_jobs": 0,
                "expected_optimizer_steps": 6_840_000,
                "checkpointed_optimizer_steps": 0,
                "complete": False,
                "losses": ["mape", "l1"],
                "assets": list(PUBLIC_ASSETS),
            },
            "results": {
                "path": None,
                "sha256": None,
                "rows": 0,
                "per_instance_method_loss": [],
                "aggregate_manifests": [],
            },
            "gpu_runtime": {
                "cumulative_job_gpu_seconds": 0.0,
                "cumulative_job_gpu_hours": 0.0,
                "job_gpu_hours_by_physical_gpu": {},
                "launch_wall_seconds": None,
                "allocated_two_gpu_hours": None,
            },
            "workers": [],
            "checkpoints": [],
            "validation": {
                "valid": False,
                "errors": [failure],
                "volume_checksums_verified": False,
                "provenance_validated_assets": [],
                "parameter_assertions_passed": False,
                "all_complete_results_streamed_512_cubed": True,
                "schema": _portable(RECEIPT_SCHEMA),
            },
            "stonefish": {
                "asset_id": STONEFISH_ASSET,
                "status": "deferred_auth_required",
                "authorization_checked": False,
                "data_access_attempted": False,
                "substitution_allowed": False,
                "substitution_used": False,
                "numeric_results_generated": False,
            },
            "table4": {"executed": False, "consolidated": False},
            "limitations": [],
            "blocked": {
                "reason": failure,
                "exact_settings_preserved": True,
                "settings_reduced": False,
                "completed_checkpoints_preserved": True,
            },
        }
        atomic_write_json(blocked_receipt_path, fallback)
        print(json.dumps(fallback, indent=2, sort_keys=True), flush=True)
        return 3
    if failure is not None:
        payload = build_receipt(
            run_manifest_path=run_dir / "run-manifest.json",
            output_path=blocked_receipt_path,
            blocked=True,
            blocker=failure,
        )
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return 3
    payload = build_receipt(
        run_manifest_path=run_dir / "run-manifest.json",
        output_path=receipt_path,
    )
    if blocked_receipt_path.is_file():
        blocked_receipt_path.unlink()
    print(
        json.dumps(
            {
                "status": payload["status"],
                "scope": payload["scope"],
                "run_id": payload["run_id"],
                "coverage": payload["coverage"],
                "gpu_runtime": payload["gpu_runtime"],
                "receipt": _portable(receipt_path),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    prepare.add_argument("--processed-root", type=Path)
    prepare.add_argument("--require-idle", action="store_true")
    worker = commands.add_parser("worker")
    worker.add_argument("--run-manifest", type=Path, required=True)
    worker.add_argument("--rank", type=int, required=True)
    worker.add_argument("--physical-gpu-id", type=int, required=True)
    worker.add_argument("--processed-root", type=Path)
    launch_parser = commands.add_parser("launch")
    launch_parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    launch_parser.add_argument("--processed-root", type=Path)
    launch_parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    launch_parser.add_argument(
        "--blocked-receipt", type=Path, default=DEFAULT_BLOCKED_RECEIPT
    )
    report = commands.add_parser("report")
    report.add_argument("--run-manifest", type=Path, required=True)
    report.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    report.add_argument("--blocked", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    validate.add_argument("--no-work", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        manifest, run_dir = prepare_run_manifest(
            work_root=arguments.work_root,
            processed_root=arguments.processed_root,
            require_idle=arguments.require_idle,
        )
        print(
            json.dumps(
                {
                    "run_id": manifest["run_id"],
                    "run_dir": str(run_dir),
                    "scope": manifest["scope"],
                    "coverage": manifest["coverage"],
                    "parallelism": manifest["parallelism"],
                    "stonefish": manifest["stonefish"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "worker":
        return run_worker(
            run_manifest_path=arguments.run_manifest.resolve(),
            rank=arguments.rank,
            physical_gpu_id=arguments.physical_gpu_id,
            processed_root=arguments.processed_root,
        )
    if arguments.command == "launch":
        return launch(
            work_root=arguments.work_root,
            processed_root=arguments.processed_root,
            receipt_path=arguments.receipt,
            blocked_receipt_path=arguments.blocked_receipt,
        )
    if arguments.command == "report":
        payload = build_receipt(
            run_manifest_path=arguments.run_manifest.resolve(),
            output_path=arguments.output.resolve(),
            blocked=arguments.blocked,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "complete" else 3
    payload = validate_receipt(
        arguments.receipt.resolve(),
        require_work=not arguments.no_work,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
