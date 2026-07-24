"""Bounded, resumable Kodak image-budget convergence calibration.

This calibration-only runner never reads or resumes the full Table 1 work
namespace and never labels pilot evidence paper-exact or verified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from apps.image.data import image_to_coords_targets, load_paper_kodak
from data.manifest import hash_file, load_manifest
from experiments.config import ExperimentConfig, MethodConfig
from experiments.runner import (
    TensorInstance,
    _assert_budget,
    _build_model,
    _compression_factor,
    _parameter_counts,
    _set_initialization_seed,
    atomic_torch_save,
    atomic_write_json,
    evaluate_metrics,
)
from peps.metrics import metric_versions
from peps.report import collect_environment, collect_git_state
from peps.train import MinibatchStream, make_paper_optimizer, paper_recipe_from_mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/paper/image_convergence_pilot.json"
DEFAULT_OUTPUT_ROOT = ROOT / "results"
DEFAULT_EVIDENCE_DIR = ROOT / "results/image_convergence"
TABLE2_AUTHORIZATION_PATH = (
    ROOT / "results/texture_repro/table2_launch_authorization.json"
)
DISJOINT_GPU_OPT_IN = "PEPS_PILOT_ALLOW_DISJOINT_GPU"
STATUS = "bounded_protocol_assumption_calibration_not_table1"
CODE_PATHS = (
    ROOT / "experiments/image_convergence.py",
    ROOT / "apps/image/build.py",
    ROOT / "apps/image/data.py",
    ROOT / "experiments/runner.py",
    ROOT / "peps/train.py",
    ROOT / "peps/metrics.py",
)
_STOP_REQUESTED = False


@dataclass(frozen=True)
class PilotJob:
    index: int
    instance_id: str
    selection_role: str
    orientation: str
    method: MethodConfig
    category: str
    seed: int

    @property
    def identity(self) -> dict[str, object]:
        return {
            "job_index": self.index,
            "instance": self.instance_id,
            "method": self.method.name,
            "category": self.category,
            "seed": self.seed,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return hash_file(path, "sha256")


def _safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _interlock(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _active_table2_workers() -> list[dict[str, object]]:
    """Return live Table 2 workers and their explicit ROCr physical pins."""

    workers = []
    for process_root in Path("/proc").iterdir():
        if not process_root.name.isdigit():
            continue
        try:
            if process_root.stat().st_uid != os.getuid():
                continue
            command_bytes = (process_root / "cmdline").read_bytes()
            command = command_bytes.replace(b"\0", b" ").decode(
                errors="replace"
            ).strip()
            cwd = (process_root / "cwd").resolve()
            environment = {
                item.partition(b"=")[0].decode(errors="replace"): item.partition(
                    b"="
                )[2].decode(errors="replace")
                for item in (process_root / "environ").read_bytes().split(b"\0")
                if b"=" in item
            }
        except OSError:
            continue
        if cwd != ROOT and ROOT not in cwd.parents:
            continue
        if (
            "experiments.texture_repro" not in command
            or "--artifact table2" not in command
        ):
            continue
        raw_physical = environment.get(
            "PEPS_TEXTURE_PHYSICAL_GPU",
            environment.get("ROCR_VISIBLE_DEVICES"),
        )
        try:
            physical = int(str(raw_physical))
        except (TypeError, ValueError):
            physical = None
        workers.append(
            {
                "pid": int(process_root.name),
                "physical_gpu": physical,
                "command_sha256": hashlib.sha256(command_bytes).hexdigest(),
            }
        )
    return sorted(workers, key=lambda item: int(item["pid"]))


def _disjoint_table2_gate(
    physical_devices: Sequence[int],
) -> dict[str, object]:
    """Permit this pilot only on GPUs disjoint from an active authorized Table 2."""

    selected = tuple(int(value) for value in physical_devices)
    _interlock(
        os.environ.get(DISJOINT_GPU_OPT_IN) == "1",
        f"{DISJOINT_GPU_OPT_IN}=1 is required for the disjoint GPU pilot",
    )
    try:
        authorization = json.loads(
            TABLE2_AUTHORIZATION_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "cannot validate the active Table 2 authorization"
        ) from exc
    _interlock(
        isinstance(authorization, Mapping)
        and authorization.get("schema")
        == "peps.texture_table2_launch_authorization"
        and authorization.get("schema_version") == 1
        and authorization.get("authorized") is True,
        "Table 2 authorization is absent or malformed",
    )
    authorization_id = authorization.get("authorization_id")
    _interlock(
        isinstance(authorization_id, str)
        and authorization_id.startswith("explicit-user-request-"),
        "Table 2 has no explicit user authorization ID",
    )
    _interlock(
        authorization.get("block_other_texture_gpu_work") is True
        and authorization.get("table2_complete") is False,
        "Table 2 authorization is not an active reserved-GPU run",
    )
    raw_reserved = authorization.get("physical_gpus")
    _interlock(
        isinstance(raw_reserved, list)
        and raw_reserved
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in raw_reserved
        ),
        "Table 2 authorization has malformed physical GPUs",
    )
    reserved = tuple(int(value) for value in raw_reserved)
    _interlock(
        len(selected) == len(set(selected))
        and len(reserved) == len(set(reserved)),
        "pilot or Table 2 physical GPU allocation contains duplicates",
    )
    overlap = sorted(set(selected) & set(reserved))
    _interlock(
        not overlap,
        f"pilot GPUs overlap Table 2 reserved GPUs: {overlap}",
    )
    workers = _active_table2_workers()
    _interlock(workers, "authorized Table 2 has no active workers")
    worker_devices = [item.get("physical_gpu") for item in workers]
    _interlock(
        None not in worker_devices
        and len(worker_devices) == len(reserved)
        and set(worker_devices) == set(reserved),
        "active Table 2 worker pins do not match its authorization",
    )
    return {
        "status": "passed",
        "opt_in_environment": DISJOINT_GPU_OPT_IN,
        "authorization": {
            "path": str(
                TABLE2_AUTHORIZATION_PATH.relative_to(ROOT)
                if TABLE2_AUTHORIZATION_PATH.is_relative_to(ROOT)
                else TABLE2_AUTHORIZATION_PATH
            ),
            "sha256": _sha(TABLE2_AUTHORIZATION_PATH),
            "authorization_id": authorization_id,
        },
        "pilot_physical_devices": list(selected),
        "table2_reserved_physical_devices": list(reserved),
        "active_table2_workers": workers,
        "overlap": overlap,
    }


def load_pilot_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _check(payload.get("schema") == "peps.image_convergence_pilot", "bad schema")
    _check(payload.get("schema_version") == 1, "bad schema version")
    _check(payload.get("verification_status") == STATUS, "bad evidence status")
    dataset = payload.get("dataset")
    _check(isinstance(dataset, dict), "dataset must be an object")
    _check(
        dataset.get("manifest") == "data/manifests/kodak.json",
        "pilot must use the Kodak manifest",
    )
    subset = dataset.get("subset")
    _check(isinstance(subset, list) and len(subset) >= 2, "subset is too small")
    ids = [item.get("id") for item in subset]
    _check(len(ids) == len(set(ids)), "duplicate subset image")
    _check(
        {"landscape", "portrait"}
        <= {str(item.get("orientation")) for item in subset},
        "both orientations are required",
    )
    methods = payload.get("methods")
    _check(isinstance(methods, list) and methods, "methods are required")
    _check(
        {method.get("category") for method in methods}
        == {"baseline", "peps", "pink"},
        "baseline/PEPS/Pink coverage is required",
    )
    _check(len({method.get("name") for method in methods}) == len(methods), "duplicate method")
    for method in methods:
        MethodConfig.from_mapping(
            {
                **{key: value for key, value in method.items() if key != "category"},
                "role": method["category"],
            }
        )
    seeds = payload.get("seeds")
    _check(
        isinstance(seeds, list)
        and len(seeds) >= 2
        and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds),
        "at least two integer seeds are required",
    )
    _check(len(seeds) == len(set(seeds)), "duplicate seed")
    training = payload.get("training")
    _check(isinstance(training, dict), "training must be an object")
    budgets = training.get("evaluation_budgets")
    _check(
        isinstance(budgets, list)
        and len(budgets) >= 3
        and budgets == sorted(set(budgets))
        and budgets[0] > 0,
        "budgets must be sorted, unique, and positive",
    )
    _check(training.get("max_steps") == budgets[-1], "max_steps mismatch")
    _check(training.get("loss") == "l2" and training.get("cosine") is True, "recipe drift")
    checkpoint_every = training.get("checkpoint_every")
    poll_every = training.get("deadline_poll_every")
    _check(isinstance(checkpoint_every, int) and checkpoint_every > 0, "bad checkpoint interval")
    _check(isinstance(poll_every, int) and 0 < poll_every <= checkpoint_every, "bad deadline poll")
    resume = payload.get("resume_from")
    _check(isinstance(resume, dict), "resume lineage is required")
    source_budgets = resume.get("evaluation_budgets")
    source_step = resume.get("completed_step")
    _check(
        isinstance(source_budgets, list)
        and source_budgets
        and source_budgets == budgets[: len(source_budgets)]
        and source_step == source_budgets[-1]
        and source_step < training["max_steps"],
        "resume budgets must be a strict prefix ending at the source checkpoint",
    )
    for field in (
        "run_manifest",
        "run_manifest_sha256",
        "receipt",
        "receipt_sha256",
        "run_id",
        "pilot_manifest_sha256",
        "code_bundle_sha256",
    ):
        _check(isinstance(resume.get(field), str) and resume[field], f"bad resume {field}")
    transition = resume.get("scheduler_transition")
    _check(
        isinstance(transition, dict)
        and transition.get("kind") == "global_cosine_rehorizon_at_resume"
        and transition.get("source_t_max") == source_step
        and transition.get("target_t_max") == training["max_steps"],
        "resume scheduler transition drift",
    )
    _check(payload.get("metrics") == ["psnr", "ssim", "lsd", "mae"], "metric drift")
    parallel = payload.get("parallelism")
    _check(
        isinstance(parallel, dict)
        and parallel.get("mode") == "independent_job_shards"
        and parallel.get("world_size") == 4
        and parallel.get("physical_devices") == [2, 3]
        and parallel.get("maximum_concurrent_workers") == 2
        and parallel.get("same_model_distributed") is False,
        "parallelism drift",
    )
    expected_jobs = len(subset) * len(methods) * len(seeds)
    bounds = payload.get("bounds")
    _check(isinstance(bounds, dict), "bounds are required")
    _check(bounds.get("expected_jobs") == expected_jobs, "job bound mismatch")
    _check(
        bounds.get("expected_optimizer_steps")
        == expected_jobs * training["max_steps"],
        "step bound mismatch",
    )
    _check(
        bounds.get("expected_additional_optimizer_steps")
        == expected_jobs * (training["max_steps"] - source_step),
        "additional step bound mismatch",
    )
    _check(
        bounds.get("expected_curve_points") == expected_jobs * len(budgets),
        "curve-point bound mismatch",
    )
    wall = bounds.get("max_wall_clock_seconds")
    grace = bounds.get("shutdown_grace_seconds")
    _check(isinstance(wall, int) and 60 <= wall <= 14400, "unsafe wall bound")
    _check(isinstance(grace, int) and 10 <= grace < wall, "bad shutdown grace")
    _check(
        payload.get("decision_rule", {}).get("candidate")
        == "maximum_evaluated_budget_only",
        "decision rule may not infer an untested budget",
    )
    payload["_source"] = str(source)
    return payload


def _methods(manifest: Mapping[str, Any]) -> dict[str, MethodConfig]:
    return {
        str(method["name"]): MethodConfig.from_mapping(
            {
                **{key: value for key, value in method.items() if key != "category"},
                "role": method["category"],
            }
        )
        for method in manifest["methods"]
    }


def enumerate_pilot_jobs(manifest: Mapping[str, Any]) -> tuple[PilotJob, ...]:
    methods = _methods(manifest)
    jobs = []
    for item in manifest["dataset"]["subset"]:
        for method in manifest["methods"]:
            for seed in manifest["seeds"]:
                jobs.append(
                    PilotJob(
                        len(jobs),
                        str(item["id"]),
                        str(item["selection_role"]),
                        str(item["orientation"]),
                        methods[str(method["name"])],
                        str(method["category"]),
                        int(seed),
                    )
                )
    return tuple(jobs)


def shard_pilot_jobs(
    jobs: Sequence[PilotJob], *, rank: int, world_size: int
) -> tuple[PilotJob, ...]:
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    return tuple(job for job in jobs if job.index % world_size == rank)


def _luma_gradient(image: torch.Tensor) -> float:
    luma = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    return float(
        0.5 * (luma.diff(dim=0).abs().mean() + luma.diff(dim=1).abs().mean())
    )


def _dataset_receipts(
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, object]], dict[str, TensorInstance]]:
    """Verify all Kodak files, then materialize only the selected subset."""

    data_manifest_path = ROOT / str(manifest["dataset"]["manifest"])
    data_manifest = load_manifest(data_manifest_path)
    loaded = load_paper_kodak()
    by_id = {image.image_id: image for image in loaded}
    specs = {item["id"]: item for item in data_manifest["images"]}
    gradients = {image.image_id: _luma_gradient(image.tensor) for image in loaded}
    population_median = statistics.median(gradients.values())
    declared_median = float(
        manifest["dataset"]["selection_statistic"]["population_median"]
    )
    if not math.isclose(population_median, declared_median, abs_tol=1e-7):
        raise ValueError("Kodak population selection statistic drifted")
    portraits = [image.image_id for image in loaded if image.height > image.width]
    expected_roles = {
        "lowest_spatial_gradient_landscape": min(gradients, key=gradients.get),
        "highest_spatial_gradient_landscape": max(gradients, key=gradients.get),
        "portrait_near_population_median": min(
            portraits,
            key=lambda image_id: abs(gradients[image_id] - population_median),
        ),
    }
    receipts: list[dict[str, object]] = []
    instances: dict[str, TensorInstance] = {}
    for item in manifest["dataset"]["subset"]:
        image_id = str(item["id"])
        role = str(item["selection_role"])
        if expected_roles.get(role) != image_id:
            raise ValueError(f"{image_id} no longer satisfies {role}")
        observed = gradients[image_id]
        if not math.isclose(observed, float(item["statistic_value"]), abs_tol=1e-7):
            raise ValueError(f"{image_id} selection statistic drifted")
        image = by_id[image_id]
        orientation = "landscape" if image.width > image.height else "portrait"
        if orientation != item["orientation"]:
            raise ValueError(f"{image_id} orientation mismatch")
        coords, targets, (height, width) = image_to_coords_targets(image.tensor)
        instances[image_id] = TensorInstance(
            image_id,
            coords,
            targets,
            shape=(height, width, 3),
            metadata={
                "num_signal_values": targets.numel(),
                "resolution_xy": [width, height],
                "color_space": image.color_space,
            },
        )
        spec = specs[image_id]
        receipts.append(
            {
                "id": image_id,
                "selection_role": role,
                "orientation": orientation,
                "selection_statistic": observed,
                "path": str(image.source_path.relative_to(ROOT)),
                "bytes": image.source_path.stat().st_size,
                "sha256": spec["checksum"]["value"],
                "resolution_xy": [width, height],
                "color_space": image.color_space,
            }
        )
    return receipts, instances


def _code_receipts() -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha(path),
            "bytes": path.stat().st_size,
        }
        for path in CODE_PATHS
    ]


def _bundle_digest(receipts: Sequence[Mapping[str, object]]) -> str:
    value = json.dumps(
        list(receipts), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(value).hexdigest()


def _active_kfd_pids() -> list[int]:
    root = Path("/sys/class/kfd/kfd/proc")
    if not root.is_dir():
        return []
    result = []
    for path in root.iterdir():
        try:
            pid = int(path.name)
        except ValueError:
            continue
        if pid != os.getpid():
            result.append(pid)
    return sorted(result)


def _gpu_receipt(
    manifest: Mapping[str, Any], *, require_idle: bool
) -> dict[str, object]:
    active = _active_kfd_pids()
    physical_devices = [
        int(value) for value in manifest["parallelism"]["physical_devices"]
    ]
    disjoint_gate = _disjoint_table2_gate(physical_devices)
    expected_visibility = ",".join(str(value) for value in physical_devices)
    visibility = {
        name: os.environ.get(name)
        for name in (
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        )
    }
    if (
        visibility["HIP_VISIBLE_DEVICES"] != expected_visibility
        or visibility["CUDA_VISIBLE_DEVICES"] != expected_visibility
        or visibility["ROCR_VISIBLE_DEVICES"] not in (None, "")
    ):
        raise RuntimeError(
            "HIP and CUDA visibility must explicitly reserve physical GPUs "
            f"{expected_visibility}, while ROCR visibility must be unset to "
            f"avoid ROCm double filtering; observed {visibility}"
        )
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != len(physical_devices)
    ):
        raise RuntimeError(
            f"pilot requires exactly {len(physical_devices)} visible ROCm/CUDA devices"
        )
    required_arch = str(manifest["parallelism"]["required_architecture"])
    minimum_free = int(manifest["bounds"]["minimum_free_device_memory_bytes"])
    devices = []
    for index, physical_index in enumerate(physical_devices):
        properties = torch.cuda.get_device_properties(index)
        architecture = str(getattr(properties, "gcnArchName", "unknown"))
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        if architecture != required_arch:
            raise RuntimeError(f"device {index}: {architecture} != {required_arch}")
        if free_bytes < minimum_free:
            raise RuntimeError(f"device {index}: only {free_bytes} bytes free")
        devices.append(
            {
                "index": index,
                "physical_index": physical_index,
                "name": properties.name,
                "architecture": architecture,
                "total_memory_bytes": int(total_bytes),
                "free_memory_bytes_at_preflight": int(free_bytes),
            }
        )
    return {
        "torch_version": torch.__version__,
        "device_count": torch.cuda.device_count(),
        "active_kfd_pids_observed": active,
        "idle_check_required": require_idle,
        "idle_check_scope": (
            "selected_device_free-memory threshold; KFD activity on reserved "
            "physical GPUs 0/1 for the authorized Table 2 workers is intentionally "
            "allowed only after the disjoint gate passes"
        ),
        "disjoint_table2_gate": disjoint_gate,
        "visibility": visibility,
        "devices": devices,
    }


def _run_identity(
    manifest_path: Path,
) -> tuple[str, str, list[dict[str, object]], str]:
    manifest_sha = _sha(manifest_path)
    receipts = _code_receipts()
    code_sha = _bundle_digest(receipts)
    return f"{manifest_sha[:12]}-{code_sha[:12]}", manifest_sha, receipts, code_sha


def _preflight(
    manifest_path: Path, *, output_root: Path, require_idle: bool
) -> tuple[dict[str, Any], Path]:
    manifest = load_pilot_manifest(manifest_path)
    dataset_receipts, _ = _dataset_receipts(manifest)
    run_id, manifest_sha, code_receipts, code_sha = _run_identity(manifest_path)
    hardware = _gpu_receipt(manifest, require_idle=require_idle)
    run_dir = output_root / "work/image-convergence" / run_id
    path = run_dir / "run-manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (
            existing.get("run_id") != run_id
            or existing.get("pilot_manifest_sha256") != manifest_sha
            or existing.get("code_bundle_sha256") != code_sha
        ):
            raise ValueError("existing run manifest identity drift")
        lineage = existing.get("resume_lineage")
        _check(isinstance(lineage, Mapping), "existing run has no resume lineage")
        receipt_snapshot = ROOT / str(lineage["receipt_snapshot"]["path"])
        curves_snapshot = ROOT / str(lineage["curves_snapshot"]["path"])
        observed = _validate_resume_source(
            manifest,
            receipt_path=receipt_snapshot,
            curves_path=curves_snapshot,
        )
        _check(
            observed == lineage.get("validated_source"),
            "source resume lineage drifted after preflight",
        )
        _check(
            not any(item["process_alive"] for item in _worker_statuses(run_dir)),
            "image convergence workers are already active",
        )
        return existing, run_dir
    jobs = enumerate_pilot_jobs(manifest)
    resume = manifest["resume_from"]
    source_receipt_path = ROOT / str(resume["receipt"])
    validated_source = _validate_resume_source(
        manifest,
        receipt_path=source_receipt_path,
        curves_path=ROOT
        / str(
            json.loads(source_receipt_path.read_text(encoding="utf-8"))[
                "curves"
            ]["path"]
        ),
    )
    snapshot_dir = run_dir / "resume-source"
    receipt_snapshot = snapshot_dir / "receipt-30k.json"
    curves_snapshot = snapshot_dir / "curves-30k.csv"
    source_curves_path = ROOT / str(validated_source["aggregate_curves"]["path"])
    _atomic_write_bytes(receipt_snapshot, source_receipt_path.read_bytes())
    _atomic_write_bytes(curves_snapshot, source_curves_path.read_bytes())
    resume_lineage = {
        "validated_source": validated_source,
        "receipt_snapshot": {
            "path": str(receipt_snapshot.relative_to(ROOT)),
            "bytes": receipt_snapshot.stat().st_size,
            "sha256": _sha(receipt_snapshot),
        },
        "curves_snapshot": {
            "path": str(curves_snapshot.relative_to(ROOT)),
            "bytes": curves_snapshot.stat().st_size,
            "sha256": _sha(curves_snapshot),
        },
        "scheduler_transition": dict(resume["scheduler_transition"]),
    }
    payload = {
        "schema": "peps.image_convergence_run_manifest",
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": _now(),
        "verification_status": STATUS,
        "pilot_manifest": str(manifest_path.relative_to(ROOT)),
        "pilot_manifest_sha256": manifest_sha,
        "code_receipts": code_receipts,
        "code_bundle_sha256": code_sha,
        "dataset": {
            "id": manifest["dataset"]["id"],
            "manifest": manifest["dataset"]["manifest"],
            "manifest_sha256": _sha(ROOT / manifest["dataset"]["manifest"]),
            "selection_note": manifest["dataset"]["sampling_note"],
            "instances": dataset_receipts,
        },
        "jobs": [
            {
                **job.identity,
                "assigned_rank": job.index
                % int(manifest["parallelism"]["world_size"]),
            }
            for job in jobs
        ],
        "expected_jobs": len(jobs),
        "expected_optimizer_steps": manifest["bounds"]["expected_optimizer_steps"],
        "expected_additional_optimizer_steps": manifest["bounds"][
            "expected_additional_optimizer_steps"
        ],
        "expected_curve_points": manifest["bounds"]["expected_curve_points"],
        "budgets": list(manifest["training"]["evaluation_budgets"]),
        "bounds": dict(manifest["bounds"]),
        "parallelism": dict(manifest["parallelism"]),
        "hardware": hardware,
        "resume_lineage": resume_lineage,
        "protocol_assumptions": list(manifest["protocol_assumptions"]),
        "git": collect_git_state(ROOT),
        "environment": collect_environment(),
        "metric_versions": metric_versions(),
        "output_dir": str(run_dir.relative_to(ROOT)),
    }
    atomic_write_json(path, payload)
    return payload, run_dir


def _experiment_config(
    manifest: Mapping[str, Any], manifest_path: Path
) -> ExperimentConfig:
    return ExperimentConfig(
        schema_version=1,
        name=str(manifest["name"]),
        paper=str(manifest["paper"]),
        task="image",
        profile="smoke",
        dataset=str(manifest["dataset"]["id"]),
        canonical=False,
        seeds=tuple(int(seed) for seed in manifest["seeds"]),
        training={},
        runner={},
        methods=tuple(_methods(manifest).values()),
        source=manifest_path,
    )


def _job_paths(run_dir: Path, job: PilotJob) -> tuple[Path, Path]:
    stem = (
        Path("raw")
        / _safe(job.instance_id)
        / _safe(job.method.name)
        / f"seed-{job.seed}"
    )
    return (
        run_dir / stem.with_suffix(".json"),
        run_dir / "checkpoints" / stem.with_suffix(".pt"),
    )


def _checkpoint_identity(
    run_manifest: Mapping[str, Any], job: PilotJob, max_steps: int
) -> dict[str, object]:
    return {
        "run_id": run_manifest["run_id"],
        **job.identity,
        "max_steps": max_steps,
        "pilot_manifest_sha256": run_manifest["pilot_manifest_sha256"],
        "code_bundle_sha256": run_manifest["code_bundle_sha256"],
    }


def _validate_checkpoint(
    state: Mapping[str, Any],
    expected_identity: Mapping[str, object],
    *,
    max_steps: int,
) -> int:
    _check(state.get("schema_version") == 1, "bad checkpoint schema")
    _check(
        isinstance(state.get("pilot_job"), Mapping)
        and dict(state["pilot_job"]) == dict(expected_identity),
        "checkpoint belongs to another job",
    )
    step = state.get("step")
    _check(
        isinstance(step, int) and not isinstance(step, bool) and 0 <= step <= max_steps,
        "bad checkpoint step",
    )
    stream = state.get("minibatch_stream")
    _check(
        isinstance(stream, Mapping) and int(stream.get("draws", -1)) == step,
        "checkpoint stream/step mismatch",
    )
    _check(isinstance(state.get("model"), Mapping) and state["model"], "empty model state")
    _check(isinstance(state.get("optimizer"), Mapping), "bad optimizer state")
    _check(isinstance(state.get("scheduler"), Mapping), "bad scheduler state")
    return step


def _validate_curve(
    curve: Mapping[str, Any],
    *,
    run_id: str,
    job: PilotJob,
    budgets: Sequence[int],
    allow_partial: bool,
) -> None:
    _check(curve.get("schema") == "peps.image_convergence_curve", "bad curve schema")
    _check(curve.get("schema_version") == 1, "bad curve schema version")
    _check(
        curve.get("run_id") == run_id and curve.get("job") == job.identity,
        "curve belongs to another job",
    )
    points = curve.get("points")
    _check(isinstance(points, list), "curve points must be a list")
    steps = [point.get("step") for point in points]
    _check(steps == list(budgets[: len(steps)]), "curve is not a budget prefix")
    for point in points:
        metrics = point.get("metrics")
        _check(
            isinstance(metrics, Mapping)
            and set(metrics) == {"psnr", "ssim", "lsd", "mae"},
            "curve metric mismatch",
        )
        _check(
            all(math.isfinite(float(value)) for value in metrics.values()),
            "non-finite curve metric",
        )
        for name in (
            "runtime_seconds_cumulative",
            "optimizer_runtime_seconds_cumulative",
            "evaluation_seconds",
            "evaluation_seconds_cumulative",
        ):
            value = point.get(name)
            _check(isinstance(value, (int, float)) and value >= 0, f"bad {name}")
    if curve.get("state") == "complete" or not allow_partial:
        _check(steps == list(budgets), "complete curve is missing points")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_resume_source(
    manifest: Mapping[str, Any],
    *,
    receipt_path: Path,
    curves_path: Path,
) -> dict[str, Any]:
    """Validate the frozen 30k source before any continuation state is created."""

    resume = manifest["resume_from"]
    source_step = int(resume["completed_step"])
    source_budgets = tuple(int(value) for value in resume["evaluation_budgets"])
    _check(_sha(receipt_path) == resume["receipt_sha256"], "source receipt digest drift")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _check(receipt.get("run_id") == resume["run_id"], "source receipt run drift")
    _check(receipt.get("paper_exact") is False, "source receipt was improperly promoted")
    _check(receipt.get("verified_table1") is False, "source receipt claims Table 1")
    _check(receipt.get("budgets") == list(source_budgets), "source receipt budget drift")
    _check(receipt.get("coverage", {}).get("complete") is True, "source receipt incomplete")
    _check(
        receipt.get("integrity", {}).get("active_workers") == 0,
        "source receipt recorded active workers",
    )
    _check(
        receipt.get("pilot_manifest_sha256") == resume["pilot_manifest_sha256"],
        "source pilot manifest digest drift",
    )
    _check(
        receipt.get("code_bundle_sha256") == resume["code_bundle_sha256"],
        "source code bundle digest drift",
    )
    _check(
        _sha(curves_path) == receipt["curves"]["sha256"],
        "source aggregate curve digest drift",
    )
    source_manifest_path = ROOT / str(resume["run_manifest"])
    _check(
        _sha(source_manifest_path) == resume["run_manifest_sha256"],
        "source run manifest digest drift",
    )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    _check(source_manifest.get("run_id") == resume["run_id"], "source run id drift")
    _check(
        source_manifest.get("pilot_manifest_sha256")
        == resume["pilot_manifest_sha256"],
        "source run pilot manifest drift",
    )
    _check(
        source_manifest.get("code_bundle_sha256")
        == resume["code_bundle_sha256"],
        "source run code bundle drift",
    )
    _check(
        source_manifest.get("budgets") == list(source_budgets),
        "source run budgets drift",
    )
    source_run_dir = source_manifest_path.parent
    _check(
        not any(item["process_alive"] for item in _worker_statuses(source_run_dir)),
        "source convergence workers are active",
    )
    checkpoint_receipts = {
        (
            int(item["job_index"]),
            str(item["instance"]),
            str(item["method"]),
            int(item["seed"]),
        ): item
        for item in receipt["checkpoints"]
    }
    jobs = []
    source_runtime = 0.0
    for job in enumerate_pilot_jobs(manifest):
        result_path, checkpoint_path = _job_paths(source_run_dir, job)
        curve = json.loads(result_path.read_text(encoding="utf-8"))
        _validate_curve(
            curve,
            run_id=str(source_manifest["run_id"]),
            job=job,
            budgets=source_budgets,
            allow_partial=False,
        )
        _check(
            curve.get("state") == "complete" and curve.get("last_step") == source_step,
            "source curve is not complete at the pinned checkpoint",
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _check(
            _validate_checkpoint(
                state,
                _checkpoint_identity(source_manifest, job, source_step),
                max_steps=source_step,
            )
            == source_step,
            "source checkpoint step drift",
        )
        scheduler = state["scheduler"]
        _check(
            scheduler.get("T_max") == source_step
            and scheduler.get("last_epoch") == source_step,
            "source checkpoint scheduler horizon drift",
        )
        _check(
            all(
                abs(float(group.get("lr", math.inf))) <= 1e-12
                for group in state["optimizer"]["param_groups"]
            ),
            "source checkpoint did not finish at zero cosine learning rate",
        )
        key = (job.index, job.instance_id, job.method.name, job.seed)
        recorded = checkpoint_receipts.get(key)
        _check(recorded is not None, "source receipt omitted a checkpoint")
        checkpoint_sha = _sha(checkpoint_path)
        _check(
            checkpoint_sha == recorded["sha256"]
            and checkpoint_path.stat().st_size == int(recorded["bytes"]),
            "source checkpoint digest or size drift",
        )
        runtime = float(curve["runtime_seconds_cumulative"])
        source_runtime += runtime
        jobs.append(
            {
                **job.identity,
                "curve": {
                    "path": str(result_path.relative_to(ROOT)),
                    "bytes": result_path.stat().st_size,
                    "sha256": _sha(result_path),
                },
                "checkpoint": {
                    "path": str(checkpoint_path.relative_to(ROOT)),
                    "bytes": checkpoint_path.stat().st_size,
                    "sha256": checkpoint_sha,
                    "step": source_step,
                },
                "runtime_seconds": runtime,
            }
        )
        del state
    _check(len(jobs) == int(receipt["coverage"]["completed_jobs"]), "source job count drift")
    return {
        "run_id": str(resume["run_id"]),
        "completed_step": source_step,
        "evaluation_budgets": list(source_budgets),
        "run_manifest": {
            "path": str(resume["run_manifest"]),
            "bytes": source_manifest_path.stat().st_size,
            "sha256": _sha(source_manifest_path),
        },
        "receipt": {
            "path": str(resume["receipt"]),
            "bytes": receipt_path.stat().st_size,
            "sha256": _sha(receipt_path),
        },
        "aggregate_curves": {
            "path": str(receipt["curves"]["path"]),
            "bytes": curves_path.stat().st_size,
            "sha256": _sha(curves_path),
            "rows": int(receipt["curves"]["rows"]),
        },
        "source_aggregate_job_runtime_seconds": source_runtime,
        "active_workers": 0,
        "jobs": jobs,
    }


def _curve_template(
    run_manifest: Mapping[str, Any],
    manifest: Mapping[str, Any],
    job: PilotJob,
    counts: Mapping[str, int],
    compression: float,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    return {
        "schema": "peps.image_convergence_curve",
        "schema_version": 1,
        "run_id": run_manifest["run_id"],
        "verification_status": STATUS,
        "job": job.identity,
        "assigned_rank": rank,
        "world_size": world_size,
        "parallelism": {
            "mode": "independent_job_shards",
            "same_model_distributed": False,
        },
        "parameters": dict(counts),
        "compression_factor": compression,
        "training": dict(manifest["training"]),
        "metric_versions": metric_versions(),
        "points": [],
        "last_step": 0,
        "state": "running",
        "started_at_utc": _now(),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resume_lineage_for_job(
    run_manifest: Mapping[str, Any], job: PilotJob
) -> Mapping[str, Any]:
    matches = [
        item
        for item in run_manifest["resume_lineage"]["validated_source"]["jobs"]
        if int(item["job_index"]) == job.index
    ]
    _check(len(matches) == 1, "resume lineage does not uniquely identify the job")
    item = matches[0]
    _check(
        all(item.get(key) == value for key, value in job.identity.items()),
        "resume lineage job identity drift",
    )
    return item


def _rehorizon_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    *,
    completed_step: int,
    target_steps: int,
) -> list[float]:
    """Move a completed short cosine onto the declared global target horizon."""

    _check(scheduler.T_max == target_steps, "target cosine horizon drift")
    factor = 0.5 * (1.0 + math.cos(math.pi * completed_step / target_steps))
    learning_rates = [
        scheduler.eta_min + (float(base) - scheduler.eta_min) * factor
        for base in scheduler.base_lrs
    ]
    _check(
        len(learning_rates) == len(optimizer.param_groups),
        "optimizer/scheduler group count drift",
    )
    for group, base, value in zip(
        optimizer.param_groups, scheduler.base_lrs, learning_rates
    ):
        group["initial_lr"] = float(base)
        group["lr"] = value
    scheduler.last_epoch = completed_step
    scheduler._step_count = completed_step + 1
    scheduler._last_lr = list(learning_rates)
    return learning_rates


def _run_job(
    *,
    job: PilotJob,
    instance: TensorInstance,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    run_manifest: Mapping[str, Any],
    run_dir: Path,
    rank: int,
    world_size: int,
    device: torch.device,
    deadline_epoch: float,
) -> str:
    """Run or resume one job, returning complete, skipped, or bounded_stop."""

    result_path, checkpoint_path = _job_paths(run_dir, job)
    budgets = tuple(int(value) for value in manifest["training"]["evaluation_budgets"])
    max_steps = int(manifest["training"]["max_steps"])
    expected_identity = _checkpoint_identity(run_manifest, job, max_steps)
    curve = None
    if result_path.is_file():
        curve = json.loads(result_path.read_text(encoding="utf-8"))
        _validate_curve(
            curve,
            run_id=str(run_manifest["run_id"]),
            job=job,
            budgets=budgets,
            allow_partial=True,
        )
        if curve["state"] == "complete":
            _check(checkpoint_path.is_file(), "complete curve has no checkpoint")
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            _check(
                _validate_checkpoint(state, expected_identity, max_steps=max_steps)
                == max_steps,
                "final checkpoint is incomplete",
            )
            return "skipped"

    _set_initialization_seed(job.seed)
    config = _experiment_config(manifest, manifest_path)
    model, _ = _build_model(config, job.method, instance)
    counts = _parameter_counts(model)
    _assert_budget(job.method, counts)
    compression = _compression_factor(instance, counts["total"])
    model = model.to(device)
    coords = instance.coords.to(device)
    targets = instance.targets.to(device)
    values = {
        key: manifest["training"][key]
        for key in ("task", "loss", "batch_size", "model_lr", "encoder_lr", "cosine")
    }
    values.update(
        {
            "steps": max_steps,
            "checkpoint_every": int(manifest["training"]["checkpoint_every"]),
            "log_every": max_steps,
            "seed": job.seed,
            "device": device,
        }
    )
    recipe = paper_recipe_from_mapping(values)
    optimizer = make_paper_optimizer(model, recipe)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    stream = MinibatchStream(coords.shape[0], recipe.batch_size, job.seed)
    curve_missing = curve is None
    if curve_missing:
        curve = _curve_template(
            run_manifest, manifest, job, counts, compression, rank, world_size
        )

    start_step = 0
    prior_runtime = 0.0
    evaluation_cumulative = 0.0
    imported_source = False
    if checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        start_step = _validate_checkpoint(
            state, expected_identity, max_steps=max_steps
        )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        stream.load_state_dict(state["minibatch_stream"])
        pilot_runtime = state.get("pilot_runtime", {})
        if isinstance(pilot_runtime, Mapping):
            prior_runtime = float(
                pilot_runtime.get("runtime_seconds_cumulative", 0.0)
            )
            evaluation_cumulative = float(
                pilot_runtime.get("evaluation_seconds_cumulative", 0.0)
            )
    elif curve_missing and "resume_lineage" in run_manifest:
        source = _resume_lineage_for_job(run_manifest, job)
        source_result_path = ROOT / str(source["curve"]["path"])
        source_checkpoint_path = ROOT / str(source["checkpoint"]["path"])
        _check(
            _sha(source_result_path) == source["curve"]["sha256"],
            "source curve changed after preflight",
        )
        _check(
            _sha(source_checkpoint_path) == source["checkpoint"]["sha256"],
            "source checkpoint changed after preflight",
        )
        source_manifest_path = (
            ROOT
            / str(
                run_manifest["resume_lineage"]["validated_source"][
                    "run_manifest"
                ]["path"]
            )
        )
        source_manifest = json.loads(
            source_manifest_path.read_text(encoding="utf-8")
        )
        source_budgets = tuple(
            int(value) for value in manifest["resume_from"]["evaluation_budgets"]
        )
        source_step = int(manifest["resume_from"]["completed_step"])
        source_curve = json.loads(
            source_result_path.read_text(encoding="utf-8")
        )
        _validate_curve(
            source_curve,
            run_id=str(source_manifest["run_id"]),
            job=job,
            budgets=source_budgets,
            allow_partial=False,
        )
        state = torch.load(
            source_checkpoint_path, map_location=device, weights_only=False
        )
        _check(
            _validate_checkpoint(
                state,
                _checkpoint_identity(source_manifest, job, source_step),
                max_steps=source_step,
            )
            == source_step,
            "source checkpoint is not at the pinned resume step",
        )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        transition_lrs = _rehorizon_cosine_scheduler(
            optimizer,
            scheduler,
            completed_step=source_step,
            target_steps=max_steps,
        )
        stream.load_state_dict(state["minibatch_stream"])
        start_step = source_step
        prior_runtime = float(source_curve["runtime_seconds_cumulative"])
        evaluation_cumulative = float(
            source_curve["points"][-1]["evaluation_seconds_cumulative"]
        )
        curve["points"] = list(source_curve["points"])
        curve["last_step"] = source_step
        curve["source_started_at_utc"] = source_curve.get("started_at_utc")
        curve["resume_lineage"] = {
            "source_run_id": source_manifest["run_id"],
            "source_curve_sha256": source["curve"]["sha256"],
            "source_checkpoint_sha256": source["checkpoint"]["sha256"],
            "source_step": source_step,
            "scheduler_transition": dict(
                run_manifest["resume_lineage"]["scheduler_transition"]
            ),
            "learning_rates_after_transition": transition_lrs,
        }
        imported_source = True
        del state
    elif curve["points"]:
        raise ValueError("partial curve has no checkpoint")
    if curve["points"]:
        last_point = curve["points"][-1]
        _check(int(last_point["step"]) <= start_step, "curve is ahead of checkpoint")
        prior_runtime = max(prior_runtime, float(last_point["runtime_seconds_cumulative"]))
        evaluation_cumulative = max(
            evaluation_cumulative,
            float(last_point["evaluation_seconds_cumulative"]),
        )
    evaluated = {int(point["step"]) for point in curve["points"]}
    _check(
        not [budget for budget in budgets if budget < start_step and budget not in evaluated],
        "checkpoint skipped an unevaluated budget",
    )
    session_started = time.perf_counter()

    def runtime_now() -> float:
        return prior_runtime + time.perf_counter() - session_started

    def save_checkpoint(step: int) -> None:
        atomic_torch_save(
            checkpoint_path,
            {
                "schema_version": 1,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "minibatch_stream": stream.state_dict(),
                "pilot_job": expected_identity,
                "pilot_runtime": {
                    "runtime_seconds_cumulative": runtime_now(),
                    "evaluation_seconds_cumulative": evaluation_cumulative,
                },
                "resume_lineage": curve.get("resume_lineage"),
            },
        )

    def persist(state: str, step: int, **extra: object) -> None:
        curve.update(
            {
                "state": state,
                "last_step": step,
                "runtime_seconds_cumulative": runtime_now(),
                "updated_at_utc": _now(),
                **extra,
            }
        )
        atomic_write_json(result_path, curve)

    def evaluate(step: int) -> None:
        nonlocal evaluation_cumulative
        _sync(device)
        started = time.perf_counter()
        model.eval()
        predictions = []
        chunk = int(manifest["training"]["render_chunk"])
        with torch.no_grad():
            for offset in range(0, instance.coords.shape[0], chunk):
                predictions.append(
                    model(instance.coords[offset : offset + chunk].to(device)).cpu()
                )
        prediction = torch.cat(predictions)
        metrics = evaluate_metrics(
            "image",
            [name for name in manifest["metrics"] if name != "mae"],
            instance,
            prediction,
        )
        metrics["mae"] = float(F.l1_loss(prediction, instance.targets).item())
        model.train()
        _sync(device)
        evaluation_seconds = time.perf_counter() - started
        evaluation_cumulative += evaluation_seconds
        current_runtime = runtime_now()
        curve["points"].append(
            {
                "step": step,
                "metrics": metrics,
                "runtime_seconds_cumulative": current_runtime,
                "optimizer_runtime_seconds_cumulative": max(
                    0.0, current_runtime - evaluation_cumulative
                ),
                "evaluation_seconds": evaluation_seconds,
                "evaluation_seconds_cumulative": evaluation_cumulative,
                "completed_at_utc": _now(),
            }
        )
        persist("running", step)
        save_checkpoint(step)

    if imported_source:
        save_checkpoint(start_step)
        persist(
            "running",
            start_step,
            resume_imported_at_utc=_now(),
        )

    if start_step in budgets and start_step not in evaluated:
        evaluate(start_step)
        evaluated.add(start_step)
    checkpoint_every = int(manifest["training"]["checkpoint_every"])
    poll_every = int(manifest["training"]["deadline_poll_every"])
    model.train()
    completed_step = start_step
    for step_index in range(start_step, max_steps):
        if (
            step_index % poll_every == 0
            and (_STOP_REQUESTED or time.time() >= deadline_epoch)
        ):
            save_checkpoint(completed_step)
            persist(
                "bounded_stop",
                completed_step,
                stop_reason=(
                    "signal_requested" if _STOP_REQUESTED else "wall_clock_deadline"
                ),
                finished_at_utc=_now(),
            )
            return "bounded_stop"
        indices = stream.next().to(device=device)
        prediction = model(coords.index_select(0, indices))
        loss = F.mse_loss(prediction, targets.index_select(0, indices))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        completed_step = step_index + 1
        if completed_step in budgets:
            save_checkpoint(completed_step)
            evaluate(completed_step)
            evaluated.add(completed_step)
        elif completed_step % checkpoint_every == 0:
            save_checkpoint(completed_step)
    persist("complete", completed_step, finished_at_utc=_now())
    save_checkpoint(completed_step)
    return "complete"


def _process_identity(pid: int) -> dict[str, object] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
        suffix = stat[stat.rfind(")") + 1 :].split()
        start_time_ticks = int(suffix[19])
    except (IndexError, OSError, ValueError):
        return None
    if not command or not boot_id:
        return None
    return {
        "boot_id": boot_id,
        "start_time_ticks": start_time_ticks,
        "command_sha256": hashlib.sha256(command).hexdigest(),
    }


def _worker_alive(status: Mapping[str, Any]) -> tuple[bool, str]:
    pid = status.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int):
        return False, "invalid_pid"
    observed = _process_identity(pid)
    expected = status.get("process_identity")
    if observed is None:
        return False, "pid_not_present_or_unreadable"
    if not isinstance(expected, Mapping):
        return False, "missing_boot_scoped_identity"
    fields = ("boot_id", "start_time_ticks", "command_sha256")
    if any(observed.get(field) != expected.get(field) for field in fields):
        return False, "boot_or_process_identity_mismatch"
    return True, "boot_scoped_identity_match"


def _signal_stop(_signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _worker(
    *,
    manifest_path: Path,
    run_dir: Path,
    rank: int,
    world_size: int,
    device: torch.device,
    physical_device_index: int,
    deadline_epoch: float,
) -> int:
    signal.signal(signal.SIGINT, _signal_stop)
    signal.signal(signal.SIGTERM, _signal_stop)
    manifest = load_pilot_manifest(manifest_path)
    _check(
        world_size == int(manifest["parallelism"]["world_size"]),
        "worker world size differs from manifest",
    )
    run_manifest = json.loads(
        (run_dir / "run-manifest.json").read_text(encoding="utf-8")
    )
    run_id, manifest_sha, _, code_sha = _run_identity(manifest_path)
    _check(
        run_manifest.get("run_id") == run_id
        and run_manifest.get("pilot_manifest_sha256") == manifest_sha
        and run_manifest.get("code_bundle_sha256") == code_sha,
        "run manifest differs from current manifest/code",
    )
    _check(device.type == "cuda", "calibration workers refuse CPU execution")
    _check(
        physical_device_index in manifest["parallelism"]["physical_devices"],
        "worker physical device is outside the bounded pilot allocation",
    )
    for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        _check(
            os.environ.get(name) == str(physical_device_index),
            f"{name} does not explicitly select physical GPU {physical_device_index}",
        )
    _check(
        os.environ.get("ROCR_VISIBLE_DEVICES") in (None, ""),
        "ROCR_VISIBLE_DEVICES must be unset to avoid ROCm double filtering",
    )
    properties = torch.cuda.get_device_properties(device)
    architecture = str(getattr(properties, "gcnArchName", "unknown"))
    _check(
        architecture == manifest["parallelism"]["required_architecture"],
        f"worker architecture is {architecture}",
    )
    _, instances = _dataset_receipts(manifest)
    assigned = shard_pilot_jobs(
        enumerate_pilot_jobs(manifest), rank=rank, world_size=world_size
    )
    path = run_dir / f"worker-rank-{rank}.json"
    status: dict[str, Any] = {
        "schema": "peps.image_convergence_worker",
        "schema_version": 1,
        "run_id": run_manifest["run_id"],
        "rank": rank,
        "world_size": world_size,
        "pid": os.getpid(),
        "process_identity": _process_identity(os.getpid()),
        "device": str(device),
        "physical_device_index": physical_device_index,
        "visibility": {
            name: os.environ.get(name)
            for name in (
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "architecture": architecture,
        "deadline_epoch": deadline_epoch,
        "expected_jobs": len(assigned),
        "job_indices": [job.index for job in assigned],
        "state": "running",
        "started_at_utc": _now(),
    }
    atomic_write_json(path, status)
    outcomes = {"complete": 0, "skipped": 0, "bounded_stop": 0}
    try:
        for job in assigned:
            outcome = _run_job(
                job=job,
                instance=instances[job.instance_id],
                manifest=manifest,
                manifest_path=manifest_path,
                run_manifest=run_manifest,
                run_dir=run_dir,
                rank=rank,
                world_size=world_size,
                device=device,
                deadline_epoch=deadline_epoch,
            )
            outcomes[outcome] += 1
            status.update({"outcomes": outcomes, "last_job_index": job.index})
            atomic_write_json(path, status)
            if outcome == "bounded_stop":
                break
    except BaseException as exc:
        status.update(
            {
                "state": "failed",
                "finished_at_utc": _now(),
                "outcomes": outcomes,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        atomic_write_json(path, status)
        raise
    bounded = outcomes["bounded_stop"] > 0 or _STOP_REQUESTED
    status.update(
        {
            "state": "bounded_stop" if bounded else "complete",
            "finished_at_utc": _now(),
            "outcomes": outcomes,
        }
    )
    atomic_write_json(path, status)
    return 3 if bounded else 0


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _relation(left: float, right: float, tolerance: float) -> int:
    delta = left - right
    return 0 if abs(delta) <= tolerance else (1 if delta > 0 else -1)


def analyse_curves(
    curves: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    complete: bool,
) -> dict[str, Any]:
    """Compute convergence, paired ordering, and the predeclared decision."""

    budgets = [int(value) for value in manifest["training"]["evaluation_budgets"]]
    methods = [str(method["name"]) for method in manifest["methods"]]
    tolerance = float(manifest["decision_rule"]["ranking_tie_tolerance_db"])
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    paired: dict[tuple[int, str, str, int], float] = {}
    for curve in curves:
        job = curve["job"]
        for point in curve["points"]:
            step = int(point["step"])
            method = str(job["method"])
            grouped.setdefault((step, method), []).append(point)
            paired[(step, method, str(job["instance"]), int(job["seed"]))] = float(
                point["metrics"]["psnr"]
            )

    aggregates = []
    means_by_step: dict[int, dict[str, float]] = {}
    for step in budgets:
        means_by_step[step] = {}
        for method in methods:
            points = grouped.get((step, method), [])
            metrics = {}
            for metric in manifest["metrics"]:
                values = [float(point["metrics"][metric]) for point in points]
                metrics[str(metric)] = {
                    "count": len(values),
                    "mean": _mean(values),
                    "std": _std(values),
                }
            if metrics["psnr"]["mean"] is not None:
                means_by_step[step][method] = float(metrics["psnr"]["mean"])
            aggregates.append(
                {
                    "step": step,
                    "method": method,
                    "metrics": metrics,
                    "runtime_seconds_mean": _mean(
                        [
                            float(point["runtime_seconds_cumulative"])
                            for point in points
                        ]
                    ),
                }
            )

    pairs = list(itertools.combinations(methods, 2))
    final_means = means_by_step[budgets[-1]]
    final_relations = {
        pair: _relation(final_means[pair[0]], final_means[pair[1]], tolerance)
        for pair in pairs
        if pair[0] in final_means and pair[1] in final_means
    }
    rankings = []
    for step in budgets:
        means = means_by_step[step]
        order = sorted(means, key=lambda name: (-means[name], name))
        comparable = agreements = 0
        relations = {}
        for pair in pairs:
            if pair[0] not in means or pair[1] not in means:
                continue
            relation = _relation(means[pair[0]], means[pair[1]], tolerance)
            relations[f"{pair[0]}__vs__{pair[1]}"] = relation
            if pair in final_relations:
                comparable += 1
                agreements += relation == final_relations[pair]
        rankings.append(
            {
                "step": step,
                "order_best_to_worst": order,
                "mean_psnr_db": means,
                "pairwise_relations": relations,
                "pairwise_agreement_with_final": (
                    agreements / comparable if comparable else None
                ),
            }
        )

    paired_ordering = []
    units = sorted(
        {
            (str(curve["job"]["instance"]), int(curve["job"]["seed"]))
            for curve in curves
        }
    )
    for step in budgets:
        for left, right in pairs:
            deltas = [
                paired[(step, right, instance, seed)]
                - paired[(step, left, instance, seed)]
                for instance, seed in units
                if (step, left, instance, seed) in paired
                and (step, right, instance, seed) in paired
            ]
            paired_ordering.append(
                {
                    "step": step,
                    "baseline": left,
                    "candidate": right,
                    "count": len(deltas),
                    "mean_delta_psnr_db": _mean(deltas),
                    "wins": sum(delta > tolerance for delta in deltas),
                    "ties": sum(abs(delta) <= tolerance for delta in deltas),
                    "losses": sum(delta < -tolerance for delta in deltas),
                }
            )

    gains = {}
    if len(budgets) > 1:
        previous = means_by_step[budgets[-2]]
        final = means_by_step[budgets[-1]]
        gains = {
            method: final[method] - previous[method]
            for method in methods
            if method in previous and method in final
        }
    trailing = int(manifest["decision_rule"]["required_trailing_rank_intervals"])
    trailing_rankings = rankings[-(trailing + 1) :]
    rankings_stable = (
        len(trailing_rankings) == trailing + 1
        and all(
            item["pairwise_agreement_with_final"] == 1.0
            for item in trailing_rankings
        )
    )
    gain_limit = float(
        manifest["decision_rule"]["maximum_absolute_mean_psnr_gain_db"]
    )
    gains_stable = len(gains) == len(methods) and all(
        abs(value) <= gain_limit for value in gains.values()
    )
    if complete and rankings_stable and gains_stable:
        outcome = "recommended_protocol_assumption"
        recommendation = budgets[-1]
        reason = (
            f"All jobs completed; ranking was unchanged over the final {trailing} "
            f"intervals and every final-interval mean PSNR change was at most "
            f"{gain_limit:.2f} dB."
        )
    else:
        failures = []
        if not complete:
            failures.append("not all jobs and points completed")
        if not rankings_stable:
            failures.append("trailing aggregate ranking was unstable")
        if not gains_stable:
            failures.append(
                f"a final-interval mean PSNR change exceeded {gain_limit:.2f} dB"
            )
        outcome = "inconclusive"
        recommendation = None
        reason = "; ".join(failures) + ". No untested budget is inferred."
    return {
        "outcome": outcome,
        "recommended_budget_steps": recommendation,
        "reason": reason,
        "decision_rule": dict(manifest["decision_rule"]),
        "rankings_stable": rankings_stable,
        "final_interval_gains_psnr_db": gains,
        "final_interval_gains_within_threshold": gains_stable,
        "aggregates": aggregates,
        "rankings": rankings,
        "paired_ordering": paired_ordering,
    }


CSV_COLUMNS = (
    "run_id",
    "job_index",
    "instance",
    "selection_role",
    "orientation",
    "method",
    "method_category",
    "seed",
    "step",
    "psnr",
    "ssim",
    "lsd",
    "mae",
    "runtime_seconds_cumulative",
    "optimizer_runtime_seconds_cumulative",
    "evaluation_seconds",
    "evaluation_seconds_cumulative",
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _worker_statuses(run_dir: Path) -> list[dict[str, Any]]:
    statuses = []
    for path in sorted(run_dir.glob("worker-rank-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        _check(payload.get("schema") == "peps.image_convergence_worker", "bad worker receipt")
        alive, reason = _worker_alive(payload)
        recorded = payload.get("state")
        payload["recorded_state"] = recorded
        payload["process_alive"] = alive
        payload["liveness_evidence"] = {
            "status": "verified_alive" if alive else "not_alive",
            "reason": reason,
        }
        payload["effective_state"] = (
            recorded
            if recorded != "running" or alive
            else "stopped_after_reboot_or_exit"
        )
        statuses.append(payload)
    return statuses


def _report(
    *, manifest_path: Path, run_dir: Path, evidence_dir: Path
) -> dict[str, Any]:
    manifest = load_pilot_manifest(manifest_path)
    run_manifest = json.loads(
        (run_dir / "run-manifest.json").read_text(encoding="utf-8")
    )
    jobs = enumerate_pilot_jobs(manifest)
    budgets = tuple(int(value) for value in manifest["training"]["evaluation_budgets"])
    max_steps = budgets[-1]
    curves = []
    checkpoints = []
    executed_steps = 0
    errors = []
    for job in jobs:
        result_path, checkpoint_path = _job_paths(run_dir, job)
        curve = None
        if result_path.is_file():
            try:
                curve = json.loads(result_path.read_text(encoding="utf-8"))
                _validate_curve(
                    curve,
                    run_id=str(run_manifest["run_id"]),
                    job=job,
                    budgets=budgets,
                    allow_partial=True,
                )
            except Exception as exc:
                errors.append(
                    {
                        "path": str(result_path.relative_to(ROOT)),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                curve = None
            else:
                curves.append(curve)
        checkpoint_step = 0
        if checkpoint_path.is_file():
            try:
                state = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                checkpoint_step = _validate_checkpoint(
                    state,
                    _checkpoint_identity(run_manifest, job, max_steps),
                    max_steps=max_steps,
                )
                source = _resume_lineage_for_job(run_manifest, job)
                checkpoint_lineage = state.get("resume_lineage")
                _check(
                    isinstance(checkpoint_lineage, Mapping)
                    and checkpoint_lineage.get("source_checkpoint_sha256")
                    == source["checkpoint"]["sha256"],
                    "extended checkpoint lost its source lineage",
                )
                checkpoints.append(
                    {
                        **job.identity,
                        "path": str(checkpoint_path.relative_to(ROOT)),
                        "step": checkpoint_step,
                        "bytes": checkpoint_path.stat().st_size,
                        "sha256": _sha(checkpoint_path),
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "path": str(checkpoint_path.relative_to(ROOT)),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        executed_steps += max(
            checkpoint_step,
            0 if curve is None else int(curve.get("last_step", 0)),
        )
    if errors:
        raise ValueError(
            "pilot evidence failed integrity validation: "
            + json.dumps(errors, sort_keys=True)
        )

    complete_curves = [curve for curve in curves if curve["state"] == "complete"]
    completed_points = sum(len(curve["points"]) for curve in curves)
    complete = (
        len(complete_curves) == len(jobs)
        and completed_points == manifest["bounds"]["expected_curve_points"]
        and len(checkpoints) == len(jobs)
        and all(checkpoint["step"] == max_steps for checkpoint in checkpoints)
    )
    rows = []
    by_index = {job.index: job for job in jobs}
    for curve in sorted(curves, key=lambda item: int(item["job"]["job_index"])):
        identity = curve["job"]
        job = by_index[int(identity["job_index"])]
        for point in curve["points"]:
            rows.append(
                {
                    "run_id": run_manifest["run_id"],
                    "job_index": identity["job_index"],
                    "instance": identity["instance"],
                    "selection_role": job.selection_role,
                    "orientation": job.orientation,
                    "method": identity["method"],
                    "method_category": identity["category"],
                    "seed": identity["seed"],
                    "step": point["step"],
                    "psnr": point["metrics"]["psnr"],
                    "ssim": point["metrics"]["ssim"],
                    "lsd": point["metrics"]["lsd"],
                    "mae": point["metrics"]["mae"],
                    "runtime_seconds_cumulative": point[
                        "runtime_seconds_cumulative"
                    ],
                    "optimizer_runtime_seconds_cumulative": point[
                        "optimizer_runtime_seconds_cumulative"
                    ],
                    "evaluation_seconds": point["evaluation_seconds"],
                    "evaluation_seconds_cumulative": point[
                        "evaluation_seconds_cumulative"
                    ],
                }
            )
    curves_path = evidence_dir / "curves.csv"
    _write_csv(curves_path, rows)
    workers = _worker_statuses(run_dir)
    starts = [
        datetime.fromisoformat(item["started_at_utc"])
        for item in workers
        if isinstance(item.get("started_at_utc"), str)
    ]
    finishes = [
        datetime.fromisoformat(item["finished_at_utc"])
        for item in workers
        if isinstance(item.get("finished_at_utc"), str)
    ]
    launch_wall = (
        (max(finishes) - min(starts)).total_seconds()
        if starts and finishes
        else None
    )
    runtimes = [
        float(curve.get("runtime_seconds_cumulative", 0.0)) for curve in curves
    ]
    source_runtime_by_job = {
        int(item["job_index"]): float(item["runtime_seconds"])
        for item in run_manifest["resume_lineage"]["validated_source"]["jobs"]
    }
    extension_runtimes = [
        max(
            0.0,
            float(curve.get("runtime_seconds_cumulative", 0.0))
            - source_runtime_by_job[int(curve["job"]["job_index"])],
        )
        for curve in curves
    ]
    source_runtime = float(
        run_manifest["resume_lineage"]["validated_source"][
            "source_aggregate_job_runtime_seconds"
        ]
    )
    aggregate_runtime = sum(runtimes)
    extension_runtime = sum(extension_runtimes)
    analysis = analyse_curves(curves, manifest, complete=complete)
    receipt = {
        "schema": "peps.image_convergence_receipt",
        "schema_version": 1,
        "generated_at_utc": _now(),
        "verification_status": STATUS,
        "paper_exact": False,
        "verified_table1": False,
        "run_id": run_manifest["run_id"],
        "run_manifest": str((run_dir / "run-manifest.json").relative_to(ROOT)),
        "pilot_manifest": str(manifest_path.relative_to(ROOT)),
        "pilot_manifest_sha256": run_manifest["pilot_manifest_sha256"],
        "code_bundle_sha256": run_manifest["code_bundle_sha256"],
        "dataset": run_manifest["dataset"],
        "methods": [
            {
                "name": method["name"],
                "category": method["category"],
                "expected_encoder_params": method["expected_encoder_params"],
                "expected_total_params": method["expected_total_params"],
            }
            for method in manifest["methods"]
        ],
        "seeds": list(manifest["seeds"]),
        "budgets": list(budgets),
        "coverage": {
            "expected_jobs": len(jobs),
            "completed_jobs": len(complete_curves),
            "expected_optimizer_steps": manifest["bounds"][
                "expected_optimizer_steps"
            ],
            "executed_optimizer_steps": executed_steps,
            "source_optimizer_steps": (
                len(jobs) * int(manifest["resume_from"]["completed_step"])
            ),
            "expected_additional_optimizer_steps": manifest["bounds"][
                "expected_additional_optimizer_steps"
            ],
            "executed_additional_optimizer_steps": max(
                0,
                executed_steps
                - len(jobs) * int(manifest["resume_from"]["completed_step"]),
            ),
            "expected_curve_points": manifest["bounds"]["expected_curve_points"],
            "completed_curve_points": completed_points,
            "complete": complete,
        },
        "runtime": {
            "latest_launch_wall_seconds": launch_wall,
            "aggregate_job_runtime_seconds": aggregate_runtime,
            "source_aggregate_job_runtime_seconds": source_runtime,
            "extension_aggregate_job_runtime_seconds": extension_runtime,
            "aggregate_gpu_hours": aggregate_runtime / 3600.0,
            "source_gpu_hours": source_runtime / 3600.0,
            "extension_gpu_hours": extension_runtime / 3600.0,
            "mean_job_runtime_seconds": _mean(runtimes),
            "max_job_runtime_seconds": max(runtimes, default=None),
            "physical_gpu_count": len(
                manifest["parallelism"]["physical_devices"]
            ),
            "physical_devices": list(
                manifest["parallelism"]["physical_devices"]
            ),
            "manifest_wall_clock_bound_seconds": manifest["bounds"][
                "max_wall_clock_seconds"
            ],
            "latest_launch_within_bound": (
                launch_wall is not None
                and launch_wall <= manifest["bounds"]["max_wall_clock_seconds"]
            ),
        },
        "workers": workers,
        "jobs": [
            {
                **curve["job"],
                "state": curve["state"],
                "last_step": curve["last_step"],
                "runtime_seconds": curve.get("runtime_seconds_cumulative"),
                "curve_points": len(curve["points"]),
            }
            for curve in sorted(
                curves, key=lambda item: int(item["job"]["job_index"])
            )
        ],
        "checkpoints": checkpoints,
        "resume_lineage": run_manifest["resume_lineage"],
        "curves": {
            "path": str(curves_path.relative_to(ROOT)),
            "sha256": _sha(curves_path),
            "rows": len(rows),
        },
        "analysis": analysis,
        "protocol_assumptions": list(manifest["protocol_assumptions"]),
        "protocol_assumption_status": {
            "status": "unresolved_local_calibration_assumptions",
            "paper_exact": False,
            "verified_table1": False,
            "full_table1_run_resumed": False,
        },
        "integrity": {
            "valid": True,
            "errors": [],
            "raw_curves_validated": len(curves),
            "checkpoints_validated": len(checkpoints),
            "active_workers": sum(item["process_alive"] for item in workers),
        },
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(evidence_dir / "receipt.json", receipt)
    _write_readme(evidence_dir / "README.md", receipt)
    return receipt


def _write_readme(path: Path, receipt: Mapping[str, Any]) -> None:
    coverage = receipt["coverage"]
    runtime = receipt["runtime"]
    analysis = receipt["analysis"]
    recommendation = analysis["recommended_budget_steps"]
    result = (
        f"Recommend **{recommendation:,} optimizer steps** only as a local "
        "protocol assumption for this frozen recipe."
        if recommendation is not None
        else "**Inconclusive**; no optimizer-step budget is recommended."
    )
    wall = runtime["latest_launch_wall_seconds"]
    wall_text = "unavailable" if wall is None else f"{float(wall):.3f} seconds"
    text = f"""# Kodak image-budget convergence pilot

This is a bounded calibration artifact, not a paper-exact result and not a
verified Table 1 reproduction. It did not resume the 648-job Table 1 run.

## Result

{result}

{analysis["reason"]}

- Jobs: {coverage["completed_jobs"]}/{coverage["expected_jobs"]}
- Optimizer steps: {coverage["executed_optimizer_steps"]}/{coverage["expected_optimizer_steps"]}
- Additional optimizer steps: {coverage["executed_additional_optimizer_steps"]}/{coverage["expected_additional_optimizer_steps"]}
- Curve points: {coverage["completed_curve_points"]}/{coverage["expected_curve_points"]}
- Latest queued two-GPU launch wall time: {wall_text}
- Total/source/extension GPU-hours: {runtime["aggregate_gpu_hours"]:.6f} / {runtime["source_gpu_hours"]:.6f} / {runtime["extension_gpu_hours"]:.6f}
- Budgets: {", ".join(str(value) for value in receipt["budgets"])}
- Methods: Grid (baseline), G-PEPS (PEPS), G-P-PEPS (Pink)
- Seeds: {", ".join(str(value) for value in receipt["seeds"])}

The deterministic three-image subset spans the lowest and highest measured
Kodak luma-gradient images and includes a portrait near the 24-image median.
It is a coverage subset, not a population estimate.

The 30k model, Adam moments, and minibatch stream were hash-validated and
retained. The source cosine had already reached zero, so the continuation
explicitly re-horizons the learning rates to the 120k global-cosine value at
step 30k. Any recommendation is therefore a checkpoint-continuation protocol
assumption, not an uninterrupted paper-exact 120k schedule.

## Evidence and commands

- `receipt.json` records protocol, hardware, runtime, checkpoint hashes,
  stability, the decision, and assumption status.
- `curves.csv` has one runtime/quality row per image/method/seed/budget.
- Local resumable checkpoints and raw curves live under the receipt's
  `run_manifest` work namespace and are intentionally git-ignored.

```bash
bash scripts/run_image_convergence_2gpu.sh
.venv/bin/python -m experiments.image_convergence validate
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_receipt(
    receipt_path: Path = DEFAULT_EVIDENCE_DIR / "receipt.json",
    *,
    require_work: bool = True,
) -> dict[str, object]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _check(receipt.get("schema") == "peps.image_convergence_receipt", "bad receipt schema")
    _check(receipt.get("schema_version") == 1, "bad receipt schema version")
    _check(
        receipt.get("verification_status") == STATUS
        and receipt.get("paper_exact") is False
        and receipt.get("verified_table1") is False,
        "receipt improperly upgrades pilot evidence",
    )
    curves_path = ROOT / str(receipt["curves"]["path"])
    _check(_sha(curves_path) == receipt["curves"]["sha256"], "curve digest mismatch")
    with curves_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    _check(len(rows) == int(receipt["curves"]["rows"]), "curve row count mismatch")
    _check(
        len(rows) == int(receipt["coverage"]["completed_curve_points"]),
        "coverage/curve row mismatch",
    )
    keys = {
        (row["instance"], row["method"], int(row["seed"]), int(row["step"]))
        for row in rows
    }
    _check(len(keys) == len(rows), "duplicate curve rows")
    for row in rows:
        _check(
            all(
                math.isfinite(float(row[metric]))
                for metric in ("psnr", "ssim", "lsd", "mae")
            ),
            "non-finite CSV metric",
        )
    work_validated = False
    if require_work:
        run_manifest_path = ROOT / str(receipt["run_manifest"])
        regenerated = _report(
            manifest_path=ROOT / str(receipt["pilot_manifest"]),
            run_dir=run_manifest_path.parent,
            evidence_dir=receipt_path.parent,
        )
        _check(regenerated["run_id"] == receipt["run_id"], "regenerated run drift")
        _check(
            regenerated["integrity"]["active_workers"] == 0,
            "pilot workers are still active",
        )
        work_validated = True
    return {
        "valid": True,
        "receipt": str(receipt_path),
        "curves_rows": len(rows),
        "work_validated": work_validated,
        "verification_status": STATUS,
    }


def _completed_count(run_dir: Path) -> int:
    count = 0
    for path in run_dir.glob("raw/**/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        count += payload.get("state") == "complete"
    return count


def _launch(
    *,
    manifest_path: Path,
    output_root: Path,
    evidence_dir: Path,
    wall_seconds: int | None,
) -> int:
    manifest = load_pilot_manifest(manifest_path)
    maximum_wall = int(manifest["bounds"]["max_wall_clock_seconds"])
    selected_wall = maximum_wall if wall_seconds is None else int(wall_seconds)
    _check(
        60 <= selected_wall <= maximum_wall,
        f"wall-seconds must be in [60, {maximum_wall}]",
    )
    run_manifest, run_dir = _preflight(
        manifest_path, output_root=output_root, require_idle=True
    )
    world_size = int(manifest["parallelism"]["world_size"])
    grace = int(manifest["bounds"]["shutdown_grace_seconds"])
    hard_deadline = time.time() + selected_wall
    worker_deadline = hard_deadline - grace
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    physical_devices = [
        int(value) for value in manifest["parallelism"]["physical_devices"]
    ]
    maximum_concurrent = int(
        manifest["parallelism"]["maximum_concurrent_workers"]
    )
    _check(
        maximum_concurrent == len(physical_devices) == 2,
        "the bounded launcher must reserve exactly two disjoint physical GPUs",
    )
    pending = list(range(world_size))
    available_devices = list(physical_devices)
    active: dict[int, tuple[subprocess.Popen, int, Any]] = {}
    return_codes_by_rank: dict[int, int] = {}

    def signal_processes(value: int) -> None:
        for process, _, _ in active.values():
            if process.poll() is None:
                try:
                    process.send_signal(value)
                except ProcessLookupError:
                    pass

    def start_worker(rank: int, physical_index: int) -> None:
        handle = (logs_dir / f"rank-{rank}.log").open("a", encoding="utf-8")
        environment = os.environ.copy()
        environment.pop("ROCR_VISIBLE_DEVICES", None)
        for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            environment[name] = str(physical_index)
        environment["PYTHONUNBUFFERED"] = "1"
        command = [
            sys.executable,
            "-m",
            "experiments.image_convergence",
            "worker",
            "--manifest",
            str(manifest_path),
            "--run-dir",
            str(run_dir),
            "--rank",
            str(rank),
            "--world-size",
            str(world_size),
            "--device",
            "cuda:0",
            "--physical-device-index",
            str(physical_index),
            "--deadline-epoch",
            repr(worker_deadline),
        ]
        active[rank] = (
            subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            ),
            physical_index,
            handle,
        )

    try:
        next_update = 0.0
        deadline_reached = False
        while pending or active:
            now = time.time()
            if now >= hard_deadline:
                print("hard wall-clock bound reached; stopping workers", flush=True)
                signal_processes(signal.SIGINT)
                deadline_reached = True
                break
            while (
                pending
                and available_devices
                and len(active) < maximum_concurrent
            ):
                start_worker(pending.pop(0), available_devices.pop(0))
            for rank, (process, physical_index, handle) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                return_codes_by_rank[rank] = code
                handle.close()
                del active[rank]
                available_devices.append(physical_index)
                available_devices.sort()
            if now >= next_update:
                print(
                    f"pilot progress: {_completed_count(run_dir)}/"
                    f"{run_manifest['expected_jobs']} jobs; "
                    f"logical ranks complete {len(return_codes_by_rank)}/{world_size}; "
                    f"{max(0, int(hard_deadline - now))}s remaining",
                    flush=True,
                )
                next_update = now + 60
            time.sleep(1)
        stop_started = time.time()
        while (
            any(process.poll() is None for process, _, _ in active.values())
            and time.time() - stop_started < grace
        ):
            time.sleep(0.25)
        if deadline_reached:
            signal_processes(signal.SIGTERM)
            time.sleep(1)
        for rank, (process, _, handle) in list(active.items()):
            if process.poll() is None:
                process.kill()
            return_codes_by_rank[rank] = process.wait()
            handle.close()
            del active[rank]
        return_codes = [
            return_codes_by_rank.get(rank, 3) for rank in range(world_size)
        ]
    except BaseException:
        signal_processes(signal.SIGINT)
        time.sleep(1)
        signal_processes(signal.SIGTERM)
        raise
    finally:
        for process, _, handle in active.values():
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            handle.close()
    receipt = _report(
        manifest_path=manifest_path, run_dir=run_dir, evidence_dir=evidence_dir
    )
    print(
        json.dumps(
            {
                "run_id": receipt["run_id"],
                "return_codes": return_codes,
                "coverage": receipt["coverage"],
                "runtime": receipt["runtime"],
                "outcome": receipt["analysis"]["outcome"],
                "recommended_budget_steps": receipt["analysis"][
                    "recommended_budget_steps"
                ],
                "reason": receipt["analysis"]["reason"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if receipt["coverage"]["complete"] and all(
        code == 0 for code in return_codes
    ) else 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    preflight.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    preflight.add_argument("--require-idle-gpus", action="store_true")
    worker = commands.add_parser("worker")
    worker.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    worker.add_argument("--run-dir", type=Path, required=True)
    worker.add_argument("--rank", type=int, required=True)
    worker.add_argument("--world-size", type=int, required=True)
    worker.add_argument("--device", required=True)
    worker.add_argument("--physical-device-index", type=int, required=True)
    worker.add_argument("--deadline-epoch", type=float, required=True)
    launch = commands.add_parser("launch")
    launch.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    launch.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    launch.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    launch.add_argument("--wall-seconds", type=int)
    report = commands.add_parser("report")
    report.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    validate = commands.add_parser("validate")
    validate.add_argument(
        "--receipt",
        type=Path,
        default=DEFAULT_EVIDENCE_DIR / "receipt.json",
    )
    validate.add_argument("--no-work", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "preflight":
        payload, run_dir = _preflight(
            arguments.manifest.resolve(),
            output_root=arguments.output_root.resolve(),
            require_idle=arguments.require_idle_gpus,
        )
        print(
            json.dumps(
                {
                    "run_id": payload["run_id"],
                    "run_dir": str(run_dir),
                    "expected_jobs": payload["expected_jobs"],
                    "expected_optimizer_steps": payload[
                        "expected_optimizer_steps"
                    ],
                    "bounds": payload["bounds"],
                    "hardware": payload["hardware"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "worker":
        return _worker(
            manifest_path=arguments.manifest.resolve(),
            run_dir=arguments.run_dir.resolve(),
            rank=arguments.rank,
            world_size=arguments.world_size,
            device=torch.device(arguments.device),
            physical_device_index=arguments.physical_device_index,
            deadline_epoch=arguments.deadline_epoch,
        )
    if arguments.command == "launch":
        return _launch(
            manifest_path=arguments.manifest.resolve(),
            output_root=arguments.output_root.resolve(),
            evidence_dir=arguments.evidence_dir.resolve(),
            wall_seconds=arguments.wall_seconds,
        )
    if arguments.command == "report":
        payload = _report(
            manifest_path=arguments.manifest.resolve(),
            run_dir=arguments.run_dir.resolve(),
            evidence_dir=arguments.evidence_dir.resolve(),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    payload = validate_receipt(
        arguments.receipt.resolve(), require_work=not arguments.no_work
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
