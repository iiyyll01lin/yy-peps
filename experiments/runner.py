"""Checkpointable, atomically-writing, GPU-sharded experiment runner."""

from __future__ import annotations

import importlib
import inspect
import json
import math
import os
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn

from peps.metrics import flip, iou, lpips, lpsd, lsd, metric_versions, psnr, ssim
from peps.train import (
    fit_paper,
    paper_recipe_from_mapping,
    render_full,
    split_encoder_decoder_parameters,
)

from .config import ExperimentConfig, MethodConfig


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class TensorInstance:
    """Data-layer-neutral tensors consumed by the experiment infrastructure."""

    name: str
    coords: torch.Tensor
    targets: torch.Tensor
    shape: tuple[int, ...] | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("instance name cannot be empty")
        if self.coords.ndim != 2 or self.targets.ndim != 2:
            raise ValueError("coords and targets must be rank-2")
        if self.coords.shape[0] != self.targets.shape[0]:
            raise ValueError("coords and targets must have matching row counts")
        if self.shape is not None and math.prod(self.shape) != self.targets.numel():
            raise ValueError("instance shape does not match target values")
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata))
        )


@dataclass(frozen=True)
class RunSpec:
    instance: TensorInstance
    method: MethodConfig
    seed: int
    index: int


def enumerate_jobs(
    config: ExperimentConfig,
    instances: Sequence[TensorInstance],
) -> tuple[RunSpec, ...]:
    jobs = []
    index = 0
    for instance in sorted(instances, key=lambda item: item.name):
        for method in config.methods:
            seeds = method.seeds or config.seeds
            for seed in seeds:
                jobs.append(RunSpec(instance, method, seed, index))
                index += 1
    return tuple(jobs)


def shard_jobs(
    jobs: Sequence[RunSpec],
    *,
    rank: int,
    world_size: int,
) -> tuple[RunSpec, ...]:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")
    return tuple(job for job in jobs if job.index % world_size == rank)


def resolve_shard(
    rank: int | None = None,
    world_size: int | None = None,
    local_rank: int | None = None,
) -> tuple[int, int, int]:
    resolved_rank = int(os.environ.get("RANK", 0) if rank is None else rank)
    resolved_world = int(
        os.environ.get("WORLD_SIZE", 1)
        if world_size is None
        else world_size
    )
    resolved_local = int(
        os.environ.get("LOCAL_RANK", resolved_rank)
        if local_rank is None
        else local_rank
    )
    if resolved_world < 1 or not 0 <= resolved_rank < resolved_world:
        raise ValueError("invalid rank/world_size")
    return resolved_rank, resolved_world, resolved_local


def atomic_write_json(path, payload: Mapping) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_plain(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_torch_save(path, payload: Mapping) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _safe_component(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )


def _git_provenance(start: Path) -> dict[str, str | bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=start.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=start.parent,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": revision, "git_dirty": dirty}


def _import_factory(path: str) -> Callable:
    if ":" not in path:
        raise ValueError("factory must use 'module:function' syntax")
    module_name, attribute = path.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"factory {path!r} is not callable")
    return factory


def _set_initialization_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)


def _build_model(
    config: ExperimentConfig,
    method: MethodConfig,
    instance: TensorInstance,
) -> tuple[nn.Module, int]:
    factory = _import_factory(method.factory)
    kwargs = _plain(method.kwargs)
    signature = inspect.signature(factory)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if (
        config.task in {"image", "texture"}
        and "out_dim" not in kwargs
        and ("out_dim" in signature.parameters or accepts_kwargs)
    ):
        kwargs["out_dim"] = instance.targets.shape[1]
    built = factory(**kwargs)
    if isinstance(built, tuple):
        model, reported_count = built
    else:
        model, reported_count = built, None
    if not isinstance(model, nn.Module):
        raise TypeError(f"factory {method.factory!r} did not return an nn.Module")
    actual_count = sum(parameter.numel() for parameter in model.parameters())
    if reported_count is not None and int(reported_count) != actual_count:
        raise AssertionError(
            f"factory reported {reported_count} parameters, found {actual_count}"
        )
    return model, actual_count


def _parameter_counts(model: nn.Module) -> dict[str, int]:
    encoder, decoder = split_encoder_decoder_parameters(model)
    encoder_count = sum(parameter.numel() for parameter in encoder)
    decoder_count = sum(parameter.numel() for parameter in decoder)
    return {
        "encoder": encoder_count,
        "decoder": decoder_count,
        "total": encoder_count + decoder_count,
    }


def _assert_budget(method: MethodConfig, counts: Mapping[str, int]) -> None:
    if (
        method.expected_encoder_params is not None
        and counts["encoder"] != method.expected_encoder_params
    ):
        raise AssertionError(
            f"{method.name}: encoder parameters {counts['encoder']} != "
            f"{method.expected_encoder_params}"
        )
    if (
        method.expected_total_params is not None
        and counts["total"] != method.expected_total_params
    ):
        raise AssertionError(
            f"{method.name}: total parameters {counts['total']} != "
            f"{method.expected_total_params}"
        )


def _image_pair(
    instance: TensorInstance,
    prediction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if instance.shape is None:
        raise ValueError("image and texture metrics require instance.shape")
    return prediction.reshape(instance.shape), instance.targets.reshape(instance.shape)


def evaluate_metrics(
    task: str,
    names: Sequence[str],
    instance: TensorInstance,
    prediction: torch.Tensor,
) -> dict[str, float]:
    results = {}
    if task == "sdf":
        for name in names:
            if name != "iou":
                raise ValueError(f"unsupported SDF metric: {name}")
            results[name] = iou(prediction < 0, instance.targets < 0)
        return results

    predicted_image, target_image = _image_pair(instance, prediction)
    functions = {
        "psnr": psnr,
        "ssim": ssim,
        "flip": flip,
        "lpips": lpips,
        "lsd": lsd,
        "lpsd": lpsd,
    }
    for name in names:
        if name not in functions:
            raise ValueError(f"unsupported image metric: {name}")
        results[name] = functions[name](predicted_image, target_image)
    return results


def _compression_factor(
    instance: TensorInstance,
    parameter_count: int,
) -> float:
    signal_values = int(
        instance.metadata.get("num_signal_values", instance.targets.numel())
    )
    return signal_values / parameter_count


def _assert_compression(method: MethodConfig, factor: float) -> None:
    if (
        method.compression_factor_min is not None
        and factor < method.compression_factor_min
    ):
        raise AssertionError(
            f"{method.name}: compression factor {factor:.4f} is below "
            f"{method.compression_factor_min}"
        )
    if (
        method.compression_factor_max is not None
        and factor > method.compression_factor_max
    ):
        raise AssertionError(
            f"{method.name}: compression factor {factor:.4f} is above "
            f"{method.compression_factor_max}"
        )


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir,
        *,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        device: torch.device | None = None,
        force: bool = False,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.force = force
        if device is None:
            if torch.cuda.is_available():
                available = torch.cuda.device_count()
                device = torch.device(
                    "cuda", local_rank if local_rank < available else 0
                )
            else:
                device = torch.device("cpu")
        self.device = device

    def _paths(self, spec: RunSpec) -> tuple[Path, Path]:
        stem = (
            Path("raw")
            / _safe_component(spec.instance.name)
            / _safe_component(spec.method.name)
            / f"seed-{spec.seed}"
        )
        return (
            self.output_dir / stem.with_suffix(".json"),
            self.output_dir / "checkpoints" / stem.with_suffix(".pt"),
        )

    def run_one(self, spec: RunSpec) -> dict:
        result_path, checkpoint_path = self._paths(spec)
        if result_path.exists() and not self.force:
            return json.loads(result_path.read_text(encoding="utf-8"))

        _set_initialization_seed(spec.seed)
        model, _ = _build_model(self.config, spec.method, spec.instance)
        counts = _parameter_counts(model)
        _assert_budget(spec.method, counts)
        compression = _compression_factor(spec.instance, counts["total"])
        _assert_compression(spec.method, compression)

        training_values = dict(self.config.training)
        training_values.update(spec.method.training)
        training_values["seed"] = spec.seed
        recipe = replace(
            paper_recipe_from_mapping(training_values),
            device=self.device,
        )
        resume_state = None
        if checkpoint_path.exists() and not self.force:
            resume_state = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )

        def checkpoint_callback(step: int, state: Mapping) -> None:
            atomic_torch_save(checkpoint_path, state)

        started = time.perf_counter()
        model = fit_paper(
            model,
            spec.instance.coords,
            spec.instance.targets,
            recipe,
            on_checkpoint=checkpoint_callback,
            resume_state=resume_state,
        )
        prediction = render_full(
            model,
            spec.instance.coords,
            chunk=int(self.config.runner.get("render_chunk", 262_144)),
            device=self.device,
        )
        metric_names = tuple(self.config.runner.get("metrics", ()))
        measured = evaluate_metrics(
            self.config.task,
            metric_names,
            spec.instance,
            prediction,
        )
        record = {
            "schema_version": 1,
            "experiment": self.config.name,
            "profile": self.config.profile,
            "paper": self.config.paper,
            "task": self.config.task,
            "dataset": self.config.dataset,
            "instance": spec.instance.name,
            "method": spec.method.name,
            "role": spec.method.role,
            "seed": spec.seed,
            "job_index": spec.index,
            "rank": self.rank,
            "world_size": self.world_size,
            "parameters": counts,
            "compression_factor": compression,
            "training": {
                **_plain(training_values),
                "total_steps": recipe.total_steps,
            },
            "metrics": measured,
            "metric_versions": metric_versions(),
            "config_source": str(self.config.source),
            "runner_config": _plain(self.config.runner),
            **_git_provenance(self.config.source),
            "elapsed_seconds": time.perf_counter() - started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(result_path, record)
        return record

    def run(self, instances: Sequence[TensorInstance]) -> list[dict]:
        jobs = shard_jobs(
            enumerate_jobs(self.config, instances),
            rank=self.rank,
            world_size=self.world_size,
        )
        records = [self.run_one(job) for job in jobs]
        atomic_write_json(
            self.output_dir / f"summary-rank-{self.rank}.json",
            summarize_records(records),
        )
        return records


def summarize_records(records: Sequence[Mapping]) -> dict:
    grouped: dict[str, dict[str, list[float]]] = {}
    for record in records:
        method_metrics = grouped.setdefault(record["method"], {})
        for name, value in record["metrics"].items():
            method_metrics.setdefault(name, []).append(float(value))
    summary = {}
    for method, metrics in grouped.items():
        summary[method] = {
            name: {
                "mean": sum(values) / len(values),
                "count": len(values),
            }
            for name, values in metrics.items()
        }
    return {"records": len(records), "methods": summary}


def collect_raw_records(output_dir) -> list[dict]:
    root = Path(output_dir) / "raw"
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("**/*.json"))
    ]


def write_global_summary(output_dir) -> dict:
    records = collect_raw_records(output_dir)
    summary = summarize_records(records)
    atomic_write_json(Path(output_dir) / "summary.json", summary)
    return summary


def paired_delta(
    records: Sequence[Mapping],
    *,
    baseline: str,
    candidate: str,
    metric: str,
) -> dict[str, float | int]:
    by_method = {}
    for record in records:
        key = (record["instance"], int(record["seed"]))
        if record["method"] in {baseline, candidate}:
            by_method.setdefault(record["method"], {})[key] = float(
                record["metrics"][metric]
            )
    baseline_values = by_method.get(baseline, {})
    candidate_values = by_method.get(candidate, {})
    keys = sorted(set(baseline_values) & set(candidate_values))
    if not keys:
        raise ValueError("no paired records found")
    deltas = [
        candidate_values[key] - baseline_values[key]
        for key in keys
    ]
    mean = sum(deltas) / len(deltas)
    if len(deltas) == 1:
        half_width = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in deltas) / (
            len(deltas) - 1
        )
        half_width = 1.96 * math.sqrt(variance / len(deltas))
    return {
        "count": len(deltas),
        "mean": mean,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }
