"""Reproducible real-data PEPS strong-scaling and full-job benchmark.

The benchmark intentionally uses the manifest-verified Kodak loader and the
same model/optimizer/minibatch/DDP components as the paper runner.  Multi-GPU
runs are strict: direct RCCL P2P must be requested explicitly and every run's
RCCL log is checked for P2P/IPC channel evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from apps.image.data import image_to_coords_targets, load_paper_kodak
from data.manifest import hash_file, load_manifest
from peps.distributed import (
    ddp_loss_scale,
    distributed_barrier,
    distributed_session,
    local_batch_slice,
    local_minibatch_indices,
    per_rank_batch_sizes,
    reduce_weighted_mean,
    seed_process,
    wrap_distributed,
)
from peps.metrics import metric_versions
from peps.train import (
    MinibatchStream,
    _paper_loss,
    make_paper_optimizer,
    paper_recipe_from_mapping,
)

from .config import load_experiment_config
from .multigpu import _pci_identifier, inspect_topology
from .runner import (
    RunSpec,
    TensorInstance,
    _assert_budget,
    _assert_compression,
    _build_model,
    _compression_factor,
    _parameter_counts,
    atomic_torch_save,
    atomic_write_json,
    enumerate_jobs,
    evaluate_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/paper/image_full.toml"
DEFAULT_AB_RECEIPT = ROOT / "results/multigpu/benchmark-p2p-ab.json"
REQUIRED_DIRECT_ENVIRONMENT = {
    "HSA_ENABLE_IPC_MODE_LEGACY": "0",
    "HSA_FORCE_FINE_GRAIN_PCIE": "1",
}
_DISTRIBUTED_ENVIRONMENT_NAMES = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)
_CHANNEL_ROUTE = re.compile(
    r"\bChannel\s+\d+/\d+\s*:\s*.*?->.*?\bvia\s+(\S+)"
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _maximum_rank_time(elapsed: float, device: torch.device) -> float:
    value = torch.tensor(elapsed, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def summary_statistics(
    values: Sequence[float | int | None],
    *,
    include_values: bool = False,
) -> dict[str, Any]:
    """Return robust dispersion statistics while preserving missing counts."""

    finite = sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    )
    result: dict[str, Any] = {
        "count": len(finite),
        "missing_count": len(values) - len(finite),
        "min": None,
        "median": None,
        "max": None,
        "mean": None,
        "sample_standard_deviation": None,
        "coefficient_of_variation_percent": None,
        "median_absolute_deviation": None,
        "iqr": None,
    }
    if include_values:
        result["values"] = finite
    if not finite:
        return result
    median = statistics.median(finite)
    mean = statistics.fmean(finite)
    standard_deviation = statistics.stdev(finite) if len(finite) > 1 else 0.0
    result.update(
        {
            "min": finite[0],
            "median": median,
            "max": finite[-1],
            "mean": mean,
            "sample_standard_deviation": standard_deviation,
            "coefficient_of_variation_percent": (
                100.0 * standard_deviation / abs(mean) if mean != 0 else None
            ),
            "median_absolute_deviation": statistics.median(
                abs(value - median) for value in finite
            ),
            "iqr": _percentile(finite, 0.75) - _percentile(finite, 0.25),
        }
    )
    return result


def _read_text(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return None


def _read_number(path: Path | None) -> float | None:
    value = _read_text(path)
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _active_dpm_mhz(text: str | None) -> float | None:
    """Parse the active ``*`` entry in an amdgpu ``pp_dpm_*`` file."""

    if text is None:
        return None
    for line in text.splitlines():
        if "*" not in line:
            continue
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[Mm][Hh][Zz]", line)
        if match is not None:
            return float(match.group(1))
    return None


def _clock_mhz(
    hwmon_path: Path | None,
    dpm_path: Path | None,
) -> float | None:
    hz = _read_number(hwmon_path)
    if hz is not None and hz > 0:
        return hz / 1_000_000.0
    return _active_dpm_mhz(_read_text(dpm_path))


def _find_labelled_files(
    directory: Path,
    *,
    prefix: str,
) -> dict[str, Path]:
    result = {}
    for label_path in sorted(directory.glob(f"{prefix}*_label")):
        label = _read_text(label_path)
        if not label:
            continue
        stem = label_path.name.removesuffix("_label")
        input_path = directory / f"{stem}_input"
        if input_path.is_file():
            result[label.strip().lower()] = input_path
    return result


def _telemetry_sources(torch_index: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(torch_index)
    pci = _pci_identifier(properties)
    device_path = (
        None
        if pci is None
        else Path("/sys/bus/pci/devices") / pci
    )
    hwmon = None
    if device_path is not None:
        candidates = sorted((device_path / "hwmon").glob("hwmon*"))
        hwmon = candidates[0] if candidates else None
    temperatures = (
        {} if hwmon is None else _find_labelled_files(hwmon, prefix="temp")
    )
    frequencies = (
        {} if hwmon is None else _find_labelled_files(hwmon, prefix="freq")
    )
    return {
        "torch_index": torch_index,
        "pci": pci,
        "name": str(properties.name),
        "architecture": getattr(properties, "gcnArchName", None),
        "device_path": device_path,
        "hwmon_path": hwmon,
        "paths": {
            "temperature_edge": temperatures.get("edge"),
            "temperature_junction": temperatures.get("junction"),
            "temperature_memory": (
                temperatures.get("mem") or temperatures.get("memory")
            ),
            "power_average": (
                None if hwmon is None else hwmon / "power1_average"
            ),
            "clock_core": frequencies.get("sclk"),
            "clock_memory": frequencies.get("mclk"),
            "clock_core_dpm": (
                None if device_path is None else device_path / "pp_dpm_sclk"
            ),
            "clock_memory_dpm": (
                None if device_path is None else device_path / "pp_dpm_mclk"
            ),
            "vram_used": (
                None
                if device_path is None
                else device_path / "mem_info_vram_used"
            ),
            "vram_total": (
                None
                if device_path is None
                else device_path / "mem_info_vram_total"
            ),
            "gpu_busy_percent": (
                None
                if device_path is None
                else device_path / "gpu_busy_percent"
            ),
        },
    }


def _sample_device(source: Mapping[str, Any]) -> dict[str, Any]:
    paths = source["paths"]
    return {
        "torch_index": source["torch_index"],
        "pci": source["pci"],
        "temperature_c": {
            "edge": _scaled_read(paths["temperature_edge"], 1_000.0),
            "junction": _scaled_read(paths["temperature_junction"], 1_000.0),
            "memory": _scaled_read(paths["temperature_memory"], 1_000.0),
        },
        "power_w": _scaled_read(paths["power_average"], 1_000_000.0),
        "clock_mhz": {
            "core": _clock_mhz(
                paths["clock_core"],
                paths["clock_core_dpm"],
            ),
            "memory": _clock_mhz(
                paths["clock_memory"],
                paths["clock_memory_dpm"],
            ),
        },
        "vram_bytes": {
            "used": _integer_read(paths["vram_used"]),
            "total": _integer_read(paths["vram_total"]),
        },
        "gpu_busy_percent": _read_number(paths["gpu_busy_percent"]),
    }


def _scaled_read(path: Path | None, divisor: float) -> float | None:
    value = _read_number(path)
    return None if value is None else value / divisor


def _integer_read(path: Path | None) -> int | None:
    value = _read_number(path)
    return None if value is None else int(value)


def _flatten_device_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "temperature_edge_c": sample["temperature_c"]["edge"],
        "temperature_junction_c": sample["temperature_c"]["junction"],
        "temperature_memory_c": sample["temperature_c"]["memory"],
        "power_w": sample["power_w"],
        "clock_core_mhz": sample["clock_mhz"]["core"],
        "clock_memory_mhz": sample["clock_mhz"]["memory"],
        "vram_used_bytes": sample["vram_bytes"]["used"],
        "vram_total_bytes": sample["vram_bytes"]["total"],
        "gpu_busy_percent": sample["gpu_busy_percent"],
    }


def _summarize_telemetry_samples(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_device: dict[str, dict[str, Any]] = {}
    for sample in samples:
        for device in sample["devices"]:
            key = str(device.get("pci") or f"torch:{device['torch_index']}")
            entry = by_device.setdefault(
                key,
                {
                    "pci": device.get("pci"),
                    "torch_indices": set(),
                    "values": {},
                    "sample_count": 0,
                },
            )
            entry["torch_indices"].add(int(device["torch_index"]))
            entry["sample_count"] += 1
            for metric, value in _flatten_device_sample(device).items():
                entry["values"].setdefault(metric, []).append(value)
    result = {}
    for key, entry in sorted(by_device.items()):
        result[key] = {
            "pci": entry["pci"],
            "torch_indices": sorted(entry["torch_indices"]),
            "sample_count": entry["sample_count"],
            "metrics": {
                metric: summary_statistics(values)
                for metric, values in sorted(entry["values"].items())
            },
        }
    return result


class SysfsTelemetrySampler:
    """Sample read-only amdgpu sysfs attributes in a background thread."""

    def __init__(self, device_count: int, interval_seconds: float) -> None:
        if device_count < 1:
            raise ValueError("telemetry device count must be positive")
        if interval_seconds <= 0:
            raise ValueError("telemetry interval must be positive")
        self.interval_seconds = float(interval_seconds)
        self.sources = [
            _telemetry_sources(index) for index in range(device_count)
        ]
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_monotonic: float | None = None
        self._started_at_utc: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("telemetry sampler was already started")
        self._started_monotonic = time.monotonic()
        self._started_at_utc = _utc_now()
        self._thread = threading.Thread(
            target=self._run,
            name="peps-sysfs-telemetry",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        assert self._started_monotonic is not None
        while not self._stop.is_set():
            sampled_at = time.monotonic()
            self.samples.append(
                {
                    "elapsed_seconds": sampled_at - self._started_monotonic,
                    "devices": [
                        _sample_device(source) for source in self.sources
                    ],
                }
            )
            elapsed = time.monotonic() - sampled_at
            self._stop.wait(max(0.0, self.interval_seconds - elapsed))

    def stop(self) -> dict[str, Any]:
        if self._thread is None:
            raise RuntimeError("telemetry sampler was not started")
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_seconds * 3.0))
        if self._thread.is_alive():
            raise RuntimeError("telemetry sampler did not stop")
        if not self.samples:
            assert self._started_monotonic is not None
            self.samples.append(
                {
                    "elapsed_seconds": (
                        time.monotonic() - self._started_monotonic
                    ),
                    "devices": [
                        _sample_device(source) for source in self.sources
                    ],
                }
            )
        tools = {
            "aligned_amd_smi": (
                str(ROOT / "/opt/rocm/bin/amd-smi")
                if Path("/opt/rocm/bin/amd-smi").is_file()
                else None
            ),
            "aligned_rocm_smi": (
                str(Path("/opt/rocm/bin/rocm-smi"))
                if Path("/opt/rocm/bin/rocm-smi").is_file()
                else None
            ),
            "path_amd_smi": shutil.which("amd-smi"),
            "path_rocm_smi": shutil.which("rocm-smi"),
        }
        return {
            "collector": "linux_amdgpu_sysfs_read_only",
            "started_at_utc": self._started_at_utc,
            "interval_seconds": self.interval_seconds,
            "sample_count": len(self.samples),
            "devices": [
                {
                    "torch_index": source["torch_index"],
                    "pci": source["pci"],
                    "name": source["name"],
                    "architecture": source["architecture"],
                    "hwmon_path": (
                        None
                        if source["hwmon_path"] is None
                        else str(source["hwmon_path"])
                    ),
                    "sources": {
                        name: (
                            None if path is None else str(path)
                        )
                        for name, path in source["paths"].items()
                    },
                }
                for source in self.sources
            ],
            "tool_discovery": tools,
            "tool_policy": (
                "No SMI command was invoked. /opt/rocm has no aligned SMI "
                "binary on this host and the PATH rocm-smi 5.7 utility is "
                "known to abort; read-only amdgpu sysfs was used."
            ),
            "samples": self.samples,
            "summary": _summarize_telemetry_samples(self.samples),
        }


def _wait_for_idle_gpus(
    *,
    device_count: int,
    timeout_seconds: float,
    maximum_busy_percent: float = 5.0,
    maximum_vram_used_bytes: int = 512 * 1024 * 1024,
) -> dict[str, Any]:
    """Require three consecutive quiescent read-only sysfs observations."""

    sources = [_telemetry_sources(index) for index in range(device_count)]
    deadline = time.monotonic() + timeout_seconds
    consecutive = 0
    observations = []
    while True:
        sample = {
            "observed_at_utc": _utc_now(),
            "devices": [_sample_device(source) for source in sources],
        }
        observations.append(sample)
        idle = all(
            (
                device["gpu_busy_percent"] is not None
                and device["gpu_busy_percent"] <= maximum_busy_percent
                and device["vram_bytes"]["used"] is not None
                and device["vram_bytes"]["used"]
                <= maximum_vram_used_bytes
            )
            for device in sample["devices"]
        )
        consecutive = consecutive + 1 if idle else 0
        if consecutive >= 3:
            return {
                "verified": True,
                "maximum_busy_percent": maximum_busy_percent,
                "maximum_vram_used_bytes": maximum_vram_used_bytes,
                "observations": observations[-3:],
            }
        if time.monotonic() >= deadline:
            latest = [
                {
                    "torch_index": device["torch_index"],
                    "pci": device["pci"],
                    "gpu_busy_percent": device["gpu_busy_percent"],
                    "vram_used_bytes": device["vram_bytes"]["used"],
                }
                for device in sample["devices"]
            ]
            raise RuntimeError(
                "GPUs are not idle; refusing a contaminated benchmark: "
                + json.dumps(latest, sort_keys=True)
            )
        time.sleep(0.25)


def _select_real_job(
    *,
    config_path: Path,
    instance_id: str,
    method_name: str,
    seed: int,
) -> tuple[Any, TensorInstance, RunSpec, dict[str, Any]]:
    config = load_experiment_config(config_path)
    loaded = load_paper_kodak(instance_ids=(instance_id,))[0]
    coords, targets, (height, width) = image_to_coords_targets(loaded.tensor)
    instance = TensorInstance(
        instance_id,
        coords,
        targets,
        shape=(height, width, 3),
        metadata={
            "num_signal_values": targets.numel(),
            "resolution_xy": [width, height],
            "color_space": loaded.color_space,
            "source_path": str(loaded.source_path),
            "source_sha256": hash_file(loaded.source_path, "sha256"),
        },
    )
    matches = [
        job
        for job in enumerate_jobs(config, (instance,))
        if job.method.name == method_name and job.seed == seed
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one real-data job for "
            f"{instance_id}/{method_name}/seed-{seed}, found {len(matches)}"
        )
    manifest_path = ROOT / "data/manifests/kodak.json"
    manifest = load_manifest("kodak")
    image_spec = next(
        item for item in manifest["images"] if item["id"] == instance_id
    )
    provenance = {
        "dataset_id": manifest["dataset_id"],
        "instance": instance_id,
        "source_path": str(loaded.source_path),
        "bytes": loaded.source_path.stat().st_size,
        "sha256": hash_file(loaded.source_path, "sha256"),
        "resolution_xy": [width, height],
        "color_space": loaded.color_space,
        "manifest_path": str(manifest_path),
        "manifest_sha256": hash_file(manifest_path, "sha256"),
        "source_url": image_spec["url"],
        "credit": _plain(image_spec["credit"]),
        "license": _plain(manifest["license"]),
        "loader": (
            "apps.image.data.load_paper_kodak -> "
            "apps.image.data.image_to_coords_targets"
        ),
    }
    return config, instance, matches[0], provenance


def _model_and_training(
    *,
    config: Any,
    instance: TensorInstance,
    job: RunSpec,
    context: Any,
) -> tuple[
    torch.nn.Module,
    torch.nn.Module,
    torch.optim.Optimizer,
    Any,
    Any,
    MinibatchStream,
    dict[str, Any],
    tuple[int, ...],
]:
    seed_process(job.seed)
    model, _ = _build_model(config, job.method, instance)
    counts = _parameter_counts(model)
    _assert_budget(job.method, counts)
    _assert_compression(job.method, _compression_factor(instance, counts["total"]))
    seed_process(job.seed, rank=context.rank, rank_offset=True)

    training_values = dict(config.training)
    training_values.update(job.method.training)
    training_values["seed"] = job.seed
    recipe = paper_recipe_from_mapping(training_values)
    effective_global_batch = min(recipe.batch_size, instance.coords.shape[0])
    local_sizes = per_rank_batch_sizes(
        effective_global_batch,
        context.world_size,
    )
    if min(local_sizes) < 1:
        raise ValueError("global batch must give each rank at least one sample")

    base_model = model.to(context.device)
    optimizer = make_paper_optimizer(base_model, recipe)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=recipe.total_steps,
        )
        if recipe.cosine
        else None
    )
    training_model = wrap_distributed(base_model, context)
    loss_function = _paper_loss(recipe)
    stream = MinibatchStream(
        instance.coords.shape[0],
        recipe.batch_size,
        recipe.seed,
    )
    workload = {
        "config_source": str(config.source),
        "config_sha256": hash_file(config.source, "sha256"),
        "experiment": config.name,
        "profile": config.profile,
        "dataset": config.dataset,
        "instance": instance.name,
        "method": job.method.name,
        "method_factory": job.method.factory,
        "method_kwargs": _plain(job.method.kwargs),
        "seed": job.seed,
        "loss": recipe.loss,
        "optimizer": "Adam",
        "model_lr": recipe.model_lr,
        "encoder_lr": recipe.encoder_lr,
        "cosine": recipe.cosine,
        "scheduler_horizon_steps": recipe.total_steps,
        "configured_total_steps": recipe.total_steps,
        "global_batch_size": effective_global_batch,
        "per_rank_batch_sizes": list(local_sizes),
        "parameter_counts": counts,
        "training_path": (
            "experiments.runner._build_model + peps.train.make_paper_optimizer "
            "+ peps.distributed.wrap_distributed"
        ),
    }
    return (
        base_model,
        training_model,
        optimizer,
        scheduler,
        loss_function,
        stream,
        workload,
        local_sizes,
    )


def _train_step(
    *,
    training_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    loss_function: Any,
    stream: MinibatchStream,
    coords: torch.Tensor,
    targets: torch.Tensor,
    context: Any,
) -> tuple[torch.Tensor, int]:
    global_indices = stream.next()
    local_indices = local_minibatch_indices(
        global_indices,
        context,
    ).to(device=context.device)
    local_count = local_indices.numel()
    prediction = training_model(coords.index_select(0, local_indices))
    local_loss = loss_function(
        prediction,
        targets.index_select(0, local_indices),
    )
    backward_loss = local_loss * ddp_loss_scale(
        local_count,
        global_indices.numel(),
        context.world_size,
    )
    optimizer.zero_grad(set_to_none=True)
    backward_loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return local_loss, local_count


@torch.no_grad()
def _full_image_mse(
    *,
    model: torch.nn.Module,
    coords: torch.Tensor,
    targets: torch.Tensor,
    context: Any,
    chunk: int = 131_072,
) -> float:
    was_training = model.training
    model.eval()
    portion = local_batch_slice(
        coords.shape[0],
        rank=context.rank,
        world_size=context.world_size,
    )
    squared_error = torch.zeros(
        (),
        dtype=torch.float64,
        device=context.device,
    )
    value_count = torch.zeros(
        (),
        dtype=torch.float64,
        device=context.device,
    )
    for start in range(portion.start, portion.stop, chunk):
        stop = min(start + chunk, portion.stop)
        prediction = model(coords[start:stop])
        difference = prediction - targets[start:stop]
        squared_error += difference.double().square().sum()
        value_count += difference.numel()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(squared_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(value_count, op=dist.ReduceOp.SUM)
    if was_training:
        model.train()
    return float((squared_error / value_count).item())


def _required_environment(rccl_p2p: str) -> dict[str, str]:
    if rccl_p2p != "on":
        raise ValueError("real-workload multi-GPU runs require --rccl-p2p on")
    missing = {
        name: {
            "expected": expected,
            "actual": os.environ.get(name),
        }
        for name, expected in REQUIRED_DIRECT_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if missing:
        raise RuntimeError(
            "strict RCCL direct P2P environment is missing or wrong: "
            + json.dumps(missing, sort_keys=True)
        )
    return {
        **REQUIRED_DIRECT_ENVIRONMENT,
        "NCCL_P2P_DISABLE": "0",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,P2P",
    }


def _validate_worker_environment(rccl_p2p: str, world_size: int) -> None:
    _required_environment(rccl_p2p)
    if os.environ.get("NCCL_P2P_DISABLE") != "0":
        raise RuntimeError("worker requires NCCL_P2P_DISABLE=0")
    if world_size > 1:
        cmdline = _read_text(Path("/proc/cmdline"))
        if cmdline is None or not any(
            token in {"iommu=pt", "amd_iommu=pt"}
            for token in cmdline.split()
        ):
            raise RuntimeError("strict RCCL run requires iommu=pt")


def run_real_worker(arguments: argparse.Namespace) -> dict[str, Any] | None:
    with distributed_session(backend=arguments.backend) as context:
        if context.device.type != "cuda":
            raise RuntimeError("real-workload benchmark requires a GPU")
        _validate_worker_environment(arguments.rccl_p2p, context.world_size)
        config, instance, job, provenance = _select_real_job(
            config_path=arguments.config,
            instance_id=arguments.instance,
            method_name=arguments.method,
            seed=arguments.seed,
        )
        (
            base_model,
            training_model,
            optimizer,
            scheduler,
            loss_function,
            stream,
            workload,
            local_sizes,
        ) = _model_and_training(
            config=config,
            instance=instance,
            job=job,
            context=context,
        )
        coords = instance.coords.to(context.device)
        targets = instance.targets.to(context.device)

        telemetry = (
            SysfsTelemetrySampler(
                context.world_size,
                arguments.telemetry_interval_seconds,
            )
            if context.is_main
            else None
        )
        loss_trace: list[dict[str, Any]] = []
        final_local_loss = None
        final_local_count = None

        if arguments.kind == "performance":
            total_prefix_steps = (
                arguments.warmup_steps + arguments.timed_steps
            )
            if total_prefix_steps > workload["configured_total_steps"]:
                raise ValueError(
                    "benchmark prefix exceeds configured paper steps"
                )
            for _ in range(arguments.warmup_steps):
                _train_step(
                    training_model=training_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_function=loss_function,
                    stream=stream,
                    coords=coords,
                    targets=targets,
                    context=context,
                )
            _synchronize(context.device)
            distributed_barrier(context)
            if telemetry is not None:
                telemetry.start()
            started = time.perf_counter()
            for _ in range(arguments.timed_steps):
                final_local_loss, final_local_count = _train_step(
                    training_model=training_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_function=loss_function,
                    stream=stream,
                    coords=coords,
                    targets=targets,
                    context=context,
                )
            _synchronize(context.device)
            elapsed = _maximum_rank_time(
                time.perf_counter() - started,
                context.device,
            )
            telemetry_record = (
                telemetry.stop() if telemetry is not None else None
            )
            completed_steps = total_prefix_steps
            measured_steps = arguments.timed_steps
        else:
            if (
                arguments.convergence_steps
                > workload["configured_total_steps"]
            ):
                raise ValueError(
                    "convergence prefix exceeds configured paper steps"
                )
            _synchronize(context.device)
            distributed_barrier(context)
            if telemetry is not None:
                telemetry.start()
            started = time.perf_counter()
            for step in range(1, arguments.convergence_steps + 1):
                final_local_loss, final_local_count = _train_step(
                    training_model=training_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_function=loss_function,
                    stream=stream,
                    coords=coords,
                    targets=targets,
                    context=context,
                )
                if (
                    step == 1
                    or step % arguments.convergence_log_every == 0
                    or step == arguments.convergence_steps
                ):
                    global_loss = reduce_weighted_mean(
                        final_local_loss,
                        final_local_count,
                        context,
                    )
                    if context.is_main:
                        loss_trace.append(
                            {
                                "step": step,
                                "global_minibatch_loss": global_loss,
                            }
                        )
            _synchronize(context.device)
            elapsed = _maximum_rank_time(
                time.perf_counter() - started,
                context.device,
            )
            telemetry_record = (
                telemetry.stop() if telemetry is not None else None
            )
            completed_steps = arguments.convergence_steps
            measured_steps = None

        assert final_local_loss is not None
        assert final_local_count is not None
        final_minibatch_loss = reduce_weighted_mean(
            final_local_loss,
            local_sizes[context.rank],
            context,
        )
        full_image_mse = _full_image_mse(
            model=base_model,
            coords=coords,
            targets=targets,
            context=context,
        )
        record = None
        if context.is_main:
            record = {
                "schema": "peps.real_workload_run",
                "schema_version": 1,
                "created_at_utc": _utc_now(),
                "kind": arguments.kind,
                "data": provenance,
                "workload": workload,
                "parallelism": {
                    "mode": (
                        "ddp_single_job"
                        if context.world_size > 1
                        else "single_gpu"
                    ),
                    "world_size": context.world_size,
                    "backend_api": context.backend,
                    "backend_runtime": (
                        "RCCL"
                        if context.backend == "nccl"
                        and torch.version.hip is not None
                        else context.backend
                    ),
                    "rccl_p2p_requested": arguments.rccl_p2p,
                    "rccl_p2p_disabled": (
                        os.environ.get("NCCL_P2P_DISABLE") == "1"
                    ),
                },
                "prefix": {
                    "completed_steps": completed_steps,
                    "warmup_steps": (
                        arguments.warmup_steps
                        if arguments.kind == "performance"
                        else 0
                    ),
                    "timed_steps": measured_steps,
                    "elapsed_seconds": elapsed,
                    "steps_per_second": (
                        None
                        if measured_steps is None
                        else measured_steps / elapsed
                    ),
                    "samples_per_second": (
                        None
                        if measured_steps is None
                        else (
                            measured_steps
                            * workload["global_batch_size"]
                            / elapsed
                        )
                    ),
                    "timing": (
                        "slowest-rank synchronized steady-state prefix; data "
                        "load, model construction, warmup, final loss, and "
                        "full-image evaluation excluded"
                        if arguments.kind == "performance"
                        else "slowest-rank deterministic convergence prefix"
                    ),
                },
                "loss": {
                    "final_global_minibatch": final_minibatch_loss,
                    "full_image_mse": full_image_mse,
                    "full_image_psnr_db": (
                        float("inf")
                        if full_image_mse == 0
                        else -10.0 * math.log10(full_image_mse)
                    ),
                    "trace": loss_trace,
                },
                "telemetry": telemetry_record,
                "environment": {
                    name: os.environ.get(name)
                    for name in (
                        *REQUIRED_DIRECT_ENVIRONMENT,
                        "NCCL_P2P_DISABLE",
                        "NCCL_DEBUG",
                        "NCCL_DEBUG_SUBSYS",
                    )
                },
                "torch_version": torch.__version__,
                "rocm_version": torch.version.hip,
            }
            atomic_write_json(arguments.output, record)
        distributed_barrier(context)
        return record


def validate_direct_transport_log(
    log_text: str,
    *,
    world_size: int,
) -> dict[str, Any]:
    """Require direct P2P/IPC route evidence and reject routed fallbacks."""

    if world_size == 1:
        return {
            "requested": "on",
            "effective": "not_applicable_single_gpu",
            "verified": True,
            "p2p_ipc_route_count": 0,
            "fallback_route_count": 0,
            "evidence": [],
        }
    route_lines: list[tuple[str, str]] = []
    for line in log_text.splitlines():
        match = _CHANNEL_ROUTE.search(line)
        if match is not None:
            route_lines.append((match.group(1), line.strip()))
    p2p_lines = [
        line for route, line in route_lines if route == "P2P/IPC"
    ]
    fallback_lines = [
        line for route, line in route_lines if route != "P2P/IPC"
    ]
    connected = "Connected all rings" in log_text
    disabled_marker = bool(
        re.search(
            r"(P2P (?:is )?disabled|NCCL_P2P_DISABLE\s*=\s*1)",
            log_text,
            flags=re.IGNORECASE,
        )
    )
    if (
        len(p2p_lines) < world_size
        or fallback_lines
        or not connected
        or disabled_marker
    ):
        raise RuntimeError(
            "strict RCCL direct-P2P verification failed: "
            f"p2p_routes={len(p2p_lines)}, "
            f"fallback_routes={len(fallback_lines)}, "
            f"connected_all_rings={connected}, "
            f"disabled_marker={disabled_marker}"
        )
    return {
        "requested": "on",
        "effective": "peer_ipc",
        "verified": True,
        "p2p_ipc_route_count": len(p2p_lines),
        "fallback_route_count": 0,
        "connected_all_rings": connected,
        "evidence": p2p_lines[: min(8, len(p2p_lines))],
    }


def _render_command(
    command: Sequence[str],
    environment: Mapping[str, str],
) -> str:
    prefix = [
        f"{name}={shlex.quote(value)}"
        for name, value in environment.items()
    ]
    return " ".join(
        [*prefix, *(shlex.quote(str(part)) for part in command)]
    )


def _clean_distributed_environment(
    overrides: Mapping[str, str],
) -> dict[str, str]:
    environment = os.environ.copy()
    for name in _DISTRIBUTED_ENVIRONMENT_NAMES:
        environment.pop(name, None)
    environment.update(overrides)
    return environment


def _run_logged(
    *,
    command: Sequence[str],
    environment: Mapping[str, str],
    log_path: Path,
    timeout_seconds: int,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {_render_command(command, environment)}\n")
        log.flush()
        try:
            completed = subprocess.run(
                list(command),
                cwd=ROOT,
                env=_clean_distributed_environment(environment),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            log.write(f"\nbenchmark timed out after {timeout_seconds}s\n")
            raise
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed with exit {completed.returncode}; see {log_path}"
        )


def _worker_command(
    *,
    arguments: argparse.Namespace,
    kind: str,
    world_size: int,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        "-m",
        "experiments.real_workload",
        "worker",
        "--kind",
        kind,
        "--config",
        str(arguments.config),
        "--instance",
        arguments.instance,
        "--method",
        arguments.method,
        "--seed",
        str(arguments.seed),
        "--output",
        str(output),
        "--rccl-p2p",
        arguments.rccl_p2p,
        "--telemetry-interval-seconds",
        str(arguments.telemetry_interval_seconds),
    ]
    if kind == "performance":
        command.extend(
            [
                "--warmup-steps",
                str(arguments.warmup_steps),
                "--timed-steps",
                str(arguments.timed_steps),
            ]
        )
    else:
        command.extend(
            [
                "--convergence-steps",
                str(arguments.convergence_steps),
                "--convergence-log-every",
                str(arguments.convergence_log_every),
            ]
        )
    return command


def _run_one_short(
    *,
    arguments: argparse.Namespace,
    kind: str,
    world_size: int,
    output_path: Path,
    log_path: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    command = _worker_command(
        arguments=arguments,
        kind=kind,
        world_size=world_size,
        output=output_path,
    )
    if output_path.is_file() and log_path.is_file():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            expected_completed_steps = (
                arguments.warmup_steps + arguments.timed_steps
                if kind == "performance"
                else arguments.convergence_steps
            )
            identity_matches = (
                existing.get("schema") == "peps.real_workload_run"
                and existing.get("kind") == kind
                and existing.get("data", {}).get("instance")
                == arguments.instance
                and existing.get("workload", {}).get("method")
                == arguments.method
                and existing.get("workload", {}).get("seed")
                == arguments.seed
                and existing.get("workload", {}).get("config_sha256")
                == hash_file(arguments.config, "sha256")
                and existing.get("parallelism", {}).get("world_size")
                == world_size
                and existing.get("prefix", {}).get("completed_steps")
                == expected_completed_steps
                and existing.get("prefix", {}).get("timed_steps")
                == (
                    arguments.timed_steps
                    if kind == "performance"
                    else None
                )
                and existing.get("prefix", {}).get("warmup_steps")
                == (
                    arguments.warmup_steps
                    if kind == "performance"
                    else 0
                )
                and existing.get("telemetry", {}).get("interval_seconds")
                == arguments.telemetry_interval_seconds
            )
            if identity_matches:
                transport = validate_direct_transport_log(
                    log_path.read_text(encoding="utf-8"),
                    world_size=world_size,
                )
                existing["command"] = _render_command(command, environment)
                existing["log"] = str(log_path)
                existing["transport_verification"] = transport
                atomic_write_json(output_path, existing)
                return existing
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            RuntimeError,
        ):
            pass
    idle_timeout = float(getattr(arguments, "idle_timeout_seconds", 30.0))
    pre_run_idle = _wait_for_idle_gpus(
        device_count=world_size,
        timeout_seconds=idle_timeout,
    )
    _run_logged(
        command=command,
        environment=environment,
        log_path=log_path,
        timeout_seconds=arguments.timeout_seconds,
    )
    post_run_idle = _wait_for_idle_gpus(
        device_count=world_size,
        timeout_seconds=idle_timeout,
    )
    log_text = log_path.read_text(encoding="utf-8")
    transport = validate_direct_transport_log(
        log_text,
        world_size=world_size,
    )
    record = json.loads(output_path.read_text(encoding="utf-8"))
    record["command"] = _render_command(command, environment)
    record["log"] = str(log_path)
    record["transport_verification"] = transport
    record["gpu_quiescence"] = {
        "pre_run": pre_run_idle,
        "post_run": post_run_idle,
    }
    atomic_write_json(output_path, record)
    return record


def _aggregate_telemetry(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    samples = []
    collectors = set()
    for record in records:
        telemetry = record.get("telemetry")
        if not isinstance(telemetry, Mapping):
            continue
        collectors.add(str(telemetry.get("collector")))
        samples.extend(telemetry.get("samples", ()))
    return {
        "collectors": sorted(collectors),
        "sample_count": len(samples),
        "devices": _summarize_telemetry_samples(samples),
    }


def _aggregate_suite(
    *,
    arguments: argparse.Namespace,
    performance: Mapping[int, Sequence[Mapping[str, Any]]],
    convergence: Mapping[int, Mapping[str, Any]],
    commands: Sequence[str],
    result_paths: Sequence[Path],
) -> dict[str, Any]:
    first = performance[min(performance)][0]
    performance_summary = {}
    for gpu_count, records in sorted(performance.items()):
        performance_summary[str(gpu_count)] = {
            "repetitions": len(records),
            "steps_per_second": summary_statistics(
                [record["prefix"]["steps_per_second"] for record in records],
                include_values=True,
            ),
            "samples_per_second": summary_statistics(
                [
                    record["prefix"]["samples_per_second"]
                    for record in records
                ],
                include_values=True,
            ),
            "elapsed_seconds": summary_statistics(
                [record["prefix"]["elapsed_seconds"] for record in records],
                include_values=True,
            ),
            "final_global_minibatch_loss": summary_statistics(
                [
                    record["loss"]["final_global_minibatch"]
                    for record in records
                ],
                include_values=True,
            ),
            "full_image_mse": summary_statistics(
                [record["loss"]["full_image_mse"] for record in records],
                include_values=True,
            ),
            "full_image_psnr_db": summary_statistics(
                [record["loss"]["full_image_psnr_db"] for record in records],
                include_values=True,
            ),
            "telemetry": _aggregate_telemetry(records),
            "transport": [
                record["transport_verification"] for record in records
            ],
        }
    baseline = performance_summary["1"]["steps_per_second"]["median"]
    scaling = {}
    for gpu_count in sorted(performance):
        median_rate = performance_summary[str(gpu_count)][
            "steps_per_second"
        ]["median"]
        speedup = median_rate / baseline
        scaling[str(gpu_count)] = {
            "median_steps_per_second": median_rate,
            "speedup_vs_1gpu": speedup,
            "parallel_efficiency": speedup / gpu_count,
        }

    baseline_loss = performance_summary["1"]["full_image_mse"]["median"]
    loss_consistency = {}
    for gpu_count in sorted(performance):
        median_loss = performance_summary[str(gpu_count)][
            "full_image_mse"
        ]["median"]
        absolute = abs(median_loss - baseline_loss)
        loss_consistency[str(gpu_count)] = {
            "median_full_image_mse": median_loss,
            "absolute_delta_vs_1gpu": absolute,
            "relative_delta_vs_1gpu": (
                absolute / abs(baseline_loss)
                if baseline_loss != 0
                else None
            ),
        }

    traces = {
        str(gpu_count): record["loss"]["trace"]
        for gpu_count, record in sorted(convergence.items())
    }
    trace_maps = {
        gpu_count: {
            int(item["step"]): float(item["global_minibatch_loss"])
            for item in record["loss"]["trace"]
        }
        for gpu_count, record in convergence.items()
    }
    common_steps = sorted(
        set.intersection(*(set(values) for values in trace_maps.values()))
    )
    convergence_comparison = []
    for step in common_steps:
        one_gpu_loss = trace_maps[1][step]
        values = {
            str(gpu_count): trace_maps[gpu_count][step]
            for gpu_count in sorted(trace_maps)
        }
        deltas = {
            str(gpu_count): abs(value - one_gpu_loss)
            for gpu_count, value in (
                (gpu_count, trace_maps[gpu_count][step])
                for gpu_count in sorted(trace_maps)
            )
        }
        convergence_comparison.append(
            {
                "step": step,
                "global_minibatch_loss": values,
                "absolute_delta_vs_1gpu": deltas,
                "max_absolute_delta_vs_1gpu": max(deltas.values()),
                "max_relative_delta_vs_1gpu": (
                    max(deltas.values()) / abs(one_gpu_loss)
                    if one_gpu_loss != 0
                    else None
                ),
            }
        )

    ab_path = arguments.ab_receipt.resolve()
    return {
        "schema": "peps.real_workload_strong_scaling",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "complete",
        "data": first["data"],
        "workload": first["workload"],
        "protocol": {
            "comparison": "fixed-global-batch strong scaling of one PEPS job",
            "gpu_counts": list(sorted(performance)),
            "repetitions": arguments.repetitions,
            "round_robin_order": list(sorted(performance)),
            "warmup_steps": arguments.warmup_steps,
            "timed_steps": arguments.timed_steps,
            "convergence_steps": arguments.convergence_steps,
            "convergence_log_every": arguments.convergence_log_every,
            "telemetry_interval_seconds": (
                arguments.telemetry_interval_seconds
            ),
            "idle_precondition": {
                "timeout_seconds": arguments.idle_timeout_seconds,
                "maximum_busy_percent": 5.0,
                "maximum_vram_used_bytes": 512 * 1024 * 1024,
                "required_consecutive_observations": 3,
            },
            "rccl_p2p": arguments.rccl_p2p,
            "required_environment": {
                **REQUIRED_DIRECT_ENVIRONMENT,
                "NCCL_P2P_DISABLE": "0",
            },
        },
        "authoritative_p2p_ab_receipt": {
            "path": str(ab_path),
            "sha256": hash_file(ab_path, "sha256"),
        },
        "system": inspect_topology(),
        "performance": performance_summary,
        "strong_scaling": scaling,
        "loss_consistency": loss_consistency,
        "deterministic_convergence": {
            "traces": traces,
            "comparison": convergence_comparison,
        },
        "commands": list(commands),
        "result_files": [str(path) for path in result_paths],
        "limitations": [
            "Temperature, power, clock, utilization, and VRAM telemetry come "
            "from read-only amdgpu sysfs. Missing driver attributes remain "
            "null and are counted; no value is synthesized.",
            "The five throughput repetitions are deterministic prefixes from "
            "fresh initialization, not five complete 120,000-step jobs.",
            "The separate convergence prefixes contain metric-reduction "
            "synchronizations and are not used as throughput measurements.",
        ],
    }


def run_suite(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.repetitions < 5:
        raise ValueError("the final real-workload benchmark requires >=5 repetitions")
    gpu_counts = tuple(sorted(set(arguments.gpu_counts)))
    if gpu_counts != arguments.gpu_counts:
        raise ValueError("GPU counts must be unique and sorted")
    if gpu_counts != (1, 2, 4):
        raise ValueError("the final strong-scaling protocol requires 1,2,4 GPUs")
    if arguments.warmup_steps < 1 or arguments.timed_steps < 1:
        raise ValueError("warmup and timed steps must be positive")
    if arguments.convergence_steps < 1:
        raise ValueError("convergence steps must be positive")
    environment = _required_environment(arguments.rccl_p2p)
    topology = inspect_topology()
    if topology["device_count"] < max(gpu_counts):
        raise RuntimeError(
            f"requested {max(gpu_counts)} GPUs, found {topology['device_count']}"
        )
    if not topology["iommu_passthrough_enabled"]:
        raise RuntimeError("iommu=pt is not active")
    if not arguments.ab_receipt.is_file():
        raise FileNotFoundError(arguments.ab_receipt)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.work_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = arguments.work_dir / "runs"
    logs_dir = arguments.work_dir / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    performance: dict[int, list[dict[str, Any]]] = {
        gpu_count: [] for gpu_count in gpu_counts
    }
    convergence: dict[int, dict[str, Any]] = {}
    commands = []
    result_paths = []
    for repetition in range(1, arguments.repetitions + 1):
        for gpu_count in gpu_counts:
            output_path = (
                runs_dir
                / f"performance-{gpu_count}gpu-r{repetition}.json"
            )
            log_path = (
                logs_dir
                / f"performance-{gpu_count}gpu-r{repetition}.log"
            )
            record = _run_one_short(
                arguments=arguments,
                kind="performance",
                world_size=gpu_count,
                output_path=output_path,
                log_path=log_path,
                environment=environment,
            )
            performance[gpu_count].append(record)
            commands.append(record["command"])
            result_paths.extend((output_path, log_path))

    for gpu_count in gpu_counts:
        output_path = runs_dir / f"convergence-{gpu_count}gpu.json"
        log_path = logs_dir / f"convergence-{gpu_count}gpu.log"
        record = _run_one_short(
            arguments=arguments,
            kind="convergence",
            world_size=gpu_count,
            output_path=output_path,
            log_path=log_path,
            environment=environment,
        )
        convergence[gpu_count] = record
        commands.append(record["command"])
        result_paths.extend((output_path, log_path))

    result = _aggregate_suite(
        arguments=arguments,
        performance=performance,
        convergence=convergence,
        commands=commands,
        result_paths=result_paths,
    )
    result["commands"].insert(
        0,
        _render_command(
            [sys.executable, "-m", "experiments.real_workload", *sys.argv[1:]],
            REQUIRED_DIRECT_ENVIRONMENT,
        ),
    )
    atomic_write_json(arguments.output, result)
    return result


def _prepare_tensor_input(
    *,
    config_path: Path,
    instance_id: str,
    method_name: str,
    seed: int,
    output: Path,
) -> tuple[dict[str, Any], TensorInstance]:
    _, instance, _, provenance = _select_real_job(
        config_path=config_path,
        instance_id=instance_id,
        method_name=method_name,
        seed=seed,
    )
    payload = {
        "schema": "peps.real_image_tensor_input",
        "schema_version": 1,
        "data": provenance,
        "instances": [
            {
                "name": instance.name,
                "coords": instance.coords,
                "targets": instance.targets,
                "shape": instance.shape,
                "metadata": dict(instance.metadata),
            }
        ],
    }
    atomic_torch_save(output, payload)
    return provenance, instance


def _metric_preflight(
    *,
    config_path: Path,
    instance: TensorInstance,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    names = tuple(config.runner.get("metrics", ()))
    measured = evaluate_metrics(
        config.task,
        names,
        instance,
        instance.targets.clone(),
    )
    normalized = {
        name: (
            float(value)
            if math.isfinite(float(value))
            else None
        )
        for name, value in measured.items()
    }
    return {
        "status": "passed",
        "metrics": list(names),
        "self_comparison": normalized,
        "note": (
            "Non-finite perfect-match metrics (PSNR +inf) are represented "
            "as null in JSON."
        ),
        "versions": metric_versions(),
    }


def run_full_job(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.gpus != 4:
        raise ValueError("the representative full paper job requires four GPUs")
    environment = _required_environment(arguments.rccl_p2p)
    topology = inspect_topology()
    if topology["device_count"] < arguments.gpus:
        raise RuntimeError(
            f"requested {arguments.gpus} GPUs, found {topology['device_count']}"
        )
    if not topology["iommu_passthrough_enabled"]:
        raise RuntimeError("iommu=pt is not active")
    config = load_experiment_config(arguments.config)
    training_steps = int(config.training.get("steps", 0))
    if training_steps < 1:
        raise ValueError("selected image config has no positive step budget")

    arguments.output_dir.parent.mkdir(parents=True, exist_ok=True)
    arguments.work_dir.mkdir(parents=True, exist_ok=True)
    input_path = arguments.work_dir / f"{arguments.instance}.pt"
    log_path = arguments.work_dir / "full-paper-job.log"
    receipt_path = arguments.receipt
    provenance, instance = _prepare_tensor_input(
        config_path=arguments.config,
        instance_id=arguments.instance,
        method_name=arguments.method,
        seed=arguments.seed,
        output=input_path,
    )
    metric_preflight = _metric_preflight(
        config_path=arguments.config,
        instance=instance,
    )
    pre_job_idle = _wait_for_idle_gpus(
        device_count=arguments.gpus,
        timeout_seconds=arguments.idle_timeout_seconds,
    )

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={arguments.gpus}",
        "-m",
        "experiments.ddp",
        "--config",
        str(arguments.config),
        "--input",
        str(input_path),
        "--output",
        str(arguments.output_dir),
        "--instance",
        arguments.instance,
        "--method",
        arguments.method,
        "--seed",
        str(arguments.seed),
    ]
    if arguments.force:
        command.append("--force")

    sampler = SysfsTelemetrySampler(
        arguments.gpus,
        arguments.telemetry_interval_seconds,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sampler.start()
    wall_started = time.perf_counter()
    try:
        _run_logged(
            command=command,
            environment=environment,
            log_path=log_path,
            timeout_seconds=arguments.timeout_seconds,
        )
    finally:
        wall_elapsed = time.perf_counter() - wall_started
        telemetry = sampler.stop()
    telemetry_path = arguments.work_dir / "full-paper-job-telemetry.json"
    atomic_write_json(telemetry_path, telemetry)
    try:
        post_job_idle = _wait_for_idle_gpus(
            device_count=arguments.gpus,
            timeout_seconds=arguments.idle_timeout_seconds,
        )
    except RuntimeError as exc:
        post_job_idle = {
            "verified": False,
            "reason": str(exc),
            "note": (
                "Post-run sysfs activity can lag process exit. This does not "
                "invalidate a run whose precondition was idle; process-table "
                "cleanup is verified separately after the wrapper exits."
            ),
        }
    log_text = log_path.read_text(encoding="utf-8")
    transport = validate_direct_transport_log(
        log_text,
        world_size=arguments.gpus,
    )
    result_path = arguments.output_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    completed_steps = int(result["training"]["total_steps"])
    resumed_step = int(result["training"]["resumed_step"])
    if completed_steps != training_steps:
        raise RuntimeError(
            f"full job completed {completed_steps} steps, expected {training_steps}"
        )
    if resumed_step != 0:
        raise RuntimeError(
            "full-job receipt requires an uninterrupted run from step zero; "
            f"this run resumed at step {resumed_step}"
        )
    receipt = {
        "schema": "peps.real_workload_full_paper_job",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "complete",
        "data": provenance,
        "workload": {
            "config_source": str(config.source),
            "config_sha256": hash_file(config.source, "sha256"),
            "experiment": config.name,
            "instance": arguments.instance,
            "method": arguments.method,
            "seed": arguments.seed,
            "global_batch_size": result["parallelism"]["global_batch_size"],
            "steps": completed_steps,
            "loss": result["training"]["loss"],
            "model_lr": result["training"]["model_lr"],
            "encoder_lr": result["training"]["encoder_lr"],
            "cosine": result["training"]["cosine"],
            "parameter_counts": result["parameters"],
        },
        "parallelism": result["parallelism"],
        "runtime": {
            "training_elapsed_seconds": result["training"]["elapsed_seconds"],
            "process_wall_seconds": wall_elapsed,
            "samples_per_second": result["training"]["samples_per_second"],
        },
        "final": {
            "loss_log_last": (
                result["loss_log"][-1] if result["loss_log"] else None
            ),
            "metrics": result["metrics"],
        },
        "metric_preflight": metric_preflight,
        "telemetry": telemetry,
        "gpu_quiescence": {
            "pre_run": pre_job_idle,
            "post_run": post_job_idle,
        },
        "transport_verification": transport,
        "environment": {
            **REQUIRED_DIRECT_ENVIRONMENT,
            "NCCL_P2P_DISABLE": "0",
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,P2P",
        },
        "system": topology,
        "command": _render_command(command, environment),
        "files": {
            "tensor_input": str(input_path),
            "tensor_input_sha256": hash_file(input_path, "sha256"),
            "telemetry": str(telemetry_path),
            "training_result": str(result_path),
            "checkpoint": result["checkpoint"],
            "log": str(log_path),
            "receipt": str(receipt_path),
        },
        "training_result": result,
    }
    atomic_write_json(receipt_path, receipt)
    return receipt


def _positive_csv(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def _add_job_identity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--instance", default="kodim01")
    parser.add_argument("--method", default="G-PEPS")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rccl-p2p",
        choices=("on",),
        required=True,
        help="require direct RCCL P2P; no fallback mode exists",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker")
    _add_job_identity(worker)
    worker.add_argument(
        "--kind",
        choices=("performance", "convergence"),
        required=True,
    )
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--warmup-steps", type=int, default=100)
    worker.add_argument("--timed-steps", type=int, default=500)
    worker.add_argument("--convergence-steps", type=int, default=600)
    worker.add_argument("--convergence-log-every", type=int, default=100)
    worker.add_argument(
        "--telemetry-interval-seconds",
        type=float,
        default=0.2,
    )
    worker.add_argument("--backend", choices=("nccl", "gloo"))

    suite = subparsers.add_parser("suite")
    _add_job_identity(suite)
    suite.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/multigpu/benchmark-real-workload.json",
    )
    suite.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "results/multigpu/real-workload",
    )
    suite.add_argument(
        "--ab-receipt",
        type=Path,
        default=DEFAULT_AB_RECEIPT,
    )
    suite.add_argument(
        "--gpu-counts",
        type=_positive_csv,
        default=(1, 2, 4),
    )
    suite.add_argument("--repetitions", type=int, default=5)
    suite.add_argument("--warmup-steps", type=int, default=100)
    suite.add_argument("--timed-steps", type=int, default=500)
    suite.add_argument("--convergence-steps", type=int, default=600)
    suite.add_argument("--convergence-log-every", type=int, default=100)
    suite.add_argument(
        "--telemetry-interval-seconds",
        type=float,
        default=0.2,
    )
    suite.add_argument(
        "--idle-timeout-seconds",
        type=float,
        default=30.0,
    )
    suite.add_argument("--timeout-seconds", type=int, default=600)

    full = subparsers.add_parser("full-job")
    _add_job_identity(full)
    full.add_argument("--gpus", type=int, default=4)
    full.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/multigpu/real-workload/full-kodim01-g-peps"
        ),
    )
    full.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "results/multigpu/real-workload/full-job",
    )
    full.add_argument(
        "--receipt",
        type=Path,
        default=(
            ROOT
            / "results/multigpu/real-workload/full-paper-job.json"
        ),
    )
    full.add_argument(
        "--telemetry-interval-seconds",
        type=float,
        default=0.5,
    )
    full.add_argument(
        "--idle-timeout-seconds",
        type=float,
        default=30.0,
    )
    full.add_argument("--timeout-seconds", type=int, default=7_200)
    full.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "worker":
        record = run_real_worker(arguments)
        if record is not None:
            print(
                json.dumps(
                    {
                        "status": "complete",
                        "output": str(arguments.output),
                        "kind": arguments.kind,
                        "world_size": record["parallelism"]["world_size"],
                    },
                    sort_keys=True,
                )
            )
        return 0
    if arguments.command == "suite":
        result = run_suite(arguments)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "output": str(arguments.output),
                    "strong_scaling": result["strong_scaling"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    receipt = run_full_job(arguments)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(arguments.receipt),
                "runtime": receipt["runtime"],
                "final": receipt["final"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
