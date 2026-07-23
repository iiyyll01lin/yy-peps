"""Topology, RCCL collective, and real PEPS DDP performance validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from apps.image.build import build_paper_image
from peps.distributed import (
    ddp_loss_scale,
    distributed_barrier,
    distributed_session,
    local_minibatch_indices,
    per_rank_batch_sizes,
    reduce_weighted_mean,
    seed_process,
    wrap_distributed,
)
from peps.train import MinibatchStream, PaperTrainConfig, make_paper_optimizer

from .runner import atomic_write_json


def _pci_identifier(properties) -> str | None:
    domain = getattr(properties, "pci_domain_id", None)
    bus = getattr(properties, "pci_bus_id", None)
    device = getattr(properties, "pci_device_id", None)
    if None in {domain, bus, device}:
        return None
    if isinstance(bus, str) and ":" in bus:
        return bus
    return f"{int(domain):04x}:{int(bus):02x}:{int(device):02x}.0"


def _read_sysfs_topology(pci_identifier: str | None) -> dict[str, object]:
    if pci_identifier is None:
        return {}
    directory = Path("/sys/bus/pci/devices") / pci_identifier
    fields = {
        "numa_node": "numa_node",
        "current_link_speed": "current_link_speed",
        "current_link_width": "current_link_width",
        "max_link_speed": "max_link_speed",
        "max_link_width": "max_link_width",
    }
    values: dict[str, object] = {}
    for output_name, filename in fields.items():
        path = directory / filename
        try:
            value = path.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError):
            continue
        if output_name in {"numa_node", "current_link_width", "max_link_width"}:
            try:
                values[output_name] = int(value)
                continue
            except ValueError:
                pass
        values[output_name] = value
    return values


def inspect_topology() -> dict[str, object]:
    """Inspect only information available through PyTorch and Linux sysfs."""

    available = bool(torch.cuda.is_available())
    count = torch.cuda.device_count() if available else 0
    devices = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        pci_identifier = _pci_identifier(properties)
        devices.append(
            {
                "index": index,
                "name": str(properties.name),
                "architecture": getattr(properties, "gcnArchName", None),
                "total_memory_bytes": int(properties.total_memory),
                "multiprocessor_count": int(properties.multi_processor_count),
                "pci": pci_identifier,
                "uuid": str(getattr(properties, "uuid", "")) or None,
                "linux_pci": _read_sysfs_topology(pci_identifier),
            }
        )
    p2p = [
        [
            (
                True
                if source == destination
                else bool(
                    torch.cuda.can_device_access_peer(source, destination)
                )
            )
            for destination in range(count)
        ]
        for source in range(count)
    ]
    try:
        kernel_command_line = Path("/proc/cmdline").read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        kernel_command_line = None
    return {
        "torch_version": torch.__version__,
        "rocm_version": torch.version.hip,
        "cuda_api_available": available,
        "device_count": count,
        "devices": devices,
        "p2p_access_matrix": p2p,
        "kernel_command_line": kernel_command_line,
        "iommu_passthrough_enabled": bool(
            kernel_command_line
            and any(
                token in {"iommu=pt", "amd_iommu=pt"}
                for token in kernel_command_line.split()
            )
        ),
        "topology_scope": (
            "PyTorch P2P capability plus Linux PCIe link/NUMA attributes. "
            "PyTorch does not expose whether a peer route is XGMI."
        ),
    }


def _sync_devices(*indices: int) -> None:
    for index in sorted(set(indices)):
        torch.cuda.synchronize(index)


def measure_pairwise_copy_bandwidth(
    *,
    tensor_mib: int = 128,
    warmup: int = 3,
    iterations: int = 10,
) -> dict[str, object]:
    """Measure directed peer copies with host timing around synchronized work."""

    if not torch.cuda.is_available():
        return {"available": False, "reason": "PyTorch reports no GPU"}
    if tensor_mib < 1 or warmup < 0 or iterations < 1:
        raise ValueError("invalid pairwise copy benchmark settings")

    count = torch.cuda.device_count()
    elements = tensor_mib * 1024 * 1024 // torch.tensor([], dtype=torch.float32).element_size()
    message_bytes = elements * torch.tensor([], dtype=torch.float32).element_size()
    measurements = []
    matrix: list[list[float | None]] = [
        [None for _ in range(count)] for _ in range(count)
    ]
    for source_index in range(count):
        for destination_index in range(count):
            if source_index == destination_index:
                continue
            if not torch.cuda.can_device_access_peer(
                source_index,
                destination_index,
            ):
                measurements.append(
                    {
                        "source": source_index,
                        "destination": destination_index,
                        "status": "unavailable",
                        "reason": "torch.cuda.can_device_access_peer is false",
                    }
                )
                continue
            try:
                with torch.cuda.device(source_index):
                    source = torch.empty(
                        elements,
                        device=f"cuda:{source_index}",
                        dtype=torch.float32,
                    )
                    source.fill_(1.0)
                with torch.cuda.device(destination_index):
                    destination = torch.empty_like(
                        source,
                        device=f"cuda:{destination_index}",
                    )
                    for _ in range(warmup):
                        destination.copy_(source, non_blocking=True)
                _sync_devices(source_index, destination_index)
                with torch.cuda.device(destination_index):
                    started = time.perf_counter()
                    for _ in range(iterations):
                        destination.copy_(source, non_blocking=True)
                _sync_devices(source_index, destination_index)
                elapsed = time.perf_counter() - started
                seconds_per_copy = elapsed / iterations
                bandwidth = message_bytes / seconds_per_copy / 1e9
                matrix[source_index][destination_index] = bandwidth
                measurements.append(
                    {
                        "source": source_index,
                        "destination": destination_index,
                        "status": "ok",
                        "seconds_per_copy": seconds_per_copy,
                        "gb_per_second": bandwidth,
                    }
                )
                del source, destination
                torch.cuda.empty_cache()
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                measurements.append(
                    {
                        "source": source_index,
                        "destination": destination_index,
                        "status": "unavailable",
                        "reason": str(exc),
                    }
                )
                torch.cuda.empty_cache()
    return {
        "available": any(item["status"] == "ok" for item in measurements),
        "direction": "source_to_destination",
        "dtype": "float32",
        "message_bytes": message_bytes,
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "timing": "host_perf_counter_with_source_and_destination_synchronize",
        "gb_per_second_matrix": matrix,
        "measurements": measurements,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _maximum_rank_time(elapsed: float, device: torch.device) -> float:
    value = torch.tensor(elapsed, device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def _collective_worker(
    *,
    output: Path,
    sizes_mib: Sequence[int],
    warmup: int,
    iterations: int,
    backend: str | None,
) -> None:
    with distributed_session(backend=backend) as context:
        if not context.is_distributed:
            raise RuntimeError("collective benchmark requires WORLD_SIZE > 1")
        check = torch.tensor(
            float(context.rank + 1),
            device=context.device,
        )
        dist.all_reduce(check, op=dist.ReduceOp.SUM)
        expected = context.world_size * (context.world_size + 1) / 2
        if check.item() != expected:
            raise RuntimeError(
                f"all-reduce correctness check failed: {check.item()} != {expected}"
            )

        rows = []
        element_size = torch.tensor([], dtype=torch.float32).element_size()
        for size_mib in sizes_mib:
            elements = size_mib * 1024 * 1024 // element_size
            tensor = torch.zeros(
                elements,
                device=context.device,
                dtype=torch.float32,
            )
            for _ in range(warmup):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            _synchronize(context.device)
            distributed_barrier(context)
            started = time.perf_counter()
            for _ in range(iterations):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            _synchronize(context.device)
            elapsed = _maximum_rank_time(
                time.perf_counter() - started,
                context.device,
            )
            seconds = elapsed / iterations
            message_bytes = elements * element_size
            algorithmic = message_bytes / seconds / 1e9
            bus_factor = 2 * (context.world_size - 1) / context.world_size
            rows.append(
                {
                    "message_mib": size_mib,
                    "message_bytes": message_bytes,
                    "seconds_per_all_reduce": seconds,
                    "algorithmic_gb_per_second": algorithmic,
                    "bus_gb_per_second": algorithmic * bus_factor,
                    "bus_bandwidth_factor": bus_factor,
                }
            )
            del tensor
            torch.cuda.empty_cache()

        if context.is_main:
            atomic_write_json(
                output,
                {
                    "schema": "peps.multigpu_collective",
                    "schema_version": 1,
                    "world_size": context.world_size,
                    "backend_api": context.backend,
                    "backend_runtime": (
                        "RCCL"
                        if context.backend == "nccl"
                        and torch.version.hip is not None
                        else context.backend
                    ),
                    "transport": {
                        "rccl_p2p_disabled": (
                            os.environ.get("NCCL_P2P_DISABLE") == "1"
                        ),
                        "nccl_p2p_disable": os.environ.get(
                            "NCCL_P2P_DISABLE"
                        ),
                    },
                    "dtype": "float32",
                    "operation": "all_reduce_sum",
                    "warmup_iterations": warmup,
                    "timed_iterations": iterations,
                    "timing": "slowest_rank_host_time_with_device_synchronize",
                    "measurements": rows,
                },
            )
        distributed_barrier(context)


def _representative_targets(coords: torch.Tensor) -> torch.Tensor:
    x = coords[:, 0]
    y = coords[:, 1]
    return torch.stack(
        (
            0.5 + 0.5 * torch.sin(8.0 * torch.pi * x),
            0.5 + 0.5 * torch.cos(6.0 * torch.pi * y),
            (x * y).sqrt(),
        ),
        dim=1,
    )


def _training_worker(
    *,
    output: Path,
    global_batch_size: int,
    dataset_size: int,
    warmup_steps: int,
    timed_steps: int,
    seed: int,
    backend: str | None,
) -> None:
    with distributed_session(backend=backend) as context:
        if context.device.type != "cuda":
            raise RuntimeError("representative performance benchmark requires a GPU")
        effective_global_batch = min(global_batch_size, dataset_size)
        local_sizes = per_rank_batch_sizes(
            effective_global_batch,
            context.world_size,
        )
        if min(local_sizes) < 1:
            raise ValueError("global batch must give every rank at least one sample")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 101)
        coords = torch.rand(dataset_size, 2, generator=generator)
        targets = _representative_targets(coords)
        coords = coords.to(context.device)
        targets = targets.to(context.device)

        seed_process(seed)
        model, parameter_count = build_paper_image(
            "g_peps",
            signal_resolution=(768, 512),
            activation="gelu",
        )
        model = model.to(context.device)
        recipe = PaperTrainConfig(
            task="image",
            loss="l2",
            steps=warmup_steps + timed_steps,
            batch_size=global_batch_size,
            model_lr=0.001,
            encoder_lr=0.1,
            seed=seed,
            device=context.device,
        )
        optimizer = make_paper_optimizer(model, recipe)
        training_model = wrap_distributed(model, context)
        stream = MinibatchStream(dataset_size, global_batch_size, seed)
        batches = []
        for _ in range(warmup_steps + timed_steps):
            global_indices = stream.next()
            batches.append(
                (
                    local_minibatch_indices(
                        global_indices,
                        context,
                    ).to(context.device),
                    global_indices.numel(),
                )
            )

        def train_step(
            local_indices: torch.Tensor,
            current_global_batch_size: int,
        ) -> torch.Tensor:
            prediction = training_model(coords.index_select(0, local_indices))
            loss = F.mse_loss(
                prediction,
                targets.index_select(0, local_indices),
            )
            scaled = loss * ddp_loss_scale(
                local_indices.numel(),
                current_global_batch_size,
                context.world_size,
            )
            optimizer.zero_grad(set_to_none=True)
            scaled.backward()
            optimizer.step()
            return loss

        for local_indices, current_global_size in batches[:warmup_steps]:
            train_step(local_indices, current_global_size)
        _synchronize(context.device)
        distributed_barrier(context)
        started = time.perf_counter()
        final_loss = None
        for local_indices, current_global_size in batches[warmup_steps:]:
            final_loss = train_step(local_indices, current_global_size)
        _synchronize(context.device)
        elapsed = _maximum_rank_time(
            time.perf_counter() - started,
            context.device,
        )
        assert final_loss is not None
        global_loss = reduce_weighted_mean(
            final_loss,
            local_sizes[context.rank],
            context,
        )
        samples_per_second = (
            timed_steps * effective_global_batch / elapsed
        )
        if context.is_main:
            atomic_write_json(
                output,
                {
                    "schema": "peps.multigpu_training",
                    "schema_version": 1,
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
                        "rccl_p2p_disabled": (
                            os.environ.get("NCCL_P2P_DISABLE") == "1"
                        ),
                    },
                    "workload": {
                        "model": "paper Table-1 G-PEPS architecture",
                        "data": "deterministic synthetic coordinate regression",
                        "parameter_count": parameter_count,
                        "dataset_size": dataset_size,
                        "loss": "l2",
                        "optimizer": "Adam dual learning rate",
                        "global_batch_size": effective_global_batch,
                        "per_rank_batch_sizes": list(local_sizes),
                    },
                    "warmup_steps": warmup_steps,
                    "timed_steps": timed_steps,
                    "elapsed_seconds": elapsed,
                    "steps_per_second": timed_steps / elapsed,
                    "samples_per_second": samples_per_second,
                    "final_global_loss": global_loss,
                    "timing": (
                        "slowest_rank_steady_state_with_device_synchronize; "
                        "deterministic minibatch indices prepared before timing"
                    ),
                },
            )
        distributed_barrier(context)


def _torchrun_command(
    *,
    nproc: int,
    arguments: Sequence[str],
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={nproc}",
        "-m",
        "experiments.multigpu",
        "worker",
        *arguments,
    ]


def _run_worker(
    *,
    nproc: int,
    arguments: Sequence[str],
    timeout_seconds: int,
    environment_overrides: dict[str, str] | None = None,
) -> tuple[list[str], str, str]:
    command = _torchrun_command(nproc=nproc, arguments=arguments)
    environment = os.environ.copy()
    for name in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(name, None)
    environment.update(environment_overrides or {})
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        rendered = " ".join(shlex.quote(part) for part in command)
        raise RuntimeError(
            f"worker failed ({completed.returncode}): {rendered}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return command, completed.stdout, completed.stderr


def _render_command(
    command: Sequence[str],
    environment_overrides: dict[str, str] | None = None,
) -> str:
    prefix = [
        f"{name}={shlex.quote(value)}"
        for name, value in sorted((environment_overrides or {}).items())
    ]
    return " ".join(
        [*prefix, *(shlex.quote(part) for part in command)]
    )


def run_suite(arguments: argparse.Namespace) -> dict[str, object]:
    """Run topology, pairwise, collective, and 1-to-N PEPS benchmarks."""

    topology = inspect_topology()
    if topology["device_count"] < arguments.gpus:
        raise RuntimeError(
            f"requested {arguments.gpus} GPUs, found {topology['device_count']}"
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    worker_dir = arguments.output.parent / ".multigpu-workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    collective_path = worker_dir / "collective.json"
    probe_path = worker_dir / "collective-probe.json"
    one_gpu_path = worker_dir / "training-1gpu.json"
    many_gpu_path = worker_dir / f"training-{arguments.gpus}gpu.json"

    pairwise = measure_pairwise_copy_bandwidth(
        tensor_mib=arguments.copy_size_mib,
        warmup=arguments.copy_warmup,
        iterations=arguments.copy_iterations,
    )
    transport_environment: dict[str, str] = {}
    transport = {
        "requested": arguments.rccl_p2p,
        "effective": "peer_ipc",
        "fallback_reason": None,
    }
    if arguments.rccl_p2p == "off":
        transport_environment["NCCL_P2P_DISABLE"] = "1"
        transport["effective"] = "p2p_disabled_host_transport"
    elif arguments.rccl_p2p == "on":
        transport_environment["NCCL_P2P_DISABLE"] = "0"
    elif arguments.rccl_p2p == "auto":
        transport_environment["NCCL_P2P_DISABLE"] = "0"
        try:
            _run_worker(
                nproc=arguments.gpus,
                arguments=(
                    "--kind",
                    "collective",
                    "--output",
                    str(probe_path),
                    "--collective-sizes-mib",
                    "1",
                    "--collective-warmup",
                    "1",
                    "--collective-iterations",
                    "1",
                ),
                timeout_seconds=arguments.timeout_seconds,
                environment_overrides={
                    "NCCL_DEBUG": "INFO",
                    "NCCL_DEBUG_SUBSYS": "INIT",
                    "NCCL_P2P_DISABLE": "0",
                },
            )
        except RuntimeError as exc:
            detail = str(exc)
            lower = detail.lower()
            if (
                "hipipcgetmemhandle" not in lower
                and "cuda failure 'invalid argument'" not in lower
            ):
                raise
            transport_environment["NCCL_P2P_DISABLE"] = "1"
            transport["effective"] = "p2p_disabled_host_transport"
            if topology["iommu_passthrough_enabled"]:
                transport["fallback_reason"] = (
                    "RCCL peer IPC preflight failed with "
                    "hipIpcGetMemHandle/invalid argument despite iommu=pt."
                )
            else:
                transport["fallback_reason"] = (
                    "RCCL peer IPC preflight failed with "
                    "hipIpcGetMemHandle/invalid argument. The kernel command "
                    "line does not enable iommu=pt, matching RCCL's diagnostic."
                )
    collective_command, _, collective_stderr = _run_worker(
        nproc=arguments.gpus,
        arguments=(
            "--kind",
            "collective",
            "--output",
            str(collective_path),
            "--collective-sizes-mib",
            ",".join(str(size) for size in arguments.collective_sizes_mib),
            "--collective-warmup",
            str(arguments.collective_warmup),
            "--collective-iterations",
            str(arguments.collective_iterations),
        ),
        timeout_seconds=arguments.timeout_seconds,
        environment_overrides=transport_environment,
    )
    one_command, _, one_stderr = _run_worker(
        nproc=1,
        arguments=(
            "--kind",
            "training",
            "--output",
            str(one_gpu_path),
            "--global-batch-size",
            str(arguments.global_batch_size),
            "--dataset-size",
            str(arguments.dataset_size),
            "--training-warmup",
            str(arguments.training_warmup),
            "--training-steps",
            str(arguments.training_steps),
            "--seed",
            str(arguments.seed),
        ),
        timeout_seconds=arguments.timeout_seconds,
        environment_overrides=transport_environment,
    )
    many_command, _, many_stderr = _run_worker(
        nproc=arguments.gpus,
        arguments=(
            "--kind",
            "training",
            "--output",
            str(many_gpu_path),
            "--global-batch-size",
            str(arguments.global_batch_size),
            "--dataset-size",
            str(arguments.dataset_size),
            "--training-warmup",
            str(arguments.training_warmup),
            "--training-steps",
            str(arguments.training_steps),
            "--seed",
            str(arguments.seed),
        ),
        timeout_seconds=arguments.timeout_seconds,
        environment_overrides=transport_environment,
    )

    collective = json.loads(collective_path.read_text(encoding="utf-8"))
    one_gpu = json.loads(one_gpu_path.read_text(encoding="utf-8"))
    many_gpu = json.loads(many_gpu_path.read_text(encoding="utf-8"))
    speedup = (
        many_gpu["samples_per_second"] / one_gpu["samples_per_second"]
    )
    result = {
        "schema": "peps.multigpu_validation",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "topology": topology,
        "pairwise_copy": pairwise,
        "rccl_transport": transport,
        "collective": collective,
        "representative_training": {
            "comparison": "fixed-global-batch strong scaling of one DDP job",
            "one_gpu": one_gpu,
            f"{arguments.gpus}_gpu": many_gpu,
            "speedup": speedup,
            "parallel_efficiency": speedup / arguments.gpus,
        },
        "commands": {
            "suite": _render_command(
                [
                    sys.executable,
                    "-m",
                    "experiments.multigpu",
                    *sys.argv[1:],
                ]
            ),
            "collective": _render_command(
                collective_command,
                transport_environment,
            ),
            "one_gpu_training": _render_command(
                one_command,
                transport_environment,
            ),
            f"{arguments.gpus}_gpu_training": _render_command(
                many_command,
                transport_environment,
            ),
        },
        "worker_warnings": {
            "collective": collective_stderr.strip(),
            "one_gpu_training": one_stderr.strip(),
            f"{arguments.gpus}_gpu_training": many_stderr.strip(),
        },
        "limitations": [
            "P2P capability and copy bandwidth do not identify the physical "
            "peer route; PyTorch does not expose XGMI link type.",
            "When RCCL peer IPC fails, auto mode explicitly retries with "
            "NCCL_P2P_DISABLE=1; this uses a host-staged transport and is "
            "reported rather than presented as direct peer collective speed.",
            "The training input is deterministic synthetic data, while the "
            "model and global batch follow the representative paper workload.",
            "Fixed global batch measures strong scaling; it is not four "
            "independent jobs and not weak-scaling aggregate throughput.",
        ],
    }
    atomic_write_json(arguments.output, result)
    return result


def _positive_csv(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("sizes must be positive")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    topology = subparsers.add_parser("topology")
    topology.add_argument("--copy", action="store_true")
    topology.add_argument("--copy-size-mib", type=int, default=128)
    topology.add_argument("--copy-warmup", type=int, default=3)
    topology.add_argument("--copy-iterations", type=int, default=10)
    topology.add_argument("--output", type=Path)

    suite = subparsers.add_parser("suite")
    suite.add_argument(
        "--output",
        type=Path,
        default=Path("results/multigpu/benchmark.json"),
    )
    suite.add_argument("--gpus", type=int, default=4)
    suite.add_argument(
        "--rccl-p2p",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "RCCL process-to-process peer transport. auto probes direct IPC "
            "and explicitly reports a host-transport fallback."
        ),
    )
    suite.add_argument("--copy-size-mib", type=int, default=128)
    suite.add_argument("--copy-warmup", type=int, default=3)
    suite.add_argument("--copy-iterations", type=int, default=10)
    suite.add_argument(
        "--collective-sizes-mib",
        type=_positive_csv,
        default=(16, 64, 256),
    )
    suite.add_argument("--collective-warmup", type=int, default=5)
    suite.add_argument("--collective-iterations", type=int, default=20)
    suite.add_argument("--global-batch-size", type=int, default=60_000)
    suite.add_argument("--dataset-size", type=int, default=768 * 512)
    suite.add_argument("--training-warmup", type=int, default=20)
    suite.add_argument("--training-steps", type=int, default=200)
    suite.add_argument("--seed", type=int, default=2026)
    suite.add_argument("--timeout-seconds", type=int, default=600)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--kind", choices=("collective", "training"), required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument(
        "--collective-sizes-mib",
        type=_positive_csv,
        default=(16, 64, 256),
    )
    worker.add_argument("--collective-warmup", type=int, default=5)
    worker.add_argument("--collective-iterations", type=int, default=20)
    worker.add_argument("--global-batch-size", type=int, default=60_000)
    worker.add_argument("--dataset-size", type=int, default=768 * 512)
    worker.add_argument("--training-warmup", type=int, default=20)
    worker.add_argument("--training-steps", type=int, default=200)
    worker.add_argument("--seed", type=int, default=2026)
    worker.add_argument("--backend", choices=("nccl", "gloo"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "topology":
        result: dict[str, object] = {"topology": inspect_topology()}
        if arguments.copy:
            result["pairwise_copy"] = measure_pairwise_copy_bandwidth(
                tensor_mib=arguments.copy_size_mib,
                warmup=arguments.copy_warmup,
                iterations=arguments.copy_iterations,
            )
        if arguments.output is not None:
            atomic_write_json(arguments.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if arguments.command == "suite":
        result = run_suite(arguments)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if arguments.kind == "collective":
        _collective_worker(
            output=arguments.output,
            sizes_mib=arguments.collective_sizes_mib,
            warmup=arguments.collective_warmup,
            iterations=arguments.collective_iterations,
            backend=arguments.backend,
        )
    else:
        _training_worker(
            output=arguments.output,
            global_batch_size=arguments.global_batch_size,
            dataset_size=arguments.dataset_size,
            warmup_steps=arguments.training_warmup,
            timed_steps=arguments.training_steps,
            seed=arguments.seed,
            backend=arguments.backend,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
