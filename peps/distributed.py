"""Small, explicit helpers for one-job PyTorch DDP execution.

PyTorch exposes AMD ROCm devices through the ``torch.cuda`` API and uses the
``nccl`` backend name for RCCL.  This module deliberately does not auto-enable
DDP from environment variables: callers must enter :func:`distributed_session`
so the existing single-process training APIs keep their original behaviour.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    """Resolved rank, device, and process-group state for one process."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str | None
    process_group_initialized: bool

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def resolve_distributed_environment(
    environment: dict[str, str] | None = None,
) -> tuple[int, int, int]:
    """Resolve torchrun rank variables, or return single-process defaults."""

    values = os.environ if environment is None else environment
    names = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    present = {name for name in names if name in values}
    if present and present != set(names):
        missing = ", ".join(sorted(set(names) - present))
        raise ValueError(f"incomplete torchrun environment; missing {missing}")

    rank = int(values.get("RANK", 0))
    world_size = int(values.get("WORLD_SIZE", 1))
    local_rank = int(values.get("LOCAL_RANK", 0))
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("RANK must be in [0, WORLD_SIZE)")
    if local_rank < 0:
        raise ValueError("LOCAL_RANK must be non-negative")
    return rank, world_size, local_rank


@contextmanager
def distributed_session(
    *,
    backend: str | None = None,
    timeout_seconds: int = 600,
) -> Iterator[DistributedContext]:
    """Initialize and reliably tear down a torchrun process group.

    GPU execution defaults to the PyTorch ``nccl`` backend, which dispatches to
    RCCL in a ROCm wheel.  CPU execution defaults to ``gloo`` and is useful for
    logic/integration tests.
    """

    rank, world_size, local_rank = resolve_distributed_environment()
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {device_count} GPU(s) are visible"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        resolved_backend = backend or "nccl"
    else:
        device = torch.device("cpu")
        resolved_backend = backend or "gloo"

    owns_process_group = False
    initialized = dist.is_available() and dist.is_initialized()
    if world_size > 1:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        if initialized:
            if dist.get_rank() != rank or dist.get_world_size() != world_size:
                raise RuntimeError(
                    "existing process group does not match torchrun environment"
                )
            active_backend = str(dist.get_backend())
            if backend is not None and active_backend != resolved_backend:
                raise RuntimeError(
                    f"existing backend {active_backend!r} != {resolved_backend!r}"
                )
            resolved_backend = active_backend
        else:
            dist.init_process_group(
                backend=resolved_backend,
                init_method="env://",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=timeout_seconds),
            )
            owns_process_group = True
            initialized = True
    else:
        # No collective is needed for a one-process baseline.  Avoid creating a
        # process group so the public training wrapper also works outside
        # torchrun without MASTER_ADDR/MASTER_PORT.
        resolved_backend = None

    context = DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        backend=resolved_backend,
        process_group_initialized=initialized and world_size > 1,
    )
    try:
        yield context
    finally:
        if owns_process_group and dist.is_initialized():
            # Do not barrier here: if one rank is unwinding an exception, a
            # barrier can turn the original failure into a ten-minute hang.
            dist.destroy_process_group()


def per_rank_batch_sizes(global_batch_size: int, world_size: int) -> tuple[int, ...]:
    """Return balanced local sizes whose sum is the global batch size."""

    if global_batch_size < 1:
        raise ValueError("global_batch_size must be positive")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    base, remainder = divmod(global_batch_size, world_size)
    return tuple(base + (rank < remainder) for rank in range(world_size))


def local_batch_slice(
    global_batch_size: int,
    *,
    rank: int,
    world_size: int,
) -> slice:
    """Return one non-overlapping, exhaustive slice of a global minibatch."""

    sizes = per_rank_batch_sizes(global_batch_size, world_size)
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    start = sum(sizes[:rank])
    return slice(start, start + sizes[rank])


def local_minibatch_indices(
    global_indices: torch.Tensor,
    context: DistributedContext,
) -> torch.Tensor:
    """Select this rank's slice without changing global sampling semantics."""

    if global_indices.ndim != 1:
        raise ValueError("global_indices must be rank-1")
    selected = local_batch_slice(
        global_indices.numel(),
        rank=context.rank,
        world_size=context.world_size,
    )
    return global_indices[selected]


def ddp_loss_scale(
    local_batch_size: int,
    global_batch_size: int,
    world_size: int,
) -> float:
    """Scale a local mean so DDP's rank average equals the global mean."""

    if local_batch_size < 0:
        raise ValueError("local_batch_size must be non-negative")
    if global_batch_size < 1 or world_size < 1:
        raise ValueError("global_batch_size and world_size must be positive")
    return world_size * local_batch_size / global_batch_size


def reduce_weighted_mean(
    value: torch.Tensor,
    weight: int,
    context: DistributedContext,
) -> float:
    """Return a sample-weighted scalar mean on every rank."""

    if value.numel() != 1:
        raise ValueError("value must be scalar")
    if weight < 0:
        raise ValueError("weight must be non-negative")
    pair = torch.stack(
        (
            value.detach().to(dtype=torch.float64) * weight,
            torch.tensor(float(weight), device=value.device, dtype=torch.float64),
        )
    )
    if context.is_distributed:
        dist.all_reduce(pair, op=dist.ReduceOp.SUM)
    if pair[1].item() == 0:
        raise ValueError("weighted mean has zero total weight")
    return float((pair[0] / pair[1]).item())


def distributed_barrier(context: DistributedContext) -> None:
    """Synchronize ranks while explicitly selecting the NCCL/RCCL device.

    ``barrier(device_ids=...)`` is available in the minimum supported
    PyTorch 2.4 API.  It avoids relying on the newer
    ``init_process_group(device_id=...)`` argument while also avoiding NCCL's
    rank-to-device inference and its associated warning.
    """

    if not context.is_distributed:
        return
    if context.device.type == "cuda":
        dist.barrier(device_ids=[context.local_rank])
    else:
        dist.barrier()


def wrap_distributed(
    model: nn.Module,
    context: DistributedContext,
) -> nn.Module:
    """Wrap a model in DDP only when more than one rank is active."""

    if not context.is_distributed:
        return model
    if context.device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
        )
    return DistributedDataParallel(model)


def unwrap_distributed(model: nn.Module) -> nn.Module:
    """Return the underlying module without a ``module.`` state-dict prefix."""

    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def seed_process(seed: int, *, rank: int = 0, rank_offset: bool = False) -> int:
    """Seed Python, NumPy (when available), and PyTorch deterministically."""

    resolved = int(seed) + (int(rank) if rank_offset else 0)
    random.seed(resolved)
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)
    try:
        import numpy as np
    except ImportError:
        return resolved
    np.random.seed(resolved)
    return resolved
