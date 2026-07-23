"""Run one PEPS experiment job with PyTorch DDP/RCCL.

This is intentionally separate from :mod:`experiments.run`, whose ranks shard
different jobs.  Launch this module with torchrun when every rank should train
the same model on disjoint slices of one global minibatch.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Sequence

import torch
import torch.distributed as dist

from data.manifest import hash_file
from peps.distributed import (
    distributed_barrier,
    distributed_session,
    per_rank_batch_sizes,
    seed_process,
)
from peps.metrics import metric_versions
from peps.train import (
    fit_paper_distributed,
    paper_recipe_from_mapping,
    render_full,
)

from .config import load_experiment_config
from .run import load_tensor_instances
from .runner import (
    RunSpec,
    _assert_budget,
    _assert_compression,
    _build_model,
    _compression_factor,
    _git_provenance,
    _parameter_counts,
    atomic_torch_save,
    atomic_write_json,
    enumerate_jobs,
    evaluate_metrics,
)


def _select_job(
    jobs: Sequence[RunSpec],
    *,
    instance_name: str | None,
    method_name: str | None,
    seed: int | None,
) -> RunSpec:
    selected = [
        job
        for job in jobs
        if (instance_name is None or job.instance.name == instance_name)
        and (method_name is None or job.method.name == method_name)
        and (seed is None or job.seed == seed)
    ]
    if not selected:
        raise ValueError("no job matches the requested instance/method/seed")
    if len(selected) != 1:
        choices = ", ".join(
            f"{job.instance.name}/{job.method.name}/seed-{job.seed}"
            for job in selected[:8]
        )
        raise ValueError(
            "DDP runs exactly one job; select --instance, --method, and --seed "
            f"more narrowly (matches: {choices})"
        )
    return selected[0]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_across_ranks(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def run_distributed_job(
    *,
    config_path: Path,
    input_path: Path,
    output_dir: Path,
    instance_name: str | None = None,
    method_name: str | None = None,
    seed: int | None = None,
    force: bool = False,
    backend: str | None = None,
) -> dict | None:
    """Execute one selected job; only rank 0 returns and writes a record."""

    config = load_experiment_config(config_path)
    config_sha256 = hash_file(config.source, "sha256")
    instances = load_tensor_instances(input_path)
    job = _select_job(
        enumerate_jobs(config, instances),
        instance_name=instance_name,
        method_name=method_name,
        seed=seed,
    )
    result_path = output_dir / "result.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    existing_record = None
    if result_path.is_file() and not force:
        existing_record = json.loads(result_path.read_text(encoding="utf-8"))
        identity = (
            existing_record.get("experiment"),
            existing_record.get("instance"),
            existing_record.get("method"),
            existing_record.get("seed"),
        )
        expected_identity = (
            config.name,
            job.instance.name,
            job.method.name,
            job.seed,
        )
        if identity != expected_identity:
            raise ValueError(
                "output already contains a different DDP job; choose another "
                "--output directory or pass --force"
            )
        existing_config_sha256 = existing_record.get("config_sha256")
        if (
            existing_config_sha256 is not None
            and existing_config_sha256 != config_sha256
        ):
            raise ValueError(
                "output was produced by a different config revision; choose "
                "another --output directory or pass --force"
            )

    with distributed_session(backend=backend) as context:
        already_complete = existing_record is not None
        if context.is_distributed:
            flag = torch.tensor(
                int(already_complete) if context.is_main else 0,
                device=context.device,
            )
            dist.broadcast(flag, src=0)
            already_complete = bool(flag.item())
        if already_complete:
            distributed_barrier(context)
            if context.is_main:
                return existing_record
            return None

        # All ranks construct the same initial model.  DDP then broadcasts rank
        # 0 parameters; rank-offset seeds are used only for later stochastic
        # model operations (the minibatch stream has its own global seed).
        seed_process(job.seed)
        model, _ = _build_model(config, job.method, job.instance)
        counts = _parameter_counts(model)
        _assert_budget(job.method, counts)
        compression = _compression_factor(job.instance, counts["total"])
        _assert_compression(job.method, compression)
        seed_process(job.seed, rank=context.rank, rank_offset=True)

        training_values = dict(config.training)
        training_values.update(job.method.training)
        training_values["seed"] = job.seed
        recipe = replace(
            paper_recipe_from_mapping(training_values),
            device=context.device,
        )
        effective_global_batch = min(
            recipe.batch_size,
            job.instance.coords.shape[0],
        )
        local_sizes = per_rank_batch_sizes(
            effective_global_batch,
            context.world_size,
        )

        resume_state = None
        if checkpoint_path.is_file() and not force:
            resume_state = torch.load(
                checkpoint_path,
                map_location=context.device,
                weights_only=False,
            )
            checkpoint_job = resume_state.get("job")
            expected_checkpoint_job = {
                "experiment": config.name,
                "instance": job.instance.name,
                "method": job.method.name,
                "seed": job.seed,
                "config_sha256": config_sha256,
            }
            if checkpoint_job is not None:
                identity_fields = (
                    "experiment",
                    "instance",
                    "method",
                    "seed",
                )
                if (
                    not isinstance(checkpoint_job, dict)
                    or any(
                        checkpoint_job.get(name)
                        != expected_checkpoint_job[name]
                        for name in identity_fields
                    )
                    or (
                        checkpoint_job.get("config_sha256") is not None
                        and checkpoint_job["config_sha256"] != config_sha256
                    )
                ):
                    raise ValueError(
                        "checkpoint belongs to a different DDP job; choose "
                        "another --output directory or pass --force"
                    )
        resumed_step = (
            0 if resume_state is None else int(resume_state.get("step", 0))
        )
        losses: list[dict[str, float | int]] = []

        def log_callback(step: int, loss: float) -> None:
            losses.append({"step": step, "loss": loss})

        def checkpoint_callback(step: int, state) -> None:
            payload = dict(state)
            payload["job"] = {
                "experiment": config.name,
                "instance": job.instance.name,
                "method": job.method.name,
                "seed": job.seed,
                "config_sha256": config_sha256,
            }
            atomic_torch_save(checkpoint_path, payload)

        distributed_barrier(context)
        _synchronize(context.device)
        started = time.perf_counter()
        model = fit_paper_distributed(
            model,
            job.instance.coords,
            job.instance.targets,
            recipe,
            context=context,
            on_log=log_callback,
            on_checkpoint=checkpoint_callback,
            resume_state=resume_state,
        )
        _synchronize(context.device)
        elapsed = _max_across_ranks(
            time.perf_counter() - started,
            context.device,
        )

        record = None
        if context.is_main:
            prediction = render_full(
                model,
                job.instance.coords,
                chunk=int(config.runner.get("render_chunk", 262_144)),
                device=context.device,
            )
            metric_names = tuple(config.runner.get("metrics", ()))
            measured = evaluate_metrics(
                config.task,
                metric_names,
                job.instance,
                prediction,
            )
            completed_steps = recipe.total_steps - resumed_step
            processed_samples = completed_steps * effective_global_batch
            record = {
                "schema": "peps.ddp_job",
                "schema_version": 1,
                "experiment": config.name,
                "profile": config.profile,
                "task": config.task,
                "dataset": config.dataset,
                "instance": job.instance.name,
                "method": job.method.name,
                "seed": job.seed,
                "config_source": str(config.source),
                "config_sha256": config_sha256,
                "parallelism": {
                    "mode": "ddp_single_job",
                    "backend_api": context.backend,
                    "backend_runtime": (
                        "RCCL"
                        if context.backend == "nccl"
                        and torch.version.hip is not None
                        else context.backend
                    ),
                    "world_size": context.world_size,
                    "global_batch_size": effective_global_batch,
                    "per_rank_batch_sizes": list(local_sizes),
                    "rccl_p2p_disabled": (
                        os.environ.get("NCCL_P2P_DISABLE") == "1"
                    ),
                },
                "training": {
                    **training_values,
                    "total_steps": recipe.total_steps,
                    "resumed_step": resumed_step,
                    "elapsed_seconds": elapsed,
                    "samples_per_second": (
                        processed_samples / elapsed if elapsed > 0 else None
                    ),
                },
                "parameters": counts,
                "compression_factor": compression,
                "metrics": measured,
                "metric_versions": metric_versions(),
                "loss_log": losses,
                "checkpoint": str(checkpoint_path),
                **_git_provenance(config.source),
                "torch_version": torch.__version__,
                "rocm_version": torch.version.hip,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_json(result_path, record)

        # Keep non-zero ranks alive while rank 0 renders and writes.
        distributed_barrier(context)
        return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instance")
    parser.add_argument("--method")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--backend", choices=("nccl", "gloo"))
    parser.add_argument(
        "--disable-rccl-p2p",
        action="store_true",
        help=(
            "set NCCL_P2P_DISABLE=1 for hosts where RCCL peer IPC fails; "
            "collectives then use a slower host transport"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.disable_rccl_p2p:
        os.environ["NCCL_P2P_DISABLE"] = "1"
    record = run_distributed_job(
        config_path=arguments.config,
        input_path=arguments.input,
        output_dir=arguments.output,
        instance_name=arguments.instance,
        method_name=arguments.method,
        seed=arguments.seed,
        force=arguments.force,
        backend=arguments.backend,
    )
    if record is not None:
        print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
