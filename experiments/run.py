"""CLI for immutable paper configs and preprocessed tensor instances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_experiment_config
from .runner import (
    ExperimentRunner,
    TensorInstance,
    enumerate_jobs,
    resolve_shard,
    summarize_records,
    write_global_summary,
)


def load_tensor_instances(path) -> tuple[TensorInstance, ...]:
    """Load the data-layer handoff schema without owning data preparation."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "instances" not in payload:
        raise ValueError("tensor input must be a dict containing 'instances'")
    instances = []
    for item in payload["instances"]:
        if not isinstance(item, dict):
            raise ValueError("every tensor instance must be a mapping")
        instances.append(
            TensorInstance(
                name=item["name"],
                coords=item["coords"],
                targets=item["targets"],
                shape=(
                    None
                    if item.get("shape") is None
                    else tuple(item["shape"])
                ),
                metadata=item.get("metadata", {}),
            )
        )
    return tuple(instances)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Shard independent experiment jobs across processes. This is not "
            "DDP; use experiments.ddp for one job on multiple GPUs."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def main(argv=None) -> int:
    arguments = _parser().parse_args(argv)
    config = load_experiment_config(arguments.config)
    rank, world_size, local_rank = resolve_shard(
        arguments.rank,
        arguments.world_size,
        arguments.local_rank,
    )
    instances = (
        ()
        if arguments.input is None
        else load_tensor_instances(arguments.input)
    )
    if arguments.dry_run:
        payload = {
            "experiment": config.name,
            "profile": config.profile,
            "parallelism": {
                "mode": "job_shard",
                "same_model_distributed": False,
            },
            "rank": rank,
            "world_size": world_size,
            "methods": [method.name for method in config.methods],
            "jobs": (
                len(enumerate_jobs(config, instances))
                if instances
                else None
            ),
        }
        print(json.dumps(payload, indent=2))
        return 0
    output = (
        Path(arguments.output)
        if arguments.output is not None
        else Path(config.runner.get("output_dir", "results/paper")) / config.name
    )
    if arguments.summarize_only:
        print(json.dumps(write_global_summary(output), indent=2))
        return 0
    if not instances:
        raise SystemExit(
            "--input is required unless --dry-run or --summarize-only is used"
        )
    device = (
        None if arguments.device is None else torch.device(arguments.device)
    )
    runner = ExperimentRunner(
        config,
        output,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        force=arguments.force,
    )
    records = runner.run(instances)
    print(json.dumps(summarize_records(records), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
