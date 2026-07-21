"""Reproducible paper experiment infrastructure."""

from .config import ExperimentConfig, MethodConfig, load_experiment_config
from .runner import (
    ExperimentRunner,
    RunSpec,
    TensorInstance,
    collect_raw_records,
    enumerate_jobs,
    paired_delta,
    resolve_shard,
    shard_jobs,
    write_global_summary,
)

__all__ = [
    "ExperimentConfig",
    "MethodConfig",
    "load_experiment_config",
    "ExperimentRunner",
    "RunSpec",
    "TensorInstance",
    "collect_raw_records",
    "enumerate_jobs",
    "shard_jobs",
    "resolve_shard",
    "paired_delta",
    "write_global_summary",
]
