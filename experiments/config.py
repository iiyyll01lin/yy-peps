"""Immutable TOML configuration schema for paper experiments."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class MethodConfig:
    name: str
    factory: str
    kwargs: Mapping[str, Any]
    seeds: tuple[int, ...] | None = None
    role: str = "canonical"
    training: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    expected_encoder_params: int | None = None
    expected_total_params: int | None = None
    compression_factor_min: float | None = None
    compression_factor_max: float | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "MethodConfig":
        allowed = set(cls.__dataclass_fields__)
        unexpected = set(values) - allowed
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ValueError(f"unknown method fields: {names}")
        if "name" not in values or "factory" not in values:
            raise ValueError("each method needs name and factory")
        seeds = values.get("seeds")
        if seeds is not None:
            seeds = tuple(int(seed) for seed in seeds)
            if not seeds:
                raise ValueError("method seeds cannot be empty")
        return cls(
            name=str(values["name"]),
            factory=str(values["factory"]),
            kwargs=_freeze(values.get("kwargs", {})),
            seeds=seeds,
            role=str(values.get("role", "canonical")),
            training=_freeze(values.get("training", {})),
            expected_encoder_params=values.get("expected_encoder_params"),
            expected_total_params=values.get("expected_total_params"),
            compression_factor_min=values.get("compression_factor_min"),
            compression_factor_max=values.get("compression_factor_max"),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    name: str
    paper: str
    task: str
    profile: str
    dataset: str
    canonical: bool
    seeds: tuple[int, ...]
    training: Mapping[str, Any]
    runner: Mapping[str, Any]
    methods: tuple[MethodConfig, ...]
    source: Path

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported experiment config schema")
        if self.task not in {"image", "texture", "sdf"}:
            raise ValueError("task must be image, texture, or sdf")
        if self.profile not in {"full", "smoke"}:
            raise ValueError("profile must be full or smoke")
        if not self.seeds:
            raise ValueError("config seeds cannot be empty")
        names = [method.name for method in self.methods]
        if len(names) != len(set(names)):
            raise ValueError("method names must be unique")
        if not self.methods:
            raise ValueError("config must define at least one method")


def load_experiment_config(path) -> ExperimentConfig:
    source = Path(path).resolve()
    with source.open("rb") as handle:
        values = tomllib.load(handle)
    required = {
        "schema_version",
        "name",
        "paper",
        "task",
        "profile",
        "dataset",
        "canonical",
        "seeds",
        "training",
        "runner",
        "methods",
    }
    missing = required - set(values)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"config is missing fields: {names}")
    unexpected = set(values) - required
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ValueError(f"unknown top-level config fields: {names}")
    return ExperimentConfig(
        schema_version=int(values["schema_version"]),
        name=str(values["name"]),
        paper=str(values["paper"]),
        task=str(values["task"]),
        profile=str(values["profile"]),
        dataset=str(values["dataset"]),
        canonical=bool(values["canonical"]),
        seeds=tuple(int(seed) for seed in values["seeds"]),
        training=_freeze(values["training"]),
        runner=_freeze(values["runner"]),
        methods=tuple(
            MethodConfig.from_mapping(method) for method in values["methods"]
        ),
        source=source,
    )
