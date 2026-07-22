"""Reproduce the public three-shape PEPS SDF benchmark.

This entry point is deliberately independent from the image/texture runner.
It implements:

* Table 3's ten MAPE-trained methods on Lucy, Thai Statue, and Armadillo;
* Table 6's published nine-method L1 subset on those same public shapes;
* exact streamed 512^3 occupancy IoU and a three-shape-only aggregate;
* deterministic checkpoint/resume and independent four-GPU job sharding;
* a fixed, explicitly non-paper-exact Armadillo render/FLIP protocol; and
* a parameter-only ``deferred_auth_required`` receipt for Stonefish/Table 4.

No code path substitutes another mesh for the canonical Pitted Stonefish.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
import tomllib
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import torch
import torch.nn as nn

from apps.sdf.build import build_paper_sdf
from apps.sdf.data import (
    iter_query_slabs,
    load_paper_sdf_volume,
    sample_sdf_tensor,
)
from apps.sdf.render import (
    FirstSurfaceAccumulator,
    OrthographicRenderProtocol,
    evaluate_flip,
    save_png_atomic,
)
from data.manifest import hash_file
from experiments.runner import atomic_torch_save, atomic_write_json
from peps.metrics import IoUAccumulator, metric_versions
from peps.report import collect_git_state
from peps.train import (
    PaperTrainConfig,
    l1_loss,
    make_paper_optimizer,
    mape_loss,
    split_encoder_decoder_parameters,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "paper" / "sdf"
DEFAULT_CONFIGS = (
    CONFIG_ROOT / "table3_mape.toml",
    CONFIG_ROOT / "table6_l1.toml",
)
DEFAULT_TABLE4_CONFIG = CONFIG_ROOT / "table4_deferred.toml"
DEFAULT_SMOKE_CONFIG = CONFIG_ROOT / "smoke.toml"
DEFAULT_WORK_ROOT = ROOT / "results" / "work" / "sdf-repro"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "sdf_repro"

PUBLIC_ASSETS = ("lucy", "thai-statue", "armadillo")
STONEFISH_ASSET = "pitted-stonefish"
PAPER = "PEPS Extended arXiv:2604.24167v1"
TABLE3_METHODS = (
    ("PE", "pe"),
    ("LPE", "lpe"),
    ("TI-Grid", "grid"),
    ("Grid-PEPS", "grid_peps"),
    ("Hash", "hash"),
    ("Hash-PEPS", "hash_peps"),
    ("M-Grid", "m_grid"),
    ("M-PEPS", "m_peps"),
    ("M-Hash", "m_hash"),
    ("M-HashPEPS", "m_hashpeps"),
)
TABLE6_METHODS = (
    ("TI-Grid", "grid"),
    ("Hash", "hash"),
    ("LPE", "lpe"),
    ("Grid-PEPS", "grid_peps"),
    ("PE", "pe"),
    ("M-PEPS", "m_peps"),
    ("M-Grid", "m_grid"),
    ("M-Hash", "m_hash"),
    ("M-HashPEPS", "m_hashpeps"),
)
TABLE4_METHODS = TABLE6_METHODS

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "artifact",
    "paper",
    "paper_table",
    "profile",
    "status",
    "scope",
    "canonical_four_shape",
    "paper_global_comparable",
    "assets",
    "seed",
    "training",
    "evaluation",
    "sharding",
    "render",
    "reporting",
    "deferred",
    "methods",
}
_REQUIRED_TOP_LEVEL_FIELDS = _TOP_LEVEL_FIELDS - {"deferred"}
_TRAINING_FIELDS = {
    "loss",
    "epochs",
    "batches_per_epoch",
    "batch_size",
    "model_lr",
    "encoder_lr",
    "optimizer",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "weight_decay",
    "activation",
    "hidden_layers",
    "hidden_width",
    "frequencies",
    "coordinate_sampling",
    "target_sampling",
    "eikonal",
    "cosine",
    "checkpoint_every",
    "log_every",
    "mape_epsilon",
}
_EVALUATION_FIELDS = {
    "metric",
    "resolution",
    "chunk_size",
    "occupancy_rule",
    "axis_order",
    "query_component_order",
}
_SHARDING_FIELDS = {"mode", "world_size", "same_model_distributed"}
_REPORTING_FIELDS = {
    "aggregate_label",
    "forbidden_aggregate_labels",
    "deferred_asset",
    "deferred_status",
}
_DEFERRED_FIELDS = {
    "reason",
    "auth_env_names",
    "substitution_allowed",
    "numeric_results_allowed",
    "canonical_source_uid",
    "required_receipt_status",
}
_METHOD_FIELDS = {
    "name",
    "key",
    "kwargs",
    "paper_iou",
    "shared_no_encoder_measurement",
    "expected_encoder_params",
    "expected_decoder_params",
    "expected_total_params",
    "expected_encoder_params_1x",
    "expected_decoder_params_1x",
    "expected_total_params_1x",
    "expected_encoder_params_8x",
    "expected_decoder_params_8x",
    "expected_total_params_8x",
}
_RENDER_FIELDS = {
    "enabled",
    "asset",
    "camera",
    "image_axes",
    "resolution",
    "surface_rule",
    "normal_estimator",
    "light_direction",
    "ambient",
    "diffuse",
    "albedo",
    "background",
    "flip_mode",
    "flip_ppd",
    "paper_camera_available",
    "verification_status",
}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        _plain(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_component(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )


def _portable_path(path: str | Path) -> str:
    source = Path(path)
    try:
        return source.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(source)


def _require_exact_fields(
    values: Mapping[str, Any],
    expected: set[str],
    label: str,
    *,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = expected - set(values)
    extra = set(values) - expected - optional
    if missing or extra:
        raise ValueError(
            f"{label} fields mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def _require_int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _require_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        relation = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {relation}")
    return result


@dataclass(frozen=True)
class SDFMethodSpec:
    name: str
    key: str
    kwargs: Mapping[str, Any]
    paper_iou: Mapping[str, float]
    expected: Mapping[str, int]
    shared_no_encoder_measurement: bool = False

    def expected_counts(self, multiplier: int = 1) -> dict[str, int]:
        suffix = "" if "encoder" in self.expected else f"_{multiplier}x"
        fields = {
            part: f"{part}{suffix}"
            for part in ("encoder", "decoder", "total")
        }
        try:
            return {
                part: int(self.expected[field])
                for part, field in fields.items()
            }
        except KeyError as exc:
            raise ValueError(
                f"{self.name}: no expected counts for {multiplier}x"
            ) from exc


@dataclass(frozen=True)
class SDFReproConfig:
    source: Path
    values: Mapping[str, Any]
    methods: tuple[SDFMethodSpec, ...]

    @property
    def artifact(self) -> str:
        return str(self.values["artifact"])

    @property
    def paper_table(self) -> str:
        return str(self.values["paper_table"])

    @property
    def profile(self) -> str:
        return str(self.values["profile"])

    @property
    def status(self) -> str:
        return str(self.values["status"])

    @property
    def scope(self) -> str:
        return str(self.values["scope"])

    @property
    def assets(self) -> tuple[str, ...]:
        return tuple(str(asset) for asset in self.values["assets"])

    @property
    def seed(self) -> int:
        return int(self.values["seed"])

    @property
    def training(self) -> Mapping[str, Any]:
        return self.values["training"]

    @property
    def evaluation(self) -> Mapping[str, Any]:
        return self.values["evaluation"]

    @property
    def sharding(self) -> Mapping[str, Any]:
        return self.values["sharding"]

    @property
    def render(self) -> Mapping[str, Any]:
        return self.values["render"]

    @property
    def reporting(self) -> Mapping[str, Any]:
        return self.values["reporting"]

    @property
    def deferred(self) -> Mapping[str, Any] | None:
        return self.values.get("deferred")

    @property
    def digest(self) -> str:
        return _canonical_digest(self.values)

    @property
    def total_steps(self) -> int:
        return int(self.training["epochs"]) * int(
            self.training["batches_per_epoch"]
        )

    def to_dict(self) -> dict[str, Any]:
        return _plain(self.values)


def _parse_method(
    values: Mapping[str, Any],
    *,
    deferred: bool,
) -> SDFMethodSpec:
    _require_exact_fields(
        values,
        {"name", "key"},
        "method",
        optional=_METHOD_FIELDS - {"name", "key"},
    )
    name = str(values["name"])
    key = str(values["key"])
    if not name or not key:
        raise ValueError("method name and key cannot be empty")
    kwargs = values.get("kwargs", {})
    paper_iou = values.get("paper_iou", {})
    if not isinstance(kwargs, Mapping) or not isinstance(paper_iou, Mapping):
        raise TypeError("method kwargs and paper_iou must be mappings")
    for asset, value in paper_iou.items():
        score = _require_number(value, f"{name}.paper_iou.{asset}")
        if not 0 <= score <= 1:
            raise ValueError("paper IoU values must lie in [0, 1]")

    expected_keys = (
        (
            "expected_encoder_params_1x",
            "expected_decoder_params_1x",
            "expected_total_params_1x",
            "expected_encoder_params_8x",
            "expected_decoder_params_8x",
            "expected_total_params_8x",
        )
        if deferred
        else (
            "expected_encoder_params",
            "expected_decoder_params",
            "expected_total_params",
        )
    )
    missing = [field for field in expected_keys if field not in values]
    if missing:
        raise ValueError(f"{name}: missing parameter assertions {missing}")
    expected: dict[str, int] = {}
    for field in expected_keys:
        short = field.removeprefix("expected_").removesuffix("_params")
        if field.endswith("_1x") or field.endswith("_8x"):
            prefix, multiplier = field.removeprefix("expected_").rsplit("_params_", 1)
            short = f"{prefix}_{multiplier}"
        expected[short] = _require_int(values[field], f"{name}.{field}")
    return SDFMethodSpec(
        name=name,
        key=key,
        kwargs=_freeze(kwargs),
        paper_iou=_freeze(
            {str(asset): float(value) for asset, value in paper_iou.items()}
        ),
        expected=MappingProxyType(expected),
        shared_no_encoder_measurement=bool(
            values.get("shared_no_encoder_measurement", False)
        ),
    )


def _validate_config_invariants(config: SDFReproConfig) -> None:
    values = config.values
    if values["paper"] != PAPER:
        raise ValueError(f"{config.source}: unexpected paper identifier")
    if values["profile"] not in {"full", "smoke"}:
        raise ValueError("profile must be full or smoke")
    if values["status"] not in {"runnable", "deferred_auth_required"}:
        raise ValueError("unsupported SDF config status")
    if bool(values["canonical_four_shape"]):
        raise ValueError("this phase must not claim a canonical four-shape run")
    if bool(values["paper_global_comparable"]):
        raise ValueError("three-shape results cannot be paper-Global comparable")
    if len(config.assets) != len(set(config.assets)):
        raise ValueError("asset IDs must be unique")
    _require_int(config.seed, "seed")

    training = config.training
    _require_exact_fields(training, _TRAINING_FIELDS, "training")
    if training["loss"] not in {"mape", "l1"}:
        raise ValueError("SDF loss must be mape or l1")
    for field in (
        "epochs",
        "batches_per_epoch",
        "batch_size",
        "hidden_layers",
        "hidden_width",
        "frequencies",
        "checkpoint_every",
        "log_every",
    ):
        _require_int(training[field], f"training.{field}", minimum=1)
    for field in ("model_lr", "encoder_lr", "adam_epsilon", "mape_epsilon"):
        _require_number(training[field], f"training.{field}", positive=True)
    weight_decay = _require_number(
        training["weight_decay"],
        "training.weight_decay",
    )
    if weight_decay < 0:
        raise ValueError("training.weight_decay must be non-negative")
    for field in ("adam_beta1", "adam_beta2"):
        value = _require_number(training[field], f"training.{field}")
        if not 0 <= value < 1:
            raise ValueError(f"training.{field} must lie in [0, 1)")
    if training["optimizer"] != "adam":
        raise ValueError("paper SDF configs require Adam")
    if training["activation"] != "silu":
        raise ValueError("paper SDF configs require SiLU")
    if bool(training["eikonal"]):
        raise ValueError("eikonal regularization is forbidden in paper SDF runs")
    if bool(training["cosine"]):
        raise ValueError("paper SDF runs use a fixed learning rate")

    evaluation = config.evaluation
    _require_exact_fields(evaluation, _EVALUATION_FIELDS, "evaluation")
    if (
        evaluation["metric"] != "iou"
        or evaluation["occupancy_rule"] != "sdf<0"
        or evaluation["axis_order"] != "zyx"
        or evaluation["query_component_order"] != "xyz"
    ):
        raise ValueError("evaluation does not match the frozen SDF IoU protocol")
    _require_int(evaluation["resolution"], "evaluation.resolution", minimum=2)
    _require_int(evaluation["chunk_size"], "evaluation.chunk_size", minimum=1)

    sharding = config.sharding
    _require_exact_fields(sharding, _SHARDING_FIELDS, "sharding")
    if sharding["mode"] != "independent_job_modulo":
        raise ValueError("SDF full matrix requires independent job sharding")
    if bool(sharding["same_model_distributed"]):
        raise ValueError("SDF matrix sharding is not same-model DDP")
    _require_int(sharding["world_size"], "sharding.world_size", minimum=1)

    reporting = config.reporting
    _require_exact_fields(reporting, _REPORTING_FIELDS, "reporting")
    if reporting["deferred_asset"] != STONEFISH_ASSET:
        raise ValueError("the only deferred SDF asset must be Pitted Stonefish")
    if reporting["deferred_status"] != "deferred_auth_required":
        raise ValueError("Stonefish must retain deferred_auth_required status")
    if reporting["aggregate_label"] in set(
        reporting["forbidden_aggregate_labels"]
    ):
        raise ValueError("aggregate label is forbidden by the config")

    render = config.render
    if not isinstance(render, Mapping):
        raise TypeError("render must be a mapping")
    extra_render = set(render) - _RENDER_FIELDS
    if extra_render:
        raise ValueError(f"unknown render fields: {sorted(extra_render)}")
    if "enabled" not in render:
        raise ValueError("render.enabled is required")

    method_pairs = tuple((method.name, method.key) for method in config.methods)
    if len(method_pairs) != len(set(method_pairs)):
        raise ValueError("method name/key pairs must be unique")
    if config.profile == "smoke":
        if config.paper_table != "smoke-only":
            raise ValueError("smoke config cannot name a paper table")
        if config.assets != ("synthetic-sphere",):
            raise ValueError("smoke config must use only synthetic-sphere")
        return

    if config.seed != 0:
        raise ValueError("paper SDF canonical run freezes the unreported seed to zero")
    if config.total_steps != 120_000 or int(training["batch_size"]) != 60_000:
        raise ValueError("full SDF configs require 3000x40 batches of 60000")
    if int(training["hidden_layers"]) != 3 or int(training["hidden_width"]) != 64:
        raise ValueError("paper SDF decoder must have three 64-wide hidden layers")
    if int(training["frequencies"]) != 3:
        raise ValueError("paper SDF PEPS methods require three frequencies")
    if int(evaluation["resolution"]) != 512:
        raise ValueError("full SDF evaluation must cover the complete 512^3 volume")
    if int(sharding["world_size"]) != 4:
        raise ValueError("full SDF configs must describe four-GPU job sharding")

    if config.paper_table in {"Table 3", "Table 6"}:
        if config.status != "runnable":
            raise ValueError("public-shape table configs must be runnable")
        if config.assets != PUBLIC_ASSETS:
            raise ValueError("runnable configs must contain only the three public shapes")
        if STONEFISH_ASSET in config.assets:
            raise ValueError("Stonefish cannot enter a runnable config in this phase")
        expected_methods = (
            TABLE3_METHODS if config.paper_table == "Table 3" else TABLE6_METHODS
        )
        if method_pairs != expected_methods:
            raise ValueError(
                f"{config.paper_table} methods/order do not match the paper"
            )
        expected_loss = "mape" if config.paper_table == "Table 3" else "l1"
        expected_lr = 0.001 if expected_loss == "mape" else 0.01
        if training["loss"] != expected_loss:
            raise ValueError(f"{config.paper_table} has the wrong loss")
        if (
            float(training["model_lr"]) != expected_lr
            or float(training["encoder_lr"]) != expected_lr
        ):
            raise ValueError(f"{config.paper_table} has the wrong learning rate")
        if reporting["aggregate_label"] != "three_shape_aggregate":
            raise ValueError("public tables must use three_shape_aggregate")
        for method in config.methods:
            if set(method.paper_iou) != set(PUBLIC_ASSETS):
                raise ValueError(
                    f"{method.name}: paper references must cover exactly public shapes"
                )
        if config.paper_table == "Table 6":
            if not bool(render["enabled"]) or render.get("asset") != "armadillo":
                raise ValueError("Table 6 must render Armadillo")
            if int(render.get("resolution", 0)) != 512:
                raise ValueError("Armadillo render must use the 512^3 evaluation")
            OrthographicRenderProtocol.from_mapping(render)
        elif bool(render["enabled"]):
            raise ValueError("only the Table 6 subset owns Armadillo render artifacts")
        return

    if config.paper_table != "Table 4":
        raise ValueError(f"unsupported full paper table {config.paper_table!r}")
    if config.status != "deferred_auth_required":
        raise ValueError("Table 4 must be deferred_auth_required")
    if config.assets != (STONEFISH_ASSET,):
        raise ValueError("Table 4 may name only canonical Pitted Stonefish")
    if method_pairs != TABLE4_METHODS:
        raise ValueError("Table 4 methods/order do not match the published table")
    if bool(render["enabled"]):
        raise ValueError("deferred Table 4 cannot generate a render")
    deferred = config.deferred
    if deferred is None:
        raise ValueError("Table 4 requires deferred metadata")
    _require_exact_fields(deferred, _DEFERRED_FIELDS, "deferred")
    if bool(deferred["substitution_allowed"]):
        raise ValueError("Table 4 must forbid mesh substitutions")
    if bool(deferred["numeric_results_allowed"]):
        raise ValueError("deferred Table 4 must forbid numeric results")
    if deferred["required_receipt_status"] != "deferred_auth_required":
        raise ValueError("deferred receipt has the wrong status")
    if deferred["canonical_source_uid"] != "0cdc3d1419384fd78fd952dc251a3169":
        raise ValueError("Table 4 must identify the canonical CT Stonefish")
    if any(method.paper_iou for method in config.methods):
        raise ValueError("Table 4 config must not embed numeric paper results")


def load_sdf_repro_config(path: str | Path) -> SDFReproConfig:
    """Load and strictly validate one SDF-specific TOML config."""

    source = Path(path).resolve()
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    _require_exact_fields(
        raw,
        _REQUIRED_TOP_LEVEL_FIELDS,
        "top-level",
        optional={"deferred"},
    )
    if int(raw["schema_version"]) != 1:
        raise ValueError("unsupported SDF reproduction config schema")
    methods_raw = raw["methods"]
    if not isinstance(methods_raw, list) or not methods_raw:
        raise ValueError("config methods must be a non-empty array")
    deferred = raw["status"] == "deferred_auth_required"
    methods = tuple(
        _parse_method(method, deferred=deferred)
        for method in methods_raw
    )
    config = SDFReproConfig(
        source=source,
        values=_freeze(raw),
        methods=methods,
    )
    _validate_config_invariants(config)
    return config


def parameter_counts(model: nn.Module) -> dict[str, int]:
    encoder, decoder = split_encoder_decoder_parameters(model)
    encoder_count = sum(parameter.numel() for parameter in encoder)
    decoder_count = sum(parameter.numel() for parameter in decoder)
    return {
        "encoder": encoder_count,
        "decoder": decoder_count,
        "total": encoder_count + decoder_count,
    }


def build_and_assert_method(
    method: SDFMethodSpec,
    *,
    multiplier: int = 1,
) -> tuple[nn.Module, dict[str, int]]:
    """Build a method and enforce config encoder/decoder/total assertions."""

    model, reported = build_paper_sdf(
        method.key,
        encoder_parameter_multiplier=multiplier,
        **_plain(method.kwargs),
    )
    counts = parameter_counts(model)
    if int(reported) != counts["total"]:
        raise AssertionError(
            f"{method.name}: builder reported {reported}, found {counts['total']}"
        )
    expected = method.expected_counts(multiplier)
    if counts != expected:
        raise AssertionError(
            f"{method.name} {multiplier}x counts {counts} != {expected}"
        )
    return model, counts


def assert_config_parameter_budgets(
    config: SDFReproConfig,
) -> tuple[dict[str, object], ...]:
    """Build every configured method and return parameter assertion rows."""

    rows: list[dict[str, object]] = []
    multipliers = (1, 8) if config.status == "deferred_auth_required" else (1,)
    for method in config.methods:
        previous: dict[str, int] | None = None
        for multiplier in multipliers:
            model, counts = build_and_assert_method(method, multiplier=multiplier)
            row = {
                "method": method.name,
                "key": method.key,
                "encoder_parameter_multiplier": multiplier,
                "parameters": counts,
                "assertion": "passed",
            }
            if previous is not None:
                expected_ratio = 1 if method.key == "pe" else 8
                actual_ratio = (
                    1
                    if previous["encoder"] == counts["encoder"] == 0
                    else counts["encoder"] // previous["encoder"]
                )
                if actual_ratio != expected_ratio:
                    raise AssertionError(
                        f"{method.name}: encoder ratio {actual_ratio} != "
                        f"{expected_ratio}"
                    )
                row["encoder_ratio_from_1x"] = actual_ratio
            rows.append(row)
            previous = counts
            del model
    return tuple(rows)


@dataclass(frozen=True)
class SDFJob:
    config: SDFReproConfig
    method: SDFMethodSpec
    asset: str
    index: int

    @property
    def identity(self) -> dict[str, object]:
        return {
            "artifact": self.config.artifact,
            "config_sha256": self.config.digest,
            "asset": self.asset,
            "method": self.method.name,
            "method_key": self.method.key,
            "loss": str(self.config.training["loss"]),
            "seed": self.config.seed,
        }


def enumerate_sdf_jobs(
    configs: Sequence[SDFReproConfig],
) -> tuple[SDFJob, ...]:
    jobs: list[SDFJob] = []
    index = 0
    for config in configs:
        if config.status != "runnable":
            continue
        for asset in config.assets:
            for method in config.methods:
                jobs.append(SDFJob(config, method, asset, index))
                index += 1
    return tuple(jobs)


def shard_sdf_jobs(
    jobs: Sequence[SDFJob],
    *,
    rank: int,
    world_size: int,
) -> tuple[SDFJob, ...]:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    return tuple(job for job in jobs if job.index % world_size == rank)


def _config_work_dir(config: SDFReproConfig, work_root: Path) -> Path:
    return work_root / config.artifact / config.digest[:16]


def _job_paths(job: SDFJob, work_root: Path) -> tuple[Path, Path]:
    stem = (
        Path(_safe_component(job.asset))
        / _safe_component(job.method.name)
        / f"seed-{job.config.seed}"
    )
    base = _config_work_dir(job.config, work_root)
    return base / "raw" / stem.with_suffix(".json"), (
        base / "checkpoints" / stem.with_suffix(".pt")
    )


@dataclass(frozen=True)
class LoadedSDF:
    asset: str
    values: np.ndarray
    volume_path: Path | None
    local_provenance_path: Path | None
    tracked_provenance_path: Path | None
    metadata: Mapping[str, Any]


def _load_public_volume(
    asset: str,
    *,
    processed_root: str | Path | None,
    verify_checksum: bool,
) -> LoadedSDF:
    loaded = load_paper_sdf_volume(
        asset,
        processed_root=processed_root,
        verify_checksum=verify_checksum,
    )
    tracked = ROOT / "data" / "provenance" / "sdf" / f"{asset}-512.json"
    if not tracked.is_file():
        raise FileNotFoundError(f"missing tracked SDF provenance: {tracked}")
    local_payload = json.loads(
        loaded.provenance_path.read_text(encoding="utf-8")
    )
    tracked_payload = json.loads(tracked.read_text(encoding="utf-8"))
    tracked_semantic = dict(tracked_payload)
    if tracked_semantic.pop("raw_volume_git_ignored", None) is not True:
        raise ValueError(f"{asset}: tracked receipt lacks git-ignore assertion")
    if tracked_semantic.pop("tracked_provenance_copy", None) is not True:
        raise ValueError(f"{asset}: tracked receipt lacks copy marker")
    if local_payload != tracked_semantic:
        raise ValueError(
            f"{asset}: local and tracked SDF provenance receipts differ"
        )
    output = local_payload["output"]
    grid = local_payload["grid"]
    metadata = {
        "asset_id": asset,
        "canonical_paper_protocol": bool(
            local_payload["canonical_paper_protocol"]
        ),
        "shape": list(loaded.values.shape),
        "dtype": str(loaded.values.dtype),
        "axis_order": grid["axis_order"],
        "query_component_order": grid["query_component_order"],
        "sign_convention": grid["sign_convention"],
        "distance_units": grid["distance_units"],
        "volume_bytes": loaded.volume_path.stat().st_size,
        "volume_sha256": output["checksum"]["value"],
        "checksum_verified": verify_checksum,
        "local_provenance_sha256": hash_file(loaded.provenance_path),
        "tracked_provenance_sha256": hash_file(tracked),
        "negative_fraction": output["negative_fraction"],
        "minimum": output["minimum"],
        "maximum": output["maximum"],
        "max_neighbor_delta_over_spacing": output[
            "max_neighbor_delta_over_spacing"
        ],
        "preprocessor_known_limit": local_payload["known_limit"],
    }
    return LoadedSDF(
        asset=asset,
        values=loaded.values,
        volume_path=loaded.volume_path,
        local_provenance_path=loaded.provenance_path,
        tracked_provenance_path=tracked,
        metadata=MappingProxyType(metadata),
    )


def _synthetic_sphere_volume(resolution: int) -> np.ndarray:
    line = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    z, y, x = np.meshgrid(line, line, line, indexing="ij")
    return (
        np.sqrt(x * x + y * y + z * z) - np.float32(0.6)
    ).astype(np.float32, copy=False)


def load_job_volume(
    config: SDFReproConfig,
    asset: str,
    *,
    processed_root: str | Path | None = None,
    verify_checksum: bool = True,
) -> LoadedSDF:
    if asset == "synthetic-sphere":
        resolution = int(config.evaluation["resolution"])
        values = _synthetic_sphere_volume(resolution)
        return LoadedSDF(
            asset=asset,
            values=values,
            volume_path=None,
            local_provenance_path=None,
            tracked_provenance_path=None,
            metadata=MappingProxyType(
                {
                    "asset_id": asset,
                    "canonical_paper_protocol": False,
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "axis_order": "zyx",
                    "query_component_order": "xyz",
                    "sign_convention": "negative_inside",
                    "distance_units": "analytic centered [-1,1] units",
                    "checksum_verified": False,
                    "synthetic": True,
                }
            ),
        )
    if asset not in PUBLIC_ASSETS:
        raise ValueError(
            f"{asset}: only public SDF assets are runnable; "
            "Pitted Stonefish is deferred_auth_required"
        )
    return _load_public_volume(
        asset,
        processed_root=processed_root,
        verify_checksum=verify_checksum,
    )


def validate_public_volumes(
    *,
    processed_root: str | Path | None = None,
    verify_checksums: bool = True,
) -> dict[str, object]:
    """Validate all three local 512^3 volumes and tracked provenance copies."""

    rows = []
    for asset in PUBLIC_ASSETS:
        loaded = _load_public_volume(
            asset,
            processed_root=processed_root,
            verify_checksum=verify_checksums,
        )
        if tuple(loaded.values.shape) != (512, 512, 512):
            raise AssertionError(f"{asset}: volume is not 512^3")
        if loaded.values.dtype != np.float32:
            raise AssertionError(f"{asset}: volume is not float32")
        rows.append(
            {
                **_plain(loaded.metadata),
                "status": (
                    "checksum_and_provenance_verified"
                    if verify_checksums
                    else "provenance_verified_checksum_skipped"
                ),
                "volume_path": _portable_path(loaded.volume_path),
                "local_provenance_path": _portable_path(
                    loaded.local_provenance_path
                ),
                "tracked_provenance_path": _portable_path(
                    loaded.tracked_provenance_path
                ),
            }
        )
    return {
        "schema": "peps.sdf_volume_validation",
        "schema_version": 1,
        "scope": "public_three_shape_subset",
        "paper": PAPER,
        "checked_at_utc": _utc_now(),
        "checksums_verified": verify_checksums,
        "status": "passed",
        "volumes": rows,
        "stonefish": {
            "asset_id": STONEFISH_ASSET,
            "status": "deferred_auth_required",
            "checked": False,
            "substitution_used": False,
        },
    }


def _volume_tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    array = np.asarray(values)
    if array.dtype != np.float32 or array.ndim != 3:
        raise ValueError("SDF volume must be a rank-3 float32 array")
    if device.type == "cpu" and array.flags.writeable:
        return torch.from_numpy(array)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable",
        )
        tensor = torch.from_numpy(array)
    return tensor.to(device=device)


def _paper_recipe(config: SDFReproConfig, device: torch.device) -> PaperTrainConfig:
    training = config.training
    return PaperTrainConfig(
        task="sdf",
        loss=str(training["loss"]),
        epochs=int(training["epochs"]),
        batches_per_epoch=int(training["batches_per_epoch"]),
        batch_size=int(training["batch_size"]),
        model_lr=float(training["model_lr"]),
        encoder_lr=float(training["encoder_lr"]),
        cosine=bool(training["cosine"]),
        log_every=int(training["log_every"]),
        checkpoint_every=int(training["checkpoint_every"]),
        seed=config.seed,
        mape_epsilon=float(training["mape_epsilon"]),
        adam_beta1=float(training["adam_beta1"]),
        adam_beta2=float(training["adam_beta2"]),
        adam_epsilon=float(training["adam_epsilon"]),
        weight_decay=float(training["weight_decay"]),
        device=device,
    )


def _seed_process(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _checkpoint_payload(
    *,
    job: SDFJob,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    losses: Sequence[Mapping[str, object]],
    elapsed_training_seconds: float,
) -> dict[str, object]:
    return {
        "schema": "peps.sdf_repro_checkpoint",
        "schema_version": 1,
        "job": job.identity,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "coordinate_generator": generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "loss_log": [dict(row) for row in losses],
        "elapsed_training_seconds": elapsed_training_seconds,
    }


def _load_checkpoint(
    path: Path,
    *,
    job: SDFJob,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[int, list[dict[str, object]], float]:
    state = torch.load(path, map_location=device, weights_only=False)
    if (
        state.get("schema") != "peps.sdf_repro_checkpoint"
        or int(state.get("schema_version", 0)) != 1
    ):
        raise ValueError(f"{path}: unsupported SDF checkpoint schema")
    if dict(state.get("job", {})) != job.identity:
        raise ValueError(f"{path}: checkpoint belongs to a different SDF job")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    generator.set_state(state["coordinate_generator"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    step = int(state["step"])
    if not 0 <= step <= job.config.total_steps:
        raise ValueError(f"{path}: checkpoint step is outside the config")
    losses = [dict(row) for row in state.get("loss_log", ())]
    elapsed = float(state.get("elapsed_training_seconds", 0.0))
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError(f"{path}: invalid elapsed training time")
    return step, losses, elapsed


@torch.inference_mode()
def evaluate_sdf_chunked(
    model: nn.Module,
    volume: np.ndarray,
    *,
    device: torch.device,
    chunk_size: int,
    mape_epsilon: float,
    render_protocol: OrthographicRenderProtocol | None = None,
) -> tuple[dict[str, float | int], tuple[np.ndarray, np.ndarray] | None]:
    """Evaluate all voxels with bounded coordinate and prediction buffers."""

    array = np.asarray(volume)
    if array.ndim != 3 or len(set(array.shape)) != 1:
        raise ValueError("SDF evaluation volume must be cubic")
    resolution = int(array.shape[0])
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    slab_depth = max(1, chunk_size // (resolution * resolution))
    accumulator = IoUAccumulator()
    absolute_sum = 0.0
    percentage_sum = 0.0
    value_count = 0
    render = (
        FirstSurfaceAccumulator(resolution)
        if render_protocol is not None
        else None
    )
    model = model.to(device).eval()
    for z_slice, coords in iter_query_slabs(
        resolution,
        slab_depth=slab_depth,
    ):
        parts = []
        for start in range(0, coords.shape[0], chunk_size):
            prediction = model(
                coords[start : start + chunk_size].to(device=device)
            ).cpu()
            if prediction.ndim != 2 or prediction.shape[1] != 1:
                raise ValueError("SDF model must return shape (N, 1)")
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("SDF model produced non-finite predictions")
            parts.append(prediction)
        predicted = torch.cat(parts, dim=0)
        reference = torch.from_numpy(
            np.array(array[z_slice], dtype=np.float32, copy=True)
        ).reshape(-1, 1)
        accumulator.update(predicted < 0, reference < 0)
        difference = (predicted - reference).abs()
        absolute_sum += float(difference.double().sum().item())
        percentage_sum += float(
            (
                difference / reference.abs().clamp_min(mape_epsilon)
            ).double().sum().item()
        )
        value_count += reference.numel()
        if render is not None:
            shape = (
                int(z_slice.stop) - int(z_slice.start),
                resolution,
                resolution,
            )
            render.update(
                z_slice,
                predicted.reshape(shape),
                reference.reshape(shape),
            )
    metrics: dict[str, float | int] = {
        "iou": accumulator.compute(),
        "intersection_voxels": accumulator.intersection,
        "union_voxels": accumulator.union,
        "evaluated_voxels": value_count,
        "l1": absolute_sum / value_count,
        "mape": 100.0 * percentage_sum / value_count,
    }
    rendered = None if render is None else render.render(render_protocol)
    return metrics, rendered


def _render_paths(
    job: SDFJob,
    render_root: Path,
) -> dict[str, Path]:
    directory = render_root / job.config.artifact / job.asset
    method = _safe_component(job.method.name)
    return {
        "reference": directory / "reference.png",
        "prediction": directory / f"{method}.png",
        "flip_map": directory / f"{method}-flip.png",
        "metadata": directory / f"{method}.json",
    }


def _write_render_artifacts(
    job: SDFJob,
    images: tuple[np.ndarray, np.ndarray],
    *,
    protocol: OrthographicRenderProtocol,
    render_root: Path,
) -> dict[str, object]:
    prediction, reference = images
    error_map, mean_error, flip_parameters = evaluate_flip(
        reference,
        prediction,
        protocol,
    )
    paths = _render_paths(job, render_root)
    save_png_atomic(paths["reference"], reference)
    save_png_atomic(paths["prediction"], prediction)
    save_png_atomic(paths["flip_map"], error_map)
    payload = {
        "schema": "peps.sdf_render_validation",
        "schema_version": 1,
        "asset": job.asset,
        "method": job.method.name,
        "loss": job.config.training["loss"],
        "config_sha256": job.config.digest,
        "verification_status": "render_protocol_assumption",
        "paper_camera_available": False,
        "render_protocol": protocol.to_dict(),
        "flip": {
            "mean": mean_error,
            "mode": protocol.flip_mode,
            "parameters": _plain(flip_parameters),
        },
        "files": {key: str(path) for key, path in paths.items() if key != "metadata"},
    }
    atomic_write_json(paths["metadata"], payload)
    return {
        "verification_status": payload["verification_status"],
        "paper_camera_available": False,
        "flip_mean": mean_error,
        "protocol": protocol.to_dict(),
        "files": payload["files"],
        "metadata": str(paths["metadata"]),
    }


def _training_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    recipe: PaperTrainConfig,
) -> torch.Tensor:
    if recipe.loss == "l1":
        return l1_loss(prediction, target)
    return mape_loss(
        prediction,
        target,
        epsilon=recipe.mape_epsilon,
    )


def run_sdf_job(
    job: SDFJob,
    loaded: LoadedSDF,
    volume_on_device: torch.Tensor,
    *,
    device: torch.device,
    work_root: Path,
    render_root: Path,
    rank: int,
    world_size: int,
    stop_after_steps: int | None = None,
    git_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Train, resume, evaluate, and atomically record one independent job."""

    result_path, checkpoint_path = _job_paths(job, work_root)
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        existing_identity = {
            "artifact": existing.get("artifact"),
            "config_sha256": existing.get("config", {}).get("sha256"),
            "asset": existing.get("instance"),
            "method": existing.get("method"),
            "method_key": existing.get("method_key"),
            "loss": existing.get("training", {}).get("loss"),
            "seed": existing.get("seed"),
        }
        if existing.get("status") != "complete" or existing_identity != job.identity:
            raise ValueError(
                f"{result_path}: result belongs to a different or incomplete job"
            )
        return existing

    _seed_process(job.config.seed, device)
    model, counts = build_and_assert_method(job.method)
    model = model.to(device)
    recipe = _paper_recipe(job.config, device)
    optimizer = make_paper_optimizer(model, recipe)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(job.config.seed)
    start_step = 0
    losses: list[dict[str, object]] = []
    previous_elapsed = 0.0
    if checkpoint_path.is_file():
        start_step, losses, previous_elapsed = _load_checkpoint(
            checkpoint_path,
            job=job,
            model=model,
            optimizer=optimizer,
            generator=generator,
            device=device,
        )

    target_step = recipe.total_steps
    if stop_after_steps is not None:
        _require_int(stop_after_steps, "stop_after_steps", minimum=1)
        target_step = min(target_step, start_step + stop_after_steps)
    started = time.perf_counter()
    completed = start_step

    def save_checkpoint(step: int) -> None:
        elapsed = previous_elapsed + (time.perf_counter() - started)
        atomic_torch_save(
            checkpoint_path,
            _checkpoint_payload(
                job=job,
                step=step,
                model=model,
                optimizer=optimizer,
                generator=generator,
                losses=losses,
                elapsed_training_seconds=elapsed,
            ),
        )

    model.train()
    try:
        for step_index in range(start_step, target_step):
            coords = torch.rand(
                recipe.batch_size,
                3,
                generator=generator,
                dtype=torch.float32,
            ).to(device=device)
            target = sample_sdf_tensor(volume_on_device, coords)
            prediction = model(coords)
            loss = _training_loss(prediction, target, recipe)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{job.method.name}/{job.asset}: non-finite training loss"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            completed = step_index + 1
            if completed == 1 or completed % recipe.log_every == 0:
                losses.append(
                    {"step": completed, "loss": float(loss.detach().item())}
                )
            if (
                completed % recipe.checkpoint_every == 0
                or completed == target_step
            ):
                save_checkpoint(completed)
    except KeyboardInterrupt:
        save_checkpoint(completed)
        raise

    if completed < recipe.total_steps:
        return {
            "status": "checkpointed",
            "job": job.identity,
            "step": completed,
            "total_steps": recipe.total_steps,
            "checkpoint": str(checkpoint_path),
        }

    render_protocol = None
    if bool(job.config.render["enabled"]) and job.asset == job.config.render["asset"]:
        render_protocol = OrthographicRenderProtocol.from_mapping(job.config.render)
    metrics, images = evaluate_sdf_chunked(
        model,
        loaded.values,
        device=device,
        chunk_size=int(job.config.evaluation["chunk_size"]),
        mape_epsilon=recipe.mape_epsilon,
        render_protocol=render_protocol,
    )
    render_record = None
    if images is not None:
        assert render_protocol is not None
        render_record = _write_render_artifacts(
            job,
            images,
            protocol=render_protocol,
            render_root=render_root,
        )
        metrics["flip"] = float(render_record["flip_mean"])

    total_elapsed = previous_elapsed + (time.perf_counter() - started)
    record = {
        "schema": "peps.sdf_repro_job",
        "schema_version": 1,
        "status": "complete",
        "paper": PAPER,
        "paper_table": job.config.paper_table,
        "artifact": job.config.artifact,
        "scope": job.config.scope,
        "canonical_four_shape": False,
        "paper_global_comparable": False,
        "instance": job.asset,
        "method": job.method.name,
        "method_key": job.method.key,
        "seed": job.config.seed,
        "job_index": job.index,
        "config": {
            "source": _portable_path(job.config.source),
            "sha256": job.config.digest,
        },
        "parallelism": {
            "mode": "independent_job_modulo",
            "rank": rank,
            "world_size": world_size,
            "same_model_distributed": False,
        },
        "parameters": counts,
        "compression": {
            "signal_values": int(loaded.values.size),
            "factor_total_parameters": loaded.values.size / counts["total"],
            "factor_encoder_parameters": (
                None
                if counts["encoder"] == 0
                else loaded.values.size / counts["encoder"]
            ),
            "paper_reported_factor": (
                227 if job.config.profile == "full" else None
            ),
            "note": (
                "The paper does not state whether decoder parameters enter its "
                "reported factor; both denominators are retained."
            ),
        },
        "training": {
            **_plain(job.config.training),
            "total_steps": recipe.total_steps,
            "resumed_from_step": start_step,
            "coordinate_stream": "CPU torch.Generator, paired by seed",
            "eikonal": False,
            "elapsed_seconds": total_elapsed,
            "loss_log": losses,
        },
        "evaluation": {
            **_plain(job.config.evaluation),
            "metrics": metrics,
            "chunked_full_volume": True,
        },
        "render": render_record,
        "volume": _plain(loaded.metadata),
        "metric_versions": metric_versions(),
        "checkpoint": str(checkpoint_path),
        "git": _plain(git_state or {}),
        "environment": {
            "torch_version": torch.__version__,
            "rocm_version": torch.version.hip,
            "device": str(device),
        },
        "completed_at_utc": _utc_now(),
    }
    atomic_write_json(result_path, record)
    return record


def run_shard(
    configs: Sequence[SDFReproConfig],
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    work_root: Path = DEFAULT_WORK_ROOT,
    render_root: Path = DEFAULT_OUTPUT_ROOT / "renders",
    processed_root: str | Path | None = None,
    verify_checksums: bool = True,
    stop_after_steps: int | None = None,
    instances: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    allow_full_cpu: bool = False,
) -> dict[str, object]:
    """Run one modulo shard while retaining at most one volume on the device."""

    if any(config.status != "runnable" for config in configs):
        raise ValueError("deferred configs cannot be passed to run")
    for config in configs:
        configured_world_size = int(config.sharding["world_size"])
        if configured_world_size != world_size:
            raise ValueError(
                f"{config.artifact}: config world_size={configured_world_size}, "
                f"invocation world_size={world_size}"
            )
    if device.type == "cpu" and any(config.profile == "full" for config in configs):
        if not allow_full_cpu:
            raise ValueError(
                "refusing the 120000-step full matrix on CPU; use the smoke config"
            )
    if stop_after_steps is not None and any(
        config.profile != "smoke" for config in configs
    ):
        raise ValueError("--stop-after-steps is restricted to smoke configs")

    jobs = enumerate_sdf_jobs(configs)
    if instances:
        requested = set(instances)
        known = {job.asset for job in jobs}
        missing = sorted(requested - known)
        if missing:
            raise ValueError(f"unknown requested SDF instances: {missing}")
        jobs = tuple(job for job in jobs if job.asset in requested)
    if methods:
        requested_methods = set(methods)
        known_methods = {job.method.name for job in jobs}
        missing_methods = sorted(requested_methods - known_methods)
        if missing_methods:
            raise ValueError(f"unknown requested SDF methods: {missing_methods}")
        jobs = tuple(job for job in jobs if job.method.name in requested_methods)
    selected = shard_sdf_jobs(jobs, rank=rank, world_size=world_size)
    selected = tuple(
        sorted(
            selected,
            key=lambda job: (job.asset, job.config.artifact, job.method.name),
        )
    )
    git_state = collect_git_state(ROOT)
    records: list[dict[str, object]] = []
    current_asset = None
    current_loaded: LoadedSDF | None = None
    current_tensor: torch.Tensor | None = None
    for job in selected:
        if current_asset != job.asset:
            del current_tensor, current_loaded
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            current_loaded = load_job_volume(
                job.config,
                job.asset,
                processed_root=processed_root,
                verify_checksum=verify_checksums,
            )
            current_tensor = _volume_tensor(current_loaded.values, device)
            current_asset = job.asset
        assert current_loaded is not None and current_tensor is not None
        records.append(
            run_sdf_job(
                job,
                current_loaded,
                current_tensor,
                device=device,
                work_root=work_root,
                render_root=render_root,
                rank=rank,
                world_size=world_size,
                stop_after_steps=stop_after_steps,
                git_state=git_state,
            )
        )
    summary = {
        "schema": "peps.sdf_repro_shard",
        "schema_version": 1,
        "rank": rank,
        "world_size": world_size,
        "parallelism_mode": "independent_job_modulo",
        "selected_jobs": len(selected),
        "complete_jobs": sum(record.get("status") == "complete" for record in records),
        "checkpointed_jobs": sum(
            record.get("status") == "checkpointed" for record in records
        ),
        "jobs": [job.identity for job in selected],
    }
    for config in configs:
        destination = (
            _config_work_dir(config, work_root)
            / f"shard-{rank}-of-{world_size}.json"
        )
        atomic_write_json(destination, summary)
    return summary


def _atomic_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_complete_records(
    config: SDFReproConfig,
    work_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    expected_jobs = enumerate_sdf_jobs((config,))
    records = []
    missing = []
    for job in expected_jobs:
        result_path, _ = _job_paths(job, work_root)
        if not result_path.is_file():
            missing.append(job.identity)
            continue
        record = json.loads(result_path.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            missing.append(job.identity)
            continue
        if record.get("config", {}).get("sha256") != config.digest:
            raise ValueError(f"{result_path}: result config digest mismatch")
        records.append(record)
    return records, missing


def aggregate_config(
    config: SDFReproConfig,
    *,
    work_root: Path = DEFAULT_WORK_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, object]:
    """Write per-shape and explicitly three-shape-only result artifacts."""

    if config.status != "runnable":
        raise ValueError("deferred configs cannot be aggregated")
    records, missing = _read_complete_records(config, work_root)
    if missing:
        raise RuntimeError(
            f"{config.artifact}: {len(missing)} jobs are incomplete; "
            "resume all four shards before aggregation"
        )
    expected_count = len(config.assets) * len(config.methods)
    if len(records) != expected_count:
        raise AssertionError("unexpected SDF record count")

    by_identity = {
        (str(record["instance"]), str(record["method"])): record
        for record in records
    }
    output_dir = output_root / config.artifact
    per_shape_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    for method in config.methods:
        reproduced_values = []
        paper_values = []
        for asset in config.assets:
            record = by_identity[(asset, method.name)]
            metrics = record["evaluation"]["metrics"]
            reproduced = float(metrics["iou"])
            paper_reference = (
                ""
                if asset not in method.paper_iou
                else float(method.paper_iou[asset])
            )
            reproduced_values.append(reproduced)
            if paper_reference != "":
                paper_values.append(float(paper_reference))
            per_shape_rows.append(
                {
                    "schema": "peps.sdf_repro_per_shape",
                    "schema_version": 1,
                    "artifact": config.artifact,
                    "paper_table": config.paper_table,
                    "scope": "three_shape_subset",
                    "training_loss": config.training["loss"],
                    "shape": asset,
                    "method": method.name,
                    "iou": reproduced,
                    "paper_iou": paper_reference,
                    "delta_iou": (
                        ""
                        if paper_reference == ""
                        else reproduced - float(paper_reference)
                    ),
                    "l1": float(metrics["l1"]),
                    "mape": float(metrics["mape"]),
                    "flip": (
                        ""
                        if "flip" not in metrics
                        else float(metrics["flip"])
                    ),
                    "encoder_params": record["parameters"]["encoder"],
                    "decoder_params": record["parameters"]["decoder"],
                    "total_params": record["parameters"]["total"],
                    "seed": config.seed,
                    "paper_global_comparable": False,
                }
            )
        reproduced_mean = sum(reproduced_values) / len(reproduced_values)
        paper_mean = (
            "" if not paper_values else sum(paper_values) / len(paper_values)
        )
        aggregate_rows.append(
            {
                "schema": "peps.sdf_repro_three_shape_aggregate",
                "schema_version": 1,
                "artifact": config.artifact,
                "paper_table": config.paper_table,
                "scope": config.reporting["aggregate_label"],
                "shape_count": len(reproduced_values),
                "method": method.name,
                "mean_iou": reproduced_mean,
                "paper_three_shape_mean_iou": paper_mean,
                "delta_iou": (
                    "" if paper_mean == "" else reproduced_mean - float(paper_mean)
                ),
                "paper_global_comparable": False,
                "omitted_shape": STONEFISH_ASSET,
                "omitted_status": "deferred_auth_required",
            }
        )
    if config.profile == "full":
        if any(row["scope"] != "three_shape_aggregate" for row in aggregate_rows):
            raise AssertionError("full aggregate lost its three-shape label")
        if any(int(row["shape_count"]) != 3 for row in aggregate_rows):
            raise AssertionError("full aggregate is not based on exactly three shapes")

    per_shape_path = output_dir / "per_shape.csv"
    aggregate_path = output_dir / "three_shape_aggregate.csv"
    _atomic_csv(
        per_shape_path,
        tuple(per_shape_rows[0]),
        per_shape_rows,
    )
    _atomic_csv(
        aggregate_path,
        tuple(aggregate_rows[0]),
        aggregate_rows,
    )
    record_files = []
    for job in enumerate_sdf_jobs((config,)):
        result_path, _ = _job_paths(job, work_root)
        record_files.append(
            {
                "job": job.identity,
                "path": str(result_path),
                "sha256": hash_file(result_path),
            }
        )
    manifest = {
        "schema": "peps.sdf_repro_aggregate",
        "schema_version": 1,
        "status": "complete",
        "paper": PAPER,
        "paper_table": config.paper_table,
        "artifact": config.artifact,
        "scope": "three_shape_subset",
        "aggregate_label": config.reporting["aggregate_label"],
        "canonical_four_shape": False,
        "paper_global_comparable": False,
        "assets": list(config.assets),
        "methods": [method.name for method in config.methods],
        "training_loss": config.training["loss"],
        "config": {
            "source": _portable_path(config.source),
            "sha256": config.digest,
        },
        "records": record_files,
        "outputs": {
            "per_shape": str(per_shape_path),
            "three_shape_aggregate": str(aggregate_path),
        },
        "stonefish": {
            "asset_id": STONEFISH_ASSET,
            "status": "deferred_auth_required",
            "substitution_used": False,
        },
        "completed_at_utc": _utc_now(),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_table4_deferred_receipt(
    config: SDFReproConfig,
) -> dict[str, object]:
    """Build a parameter-only receipt; never inspect credentials or data."""

    if config.paper_table != "Table 4" or config.status != "deferred_auth_required":
        raise ValueError("receipt requires the deferred Table 4 config")
    assertions = assert_config_parameter_budgets(config)
    receipt = {
        "schema": "peps.sdf_table4_deferred",
        "schema_version": 1,
        "paper": PAPER,
        "paper_table": "Table 4",
        "status": "deferred_auth_required",
        "scope": "canonical_pitted_stonefish_only",
        "config": {
            "source": _portable_path(config.source),
            "sha256": config.digest,
            "schema": _portable_path(CONFIG_ROOT / "config.schema.json"),
        },
        "asset": {
            "id": STONEFISH_ASSET,
            "canonical_source_uid": config.deferred["canonical_source_uid"],
            "auth_env_names": list(config.deferred["auth_env_names"]),
            "data_access_attempted": False,
        },
        "reason": config.deferred["reason"],
        "substitution": {
            "allowed": False,
            "used": False,
        },
        "numeric_results": {
            "allowed": False,
            "generated": False,
            "paper_values_embedded": False,
        },
        "claims": {
            "table4_reproduced": False,
            "canonical_stonefish_run_completed": False,
        },
        "parameter_assertions": list(assertions),
        "authorized_job_plan": {
            "methods": len(config.methods),
            "encoder_budget_rows": [1, 8],
            "pe_shared_between_rows": True,
            "jobs_if_authorized": 17,
        },
    }
    validate_table4_deferred_receipt(receipt)
    return receipt


def validate_table4_deferred_receipt(receipt: Mapping[str, object]) -> None:
    if receipt.get("schema") != "peps.sdf_table4_deferred":
        raise ValueError("invalid Table 4 receipt schema")
    if receipt.get("schema_version") != 1:
        raise ValueError("unsupported Table 4 receipt schema version")
    if receipt.get("status") != "deferred_auth_required":
        raise ValueError("Table 4 receipt must be deferred_auth_required")
    substitution = receipt.get("substitution")
    numeric = receipt.get("numeric_results")
    claims = receipt.get("claims")
    if not isinstance(substitution, Mapping) or any(substitution.values()):
        raise ValueError("Table 4 receipt cannot allow or use a substitution")
    if not isinstance(numeric, Mapping):
        raise ValueError("Table 4 receipt is missing numeric-results policy")
    if numeric.get("allowed") or numeric.get("generated"):
        raise ValueError("Table 4 receipt cannot contain numeric results")
    if not isinstance(claims, Mapping) or any(claims.values()):
        raise ValueError("Table 4 receipt cannot claim reproduction")
    assertions = receipt.get("parameter_assertions")
    if not isinstance(assertions, list) or len(assertions) != 18:
        raise ValueError("Table 4 receipt must contain 1x/8x assertions for 9 methods")
    if any(row.get("assertion") != "passed" for row in assertions):
        raise ValueError("Table 4 parameter assertion did not pass")


def estimate_cost(configs: Sequence[SDFReproConfig]) -> dict[str, object]:
    runnable = tuple(config for config in configs if config.status == "runnable")
    jobs = enumerate_sdf_jobs(runnable)
    total_steps = sum(job.config.total_steps for job in jobs)
    total_samples = sum(
        job.config.total_steps * int(job.config.training["batch_size"])
        for job in jobs
    )
    evaluated_queries = sum(
        int(job.config.evaluation["resolution"]) ** 3 for job in jobs
    )
    world_size = 4 if any(config.profile == "full" for config in runnable) else 1
    shard_counts = [
        len(shard_sdf_jobs(jobs, rank=rank, world_size=world_size))
        for rank in range(world_size)
    ]
    return {
        "schema": "peps.sdf_repro_cost",
        "schema_version": 1,
        "artifacts": [config.artifact for config in runnable],
        "jobs": len(jobs),
        "optimizer_steps": total_steps,
        "sampled_training_points": total_samples,
        "full_volume_queries": evaluated_queries,
        "full_volume_queries_per_job": {
            config.artifact: int(config.evaluation["resolution"]) ** 3
            for config in runnable
        },
        "armadillo_flip_renders": sum(
            len(config.methods)
            for config in runnable
            if bool(config.render["enabled"])
        ),
        "sharding": {
            "mode": "independent_job_modulo",
            "world_size": world_size,
            "jobs_per_rank": shard_counts,
            "same_model_distributed": False,
        },
        "memory_floor_per_worker": {
            "volume_payload_bytes": 512**3 * 4,
            "note": "Model, Adam state, activations, and framework workspace are additional.",
        },
        "runtime_note": (
            "Wall time is hardware/method dependent. Measure representative "
            "step and 512^3 inference rates before assigning an hour estimate."
        ),
    }


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _config_arguments(values: Sequence[Path] | None) -> tuple[SDFReproConfig, ...]:
    paths = tuple(values) if values else DEFAULT_CONFIGS
    return tuple(load_sdf_repro_config(path) for path in paths)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--config", type=Path, action="append")
    validate.add_argument("--table4-config", type=Path, default=DEFAULT_TABLE4_CONFIG)
    validate.add_argument("--processed-root", type=Path)
    validate.add_argument("--skip-volume-checksums", action="store_true")
    validate.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "volume_validation.json",
    )
    validate.add_argument(
        "--table4-receipt",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "table4_deferred_auth_required.json",
    )

    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, action="append", required=True)
    run.add_argument("--rank", type=int, default=0)
    run.add_argument("--world-size", type=int, default=1)
    run.add_argument("--device", default="auto")
    run.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    run.add_argument(
        "--render-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "renders",
    )
    run.add_argument("--processed-root", type=Path)
    run.add_argument("--skip-volume-checksums", action="store_true")
    run.add_argument("--instance", action="append")
    run.add_argument("--method", action="append")
    run.add_argument("--stop-after-steps", type=int)
    run.add_argument("--allow-full-cpu", action="store_true")

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--config", type=Path, action="append", required=True)
    aggregate.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    aggregate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    receipt = subparsers.add_parser("table4-receipt")
    receipt.add_argument("--config", type=Path, default=DEFAULT_TABLE4_CONFIG)
    receipt.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "table4_deferred_auth_required.json",
    )

    estimate = subparsers.add_parser("estimate")
    estimate.add_argument("--config", type=Path, action="append")

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--config", type=Path, default=DEFAULT_SMOKE_CONFIG)
    smoke.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    smoke.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    smoke.add_argument("--stop-after-steps", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate":
        configs = _config_arguments(arguments.config)
        table4 = load_sdf_repro_config(arguments.table4_config)
        parameter_receipts = {
            config.artifact: list(assert_config_parameter_budgets(config))
            for config in (*configs, table4)
        }
        volume_receipt = validate_public_volumes(
            processed_root=arguments.processed_root,
            verify_checksums=not arguments.skip_volume_checksums,
        )
        table4_receipt = build_table4_deferred_receipt(table4)
        atomic_write_json(arguments.output, volume_receipt)
        atomic_write_json(arguments.table4_receipt, table4_receipt)
        payload = {
            "schema": "peps.sdf_repro_validation",
            "schema_version": 1,
            "status": "passed",
            "configs": {
                config.artifact: {
                    "source": _portable_path(config.source),
                    "sha256": config.digest,
                }
                for config in (*configs, table4)
            },
            "parameter_assertions": parameter_receipts,
            "volumes": volume_receipt,
            "table4": table4_receipt,
            "outputs": {
                "volume_validation": str(arguments.output),
                "table4_receipt": str(arguments.table4_receipt),
            },
        }
        print(json.dumps(_plain(payload), indent=2, sort_keys=True))
        return 0

    if arguments.command == "run":
        configs = tuple(
            load_sdf_repro_config(path) for path in arguments.config
        )
        payload = run_shard(
            configs,
            rank=arguments.rank,
            world_size=arguments.world_size,
            device=_device(arguments.device),
            work_root=arguments.work_root,
            render_root=arguments.render_root,
            processed_root=arguments.processed_root,
            verify_checksums=not arguments.skip_volume_checksums,
            stop_after_steps=arguments.stop_after_steps,
            instances=arguments.instance,
            methods=arguments.method,
            allow_full_cpu=arguments.allow_full_cpu,
        )
        print(json.dumps(_plain(payload), indent=2, sort_keys=True))
        return 0

    if arguments.command == "aggregate":
        outputs = [
            aggregate_config(
                load_sdf_repro_config(path),
                work_root=arguments.work_root,
                output_root=arguments.output_root,
            )
            for path in arguments.config
        ]
        print(json.dumps(_plain(outputs), indent=2, sort_keys=True))
        return 0

    if arguments.command == "table4-receipt":
        receipt = build_table4_deferred_receipt(
            load_sdf_repro_config(arguments.config)
        )
        atomic_write_json(arguments.output, receipt)
        print(json.dumps(_plain(receipt), indent=2, sort_keys=True))
        return 0

    if arguments.command == "estimate":
        configs = _config_arguments(arguments.config)
        print(json.dumps(estimate_cost(configs), indent=2, sort_keys=True))
        return 0

    config = load_sdf_repro_config(arguments.config)
    summary = run_shard(
        (config,),
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        work_root=arguments.work_root,
        render_root=arguments.output_root / "renders",
        verify_checksums=False,
        stop_after_steps=arguments.stop_after_steps,
    )
    if summary["checkpointed_jobs"] == 0:
        aggregate_config(
            config,
            work_root=arguments.work_root,
            output_root=arguments.output_root,
        )
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
