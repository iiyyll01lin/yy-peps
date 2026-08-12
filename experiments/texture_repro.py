"""Texture-specific PEPS Table 2 and Figure 8 reproduction runner.

The generic experiment runner owns model fitting and checkpoint state.  This
module adds the texture-specific contracts that cannot be inferred from a
generic tensor input: the frozen 18-set manifest, dynamic ``R^(3k)`` outputs,
asset/method sharding, the 3F/4F sweep, map-weighted Table 2 aggregation, and
the Paving Stones qualitative/FLIP artifact.

Full profiles refuse CPU execution.  Protocol, status, manifest validation,
report generation, and the synthetic smoke profile are safe CPU operations.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import random
import signal
import shutil
import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from apps.texture.data import bundle_to_coords_targets, load_paper_texture_set
from data.manifest import (
    DEFAULT_RAW_ROOT,
    hash_file,
    load_manifest,
    resolve_local_path,
    texture_set_spec,
    verify_file,
)
from experiments.config import ExperimentConfig, MethodConfig, load_experiment_config
from experiments.runner import (
    ExperimentRunner,
    RunSpec,
    TensorInstance,
    _build_model,
    atomic_torch_save,
    atomic_write_json,
    enumerate_jobs,
    evaluate_metrics,
    summarize_records,
)
from peps.report import collect_git_state
from peps.train import (
    MinibatchStream,
    make_paper_optimizer,
    paper_recipe_from_mapping,
    split_encoder_decoder_parameters,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results"
SCHEMA_VERSION = 1
SEMANTICS = (
    "AO",
    "ARM",
    "DIFF",
    "Displacement",
    "metal",
    "normal",
    "rough",
    "specular",
)
ARTIFACT_CONFIGS = {
    "table2": ROOT / "configs/paper/texture_full.toml",
    "sweep": ROOT / "configs/paper/texture_sweep_full.toml",
    "smoke": ROOT / "configs/paper/texture_smoke.toml",
}
PILOT_CONFIG = ROOT / "configs/paper/texture/convergence_pilot.toml"
_PILOT_STOP_REQUESTED = False
CODE_RECEIPT_PATHS = (
    ROOT / "experiments/texture_repro.py",
    ROOT / "experiments/runner.py",
    ROOT / "apps/texture/build.py",
    ROOT / "apps/texture/data.py",
    ROOT / "peps/train.py",
    ROOT / "peps/projector.py",
    ROOT / "peps/aggregate.py",
    ROOT / "peps/wrapper.py",
    ROOT / "peps/encoders/grid.py",
    ROOT / "peps/encoders/lpe.py",
    ROOT / "peps/encoders/ntc.py",
    ROOT / "peps/models/mlp.py",
    ROOT / "peps/metrics.py",
)
PAPER_TABLE2 = {
    "LPE": {
        "psnr": 40.21,
        "ssim": 0.95,
        "AO": 42.16,
        "ARM": 41.18,
        "DIFF": 36.38,
        "Displacement": 50.34,
        "metal": 46.83,
        "normal": 34.99,
        "rough": 39.05,
        "specular": 44.76,
    },
    "NTC_N": {
        "psnr": 40.20,
        "ssim": 0.95,
        "AO": 42.59,
        "ARM": 41.29,
        "DIFF": 36.78,
        "Displacement": 48.99,
        "metal": 46.07,
        "normal": 34.99,
        "rough": 39.18,
        "specular": 46.13,
    },
    "BI-Grid": {
        "psnr": 41.25,
        "ssim": 0.95,
        "AO": 43.59,
        "ARM": 42.24,
        "DIFF": 37.42,
        "Displacement": 51.18,
        "metal": 47.35,
        "normal": 36.03,
        "rough": 39.98,
        "specular": 46.07,
    },
    "Grid-PEPS4F": {
        "psnr": 41.23,
        "ssim": 0.95,
        "AO": 43.42,
        "ARM": 42.11,
        "DIFF": 37.09,
        "Displacement": 50.81,
        "metal": 49.90,
        "normal": 35.90,
        "rough": 39.67,
        "specular": 47.74,
    },
    "Grid-PinkPEPS4F": {
        "psnr": 41.44,
        "ssim": 0.95,
        "AO": 43.55,
        "ARM": 42.48,
        "DIFF": 37.48,
        "Displacement": 50.64,
        "metal": 49.43,
        "normal": 36.24,
        "rough": 40.10,
        "specular": 47.73,
    },
    "NTC_PEPS": {
        "psnr": 41.79,
        "ssim": 0.95,
        "AO": 44.17,
        "ARM": 42.73,
        "DIFF": 37.91,
        "Displacement": 50.28,
        "metal": 50.47,
        "normal": 36.64,
        "rough": 40.23,
        "specular": 48.56,
    },
    "NTC_PinkPEPS": {
        "psnr": 41.89,
        "ssim": 0.95,
        "AO": 44.34,
        "ARM": 42.79,
        "DIFF": 38.09,
        "Displacement": 50.42,
        "metal": 50.10,
        "normal": 36.71,
        "rough": 40.47,
        "specular": 48.24,
    },
    "Grid-PEPS4F-25": {
        "psnr": 39.86,
        "ssim": 0.93,
        "AO": 42.19,
        "ARM": 40.83,
        "DIFF": 35.86,
        "Displacement": 49.15,
        "metal": 48.63,
        "normal": 34.58,
        "rough": 38.50,
        "specular": 45.41,
    },
    "Grid-PinkPEPS4F-25": {
        "psnr": 40.03,
        "ssim": 0.94,
        "AO": 42.32,
        "ARM": 41.26,
        "DIFF": 36.20,
        "Displacement": 48.83,
        "metal": 48.07,
        "normal": 34.81,
        "rough": 38.83,
        "specular": 45.31,
    },
    "NTC_PEPS-25": {
        "psnr": 40.59,
        "ssim": 0.94,
        "AO": 43.17,
        "ARM": 41.60,
        "DIFF": 36.70,
        "Displacement": 49.65,
        "metal": 49.23,
        "normal": 35.27,
        "rough": 39.10,
        "specular": 46.15,
    },
    "NTC_PinkPEPS-25": {
        "psnr": 40.56,
        "ssim": 0.94,
        "AO": 43.18,
        "ARM": 41.65,
        "DIFF": 36.74,
        "Displacement": 49.61,
        "metal": 47.90,
        "normal": 35.30,
        "rough": 39.19,
        "specular": 46.12,
    },
}
FIGURE8_METHODS = (
    "BI-Grid",
    "Grid-PEPS4F",
    "Grid-PinkPEPS4F",
    "NTC_N",
    "NTC_PEPS",
    "NTC_PinkPEPS",
)
PROTOCOL_ASSUMPTIONS = (
    {
        "code": "texture_file_selection_not_published",
        "detail": (
            "The paper names 18 sets and eight map categories but not exact "
            "source files. data/manifests/textures.json freezes a "
            "non-duplicating 78-map selection."
        ),
    },
    {
        "code": "optimizer_and_seed_not_published",
        "detail": (
            "Adam and seeds 0/1/2 are explicit reproduction choices. The paper "
            "publishes the dual learning rates, cosine schedule, batch size, "
            "epochs, and batches per epoch, but not optimizer or seeds."
        ),
    },
    {
        "code": "sampling_text_is_ambiguous",
        "detail": (
            "The main text says bilinear-filtered targets while the appendix "
            "says random pixel locations. The runner samples random native "
            "pixel locations; at those locations the bilinear target is exactly "
            "the stored texel."
        ),
    },
    {
        "code": "pytorch_latency_not_fused_paper_latency",
        "detail": (
            "The 3F/4F sweep has not been run, so no latency is recorded "
            "anywhere in this artifact; the empty latency columns are absent "
            "measurements, not fast ones. If run, it would time checkpoint "
            "inference in PyTorch, which is not the paper's fused HIP/WMMA "
            "three-channel decoder. results/hip_texture_geometry.json measures "
            "the repository's own fused kernel at the Grid family's decoder "
            "geometry (input widths 119, 153, 45 and 47, matching this "
            "receipt) but with random weights, so it gives cost without "
            "quality. That kernel cannot express the NTC family at all: its "
            "68-channel aggregate exceeds the kernel's 32-channel limit, and "
            "its 12-dimension tiled encoding has no counterpart in the "
            "kernel's aggregation."
        ),
    },
    {
        "code": "figure8_crop_not_published",
        "detail": (
            "The paper states a 100-pixel Paving Stones sample but does not "
            "publish crop coordinates. The generator defaults to a centered "
            "100x100 crop and records the exact rectangle."
        ),
    },
)
PILOT_LIMITATIONS = (
    {
        "code": "bounded_early_schedule_observation",
        "detail": (
            "The pilot stops far before the 120,000-step Table 2 schedule and "
            "preserves that full cosine horizon. It can reject an undersized "
            "budget, but cannot establish convergence unless the measured "
            "curves actually plateau with stable rankings."
        ),
    },
    {
        "code": "deterministic_lattice_evaluation_proxy",
        "detail": (
            "Quality is evaluated on a fixed native-pixel lattice rather than "
            "all 4096x4096 pixels. Per-map PSNR and SSIM retain map semantics "
            "but are runtime-quality pilot proxies, not Table 2 values."
        ),
    },
    {
        "code": "representative_matrix_not_full_table",
        "detail": (
            "Two sets cover both providers and all eight map semantics, and "
            "five methods cover the major architecture families. The omitted "
            "sets and methods prevent promotion to a complete Table 2 result."
        ),
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_identity(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> dict[str, object] | None:
    """Return identity evidence that cannot survive a host reboot or PID reuse."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        return None
    process_root = proc_root / str(pid)
    try:
        stat_text = (process_root / "stat").read_text(encoding="utf-8")
        command = (process_root / "cmdline").read_bytes()
        boot_id = (
            proc_root / "sys/kernel/random/boot_id"
        ).read_text(encoding="utf-8").strip()
        closing_parenthesis = stat_text.rfind(")")
        if closing_parenthesis < 0:
            return None
        # The suffix starts at proc(5) field 3; starttime is field 22.
        stat_fields = stat_text[closing_parenthesis + 1 :].split()
        start_time_ticks = int(stat_fields[19])
        if not command or not boot_id:
            return None
    except (IndexError, OSError, ValueError):
        return None
    return {
        "boot_id": boot_id,
        "start_time_ticks": start_time_ticks,
        "command_sha256": hashlib.sha256(command).hexdigest(),
    }


def _worker_liveness(
    payload: Mapping[str, object],
) -> tuple[bool, dict[str, object]]:
    pid = payload.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        return False, {"status": "not_alive", "reason": "invalid_pid"}
    observed = _process_identity(pid)
    if observed is None:
        return False, {
            "status": "not_alive",
            "reason": "pid_not_present_or_unreadable",
        }
    expected = payload.get("process_identity")
    if not isinstance(expected, Mapping):
        return False, {
            "status": "unverified",
            "reason": "worker_record_has_no_process_identity",
        }
    fields = ("boot_id", "start_time_ticks", "command_sha256")
    if any(expected.get(name) != observed[name] for name in fields):
        return False, {
            "status": "not_alive",
            "reason": "pid_identity_mismatch",
        }
    return True, {
        "status": "verified_alive",
        "reason": "boot_id_start_time_and_command_match",
    }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256(path: Path) -> str:
    return hash_file(path, "sha256")


def _code_receipts() -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in CODE_RECEIPT_PATHS
    ]


@lru_cache(maxsize=1)
def _code_digest() -> str:
    import hashlib

    digest = hashlib.sha256()
    for receipt in _code_receipts():
        digest.update(str(receipt["path"]).encode("utf-8"))
        digest.update(str(receipt["sha256"]).encode("ascii"))
    return digest.hexdigest()


def _config_payload(config: ExperimentConfig) -> dict[str, object]:
    return {
        "schema_version": config.schema_version,
        "name": config.name,
        "paper": config.paper,
        "task": config.task,
        "profile": config.profile,
        "dataset": config.dataset,
        "canonical": config.canonical,
        "seeds": list(config.seeds),
        "training": _plain(config.training),
        "runner": _plain(config.runner),
        "methods": [
            {
                "name": method.name,
                "factory": method.factory,
                "kwargs": _plain(method.kwargs),
                "seeds": None if method.seeds is None else list(method.seeds),
                "role": method.role,
                "training": _plain(method.training),
                "expected_encoder_params": method.expected_encoder_params,
                "expected_total_params": method.expected_total_params,
            }
            for method in config.methods
        ],
        "source": str(config.source),
    }


def _artifact_output(output_root: Path, artifact: str) -> Path:
    config_path = ARTIFACT_CONFIGS[artifact]
    manifest_path = ROOT / "data/manifests/textures.json"
    digest = (
        f"{_sha256(config_path)[:12]}-"
        f"{_sha256(manifest_path)[:12]}-"
        f"{_code_digest()[:12]}"
    )
    return output_root / "work/texture-repro" / artifact / digest


def _safe_component(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )


def _job_paths(
    output_dir: Path,
    instance: str,
    method: str,
    seed: int,
) -> tuple[Path, Path]:
    stem = (
        Path("raw")
        / _safe_component(instance)
        / _safe_component(method)
        / f"seed-{seed}"
    )
    return (
        output_dir / stem.with_suffix(".json"),
        output_dir / "checkpoints" / stem.with_suffix(".pt"),
    )


def _map_metadata(set_spec: Mapping[str, object]) -> list[dict[str, object]]:
    result = []
    for index, texture_map in enumerate(set_spec["maps"]):
        result.append(
            {
                "map_id": texture_map["id"],
                "semantic": texture_map["semantic"],
                "channel_start": 3 * index,
                "channel_stop": 3 * (index + 1),
            }
        )
    return result


def validate_manifest_consumption(
    *,
    verify_files: bool = False,
    decode_size: int | None = None,
) -> dict[str, object]:
    """Validate the complete 18-set/78-map contract without starting training."""

    manifest_path = ROOT / "data/manifests/textures.json"
    manifest = load_manifest(manifest_path)
    semantic_counts: dict[str, int] = defaultdict(int)
    set_rows = []
    file_rows = []
    for texture_set in manifest["sets"]:
        maps = texture_set["maps"]
        for texture_map in maps:
            source = resolve_local_path(DEFAULT_RAW_ROOT, texture_map)
            if verify_files:
                verify_file(source, texture_map)
            semantic_counts[str(texture_map["semantic"])] += 1
            stat = source.stat() if source.is_file() else None
            file_rows.append(
                {
                    "set_id": texture_set["id"],
                    "map_id": texture_map["id"],
                    "semantic": texture_map["semantic"],
                    "path": str(source),
                    "sha256": texture_map["checksum"]["value"],
                    "manifest_bytes": texture_map["bytes"],
                    "mtime_ns": None if stat is None else stat.st_mtime_ns,
                }
            )
        decoded_shape = None
        if decode_size is not None:
            loaded = load_paper_texture_set(
                str(texture_set["id"]),
                size=decode_size,
                verify_checksums=not verify_files,
            )
            decoded_shape = list(loaded.tensor.shape)
            expected = [decode_size, decode_size, 3 * len(maps)]
            if decoded_shape != expected:
                raise AssertionError(
                    f"{texture_set['id']}: decoded {decoded_shape}, expected {expected}"
                )
            del loaded
            gc.collect()
        set_rows.append(
            {
                "id": texture_set["id"],
                "paper_name": texture_set["paper_name"],
                "provider": texture_set["source"]["provider"],
                "map_count": len(maps),
                "output_channels": 3 * len(maps),
                "semantics": [item["semantic"] for item in maps],
                "decoded_shape": decoded_shape,
            }
        )
    if len(set_rows) != 18 or len(file_rows) != 78:
        raise AssertionError("paper texture manifest must contain 18 sets and 78 maps")
    if tuple(name for name in SEMANTICS if semantic_counts[name]) != SEMANTICS:
        raise AssertionError("paper texture manifest does not cover all semantics")
    return {
        "schema": "peps.texture_manifest_validation",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": _sha256(manifest_path),
        "dataset_id": manifest["dataset_id"],
        "set_count": len(set_rows),
        "map_count": len(file_rows),
        "total_output_channels_across_sets": sum(
            int(row["output_channels"]) for row in set_rows
        ),
        "semantic_counts": dict(sorted(semantic_counts.items())),
        "all_native_4k": all(
            item["width"] == 4096 and item["height"] == 4096
            for texture_set in manifest["sets"]
            for item in texture_set["maps"]
        ),
        "verified_files": len(file_rows) if verify_files else 0,
        "decode_size": decode_size,
        "decoded_sets": len(set_rows) if decode_size is not None else 0,
        "sets": set_rows,
        "files": file_rows,
    }


def verification_receipt_is_current(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "peps.texture_manifest_validation":
            return False
        if payload.get("manifest_sha256") != _sha256(
            ROOT / "data/manifests/textures.json"
        ):
            return False
        if int(payload.get("verified_files", 0)) != 78:
            return False
        for item in payload["files"]:
            source = Path(item["path"])
            stat = source.stat()
            if stat.st_size != int(item["manifest_bytes"]):
                return False
            if stat.st_mtime_ns != int(item["mtime_ns"]):
                return False
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True


def _dummy_texture_instances() -> tuple[TensorInstance, ...]:
    manifest = load_manifest("textures")
    return tuple(
        TensorInstance(
            str(texture_set["id"]),
            torch.zeros(1, 2),
            torch.zeros(1, 3 * len(texture_set["maps"])),
            metadata={"texture_maps": _map_metadata(texture_set)},
        )
        for texture_set in manifest["sets"]
    )


def _synthetic_instance() -> TensorInstance:
    height = width = 16
    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, height),
        torch.linspace(0.0, 1.0, width),
        indexing="ij",
    )
    diffuse = torch.stack((x, y, 0.5 * (x + y)), dim=-1)
    normal = torch.stack(
        (
            0.5 + 0.25 * torch.sin(2 * math.pi * x),
            0.5 + 0.25 * torch.cos(2 * math.pi * y),
            torch.ones_like(x),
        ),
        dim=-1,
    ).clamp(0.0, 1.0)
    rough = (0.25 + 0.5 * x * y).unsqueeze(-1).expand(-1, -1, 3)
    texture = torch.cat((diffuse, normal, rough), dim=-1)
    coords, targets, _ = bundle_to_coords_targets(texture)
    return TensorInstance(
        "synthetic-texture",
        coords,
        targets,
        shape=(height, width, 9),
        metadata={
            "num_signal_values": targets.numel(),
            "resolution_xy": [width, height],
            "texture_maps": [
                {
                    "map_id": "diffuse",
                    "semantic": "DIFF",
                    "channel_start": 0,
                    "channel_stop": 3,
                },
                {
                    "map_id": "normal",
                    "semantic": "normal",
                    "channel_start": 3,
                    "channel_stop": 6,
                },
                {
                    "map_id": "rough",
                    "semantic": "rough",
                    "channel_start": 6,
                    "channel_stop": 9,
                },
            ],
        },
    )


def _dummy_instances(config: ExperimentConfig) -> tuple[TensorInstance, ...]:
    if config.profile == "smoke":
        instance = _synthetic_instance()
        return (
            TensorInstance(
                instance.name,
                torch.zeros(1, 2),
                torch.zeros(1, instance.targets.shape[1]),
                metadata=instance.metadata,
            ),
        )
    return _dummy_texture_instances()


def _load_texture_instance(
    set_id: str,
    *,
    verify_checksums: bool,
) -> TensorInstance:
    set_spec = texture_set_spec(set_id)
    loaded = load_paper_texture_set(
        set_id,
        verify_checksums=verify_checksums,
    )
    coords, targets, (height, width) = bundle_to_coords_targets(loaded.tensor)
    maps = [
        {
            "map_id": texture_map.map_id,
            "semantic": texture_map.semantic,
            "channel_start": texture_map.channel_slice.start,
            "channel_stop": texture_map.channel_slice.stop,
            "source_path": str(texture_map.source_path),
        }
        for texture_map in loaded.maps
    ]
    return TensorInstance(
        set_id,
        coords,
        targets,
        shape=(height, width, targets.shape[1]),
        metadata={
            "num_signal_values": targets.numel(),
            "resolution_xy": [width, height],
            "provider": set_spec["source"]["provider"],
            "texture_maps": maps,
        },
    )


def _normalise_method(value: str) -> str:
    name = value.lower().replace("-", "_")
    aliases = {
        "grid_peps": "grid_peps4f",
        "grid_pink_peps": "grid_pinkpeps4f",
        "grid_peps_25": "grid_peps4f_25",
        "grid_pink_peps_25": "grid_pinkpeps4f_25",
        "grid_pink_peps3f": "grid_pinkpeps3f",
        "grid_pink_peps4f": "grid_pinkpeps4f",
        "ntc_peps4f": "ntc_peps",
        "ntc_pink_peps": "ntc_pinkpeps",
        "ntc_pink_peps_25": "ntc_pinkpeps_25",
        "ntc_pinkpeps4f": "ntc_pinkpeps",
        "ntc_pink_peps3f": "ntc_pinkpeps3f",
        "ntc_pink_peps4f": "ntc_pinkpeps",
        "grid_pinkpeps": "grid_pinkpeps4f",
    }
    return aliases.get(name, name)


def _resolution_product(value: object, dim: int = 2) -> int:
    if isinstance(value, int):
        return value**dim
    values = tuple(int(item) for item in value)
    if len(values) != dim:
        raise ValueError(f"resolution must have {dim} entries")
    return math.prod(values)


def _decoder_parameters(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
) -> int:
    if num_layers == 1:
        return input_dim * output_dim + output_dim
    hidden_layers = num_layers - 1
    return (
        input_dim * hidden_dim
        + hidden_dim
        + (hidden_layers - 1) * (hidden_dim * hidden_dim + hidden_dim)
        + hidden_dim * output_dim
        + output_dim
    )


def architecture_receipt(
    method: MethodConfig,
    *,
    output_channels: int,
) -> dict[str, object]:
    """Return exact encoder/decoder widths without allocating 4K grids."""

    values = dict(_plain(method.kwargs))
    name = _normalise_method(str(values.pop("method")))
    hidden_dim = int(values.get("hidden_dim", 64))
    num_layers = int(values.get("num_layers", 4))
    frequencies: int | None = None
    aggregation = "baseline"
    tiled_dim = 0
    if name == "lpe":
        resolution = values.get("resolution", 1024)
        slots = int(values.get("num_frequencies", 4))
        input_dim = 2 * 2 * slots
        encoder_params = _resolution_product(resolution) * input_dim
    elif name == "ntc_n":
        g0_resolution = values.get("g0_resolution", 1024)
        g0_dim = int(values.get("g0_feature_dim", 12))
        g1_resolution = values.get("g1_resolution", 512)
        g1_dim = int(values.get("g1_feature_dim", 20))
        tiled_dim = 2 * int(values.get("num_octaves", 3)) * 2
        input_dim = 4 * g0_dim + g1_dim + tiled_dim
        encoder_params = (
            _resolution_product(g0_resolution) * g0_dim
            + _resolution_product(g1_resolution) * g1_dim
        )
    elif name == "bi_grid":
        resolution = values.get("resolution", 1024)
        feature_dim = int(values.get("feature_dim", 17))
        input_dim = feature_dim
        encoder_params = _resolution_product(resolution) * feature_dim
    elif name.startswith("grid_"):
        reduced = name.endswith("_25")
        resolution = values.get("resolution", 1024)
        feature_dim = int(values.get("feature_dim", 13 if reduced else 17))
        frequencies = int(
            values.get("num_frequencies", 3 if "3f" in name else 4)
        )
        aggregation = "pink" if "pink" in name else "concat"
        if aggregation == "pink":
            widths = [
                max(1, math.floor(feature_dim / (2**index)))
                for index in range(1, frequencies + 1)
            ]
            input_dim = feature_dim + 2 * sum(widths)
        else:
            widths = [feature_dim] * frequencies
            input_dim = (2 * frequencies + 1) * feature_dim
        encoder_params = _resolution_product(resolution) * feature_dim
    elif name.startswith("ntc_"):
        reduced = name.endswith("_25")
        g0_resolution = values.get("g0_resolution", 1024)
        g0_dim = int(values.get("g0_feature_dim", 9 if reduced else 12))
        g1_resolution = values.get("g1_resolution", 512)
        g1_dim = int(values.get("g1_feature_dim", 15 if reduced else 20))
        frequencies = int(
            values.get("num_frequencies", 3 if "3f" in name else 4)
        )
        aggregation = "pink" if "pink" in name else "concat"
        tiled_dim = 2 * int(values.get("num_octaves", 3)) * 2
        grid_dim = 4 * g0_dim + g1_dim
        if aggregation == "pink":
            widths = [
                max(1, math.floor(grid_dim / (2**index)))
                for index in range(1, frequencies + 1)
            ]
            input_dim = grid_dim + 2 * sum(widths) + tiled_dim
        else:
            widths = [grid_dim] * frequencies
            input_dim = (2 * frequencies + 1) * grid_dim + tiled_dim
        encoder_params = (
            _resolution_product(g0_resolution) * g0_dim
            + _resolution_product(g1_resolution) * g1_dim
        )
    else:
        raise ValueError(f"unsupported texture method for receipt: {name}")
    decoder_params = _decoder_parameters(
        input_dim,
        output_channels,
        hidden_dim,
        num_layers,
    )
    if (
        method.expected_encoder_params is not None
        and encoder_params != method.expected_encoder_params
    ):
        raise AssertionError(
            f"{method.name}: receipt encoder parameters {encoder_params} != "
            f"{method.expected_encoder_params}"
        )
    return {
        "builder_method": name,
        "aggregation": aggregation,
        "peps_frequencies": frequencies,
        "tiled_encoding_dim": tiled_dim,
        "decoder_input_dim": input_dim,
        "output_channels": output_channels,
        "hidden_dim": hidden_dim,
        "linear_layers": num_layers,
        "encoder_params": encoder_params,
        "decoder_params": decoder_params,
        "total_params": encoder_params + decoder_params,
    }


def _job_total_steps(config: ExperimentConfig, method: MethodConfig) -> int:
    values = dict(config.training)
    values.update(method.training)
    return paper_recipe_from_mapping(values).total_steps


def _job_batch_size(config: ExperimentConfig, method: MethodConfig) -> int:
    values = dict(config.training)
    values.update(method.training)
    return paper_recipe_from_mapping(values).batch_size


def assigned_job_ranks(
    jobs: Sequence[RunSpec],
    *,
    world_size: int,
) -> dict[int, int]:
    """Keep all seeds of an asset/method pair on one deterministic rank."""

    if world_size < 1:
        raise ValueError("world_size must be positive")
    pair_indices: dict[tuple[str, str], int] = {}
    result = {}
    for job in jobs:
        key = (job.instance.name, job.method.name)
        if key not in pair_indices:
            pair_indices[key] = len(pair_indices)
        result[job.index] = pair_indices[key] % world_size
    return result


def _checkpoint_retained(
    config: ExperimentConfig,
    spec: RunSpec,
) -> bool:
    policy = str(config.runner.get("completed_checkpoint_retention", "all"))
    if policy == "all":
        return True
    if policy == "none":
        return False
    if policy == "figure8_seed0":
        return (
            spec.instance.name
            == str(config.runner.get("figure8_instance", "paving-stones-070"))
            and spec.seed == 0
        )
    raise ValueError(f"unknown completed checkpoint retention policy: {policy}")


def job_plan(
    artifact: str,
    *,
    world_size: int,
    include_jobs: bool = False,
) -> dict[str, object]:
    config = load_experiment_config(ARTIFACT_CONFIGS[artifact])
    instances = _dummy_instances(config)
    jobs = enumerate_jobs(config, instances)
    assignments = assigned_job_ranks(jobs, world_size=world_size)
    total_steps = sum(_job_total_steps(config, job.method) for job in jobs)
    total_samples = sum(
        _job_total_steps(config, job.method)
        * _job_batch_size(config, job.method)
        for job in jobs
    )
    raw_checkpoint_bytes = 0
    retained_checkpoint_bytes = 0
    for job in jobs:
        architecture = architecture_receipt(
            job.method,
            output_channels=int(job.instance.targets.shape[1]),
        )
        estimated = 12 * int(architecture["total_params"])
        raw_checkpoint_bytes += estimated
        if _checkpoint_retained(config, job):
            retained_checkpoint_bytes += estimated
    per_rank = {
        str(rank): {
            "jobs": sum(value == rank for value in assignments.values()),
            "optimizer_steps": sum(
                _job_total_steps(config, job.method)
                for job in jobs
                if assignments[job.index] == rank
            ),
        }
        for rank in range(world_size)
    }
    payload: dict[str, object] = {
        "schema": "peps.texture_job_plan",
        "schema_version": SCHEMA_VERSION,
        "artifact": artifact,
        "paper": config.paper,
        "profile": config.profile,
        "canonical": config.canonical,
        "config": _config_payload(config),
        "config_sha256": _sha256(config.source),
        "manifest_sha256": _sha256(ROOT / "data/manifests/textures.json"),
        "code_digest": _code_digest(),
        "code_receipts": _code_receipts(),
        "parallelism": {
            "mode": "asset_method_job_shard",
            "world_size": world_size,
            "assignment": "asset_method_pair_index_mod_world_size",
            "all_seeds_of_pair_share_rank": True,
            "same_model_distributed": False,
        },
        "instances": len(instances),
        "methods": len(config.methods),
        "seeds": list(config.seeds),
        "expected_jobs": len(jobs),
        "expected_optimizer_steps": total_steps,
        "expected_training_samples": total_samples,
        "per_rank": per_rank,
        "checkpoint_storage_estimate": {
            "bytes_per_parameter": 12,
            "basis": "fp32 weights plus two Adam moment tensors",
            "all_completed_checkpoints_bytes": raw_checkpoint_bytes,
            "retained_completed_checkpoints_bytes": retained_checkpoint_bytes,
            "policy": config.runner.get(
                "completed_checkpoint_retention", "all"
            ),
            "incomplete_jobs_are_always_retained": True,
        },
        "wall_clock_estimate": {
            "formula": (
                "expected_optimizer_steps / measured_aggregate_steps_per_second"
            ),
            "seconds": None,
            "reason": (
                "No texture checkpoint throughput has been measured yet; "
                "CPU smoke timing is not extrapolated to RDNA4 full jobs."
            ),
            "reference_scenarios_not_predictions": [
                {
                    "aggregate_steps_per_second": rate,
                    "wall_days": total_steps / rate / 86_400,
                }
                for rate in (50, 100, 200)
            ],
        },
        "limitations": list(PROTOCOL_ASSUMPTIONS),
    }
    if include_jobs:
        payload["jobs"] = [
            {
                "job_index": job.index,
                "rank": assignments[job.index],
                "instance": job.instance.name,
                "method": job.method.name,
                "seed": job.seed,
                "optimizer_steps": _job_total_steps(config, job.method),
                "output_channels": int(job.instance.targets.shape[1]),
            }
            for job in jobs
        ]
    return payload


def _load_pilot_config() -> ExperimentConfig:
    config = load_experiment_config(PILOT_CONFIG)
    runner = config.runner
    budgets = tuple(int(value) for value in runner.get("step_budgets", ()))
    if config.canonical or config.profile != "full":
        raise ValueError("the convergence pilot must be non-canonical and GPU-only")
    if budgets != (10, 50, 200, 1_000, 2_000, 5_000):
        raise ValueError("pilot step budgets drifted from the bounded extension")
    full_schedule = int(runner.get("full_schedule_steps", 0))
    if budgets[0] < 1 or budgets[-1] >= full_schedule:
        raise ValueError("pilot budgets must be positive and below the full schedule")
    if any(_job_total_steps(config, method) != full_schedule for method in config.methods):
        raise ValueError("pilot methods must preserve the full Table 2 schedule horizon")
    if tuple(config.seeds) != (0, 1, 2):
        raise ValueError("the pilot freezes the Table 2 seed set 0/1/2")
    if int(runner.get("evaluation_side", 0)) < 32:
        raise ValueError("pilot evaluation lattice is unexpectedly small")
    if tuple(int(value) for value in runner.get("physical_devices", ())) != (0, 1):
        raise ValueError("the pilot must reserve only physical GPUs 0 and 1")
    if int(runner.get("maximum_concurrent_workers", 0)) != 2:
        raise ValueError("the pilot must run at most two workers concurrently")
    checkpoint_every = int(runner.get("checkpoint_every", 0))
    if checkpoint_every < 10 or checkpoint_every > 500:
        raise ValueError("pilot checkpoint cadence is outside the resumable bound")
    resume = runner.get("resume_from")
    if not isinstance(resume, Mapping):
        raise ValueError("the pilot extension requires a pinned resume source")
    source_budgets = tuple(int(value) for value in resume.get("step_budgets", ()))
    if (
        source_budgets != budgets[: len(source_budgets)]
        or source_budgets != (10, 50, 200)
        or int(resume.get("maximum_budget", -1)) != source_budgets[-1]
    ):
        raise ValueError("texture resume budgets do not match the 200-step source")
    if int(resume.get("full_schedule_steps", -1)) != full_schedule:
        raise ValueError("texture resume schedule horizon drifted")
    if int(resume.get("evaluation_side", -1)) != int(runner["evaluation_side"]):
        raise ValueError("texture resume evaluation lattice drifted")
    for field in ("manifest", "manifest_sha256", "config_sha256", "code_digest"):
        if not isinstance(resume.get(field), str) or not resume[field]:
            raise ValueError(f"texture resume {field} is missing")
    return config


def _pilot_instance_specs(
    config: ExperimentConfig,
) -> tuple[Mapping[str, object], ...]:
    manifest = load_manifest("textures")
    by_id = {str(item["id"]): item for item in manifest["sets"]}
    selected_ids = tuple(str(value) for value in config.runner["instance_ids"])
    if len(selected_ids) < 2 or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("pilot needs at least two distinct texture sets")
    unknown = sorted(set(selected_ids) - set(by_id))
    if unknown:
        raise ValueError(f"pilot references unknown texture sets: {unknown}")
    selected = tuple(by_id[value] for value in selected_ids)
    covered = {
        str(texture_map["semantic"])
        for texture_set in selected
        for texture_map in texture_set["maps"]
    }
    if covered != set(SEMANTICS):
        missing = sorted(set(SEMANTICS) - covered)
        raise ValueError(f"pilot sets do not cover all map semantics: {missing}")
    providers = {str(item["source"]["provider"]) for item in selected}
    if providers != {"polyhaven", "ambientcg"}:
        raise ValueError("pilot sets must cover Poly Haven and ambientCG")
    return selected


def _pilot_dummy_instances(
    config: ExperimentConfig,
) -> tuple[TensorInstance, ...]:
    return tuple(
        TensorInstance(
            str(item["id"]),
            torch.zeros(1, 2),
            torch.zeros(1, 3 * len(item["maps"])),
            metadata={
                "provider": item["source"]["provider"],
                "texture_maps": _map_metadata(item),
            },
        )
        for item in _pilot_instance_specs(config)
    )


def _pilot_output(output_root: Path) -> Path:
    digest = (
        f"{_sha256(PILOT_CONFIG)[:12]}-"
        f"{_sha256(ROOT / 'data/manifests/textures.json')[:12]}-"
        f"{_code_digest()[:12]}"
    )
    return output_root / "work/texture-repro/convergence-pilot" / digest


def pilot_job_plan(
    *,
    world_size: int = 4,
    include_jobs: bool = False,
) -> dict[str, object]:
    config = _load_pilot_config()
    instances = _pilot_dummy_instances(config)
    jobs = enumerate_jobs(config, instances)
    assignments = assigned_job_ranks(jobs, world_size=world_size)
    budgets = tuple(int(value) for value in config.runner["step_budgets"])
    maximum_budget = budgets[-1]
    total_steps = len(jobs) * maximum_budget
    source_budget = int(config.runner["resume_from"]["maximum_budget"])
    additional_steps = len(jobs) * (maximum_budget - source_budget)
    maximum_trajectories = int(config.runner["max_trajectories"])
    maximum_total_steps = int(config.runner["max_total_optimizer_steps"])
    if len(jobs) > maximum_trajectories or total_steps > maximum_total_steps:
        raise ValueError("pilot matrix exceeds its checked-in safety cap")

    specs = _pilot_instance_specs(config)
    semantic_counts: dict[str, int] = defaultdict(int)
    for item in specs:
        for texture_map in item["maps"]:
            semantic_counts[str(texture_map["semantic"])] += 1

    checkpoint_bytes = 0
    for job in jobs:
        receipt = architecture_receipt(
            job.method,
            output_channels=int(job.instance.targets.shape[1]),
        )
        checkpoint_bytes += 12 * int(receipt["total_params"])

    per_rank = {
        str(rank): {
            "trajectories": sum(
                assignment == rank for assignment in assignments.values()
            ),
            "optimizer_steps": sum(
                maximum_budget
                for job in jobs
                if assignments[job.index] == rank
            ),
        }
        for rank in range(world_size)
    }
    payload: dict[str, object] = {
        "schema": "peps.texture_convergence_pilot_plan",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "paper": config.paper,
        "profile": "bounded_convergence_pilot",
        "canonical": False,
        "config": str(PILOT_CONFIG.relative_to(ROOT)),
        "config_sha256": _sha256(PILOT_CONFIG),
        "texture_manifest": "data/manifests/textures.json",
        "texture_manifest_sha256": _sha256(
            ROOT / "data/manifests/textures.json"
        ),
        "code_digest": _code_digest(),
        "instances": [
            {
                "id": item["id"],
                "provider": item["source"]["provider"],
                "map_count": len(item["maps"]),
                "output_channels": 3 * len(item["maps"]),
                "semantics": [entry["semantic"] for entry in item["maps"]],
            }
            for item in specs
        ],
        "semantic_coverage": {
            name: semantic_counts[name] for name in SEMANTICS
        },
        "methods": [method.name for method in config.methods],
        "seeds": list(config.seeds),
        "step_budgets": list(budgets),
        "full_schedule_steps": int(config.runner["full_schedule_steps"]),
        "evaluation": {
            "side": int(config.runner["evaluation_side"]),
            "sampling": config.runner["evaluation_sampling"],
            "native_resolution": [4096, 4096],
            "metrics": list(config.runner["metrics"]),
            "table2_numeric_comparable": False,
        },
        "parallelism": {
            "mode": "independent_trajectory_job_shard",
            "world_size": world_size,
            "physical_devices": list(config.runner["physical_devices"]),
            "maximum_concurrent_workers": int(
                config.runner["maximum_concurrent_workers"]
            ),
            "assignment": config.runner["assignment"],
            "all_seeds_of_asset_method_pair_share_rank": True,
            "same_model_distributed": False,
        },
        "expected_trajectories": len(jobs),
        "expected_observations": len(jobs) * len(budgets),
        "expected_optimizer_steps": total_steps,
        "source_optimizer_steps": len(jobs) * source_budget,
        "expected_additional_optimizer_steps": additional_steps,
        "expected_training_samples": (
            total_steps * _job_batch_size(config, config.methods[0])
        ),
        "per_rank": per_rank,
        "checkpoint_storage_estimate": {
            "bytes": checkpoint_bytes,
            "bytes_per_parameter": 12,
            "basis": "fp32 weights plus two Adam moment tensors",
            "retention": "one resumable checkpoint per trajectory",
        },
        "safety": {
            "launches_full_table2": False,
            "full_table2_optimizer_steps": 71_280_000,
            "optimizer_step_fraction_of_full_table2": total_steps / 71_280_000,
            "maximum_trajectories": maximum_trajectories,
            "maximum_total_optimizer_steps": maximum_total_steps,
            "max_wall_seconds_per_rank": int(
                config.runner["max_wall_seconds_per_rank"]
            ),
            "requires_gpu": bool(config.runner["require_gpu"]),
            "preferred_architecture": config.runner["preferred_architecture"],
        },
        "resume_from": {
            "manifest": str(config.runner["resume_from"]["manifest"]),
            "manifest_sha256": str(
                config.runner["resume_from"]["manifest_sha256"]
            ),
            "maximum_budget": source_budget,
            "step_budgets": list(
                config.runner["resume_from"]["step_budgets"]
            ),
            "schedule_continuity": (
                "source and extension both retain the uninterrupted "
                "120000-step cosine horizon"
            ),
        },
        "limitations": list(PILOT_LIMITATIONS),
    }
    if include_jobs:
        payload["jobs"] = [
            {
                "job_index": job.index,
                "rank": assignments[job.index],
                "instance": job.instance.name,
                "provider": job.instance.metadata["provider"],
                "method": job.method.name,
                "seed": job.seed,
                "maximum_budget": maximum_budget,
                "output_channels": int(job.instance.targets.shape[1]),
            }
            for job in jobs
        ]
    return payload


def _pilot_source_identity(
    config: ExperimentConfig,
    spec: RunSpec,
    *,
    output_channels: int,
) -> dict[str, object]:
    resume = config.runner["resume_from"]
    return {
        "experiment": config.name,
        "instance": spec.instance.name,
        "method": spec.method.name,
        "seed": spec.seed,
        "job_index": spec.index,
        "config_sha256": str(resume["config_sha256"]),
        "texture_manifest_sha256": _sha256(
            ROOT / "data/manifests/textures.json"
        ),
        "code_digest": str(resume["code_digest"]),
        "step_budgets": [
            int(value) for value in resume["step_budgets"]
        ],
        "full_schedule_steps": int(resume["full_schedule_steps"]),
        "evaluation_side": int(resume["evaluation_side"]),
        "output_channels": output_channels,
    }


def _validate_pilot_resume_source(
    config: ExperimentConfig,
    jobs: Sequence[RunSpec],
) -> dict[str, object]:
    """Validate all manifest-backed 200-step trajectories before extension."""

    resume = config.runner["resume_from"]
    source_manifest_path = ROOT / str(resume["manifest"])
    if _sha256(source_manifest_path) != resume["manifest_sha256"]:
        raise ValueError("texture source manifest digest drift")
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    if source_manifest.get("schema") != "peps.texture_convergence_pilot_manifest":
        raise ValueError("texture source manifest schema drift")
    if source_manifest["config"]["sha256"] != resume["config_sha256"]:
        raise ValueError("texture source config digest drift")
    if source_manifest["code"]["digest"] != resume["code_digest"]:
        raise ValueError("texture source code digest drift")
    if source_manifest["dataset"]["manifest_sha256"] != _sha256(
        ROOT / "data/manifests/textures.json"
    ):
        raise ValueError("texture source dataset manifest drift")
    if source_manifest["plan"]["step_budgets"] != list(
        resume["step_budgets"]
    ):
        raise ValueError("texture source budget drift")
    if source_manifest["claims"] != {
        "full_table2_run": False,
        "paper_numeric_reproduction": False,
        "budget_calibration_only": True,
    }:
        raise ValueError("texture source claim scope drift")
    source_dir = source_manifest_path.parent
    active_workers = 0
    for path in sorted(source_dir.glob("worker-rank-*.json")):
        worker = json.loads(path.read_text(encoding="utf-8"))
        alive, _ = _worker_liveness(worker)
        active_workers += alive
    if active_workers:
        raise RuntimeError("texture source workers are still active")

    source_budget = int(resume["maximum_budget"])
    source_budgets = tuple(int(value) for value in resume["step_budgets"])
    lineages = []
    source_train_seconds = 0.0
    for job in jobs:
        result_path, checkpoint_path = _pilot_job_paths(
            source_dir,
            job.instance.name,
            job.method.name,
            job.seed,
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        identity = _pilot_source_identity(
            config,
            job,
            output_channels=int(job.instance.targets.shape[1]),
        )
        _validate_pilot_result_identity(result, identity)
        observation_steps = tuple(
            int(item["budget_steps"]) for item in result["observations"]
        )
        if (
            result.get("status") != "complete"
            or int(result.get("completed_steps", -1)) != source_budget
            or observation_steps != source_budgets
        ):
            raise ValueError("texture source result is not complete at 200 steps")
        expected_metrics = _expected_metric_keys(config, job.instance)
        if any(
            set(item["metrics"]) != expected_metrics
            or not all(
                math.isfinite(float(value))
                for value in item["metrics"].values()
            )
            for item in result["observations"]
        ):
            raise ValueError("texture source observation metrics are invalid")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if (
            checkpoint.get("schema_version") != 1
            or checkpoint.get("job") != identity
            or int(checkpoint.get("step", -1)) != source_budget
            or int(checkpoint.get("minibatch_stream", {}).get("draws", -1))
            != source_budget
            or checkpoint.get("observations") != result["observations"]
        ):
            raise ValueError("texture source checkpoint identity drift")
        scheduler = checkpoint.get("scheduler")
        if (
            not isinstance(scheduler, Mapping)
            or int(scheduler.get("T_max", -1))
            != int(resume["full_schedule_steps"])
            or int(scheduler.get("last_epoch", -1)) != source_budget
        ):
            raise ValueError("texture source scheduler continuity drift")
        elapsed = float(result["elapsed_train_seconds"])
        source_train_seconds += elapsed
        lineages.append(
            {
                "job_index": job.index,
                "instance": job.instance.name,
                "method": job.method.name,
                "seed": job.seed,
                "source_step": source_budget,
                "source_elapsed_train_seconds": elapsed,
                "result": {
                    "path": str(result_path.relative_to(ROOT)),
                    "bytes": result_path.stat().st_size,
                    "sha256": _sha256(result_path),
                },
                "checkpoint": {
                    "path": str(checkpoint_path.relative_to(ROOT)),
                    "bytes": checkpoint_path.stat().st_size,
                    "sha256": _sha256(checkpoint_path),
                },
            }
        )
        del checkpoint
    if len(lineages) != int(source_manifest["plan"]["expected_trajectories"]):
        raise ValueError("texture source trajectory count drift")
    return {
        "source_manifest": {
            "path": str(resume["manifest"]),
            "bytes": source_manifest_path.stat().st_size,
            "sha256": _sha256(source_manifest_path),
        },
        "source_step_budgets": list(source_budgets),
        "source_maximum_budget": source_budget,
        "source_aggregate_train_seconds": source_train_seconds,
        "active_workers": 0,
        "schedule_continuity": (
            "The source and extension retain the same 120000-step cosine "
            "horizon, optimizer state, minibatch stream, and model state."
        ),
        "jobs": lineages,
    }


def _pilot_manifest_is_current(
    payload: Mapping[str, object],
    *,
    world_size: int,
) -> bool:
    return (
        payload.get("schema") == "peps.texture_convergence_pilot_manifest"
        and payload.get("config", {}).get("sha256") == _sha256(PILOT_CONFIG)
        and payload.get("dataset", {}).get("manifest_sha256")
        == _sha256(ROOT / "data/manifests/textures.json")
        and payload.get("code", {}).get("digest") == _code_digest()
        and payload.get("plan", {}).get("parallelism", {}).get("world_size")
        == world_size
        and isinstance(payload.get("resume_lineage"), Mapping)
    )


def _write_pilot_manifest(
    *,
    output_root: Path,
    world_size: int,
    verification_receipt: Path,
) -> dict[str, object]:
    output_dir = _pilot_output(output_root)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _pilot_manifest_is_current(existing, world_size=world_size):
            raise ValueError("existing texture pilot manifest identity drift")
        return existing
    plan = pilot_job_plan(world_size=world_size, include_jobs=True)
    config = _load_pilot_config()
    jobs = enumerate_jobs(config, _pilot_dummy_instances(config))
    resume_lineage = _validate_pilot_resume_source(config, jobs)
    payload = {
        "schema": "peps.texture_convergence_pilot_manifest",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "artifact": "texture-table2-convergence-pilot",
        "config": {
            "path": str(PILOT_CONFIG.relative_to(ROOT)),
            "sha256": _sha256(PILOT_CONFIG),
        },
        "dataset": {
            "manifest": "data/manifests/textures.json",
            "manifest_sha256": _sha256(
                ROOT / "data/manifests/textures.json"
            ),
            "verification_receipt": str(verification_receipt),
            "verification_receipt_sha256": _sha256(verification_receipt),
            "verification_receipt_current": verification_receipt_is_current(
                verification_receipt
            ),
        },
        "code": {
            "digest": _code_digest(),
            "receipts": _code_receipts(),
        },
        "plan": plan,
        "resume_lineage": resume_lineage,
        "output_dir": str(output_dir),
        "git": collect_git_state(ROOT),
        "claims": {
            "full_table2_run": False,
            "paper_numeric_reproduction": False,
            "budget_calibration_only": True,
        },
    }
    atomic_write_json(manifest_path, payload)
    return payload


def _ensure_pilot_manifest(
    *,
    output_root: Path,
    world_size: int,
    verification_receipt: Path,
) -> dict[str, object]:
    output_dir = _pilot_output(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".manifest.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            return _write_pilot_manifest(
                output_root=output_root,
                world_size=world_size,
                verification_receipt=verification_receipt,
            )
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def pilot_preflight(
    *,
    output_root: Path,
    world_size: int,
    rank: int,
    device: torch.device,
    physical_device_index: int,
    verification_receipt: Path,
    max_wall_seconds: int,
) -> dict[str, object]:
    config = _load_pilot_config()
    plan = pilot_job_plan(world_size=world_size)
    errors = []
    if world_size != int(config.runner["world_size"]):
        errors.append(
            f"pilot requires world_size={config.runner['world_size']}, got {world_size}"
        )
    if not 0 <= rank < world_size:
        errors.append("rank must be in [0, world_size)")
    physical_devices = tuple(
        int(value) for value in config.runner["physical_devices"]
    )
    if physical_device_index not in physical_devices:
        errors.append("physical GPU is outside the bounded 0/1 allocation")
    expected_visibility = str(physical_device_index)
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
        errors.append(
            "HIP and CUDA visibility must explicitly select physical GPU "
            f"{physical_device_index}, while ROCR visibility must be unset to "
            f"avoid ROCm double filtering; observed {visibility}"
        )
    wall_cap = int(config.runner["max_wall_seconds_per_rank"])
    if max_wall_seconds < 1 or max_wall_seconds > wall_cap:
        errors.append(f"wall limit must be in [1, {wall_cap}] seconds")
    receipt_current = verification_receipt_is_current(verification_receipt)
    if not receipt_current:
        errors.append("texture dataset verification receipt is absent or stale")

    device_receipt: dict[str, object] = {"requested": str(device)}
    if device.type != "cuda" or not torch.cuda.is_available():
        errors.append("the native-4K pilot requires a ROCm/CUDA GPU")
    else:
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            errors.append(f"GPU index {index} is unavailable")
        else:
            properties = torch.cuda.get_device_properties(index)
            architecture = getattr(properties, "gcnArchName", None)
            device_receipt.update(
                {
                    "index": index,
                    "name": str(properties.name),
                    "architecture": architecture,
                    "total_memory_bytes": int(properties.total_memory),
                    "preferred_architecture": config.runner[
                        "preferred_architecture"
                    ],
                    "preferred_architecture_match": (
                        architecture == config.runner["preferred_architecture"]
                    ),
                }
            )
            if int(properties.total_memory) < int(
                config.runner["minimum_vram_bytes"]
            ):
                errors.append("GPU VRAM is below the checked-in pilot minimum")

    disk_root = output_root if output_root.exists() else output_root.parent
    disk = shutil.disk_usage(disk_root)
    if disk.free < int(config.runner["minimum_free_disk_bytes"]):
        errors.append("free disk space is below the checked-in pilot minimum")
    full_progress = artifact_progress("table2", output_root=output_root)
    if int(full_progress["active_workers"]) != 0:
        errors.append("full Table 2 workers are active; pilot refuses contention")
    table2_authorization_path = (
        output_root / "texture_repro/table2_launch_authorization.json"
    )
    table2_authorization: dict[str, object] = {
        "path": str(table2_authorization_path),
        "present": table2_authorization_path.is_file(),
        "blocks_pilot": False,
    }
    if table2_authorization_path.is_file():
        try:
            authorization_payload = json.loads(
                table2_authorization_path.read_text(encoding="utf-8")
            )
            blocks_pilot = (
                authorization_payload.get("schema")
                == "peps.texture_table2_launch_authorization"
                and authorization_payload.get("authorized") is True
                and authorization_payload.get("block_other_texture_gpu_work")
                is True
                and authorization_payload.get("table2_complete") is not True
            )
            table2_authorization.update(
                {
                    "blocks_pilot": blocks_pilot,
                    "authorization_id": authorization_payload.get(
                        "authorization_id"
                    ),
                }
            )
            if blocks_pilot:
                errors.append(
                    "full Table 2 has explicit launch authorization; "
                    "convergence pilot is blocked until Table 2 completes"
                )
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot validate Table 2 launch authorization: {exc}")

    payload = {
        "schema": "peps.texture_convergence_pilot_preflight",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "rank": rank,
        "world_size": world_size,
        "status": "passed" if not errors else "refused",
        "errors": errors,
        "config_sha256": plan["config_sha256"],
        "texture_manifest_sha256": plan["texture_manifest_sha256"],
        "dataset_verification": {
            "path": str(verification_receipt),
            "current": receipt_current,
            "sha256": (
                _sha256(verification_receipt)
                if verification_receipt.is_file()
                else None
            ),
        },
        "device": device_receipt,
        "physical_device_index": physical_device_index,
        "visibility": visibility,
        "disk": {
            "path": str(disk_root),
            "free_bytes": disk.free,
            "required_free_bytes": int(
                config.runner["minimum_free_disk_bytes"]
            ),
        },
        "wall_clock": {
            "requested_seconds": max_wall_seconds,
            "hard_cap_seconds": int(
                config.runner["max_wall_seconds_per_rank"]
            ),
        },
        "full_table2_active_workers": full_progress["active_workers"],
        "full_table2_launch_authorization": table2_authorization,
        "bounded_plan": {
            "trajectories": plan["expected_trajectories"],
            "optimizer_steps": plan["expected_optimizer_steps"],
            "full_table2_optimizer_steps": plan["safety"][
                "full_table2_optimizer_steps"
            ],
            "launches_full_table2": False,
        },
    }
    atomic_write_json(
        _pilot_output(output_root) / f"preflight-rank-{rank}.json",
        payload,
    )
    if errors:
        raise RuntimeError("; ".join(errors))
    return payload


def protocol_report(output_root: Path) -> dict[str, object]:
    manifest = validate_manifest_consumption()
    configs = {}
    for artifact, config_path in ARTIFACT_CONFIGS.items():
        config = load_experiment_config(config_path)
        representative_channels = 9 if config.profile == "smoke" else 15
        configs[artifact] = {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
            "profile": config.profile,
            "canonical": config.canonical,
            "methods": [
                {
                    "name": method.name,
                    "role": method.role,
                    **architecture_receipt(
                        method,
                        output_channels=representative_channels,
                    ),
                }
                for method in config.methods
            ],
            "plan_4gpu": job_plan(artifact, world_size=4),
            "output_dir": str(_artifact_output(output_root, artifact)),
        }
    return {
        "schema": "peps.texture_reproduction_protocol",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "paper": {
            "reference": "PEPS Extended arXiv:2604.24167v1",
            "source": "https://arxiv.org/src/2604.24167",
            "table2": PAPER_TABLE2,
            "methods": {
                "grid": "1024^2 x 17; -25 row uses 13 features",
                "lpe": "1024^2 x 16",
                "ntc": (
                    "G0 1024^2 x 12 plus G1 512^2 x 20; -25 row "
                    "uses 9 and 15"
                ),
                "peps_frequencies": 4,
                "mlp": "three hidden layers, width 64, GELU",
                "training": (
                    "L1, grid LR 0.1, MLP LR 0.001, cosine, "
                    "3000 epochs x 40 batches x 60000 samples"
                ),
            },
        },
        "dataset": {
            key: value
            for key, value in manifest.items()
            if key not in {"files", "sets", "generated_at_utc"}
        },
        "dataset_sets": manifest["sets"],
        "configs": configs,
        "figure8": {
            "instance": "paving-stones-070",
            "default_methods": list(FIGURE8_METHODS),
            "crop_size": 100,
            "dependency": "retained final Table 2 seed-0 checkpoints",
            "metric": "official flip_evaluator LDR error map",
        },
        "limitations": list(PROTOCOL_ASSUMPTIONS),
        "git": collect_git_state(ROOT),
        "code_digest": _code_digest(),
        "code_receipts": _code_receipts(),
    }


def _filter_config(
    config: ExperimentConfig,
    methods: Sequence[str] | None,
) -> ExperimentConfig:
    if not methods:
        return config
    requested = set(methods)
    selected = tuple(
        method for method in config.methods if method.name in requested
    )
    missing = sorted(requested - {method.name for method in selected})
    if missing:
        raise ValueError(f"unknown methods for {config.name}: {missing}")
    return replace(config, methods=selected)


@torch.no_grad()
def _benchmark_checkpoint(
    *,
    config: ExperimentConfig,
    spec: RunSpec,
    instance: TensorInstance,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    model, _ = _build_model(config, spec.method, instance)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_steps = _job_total_steps(config, spec.method)
    if int(state["step"]) != expected_steps:
        raise ValueError("latency benchmark requires a final checkpoint")
    model.load_state_dict(state["model"])
    model = model.to(device).eval()
    query_count = int(config.runner.get("latency_queries", 1_048_576))
    side = math.ceil(math.sqrt(query_count))
    axis = torch.linspace(0.0, 1.0, side, device=device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack((x.reshape(-1), y.reshape(-1)), dim=1)[:query_count]
    chunk = int(config.runner.get("latency_chunk", 65_536))
    warmup = int(config.runner.get("latency_warmup", 3))
    repeats = int(config.runner.get("latency_repeats", 7))

    def evaluate_once() -> None:
        for start in range(0, query_count, chunk):
            output = model(coords[start : start + chunk])
            del output

    for _ in range(warmup):
        evaluate_once()
    torch.cuda.synchronize(device)
    seconds = []
    for _ in range(repeats):
        started = time.perf_counter()
        evaluate_once()
        torch.cuda.synchronize(device)
        seconds.append(time.perf_counter() - started)
    ordered = sorted(seconds)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    median = statistics.median(seconds)
    properties = torch.cuda.get_device_properties(device)
    result = {
        "schema": "peps.texture_pytorch_latency",
        "schema_version": SCHEMA_VERSION,
        "instance": spec.instance.name,
        "seed": spec.seed,
        "queries": query_count,
        "output_channels": int(instance.targets.shape[1]),
        "chunk": chunk,
        "warmup": warmup,
        "repeats": repeats,
        "seconds": seconds,
        "median_ms": 1000.0 * median,
        "p95_ms": 1000.0 * p95,
        "million_queries_per_second": query_count / median / 1e6,
        "device": {
            "name": str(properties.name),
            "architecture": getattr(properties, "gcnArchName", None),
        },
        "precision": str(next(model.parameters()).dtype),
        "coordinates_preloaded_on_device": True,
        "checkpoint_load_excluded": True,
        "kernel_fused": False,
        "comparable_to_paper_fused_hip": False,
    }
    del model, state, coords, x, y, axis
    torch.cuda.empty_cache()
    return result


def _finalize_record(
    *,
    config: ExperimentConfig,
    spec: RunSpec,
    instance: TensorInstance,
    record: dict[str, object],
    result_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    verification_receipt: Path | None,
) -> dict[str, object]:
    record["architecture"] = architecture_receipt(
        spec.method,
        output_channels=int(instance.targets.shape[1]),
    )
    record["dataset_verification"] = {
        "manifest_sha256": _sha256(ROOT / "data/manifests/textures.json"),
        "receipt": (
            None if verification_receipt is None else str(verification_receipt)
        ),
        "receipt_current": bool(
            verification_receipt is not None
            and verification_receipt_is_current(verification_receipt)
        ),
    }
    record["texture_parallelism"] = {
        "mode": "asset_method_job_shard",
        "assignment": "asset_method_pair_index_mod_world_size",
        "all_seeds_of_pair_share_rank": True,
        "rank": record["rank"],
        "world_size": record["world_size"],
    }
    record["texture_reproduction"] = {
        "config_sha256": _sha256(config.source),
        "manifest_sha256": _sha256(ROOT / "data/manifests/textures.json"),
        "code_digest": _code_digest(),
    }
    latency_instances = set(config.runner.get("latency_instance_ids", ()))
    should_benchmark = (
        bool(config.runner.get("measure_latency", False))
        and spec.instance.name in latency_instances
        and spec.seed == int(config.runner.get("latency_seed", 0))
    )
    if should_benchmark and "inference_benchmark" not in record:
        record["inference_benchmark"] = _benchmark_checkpoint(
            config=config,
            spec=spec,
            instance=instance,
            checkpoint_path=checkpoint_path,
            device=device,
        )
    retained = _checkpoint_retained(config, spec)
    record["checkpoint"] = {
        "policy": config.runner.get("completed_checkpoint_retention", "all"),
        "retained_after_result": retained,
        "path": str(checkpoint_path) if retained else None,
        "resume_supported_while_incomplete": True,
    }
    atomic_write_json(result_path, record)
    if not retained:
        checkpoint_path.unlink(missing_ok=True)
    return record


def run_artifact(
    artifact: str,
    *,
    output_root: Path,
    rank: int,
    world_size: int,
    device: torch.device,
    instance_ids: Sequence[str] | None,
    methods: Sequence[str] | None,
    force: bool,
    allow_protocol_assumptions: bool,
    verification_receipt: Path | None,
) -> dict[str, object]:
    complete_config = load_experiment_config(ARTIFACT_CONFIGS[artifact])
    if complete_config.profile == "full":
        if not allow_protocol_assumptions:
            raise ValueError(
                "full texture runs require --allow-protocol-assumptions"
            )
        if device.type != "cuda":
            raise ValueError("full texture runs refuse CPU execution")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    selected_config = _filter_config(complete_config, methods)
    dummy_instances = _dummy_instances(complete_config)
    known_instances = {instance.name for instance in dummy_instances}
    selected_instances = (
        known_instances if instance_ids is None else set(instance_ids)
    )
    unknown_instances = sorted(selected_instances - known_instances)
    if unknown_instances:
        raise ValueError(f"unknown texture instances: {unknown_instances}")
    selected_methods = {method.name for method in selected_config.methods}
    complete_jobs = enumerate_jobs(complete_config, dummy_instances)
    assignments = assigned_job_ranks(complete_jobs, world_size=world_size)
    selected_jobs = [
        job
        for job in complete_jobs
        if job.instance.name in selected_instances
        and job.method.name in selected_methods
        and assignments[job.index] == rank
    ]
    output_dir = _artifact_output(output_root, artifact)
    if rank == 0:
        atomic_write_json(
            output_dir / "job-plan.json",
            job_plan(artifact, world_size=world_size, include_jobs=True),
        )
    receipt_current = bool(
        verification_receipt is not None
        and verification_receipt_is_current(verification_receipt)
    )
    worker_path = output_dir / f"worker-rank-{rank}.json"
    worker = {
        "schema": "peps.texture_worker_status",
        "schema_version": SCHEMA_VERSION,
        "artifact": artifact,
        "rank": rank,
        "world_size": world_size,
        "pid": os.getpid(),
        "process_identity": _process_identity(os.getpid()),
        "device": str(device),
        "state": "running",
        "started_at_utc": _utc_now(),
        "selected_instances": sorted(selected_instances),
        "selected_methods": sorted(selected_methods),
        "selected_jobs": len(selected_jobs),
        "verification_receipt": (
            None if verification_receipt is None else str(verification_receipt)
        ),
        "verification_receipt_current": receipt_current,
        "output_dir": str(output_dir),
    }
    atomic_write_json(worker_path, worker)
    runner = ExperimentRunner(
        complete_config,
        output_dir,
        rank=rank,
        world_size=world_size,
        local_rank=0,
        device=device,
        force=force,
    )
    records = []
    try:
        by_instance: dict[str, list[RunSpec]] = defaultdict(list)
        for job in selected_jobs:
            by_instance[job.instance.name].append(job)
        for instance_name in sorted(by_instance):
            if complete_config.profile == "smoke":
                instance = _synthetic_instance()
            else:
                instance = _load_texture_instance(
                    instance_name,
                    verify_checksums=not receipt_current,
                )
            for planned in by_instance[instance_name]:
                spec = RunSpec(
                    instance=instance,
                    method=planned.method,
                    seed=planned.seed,
                    index=planned.index,
                )
                record = runner.run_one(spec)
                result_path, checkpoint_path = _job_paths(
                    output_dir,
                    instance.name,
                    spec.method.name,
                    spec.seed,
                )
                records.append(
                    _finalize_record(
                        config=complete_config,
                        spec=spec,
                        instance=instance,
                        record=record,
                        result_path=result_path,
                        checkpoint_path=checkpoint_path,
                        device=device,
                        verification_receipt=verification_receipt,
                    )
                )
            del instance
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        atomic_write_json(
            output_dir / f"summary-rank-{rank}.json",
            summarize_records(records),
        )
    except BaseException as exc:
        worker.update(
            {
                "state": (
                    "interrupted"
                    if isinstance(exc, KeyboardInterrupt)
                    else "failed"
                ),
                "finished_at_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_records": len(records),
            }
        )
        atomic_write_json(worker_path, worker)
        raise
    worker.update(
        {
            "state": "complete",
            "finished_at_utc": _utc_now(),
            "completed_records": len(records),
        }
    )
    atomic_write_json(worker_path, worker)
    return {
        "artifact": artifact,
        "rank": rank,
        "world_size": world_size,
        "records": len(records),
        "output_dir": str(output_dir),
    }


def _pilot_job_paths(
    output_dir: Path,
    instance: str,
    method: str,
    seed: int,
) -> tuple[Path, Path]:
    stem = (
        Path("raw")
        / _safe_component(instance)
        / _safe_component(method)
        / f"seed-{seed}"
    )
    return (
        output_dir / stem.with_suffix(".json"),
        output_dir / "checkpoints" / stem.with_suffix(".pt"),
    )


def _pilot_evaluation_instance(
    instance: TensorInstance,
    *,
    side: int,
) -> TensorInstance:
    if instance.shape is None or len(instance.shape) != 3:
        raise ValueError("pilot texture instance needs an HWC shape")
    height, width, channels = (int(value) for value in instance.shape)
    if side > min(height, width):
        raise ValueError("pilot evaluation side exceeds native texture size")
    y_indices = torch.linspace(0, height - 1, side).round().to(torch.long)
    x_indices = torch.linspace(0, width - 1, side).round().to(torch.long)
    yy, xx = torch.meshgrid(y_indices, x_indices, indexing="ij")
    flat = (yy * width + xx).reshape(-1)
    coords = torch.stack(
        (
            xx.reshape(-1).to(torch.float32) / (width - 1),
            yy.reshape(-1).to(torch.float32) / (height - 1),
        ),
        dim=1,
    )
    targets = instance.targets.index_select(0, flat)
    return TensorInstance(
        f"{instance.name}-pilot-lattice-{side}",
        coords,
        targets,
        shape=(side, side, channels),
        metadata=instance.metadata,
    )


@torch.no_grad()
def _evaluate_pilot_checkpoint(
    model: torch.nn.Module,
    evaluation: TensorInstance,
    *,
    device: torch.device,
    chunk: int,
    metric_names: Sequence[str],
) -> tuple[dict[str, float], float]:
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    predictions = []
    for start in range(0, evaluation.coords.shape[0], chunk):
        predictions.append(
            model(evaluation.coords[start : start + chunk].to(device)).cpu()
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    prediction = torch.cat(predictions, dim=0)
    metrics = evaluate_metrics(
        "texture",
        metric_names,
        evaluation,
        prediction,
    )
    model.train()
    return metrics, elapsed


def _set_pilot_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pilot_parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    encoder, decoder = split_encoder_decoder_parameters(model)
    encoder_count = sum(parameter.numel() for parameter in encoder)
    decoder_count = sum(parameter.numel() for parameter in decoder)
    return {
        "encoder": encoder_count,
        "decoder": decoder_count,
        "total": encoder_count + decoder_count,
    }


def _pilot_job_identity(
    config: ExperimentConfig,
    spec: RunSpec,
    *,
    output_channels: int,
) -> dict[str, object]:
    return {
        "experiment": config.name,
        "instance": spec.instance.name,
        "method": spec.method.name,
        "seed": spec.seed,
        "job_index": spec.index,
        "config_sha256": _sha256(PILOT_CONFIG),
        "texture_manifest_sha256": _sha256(
            ROOT / "data/manifests/textures.json"
        ),
        "code_digest": _code_digest(),
        "step_budgets": [
            int(value) for value in config.runner["step_budgets"]
        ],
        "full_schedule_steps": int(config.runner["full_schedule_steps"]),
        "evaluation_side": int(config.runner["evaluation_side"]),
        "output_channels": output_channels,
    }


def _validate_pilot_result_identity(
    payload: Mapping[str, object],
    identity: Mapping[str, object],
) -> None:
    if payload.get("identity") != identity:
        raise ValueError("pilot result belongs to a different trajectory")
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise ValueError("pilot result has no observation list")
    steps = [int(item["budget_steps"]) for item in observations]
    if steps != sorted(set(steps)):
        raise ValueError("pilot result observations are duplicated or unordered")


def _pilot_resume_lineage_for_job(
    output_dir: Path,
    spec: RunSpec,
) -> Mapping[str, object]:
    manifest_path = output_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [
        item
        for item in payload["resume_lineage"]["jobs"]
        if int(item["job_index"]) == spec.index
    ]
    if len(matches) != 1:
        raise ValueError("texture resume lineage does not uniquely identify the job")
    lineage = matches[0]
    if (
        lineage["instance"] != spec.instance.name
        or lineage["method"] != spec.method.name
        or int(lineage["seed"]) != spec.seed
    ):
        raise ValueError("texture resume lineage job identity drift")
    return lineage


def _run_pilot_trajectory(
    *,
    config: ExperimentConfig,
    spec: RunSpec,
    output_dir: Path,
    device: torch.device,
    verification_receipt: Path,
    deadline: float,
) -> dict[str, object]:
    result_path, checkpoint_path = _pilot_job_paths(
        output_dir,
        spec.instance.name,
        spec.method.name,
        spec.seed,
    )
    identity = _pilot_job_identity(
        config,
        spec,
        output_channels=int(spec.instance.targets.shape[1]),
    )
    existing: dict[str, object] | None = None
    if result_path.is_file():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            _validate_pilot_result_identity(existing, identity)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"existing pilot result is incomplete or corrupt: {result_path}"
            ) from exc
        if existing.get("status") == "complete":
            return existing

    _set_pilot_seed(spec.seed)
    model, _ = _build_model(config, spec.method, spec.instance)
    parameters = _pilot_parameter_counts(model)
    if (
        spec.method.expected_encoder_params is not None
        and parameters["encoder"] != spec.method.expected_encoder_params
    ):
        raise AssertionError(
            f"{spec.method.name}: encoder parameters {parameters['encoder']} != "
            f"{spec.method.expected_encoder_params}"
        )
    architecture = architecture_receipt(
        spec.method,
        output_channels=int(spec.instance.targets.shape[1]),
    )
    recipe = replace(
        paper_recipe_from_mapping(config.training),
        seed=spec.seed,
        device=device,
    )
    if recipe.total_steps != int(config.runner["full_schedule_steps"]):
        raise ValueError("pilot recipe does not preserve the full schedule horizon")

    model = model.to(device)
    optimizer = make_paper_optimizer(model, recipe)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=recipe.total_steps,
        )
        if recipe.cosine
        else None
    )
    stream = MinibatchStream(
        spec.instance.coords.shape[0],
        recipe.batch_size,
        recipe.seed,
    )
    observations = list(existing.get("observations", [])) if existing else []
    completed_step = int(existing.get("completed_steps", 0)) if existing else 0
    elapsed_train = (
        float(existing.get("elapsed_train_seconds", 0.0)) if existing else 0.0
    )
    interval_train = 0.0
    last_observation_step = (
        int(observations[-1]["budget_steps"]) if observations else 0
    )
    resume_lineage: Mapping[str, object] | None = None
    imported_source = False

    if checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("job") != identity:
            raise ValueError(
                f"checkpoint belongs to a different pilot job: {checkpoint_path}"
            )
        checkpoint_observations = list(checkpoint.get("observations", []))
        if observations and checkpoint_observations[: len(observations)] != observations:
            raise ValueError("pilot result and checkpoint observations disagree")
        observations = checkpoint_observations
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            if checkpoint.get("scheduler") is None:
                raise ValueError("pilot checkpoint is missing scheduler state")
            scheduler.load_state_dict(checkpoint["scheduler"])
        stream.load_state_dict(checkpoint["minibatch_stream"])
        completed_step = int(checkpoint["step"])
        elapsed_train = float(checkpoint.get("elapsed_train_seconds", 0.0))
        interval_train = float(
            checkpoint.get("interval_train_seconds", 0.0)
        )
        last_observation_step = int(
            checkpoint.get("last_observation_step", last_observation_step)
        )
        resume_lineage = checkpoint.get("resume_lineage")
        if not isinstance(resume_lineage, Mapping):
            raise ValueError("extended texture checkpoint lost its resume lineage")
    elif existing is not None and completed_step:
        raise ValueError("partial pilot result exists without its checkpoint")
    else:
        resume_lineage = _pilot_resume_lineage_for_job(output_dir, spec)
        source_result_path = ROOT / str(resume_lineage["result"]["path"])
        source_checkpoint_path = ROOT / str(
            resume_lineage["checkpoint"]["path"]
        )
        if _sha256(source_result_path) != resume_lineage["result"]["sha256"]:
            raise ValueError("texture source result changed after preflight")
        if (
            _sha256(source_checkpoint_path)
            != resume_lineage["checkpoint"]["sha256"]
        ):
            raise ValueError("texture source checkpoint changed after preflight")
        source_result = json.loads(
            source_result_path.read_text(encoding="utf-8")
        )
        source_identity = _pilot_source_identity(
            config,
            spec,
            output_channels=int(spec.instance.targets.shape[1]),
        )
        _validate_pilot_result_identity(source_result, source_identity)
        checkpoint = torch.load(
            source_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if (
            checkpoint.get("job") != source_identity
            or int(checkpoint.get("step", -1))
            != int(config.runner["resume_from"]["maximum_budget"])
            or checkpoint.get("observations")
            != source_result["observations"]
        ):
            raise ValueError("texture source checkpoint/result continuity drift")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            if checkpoint.get("scheduler") is None:
                raise ValueError("texture source checkpoint has no scheduler")
            scheduler.load_state_dict(checkpoint["scheduler"])
        stream.load_state_dict(checkpoint["minibatch_stream"])
        observations = list(checkpoint["observations"])
        completed_step = int(checkpoint["step"])
        elapsed_train = float(checkpoint["elapsed_train_seconds"])
        interval_train = float(
            checkpoint.get("interval_train_seconds", 0.0)
        )
        last_observation_step = int(
            checkpoint.get(
                "last_observation_step",
                observations[-1]["budget_steps"],
            )
        )
        imported_source = True
        del checkpoint

    budgets = tuple(int(value) for value in config.runner["step_budgets"])
    maximum_budget = budgets[-1]
    if completed_step > maximum_budget:
        raise ValueError("pilot checkpoint exceeds the maximum bounded budget")
    observation_steps = {int(item["budget_steps"]) for item in observations}
    if any(step > completed_step for step in observation_steps):
        raise ValueError("pilot observation is ahead of its checkpoint")

    evaluation = _pilot_evaluation_instance(
        spec.instance,
        side=int(config.runner["evaluation_side"]),
    )
    training_coords = spec.instance.coords.to(device)
    training_targets = spec.instance.targets.to(device)
    metric_names = tuple(str(value) for value in config.runner["metrics"])
    render_chunk = int(config.runner["render_chunk"])
    map_receipt = [
        {
            "map_id": item["map_id"],
            "semantic": item["semantic"],
            "channel_start": item["channel_start"],
            "channel_stop": item["channel_stop"],
        }
        for item in spec.instance.metadata["texture_maps"]
    ]

    def checkpoint_payload() -> dict[str, object]:
        return {
            "schema_version": 1,
            "step": completed_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": (
                None if scheduler is None else scheduler.state_dict()
            ),
            "minibatch_stream": stream.state_dict(),
            "job": identity,
            "observations": observations,
            "elapsed_train_seconds": elapsed_train,
            "interval_train_seconds": interval_train,
            "last_observation_step": last_observation_step,
            "resume_lineage": resume_lineage,
        }

    def result_payload(status: str) -> dict[str, object]:
        properties = torch.cuda.get_device_properties(device)
        return {
            "schema": "peps.texture_convergence_pilot_trajectory",
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "identity": identity,
            "paper": config.paper,
            "profile": "bounded_convergence_pilot",
            "canonical": False,
            "instance": spec.instance.name,
            "provider": spec.instance.metadata["provider"],
            "method": spec.method.name,
            "role": spec.method.role,
            "seed": spec.seed,
            "job_index": spec.index,
            "parameters": parameters,
            "architecture": architecture,
            "maps": map_receipt,
            "training": {
                **_plain(config.training),
                "full_schedule_steps": recipe.total_steps,
                "maximum_pilot_steps": maximum_budget,
                "cosine_horizon_preserved": True,
            },
            "evaluation": {
                "side": int(config.runner["evaluation_side"]),
                "sampling": config.runner["evaluation_sampling"],
                "metrics": list(metric_names),
                "table2_numeric_comparable": False,
            },
            "completed_steps": completed_step,
            "elapsed_train_seconds": elapsed_train,
            "source_elapsed_train_seconds": float(
                resume_lineage["source_elapsed_train_seconds"]
            ),
            "extension_elapsed_train_seconds": max(
                0.0,
                elapsed_train
                - float(resume_lineage["source_elapsed_train_seconds"]),
            ),
            "observations": observations,
            "resume_lineage": resume_lineage,
            "dataset_verification": {
                "texture_manifest_sha256": identity[
                    "texture_manifest_sha256"
                ],
                "receipt": str(verification_receipt),
                "receipt_sha256": _sha256(verification_receipt),
                "receipt_current": verification_receipt_is_current(
                    verification_receipt
                ),
            },
            "device": {
                "name": str(properties.name),
                "architecture": getattr(properties, "gcnArchName", None),
                "index": device.index,
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "retained": True,
                "resume_supported": True,
                "step": completed_step,
            },
            "updated_at_utc": _utc_now(),
            "limitations": list(PILOT_LIMITATIONS),
        }

    def record_observation() -> None:
        nonlocal interval_train, last_observation_step
        if completed_step in observation_steps:
            return
        metrics, evaluation_seconds = _evaluate_pilot_checkpoint(
            model,
            evaluation,
            device=device,
            chunk=render_chunk,
            metric_names=metric_names,
        )
        step_delta = completed_step - last_observation_step
        observations.append(
            {
                "budget_steps": completed_step,
                "step_delta": step_delta,
                "interval_train_seconds": interval_train,
                "cumulative_train_seconds": elapsed_train,
                "interval_steps_per_second": (
                    step_delta / interval_train
                    if interval_train > 0
                    else None
                ),
                "evaluation_seconds": evaluation_seconds,
                "learning_rates": {
                    str(group.get("group_name", index)): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
                "metrics": metrics,
            }
        )
        observation_steps.add(completed_step)
        last_observation_step = completed_step
        interval_train = 0.0

    if imported_source:
        atomic_torch_save(checkpoint_path, checkpoint_payload())
        atomic_write_json(result_path, result_payload("partial"))

    if completed_step in budgets and completed_step not in observation_steps:
        record_observation()
        atomic_torch_save(checkpoint_path, checkpoint_payload())
        status = "complete" if completed_step == maximum_budget else "partial"
        payload = result_payload(status)
        atomic_write_json(result_path, payload)
        if status == "complete":
            del training_coords, training_targets, model, optimizer
            torch.cuda.empty_cache()
            return payload

    check_every = 10
    checkpoint_every = int(config.runner["checkpoint_every"])
    timing_started = time.perf_counter()
    while completed_step < maximum_budget:
        model.train()
        indices = stream.next().to(device=device)
        prediction = model(training_coords.index_select(0, indices))
        loss = torch.nn.functional.l1_loss(
            prediction,
            training_targets.index_select(0, indices),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        completed_step += 1

        should_measure = (
            completed_step in budgets
            or completed_step % check_every == 0
            or completed_step == maximum_budget
            or _PILOT_STOP_REQUESTED
        )
        if not should_measure:
            continue
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        measured = time.perf_counter() - timing_started
        elapsed_train += measured
        interval_train += measured
        timing_started = time.perf_counter()

        if completed_step in budgets:
            record_observation()
            atomic_torch_save(checkpoint_path, checkpoint_payload())
            status = (
                "complete" if completed_step == maximum_budget else "partial"
            )
            payload = result_payload(status)
            atomic_write_json(result_path, payload)
            if status == "complete":
                del training_coords, training_targets, model, optimizer
                torch.cuda.empty_cache()
                return payload

        if (
            completed_step % checkpoint_every == 0
            and completed_step not in budgets
        ):
            atomic_torch_save(checkpoint_path, checkpoint_payload())
            atomic_write_json(result_path, result_payload("partial"))

        if _PILOT_STOP_REQUESTED or time.monotonic() >= deadline:
            atomic_torch_save(checkpoint_path, checkpoint_payload())
            payload = result_payload(
                "interrupted" if _PILOT_STOP_REQUESTED else "wall_clock_limited"
            )
            payload["stop_reason"] = (
                "signal_requested"
                if _PILOT_STOP_REQUESTED
                else "wall_clock_deadline"
            )
            atomic_write_json(result_path, payload)
            del training_coords, training_targets, model, optimizer
            torch.cuda.empty_cache()
            return payload

    raise AssertionError("pilot trajectory loop exited without a receipt")


def run_convergence_pilot(
    *,
    output_root: Path,
    rank: int,
    world_size: int,
    device: torch.device,
    physical_device_index: int,
    verification_receipt: Path,
    max_wall_seconds: int,
    allow_protocol_assumptions: bool,
) -> dict[str, object]:
    global _PILOT_STOP_REQUESTED
    _PILOT_STOP_REQUESTED = False

    def request_stop(_signum, _frame) -> None:
        global _PILOT_STOP_REQUESTED
        _PILOT_STOP_REQUESTED = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if not allow_protocol_assumptions:
        raise ValueError(
            "texture pilot requires --allow-protocol-assumptions"
        )
    config = _load_pilot_config()
    started_monotonic = time.monotonic()
    deadline = started_monotonic + max_wall_seconds
    preflight = pilot_preflight(
        output_root=output_root,
        world_size=world_size,
        rank=rank,
        device=device,
        physical_device_index=physical_device_index,
        verification_receipt=verification_receipt,
        max_wall_seconds=max_wall_seconds,
    )
    output_dir = _pilot_output(output_root)
    _ensure_pilot_manifest(
        output_root=output_root,
        world_size=world_size,
        verification_receipt=verification_receipt,
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dummy_instances = _pilot_dummy_instances(config)
    jobs = enumerate_jobs(config, dummy_instances)
    assignments = assigned_job_ranks(jobs, world_size=world_size)
    selected_jobs = [
        job for job in jobs if assignments[job.index] == rank
    ]
    worker_path = output_dir / f"worker-rank-{rank}.json"
    worker = {
        "schema": "peps.texture_convergence_pilot_worker",
        "schema_version": SCHEMA_VERSION,
        "rank": rank,
        "world_size": world_size,
        "pid": os.getpid(),
        "process_identity": _process_identity(os.getpid()),
        "state": "running",
        "started_at_utc": _utc_now(),
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
        "selected_trajectories": len(selected_jobs),
        "max_wall_seconds": max_wall_seconds,
        "preflight": preflight,
        "output_dir": str(output_dir),
    }
    atomic_write_json(worker_path, worker)
    completed = 0
    wall_limited = 0
    attempted = 0
    try:
        by_instance: dict[str, list[RunSpec]] = defaultdict(list)
        for job in selected_jobs:
            by_instance[job.instance.name].append(job)
        for instance_name in sorted(by_instance):
            if _PILOT_STOP_REQUESTED or time.monotonic() >= deadline:
                wall_limited += len(by_instance[instance_name])
                break
            instance = _load_texture_instance(
                instance_name,
                verify_checksums=False,
            )
            for planned in by_instance[instance_name]:
                if _PILOT_STOP_REQUESTED or time.monotonic() >= deadline:
                    wall_limited += 1
                    break
                spec = RunSpec(
                    instance=instance,
                    method=planned.method,
                    seed=planned.seed,
                    index=planned.index,
                )
                attempted += 1
                payload = _run_pilot_trajectory(
                    config=config,
                    spec=spec,
                    output_dir=output_dir,
                    device=device,
                    verification_receipt=verification_receipt,
                    deadline=deadline,
                )
                if payload["status"] == "complete":
                    completed += 1
                else:
                    wall_limited += 1
                    if _PILOT_STOP_REQUESTED:
                        break
            del instance
            gc.collect()
            torch.cuda.empty_cache()
            if _PILOT_STOP_REQUESTED:
                break
    except BaseException as exc:
        worker.update(
            {
                "state": (
                    "interrupted"
                    if isinstance(exc, KeyboardInterrupt)
                    else "failed"
                ),
                "finished_at_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "attempted_trajectories": attempted,
                "completed_trajectories": completed,
            }
        )
        atomic_write_json(worker_path, worker)
        raise

    state = (
        "complete"
        if completed == len(selected_jobs)
        else "interrupted"
        if _PILOT_STOP_REQUESTED
        else "wall_clock_limited"
    )
    worker.update(
        {
            "state": state,
            "finished_at_utc": _utc_now(),
            "attempted_trajectories": attempted,
            "completed_trajectories": completed,
            "wall_clock_limited_trajectories": wall_limited,
            "elapsed_wall_seconds": time.monotonic() - started_monotonic,
        }
    )
    atomic_write_json(worker_path, worker)
    summary = {
        "schema": "peps.texture_convergence_pilot_rank_summary",
        "schema_version": SCHEMA_VERSION,
        "rank": rank,
        "world_size": world_size,
        "state": state,
        "selected_trajectories": len(selected_jobs),
        "attempted_trajectories": attempted,
        "completed_trajectories": completed,
        "wall_clock_limited_trajectories": wall_limited,
        "elapsed_wall_seconds": worker["elapsed_wall_seconds"],
        "output_dir": str(output_dir),
    }
    atomic_write_json(output_dir / f"summary-rank-{rank}.json", summary)
    return summary


def _expected_metric_keys(
    config: ExperimentConfig,
    instance: TensorInstance,
) -> set[str]:
    names = tuple(config.runner.get("metrics", ()))
    maps = tuple(instance.metadata.get("texture_maps", ()))
    semantics = {str(item["semantic"]) for item in maps}
    keys = set(names)
    for name in names:
        keys.update(
            f"{name}/map/{item['map_id']}/{item['semantic']}" for item in maps
        )
        keys.update(f"{name}/semantic/{semantic}" for semantic in semantics)
    return keys


def _validate_result_record(
    path: Path,
    *,
    config: ExperimentConfig,
    job: RunSpec,
) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["instance"] != job.instance.name:
            raise ValueError("instance mismatch")
        if payload["method"] != job.method.name:
            raise ValueError("method mismatch")
        if int(payload["seed"]) != job.seed:
            raise ValueError("seed mismatch")
        if int(payload["job_index"]) != job.index:
            raise ValueError("global job index mismatch")
        if int(payload["training"]["total_steps"]) != _job_total_steps(
            config, job.method
        ):
            raise ValueError("optimizer-step budget mismatch")
        if set(payload["metrics"]) != _expected_metric_keys(config, job.instance):
            raise ValueError("texture metric key set mismatch")
        if not all(
            math.isfinite(float(value)) for value in payload["metrics"].values()
        ):
            raise ValueError("non-finite metric")
        if (
            job.method.expected_encoder_params is not None
            and int(payload["parameters"]["encoder"])
            != job.method.expected_encoder_params
        ):
            raise ValueError("encoder parameter budget mismatch")
        architecture = payload["architecture"]
        if int(architecture["output_channels"]) != job.instance.targets.shape[1]:
            raise ValueError("dynamic output width mismatch")
        if payload["parallelism"]["mode"] != "job_shard":
            raise ValueError("missing generic job-shard provenance")
        if payload["texture_parallelism"]["mode"] != "asset_method_job_shard":
            raise ValueError("missing texture asset/method shard provenance")
        if payload["texture_reproduction"] != {
            "config_sha256": _sha256(config.source),
            "manifest_sha256": _sha256(ROOT / "data/manifests/textures.json"),
            "code_digest": _code_digest(),
        }:
            raise ValueError("texture reproduction digest mismatch")
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _checkpoint_step(path: Path, total_steps: int) -> tuple[int, str | None]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        step = int(payload["step"])
    except Exception as exc:
        return 0, f"{type(exc).__name__}: {exc}"
    if not 0 <= step <= total_steps:
        return 0, f"checkpoint step {step} outside [0, {total_steps}]"
    return step, None


def _pilot_checkpoint_step(
    path: Path,
    total_steps: int,
    identity: Mapping[str, object],
) -> tuple[int, str | None]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        step = int(payload["step"])
        if payload.get("job") != identity:
            raise ValueError("pilot checkpoint identity mismatch")
        if int(payload.get("minibatch_stream", {}).get("draws", -1)) != step:
            raise ValueError("pilot checkpoint stream/step mismatch")
        if not isinstance(payload.get("resume_lineage"), Mapping):
            raise ValueError("pilot checkpoint has no source resume lineage")
    except Exception as exc:
        return 0, f"{type(exc).__name__}: {exc}"
    if not 0 <= step <= total_steps:
        return 0, f"checkpoint step {step} outside [0, {total_steps}]"
    return step, None


def artifact_progress(
    artifact: str,
    *,
    output_root: Path,
) -> dict[str, object]:
    config = load_experiment_config(ARTIFACT_CONFIGS[artifact])
    output_dir = _artifact_output(output_root, artifact)
    jobs = enumerate_jobs(config, _dummy_instances(config))
    plan_path = output_dir / "job-plan.json"
    world_size = int(config.runner.get("world_size", 4))
    if plan_path.is_file():
        try:
            world_size = int(
                json.loads(plan_path.read_text(encoding="utf-8"))[
                    "parallelism"
                ]["world_size"]
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    assignments = assigned_job_ranks(jobs, world_size=world_size)
    completed = 0
    checkpointed = 0
    accounted_steps = 0
    expected_steps = 0
    result_errors = []
    checkpoint_errors = []
    per_rank = {
        rank: {"expected_jobs": 0, "completed_jobs": 0, "optimizer_steps": 0}
        for rank in range(world_size)
    }
    for job in jobs:
        total = _job_total_steps(config, job.method)
        expected_steps += total
        rank = assignments[job.index]
        per_rank[rank]["expected_jobs"] += 1
        result_path, checkpoint_path = _job_paths(
            output_dir,
            job.instance.name,
            job.method.name,
            job.seed,
        )
        if result_path.is_file():
            error = _validate_result_record(
                result_path,
                config=config,
                job=job,
            )
            if error is None:
                completed += 1
                accounted_steps += total
                per_rank[rank]["completed_jobs"] += 1
                per_rank[rank]["optimizer_steps"] += total
                continue
            result_errors.append(
                {
                    "instance": job.instance.name,
                    "method": job.method.name,
                    "seed": job.seed,
                    "error": error,
                }
            )
        if checkpoint_path.is_file():
            step, error = _checkpoint_step(checkpoint_path, total)
            accounted_steps += step
            per_rank[rank]["optimizer_steps"] += step
            if step:
                checkpointed += 1
            if error is not None:
                checkpoint_errors.append(
                    {
                        "instance": job.instance.name,
                        "method": job.method.name,
                        "seed": job.seed,
                        "error": error,
                    }
                )
    workers = []
    for path in sorted(output_dir.glob("worker-rank-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            process_alive, evidence = _worker_liveness(payload)
            payload["process_alive"] = process_alive
            payload["liveness_evidence"] = evidence
            if payload.get("state") == "running" and not process_alive:
                payload["effective_state"] = "stale"
            else:
                payload["effective_state"] = payload.get("state")
        except (OSError, json.JSONDecodeError) as exc:
            payload = {
                "path": str(path),
                "state": "unreadable",
                "effective_state": "unreadable",
                "error": f"{type(exc).__name__}: {exc}",
                "process_alive": False,
                "liveness_evidence": {
                    "status": "unverified",
                    "reason": "worker_record_unreadable",
                },
            }
        workers.append(payload)
    started = []
    for worker in workers:
        value = worker.get("started_at_utc")
        if isinstance(value, str):
            try:
                started.append(datetime.fromisoformat(value))
            except ValueError:
                pass
    observed = None
    if started and accounted_steps:
        elapsed = (datetime.now(timezone.utc) - min(started)).total_seconds()
        if elapsed > 0:
            rate = accounted_steps / elapsed
            observed = {
                "aggregate_checkpointed_steps_per_second": rate,
                "naive_remaining_seconds_at_current_job_mix": (
                    (expected_steps - accounted_steps) / rate
                ),
                "warning": (
                    "Not a guaranteed ETA: method/output widths differ and "
                    "progress appears only at checkpoint boundaries."
                ),
            }
    return {
        "artifact": artifact,
        "profile": config.profile,
        "canonical": config.canonical,
        "config": str(config.source.relative_to(ROOT)),
        "config_sha256": _sha256(config.source),
        "output_dir": str(output_dir),
        "world_size": world_size,
        "expected_jobs": len(jobs),
        "completed_jobs": completed,
        "checkpointed_incomplete_jobs": checkpointed,
        "job_completion_fraction": completed / len(jobs),
        "expected_optimizer_steps": expected_steps,
        "accounted_optimizer_steps": accounted_steps,
        "optimizer_step_completion_fraction": (
            accounted_steps / expected_steps if expected_steps else 0.0
        ),
        "per_rank": {
            str(rank): values for rank, values in sorted(per_rank.items())
        },
        "workers": workers,
        "active_workers": sum(
            worker.get("state") == "running"
            and bool(worker.get("process_alive"))
            for worker in workers
        ),
        "throughput_observation": observed,
        "result_errors": result_errors,
        "checkpoint_errors": checkpoint_errors,
        "complete": completed == len(jobs),
    }


def pilot_progress(
    *,
    output_root: Path,
) -> dict[str, object]:
    config = _load_pilot_config()
    output_dir = _pilot_output(output_root)
    jobs = enumerate_jobs(config, _pilot_dummy_instances(config))
    world_size = int(config.runner["world_size"])
    assignments = assigned_job_ranks(jobs, world_size=world_size)
    budgets = tuple(int(value) for value in config.runner["step_budgets"])
    expected_metric_keys = {
        job.index: _expected_metric_keys(config, job.instance) for job in jobs
    }
    completed = 0
    checkpointed = 0
    accounted_steps = 0
    observations = 0
    result_errors = []
    checkpoint_errors = []
    per_rank = {
        rank: {
            "expected_trajectories": 0,
            "completed_trajectories": 0,
            "optimizer_steps": 0,
        }
        for rank in range(world_size)
    }
    for job in jobs:
        rank = assignments[job.index]
        per_rank[rank]["expected_trajectories"] += 1
        result_path, checkpoint_path = _pilot_job_paths(
            output_dir,
            job.instance.name,
            job.method.name,
            job.seed,
        )
        identity = _pilot_job_identity(
            config,
            job,
            output_channels=int(job.instance.targets.shape[1]),
        )
        result_step = 0
        result_complete = False
        if result_path.is_file():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                _validate_pilot_result_identity(payload, identity)
                result_step = int(payload["completed_steps"])
                if not 0 <= result_step <= budgets[-1]:
                    raise ValueError("completed step is outside the pilot budget")
                seen_steps = []
                for observation in payload["observations"]:
                    step = int(observation["budget_steps"])
                    if step not in budgets or step > result_step:
                        raise ValueError("observation step is outside the receipt")
                    metrics = observation["metrics"]
                    if set(metrics) != expected_metric_keys[job.index]:
                        raise ValueError("observation metric key set mismatch")
                    if not all(
                        math.isfinite(float(value))
                        for value in metrics.values()
                    ):
                        raise ValueError("observation contains a non-finite metric")
                    seen_steps.append(step)
                observations += len(seen_steps)
                result_complete = (
                    payload["status"] == "complete"
                    and result_step == budgets[-1]
                    and tuple(seen_steps) == budgets
                )
            except Exception as exc:
                result_errors.append(
                    {
                        "instance": job.instance.name,
                        "method": job.method.name,
                        "seed": job.seed,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                result_step = 0
        checkpoint_step = 0
        if checkpoint_path.is_file():
            checkpoint_step, error = _pilot_checkpoint_step(
                checkpoint_path,
                budgets[-1],
                identity,
            )
            if error is not None:
                checkpoint_errors.append(
                    {
                        "instance": job.instance.name,
                        "method": job.method.name,
                        "seed": job.seed,
                        "error": error,
                    }
                )
                checkpoint_step = 0
            elif checkpoint_step:
                checkpointed += 1
        if result_complete and checkpoint_step != budgets[-1]:
            checkpoint_errors.append(
                {
                    "instance": job.instance.name,
                    "method": job.method.name,
                    "seed": job.seed,
                    "error": "complete result lacks a matching final checkpoint",
                }
            )
            result_complete = False
        effective_step = max(result_step, checkpoint_step)
        accounted_steps += effective_step
        per_rank[rank]["optimizer_steps"] += effective_step
        if result_complete:
            completed += 1
            per_rank[rank]["completed_trajectories"] += 1

    workers = []
    for path in sorted(output_dir.glob("worker-rank-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            process_alive, evidence = _worker_liveness(payload)
            payload["process_alive"] = process_alive
            payload["liveness_evidence"] = evidence
            payload["effective_state"] = (
                "stale"
                if payload.get("state") == "running" and not process_alive
                else payload.get("state")
            )
        except (OSError, json.JSONDecodeError) as exc:
            payload = {
                "path": str(path),
                "state": "unreadable",
                "effective_state": "unreadable",
                "process_alive": False,
                "error": f"{type(exc).__name__}: {exc}",
                "liveness_evidence": {
                    "status": "unverified",
                    "reason": "worker_record_unreadable",
                },
            }
        workers.append(payload)

    manifest_path = output_dir / "manifest.json"
    manifest_current = False
    manifest_sha256 = None
    if manifest_path.is_file():
        try:
            manifest_payload = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            manifest_current = (
                _pilot_manifest_is_current(
                    manifest_payload,
                    world_size=world_size,
                )
                and manifest_payload["plan"]["expected_trajectories"]
                == len(jobs)
            )
            manifest_sha256 = _sha256(manifest_path)
        except (OSError, KeyError, json.JSONDecodeError):
            manifest_current = False

    expected_steps = len(jobs) * budgets[-1]
    source_steps = len(jobs) * int(
        config.runner["resume_from"]["maximum_budget"]
    )
    expected_additional_steps = expected_steps - source_steps
    current_identity = _process_identity(os.getpid())
    return {
        "schema": "peps.texture_convergence_pilot_progress",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "artifact": "texture-table2-convergence-pilot",
        "output_dir": str(output_dir),
        "config": str(PILOT_CONFIG.relative_to(ROOT)),
        "config_sha256": _sha256(PILOT_CONFIG),
        "texture_manifest_sha256": _sha256(
            ROOT / "data/manifests/textures.json"
        ),
        "code_digest": _code_digest(),
        "boot_id": (
            None if current_identity is None else current_identity["boot_id"]
        ),
        "world_size": world_size,
        "physical_devices": list(config.runner["physical_devices"]),
        "maximum_concurrent_workers": int(
            config.runner["maximum_concurrent_workers"]
        ),
        "step_budgets": list(budgets),
        "expected_trajectories": len(jobs),
        "completed_trajectories": completed,
        "checkpointed_trajectories": checkpointed,
        "expected_observations": len(jobs) * len(budgets),
        "observations": observations,
        "expected_optimizer_steps": expected_steps,
        "accounted_optimizer_steps": accounted_steps,
        "source_optimizer_steps": source_steps,
        "expected_additional_optimizer_steps": expected_additional_steps,
        "accounted_additional_optimizer_steps": max(
            0, accounted_steps - source_steps
        ),
        "optimizer_step_completion_fraction": (
            accounted_steps / expected_steps if expected_steps else 0.0
        ),
        "per_rank": {
            str(rank): values for rank, values in sorted(per_rank.items())
        },
        "workers": workers,
        "active_workers": sum(
            worker.get("state") == "running"
            and bool(worker.get("process_alive"))
            for worker in workers
        ),
        "manifest": {
            "path": str(manifest_path),
            "present": manifest_path.is_file(),
            "current": manifest_current,
            "sha256": manifest_sha256,
        },
        "result_errors": result_errors,
        "checkpoint_errors": checkpoint_errors,
        "complete": (
            completed == len(jobs)
            and not result_errors
            and not checkpoint_errors
            and manifest_current
        ),
    }


def _collect_valid_records(
    config: ExperimentConfig,
    output_dir: Path,
) -> list[dict[str, object]]:
    """Read only expected records that pass the complete job contract."""

    records = []
    for job in enumerate_jobs(config, _dummy_instances(config)):
        result_path, _ = _job_paths(
            output_dir,
            job.instance.name,
            job.method.name,
            job.seed,
        )
        if not result_path.is_file():
            continue
        error = _validate_result_record(
            result_path,
            config=config,
            job=job,
        )
        if error is None:
            records.append(json.loads(result_path.read_text(encoding="utf-8")))
    return records


def _map_observations(
    records: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str], list[float]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in records:
        method = str(record["method"])
        for key, value in record["metrics"].items():
            if "/map/" not in key:
                continue
            metric, suffix = str(key).split("/map/", 1)
            _, semantic = suffix.rsplit("/", 1)
            grouped[(method, metric, semantic)].append(float(value))
            grouped[(method, metric, "global")].append(float(value))
    return grouped


def _method_record_groups(
    records: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    result: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        result[str(record["method"])].append(record)
    return result


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _table2_rows(
    config: ExperimentConfig,
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    observations = _map_observations(records)
    by_method = _method_record_groups(records)
    manifest = load_manifest("textures")
    expected_jobs = len(manifest["sets"]) * len(config.seeds)
    expected_maps = sum(len(item["maps"]) for item in manifest["sets"]) * len(
        config.seeds
    )
    rows = []
    for method in config.methods:
        method_records = by_method.get(method.name, [])
        encoder_counts = {
            int(record["parameters"]["encoder"]) for record in method_records
        }
        decoder_counts = [
            int(record["parameters"]["decoder"]) for record in method_records
        ]
        total_counts = [
            int(record["parameters"]["total"]) for record in method_records
        ]
        compression = [
            float(record["compression_factor"]) for record in method_records
        ]
        psnr = observations.get((method.name, "psnr", "global"), [])
        ssim = observations.get((method.name, "ssim", "global"), [])
        complete = (
            len(method_records) == expected_jobs
            and len(psnr) == expected_maps
            and len(ssim) == expected_maps
        )
        row: dict[str, object] = {
            "method": method.name,
            "profile": config.profile,
            "complete": complete,
            "job_count": len(method_records),
            "expected_job_count": expected_jobs,
            "map_observation_count": len(psnr),
            "expected_map_observation_count": expected_maps,
            "psnr": _mean(psnr),
            "ssim": _mean(ssim),
            "encoder_params": (
                next(iter(encoder_counts)) if len(encoder_counts) == 1 else None
            ),
            "decoder_params_min": min(decoder_counts) if decoder_counts else None,
            "decoder_params_max": max(decoder_counts) if decoder_counts else None,
            "total_params_min": min(total_counts) if total_counts else None,
            "total_params_max": max(total_counts) if total_counts else None,
            "compression_factor_mean": _mean(compression),
            "verification_status": (
                "complete_protocol_assumption"
                if complete
                else "partial_do_not_interpret"
            ),
        }
        for semantic in SEMANTICS:
            row[semantic] = _mean(
                observations.get((method.name, "psnr", semantic), [])
            )
        paper = PAPER_TABLE2[method.name]
        row["paper_psnr"] = paper["psnr"]
        row["paper_ssim"] = paper["ssim"]
        row["delta_psnr"] = (
            None if row["psnr"] is None else float(row["psnr"]) - paper["psnr"]
        )
        row["delta_ssim"] = (
            None if row["ssim"] is None else float(row["ssim"]) - paper["ssim"]
        )
        rows.append(row)
    return rows


TABLE2_COLUMNS = (
    "method",
    "profile",
    "complete",
    "job_count",
    "expected_job_count",
    "map_observation_count",
    "expected_map_observation_count",
    "psnr",
    "ssim",
    *SEMANTICS,
    "paper_psnr",
    "paper_ssim",
    "delta_psnr",
    "delta_ssim",
    "encoder_params",
    "decoder_params_min",
    "decoder_params_max",
    "total_params_min",
    "total_params_max",
    "compression_factor_mean",
    "verification_status",
)


def _atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            writer.writerows(
                {column: row.get(column) for column in columns} for row in rows
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _pairwise_rank_agreement(
    left: Sequence[str],
    right: Sequence[str],
) -> float | None:
    common = sorted(set(left) & set(right))
    if len(common) < 2:
        return None
    left_rank = {name: index for index, name in enumerate(left)}
    right_rank = {name: index for index, name in enumerate(right)}
    agreements = 0
    comparisons = 0
    for index, first in enumerate(common):
        for second in common[index + 1 :]:
            comparisons += 1
            if (
                (left_rank[first] - left_rank[second])
                * (right_rank[first] - right_rank[second])
                > 0
            ):
                agreements += 1
    return agreements / comparisons if comparisons else None


def _collect_pilot_records(
    config: ExperimentConfig,
    output_dir: Path,
) -> list[dict[str, object]]:
    records = []
    for job in enumerate_jobs(config, _pilot_dummy_instances(config)):
        result_path, _ = _pilot_job_paths(
            output_dir,
            job.instance.name,
            job.method.name,
            job.seed,
        )
        if not result_path.is_file():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        identity = _pilot_job_identity(
            config,
            job,
            output_channels=int(job.instance.targets.shape[1]),
        )
        _validate_pilot_result_identity(payload, identity)
        records.append(payload)
    return records


def write_pilot_report(
    *,
    output_root: Path,
    output_path: Path,
    csv_path: Path,
) -> dict[str, object]:
    config = _load_pilot_config()
    output_dir = _pilot_output(output_root)
    progress = pilot_progress(output_root=output_root)
    records = _collect_pilot_records(config, output_dir)
    budgets = tuple(int(value) for value in config.runner["step_budgets"])
    manifest_path = output_dir / "manifest.json"
    manifest_payload = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    jobs_by_index = {
        job.index: job
        for job in enumerate_jobs(config, _pilot_dummy_instances(config))
    }
    checkpoint_receipts = []
    for record in records:
        job = jobs_by_index[int(record["job_index"])]
        _, checkpoint_path = _pilot_job_paths(
            output_dir,
            job.instance.name,
            job.method.name,
            job.seed,
        )
        identity = _pilot_job_identity(
            config,
            job,
            output_channels=int(job.instance.targets.shape[1]),
        )
        step, error = _pilot_checkpoint_step(
            checkpoint_path,
            budgets[-1],
            identity,
        )
        if error is not None:
            raise ValueError(
                f"cannot receipt texture checkpoint {checkpoint_path}: {error}"
            )
        checkpoint_receipts.append(
            {
                "job_index": job.index,
                "instance": job.instance.name,
                "method": job.method.name,
                "seed": job.seed,
                "step": step,
                "path": str(checkpoint_path.resolve().relative_to(ROOT)),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": _sha256(checkpoint_path),
            }
        )

    raw_rows = []
    for record in records:
        for observation in record["observations"]:
            metrics = observation["metrics"]
            row = {
                "instance": record["instance"],
                "provider": record["provider"],
                "method": record["method"],
                "seed": record["seed"],
                "budget_steps": observation["budget_steps"],
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "interval_train_seconds": observation[
                    "interval_train_seconds"
                ],
                "cumulative_train_seconds": observation[
                    "cumulative_train_seconds"
                ],
                "interval_steps_per_second": observation[
                    "interval_steps_per_second"
                ],
                "evaluation_seconds": observation["evaluation_seconds"],
                "completed_trajectory": record["status"] == "complete",
            }
            for semantic in SEMANTICS:
                row[f"psnr_{semantic}"] = metrics.get(
                    f"psnr/semantic/{semantic}"
                )
                row[f"ssim_{semantic}"] = metrics.get(
                    f"ssim/semantic/{semantic}"
                )
            raw_rows.append(row)
    raw_rows.sort(
        key=lambda row: (
            int(row["budget_steps"]),
            str(row["method"]),
            str(row["instance"]),
            int(row["seed"]),
        )
    )
    columns = (
        "instance",
        "provider",
        "method",
        "seed",
        "budget_steps",
        "psnr",
        "ssim",
        *(f"psnr_{semantic}" for semantic in SEMANTICS),
        *(f"ssim_{semantic}" for semantic in SEMANTICS),
        "interval_train_seconds",
        "cumulative_train_seconds",
        "interval_steps_per_second",
        "evaluation_seconds",
        "completed_trajectory",
    )
    _atomic_csv(csv_path, raw_rows, columns)

    grouped: dict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(int(row["budget_steps"]), str(row["method"]))].append(row)
    expected_per_curve = len(_pilot_instance_specs(config)) * len(config.seeds)
    curves = []
    rankings: dict[int, list[str]] = {}
    for budget in budgets:
        budget_rows = []
        for method in config.methods:
            values = grouped.get((budget, method.name), [])
            psnr_values = [float(row["psnr"]) for row in values]
            ssim_values = [float(row["ssim"]) for row in values]
            interval_seconds = sum(
                float(row["interval_train_seconds"]) for row in values
            )
            interval_steps = sum(
                budget if budget == budgets[0] else budget - budgets[
                    budgets.index(budget) - 1
                ]
                for _ in values
            )
            curve = {
                "budget_steps": budget,
                "method": method.name,
                "trajectory_count": len(values),
                "expected_trajectory_count": expected_per_curve,
                "complete": len(values) == expected_per_curve,
                "psnr_mean": (
                    statistics.fmean(psnr_values) if psnr_values else None
                ),
                "psnr_stddev": (
                    statistics.stdev(psnr_values)
                    if len(psnr_values) > 1
                    else 0.0 if psnr_values else None
                ),
                "ssim_mean": (
                    statistics.fmean(ssim_values) if ssim_values else None
                ),
                "ssim_stddev": (
                    statistics.stdev(ssim_values)
                    if len(ssim_values) > 1
                    else 0.0 if ssim_values else None
                ),
                "interval_train_seconds": interval_seconds,
                "aggregate_steps_per_second": (
                    interval_steps / interval_seconds
                    if interval_seconds > 0
                    else None
                ),
                "semantic_psnr_mean": {
                    semantic: (
                        statistics.fmean(
                            float(row[f"psnr_{semantic}"])
                            for row in values
                            if row[f"psnr_{semantic}"] is not None
                        )
                        if any(
                            row[f"psnr_{semantic}"] is not None
                            for row in values
                        )
                        else None
                    )
                    for semantic in SEMANTICS
                },
            }
            curves.append(curve)
            if curve["psnr_mean"] is not None:
                budget_rows.append(curve)
        rankings[budget] = [
            str(item["method"])
            for item in sorted(
                budget_rows,
                key=lambda item: float(item["psnr_mean"]),
                reverse=True,
            )
        ]

    ranking_agreements = []
    for previous, current in zip(budgets, budgets[1:]):
        ranking_agreements.append(
            {
                "from_budget": previous,
                "to_budget": current,
                "pairwise_agreement": _pairwise_rank_agreement(
                    rankings[previous],
                    rankings[current],
                ),
            }
        )
    by_curve = {
        (int(item["budget_steps"]), str(item["method"])): item
        for item in curves
    }
    final_gains = {}
    if len(budgets) >= 2:
        previous, final = budgets[-2:]
        for method in config.methods:
            left = by_curve.get((previous, method.name), {}).get("psnr_mean")
            right = by_curve.get((final, method.name), {}).get("psnr_mean")
            final_gains[method.name] = (
                None
                if left is None or right is None
                else float(right) - float(left)
            )

    final_agreement = (
        ranking_agreements[-1]["pairwise_agreement"]
        if ranking_agreements
        else None
    )
    finite_gains = [
        abs(float(value))
        for value in final_gains.values()
        if value is not None
    ]
    maximum_final_gain = max(finite_gains) if finite_gains else None
    ranking_stable = (
        final_agreement is not None and float(final_agreement) >= 0.9
    )
    plateau = (
        maximum_final_gain is not None and maximum_final_gain <= 0.1
    )
    minimum_recommendation_steps = 5_000
    recommendable = (
        bool(progress["complete"])
        and budgets[-1] >= minimum_recommendation_steps
        and ranking_stable
        and plateau
    )
    reasons = []
    if not progress["complete"]:
        reasons.append("not every planned trajectory and checkpoint is complete")
    if budgets[-1] < minimum_recommendation_steps:
        reasons.append(
            "the largest budget is below the conservative 5,000-step "
            "recommendation floor"
        )
    if not ranking_stable:
        reasons.append("method ordering is not yet stable across the last budgets")
    if not plateau:
        reasons.append("at least one method still changes by more than 0.1 dB")
    if not reasons:
        reasons.append("quality curves plateaued with stable method ordering")

    trajectory_seconds = []
    source_trajectory_seconds = []
    extension_trajectory_seconds = []
    trajectory_steps = []
    for record in records:
        if record["observations"]:
            final_observation = record["observations"][-1]
            trajectory_seconds.append(
                float(final_observation["cumulative_train_seconds"])
            )
            source_trajectory_seconds.append(
                float(record["source_elapsed_train_seconds"])
            )
            extension_trajectory_seconds.append(
                float(record["extension_elapsed_train_seconds"])
            )
            trajectory_steps.append(int(final_observation["budget_steps"]))
    total_train_seconds = sum(trajectory_seconds)
    source_train_seconds = sum(source_trajectory_seconds)
    extension_train_seconds = sum(extension_trajectory_seconds)
    total_measured_steps = sum(trajectory_steps)
    source_measured_steps = len(trajectory_steps) * int(
        config.runner["resume_from"]["maximum_budget"]
    )
    extension_measured_steps = max(0, total_measured_steps - source_measured_steps)
    gpu_seconds_per_step = (
        total_train_seconds / total_measured_steps
        if total_measured_steps
        else None
    )
    extension_gpu_seconds_per_step = (
        extension_train_seconds / extension_measured_steps
        if extension_measured_steps
        else None
    )
    projected_full_seconds = (
        71_280_000 * gpu_seconds_per_step / 4
        if gpu_seconds_per_step is not None
        else None
    )
    next_budgets = sorted(
        {
            min(int(config.runner["full_schedule_steps"]) - 1, budgets[-1] * scale)
            for scale in (5, 10, 25)
        }
    )
    worker_elapsed = [
        float(worker["elapsed_wall_seconds"])
        for worker in progress["workers"]
        if worker.get("elapsed_wall_seconds") is not None
    ]
    worker_starts = [
        datetime.fromisoformat(str(worker["started_at_utc"]))
        for worker in progress["workers"]
        if worker.get("started_at_utc") is not None
    ]
    worker_finishes = [
        datetime.fromisoformat(str(worker["finished_at_utc"]))
        for worker in progress["workers"]
        if worker.get("finished_at_utc") is not None
    ]
    observed_launch_wall = (
        (max(worker_finishes) - min(worker_starts)).total_seconds()
        if worker_starts and worker_finishes
        else None
    )

    payload = {
        "schema": "peps.texture_convergence_pilot_report",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "artifact": "texture-table2-convergence-pilot",
        "paper": config.paper,
        "profile": "bounded_convergence_pilot",
        "canonical": False,
        "progress": progress,
        "protocol": {
            "sets": [item["id"] for item in _pilot_instance_specs(config)],
            "providers": ["polyhaven", "ambientcg"],
            "semantic_coverage": list(SEMANTICS),
            "methods": [method.name for method in config.methods],
            "seeds": list(config.seeds),
            "step_budgets": list(budgets),
            "full_schedule_steps": int(config.runner["full_schedule_steps"]),
            "evaluation_side": int(config.runner["evaluation_side"]),
            "evaluation_sampling": config.runner["evaluation_sampling"],
            "table2_numeric_comparable": False,
            "resume_from_steps": int(
                config.runner["resume_from"]["maximum_budget"]
            ),
            "resume_schedule_continuity": (
                "model, Adam moments, minibatch stream, and the original "
                "120000-step cosine scheduler are retained"
            ),
            "physical_devices": list(config.runner["physical_devices"]),
        },
        "curves": curves,
        "rankings": [
            {"budget_steps": budget, "methods": rankings[budget]}
            for budget in budgets
        ],
        "ranking_agreements": ranking_agreements,
        "final_interval_psnr_gains": final_gains,
        "runtime": {
            "measured_optimizer_steps": total_measured_steps,
            "source_optimizer_steps": source_measured_steps,
            "extension_optimizer_steps": extension_measured_steps,
            "summed_gpu_train_seconds": total_train_seconds,
            "source_gpu_train_seconds": source_train_seconds,
            "extension_gpu_train_seconds": extension_train_seconds,
            "summed_gpu_hours": total_train_seconds / 3600.0,
            "source_gpu_hours": source_train_seconds / 3600.0,
            "extension_gpu_hours": extension_train_seconds / 3600.0,
            "mean_gpu_seconds_per_optimizer_step": gpu_seconds_per_step,
            "extension_gpu_seconds_per_optimizer_step": (
                extension_gpu_seconds_per_step
            ),
            "observed_parallel_wall_seconds": observed_launch_wall,
            "maximum_worker_elapsed_seconds": (
                max(worker_elapsed) if worker_elapsed else None
            ),
            "physical_gpu_count": len(config.runner["physical_devices"]),
            "physical_devices": list(config.runner["physical_devices"]),
            "rough_full_table2_four_gpu_seconds": projected_full_seconds,
            "full_projection_warning": (
                "Representative early-step throughput only; method mix, output "
                "width, evaluation, I/O, and long-run effects make this non-binding."
            ),
        },
        "decision": {
            "status": (
                "budget_recommended"
                if recommendable
                else "inconclusive_bounded_pilot"
            ),
            "recommended_table2_steps": budgets[-1] if recommendable else None,
            "recommendation_scope": (
                "local_checkpoint_continuation_protocol_assumption"
                if recommendable
                else "none_inconclusive"
            ),
            "ranking_stable": ranking_stable,
            "last_interval_max_abs_psnr_gain_db": maximum_final_gain,
            "plateau_threshold_db": 0.1,
            "minimum_recommendation_steps": minimum_recommendation_steps,
            "reasons": reasons,
            "next_pilot_budgets": next_budgets,
            "full_71m_step_run_authorized": False,
        },
        "evidence": {
            "pilot_manifest": progress["manifest"],
            "config": {
                "path": str(PILOT_CONFIG.relative_to(ROOT)),
                "sha256": _sha256(PILOT_CONFIG),
            },
            "texture_manifest": {
                "path": "data/manifests/textures.json",
                "sha256": _sha256(ROOT / "data/manifests/textures.json"),
            },
            "code_digest": _code_digest(),
            "code_receipts": _code_receipts(),
            "resume_lineage": (
                None
                if manifest_payload is None
                else manifest_payload["resume_lineage"]
            ),
            "checkpoints": checkpoint_receipts,
            "raw_csv": {
                "path": str(csv_path),
                "bytes": csv_path.stat().st_size,
                "sha256": _sha256(csv_path),
                "rows": len(raw_rows),
            },
        },
        "limitations": list(PILOT_LIMITATIONS),
    }
    atomic_write_json(output_path, payload)
    return payload


def _instance_rows(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for record in records:
        metrics = record["metrics"]
        row = {
            "instance": record["instance"],
            "method": record["method"],
            "seed": record["seed"],
            "psnr": metrics.get("psnr"),
            "ssim": metrics.get("ssim"),
            "encoder_params": record["parameters"]["encoder"],
            "decoder_params": record["parameters"]["decoder"],
            "total_params": record["parameters"]["total"],
            "compression_factor": record["compression_factor"],
            "elapsed_seconds": record["elapsed_seconds"],
            "job_index": record["job_index"],
            "job_shard_rank": record["rank"],
            "job_shard_world_size": record["world_size"],
            "git_commit": record.get("git_commit"),
            "git_dirty": record.get("git_dirty"),
        }
        for semantic in SEMANTICS:
            row[semantic] = metrics.get(f"psnr/semantic/{semantic}")
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (row["instance"], row["method"], row["seed"]),
    )


INSTANCE_COLUMNS = (
    "instance",
    "method",
    "seed",
    "psnr",
    "ssim",
    *SEMANTICS,
    "encoder_params",
    "decoder_params",
    "total_params",
    "compression_factor",
    "elapsed_seconds",
    "job_index",
    "job_shard_rank",
    "job_shard_world_size",
    "git_commit",
    "git_dirty",
)


def _paired_sweep_deltas(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    pairs = (
        ("Grid-PEPS3F", "Grid-PEPS4F"),
        ("Grid-PinkPEPS3F", "Grid-PinkPEPS4F"),
        ("NTC_PEPS3F", "NTC_PEPS4F"),
        ("NTC_PinkPEPS3F", "NTC_PinkPEPS4F"),
    )
    by_key = {
        (str(record["instance"]), int(record["seed"]), str(record["method"])): record
        for record in records
    }
    rows = []
    for three, four in pairs:
        for metric in ("psnr", "ssim"):
            deltas = []
            for instance, seed, method in sorted(by_key):
                if method != three:
                    continue
                left = by_key[(instance, seed, three)]
                right = by_key.get((instance, seed, four))
                if right is not None:
                    deltas.append(
                        float(right["metrics"][metric])
                        - float(left["metrics"][metric])
                    )
            rows.append(
                {
                    "three_frequency_method": three,
                    "four_frequency_method": four,
                    "metric": metric,
                    "count": len(deltas),
                    "mean_delta_4f_minus_3f": _mean(deltas),
                }
            )
    return rows


def _sweep_rows(
    config: ExperimentConfig,
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    observations = _map_observations(records)
    by_method = _method_record_groups(records)
    expected_jobs = 18 * len(config.seeds)
    expected_maps = 78 * len(config.seeds)
    rows = []
    for method in config.methods:
        method_records = by_method.get(method.name, [])
        benchmarks = [
            record["inference_benchmark"]
            for record in method_records
            if "inference_benchmark" in record
        ]
        architecture = (
            method_records[0].get("architecture")
            if method_records
            else architecture_receipt(method, output_channels=15)
        )
        psnr = observations.get((method.name, "psnr", "global"), [])
        ssim = observations.get((method.name, "ssim", "global"), [])
        rows.append(
            {
                "method": method.name,
                "frequencies": architecture["peps_frequencies"],
                "aggregation": architecture["aggregation"],
                "decoder_input_dim": architecture["decoder_input_dim"],
                "encoder_params": architecture["encoder_params"],
                "job_count": len(method_records),
                "expected_job_count": expected_jobs,
                "map_observation_count": len(psnr),
                "expected_map_observation_count": expected_maps,
                "psnr": _mean(psnr),
                "ssim": _mean(ssim),
                "latency_observation_count": len(benchmarks),
                "latency_median_ms": _mean(
                    [float(item["median_ms"]) for item in benchmarks]
                ),
                "latency_p95_ms": _mean(
                    [float(item["p95_ms"]) for item in benchmarks]
                ),
                "latency_million_queries_per_second": _mean(
                    [
                        float(item["million_queries_per_second"])
                        for item in benchmarks
                    ]
                ),
                "latency_comparable_to_paper": False,
                "complete": (
                    len(method_records) == expected_jobs
                    and len(psnr) == expected_maps
                    and len(ssim) == expected_maps
                    and len(benchmarks) == 1
                ),
            }
        )
    return rows


SWEEP_COLUMNS = (
    "method",
    "frequencies",
    "aggregation",
    "decoder_input_dim",
    "encoder_params",
    "job_count",
    "expected_job_count",
    "map_observation_count",
    "expected_map_observation_count",
    "psnr",
    "ssim",
    "latency_observation_count",
    "latency_median_ms",
    "latency_p95_ms",
    "latency_million_queries_per_second",
    "latency_comparable_to_paper",
    "complete",
)


def write_reports(
    *,
    output_root: Path,
    destination_dir: Path,
) -> dict[str, object]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    table_config = load_experiment_config(ARTIFACT_CONFIGS["table2"])
    table_records = _collect_valid_records(
        table_config,
        _artifact_output(output_root, "table2"),
    )
    table_rows = _table2_rows(table_config, table_records)
    table_progress = artifact_progress("table2", output_root=output_root)
    table_payload = {
        "schema": "peps.texture_table2_report",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "paper": table_config.paper,
        "artifact": "texture-table2",
        "profile": table_config.profile,
        "aggregation": {
            "unit": "individual RGB map, then mean over all maps and seeds",
            "global_weighting": "map_weighted",
            "semantic_weighting": "map_weighted_within_category",
        },
        "dataset": {
            "set_count": 18,
            "map_count": 78,
            "semantics": list(SEMANTICS),
            "manifest_sha256": _sha256(
                ROOT / "data/manifests/textures.json"
            ),
        },
        "progress": table_progress,
        "rows": table_rows,
        "complete": table_progress["complete"],
        "verification_status": (
            "complete_protocol_assumption"
            if table_progress["complete"]
            else "partial_do_not_interpret"
        ),
        "limitations": list(PROTOCOL_ASSUMPTIONS),
    }
    table_json = destination_dir / "table2.json"
    table_csv = destination_dir / "table2.csv"
    instance_csv = destination_dir / "table2_instances.csv"
    atomic_write_json(table_json, table_payload)
    _atomic_csv(table_csv, table_rows, TABLE2_COLUMNS)
    _atomic_csv(instance_csv, _instance_rows(table_records), INSTANCE_COLUMNS)

    sweep_config = load_experiment_config(ARTIFACT_CONFIGS["sweep"])
    sweep_records = _collect_valid_records(
        sweep_config,
        _artifact_output(output_root, "sweep"),
    )
    sweep_rows = _sweep_rows(sweep_config, sweep_records)
    sweep_progress = artifact_progress("sweep", output_root=output_root)
    sweep_payload = {
        "schema": "peps.texture_frequency_sweep_report",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "paper": sweep_config.paper,
        "artifact": "texture-frequency-sweep",
        "profile": sweep_config.profile,
        "progress": sweep_progress,
        "rows": sweep_rows,
        "paired_deltas": _paired_sweep_deltas(sweep_records),
        "latency_scope": (
            "PyTorch checkpoint inference on Paving Stones 070 seed 0, "
            "1048576 queries; not fused-HIP comparable"
        ),
        "complete": sweep_progress["complete"],
        "limitations": list(PROTOCOL_ASSUMPTIONS),
    }
    sweep_json = destination_dir / "frequency_sweep.json"
    sweep_csv = destination_dir / "frequency_sweep.csv"
    atomic_write_json(sweep_json, sweep_payload)
    _atomic_csv(sweep_csv, sweep_rows, SWEEP_COLUMNS)
    status = {
        "schema": "peps.texture_reproduction_status",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "table2": table_progress,
        "sweep": sweep_progress,
        "reports": {
            "table2_json": str(table_json),
            "table2_csv": str(table_csv),
            "table2_instances_csv": str(instance_csv),
            "sweep_json": str(sweep_json),
            "sweep_csv": str(sweep_csv),
        },
    }
    atomic_write_json(destination_dir / "status.json", status)
    return status


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(descriptor)
    try:
        fig.savefig(temporary, dpi=180, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def generate_figure8(
    *,
    output_root: Path,
    output_path: Path,
    status_path: Path,
    device: torch.device,
    methods: Sequence[str] | None,
    seed: int,
    crop_x: int | None,
    crop_y: int | None,
    crop_size: int,
    verification_receipt: Path | None,
) -> dict[str, object]:
    config = load_experiment_config(ARTIFACT_CONFIGS["table2"])
    requested = tuple(methods or FIGURE8_METHODS)
    method_configs = {
        method.name: method for method in config.methods if method.name in requested
    }
    missing_methods = sorted(set(requested) - set(method_configs))
    if missing_methods:
        raise ValueError(f"unknown Figure 8 methods: {missing_methods}")
    instance_id = str(config.runner.get("figure8_instance", "paving-stones-070"))
    output_dir = _artifact_output(output_root, "table2")
    planned_jobs = {
        (job.instance.name, job.method.name, job.seed): job
        for job in enumerate_jobs(config, _dummy_instances(config))
    }
    blockers = []
    checkpoints = {}
    for name in requested:
        method = method_configs[name]
        planned_job = planned_jobs.get((instance_id, name, seed))
        if planned_job is None:
            raise ValueError(
                f"seed {seed} is not configured for Figure 8 method {name}"
            )
        result_path, checkpoint_path = _job_paths(
            output_dir,
            instance_id,
            name,
            seed,
        )
        expected = _job_total_steps(config, method)
        if not result_path.is_file():
            blockers.append(f"{name}: completed result is missing")
            continue
        result_error = _validate_result_record(
            result_path,
            config=config,
            job=planned_job,
        )
        if result_error is not None:
            blockers.append(f"{name}: invalid result: {result_error}")
            continue
        if not checkpoint_path.is_file():
            blockers.append(f"{name}: retained checkpoint is missing")
            continue
        step, error = _checkpoint_step(checkpoint_path, expected)
        if error is not None or step != expected:
            blockers.append(f"{name}: {error or f'checkpoint step is {step}'}")
            continue
        checkpoints[name] = checkpoint_path
    if blockers:
        payload = {
            "schema": "peps.texture_figure8_status",
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": _utc_now(),
            "status": "blocked",
            "instance": instance_id,
            "seed": seed,
            "methods": list(requested),
            "blockers": blockers,
            "required_artifact": "complete retained Table 2 seed checkpoint",
            "output": str(output_path),
        }
        atomic_write_json(status_path, payload)
        return payload

    receipt_current = bool(
        verification_receipt is not None
        and verification_receipt_is_current(verification_receipt)
    )
    instance = _load_texture_instance(
        instance_id,
        verify_checksums=not receipt_current,
    )
    height, width, _ = instance.shape
    if crop_size < 11 or crop_size > min(height, width):
        raise ValueError("crop size must be between 11 and the texture extent")
    left = (width - crop_size) // 2 if crop_x is None else crop_x
    top = (height - crop_size) // 2 if crop_y is None else crop_y
    if left < 0 or top < 0 or left + crop_size > width or top + crop_size > height:
        raise ValueError("Figure 8 crop is outside the texture")
    x_axis = torch.arange(left, left + crop_size, dtype=torch.float32) / (width - 1)
    y_axis = torch.arange(top, top + crop_size, dtype=torch.float32) / (height - 1)
    y, x = torch.meshgrid(y_axis, x_axis, indexing="ij")
    crop_coords = torch.stack((x.reshape(-1), y.reshape(-1)), dim=1)
    target_crop = instance.targets.reshape(instance.shape)[
        top : top + crop_size,
        left : left + crop_size,
    ]
    predictions = {}
    checkpoint_receipts = []
    for name in requested:
        method = method_configs[name]
        model, _ = _build_model(config, method, instance)
        state = torch.load(
            checkpoints[name],
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(state["model"])
        model = model.to(device).eval()
        with torch.no_grad():
            prediction = model(crop_coords.to(device)).cpu().reshape(
                crop_size,
                crop_size,
                instance.targets.shape[1],
            )
        predictions[name] = prediction.clamp(0.0, 1.0)
        checkpoint_receipts.append(
            {
                "method": name,
                "path": str(checkpoints[name]),
                "sha256": _sha256(checkpoints[name]),
                "step": int(state["step"]),
            }
        )
        del model, state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    import flip_evaluator
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    maps = tuple(instance.metadata["texture_maps"])
    columns = ("reference", *requested)
    fig, axes = plt.subplots(
        2 * len(maps),
        len(columns),
        figsize=(2.25 * len(columns), 3.1 * len(maps)),
        squeeze=False,
    )
    flip_rows = []
    for map_index, texture_map in enumerate(maps):
        start = int(texture_map["channel_start"])
        stop = int(texture_map["channel_stop"])
        reference = target_crop[..., start:stop].clamp(0.0, 1.0)
        for column, name in enumerate(columns):
            image_axis = axes[2 * map_index, column]
            flip_axis = axes[2 * map_index + 1, column]
            image = reference if name == "reference" else predictions[name][..., start:stop]
            image_axis.imshow(image.numpy())
            image_axis.axis("off")
            image_axis.set_title(
                f"{texture_map['map_id']}\n{name}",
                fontsize=8,
            )
            flip_axis.axis("off")
            if name == "reference":
                flip_axis.text(0.5, 0.5, "reference", ha="center", va="center")
                continue
            error_map, mean_error, _ = flip_evaluator.evaluate(
                reference.numpy(),
                image.numpy(),
                "LDR",
            )
            error = torch.as_tensor(error_map)
            if error.ndim == 3:
                error = error.mean(dim=-1)
            flip_axis.imshow(error.numpy(), cmap="magma", vmin=0.0, vmax=1.0)
            flip_rows.append(
                {
                    "map_id": texture_map["map_id"],
                    "semantic": texture_map["semantic"],
                    "method": name,
                    "mean_flip": float(mean_error),
                }
            )
    fig.suptitle(
        f"Figure 8 — Paving Stones 070 {crop_size}x{crop_size} crops "
        "and official LDR-FLIP maps"
    )
    _save_figure(fig, output_path)
    plt.close(fig)
    flip_csv = status_path.with_name("figure8_flip.csv")
    _atomic_csv(
        flip_csv,
        flip_rows,
        ("map_id", "semantic", "method", "mean_flip"),
    )
    payload = {
        "schema": "peps.texture_figure8_status",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "status": "generated",
        "instance": instance_id,
        "seed": seed,
        "methods": list(requested),
        "crop_xywh": [left, top, crop_size, crop_size],
        "crop_coordinates_published_by_paper": False,
        "output": str(output_path),
        "flip_rows": str(flip_csv),
        "checkpoints": checkpoint_receipts,
        "verification_receipt": (
            None if verification_receipt is None else str(verification_receipt)
        ),
    }
    atomic_write_json(status_path, payload)
    return payload


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    protocol = subparsers.add_parser("protocol")
    protocol.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    protocol.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "texture_repro/protocol.json",
    )

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--verify-files", action="store_true")
    manifest.add_argument("--decode-size", type=int)
    manifest.add_argument(
        "--output",
        type=Path,
    )

    plan = subparsers.add_parser("plan")
    plan.add_argument("--artifact", choices=ARTIFACT_CONFIGS, required=True)
    plan.add_argument("--world-size", type=int, default=4)
    plan.add_argument("--include-jobs", action="store_true")
    plan.add_argument("--output", type=Path)

    run = subparsers.add_parser("run")
    run.add_argument("--artifact", choices=ARTIFACT_CONFIGS, required=True)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--rank", type=int, default=0)
    run.add_argument("--world-size", type=int, default=1)
    run.add_argument("--device", default="auto")
    run.add_argument("--instance", action="append")
    run.add_argument("--method", action="append")
    run.add_argument("--force", action="store_true")
    run.add_argument("--allow-protocol-assumptions", action="store_true")
    run.add_argument(
        "--verification-receipt",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT
        / "texture_repro/dataset_verification.json",
    )

    status = subparsers.add_parser("status")
    status.add_argument("--artifact", choices=ARTIFACT_CONFIGS, required=True)
    status.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    status.add_argument("--output", type=Path)

    report = subparsers.add_parser("report")
    report.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    report.add_argument(
        "--destination-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "texture_repro",
    )

    pilot_plan = subparsers.add_parser("pilot-plan")
    pilot_plan.add_argument("--world-size", type=int, default=4)
    pilot_plan.add_argument("--include-jobs", action="store_true")
    pilot_plan.add_argument("--output", type=Path)

    pilot_run = subparsers.add_parser("pilot-run")
    pilot_run.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    pilot_run.add_argument("--rank", type=int, default=0)
    pilot_run.add_argument("--world-size", type=int, default=4)
    pilot_run.add_argument("--device", default="auto")
    pilot_run.add_argument("--physical-device-index", type=int, required=True)
    pilot_run.add_argument("--max-wall-seconds", type=int, default=1200)
    pilot_run.add_argument("--allow-protocol-assumptions", action="store_true")
    pilot_run.add_argument(
        "--verification-receipt",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT
        / "texture_repro/dataset_verification.json",
    )

    pilot_status = subparsers.add_parser("pilot-status")
    pilot_status.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    pilot_status.add_argument("--output", type=Path)

    pilot_report = subparsers.add_parser("pilot-report")
    pilot_report.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    pilot_report.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT
        / "texture_repro/convergence_pilot.json",
    )
    pilot_report.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT
        / "texture_repro/convergence_pilot_observations.csv",
    )

    figure = subparsers.add_parser("figure8")
    figure.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    figure.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "texture_repro/figure8.png",
    )
    figure.add_argument(
        "--status-output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "texture_repro/figure8_status.json",
    )
    figure.add_argument("--device", default="auto")
    figure.add_argument("--method", action="append")
    figure.add_argument("--seed", type=int, default=0)
    figure.add_argument("--crop-x", type=int)
    figure.add_argument("--crop-y", type=int)
    figure.add_argument("--crop-size", type=int, default=100)
    figure.add_argument(
        "--verification-receipt",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT
        / "texture_repro/dataset_verification.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "protocol":
        payload = protocol_report(arguments.output_root)
        atomic_write_json(arguments.output, payload)
    elif arguments.command == "manifest":
        payload = validate_manifest_consumption(
            verify_files=arguments.verify_files,
            decode_size=arguments.decode_size,
        )
        destination = arguments.output or (
            DEFAULT_OUTPUT_ROOT
            / "texture_repro"
            / (
                "dataset_verification.json"
                if arguments.verify_files
                else "manifest_validation.json"
            )
        )
        atomic_write_json(destination, payload)
    elif arguments.command == "plan":
        payload = job_plan(
            arguments.artifact,
            world_size=arguments.world_size,
            include_jobs=arguments.include_jobs,
        )
        if arguments.output is not None:
            atomic_write_json(arguments.output, payload)
    elif arguments.command == "run":
        payload = run_artifact(
            arguments.artifact,
            output_root=arguments.output_root,
            rank=arguments.rank,
            world_size=arguments.world_size,
            device=_device(arguments.device),
            instance_ids=arguments.instance,
            methods=arguments.method,
            force=arguments.force,
            allow_protocol_assumptions=arguments.allow_protocol_assumptions,
            verification_receipt=arguments.verification_receipt,
        )
    elif arguments.command == "status":
        payload = artifact_progress(
            arguments.artifact,
            output_root=arguments.output_root,
        )
        if arguments.output is not None:
            atomic_write_json(arguments.output, payload)
    elif arguments.command == "report":
        payload = write_reports(
            output_root=arguments.output_root,
            destination_dir=arguments.destination_dir,
        )
    elif arguments.command == "pilot-plan":
        payload = pilot_job_plan(
            world_size=arguments.world_size,
            include_jobs=arguments.include_jobs,
        )
        if arguments.output is not None:
            atomic_write_json(arguments.output, payload)
    elif arguments.command == "pilot-run":
        payload = run_convergence_pilot(
            output_root=arguments.output_root,
            rank=arguments.rank,
            world_size=arguments.world_size,
            device=_device(arguments.device),
            physical_device_index=arguments.physical_device_index,
            verification_receipt=arguments.verification_receipt,
            max_wall_seconds=arguments.max_wall_seconds,
            allow_protocol_assumptions=arguments.allow_protocol_assumptions,
        )
    elif arguments.command == "pilot-status":
        payload = pilot_progress(output_root=arguments.output_root)
        if arguments.output is not None:
            atomic_write_json(arguments.output, payload)
    elif arguments.command == "pilot-report":
        payload = write_pilot_report(
            output_root=arguments.output_root,
            output_path=arguments.output,
            csv_path=arguments.csv_output,
        )
    elif arguments.command == "figure8":
        payload = generate_figure8(
            output_root=arguments.output_root,
            output_path=arguments.output,
            status_path=arguments.status_output,
            device=_device(arguments.device),
            methods=arguments.method,
            seed=arguments.seed,
            crop_x=arguments.crop_x,
            crop_y=arguments.crop_y,
            crop_size=arguments.crop_size,
            verification_receipt=arguments.verification_receipt,
        )
    else:
        raise AssertionError(f"unhandled command: {arguments.command}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
