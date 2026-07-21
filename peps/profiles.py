"""Immutable experiment profiles for paper and teaching runs.

``paper_exact`` records the protocol reported by PEPS arXiv:2604.24167v1.
Values the paper does not specify are represented explicitly as
``"not_reported"`` rather than filled with guesses. ``course_fast`` captures the
smaller workloads used by the teaching notebooks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar


PAPER_REFERENCE = "arXiv:2604.24167v1"
PROFILE_SCHEMA = "peps.experiment_profile"
PROFILE_SCHEMA_VERSION = 1
PROFILE_NAMES = ("paper_exact", "course_fast")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("profile mapping keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ExperimentProfile:
    """Recursively immutable configuration for one fidelity/runtime budget."""

    schema: ClassVar[str] = PROFILE_SCHEMA
    schema_version: ClassVar[int] = PROFILE_SCHEMA_VERSION

    name: str
    summary: str
    paper_reference: str
    image: Mapping[str, Any]
    texture: Mapping[str, Any]
    sdf: Mapping[str, Any]
    quantization: Mapping[str, Any]
    runtime: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.name not in PROFILE_NAMES:
            raise ValueError(f"unknown experiment profile {self.name!r}")
        for section in ("image", "texture", "sdf", "quantization", "runtime"):
            object.__setattr__(self, section, _freeze(getattr(self, section)))

    def section(self, name: str) -> Mapping[str, Any]:
        """Return an immutable named section."""

        if name not in {"image", "texture", "sdf", "quantization", "runtime"}:
            raise KeyError(f"unknown profile section {name!r}")
        return getattr(self, name)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible copy."""

        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "name": self.name,
            "summary": self.summary,
            "paper_reference": self.paper_reference,
            "image": _thaw(self.image),
            "texture": _thaw(self.texture),
            "sdf": _thaw(self.sdf),
            "quantization": _thaw(self.quantization),
            "runtime": _thaw(self.runtime),
        }


_PAPER_MLP = {
    "hidden_layers": 3,
    "hidden_dim": 64,
    "activation": "leaky_relu",
}

PAPER_EXACT = ExperimentProfile(
    name="paper_exact",
    summary="Protocol reported by the PEPS paper; no teaching downscaling.",
    paper_reference=PAPER_REFERENCE,
    image={
        "coordinates": "normalized_[0,1]^2",
        "frequency_schedule": "phi_i=2^i*pi_for_i=1..L",
        "capacity_sweep": {
            "dataset": {
                "name": "paper_4k_image_suite",
                "instance_count": "not_reported",
                "resolution": "native_4k",
            },
            "grid_resolutions": [16, 32, 64, 128],
            "feature_dimensions": [8, 16, 32, 64],
            "parameter_range": [10_000, 1_000_000],
            "peps_frequencies": 3,
            "loss": "l1",
            "learning_rate": 0.01,
            "optimizer": "not_reported",
            "batch_size": "not_reported",
            "training_steps": "same_across_methods_but_not_reported",
            "network": _PAPER_MLP,
            "metrics": ["psnr"],
        },
        "kodak_table_1": {
            "dataset": {
                "name": "Kodak",
                "instance_ids": [f"kodim{index:02d}" for index in range(1, 25)],
                "instance_count": 24,
                "resolution_xy": [768, 512],
                "color_space": "sRGB",
            },
            "models": {
                "pe": {
                    "frequencies": 10,
                    "hidden_layers": 3,
                    "hidden_dim": 300,
                },
                "lpe": {"grid_resolution_xy": [196, 128], "frequencies": 4},
                "ntc_n": {
                    "g0_resolution_xy": [192, 128],
                    "g0_feature_dim": 12,
                    "g1_resolution_xy": [96, 64],
                    "g1_feature_dim": 20,
                },
                "grid": {"grid_resolution_xy": [196, 128], "feature_dim": 17},
                "g_peps": {
                    "grid_resolution_xy": [196, 128],
                    "feature_dim": 17,
                    "frequencies": 3,
                    "aggregation": "concat",
                },
                "g_pink_peps": {
                    "grid_resolution_xy": [196, 128],
                    "feature_dim": 17,
                    "frequencies": 3,
                    "aggregation": "pink",
                },
                "g_pink_peps_25": {
                    "grid_resolution_xy": [196, 128],
                    "feature_dim": 13,
                    "frequencies": 3,
                    "aggregation": "pink",
                },
                "ntc_peps": {
                    "g0_resolution_xy": [192, 128],
                    "g0_feature_dim": 12,
                    "g1_resolution_xy": [96, 64],
                    "g1_feature_dim": 20,
                    "frequencies": 3,
                    "aggregation": "concat",
                },
                "ntc_pink_peps": {
                    "g0_resolution_xy": [192, 128],
                    "g0_feature_dim": 12,
                    "g1_resolution_xy": [96, 64],
                    "g1_feature_dim": 20,
                    "frequencies": 3,
                    "aggregation": "pink",
                },
            },
            "network": {**_PAPER_MLP, "output_activation": "sigmoid"},
            "training": {
                "fixed_learning_rate": 0.01,
                "optimizer": "not_reported",
                "batch_size": "not_reported",
                "training_steps": "not_reported",
                "narrative_loss": "l1",
                "table_1_value_matching_loss": "l2",
            },
            "metrics": {
                "psnr": {"data_range": 1.0, "aggregation": "per_image_then_mean"},
                "flip": {
                    "implementation": "flip_evaluator_LDR",
                    "reduction": "image_mean_then_dataset_mean",
                },
                "lpips": {
                    "backbone": "alexnet",
                    "aggregation": "per_image_then_mean",
                },
                "lsd": {
                    "implementation": "peps.metrics.lsd",
                    "domain": "log1p_2d_orthonormal_fourier_amplitude",
                    "reduction": "channel_mean_of_spectral_rmse",
                },
                "ssim": {
                    "implementation": "torchmetrics.image.StructuralSimilarityIndexMeasure",
                    "windowed": True,
                    "data_range": 1.0,
                    "aggregation": "per_image_then_mean",
                },
            },
            "paper_ambiguity": (
                "The paper describes L1 as the stable image protocol, while the "
                "published Table 1 PSNR values match the appendix's L2 row."
            ),
        },
    },
    texture={
        "coordinates": "normalized_[0,1]^2",
        "frequency_schedule": "phi_i=2^i*pi_for_i=1..L",
        "dataset": {
            "name": "paper_texture_sets",
            "instance_count": 18,
            "resolution_xy": [4096, 4096],
            "instance_ids": [
                "bench_vice_01",
                "cardboard_box_01",
                "cannon_01",
                "clay_roof_tiles_02",
                "fabric_pattern_07",
                "garden_gnome",
                "garden_sprinkler_01",
                "wood_planks",
                "treasure_chest",
                "paving_stones_070",
                "rails_001",
                "red_dirt_mud_01",
                "aerial_rocks_02",
                "bricks_090",
                "forest_sand_01",
                "metal_plates_013",
                "roof_09",
                "wood_063",
            ],
            "sources": ["polyhaven.com", "ambientcg.com"],
            "target_channels": "3_per_available_texture_map",
        },
        "models": {
            "bi_grid": {"grid_resolution": 1024, "feature_dim": 17},
            "lpe": {
                "grid_resolution": 1024,
                "feature_dim": 16,
                "frequencies": "not_reported",
            },
            "ntc_n": {
                "g0_resolution": 1024,
                "g0_feature_dim": 12,
                "g1_resolution": 512,
                "g1_feature_dim": 20,
                "tiled_pe_octaves": 3,
            },
            "grid_peps": {
                "grid_resolution": 1024,
                "feature_dim": 17,
                "frequencies": 4,
                "aggregation": "concat",
            },
            "grid_pink_peps": {
                "grid_resolution": 1024,
                "feature_dim": 17,
                "frequencies": 4,
                "aggregation": "pink",
            },
            "ntc_peps": {
                "g0_resolution": 1024,
                "g0_feature_dim": 12,
                "g1_resolution": 512,
                "g1_feature_dim": 20,
                "tiled_pe_octaves": 3,
                "frequencies": 4,
                "aggregation": "concat",
            },
            "ntc_pink_peps": {
                "g0_resolution": 1024,
                "g0_feature_dim": 12,
                "g1_resolution": 512,
                "g1_feature_dim": 20,
                "tiled_pe_octaves": 3,
                "frequencies": 4,
                "aggregation": "pink",
            },
            "grid_peps_25": {
                "grid_resolution": 1024,
                "feature_dim": 13,
                "frequencies": 4,
                "aggregation": "concat",
            },
            "grid_pink_peps_25": {
                "grid_resolution": 1024,
                "feature_dim": 13,
                "frequencies": 4,
                "aggregation": "pink",
            },
            "ntc_peps_25": {
                "g0_resolution": 1024,
                "g0_feature_dim": 9,
                "g1_resolution": 512,
                "g1_feature_dim": 15,
                "tiled_pe_octaves": 3,
                "frequencies": 4,
                "aggregation": "concat",
            },
            "ntc_pink_peps_25": {
                "g0_resolution": 1024,
                "g0_feature_dim": 9,
                "g1_resolution": 512,
                "g1_feature_dim": 15,
                "tiled_pe_octaves": 3,
                "frequencies": 4,
                "aggregation": "pink",
            },
        },
        "network": {"hidden_layers": 3, "hidden_dim": 64, "activation": "gelu"},
        "training": {
            "optimizer": "not_reported",
            "loss": "l1",
            "grid_learning_rate": 0.1,
            "mlp_learning_rate": 0.001,
            "scheduler": "cosine",
            "activation": "gelu",
            "batch_size": 60_000,
            "epochs": 3_000,
            "batches_per_epoch": 40,
            "optimizer_steps": 120_000,
            "sampling": "random_pixel_coordinates",
        },
        "metrics": {
            "global": ["psnr", "ssim"],
            "per_texture_type": [
                "ao",
                "arm",
                "diffuse",
                "displacement",
                "metal",
                "normal",
                "roughness",
                "specular",
            ],
        },
        "quantization": "none",
    },
    sdf={
        "coordinates": "normalized_[0,1]^3",
        "frequency_schedule": "phi_i=2^i*pi_for_i=1..L",
        "dataset": {
            "instance_ids": [
                "lucy",
                "pitted-stonefish",
                "thai-statue",
                "armadillo",
            ],
            "volume_resolution": [512, 512, 512],
            "sign_convention": "negative_inside",
        },
        "models": {
            "pe": {"frequencies": 10},
            "ti_grid": {"resolution": 32, "feature_dim": 18},
            "grid_peps": {"resolution": 32, "feature_dim": 18, "frequencies": 3},
            "hash": {
                "resolution": 64,
                "feature_dim": 18,
                "hash_table_entries": 32**3,
            },
            "hash_peps": {
                "resolution": 64,
                "feature_dim": 18,
                "hash_table_entries": 32**3,
                "frequencies": 3,
            },
            "multi_grid": {"resolutions": [16, 32, 64], "feature_dim": 2},
            "multi_grid_peps": {
                "resolutions": [16, 32, 64],
                "feature_dim": 2,
                "frequencies": 3,
            },
            "multi_hash": {
                "resolutions": [16, 32, 64, 128],
                "feature_dim": 2,
                "max_hash_table_log2": 17,
            },
            "multi_hash_peps": {
                "resolutions": [16, 32, 64, 128],
                "feature_dim": 2,
                "max_hash_table_log2": 17,
                "frequencies": 3,
            },
        },
        "network": {"hidden_layers": 3, "hidden_dim": 64, "activation": "silu"},
        "training": {
            "primary": {"loss": "mape", "learning_rate": 0.001},
            "appendix": {"loss": "l1", "learning_rate": 0.01},
            "optimizer": "not_reported",
            "batch_size": 60_000,
            "epochs": 3_000,
            "batches_per_epoch": 40,
            "optimizer_steps": 120_000,
            "sampling": "uniform_random_[0,1]^3",
        },
        "metrics": ["occupancy_iou"],
        "eight_x_ablation": {
            "instance_id": "pitted-stonefish",
            "encoder_parameter_multiplier": 8,
            "ti_grid_resolution": 64,
            "loss": "l1",
        },
    },
    quantization={
        "enabled": False,
        "status": "not_in_paper",
        "reason": "The PEPS paper evaluates full-precision models only.",
    },
    runtime={
        "target": {
            "gpu": "AMD Radeon RX 9070 XT",
            "architecture": "gfx1201",
            "output_resolution": [1024, 1024],
            "output_channels": 3,
        },
        "model": {
            "grid_resolution": [1024, 1024],
            "feature_dim": 16,
            "mlp_hidden_layers": 3,
            "mlp_hidden_dim": 64,
            "fused_encoder_and_mlp": True,
            "precision": "not_reported",
        },
        "benchmark": {
            "warmup_iterations": "not_reported",
            "timed_iterations": "not_reported",
            "synchronization": "not_reported",
            "reported_ms": {
                "bi_grid_0_frequencies": 4.32,
                "grid_peps_3_frequencies": 5.47,
                "grid_pink_peps_3_frequencies": 4.86,
                "grid_pink_peps_4_frequencies": 4.99,
            },
        },
    },
)


COURSE_FAST = ExperimentProfile(
    name="course_fast",
    summary="Small CPU/GPU-friendly workloads used by the course notebooks.",
    paper_reference=PAPER_REFERENCE,
    image={
        "dataset": {
            "name": "Kodak_subset",
            "instance_ids": ["kodim01", "kodim05", "kodim19"],
            "max_image_side": 384,
        },
        "models": {
            "grid": {"resolution": 128, "feature_dim": 8},
            "grid_peps": {
                "resolution": 128,
                "feature_dim": 8,
                "frequencies": 6,
                "aggregation": "concat",
            },
            "pink_peps": {
                "resolution": 128,
                "feature_dim": 8,
                "frequencies": 6,
                "aggregation": "pink",
            },
        },
        "training": {
            "loss": "mse",
            "optimizer": "adam",
            "learning_rate": 0.01,
            "batch_size": 32_768,
            "optimizer_steps": 2_000,
        },
        "network": {
            "hidden_layers": 3,
            "hidden_dim": 64,
            "activation": "relu",
        },
        "metrics": ["psnr", "ssim", "lsd"],
    },
    texture={
        "dataset": {
            "instance_ids": [
                "MetalPlates013",
                "Metal032",
                "Planks020",
                "Rock023",
            ],
            "output_resolution": [512, 512],
        },
        "models": {
            "grid": {"resolution": 256, "feature_dim": 8},
            "grid_peps": {"resolution": 256, "feature_dim": 8, "frequencies": 6},
            "pink_peps": {"resolution": 256, "feature_dim": 8, "frequencies": 6},
        },
        "training": {
            "loss": "mse",
            "optimizer": "adam",
            "learning_rate": 0.01,
            "batch_size": 32_768,
            "optimizer_steps": 2_000,
        },
        "network": {
            "hidden_layers": 3,
            "hidden_dim": 64,
            "activation": "relu",
        },
        "metrics": ["psnr"],
    },
    sdf={
        "dataset": {
            "name": "analytic_torus",
            "training_samples": 120_000,
            "near_surface_fraction": 0.7,
            "near_surface_sigma": 0.05,
            "query_resolution": 64,
        },
        "models": {
            "grid": {"resolution": 48, "feature_dim": 4},
            "grid_peps": {"resolution": 48, "feature_dim": 4, "frequencies": 6},
            "multi_grid": {"base_resolution": 16, "levels": 4, "feature_dim": 2},
            "hash": {"levels": 8, "feature_dim": 2, "max_hash_table_log2": 17},
        },
        "training": {
            "loss": "mse_plus_eikonal",
            "optimizer": "adam",
            "learning_rate": 0.01,
            "batch_size": 16_384,
            "optimizer_steps": 800,
            "render_optimizer_steps": 1_200,
            "eikonal_weight": 0.1,
            "eikonal_epsilon": 0.01,
        },
        "network": {
            "hidden_layers": 3,
            "hidden_dim": 64,
            "activation": "relu",
        },
        "metrics": ["occupancy_iou"],
    },
    quantization={
        "enabled": True,
        "status": "course_extension_not_in_paper",
        "dataset": {"instance_ids": ["kodim01"], "max_image_side": 384},
        "models": {
            "grid": {"resolution": 128, "feature_dim": 8},
            "grid_peps": {
                "resolution": 128,
                "feature_dim": 8,
                "frequencies": 6,
                "aggregation": "concat",
            },
        },
        "training": {
            "loss": "mse",
            "optimizer": "adam",
            "learning_rate": 0.01,
            "batch_size": 32_768,
            "optimizer_steps": 2_500,
        },
        "reference_precision_bits": 32,
        "bit_widths": [8, 6, 4],
        "scheme": "symmetric_per_tensor_fake_ptq",
        "targets": ["grid_latents", "mlp_weights_and_biases"],
        "storage_accounting": "parameter_bits_only_scales_and_metadata_excluded",
        "metrics": ["effective_bits_per_parameter", "psnr"],
    },
    runtime={
        "status": "teaching_microbenchmarks_not_paper_comparable",
        "architectures": ["gfx1151", "gfx1201"],
        "wmma_workloads": [
            {"m": 4096, "k": 64, "n": 64, "iterations": 1_000},
            {"m": 2048, "k": 2048, "n": 2048, "iterations": 200},
        ],
        "precisions": ["fp16", "int8"],
        "timing": {
            "warmup_iterations": 1,
            "timer": "host_chrono",
            "device_synchronize_before_and_after": True,
        },
        "fused_teaching_kernel": {
            "query_points": 262_144,
            "iterations": 1_000,
            "scope": "one_bilinear_grid_sample_plus_first_mlp_layer",
        },
    },
)


PROFILES: Mapping[str, ExperimentProfile] = MappingProxyType(
    {profile.name: profile for profile in (PAPER_EXACT, COURSE_FAST)}
)


def get_profile(name: str) -> ExperimentProfile:
    """Resolve ``paper_exact`` or ``course_fast`` by stable name."""

    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(PROFILE_NAMES)
        raise KeyError(f"unknown profile {name!r}; choose one of: {choices}") from exc
