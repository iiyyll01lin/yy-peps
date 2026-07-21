"""Focused oracles for the paper-artifact reproduction layer."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch

from apps.image.build import build_paper_fig5, build_paper_image
from apps.image.data import orient_resolution_xy
from apps.sdf.build import build_paper_sdf
from apps.sdf.data import SDF_COORDINATE_SCALE, sample_sdf_tensor
from apps.texture.data import (
    aggregate_texture_map_metrics,
    texture_map_metric_rows,
)
from data.manifest import LoadedMap, LoadedTextureSet
from experiments.reproduce import check_prerequisites, run_course_smoke
from peps.train import SDFTrainConfig, fit_sdf, split_encoder_decoder_parameters


def test_kodak_paper_grids_rotate_with_original_portrait_orientation():
    assert orient_resolution_xy((196, 128), (768, 512)) == (196, 128)
    assert orient_resolution_xy((196, 128), (512, 768)) == (128, 196)

    landscape, _ = build_paper_image("grid", signal_resolution=(768, 512))
    portrait, _ = build_paper_image("grid", signal_resolution=(512, 768))
    assert landscape[0].resolution == (128, 196)
    assert portrait[0].resolution == (196, 128)

    portrait_ntc, _ = build_paper_image(
        "ntc_n",
        signal_resolution=(512, 768),
    )
    assert portrait_ntc[0].g0.resolution == (128, 192)
    # G1 is a GridEncoder and therefore stores (H, W).
    assert portrait_ntc[0].g1.resolution == (96, 64)


def test_fig5_builder_covers_exact_resolution_feature_factorial():
    for resolution in (16, 32, 64, 128):
        for feature_dim in (8, 16, 32, 64):
            grid, _ = build_paper_fig5(
                "bi_grid",
                resolution=resolution,
                feature_dim=feature_dim,
            )
            lpe, _ = build_paper_fig5(
                "lpe",
                resolution=resolution,
                feature_dim=feature_dim,
            )
            peps, _ = build_paper_fig5(
                "grid_peps",
                resolution=resolution,
                feature_dim=feature_dim,
            )
            assert grid[0].feature_dim == feature_dim
            assert lpe[0].feature_dim == feature_dim
            assert peps.encoder.feature_dim == feature_dim
            assert peps.projector.num_frequencies == 3


@pytest.mark.parametrize(
    "method",
    (
        "lpe",
        "grid",
        "hash",
        "grid_peps",
        "hash_peps",
        "m_grid",
        "m_peps",
        "m_hash",
        "m_hashpeps",
    ),
)
def test_table4_large_row_has_exactly_eight_times_encoder_parameters(method):
    base, _ = build_paper_sdf(method, encoder_parameter_multiplier=1)
    large, _ = build_paper_sdf(method, encoder_parameter_multiplier=8)
    base_encoder, _ = split_encoder_decoder_parameters(base)
    large_encoder, _ = split_encoder_decoder_parameters(large)
    base_count = sum(parameter.numel() for parameter in base_encoder)
    large_count = sum(parameter.numel() for parameter in large_encoder)
    assert large_count == 8 * base_count


def test_sdf_volume_sampling_uses_zyx_storage_and_inclusive_boundaries():
    z, y, x = torch.meshgrid(
        torch.arange(2, dtype=torch.float32),
        torch.arange(2, dtype=torch.float32),
        torch.arange(2, dtype=torch.float32),
        indexing="ij",
    )
    volume = x + 10 * y + 100 * z
    coords = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]
    )
    sampled = sample_sdf_tensor(volume, coords).squeeze(1)
    assert torch.allclose(sampled, torch.tensor([0.0, 111.0, 55.5]))


def test_course_eikonal_uses_centered_distance_scale_without_boundary_bias():
    assert SDF_COORDINATE_SCALE == 2.0

    class LinearCenteredSDF(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def forward(self, coords):
            return 2.0 * coords[:, :1] + self.bias

    coords = torch.rand(64, 3)
    targets = 2.0 * coords[:, :1]
    logged = []
    fit_sdf(
        LinearCenteredSDF(),
        coords,
        targets,
        SDFTrainConfig(
            steps=1,
            batch_size=4096,
            lr=0.0,
            eikonal_weight=1.0,
            eikonal_eps=0.1,
            eikonal_target_norm=2.0,
            device=torch.device("cpu"),
            log_every=1,
        ),
        on_log=lambda _step, value: logged.append(value),
    )
    assert logged[0] == pytest.approx(0.0, abs=1e-10)


def test_texture_table2_metrics_are_per_rgb_map_then_aggregated_by_type(tmp_path):
    target = torch.zeros(4, 4, 6)
    prediction = target.clone()
    prediction[..., :3] = 0.1
    prediction[..., 3:] = 0.2
    loaded = LoadedTextureSet(
        set_id="synthetic",
        tensor=target,
        maps=(
            LoadedMap("color", "DIFF", slice(0, 3), tmp_path / "color.png"),
            LoadedMap("roughness", "rough", slice(3, 6), tmp_path / "rough.png"),
        ),
        source_size=(4, 4),
        output_size=(4, 4),
    )
    rows = texture_map_metric_rows(
        prediction,
        loaded,
        metrics=("psnr",),
    )
    assert [row["map_id"] for row in rows] == ["color", "roughness"]
    aggregated = aggregate_texture_map_metrics(rows)
    global_row = next(row for row in aggregated if row["semantic"] == "global")
    assert global_row["count"] == 2
    assert {row["semantic"] for row in aggregated} == {
        "DIFF",
        "rough",
        "global",
    }


def test_course_smoke_writes_numeric_rows_only_beside_manifest(tmp_path):
    prerequisites = check_prerequisites(
        profile="course_fast",
        artifacts=("image-table1",),
    )
    assert prerequisites["ready"]

    output = run_course_smoke(
        "image",
        output_root=tmp_path,
        device=torch.device("cpu"),
    )
    run_dir = tmp_path / "runs" / output["run_id"]
    with (run_dir / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["profile"] == "course_fast"
    assert manifest["metadata"]["verification_status"].startswith("course_fast")
    with (run_dir / "instances.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["value"] for row in rows)
    assert (run_dir / "summary.csv").is_file()
