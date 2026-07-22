"""Focused CPU tests for the public-shape SDF reproduction path."""

from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import tomllib

import numpy as np
import pytest
import torch

from apps.sdf.render import (
    FirstSurfaceAccumulator,
    OrthographicRenderProtocol,
    evaluate_flip,
)
from experiments.sdf_repro import (
    CONFIG_ROOT,
    DEFAULT_TABLE4_CONFIG,
    PUBLIC_ASSETS,
    TABLE3_METHODS,
    TABLE4_METHODS,
    TABLE6_METHODS,
    _job_paths,
    _synthetic_sphere_volume,
    aggregate_config,
    assert_config_parameter_budgets,
    build_table4_deferred_receipt,
    enumerate_sdf_jobs,
    estimate_cost,
    evaluate_sdf_chunked,
    load_sdf_repro_config,
    run_shard,
    shard_sdf_jobs,
    validate_table4_deferred_receipt,
)


ROOT = Path(__file__).resolve().parents[1]


def _configs():
    return (
        load_sdf_repro_config(CONFIG_ROOT / "table3_mape.toml"),
        load_sdf_repro_config(CONFIG_ROOT / "table6_l1.toml"),
        load_sdf_repro_config(CONFIG_ROOT / "table4_deferred.toml"),
        load_sdf_repro_config(CONFIG_ROOT / "smoke.toml"),
    )


def test_sdf_configs_freeze_exact_public_subsets_and_paper_protocols() -> None:
    table3, table6, table4, smoke = _configs()
    assert table3.assets == PUBLIC_ASSETS
    assert table6.assets == PUBLIC_ASSETS
    assert tuple((method.name, method.key) for method in table3.methods) == (
        TABLE3_METHODS
    )
    assert tuple((method.name, method.key) for method in table6.methods) == (
        TABLE6_METHODS
    )
    assert tuple((method.name, method.key) for method in table4.methods) == (
        TABLE4_METHODS
    )
    assert "Hash-PEPS" in {method.name for method in table3.methods}
    assert "Hash-PEPS" not in {method.name for method in table6.methods}
    assert table3.training["loss"] == "mape"
    assert table6.training["loss"] == "l1"
    assert table3.total_steps == table6.total_steps == 120_000
    assert table3.training["batch_size"] == table6.training["batch_size"] == 60_000
    assert table3.evaluation["resolution"] == table6.evaluation["resolution"] == 512
    assert table3.sharding["world_size"] == table6.sharding["world_size"] == 4
    assert table3.reporting["aggregate_label"] == "three_shape_aggregate"
    assert table6.reporting["aggregate_label"] == "three_shape_aggregate"
    assert table4.status == "deferred_auth_required"
    assert table4.assets == ("pitted-stonefish",)
    assert smoke.profile == "smoke"


def test_sdf_toml_files_validate_against_checked_in_config_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (CONFIG_ROOT / "config.schema.json").read_text(encoding="utf-8")
    )
    for name in (
        "table3_mape.toml",
        "table6_l1.toml",
        "table4_deferred.toml",
        "smoke.toml",
    ):
        with (CONFIG_ROOT / name).open("rb") as handle:
            jsonschema.validate(tomllib.load(handle), schema)


def test_all_sdf_parameter_budgets_and_table4_8x_ratios_are_exact() -> None:
    table3, table6, table4, smoke = _configs()
    assert len(assert_config_parameter_budgets(table3)) == 10
    assert len(assert_config_parameter_budgets(table6)) == 9
    assert len(assert_config_parameter_budgets(smoke)) == 2
    rows = assert_config_parameter_budgets(table4)
    assert len(rows) == 18
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], {})[
            row["encoder_parameter_multiplier"]
        ] = row["parameters"]
    for method, budgets in by_method.items():
        if method == "PE":
            assert budgets[1] == budgets[8]
            assert budgets[1]["encoder"] == 0
        else:
            assert budgets[8]["encoder"] == 8 * budgets[1]["encoder"]


def test_table4_receipt_is_parameter_only_and_never_claims_reproduction() -> None:
    receipt = build_table4_deferred_receipt(
        load_sdf_repro_config(DEFAULT_TABLE4_CONFIG)
    )
    validate_table4_deferred_receipt(receipt)
    assert receipt["status"] == "deferred_auth_required"
    assert receipt["asset"]["data_access_attempted"] is False
    assert receipt["substitution"] == {"allowed": False, "used": False}
    assert receipt["numeric_results"] == {
        "allowed": False,
        "generated": False,
        "paper_values_embedded": False,
    }
    assert not any(receipt["claims"].values())
    encoded = json.dumps(receipt).lower()
    assert '"metrics"' not in encoded
    assert '"iou"' not in encoded
    assert json.loads(
        (
            ROOT
            / "results"
            / "sdf_repro"
            / "table4_deferred_auth_required.json"
        ).read_text(encoding="utf-8")
    ) == receipt

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (
            ROOT
            / "results"
            / "schemas"
            / "sdf_table4_deferred.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(receipt, schema)


def test_checked_in_volume_validation_receipt_covers_public_three_only() -> None:
    receipt = json.loads(
        (
            ROOT / "results" / "sdf_repro" / "volume_validation.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "passed"
    assert receipt["checksums_verified"] is True
    assert [row["asset_id"] for row in receipt["volumes"]] == list(PUBLIC_ASSETS)
    assert all(row["shape"] == [512, 512, 512] for row in receipt["volumes"])
    assert all(
        row["status"] == "checksum_and_provenance_verified"
        for row in receipt["volumes"]
    )
    assert receipt["stonefish"] == {
        "asset_id": "pitted-stonefish",
        "checked": False,
        "status": "deferred_auth_required",
        "substitution_used": False,
    }


def test_full_matrix_shards_are_disjoint_complete_and_costed() -> None:
    table3 = load_sdf_repro_config(CONFIG_ROOT / "table3_mape.toml")
    table6 = load_sdf_repro_config(CONFIG_ROOT / "table6_l1.toml")
    jobs = enumerate_sdf_jobs((table3, table6))
    shards = [
        shard_sdf_jobs(jobs, rank=rank, world_size=4)
        for rank in range(4)
    ]
    assert len(jobs) == 57
    assert [len(shard) for shard in shards] == [15, 14, 14, 14]
    assert {
        job.index for shard in shards for job in shard
    } == set(range(57))
    for left in range(4):
        for right in range(left + 1, 4):
            assert {job.index for job in shards[left]}.isdisjoint(
                {job.index for job in shards[right]}
            )
    cost = estimate_cost((table3, table6))
    assert cost["optimizer_steps"] == 6_840_000
    assert cost["sampled_training_points"] == 410_400_000_000
    assert cost["full_volume_queries"] == 57 * 512**3
    assert cost["armadillo_flip_renders"] == 9
    assert json.loads(
        (ROOT / "results" / "sdf_repro" / "cost.json").read_text(
            encoding="utf-8"
        )
    ) == cost


class _AnalyticSphere(torch.nn.Module):
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        centered = coords * 2.0 - 1.0
        return centered.norm(dim=1, keepdim=True) - 0.6


def test_chunked_iou_visits_every_voxel_without_materializing_full_grid() -> None:
    volume = _synthetic_sphere_volume(9)
    metrics, images = evaluate_sdf_chunked(
        _AnalyticSphere(),
        volume,
        device=torch.device("cpu"),
        chunk_size=37,
        mape_epsilon=1e-6,
    )
    assert images is None
    assert metrics["evaluated_voxels"] == 9**3
    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["l1"] < 2e-7


def test_first_surface_render_and_official_flip_identity() -> None:
    pytest.importorskip("flip_evaluator")
    resolution = 8
    reference = torch.ones(resolution, resolution, resolution)
    reference[3:] = -1
    accumulator = FirstSurfaceAccumulator(resolution)
    accumulator.update(slice(0, 2), reference[:2], reference[:2])
    accumulator.update(slice(2, 8), reference[2:], reference[2:])
    protocol = OrthographicRenderProtocol()
    prediction_image, reference_image = accumulator.render(protocol)
    np.testing.assert_array_equal(prediction_image, reference_image)
    error_map, mean_error, parameters = evaluate_flip(
        reference_image,
        prediction_image,
        protocol,
    )
    assert error_map.shape == (resolution, resolution, 3)
    assert mean_error == pytest.approx(0.0, abs=1e-8)
    assert parameters["ppd"] == pytest.approx(67.0)


def test_cpu_smoke_checkpoint_resume_matches_uninterrupted(tmp_path: Path) -> None:
    original = load_sdf_repro_config(CONFIG_ROOT / "smoke.toml")
    config = replace(original, methods=(original.methods[0],))
    interrupted_work = tmp_path / "interrupted"
    complete_work = tmp_path / "complete"
    render_root = tmp_path / "renders"

    first = run_shard(
        (config,),
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        work_root=interrupted_work,
        render_root=render_root,
        verify_checksums=False,
        stop_after_steps=2,
    )
    assert first["checkpointed_jobs"] == 1
    resumed = run_shard(
        (config,),
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        work_root=interrupted_work,
        render_root=render_root,
        verify_checksums=False,
    )
    assert resumed["complete_jobs"] == 1
    uninterrupted = run_shard(
        (config,),
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        work_root=complete_work,
        render_root=render_root,
        verify_checksums=False,
    )
    assert uninterrupted["complete_jobs"] == 1

    job = enumerate_sdf_jobs((config,))[0]
    _, resumed_checkpoint = _job_paths(job, interrupted_work)
    _, full_checkpoint = _job_paths(job, complete_work)
    resumed_state = torch.load(
        resumed_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    full_state = torch.load(
        full_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert resumed_state["step"] == full_state["step"] == 4
    for name, expected in full_state["model"].items():
        assert torch.equal(resumed_state["model"][name], expected)


def test_aggregate_is_explicitly_three_shape_not_paper_global(
    tmp_path: Path,
) -> None:
    config = load_sdf_repro_config(CONFIG_ROOT / "table3_mape.toml")
    work_root = tmp_path / "work"
    output_root = tmp_path / "output"
    for job in enumerate_sdf_jobs((config,)):
        result_path, _ = _job_paths(job, work_root)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        expected = job.method.expected_counts()
        result_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "config": {"sha256": config.digest},
                    "instance": job.asset,
                    "method": job.method.name,
                    "parameters": expected,
                    "evaluation": {
                        "metrics": {
                            "iou": job.method.paper_iou[job.asset],
                            "l1": 0.0,
                            "mape": 0.0,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
    manifest = aggregate_config(
        config,
        work_root=work_root,
        output_root=output_root,
    )
    assert manifest["aggregate_label"] == "three_shape_aggregate"
    assert manifest["canonical_four_shape"] is False
    assert manifest["paper_global_comparable"] is False
    aggregate_path = Path(manifest["outputs"]["three_shape_aggregate"])
    with aggregate_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert {row["scope"] for row in rows} == {"three_shape_aggregate"}
    assert {row["shape_count"] for row in rows} == {"3"}
    assert all(row["omitted_shape"] == "pitted-stonefish" for row in rows)
