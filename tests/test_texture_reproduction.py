"""Focused CPU oracles for the texture-specific reproduction phase."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import pytest
import torch

from apps.texture.build import build_paper_texture
from experiments.config import load_experiment_config
from experiments.runner import ExperimentRunner, RunSpec, enumerate_jobs
from experiments.texture_repro import (
    ARTIFACT_CONFIGS,
    PAPER_TABLE2,
    PILOT_CONFIG,
    SEMANTICS,
    _artifact_output,
    _load_pilot_config,
    _map_observations,
    _pilot_evaluation_instance,
    _process_identity,
    _synthetic_instance,
    _worker_liveness,
    architecture_receipt,
    artifact_progress,
    generate_figure8,
    job_plan,
    pilot_job_plan,
    run_artifact,
    validate_manifest_consumption,
    write_pilot_report,
    write_reports,
)
from peps.train import split_encoder_decoder_parameters


ROOT = Path(__file__).resolve().parents[1]


def test_texture_manifest_consumption_covers_all_18_dynamic_sets() -> None:
    receipt = validate_manifest_consumption()
    assert receipt["set_count"] == 18
    assert receipt["map_count"] == 78
    assert receipt["total_output_channels_across_sets"] == 234
    assert receipt["all_native_4k"]
    assert tuple(receipt["semantic_counts"]) == SEMANTICS
    assert {row["provider"] for row in receipt["sets"]} == {
        "polyhaven",
        "ambientcg",
    }
    assert {row["output_channels"] for row in receipt["sets"]} > {9}
    assert all(
        row["output_channels"] == 3 * row["map_count"]
        for row in receipt["sets"]
    )


EXPECTED_TABLE_INPUTS = {
    "LPE": 16,
    "NTC_N": 80,
    "BI-Grid": 17,
    "Grid-PEPS4F": 153,
    "Grid-PinkPEPS4F": 47,
    "NTC_PEPS": 624,
    "NTC_PinkPEPS": 206,
    "Grid-PEPS4F-25": 117,
    "Grid-PinkPEPS4F-25": 35,
    "NTC_PEPS-25": 471,
    "NTC_PinkPEPS-25": 155,
}


@pytest.mark.parametrize(
    "method_name,expected_input",
    tuple(EXPECTED_TABLE_INPUTS.items()),
)
def test_table2_builders_match_paper_budgets_and_decoder_inputs(
    method_name: str,
    expected_input: int,
) -> None:
    config = load_experiment_config(ARTIFACT_CONFIGS["table2"])
    method = next(item for item in config.methods if item.name == method_name)
    model, reported = build_paper_texture(
        str(method.kwargs["method"]),
        num_textures=5,
    )
    encoder, decoder = split_encoder_decoder_parameters(model)
    decoder_module = model.model if hasattr(model, "model") else model[-1]
    assert decoder_module.in_dim == expected_input
    assert sum(parameter.numel() for parameter in encoder) == (
        method.expected_encoder_params
    )
    assert reported == sum(parameter.numel() for parameter in model.parameters())
    assert reported == sum(parameter.numel() for parameter in (*encoder, *decoder))
    del model, encoder, decoder, decoder_module
    gc.collect()


def test_table2_and_frequency_sweep_receipts_are_exact() -> None:
    table = load_experiment_config(ARTIFACT_CONFIGS["table2"])
    assert tuple(method.name for method in table.methods) == tuple(PAPER_TABLE2)
    receipts = {
        method.name: architecture_receipt(method, output_channels=15)
        for method in table.methods
    }
    assert {
        name: receipt["decoder_input_dim"] for name, receipt in receipts.items()
    } == EXPECTED_TABLE_INPUTS
    assert receipts["BI-Grid"]["encoder_params"] == 17_825_792
    assert receipts["Grid-PEPS4F-25"]["encoder_params"] == 13_631_488
    assert receipts["NTC_PEPS"]["encoder_params"] == 17_825_792
    assert receipts["NTC_PEPS-25"]["encoder_params"] == 13_369_344
    assert receipts["Grid-PEPS4F-25"]["encoder_params"] / receipts[
        "Grid-PEPS4F"
    ]["encoder_params"] == pytest.approx(13 / 17)
    assert receipts["NTC_PEPS-25"]["encoder_params"] / receipts[
        "NTC_PEPS"
    ]["encoder_params"] == pytest.approx(0.75)

    sweep = load_experiment_config(ARTIFACT_CONFIGS["sweep"])
    sweep_receipts = {
        method.name: architecture_receipt(method, output_channels=15)
        for method in sweep.methods
    }
    assert {
        name: receipt["decoder_input_dim"]
        for name, receipt in sweep_receipts.items()
    } == {
        "Grid-PEPS3F": 119,
        "Grid-PEPS4F": 153,
        "Grid-PinkPEPS3F": 45,
        "Grid-PinkPEPS4F": 47,
        "NTC_PEPS3F": 488,
        "NTC_PEPS4F": 624,
        "NTC_PinkPEPS3F": 198,
        "NTC_PinkPEPS4F": 206,
    }
    assert {
        receipt["peps_frequencies"] for receipt in sweep_receipts.values()
    } == {3, 4}


def test_three_frequency_builders_are_executable() -> None:
    grid, _ = build_paper_texture(
        "grid_pinkpeps3f",
        num_textures=2,
        resolution=8,
        feature_dim=4,
        hidden_dim=8,
        num_layers=2,
    )
    ntc, _ = build_paper_texture(
        "ntc_peps3f",
        num_textures=2,
        signal_resolution=16,
        g0_resolution=8,
        g0_feature_dim=2,
        g1_resolution=4,
        g1_feature_dim=3,
        hidden_dim=8,
        num_layers=2,
    )
    assert grid.projector.num_frequencies == 3
    assert grid(torch.rand(2, 2)).shape == (2, 6)
    assert ntc[0].projector.num_frequencies == 3
    assert ntc(torch.rand(2, 2)).shape == (2, 6)


def test_asset_method_sharding_is_disjoint_balanced_and_seed_paired() -> None:
    table = job_plan("table2", world_size=4, include_jobs=True)
    assert table["expected_jobs"] == 18 * 11 * 3
    assert table["expected_optimizer_steps"] == 71_280_000
    assert table["expected_training_samples"] == 4_276_800_000_000
    assert sorted(item["jobs"] for item in table["per_rank"].values()) == [
        147,
        147,
        150,
        150,
    ]
    jobs = table["jobs"]
    assert {job["job_index"] for job in jobs} == set(range(594))
    pair_ranks: dict[tuple[str, str], set[int]] = {}
    pair_seeds: dict[tuple[str, str], set[int]] = {}
    for job in jobs:
        key = (job["instance"], job["method"])
        pair_ranks.setdefault(key, set()).add(job["rank"])
        pair_seeds.setdefault(key, set()).add(job["seed"])
    assert all(len(ranks) == 1 for ranks in pair_ranks.values())
    assert all(seeds == {0, 1, 2} for seeds in pair_seeds.values())

    sweep = job_plan("sweep", world_size=4)
    assert sweep["expected_jobs"] == 18 * 8 * 3
    assert {item["jobs"] for item in sweep["per_rank"].values()} == {108}


def test_convergence_pilot_is_bounded_semantic_and_seed_paired() -> None:
    config = _load_pilot_config()
    assert config.source == PILOT_CONFIG.resolve()
    plan = pilot_job_plan(world_size=4, include_jobs=True)
    assert plan["canonical"] is False
    assert plan["expected_trajectories"] == 2 * 5 * 3
    assert plan["expected_optimizer_steps"] == 150_000
    assert plan["source_optimizer_steps"] == 6_000
    assert plan["expected_additional_optimizer_steps"] == 144_000
    assert plan["step_budgets"] == [10, 50, 200, 1_000, 2_000, 5_000]
    assert plan["parallelism"]["physical_devices"] == [0, 1]
    assert plan["parallelism"]["maximum_concurrent_workers"] == 2
    assert plan["resume_from"]["maximum_budget"] == 200
    assert tuple(plan["semantic_coverage"]) == SEMANTICS
    assert not plan["safety"]["launches_full_table2"]
    assert (
        plan["expected_optimizer_steps"]
        < plan["safety"]["full_table2_optimizer_steps"] / 400
    )
    pair_ranks: dict[tuple[str, str], set[int]] = {}
    pair_seeds: dict[tuple[str, str], set[int]] = {}
    for job in plan["jobs"]:
        key = (job["instance"], job["method"])
        pair_ranks.setdefault(key, set()).add(job["rank"])
        pair_seeds.setdefault(key, set()).add(job["seed"])
    assert all(len(ranks) == 1 for ranks in pair_ranks.values())
    assert all(seeds == {0, 1, 2} for seeds in pair_seeds.values())


def test_pilot_evaluation_uses_fixed_native_pixel_lattice() -> None:
    instance = _synthetic_instance()
    evaluation = _pilot_evaluation_instance(instance, side=8)
    assert evaluation.shape == (8, 8, 9)
    assert evaluation.coords.shape == (64, 2)
    assert evaluation.targets.shape == (64, 9)
    assert evaluation.coords.min().item() == 0.0
    assert evaluation.coords.max().item() == 1.0
    assert evaluation.metadata["texture_maps"] == instance.metadata["texture_maps"]


def test_texture_worker_liveness_requires_boot_scoped_identity(
) -> None:
    alive, evidence = _worker_liveness({"pid": os.getpid()})
    assert not alive
    assert evidence["reason"] == "worker_record_has_no_process_identity"

    identity = _process_identity(os.getpid())
    assert identity is not None
    alive, evidence = _worker_liveness(
        {"pid": os.getpid(), "process_identity": identity}
    )
    assert alive
    assert evidence["reason"] == "boot_id_start_time_and_command_match"

    stale = dict(identity)
    stale["boot_id"] = "a-different-boot"
    alive, evidence = _worker_liveness(
        {"pid": os.getpid(), "process_identity": stale}
    )
    assert not alive
    assert evidence["reason"] == "pid_identity_mismatch"


def test_texture_smoke_runner_writes_resumable_sharded_records(tmp_path: Path) -> None:
    result = run_artifact(
        "smoke",
        output_root=tmp_path,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        instance_ids=None,
        methods=None,
        force=False,
        allow_protocol_assumptions=False,
        verification_receipt=None,
    )
    assert result["records"] == 3
    output = _artifact_output(tmp_path, "smoke")
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in output.glob("raw/**/*.json")
    ]
    assert len(records) == 3
    assert all(record["architecture"]["output_channels"] == 9 for record in records)
    assert all(
        record["texture_parallelism"]["mode"] == "asset_method_job_shard"
        for record in records
    )
    assert len(list(output.glob("checkpoints/**/*.pt"))) == 3
    assert artifact_progress("smoke", output_root=tmp_path)["complete"]

    repeated = run_artifact(
        "smoke",
        output_root=tmp_path,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        instance_ids=None,
        methods=None,
        force=False,
        allow_protocol_assumptions=False,
        verification_receipt=None,
    )
    assert repeated["records"] == 3


def test_generic_checkpoint_interrupt_resumes_texture_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.runner as runner_module

    config = load_experiment_config(ARTIFACT_CONFIGS["smoke"])
    instance = _synthetic_instance()
    planned = enumerate_jobs(config, (instance,))[0]
    spec = RunSpec(instance, planned.method, planned.seed, planned.index)
    runner = ExperimentRunner(
        config,
        tmp_path,
        device=torch.device("cpu"),
    )
    real_save = runner_module.atomic_torch_save
    interrupted = False

    def save_then_interrupt(path, payload):
        nonlocal interrupted
        real_save(path, payload)
        if int(payload["step"]) == 1 and not interrupted:
            interrupted = True
            raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(runner_module, "atomic_torch_save", save_then_interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        runner.run_one(spec)
    result_path, checkpoint_path = runner._paths(spec)
    assert not result_path.exists()
    assert torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )["step"] == 1

    monkeypatch.setattr(runner_module, "atomic_torch_save", real_save)
    record = runner.run_one(spec)
    assert record["training"]["total_steps"] == 2
    assert result_path.is_file()
    assert torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )["step"] == 2


def test_table2_aggregation_weights_individual_maps_not_sets() -> None:
    records = [
        {
            "method": "LPE",
            "metrics": {
                "psnr/map/a/DIFF": 10.0,
                "psnr/map/b/DIFF": 20.0,
            },
        },
        {
            "method": "LPE",
            "metrics": {
                "psnr/map/c/rough": 40.0,
            },
        },
    ]
    grouped = _map_observations(records)
    assert grouped[("LPE", "psnr", "global")] == [10.0, 20.0, 40.0]
    assert sum(grouped[("LPE", "psnr", "global")]) / 3 == pytest.approx(
        70 / 3
    )


def test_empty_reports_validate_schema_and_figure8_blocks(tmp_path: Path) -> None:
    status = write_reports(
        output_root=tmp_path,
        destination_dir=tmp_path / "texture_repro",
    )
    assert not status["table2"]["complete"]
    table_path = tmp_path / "texture_repro/table2.json"
    payload = json.loads(table_path.read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 11
    assert all(row["verification_status"] == "partial_do_not_interpret" for row in payload["rows"])

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (
            ROOT / "results/schemas/texture_table2_report.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(payload, schema)

    figure_path = tmp_path / "texture_repro/figure8.png"
    figure_status = generate_figure8(
        output_root=tmp_path,
        output_path=figure_path,
        status_path=tmp_path / "texture_repro/figure8_status.json",
        device=torch.device("cpu"),
        methods=None,
        seed=0,
        crop_x=None,
        crop_y=None,
        crop_size=100,
        verification_receipt=None,
    )
    assert figure_status["status"] == "blocked"
    assert not figure_path.exists()


def test_empty_convergence_pilot_report_is_schema_valid(
    tmp_path: Path,
) -> None:
    output = tmp_path / "convergence_pilot.json"
    csv_output = tmp_path / "convergence_pilot_observations.csv"
    payload = write_pilot_report(
        output_root=tmp_path,
        output_path=output,
        csv_path=csv_output,
    )
    assert payload["decision"]["status"] == "inconclusive_bounded_pilot"
    assert payload["decision"]["recommended_table2_steps"] is None
    assert not payload["decision"]["full_71m_step_run_authorized"]
    assert payload["evidence"]["raw_csv"]["rows"] == 0

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (
            ROOT
            / "results/schemas/texture_convergence_pilot.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(payload, schema)
