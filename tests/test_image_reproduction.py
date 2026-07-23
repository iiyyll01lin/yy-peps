"""Oracles for the image/core reproduction matrix."""

from __future__ import annotations

import importlib
import inspect
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest
import torch

import experiments.full_run_authorization as full_auth
from data.manifest import load_manifest
from experiments.config import load_experiment_config
from experiments.image_figures import odd_alignment_report
from experiments.image_models import (
    FrequencyPairSumAggregator,
    FullSumAggregator,
    build_paper_image_ablation,
)
from experiments.image_repro import (
    ARTIFACT_CONFIGS,
    PAPER_TABLE1,
    PAPER_TABLE5,
    _artifact_output,
    _process_identity,
    artifact_progress,
    run_artifact,
    write_summary_report,
)
from experiments.full_run_authorization import (
    IMAGE_TABLE1_CONFIG,
    validate_image_table1_authorization,
)
from experiments.runner import TensorInstance, enumerate_jobs
from peps.train import split_encoder_decoder_parameters


EXPECTED_JOBS = {
    "table1": 24 * 9 * 3,
    "table5": 24 * 3 * 3,
    "core-ablations": 24 * 5 * 3,
    "recipe-ablations": 24 * 5 * 3,
    "smoke": 24 * 9,
    "appendix-smoke": 13,
}


def _dummy_instances(config):
    requested = config.runner.get("instance_ids")
    requested_ids = None if requested is None else set(requested)
    return tuple(
        TensorInstance(item["id"], torch.zeros(1, 2), torch.zeros(1, 3))
        for item in load_manifest("kodak")["images"]
        if requested_ids is None or item["id"] in requested_ids
    )


def _build_method(method, signal_resolution):
    module_name, function_name = method.factory.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    kwargs = dict(method.kwargs)
    signature = inspect.signature(factory)
    if (
        "signal_resolution" in signature.parameters
        or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    ):
        kwargs["signal_resolution"] = signal_resolution
    return factory(**kwargs)[0]


@pytest.mark.parametrize("artifact", tuple(ARTIFACT_CONFIGS))
def test_image_configs_freeze_job_counts_and_parameter_budgets(artifact):
    config = load_experiment_config(ARTIFACT_CONFIGS[artifact])
    assert len(enumerate_jobs(config, _dummy_instances(config))) == EXPECTED_JOBS[artifact]
    for method in config.methods:
        landscape = _build_method(method, (768, 512))
        portrait = _build_method(method, (512, 768))
        for model in (landscape, portrait):
            encoder, decoder = split_encoder_decoder_parameters(model)
            encoder_count = sum(parameter.numel() for parameter in encoder)
            total_count = encoder_count + sum(
                parameter.numel() for parameter in decoder
            )
            assert encoder_count == method.expected_encoder_params
            assert total_count == method.expected_total_params


def test_appendix_sum_aggregators_follow_frozen_interpretation():
    latents = torch.arange(2 * 7 * 3, dtype=torch.float32).reshape(2, 7, 3)
    full = FullSumAggregator(7, 3)
    assert torch.equal(full(latents), latents.sum(dim=1))

    paired = FrequencyPairSumAggregator(3, 3)
    expected = torch.cat(
        (
            latents[:, 0],
            latents[:, 1] + latents[:, 4],
            latents[:, 2] + latents[:, 5],
            latents[:, 3] + latents[:, 6],
        ),
        dim=1,
    )
    assert torch.equal(paired(latents), expected)


def test_no_sharing_ablation_is_independent_and_budget_matched():
    model, _ = build_paper_image_ablation(
        "no_sharing",
        no_sharing_resolution=(74, 48),
    )
    assert not model.shared_encoder
    assert model.encoders is not None
    assert len(model.encoders) == 7
    assert len({id(encoder.grid) for encoder in model.encoders}) == 7
    encoder, _ = split_encoder_decoder_parameters(model)
    count = sum(parameter.numel() for parameter in encoder)
    assert count == 422_688
    assert abs(count / 426_496 - 1.0) < 0.01

    loss = model(torch.rand(4, 2)).sum()
    loss.backward()
    assert all(encoder.grid.grad is not None for encoder in model.encoders)


def test_remove_original_and_brownian_ablation_shapes():
    no_original, _ = build_paper_image_ablation("no_original_point")
    assert no_original.projector.include_input is False
    assert no_original.projector.num_points == 6
    assert no_original(torch.rand(5, 2)).shape == (5, 3)

    brownian, _ = build_paper_image_ablation("brownian")
    assert brownian.aggregator.out_dim == 29
    assert brownian(torch.rand(5, 2)).shape == (5, 3)


def test_wire_sensitivity_is_finite_and_explicit():
    model, _ = build_paper_image_ablation(
        "wire",
        wire_omega=20.0,
        wire_scale=10.0,
    )
    output = model(torch.rand(8, 2))
    assert output.shape == (8, 3)
    assert torch.isfinite(output).all()


def test_odd_grid_coordinate_contract_has_no_half_texel_shift():
    report = odd_alignment_report()
    assert report["aligned_max_abs_error"] <= 1e-5
    assert report["half_texel_mean_abs_error"] > 0.1


def test_empty_progress_is_zero_not_a_synthetic_result(tmp_path):
    progress = artifact_progress("table1", output_root=tmp_path)
    assert progress["expected_jobs"] == EXPECTED_JOBS["table1"]
    assert progress["completed_jobs"] == 0
    assert progress["accounted_optimizer_steps"] == 0
    assert progress["job_completion_fraction"] == 0.0
    assert not progress["complete"]


def test_worker_liveness_requires_boot_scoped_process_identity(tmp_path):
    output_dir = _artifact_output(tmp_path, "table1")
    output_dir.mkdir(parents=True)
    worker_path = output_dir / "worker-rank-0.json"
    worker_path.write_text(
        json.dumps(
            {
                "schema": "peps.image_worker_status",
                "schema_version": 1,
                "artifact": "table1",
                "rank": 0,
                "world_size": 4,
                "pid": os.getpid(),
                "state": "running",
                "selected_dataset_receipts": [],
            }
        ),
        encoding="utf-8",
    )

    progress = artifact_progress("table1", output_root=tmp_path)
    assert progress["active_workers"] == 0
    worker = progress["workers"][0]
    assert not worker["process_alive"]
    assert worker["effective_state"] == "stopped_incomplete"
    assert worker["state"] == "stopped_incomplete"
    assert (
        worker["liveness_evidence"]["reason"]
        == "worker_record_has_no_process_identity"
    )

    identity = _process_identity(os.getpid())
    if identity is None:
        pytest.skip("Linux /proc process identity is unavailable")
    payload = json.loads(worker_path.read_text(encoding="utf-8"))
    payload["process_identity"] = identity
    worker_path.write_text(json.dumps(payload), encoding="utf-8")
    progress = artifact_progress("table1", output_root=tmp_path)
    assert progress["active_workers"] == 1
    assert progress["workers"][0]["effective_state"] == "running"
    assert progress["workers"][0]["state"] == "running"


def _table1_authorization(now: datetime, boot_id: str) -> dict[str, object]:
    return {
        "schema": "peps.full_run_authorization",
        "schema_version": 1,
        "artifact": "image-table1",
        "authorization_scope": "launch-or-resume-full-matrix",
        "authorized": True,
        "authorization_basis": (
            "independent-explicit-approval-after-full-reproduction-gate"
        ),
        "bounded_pilot_authorizes_full_run": False,
        "config_sha256": hashlib.sha256(
            IMAGE_TABLE1_CONFIG.read_bytes()
        ).hexdigest(),
        "expected_jobs": 648,
        "expected_optimizer_steps": 77_760_000,
        "boot_id": boot_id,
        "issued_at_utc": now.isoformat(),
        "expires_at_utc": (now + timedelta(minutes=30)).isoformat(),
        "approved_by": "test-approver",
        "reason": "explicit test-only approval",
        "approval_id": "test-approval-001",
    }


def test_table1_authorization_is_independent_boot_scoped_and_short_lived(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(full_auth, "IMAGE_TABLE1_RECOVERY_ENABLED", True)
    now = datetime(2026, 7, 22, 2, 0, tzinfo=timezone.utc)
    boot_id = "11111111-2222-3333-4444-555555555555"
    proc_root = tmp_path / "proc"
    boot_path = proc_root / "sys/kernel/random/boot_id"
    boot_path.parent.mkdir(parents=True)
    boot_path.write_text(boot_id, encoding="utf-8")
    receipt_path = tmp_path / "authorization.json"
    payload = _table1_authorization(now, boot_id)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    assert validate_image_table1_authorization(
        receipt_path,
        proc_root=proc_root,
        now=now + timedelta(minutes=1),
    )["approval_id"] == "test-approval-001"

    payload["boot_id"] = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="another boot"):
        validate_image_table1_authorization(
            receipt_path,
            proc_root=proc_root,
            now=now + timedelta(minutes=1),
        )

    payload = _table1_authorization(now, boot_id)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expired"):
        validate_image_table1_authorization(
            receipt_path,
            proc_root=proc_root,
            now=now + timedelta(hours=1),
        )


def test_bounded_pilot_and_bare_launcher_cannot_authorize_table1(tmp_path):
    with pytest.raises(ValueError, match="authorization receipt"):
        validate_image_table1_authorization(
            Path("results/image_convergence/receipt.json")
        )

    with pytest.raises(ValueError, match="launch and recovery are disabled"):
        run_artifact(
            "table1",
            output_root=tmp_path,
            rank=0,
            world_size=4,
            device=torch.device("cuda:0"),
            instance_ids=None,
            methods=None,
            force=False,
            allow_protocol_assumptions=True,
        )

    environment = os.environ.copy()
    environment["PEPS_OUTPUT_ROOT"] = str(tmp_path / "script-output")
    environment["PEPS_PYTHON"] = sys.executable
    result = subprocess.run(
        ["bash", "scripts/run_image_repro_4gpu.sh", "table1"],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 3
    assert "Table 1 launch and recovery are disabled" in result.stderr
    assert not (tmp_path / "script-output").exists()


def test_progress_quarantines_corrupt_and_unplanned_outputs(tmp_path):
    output_dir = _artifact_output(tmp_path, "table1")
    result_path = output_dir / "raw/kodim01/PE/seed-0.json"
    checkpoint_path = (
        output_dir / "checkpoints/raw/kodim01/PE/seed-0.pt"
    )
    result_path.parent.mkdir(parents=True)
    checkpoint_path.parent.mkdir(parents=True)
    result_path.write_text('{"schema_version":', encoding="utf-8")
    checkpoint_path.write_bytes(b"not a torch checkpoint")
    (output_dir / "raw/unplanned.json").write_text("{}", encoding="utf-8")
    temporary = output_dir / "checkpoints/.interrupted.tmp"
    temporary.write_bytes(b"partial")

    progress = artifact_progress("table1", output_root=tmp_path)
    assert progress["completed_jobs"] == 0
    assert progress["accounted_optimizer_steps"] == 0
    assert len(progress["result_errors"]) == 1
    assert len(progress["checkpoint_errors"]) == 1
    assert progress["unexpected_result_files"] == [
        str(output_dir / "raw/unplanned.json")
    ]
    assert progress["incomplete_temporary_outputs"] == [str(temporary)]
    assert not progress["output_integrity_ok"]

    rows = write_summary_report(
        output_root=tmp_path,
        destination=tmp_path / "summary.csv",
    )
    assert rows == []


def test_paper_reference_values_are_transcribed_from_extended_source():
    assert PAPER_TABLE1["G-P-PEPS"]["psnr"] == 47.83
    assert PAPER_TABLE1["G-P-PEPS-25"]["psnr"] == 44.89
    assert PAPER_TABLE5["Grid"]["psnr_l1"] == 40.871
    assert PAPER_TABLE5["G-P-PEPS"]["ssim_l2"] == 0.993
