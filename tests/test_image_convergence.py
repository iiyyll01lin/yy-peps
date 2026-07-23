"""Focused oracles for the bounded Kodak convergence pilot."""

from __future__ import annotations

import copy
import os
import time

import pytest
import torch

import experiments.image_convergence as pilot
from experiments.config import MethodConfig
from experiments.runner import TensorInstance


def test_pilot_manifest_freezes_bounded_representative_matrix():
    manifest = pilot.load_pilot_manifest()
    jobs = pilot.enumerate_pilot_jobs(manifest)
    assert len(jobs) == 18
    assert {job.category for job in jobs} == {"baseline", "peps", "pink"}
    assert {job.seed for job in jobs} == {0, 1}
    assert {job.orientation for job in jobs} == {"landscape", "portrait"}
    assert manifest["training"]["evaluation_budgets"] == [
        1000,
        3000,
        10000,
        20000,
        30000,
        60000,
        90000,
        120000,
    ]
    assert manifest["bounds"]["expected_optimizer_steps"] == 2_160_000
    assert manifest["bounds"]["expected_additional_optimizer_steps"] == 1_620_000
    assert manifest["bounds"]["max_wall_clock_seconds"] == 14_400
    assert manifest["parallelism"]["physical_devices"] == [0, 1]
    assert manifest["parallelism"]["maximum_concurrent_workers"] == 2
    assert manifest["resume_from"]["completed_step"] == 30_000
    assert manifest["verification_status"] == pilot.STATUS


def test_pilot_four_way_shards_are_disjoint_and_complete():
    manifest = pilot.load_pilot_manifest()
    jobs = pilot.enumerate_pilot_jobs(manifest)
    shards = [
        pilot.shard_pilot_jobs(jobs, rank=rank, world_size=4)
        for rank in range(4)
    ]
    assert [len(shard) for shard in shards] == [5, 5, 4, 4]
    indices = [{job.index for job in shard} for shard in shards]
    assert set.union(*indices) == set(range(18))
    assert all(
        left.isdisjoint(right)
        for position, left in enumerate(indices)
        for right in indices[position + 1 :]
    )


def _synthetic_curves(manifest):
    budgets = manifest["training"]["evaluation_budgets"]
    offsets = {"Grid": 0.0, "G-PEPS": 1.0, "G-P-PEPS": 2.0}
    increments = [0.0, 0.4, 0.7, 0.9, 0.95, 1.0, 1.04, 1.08]
    curves = []
    for index, method in enumerate(offsets):
        curves.append(
            {
                "job": {
                    "job_index": index,
                    "instance": "synthetic",
                    "method": method,
                    "category": "synthetic",
                    "seed": 0,
                },
                "points": [
                    {
                        "step": step,
                        "metrics": {
                            "psnr": 20.0 + offsets[method] + increment,
                            "ssim": 0.8,
                            "lsd": 0.2,
                            "mae": 0.1,
                        },
                        "runtime_seconds_cumulative": float(position + 1),
                    }
                    for position, (step, increment) in enumerate(
                        zip(budgets, increments)
                    )
                ],
            }
        )
    return curves


def test_decision_rule_recommends_only_stable_maximum_budget():
    manifest = pilot.load_pilot_manifest()
    curves = _synthetic_curves(manifest)
    stable = pilot.analyse_curves(curves, manifest, complete=True)
    assert stable["outcome"] == "recommended_protocol_assumption"
    assert stable["recommended_budget_steps"] == 120_000
    drifting = copy.deepcopy(curves)
    for curve in drifting:
        curve["points"][-1]["metrics"]["psnr"] += 0.5
    inconclusive = pilot.analyse_curves(drifting, manifest, complete=True)
    assert inconclusive["outcome"] == "inconclusive"
    assert inconclusive["recommended_budget_steps"] is None


def test_checkpoint_cosine_rehorizon_is_explicit_and_global():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=120_000,
    )
    optimizer.param_groups[0]["lr"] = 0.0
    learning_rates = pilot._rehorizon_cosine_scheduler(
        optimizer,
        scheduler,
        completed_step=30_000,
        target_steps=120_000,
    )
    expected = 0.05 * (1.0 + 2**-0.5)
    assert learning_rates == [pytest.approx(expected)]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected)
    assert scheduler.last_epoch == 30_000


def test_worker_liveness_rejects_pre_reboot_identity():
    identity = pilot._process_identity(os.getpid())
    assert identity is not None
    stale = dict(identity)
    stale["boot_id"] = "pre-reboot-boot-id"
    alive, reason = pilot._worker_alive(
        {"pid": os.getpid(), "process_identity": stale}
    )
    assert not alive
    assert reason == "boot_or_process_identity_mismatch"


def test_job_checkpoint_resumes_after_bounded_stop(tmp_path, monkeypatch):
    method = MethodConfig.from_mapping(
        {
            "name": "tiny-grid",
            "factory": "apps.image.build:build_grid",
            "kwargs": {
                "resolution": 4,
                "feature_dim": 2,
                "hidden_dim": 4,
                "num_layers": 2,
                "output_activation": None,
            },
        }
    )
    job = pilot.PilotJob(0, "tiny", "test", "landscape", method, "baseline", 3)
    y, x = torch.meshgrid(
        torch.linspace(0, 1, 16),
        torch.linspace(0, 1, 16),
        indexing="ij",
    )
    coords = torch.stack((x.reshape(-1), y.reshape(-1)), dim=1)
    targets = torch.stack((coords[:, 0], coords[:, 1], coords.mean(dim=1)), dim=1)
    instance = TensorInstance("tiny", coords, targets, shape=(16, 16, 3))
    manifest = {
        "name": "tiny-pilot",
        "paper": "test",
        "dataset": {"id": "synthetic"},
        "methods": [
            {
                "name": "tiny-grid",
                "category": "baseline",
                "factory": method.factory,
                "kwargs": dict(method.kwargs),
            }
        ],
        "seeds": [3],
        "training": {
            "task": "image",
            "loss": "l2",
            "batch_size": 32,
            "model_lr": 0.01,
            "encoder_lr": 0.01,
            "cosine": True,
            "max_steps": 2,
            "evaluation_budgets": [1, 2],
            "checkpoint_every": 1,
            "deadline_poll_every": 1,
            "render_chunk": 256,
        },
        "metrics": ["psnr", "ssim", "lsd", "mae"],
    }
    run_manifest = {
        "run_id": "test-run",
        "pilot_manifest_sha256": "a",
        "code_bundle_sha256": "b",
    }
    monkeypatch.setattr(
        pilot,
        "evaluate_metrics",
        lambda *_args, **_kwargs: {"psnr": 1.0, "ssim": 0.5, "lsd": 0.25},
    )
    first = pilot._run_job(
        job=job,
        instance=instance,
        manifest=manifest,
        manifest_path=pilot.DEFAULT_MANIFEST,
        run_manifest=run_manifest,
        run_dir=tmp_path,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        deadline_epoch=time.time() - 1,
    )
    assert first == "bounded_stop"
    second = pilot._run_job(
        job=job,
        instance=instance,
        manifest=manifest,
        manifest_path=pilot.DEFAULT_MANIFEST,
        run_manifest=run_manifest,
        run_dir=tmp_path,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        deadline_epoch=time.time() + 60,
    )
    assert second == "complete"
    third = pilot._run_job(
        job=job,
        instance=instance,
        manifest=manifest,
        manifest_path=pilot.DEFAULT_MANIFEST,
        run_manifest=run_manifest,
        run_dir=tmp_path,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        deadline_epoch=time.time() + 60,
    )
    assert third == "skipped"
