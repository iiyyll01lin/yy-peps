"""CPU logic tests plus an opt-in four-GPU DDP smoke test."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from peps.distributed import (
    DistributedContext,
    ddp_loss_scale,
    distributed_barrier,
    local_batch_slice,
    local_minibatch_indices,
    per_rank_batch_sizes,
    resolve_distributed_environment,
)
from peps.train import PaperTrainConfig, fit_paper, fit_paper_distributed


ROOT = Path(__file__).resolve().parents[1]


def _cpu_context(rank: int = 0, world_size: int = 1) -> DistributedContext:
    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        device=torch.device("cpu"),
        backend=None,
        process_group_initialized=False,
    )


def test_global_batch_is_split_without_replication():
    global_indices = torch.arange(60_000)
    slices = [
        local_minibatch_indices(
            global_indices,
            _cpu_context(rank, 4),
        )
        for rank in range(4)
    ]
    assert [part.numel() for part in slices] == [15_000] * 4
    assert torch.equal(torch.cat(slices), global_indices)
    assert per_rank_batch_sizes(60_003, 4) == (15_001, 15_001, 15_001, 15_000)
    assert local_batch_slice(10, rank=1, world_size=3) == slice(4, 7)


def test_uneven_ddp_loss_scale_recovers_global_sample_mean():
    local_sizes = per_rank_batch_sizes(10, 3)
    local_mean_gradients = (2.0, 5.0, 11.0)
    ddp_average = sum(
        mean * ddp_loss_scale(size, 10, 3)
        for mean, size in zip(local_mean_gradients, local_sizes)
    ) / 3
    expected = sum(
        mean * size
        for mean, size in zip(local_mean_gradients, local_sizes)
    ) / 10
    assert ddp_average == pytest.approx(expected)


def test_torchrun_environment_resolution_is_strict():
    assert resolve_distributed_environment({}) == (0, 1, 0)
    assert resolve_distributed_environment(
        {"RANK": "2", "WORLD_SIZE": "4", "LOCAL_RANK": "2"}
    ) == (2, 4, 2)
    with pytest.raises(ValueError, match="incomplete torchrun environment"):
        resolve_distributed_environment({"RANK": "0", "WORLD_SIZE": "4"})


def test_distributed_barrier_selects_gpu_but_not_cpu_device(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda **kwargs: calls.append(kwargs),
    )
    distributed_barrier(
        DistributedContext(
            rank=2,
            world_size=4,
            local_rank=2,
            device=torch.device("cuda", 2),
            backend="nccl",
            process_group_initialized=True,
        )
    )
    distributed_barrier(
        DistributedContext(
            rank=0,
            world_size=2,
            local_rank=0,
            device=torch.device("cpu"),
            backend="gloo",
            process_group_initialized=True,
        )
    )
    distributed_barrier(_cpu_context())
    assert calls == [{"device_ids": [2]}, {}]


def test_distributed_checkpoint_has_clean_state_dict_and_resumes_single_cpu():
    coords = torch.linspace(0, 1, 20).unsqueeze(1)
    targets = coords.square()
    initial = torch.nn.Linear(1, 1)
    distributed_model = torch.nn.Linear(1, 1)
    distributed_model.load_state_dict(initial.state_dict())
    uninterrupted = torch.nn.Linear(1, 1)
    uninterrupted.load_state_dict(initial.state_dict())

    first_config = PaperTrainConfig(
        task="image",
        loss="l2",
        steps=2,
        batch_size=7,
        model_lr=0.01,
        seed=9,
        checkpoint_every=1,
        device=torch.device("cpu"),
    )
    _, checkpoint = fit_paper_distributed(
        distributed_model,
        coords,
        targets,
        first_config,
        context=_cpu_context(),
        return_state=True,
    )
    assert checkpoint["parallelism"]["mode"] == "ddp_single_job"
    assert checkpoint["parallelism"]["global_batch_size"] == 7
    assert all(not key.startswith("module.") for key in checkpoint["model"])

    full_config = PaperTrainConfig(
        task="image",
        loss="l2",
        steps=4,
        batch_size=7,
        model_lr=0.01,
        seed=9,
        checkpoint_every=1,
        device=torch.device("cpu"),
    )
    fit_paper(
        distributed_model,
        coords,
        targets,
        full_config,
        resume_state=checkpoint,
    )
    fit_paper(
        uninterrupted,
        coords,
        targets,
        full_config,
    )
    for expected, actual in zip(
        uninterrupted.parameters(),
        distributed_model.parameters(),
    ):
        assert torch.equal(expected, actual)


def test_four_gpu_representative_ddp_smoke(tmp_path: Path):
    if os.environ.get("PEPS_RUN_4GPU_TESTS") != "1":
        pytest.skip("set PEPS_RUN_4GPU_TESTS=1 for the hardware smoke")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        pytest.skip("four visible PyTorch GPUs are required")

    output = tmp_path / "training.json"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        "-m",
        "experiments.multigpu",
        "worker",
        "--kind",
        "training",
        "--output",
        str(output),
        "--global-batch-size",
        "1024",
        "--dataset-size",
        "4096",
        "--training-warmup",
        "1",
        "--training-steps",
        "2",
    ]
    environment = os.environ.copy()
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    # This runner boots with iommu=pt and uses ROCr dma-buf IPC. Require the
    # same strict direct transport as the real-workload benchmark.
    environment["HSA_ENABLE_IPC_MODE_LEGACY"] = "0"
    environment["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
    environment["NCCL_P2P_DISABLE"] = "0"
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "barrier(): using the device under current context" not in completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["parallelism"]["mode"] == "ddp_single_job"
    assert record["parallelism"]["world_size"] == 4
    assert record["workload"]["per_rank_batch_sizes"] == [256] * 4
    assert record["samples_per_second"] > 0


def test_four_gpu_experiment_entrypoint_checkpoint_smoke(tmp_path: Path):
    if os.environ.get("PEPS_RUN_4GPU_TESTS") != "1":
        pytest.skip("set PEPS_RUN_4GPU_TESTS=1 for the hardware smoke")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        pytest.skip("four visible PyTorch GPUs are required")

    line = torch.linspace(0, 1, 8)
    y, x = torch.meshgrid(line, line, indexing="ij")
    coords = torch.stack((x.reshape(-1), y.reshape(-1)), dim=1)
    targets = torch.stack(
        (coords[:, 0], coords[:, 1], (coords[:, 0] + coords[:, 1]) / 2),
        dim=1,
    )
    input_path = tmp_path / "input.pt"
    torch.save(
        {
            "instances": [
                {
                    "name": "synthetic",
                    "coords": coords,
                    "targets": targets,
                    "shape": (8, 8, 3),
                }
            ]
        },
        input_path,
    )
    output = tmp_path / "ddp-output"
    config_path = ROOT / "configs/paper/image_smoke.toml"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        "-m",
        "experiments.ddp",
        "--config",
        str(config_path),
        "--input",
        str(input_path),
        "--output",
        str(output),
        "--instance",
        "synthetic",
        "--method",
        "G-PEPS-smoke",
        "--seed",
        "0",
    ]
    environment = os.environ.copy()
    environment["HSA_ENABLE_IPC_MODE_LEGACY"] = "0"
    environment["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
    environment["NCCL_P2P_DISABLE"] = "0"
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "barrier(): using the device under current context" not in completed.stderr
    record = json.loads((output / "result.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        output / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert record["parallelism"]["mode"] == "ddp_single_job"
    assert record["parallelism"]["global_batch_size"] == 32
    assert record["parallelism"]["per_rank_batch_sizes"] == [8] * 4
    assert record["parallelism"]["rccl_p2p_disabled"] is False
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert record["config_sha256"] == config_sha256
    assert record["git_commit"]
    assert checkpoint["parallelism"]["mode"] == "ddp_single_job"
    assert checkpoint["job"]["method"] == "G-PEPS-smoke"
    assert checkpoint["job"]["config_sha256"] == config_sha256
    assert all(not key.startswith("module.") for key in checkpoint["model"])
