"""Contracts for the persistent four-GPU image reproduction manager."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANAGER_PATH = ROOT / "scripts" / "manage_image_repro_4gpu.py"
RUNNER_PATH = ROOT / "scripts" / "run_image_repro_4gpu.sh"


def _load_manager():
    spec = importlib.util.spec_from_file_location(
        "peps_image_repro_manager",
        MANAGER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


manager = _load_manager()


def test_virtualenv_python_path_is_not_resolved_to_system_interpreter():
    path = ROOT / ".venv/bin/python"

    assert manager._absolute_without_symlink_resolution(path) == path


def _healthy_status():
    workers = [
        {
            "rank": rank,
            "pid": 1000 + rank,
            "process_alive": True,
            "state": "running",
            "recorded_state": "running",
            "liveness_evidence": {
                "status": "verified_alive",
            },
        }
        for rank in range(4)
    ]
    return {
        "schema": "peps.image_repro_manager_status",
        "schema_version": 1,
        "service": {
            "ActiveState": "active",
            "SubState": "running",
            "MainPID": 999,
        },
        "progress": {
            "accounted_optimizer_steps": 200,
            "expected_optimizer_steps": 1000,
            "optimizer_step_completion_fraction": 0.2,
            "completed_jobs": 0,
            "expected_jobs": 8,
            "checkpointed_incomplete_jobs": 4,
            "active_workers": 4,
            "per_rank": {},
            "workers": workers,
            "checkpoint_errors": [],
            "checkpoint_warnings": [],
            "result_errors": [],
            "worker_status_errors": [],
            "unexpected_result_files": [],
            "unexpected_checkpoint_files": [],
            "incomplete_temporary_outputs": [],
            "output_integrity_ok": True,
        },
        "receipt": {
            "start_accounted_optimizer_steps": 120,
        },
        "competitors": [],
    }


def test_systemd_command_owns_one_control_group(tmp_path: Path):
    authorization = tmp_path / "authorization.json"
    command = manager._build_systemd_run(
        unit="peps-image-repro-table1.service",
        artifact="table1",
        output_root=tmp_path / "results",
        python_bin=ROOT / ".venv/bin/python",
        service_log=tmp_path / "service.log",
        authorization_receipt=authorization,
    )

    assert command[:2] == ["systemd-run", "--user"]
    assert "--collect" in command
    assert "--service-type=exec" in command
    assert "--property=KillMode=control-group" in command
    assert "--property=Restart=no" in command
    assert f"--working-directory={ROOT}" in command
    assert command[-3:] == [
        "table1",
        "--authorization-receipt",
        str(authorization),
    ]
    assert "nohup" not in command
    assert "setsid" not in command


def test_health_requires_four_live_workers_on_distinct_gpus(
    monkeypatch: pytest.MonkeyPatch,
):
    status = _healthy_status()
    monkeypatch.setattr(
        manager,
        "_worker_gpu",
        lambda pid: str(pid - 1000),
    )

    health = manager._health_payload(status)

    assert health["healthy"] is True
    assert health["steps_since_managed_start"] == 80
    assert all(check["ok"] for check in health["rank_checks"].values())


def test_health_rejects_wrong_gpu_or_competing_job(
    monkeypatch: pytest.MonkeyPatch,
):
    status = _healthy_status()
    status["competitors"] = [{"pid": 2000, "command": "texture pilot"}]
    monkeypatch.setattr(manager, "_worker_gpu", lambda _pid: "0")

    health = manager._health_payload(status)

    assert health["healthy"] is False
    assert health["rank_checks"]["1"]["ok"] is False


def test_reproduction_filter_is_repo_command_specific():
    records = [
        manager.ProcessRecord(
            pid=1,
            ppid=0,
            pgid=1,
            sid=1,
            cwd=str(ROOT),
            command=".venv/bin/python -m experiments.image_repro run",
        ),
        manager.ProcessRecord(
            pid=2,
            ppid=0,
            pgid=2,
            sid=2,
            cwd=str(ROOT),
            command="/usr/bin/python unrelated.py",
        ),
    ]

    assert manager._reproduction_processes(records) == [records[0]]


def test_table1_start_is_disabled_even_with_authorization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(manager, "_systemd_available", lambda: (True, "running"))
    monkeypatch.setattr(
        manager,
        "_unit_properties",
        lambda _unit: {
            "LoadState": "not-found",
            "ActiveState": "inactive",
        },
    )
    monkeypatch.setattr(manager, "_reproduction_processes", lambda: [])

    for authorization in (None, tmp_path / "authorization.json"):
        with pytest.raises(RuntimeError, match="launch and recovery are disabled"):
            manager.start(
                artifact="table1",
                output_root=tmp_path,
                unit="peps-image-repro-table1.service",
                python_bin=ROOT / ".venv/bin/python",
                authorization_receipt=authorization,
            )


def test_shell_launcher_has_signal_cleanup_and_valid_syntax():
    completed = subprocess.run(
        ["bash", "-n", str(RUNNER_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "trap terminate_workers HUP INT TERM" in source
    assert 'kill -TERM "${pid}"' in source
