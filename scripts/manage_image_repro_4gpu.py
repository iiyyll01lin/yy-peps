#!/usr/bin/env python3
"""Manage four-GPU image reproduction as a persistent user systemd service."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_image_repro_4gpu.sh"
DEFAULT_OUTPUT_ROOT = ROOT / "results"
TABLE1_RECOVERY_ENABLED = False
ALLOWED_ARTIFACTS = {
    "table1",
    "table5",
    "core-ablations",
    "recipe-ablations",
    "smoke",
    "appendix-smoke",
}
UNIT_PATTERN = re.compile(r"^[A-Za-z0-9_.@:-]+\.service$")
REPRODUCTION_MARKERS = (
    "experiments.image_repro",
    "scripts/run_image_repro_4gpu.sh",
    "experiments.image_convergence",
    "experiments.real_workload",
    "experiments.multigpu",
    "experiments.texture_repro",
    "experiments.sdf_repro",
    "scripts/run_texture_repro_4gpu.sh",
    "scripts/run_sdf_repro_4gpu.sh",
)
UNIT_PROPERTIES = (
    "LoadState",
    "ActiveState",
    "SubState",
    "MainPID",
    "ControlGroup",
    "ExecMainCode",
    "ExecMainStatus",
    "Result",
    "StateChangeTimestamp",
)


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    ppid: int
    pgid: int
    sid: int
    cwd: str
    command: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(
    arguments: Sequence[str],
    *,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(arguments),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=check,
    )


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _unit_name(artifact: str, override: str | None = None) -> str:
    raw = override or f"peps-image-repro-{artifact}.service"
    if not raw.endswith(".service"):
        raw = f"{raw}.service"
    if not UNIT_PATTERN.fullmatch(raw):
        raise ValueError(f"invalid systemd unit name: {raw}")
    return raw


def _resolved_output_root(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _absolute_without_symlink_resolution(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = ROOT / expanded
    return Path(os.path.abspath(expanded))


def _state_dir(output_root: Path) -> Path:
    return output_root / "work" / "image-repro" / "manager"


def _receipt_path(output_root: Path, artifact: str) -> Path:
    return _state_dir(output_root) / f"{artifact}.json"


def _service_log_path(output_root: Path, artifact: str) -> Path:
    return _state_dir(output_root) / f"{artifact}.service.log"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_receipt(output_root: Path, artifact: str) -> dict[str, Any] | None:
    path = _receipt_path(output_root, artifact)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _systemd_available() -> tuple[bool, str]:
    result = _run(("systemctl", "--user", "is-system-running"))
    state = result.stdout.strip()
    return result.returncode == 0 and state in {"running", "degraded"}, state


def _unit_properties(unit: str) -> dict[str, Any]:
    result = _run(
        (
            "systemctl",
            "--user",
            "show",
            unit,
            "--no-pager",
            f"--property={','.join(UNIT_PROPERTIES)}",
        )
    )
    payload: dict[str, Any] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        payload[name] = value
    payload.setdefault("LoadState", "not-found")
    for name in ("MainPID", "ExecMainCode", "ExecMainStatus"):
        try:
            payload[name] = int(payload.get(name, 0))
        except (TypeError, ValueError):
            payload[name] = 0
    return payload


def _read_processes(
    *,
    proc_root: Path = Path("/proc"),
    uid: int | None = None,
) -> list[ProcessRecord]:
    expected_uid = os.getuid() if uid is None else uid
    records: list[ProcessRecord] = []
    for process_root in proc_root.iterdir():
        if not process_root.name.isdigit():
            continue
        try:
            if process_root.stat().st_uid != expected_uid:
                continue
            cwd = (process_root / "cwd").resolve()
            if cwd != ROOT and ROOT not in cwd.parents:
                continue
            command = (
                (process_root / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
                .strip()
            )
            stat = (process_root / "stat").read_text(encoding="utf-8")
            closing_parenthesis = stat.rfind(")")
            fields = stat[closing_parenthesis + 1 :].split()
            records.append(
                ProcessRecord(
                    pid=int(process_root.name),
                    ppid=int(fields[1]),
                    pgid=int(fields[2]),
                    sid=int(fields[3]),
                    cwd=str(cwd),
                    command=command,
                )
            )
        except (IndexError, OSError, ValueError):
            continue
    return records


def _reproduction_processes(
    processes: Sequence[ProcessRecord] | None = None,
) -> list[ProcessRecord]:
    candidates = _read_processes() if processes is None else list(processes)
    return [
        process
        for process in candidates
        if any(marker in process.command for marker in REPRODUCTION_MARKERS)
    ]


def _process_payload(process: ProcessRecord) -> dict[str, Any]:
    return {
        "pid": process.pid,
        "ppid": process.ppid,
        "pgid": process.pgid,
        "sid": process.sid,
        "cwd": process.cwd,
        "command": process.command,
    }


def _artifact_progress(artifact: str, output_root: Path) -> dict[str, Any]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from experiments.image_repro import artifact_progress

    return artifact_progress(artifact, output_root=output_root)


def _progress_summary(progress: dict[str, Any]) -> dict[str, Any]:
    workers = []
    for worker in progress.get("workers", ()):
        workers.append(
            {
                "rank": worker.get("rank"),
                "pid": worker.get("pid"),
                "process_alive": worker.get("process_alive"),
                "state": worker.get("state"),
                "recorded_state": worker.get("recorded_state"),
                "liveness_evidence": worker.get("liveness_evidence"),
            }
        )
    return {
        "accounted_optimizer_steps": progress["accounted_optimizer_steps"],
        "expected_optimizer_steps": progress["expected_optimizer_steps"],
        "optimizer_step_completion_fraction": progress[
            "optimizer_step_completion_fraction"
        ],
        "completed_jobs": progress["completed_jobs"],
        "expected_jobs": progress["expected_jobs"],
        "checkpointed_incomplete_jobs": progress[
            "checkpointed_incomplete_jobs"
        ],
        "active_workers": progress["active_workers"],
        "per_rank": progress["per_rank"],
        "workers": workers,
        "checkpoint_errors": progress["checkpoint_errors"],
        "checkpoint_warnings": progress["checkpoint_warnings"],
        "result_errors": progress["result_errors"],
        "worker_status_errors": progress["worker_status_errors"],
        "unexpected_result_files": progress["unexpected_result_files"],
        "unexpected_checkpoint_files": progress[
            "unexpected_checkpoint_files"
        ],
        "incomplete_temporary_outputs": progress[
            "incomplete_temporary_outputs"
        ],
        "output_integrity_ok": progress["output_integrity_ok"],
    }


def _worker_gpu(pid: int) -> str | None:
    try:
        values = (Path("/proc") / str(pid) / "environ").read_bytes().split(
            b"\0"
        )
    except OSError:
        return None
    for value in values:
        if value.startswith(b"HIP_VISIBLE_DEVICES="):
            return value.split(b"=", 1)[1].decode(errors="replace")
    return None


def _build_systemd_run(
    *,
    unit: str,
    artifact: str,
    output_root: Path,
    python_bin: Path,
    service_log: Path,
    authorization_receipt: Path | None,
) -> list[str]:
    arguments = [
        "systemd-run",
        "--user",
        f"--unit={unit}",
        "--collect",
        "--service-type=exec",
        f"--description=PEPS image reproduction ({artifact})",
        f"--working-directory={ROOT}",
        f"--setenv=PEPS_PYTHON={python_bin}",
        f"--setenv=PEPS_OUTPUT_ROOT={output_root}",
        "--property=KillMode=control-group",
        "--property=TimeoutStopSec=45s",
        "--property=SendSIGKILL=yes",
        "--property=Restart=no",
        "--property=UMask=0002",
        f"--property=StandardOutput=append:{service_log}",
        f"--property=StandardError=append:{service_log}",
        "/usr/bin/bash",
        str(RUNNER),
        artifact,
    ]
    if authorization_receipt is not None:
        arguments.extend(
            ("--authorization-receipt", str(authorization_receipt))
        )
    return arguments


def _status_payload(
    *,
    artifact: str,
    output_root: Path,
    unit: str,
) -> dict[str, Any]:
    service = _unit_properties(unit)
    progress = _artifact_progress(artifact, output_root)
    receipt = _read_receipt(output_root, artifact)
    expected_pids = {
        int(worker["pid"])
        for worker in progress.get("workers", ())
        if isinstance(worker.get("pid"), int)
        and bool(worker.get("process_alive"))
    }
    main_pid = int(service.get("MainPID", 0))
    competitors = [
        process
        for process in _reproduction_processes()
        if process.pid not in expected_pids and process.pid != main_pid
    ]
    return {
        "schema": "peps.image_repro_manager_status",
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "backend": "systemd-user",
        "artifact": artifact,
        "unit": unit,
        "output_root": str(output_root),
        "receipt": receipt,
        "service": service,
        "progress": _progress_summary(progress),
        "competitors": [
            _process_payload(process) for process in competitors
        ],
        "logs": {
            "service": str(_service_log_path(output_root, artifact)),
            "ranks": [
                str(
                    output_root
                    / "work"
                    / "image-repro"
                    / "launch-logs"
                    / artifact
                    / f"rank-{rank}.log"
                )
                for rank in range(4)
            ],
        },
    }


def _health_payload(status: dict[str, Any]) -> dict[str, Any]:
    progress = status["progress"]
    service = status["service"]
    workers = {
        worker.get("rank"): worker
        for worker in progress["workers"]
        if isinstance(worker.get("rank"), int)
    }
    rank_checks: dict[str, dict[str, Any]] = {}
    for rank in range(4):
        worker = workers.get(rank, {})
        pid = worker.get("pid")
        gpu = _worker_gpu(pid) if isinstance(pid, int) else None
        alive = bool(worker.get("process_alive"))
        rank_checks[str(rank)] = {
            "pid": pid,
            "process_alive": alive,
            "expected_gpu": str(rank),
            "hip_visible_devices": gpu,
            "ok": alive and gpu == str(rank),
        }
    error_fields = (
        "checkpoint_errors",
        "result_errors",
        "worker_status_errors",
        "unexpected_result_files",
        "unexpected_checkpoint_files",
        "incomplete_temporary_outputs",
    )
    no_errors = not any(progress[name] for name in error_fields)
    healthy = all(
        (
            service.get("ActiveState") == "active",
            service.get("SubState") == "running",
            progress["active_workers"] == 4,
            progress["output_integrity_ok"],
            no_errors,
            not status["competitors"],
            all(check["ok"] for check in rank_checks.values()),
        )
    )
    receipt = status.get("receipt") or {}
    start_steps = receipt.get("start_accounted_optimizer_steps")
    current_steps = progress["accounted_optimizer_steps"]
    return {
        **status,
        "schema": "peps.image_repro_manager_health",
        "healthy": healthy,
        "rank_checks": rank_checks,
        "start_accounted_optimizer_steps": start_steps,
        "steps_since_managed_start": (
            current_steps - start_steps
            if isinstance(start_steps, int)
            else None
        ),
    }


def start(
    *,
    artifact: str,
    output_root: Path,
    unit: str,
    python_bin: Path,
    authorization_receipt: Path | None,
) -> dict[str, Any]:
    if artifact == "table1" and not TABLE1_RECOVERY_ENABLED:
        raise RuntimeError(
            "Table 1 launch and recovery are disabled after the external "
            "recovery incident; the systemd manager cannot start this artifact"
        )
    available, systemd_state = _systemd_available()
    if not available:
        raise RuntimeError(
            f"user systemd manager is not running: {systemd_state!r}"
        )
    existing = _unit_properties(unit)
    if existing.get("ActiveState") in {"active", "activating", "reloading"}:
        payload = _status_payload(
            artifact=artifact,
            output_root=output_root,
            unit=unit,
        )
        payload["already_running"] = True
        return payload
    competitors = _reproduction_processes()
    if competitors:
        details = "; ".join(
            f"{process.pid}: {process.command}" for process in competitors
        )
        raise RuntimeError(
            "refusing to start while repo-local reproduction processes exist: "
            f"{details}"
        )
    if artifact == "table1":
        if authorization_receipt is None:
            raise RuntimeError(
                "Table 1 requires --authorization-receipt"
            )
        authorization = _run(
            (
                str(python_bin),
                "-m",
                "experiments.image_repro",
                "authorization-check",
                "--artifact",
                "table1",
                "--authorization-receipt",
                str(authorization_receipt),
            )
        )
        if authorization.returncode != 0:
            raise RuntimeError(
                "Table 1 authorization failed: "
                f"{authorization.stderr.strip() or authorization.stdout.strip()}"
            )

    output_root.mkdir(parents=True, exist_ok=True)
    service_log = _service_log_path(output_root, artifact)
    service_log.parent.mkdir(parents=True, exist_ok=True)
    service_log.touch(exist_ok=True)
    progress = _artifact_progress(artifact, output_root)
    arguments = _build_systemd_run(
        unit=unit,
        artifact=artifact,
        output_root=output_root,
        python_bin=python_bin,
        service_log=service_log,
        authorization_receipt=authorization_receipt,
    )
    completed = _run(arguments)
    if completed.returncode != 0:
        raise RuntimeError(
            "systemd-run failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )

    service: dict[str, Any] = {}
    for _ in range(50):
        service = _unit_properties(unit)
        if service.get("ActiveState") == "active" and int(
            service.get("MainPID", 0)
        ):
            break
        if service.get("ActiveState") == "failed":
            break
        time.sleep(0.1)

    receipt = {
        "schema": "peps.image_repro_manager_receipt",
        "schema_version": 1,
        "backend": "systemd-user",
        "artifact": artifact,
        "unit": unit,
        "root": str(ROOT),
        "output_root": str(output_root),
        "python": str(python_bin),
        "runner": str(RUNNER),
        "authorization_receipt": (
            None
            if authorization_receipt is None
            else str(authorization_receipt)
        ),
        "service_log": str(service_log),
        "started_at_utc": _utc_now(),
        "start_accounted_optimizer_steps": progress[
            "accounted_optimizer_steps"
        ],
        "systemd_run": arguments,
        "service": service,
    }
    _atomic_write_json(_receipt_path(output_root, artifact), receipt)
    payload: dict[str, Any] = {}
    for _ in range(50):
        payload = _status_payload(
            artifact=artifact,
            output_root=output_root,
            unit=unit,
        )
        if payload["progress"]["active_workers"] == 4:
            break
        if payload["service"].get("ActiveState") != "active":
            break
        time.sleep(0.1)
    payload["already_running"] = False
    return payload


def stop(
    *,
    artifact: str,
    output_root: Path,
    unit: str,
) -> dict[str, Any]:
    before = _unit_properties(unit)
    if before.get("LoadState") != "not-found":
        completed = _run(("systemctl", "--user", "stop", unit))
        if completed.returncode != 0:
            raise RuntimeError(
                "systemctl stop failed: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        for _ in range(100):
            current = _unit_properties(unit)
            if current.get("ActiveState") not in {
                "active",
                "activating",
                "deactivating",
            }:
                break
            time.sleep(0.1)
    payload = _status_payload(
        artifact=artifact,
        output_root=output_root,
        unit=unit,
    )
    payload["stopped"] = True
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("start", "stop", "status", "health", "logs"),
    )
    parser.add_argument("artifact", nargs="?", default="table1")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get("PEPS_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))
        ),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(
            os.environ.get("PEPS_PYTHON", str(ROOT / ".venv/bin/python"))
        ),
    )
    parser.add_argument(
        "--unit",
        default=os.environ.get("PEPS_IMAGE_MANAGER_UNIT"),
    )
    parser.add_argument(
        "--authorization-receipt",
        type=Path,
        default=(
            Path(os.environ["PEPS_IMAGE_AUTHORIZATION_RECEIPT"])
            if "PEPS_IMAGE_AUTHORIZATION_RECEIPT" in os.environ
            else None
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.artifact not in ALLOWED_ARTIFACTS:
        raise SystemExit(f"unknown image artifact: {arguments.artifact}")
    artifact = arguments.artifact
    output_root = _resolved_output_root(arguments.output_root)
    python_bin = _absolute_without_symlink_resolution(arguments.python)
    unit = _unit_name(artifact, arguments.unit)
    authorization_receipt = (
        None
        if arguments.authorization_receipt is None
        else arguments.authorization_receipt.expanduser().resolve()
    )

    try:
        if arguments.action == "start":
            payload = start(
                artifact=artifact,
                output_root=output_root,
                unit=unit,
                python_bin=python_bin,
                authorization_receipt=authorization_receipt,
            )
        elif arguments.action == "stop":
            payload = stop(
                artifact=artifact,
                output_root=output_root,
                unit=unit,
            )
        else:
            payload = _status_payload(
                artifact=artifact,
                output_root=output_root,
                unit=unit,
            )
            if arguments.action == "health":
                payload = _health_payload(payload)
            elif arguments.action == "logs":
                payload = {
                    "artifact": artifact,
                    "unit": unit,
                    "logs": payload["logs"],
                }
    except (OSError, RuntimeError, ValueError) as exc:
        _print(
            {
                "schema": "peps.image_repro_manager_error",
                "schema_version": 1,
                "action": arguments.action,
                "artifact": artifact,
                "unit": unit,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return 1

    _print(payload)
    if arguments.action == "health" and not payload["healthy"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
