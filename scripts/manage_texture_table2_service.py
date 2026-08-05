#!/usr/bin/env python3
"""Run Texture Table 2 on GPU 0/1 under one persistent systemd service."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
OUTPUT_ROOT = ROOT / "results"
TEXTURE_RESULTS = OUTPUT_ROOT / "texture_repro"
SERVICE_ROOT = OUTPUT_ROOT / "work/texture-repro/service"
LOG_ROOT = SERVICE_ROOT / "logs"
STATE_PATH = TEXTURE_RESULTS / "table2_service_state.json"
PREFLIGHT_PATH = TEXTURE_RESULTS / "table2_service_preflight.json"
AUTHORIZATION_PATH = TEXTURE_RESULTS / "table2_launch_authorization.json"
LOCK_PATH = SERVICE_ROOT / "table2.lock"
STOP_REQUEST_PATH = SERVICE_ROOT / "stop-request.json"
RECEIPT_PATH = TEXTURE_RESULTS / "dataset_verification.json"
IMAGE_TABLE1_UNIT = (
    Path.home() / ".config/systemd/user/peps-image-repro-table1.service"
)
PHYSICAL_GPUS = (0, 1)
WORLD_SIZE = 2
EXPECTED_JOBS = 594
EXPECTED_OPTIMIZER_STEPS = 71_280_000
EXPECTED_MAPS = 78
EXPECTED_SETS = 18
EXPECTED_SMOKE_JOBS = 3
MINIMUM_FREE_DISK_BYTES = 20 * 1024**3
MINIMUM_VRAM_BYTES = 8 * 1024**3
HARD_INTERLOCK_EXIT = 78
SERVICE_UNIT = "peps-texture-table2.service"
PIN_ENV = {
    "code_digest": "PEPS_TEXTURE_EXPECTED_CODE_DIGEST",
    "config_sha256": "PEPS_TEXTURE_EXPECTED_CONFIG_SHA256",
    "manifest_sha256": "PEPS_TEXTURE_EXPECTED_MANIFEST_SHA256",
    "manager_sha256": "PEPS_TEXTURE_EXPECTED_MANAGER_SHA256",
    "receipt_sha256": "PEPS_TEXTURE_EXPECTED_RECEIPT_SHA256",
}


class HardInterlock(RuntimeError):
    """A launch condition that must not be retried or bypassed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HardInterlock(f"cannot read required receipt {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HardInterlock(f"required receipt is not an object: {path}")
    return payload


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise HardInterlock(message)


def _texture_runner_processes() -> list[dict[str, object]]:
    found = []
    for process_root in Path("/proc").iterdir():
        if not process_root.name.isdigit():
            continue
        try:
            if process_root.stat().st_uid != os.getuid():
                continue
            command = (
                (process_root / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
                .strip()
            )
            cwd = (process_root / "cwd").resolve()
            stat = (process_root / "stat").read_text(encoding="utf-8")
        except OSError:
            continue
        if cwd != ROOT and ROOT not in cwd.parents:
            continue
        if (
            "experiments.texture_repro run" not in command
            or "--artifact table2" not in command
        ):
            continue
        fields = stat[stat.rfind(")") + 1 :].split()
        found.append(
            {
                "pid": int(process_root.name),
                "ppid": int(fields[1]),
                "pgid": int(fields[2]),
                "command": command,
            }
        )
    return found


def _gpu_snapshot() -> dict[str, object]:
    completed = subprocess.run(
        ("rocm-smi", "--showuse", "--showmemuse", "--json"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise HardInterlock(
            f"cannot inspect GPU occupancy: {completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise HardInterlock("rocm-smi returned invalid JSON") from exc
    for physical in PHYSICAL_GPUS:
        card = payload.get(f"card{physical}")
        _check(isinstance(card, Mapping), f"GPU {physical} is not visible")
        try:
            use = int(str(card["GPU use (%)"]))
            # ROCm 7.2.3 renamed this field; both spellings mean VRAM percent.
            memory_field = card.get("GPU memory use (%)")
            if memory_field is None:
                memory_field = card["GPU Memory Allocated (VRAM%)"]
            memory = int(str(memory_field))
        except (KeyError, TypeError, ValueError) as exc:
            raise HardInterlock(
                f"GPU {physical} occupancy fields are malformed"
            ) from exc
        _check(
            use == 0 and memory == 0,
            f"GPU {physical} is not idle (use={use}%, memory={memory}%)",
        )
    return payload


def _visible_gpu_probe(physical: int) -> dict[str, object]:
    script = (
        "import json, torch; "
        "p=torch.cuda.get_device_properties(0) if torch.cuda.is_available() "
        "and torch.cuda.device_count()==1 else None; "
        "print(json.dumps({'available':torch.cuda.is_available(),"
        "'count':torch.cuda.device_count(),"
        "'name':None if p is None else str(p.name),"
        "'architecture':None if p is None else getattr(p,'gcnArchName',None),"
        "'total_memory_bytes':None if p is None else int(p.total_memory)}))"
    )
    environment = os.environ.copy()
    for name in (
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        environment.pop(name, None)
    environment["ROCR_VISIBLE_DEVICES"] = str(physical)
    completed = subprocess.run(
        (str(PYTHON), "-c", script),
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise HardInterlock(
            f"GPU {physical} visibility probe failed: {completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise HardInterlock(
            f"GPU {physical} visibility probe returned invalid JSON"
        ) from exc
    _check(payload.get("available") is True, f"GPU {physical} is unavailable")
    _check(payload.get("count") == 1, f"GPU {physical} pin exposes != 1 device")
    _check(
        str(payload.get("architecture", "")).split(":", 1)[0] == "gfx1201",
        f"GPU {physical} is not gfx1201",
    )
    _check(
        int(payload.get("total_memory_bytes") or 0) >= MINIMUM_VRAM_BYTES,
        f"GPU {physical} has less than 8 GiB VRAM",
    )
    return payload


def _pin_payload(require_service_pins: bool) -> dict[str, str]:
    from experiments.texture_repro import (
        ARTIFACT_CONFIGS,
        _code_digest,
    )
    from data.manifest import hash_file

    current = {
        "code_digest": _code_digest(),
        "config_sha256": hash_file(ARTIFACT_CONFIGS["table2"], "sha256"),
        "manifest_sha256": hash_file(
            ROOT / "data/manifests/textures.json", "sha256"
        ),
        "manager_sha256": _sha256(Path(__file__).resolve()),
        "receipt_sha256": _sha256(RECEIPT_PATH),
    }
    if require_service_pins:
        for key, environment_name in PIN_ENV.items():
            expected = os.environ.get(environment_name)
            _check(bool(expected), f"systemd pin {environment_name} is missing")
            _check(
                expected == current[key],
                f"systemd pin drift for {key}: expected {expected}, "
                f"found {current[key]}",
            )
    return current


def preflight(*, require_service_pins: bool = False) -> dict[str, object]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from data.manifest import load_manifest
    from experiments.config import load_experiment_config
    from experiments.texture_repro import (
        ARTIFACT_CONFIGS,
        PAPER_TABLE2,
        _artifact_output,
        architecture_receipt,
        artifact_progress,
        job_plan,
        verification_receipt_is_current,
    )

    _check(PYTHON.is_file() and os.access(PYTHON, os.X_OK), "venv Python missing")
    _check(
        IMAGE_TABLE1_UNIT.is_symlink()
        and os.readlink(IMAGE_TABLE1_UNIT) == "/dev/null",
        "Table 1 user service is not masked",
    )
    _check(
        not _texture_runner_processes(),
        "an unmanaged Texture Table 2 process is already running",
    )
    _check(
        verification_receipt_is_current(RECEIPT_PATH),
        "18-set texture verification receipt is stale",
    )
    receipt = _load_json(RECEIPT_PATH)
    _check(receipt.get("set_count") == EXPECTED_SETS, "texture set count != 18")
    _check(receipt.get("map_count") == EXPECTED_MAPS, "texture map count != 78")
    _check(receipt.get("verified_files") == EXPECTED_MAPS, "78 maps not verified")
    _check(receipt.get("decoded_sets") == EXPECTED_SETS, "18 sets not decoded")

    manifest = load_manifest("textures")
    license_names = {
        key: str(value.get("name", ""))
        for key, value in manifest["licenses"].items()
    }
    _check(
        all("CC0" in value for value in license_names.values()),
        "texture sources are not all covered by CC0 receipts",
    )
    _check(
        all(item["license"] in license_names for item in manifest["sets"]),
        "a texture set has no recognized license receipt",
    )

    config = load_experiment_config(ARTIFACT_CONFIGS["table2"])
    _check(config.canonical and config.profile == "full", "Table 2 config drift")
    _check(tuple(config.seeds) == (0, 1, 2), "Table 2 seeds drift")
    _check(tuple(method.name for method in config.methods) == tuple(PAPER_TABLE2), "Table 2 method set drift")
    architecture = [
        architecture_receipt(method, output_channels=15)
        for method in config.methods
    ]
    _check(
        all(
            method.expected_encoder_params == item["encoder_params"]
            for method, item in zip(config.methods, architecture)
        ),
        "a Table 2 parameter budget does not match its builder",
    )

    plan = job_plan("table2", world_size=WORLD_SIZE)
    _check(plan["expected_jobs"] == EXPECTED_JOBS, "Table 2 jobs != 594")
    _check(
        plan["expected_optimizer_steps"] == EXPECTED_OPTIMIZER_STEPS,
        "Table 2 optimizer-step budget drift",
    )
    _check(
        plan["parallelism"]["world_size"] == WORLD_SIZE,
        "Table 2 world size != 2",
    )
    _check(
        [plan["per_rank"][str(rank)]["jobs"] for rank in range(WORLD_SIZE)]
        == [297, 297],
        "Table 2 two-rank sharding is not balanced",
    )

    smoke = _load_json(TEXTURE_RESULTS / "smoke_status.json")
    _check(smoke.get("complete") is True, "texture smoke is incomplete")
    _check(
        smoke.get("completed_jobs") == EXPECTED_SMOKE_JOBS
        and smoke.get("expected_jobs") == EXPECTED_SMOKE_JOBS,
        "texture smoke is not 3/3",
    )
    _check(
        not smoke.get("result_errors") and not smoke.get("checkpoint_errors"),
        "texture smoke contains errors",
    )
    _check(
        Path(str(smoke.get("output_dir", ""))).resolve()
        == _artifact_output(OUTPUT_ROOT, "smoke").resolve(),
        "texture smoke belongs to a stale code digest",
    )

    sweep = artifact_progress("sweep", output_root=OUTPUT_ROOT)
    _check(int(sweep["active_workers"]) == 0, "3F/4F sweep is active")
    table = artifact_progress("table2", output_root=OUTPUT_ROOT)
    _check(int(table["active_workers"]) == 0, "Table 2 workers already active")

    disk = os.statvfs(OUTPUT_ROOT)
    free_bytes = disk.f_bavail * disk.f_frsize
    _check(
        free_bytes >= MINIMUM_FREE_DISK_BYTES,
        "free disk is below the 20 GiB Table 2 interlock",
    )
    gpu_snapshot = _gpu_snapshot()
    gpu_probes = {
        str(physical): _visible_gpu_probe(physical)
        for physical in PHYSICAL_GPUS
    }
    pins = _pin_payload(require_service_pins)
    authorization_id = os.environ.get(
        "PEPS_TEXTURE_AUTHORIZATION_ID",
        "interactive-preflight-only",
    )
    if require_service_pins:
        _check(
            authorization_id.startswith("explicit-user-request-"),
            "explicit user authorization ID is missing",
        )
    return {
        "schema": "peps.texture_table2_service_preflight",
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "status": "passed",
        "service_unit": SERVICE_UNIT,
        "authorization": {
            "id": authorization_id,
            "scope": "Texture Table 2 full matrix only",
            "sweep_authorized": False,
        },
        "pins": pins,
        "dataset": {
            "sets": receipt["set_count"],
            "maps": receipt["map_count"],
            "verified_files": receipt["verified_files"],
            "decoded_sets": receipt["decoded_sets"],
            "licenses": license_names,
        },
        "matrix": {
            "jobs": plan["expected_jobs"],
            "optimizer_steps": plan["expected_optimizer_steps"],
            "world_size": WORLD_SIZE,
            "per_rank": plan["per_rank"],
            "checkpoint_storage_estimate": plan[
                "checkpoint_storage_estimate"
            ],
        },
        "smoke": {
            "complete": smoke["complete"],
            "completed_jobs": smoke["completed_jobs"],
            "expected_jobs": smoke["expected_jobs"],
            "output_dir": smoke["output_dir"],
        },
        "parameter_budgets": architecture,
        "gpu": {
            "physical_devices": list(PHYSICAL_GPUS),
            "snapshot": gpu_snapshot,
            "visibility_probes": gpu_probes,
        },
        "guards": {
            "table1_masked": True,
            "sweep_active_workers": sweep["active_workers"],
            "existing_table2_active_workers": table["active_workers"],
            "unmanaged_table2_processes": [],
        },
        "disk": {
            "free_bytes": free_bytes,
            "minimum_free_bytes": MINIMUM_FREE_DISK_BYTES,
        },
        "output_dir": str(_artifact_output(OUTPUT_ROOT, "table2")),
    }


def _worker_command(rank: int) -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "experiments.texture_repro",
        "run",
        "--artifact",
        "table2",
        "--rank",
        str(rank),
        "--world-size",
        str(WORLD_SIZE),
        "--device",
        "cuda:0",
        "--output-root",
        str(OUTPUT_ROOT),
        "--verification-receipt",
        str(RECEIPT_PATH),
        "--allow-protocol-assumptions",
    ]


def _worker_environment(physical: int) -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        environment.pop(name, None)
    environment["ROCR_VISIBLE_DEVICES"] = str(physical)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PEPS_TEXTURE_PHYSICAL_GPU"] = str(physical)
    return environment


def run_service() -> int:
    SERVICE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(
                lock_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise HardInterlock("Texture Table 2 manager lock is held") from exc

        checked = preflight(require_service_pins=True)
        _atomic_json(PREFLIGHT_PATH, checked)
        _atomic_json(
            STOP_REQUEST_PATH,
            {
                "schema": "peps.texture_table2_stop_request",
                "schema_version": 1,
                "requested": False,
                "updated_at_utc": _utc_now(),
            },
        )
        authorization = {
            "schema": "peps.texture_table2_launch_authorization",
            "schema_version": 1,
            "authorized": True,
            "authorization_id": checked["authorization"]["id"],
            "scope": "Texture Table 2 full matrix only",
            "issued_at_utc": _utc_now(),
            "world_size": WORLD_SIZE,
            "physical_gpus": list(PHYSICAL_GPUS),
            "expected_jobs": EXPECTED_JOBS,
            "expected_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
            "pins": checked["pins"],
            "block_other_texture_gpu_work": True,
            "table2_complete": False,
        }
        _atomic_json(AUTHORIZATION_PATH, authorization)
        from experiments.texture_repro import artifact_progress

        progress = artifact_progress("table2", output_root=OUTPUT_ROOT)
        if progress["complete"]:
            authorization.update(
                {
                    "table2_complete": True,
                    "completed_at_utc": _utc_now(),
                }
            )
            _atomic_json(AUTHORIZATION_PATH, authorization)
            _atomic_json(
                STATE_PATH,
                {
                    "schema": "peps.texture_table2_service_state",
                    "schema_version": 1,
                    "updated_at_utc": _utc_now(),
                    "state": "already_complete",
                    "service_pid": os.getpid(),
                    "preflight": str(PREFLIGHT_PATH),
                    "progress": progress,
                },
            )
            return 0

        children: list[tuple[int, int, subprocess.Popen[str], Any, Path]] = []
        stopping = False

        def stop_children(signum: int, _frame: object) -> None:
            nonlocal stopping
            stopping = True
            for _rank, _physical, child, _handle, _path in children:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

        signal.signal(signal.SIGTERM, stop_children)
        signal.signal(signal.SIGINT, stop_children)

        for rank, physical in enumerate(PHYSICAL_GPUS):
            log_path = LOG_ROOT / f"rank-{rank}-gpu-{physical}.log"
            log_handle = log_path.open("a", encoding="utf-8", buffering=1)
            command = _worker_command(rank)
            log_handle.write(
                json.dumps(
                    {
                        "event": "worker_launch",
                        "at_utc": _utc_now(),
                        "rank": rank,
                        "physical_gpu": physical,
                        "command": command,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                env=_worker_environment(physical),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            children.append((rank, physical, child, log_handle, log_path))

        def state_payload(state: str) -> dict[str, object]:
            return {
                "schema": "peps.texture_table2_service_state",
                "schema_version": 1,
                "updated_at_utc": _utc_now(),
                "state": state,
                "service_unit": SERVICE_UNIT,
                "service_pid": os.getpid(),
                "authorization": checked["authorization"],
                "output_dir": checked["output_dir"],
                "world_size": WORLD_SIZE,
                "sweep_started": False,
                "workers": [
                    {
                        "rank": rank,
                        "physical_gpu": physical,
                        "pid": child.pid,
                        "pgid": child.pid,
                        "returncode": child.poll(),
                        "alive": child.poll() is None,
                        "log": str(log_path),
                        "command": _worker_command(rank),
                    }
                    for rank, physical, child, _handle, log_path in children
                ],
                "preflight": str(PREFLIGHT_PATH),
            }

        _atomic_json(STATE_PATH, state_payload("running"))
        print(json.dumps(state_payload("running"), sort_keys=True), flush=True)
        failure: int | None = None
        pin_error: str | None = None
        requested_stop = False
        try:
            while True:
                try:
                    _pin_payload(require_service_pins=True)
                except HardInterlock as exc:
                    pin_error = str(exc)
                    stop_children(signal.SIGTERM, None)
                try:
                    stop_request = json.loads(
                        STOP_REQUEST_PATH.read_text(encoding="utf-8")
                    )
                    if stop_request.get("requested") is True:
                        requested_stop = True
                        stop_children(signal.SIGTERM, None)
                except (OSError, json.JSONDecodeError):
                    pass
                statuses = [child.poll() for _, _, child, _, _ in children]
                if all(status is not None for status in statuses):
                    failure = next(
                        (status for status in statuses if status != 0),
                        None,
                    )
                    break
                if any(
                    status is not None and status != 0 for status in statuses
                ):
                    failure = next(
                        status
                        for status in statuses
                        if status is not None and status != 0
                    )
                    stop_children(signal.SIGTERM, None)
                _atomic_json(
                    STATE_PATH,
                    state_payload("stopping" if stopping else "running"),
                )
                time.sleep(5)
        finally:
            deadline = time.monotonic() + 120
            for _rank, _physical, child, _handle, _path in children:
                if child.poll() is None:
                    try:
                        child.wait(timeout=max(0.0, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait()
            for _rank, _physical, _child, handle, _path in children:
                handle.close()

        if pin_error is not None:
            raise HardInterlock(
                f"code/config/data pin changed while workers ran: {pin_error}"
            )
        final_state = (
            "stopped_by_request"
            if requested_stop
            else (
                "interrupted"
                if stopping
                else ("failed" if failure is not None else "complete")
            )
        )
        _atomic_json(STATE_PATH, state_payload(final_state))
        if final_state == "complete":
            authorization.update(
                {
                    "table2_complete": True,
                    "completed_at_utc": _utc_now(),
                }
            )
            _atomic_json(AUTHORIZATION_PATH, authorization)
        if requested_stop:
            return 0
        return 1 if failure is not None or stopping else 0


def request_stop() -> int:
    SERVICE_ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ("systemctl", "--user", "disable", SERVICE_UNIT),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    _atomic_json(
        STOP_REQUEST_PATH,
        {
            "schema": "peps.texture_table2_stop_request",
            "schema_version": 1,
            "requested": True,
            "requested_at_utc": _utc_now(),
            "requested_by_pid": os.getpid(),
        },
    )
    print(
        json.dumps(
            {
                "service_unit": SERVICE_UNIT,
                "stop_request": str(STOP_REQUEST_PATH),
                "requested": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "run", "request-stop"),
        nargs="?",
        default="run",
    )
    parser.add_argument(
        "--require-service-pins",
        action="store_true",
        help="Validate systemd-pinned code/config/data digests.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "request-stop":
            return request_stop()
        if arguments.command == "preflight":
            payload = preflight(
                require_service_pins=arguments.require_service_pins
            )
            _atomic_json(PREFLIGHT_PATH, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        return run_service()
    except HardInterlock as exc:
        payload = {
            "schema": "peps.texture_table2_service_state",
            "schema_version": 1,
            "updated_at_utc": _utc_now(),
            "state": "hard_interlock",
            "service_unit": SERVICE_UNIT,
            "service_pid": os.getpid(),
            "error": str(exc),
            "sweep_started": False,
        }
        _atomic_json(STATE_PATH, payload)
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return HARD_INTERLOCK_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
