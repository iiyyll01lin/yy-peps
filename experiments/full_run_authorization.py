"""Explicit, boot-scoped authorization for the full image Table 1 matrix."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
IMAGE_TABLE1_CONFIG = ROOT / "configs/paper/image_full.toml"
IMAGE_TABLE1_ARTIFACT = "image-table1"
IMAGE_TABLE1_EXPECTED_JOBS = 648
IMAGE_TABLE1_EXPECTED_OPTIMIZER_STEPS = 77_760_000
MAX_AUTHORIZATION_WINDOW_SECONDS = 3600
IMAGE_TABLE1_RECOVERY_ENABLED = False


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_utc(value: object, field: str) -> datetime:
    _check(isinstance(value, str) and bool(value), f"{field} must be a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid ISO-8601 timestamp") from exc
    _check(parsed.tzinfo is not None, f"{field} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image_table1_authorization(
    receipt_path: str | Path | None,
    *,
    config_path: Path = IMAGE_TABLE1_CONFIG,
    proc_root: Path = Path("/proc"),
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate an independent, short-lived approval before any full run starts."""

    if not IMAGE_TABLE1_RECOVERY_ENABLED:
        raise ValueError(
            "full image Table 1 launch and recovery are disabled after the "
            "external-recovery incident; no authorization receipt is accepted "
            "until a separate full-reproduction gate deliberately changes the "
            "code-level interlock"
        )
    if receipt_path is None:
        raise ValueError(
            "full image Table 1 launch is disabled without an explicit "
            "--authorization-receipt"
        )
    path = Path(receipt_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Table 1 authorization receipt: {exc}") from exc
    _check(isinstance(payload, Mapping), "authorization receipt must be an object")

    expected_fields = {
        "schema",
        "schema_version",
        "artifact",
        "authorization_scope",
        "authorized",
        "authorization_basis",
        "bounded_pilot_authorizes_full_run",
        "config_sha256",
        "expected_jobs",
        "expected_optimizer_steps",
        "boot_id",
        "issued_at_utc",
        "expires_at_utc",
        "approved_by",
        "reason",
        "approval_id",
    }
    _check(
        set(payload) == expected_fields,
        "authorization receipt fields do not match the frozen contract",
    )
    _check(payload["schema"] == "peps.full_run_authorization", "bad authorization schema")
    _check(payload["schema_version"] == 1, "bad authorization schema version")
    _check(payload["artifact"] == IMAGE_TABLE1_ARTIFACT, "authorization is for another artifact")
    _check(
        payload["authorization_scope"] == "launch-or-resume-full-matrix",
        "authorization scope does not permit a full launch",
    )
    _check(payload["authorized"] is True, "full Table 1 run is not authorized")
    _check(
        payload["authorization_basis"]
        == "independent-explicit-approval-after-full-reproduction-gate",
        "bounded evidence cannot serve as full-run authorization",
    )
    _check(
        payload["bounded_pilot_authorizes_full_run"] is False,
        "authorization must acknowledge that the bounded pilot is non-authorizing",
    )
    _check(
        payload["config_sha256"] == _sha256(config_path),
        "authorization config digest does not match the current Table 1 config",
    )
    _check(
        payload["expected_jobs"] == IMAGE_TABLE1_EXPECTED_JOBS,
        "authorization job count does not match the full Table 1 matrix",
    )
    _check(
        payload["expected_optimizer_steps"]
        == IMAGE_TABLE1_EXPECTED_OPTIMIZER_STEPS,
        "authorization optimizer-step count does not match Table 1",
    )

    try:
        boot_id = (proc_root / "sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except OSError as exc:
        raise ValueError(f"cannot read current boot identity: {exc}") from exc
    _check(bool(boot_id), "current boot identity is empty")
    _check(payload["boot_id"] == boot_id, "authorization belongs to another boot")

    issued = _parse_utc(payload["issued_at_utc"], "issued_at_utc")
    expires = _parse_utc(payload["expires_at_utc"], "expires_at_utc")
    observed_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    _check(issued <= observed_now, "authorization is not yet valid")
    _check(expires > observed_now, "authorization has expired")
    _check(expires > issued, "authorization expiry must follow issuance")
    _check(
        (expires - issued).total_seconds() <= MAX_AUTHORIZATION_WINDOW_SECONDS,
        "authorization window exceeds one hour",
    )
    for field in ("approved_by", "reason", "approval_id"):
        _check(
            isinstance(payload[field], str) and bool(payload[field].strip()),
            f"{field} must be a non-empty string",
        )
    return dict(payload)
