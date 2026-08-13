"""Static contracts for the bounded course release evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from peps.report import INSTANCE_COLUMNS, validate_run_manifest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RECEIPT_PATH = RESULTS / "course_release" / "receipt.json"

COURSE_SMOKES = {
    "course-fast-image-smoke": (
        "20260721T175053994434Z-smoke-image-course_fast-s0-3cfadfa6",
        "smoke-image",
        9,
    ),
    "course-fast-texture-smoke": (
        "20260721T175141764666Z-smoke-texture-course_fast-s0-737949da",
        "smoke-texture",
        15,
    ),
    "course-fast-sdf-smoke": (
        "20260721T175154356740Z-smoke-sdf-course_fast-s0-d66561b7",
        "smoke-sdf",
        5,
    ),
}

EXPECTED_EVIDENCE_STATUSES = {
    **{
        evidence_id: "validated-course-smoke-not-paper-comparable"
        for evidence_id in COURSE_SMOKES
    },
    "kodak-convergence-pilot": (
        "validated-inconclusive-pilot-not-paper-comparable"
    ),
    "texture-convergence-pilot": (
        "validated-inconclusive-pilot-not-paper-comparable"
    ),
    "sdf-public-512-provenance": (
        "validated-input-provenance-not-numeric-result"
    ),
    "texture-table2-complete-run": (
        "validated-complete-run-not-paper-matching"
    ),
    "texture-table2-shortfall-diagnosis": (
        "validated-sufficient-cause-not-established-cause"
    ),
    "sdf-table3-mape-public-subset": (
        "validated-public-subset-not-global"
    ),
    "sdf-table6-l1-public-subset": (
        "validated-public-subset-not-global"
    ),
}

REQUIRED_PAPER_BLOCKERS = {
    "fig5_dataset_not_reported",
    "fig5_training_budget_not_reported",
    "image_training_steps_not_reported",
    "table1_loss_recipe_conflict",
    "optimizer_and_seed_not_reported",
    "unreleased_sdf_converter",
    "pitted_stonefish_authorization_required",
    "table1_incomplete",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_course_release_receipt_matches_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load(RESULTS / "schemas" / "course_release_receipt.schema.json")
    jsonschema.validate(_load(RECEIPT_PATH), schema)


def test_all_checked_in_result_schemas_are_valid_json_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_paths = sorted((RESULTS / "schemas").glob("*.json"))
    assert schema_paths
    for path in schema_paths:
        schema = _load(path)
        jsonschema.validators.validator_for(schema).check_schema(schema)


def test_release_index_promotes_only_bounded_non_paper_evidence() -> None:
    manifest = _load(RESULTS / "manifest.json")
    receipt = _load(RECEIPT_PATH)
    indexed = manifest["release_evidence"]
    promoted = {item["id"]: item for item in receipt["promoted_evidence"]}

    assert set(indexed) == set(promoted) == set(EXPECTED_EVIDENCE_STATUSES)
    assert receipt["paper_comparable_results"] == 0
    assert receipt["paper_exact"]["ready"] is False
    assert receipt["paper_exact"]["paper_comparable_results"] == 0
    for evidence_id, expected_status in EXPECTED_EVIDENCE_STATUSES.items():
        assert indexed[evidence_id]["status"] == expected_status
        assert promoted[evidence_id]["status"] == expected_status
        assert indexed[evidence_id]["paper_comparable"] is False
        assert promoted[evidence_id]["paper_comparable"] is False


def test_every_top_level_csv_remains_legacy_unverified() -> None:
    manifest = _load(RESULTS / "manifest.json")
    csv_files = sorted(path.name for path in RESULTS.glob("*.csv"))
    manifested = sorted(
        name
        for name in manifest["artifacts"]
        if Path(name).suffix.lower() == ".csv"
    )
    assert manifested == csv_files
    assert csv_files
    assert all(
        manifest["artifacts"][name]["status"] == "legacy-unverified"
        for name in csv_files
    )

    receipt = _load(RECEIPT_PATH)["legacy_top_level_csvs"]
    assert receipt["status"] == "legacy-unverified"
    assert receipt["files"] == csv_files
    assert receipt["count"] == len(csv_files)


@pytest.mark.parametrize(
    ("evidence_id", "run_id", "artifact", "expected_rows"),
    [
        (evidence_id, *values)
        for evidence_id, values in COURSE_SMOKES.items()
    ],
)
def test_course_smoke_bundle_has_manifest_raw_rows_and_summary(
    evidence_id: str,
    run_id: str,
    artifact: str,
    expected_rows: int,
) -> None:
    del evidence_id
    run_dir = RESULTS / "runs" / run_id
    manifest = _load(run_dir / "manifest.json")
    validate_run_manifest(manifest)
    assert manifest["run_id"] == run_id
    assert manifest["profile"] == "course_fast"
    assert manifest["metadata"]["verification_status"] == (
        "course_fast_smoke_not_paper_comparable"
    )
    assert manifest["instances"]["row_count"] == expected_rows

    with (run_dir / "instances.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(INSTANCE_COLUMNS)
        rows = list(reader)
    assert len(rows) == expected_rows
    assert all(row["run_id"] == run_id for row in rows)
    assert all(row["profile"] == "course_fast" for row in rows)
    assert all(row["status"] == "ok" for row in rows)
    assert all(math.isfinite(float(row["value"])) for row in rows)

    summary = _load(run_dir / "summary.json")
    assert summary["schema"] == "peps.paper_artifact_summary"
    assert summary["schema_version"] == 1
    assert summary["run_id"] == run_id
    assert summary["artifact"] == artifact
    assert summary["rows"]
    assert all(row["run_id"] == run_id for row in summary["rows"])
    assert all(row["profile"] == "course_fast" for row in summary["rows"])


def test_pilots_remain_complete_inconclusive_and_unauthorized() -> None:
    image = _load(RESULTS / "image_convergence" / "receipt.json")
    assert image["integrity"]["valid"] is True
    assert image["integrity"]["active_workers"] == 0
    assert image["coverage"]["complete"] is True
    assert image["analysis"]["outcome"] == "inconclusive"
    assert image["analysis"]["recommended_budget_steps"] is None
    assert image["paper_exact"] is False
    assert image["verified_table1"] is False

    incident = _load(
        RESULTS
        / "image_convergence"
        / "external_table1_recovery_incident.json"
    )
    assert incident["schema_version"] == 2
    assert incident["status"] == "resolved_launcher_disabled"
    assert incident["root_cause"]["mechanism_type"] == (
        "cursor_recovery_policy_via_wake_chain_then_user_systemd_service"
    )
    assert incident["root_cause"]["bounded_pilot_involved"] is False
    assert all(
        item["final_state"] == "exited_0"
        for item in incident["wake_sleepers"]
    )
    assert incident["final_process_group"]["termination"]["survivors"] == []
    assert incident["final_process_group"]["termination"][
        "unrelated_processes_signalled"
    ] == 0
    assert incident["table1_status_after_all_termination"]["active_workers"] == 0
    assert incident["table1_status_after_all_termination"][
        "output_integrity_ok"
    ] is True
    assert incident["zero_process_observation"]["sample_count"] == 25
    assert incident["zero_process_observation"]["all_samples_zero"] is True
    assert incident["zero_process_observation"][
        "systemd_mask_present_in_every_sample"
    ] is True
    guard = incident["launcher_disablement"]["repository_guard"]
    assert guard["bare_launch_allowed"] is False
    assert guard["bounded_pilot_receipt_allowed"] is False
    assert guard["any_authorization_receipt_currently_allowed"] is False

    texture = _load(RESULTS / "texture_repro" / "convergence_pilot.json")
    progress = _load(
        RESULTS / "texture_repro" / "convergence_pilot_progress.json"
    )
    recovery = _load(
        RESULTS / "texture_repro" / "convergence_pilot_recovery.json"
    )
    assert progress["complete"] is True
    assert progress["active_workers"] == 0
    assert progress["observations"] == progress["expected_observations"] == 180
    assert texture["decision"]["status"] == "inconclusive_bounded_pilot"
    assert texture["decision"]["recommended_table2_steps"] is None
    assert texture["decision"]["full_71m_step_run_authorized"] is False
    assert recovery["process_reconciliation"]["active_related_processes"] == 0
    assert recovery["safety"]["full_table2_run_launched"] is False
    assert recovery["safety"]["recommendation_promoted"] is False


def test_public_sdf_records_are_exactly_three_and_stonefish_is_blocked() -> None:
    validation = _load(RESULTS / "sdf_repro" / "volume_validation.json")
    receipt = _load(RECEIPT_PATH)
    assert validation["status"] == "passed"
    assert validation["checksums_verified"] is True
    assert validation["stonefish"] == {
        "asset_id": "pitted-stonefish",
        "checked": False,
        "status": "deferred_auth_required",
        "substitution_used": False,
    }

    expected_assets = {"lucy", "thai-statue", "armadillo"}
    validation_rows = {
        item["asset_id"]: item for item in validation["volumes"]
    }
    release_rows = {
        item["asset_id"]: item for item in receipt["sdf_public_provenance"]
    }
    assert set(validation_rows) == set(release_rows) == expected_assets
    for asset_id in expected_assets:
        validation_row = validation_rows[asset_id]
        release_row = release_rows[asset_id]
        provenance_path = ROOT / release_row["receipt"]
        assert _sha256(provenance_path) == validation_row[
            "tracked_provenance_sha256"
        ]
        assert release_row["volume_sha256"] == validation_row["volume_sha256"]
        assert release_row["shape"] == [512, 512, 512]
        assert release_row["known_limit"] == validation_row[
            "preprocessor_known_limit"
        ]


def test_paper_omissions_and_authorization_blocker_stay_visible() -> None:
    receipt = _load(RECEIPT_PATH)
    blocker_codes = {
        item["code"] for item in receipt["paper_exact"]["blockers"]
    }
    assert REQUIRED_PAPER_BLOCKERS.issubset(blocker_codes)
    assert receipt["process_audit"][
        "active_recovery_or_full_training_processes"
    ] == 0

    prerequisites = _load(RESULTS / "paper_exact_prerequisites.json")
    by_artifact = {
        item["artifact"]: item for item in prerequisites["artifacts"]
    }
    for artifact in ("sdf-table3-mape", "sdf-table3-l1"):
        report = by_artifact[artifact]
        assert report["checks"]["verified_volumes"] == [
            "lucy",
            "thai-statue",
            "armadillo",
        ]
        assert {
            blocker["code"] for blocker in report["blockers"]
        } == {"pitted_stonefish_authorization_required"}
    assert {
        blocker["code"]
        for blocker in by_artifact["sdf-table4"]["blockers"]
    } == {"pitted_stonefish_authorization_required"}
