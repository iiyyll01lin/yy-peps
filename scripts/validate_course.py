#!/usr/bin/env python3
"""Static release/course validation with no dataset or GPU requirement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WEEK_IDS = tuple(f"W{week:02d}" for week in range(1, 15))
CHECKSUM_LENGTHS = {"md5": 32, "sha256": 64}
RELEASE_EVIDENCE_STATUSES = {
    "course-fast-image-smoke": "validated-course-smoke-not-paper-comparable",
    "course-fast-texture-smoke": "validated-course-smoke-not-paper-comparable",
    "course-fast-sdf-smoke": "validated-course-smoke-not-paper-comparable",
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
COURSE_SMOKE_RUNS = {
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


class Validator:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def load_json(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.errors.append(f"missing JSON file: {path.relative_to(ROOT)}")
        except json.JSONDecodeError as exc:
            self.errors.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
        return None


def _source(cell: dict[str, Any]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_file(
    validator: Validator,
    value: Any,
    context: str,
) -> Path | None:
    valid = (
        isinstance(value, str)
        and bool(value)
        and not Path(value).is_absolute()
        and ".." not in Path(value).parts
    )
    validator.check(valid, f"{context}: unsafe or missing path")
    if not valid:
        return None
    path = ROOT / value
    validator.check(path.is_file(), f"{context}: file does not exist: {value}")
    return path if path.is_file() else None


def validate_course_smoke(
    validator: Validator,
    evidence_id: str,
    run_id: str,
    artifact: str,
    expected_rows: int,
) -> None:
    run_dir = ROOT / "results" / "runs" / run_id
    manifest = validator.load_json(run_dir / "manifest.json")
    if not isinstance(manifest, dict):
        return
    validator.check(
        manifest.get("schema") == "peps.run_manifest"
        and manifest.get("schema_version") == 1,
        f"{evidence_id}: unsupported run manifest",
    )
    validator.check(
        manifest.get("run_id") == run_id,
        f"{evidence_id}: run_id mismatch",
    )
    validator.check(
        manifest.get("profile") == "course_fast",
        f"{evidence_id}: profile must be course_fast",
    )
    metadata = manifest.get("metadata")
    validator.check(
        isinstance(metadata, dict)
        and metadata.get("verification_status")
        == "course_fast_smoke_not_paper_comparable",
        f"{evidence_id}: missing non-paper-comparable verification status",
    )
    descriptor = manifest.get("instances")
    validator.check(
        isinstance(descriptor, dict)
        and descriptor.get("row_count") == expected_rows,
        f"{evidence_id}: manifest row count mismatch",
    )
    instances_path = run_dir / "instances.csv"
    if not instances_path.is_file():
        validator.errors.append(f"{evidence_id}: missing instances.csv")
        return
    try:
        with instances_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        validator.errors.append(f"{evidence_id}: invalid instances.csv: {exc}")
        return
    validator.check(
        len(rows) == expected_rows,
        f"{evidence_id}: instances.csv row count mismatch",
    )
    for index, row in enumerate(rows):
        prefix = f"{evidence_id}: row {index}"
        validator.check(row.get("run_id") == run_id, f"{prefix}: run_id mismatch")
        validator.check(
            row.get("profile") == "course_fast",
            f"{prefix}: profile mismatch",
        )
        validator.check(row.get("status") == "ok", f"{prefix}: status is not ok")
        try:
            finite = math.isfinite(float(row.get("value", "")))
        except (TypeError, ValueError):
            finite = False
        validator.check(finite, f"{prefix}: value is not finite")

    summary = validator.load_json(run_dir / "summary.json")
    validator.check(
        isinstance(summary, dict)
        and summary.get("schema") == "peps.paper_artifact_summary"
        and summary.get("schema_version") == 1
        and summary.get("run_id") == run_id
        and summary.get("artifact") == artifact
        and isinstance(summary.get("rows"), list)
        and bool(summary["rows"]),
        f"{evidence_id}: summary contract mismatch",
    )


def validate_notebook(validator: Validator, path: Path) -> None:
    notebook = validator.load_json(path)
    if not isinstance(notebook, dict):
        return
    relative = path.relative_to(ROOT)
    validator.check(notebook.get("nbformat") == 4, f"{relative}: nbformat must be 4")
    cells = notebook.get("cells")
    validator.check(isinstance(cells, list), f"{relative}: cells must be a list")
    if not isinstance(cells, list):
        return
    validator.check(bool(cells), f"{relative}: notebook has no cells")
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            validator.errors.append(f"{relative}: cell {index} is not an object")
            continue
        if cell.get("cell_type") != "code":
            continue
        source = _source(cell)
        try:
            compile(source, f"{relative}:cell-{index}", "exec")
        except SyntaxError as exc:
            validator.errors.append(
                f"{relative}: code cell {index} does not compile: "
                f"{exc.msg} (line {exc.lineno})"
            )


def validate_labs(validator: Validator) -> dict[str, dict[str, Any]]:
    path = ROOT / "course" / "labs.json"
    document = validator.load_json(path)
    if not isinstance(document, dict):
        return {}
    validator.check(
        document.get("schema_version") == 1,
        "course/labs.json: schema_version must be 1",
    )
    weeks = document.get("weeks")
    validator.check(isinstance(weeks, list), "course/labs.json: weeks must be a list")
    if not isinstance(weeks, list):
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    for index, week in enumerate(weeks):
        if not isinstance(week, dict):
            validator.errors.append(f"course/labs.json: week {index} is not an object")
            continue
        week_id = week.get("id")
        if not isinstance(week_id, str):
            validator.errors.append(f"course/labs.json: week {index} has no ID")
            continue
        validator.check(week_id not in by_id, f"duplicate lab ID: {week_id}")
        by_id[week_id] = week
        notebook = week.get("notebook")
        validator.check(
            isinstance(notebook, str) and (ROOT / notebook).is_file(),
            f"{week_id}: notebook path is missing",
        )
        readings = week.get("readings")
        validator.check(
            isinstance(readings, list) and bool(readings),
            f"{week_id}: at least one reading is required",
        )
        if isinstance(readings, list):
            for reading_index, reading in enumerate(readings):
                valid = (
                    isinstance(reading, dict)
                    and isinstance(reading.get("title"), str)
                    and bool(reading["title"].strip())
                    and isinstance(reading.get("url"), str)
                    and reading["url"].startswith("https://")
                )
                validator.check(
                    valid, f"{week_id}: reading {reading_index} is incomplete"
                )
        criteria = week.get("success_criteria")
        validator.check(
            isinstance(criteria, list) and len(criteria) >= 2,
            f"{week_id}: at least two success criteria are required",
        )
        if isinstance(criteria, list):
            criterion_ids: set[str] = set()
            for criterion in criteria:
                valid = (
                    isinstance(criterion, dict)
                    and isinstance(criterion.get("id"), str)
                    and bool(criterion["id"].strip())
                    and isinstance(criterion.get("description"), str)
                    and bool(criterion["description"].strip())
                    and isinstance(criterion.get("evidence"), str)
                    and bool(criterion["evidence"].strip())
                )
                validator.check(valid, f"{week_id}: incomplete success criterion")
                if valid:
                    validator.check(
                        criterion["id"] not in criterion_ids,
                        f"{week_id}: duplicate criterion {criterion['id']}",
                    )
                    criterion_ids.add(criterion["id"])
    validator.check(
        tuple(sorted(by_id)) == WEEK_IDS,
        "course/labs.json must define W01 through W14 exactly once",
    )
    return by_id


def _iter_file_specs(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if "local_path" in value:
            yield value
        for nested in value.values():
            yield from _iter_file_specs(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_file_specs(nested)


def validate_data_manifests(validator: Validator) -> None:
    manifest_dir = ROOT / "data" / "manifests"
    for name in ("kodak.json", "textures.json", "sdf.json"):
        path = manifest_dir / name
        document = validator.load_json(path)
        if not isinstance(document, dict):
            continue
        validator.check(
            document.get("schema_version") == 1,
            f"data/manifests/{name}: schema_version must be 1",
        )
        validator.check(
            isinstance(document.get("dataset_id"), str),
            f"data/manifests/{name}: dataset_id is required",
        )
        for spec in _iter_file_specs(document):
            local_path = spec.get("local_path")
            validator.check(
                isinstance(local_path, str)
                and bool(local_path)
                and not Path(local_path).is_absolute()
                and ".." not in Path(local_path).parts,
                f"data/manifests/{name}: unsafe or missing local_path",
            )
            checksum = spec.get("checksum")
            if checksum is None:
                continue
            algorithm = checksum.get("algorithm") if isinstance(checksum, dict) else None
            value = checksum.get("value") if isinstance(checksum, dict) else None
            length = CHECKSUM_LENGTHS.get(algorithm)
            valid = (
                length is not None
                and isinstance(value, str)
                and len(value) == length
                and re.fullmatch(r"[0-9a-fA-F]+", value) is not None
            )
            validator.check(
                valid, f"data/manifests/{name}: malformed checksum for {local_path}"
            )


def validate_results(validator: Validator) -> None:
    path = ROOT / "results" / "manifest.json"
    document = validator.load_json(path)
    if not isinstance(document, dict):
        return
    validator.check(
        document.get("schema_version") == 1,
        "results/manifest.json: schema_version must be 1",
    )
    artifacts = document.get("artifacts")
    validator.check(
        isinstance(artifacts, dict), "results/manifest.json: artifacts must be an object"
    )
    if not isinstance(artifacts, dict):
        return
    csv_files = {item.name for item in (ROOT / "results").glob("*.csv")}
    manifested_csvs = {
        name for name in artifacts if Path(name).suffix.lower() == ".csv"
    }
    validator.check(
        manifested_csvs == csv_files,
        "results/manifest.json must list every top-level results CSV exactly once",
    )
    # Schema files are contracts rather than results, and the manifest does not
    # index itself. Without this the CSV half was enforced and the JSON half was
    # not, which let ten receipts drift out of the index unnoticed.
    json_files = {
        item.name
        for item in (ROOT / "results").glob("*.json")
        if item.name != "manifest.json" and not item.name.endswith(".schema.json")
    }
    manifested_jsons = {
        name
        for name in artifacts
        if Path(name).suffix.lower() == ".json" and not name.endswith(".schema.json")
    }
    validator.check(
        manifested_jsons == json_files,
        "results/manifest.json must list every top-level results JSON exactly once",
    )
    for name, metadata in artifacts.items():
        validator.check(
            Path(name).name == name and (ROOT / "results" / name).is_file(),
            f"results/manifest.json: {name} is not a top-level result file",
        )
        status = metadata.get("status") if isinstance(metadata, dict) else None
        validator.check(
            status
            in {
                "legacy-unverified",
                "measured-not-verifiable-by-this-policy",
                "falsification-result-not-paper-comparable",
                "superseded-retained",
                "verified",
                "blocked-performance",
            },
            f"results/manifest.json: {name} has invalid status",
        )
        if name in csv_files:
            validator.check(
                status == "legacy-unverified",
                f"results/manifest.json: top-level CSV {name} must remain legacy-unverified",
            )
    validate_course_release(validator, document, sorted(csv_files))


def validate_course_release(
    validator: Validator,
    results_manifest: dict[str, Any],
    top_level_csvs: list[str],
) -> None:
    indexed = results_manifest.get("release_evidence")
    validator.check(
        isinstance(indexed, dict),
        "results/manifest.json: release_evidence must be an object",
    )
    if not isinstance(indexed, dict):
        return
    validator.check(
        set(indexed) == set(RELEASE_EVIDENCE_STATUSES),
        "results/manifest.json: release evidence IDs do not match the course bundle",
    )
    for evidence_id, expected_status in RELEASE_EVIDENCE_STATUSES.items():
        metadata = indexed.get(evidence_id)
        validator.check(
            isinstance(metadata, dict),
            f"release evidence {evidence_id}: metadata must be an object",
        )
        if not isinstance(metadata, dict):
            continue
        validator.check(
            metadata.get("status") == expected_status,
            f"release evidence {evidence_id}: status mismatch",
        )
        validator.check(
            metadata.get("paper_comparable") is False,
            f"release evidence {evidence_id}: must not be paper-comparable",
        )
        validator.check(
            isinstance(metadata.get("claim_scope"), str)
            and bool(metadata["claim_scope"].strip()),
            f"release evidence {evidence_id}: claim_scope is required",
        )
        for key in ("receipt", "raw_rows", "summary", "recovery_receipt"):
            if key in metadata:
                _release_file(
                    validator,
                    metadata[key],
                    f"release evidence {evidence_id}.{key}",
                )
        provenance = metadata.get("provenance", [])
        validator.check(
            isinstance(provenance, list),
            f"release evidence {evidence_id}: provenance must be an array",
        )
        if isinstance(provenance, list):
            for index, value in enumerate(provenance):
                _release_file(
                    validator,
                    value,
                    f"release evidence {evidence_id}.provenance[{index}]",
                )

    release = results_manifest.get("course_release")
    validator.check(
        isinstance(release, dict),
        "results/manifest.json: course_release must be an object",
    )
    if not isinstance(release, dict):
        return
    validator.check(
        release.get("status") == "course-ready-paper-exact-blocked"
        and release.get("paper_exact_ready") is False
        and release.get("paper_comparable_results") == 0,
        "results/manifest.json: course release status is unsafe",
    )
    for key in ("receipt", "schema", "checklist"):
        _release_file(
            validator,
            release.get(key),
            f"results/manifest.json course_release.{key}",
        )
    receipt_path = _release_file(
        validator,
        release.get("receipt"),
        "course release receipt",
    )
    if receipt_path is None:
        return
    receipt = validator.load_json(receipt_path)
    if not isinstance(receipt, dict):
        return
    validator.check(
        receipt.get("schema") == "peps.course_release_receipt"
        and receipt.get("schema_version") == 1,
        "course release receipt: unsupported schema",
    )
    validator.check(
        receipt.get("status") == "course-ready-paper-exact-blocked"
        and receipt.get("paper_comparable_results") == 0,
        "course release receipt: unsafe release status",
    )

    promoted = receipt.get("promoted_evidence")
    validator.check(
        isinstance(promoted, list),
        "course release receipt: promoted_evidence must be an array",
    )
    promoted_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(promoted, list):
        for item in promoted:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                validator.errors.append(
                    "course release receipt: malformed promoted evidence item"
                )
                continue
            evidence_id = item["id"]
            validator.check(
                evidence_id not in promoted_by_id,
                f"course release receipt: duplicate evidence ID {evidence_id}",
            )
            promoted_by_id[evidence_id] = item
            validator.check(
                item.get("status") == RELEASE_EVIDENCE_STATUSES.get(evidence_id),
                f"course release receipt: status mismatch for {evidence_id}",
            )
            validator.check(
                item.get("paper_comparable") is False,
                f"course release receipt: {evidence_id} must not be paper-comparable",
            )
            files = item.get("files")
            validator.check(
                isinstance(files, list) and bool(files),
                f"course release receipt: {evidence_id} needs evidence files",
            )
            if isinstance(files, list):
                for index, file_spec in enumerate(files):
                    path_value = (
                        file_spec.get("path")
                        if isinstance(file_spec, dict)
                        else None
                    )
                    _release_file(
                        validator,
                        path_value,
                        f"course release receipt {evidence_id}.files[{index}]",
                    )
    validator.check(
        set(promoted_by_id) == set(RELEASE_EVIDENCE_STATUSES),
        "course release receipt: promoted IDs differ from results manifest",
    )

    legacy = receipt.get("legacy_top_level_csvs")
    validator.check(
        isinstance(legacy, dict)
        and legacy.get("status") == "legacy-unverified"
        and legacy.get("files") == top_level_csvs
        and legacy.get("count") == len(top_level_csvs),
        "course release receipt: legacy top-level CSV inventory mismatch",
    )

    for evidence_id, values in COURSE_SMOKE_RUNS.items():
        validate_course_smoke(validator, evidence_id, *values)

    image_pilot = validator.load_json(
        ROOT / "results" / "image_convergence" / "receipt.json"
    )
    if isinstance(image_pilot, dict):
        analysis = image_pilot.get("analysis", {})
        coverage = image_pilot.get("coverage", {})
        integrity = image_pilot.get("integrity", {})
        validator.check(
            isinstance(analysis, dict)
            and analysis.get("outcome") == "inconclusive"
            and analysis.get("recommended_budget_steps") is None,
            "Kodak convergence pilot must remain inconclusive",
        )
        validator.check(
            isinstance(coverage, dict)
            and coverage.get("complete") is True
            and isinstance(integrity, dict)
            and integrity.get("valid") is True
            and integrity.get("active_workers") == 0
            and image_pilot.get("paper_exact") is False
            and image_pilot.get("verified_table1") is False,
            "Kodak convergence pilot coverage/integrity status is invalid",
        )

    texture_pilot = validator.load_json(
        ROOT / "results" / "texture_repro" / "convergence_pilot.json"
    )
    texture_progress = validator.load_json(
        ROOT / "results" / "texture_repro" / "convergence_pilot_progress.json"
    )
    texture_recovery = validator.load_json(
        ROOT / "results" / "texture_repro" / "convergence_pilot_recovery.json"
    )
    if isinstance(texture_pilot, dict):
        decision = texture_pilot.get("decision", {})
        validator.check(
            isinstance(decision, dict)
            and decision.get("status") == "inconclusive_bounded_pilot"
            and decision.get("recommended_table2_steps") is None
            and decision.get("full_71m_step_run_authorized") is False,
            "texture convergence pilot must remain inconclusive and unauthorized",
        )
    validator.check(
        isinstance(texture_progress, dict)
        and texture_progress.get("complete") is True
        and texture_progress.get("active_workers") == 0
        and texture_progress.get("observations") == 180
        and isinstance(texture_recovery, dict)
        and texture_recovery.get("process_reconciliation", {}).get(
            "active_related_processes"
        )
        == 0
        and texture_recovery.get("safety", {}).get("full_table2_run_launched")
        is False,
        "texture pilot progress/recovery receipt is invalid",
    )

    sdf_validation = validator.load_json(
        ROOT / "results" / "sdf_repro" / "volume_validation.json"
    )
    release_sdf = receipt.get("sdf_public_provenance")
    if isinstance(sdf_validation, dict) and isinstance(release_sdf, list):
        source_rows = {
            item.get("asset_id"): item
            for item in sdf_validation.get("volumes", [])
            if isinstance(item, dict)
        }
        release_rows = {
            item.get("asset_id"): item
            for item in release_sdf
            if isinstance(item, dict)
        }
        expected_assets = {"lucy", "thai-statue", "armadillo"}
        validator.check(
            sdf_validation.get("status") == "passed"
            and sdf_validation.get("checksums_verified") is True
            and set(source_rows) == set(release_rows) == expected_assets,
            "course release receipt: public SDF inventory mismatch",
        )
        validator.check(
            sdf_validation.get("stonefish")
            == {
                "asset_id": "pitted-stonefish",
                "checked": False,
                "status": "deferred_auth_required",
                "substitution_used": False,
            },
            "course release receipt: Stonefish blocker was weakened",
        )
        for asset_id in expected_assets:
            source = source_rows.get(asset_id, {})
            released = release_rows.get(asset_id, {})
            provenance_path = _release_file(
                validator,
                released.get("receipt"),
                f"course release SDF provenance {asset_id}",
            )
            if provenance_path is not None:
                validator.check(
                    _sha256(provenance_path)
                    == source.get("tracked_provenance_sha256"),
                    f"course release SDF provenance {asset_id}: checksum mismatch",
                )
            validator.check(
                released.get("status") == "checksum_and_provenance_verified"
                and released.get("shape") == [512, 512, 512]
                and released.get("volume_sha256") == source.get("volume_sha256")
                and released.get("known_limit")
                == source.get("preprocessor_known_limit"),
                f"course release SDF provenance {asset_id}: receipt mismatch",
            )

    pilot_decisions = receipt.get("pilot_decisions")
    validator.check(
        isinstance(pilot_decisions, dict)
        and pilot_decisions.get("kodak", {}).get("status") == "inconclusive"
        and pilot_decisions.get("kodak", {}).get("recommended_budget_steps")
        is None
        and pilot_decisions.get("kodak", {}).get("full_run_authorized") is False
        and pilot_decisions.get("texture", {}).get("status")
        == "inconclusive_bounded_pilot"
        and pilot_decisions.get("texture", {}).get(
            "recommended_budget_steps"
        )
        is None
        and pilot_decisions.get("texture", {}).get("full_run_authorized") is False,
        "course release receipt: pilot decisions were promoted unsafely",
    )

    paper_exact = receipt.get("paper_exact")
    blocker_codes = {
        item.get("code")
        for item in paper_exact.get("blockers", [])
        if isinstance(item, dict)
    } if isinstance(paper_exact, dict) else set()
    validator.check(
        isinstance(paper_exact, dict)
        and paper_exact.get("ready") is False
        and paper_exact.get("paper_comparable_results") == 0
        and REQUIRED_PAPER_BLOCKERS.issubset(blocker_codes),
        "course release receipt: paper_exact blockers are incomplete",
    )
    process_audit = receipt.get("process_audit")
    validator.check(
        isinstance(process_audit, dict)
        and process_audit.get("active_recovery_or_full_training_processes") == 0,
        "course release receipt: active long-running process audit is not clean",
    )
    validation = receipt.get("validation")
    validator.check(
        isinstance(validation, dict)
        and validation.get("status")
        in {"passed", "passed-with-explicit-blocker"},
        "course release receipt: final validation status is not passed",
    )


def validate_slides(validator: Validator) -> None:
    slide_dir = ROOT / "slides"
    decks = sorted(slide_dir.glob("[0-9][0-9]_*.md"))
    validator.check(bool(decks), "slides: no numbered decks found")
    for path in decks:
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        validator.check(text.startswith("---\n"), f"{relative}: missing front matter")
        front_matter = text.split("---", 2)[1] if text.startswith("---") else ""
        validator.check("marp: true" in front_matter, f"{relative}: marp must be true")
        validator.check("title:" in front_matter, f"{relative}: title is required")
        validator.check(
            "\n---\n" in text[4:], f"{relative}: deck has no slide separator"
        )


def validate_release_files(validator: Validator) -> None:
    required = (
        "LICENSE",
        "CITATION.cff",
        "references.bib",
        "README.md",
        "env/constraints.txt",
        "course/RELEASE_CHECKLIST.md",
        "docs/07_readings_and_labs.md",
        "docs/08_midterm.md",
        "results/course_release/README.md",
        "results/course_release/receipt.json",
        "results/schemas/course_release_receipt.schema.json",
        ".github/workflows/ci.yml",
        ".github/workflows/amd-gpu.yml",
    )
    for value in required:
        path = ROOT / value
        validator.check(
            path.is_file() and path.stat().st_size > 0,
            f"missing release/course file: {value}",
        )
    if (ROOT / "references.bib").is_file():
        references = (ROOT / "references.bib").read_text(encoding="utf-8")
        validator.check(
            "10.1145/3806062" in references,
            "references.bib must contain the publisher PEPS DOI",
        )
    if (ROOT / "CITATION.cff").is_file():
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        validator.check(
            "10.1145/3806062" in citation,
            "CITATION.cff must reference the publisher PEPS DOI",
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        metavar="WEEK_OR_PATH",
        help="validate one notebook instead of the complete course",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    validator = Validator()
    labs = validate_labs(validator)

    if args.notebook:
        requested = args.notebook.upper()
        if requested in labs:
            path = ROOT / labs[requested]["notebook"]
        else:
            candidate = Path(args.notebook)
            path = candidate if candidate.is_absolute() else ROOT / candidate
        if not path.is_file():
            validator.errors.append(f"notebook not found: {args.notebook}")
        else:
            validate_notebook(validator, path)
    else:
        for week_id in WEEK_IDS:
            week = labs.get(week_id)
            if week and isinstance(week.get("notebook"), str):
                validate_notebook(validator, ROOT / week["notebook"])
        validate_data_manifests(validator)
        validate_results(validator)
        validate_slides(validator)
        validate_release_files(validator)

    if validator.errors:
        for error in validator.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"course validation failed: {len(validator.errors)} error(s)", file=sys.stderr)
        return 1
    scope = args.notebook or "release + course"
    print(f"course validation passed: {scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
