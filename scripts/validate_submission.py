#!/usr/bin/env python3
"""Validate midterm or capstone evidence without running experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = re.compile(
    r"(?:REPLACE_ME|TODO(?:\(student\))?|"
    r"<(?:your|replace|metric|number|title|name|one sentence)[^>]*>)",
    re.IGNORECASE,
)
SHA256 = re.compile(r"[0-9a-f]{64}", re.IGNORECASE)
GIT_COMMIT = re.compile(r"[0-9a-f]{7,40}", re.IGNORECASE)


class ValidationError(ValueError):
    """A submission is incomplete or internally inconsistent."""


def _nonblank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or PLACEHOLDER.search(value):
        raise ValidationError(f"{field} is blank or still contains a placeholder")
    return value.strip()


def _finite(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValidationError(f"{field} must be finite")
    return number


def _artifact_path(value: Any, field: str) -> Path:
    relative = Path(_nonblank(value, field))
    if relative.is_absolute():
        raise ValidationError(f"{field} must be repository-relative")
    resolved = (ROOT / relative).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValidationError(f"{field} escapes the repository") from exc
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValidationError(f"{field} does not name a non-empty file: {relative}")
    return resolved


def _load_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{field} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{field} must contain a JSON object")
    return value


def _has_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return PLACEHOLDER.search(value) is not None
    if isinstance(value, dict):
        return any(_has_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_placeholder(item) for item in value)
    return False


def _validate_common(document: dict[str, Any], kind: str) -> None:
    if document.get("schema_version") != 1:
        raise ValidationError("schema_version must be 1")
    if document.get("submission_kind") != kind:
        raise ValidationError(f"submission_kind must be {kind!r}")
    if document.get("status") != "verified":
        raise ValidationError("status must be 'verified'; draft/legacy evidence cannot pass")
    _nonblank(document.get("student_id"), "student_id")
    if document.get("profile") not in {"course_fast", "paper_exact"}:
        raise ValidationError("profile must be course_fast or paper_exact")
    commit = _nonblank(document.get("git_commit"), "git_commit")
    if not GIT_COMMIT.fullmatch(commit):
        raise ValidationError("git_commit must be a 7-40 character hexadecimal revision")

    seeds = document.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise ValidationError("seeds must be a non-empty list of integers")

    checksums = document.get("data_checksums")
    if not isinstance(checksums, list) or not checksums:
        raise ValidationError("data_checksums must be a non-empty list")
    for index, item in enumerate(checksums):
        if not isinstance(item, dict):
            raise ValidationError(f"data_checksums[{index}] must be an object")
        _nonblank(item.get("asset"), f"data_checksums[{index}].asset")
        checksum = _nonblank(
            item.get("sha256"), f"data_checksums[{index}].sha256"
        )
        if not SHA256.fullmatch(checksum):
            raise ValidationError(
                f"data_checksums[{index}].sha256 must be 64 hexadecimal characters"
            )

    _nonblank(document.get("conclusion"), "conclusion")
    _nonblank(document.get("limitations"), "limitations")


def _validate_artifacts(document: dict[str, Any], kind: str) -> dict[str, Path]:
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValidationError("artifacts must be an object")
    names = ["notebook", "results_csv", "run_manifest"]
    if kind == "capstone":
        names.append("slide")
    paths = {
        name: _artifact_path(artifacts.get(name), f"artifacts.{name}")
        for name in names
    }

    if paths["notebook"].suffix != ".ipynb":
        raise ValidationError("artifacts.notebook must be an .ipynb file")
    notebook = _load_json(paths["notebook"], "artifacts.notebook")
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValidationError("artifacts.notebook is not a version-4 notebook")
    executed = [
        cell
        for cell in notebook.get("cells", [])
        if isinstance(cell, dict)
        and cell.get("cell_type") == "code"
        and cell.get("execution_count") is not None
    ]
    if not executed:
        raise ValidationError("artifacts.notebook has no executed code cells")

    if paths["results_csv"].suffix != ".csv":
        raise ValidationError("artifacts.results_csv must be a .csv file")
    run = _load_json(paths["run_manifest"], "artifacts.run_manifest")
    if _has_placeholder(run):
        raise ValidationError("artifacts.run_manifest still contains a placeholder")
    run_profile = run.get("profile")
    if run_profile is not None and run_profile != document["profile"]:
        raise ValidationError("run-manifest profile does not match submission profile")

    if kind == "capstone":
        slide_text = paths["slide"].read_text(encoding="utf-8")
        if not slide_text.startswith("---\n") or "marp: true" not in slide_text[:300]:
            raise ValidationError("artifacts.slide is not a Marp deck")
        if PLACEHOLDER.search(slide_text):
            raise ValidationError("artifacts.slide still contains a placeholder")
    return paths


def _validate_csv(
    document: dict[str, Any],
    path: Path,
    kind: str,
) -> None:
    contract = document.get("csv_contract")
    if not isinstance(contract, dict):
        raise ValidationError("csv_contract must be an object")
    required = contract.get("required_columns")
    numeric = contract.get("numeric_columns")
    minimum_rows = contract.get("minimum_rows")
    if (
        not isinstance(required, list)
        or not required
        or any(not isinstance(column, str) for column in required)
    ):
        raise ValidationError("csv_contract.required_columns must be non-empty")
    if not isinstance(numeric, list) or any(
        not isinstance(column, str) for column in numeric
    ):
        raise ValidationError("csv_contract.numeric_columns must be a list")
    if isinstance(minimum_rows, bool) or not isinstance(minimum_rows, int):
        raise ValidationError("csv_contract.minimum_rows must be an integer")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = sorted(set(required) - set(fieldnames))
        if missing:
            raise ValidationError(f"results CSV is missing columns: {', '.join(missing)}")
        rows = list(reader)
    if len(rows) < minimum_rows:
        raise ValidationError(
            f"results CSV has {len(rows)} rows; expected at least {minimum_rows}"
        )

    for index, row in enumerate(rows, start=2):
        for column in required:
            _nonblank(row.get(column), f"CSV row {index}.{column}")
        for column in numeric:
            _finite(row.get(column), f"CSV row {index}.{column}")
        if row.get("profile") != document["profile"]:
            raise ValidationError(f"CSV row {index} profile does not match submission")
        if row.get("status") != "verified":
            raise ValidationError(f"CSV row {index} status must be verified")

    if kind == "midterm":
        tasks = {row["task"].strip().lower() for row in rows}
        if not {"fig5", "table1"}.issubset(tasks):
            raise ValidationError("midterm CSV must contain both fig5 and table1 rows")
    else:
        stages = {row["stage"].strip().lower() for row in rows}
        if not {"baseline", "extension"}.issubset(stages):
            raise ValidationError(
                "capstone CSV must contain baseline and extension stages"
            )
        if any(row["metric"].strip() != document["primary_metric"] for row in rows):
            raise ValidationError(
                "capstone CSV metric does not match primary_metric"
            )


def validate(document: dict[str, Any], kind: str) -> None:
    _validate_common(document, kind)
    if kind == "midterm":
        conditions = document.get("matched_conditions")
        if not isinstance(conditions, list) or len(conditions) < 3:
            raise ValidationError("matched_conditions must contain at least three items")
        for index, condition in enumerate(conditions):
            _nonblank(condition, f"matched_conditions[{index}]")
    else:
        if document.get("track") not in {"A", "B", "C", "D"}:
            raise ValidationError("track must be A, B, C, or D")
        _nonblank(document.get("hypothesis"), "hypothesis")
        metric = _nonblank(document.get("primary_metric"), "primary_metric")
        for stage in ("baseline", "extension"):
            value = document.get(stage)
            if not isinstance(value, dict):
                raise ValidationError(f"{stage} must be an object")
            _nonblank(value.get("label"), f"{stage}.label")
            _finite(value.get("value"), f"{stage}.value")
            _nonblank(value.get("units"), f"{stage}.units")
        if document["baseline"].get("source") != "rerun":
            raise ValidationError("baseline.source must be 'rerun', not a legacy CSV")
        if not metric:
            raise ValidationError("primary_metric is required")

    paths = _validate_artifacts(document, kind)
    _validate_csv(document, paths["results_csv"], kind)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--kind", choices=("midterm", "capstone"), required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    try:
        document = _load_json(path, "submission manifest")
        validate(document, args.kind)
    except ValidationError as exc:
        print(f"submission invalid: {exc}", file=sys.stderr)
        return 1
    print(f"submission valid: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
