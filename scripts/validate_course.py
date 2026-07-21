#!/usr/bin/env python3
"""Static release/course validation with no dataset or GPU requirement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WEEK_IDS = tuple(f"W{week:02d}" for week in range(1, 15))
CHECKSUM_LENGTHS = {"md5": 32, "sha256": 64}


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
    validator.check(
        set(artifacts) == csv_files,
        "results/manifest.json must list every top-level results CSV exactly once",
    )
    for name, metadata in artifacts.items():
        status = metadata.get("status") if isinstance(metadata, dict) else None
        validator.check(
            status in {"legacy-unverified", "verified"},
            f"results/manifest.json: {name} has invalid status",
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
        "docs/07_readings_and_labs.md",
        "docs/08_midterm.md",
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
