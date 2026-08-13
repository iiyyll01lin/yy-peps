"""Every evidence path a success criterion cites must resolve.

course/labs.json says what a student has to show and where the reference
evidence lives. Nothing checked that those paths existed, so renaming a file
under results/ would leave the course grading against nothing, silently and
for as long as nobody opened the JSON.

The second half pins the field values the newer criteria quote. A criterion
that states "count 0 against expected_pairs 72" is making a factual claim
about committed evidence, and it should fail loudly when that stops being
true rather than quietly misinform a student.

Whether the named fields *exist* is checked by scripts/check_criteria_evidence.py,
which derives the names from labs.json. A hand-written list lived here and was
removed: it covered four files and could not notice a criterion it had never
been told about. What is quoted here are values, which cannot be derived.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LABS = json.loads((ROOT / "course" / "labs.json").read_text(encoding="utf-8"))

# Paths appear inside prose, so match on the extension rather than position.
PATH_TOKEN = re.compile(r"[A-Za-z0-9_./*-]+\.(?:json|csv|ipynb|md|py|toml)")
# Prose references to commands and to files a student is told to create.
IGNORE_PREFIXES = ("python ", "pytest ", "bash ")


def cited_paths() -> list[tuple[str, str, str]]:
    found = []
    for week in LABS["weeks"]:
        for criterion in week["success_criteria"]:
            evidence = criterion["evidence"]
            if evidence.startswith(IGNORE_PREFIXES):
                continue
            for token in PATH_TOKEN.findall(evidence):
                if "/" not in token:
                    continue
                found.append((week["id"], criterion["id"], token))
    return found


def test_the_scan_finds_something_to_check():
    # A regex that silently matches nothing would make the next test vacuous.
    assert len(cited_paths()) >= 10


@pytest.mark.parametrize("week,criterion,token", cited_paths())
def test_cited_evidence_path_exists(week, criterion, token):
    if "*" in token:
        parent = ROOT / Path(token).parent
        matches = list(parent.glob(Path(token).name)) if parent.exists() else []
        assert matches, f"{week}/{criterion} cites {token}, which matches nothing"
    else:
        assert (ROOT / token).exists(), f"{week}/{criterion} cites a missing {token}"


# --- the factual claims the newer criteria make ------------------------

QUOTED = [
    ("results/image_repro_paired.json", "comparisons.0.count", 0),
    ("results/image_repro_paired.json", "comparisons.0.expected_pairs", 72),
    ("results/image_repro_paired.json", "comparisons.0.complete", False),
    ("results/image_repro_paired.json", "comparisons.0.mean_delta", None),
    ("results/sdf_repro/table4_deferred_auth_required.json",
     "asset.data_access_attempted", False),
    ("results/sdf_repro/cost.json", "jobs", 57),
    ("results/sdf_repro/volume_validation.json", "checksums_verified", True),
    ("results/sdf_repro/volume_validation.json",
     "recovery.partial_files_found", 0),
    ("results/sdf_repro/volume_validation.json",
     "recovery.active_related_processes", 0),
]


def dig(payload, dotted: str):
    for part in dotted.split("."):
        payload = payload[int(part)] if part.isdigit() else payload[part]
    return payload


@pytest.mark.parametrize("path,dotted,expected", QUOTED)
def test_quoted_field_still_holds(path, dotted, expected):
    payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
    assert dig(payload, dotted) == expected


def test_no_week_is_graded_by_fewer_than_two_criteria():
    for week in LABS["weeks"]:
        assert len(week["success_criteria"]) >= 2, week["id"]
