"""The criteria-evidence check must pass, and must be capable of failing.

A success criterion tells a student where to look to find out whether they have
met the bar. scripts/check_criteria_evidence.py holds the criteria to that
promise by opening the artefact and looking for the field. These tests run it
against the repository, then build small scratch trees where the promise is
broken in each of the ways it can break, and require each one to be caught.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

check_criteria_evidence = pytest.importorskip("check_criteria_evidence")


def build(tmp_path, evidence, artefact=None):
    """A one-criterion course, optionally with the artefact it names."""
    labs = {
        "labs": [
            {
                "id": "W01",
                "topic": "scratch",
                "success_criteria": [{"id": "only", "evidence": evidence}],
            }
        ]
    }
    (tmp_path / "course").mkdir(parents=True, exist_ok=True)
    (tmp_path / "course" / "labs.json").write_text(
        json.dumps(labs), encoding="utf-8"
    )
    if artefact is not None:
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)
        (tmp_path / "results" / "thing.json").write_text(
            json.dumps(artefact), encoding="utf-8"
        )
    return str(tmp_path)


def test_every_criterion_promise_in_this_repository_is_kept():
    problems, _ = check_criteria_evidence.check(ROOT)
    assert problems == []
    assert check_criteria_evidence.main(str(ROOT)) == 0


def test_the_field_half_of_the_check_is_actually_running():
    """Paths and fields are counted separately on purpose. If the field clause
    ever stops parsing, every path promise still resolves and the check goes on
    passing while verifying half of what it claims to. The floors are set below
    the current counts so that adding criteria cannot trip them, and removing a
    body of evidence has to be noticed rather than absorbed."""
    _, counts = check_criteria_evidence.check(ROOT)
    assert counts["paths"] >= 30
    assert counts["fields"] >= 20


def test_a_kept_promise_passes(tmp_path):
    root = build(
        tmp_path,
        "results/thing.json, fields alpha and beta",
        {"alpha": 1, "beta": 2},
    )
    assert check_criteria_evidence.main(root) == 0


def test_a_named_file_that_is_not_there_is_caught(tmp_path, capsys):
    root = build(tmp_path, "results/thing.json, fields alpha and beta")
    assert check_criteria_evidence.main(root) == 1
    assert "results/thing.json" in capsys.readouterr().out


def test_a_promised_field_that_is_not_in_the_artefact_is_caught(
    tmp_path, capsys
):
    root = build(
        tmp_path, "results/thing.json, fields alpha and beta", {"alpha": 1}
    )
    assert check_criteria_evidence.main(root) == 1
    out = capsys.readouterr().out
    assert "beta" in out
    assert "alpha" not in out


def test_a_field_nested_inside_the_artefact_still_counts(tmp_path):
    """Receipts nest under run names and under per-case rows. A criterion that
    says "field answer" means the receipt records one, not that it sits at the
    top level."""
    root = build(
        tmp_path,
        "results/thing.json, field answer",
        {"runs": [{"case": "a", "answer": "no"}]},
    )
    assert check_criteria_evidence.main(root) == 0


def test_a_dotted_field_must_match_the_nesting(tmp_path, capsys):
    root = build(
        tmp_path,
        "results/thing.json, fields recovery.action",
        {"recovery": {"note": "x"}, "action": "y"},
    )
    assert check_criteria_evidence.main(root) == 1
    assert "recovery.action" in capsys.readouterr().out


def test_a_course_promising_nothing_is_reported_rather_than_passing(
    tmp_path, capsys
):
    """Criteria with no file and no field leave nothing to verify. That looks
    identical to a clean repository from the outside, so it is a failure."""
    root = build(tmp_path, "executed notebook output")
    assert check_criteria_evidence.main(root) == 1
    assert "without checking anything" in capsys.readouterr().out
