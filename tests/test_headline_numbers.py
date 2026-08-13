"""The headline-number check must pass, and must be capable of failing.

scripts/check_headline_numbers.py recomputes each figure the README argues from
and requires the prose to agree. These tests run it against the repository, then
break the agreement from each side in a scratch tree and require each break to be
caught -- because a check that only ever passes is indistinguishable from one
that has stopped looking.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

check_headline_numbers = pytest.importorskip("check_headline_numbers")


def mirror(tmp_path, edit_receipt=None, edit_readme=None, edit_csv=None):
    """A copy of the sources this check reads, optionally perturbed."""
    for claim in check_headline_numbers.CLAIMS:
        source = tmp_path / claim.source
        if source.exists():
            continue
        source.parent.mkdir(parents=True, exist_ok=True)
        original = ROOT / claim.source
        if original.suffix == ".csv":
            text = original.read_text(encoding="utf-8")
            if edit_csv is not None:
                text = edit_csv(claim.source, text)
            source.write_text(text, encoding="utf-8")
            continue
        payload = json.loads(original.read_text(encoding="utf-8"))
        if edit_receipt is not None:
            payload = edit_receipt(claim.source, payload)
        source.write_text(json.dumps(payload), encoding="utf-8")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if edit_readme is not None:
        readme = edit_readme(readme)
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    return str(tmp_path)


def test_every_headline_number_still_comes_out_of_its_artefact():
    problems, checked = check_headline_numbers.check(ROOT)
    assert problems == []
    assert checked == len(check_headline_numbers.CLAIMS)


def test_the_mirror_of_the_repository_passes(tmp_path):
    assert check_headline_numbers.main(mirror(tmp_path)) == 0


def test_a_receipt_that_moves_away_from_the_prose_is_caught(tmp_path, capsys):
    def bump(source, payload):
        if source.endswith("table2.json"):
            payload["progress"]["completed_jobs"] = 593
        return payload

    assert check_headline_numbers.main(mirror(tmp_path, edit_receipt=bump)) == 1
    assert "593" in capsys.readouterr().out


def test_prose_that_moves_away_from_the_receipt_is_caught(tmp_path, capsys):
    def retype(readme):
        return readme.replace("2.95 ms", "2.59 ms")

    assert check_headline_numbers.main(mirror(tmp_path, edit_readme=retype)) == 1
    assert "bi-grid specialised latency" in capsys.readouterr().out


def test_a_renamed_field_is_reported_rather_than_skipped(tmp_path, capsys):
    """A path that stops resolving must fail. Silently skipping it would leave
    the check green while measuring nothing."""

    def rename(source, payload):
        if source.endswith("receipt.json") and "method_means" in payload:
            payload["means"] = payload.pop("method_means")
        return payload

    assert check_headline_numbers.main(mirror(tmp_path, edit_receipt=rename)) == 1
    assert "method_means.paper" in capsys.readouterr().out


def test_an_empty_claim_list_is_reported_rather_than_passing(monkeypatch, capsys):
    monkeypatch.setattr(check_headline_numbers, "CLAIMS", [])
    assert check_headline_numbers.main(str(ROOT)) == 1
    assert "without comparing anything" in capsys.readouterr().out


def test_a_renamed_csv_column_is_reported_rather_than_skipped(tmp_path, capsys):
    def rename(source, text):
        return text.replace("delta_iou", "delta")

    assert check_headline_numbers.main(mirror(tmp_path, edit_csv=rename)) == 1
    assert "delta_iou" in capsys.readouterr().out


def test_a_counted_phrase_that_stops_matching_the_prose_is_caught(
    tmp_path, capsys
):
    """The counts are checked as phrases. A bare integer would match almost any
    line in the README, so the phrase is what makes the claim falsifiable."""
    rows = check_headline_numbers.load(ROOT / check_headline_numbers.SDF_MAPE)
    phrase = f"{check_headline_numbers.above_paper(rows)} of the {len(rows)}"

    def retype(readme):
        return readme.replace(phrase, "every one of the")

    assert check_headline_numbers.main(mirror(tmp_path, edit_readme=retype)) == 1
    assert "under MAPE" in capsys.readouterr().out


def test_the_oracle_rejection_count_cannot_be_softened_in_prose(tmp_path, capsys):
    def retype(readme):
        return readme.replace("5 of the 7", "6 of the 7")

    assert check_headline_numbers.main(mirror(tmp_path, edit_readme=retype)) == 1
    assert "oracle pass count" in capsys.readouterr().out
