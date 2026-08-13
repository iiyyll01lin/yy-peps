"""The superseded-claim check must pass, and must be capable of failing.

Three claims here were corrected after measurement disproved them, and each
took several hand-run searches to chase down. scripts/check_superseded_claims.py
makes that mechanical. A check nobody runs is worth nothing, so this runs it;
and a check that cannot fail is worth less than nothing, so this plants a stray
claim in a scratch tree and requires it to be caught.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

check_superseded_claims = pytest.importorskip("check_superseded_claims")


def test_no_superseded_claim_has_leaked_back():
    assert check_superseded_claims.main(str(ROOT)) == 0


def test_the_check_catches_a_planted_claim(tmp_path, capsys):
    # A scratch tree, so nothing is written into the repository. Nothing here
    # is on any allowlist, so a single occurrence has to be a failure.
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "stray.md").write_text(
        "the kernel measured 42.304207 ms\n", encoding="utf-8"
    )
    assert check_superseded_claims.main(str(tmp_path)) != 0
    assert "docs/stray.md" in capsys.readouterr().out


def test_a_longer_number_is_not_a_match(tmp_path):
    # 42.30421821514322 is a PSNR in results/texture_repro/table2_instances.csv.
    # An unanchored search calls it the latency median, and that false alarm is
    # what gets a check switched off.
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "psnr.md").write_text(
        "psnr 42.30421821514322 and 148.200451 and 253.787079\n", encoding="utf-8"
    )
    claim = next(
        c for c in check_superseded_claims.CLAIMS
        if c.name == "artefact_latency_medians"
    )
    assert claim.regex.search((tmp_path / "docs" / "psnr.md").read_text()) is None


def test_rotted_patterns_are_reported_rather_than_passing(tmp_path):
    # An empty tree matches nothing. That must fail, because a check whose
    # patterns have stopped matching looks identical to a clean repository.
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "unrelated.md").write_text("nothing here\n", encoding="utf-8")
    assert check_superseded_claims.main(str(tmp_path)) != 0
