"""The liveness check must pass, and must be capable of failing.

A status file in this repository recorded two workers as alive and was then
committed. The processes died the same day; the assertion did not. Three weeks
later the file still said ``"process_alive": true`` while the finished result
sat beside it saying the run was complete, and nothing noticed, because no
script and no test read that file.

So this runs the check, and then plants an unqualified live assertion to prove
the check can still fail. It also pins the two things that made the original
defect invisible: that the good pattern already in the repository is accepted,
and that the resolution now attached to the offending file agrees with the
authoritative result rather than restating the snapshot's own counters.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

check_status_liveness = pytest.importorskip("check_status_liveness")

STATUS = ROOT / "results" / "texture_repro" / "table2_status.json"
RESULT = ROOT / "results" / "texture_repro" / "table2.json"


def test_no_committed_record_asserts_an_unqualified_live_process():
    assert check_status_liveness.main(str(ROOT)) == 0


def test_the_check_catches_an_unqualified_assertion():
    planted = {"workers": [{"state": "running", "pid": 1}]}
    found = check_status_liveness.violations_in(planted, False, "planted.json")
    assert found, "an unqualified live assertion must be reported"
    assert "state" in found[0]


def test_a_qualified_assertion_is_accepted():
    # The same assertion, next to a recorded outcome, is a record and not a report.
    qualified = {"workers": [{"state": "running", "pid": 1, "outcome": "terminated"}]}
    assert check_status_liveness.violations_in(qualified, False, "q.json") == []


def test_qualification_is_inherited_by_nested_records():
    nested = {"outcome": "terminated", "workers": [{"process_alive": True}]}
    assert check_status_liveness.violations_in(nested, False, "n.json") == []


def test_a_dead_assertion_is_never_a_violation():
    dead = {"workers": [{"alive": False, "state": "exited"}]}
    assert check_status_liveness.violations_in(dead, False, "d.json") == []


def test_the_pattern_the_repository_already_had_right_still_passes():
    # The image status writer names its field recorded_state and re-verifies the
    # pid. That is the pattern this check was written to make universal, so it
    # must not be what the check flags.
    image = json.loads(
        (ROOT / "results" / "image_repro_status.json").read_text(encoding="utf-8")
    )
    assert check_status_liveness.violations_in(image, False, "image") == []


def test_the_resolution_defers_to_the_authoritative_result():
    resolution = json.loads(STATUS.read_text(encoding="utf-8"))["liveness_resolution"]
    assert resolution["workers_alive_at_verification"] is False
    assert resolution["boot_id_unchanged_since_record"] is True
    assert resolution["authoritative_result"] == "results/texture_repro/table2.json"
    assert json.loads(RESULT.read_text(encoding="utf-8"))["complete"] is True


def test_the_snapshot_counters_are_not_quietly_rewritten():
    # The record of the superseded attempt stays as it was observed; only the
    # later-known resolution is added. Rewriting it would falsify the history.
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    assert status["complete"] is False
    assert status["completed_jobs"] == 0
