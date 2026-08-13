"""The occupancy preregistration and the result that resolves it must agree.

results/hip_occupancy_prediction.json is registered before the measurement it
describes. Two things can quietly ruin that. The numbers can drift away from the
formula they claim to come from, so they are recomputed here. The result must
also retain the failed gate: otherwise a CDNA runtime answer could be promoted
to a hardware-granule claim after the oracle that produced it was rejected.
"""

import json
import hashlib
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTION = ROOT / "results" / "hip_occupancy_prediction.json"
CAPS = ROOT / "results" / "hip_specialised_caps.json"
RESULT = ROOT / "results" / "hip_occupancy_probe_result.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def effective(footprint, granule):
    return math.ceil(footprint / granule) * granule


def test_rdna_predictions_follow_the_committed_model():
    doc = load(PREDICTION)
    pool = doc["model"]["pool_bytes_rdna"]
    granule = doc["model"]["granule_bytes_rdna"]
    for row in doc["rdna_predictions"]:
        expected = effective(row["footprint"], granule)
        assert row["effective"] == expected, row["footprint"]
        assert row["workgroups_per_wgp"] == pool // expected, row["footprint"]
        assert row["if_multiprocessor_means_cu"] == (pool // expected) // 2


def test_the_rdna_predictions_match_the_counters_that_exist():
    """The seven footprints with counter measurements are the reason to believe
    the model at all. If the file disagrees with them it is not the model."""
    doc = load(PREDICTION)
    measured = {
        row["footprint"]: row["measured_waves"]
        for row in load(CAPS)["occupancy"]["footprints"]
    }
    checked = 0
    for row in doc["rdna_predictions"]:
        if row["measured_waves_per_cu"] is None:
            continue
        checked += 1
        assert row["measured_waves_per_cu"] == measured[row["footprint"]]
        assert abs(row["workgroups_per_wgp"] - row["measured_waves_per_cu"]) < 0.1
    assert checked == 7, "all seven counter footprints must be carried over"


def test_cdna_predictions_follow_both_granule_hypotheses():
    doc = load(PREDICTION)
    pool = 65536
    for row in doc["cdna_predictions"]:
        for granule, key in ((1024, "1024"), (128, "128")):
            expected = effective(row["footprint"], granule)
            assert row[f"effective_{key}"] == expected, row["footprint"]
            assert row[f"blocks_per_cu_if_granule_{key}"] == pool // expected


def test_exactly_one_footprint_separates_the_two_granules():
    """A sweep where every point agrees under both hypotheses would run fine and
    decide nothing. The decisive footprint is named in the file, so it has to be
    the one that is actually decisive."""
    doc = load(PREDICTION)
    separating = [
        row["footprint"]
        for row in doc["cdna_predictions"]
        if row["blocks_per_cu_if_granule_1024"]
        != row["blocks_per_cu_if_granule_128"]
    ]
    named = doc["test_two_does_the_form_survive_a_change_of_architecture"][
        "the_decisive_footprint"
    ]
    assert separating == [named["footprint"]]


def test_the_preregistration_resolves_to_the_result():
    doc = load(PREDICTION)
    assert doc["status"] == "resolved_oracle_rejected"
    assert ROOT / doc["resolved_by"] == RESULT


def test_stage_one_rejects_the_runtime_oracle_on_two_odd_wgp_cases():
    result = load(RESULT)
    stage = result["stage_one"]
    assert stage["verdict"] == "oracle_rejected"
    assert (stage["passed"], stage["total"]) == (5, 7)
    failed = [
        row for row in stage["comparisons"] if not row["matches_within_0_1"]
    ]
    assert [row["footprint"] for row in failed] == [10752, 13312]
    assert [row["api_waves_per_multiprocessor"] for row in failed] == [12, 8]
    assert [row["counter_waves_per_cu"] for row in failed] == [10.97, 8.97]


def test_the_registered_blocks_versus_waves_unit_error_is_disclosed():
    result = load(RESULT)
    correction = result["stage_one"]["pre_registration_correction"]
    assert "blocks_per_multiprocessor" in correction["original_wording"]
    assert "different units" in correction["problem"]
    assert "waves_per_multiprocessor" in correction["corrected_gate"]
    assert correction["predictions_rewritten"] is False


def test_all_three_runtime_sweeps_have_identical_block_counts():
    result = load(RESULT)
    rows = []
    for raw in result["raw_receipts"]:
        payload = load(ROOT / raw["path"])
        rows.append(
            [row["blocks_per_multiprocessor"] for row in payload["rows"]]
        )
    assert rows[0] == rows[1] == rows[2]


def test_raw_receipts_keep_the_hashes_recorded_at_capture():
    result = load(RESULT)
    for raw in result["raw_receipts"]:
        payload = ROOT / raw["path"]
        assert hashlib.sha256(payload.read_bytes()).hexdigest() == raw["sha256"]


def test_runtime_rows_are_plain_64_kib_dynamic_lds_divisions():
    result = load(RESULT)
    for raw in result["raw_receipts"]:
        payload = load(ROOT / raw["path"])
        for row in payload["rows"]:
            assert row["blocks_per_multiprocessor"] == 65536 // row["footprint"]


def test_the_gfx942_six_is_not_promoted_to_a_granule_verdict():
    result = load(RESULT)
    cross = result["cross_architecture_observation"]
    assert cross["verdict"] == "not_admissible_as_a_granule_test"
    assert cross["decisive_10752_row"] == {
        "gfx1151": 6,
        "gfx1201": 6,
        "gfx942": 6,
    }
    assert "does not establish" in cross["what_six_means"]
