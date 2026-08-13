"""The occupancy predictions must follow the model, and must stay predictions.

results/hip_occupancy_prediction.json is registered before the measurement it
describes. Two things can quietly ruin that. The numbers can drift away from the
formula they claim to come from, so they are recomputed here. And measured values
can be filed into it while the status still says nothing was measured, which
would turn a prediction into a result without anyone deciding to.
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTION = ROOT / "results" / "hip_occupancy_prediction.json"
CAPS = ROOT / "results" / "hip_specialised_caps.json"


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


def test_it_is_still_a_prediction():
    doc = load(PREDICTION)
    assert doc["status"] == "predicted_not_yet_measured"
    blob = json.dumps(doc["cdna_predictions"])
    assert "measured" not in blob, (
        "a measured value appeared in the CDNA rows while the status still "
        "says nothing has been measured"
    )
