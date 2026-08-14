"""The census predictions must follow the model, and must stay predictions.

results/hip_occupancy_census_prediction.json is registered before the run, like
the runtime-API attempt before it. That one was resolved by rejecting its own
oracle, so the temptation this time is to leave the gate vague enough to pass.
These tests recompute every row from the stated model and pin the gate.

results/hip_lds_codeobject_scan.json is already complete. Its value is a
negative result, which is the kind that quietly gets softened later, so the
unaligned sizes that make it conclusive are asserted here too.
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "results" / "hip_occupancy_census_prediction.json"
SCAN = ROOT / "results" / "hip_lds_codeobject_scan.json"
CAPS = ROOT / "results" / "hip_specialised_caps.json"

# multi_processor_count as HIP reports it, and the pool that count indexes.
PARTS = {
    "gfx1151": {"mp": 20, "pool": 131072},
    "gfx1201": {"mp": 32, "pool": 131072},
    "gfx942": {"mp": 152, "pool": 65536},
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def effective(footprint, granule):
    return math.ceil(footprint / granule) * granule


def test_every_predicted_row_follows_the_model():
    doc = load(CENSUS)
    for arch, rows in doc["predictions"].items():
        pool = PARTS[arch]["pool"]
        count = PARTS[arch]["mp"]
        for row in rows:
            per_mp = pool // effective(row["footprint"], 1024)
            assert row["per_multiprocessor"] == per_mp, (arch, row["footprint"])
            assert row["peak_resident_blocks"] == per_mp * count
            alt = pool // effective(row["footprint"], 128)
            assert row["if_granule_128_peak"] == alt * count
            assert row["discriminating"] is (per_mp != alt)


def test_the_rdna_predictions_agree_with_the_counters_that_exist():
    """peak divided by multi_processor_count is waves per CU on RDNA. If the
    prediction disagrees with the counters, the file is not the model."""
    doc = load(CENSUS)
    measured = {
        row["footprint"]: row["measured_waves"]
        for row in load(CAPS)["occupancy"]["footprints"]
    }
    checked = 0
    for row in doc["predictions"]["gfx1151"]:
        if row["counter_waves_per_cu"] is None:
            continue
        checked += 1
        assert row["counter_waves_per_cu"] == measured[row["footprint"]]
        per_mp = row["peak_resident_blocks"] / PARTS["gfx1151"]["mp"]
        assert abs(per_mp - row["counter_waves_per_cu"]) < 0.1
    assert checked == 7, "all seven counter footprints must be carried over"


def test_the_gate_names_the_part_that_has_counters():
    gate = load(CENSUS)["gate_before_any_cdna_reading"]
    assert "gfx1151" in gate["run_on"]
    assert "Stop" in gate["if_it_fails"]


def test_the_cdna_discriminator_is_the_only_one_that_separates():
    doc = load(CENSUS)
    separating = [
        row["footprint"]
        for row in doc["predictions"]["gfx942"]
        if row["discriminating"]
    ]
    named = doc["decisive_footprint_on_cdna"]
    assert separating == [named["footprint"]]
    assert named["predicts_if_granule_is_1024"] != named["predicts_if_granule_is_128"]


def test_rdna_has_four_separating_footprints_unlike_the_rejected_api_test():
    """The runtime API sweep could not separate the granules anywhere the
    counters exist. If that were still true the census would be no better."""
    doc = load(CENSUS)
    separating = [
        row["footprint"]
        for row in doc["predictions"]["gfx1151"]
        if row["discriminating"]
    ]
    assert separating == doc["decisive_footprints_on_rdna"]["footprints"]
    assert len(separating) == 4


def test_the_census_is_still_a_prediction():
    doc = load(CENSUS)
    assert doc["status"] == "predicted_not_yet_measured"
    blob = json.dumps(doc["predictions"])
    assert "observed" not in blob and "measured_peak" not in blob


def test_the_code_object_scan_records_the_request_verbatim():
    doc = load(SCAN)
    assert doc["status"] == "complete-route-cannot-answer"
    for row in doc["observations"]:
        assert row["recorded"] == row["requested"], row
        assert row["ceil_1024"] == effective(row["requested"], 1024)
        assert row["ceil_128"] == effective(row["requested"], 128)


def test_the_scan_includes_sizes_aligned_to_neither_granule():
    """Every other footprint in the sweep is a multiple of 128, so without an
    unaligned size the scan cannot tell 'verbatim' from 'rounded to 128'."""
    doc = load(SCAN)
    unaligned = {
        row["requested"]
        for row in doc["observations"]
        if row["requested"] % 128 != 0 and row["requested"] % 1024 != 0
    }
    assert unaligned, "the conclusion needs at least one unaligned size"
    for arch in ("gfx1201", "gfx942"):
        seen = {
            row["requested"]
            for row in doc["observations"]
            if row["arch"] == arch and row["requested"] in unaligned
        }
        assert seen == unaligned, f"{arch} must cover the unaligned sizes too"
