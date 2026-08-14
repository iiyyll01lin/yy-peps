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
import hashlib
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "results" / "hip_occupancy_census_prediction.json"
RESULT = ROOT / "results" / "hip_occupancy_census_result.json"
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


def test_the_census_resolves_to_its_result():
    doc = load(CENSUS)
    assert doc["status"] == "resolved_gate_passed"
    assert ROOT / doc["resolved_by"] == RESULT
    assert doc["predictions_rewritten"] is False


def test_the_gate_passed_on_every_counter_footprint():
    gate = load(RESULT)["gate_on_the_part_with_counters"]
    assert gate["verdict"] == "passed"
    assert (gate["passed"], gate["total"]) == (7, 7)
    assert all(row["within_0_1"] for row in gate["comparisons"])


def test_the_raw_captures_keep_the_hashes_recorded_at_capture():
    result = load(RESULT)
    for raw in result["raw_receipts"]:
        payload = ROOT / raw["path"]
        assert hashlib.sha256(payload.read_bytes()).hexdigest() == raw["sha256"]


def test_rdna_captures_match_the_registered_predictions_exactly():
    """Both RDNA parts came out on every row. If a capture is ever replaced by
    one that does not, the confirmation of the granule goes with it."""
    doc = load(CENSUS)
    result = load(RESULT)
    paths = {raw["arch"]: raw["path"] for raw in result["raw_receipts"]}
    for arch in ("gfx1151", "gfx1201"):
        capture = load(ROOT / paths[arch])
        predicted = {r["footprint"]: r for r in doc["predictions"][arch]}
        for row in capture["rows"]:
            assert (row["peak_resident_blocks"]
                    == predicted[row["footprint"]]["peak_resident_blocks"])


def test_the_cdna_refutation_is_not_softened():
    """The one row that missed is the whole CDNA finding. A capture or a reading
    that quietly agreed with 1024 again would erase it."""
    result = load(RESULT)
    cdna = result["cdna_result"]
    assert cdna["observed"] != cdna["predicted_if_granule_1024"]
    assert cdna["observed"] == cdna["predicted_if_no_rounding"]
    assert 1024 in cdna["granules_ruled_out"]

    paths = {raw["arch"]: raw["path"] for raw in result["raw_receipts"]}
    capture = load(ROOT / paths["gfx942"])
    decisive = next(r for r in capture["rows"]
                    if r["footprint"] == cdna["decisive_footprint"])
    assert decisive["peak_resident_blocks"] == cdna["observed"]


def test_the_cdna_granule_is_reported_as_a_bound_not_a_measurement():
    """Every footprint swept is a multiple of 512, so nothing at or below that
    is distinguishable. Claiming a specific CDNA granule would overstate."""
    result = load(RESULT)
    cdna = result["cdna_result"]
    assert cdna["largest_granule_still_consistent"] == 512
    survivors = cdna["granules_still_consistent_with_every_row"]
    assert survivors == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    capture_paths = {raw["arch"]: raw["path"] for raw in result["raw_receipts"]}
    capture = load(ROOT / capture_paths["gfx942"])
    for row in capture["rows"]:
        assert row["footprint"] % 512 == 0, (
            "a footprint that is not a multiple of 512 would narrow the bound "
            "and this claim would need redoing"
        )


def test_lds_was_the_binding_limiter():
    check = load(RESULT)["limiter_check"]
    assert check["rdna_peak_waves_per_cu"] < check["rdna_wave_ceiling_per_cu"]
    assert check["cdna_peak_waves_per_cu"] < check["rdna_wave_ceiling_per_cu"]


def test_the_wave_ceiling_is_observed_rather_than_quoted():
    """It was a specification number until rocminfo was read on both parts.
    Losing that provenance would put the headroom argument back on recall."""
    source = load(RESULT)["limiter_check"]["ceiling_source"]
    assert source["tool"] == "rocminfo"
    ceiling = load(RESULT)["limiter_check"]["rdna_wave_ceiling_per_cu"]
    assert source["gfx1201"] == source["gfx942"] == ceiling


def test_the_multiprocessor_factor_of_two_is_shown_by_two_tools():
    """A HIP multiprocessor is a workgroup processor on RDNA and a compute unit
    on CDNA. That is the misreading that once cost this repository a factor of
    two, and it is checkable arithmetic rather than a remembered caveat."""
    trap = load(RESULT)["limiter_check"][
        "the_multiprocessor_trap_confirmed_by_a_second_tool"
    ]
    cu = trap["rocminfo_compute_unit"]
    mp = trap["hip_multi_processor_count"]
    per_cu = trap["rocminfo_max_work_item_per_cu"]
    per_mp = trap["hip_max_threads_per_multiprocessor"]
    for part in ("gfx1201", "gfx942"):
        # Total work-items must agree however the device is carved up.
        assert cu[part] * per_cu[part] == mp[part] * per_mp[part], part
    assert cu["gfx1201"] == 2 * mp["gfx1201"]
    assert cu["gfx942"] == mp["gfx942"]


def test_the_workgroup_cap_is_recorded_as_unavailable():
    """The wave ceiling can be looked up. The workgroup-count cap turned out to
    be reportable too, after I had said it was not, and the correction has to
    stay visible: the same header documents a property that returns zero on
    some paths, which is why the probe still measures it."""
    gap = load(RESULT)["limiter_check"]["what_no_tool_reports"]
    assert "workgroup" in gap["quantity"]
    assert "rocminfo" in gap["checked"]
    assert "maxBlocksPerMultiProcessor" in gap["correction"]
    assert "@bug" in gap["why_the_measurement_is_still_worth_making"]


def test_the_cdna_card_was_not_shared():
    """A tenant holding part of the card is the reading that would look like a
    refuted model. Every peak being an exact multiple of the processor count
    rules it out."""
    result = load(RESULT)
    paths = {raw["arch"]: raw["path"] for raw in result["raw_receipts"]}
    capture = load(ROOT / paths["gfx942"])
    count = capture["multi_processor_count"]
    for row in capture["rows"]:
        assert row["peak_resident_blocks"] % count == 0


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
