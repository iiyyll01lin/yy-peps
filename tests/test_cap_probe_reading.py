"""The cap probe must be read by the rule, not by whoever sees the numbers.

The verdict these tests pin was fixed in 183d27e before the probe was run: four
agreeing low footprints give the cap, a disagreement with what HIP reports is
the finding rather than an error, and a cap above six per compute unit leaves
the published separators standing.

Encoding it here means the reading cannot be adjusted after the data arrives.
The fixtures are synthetic on purpose; the point is the rule, not the numbers.
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import read_cap_probe as probe  # noqa: E402


def capture(arch, multiprocessors, peaks, reported=None):
    rows = [
        {"footprint": f, "grid": 4096, "peak_resident_blocks": p, "per_multiprocessor": p / multiprocessors}
        for f, p in peaks.items()
    ]
    out = {"gcn_arch": arch, "multi_processor_count": multiprocessors, "rows": rows}
    if reported is not None:
        out["max_blocks_per_multiprocessor"] = reported
    return out


def agreeing(arch, multiprocessors, peak, reported=None):
    return capture(arch, multiprocessors, {f: peak for f in probe.LOW_FOOTPRINTS}, reported)


def test_the_rdna_multiprocessor_is_halved_into_compute_units():
    # 32 per WGP is 16 per CU. Reading the WGP number as a CU number is the
    # mistake that already cost this repository a factor of two.
    result = probe.read_probe(agreeing("gfx1201", 32, 32 * 32))
    assert result["measured_cap_per_multiprocessor"] == 32
    assert result["measured_cap_per_compute_unit"] == 16


def test_the_cdna_multiprocessor_is_a_compute_unit():
    result = probe.read_probe(agreeing("gfx942:sramecc+:xnack-", 152, 32 * 152))
    assert result["measured_cap_per_multiprocessor"] == 32
    assert result["measured_cap_per_compute_unit"] == 32


def test_disagreeing_footprints_establish_nothing():
    peaks = dict(zip(probe.LOW_FOOTPRINTS, (448, 448, 384, 320)))
    result = probe.read_probe(capture("gfx1201", 32, peaks))
    assert result["cap_established"] is False
    assert result["footprints_agree"] is False
    # The reading must not average them into a number that looks measured.
    assert "measured_cap_per_multiprocessor" not in result


def test_a_disagreement_with_hip_is_reported_as_the_finding():
    result = probe.read_probe(agreeing("gfx942", 152, 16 * 152, reported=32))
    assert result["reported_matches_measured"] is False
    text = " ".join(probe.describe(result))
    assert "HIP reports 32" in text and "measurement gives 16" in text
    assert "the finding" in text


def test_agreement_with_hip_is_reported_as_confirmation():
    result = probe.read_probe(agreeing("gfx942", 152, 32 * 152, reported=32))
    assert result["reported_matches_measured"] is True
    assert "confirms" in " ".join(probe.describe(result))


def test_a_cap_above_six_leaves_the_separators_standing():
    result = probe.read_probe(agreeing("gfx942", 152, 7 * 152))
    assert result["separators_survive"] is True


def test_a_cap_at_six_reopens_every_separator_result():
    # Six is the residency the separators themselves reach, so a cap there is
    # indistinguishable from the granule and cannot be waved through.
    result = probe.read_probe(agreeing("gfx942", 152, 6 * 152))
    assert result["separators_survive"] is False
    assert "reopened" in " ".join(probe.describe(result))


def test_an_old_capture_is_refused_rather_than_guessed_at():
    old = capture("gfx1201", 32, {8704: 448, 9216: 448})
    result = probe.read_probe(old)
    assert result["readable"] is False
    assert "predates" in result["why"]


def test_exit_status_separates_a_threat_from_an_unreadable_capture(tmp_path):
    safe = tmp_path / "safe.json"
    safe.write_text(json.dumps(agreeing("gfx942", 152, 32 * 152, reported=32)), encoding="utf-8")
    assert probe.main([str(safe)]) == 0

    threat = tmp_path / "threat.json"
    threat.write_text(json.dumps(agreeing("gfx942", 152, 5 * 152)), encoding="utf-8")
    assert probe.main([str(threat)]) == 1

    old = tmp_path / "old.json"
    old.write_text(json.dumps(capture("gfx1201", 32, {8704: 448})), encoding="utf-8")
    assert probe.main([str(old)]) == 2

    assert probe.main([str(tmp_path / "absent.json")]) == 2


def test_the_real_captures_are_still_the_pre_probe_ones(tmp_path):
    # Guards the claim in the receipt that the cap probe has not been run yet.
    # When it is, this test is the one that should be updated deliberately.
    for name in ("gfx1151", "gfx1201", "gfx942"):
        path = ROOT / "results" / "hip_occupancy_census" / f"{name}.json"
        result = probe.read_probe(json.loads(path.read_text(encoding="utf-8")))
        assert result["readable"] is False
