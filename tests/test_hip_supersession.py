"""Guards for the supersession markers on the HIP receipts.

docs/05_amd_hardware.md has always said that the first latency receipt's
ordering is an artefact of measurement order. The receipt itself said nothing,
so anything reading results/ rather than docs/ saw a 42.30 ms median presented
with status "passed". W14 grades students on retracting in place rather than
deleting; these tests hold the repository to the same rule.

The markers are checked against the numbers they describe, so a marker cannot
drift into claiming a correction that did not happen.
"""

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARTEFACT = RESULTS / "hip_benchmark_gfx1201.json"
SETTLED = RESULTS / "hip_stable_latency.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_artefact_receipt_says_it_is_superseded():
    marker = load(ARTEFACT)["superseded_by"]
    target = ROOT / marker["receipt"]
    assert target.exists(), f"{marker['receipt']} does not exist"
    assert target.resolve() == SETTLED.resolve()


def test_the_recorded_overstatement_matches_both_receipts():
    # A marker that merely asserts "superseded" is a label. This one carries
    # numbers, so it can be checked against the receipts it spans.
    marker = load(ARTEFACT)["superseded_by"]["overstatement_by_method"]
    here = {row["method"]: row["median_ms"] for row in load(ARTEFACT)["measurements"]}
    settled = {
        method: row["median_of_round_medians_ms"]
        for method, row in load(SETTLED)["summary"].items()
    }
    assert set(marker) == set(here) == set(settled)
    for method, claim in marker.items():
        assert claim["here_ms"] == pytest.approx(here[method], abs=1e-6)
        assert claim["settled_ms"] == pytest.approx(settled[method], abs=5e-3)
        assert claim["factor"] == pytest.approx(
            here[method] / settled[method], abs=0.01
        )


def test_the_overstatement_decays_with_measurement_order():
    # The claim is not that one number was wrong but that each method
    # inherited a warmer card than the one before it. That predicts a
    # monotone decay, and without it the diagnosis would be unsupported.
    order = ["bi-grid", "grid-peps-3f", "grid-pink-peps-3f", "grid-pink-peps-4f"]
    marker = load(ARTEFACT)["superseded_by"]["overstatement_by_method"]
    factors = [marker[method]["factor"] for method in order]
    assert factors == sorted(factors, reverse=True)
    assert factors[0] > 5.0 and factors[-1] < 3.0


def test_every_supersession_path_resolves():
    # A pointer to a file that has been renamed is worse than no pointer.
    keys = ("superseded_by", "supersedes", "later_work", "supersedes_receipt")
    checked = 0
    for path in sorted(RESULTS.glob("hip_*.json")):
        payload = load(path)
        if not isinstance(payload, dict):
            continue
        for key in keys:
            block = payload.get(key)
            if isinstance(block, dict) and "receipt" in block:
                assert (ROOT / block["receipt"]).exists(), (
                    f"{path.name}.{key} points at a missing file"
                )
                checked += 1
        # supersedes blocks keyed by path, as results/hip_lds_ab.json uses.
        block = payload.get("supersedes")
        if isinstance(block, dict) and "receipt" not in block:
            for target in block:
                assert (ROOT / target).exists(), (
                    f"{path.name}.supersedes names a missing file: {target}"
                )
                checked += 1
    assert checked >= 3


def test_the_csv_twin_is_marked_and_only_where_it_should_be():
    bad_binary = load(ARTEFACT)["build"]["binary_sha256"]
    with (RESULTS / "hip_latency.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    marked = [row for row in rows if row.get("superseded_by")]
    assert len(marked) == 4, "the four schema-3 rows from that build carry the numbers"
    for row in rows:
        expected = row["binary_sha256"] == bad_binary
        assert bool(row.get("superseded_by")) is expected
        if expected:
            assert row["superseded_by"] == "results/hip_stable_latency.json"
            assert row["schema_version"] == "3"


def test_measurement_records_match_their_own_schema():
    # The schema files were present but nothing validated against them, so
    # they could drift from the receipts indefinitely without anyone noticing.
    jsonschema = pytest.importorskip("jsonschema")
    schema = load(RESULTS / "hip_benchmark_receipt.schema.json")
    for record in load(ARTEFACT)["measurements"]:
        jsonschema.validate(record, schema)


def test_the_kernel_says_what_compute_units_actually_counts():
    # hip/README.md read compute_units as a CU count and halved the part,
    # which made every per-CU ratio in it wrong by two. The field is
    # hipDeviceProp_t multiProcessorCount, which counts WGPs on RDNA. The
    # name is kept for schema compatibility, so the receipt has to carry its
    # own explanation or the same misreading is available to the next reader.
    source = (ROOT / "hip" / "fused_peps_kernel.hip").read_text(encoding="utf-8")
    assert "compute_units_semantics" in source
    assert "multiProcessorCount" in source
    assert "each WGP is two compute units" in source

    schema = load(RESULTS / "hip_benchmark_receipt.schema.json")
    # additionalProperties is false, so an unlisted field would make every
    # fresh receipt fail its own schema.
    assert schema.get("additionalProperties") is False
    assert "compute_units_semantics" in schema["properties"]
    # Optional, so the receipts written before this stays valid.
    assert "compute_units_semantics" not in schema["required"]
    assert "compute_units" in schema["required"]
