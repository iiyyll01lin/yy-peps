"""Machine-readable HIP benchmark artifact consistency checks."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gfx1201_benchmark_receipt_and_csv_agree():
    receipt = json.loads(
        (ROOT / "results" / "hip_benchmark_gfx1201.json").read_text()
    )
    assert receipt["schema_version"] == 3
    assert receipt["status"] == "passed"
    assert receipt["build"]["code_object_targets"] == ["gfx1201"]
    assert receipt["hardware"]["detected_isa"] == "gfx1201"
    assert receipt["paper_comparison"]["directly_comparable"] is False
    assert receipt["rdna35_validation"]["status"] == "deferred"

    build = receipt["build"]
    binary_name = Path(build["binary"]).name
    assert build["git_sha"] in binary_name
    assert build["target_isa"] in binary_name
    assert len(build["binary_sha256"]) == len(build["source_sha256"]) == 64
    rendered_command = " ".join(build["command"]).lower()
    assert "password" not in rendered_command

    measurements = {row["method"]: row for row in receipt["measurements"]}
    assert set(measurements) == {
        "bi-grid",
        "grid-peps-3f",
        "grid-pink-peps-3f",
        "grid-pink-peps-4f",
    }
    assert {
        method: row["selected_feature_dim"]
        for method, row in measurements.items()
    } == {
        "bi-grid": 16,
        "grid-peps-3f": 112,
        "grid-pink-peps-3f": 44,
        "grid-pink-peps-4f": 46,
    }
    assert all(row["timing"] == "hip_events" for row in measurements.values())
    assert all(row["iters"] == 100 and row["warmup"] == 30 for row in measurements.values())

    with (ROOT / "results" / "hip_latency.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    current = {
        row["method"]: row
        for row in rows
        if row["schema_version"] == "3"
        and row["source_sha256"] == build["source_sha256"]
    }
    assert set(current) == set(measurements)
    for method, measurement in measurements.items():
        row = current[method]
        assert float(row["median_ms"]) == measurement["median_ms"]
        assert float(row["p95_ms"]) == measurement["p95_ms"]
        assert row["binary_sha256"] == build["binary_sha256"]
        assert row["code_object_target"] == "gfx1201"
        assert row["parity_status"] == "passed"
        assert row["comparable_to_paper"] == "false"


def test_recorded_full_output_parity_meets_precision_contracts():
    receipt = json.loads(
        (ROOT / "results" / "hip_benchmark_gfx1201.json").read_text()
    )
    assert len(receipt["parity"]) == 10
    for record in receipt["parity"]:
        assert record["passed"] is True
        assert record["max_abs_error"] < record["tolerance"]
        if record["precision"] == "fp32":
            assert record["tolerance"] == 1e-3
        else:
            assert record["precision"] == "fp16"
            assert record["tolerance"] == 4e-3
