"""Tests for the reproducible-reporting helper (peps.report).

繁體中文:驗證 report 工具能寫出/讀回 CSV、產生 markdown 表、以及在無顯示環境
存 PNG(Agg backend)。這些是 Phase 0「每個數字都有產出檔」的地基。
"""

import csv
import hashlib
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import pytest

from peps import report
from peps.profiles import get_profile


def test_write_read_roundtrip(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(report, "RESULTS_DIR", d)
        rows = [
            {"method": "grid", "psnr": 37.7, "lsd": 0.82},
            {"method": "pink_peps", "psnr": 42.2, "lsd": 0.51},
        ]
        path = report.write_table("t.csv", rows)
        assert os.path.exists(path)
        back = report.read_table("t.csv")
        assert back[0]["method"] == "grid"
        assert back[1]["method"] == "pink_peps"
        # values come back as strings
        assert back[1]["psnr"] == "42.2"


def test_markdown_table():
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    md = report.markdown_table(rows)
    assert "| a | b |" in md
    assert "| 1 | 2 |" in md
    assert md.count("\n") == 3  # header + sep + 2 rows


def test_explicit_columns_order(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(report, "RESULTS_DIR", d)
        rows = [{"b": 2, "a": 1}]
        report.write_table("o.csv", rows, columns=["a", "b"])
        with open(os.path.join(d, "o.csv")) as f:
            header = f.readline().strip()
        assert header == "a,b"


def test_plot_xy_writes_png(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(report, "RESULTS_DIR", d)
        p = report.plot_xy(
            "fig.png",
            {"grid": ([1, 2, 3], [30, 32, 33]),
             "peps": ([1, 2, 3], [31, 35, 40])},
            xlabel="params", ylabel="PSNR",
        )
        assert os.path.exists(p)
        assert os.path.getsize(p) > 0


def _test_manifest(config=None):
    profile = get_profile("course_fast")
    return report.build_run_manifest(
        experiment="image.table_1",
        profile=profile,
        config=profile.image if config is None else config,
        seed=7,
        git_state={
            "available": True,
            "sha": "a" * 40,
            "branch": "test",
            "dirty": True,
            "tracked_changes": 2,
            "untracked_files": 1,
            "error": None,
        },
        dataset_hashes=[
            {
                "id": "kodim01",
                "path": "data/kodim01.png",
                "algorithm": "sha256",
                "digest": "b" * 64,
                "bytes": 3,
            }
        ],
        environment={
            "platform": {"python_version": "3.12"},
            "packages": {"torch": "2.test"},
            "pytorch": {"version": "2.test"},
            "rocm": {"torch_hip_version": "7.test"},
            "gpu": {"available": False, "count": 0, "devices": []},
            "collection_errors": [],
        },
        timestamp="2026-07-21T12:34:56+00:00",
        run_id="image-table1-course-fast-s7",
    )


def test_hash_dataset_files_is_stable_and_sorted(tmp_path):
    second = tmp_path / "second.bin"
    first = tmp_path / "first.bin"
    second.write_bytes(b"second")
    first.write_bytes(b"first")

    hashes = report.hash_dataset_files({"z": second, "a": first})
    assert [item["id"] for item in hashes] == ["a", "z"]
    assert hashes[0]["algorithm"] == "sha256"
    assert hashes[0]["digest"] == hashlib.sha256(b"first").hexdigest()
    assert hashes[0]["bytes"] == 5


def test_run_manifest_has_versioned_config_and_provenance_schema():
    manifest = _test_manifest()
    assert manifest["schema"] == "peps.run_manifest"
    assert manifest["schema_version"] == 1
    assert manifest["profile"] == "course_fast"
    assert manifest["created_at_utc"] == "2026-07-21T12:34:56Z"
    assert manifest["provenance"]["git"]["dirty"] is True
    assert manifest["provenance"]["datasets"][0]["digest"] == "b" * 64
    assert manifest["provenance"]["environment"]["rocm"]["torch_hip_version"] == "7.test"
    assert manifest["instances"]["columns"] == list(report.INSTANCE_COLUMNS)
    assert len(manifest["config_sha256"]) == 64
    report.validate_run_manifest(manifest)

    manifest["config"]["training"]["optimizer_steps"] = 1
    with pytest.raises(ValueError, match="config_sha256"):
        report.validate_run_manifest(manifest)


def test_collect_run_manifest_hashes_data_and_uses_collectors(monkeypatch, tmp_path):
    data = tmp_path / "sample.bin"
    data.write_bytes(b"dataset")
    monkeypatch.setattr(
        report,
        "collect_git_state",
        lambda repo_root=None: {
            "available": True,
            "sha": "c" * 40,
            "branch": "test",
            "dirty": False,
            "tracked_changes": 0,
            "untracked_files": 0,
            "error": None,
        },
    )
    monkeypatch.setattr(
        report,
        "collect_environment",
        lambda package_names=(): {
            "platform": {"python_version": "test"},
            "packages": {"torch": "test"},
            "pytorch": {"version": "test"},
            "rocm": {"torch_hip_version": None},
            "gpu": {"available": False, "count": 0, "devices": []},
            "collection_errors": [],
        },
    )
    profile = get_profile("course_fast")
    manifest = report.collect_run_manifest(
        experiment="sdf.table_3",
        profile=profile,
        config=profile.sdf,
        seed=0,
        dataset_files={"torus": data},
        timestamp="2026-07-21T00:00:00Z",
        run_id="sdf-course-fast-s0",
    )
    dataset = manifest["provenance"]["datasets"][0]
    assert dataset["digest"] == hashlib.sha256(b"dataset").hexdigest()
    assert manifest["provenance"]["git"]["dirty"] is False


def test_collect_environment_records_pytorch_rocm_and_gpu_without_gpu_work(
    monkeypatch,
):
    properties = SimpleNamespace(
        name="Fake Radeon",
        gcnArchName="gfx-test",
        total_memory=1024,
        multi_processor_count=8,
    )
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda index: properties,
        get_device_capability=lambda index: (12, 0),
    )
    fake_torch = SimpleNamespace(
        __version__="2.test",
        version=SimpleNamespace(cuda=None, hip="7.test"),
        cuda=fake_cuda,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        report,
        "_distribution_version",
        lambda name: {"torch": "2.dist", "numpy": "1.test"}.get(name),
    )
    monkeypatch.setattr(report, "_rocm_runtime_version", lambda: "7.runtime")

    environment = report.collect_environment(("torch", "numpy"))
    assert environment["packages"] == {"numpy": "1.test", "torch": "2.dist"}
    assert environment["pytorch"]["version"] == "2.test"
    assert environment["rocm"] == {
        "runtime_version": "7.runtime",
        "torch_hip_version": "7.test",
        "rocm_home": None,
    }
    assert environment["gpu"]["devices"][0]["architecture"] == "gfx-test"
    assert environment["gpu"]["devices"][0]["total_memory_bytes"] == 1024
    assert environment["collection_errors"] == []


def test_collect_git_state_records_sha_and_dirty_counts(monkeypatch, tmp_path):
    def fake_git_command(repo_root, *args):
        if args[:2] == ("rev-parse", "HEAD"):
            return SimpleNamespace(returncode=0, stdout="d" * 40 + "\n", stderr="")
        if args[0] == "status":
            return SimpleNamespace(
                returncode=0,
                stdout=" M peps/report.py\n?? tests/test_profiles.py\n",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="feature/profiles\n", stderr="")

    monkeypatch.setattr(report, "_git_command", fake_git_command)
    state = report.collect_git_state(tmp_path)
    assert tuple(state) == report.GIT_STATE_FIELDS
    assert state["sha"] == "d" * 40
    assert state["branch"] == "feature/profiles"
    assert state["dirty"] is True
    assert state["tracked_changes"] == 1
    assert state["untracked_files"] == 1


def test_write_run_uses_stable_tidy_instance_schema(tmp_path):
    manifest = _test_manifest()
    rows = [
        report.InstanceRow(
            instance_id="kodim01",
            method="g_peps",
            metric="psnr",
            value=47.25,
            unit="dB",
            duration_seconds=1.5,
            metadata={"image_size": [512, 768]},
        ),
        {
            "instance_id": "kodim01",
            "method": "g_peps",
            "metric": "ssim",
            "value": 0.993,
        },
    ]
    artifacts = report.write_run(manifest, rows, output_dir=tmp_path)

    with open(artifacts.manifest_path, encoding="utf-8") as handle:
        written_manifest = json.load(handle)
    assert written_manifest["instances"]["row_count"] == 2
    assert written_manifest["config_sha256"] == manifest["config_sha256"]

    with open(artifacts.instances_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        written_rows = list(reader)
        assert reader.fieldnames == list(report.INSTANCE_COLUMNS)
    assert written_rows[0]["run_id"] == manifest["run_id"]
    assert written_rows[0]["seed"] == "7"
    assert written_rows[0]["value"] == "47.25"
    assert json.loads(written_rows[0]["metadata_json"]) == {
        "image_size": [512, 768]
    }
    assert written_rows[1]["split"] == "test"

    with pytest.raises(FileExistsError):
        report.write_run(manifest, rows, output_dir=tmp_path)


def test_instance_rows_reject_ad_hoc_columns(tmp_path):
    manifest = _test_manifest()
    with pytest.raises(ValueError, match="unknown instance row fields"):
        report.write_run(
            manifest,
            [
                {
                    "instance_id": "kodim01",
                    "method": "grid",
                    "metric": "psnr",
                    "value": 40.0,
                    "custom": "put this in metadata instead",
                }
            ],
            output_dir=tmp_path,
        )
