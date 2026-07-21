"""Tests for the reproducible-reporting helper (peps.report).

繁體中文:驗證 report 工具能寫出/讀回 CSV、產生 markdown 表、以及在無顯示環境
存 PNG(Agg backend)。這些是 Phase 0「每個數字都有產出檔」的地基。
"""

import os
import tempfile

from peps import report


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
