"""Reproducible reporting — write tables/figures to ``results/``.

繁體中文:可重現報告工具。所有 notebook 透過本模組把 Table/Fig 寫成
``results/<name>.csv``(數字)與 ``results/<name>.png``(圖),讓 docs 引用的每個
dB/IoU 數字都有對應產出檔背書,而非寫死在文字裡。

設計原則:
- CSV 是「真相來源」:純文字、進 git、可 diff。
- PNG 為視覺化:被 .gitignore 忽略(可由 CSV 重新生成)。
- 不依賴 pandas(遠端 venv 未必有);只用 csv 標準庫 + 可選 matplotlib。
"""

from __future__ import annotations

import csv
import os
from typing import Iterable, Mapping, Sequence

# results/ lives at repo root, one level up from this file's package.
RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results")
)


def results_path(name: str) -> str:
    """Absolute path under ``results/`` for a given basename (dirs auto-created)."""
    p = os.path.join(RESULTS_DIR, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def write_table(name: str, rows: Sequence[Mapping[str, object]],
                columns: Sequence[str] | None = None) -> str:
    """Write a list-of-dicts as a CSV under ``results/``.

    Args:
        name: filename, e.g. ``"table1_image.csv"``.
        rows: sequence of dict rows.
        columns: explicit column order; inferred from the first row if omitted.
    Returns the written path.
    """
    if not rows:
        raise ValueError("write_table: rows is empty")
    if columns is None:
        columns = list(rows[0].keys())
    path = results_path(name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
    return path


def read_table(name: str) -> list[dict[str, str]]:
    """Read back a CSV written by :func:`write_table` (values as strings)."""
    path = results_path(name)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def markdown_table(rows: Sequence[Mapping[str, object]],
                   columns: Sequence[str] | None = None) -> str:
    """Render rows as a GitHub-flavored markdown table (for docs/notebooks)."""
    if not rows:
        return ""
    if columns is None:
        columns = list(rows[0].keys())
    cols = list(columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
        for r in rows
    ]
    return "\n".join([head, sep, *body])


def save_figure(name: str, fig=None) -> str:
    """Save a matplotlib figure under ``results/`` (PNG). No-op-safe if headless.

    Uses the Agg backend implicitly when called on a remote box without a display.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    path = results_path(name)
    (fig or plt.gcf()).savefig(path, dpi=120, bbox_inches="tight")
    return path


def plot_xy(name: str, series: Mapping[str, tuple[Iterable[float], Iterable[float]]],
            xlabel: str = "", ylabel: str = "", title: str = "",
            logx: bool = False, logy: bool = False) -> str:
    """Convenience: line plot of multiple named ``(xs, ys)`` series -> PNG.

    Used for params-vs-PSNR (Fig.5) and rate-distortion (W10) curves.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, (xs, ys) in series.items():
        ax.plot(list(xs), list(ys), marker="o", label=label)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = results_path(name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
