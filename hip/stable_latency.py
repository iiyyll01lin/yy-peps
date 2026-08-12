#!/usr/bin/env python3
"""Latency measurement that survives a cold GPU.

The previous protocol ran every iteration of method 1, then every iteration of
method 2, and so on, from an idle card. The card spends the first seconds
ramping clocks, so whichever method is measured first absorbs the ramp: in
results/hip_benchmark_gfx1201.json the first method has a min/median ratio of
5.8 and a standard deviation of 16.9 ms, while the last two methods are stable
to 0.07 and 0.50 ms. The ordering that receipt reports is therefore an artefact
of measurement order.

Two changes fix it. Spin the GPU until the shader clock stops rising, then
interleave the methods round by round so any residual drift is shared equally
instead of being charged to whoever went first.

Usage:
    python hip/stable_latency.py --binary hip/build/<binary> --rounds 8
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import subprocess
import time
from pathlib import Path

METHODS = ("bi-grid", "grid-peps-3f", "grid-pink-peps-3f", "grid-pink-peps-4f")
PAPER_MS = {"bi-grid": 4.32, "grid-peps-3f": 5.47,
            "grid-pink-peps-3f": 4.86, "grid-pink-peps-4f": 4.99}
NAMED = [(name, ["benchmark", name]) for name in METHODS]


def parse_geometry(spec: str) -> list[tuple[str, list[str]]]:
    """Turn `peps:17:3,pink:17:4` into labelled geometry invocations."""
    entries = []
    for item in spec.split(","):
        mode, channels, frequencies = item.strip().split(":")
        label = f"geometry-{mode}-c{channels}-{frequencies}f"
        entries.append((label, ["geometry", mode, channels, frequencies]))
    return entries


def sclk_mhz(device: int) -> float | None:
    """Current shader clock, the thing that betrays a cold card."""
    out = subprocess.run(["rocm-smi", "-c"], capture_output=True, text=True).stdout
    current = None
    for line in out.splitlines():
        if f"GPU[{device}]" not in line or "sclk" not in line.lower():
            continue
        match = re.search(r"\((\d+)Mhz\)", line)
        if match:
            current = float(match.group(1))
    return current


def run_once(binary: Path, invocation: list[str], side: int, warmup: int,
             iters: int, device: int) -> dict:
    proc = subprocess.run(
        [str(binary), *invocation, str(side), str(warmup), str(iters)],
        capture_output=True, text=True, timeout=900,
        env={"ROCR_VISIBLE_DEVICES": str(device), "PATH": "/usr/bin:/bin:/opt/rocm/bin"},
    )
    if proc.returncode != 0:
        raise SystemExit(f"{' '.join(invocation)} failed: {proc.stderr[-400:]}")
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise SystemExit(f"{' '.join(invocation)} produced no JSON record")


def settle(binary: Path, methods: list[tuple[str, list[str]]], side: int,
           device: int, limit: int = 12) -> dict:
    """Spin until the shader clock stops climbing."""
    history = []
    for attempt in range(limit):
        run_once(binary, methods[0][1], side, 5, 20, device)
        clock = sclk_mhz(device)
        history.append(clock)
        if len(history) >= 3 and all(c is not None for c in history[-3:]):
            recent = history[-3:]
            if max(recent) - min(recent) <= 0.02 * max(recent):
                return {"settled": True, "rounds_to_settle": attempt + 1,
                        "sclk_history_mhz": history}
        time.sleep(0.5)
    return {"settled": False, "rounds_to_settle": limit, "sclk_history_mhz": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--side", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--geometry",
        help="comma-separated mode:channels:frequencies, e.g. peps:17:3,pink:17:4; "
             "defaults to the four named paper methods",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    methods = parse_geometry(args.geometry) if args.geometry else NAMED
    labels = [label for label, _ in methods]

    print(f"idle sclk: {sclk_mhz(args.device)} MHz")
    warm = settle(args.binary, methods, args.side, args.device)
    print(f"settle: {warm['settled']} after {warm['rounds_to_settle']} spins, "
          f"sclk {warm['sclk_history_mhz']}")

    rounds: list[dict] = []
    for index in range(args.rounds):
        # Rotating the start position keeps any residual drift from always
        # landing on the same method.
        order = methods[index % len(methods):] + methods[:index % len(methods)]
        for label, invocation in order:
            record = run_once(args.binary, invocation, args.side,
                              args.warmup, args.iters, args.device)
            rounds.append({
                "round": index, "method": label,
                "median_ms": record["median_ms"], "min_ms": record["min_ms"],
                "stddev_ms": record["stddev_ms"],
                "selected_feature_dim": record.get("selected_feature_dim"),
                "sclk_mhz": sclk_mhz(args.device),
            })
        print(f"  round {index}: " + "  ".join(
            f"{r['method'].split('-')[-1]}={r['median_ms']:.2f}"
            for r in rounds[-len(methods):]))

    summary = {}
    for label in labels:
        medians = [r["median_ms"] for r in rounds if r["method"] == label]
        mins = [r["min_ms"] for r in rounds if r["method"] == label]
        paper = PAPER_MS.get(label)
        summary[label] = {
            "median_of_round_medians_ms": st.median(medians),
            "best_round_median_ms": min(medians),
            "worst_round_median_ms": max(medians),
            "round_spread_ratio": max(medians) / min(medians),
            "min_observed_ms": min(mins),
            "paper_reference_ms": paper,
            "ratio_to_paper": st.median(medians) / paper if paper else None,
        }

    print(f"\n{'method':<22}{'median':>9}{'spread':>9}{'paper':>8}{'ratio':>8}")
    for method, s in summary.items():
        paper = s["paper_reference_ms"]
        ratio = s["ratio_to_paper"]
        print(f"{method:<22}{s['median_of_round_medians_ms']:>9.2f}"
              f"{s['round_spread_ratio']:>9.2f}x"
              f"{format(paper, '>7.2f') if paper else '     --'}"
              f"{format(ratio, '>7.1f') + 'x' if ratio else '      --'}")

    worst = max(s["round_spread_ratio"] for s in summary.values())
    print(f"\nworst round-to-round spread: {worst:.2f}x "
          f"({'stable' if worst < 1.10 else 'STILL UNSTABLE'})")

    payload = {
        "schema": "peps.stable_latency",
        "schema_version": 1,
        "protocol": {
            "side": args.side, "rounds": args.rounds,
            "warmup_per_round": args.warmup, "iters_per_round": args.iters,
            "method_order": "rotated each round",
            "methods": labels,
            "geometry": args.geometry,
            "clock_settling": warm,
            "note": ("Interleaved rounds with a rotating start, after spinning "
                     "the card until the shader clock stopped climbing."),
        },
        "rounds": rounds,
        "summary": summary,
        "stable": worst < 1.10,
        "worst_round_spread_ratio": worst,
    }
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
