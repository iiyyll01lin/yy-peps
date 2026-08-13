#!/usr/bin/env python3
"""Fail if a superseded claim reappears outside the places that correct it.

Three claims in this repository were corrected after measurement disproved
them, and each took several passes to chase down because the search was redone
by hand every time. The latency medians were found in five places, the last
being a README that also told the reader to re-run the tool that produced them.
The compute-unit misreading was found at its source only after the prose had
been fixed. The per-CU occupancy vocabulary was found in the course itself.

Nothing prevented any of them coming back. This makes the check mechanical.

A claim registers the patterns that express it and the files allowed to contain
them: the receipt holding the original under a supersession marker, and the
documents that quote it in order to correct it. Anywhere else is a regression.

The patterns are anchored so a longer number is not a match. A PSNR of
42.30421821514322 is not the latency median 42.304207, and an unanchored
substring search says it is -- which is the kind of false alarm that gets a
check switched off.

Run from the repository root, or pass the root as argv[1].
"""
from __future__ import annotations

import pathlib
import re
import sys

SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "site-packages"}
# Historical job records: they describe what a past run emitted, and rewriting
# them would falsify that history, so they are out of scope by design.
SKIP_PREFIXES = ("results/work/",)
SUFFIXES = {".md", ".py", ".json", ".csv", ".ipynb", ".hip", ".sh", ".toml"}
SELF = "scripts/check_superseded_claims.py"


def number(value: str) -> str:
    """A decimal that is not part of a longer one."""
    return r"(?<![\d.])" + value.replace(".", r"\.") + r"(?![\d])"


class Claim:
    def __init__(self, name: str, patterns: list[str], allowed: set[str],
                 corrected_to: str) -> None:
        self.name = name
        self.regex = re.compile("|".join(patterns))
        self.allowed = allowed
        self.corrected_to = corrected_to


CLAIMS = [
    Claim(
        "artefact_latency_medians",
        [number(v) for v in (
            "42.304207", "42.304", "42.30",
            "48.200451", "48.200",
            "51.069656", "51.070",
            "53.787079", "53.787",
        )],
        {
            # The receipt holding the original under a supersession marker,
            # and its CSV twin.
            "results/hip_benchmark_gfx1201.json",
            "results/hip_latency.csv",
            # Prose that quotes the number in order to correct it.
            "hip/README.md",
            "docs/05_amd_hardware.md",
            # The index entry that says what superseded it.
            "results/manifest.json",
            # The guard that checks the recorded overstatement against both
            # receipts, which cannot do that without naming the values.
            "tests/test_hip_supersession.py",
            # This check's own tests, which plant one of these values in a
            # scratch tree to prove the check is capable of failing.
            "tests/test_superseded_claims.py",
        },
        "7.40 / 16.06 / 18.36 / 21.27 ms settled; results/hip_stable_latency.json",
    ),
    Claim(
        "wgp_count_read_as_compute_units",
        [r"32 compute units", r"20 compute units"],
        {"hip/README.md", "docs/05_amd_hardware.md"},
        "64 CU on gfx1201 and 40 on gfx1151; multiProcessorCount counts WGPs",
    ),
    Claim(
        "per_cu_occupancy_vocabulary",
        [r"workgroups_per_cu_by", r"workgroups per compute unit three ways"],
        {"results/hip_occupancy.json"},
        "workgroups per WGP after rounding to the 1024-byte allocation granule",
    ),
]


def tracked_files(root: pathlib.Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith(SKIP_PREFIXES) or relative == SELF:
            continue
        yield relative, path


def main(root_arg: str | None = None) -> int:
    argument = root_arg or (sys.argv[1] if len(sys.argv) > 1 else ".")
    root = pathlib.Path(argument).resolve()
    files = list(tracked_files(root))
    if not files:
        print("FAIL: scanned nothing, so the file filter is wrong")
        return 2

    found: dict[str, set[str]] = {claim.name: set() for claim in CLAIMS}
    for relative, path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for claim in CLAIMS:
            if claim.regex.search(text):
                found[claim.name].add(relative)

    failures = 0
    for claim in CLAIMS:
        hits = found[claim.name]
        stray = sorted(hits - claim.allowed)
        # A claim matching nothing means the patterns have rotted and the check
        # has quietly stopped checking, which is worse than a stray hit.
        if not hits:
            print(f"FAIL {claim.name}: matched nothing, so its patterns have rotted")
            failures += 1
        elif stray:
            print(f"FAIL {claim.name}: reappeared outside the documents that correct it")
            for name in stray:
                print(f"       {name}")
            print(f"     corrected to: {claim.corrected_to}")
            failures += 1
        else:
            print(f"ok   {claim.name}: {len(hits)} file(s), all expected")

    print(f"\nscanned {len(files)} files")
    if failures:
        print(f"{failures} superseded claim(s) leaked back")
        return 1
    print("no superseded claim appears outside the documents that correct it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
