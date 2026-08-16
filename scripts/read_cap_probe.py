#!/usr/bin/env python3
"""Read the cap probe against the rule fixed before the run.

The census measures how many workgroups sit on a multiprocessor at once, and
two things can stop that number rising: the LDS pool, and a hard limit on how
many workgroups may be resident at all. Every granule conclusion in this
repository assumes the second one never bound. The four low footprints test
that assumption, because at 512 to 4096 bytes the pool would allow on the order
of 128 workgroups and therefore cannot be what the count is reporting.

The rule, fixed in 183d27e before any of this was run:

  * the four low footprints agreeing on one value makes that value the cap;
  * the value is then compared with what HIP reports, and a disagreement is
    the finding rather than an error, because the same header documents a
    neighbouring field as always returning zero;
  * the separators peak at six workgroups per compute unit, so any cap above
    six leaves every published separator result standing.

Exit status is 0 when the probe reads and the separators survive, 1 when a
measured cap is low enough to threaten them, and 2 when the capture cannot
answer the question at all.
"""

from __future__ import annotations

import json
import pathlib
import sys

# Small enough that the LDS pool cannot be the limiter at any granule in play.
LOW_FOOTPRINTS = (512, 1024, 2048, 4096)

# The largest residency any published separator relies on. A cap above this
# cannot have produced those readings.
SEPARATOR_PEAK_BLOCKS_PER_CU = 6


def compute_units_per_multiprocessor(arch: str) -> int:
    """RDNA's multiprocessor is a workgroup processor spanning two compute units."""
    family = arch.split(":")[0]
    return 2 if family.startswith(("gfx10", "gfx11", "gfx12")) else 1


def read_probe(capture: dict) -> dict:
    arch = capture["gcn_arch"]
    multiprocessors = capture["multi_processor_count"]
    per_mp_to_per_cu = compute_units_per_multiprocessor(arch)

    rows = {row["footprint"]: row for row in capture["rows"]}
    missing = [f for f in LOW_FOOTPRINTS if f not in rows]
    if missing:
        return {
            "arch": arch,
            "readable": False,
            "why": f"capture has no rows at {missing}, so it predates the cap probe",
        }

    peaks = {f: rows[f]["peak_resident_blocks"] for f in LOW_FOOTPRINTS}
    agreed = len(set(peaks.values())) == 1
    reported = capture.get("max_blocks_per_multiprocessor")

    result = {
        "arch": arch,
        "readable": True,
        "peaks": peaks,
        "footprints_agree": agreed,
        "reported_cap_per_multiprocessor": reported,
        "compute_units_per_multiprocessor": per_mp_to_per_cu,
    }

    if not agreed:
        # Disagreement means something other than a fixed cap is moving, so the
        # run does not establish one and must not be averaged into looking like
        # it does.
        result["cap_established"] = False
        result["why"] = "the four low footprints disagree, so no single cap is measured"
        return result

    peak = next(iter(peaks.values()))
    measured_per_mp = peak / multiprocessors
    measured_per_cu = measured_per_mp / per_mp_to_per_cu

    result["cap_established"] = True
    result["measured_cap_per_multiprocessor"] = measured_per_mp
    result["measured_cap_per_compute_unit"] = measured_per_cu
    result["separators_survive"] = measured_per_cu > SEPARATOR_PEAK_BLOCKS_PER_CU

    if reported is not None:
        result["reported_matches_measured"] = reported == measured_per_mp
    return result


def describe(result: dict) -> list[str]:
    arch = result["arch"]
    if not result["readable"]:
        return [f"{arch}: {result['why']}"]

    lines = [f"{arch}: peaks at the four low footprints are {result['peaks']}"]
    if not result["cap_established"]:
        lines.append(f"{arch}: {result['why']}")
        return lines

    per_mp = result["measured_cap_per_multiprocessor"]
    per_cu = result["measured_cap_per_compute_unit"]
    unit = "WGP" if result["compute_units_per_multiprocessor"] == 2 else "CU"
    lines.append(
        f"{arch}: measured cap {per_mp:g} per multiprocessor ({unit}), "
        f"{per_cu:g} per compute unit"
    )

    reported = result["reported_cap_per_multiprocessor"]
    if reported is None:
        lines.append(f"{arch}: HIP reported no cap in this capture, nothing to compare")
    elif result["reported_matches_measured"]:
        lines.append(f"{arch}: HIP reports {reported}, which the measurement confirms")
    else:
        lines.append(
            f"{arch}: HIP reports {reported} but the measurement gives {per_mp:g}. "
            "The rule fixed before the run treats this disagreement as the finding"
        )

    if result["separators_survive"]:
        lines.append(
            f"{arch}: separators need at most {SEPARATOR_PEAK_BLOCKS_PER_CU} per compute "
            f"unit and the cap allows {per_cu:g}, so they stand"
        )
    else:
        lines.append(
            f"{arch}: the cap allows {per_cu:g} per compute unit, at or below the "
            f"{SEPARATOR_PEAK_BLOCKS_PER_CU} the separators rely on. Every conclusion "
            "drawn from them has to be reopened"
        )
    return lines


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    unusable = False
    threatened = False
    for arg in argv:
        path = pathlib.Path(arg)
        if not path.exists():
            print(f"{arg}: missing")
            unusable = True
            continue
        result = read_probe(json.loads(path.read_text(encoding="utf-8")))
        for line in describe(result):
            print(line)
        if not result["readable"]:
            unusable = True
        elif result["cap_established"] and not result["separators_survive"]:
            threatened = True

    if threatened:
        return 1
    return 2 if unusable else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
