"""Check that the numbers the README argues from still come out of the artefacts.

The README leads with figures -- 594 of 594 jobs, a 1.154 dB shortfall, a 19.4 dB
category spread, bi-grid at 2.95 ms, a 43.75 percent ceiling. A reader takes
those as read. Nothing made them stay true if a receipt were regenerated.

The dependency runs one way on purpose. Each claim below records where its number
comes from and how it is rounded, and the expected text is computed from the
artefact every run. Writing the expected values in here instead would be a copy
that keeps passing after the measurement moves, which is the same trap as a
regex that has stopped matching: the check would be reporting on itself.

So an artefact edit fails this until the prose is updated, and a prose edit fails
it until the artefact agrees. Neither can drift quietly away from the other.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys


class Claim:
    def __init__(self, name, source, compute, render, documents):
        self.name = name
        self.source = source
        self.compute = compute
        self.render = render
        self.documents = documents


def dig(doc, dotted: str):
    """Resolve a dotted path, raising KeyError with the path that failed."""
    node = doc
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


TABLE2 = "results/texture_repro/table2.json"
SHORTFALL = "results/texture_repro/shortfall_analysis/receipt.json"
COMPOSITION = "results/texture_repro/shortfall_analysis/implied_composition.json"
CAPS = "results/hip_specialised_caps.json"

CLAIMS = [
    Claim(
        "table 2 job count",
        TABLE2,
        lambda d: dig(d, "progress.completed_jobs"),
        lambda v: f"{v:d}",
        ["README.md"],
    ),
    Claim(
        "table 2 expected job count",
        TABLE2,
        lambda d: dig(d, "progress.expected_jobs"),
        lambda v: f"{v:d}",
        ["README.md"],
    ),
    # Not stored anywhere: the README's shortfall is the difference of the two
    # means, so it has to be recomputed rather than looked up.
    Claim(
        "mean shortfall against the published table",
        SHORTFALL,
        lambda d: dig(d, "method_means.paper") - dig(d, "method_means.ours"),
        lambda v: f"{v:.3f}",
        ["README.md"],
    ),
    Claim(
        "map category spread",
        SHORTFALL,
        lambda d: dig(d, "composition.category_spread_db"),
        lambda v: f"{v:.1f}",
        ["README.md"],
    ),
    Claim(
        "share of maps in the two lowest categories",
        SHORTFALL,
        lambda d: dig(d, "composition.share_in_two_lowest_categories") * 100,
        lambda v: f"{v:.0f}",
        ["README.md"],
    ),
    Claim(
        "held-out improvement from reweighting",
        COMPOSITION,
        lambda d: dig(d, "fit_quality.held_out_improvement_factor"),
        lambda v: f"{v:.1f}",
        ["README.md"],
    ),
    Claim(
        "bi-grid specialised latency",
        CAPS,
        lambda d: dig(d, "latency.methods.bi-grid.specialised"),
        lambda v: f"{v:.2f}",
        ["README.md"],
    ),
    Claim(
        "bi-grid paper reference latency",
        CAPS,
        lambda d: dig(d, "latency.methods.bi-grid.paper"),
        lambda v: f"{v:.2f}",
        ["README.md"],
    ),
    Claim(
        "occupancy ceiling",
        CAPS,
        lambda d: dig(d, "ceiling_of_this_technique.ceiling.occupancy_fraction")
        * 100,
        lambda v: f"{v:.2f}",
        ["README.md"],
    ),
    Claim(
        "measured waves per compute unit at the ceiling",
        CAPS,
        lambda d: dig(
            d, "ceiling_of_this_technique.ceiling.measured_waves_per_cu"
        ),
        lambda v: f"{v:.2f}",
        ["README.md"],
    ),
    Claim(
        "fixed tile cost",
        CAPS,
        lambda d: dig(d, "ceiling_of_this_technique.fixed_tiles_bytes"),
        lambda v: f"{v:d}",
        ["README.md"],
    ),
]


def anchored(value: str) -> re.Pattern:
    """A longer number must not satisfy a shorter claim: 1.15 is not 1.1538."""
    return re.compile(r"(?<![\d.])" + re.escape(value) + r"(?![\d])")


def check(root: pathlib.Path) -> tuple[list[str], int]:
    problems: list[str] = []
    checked = 0
    cache: dict[str, object] = {}

    for claim in CLAIMS:
        path = root / claim.source
        if not path.exists():
            problems.append(f"{claim.name}: {claim.source} does not exist")
            continue
        if claim.source not in cache:
            cache[claim.source] = json.loads(path.read_text(encoding="utf-8"))
        try:
            value = claim.compute(cache[claim.source])
        except KeyError as exc:
            problems.append(
                f"{claim.name}: {claim.source} no longer has {exc.args[0]}"
            )
            continue

        expected = claim.render(value)
        for document in claim.documents:
            checked += 1
            target = root / document
            if not target.exists():
                problems.append(f"{claim.name}: {document} does not exist")
                continue
            if not anchored(expected).search(target.read_text(encoding="utf-8")):
                problems.append(
                    f"{claim.name}: {claim.source} gives {expected}, "
                    f"which does not appear in {document}"
                )

    return problems, checked


def main(root_arg: str | None = None) -> int:
    root = pathlib.Path(root_arg or (sys.argv[1] if len(sys.argv) > 1 else "."))
    problems, checked = check(root)

    if checked == 0:
        print(
            "no headline number was checked. The claim list is empty or every "
            "source is missing, either of which leaves this passing without "
            "comparing anything."
        )
        return 1

    for problem in problems:
        print(problem)

    if problems:
        print(f"\n{len(problems)} headline number(s) adrift, {checked} checked")
        return 1

    print(f"headline numbers verified: {checked} still come out of the artefacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
