#!/usr/bin/env python3
"""Fail if a committed record asserts that a process is alive.

Liveness is true only at the instant it is checked. Git is permanent. A file
that records ``"process_alive": true`` and is then committed has frozen a
time-sensitive observation into a claim the repository will keep making long
after the process has gone, and nothing downstream will notice: the file that
prompted this check is read by no script and no test, so its assertion went
three weeks past the death of the processes it described, while the completed
result sat beside it saying the opposite.

The repository already had the answer. The image reproduction status writer
names its field ``recorded_state``, which reads as a record rather than a
report, and re-verifies each pid when the status is produced, emitting
``pid_not_present_or_unreadable`` when the process has gone. The texture
Table 2 writer did neither. This check makes the good pattern the enforced one.

A record may say a process was alive if it also says how that observation was
qualified: a past-tense field name, a re-verification result, a recorded
outcome, or a supersession marker. An unqualified live assertion is the defect.

Run from the repository root, or pass the root as argv[1].
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

# Historical job records describe what a past run emitted; rewriting them would
# falsify that history, so they are out of scope exactly as they are for the
# superseded-claim check.
SKIP_PREFIXES = ("results/work/",)

# Fields whose truthy value asserts, in the present tense, that a process runs.
LIVE_ASSERTIONS = {
    "process_alive": (True,),
    "alive": (True,),
    "state": ("running",),
    "effective_state": ("running",),
    "status": ("verified_alive",),
}

# Fields that qualify such an assertion as a record rather than a live report.
QUALIFIERS = frozenset(
    {
        "recorded_state",
        "recorded_at_utc",
        "outcome",
        "returncode",
        "superseded",
        "superseded_by",
        "liveness_resolution",
        "verified_at_utc",
    }
)


def tracked_json(root: pathlib.Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "results"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [
        line
        for line in out.splitlines()
        if line.endswith(".json") and not line.startswith(SKIP_PREFIXES)
    ]


def violations_in(node: object, qualified: bool, path: str) -> list[str]:
    """Report unqualified live assertions, inheriting qualification from ancestors."""
    found: list[str] = []
    if isinstance(node, dict):
        here = qualified or bool(QUALIFIERS & node.keys())
        for key, value in node.items():
            if key in LIVE_ASSERTIONS and value in LIVE_ASSERTIONS[key] and not here:
                found.append(f"{path}.{key} = {value!r}")
        for key, value in node.items():
            found += violations_in(value, here, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found += violations_in(value, qualified, f"{path}[{index}]")
    return found


def main(root_arg: str = ".") -> int:
    root = pathlib.Path(root_arg).resolve()
    failures: list[str] = []
    for rel in tracked_json(root):
        try:
            data = json.loads((root / rel).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{rel}: unreadable ({exc})")
            continue
        for hit in violations_in(data, False, rel):
            failures.append(f"unqualified live assertion: {hit}")

    if failures:
        print("committed records assert a live process without qualifying it:")
        for line in failures:
            print(f"  {line}")
        print()
        print("A record may say a process was alive if it also records how that")
        print("was qualified: a past-tense field name such as recorded_state, a")
        print("re-verification timestamp, an outcome, or a supersession marker.")
        return 1

    print("status liveness check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
