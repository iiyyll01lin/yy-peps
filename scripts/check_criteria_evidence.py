"""Check that every success criterion in course/labs.json points at evidence
that actually exists.

A criterion is a promise to the student: run this, look at that field, and you
will be able to tell whether you have met the bar. The promise is only worth
anything if the file is there and the field is in it -- a key in the JSON, or a
column in the CSV header. This repository has already had one case of a document
confidently describing an artefact that said something else, so the criteria are
checked against the artefacts rather than trusted.

The claims are read out of labs.json itself. Transcribing them into this file
would mean the check kept passing after someone edited a criterion, which is
the failure mode where a check quietly stops checking anything.
"""

from __future__ import annotations

import csv
import json
import pathlib
import re
import sys

# Directories that make a token a path into this repository rather than prose.
REPO_DIR = r"(?:results|docs|tests|hip|course|notebooks|scripts|data|env|slides)"
PATH = re.compile(rf"\b{REPO_DIR}/[A-Za-z0-9_.*/-]+")

# "fields a, b and c" / "field answer"
FIELD_CLAUSE = re.compile(r"\bfields?\b")

# English glue inside a field clause. Anything else is taken to be a field name.
STOPWORDS = {
    "and", "or", "the", "a", "an", "plus", "its", "this", "that", "these",
    "flags", "values", "entries", "section", "fields", "field", "in", "of",
    "with", "for", "any", "all", "each", "both",
}


def paths_in(text: str) -> list[str]:
    out = []
    for hit in PATH.findall(text):
        out.append(hit.rstrip(".,;"))
    return out


def fields_in(clause: str) -> list[str]:
    out = []
    for token in re.split(r"[\s,;]+", clause):
        token = token.strip().rstrip(".,;")
        if not token or token.lower() in STOPWORDS:
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", token):
            continue
        out.append(token)
    return out


def resolves(doc, dotted: str) -> bool:
    """True if the dotted key path resolves anywhere in the document.

    Receipts here nest under run names and under lists of per-case rows, and a
    criterion that says "field answer" means "the receipt records an answer",
    not "answer is at the top level". So the search descends.
    """
    head, _, rest = dotted.partition(".")

    def walk(node, key):
        found = []
        if isinstance(node, dict):
            if key in node:
                found.append(node[key])
            for value in node.values():
                found.extend(walk(value, key))
        elif isinstance(node, list):
            for value in node:
                found.extend(walk(value, key))
        return found

    hits = walk(doc, head)
    if not rest:
        return bool(hits)
    return any(resolves(hit, rest) for hit in hits)


def check(root: pathlib.Path) -> tuple[list[str], dict[str, int]]:
    problems: list[str] = []
    counts = {"paths": 0, "fields": 0}

    labs = json.loads((root / "course" / "labs.json").read_text(encoding="utf-8"))
    if isinstance(labs, dict):
        for key in ("labs", "weeks", "items"):
            if key in labs:
                labs = labs[key]
                break

    for lab in labs:
        week = lab.get("id", "?")
        for crit in lab.get("success_criteria", []):
            name = f"{week} {crit.get('id')}"
            evidence = crit.get("evidence", "")

            split = FIELD_CLAUSE.search(evidence)
            head = evidence[: split.start()] if split else evidence
            tail = evidence[split.end():] if split else ""

            named = paths_in(head) + paths_in(tail)
            for rel in named:
                counts["paths"] += 1
                if "*" in rel:
                    if not list(root.glob(rel)):
                        problems.append(f"{name}: no file matches {rel}")
                elif not (root / rel).exists():
                    problems.append(f"{name}: names {rel}, which does not exist")

            if not split:
                continue

            targets = [
                p for p in paths_in(head) if p.endswith((".json", ".csv"))
            ]
            if not targets:
                continue
            target = targets[-1]
            if not (root / target).exists():
                continue

            if target.endswith(".csv"):
                with (root / target).open(newline="", encoding="utf-8") as handle:
                    header = next(csv.reader(handle), [])
                for field in fields_in(tail):
                    counts["fields"] += 1
                    if field not in header:
                        problems.append(
                            f"{name}: promises column {field!r} in {target}, "
                            f"which is not there"
                        )
                continue

            try:
                doc = json.loads((root / target).read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                problems.append(f"{name}: {target} is not readable JSON ({exc})")
                continue
            for field in fields_in(tail):
                counts["fields"] += 1
                if not resolves(doc, field):
                    problems.append(
                        f"{name}: promises field {field!r} in {target}, "
                        f"which is not there"
                    )

    return problems, counts


def main(root_arg: str | None = None) -> int:
    root = pathlib.Path(root_arg or (sys.argv[1] if len(sys.argv) > 1 else "."))
    problems, counts = check(root)

    if counts["paths"] + counts["fields"] == 0:
        print(
            "no criterion named a file or a field. Either labs.json lost its "
            "evidence lines or the parser stopped matching them; both leave "
            "this check passing without checking anything.",
        )
        return 1

    for problem in problems:
        print(problem)

    if problems:
        print(
            f"\n{len(problems)} criterion promise(s) not backed "
            f"({counts['paths']} files and {counts['fields']} fields checked)"
        )
        return 1

    print(
        f"criteria evidence verified: {counts['paths']} files and "
        f"{counts['fields']} fields promised, all present"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
