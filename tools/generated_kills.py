#!/usr/bin/env python
r"""The record of which drawn case kills each historical entry of the main line.

``python tools/generated_kills.py write OUT [OUT ...]`` reads the
``kills.json`` of runs of ``tools/mutant_fuzz.py`` (and the saved case of
each kill, ``OUT/kill_cases/<label>.json``) and writes
``docs/verification/generated_kills.json``: for every entry of
``mutant_fuzz.TARGETS``, the first counted kill in the order the runs are
given, with the case's seed, mode, route and integrator, the rule that
counted it, the oracle's value, the mutant's value and the tolerance at the
worst compared row, the unmutated tree's value there (a value kill) or its
refusal (a refusal kill), the case's features, ``t_min`` and ``t_final``,
and the seeds the runs tried.  Each entry carries
the digest of its corpus entry (``kill_matrix.entry_digest``), so a changed
entry makes the record stale, and the record carries the digest of the
oracle's code (``tools/mutant_oracle.py``).  ``python
tools/generated_kills.py check`` lists the problems of the record against
the corpus and the oracle.

The record holds no local path.  This process imports no sft_wick code.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import sys
from collections.abc import Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RECORD = ROOT / "docs" / "verification" / "generated_kills.json"
#: The longest refusal text the record keeps.
MAX_ERROR = 300
RULE = (
    "value: the unmutated tree passes against the oracle and the mutant "
    "fails; refusal: the unmutated tree refuses with the text of the "
    "refusal the entry removes and the mutant returns a value the oracle "
    "rejects; a forced case counts only when its admissible control passes "
    "under both trees")
_LOCAL = re.compile(r"(/Users/|/home/|/private/|/tmp/|[A-Za-z]:\\)")


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tools():
    return _load("mutant_fuzz"), _load("kill_matrix")


def current_oracle_sha() -> str:
    """The digest of the oracle's code now (``tools/mutant_oracle.py``)."""
    return _load("mutant_oracle").oracle_sha()


def target_entries() -> list:
    """The corpus entries of ``mutant_fuzz.TARGETS``, in that order."""
    mf, _ = _tools()
    corpus = {m.label: m for m in mf.load_corpus()}
    return [corpus[label] for label in mf.TARGETS]


def _error_text(error: str | None) -> str | None:
    if not error:
        return None
    first = error.strip().splitlines()[0]
    return first if len(first) <= MAX_ERROR else first[:MAX_ERROR] + " ..."


def _kill_of(entry: dict, row: dict, facts: dict) -> dict:
    from validation.compare import tolerance

    mutant, head = row.get("mutant_row") or {}, row.get("head_row") or {}
    tol = tolerance(row["method"], "oracle")
    return dict(
        seed=row["seed"], mode=row["mode"],
        route=row.get("route") or entry["target"]["route"],
        method=row["method"], rule=row["reason"], tag=mutant.get("tag"),
        oracle=mutant.get("oracle"), mutant=mutant.get("value"),
        mutant_error_estimate=mutant.get("error"),
        allowed=mutant.get("allowed"), ratio=mutant.get("ratio"),
        head=head.get("value"), head_allowed=head.get("allowed"),
        head_error=_error_text(row.get("head_error")),
        control=row.get("control"), **facts,
        tolerance=dict(rel=tol.rel, k_err=tol.k_err, floor=tol.floor))


def _facts(out: pathlib.Path, label: str, seed: int, mode: str) -> dict:
    """The features of the saved case (``validation.fuzz.features``), its
    ``t_min`` and ``t_final``."""
    mf, _ = _tools()
    from validation import fuzz

    path = out / "kill_cases" / f"{label}.json"
    if path.exists():
        _label, items = mf.items_from_file(path)
        for item in items:
            if item.case.seed == seed and item.mode == mode:
                return dict(features=fuzz.features(item.case),
                            t_min=float(item.case.model.t_min),
                            t_final=float(item.case.observable.t_final))
    return dict(features=[], t_min=None, t_final=None)


def _first(out: pathlib.Path, label: str, counted: list) -> dict:
    """The counted row reported for a run: of the case saved in
    ``OUT/kill_cases/<label>.json`` when there is one, else of the lowest
    seed; Gauss-Legendre before QMC."""
    mf, _ = _tools()
    path = out / "kill_cases" / f"{label}.json"
    if path.exists():
        saved = json.loads(path.read_text())["kill"]
        same = [r for r in counted
                if (r["seed"], r["mode"]) == (saved["seed"], saved["mode"])]
        counted = same or counted
    order = {m: k for k, m in enumerate(mf.METHODS)}
    return min(counted, key=lambda r: (r["seed"], r["mode"],
                                       order.get(r["method"], 99)))


def _run_of(out: pathlib.Path, label: str, entry: dict) -> dict:
    """What one run of ``label`` tried: its modes and routes (from
    ``OUT/cases.jsonl``), seeds, rows, counted kills and the reasons of the
    rows that did not kill."""
    ways = set()
    cases = out / "cases.jsonl"
    if cases.exists():
        for line in cases.read_text().splitlines():
            case = json.loads(line)
            if case["label"] == label and not case["mode"].endswith(
                    "/control"):
                ways.add((case["mode"], case["route"]))
    rows = entry["rows_detail"]
    return dict(modes=sorted({m for m, _ in ways}),
                routes=sorted({r for _, r in ways}),
                seeds=sorted({r["seed"] for r in rows}), rows=entry["rows"],
                counted=sum(1 for r in rows if r["counted"]),
                not_killed=entry["not_killed"])


def _entry_row(mut, seen: dict) -> dict:
    mf, km = _tools()
    target = mf.TARGETS[mut.label]
    runs = seen.get("runs", [])
    return dict(
        label=mut.label, kind=mut.kind, path=mut.relpath,
        sha256=km.entry_digest(mut),
        mutated_sha256=seen.get("mutated_sha256"),
        target=dict(mode=target.mode, route=target.route,
                    expect=target.expect, methods=list(target.methods),
                    note=target.note),
        killed=seen.get("kill") is not None, kill=seen.get("kill"),
        methods_killed=seen.get("methods", []),
        searched=dict(seeds=sorted({s for r in runs for s in r["seeds"]}),
                      rows=sum(r["rows"] for r in runs), runs=runs))


def build(outs: Sequence[pathlib.Path]) -> dict:
    """The record from the runs in ``outs`` (see the module docstring)."""
    _, km = _tools()
    entries = target_entries()
    labels: dict = {}
    oracle_sha = set()
    for out in map(pathlib.Path, outs):
        summary = json.loads((out / "kills.json").read_text())
        oracle_sha.add(summary["oracle_sha"])
        for label, entry in summary["labels"].items():
            seen = labels.setdefault(label, dict(kill=None, runs=[]))
            seen["runs"].append(_run_of(out, label, entry))
            counted = [r for r in entry["rows_detail"] if r["counted"]]
            if not counted or seen["kill"] is not None:
                continue
            first = _first(out, label, counted)
            seen["kill"] = _kill_of(entry, first, _facts(
                out, label, first["seed"], first["mode"]))
            seen["methods"] = sorted({
                r["method"] for r in counted
                if (r["seed"], r["mode"]) == (first["seed"], first["mode"])})
            seen["mutated_sha256"] = entry["mutated_sha256"]
    if len(oracle_sha) > 1:
        raise SystemExit(f"the runs used {len(oracle_sha)} oracle codes")
    return dict(rule=RULE, oracle_sha256=next(iter(oracle_sha), None),
                corpus_sha256=km.corpus_digest(entries),
                entries=[_entry_row(mut, labels.get(mut.label, {}))
                         for mut in entries])


def check(path: pathlib.Path = RECORD) -> list:
    """Problems of the record at ``path`` against the current corpus."""
    _, km = _tools()
    text = pathlib.Path(path).read_text(encoding="utf-8")
    record = json.loads(text)
    entries = target_entries()
    problems = []
    if _LOCAL.search(text):
        problems.append("the record holds a local path")
    if record["corpus_sha256"] != km.corpus_digest(entries):
        problems.append("the target entries changed since the record")
    if record.get("oracle_sha256") != current_oracle_sha():
        problems.append("the oracle's code changed since the record")
    rows = {r["label"]: r for r in record["entries"]}
    for mut in entries:
        row = rows.get(mut.label)
        if row is None:
            problems.append(f"{mut.label}: no record")
            continue
        if row["sha256"] != km.entry_digest(mut):
            problems.append(f"{mut.label}: the entry changed since its kill")
        if not row["killed"]:
            problems.append(f"{mut.label}: not killed")
    extra = set(rows) - {m.label for m in entries}
    if extra:
        problems.append(f"entries not in the target list: {sorted(extra)}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python tools/generated_kills.py")
    sub = ap.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write", help="write the record from mutant_fuzz runs")
    w.add_argument("outs", nargs="+", help="OUT directories of mutant_fuzz")
    w.add_argument("--out", default=str(RECORD))
    c = sub.add_parser("check", help="check the record against the corpus")
    c.add_argument("path", nargs="?", default=str(RECORD))
    args = ap.parse_args(argv)
    if args.command == "write":
        record = build([pathlib.Path(o) for o in args.outs])
        text = json.dumps(record, indent=1) + "\n"
        if _LOCAL.search(text):
            raise SystemExit("the record would hold a local path")
        pathlib.Path(args.out).write_text(text, encoding="utf-8")
        for row in record["entries"]:
            kill = row["kill"]
            print(f"{'KILLED' if row['killed'] else 'NOT KILLED':10s} "
                  f"{row['label']}: " + (
                      f"seed {kill['seed']} {kill['mode']}/{kill['route']} "
                      f"{','.join(row['methods_killed'])} {kill['rule']}"
                      if kill else f"{len(row['searched']['seeds'])} seeds"))
        return 0 if all(r["killed"] for r in record["entries"]) else 1
    problems = check(pathlib.Path(args.path))
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
