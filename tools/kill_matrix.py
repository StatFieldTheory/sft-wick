#!/usr/bin/env python
"""Write ``docs/verification/kill_matrix.json`` from a sweep of the corpus.

The input is the ``--json`` report of ``tools/mutation_check.py`` run with
``--matrix`` and a selector that runs the whole local default suite, for
example::

    python tools/mutation_check.py --matrix --tests 'tests/ -n 4' \\
        --json kills.json
    python tools/kill_matrix.py kills.json

The matrix records, per entry of the corpus, its kind, its verdict, the test
files that failed under it and, for an entry that was not killed, a
classification (``coverage gap`` or ``equivalent``) and a reason, written by
hand, with the selector, the pytest command and the number of entries per
verdict, overall and per kind.  Each entry carries a digest of its path,
``old`` and ``new`` (:func:`entry_digest`).  A rebuilt matrix keeps the
classification of an entry that was not killed before and is not killed
now, when its digest is unchanged.  The matrix also records a digest of the
corpus (:func:`corpus_digest`), checked by ``tests/test_mutation_corpus.py``,
and the number and a digest of the node IDs of the local default suite
(:func:`node_ids_digest`), which say which suite the matrix was built
against and are not checked against the current one.  Each row records the
number of tests the fast suite collected when that row was swept.

A kill whose failing test files all hold a test known to fail on unmutated
source (:data:`FLAKY_FILES`) is refused: that mutant is to be run again.

Several reports, for example the shards of one sweep, are joined
(:func:`join_reports`).  With ``--reuse`` the rows of the existing matrix
stand in for the entries no report covers, when their entry digest still
matches (:func:`reuse_rows`), so a changed entry is swept alone::

    python tools/mutation_check.py -k new-entry --matrix \\
        --tests 'tests/ -n 4' --json new.json
    python tools/kill_matrix.py --reuse new.json

Exit status: 0 when the matrix was written and every entry that was not
killed is classified, 1 when some are not (the file is written anyway), 2
when the report cannot be used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
from collections.abc import Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "verification" / "kill_matrix.json"
#: The values an entry that was not killed may be classified as.
CLASSIFICATIONS = ("coverage gap", "equivalent")
KILLED = "KILLED"
#: Test files holding a test seen failing on unmutated source, with the
#: test and what was seen.  A kill by these files alone may be the flake.
FLAKY_FILES = {
    "tests/test_validation_fuzz.py":
        "test_VZ18_a_case_that_kills_its_pool_takes_no_other_case_down "
        "failed 1 of 7 runs of the fast suite on unmutated source under load",
}


def _load_harness():
    path = pathlib.Path(__file__).resolve().with_name("mutation_check.py")
    spec = importlib.util.spec_from_file_location("mutation_check", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mutation_check"] = module
    spec.loader.exec_module(module)
    return module


def corpus_digest(corpus: Sequence) -> str:
    """SHA-256 of the label, kind, path, ``old`` and ``new`` of every entry,
    in corpus order.  The selector and ``why`` are left out: the matrix is
    run with one selector for every entry."""
    rows = [[m.label, m.kind, m.relpath, m.old, m.new] for m in corpus]
    data = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def entry_digest(mut) -> str:
    """SHA-256 of the path, ``old`` and ``new`` of one entry."""
    data = json.dumps([mut.relpath, mut.old, mut.new], ensure_ascii=False,
                      separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def fast_suite_node_ids(root: pathlib.Path = ROOT) -> list[str]:
    """The node IDs pytest collects from ``tests/`` in ``root`` under the
    project's ``addopts``, which deselect the ``slow`` and ``heavy`` tests."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", "tests/"],
        cwd=root, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"pytest collection failed (exit {proc.returncode})"
                           f":\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return [line for line in proc.stdout.splitlines()
            if line.startswith("tests/") and "::" in line]


def node_ids_digest(node_ids: Sequence[str]) -> str:
    """SHA-256 of the sorted node IDs, one per line."""
    data = "\n".join(sorted(node_ids))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _previous_classification(previous: dict | None) -> dict[str, dict]:
    if not previous:
        return {}
    return {(m["label"], m.get("sha256")): m
            for m in previous.get("mutants", ())
            if m.get("verdict") != KILLED and m.get("classification")}


def _only_flaky(failed_files: Sequence[str]) -> bool:
    return bool(failed_files) and all(f in FLAKY_FILES for f in failed_files)


def join_reports(reports: Sequence[dict]) -> dict:
    """One report from several, such as the shards of one sweep.

    Raises:
        ValueError: if no report is given or two reports hold one label.
    """
    if not reports:
        raise ValueError("no report to join")
    mutants: list[dict] = []
    seen: set[str] = set()
    for report in reports:
        for record in report.get("mutants", ()):
            if record["label"] in seen:
                raise ValueError(f"{record['label']!r} is in two reports")
            seen.add(record["label"])
            mutants.append(record)
    changed = sorted({c for r in reports for c in r.get("changed") or ()})
    return {"matrix": all(r.get("matrix") for r in reports),
            "changed": changed, "mutants": mutants}


def reuse_rows(report: dict, corpus: Sequence, previous: dict | None) -> dict:
    """``report`` with a record, taken from the ``previous`` matrix, for
    every entry of ``corpus`` it lacks whose label and entry digest
    (:func:`entry_digest`) the previous matrix holds.  The record keeps the
    previous verdict, exit code and failing files, the number of tests
    the fast suite collected when it was swept (the previous matrix's own
    count for a row that does not record one), and the previous selector.  An entry whose digest changed gets no record, so
    :func:`build` refuses the report until that entry is swept again.

    Raises:
        ValueError: if a row is to be reused and the previous matrix has no
            single selector.
    """
    have = {r["label"] for r in report.get("mutants", ())}
    rows = {(m["label"], m.get("sha256")): m
            for m in (previous or {}).get("mutants", ())}
    reused = [rows[(mut.label, entry_digest(mut))] for mut in corpus
              if mut.label not in have
              and (mut.label, entry_digest(mut)) in rows]
    if not reused:
        return report
    selector = previous.get("selector")
    if not isinstance(selector, str):
        raise ValueError("the previous matrix ran more than one selector, so "
                         "its rows cannot be reused under one")
    records = [{"label": m["label"], "path": m["path"], "tests": selector,
                "verdict": m["verdict"], "returncode": m.get("returncode"),
                "failed_files": list(m.get("failed_files", ())),
                "fast_suite_tests": m.get("fast_suite_tests",
                                          previous.get("fast_suite_tests"))}
               for m in reused]
    return {**report, "mutants": list(report.get("mutants", ())) + records}


def build(report: dict, corpus: Sequence, node_ids: Sequence[str],
          previous: dict | None) -> dict:
    """The kill matrix of ``corpus`` from a ``--matrix`` ``report``.

    Raises:
        ValueError: if the report was not made with ``--matrix``, a target
            file changed during the sweep, or an entry of the corpus has no
            record in the report, or a kill comes from
            :data:`FLAKY_FILES` alone.
    """
    if not report.get("matrix"):
        raise ValueError("the report was not made with --matrix, so its "
                         "failing test files are not the whole list")
    if report.get("changed"):
        raise ValueError(f"target files changed during the sweep: "
                         f"{report['changed']}")
    records = {r["label"]: r for r in report.get("mutants", ())}
    missing = [m.label for m in corpus if m.label not in records]
    if missing:
        raise ValueError(f"the report has no record of {missing[0]!r}"
                         + (f" and {len(missing) - 1} more" if len(missing) > 1
                            else ""))
    flaky = [m.label for m in corpus if records[m.label]["verdict"] == KILLED
             and _only_flaky(records[m.label].get("failed_files", ()))]
    if flaky:
        raise ValueError(f"{flaky} killed only by {sorted(FLAKY_FILES)}, "
                         "which hold a test seen failing on unmutated source; "
                         "run them again")
    kept = _previous_classification(previous)
    mutants = []
    for mut in corpus:
        record = records[mut.label]
        verdict = record["verdict"]
        digest = entry_digest(mut)
        old = kept.get((mut.label, digest)) if verdict != KILLED else None
        mutants.append({
            "label": mut.label,
            "kind": mut.kind,
            "path": mut.relpath,
            "sha256": digest,
            "verdict": verdict,
            "returncode": record.get("returncode"),
            "failed_files": sorted(record.get("failed_files", ())),
            "fast_suite_tests": record.get("fast_suite_tests", len(node_ids)),
            "classification": old["classification"] if old else None,
            "reason": old.get("reason", "") if old else "",
        })
    selectors = sorted({r.get("tests", "") for r in records.values()})
    return {
        "selector": selectors[0] if len(selectors) == 1 else selectors,
        "pytest_command": [_pytest_command(s) for s in selectors][0]
        if len(selectors) == 1 else [_pytest_command(s) for s in selectors],
        "corpus_sha256": corpus_digest(corpus),
        "fast_suite_tests": len(node_ids),
        "fast_suite_sha256": node_ids_digest(node_ids),
        "killed": sum(m["verdict"] == KILLED for m in mutants),
        "totals": _totals(mutants),
        "mutants": mutants,
    }


#: The flags ``tools/mutation_tree.py`` adds to every pytest run of a
#: ``--matrix`` sweep, after the selector.  The project's ``addopts``
#: (``-m 'not slow and not heavy'``) apply as well.
MATRIX_FLAGS = "--no-header -p no:cacheprovider -o junit_family=xunit1"


def _pytest_command(selector: str) -> str:
    return f"python -m pytest {selector} {MATRIX_FLAGS}"


def _totals(mutants: Sequence[dict]) -> dict:
    """Entries per verdict, overall and per kind."""
    totals: dict = {"entries": len(mutants), "by_verdict": {}, "by_kind": {}}
    for m in mutants:
        verdict, kind = m["verdict"], m["kind"] or "none"
        totals["by_verdict"][verdict] = totals["by_verdict"].get(verdict, 0) + 1
        per_kind = totals["by_kind"].setdefault(kind, {})
        per_kind[verdict] = per_kind.get(verdict, 0) + 1
    return totals


def unclassified(matrix: dict) -> list[str]:
    """Labels of the entries not killed that lack a classification or a
    reason."""
    return [m["label"] for m in matrix["mutants"]
            if m["verdict"] != KILLED
            and (m.get("classification") not in CLASSIFICATIONS
                 or not str(m.get("reason") or "").strip())]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reports", type=pathlib.Path, nargs="+",
                    help="the --json reports of a --matrix sweep, joined")
    ap.add_argument("--reuse", action="store_true",
                    help="take the rows of the existing matrix for the "
                    "entries no report covers and whose digest is unchanged")
    ap.add_argument("--out", type=pathlib.Path, default=OUT,
                    help=f"the matrix to write; default {OUT.relative_to(ROOT)}")
    args = ap.parse_args(argv)
    harness = _load_harness()
    try:
        report = join_reports([json.loads(r.read_text(encoding="utf-8"))
                               for r in args.reports])
        previous = (json.loads(args.out.read_text(encoding="utf-8"))
                    if args.out.exists() else None)
        corpus = harness.load_corpus()
        if args.reuse:
            report = reuse_rows(report, corpus, previous)
        matrix = build(report, corpus, fast_suite_node_ids(ROOT), previous)
    except (OSError, ValueError, RuntimeError, harness.HarnessError) as exc:
        print(f"cannot write the matrix: {exc}")
        return 2
    args.out.write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out}: {matrix['killed']} of {len(matrix['mutants'])} "
          "killed")
    missing = unclassified(matrix)
    if missing:
        print("classify these in the file (classification and reason): "
              + ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
