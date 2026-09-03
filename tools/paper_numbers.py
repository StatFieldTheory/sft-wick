#!/usr/bin/env python
"""Print the paper-facing figures, read from the files that own them.

WHY THIS EXISTS.  Over one revision round, four separate figures reached the
manuscript wrong -- a runtime attached to the wrong row and unit, a test count,
a per-subject count, and an assertion partition that did not sum -- and every
one of them was correct in the repository and wrong only in transit.  0.4.2
fixed the hop from *measurement* into the catalogue, by having
``tools/gen_test_catalog.py`` read ``approx_audit_summary.json`` instead of
carrying transcribed prose.  This closes the next hop: catalogue into a
message, a commit, or a paragraph of LaTeX.

Nothing here computes anything.  It reads the committed files and prints what
they say, so that quoting a figure requires no retyping::

    python tools/paper_numbers.py            # human-readable
    python tools/paper_numbers.py --json     # machine-readable

If a figure you need is not printed here, add it here rather than reading it
off by eye -- that is the whole point.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs" / "verification" / "catalog.rst"
SUMMARY = ROOT / "docs" / "verification" / "approx_audit_summary.json"


def _version() -> str:
    m = re.search(r'(?m)^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text())
    return m.group(1) if m else "?"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                                        # pragma: no cover
        return ""


def collect() -> dict:
    cat = CATALOG.read_text()
    total = re.search(r"has \*\*(\d+) tests\*\* in (\d+) files", cat)
    subjects = re.findall(
        r"(?m)^([A-Z][^\n*]+?)\n[-~^]{3,}\n\n\*(\d+) tests in (\d+) files\.\*", cat)
    audit = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    b = audit.get("buckets", {})
    return {
        "version": _version(),
        "git_describe": _git("describe", "--tags", "--always"),
        "catalog": {
            "total_tests": int(total.group(1)) if total else None,
            "total_files": int(total.group(2)) if total else None,
            "by_subject": {name.strip(): {"tests": int(t), "files": int(f)}
                           for name, t, f in subjects},
        },
        "approx_audit": {
            "total_sites": audit.get("total_sites"),
            "rel_in_force": b.get("rel_in_force"),
            "explicit_abs": b.get("explicit_abs"),
            "explicit_abs_zero": audit.get("explicit_abs_zero"),
            "compared_to_zero": b.get("compared_to_zero"),
            "weakened": b.get("weakened"),
            "vacuous": audit.get("vacuous"),
            "bucket_priority": audit.get("bucket_priority"),
        },
    }


def check(d: dict) -> list[str]:
    """Refuse to print figures that contradict each other."""
    problems = []
    cat = d["catalog"]
    if cat["by_subject"]:
        s = sum(v["tests"] for v in cat["by_subject"].values())
        if cat["total_tests"] is not None and s != cat["total_tests"]:
            problems.append(
                f"catalogue subject rows sum to {s} but the header says "
                f"{cat['total_tests']}")
    a = d["approx_audit"]
    buckets = [a[k] for k in ("rel_in_force", "explicit_abs",
                              "compared_to_zero", "weakened")]
    if all(x is not None for x in buckets) and a["total_sites"] is not None:
        if sum(buckets) != a["total_sites"]:
            problems.append(
                f"approx buckets sum to {sum(buckets)} but the total is "
                f"{a['total_sites']} -- this is the 0.4.1 defect recurring")
    return problems


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args()

    d = collect()
    problems = check(d)

    if args.json:
        print(json.dumps({**d, "problems": problems}, indent=2, sort_keys=True))
        return 1 if problems else 0

    cat, a = d["catalog"], d["approx_audit"]
    print(f"sft-wick {d['version']}  ({d['git_describe']})")
    print("  source: docs/verification/{catalog.rst, approx_audit_summary.json}")
    print()
    print(f"  Test suite: {cat['total_tests']} tests in {cat['total_files']} files")
    for name, v in cat["by_subject"].items():
        print(f"    {name:<34} {v['tests']:>5} tests in {v['files']:>2} files")
    print()
    print(f"  pytest.approx sites: {a['total_sites']}")
    print(f"    relative tolerance in force      {a['rel_in_force']:>5}")
    print(f"    explicit abs=                    {a['explicit_abs']:>5}"
          f"  ({a['explicit_abs_zero']} of them abs=0.0)")
    print(f"    compares against 0 (default floor){a['compared_to_zero']:>5}")
    print(f"    WEAKENED by the 1e-12 floor      {a['weakened']:>5}")
    print(f"    vacuous (cannot fail)            {a['vacuous']:>5}")
    print(f"    bucket priority: {' > '.join(a['bucket_priority'] or [])}")
    print()
    if problems:
        print("  PROBLEMS -- do not quote these figures:")
        for x in problems:
            print(f"    ! {x}")
        return 1
    print("  self-consistent: subject rows sum to the total; buckets partition the sites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
