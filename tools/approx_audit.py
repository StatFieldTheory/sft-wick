"""pytest plugin: measure what every ``pytest.approx`` site ACTUALLY enforces.

``approx`` compares against ``max(rel * expected, abs)`` with ``abs``
defaulting to 1e-12, so a site written with a tight ``rel`` on a small
quantity silently enforces the floor.  Which sites those are depends on
the runtime magnitude of the compared value, so it cannot be read off
the source -- this wraps ``approx`` and records it.

Measured on the suite 2026-09-02 (173 runtime sites):

* 15 compare against ``0.0`` -- the floor is the point there, not a bug;
* 29 pass an explicit ``abs=``, i.e. a deliberate absolute tolerance;
* 108 have their ``rel`` genuinely in force;
* **21 are weakened by the default floor**, of which 20 still enforce
  2.0e-12 to 2.4e-10 relative -- looser than written, but real checks.

So the headline risk is much narrower than a static grep suggests, and
it is NOT concentrated where one would guess: the 10 affected sites in
``test_msr_numerics_regressions.py`` compare quantities of 1.8e-02 to
0.5 and still enforce 2.0e-12 to 5.5e-11.

Usage::

    PYTHONPATH=tools pytest tests/ -p approx_audit \
        --approx-audit-out=audit.csv

Read the ``effective_rel`` column: it is what the site actually enforces
as a relative tolerance, against ``rel_written`` which is what it reads
as.  ``vacuous`` marks a site whose tolerance exceeds the quantity being
compared, i.e. one that cannot fail.
"""
import csv, inspect, json, os
import numpy as np
import pytest as _pytest

_ROWS = []
_orig = _pytest.approx
DEFAULT_ABS = 1e-12


def _record(expected, rel, abs_):
    # Caller frame outside this file.
    fr = inspect.currentframe().f_back.f_back
    while fr and ("approx_audit" in (fr.f_code.co_filename or "")):
        fr = fr.f_back
    where = f"{os.path.relpath(fr.f_code.co_filename)}:{fr.f_lineno}" if fr else "?"
    try:
        arr = np.abs(np.asarray(expected, dtype=float)).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        mag = float(np.min(arr[arr > 0])) if np.any(arr > 0) else 0.0
    except Exception:
        return
    r = 1e-6 if rel is None else float(rel)          # pytest's own default rel
    a = DEFAULT_ABS if abs_ is None else float(abs_)
    enforced = max(r * mag, a)
    _ROWS.append(dict(
        site=where, min_abs_expected=mag, rel_written=r, abs_written=("default"
        if abs_ is None else a), rel_times_expected=r * mag, enforced=enforced,
        effective_rel=(enforced / mag if mag > 0 else float("inf")),
        floor_dominates=(a > r * mag), vacuous=(mag > 0 and enforced >= mag)))


def _approx(expected, rel=None, abs=None, nan_ok=False):
    try:
        _record(expected, rel, abs)
    except Exception:
        pass
    return _orig(expected, rel=rel, abs=abs, nan_ok=nan_ok)


def pytest_addoption(parser):
    parser.addoption("--approx-audit-out", default="approx_audit.csv")


def pytest_configure(config):
    _pytest.approx = _approx


def pytest_unconfigure(config):
    _pytest.approx = _orig
    out = config.getoption("--approx-audit-out")
    if not _ROWS:
        return
    # Collapse repeated hits of the same site to its WORST case.
    best = {}
    for r in _ROWS:
        k = r["site"]
        if k not in best or r["effective_rel"] > best[k]["effective_rel"]:
            best[k] = r
    rows = sorted(best.values(), key=lambda r: -r["effective_rel"])
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\n[approx-audit] {len(rows)} distinct sites -> {out}")
    _write_summary(rows)


#: The catalogue quotes these figures.  They are WRITTEN HERE by the
#: measurement rather than retyped into prose, because the 0.4.1 catalogue
#: restated a partition by hand and it did not add up: it carried 29 and 15
#: from an earlier 173-site run while updating the total to 207, leaving 31
#: sites in no bucket.  A figure that describes a measurement should come from
#: that measurement.
SUMMARY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "verification", "approx_audit_summary.json",
)


def classify(row):
    """Bucket one row.  Priority, stated so the partition is reproducible:

    explicit ``abs=`` > compares against 0 > floor dominates > ``rel`` in force.

    The priority matters only for sites that are BOTH explicit-``abs`` and
    zero-compared; putting them under "explicit" keeps the buckets answering
    "what did the author write" first and "what did it enforce" second.
    """
    if row["abs_written"] != "default":
        return "explicit_abs"
    if float(row["min_abs_expected"]) == 0.0:
        return "compared_to_zero"
    if row["floor_dominates"]:
        return "weakened"
    return "rel_in_force"


def summarise(rows):
    counts = {k: 0 for k in
              ("rel_in_force", "explicit_abs", "compared_to_zero", "weakened")}
    weakened, per_file = [], {}
    for r in rows:
        b = classify(r)
        counts[b] += 1
        if b == "weakened":
            weakened.append(r)
            f = r["site"].split(":")[0]
            per_file[f] = per_file.get(f, 0) + 1
    effs = sorted(r["effective_rel"] for r in weakened)
    return {
        "total_sites": len(rows),
        "buckets": counts,
        "bucket_priority": ["explicit_abs", "compared_to_zero", "weakened",
                            "rel_in_force"],
        "explicit_abs_zero": sum(
            1 for r in rows if r["abs_written"] != "default"
            and float(r["abs_written"]) == 0.0),
        "vacuous": sum(1 for r in rows if r["vacuous"]),
        "weakened_effective_rel_min": effs[0] if effs else None,
        "weakened_effective_rel_max": effs[-1] if effs else None,
        "weakened_per_file": dict(sorted(per_file.items())),
    }


def _write_summary(rows):
    summary = summarise(rows)
    assert sum(summary["buckets"].values()) == summary["total_sites"], (
        "the buckets must partition the sites exactly -- that they did not is "
        "the defect this summary exists to prevent")
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"[approx-audit] summary -> {SUMMARY_PATH}")
