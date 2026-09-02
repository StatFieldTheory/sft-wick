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
import csv, inspect, os
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
