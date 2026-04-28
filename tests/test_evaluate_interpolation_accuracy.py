"""Regression-lock for the ``cubic -> linear`` switch in ``evaluate.py``.

Three ``RegularGridInterpolator`` instances inside
``PropagatorCache.precompute_C_table_*`` were originally constructed
with ``method='cubic'``.  In production runs we observed wildly wrong
(often negative) interpolation values in the steep tail of the C
propagator -- cosmologically-relevant kernels like a Gaussian
``exp(-c*r**2)`` decay across ~30 orders of magnitude over the gridded
``r`` range, and tensor-product cubic splines overshoot / undershoot
across that dynamic range.  Linear interpolation is monotone and
faithful to the grid endpoints, so we switched to ``method='linear'``.

This file regression-locks two facts on a closed-form test problem:

1. Linear interpolation reproduces a steeply-decaying function within
   a sane tolerance everywhere inside the grid.
2. Cubic interpolation, on the same grid, produces sign / magnitude
   errors in the tail that motivated the switch.

If a future ``scipy`` release fixes the cubic overshoot (or our grid
discretization gets dense enough that cubic no longer breaks), the
``cubic`` assertion below is what will start to fail -- which is a
**good** signal that the switch back to cubic should be re-evaluated.

The interpolator construction here mirrors evaluate.py's call sites
exactly: ``bounds_error=False, fill_value=None`` (so out-of-range
queries extrapolate rather than raise).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator


_T_MAX = 2.0
_R_MAX = 5.0
_N_GRID_T = 12
_N_GRID_R = 6
# A steeply-decaying axisymmetric C on a (t1, t2, r) grid -- the
# canonical shape of the spatial Gaussian kernel times an exponential
# time dependence.
_C_DECAY = 3.0


def _truth(t1: float, t2: float, r: float) -> float:
    return float(np.exp(-(t1 + t2)) * np.exp(-_C_DECAY * r * r))


def _build_interpolators() -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
    ts = np.linspace(0.0, _T_MAX, _N_GRID_T)
    rs = np.linspace(0.0, _R_MAX, _N_GRID_R)
    t1, t2, r = np.meshgrid(ts, ts, rs, indexing="ij")
    grid = np.exp(-(t1 + t2)) * np.exp(-_C_DECAY * r * r)
    common = dict(bounds_error=False, fill_value=None)
    linear = RegularGridInterpolator((ts, ts, rs), grid, method="linear", **common)
    cubic = RegularGridInterpolator((ts, ts, rs), grid, method="cubic", **common)
    return linear, cubic


def test_linear_interpolator_qualitatively_faithful_in_bulk() -> None:
    """Linear interpolation must stay positive and within an
    order-of-magnitude of truth across the bulk of the grid.

    The tolerance is loose by design: with ``_N_GRID_R = 6`` over
    ``r in [0, 5]`` and a steep ``exp(-3 r^2)`` decay, no first-order
    interpolant can be tight in absolute terms.  What this test
    locks is the *qualitative* property -- linear does not flip
    sign, does not blow up, and stays in the right ballpark.  The
    counterpart test ``test_cubic_interpolator_overshoots_in_tail``
    documents that cubic violates these properties.
    """
    linear, _ = _build_interpolators()
    for t1, t2, r in [(0.5, 0.5, 0.2), (1.0, 0.5, 0.5), (0.8, 1.2, 0.8)]:
        truth = _truth(t1, t2, r)
        got = float(linear((t1, t2, r)))
        assert got > 0.0, f"linear must stay positive in the bulk, got {got}"
        rel = abs(got - truth) / abs(truth)
        assert rel < 5.0, (
            f"linear at (t1={t1}, t2={t2}, r={r}) deviates by {rel:.2%} "
            f"(truth={truth:.3e}, got={got:.3e})"
        )


def test_linear_interpolator_stays_nonnegative_in_tail() -> None:
    """In the steep r-tail, linear interpolation must not flip sign.

    The truth there is microscopic (~1e-20) so any reasonable
    interpolant will be off by orders of magnitude in absolute terms,
    but staying non-negative is the property that callers rely on
    (a propagator squared cannot be negative).
    """
    linear, _ = _build_interpolators()
    for t1, t2, r in [(0.5, 0.5, 3.5), (1.0, 1.0, 4.0), (1.5, 1.5, 4.8)]:
        got = float(linear((t1, t2, r)))
        assert got >= -1e-20, (
            f"linear flipped sign at (t1={t1}, t2={t2}, r={r}): {got:.3e}"
        )


def test_cubic_interpolator_overshoots_in_tail() -> None:
    """Cubic interpolation produces sign / magnitude errors in the
    steep r-tail on the same grid.  This is the bug that motivated
    switching to ``method='linear'`` in evaluate.py.

    The assertion is: at *some* point in the tail, cubic produces a
    value with the wrong sign, or a magnitude > 1e3 * truth.  If a
    future scipy release fixes the cubic tensor-product overshoot
    behaviour, this test will start to fail and we should re-evaluate
    whether we can switch back.
    """
    _, cubic = _build_interpolators()
    saw_artefact = False
    for t1, t2, r in [(0.5, 0.5, 1.5), (0.5, 0.5, 4.0), (1.0, 1.0, 4.5)]:
        truth = _truth(t1, t2, r)
        got = float(cubic((t1, t2, r)))
        # Sign flip OR > 1e3 relative error count as an "artefact".
        if got < 0.0 or abs(got - truth) > 1e3 * abs(truth):
            saw_artefact = True
            break
    assert saw_artefact, (
        "Expected at least one cubic-interpolator artefact (sign flip or "
        "huge overshoot) in the steep r-tail; if this assertion fails, the "
        "scipy issue motivating the cubic->linear switch in evaluate.py "
        "may have been fixed -- consider re-evaluating the switch."
    )


def test_linear_extrapolation_does_not_explode() -> None:
    """For r values just outside the grid, linear extrapolation must
    stay bounded; cubic extrapolation is known to blow up.

    This is the second motivation for the switch: lazy-mode caches and
    QMC samples occasionally probe parameter values right at (or just
    past) the grid edge.
    """
    linear, cubic = _build_interpolators()
    pt = (1.0, 1.0, _R_MAX + 0.5)  # 0.5 past r_max
    lin_val = float(linear(pt))
    cub_val = float(cubic(pt))
    # Linear extrapolation magnitude bounded by the last-step gradient;
    # cubic is allowed to be wild.
    assert abs(lin_val) < 1.0, (
        f"linear extrapolation exploded at {pt}: {lin_val:.3e}"
    )
    # The diagnostic value of `cub_val` is the relative explosion.
    # Use a print rather than an assertion so the test stays focused
    # on the actually-locked invariant.
    if abs(cub_val) >= 1.0:  # pragma: no cover - diagnostic only
        pytest.skip(
            f"diagnostic: cubic extrapolation = {cub_val:.3e} (expected "
            f"to be wild, this is informational only)"
        )
