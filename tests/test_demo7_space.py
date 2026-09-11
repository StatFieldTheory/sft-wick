"""Demo 7: observables in space, angle and time.

``R`` is local in space and the local vertices act at one point, so an
n-point function at the observation points reduces to a finite-dimensional
Itô SDE there, with the spatial kernel at the points' separations as the
noise covariance (``examples/demo7/space7_reference.py``; no sft-wick code).
That one reference covers every homogeneity, kernel and dimension.  Checked
here:

* the reference itself: its two-time propagation and its integrated fields
  against quadrature of the same closed form, and the hand-written Legendre
  polynomials against numpy's;
* (a) the two-time ``⟨φ_a(x, t) φ_b(y, t')⟩`` at orders 0-2, ``a ≠ b``,
  ``r ≠ 0``, ``t ≠ t'`` in both orders, on every integrator, with a scalar
  and a matrix R and with a component-mixing white noise (which makes
  ``C_{01}(t, t')`` asymmetric under swapping the two times);
* (b) ``SeparableRotation`` with a four-coefficient ``LegendreAngular``
  kernel at three angles, where C changes sign with direction;
* (c) the package's own quadrature tables for a Gaussian and a custom
  (damped-cosine, negative at the separation used) translation kernel and
  for a ``GeneralKappa2`` with no spatial symmetry and a different kernel
  per component;
* (e) ``integrate_over='all'`` and a one-point subset at ``N = 2`` with
  ``a ≠ b`` at order 2, and the three-point function at order 2.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import dblquad, quad

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo7"
sys.path.insert(0, str(DEMO))
import space7_model as m7  # noqa: E402
import space7_reference as r7  # noqa: E402
import space7_run as run7  # noqa: E402

T, T_EARLY = run7.T, run7.T_EARLY
POS = run7.POS
TAGS = run7.TAGS
TWO, THREE = run7.TWO, run7.THREE


def _phi_quad(gamma, rho, t1, t2, rtol=1e-13):
    """``∫_0^{t1}∫_0^{t2} e^{−γ(t1−λ1)} e^{−ρ|λ1−λ2|} e^{−γ(t2−λ2)}``, split
    at the kernel's cusp."""
    def f(l2, l1):
        return np.exp(-gamma * (t1 - l1) - rho * abs(l1 - l2)
                      - gamma * (t2 - l2))
    td = min(t1, t2)
    opts = dict(epsabs=0.1 * rtol, epsrel=rtol)
    total = (dblquad(f, 0, td, lambda l1: 0, lambda l1: l1, **opts)[0]
             + dblquad(f, 0, td, lambda l1: l1, lambda l1: td, **opts)[0])
    if t1 > t2:
        total += dblquad(f, td, t1, lambda l1: 0, lambda l1: td, **opts)[0]
    elif t2 > t1:
        total += dblquad(f, 0, td, lambda l1: td, lambda l1: t2, **opts)[0]
    return total


def _C_quad(cfg, points, a, b, i, j, t1, t2, rtol=1e-13):
    """``C_ab(x_i, t1; x_j, t2)`` of the free theory by quadrature: the
    coloured part ``δ_ab Σ^a_ij Φ`` plus the white part ``S_ab Ψ``."""
    sigma = cfg.sigma(points)
    rho = cfg.rho
    out = 0.0
    if a == b:
        out += sigma[a][i, j] * _phi_quad(cfg.gamma[a], rho[a], t1, t2,
                                          rtol=rtol)
    if cfg.white is not None:
        S = np.asarray(cfg.white, float)
        out += S[a, b] * quad(
            lambda tau: np.exp(-cfg.gamma[a] * (t1 - tau)
                               - cfg.gamma[b] * (t2 - tau)),
            0.0, min(t1, t2), epsabs=1e-15, epsrel=1e-13)[0]
    return out


# ---------------------------------------------------------------------------
# The reference
# ---------------------------------------------------------------------------

MIX = "translation-exp-white-matrixR"


@pytest.mark.parametrize("ab", list(itertools.product(range(2), repeat=2)))
@pytest.mark.parametrize("times", [(T, T_EARLY), (T_EARLY, T)])
def test_reference_two_time_matches_quadrature(ab, times):
    """The hierarchy's two-time propagation at ``F = G = 0`` against the
    double integral of the same free theory."""
    cfg = m7.CONFIGS[MIX]
    points = [POS["x"], POS["y"]]
    H = cfg.hierarchy(points)
    a, b = ab
    t1, t2 = times
    if t1 >= t2:
        got = H.two_time([(a, 0)], [(b, 1)], [(0, 0)], t1, t2)[(0, 0)]
    else:
        got = H.two_time([(b, 1)], [(a, 0)], [(0, 0)], t2, t1)[(0, 0)]
    ref = _C_quad(cfg, points, a, b, 0, 1, t1, t2)
    assert got == pytest.approx(ref, rel=1e-10, abs=0.0)


def test_the_mixing_white_noise_makes_C01_asymmetric_in_the_two_times():
    """Swapping the two times changes ``C_{01}`` by more than 10 %: the
    two-time checks below are not blind to a swap of the external times.
    With one decay rate for both components it would be exactly symmetric."""
    cfg = m7.CONFIGS[MIX]
    H = cfg.hierarchy([POS["x"], POS["y"]])
    late = H.two_time([(0, 0)], [(1, 1)], [(0, 0)], T, T_EARLY)[(0, 0)]
    early = H.two_time([(1, 1)], [(0, 0)], [(0, 0)], T, T_EARLY)[(0, 0)]
    assert abs(late - early) / abs(late) > 0.1


def test_reference_integrated_fields_match_time_integrals():
    """``I_{a,i}(T) = ∫_0^T φ_{a,i}``: the extra states behind the package's
    ``integrate_over``."""
    cfg = m7.CONFIGS[MIX]
    points = [POS["x"], POS["y"]]
    H = cfg.hierarchy(points, integrated=[(0, 0), (1, 1)])
    one = H.moments([("I", 0, 0), (1, 1)], [(0, 0)], T)[(0, 0)]
    ref_one = quad(lambda t: _C_quad(cfg, points, 0, 1, 0, 1, t, T,
                                     rtol=1e-11),
                   0.0, T, epsabs=1e-13, epsrel=1e-11)[0]
    assert one == pytest.approx(ref_one, rel=1e-8, abs=0.0)
    both = H.moments([("I", 0, 0), ("I", 1, 1)], [(0, 0)], T)[(0, 0)]
    ref_both = dblquad(lambda t2, t1: _C_quad(cfg, points, 0, 1, 0, 1, t1, t2,
                                              rtol=1e-9),
                       0.0, T, 0.0, T, epsabs=1e-10, epsrel=1e-8)[0]
    assert both == pytest.approx(ref_both, rel=1e-6, abs=0.0)


def test_legendre_series_matches_numpy():
    from numpy.polynomial.legendre import legval
    coeffs = m7.CONFIGS["rotation-legendre"].coeffs
    for cos in (-1.0, -0.42, 0.0, 0.77, 1.0):
        assert r7.legendre_series(coeffs, cos) == pytest.approx(
            float(legval(cos, np.asarray(coeffs))), rel=1e-13, abs=1e-15)


# ---------------------------------------------------------------------------
# The package against the reference
# ---------------------------------------------------------------------------

def _check(setup, observable, comps, order, route, ref, rel, *,
           positions=POS, t_final=None, orders=(0, 1, 2), **kw):
    row = run7.run_case(
        setup, observable, comps, order, route, ref=ref, positions=positions,
        t_final=(setup.cfg.t_min + T if t_final is None else t_final),
        orders=orders, **kw)
    if row["status"] == "refused":
        pytest.skip(f"{route[0]}: {row['reason']}")
    assert row["error"] <= rel, row
    assert row["zero"] <= 1e-12, row
    return row


@pytest.fixture(scope="module")
def setups():
    """One :class:`space7_run.Setup` per configuration, built once.  The
    quadrature-table configurations use coarser tables than
    ``space7_run.py`` so the file stays fast; the tolerances below are the
    measured errors of these settings with margin."""
    cfgs = {
        "exp": m7.CONFIGS["translation-exp"],
        "mix": m7.CONFIGS[MIX],
        "rot": m7.CONFIGS["rotation-legendre"].with_(n_grid_t=16),
        "custom": m7.CONFIGS["translation-custom-quad"].with_(n_grid_t=20),
        "general": m7.CONFIGS["general-quad"].with_(n_grid_t=14),
    }
    return {k: run7.Setup(c) for k, c in cfgs.items()}


# -- (a) the two-time function ------------------------------------------------

@pytest.mark.parametrize("route,rel", [(run7.GL(32), 1e-6), (run7.NQ, 1e-6),
                                       (run7.QV(12), 1e-4)])
@pytest.mark.parametrize("times", [(T, T_EARLY), (T_EARLY, T)])
@pytest.mark.parametrize("ab", [(0, 1), (1, 1)])
def test_two_time_scalar_R(setups, ab, times, route, rel):
    """(a) ``⟨φ_a(x, t) φ_b(y, t')⟩`` at orders 0-2 through
    ``external_times``, both time orders."""
    s = setups["exp"]
    H = s.cfg.hierarchy([POS["x"], POS["y"]])
    tx, ty = times
    for order in (0, 1, 2):
        if tx >= ty:
            ref = H.two_time([(ab[0], 0)], [(ab[1], 1)], TAGS[order], tx, ty)
        else:
            ref = H.two_time([(ab[1], 1)], [(ab[0], 0)], TAGS[order], ty, tx)
        _check(s, TWO, ab, order, route, ref, rel,
               t_final=s.cfg.t_min + max(tx, ty),
               external_times={"x": s.cfg.t_min + tx,
                               "y": s.cfg.t_min + ty})


@pytest.mark.parametrize("route,rel", [(run7.GL(32), 1e-6), (run7.NQ, 1e-6),
                                       (run7.QV(14), 1e-5),
                                       (run7.QS(10), 1e-3)])
@pytest.mark.parametrize("order", [0, 1])
def test_two_time_matrix_R_and_mixing_white_noise(setups, order, route, rel):
    """(a) with a rate per component (matrix R) and a dense ``ConstantImpulse``:
    ``C_{01}`` is then non-zero at order 0 and asymmetric in the two times.
    The batched backends refuse a matrix R at the interacting orders (that
    is being added elsewhere); those cases skip."""
    s = setups["mix"]
    H = s.cfg.hierarchy([POS["x"], POS["y"]])
    for ab in [(0, 1), (1, 0)]:
        ref = H.two_time([(ab[0], 0)], [(ab[1], 1)], TAGS[order], T, T_EARLY)
        _check(s, TWO, ab, order, route, ref, rel,
               t_final=s.cfg.t_min + T,
               external_times={"x": s.cfg.t_min + T,
                               "y": s.cfg.t_min + T_EARLY})


@pytest.mark.slow
def test_two_time_matrix_R_order_2(setups):
    s = setups["mix"]
    H = s.cfg.hierarchy([POS["x"], POS["y"]])
    ref = H.two_time([(0, 0)], [(1, 1)], TAGS[2], T, T_EARLY)
    _check(s, TWO, (0, 1), 2, run7.QS(11), ref, 1e-3,
           t_final=s.cfg.t_min + T,
           external_times={"x": s.cfg.t_min + T, "y": s.cfg.t_min + T_EARLY})


# -- (b) rotation-invariant noise ---------------------------------------------

@pytest.mark.parametrize("theta", [35.0, 150.0])
@pytest.mark.parametrize("ab", [(0, 1), (1, 1)])
def test_rotation_legendre_kernel(setups, ab, theta):
    """(b) ``SeparableRotation`` with four Legendre coefficients, through the
    package's own quadrature tables."""
    s = setups["rot"]
    y = run7.direction(theta)
    H = s.cfg.hierarchy([run7.NORTH, y])
    for order in (0, 1, 2):
        ref = H.moments([(ab[0], 0), (ab[1], 1)], TAGS[order], T)
        _check(s, TWO, ab, order, run7.GL(24), ref, 2e-4,
               positions={"x": run7.NORTH, "y": y})


def test_C_changes_sign_with_the_direction(setups):
    """The four-coefficient kernel is positive at 35° and negative at 150°,
    so an angle-blind C could not pass the check above.  Every earlier test
    used ``coeffs=[1.0]``, where C does not depend on direction at all."""
    s = setups["rot"]
    vals = []
    for theta in (35.0, 150.0):
        y = run7.direction(theta)
        vals.append(s.expansion(TWO, (0, 1, 2)).evaluate(
            s.props, positions={"x": run7.NORTH, "y": y},
            t_final=s.cfg.t_min + T, component_pair=(1, 1), orders=[0],
            method="gauss_legendre", n_gauss=8).total)
    assert vals[0] > 0.0 > vals[1]
    assert abs(vals[1] / vals[0]) > 0.05


# -- (c) the package's quadrature tables --------------------------------------

@pytest.mark.parametrize("key,rel", [("custom", 2e-4), ("general", 2e-4)])
@pytest.mark.parametrize("ab", [(0, 1), (1, 1)])
def test_quadrature_tables_for_custom_and_general_kernels(setups, key, ab,
                                                          rel):
    """(c) a damped-cosine ``CustomKernel`` (translation, negative at the
    separation used) and a ``GeneralKappa2`` with no spatial symmetry and a
    different amplitude, correlation time and width per component."""
    s = setups[key]
    H = s.cfg.hierarchy([POS["x"], POS["y"]])
    for order in (0, 1, 2):
        ref = H.moments([(ab[0], 0), (ab[1], 1)], TAGS[order], T)
        _check(s, TWO, ab, order, run7.GL(24), ref, rel)


def test_the_custom_kernel_is_negative_at_the_separation_used():
    cfg = m7.CONFIGS["translation-custom-quad"]
    r = abs(POS["y"] - POS["x"])
    assert cfg.custom(r) < -0.1 * cfg.custom(0.0)


def test_the_general_kernel_is_not_translation_invariant():
    """``κ²(x, x')`` of the general configuration depends on the pair, not
    only on ``x − x'``; and its two components differ."""
    g = m7.CONFIGS["general-quad"].general
    same_sep = (g.spatial(0, 0.0, 1.3), g.spatial(0, -0.6, 0.7))
    assert abs(same_sep[0] / same_sep[1] - 1.0) > 0.1
    assert abs(g.spatial(0, 0.0, 1.3) / g.spatial(1, 0.0, 1.3) - 1.0) > 0.1


# -- (e) integrate_over and the three-point function ---------------------------

@pytest.mark.parametrize("ab", [(0, 1), (1, 1)])
@pytest.mark.parametrize("over,legs,route,rel", [
    ("all", ["I", "I"], run7.GL(12), 1e-4),
    ("all", ["I", "I"], run7.QV(12), 1e-2),
    (("x",), ["I", "phi"], run7.GL(16), 1e-6),
    (("x",), ["I", "phi"], run7.QV(12), 1e-3),
])
def test_integrate_over_at_order_2(setups, ab, over, legs, route, rel):
    """(e) the time-integrated moment ``⟨∫φ_a(x) ∫φ_b(y)⟩`` and the mixed
    ``⟨∫φ_a(x) φ_b(y, t_f)⟩`` at ``N = 2``."""
    s = setups["exp"]
    H = s.cfg.hierarchy([POS["x"], POS["y"]],
                        integrated=[(ab[0], 0), (ab[1], 1)])
    ref_legs = [("I", ab[0], 0) if legs[0] == "I" else (ab[0], 0),
                ("I", ab[1], 1) if legs[1] == "I" else (ab[1], 1)]
    ref = H.moments(ref_legs, TAGS[2], T)
    _check(s, TWO, ab, 2, route, ref, rel,
           integrate_over=(over if over == "all" else set(over)))


@pytest.mark.parametrize("route,rel", [(run7.GL(16), 1e-5),
                                       (run7.QV(12), 1e-3)])
@pytest.mark.parametrize("abc", [(0, 1, 1), (1, 1, 0)])
def test_three_point_function(setups, abc, route, rel):
    """(e) ``⟨φ_a(x) φ_b(y) φ_c(z)⟩`` at orders 1 (the ``F`` channel) and 2
    (the ``FG`` channel: ``FF`` and ``GG`` have an odd number of fields and
    vanish)."""
    s = setups["exp"]
    H = s.cfg.hierarchy([POS["x"], POS["y"], POS["z"]])
    for order in (1, 2):
        ref = H.moments([(c, i) for i, c in enumerate(abc)], TAGS[order], T)
        _check(s, THREE, abc, order, route, ref, rel, orders=(1, 2))
