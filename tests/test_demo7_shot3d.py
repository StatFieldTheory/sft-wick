"""Demo 7, item (d): 3-D positions with a callable κ³.

Compound-Poisson shot noise in ``R^3`` (``examples/demo7/space7_shot.py``):
every cumulant is a Gaussian overlap integral in closed form, so the ``κ³``
vertex is a callable evaluated at 3-D leg positions and C is an exact
closed form of the 3-vector separation.  Checked here:

* the closed forms against direct quadrature (the 3-D overlap coordinate by
  coordinate, ``T̃`` against ``scipy.quad``);
* the moment hierarchy against them at ``F = 0``;
* the package's ``⟨φ_a(x) φ_b(y)⟩`` (order 0, and the ``F`` and ``FK3``
  channels of order 2) and ``⟨φ_a(x) φ_b(y) φ_c(z)⟩`` (the ``F`` and ``K3``
  channels of order 1) against the hierarchy, at off-diagonal component
  tuples, for white pulses (raw ``equal_time`` κ³), exponential pulses
  (``already_R_contracted``) and a matrix-valued R;
* that the check sees the third dimension at all: the same evaluation with
  the points' first coordinates only differs by more than 10 %.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo7"
sys.path.insert(0, str(DEMO))
import space7_shot as sh  # noqa: E402
import space7_shot_system as ss  # noqa: E402
import space7_shot3d as d7  # noqa: E402

T = d7.T
POINTS = d7.POINTS
X3 = np.array([sh.POINTS["x"], sh.POINTS["y"], sh.POINTS["z"]])


def _rel(a, b):
    return abs(a - b) / abs(b)


# ---------------------------------------------------------------------------
# The closed forms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("comps,powers", [((0, 1, 1), None), ((1, 0), (2, 1)),
                                          ((0, 1, 0), (1, 2, 1))])
def test_spatial_overlap_matches_coordinate_quadrature(comps, powers):
    """``∫d^3z Π w(x_j − z)^{k_j}`` factorises over the coordinates; each
    factor is a 1-D quadrature."""
    p = sh.PARAMS_WHITE
    xs = X3[:len(comps)]
    k = np.ones(len(comps)) if powers is None else np.asarray(powers, float)
    ref = 1.0
    for axis in range(3):
        def integrand(z, axis=axis):
            return np.prod([np.exp(-kj * (x[axis] - z) ** 2
                                   / (2.0 * p.s[a] ** 2))
                            for a, kj, x in zip(comps, k, xs)])
        ref *= quad(integrand, -30.0, 30.0, epsabs=1e-15, epsrel=1e-13,
                    limit=200)[0]
    got = sh.spatial_overlap(comps, xs, p, powers=powers)[0]
    assert got == pytest.approx(ref, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("comps,ts", [((0, 1), (1.4, 0.9)),
                                      ((1, 0, 1), (1.7, 1.1, 0.6))])
def test_t_tilde_matches_quadrature(comps, ts):
    """``T̃ = ∫ds Π_j J_{a_j}(t_j, s)`` for white pulses with a rate per
    component and for exponential pulses."""
    pw = sh.PARAMS_WHITE_MATRIX
    tau = np.array(ts) - pw.t_min
    ref = quad(lambda s: np.prod([np.exp(-pw.gamma[a] * (t - s))
                                  for a, t in zip(comps, tau)]),
               0.0, tau.min(), epsabs=1e-15, epsrel=1e-13)[0]
    assert sh.t_tilde(comps, np.array(ts), pw)[0] == pytest.approx(
        ref, rel=1e-12, abs=0.0)

    pe = sh.PARAMS_EXP
    tau = np.array(ts) - pe.t_min

    def J(a, t, s):
        lo = max(0.0, s)
        if lo >= t:
            return 0.0
        return quad(lambda u: np.exp(-pe.gamma[0] * (t - u)
                                     - (u - s) * pe.rates[a]),
                    lo, t, epsabs=1e-15, epsrel=1e-13)[0]

    ref = quad(lambda s: np.prod([J(a, t, s) for a, t in zip(comps, tau)]),
               -40.0, tau.min(), epsabs=1e-16, epsrel=1e-13, limit=200,
               points=[0.0])[0]
    assert sh.t_tilde(comps, np.array(ts), pe)[0] == pytest.approx(
        ref, rel=1e-11, abs=0.0)


@pytest.mark.parametrize("variant", sorted(d7.PULSES))
def test_hierarchy_reproduces_C_and_the_three_point_function(variant):
    """At ``F = 0`` the hierarchy's ``(0, 0)`` and ``(0, 1)`` coefficients are
    the closed-form C and ``K_R``."""
    p = d7.PULSES[variant]
    H2 = sh.ShotHierarchy(p, X3[:2])
    for a, b in itertools.product(range(2), repeat=2):
        got = H2.moments([(a, 0), (b, 1)], [(0, 0)], T)[(0, 0)]
        ref = sh.C_matrix(X3[0], p.t_min + T, X3[1], p.t_min + T, p)[0, a, b]
        assert got == pytest.approx(ref, rel=1e-11, abs=0.0)
    H3 = sh.ShotHierarchy(p, X3)
    for comps in [(0, 1, 1), (1, 0, 1)]:
        got = H3.moments([(c, i) for i, c in enumerate(comps)], [(0, 1)],
                         T)[(0, 1)]
        ref = sh.K_R(comps, X3, np.full(3, p.t_min + T), p)[0]
        assert got == pytest.approx(ref, rel=1e-11, abs=0.0)


def test_the_kernel_is_asymmetric_in_points_and_components():
    """Permuting the points at fixed components, or the components at fixed
    points, changes ``K_R`` by more than 1 %: a leg-order or routing error
    cannot hide behind a symmetry."""
    p = sh.PARAMS_EXP
    ts = np.array([1.7, 1.2, 0.8])
    base = sh.K_R((0, 1, 1), X3, ts, p)[0]
    assert _rel(sh.K_R((0, 1, 1), X3[[2, 0, 1]], ts[[2, 0, 1]], p)[0],
                base) > 1e-2
    assert _rel(sh.K_R((1, 1, 0), X3, ts, p)[0], base) > 1e-2
    joint = sh.K_R((1, 0, 1), X3[[2, 0, 1]], ts[[2, 0, 1]], p)[0]
    assert joint == pytest.approx(base, rel=1e-12, abs=0.0)


# ---------------------------------------------------------------------------
# The package against the hierarchy
# ---------------------------------------------------------------------------

ROUTES = {"white": [d7.GL(16), d7.QV(13), d7.QS(10)],
          "exponential": [d7.GL(16), d7.QV(13)],
          "white-matrixR": [d7.QS(11), d7.GL(16)]}
TOL = {"gauss_legendre": 1e-10, "qmc_vectorized": 1e-4, "qmc_scalar": 3e-3}


@pytest.fixture(scope="module")
def runs():
    """``{variant: rows}`` for the two-point pair (0, 1) and the triple
    (0, 1, 1)."""
    return {v: d7.run(v, pairs=((0, 1),), triples=((0, 1, 1),),
                      routes=ROUTES[v]) for v in ROUTES}


@pytest.mark.parametrize("variant", sorted(ROUTES))
def test_package_matches_the_hierarchy(runs, variant):
    for row in runs[variant]:
        if row["status"] == "refused":
            # Matrix R on the batched backends: being added elsewhere.
            continue
        assert row["error"] <= TOL[row["method"]], row


def test_the_three_dimensions_are_really_used():
    """Evaluating at the points' first coordinates only changes the order-2
    FK3 channel by more than 10 %: the 3-D positions reach the callable and
    the closed-form C, and a projection would not pass the check above."""
    p = sh.PARAMS_WHITE
    system = ss.make_system(p, f_amplitude=1.0, cumulants=(3,))
    props = ss.propagators_for(system, p, t_max=p.t_min + T + 0.3)
    exp = system.expand(d7.TWO, orders=[2], diag_C=False, progress=False)
    kw = dict(t_final=p.t_min + T, component_pair=(0, 1), orders=[2],
              vertex_types={"FK3"}, method="gauss_legendre", n_gauss=16)
    full = exp.evaluate(props, positions=POINTS, **kw).total
    flat = exp.evaluate(props, positions={k: np.array([v[0]])
                                          for k, v in POINTS.items()},
                        **kw).total
    assert _rel(flat, full) > 0.1
