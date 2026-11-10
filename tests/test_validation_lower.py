"""The lowering of a ModelSpec onto sft-wick (``validation/lower.py``, with
the closed form of C in ``validation/lowering_parts.py``) and the comparison
with the oracle (``validation/compare.py``).

* The closed-form integrals of C against mpmath, near resonant rates, at
  negative rates and at large rate times time; each method of the
  triangle integral and of its moments against the others at every switch
  between them, with the switch bypassed.
* C before ``t_min``, the closed-form C against the oracle's order-0
  two-time function, and the noise callables the package integrates
  against the closed-form C by an independent quadrature.
* The package against the oracle on one fixed model per lowered feature,
  channel by channel, with Gauss-Legendre at its declared tolerance; the
  C routes; the joint cumulant against the scale of its cancellation.
* The comparison rules, the declared tolerances, and negative controls that
  a shifted model fails.
"""
from __future__ import annotations

import dataclasses
import math

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import dblquad, quad

import validation.lower as lw
import validation.lowering_parts as lp
from validation import demo_cases as dc
from validation.compare import (TABLE, TOLERANCES, Tolerance, compare_values,
                                tolerance)
from validation.hierarchy import moments, moments_with_scale, tags_up_to
from validation.modelspec import (Cumulant, Kernel, Leg, ModelSpec,
                                  Multiplicative, Observable, OUBlock, Poisson)

# ---------------------------------------------------------------------------
# Closed-form integrals
# ---------------------------------------------------------------------------


def _coloured_mp(al, be, p, q, T1, T2):
    with mp.workdps(30):
        al, be, p, q, T1, T2 = map(mp.mpf, (al, be, p, q, T1, T2))

        def inner(s):
            m = min(s, T2)
            a = mp.quad(lambda u: mp.e ** (-be * (T2 - u) - p * (s - u)),
                        [0, m])
            b = (mp.quad(lambda u: mp.e ** (-be * (T2 - u) - q * (u - s)),
                         [m, T2]) if m < T2 else 0)
            return mp.e ** (-al * (T1 - s)) * (a + b)

        return float(mp.quad(inner, [0, min(T1, T2), T1] if T2 < T1
                             else [0, T1]))


@pytest.mark.parametrize("al,be,p,q,T1,T2", [
    (0.7, 1.3, 1.6, 1.6, 1.4, 0.9), (0.7, 1.3, 1.6, 1.6, 0.9, 1.4),
    (1.0, 1.0, 1.0, 1.0, 1.2, 1.2), (0.7, 1.3, 0.7, 1.3, 1.1, 0.4),
    (0.7, 1.3, 0.7 + 1e-9, 1.3 - 1e-7, 1.1, 0.4),
    (-0.3, 1.3, 1.1, 0.8, 1.5, 1.0), (0.0, 0.0, 0.5, 0.5, 2.0, 1.0),
    (5.0, 4.0, 2.0, 9.0, 3.0, 2.5),
    (49.97, 50.0, 50.0, 50.0, 20.0, 20.0),
    (-0.3, -0.3, 200.0, 200.0, 20.0, 20.0),
    (0.01, 0.01, 50.0, 50.0, 20.0, 20.0)])
def test_VL1_coloured_integral_matches_mpmath(al, be, p, q, T1, T2):
    got = lp.coloured_integral(al, be, p, q, np.array([T1]), np.array([T2]))
    assert got[0] == pytest.approx(_coloured_mp(al, be, p, q, T1, T2),
                                   rel=1e-13, abs=0.0)


def _triangle_mp(x, y, T):
    """``∫∫_{0 ≤ a ≤ b ≤ T} e^{−xa − yb}`` in closed form, at 60 digits more
    than ``x`` needs to survive in ``x + y``."""
    extra = 0 if x == 0 else max(0, math.ceil(math.log10(
        max(abs(y) * T, 1.0) / abs(x * T))))
    with mp.workdps(60 + extra):
        x, y, T = map(mp.mpf, (x, y, T))

        def E(k):
            return T if k == 0 else -mp.expm1(-k * T) / k

        if x == 0:
            return float(T * T / 2 if y == 0 else
                         (1 - (1 + y * T) * mp.e ** (-y * T)) / (y * y))
        return float((E(y) - E(x + y)) / x)


def _arr(*values):
    return tuple(np.array([float(v)]) for v in values)


def _dispatcher_matches_mpmath_on_both_sides(x, y, T):
    for xi in (x * (1 - 1e-9), x * (1 + 1e-9)):
        assert lp._triangle(*_arr(xi, y, T)).item() == pytest.approx(
            _triangle_mp(xi, y, T), rel=1e-13, abs=0.0)


@pytest.mark.parametrize("y,T", [(0.0, 1.0), (1.0, 1.0), (1.5, 2.0),
                                 (-2.0, 1.5), (1e-9, 5.0), (0.4, 1e-6),
                                 (3.9, 1.0), (0.15, 20.0), (-20.0, 2.0),
                                 (-2.0, 20.0), (0.0, 1e-9)])
@pytest.mark.parametrize("sign", [1, -1])
def test_VL2_triangle_methods_agree_at_the_rate_time_switch(y, T, sign):
    """At ``|xT| = TRIANGLE_SWITCH`` (``yT < 4``, so the rate-ratio switch
    does not apply) the difference formula and the series both hold."""
    x = sign * lp.TRIANGLE_SWITCH / T
    want = _triangle_mp(x, y, T)
    far = lp._triangle_far(*_arr(x, y, T, x + y)).item()
    series = lp._triangle_series(*_arr(x, y, T)).item()
    assert far == pytest.approx(want, rel=1e-13, abs=0.0)
    assert series == pytest.approx(want, rel=1e-13, abs=0.0)
    _dispatcher_matches_mpmath_on_both_sides(x, y, T)


@pytest.mark.parametrize("yT", [4.0, 50.0, 1e3, 1e4, 1e6])
@pytest.mark.parametrize("T", [1.0, 20.0])
@pytest.mark.parametrize("sign", [1, -1])
def test_VL3_triangle_methods_agree_at_the_rate_ratio_switch(yT, T, sign):
    """At ``|x| = y/RATE_RATIO`` with ``|xT| ≥ TRIANGLE_SWITCH`` the
    difference formula loses at most a factor 9 and the series converges
    as ``RATE_RATIO^{−n}``; inside, the difference would cancel as ``yT``."""
    y = yT / T
    x = sign * y / lp.RATE_RATIO
    want = _triangle_mp(x, y, T)
    far = lp._triangle_far(*_arr(x, y, T, x + y)).item()
    series = lp._triangle_series(*_arr(x, y, T)).item()
    assert far == pytest.approx(want, rel=1e-13, abs=0.0)
    assert series == pytest.approx(want, rel=1e-13, abs=0.0)
    _dispatcher_matches_mpmath_on_both_sides(x, y, T)


def _moment_mp(k, z):
    with mp.workdps(50):
        z = mp.mpf(z)
        if z == 0:
            return float(mp.mpf(1) / (k + 1))
        if z > 0:
            return float(mp.factorial(k) / z ** (k + 1)
                         * mp.gammainc(k + 1, 0, z, regularized=True))
        return float(mp.quad(lambda s: s ** k * mp.e ** (-z * s), [0, 1]))


def _all_moments_match(got, K, z, rel=1e-14):
    for k in range(1, K + 1):
        assert got[k - 1] == pytest.approx(_moment_mp(k, z), rel=rel, abs=0.0)


@pytest.mark.parametrize("K", [1, 2, 8, 20])
def test_VL4_moment_methods_agree_at_their_switches(K):
    """``z = K`` (upwards against downwards), ``z = 1`` (the top moment from
    the series against the incomplete gamma function) and ``z = 0`` (the
    series against the downward recurrence), each method called directly,
    then the dispatcher at the extremes."""
    for z in (K * (1 - 1e-9), float(K), K * (1 + 1e-9)):
        zs = np.array([z])
        _all_moments_match(lp._moments_upward(K, zs)[:, 0], K, z)
        _all_moments_match(lp._moments_downward(K, zs)[:, 0], K, z)
    for z in (1 - 1e-9, 1.0, 1 + 1e-9):
        zs = np.array([z])
        want = _moment_mp(K, z)
        assert lp._moments_series(K, zs)[K - 1, 0] == pytest.approx(
            want, rel=1e-14, abs=0.0)
        assert lp._moment_gamma(K, zs)[0] == pytest.approx(
            want, rel=1e-14, abs=0.0)
    for z in (0.0, 1e-300, -1e-12):
        zs = np.array([z])
        _all_moments_match(lp._moments_series(K, zs)[:, 0], K, z)
        if z >= 0:
            _all_moments_match(lp._moments_downward(K, zs)[:, 0], K, z)
    zs = np.array([-300.0, -40.0, -1.0, 0.5, 2.5, 37.0, 1e6])
    got = lp._moments(K, zs)
    for j, z in enumerate(zs):
        _all_moments_match(got[:, j], K, z, rel=3e-15 * max(1.0, -z / 40))


def test_VL5_white_integral_and_the_zero_rate():
    T1, T2 = np.array([1.4, 0.3]), np.array([0.9, 2.0])
    got = lp.white_integral(0.0, 0.0, T1, T2)
    assert got == pytest.approx(np.minimum(T1, T2), rel=1e-15, abs=0.0)
    got = lp.white_integral(-0.4, 0.4, np.array([1.0]), np.array([2.0]))
    assert got[0] == pytest.approx(np.exp(-0.4), rel=1e-14, abs=0.0)


# ---------------------------------------------------------------------------
# Closed-form C: before t_min, against the oracle, against the callables
# ---------------------------------------------------------------------------

GAUSSIAN = ModelSpec(
    gamma=(0.7, 1.3),
    ou=(OUBlock(0.45, 0.6, Kernel("exponential", (0.8,)), (1.0, -0.4)),
        OUBlock(0.3, 1.1, Kernel("legendre", (0.25, 0.6, 0.3)), (0.2, 0.9))),
    white=((0.30, 0.10), (0.10, 0.20)),
    poisson=Poisson(1.2, (0.9, -0.6), (0.5, 0.9), "exponential", (0.35, 1.8)),
    t_min=0.35)
X3, Y3 = (0.2, -0.3, 0.5), (-0.4, 0.6, 0.1)


def test_VL6_c_is_zero_before_t_min_and_earlier_times_are_refused():
    c = lp.closed_form_c(GAUSSIAN)
    t0 = GAUSSIAN.t_min
    times = np.array([t0 - 2.0, t0 - 1e-9, t0, t0 + 0.4])
    got = c(np.array([X3] * 4), times, np.array([Y3] * 4), np.full(4, 1.2))
    assert np.all(got[:3] == 0.0) and np.all(np.abs(got[3]) > 0.0)
    assert np.all(c(np.array(X3), 1.2, np.array(Y3), t0 - 0.5) == 0.0)
    blowing = lp.closed_form_c(ModelSpec(
        gamma=(-50.0,), ou=(OUBlock(1.0, 0.5, Kernel("constant", ()), (1.0,)),),
        white=((0.1,),)))
    for t1, t2 in ((20.0, -1.0), (20.0, 0.0), (-1.0, 20.0)):
        assert blowing(0.0, t1, 0.0, t2)[0, 0] == 0.0
    pos = {"x": X3, "y": Y3}
    for obs in (Observable([(0, "x"), (1, "y")], pos, t0 - 0.1),
                Observable([(0, "x"), (1, "y")], pos, 1.2,
                           times={"y": t0 - 0.1})):
        with pytest.raises(ValueError, match="before t_min"):
            lw.Lowered(GAUSSIAN, obs, 0)


@pytest.mark.parametrize("pulse", ["white", "exponential"])
def test_VL7_closed_form_c_is_the_free_two_time_function(pulse):
    p = GAUSSIAN.poisson
    model = dataclasses.replace(GAUSSIAN, poisson=Poisson(
        p.nu, p.amplitude, p.width, pulse,
        p.tau if pulse == "exponential" else None))
    c = lp.closed_form_c(model)
    times = [(1.6, 1.6), (1.6, 0.9), (0.9, 1.6), (0.35, 1.2)]
    batch = c(np.array([X3] * 4), np.array([t for t, _ in times]),
              np.array([Y3] * 4), np.array([s for _, s in times]))
    assert batch.shape == (4, 2, 2)
    for k, (t1, t2) in enumerate(times):
        single = c(np.array(X3), t1, np.array(Y3), t2)
        np.testing.assert_allclose(single, batch[k], rtol=1e-15, atol=0.0)
        for a in range(2):
            for b in range(2):
                obs = Observable([(a, "x"), (b, "y")], {"x": X3, "y": Y3},
                                 max(t1, t2), times={"x": t1, "y": t2})
                want = moments(model, obs, [(0,)])[(0,)]
                assert single[a, b] == pytest.approx(want, rel=1e-12,
                                                     abs=1e-16)


def _c_by_quadrature(model, a, b, x, t1, y, t2):
    """``∫∫ R_a κ²_ab R_b + ∫ R_a σ²_ab R_b`` from the callables the package
    is given, with the square split on its diagonal."""
    kappa2, sigma2 = lp.Kappa2Of(model), lp.Sigma2Of(model)
    g, t0 = model.gamma, model.t_min

    def coloured(u, s):
        return (math.exp(-g[a] * (t1 - s) - g[b] * (t2 - u))
                * kappa2(x, s, y, u)[a, b])

    opts = dict(epsabs=0.0, epsrel=1e-12)
    below = dblquad(coloured, t0, t1, lambda s: t0, lambda s: min(s, t2),
                    **opts)[0]
    above = dblquad(coloured, t0, t1, lambda s: min(s, t2), lambda s: t2,
                    **opts)[0]
    white = quad(lambda s: math.exp(-g[a] * (t1 - s) - g[b] * (t2 - s))
                 * sigma2(x, s, y)[a, b], t0, min(t1, t2), **opts)[0]
    return below + above + white


@pytest.mark.parametrize("model", [
    ModelSpec(gamma=(0.7, 1.6), ou=GAUSSIAN.ou[:1], poisson=GAUSSIAN.poisson,
              t_min=0.35),
    ModelSpec(gamma=(0.7, 1.6), white=GAUSSIAN.white,
              poisson=Poisson(1.2, (0.9, -0.6), (0.5, 0.9), "white", None),
              t_min=0.35)], ids=["coloured", "white"])
def test_VL8_the_noise_callables_integrate_to_the_closed_form(model):
    """``Kappa2Of`` and ``Sigma2Of`` feed the package's tables and its
    diagonal probes; ``ClosedFormC`` is written separately."""
    c = lp.closed_form_c(model)
    x, y = np.array([0.2, -0.3]), np.array([-0.4, 0.6])
    for t1, t2 in ((1.6, 0.9), (0.9, 1.6)):
        want = c(x, t1, y, t2)
        for a in range(2):
            for b in range(2):
                got = _c_by_quadrature(model, a, b, x, t1, y, t2)
                assert got == pytest.approx(want[a, b], rel=1e-9, abs=0.0)


# ---------------------------------------------------------------------------
# Package against the oracle, one fixed model per feature
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
F = RNG.normal(scale=0.4, size=(2, 2, 2))
G = RNG.normal(scale=0.2, size=(2, 2, 2, 2))
OU = (OUBlock(0.45, 0.6, Kernel("exponential", (0.8,)), (1.0, -0.4)),)
POS = {"x": 0.0, "y": 0.9, "z": -0.5}
ONE_POINT = {"x": 0.0, "y": 0.0}
VECTORS = {"x": X3, "y": Y3, "z": (0.3, 0.1, -0.6)}

FEATURES = {
    "source, F, G": (ModelSpec(gamma=(0.7, 1.3), source=(0.3, -0.5),
                               quadratic=F, cubic=G, ou=OU, t_min=0.3), 2,
                     POS),
    "static S1, S2, S3": (ModelSpec(
        gamma=(0.7, 1.3), quadratic=F, cumulants=(
            Cumulant(1, (0.3, -0.2)), Cumulant(2, ((0.4, 0.1), (0.1, 0.25))),
            Cumulant(3, RNG.normal(size=(2, 2, 2))))), 2, POS),
    "white W2, W3": (ModelSpec(
        gamma=(0.7, 1.3), quadratic=F, ou=OU, cumulants=(
            Cumulant(2, ((0.3, -0.1), (-0.1, 0.2)), "white"),
            Cumulant(3, RNG.normal(size=(2, 2, 2)), "white"))), 2, POS),
    "Poisson white P3, P4": (ModelSpec(
        gamma=(0.8, 1.3), quadratic=F, poisson=Poisson(
            1.2, (0.9, -0.6), (0.5, 0.9), "white", None, (3, 4))), 2,
        VECTORS),
    "Poisson exponential P3": (ModelSpec(
        gamma=(1.0, 1.0), quadratic=F, poisson=Poisson(
            1.2, (0.9, -0.6), (0.5, 0.9), "exponential", (0.35, 1.8), (3,)),
        t_min=0.2), 2, POS),
    "multiplicative Stratonovich and white": (ModelSpec(
        gamma=(0.7, 1.6), quadratic=F, white=((0.2, 0.05), (0.05, 0.1)),
        multiplicative=Multiplicative(RNG.normal(scale=0.4, size=(2, 2)),
                                      RNG.normal(scale=0.3, size=(2, 2, 2)),
                                      "stratonovich")), 2, ONE_POINT),
    "every Gaussian block in 3-D, three points": (GAUSSIAN, 1, VECTORS),
    "one component": (ModelSpec(
        gamma=(0.8,), source=(0.3,), quadratic=[[[0.4]]], cubic=[[[[-0.2]]]],
        ou=(OUBlock(0.45, 0.6, Kernel("exponential", (0.8,)), (1.0,)),),
        white=((0.25,),), t_min=0.3), 2, POS),
    "diagonal blocks under the default diag_C": (ModelSpec(
        gamma=(0.7, 1.3), quadratic=F,
        ou=(OUBlock(0.45, 0.6, Kernel("exponential", (0.8,)), (1.0, 0.0)),
            OUBlock(0.3, 1.1, Kernel("legendre", (0.25, 0.6, 0.3)),
                    (0.0, 0.9))),
        white=((0.3, 0.0), (0.0, 0.2)), cumulants=(
            Cumulant(2, ((0.3, -0.1), (-0.1, 0.2)), "white"),)), 2, VECTORS),
}


def _observable(model, pos, comps=None):
    comps = comps or ((0, 1, 1) if "z" in pos else (0, 1))
    comps = tuple(min(c, model.n_components - 1) for c in comps)
    return Observable(list(zip(comps, "xyz")), pos, model.t_min + 1.3)


@pytest.mark.parametrize("name", list(FEATURES))
def test_VL9_package_matches_the_oracle_channel_by_channel(name):
    model, weight, pos = FEATURES[name]
    obs = _observable(model, pos)
    low = lw.Lowered(model, obs, weight)
    got = low.evaluate("gauss_legendre", n_gauss=16)
    ref = moments(model, obs, tags_up_to(model, weight))
    rows = compare_values(ref, got, tolerance("gauss_legendre", "oracle"))
    assert any(sum(t) and abs(v) > 1e-8 for t, (v, _) in got.items()), got
    bad = [r for r in rows if not r.ok]
    assert not bad, bad
    if name.startswith("diagonal"):
        assert low.flags["diag_C"]


def test_VL10_two_times_integrated_legs_and_the_connected_target():
    """The connected target is compared against the scale of its own
    cancellation.  The last model's three-leg raw moment is 0 at every tag,
    and the oracle's cumulant keeps a residue that a zero scale would fail.
    On a static model the scale is the largest product ``κ(B) m(S∖B)``,
    above every raw moment, and is pinned by hand."""
    model = FEATURES["source, F, G"][0]
    cases = [
        (model, Observable([(0, "x"), (1, "y")], POS, 1.6, times={"y": 1.0})),
        (model, Observable([(0, "x"), (1, "y")], POS, 1.6,
                           integrate_over=("x",))),
        (model, Observable([(1, "x"), (0, "y"), (1, "z")], POS, 1.6,
                           target="connected")),
        (ModelSpec(gamma=(0.7, 1.2), cumulants=(
            Cumulant(1, (0.3, -0.2)), Cumulant(2, ((0.3, 0.1), (0.1, 0.2))))),
         Observable([(0, "x"), (1, "y"), (0, "z")],
                    {"x": 0.0, "y": 0.0, "z": 0.0}, 1.0, target="connected")),
    ]
    for m, obs in cases:
        got = lw.Lowered(m, obs, 2).evaluate("gauss_legendre", n_gauss=16)
        ref, scale = moments_with_scale(m, obs, tags_up_to(m, 2))
        rows = compare_values(ref, got, tolerance("gauss_legendre", "oracle"),
                              scale=scale)
        assert all(r.ok for r in rows), [r for r in rows if not r.ok]
    assert max(abs(v) for v in ref.values()) > 0.0 and got == {}
    static = ModelSpec(gamma=(1.0, 1.0), cumulants=(
        Cumulant(1, (3.0, 3.0)), Cumulant(2, ((20.0, -15.0), (-15.0, 20.0)))))
    _, scale = moments_with_scale(static, cases[-1][1],
                                  tags_up_to(static, 2))
    r = 1.0 - math.exp(-1.0)   # the response of each leg to a static source
    # κ({0, 2}) at S2 is r² C₀₀ = 20 r², m({1}) at S1 is r μ₁ = 3 r; the
    # largest raw moment, r² C₀₀, is smaller.
    assert scale == pytest.approx(60.0 * r ** 3, rel=1e-12, abs=0.0)


def test_VL11_the_c_routes_and_the_flags_they_admit():
    """Each C source takes its own route; the built-in closed form under
    ``diag_C`` and ``iso_C``, and beside multiplicative noise where white
    noise is folded into extra ``g0`` columns; the tables at a coarse grid;
    the refusal of ``diag_C`` on loaded blocks."""
    model = dc.demo5_model("isotropic")
    obs = Observable([(0, "x"), (0, "y")], {"x": 0.0, "y": 0.7}, 2.1)
    assert lp.diagonal_c(model) and not lp.diagonal_c(GAUSSIAN)
    low = lw.Lowered(model, obs, 2, dict(iso_C=True), "auto")
    assert low.props.c_source == "closed_form:builtin"
    got = low.evaluate("gauss_legendre", n_gauss=12)
    ref = moments(model, obs, [(0,), (1,), (2,)])
    assert all(r.ok for r in compare_values(
        ref, got, tolerance("gauss_legendre", "auto")))
    mult, weight, pos = FEATURES["multiplicative Stratonovich and white"]
    obs = _observable(mult, pos)
    ref = moments(mult, obs, tags_up_to(mult, weight))
    for source, route in (("oracle", "closed_form:user"),
                          ("auto", "closed_form:builtin")):
        low = lw.Lowered(mult, obs, weight, c_source=source)
        assert low.props.c_source == route
        rows = compare_values(ref, low.evaluate("gauss_legendre", n_gauss=16),
                              tolerance("gauss_legendre", source))
        assert all(r.ok for r in rows), [r for r in rows if not r.ok]
    white = ModelSpec(gamma=(0.7, 1.3), quadratic=F,
                      white=((0.3, 0.0), (0.0, 0.2)), t_min=0.2)
    obs = Observable([(0, "x"), (1, "y")], {"x": 0.0, "y": 0.9}, 1.5)
    low = lw.Lowered(white, obs, 2, c_source="table", n_grid_t=9)
    assert low.props.c_source.startswith("quadrature")
    rows = compare_values(moments(white, obs, tags_up_to(white, 2)),
                          low.evaluate("gauss_legendre", n_gauss=12), TABLE)
    assert all(r.ok for r in rows), [r for r in rows if not r.ok]
    loaded = FEATURES["source, F, G"][0]
    assert not lp.diagonal_c(loaded)
    with pytest.raises(ValueError, match="diag_C"):
        lw.Lowered(loaded, _observable(loaded, POS), 0, dict(diag_C=True))


def test_VL12_orders_and_the_legs_the_lowering_refuses():
    """A subset of orders is complete per tag except for the ``g`` channel,
    whose vertices carry weight 1 or 2; response legs are not lowered."""
    model, weight, pos = FEATURES["source, F, G"]
    low = lw.Lowered(model, _observable(model, pos), weight)
    full = low.evaluate("gauss_legendre", n_gauss=12)
    part = low.evaluate("gauss_legendre", orders=[1], n_gauss=12)
    assert part and all(sum(t) == 1 and part[t] == full[t] for t in part)
    mult, weight, pos = FEATURES["multiplicative Stratonovich and white"]
    low = lw.Lowered(mult, _observable(mult, pos), weight)
    assert low.evaluate("gauss_legendre", orders=[0, 1, 2], n_gauss=8)
    with pytest.raises(ValueError, match="subset of orders"):
        low.evaluate("gauss_legendre", orders=[1], n_gauss=8)
    with pytest.raises(NotImplementedError, match="response legs"):
        lw.Lowered(model, Observable([Leg(0, "x", "psi"), (1, "y")], POS,
                                     1.6), 1)
    pulses = dataclasses.replace(mult, poisson=Poisson(
        1.2, (0.9, -0.6), (0.5, 0.9), "white", None))
    with pytest.raises(NotImplementedError, match="white Poisson"):
        lw.Lowered(pulses, _observable(pulses, pos), 1)


def test_VL13_errors_of_one_tag_add_in_quadrature():
    model, weight, pos = FEATURES["source, F, G"]
    low = lw.Lowered(model, _observable(model, pos), weight)
    settings = dict(n_samples=2 ** 10, seed=1)
    rows = low.rows("qmc_vectorized", **settings)
    got = low.evaluate("qmc_vectorized", **settings)
    shared = 0
    for tag, (value, error) in got.items():
        mine = [(v, e) for t, v, e in rows if t == tag]
        assert value == pytest.approx(sum(v for v, _ in mine), rel=1e-15,
                                      abs=1e-300)
        assert error == pytest.approx(math.sqrt(sum(e * e for _, e in mine)),
                                      rel=1e-15, abs=0.0)
        shared += sum(e > 0 for _, e in mine) >= 2
    assert shared, "no tag where a sum, a maximum and quadrature differ"


# ---------------------------------------------------------------------------
# Comparison rules, declared tolerances, negative controls
# ---------------------------------------------------------------------------


def test_VL14_comparison_rules():
    tol = Tolerance(rel=1e-10, k_err=10.0, floor=1e-12)
    oracle = {(0,): 0.5, (1,): 0.0}
    rows = compare_values(oracle, {(0,): (0.5 + 4e-11, 0.0),
                                   (1,): (3e-13, 0.0)}, tol)
    assert [r.ok for r in rows] == [True, True]
    for sign in (1, -1):
        row = compare_values(oracle, {(0,): (0.5 + sign * 1e-6, 0.0)}, tol)[0]
        assert not row.ok and row.ratio > 1
    rows = compare_values(oracle, {(0,): (0.5, 1e-6)}, tol)
    assert rows[1].package == 0.0 and rows[0].allowed > 1e-5
    rows = compare_values(oracle, {(1,): (2e-12, 0.0)}, tol)
    assert not rows[1].ok and rows[1].ratio > 1
    small = compare_values({(0,): 1e-6, (1,): 0.0}, {
        (0,): (1e-6, 0.0), (1,): (1e-15, 0.0)},
        Tolerance(rel=1e-10, k_err=0.0, floor=1e-12))
    assert small[0].ok and not small[1].ok
    with pytest.raises(ValueError, match="no oracle value"):
        compare_values(oracle, {(2,): (1.0, 0.0)}, tol)
    for package in ({(0,): (math.inf, 0.0), (1,): (0.5, 0.0)},
                    {(0,): (0.5, math.inf)}, {(0,): (math.nan, 0.0)}):
        rows = compare_values(oracle, package, tol)
        assert not all(r.ok for r in rows), package
    rows = compare_values({(0,): 0.5, (1,): 0.25},
                          {(0,): (math.inf, 0.0), (1,): (0.25, 0.0)}, tol)
    assert [r.ok for r in rows] == [False, True]
    rows = compare_values(oracle, {(0,): (0.5, 0.0), (1,): (1.0, 0.0)}, tol,
                          scale=math.inf)
    assert not any(r.ok for r in rows)
    rows = compare_values({(0,): math.inf}, {(0,): (math.inf, 0.0)}, tol)
    assert not rows[0].ok


def test_VL15_the_declared_tolerances():
    assert TOLERANCES == {
        ("gauss_legendre", "closed"): Tolerance(1e-10, 0.0, 1e-12),
        ("nquad", "closed"): Tolerance(1e-7, 10.0, 1e-12),
        ("qmc_vectorized", "closed"): Tolerance(1e-9, 10.0, 1e-8),
        ("qmc_scalar", "closed"): Tolerance(1e-9, 10.0, 1e-8),
        ("qmc", "closed"): Tolerance(1e-9, 10.0, 1e-8)}
    assert TABLE == Tolerance(1e-4, 10.0, 1e-6)
    assert tolerance("nquad", "auto") == tolerance("nquad", "oracle")
    assert tolerance("gauss_legendre", "table") == TABLE
    with pytest.raises(ValueError, match="no tolerance"):
        tolerance("simpson", "oracle")


@pytest.mark.parametrize("method,settings,shift,target", [
    ("gauss_legendre", dict(n_gauss=12), 1e-8, "moment"),
    ("nquad", {}, 1e-5, "moment"),
    ("gauss_legendre", dict(n_gauss=12), 1e-8, "connected")])
def test_VL16_a_shifted_model_fails_the_declared_tolerance(method, settings,
                                                            shift, target):
    """The joint cumulant is compared at the scale of its cancellation, so
    a scale that grew would let the shift through."""
    model = ModelSpec(gamma=(0.7, 1.3), source=(0.3, -0.5), quadratic=F,
                      white=((0.3, 0.1), (0.1, 0.2)), t_min=0.3)
    legs = [(0, "x"), (1, "y")] + ([(1, "z")] if target == "connected"
                                   else [])
    obs = Observable(legs, POS, 1.6, target=target)
    got = lw.Lowered(model, obs, 2).evaluate(method, **settings)
    tol = tolerance(method, "oracle")
    tags = tags_up_to(model, 2)
    ref, scale = moments_with_scale(model, obs, tags)
    assert all(r.ok for r in compare_values(ref, got, tol, scale=scale))
    shifted = dataclasses.replace(model, quadratic=F * (1 + shift))
    ref, scale = moments_with_scale(shifted, obs, tags)
    rows = compare_values(ref, got, tol, scale=scale)
    assert not all(r.ok for r in rows)
