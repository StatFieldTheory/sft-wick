r"""Demo 3 --- filtered-Poisson (shot) noise: cumulants and R-contracted kernels.

Every closed form in ``examples/demo3/shot_noise.py`` is pinned against an
**independent** reference:

===========================  =================================================
``X_m``                      adaptive quadrature of ``∫dx' Π_i w(x_i − x')``
``κ₂ / κ₃ / κ₄``             direct Monte Carlo of the *event process*
``κ₂``                       the package's own ``ClosedFormC`` (T̃₂ route)
``K_R = ν h^m X_m T̃_m``     ``m``-dimensional quadrature of the raw leg
                             integral ``∫Π du_i R(t_i,u_i) κ_m(u)``, split by
                             which ``u_i`` is smallest so the ``|u_i − u_min|``
                             kinks sit on region boundaries
``t_tilde`` dispatch         the two branches compared *directly* at the
                             threshold, per the boundary-validation rule
===========================  =================================================

The Monte-Carlo tests assert both ``|estimate − closed form| < 4 SE`` **and**
``SE < 10 % of the closed form``, so a broken formula cannot pass by hiding
behind a large error bar.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy import integrate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "demo3"))

import shot_noise as sn          # noqa: E402
import simulate as sim           # noqa: E402

P = sn.PARAMS


# =====================================================================
# Independent references
# =====================================================================

def _X_m_quadrature(xs, sigma_x):
    """``∫dx' Π_i e^{−|x_i−x'|/σ_x}`` by adaptive quadrature."""
    xs = np.asarray(xs, dtype=float)
    lo, hi = xs.min() - 60 * sigma_x, xs.max() + 60 * sigma_x
    val, _ = integrate.quad(
        lambda xp: np.exp(-np.abs(xs - xp).sum() / sigma_x),
        lo, hi, points=list(xs), limit=400)
    return val


def _leg_integral_reference(tvals, p, n_gl=24):
    r"""``T̃_m`` by direct ``m``-dimensional quadrature of the leg integral.

    ``∫ Π_i du_i R(t_i, u_i) T_m(u)`` over ``u_i ∈ [0, t_i]``.  ``T_m``
    kinks on the ``u_i = u_j`` planes, so the domain is split by *which*
    ``u_i`` is the minimum; on each region the integrand is a smooth
    product of exponentials and a tensor Gauss-Legendre rule converges
    exponentially.  Shares no code with the ``s``-factorisation.
    """
    tvals = np.asarray(tvals, dtype=float)
    m = tvals.size
    x, w = leggauss(n_gl)
    xu, wu = 0.5 * (x + 1.0), 0.5 * w          # nodes/weights on [0, 1]
    total = 0.0
    for jmin in range(m):
        others = [i for i in range(m) if i != jmin]
        u_j = xu * tvals.min()                  # (n,)
        jac_j = wu * tvals.min()
        grids, jac = [u_j], [jac_j]
        for i in others:                        # u_i ∈ [u_j, t_i]
            span = tvals[i] - u_j               # (n,) depends on u_j
            grids.append(span[:, None] * xu[None, :] + u_j[:, None])
            jac.append(span[:, None] * wu[None, :])
        # broadcast: axis 0 is u_j, axes 1.. are the others
        u = np.empty((m,) + (n_gl,) * m)
        shape_j = (n_gl,) + (1,) * (m - 1)
        u[jmin] = u_j.reshape(shape_j)
        weight = jac_j.reshape(shape_j)
        for k, i in enumerate(others):
            sh = [1] * m
            sh[0], sh[k + 1] = n_gl, n_gl
            u[i] = grids[k + 1].reshape(sh)
            weight = weight * jac[k + 1].reshape(sh)
        integ = np.exp(-p.gamma * (tvals.reshape((m,) + (1,) * m) - u)).prod(axis=0)
        integ = integ * sn.T_m(u.reshape(m, -1), p.sigma_t).reshape((n_gl,) * m)
        total += float((weight * integ).sum())
    return total


# =====================================================================
# X_m
# =====================================================================

@pytest.mark.parametrize("xs", [
    [0.0, 0.0], [0.0, 0.6], [0.0, 0.0, 0.0], [0.0, 0.7, -1.3],
    [0.0] * 4, [0.0, 0.3, 0.3, 1.9], [-2.0, -0.5, 0.5, 2.0],
    [0.0, 0.0, 1.0, 1.0, 3.0],
])
def test_X_m_matches_quadrature(xs):
    """The piecewise-analytic spatial overlap is exact for every ``m``."""
    got = float(sn.X_m(np.array(xs, float)[:, None], P.sigma_x)[0])
    ref = _X_m_quadrature(xs, P.sigma_x)
    assert got == pytest.approx(ref, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("m", [2, 3, 4, 5, 6])
def test_X_m_coincident_is_two_sigma_over_m(m):
    """Coincident points: ``X_m = 2 σ_x / m`` --- the source of the ``1/m²``
    in ``κ_m`` and hence of the ``1/√n`` skewness law."""
    got = float(sn.X_m(np.zeros((m, 1)), P.sigma_x)[0])
    assert got == pytest.approx(2.0 * P.sigma_x / m, rel=1e-14, abs=0.0)


def test_T_m_coincident_and_shift():
    """``T_m = (σ_t/m) e^{−Σ(t_i−t_min)/σ_t}``, and ``T₂`` is ``e^{−|Δt|/σ_t}``."""
    assert float(sn.T_m(np.zeros((3, 1)), P.sigma_t)[0]) == pytest.approx(P.sigma_t / 3, abs=0.0)
    dt = 0.4
    got = float(sn.T_m(np.array([[0.0], [dt]]), P.sigma_t)[0])
    assert got == pytest.approx(P.sigma_t / 2 * np.exp(-dt / P.sigma_t), rel=1e-14, abs=0.0)


# =====================================================================
# Cumulants vs Monte Carlo of the event process
# =====================================================================

@pytest.mark.parametrize("m,xs,ts", [
    (2, [0.0, 0.0], [0.0, 0.0]),
    (2, [0.0, 0.6], [0.0, 0.0]),
    (2, [0.0, 0.0], [0.0, 0.4]),
    (3, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    (3, [0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
    (3, [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]),
])
def test_kappa_m_matches_event_monte_carlo(m, xs, ts):
    """Campbell's-theorem cumulants against a direct MC of the event process.

    For ``m ≤ 3`` the joint cumulant equals the joint *central* moment, so
    the estimator needs only the (exactly known, truncated-window) mean.
    """
    rng = np.random.default_rng(20260902 + m)
    window = sim.Window.for_times(max(ts), P)
    samples = sim.sample_points(rng, xs, ts, 120_000, window, P, kind="eta")[:, 0, :]
    est, se = sim.central_moments(samples, m)
    exact = float(sn.kappa_m(np.array(xs, float)[:, None],
                             np.array(ts, float)[:, None], P)[0])
    assert se < 0.1 * abs(exact), "MC error bar too wide for the test to bite"
    assert abs(est - exact) < 4.0 * se, f"{est} vs {exact} ({abs(est-exact)/se:.1f} sigma)"


def test_kappa4_matches_event_monte_carlo():
    """The connected fourth cumulant, the channel that drives ``ξ_aa``."""
    rng = np.random.default_rng(4004)
    window = sim.Window.for_times(0.0, P)
    samples = sim.sample_points(rng, [0.0] * 4, [0.0] * 4, 400_000, window, P,
                                kind="eta")[:, 0, :]
    est, se = sim.connected_cumulant(samples)
    exact = float(sn.kappa_m(np.zeros((4, 1)), np.zeros((4, 1)), P)[0])
    assert se < 0.1 * abs(exact)
    assert abs(est - exact) < 4.0 * se, f"{est} vs {exact} ({abs(est-exact)/se:.1f} sigma)"


def test_window_truncation_bound_is_negligible():
    """The only approximation in the exact simulation, bounded analytically.

    The ``m``-th cumulant is an ``m``-fold product of pulse profiles, so
    the discarded window costs ``e^{−m L/σ_x}`` --- ``m`` times faster than
    the field's own tail.
    """
    window = sim.Window.for_times(5.0, P)
    assert sim.truncation_bound(window, 3, P)["total"] < 1e-9
    assert sim.truncation_bound(window, 2, P)["total"] < 1e-6


# =====================================================================
# kappa2 against the package's own closed-form C
# =====================================================================

@pytest.mark.parametrize("t1,t2,r", [
    (1.0, 1.0, 0.0), (2.0, 0.7, 0.0), (5.0, 5.0, 0.9),
    (0.3, 0.3, 0.0), (12.0, 9.0, 2.0),
])
def test_kappa2_reproduces_package_closed_form_C(t1, t2, r):
    """``ν h² X₂(r) T̃₂(t₁,t₂)`` is the package's ``C`` --- and the package
    reaches it through a completely different derivation.

    Also locks that ``builtin_closed_form_for`` *accepts* this system: the
    temporal kernel is exponential and the (non-exponential) spatial
    envelope factors out, so no C quadrature ever runs.
    """
    from sft_wick.workflow.closed_forms import builtin_closed_form_for
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "demo3"))
    import system as dsys

    closed = builtin_closed_form_for(dsys.make_system(P))
    assert closed is not None, "the built-in closed form must accept demo 3's kernel"
    pkg = float(np.asarray(closed(0.0, t1, r, t2))[0, 0])
    mine = float(sn.K_R(np.array([[0.0], [r]]), np.array([[t1], [t2]]), P)[0])
    assert mine == pytest.approx(pkg, rel=1e-13, abs=0.0)


# =====================================================================
# K_R against the raw leg integral
# =====================================================================

@pytest.mark.parametrize("tvals", [
    [1.0, 1.0, 1.0], [3.0, 1.5, 1.5], [2.0, 2.0, 0.5],
    [0.2, 0.2, 0.2], [12.0, 3.0, 0.7], [20.0, 20.0, 20.0],
])
def test_t_tilde_matches_leg_integral_m3(tvals):
    """``T̃₃`` against direct 3-D quadrature, incl. distinct and large ``t'``."""
    got = float(sn.t_tilde(np.array(tvals, float)[:, None], P)[0])
    ref = _leg_integral_reference(tvals, P, n_gl=28)
    assert got == pytest.approx(ref, rel=1e-9, abs=0.0)


@pytest.mark.parametrize("tvals", [
    [1.0, 1.0, 1.0, 1.0], [2.0, 1.3, 0.9, 3.1], [8.0, 8.0, 8.0, 8.0],
])
def test_t_tilde_matches_leg_integral_m4(tvals):
    """``T̃₄`` against direct 4-D quadrature."""
    got = float(sn.t_tilde(np.array(tvals, float)[:, None], P)[0])
    ref = _leg_integral_reference(tvals, P, n_gl=24)
    assert got == pytest.approx(ref, rel=1e-8, abs=0.0)


def test_leg_integral_reference_is_converged():
    """The reference itself must be converged, or the tests above are vacuous."""
    for tvals in ([1.0, 1.0, 1.0], [3.0, 1.5, 1.5]):
        lo = _leg_integral_reference(tvals, P, n_gl=18)
        hi = _leg_integral_reference(tvals, P, n_gl=28)
        assert lo == pytest.approx(hi, rel=1e-10, abs=0.0)


def test_K_R_factorises_into_spatial_and_temporal():
    """``K_R`` is ``ν h^m X_m(x') T̃_m(t')`` --- the factorisation that makes
    the outer integral of an FK diagram one-dimensional."""
    xs = np.array([[0.0], [0.6], [1.3]])
    ts = np.array([[1.0], [2.0], [0.7]])
    got = float(sn.K_R(xs, ts, P)[0])
    expect = (P.nu * P.h ** 3 * float(sn.X_m(xs, P.sigma_x)[0])
              * float(sn.t_tilde(ts, P)[0]))
    assert got == pytest.approx(expect, rel=1e-15, abs=0.0)


# =====================================================================
# Boundary validation of the t_tilde dispatcher
# =====================================================================

_CORNERS = [
    (3, [0.02, 0.02, 0.02]), (3, [1.0, 1.0, 1.0]), (3, [40.0, 40.0, 40.0]),
    (3, [40.0, 2.0, 0.05]), (4, [1.0] * 4), (4, [40.0] * 4), (4, [0.02] * 4),
    (4, [25.0, 3.0, 0.6, 0.02]),
]


def _t_tilde_mpmath(tvals, p, dps=60):
    """``T̃_m`` by the *same* closed form in 60-digit arithmetic.

    Extended precision removes the ``(γ−a)^{−m}`` cancellation entirely,
    so this is the arbiter that says which double-precision branch is
    right in a corner where they disagree.
    """
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = dps
    a, g = mpmath.mpf(1) / mpmath.mpf(p.sigma_t), mpmath.mpf(p.gamma)
    eps = g - a
    t = [mpmath.mpf(x) for x in tvals]
    m, T = len(t), min(t)
    out = mpmath.mpf(1)
    for x in t:
        out *= (mpmath.exp(-a * x) - mpmath.exp(-g * x)) / eps
    out /= m * a
    acc = mpmath.mpf(0)
    for mask in itertools.product((0, 1), repeat=m):
        k = sum(mask)
        b = a * k + g * (m - k)
        c = sum((a if mask[i] else g) * t[i] for i in range(m))
        E = sum((a if mask[i] else g) * (t[i] - T) for i in range(m))
        acc += (-1) ** (m - k) * (mpmath.exp(-E) - mpmath.exp(-c)) / b
    return float(out + acc / eps ** m)


def _p_at(mult, p=P):
    """A parameter set with ``γ = mult · (1/σ_t)``.

    Parametrising by the *ratio* rather than by ``|γ − a|/max(γ,a)`` keeps
    ``γ > 0``: the latter sends ``γ → 0`` (no drift at all) as it
    approaches 1, which is not a point in the model's parameter space.
    ``mult = 1`` is the removable singularity.
    """
    a = 1.0 / p.sigma_t
    return sn.ShotNoise(nu=p.nu, h=p.h, sigma_t=p.sigma_t, sigma_x=p.sigma_x,
                        gamma=a * mult)


@pytest.mark.parametrize("mult", [0.1, 0.5, 0.7, 0.9, 0.99, 0.9999, 1.0,
                                  1.5, 3.0])
@pytest.mark.parametrize("m,tvals", _CORNERS)
def test_auto_dispatch_matches_extended_precision(mult, m, tvals):
    """The dispatcher must be right at *every* corner, including the ones
    where each individual branch fails.

    This is the boundary check in its strongest form: rather than trust a
    threshold in ``|γ − a|``, both branches are evaluated directly and the
    dispatcher's choice is compared against 60-digit arithmetic.  The
    closed form degrades at small ``t'`` (its terms dwarf the result); the
    quadrature degrades where its panel stack is short.  ``mult = 1`` is
    the exactly-removable singularity ``γ = 1/σ_t``.
    """
    p = _p_at(mult)
    arr = np.array(tvals, float)[:, None]
    got = float(sn.t_tilde(arr, p)[0])
    assert np.isfinite(got)
    if mult == 1.0:
        # No closed form to compare against: the quadrature branch is
        # pinned separately by test_degenerate_limit_against_independent_analytic_form.
        # abs=0.0 is required, exactly as on the mpmath comparison below:
        # at the (m=3, t=[40, 2, 0.05]) corner the compared value is
        # 2.3e-37, so approx's 1e-12 default absolute floor would enforce
        # 4.3e+24 relative -- i.e. nothing at all.  The two branches are
        # bit-identical at every corner, so rel=1e-12 is comfortable.
        assert got == pytest.approx(
            float(sn.t_tilde_quad(arr, p)[0]), rel=1e-12, abs=0.0)
        return
    ref = _t_tilde_mpmath(tvals, p)
    assert got == pytest.approx(ref, rel=1e-10, abs=0.0), (
        f"gamma/a={mult} t={tvals}: auto={got!r} ref={ref!r}")


@pytest.mark.parametrize("mult", [0.1, 0.5, 0.7, 0.9, 0.99, 1.5, 3.0])
@pytest.mark.parametrize("m,tvals", _CORNERS)
def test_closed_form_error_estimate_is_an_upper_bound(mult, m, tvals):
    """The load-bearing property of the design: the closed form's *own*
    conditioning estimate must not under-report its true error.

    ``est = ε · Σ_S|term_S| / |result|`` is the cancellation amplification
    factor; if it were optimistic the dispatcher would keep a bad value.
    A factor-4 slack is allowed for the estimate being a bound on a sum of
    rounding errors rather than a realisation of them.
    """
    p = _p_at(mult)
    arr = np.array(tvals, float)[:, None]
    value, est = sn.t_tilde_closed(arr, p, return_error=True)
    ref = _t_tilde_mpmath(tvals, p)
    actual = abs(float(value[0]) - ref) / abs(ref)
    # An order of magnitude of slack.  The estimate is a *bound* on a sum
    # of rounding errors, not a prediction of their realisation, so the
    # ratio is expected to sit below 1 most of the time and to exceed it
    # occasionally; the worst measured over this grid is 3.45, at
    # gamma/a = 0.1 and t = [40, 2, 0.05].  A tighter factor would be
    # asserting a rounding outcome rather than the bound.
    assert actual <= 10.0 * max(float(est[0]), 1e-16), (
        f"estimate {float(est[0]):.2e} under-reports actual {actual:.2e}")


@pytest.mark.parametrize("m,tvals", _CORNERS)
def test_quad_branch_is_converged(m, tvals):
    """The quadrature branch is the fallback, so its own convergence matters.

    Node count is varied at fixed panel stack: an error that does *not*
    shrink with ``n_gl`` is panel truncation, not quadrature order --- the
    signature that caught the original off-by-one in ``_panel_edges``.
    """
    p = _p_at(1.0)                                   # exactly degenerate
    arr = np.array(tvals, float)[:, None]
    lo = float(sn.t_tilde_quad(arr, p)[0])           # the shipped node count
    hi = float(sn.t_tilde_quad(arr, p, n_gl=24)[0])
    assert lo == pytest.approx(hi, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("m", [3, 4, 5])
@pytest.mark.parametrize("T", [40.0, 100.0])
def test_quad_branch_matches_the_exact_saturated_limit(m, T):
    r"""At ``γ = 1/σ_t`` and large ``T``, ``T̃_m`` has an exactly-known limit.

    ``J(t,s) → v e^{−a v}`` with ``v = T − s``, so as ``T → ∞``

        ``T̃_m → ∫_0^∞ v^m e^{−m a v} dv = m! / (m a)^{m+1}``.

    A *pin* against a closed form, which is much stronger than a
    self-convergence check --- and it is what showed that 8 nodes per
    panel left 4.3e-11 here while 12 is exact.  This is precisely the
    corner where the quadrature is the only available method, since the
    closed form is 0/0 at ``γ = a``.
    """
    import math
    a = 1.0 / P.sigma_t
    p = _p_at(1.0)
    exact = math.factorial(m) / (m * a) ** (m + 1)
    got = float(sn.t_tilde_quad(np.full((m, 1), T), p)[0])
    assert got == pytest.approx(exact, rel=1e-13, abs=0.0)


@pytest.mark.parametrize("m,tvals", _CORNERS)
def test_quad_branch_panel_stack_reaches_the_tail(m, tvals):
    """Doubling the covered decades must not move the answer."""
    p = _p_at(0.7)
    arr = np.array(tvals, float)[:, None]
    base = float(sn.t_tilde_quad(arr, p)[0])
    old_dec = sn.DECADES
    try:
        sn.DECADES = 2 * old_dec
        wide = float(sn.t_tilde_quad(arr, p)[0])
    finally:
        sn.DECADES = old_dec
    assert base == pytest.approx(wide, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("offset", [0.0, 1e-9, -1e-9, 1e-6, -1e-6])
def test_degenerate_gamma_is_finite_and_continuous(offset):
    """``γ → 1/σ_t`` is a removable singularity: the dispatcher must stay
    finite and continuous through it."""
    a = 1.0 / P.sigma_t
    p = sn.ShotNoise(nu=P.nu, h=P.h, sigma_t=P.sigma_t, sigma_x=P.sigma_x,
                     gamma=a + offset)
    arr = np.array([[1.0], [2.0], [0.5]])
    got = float(sn.t_tilde(arr, p)[0])
    ref = float(sn.t_tilde_quad(arr, sn.ShotNoise(
        nu=P.nu, h=P.h, sigma_t=P.sigma_t, sigma_x=P.sigma_x, gamma=a))[0])
    assert np.isfinite(got)
    # Continuity as a Lipschitz statement rather than a fixed tolerance:
    # T̃ genuinely moves by ~1.48·|δγ| here, so a constant bound would
    # either be vacuous at small offsets or fail at large ones.
    assert abs(got - ref) / ref <= 5.0 * abs(offset) + 1e-14


def test_degenerate_limit_against_independent_analytic_form():
    """At ``γ = a`` exactly, ``J(t,s) = (t−s)e^{−a(t−s)}`` --- integrate that
    directly and compare, sharing no code with :func:`shot_noise.t_tilde_quad`."""
    a = 1.0 / P.sigma_t
    p = sn.ShotNoise(nu=P.nu, h=P.h, sigma_t=P.sigma_t, sigma_x=P.sigma_x, gamma=a)
    for tvals in ([1.0, 1.0, 1.0], [2.0, 1.0, 0.5], [1.5] * 4):
        m = len(tvals)
        tail = (np.prod([t * np.exp(-a * t) for t in tvals]) / (m * a))
        body, _ = integrate.quad(
            lambda s: np.prod([(t - s) * np.exp(-a * (t - s)) for t in tvals]),
            0.0, min(tvals), epsabs=1e-16, epsrel=1e-13, limit=300)
        got = float(sn.t_tilde(np.array(tvals, float)[:, None], p)[0])
        assert got == pytest.approx(tail + body, rel=1e-10, abs=0.0)


def test_auto_dispatch_uses_the_closed_form_where_it_is_sound():
    """At the demo's own parameters the fast branch is chosen for the whole
    time range, so ``auto`` costs one closed-form evaluation."""
    assert P.rel_split == pytest.approx(0.5, abs=0.0)
    ts = np.linspace(0.2, 30.0, 200)[None, :].repeat(3, axis=0)
    _, est = sn.t_tilde_closed(ts, P, return_error=True)
    assert np.all(est < sn.AUTO_TOL), f"worst estimate {est.max():.2e}"
    # Agreement, not bit-identity: if a sample did cross the tolerance the
    # dispatcher would repair it with the quadrature and the ANSWER would
    # be unchanged, which is the part worth pinning.  Requiring identical
    # bits would make this test depend on the branch choice, which is a
    # rounding outcome.
    assert np.allclose(sn.t_tilde(ts, P), sn.t_tilde_closed(ts, P),
                       rtol=1e-12, atol=0.0)


def test_auto_dispatch_repairs_the_small_time_corner():
    """Below ``t' ~ 0.1`` the closed form's terms dwarf the result, so the
    dispatcher must hand those samples to the quadrature --- automatically,
    at the demo's own ``γ``."""
    ts = np.full((3, 1), 0.02)
    _, est = sn.t_tilde_closed(ts, P, return_error=True)
    assert est[0] > sn.AUTO_TOL
    # Tolerance audit (issue #5): ``abs`` used to default to 1e-12 on a
    # quantity of ~1.3e-6, so this enforced ~8e-7 relative, not the 1e-15
    # written.  The repair is a straight assignment of the quadrature value
    # into the bad slot, so the two agree bit-for-bit --- ``abs=0.0`` puts the
    # written 1e-15 actually in force (a ~9-order-of-magnitude tightening).
    assert sn.t_tilde(ts, P)[0] == pytest.approx(
        float(sn.t_tilde_quad(ts, P)[0]), rel=1e-15, abs=0.0)


def test_auto_dispatch_repairs_only_the_bad_samples():
    """A mixed batch: good samples keep the closed-form value, bad ones are
    replaced --- the repair is per-sample, not all-or-nothing."""
    p = _p_at(0.7)
    ts = np.array([[1.0, 0.02]] * 3, dtype=float)
    _, est = sn.t_tilde_closed(ts, p, return_error=True)
    assert est[0] < sn.AUTO_TOL < est[1], "the fixture must contain one of each"
    auto = sn.t_tilde(ts, p)
    assert auto[0] == float(sn.t_tilde_closed(ts, p)[0])
    # Tolerance audit (issue #5): same story as the previous test --- the
    # default ``abs=1e-12`` floor made this ~8e-7 relative on a ~1.2e-6
    # quantity.  The bad sample is recomputed by the vectorised quadrature on
    # the bad *subset*, which is elementwise-identical to the full-batch call,
    # so ``abs=0.0`` holds the written 1e-15 (was ~8e-7 effective).
    assert auto[1] == pytest.approx(float(sn.t_tilde_quad(ts, p)[1]), rel=1e-15,
                                    abs=0.0)


# =====================================================================
# Scaling laws and vertex structure
# =====================================================================

@pytest.mark.parametrize("n", [0.25, 1.0, 4.0, 16.0])
def test_non_gaussianity_scaling_laws(n):
    """``n = ν σ_t σ_x`` is the *only* knob: skewness ``∝ 1/√n``, excess
    kurtosis ``∝ 1/n``, and ``κ₂`` held fixed by compensating ``h``."""
    q = P.with_n(n)
    assert q.n_dimensionless == pytest.approx(n, abs=0.0)
    assert q.variance == pytest.approx(P.variance, rel=1e-12, abs=0.0)
    assert q.skewness == pytest.approx(0.6285393610547089 / np.sqrt(n), rel=1e-12, abs=0.0)
    assert q.excess_kurtosis == pytest.approx(0.5 / n, rel=1e-12, abs=0.0)


def test_skewness_and_kurtosis_match_the_cumulant_formulas():
    """The advertised shape statistics are the cumulant ratios, not fits."""
    k2 = float(sn.kappa_m(np.zeros((2, 1)), np.zeros((2, 1)), P)[0])
    k3 = float(sn.kappa_m(np.zeros((3, 1)), np.zeros((3, 1)), P)[0])
    k4 = float(sn.kappa_m(np.zeros((4, 1)), np.zeros((4, 1)), P)[0])
    assert k2 == pytest.approx(P.variance, rel=1e-14, abs=0.0)
    assert k3 / k2 ** 1.5 == pytest.approx(P.skewness, rel=1e-12, abs=0.0)
    assert k4 / k2 ** 2 == pytest.approx(P.excess_kurtosis, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("m", [3, 4, 5])
def test_coupling_tensor_is_component_diagonal(m):
    """Each component is an independent event process, so ``κ_m ∝ δ_{a_1…a_m}``."""
    fn = sn.RContractedCoupling(m=m, params=P)
    K = fn(np.zeros((m, 3)), np.ones((m, 3)))
    assert K.shape == (3,) + (P.n_components,) * m
    amp = float(sn.K_R(np.zeros((m, 1)), np.ones((m, 1)), P)[0])
    for idx in itertools.product(range(P.n_components), repeat=m):
        expect = amp if len(set(idx)) == 1 else 0.0
        assert K[(0,) + idx] == pytest.approx(expect, rel=1e-14, abs=1e-300)


def test_kappa_ladder_ratio_closed_form():
    """``κ_m/κ₃ = h^{m−3}(3/m)²`` at coincident points --- the neglected-cumulant
    ladder, in closed form, for the level-B error budget."""
    k3 = float(sn.kappa_m(np.zeros((3, 1)), np.zeros((3, 1)), P)[0])
    for m in (4, 5, 6):
        km = float(sn.kappa_m(np.zeros((m, 1)), np.zeros((m, 1)), P)[0])
        assert km / k3 == pytest.approx(sn.kappa_ratio(m, P), rel=1e-13, abs=0.0)


def test_per_sample_and_vectorised_couplings_agree():
    """The two ``NonLocalVertex`` contracts must return the same tensor."""
    n_list, t_list = [0.0, 0.4, 1.1], [1.0, 2.0, 0.5]
    per = sn.coupling_k3(n_list, t_list, P)
    vec = sn.coupling_k3_vectorized(np.array(n_list)[:, None],
                                    np.array(t_list)[:, None], P)[0]
    assert np.allclose(per, vec, rtol=1e-15, atol=0.0)
