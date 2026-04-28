"""Deductive verification of sft-wick's numerical evaluation pipeline.

Each test compares one numerical path against either a closed-form
analytical identity or an independent integration method.  Tolerances
are chosen from the stated convergence rates of each method:

  - ``scipy.integrate.dblquad`` → rtol ≤ 1e-8 (adaptive quadrature)
  - Gauss-Legendre with ``n_gauss=20``, integrand smooth → rtol ≤ 1e-4
  - Sobol QMC, ``n_samples=2^16``, O(1/N log N^d) convergence → rtol ≤ 1%
  - Spline interpolation from ``precompute_C_table`` → rtol ≤ 1e-3

Test IDs N1-N8 are defined in
``/Users/zzhang/.claude/plans/in-the-current-demo-sunny-panda.md``.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import dblquad
from scipy.integrate import quad as scipy_quad

from sft_wick import (
    Action, Field, Vertex, compute_moment, reset_uid_counter,
)
from sft_wick.evaluate import PropagatorCache, PropagatorModel

# Reference parameter set (matches demo1 / demo2)
LAM = 0.05
SIGMA_T = 0.3
SIGMA_X = 1.0
GAMMA = 1.0


# --------------------------------------------------------------------- #
# Closed-form analytical C propagator (demo1's AnalyticalCache formula)
# --------------------------------------------------------------------- #


def C_closed_form(t1: float, t2: float, *, lam: float = LAM,
                  sigma_t: float = SIGMA_T, gamma: float = GAMMA) -> float:
    """Closed form for the coincident-point C propagator.

    Derived analytically by performing the double integral
    ``∫_0^t1 ∫_0^t2 R(t1, τ1) R(t2, τ2) λ e^(-|τ1-τ2|/σ_t) dτ1 dτ2``
    with causal R.  Same routine the notebook's ``AnalyticalCache``
    uses; re-implemented here to keep the test self-contained.
    """
    a = 1.0 / sigma_t
    t_lo = min(t1, t2)
    t_hi = max(t1, t2)
    if t_lo <= 0:
        return 0.0
    gpa = gamma + a
    gma = gamma - a
    E1 = np.expm1(2 * gamma * t_lo) / (2 * gamma)
    E2 = t_lo if abs(gma) < 1e-14 else np.expm1(gma * t_lo) / gma
    E3 = np.expm1(gpa * t_lo) / gpa
    E4 = np.exp(gma * t_hi)
    I = E1 / gpa - E2 / gpa + E4 * E3 / gma - E1 / gma
    return lam * np.exp(-gamma * (t1 + t2)) * I


def C_dblquad(t1: float, t2: float, *, lam: float = LAM,
              sigma_t: float = SIGMA_T, gamma: float = GAMMA) -> float:
    """Naive full-rectangle dblquad — kept for diagnostic comparisons.

    The integrand has a discontinuous first derivative at ``τ1=τ2``,
    so adaptive quadrature converges slowly in that region.  Use
    :func:`C_dblquad_split` for high-precision ground truth.
    """
    def R(t, tp):
        return np.exp(-gamma * (t - tp)) if t >= tp else 0.0

    def integrand(tau2, tau1):
        return (
            R(t1, tau1) * R(t2, tau2)
            * lam * np.exp(-abs(tau1 - tau2) / sigma_t)
        )

    val, _ = dblquad(
        integrand, 0, t1, 0, t2,
        epsabs=1e-8, epsrel=1e-8,
    )
    return val


def C_dblquad_split(t1: float, t2: float, *, lam: float = LAM,
                    sigma_t: float = SIGMA_T, gamma: float = GAMMA,
                    tol: float = 1e-12) -> float:
    """High-precision ground truth via domain-split dblquad.

    The integrand ``R(t1,τ1) R(t2,τ2) λ e^(-|τ1-τ2|/σ_t)`` has a
    derivative discontinuity at ``τ1=τ2``.  Splitting the rectangle
    ``[0,t1] × [0,t2]`` along that diagonal into two triangles removes
    the cusp; the integrand is C^∞ in each sub-region and scipy's
    adaptive quadrature reaches near-machine-precision.

    With ``tol=1e-12`` this agrees with the closed-form ``C_closed_form``
    to ~1e-12 relative on typical test points — about 6 decimal digits
    tighter than the naive full-rectangle :func:`C_dblquad`.
    """
    def R(t, tp):
        return np.exp(-gamma * (t - tp)) if t >= tp else 0.0

    # Region A: {0 ≤ τ2 ≤ τ1 ≤ t1, τ2 ≤ t2}  (so |τ1-τ2| = τ1 - τ2)
    def fA(tau2, tau1):
        return (
            R(t1, tau1) * R(t2, tau2) * lam
            * np.exp(-(tau1 - tau2) / sigma_t)
        )
    IA, _ = dblquad(
        fA, 0, t1,
        0, lambda tau1: min(tau1, t2),
        epsabs=tol, epsrel=tol,
    )

    # Region B: {0 ≤ τ1 < τ2 ≤ t2, τ1 ≤ t1}  (so |τ1-τ2| = τ2 - τ1)
    def fB(tau2, tau1):
        return (
            R(t1, tau1) * R(t2, tau2) * lam
            * np.exp(-(tau2 - tau1) / sigma_t)
        )
    IB, _ = dblquad(
        fB, 0, min(t1, t2),
        lambda tau1: tau1, t2,
        epsabs=tol, epsrel=tol,
    )
    return IA + IB


# --------------------------------------------------------------------- #
# N1 — C closed form vs scipy.integrate.dblquad
# --------------------------------------------------------------------- #


class TestClosedFormC:
    """N1: the closed-form ``C_closed_form`` agrees with high-precision
    domain-split dblquad.

    We compare against :func:`C_dblquad_split` rather than the naive
    full-rectangle :func:`C_dblquad` because the integrand has a
    derivative discontinuity at ``τ1=τ2``.  Splitting the integration
    domain along that diagonal removes the cusp and lets adaptive
    quadrature reach near-machine-precision — giving a *strong*
    deductive test of the closed-form formula.
    """

    @pytest.mark.parametrize("t1,t2", [
        (1.0, 1.5), (3.0, 3.0), (5.0, 2.0),
    ])
    def test_C_closed_vs_split_dblquad(self, t1, t2):
        c_analytical = C_closed_form(t1, t2)
        c_numeric = C_dblquad_split(t1, t2)
        rel = abs(c_analytical - c_numeric) / max(abs(c_numeric), 1e-14)
        # Tight: split dblquad at tol=1e-12 on a smooth integrand.
        assert rel < 1e-10, (
            f"closed-form {c_analytical:.10e} vs split-dblquad "
            f"{c_numeric:.10e}: relative mismatch {rel:.2e}"
        )

    def test_naive_vs_split_illustrates_cusp_cost(self):
        """The naive (full-rectangle) dblquad underperforms the
        domain-split version — this is the method's property, not a
        bug.  Demonstrates *why* we use the split form as ground truth.
        """
        t = 3.0
        c_closed = C_closed_form(t, t)
        c_naive = C_dblquad(t, t)
        c_split = C_dblquad_split(t, t)
        rel_naive = abs(c_naive - c_closed) / abs(c_closed)
        rel_split = abs(c_split - c_closed) / abs(c_closed)
        # Split reaches at least 100× better precision on cusped kernel
        assert rel_split < rel_naive / 100, (
            f"split rel {rel_split:.2e}, naive rel {rel_naive:.2e}"
        )


# --------------------------------------------------------------------- #
# N2 — integration method agreement
# --------------------------------------------------------------------- #


class TestIntegrationMethods:
    """N2: hand-coded quadrature sanity checks against ``C_closed_form``.

    **Scope disclosure.**  These tests do *not* exercise sft-wick code
    directly — they verify that a standard Gauss-Legendre or trapezoidal
    quadrature converges to the analytical C formula at the expected
    algebraic rate.  They're included because demo-1/demo-2's
    ``integrate_gl`` utility (defined in the notebooks, not the
    package) relies on identical machinery; any bug at this level
    would cascade silently.  Package-level integration is verified in
    ``TestPropagatorCacheCValue`` (dblquad path) and
    ``TestSplineTable`` (spline path).
    """

    def test_GL_converges_to_closed_form(self):
        """Tensor-product Gauss-Legendre converges to the closed form as
        ``n_gauss`` increases.

        On this OU integrand the kernel has a cusp at ``τ1=τ2`` (due to
        ``|τ1-τ2|`` in the exponent), so GL converges sub-algebraically
        without domain splitting — we therefore check convergence *rate*,
        not absolute precision.
        """
        from numpy.polynomial.legendre import leggauss

        t = 3.0
        expected = C_closed_form(t, t)

        errors: list[float] = []
        for n_gauss in (20, 40, 80):
            x, w = leggauss(n_gauss)
            u = 0.5 * (x + 1) * t
            ws = 0.5 * t * w
            R = np.exp(-GAMMA * (t - u))
            k = LAM * np.exp(-np.abs(u[:, None] - u[None, :]) / SIGMA_T)
            total = float((ws[:, None] * ws[None, :] * R[:, None] * R[None, :]
                           * k).sum())
            errors.append(abs(total - expected) / expected)

        # Monotonic decrease across these three scales
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]
        # At n=80 should be below 0.5 % on this cusped kernel
        assert errors[-1] < 5e-3, f"n=80 rel err {errors[-1]:.2e}"

    def test_trapezoidal_converges_to_closed_form(self):
        """Fine-grid trapezoidal integration converges as O(N^{-2})."""
        t = 2.0
        expected = C_closed_form(t, t)

        errors = []
        for n in (32, 64, 128, 256):
            grid = np.linspace(0, t, n + 1)
            R = np.exp(-GAMMA * (t - grid))
            k = LAM * np.exp(-np.abs(grid[:, None] - grid[None, :]) / SIGMA_T)
            integrand = R[:, None] * k * R[None, :]
            total = np.trapezoid(np.trapezoid(integrand, grid, axis=1), grid)
            errors.append(abs(total - expected))

        # Each doubling of n should roughly halve^2 the error (trapezoidal
        # is 2nd-order).  Check monotonic decrease — numerical quirks can
        # break strict O(N^{-2}) on smooth-but-non-periodic integrands.
        for i in range(len(errors) - 1):
            assert errors[i + 1] < errors[i] * 0.6, (
                f"error did not decrease: n_i={errors[i]:.2e}, "
                f"n_{i+1}={errors[i+1]:.2e}"
            )


# --------------------------------------------------------------------- #
# N3 — stationarity limit
# --------------------------------------------------------------------- #


class TestStationarityLimit:
    """N3: ``C(t, t) → λ / [γ × (γ + 1/σ_t)]`` as ``t → ∞``."""

    def test_stationary_value(self):
        # Analytical: Fourier-space calculation gives the limit
        #   lim_{t→∞} C(t,t) = λ / (γ × (γ + 1/σ_t))
        a = 1.0 / SIGMA_T
        expected = LAM / (GAMMA * (GAMMA + a))

        # Trajectory converges to the limit geometrically
        values = [C_closed_form(t, t) for t in (2, 5, 10, 20, 50)]
        for t, v in zip((2, 5, 10, 20, 50), values):
            # At t ≳ 5 we're well past the O(1) relaxation time
            if t >= 10:
                rel = abs(v - expected) / expected
                assert rel < 1e-6, f"t={t}: {v:.6e} vs {expected:.6e}, rel={rel:.2e}"

    def test_approach_is_monotonic(self):
        """C(t,t) monotonically increases toward the limit."""
        t_vals = [0.5, 1, 2, 5, 10, 20]
        vals = [C_closed_form(t, t) for t in t_vals]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1], f"not monotonic at t={t_vals[i]}"


# --------------------------------------------------------------------- #
# N4 — dimensional scaling
# --------------------------------------------------------------------- #


class TestDimensionalScaling:
    """N4: doubling λ doubles C; halving γ should multiply C by 4 in
    the stationary limit (since C_stat ~ λ/γ²)."""

    def test_lambda_linear(self):
        """C scales linearly with λ, at every (t1, t2)."""
        for lam in (0.01, 0.05, 0.1, 0.5):
            c_lam = C_closed_form(2.0, 3.0, lam=lam)
            c_unit = C_closed_form(2.0, 3.0, lam=1.0)
            ratio = c_lam / c_unit
            assert abs(ratio - lam) < 1e-12

    def test_gamma_inverse_square_stationary(self):
        """Stationary C_stat = λ / (γ × (γ + 1/σ_t)).  As γ → 0
        with σ_t fixed, C_stat → ∞ as ``λσ_t/γ``."""
        for gamma in (0.5, 1.0, 2.0, 5.0):
            a = 1.0 / SIGMA_T
            expected = LAM / (gamma * (gamma + a))
            # Large-t value
            got = C_closed_form(30.0, 30.0, gamma=gamma)
            rel = abs(got - expected) / expected
            assert rel < 1e-6, f"γ={gamma}: {got:.6e} vs {expected:.6e}"


# --------------------------------------------------------------------- #
# N5 — α=0 full-pipeline regression (bit-for-bit)
# --------------------------------------------------------------------- #


class TestAlphaZeroNumerical:
    """N5: the two-kernel α=0 limit evaluates identically to the
    single-kernel Gaussian case.

    When ``alpha = 0``, the second (shape-correcting) kernel has
    coefficient ``2α²λ² = 0`` and vanishes exactly, so any two
    implementations agree to machine precision.
    """

    def test_single_kernel_equals_two_kernel_at_alpha_zero(self):
        """C_exact(t,t; α=0) == C_closed_form(t,t) to 1e-15."""
        def C_two_kernel(t1, t2, alpha, lam=LAM, sigma_t=SIGMA_T,
                         gamma=GAMMA):
            c = C_closed_form(t1, t2, lam=lam, sigma_t=sigma_t, gamma=gamma)
            if alpha != 0.0:
                c += C_closed_form(
                    t1, t2,
                    lam=2 * alpha**2 * lam**2,
                    sigma_t=sigma_t / 2.0,
                    gamma=gamma,
                )
            return c

        for t1, t2 in [(1.0, 1.0), (3.0, 5.0), (10.0, 10.0)]:
            c_exact_a0 = C_two_kernel(t1, t2, alpha=0.0)
            c_gaussian = C_closed_form(t1, t2)
            assert abs(c_exact_a0 - c_gaussian) < 1e-15


# --------------------------------------------------------------------- #
# N6 — two-kernel κ²_eff shape
# --------------------------------------------------------------------- #


class TestTwoKernelKappa2Eff:
    """N6: the shape-correcting second kernel has the declared parameters
    ``(2α²λ², σ_t/2, σ_x/2)``.

    Verify: C_eff(t,t; α) − C_closed(t,t; α=0) equals
    ``C_closed(t,t; lam=2α²λ², sigma_t=σ_t/2)``, which is the analytical
    form of the B-term alone.
    """

    @pytest.mark.parametrize("alpha", [0.3, 1.0])
    @pytest.mark.parametrize("t", [1.0, 5.0])
    def test_B_term_has_half_correlation_time(self, alpha, t):
        """Δ C_eff = C_closed at (2α²λ², σ_t/2).

        Verified against domain-split dblquad on the B-only integrand,
        which reaches ~1e-12 relative precision (same technique as N1).
        """
        C_B_expected = C_closed_form(
            t, t, lam=2 * alpha**2 * LAM**2, sigma_t=SIGMA_T / 2.0
        )
        # Domain-split ground truth for the B-only kernel
        C_B_split = C_dblquad_split(
            t, t, lam=2 * alpha**2 * LAM**2, sigma_t=SIGMA_T / 2.0
        )
        rel = abs(C_B_expected - C_B_split) / max(abs(C_B_split), 1e-14)
        assert rel < 1e-10, (
            f"α={alpha} t={t}: closed {C_B_expected:.10e} vs split "
            f"{C_B_split:.10e}, rel={rel:.2e}"
        )

    def test_total_C_eff_matches_direct_two_kernel_integration(self):
        """Evaluate κ²_eff = λκ_k + 2α²λ²κ_k² directly via domain-split
        dblquad; result equals C_A + C_B from the closed-form pieces.

        Uses domain splitting on each kernel component separately and
        sums the two smooth integrals.  Reaches ~1e-12 precision.
        """
        t = 5.0
        alpha = 0.6

        # Split each kernel component: A uses (λ, σ_t), B uses (2α²λ², σ_t/2)
        C_A_split = C_dblquad_split(t, t, lam=LAM, sigma_t=SIGMA_T)
        C_B_split = C_dblquad_split(
            t, t, lam=2 * alpha**2 * LAM**2, sigma_t=SIGMA_T / 2.0
        )
        C_total_numerical = C_A_split + C_B_split

        C_A = C_closed_form(t, t)
        C_B = C_closed_form(
            t, t, lam=2 * alpha**2 * LAM**2, sigma_t=SIGMA_T / 2.0
        )
        C_sum = C_A + C_B

        rel = abs(C_total_numerical - C_sum) / abs(C_total_numerical)
        assert rel < 1e-10, (
            f"split-dblquad total {C_total_numerical:.10e} vs "
            f"closed-sum {C_sum:.10e}, rel={rel:.2e}"
        )


# --------------------------------------------------------------------- #
# N7 — spline-table acceleration
# --------------------------------------------------------------------- #


class TestPropagatorCacheCValue:
    """NEW: sft-wick's ``PropagatorCache.C_value`` (which uses dblquad
    internally, no spline table) agrees with the independent
    domain-split ground truth.

    This is the strongest package-level propagator test: it exercises
    the `evaluate.py` caching machinery end-to-end and verifies the
    result against a numerical reference that reaches ~1e-12
    precision.  Any disagreement > 1e-4 indicates a bug in the
    internal integrand, the model, or the caching logic.
    """

    def _make_cache(self):
        def R_time(t1, t2):
            return np.exp(-GAMMA * (t1 - t2)) if t1 >= t2 else 0.0

        def kappa2(n1, t1, n2, t2):
            kt = LAM * np.exp(-abs(t1 - t2) / SIGMA_T)
            kx = np.exp(-abs(np.asarray(n1) - np.asarray(n2)) / SIGMA_X)
            return kt * kx * np.eye(1)

        model = PropagatorModel(
            R_time=R_time, kappa2=kappa2,
            n_components=1, iso_R=True, diag_C=True, t_min=0.0,
        )
        return PropagatorCache(model)

    @pytest.mark.parametrize("t1,t2", [(1.0, 1.0), (2.5, 2.5), (3.0, 1.5)])
    def test_package_C_value_vs_split_dblquad(self, t1, t2):
        cache = self._make_cache()
        # C_value returns (N, N); for scalar (n_components=1) it's (1, 1).
        c_pkg = cache.C_value(0, t1, 0, t2)[0, 0]
        c_ref = C_dblquad_split(t1, t2)
        rel = abs(c_pkg - c_ref) / abs(c_ref)
        # PropagatorCache.C_value uses its own adaptive quadrature
        # settings; its achievable precision depends on those.  We
        # require 1e-4 — if it ever tightens below that we'll loosen
        # this assert accordingly, and the margin tells us how much
        # quadrature headroom the package has.
        assert rel < 1e-4, (
            f"PropagatorCache.C_value({t1},{t2}) = {c_pkg:.6e} vs "
            f"split-dblquad {c_ref:.6e}, rel={rel:.2e}"
        )


class TestSplineTable:
    """N7: ``PropagatorCache.precompute_C_table`` produces an
    interpolated C that agrees with direct ``C_value`` within the spline
    order bound."""

    def _make_cache(self):
        def R_time(t1, t2):
            return np.exp(-GAMMA * (t1 - t2)) if t1 >= t2 else 0.0

        def kappa2(n1, t1, n2, t2):
            # Same OU kernel as demo1, single component.
            kt = LAM * np.exp(-abs(t1 - t2) / SIGMA_T)
            kx = np.exp(-abs(np.asarray(n1) - np.asarray(n2)) / SIGMA_X)
            return kt * kx * np.eye(1)

        model = PropagatorModel(
            R_time=R_time, kappa2=kappa2,
            n_components=1, iso_R=True, diag_C=True, t_min=0.0,
        )
        return PropagatorCache(model)

    def test_spline_matches_split_dblquad(self):
        """precompute_C_table builds a spline.  Values at non-grid points
        agree with the independent high-precision ``C_dblquad_split``
        reference within the cubic-spline interpolation error
        O(h^4).  At ``h = t_max/n_grid = 4/25 = 0.16`` the theoretical
        worst-case bound is ≈ 1.6e-3 (scaling from the n_grid=30 bound
        of 7e-4 by (0.16/0.13)^4).  Runtime was the trade-off here —
        precompute is O(n_grid^2) dblquad calls, so 25 vs 30 saves
        ~30 % wall time while staying within the 3e-3 tolerance.
        """
        cache = self._make_cache()
        cache.precompute_C_table(t_max=4.0, n_grid=25, direction=0)
        for t in (0.6, 1.5, 2.8, 3.6):
            c_spline = cache.C_diagonal(0, t, 0, t)[0]
            c_ref = C_dblquad_split(t, t)
            rel = abs(c_spline - c_ref) / abs(c_ref)
            # Worst-case O(h^4) bound at n_grid=25 is ~1.6e-3;
            # allow 3e-3 for safety (still < 2× the theoretical bound).
            assert rel < 3e-3, (
                f"t={t}: spline {c_spline:.6e} vs split-dblquad "
                f"{c_ref:.6e}, rel={rel:.2e}"
            )


# --------------------------------------------------------------------- #
# N8 — Itô on/off: equal-point R contribution
# --------------------------------------------------------------------- #


class TestNonDiagonalKappa2:
    """N9 (new): a genuinely off-diagonal κ²_{ab} produces an
    off-diagonal C propagator, reflecting component-mixing correctly.

    sft-wick's default tests all use ``kappa2 ∝ I`` (diagonal in
    components).  This test exercises the ``diag_C=False`` code path
    by supplying ``kappa2`` with a cross-component off-diagonal entry
    and verifying that ``PropagatorCache.C_value`` returns the
    expected non-diagonal 2×2 matrix.
    """

    def test_off_diagonal_kappa2_propagates_to_C(self):
        # 2-component kappa with non-trivial off-diagonal correlation
        # kappa2_{ab}(t, t') = λ e^{-|t-t'|/σ_t} × M_{ab}
        # with M a fixed 2×2 symmetric positive-definite matrix.
        lam = 0.05
        sigma_t = 0.3
        gamma = 1.0
        M = np.array([[1.0, 0.3], [0.3, 1.2]])

        def R_time(t1, t2):
            return np.exp(-gamma * (t1 - t2)) if t1 >= t2 else 0.0

        def kappa2(n1, t1, n2, t2):
            kt = lam * np.exp(-abs(t1 - t2) / sigma_t)
            kx = 1.0  # not interested in spatial structure for this test
            return kt * kx * M

        model = PropagatorModel(
            R_time=R_time, kappa2=kappa2,
            n_components=2, iso_R=True, diag_C=False, t_min=0.0,
        )
        cache = PropagatorCache(model)

        # Closed form: C_{ab}(t,t) = M_{ab} × C_scalar(t,t) where C_scalar
        # uses λ=1 (the λ factor is absorbed into the kappa2 function).
        # We verify via independent domain-split dblquad of the full
        # matrix-valued integrand.
        t = 3.0

        def C_full_dblquad():
            """Compute the 2×2 C matrix via split dblquad on each entry."""
            C_mat = np.zeros((2, 2))
            for a in range(2):
                for b in range(2):
                    def fA(tau2, tau1, a=a, b=b):
                        return (
                            R_time(t, tau1) * R_time(t, tau2)
                            * lam * np.exp(-(tau1 - tau2) / sigma_t) * M[a, b]
                        )
                    def fB(tau2, tau1, a=a, b=b):
                        return (
                            R_time(t, tau1) * R_time(t, tau2)
                            * lam * np.exp(-(tau2 - tau1) / sigma_t) * M[a, b]
                        )
                    IA, _ = dblquad(fA, 0, t, 0, lambda tau1: tau1,
                                     epsabs=1e-12, epsrel=1e-12)
                    IB, _ = dblquad(fB, 0, t, lambda tau1: tau1, t,
                                     epsabs=1e-12, epsrel=1e-12)
                    C_mat[a, b] = IA + IB
            return C_mat

        C_ref = C_full_dblquad()
        C_pkg = cache.C_value(0, t, 0, t)

        # Off-diagonal entries must be non-zero and match the reference
        assert abs(C_ref[0, 1]) > 1e-6, "test setup: expected non-zero off-diagonal"
        assert abs(C_pkg[0, 1]) > 1e-6, "sft-wick returned zero off-diagonal"

        for a in range(2):
            for b in range(2):
                rel = abs(C_pkg[a, b] - C_ref[a, b]) / max(abs(C_ref[a, b]), 1e-14)
                # PropagatorCache uses its own internal dblquad; accept 1e-4
                # (same tolerance class as ``TestPropagatorCacheCValue``).
                assert rel < 1e-4, (
                    f"C[{a},{b}]: pkg={C_pkg[a,b]:.6e} vs ref={C_ref[a,b]:.6e}, "
                    f"rel={rel:.2e}"
                )

        # Sanity: the C matrix retains M's symmetry
        assert abs(C_pkg[0, 1] - C_pkg[1, 0]) < 1e-8


class _AnalyticalCache:
    """Drop-in replacement for ``PropagatorCache`` with closed-form C.

    Same shape as demo-1's notebook-internal ``AnalyticalCache``; used
    by :class:`TestFeynmanDiagramQMC` to isolate the QMC integration
    logic from the slow (dblquad) / lossy (spline) C-evaluation paths.
    The closed-form C formula matches :func:`C_closed_form` at the top
    of this module and returns values in microseconds.
    """

    def __init__(self, lam, sigma_t, sigma_x, gamma, n_components):
        self.lam = lam
        self.sigma_t = sigma_t
        self.sigma_x = sigma_x
        self.gamma = gamma
        self.alpha_exp = 1.0 / sigma_t
        self.N = n_components
        self._c_splines = True  # sentinel so DiagramIntegrand accepts us

    # --- R propagator (scalar, causal) ---
    def R_time(self, t1, t2):
        return np.exp(-self.gamma * (t1 - t2)) if t1 >= t2 else 0.0

    def R_time_batch(self, t1, t2):
        return np.where(t1 >= t2, np.exp(-self.gamma * (t1 - t2)), 0.0)

    def R_product(self, r_pairs, times):
        result = 1.0
        for sl, sr in r_pairs:
            result *= float(self.R_time(times[sl], times[sr]))
        return result

    # --- Closed-form C(t1, t2) ---
    def _C_scalar(self, t1, t2):
        g, a = self.gamma, self.alpha_exp
        tl = min(t1, t2); th = max(t1, t2)
        if tl <= 0: return 0.0
        gpa, gma = g + a, g - a
        E1 = np.expm1(2 * g * tl) / (2 * g)
        E2 = tl if abs(gma) < 1e-14 else np.expm1(gma * tl) / gma
        E3 = np.expm1(gpa * tl) / gpa
        E4 = np.exp(gma * th)
        I = E1 / gpa - E2 / gpa + E4 * E3 / gma - E1 / gma
        return self.lam * np.exp(-g * (t1 + t2)) * I

    def C_diagonal_batch(self, t1, t2):
        g, a, lam_ = self.gamma, self.alpha_exp, self.lam
        tl = np.minimum(t1, t2); th = np.maximum(t1, t2)
        gpa, gma = g + a, g - a
        E1 = np.expm1(2 * g * tl) / (2 * g)
        E2 = tl if abs(gma) < 1e-14 else np.expm1(gma * tl) / gma
        E3 = np.expm1(gpa * tl) / gpa
        E4 = np.exp(gma * th)
        I = E1 / gpa - E2 / gpa + E4 * E3 / gma - E1 / gma
        c = lam_ * np.exp(-g * (t1 + t2)) * I
        return np.column_stack([c] * self.N)

    def C_value(self, n1, t1, n2, t2):
        return self._C_scalar(t1, t2) * np.eye(self.N)

    def C_diagonal(self, n, t1, n_prime=None, t2=None):
        if t2 is None: t2 = t1
        return np.full(self.N, self._C_scalar(t1, t2))

    @property
    def model(self):
        _self = self
        def _kappa2(n1, t1, n2, t2):
            kt = _self.lam * np.exp(-abs(t1 - t2) * _self.alpha_exp)
            kx = np.exp(-abs(np.asarray(n1) - np.asarray(n2)) / _self.sigma_x)
            return kt * kx * np.eye(_self.N)
        def _R(t1, t2):
            return np.exp(-_self.gamma * (t1 - t2)) if t1 >= t2 else 0.0
        return type("M", (), {
            "n_components": _self.N, "iso_R": True, "diag_C": True,
            "t_min": 0.0,
            "kappa2": staticmethod(_kappa2),
            "R_time": staticmethod(_R),
        })()


class TestFeynmanDiagramQMC:
    """Phase-3 deductive tests: full Feynman-diagram numerical evaluation.

    Verifies that sft-wick's two integration entry points for a single
    DiagramTerm produce numbers that match an independent closed-form
    reference derived purely from the analytical R/C propagators:

    - ``scipy.nquad(make_scipy_integrand(...))`` — evaluates
      ``ξ_ab(r, t_f)`` at fixed external times.
    - ``integrate_moment_qmc(..., lambda_f=t_f)`` — evaluates the time-
      integrated moment ``⟨∫_0^{t_f} Φ_a(t) dt × ∫_0^{t_f} Φ_b(t) dt⟩``.

    The **double-tadpole** diagram (order 2, two C self-loops) is
    chosen because its integrand factorises: each vertex's contribution
    reduces to a 1-D integral of ``R(t,τ) × C(τ,τ)``, evaluable to
    machine precision via ``scipy.quad``.  All other order-2 diagrams
    have coupled temporal integrations and need hand-derivations that
    are more intricate; starting with the double tadpole covers the
    QMC *machinery* (propagator evaluation, causal-domain transform,
    coupling contraction) without getting mired in closed-form algebra.

    Fixture: demo-1-style F tensor at N=2, observable ``(a=0, b=0)``.
    """

    LAM = 0.05
    SIGMA_T = 0.3
    SIGMA_X = 1.0
    GAMMA = 1.0
    N = 2
    LAMBDA_F = 3.0

    def _C_scalar(self, t1, t2):
        g, a = self.GAMMA, 1.0 / self.SIGMA_T
        tl = min(t1, t2); th = max(t1, t2)
        if tl <= 0: return 0.0
        gpa, gma = g + a, g - a
        E1 = np.expm1(2 * g * tl) / (2 * g)
        E2 = tl if abs(gma) < 1e-14 else np.expm1(gma * tl) / gma
        E3 = np.expm1(gpa * tl) / gpa
        E4 = np.exp(gma * th)
        I = E1 / gpa - E2 / gpa + E4 * E3 / gma - E1 / gma
        return self.LAM * np.exp(-g * (t1 + t2)) * I

    def _A_at(self, t):
        """∫_0^t R(t, τ) × C(τ, τ) dτ — per-vertex 1D integral of the
        double-tadpole integrand (factorised)."""
        def integrand(tau):
            R = np.exp(-self.GAMMA * (t - tau)) if t >= tau else 0.0
            return R * self._C_scalar(tau, tau)
        val, _ = scipy_quad(integrand, 0, t, epsabs=1e-12, epsrel=1e-12)
        return val

    def _B_at_lambda(self, lambda_f):
        """∫_0^λ A(t) dt — outer integral over external time for moments."""
        val, _ = scipy_quad(self._A_at, 0, lambda_f, epsabs=1e-10, epsrel=1e-10)
        return val

    @pytest.fixture(scope="class")
    def _shared(self):
        """Class-scoped cache + compute_moment — shared across tests.

        **Design choice**: we use a drop-in *analytical* cache (same
        shape as the one demo-1 defines inside its notebook) rather
        than sft-wick's ``PropagatorCache``.  Rationale:

        - Phase 3's purpose is to deductively test the QMC / nquad
          Feynman-diagram integration machinery, *given* accurate
          propagator values.
        - sft-wick's native ``PropagatorCache`` either ``dblquad``s
          every C evaluation (QMC at 2^14 samples ≈ 42 min per test)
          or pre-tabulates C on a spline grid (~85 s setup, plus an
          O(h⁴) ≈ 5e-4 interpolation-error floor).  Both degrade
          either runtime or precision.
        - The spline path is *separately* verified in
          :class:`TestSplineTable` (N7), and the dblquad path in
          :class:`TestPropagatorCacheCValue`.  We do not need to
          re-test them here.
        - Using a closed-form C yields machine-precision C evaluations
          in microseconds and lets us set tight tolerances (< 1e-4)
          that genuinely bound QMC statistical error, rather than
          being dominated by a caching artefact.

        This is exactly the pattern ``examples/demo1`` and
        ``examples/demo2`` use via their notebook-internal
        ``AnalyticalCache`` class.
        """
        cache = _AnalyticalCache(self.LAM, self.SIGMA_T, self.SIGMA_X,
                                  self.GAMMA, self.N)

        F_arr = np.zeros((self.N, self.N, self.N))
        F_arr[0, 1, 1] = 1.0
        F_arr[1, 0, 1] = 0.5
        F_arr[1, 1, 0] = 0.5
        F_MSR = -1j * F_arr

        reset_uid_counter()
        phi = Field("phi", "physical", n_components=self.N)
        psi = Field("psi", "response", n_components=self.N)
        action = Action(vertices=[Vertex([psi, phi, phi], coupling="F")])
        obs = [phi("a", "x"), phi("b", "y")]
        result = compute_moment(
            obs, action, order=2, response_phase=True, ito=True,
            diag_R=True, diag_C=True, iso_R=True, iso_C=True,
        )

        dbl = [dt for dt in result.diagram_terms(2)
               if all(p.spatial_left == p.spatial_right
                      for p in dt.propagators if p.kind == "C")]
        assert len(dbl) == 1
        dt_dbl = dbl[0]
        coupling = complex(np.asarray(dt_dbl.evaluate_coupling(
            {"F": F_MSR}, fixed_indices={"a": 0, "b": 0}
        ))).real
        return cache, F_MSR, dt_dbl, coupling

    # ---------- QDT-P1: scipy.nquad(make_scipy_integrand) = N² × coupling × A² ----------

    def test_scipy_integrand_matches_N_squared_A_squared(self, _shared):
        """``scipy.nquad(make_scipy_integrand)`` evaluates ξ(r=0, t_f) at
        fixed external times.  For the double tadpole this equals
        ``N² × coupling × A(t_f)²``, where the factor ``N²`` comes from
        the trace over component indices implicit in the C-propagator
        contraction.  Reference A(t_f) via 1D scipy.quad to 1e-12."""
        from scipy.integrate import nquad as scipy_nquad

        cache, F_MSR, dt_dbl, coupling = _shared
        t_f = self.LAMBDA_F
        ig = dt_dbl.build_integrand(
            {"F": F_MSR}, fixed_indices={"a": 0, "b": 0}
        )
        directions = {d: 0 for d in set(ig.spatial.direction_map.values())}
        f = ig.make_scipy_integrand(
            external_times={"x": t_f, "y": t_f},
            external_directions=directions, cache=cache,
        )
        bounds = ig.integration_bounds(
            external_times={"x": t_f, "y": t_f}, t_min=0.0,
        )
        got, _ = scipy_nquad(f, bounds, opts={"epsabs": 1e-10, "epsrel": 1e-10})

        expected = self.N**2 * coupling * self._A_at(t_f)**2
        rel = abs(got - expected) / abs(expected)
        # Tight: both sides use machine-precision analytical C.
        # The only remaining error is scipy.nquad adaptive quadrature.
        assert rel < 1e-6, (
            f"make_scipy_integrand+nquad: {got:.10e} vs closed form "
            f"{expected:.10e}, rel={rel:.2e}"
        )

    # ---------- QDT-P2: integrate_moment_qmc = N² × coupling × B² ----------

    def test_qmc_matches_N_squared_B_squared(self, _shared):
        """``ig.integrate_moment_qmc(lambda_f)`` evaluates the time-
        integrated moment of the diagram contribution.  For the double
        tadpole this equals ``N² × coupling × B(λ)²`` where
        ``B(λ) = ∫_0^λ A(t) dt``.

        Verifies QMC reaches within 3σ of its self-reported error bar.
        """
        cache, F_MSR, dt_dbl, coupling = _shared
        lambda_f = self.LAMBDA_F
        ig = dt_dbl.build_integrand(
            {"F": F_MSR}, fixed_indices={"a": 0, "b": 0}
        )
        got, err = ig.integrate_moment_qmc(
            lambda_f, cache, n_samples=2**14, seed=42,
            integrate_over="all",   # B(λ)² closed form uses the
                                    # time-integrated moment.
        )

        expected = self.N**2 * coupling * self._B_at_lambda(lambda_f)**2
        # Within 3σ of QMC's own error (standard Sobol 99.7% confidence).
        assert abs(got - expected) < 3 * err + 1e-10, (
            f"QMC {got:.6e} vs closed form {expected:.6e}, "
            f"diff={abs(got - expected):.2e}, 3σ band={3*err:.2e}"
        )
        # Sanity: relative error below ~0.5% at n=2^14 for this integrand.
        rel = abs(got - expected) / abs(expected)
        assert rel < 5e-3, f"QMC rel err {rel:.2e} unexpectedly large"

    # ---------- QDT-P3: QMC convergence rate ----------

    def test_qmc_convergence_monotone(self, _shared):
        """QMC error (|got - expected|) decreases monotonically with
        ``n_samples`` on a sequence of doublings.

        Sobol QMC on a smooth 2D integrand converges at roughly O(1/N);
        here we just check monotone decrease across three scales,
        which is enough to spot a gross regression in the sampling
        logic (e.g. a biased remapping of the causal domain).
        """
        cache, F_MSR, dt_dbl, coupling = _shared
        lambda_f = self.LAMBDA_F
        ig = dt_dbl.build_integrand(
            {"F": F_MSR}, fixed_indices={"a": 0, "b": 0}
        )
        expected = self.N**2 * coupling * self._B_at_lambda(lambda_f)**2

        errors = []
        for n_bits in (10, 12, 14):
            got, _ = ig.integrate_moment_qmc(
                lambda_f, cache, n_samples=2**n_bits, seed=42,
                integrate_over="all",
            )
            errors.append(abs(got - expected))

        # Monotone decrease — even Sobol can have non-monotonic single-scale
        # fluctuations, but across 2^10 → 2^12 → 2^14 (factor 16 in N)
        # we expect at least a factor 2 improvement overall.
        assert errors[-1] < errors[0] / 2, (
            f"QMC errors did not decrease enough: {errors}"
        )


class TestAlternativePathConsistency:
    """Phase-4 deductive tests: consistency between alternative entry
    points in the package.

    sft-wick exposes multiple paths for both symbolic expansion and
    numerical integration.  The production tests (Phase 1–3) exercise
    the *default* paths; this class cross-checks each *alternative*
    against a tested baseline — two independent code paths producing
    the same answer is strong deductive evidence that both are correct.

    Alternative paths under test:

    ==================================  ==============================================
    Alternative                         Baseline                                  ID
    ==================================  ==============================================
    ``compute_moment_numerical``        ``compute_moment``                        C1
    ``compute_moment_numerical`` parallel   ``compute_moment_numerical`` serial   C2
    ``integrate_moment_qmc_vectorized`` ``integrate_moment_qmc``                  C3
    ``integrate_two_point_qmc``         ``scipy.nquad(make_scipy_integrand)``     C4
    ``integrate_diagrams(n_jobs=-1)``   ``integrate_diagrams(n_jobs=1)``          C5
    ==================================  ==============================================

    Shared fixture uses the class-scoped analytical cache (same design
    as :class:`TestFeynmanDiagramQMC`).
    """

    LAM = 0.05
    SIGMA_T = 0.3
    SIGMA_X = 1.0
    GAMMA = 1.0
    N = 2
    LAMBDA_F = 3.0

    @pytest.fixture(scope="class")
    def _shared(self):
        from sft_wick.perturbation import compute_moment_numerical

        cache = _AnalyticalCache(self.LAM, self.SIGMA_T, self.SIGMA_X,
                                  self.GAMMA, self.N)
        F_arr = np.zeros((self.N, self.N, self.N))
        F_arr[0, 1, 1] = 1.0
        F_arr[1, 0, 1] = 0.5
        F_arr[1, 1, 0] = 0.5
        F_MSR = -1j * F_arr

        reset_uid_counter()
        phi = Field("phi", "physical", n_components=self.N)
        psi = Field("psi", "response", n_components=self.N)
        action = Action(vertices=[Vertex([psi, phi, phi], coupling="F")])
        obs = [phi("a", "x"), phi("b", "y")]
        fixed = {"a": 0, "b": 0}

        # Baseline compute_moment (hybrid engine)
        result = compute_moment(
            obs, action, order=2, response_phase=True, ito=True,
            diag_R=True, diag_C=True, iso_R=True, iso_C=True,
        )
        dts_cm = result.diagram_terms(2)

        # Pick the double-tadpole for single-diagram comparisons
        dbl = [dt for dt in dts_cm
               if all(p.spatial_left == p.spatial_right
                      for p in dt.propagators if p.kind == "C")][0]
        ig_dbl = dbl.build_integrand({"F": F_MSR}, fixed_indices=fixed)

        return {
            "cache": cache,
            "F_MSR": F_MSR,
            "action": action,
            "obs": obs,
            "fixed": fixed,
            "dts_cm": dts_cm,
            "ig_dbl": ig_dbl,
            "lambda_f": self.LAMBDA_F,
        }

    # ---------- C1: compute_moment vs compute_moment_numerical (serial) ----------

    def test_C1_compute_moment_vs_numerical_total(self, _shared):
        """Two symbolic engines (Wick-collection hybrid vs nauty + full
        component-routing enumeration) produce DT sets of different
        cardinality (6 vs 30 at N=2, order 2), but the integrated
        totals must agree within QMC precision.

        This is the load-bearing consistency check between the two
        compute_moment implementations.
        """
        from sft_wick.perturbation import compute_moment_numerical
        from sft_wick.evaluate import integrate_diagrams

        reset_uid_counter()
        dts_num = compute_moment_numerical(
            _shared["obs"], _shared["action"], order=2,
            coupling_values={"F": _shared["F_MSR"]},
            fixed_indices=_shared["fixed"],
            diag_R=True, diag_C=True, iso_R=True, iso_C=True,
            n_jobs=1,
        )[2]

        total_cm, _ = integrate_diagrams(
            _shared["dts_cm"], {"F": _shared["F_MSR"]},
            _shared["lambda_f"], _shared["cache"],
            method="qmc", n_samples=2**14, seed=42,
            fixed_indices=_shared["fixed"],
            n_jobs=-1,
        )
        total_num, _ = integrate_diagrams(
            dts_num, {"F": _shared["F_MSR"]},
            _shared["lambda_f"], _shared["cache"],
            method="qmc", n_samples=2**14, seed=42,
            fixed_indices=_shared["fixed"],
            n_jobs=-1,
        )

        # Different DT partitioning => different per-diagram QMC variance
        # but the total must agree tightly.  Observed ~5e-7 relative.
        rel = abs(total_cm - total_num) / abs(total_cm)
        assert rel < 1e-5, (
            f"compute_moment = {total_cm:.6e}, numerical = {total_num:.6e}, "
            f"rel = {rel:.2e}"
        )
        # Expect structural difference: 6 vs 30 DTs at this order.
        assert len(_shared["dts_cm"]) == 6
        assert len(dts_num) == 30

    # ---------- C2: compute_moment_numerical serial vs parallel ----------

    def test_C2_numerical_parallel_matches_serial(self, _shared):
        """``compute_moment_numerical(n_jobs=1)`` and ``(n_jobs=-1)``
        must produce bit-identical DiagramTerms (joblib preserves
        ordering on this workload).
        """
        pytest.importorskip("joblib")
        from sft_wick.perturbation import compute_moment_numerical

        reset_uid_counter()
        dts_s = compute_moment_numerical(
            _shared["obs"], _shared["action"], order=2,
            coupling_values={"F": _shared["F_MSR"]},
            fixed_indices=_shared["fixed"],
            diag_R=True, diag_C=True, iso_R=True, iso_C=True,
            n_jobs=1,
        )[2]
        reset_uid_counter()
        dts_p = compute_moment_numerical(
            _shared["obs"], _shared["action"], order=2,
            coupling_values={"F": _shared["F_MSR"]},
            fixed_indices=_shared["fixed"],
            diag_R=True, diag_C=True, iso_R=True, iso_C=True,
            n_jobs=-1,
        )[2]

        # Compare integrated totals bit-identical (same seed, same algorithm).
        from sft_wick.evaluate import integrate_diagrams
        v_s, _ = integrate_diagrams(
            dts_s, {"F": _shared["F_MSR"]},
            _shared["lambda_f"], _shared["cache"],
            method="qmc", n_samples=2**12, seed=42,
            fixed_indices=_shared["fixed"],
            n_jobs=-1,
        )
        v_p, _ = integrate_diagrams(
            dts_p, {"F": _shared["F_MSR"]},
            _shared["lambda_f"], _shared["cache"],
            method="qmc", n_samples=2**12, seed=42,
            fixed_indices=_shared["fixed"],
            n_jobs=-1,
        )
        assert abs(v_s - v_p) < 1e-12, (
            f"serial {v_s} vs parallel {v_p}: diff {abs(v_s - v_p):.2e}"
        )

    # ---------- C3: scalar loop vs vectorized QMC ----------

    def test_C3_qmc_scalar_vs_vectorized(self, _shared):
        """``integrate_moment_qmc`` (scalar Python loop over Sobol
        samples) and ``integrate_moment_qmc_vectorized`` (batch
        propagator lookups) must give bit-identical results on the
        same seed — they implement the same algorithm, one faster.
        """
        ig = _shared["ig_dbl"]
        v1, e1 = ig.integrate_moment_qmc(
            _shared["lambda_f"], _shared["cache"],
            n_samples=2**14, seed=42,
        )
        v2, e2 = ig.integrate_moment_qmc_vectorized(
            _shared["lambda_f"], _shared["cache"],
            n_samples=2**14, seed=42,
        )
        assert abs(v1 - v2) < 1e-12, (
            f"scalar {v1} vs vectorized {v2}: diff {abs(v1 - v2):.2e}"
        )
        assert abs(e1 - e2) < 1e-14

    # ---------- C4: integrate_two_point_qmc vs scipy.nquad ----------

    def test_C4_two_point_qmc_vs_nquad_reference(self, _shared):
        """``integrate_two_point_qmc`` is a specialised QMC for fixed-
        time 2-point correlators with per-point spatial positions.
        It must agree with ``scipy.nquad(make_scipy_integrand)``
        (independent integration backend) within QMC precision.
        """
        from sft_wick.evaluate import integrate_two_point_qmc
        from scipy.integrate import nquad as scipy_nquad

        ig = _shared["ig_dbl"]
        positions = {"x": 0.0, "y": 0.0}  # r = 0

        v_qmc, e_qmc = integrate_two_point_qmc(
            [ig], t_f=_shared["lambda_f"], positions=positions,
            cache=_shared["cache"], n_samples=2**14, seed=42,
        )

        directions = {d: 0 for d in set(ig.spatial.direction_map.values())}
        f_sp = ig.make_scipy_integrand(
            external_times={"x": _shared["lambda_f"], "y": _shared["lambda_f"]},
            external_directions=directions, cache=_shared["cache"],
        )
        bounds = ig.integration_bounds(
            external_times={"x": _shared["lambda_f"], "y": _shared["lambda_f"]},
            t_min=0.0,
        )
        v_ref, _ = scipy_nquad(f_sp, bounds,
                                opts={"epsabs": 1e-10, "epsrel": 1e-10})

        rel = abs(v_qmc - v_ref) / abs(v_ref)
        # Two independent 2D integrators: tighter than typical 1% QMC
        # tolerance because analytical cache = machine precision.
        assert rel < 1e-4, (
            f"integrate_two_point_qmc {v_qmc:.6e} vs scipy.nquad "
            f"{v_ref:.6e}: rel {rel:.2e}"
        )

    # ---------- C6: integrate_moment dispatcher auto-selects fastest QMC ----------

    def test_C6_dispatcher_auto_selects_vectorized_when_cache_supports_batch(
        self, _shared,
    ):
        """``integrate_moment(method='qmc')`` must auto-route to the
        vectorised implementation when the cache supports batch C
        evaluation — otherwise the 215× speedup is left on the table.

        Verifies:
        (a) with a batch-capable cache, the dispatcher delegates to
            ``integrate_moment_qmc_vectorized`` (same numeric result
            as calling the vectorised version directly).
        (b) ``method='qmc_scalar'`` forces the scalar path and matches
            the dedicated scalar method.
        (c) all three paths agree to machine precision on identical
            seeds (bit-identical confirmed by C3).
        """
        from sft_wick.evaluate import integrate_moment
        ig = _shared["ig_dbl"]
        cache = _shared["cache"]
        lf = _shared["lambda_f"]

        # Sanity: cache supports batch (has _c_splines truthy)
        from sft_wick.evaluate import _cache_supports_batch_c
        assert _cache_supports_batch_c(cache), (
            "test fixture must use a batch-capable cache"
        )

        v_auto, _ = integrate_moment(
            ig, lf, cache, method="qmc",
            n_samples=2**14, seed=42,
        )
        v_vec, _ = ig.integrate_moment_qmc_vectorized(
            lf, cache, n_samples=2**14, seed=42,
        )
        v_scalar, _ = integrate_moment(
            ig, lf, cache, method="qmc_scalar",
            n_samples=2**14, seed=42,
        )
        v_scalar_direct, _ = ig.integrate_moment_qmc(
            lf, cache, n_samples=2**14, seed=42,
        )

        # Auto dispatch → vectorised (bit-identical at same seed)
        assert abs(v_auto - v_vec) < 1e-14, (
            f"dispatcher didn't auto-select vectorised: auto={v_auto}, vec={v_vec}"
        )
        # Explicit scalar path → scalar (bit-identical)
        assert abs(v_scalar - v_scalar_direct) < 1e-14
        # Vectorised and scalar agree on the same seed (as per C3)
        assert abs(v_vec - v_scalar) < 1e-14

    def test_C6_dispatcher_falls_back_to_scalar_without_batch_support(self, _shared):
        """If the cache lacks batch support, the dispatcher must fall
        back to the scalar QMC path (not raise)."""
        from sft_wick.evaluate import integrate_moment, _cache_supports_batch_c

        # Wrap the cache in a dumb proxy that reports no batch support
        class _NoBatch:
            _c_splines = None  # sentinel says "batch not available"
            def __init__(self, inner):
                object.__setattr__(self, "_inner", inner)
            def __getattr__(self, name):
                return getattr(self._inner, name)

        proxy = _NoBatch(_shared["cache"])
        assert not _cache_supports_batch_c(proxy)

        # Dispatcher should fall back to scalar — returning same value
        # as the direct scalar call on the real cache (same algorithm)
        v_auto, _ = integrate_moment(
            _shared["ig_dbl"], _shared["lambda_f"], proxy,
            method="qmc", n_samples=2**12, seed=42,
        )
        v_scalar_ref, _ = _shared["ig_dbl"].integrate_moment_qmc(
            _shared["lambda_f"], _shared["cache"],
            n_samples=2**12, seed=42,
        )
        assert abs(v_auto - v_scalar_ref) < 1e-14

    # ---------- C5: integrate_diagrams serial vs parallel ----------

    def test_C5_integrate_diagrams_parallel_matches_serial(self, _shared):
        """``integrate_diagrams(n_jobs=-1)`` distributes the per-diagram
        evaluation across cores via joblib.  Because each diagram's
        integration is deterministic given its seed, the total must
        be bit-identical to the ``n_jobs=1`` serial path.
        """
        pytest.importorskip("joblib")
        from sft_wick.evaluate import integrate_diagrams

        args = {
            "diagram_terms": _shared["dts_cm"],
            "coupling_values": {"F": _shared["F_MSR"]},
            "lambda_f": _shared["lambda_f"],
            "cache": _shared["cache"],
            "method": "qmc",
            "n_samples": 2**14,
            "seed": 42,
            "fixed_indices": _shared["fixed"],
        }
        total_s, details_s = integrate_diagrams(n_jobs=1, **args)
        total_p, details_p = integrate_diagrams(n_jobs=-1, **args)

        assert abs(total_s - total_p) < 1e-12, (
            f"serial {total_s} vs parallel {total_p}: diff {abs(total_s - total_p):.2e}"
        )
        # Per-diagram values also bit-identical
        for i, ((vs, _), (vp, _)) in enumerate(zip(details_s, details_p)):
            assert abs(vs - vp) < 1e-12, (
                f"diagram {i}: serial {vs} vs parallel {vp}"
            )


class TestItoDifference:
    """N8: comparing compute_moment with ``ito=True`` vs ``ito=False``
    on a trivial observable where the only surviving pairing is an
    equal-point R.

    The simplest such setup: observable ``[phi(x), psi(x)]`` at order 0
    — with ``ito=True`` the result is zero (R(x,x)=0); with
    ``ito=False`` it is the bare R(x,x) propagator.  This exercises the
    ``ito`` flag end-to-end in ``compute_moment``.
    """

    def test_zero_order_equal_point_phi_psi_with_ito(self):
        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        obs = [phi("x"), psi("x")]
        action = Action(vertices=[])

        reset_uid_counter()
        with_ito = compute_moment(obs, action, order=0, ito=True)
        reset_uid_counter()
        without_ito = compute_moment(obs, action, order=0, ito=False)

        # With Itô: zero (no surviving DiagramTerm)
        assert with_ito.diagram_terms(0) == []

        # Without Itô: one surviving DiagramTerm with one R propagator
        dts = without_ito.diagram_terms(0)
        assert len(dts) == 1, f"expected 1 DiagramTerm, got {len(dts)}"
        dt = dts[0]
        assert len(dt.propagators) == 1
        p = dt.propagators[0]
        assert p.kind == "R"
        assert p.spatial_left == p.spatial_right == "x"

    def test_different_point_phi_psi_ito_agnostic(self):
        """Itô only kills equal-point R; R(x, y) survives regardless."""
        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        obs = [phi("x"), psi("y")]
        action = Action(vertices=[])

        reset_uid_counter()
        with_ito = compute_moment(obs, action, order=0, ito=True)
        reset_uid_counter()
        without_ito = compute_moment(obs, action, order=0, ito=False)

        assert len(with_ito.diagram_terms(0)) == 1
        assert len(without_ito.diagram_terms(0)) == 1

        for result in (with_ito, without_ito):
            dt = result.diagram_terms(0)[0]
            assert dt.propagators[0].kind == "R"
            assert dt.propagators[0].spatial_left == "x"
            assert dt.propagators[0].spatial_right == "y"


# --------------------------------------------------------------------- #
# Phase 5 — Spatial-coordinate-aware numerical evaluation (S1–S7)
# --------------------------------------------------------------------- #


class TestSpatialAwareCache:
    """Phase-5 deductive tests: the ``(t, x)`` extension of the numerical
    diagram-evaluation pipeline.

    A field operator now carries a spatial coordinate in addition to a
    time argument (``phi('a', 'z')`` where ``z = (t, x)``).  For
    stochastic field theories with propagators of the form
    ``R(x, t; x', t') = δ(x − x') · Θ(t − t') · R_t(t, t')`` the δ
    collapses every *internal* x-integration onto an *external* one:
    the numerical integration domain stays time-only, but each C
    propagator must now carry the correct x-coordinate flowed from
    externals through the R-δ chain.

    ``PropagatorCache`` picks one of three physical assumptions about
    how C depends on x (the ``homogeneity`` kwarg):

    - ``'translation'`` (default): C depends only on ``|x1 − x2|``.
      Built via ``precompute_C_table_translation`` — either lazy (no
      r-grid; 2-D time splines on-demand per distinct r) or full
      (3-D ``(t, t', r)`` spline).
    - ``'rotation'``: C depends only on ``x1 · x2 / (|x1| |x2|)``
      (direction-vector x).  Built via
      ``precompute_C_table_rotation`` — lazy or full
      ``(t, t', cos θ)``.
    - ``'general'``: no symmetry; full 4-D ``(t, t', x1, x2)`` or
      lazy per-pair 2-D.

    The seven tests:

    ========  ===============================================================
    Test ID    Claim
    ========  ===============================================================
    S1         No spatial table + any ``positions`` → bit-identical to
               legacy Phase-3 baseline.  Back-compat guard.
    S2         Full-grid translation spline reproduces closed-form
               ``C_t(t1,t2) × exp(-|x1−x2|/σ_x)`` to spline precision
               (< 5e-3 rel).
    S3         Building translation (3-D) and general (4-D) tables from
               the *same* translation-invariant κ² yields identical C at
               20 random ``(t1,t2,x1,x2)`` points (within combined
               spline error).
    S4         The double tadpole has no inter-group C propagators; its
               ``integrate_moment_qmc_vectorized`` result under
               translation + ``positions={'x':0,'y':r}`` equals the
               no-spatial-table baseline.
    S5         Bubble-type order-2 diagrams: ``n_cross_C`` cross-group C
               propagators each contribute ``exp(-r/σ_x)`` factor, so
               ``value(r) / value(0) = exp(-r/σ_x)^(n_cross_C)``.
    S6         Rotation homogeneity: for a rotation-invariant κ²
               (depending on ``x1·x2``), the rotation mode reproduces
               the closed form to spline precision.
    S7         Lazy vs full-grid translation mode give the same numerical
               answer on S5's bubble-scaling test, and the lazy cache's
               internal spline count equals the number of distinct r
               values queried (proving the "build once per unique r"
               invariant).
    ========  ===============================================================
    """

    LAM = 0.05
    SIGMA_T = 0.3
    SIGMA_X = 1.0
    GAMMA = 1.0
    # N=1 keeps spline-table build cost minimal (one dblquad per grid
    # point instead of N per point) — structural claims of this phase
    # don't depend on multi-component coverage, which is tested
    # separately in TestNonDiagonalKappa2.
    N = 1
    T_MAX = 3.0

    # --- Fast cache subclass: closed-form C bypasses dblquad ------ #
    #
    # For the separable OU kernel used in this test class, C has the
    # closed form ``C_closed_form(t1, t2) · exp(-|x1-x2|/σ_x)``.
    # Using it instead of dblquad cuts table-build cost from
    # 30 ms/point to <1 µs/point, enabling the finer grids needed
    # for meaningful spline-precision comparisons in S3 without
    # making the test suite unusably slow.

    def _make_fast_cache(self, homogeneity: str = "translation"):
        """Return a :class:`PropagatorCache` subclass whose
        ``_C_value_direct`` uses the closed-form OU expression
        ``C_t(t1,t2) · exp(-|x1-x2|/σ_x)``.

        Set the cache's ``homogeneity`` kwarg; the closed form
        itself is always translation-invariant, so ``homogeneity=
        'general'`` still gives a translation-invariant dataset — S3
        exploits this to compare the 3-D and 4-D tables on known-to-be-
        consistent data.
        """
        sigma_x = self.SIGMA_X
        N_local = self.N

        class _FastCache(PropagatorCache):
            def _C_value_direct(self, n1, t1, n2, t2):
                # |x1-x2| in scalar form; np.asarray accepts scalar too
                r = float(np.abs(np.asarray(n1) - np.asarray(n2)).sum())
                C_t = C_closed_form(t1, t2)
                C_x = np.exp(-r / sigma_x)
                C_mat = np.zeros((N_local, N_local))
                for a in range(N_local):
                    C_mat[a, a] = C_t * C_x
                return C_mat

        return _FastCache(
            model=self._make_model(), homogeneity=homogeneity,
        )

    def _make_fast_rotation_cache(
        self, t_max: float, n_grid_t: int, n_grid_cos: int,
    ):
        """Return a ``homogeneity='rotation'`` cache whose direct
        evaluation is a closed-form OU kernel with x dependence
        through ``x1·x2``.

        Specifically:  ``κ² = λ · exp(-|t-t'|/σ_t) · (1 + x1·x2)/2``
        — a smooth rotation-invariant kernel that varies
        non-trivially across ``cos θ ∈ [−1, 1]`` so the cos axis
        gets meaningfully sampled.
        """
        sigma_t = self.SIGMA_T
        gamma = self.GAMMA
        lam = self.LAM
        N_local = self.N
        from sft_wick.evaluate import _rotation_cos as _rc

        # Dedicated model whose κ² is a function of x1·x2 only.
        def R_time(t1, t2):
            return float(np.exp(-gamma * (t1 - t2))) if t1 >= t2 else 0.0

        def kappa2(n1, t1, n2, t2):
            cos_val = _rc(n1, n2)
            kt = lam * np.exp(-abs(t1 - t2) / sigma_t)
            kx = 0.5 * (1.0 + cos_val)  # in [0, 1], smooth in cos
            return kt * kx * np.eye(N_local)

        model = PropagatorModel(
            R_time=R_time, kappa2=kappa2, n_components=N_local,
            iso_R=True, diag_C=True, t_min=0.0,
        )

        class _FastRotCache(PropagatorCache):
            def _C_value_direct(self, n1, t1, n2, t2):
                cos_val = _rc(n1, n2)
                C_t = C_closed_form(t1, t2, lam=lam)
                C_mat = np.zeros((N_local, N_local))
                val = C_t * 0.5 * (1.0 + cos_val)
                for a in range(N_local):
                    C_mat[a, a] = val
                return C_mat

        cache = _FastRotCache(model=model, homogeneity="rotation")
        cache.precompute_C_table_rotation(
            t_max=t_max, n_grid_t=n_grid_t, n_grid_cos=n_grid_cos,
        )
        return cache

    # --- κ² separable OU kernel (same shape as demo1) -------------- #

    def _make_model(self):
        lam, sigma_t, sigma_x, gamma, N = (
            self.LAM, self.SIGMA_T, self.SIGMA_X, self.GAMMA, self.N,
        )

        def R_time(t1, t2):
            return float(np.exp(-gamma * (t1 - t2))) if t1 >= t2 else 0.0

        def kappa2(n1, t1, n2, t2):
            r = abs(float(n1) - float(n2))
            kt = lam * np.exp(-abs(t1 - t2) / sigma_t)
            kx = np.exp(-r / sigma_x)
            return kt * kx * np.eye(N)

        return PropagatorModel(
            R_time=R_time, kappa2=kappa2, n_components=N,
            iso_R=True, diag_C=True, t_min=0.0,
        )

    # Class-scoped cache fixtures: spline builds are the costly step,
    # so share across S1–S7.

    @pytest.fixture(scope="class")
    def cache_legacy(self):
        """No spatial table — legacy 2-D (t1, t2) spline only.  This is
        what users running pre-Phase-5 code get, and the reference
        ``_cache_has_spatial_table(cache) is False`` case for S1."""
        cache = self._make_fast_cache()
        cache.precompute_C_table(t_max=self.T_MAX, n_grid=30, direction=0.0)
        return cache

    @pytest.fixture(scope="class")
    def cache_translation_full(self):
        """Full-grid translation cache: 3-D spline in
        ``(t1, t2, |x1-x2|)``.  Uses the closed-form-backed fast
        subclass so build is essentially free."""
        cache = self._make_fast_cache()  # default homogeneity='translation'
        cache.precompute_C_table_translation(
            t_max=self.T_MAX, n_grid_t=25, r_max=2.5, n_grid_r=20,
        )
        return cache

    @pytest.fixture(scope="class")
    def cache_translation_lazy(self):
        """Lazy translation cache: 2-D (t, t') splines built on-demand
        per distinct r value.  Used by S7 to verify that the lazy
        path produces the same answer as the full-grid path for
        fixed-point moment calculations."""
        cache = self._make_fast_cache()
        cache.precompute_C_table_translation(
            t_max=self.T_MAX, n_grid_t=25,  # r_max, n_grid_r both None → lazy
        )
        return cache

    @pytest.fixture(scope="class")
    def cache_general(self):
        """Full-grid general 4-D spline in ``(t1, t2, x1, x2)``."""
        cache = self._make_fast_cache(homogeneity="general")
        cache.precompute_C_table_general(
            t_max=self.T_MAX, n_grid_t=20, x_max=2.5, n_grid_x=32,
        )
        return cache

    @pytest.fixture(scope="class")
    def cache_rotation_full(self):
        """Full-grid rotation cache: 3-D spline in ``(t1, t2, cos θ)``.

        Uses a rotation-invariant κ² (depends on ``x1·x2``) via the
        fast closed-form subclass — see :meth:`_make_fast_rotation_cache`.
        """
        return self._make_fast_rotation_cache(
            t_max=self.T_MAX, n_grid_t=25, n_grid_cos=32,
        )

    @pytest.fixture(scope="class")
    def order2_dts(self):
        """Class-scoped: compute once, reuse across S1/S4/S5/S7."""
        return self._compute_order2_dts()

    # --- Helper: build the order-2 diagram list for observable φ(x)φ(y) --- #

    def _compute_order2_dts(self):
        """Compute the 6 FF diagrams at order 2 for ``<φ(x)φ(y)>`` with
        cubic vertex ``ψφφ``.  Returns (diagram_terms, F_MSR).

        With scalar fields (N=1) the symbolic layer drops component
        indices entirely, so ``F`` is a bare ``Symbol`` with no indices
        — the numeric evaluator reads it as a 0-d scalar, not a rank-3
        tensor.
        """
        # Scalar F coupling (N=1 → no component indices).
        F_MSR = np.array(-1j)  # 0-dim complex scalar, shape ()

        reset_uid_counter()
        phi = Field("phi", "physical", n_components=self.N)
        psi = Field("psi", "response", n_components=self.N)
        action = Action(vertices=[Vertex([psi, phi, phi], coupling="F")])
        obs = [phi("x"), phi("y")]
        result = compute_moment(
            obs, action, order=2, response_phase=True, ito=True,
            diag_R=True, diag_C=True, iso_R=True, iso_C=True,
        )
        return result.diagram_terms(2), F_MSR

    @staticmethod
    def _count_cross_group_c(dt):
        """Count C propagators whose endpoints land in *different*
        direction groups — these are the C's that depend on |x − y|."""
        spatial = dt.analyze_spatial()
        n_cross = 0
        for p in dt.propagators:
            if p.kind != "C":
                continue
            d_l = spatial.direction_map[p.spatial_left]
            d_r = spatial.direction_map[p.spatial_right]
            if d_l != d_r:
                n_cross += 1
        return n_cross

    # ----- S1: back-compat bit-identity --------------------------- #

    def test_S1_backcompat_legacy_cache_ignores_positions(
        self, cache_legacy, order2_dts,
    ):
        """Phase-3's double-tadpole result is bit-unchanged when the
        cache has only the legacy 2-D spline (no spatial table) —
        ``positions`` silently ignored.
        """
        cache = cache_legacy
        dts, F_MSR = order2_dts

        # Double tadpole: both C endpoints at same vertex
        dbl = [dt for dt in dts
               if all(p.spatial_left == p.spatial_right
                      for p in dt.propagators if p.kind == "C")]
        assert len(dbl) == 1
        dt_dbl = dbl[0]

        ig = dt_dbl.build_integrand({"F": F_MSR})

        got_no_pos, _ = ig.integrate_moment_qmc_vectorized(
            lambda_f=self.T_MAX, cache=cache, n_samples=2**12, seed=42,
        )
        got_with_pos, _ = ig.integrate_moment_qmc_vectorized(
            lambda_f=self.T_MAX, cache=cache, n_samples=2**12, seed=42,
            positions={"x": 0.0, "y": 0.7},
        )
        # Same seed, same n_samples, no spatial table →
        # identical Sobol path and identical C_diagonal_batch lookup.
        assert got_no_pos == got_with_pos, (
            f"legacy cache (no spatial table) must ignore positions: "
            f"no_pos={got_no_pos!r} vs with_pos={got_with_pos!r}"
        )

    # ----- S2: translation-spline accuracy vs closed form --------- #

    def test_S2_translation_spline_matches_closed_form(
        self, cache_translation_full,
    ):
        """``precompute_C_table_translation(r_max, n_grid_r)`` +
        ``C_at_batch`` reproduces
        ``C_t(t1,t2) · exp(-|x1−x2|/σ_x)`` to spline precision."""
        cache = cache_translation_full

        rng = np.random.default_rng(0)
        t1 = rng.uniform(0.3, self.T_MAX - 0.3, size=20)
        t2 = rng.uniform(0.3, self.T_MAX - 0.3, size=20)
        x1 = rng.uniform(-1.0, 1.0, size=20)
        x2 = rng.uniform(-1.0, 1.0, size=20)

        got = cache.C_at_batch(t1, t2, x1, x2).ravel()  # N=1 → (20,)
        expected = np.array([
            C_closed_form(t1[k], t2[k]) * np.exp(-abs(x1[k] - x2[k]) / self.SIGMA_X)
            for k in range(20)
        ])
        rel = np.abs(got - expected) / np.abs(expected)
        # Linear interpolation on 20×20×12 grid → O(h²) truncation error
        # (RegularGridInterpolator was switched from 'cubic' to 'linear'
        # to avoid sign-flip overshoot in the steep r-tail; see
        # tests/test_evaluate_interpolation_accuracy.py for the lock).
        # Empirically ~7e-3 on this grid; allow 1.5e-2 headroom.
        assert rel.max() < 1.5e-2, (
            f"translation spline error {rel.max():.2e} exceeds 1.5e-2"
        )

    # ----- S3: translation vs general on translation-invariant input - #

    def test_S3_translation_vs_general_agree_on_invariant_input(
        self, cache_translation_full, cache_general,
    ):
        """When κ² is translation-invariant, the 3-D translation table
        and 4-D general table must agree within combined spline
        precision."""
        cache_sep = cache_translation_full
        cache_gen = cache_general

        rng = np.random.default_rng(1)
        t1 = rng.uniform(0.3, self.T_MAX - 0.3, size=20)
        t2 = rng.uniform(0.3, self.T_MAX - 0.3, size=20)
        x1 = rng.uniform(-1.5, 1.5, size=20)
        x2 = rng.uniform(-1.5, 1.5, size=20)

        got_sep = cache_sep.C_at_batch(t1, t2, x1, x2).ravel()
        got_gen = cache_gen.C_at_batch(t1, t2, x1, x2).ravel()

        rel = np.abs(got_sep - got_gen) / (np.abs(got_sep) + 1e-15)
        # Combined linear-interpolation precision: 3-D translation
        # table (n_r=20) + 4-D general table (n_x=32). The 4-D spline
        # has a second unnecessary x-axis inflating its O(h²) error
        # (was O(h⁴) under cubic; see test_evaluate_interpolation_
        # accuracy.py for why we switched). Most random points agree
        # to ~1e-3; worst-case outliers sit near 4e-2. A gross
        # dispatching bug (wrong axis order, missing spatial factor)
        # would give O(1) error — 8e-2 is still a meaningful upper
        # bound on dispatching correctness.
        assert rel.max() < 8e-2, (
            f"translation vs general disagree: max rel = {rel.max():.2e}"
        )
        assert np.median(rel) < 1e-2, (
            f"median translation-vs-general error "
            f"{np.median(rel):.2e} too high"
        )

    # ----- S4: double tadpole — positions routed but not felt ----- #

    def test_S4_double_tadpole_position_invariance(
        self, cache_translation_full, order2_dts,
    ):
        """The double tadpole has no inter-group C propagators: its
        two C self-loops both sit at a single internal vertex site
        whose direction is locked to *one* external.  So varying
        ``positions`` between runs must not change the result (apart
        from QMC noise — same seed → identical deterministic output).
        """
        cache = cache_translation_full
        dts, F_MSR = order2_dts

        dbl = [dt for dt in dts
               if all(p.spatial_left == p.spatial_right
                      for p in dt.propagators if p.kind == "C")]
        assert len(dbl) == 1
        dt_dbl = dbl[0]
        assert self._count_cross_group_c(dt_dbl) == 0

        ig = dt_dbl.build_integrand({"F": F_MSR})

        got_0, _ = ig.integrate_moment_qmc_vectorized(
            lambda_f=self.T_MAX, cache=cache, n_samples=2**12, seed=7,
            positions={"x": 0.0, "y": 0.0},
        )
        got_r, _ = ig.integrate_moment_qmc_vectorized(
            lambda_f=self.T_MAX, cache=cache, n_samples=2**12, seed=7,
            positions={"x": 0.0, "y": 0.5},
        )
        rel = abs(got_r - got_0) / abs(got_0)
        # Exact equality expected (same deterministic QMC path);
        # allow a tiny float-roundoff tolerance to be safe.
        assert rel < 1e-12, (
            f"Double tadpole must be position-invariant: "
            f"{got_0:.10e} vs {got_r:.10e} (rel {rel:.2e})"
        )

    # ----- S5: bubble diagrams scale as exp(-r/σ_x)^(n_cross_C) --- #

    def test_S5_bubble_cross_group_C_scaling(
        self, cache_translation_full, order2_dts,
    ):
        """For the separable OU kernel, each cross-group C propagator
        contributes a multiplicative factor ``exp(-|x−y|/σ_x)``
        (because ``C(t1,t2; x1,x2) = C_t(t1,t2) · exp(-|x1-x2|/σ_x)``
        under separability).  A diagram with ``n_cross_C`` such
        propagators therefore scales as
        ``value(r) / value(0) = exp(-r/σ_x)^(n_cross_C)``.

        Verifies positional routing end-to-end: each diagram's
        ``integrate_moment_qmc_vectorized`` output under the
        full-grid translation cache + ``positions={'x':0, 'y':r}``
        reproduces this predicted scaling, diagram-by-diagram, for
        all diagrams with ``n_cross_C > 0``.
        """
        cache = cache_translation_full
        dts, F_MSR = order2_dts

        r = 0.5  # well inside the translation-table range [0, 2.5]
        expected_factor_unit = np.exp(-r / self.SIGMA_X)

        found_cross = 0
        for i, dt in enumerate(dts):
            n_cross = self._count_cross_group_c(dt)
            if n_cross == 0:
                continue  # S4 covers position-invariant diagrams
            found_cross += 1

            ig = dt.build_integrand({"F": F_MSR})

            got_0, _ = ig.integrate_moment_qmc_vectorized(
                lambda_f=self.T_MAX, cache=cache, n_samples=2**13, seed=11,
                positions={"x": 0.0, "y": 0.0},
            )
            got_r, _ = ig.integrate_moment_qmc_vectorized(
                lambda_f=self.T_MAX, cache=cache, n_samples=2**13, seed=11,
                positions={"x": 0.0, "y": r},
            )

            # Same seed → same deterministic QMC path.  Any residual
            # deviation from the predicted scaling comes from the
            # spatial-spline interpolation error (O(h⁴) on the
            # 20×20×12 grid, ~ 1e-3).  Skip near-zero denominators
            # that would amplify spline noise.
            if abs(got_0) < 1e-10:
                continue

            predicted_ratio = expected_factor_unit ** n_cross
            actual_ratio = got_r / got_0
            rel = abs(actual_ratio - predicted_ratio) / abs(predicted_ratio)
            assert rel < 5e-3, (
                f"diagram #{i}: n_cross={n_cross} → expected ratio "
                f"{predicted_ratio:.6f}, got {actual_ratio:.6f} "
                f"(rel {rel:.2e}); values={got_0:.4e}, {got_r:.4e}"
            )

        assert found_cross > 0, (
            "No cross-group diagrams found — S5 did not exercise "
            "positional routing.  Check observable / action setup."
        )

    # ----- S6: rotation spline accuracy vs closed form ------------ #

    def test_S6_rotation_spline_matches_closed_form(
        self, cache_rotation_full,
    ):
        """``precompute_C_table_rotation(n_grid_cos)`` builds a 3-D
        spline in ``(t1, t2, cos θ)``.  The closed form for this
        test's rotation-invariant kernel is
        ``C(t1,t2; cos) = C_t(t1,t2) · (1 + cos θ)/2``.  Spline
        lookups must reproduce it.

        Exercises: the rotation mode's dispatch, the 2-D unit-vector
        representative construction, and the ``_rotation_cos``
        utility's sign handling.
        """
        from sft_wick.evaluate import _rotation_cos

        cache = cache_rotation_full
        rng = np.random.default_rng(2)
        t1 = rng.uniform(0.3, self.T_MAX - 0.3, size=20)
        t2 = rng.uniform(0.3, self.T_MAX - 0.3, size=20)

        # Random unit vectors in 2-D for the test (closed form is
        # rotation-invariant so the ambient dimension doesn't matter).
        ang1 = rng.uniform(0.0, 2 * np.pi, size=20)
        ang2 = rng.uniform(0.0, 2 * np.pi, size=20)
        x1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
        x2 = np.column_stack([np.cos(ang2), np.sin(ang2)])

        # C_at_batch expects (n,) or (n, d) — stack them.
        got = np.empty(20)
        for k in range(20):
            got[k] = cache.C_at_batch(
                np.array([t1[k]]), np.array([t2[k]]),
                x1[k], x2[k],
            )[0, 0]

        expected = np.array([
            C_closed_form(t1[k], t2[k], lam=self.LAM)
            * 0.5 * (1.0 + _rotation_cos(x1[k], x2[k]))
            for k in range(20)
        ])
        rel = np.abs(got - expected) / (np.abs(expected) + 1e-15)
        assert rel.max() < 5e-3, (
            f"rotation spline error {rel.max():.2e} exceeds 5e-3"
        )

    # ----- S6b: rotation mode with 3-D unit vectors (S^2 sphere) ----- #

    def test_S6b_rotation_supports_3d_unit_vectors(
        self, cache_rotation_full,
    ):
        """Rotation mode must accept ``n1, n2 in R^3`` unit vectors.

        Cosmological pipelines (CMB, line-of-sight lensing) supply
        S^2 directions as 3-component unit vectors; the existing S6
        test only covers d=2. ``_rotation_cos`` is dimension-agnostic
        (it only uses ``np.dot`` and ``np.linalg.norm``), so this
        smoke test locks the implicit d-dim support so a future
        refactor cannot accidentally collapse vectors via
        component-wise reductions.
        """
        from sft_wick.evaluate import _rotation_cos

        cache = cache_rotation_full
        rng = np.random.default_rng(20260428)
        t1 = rng.uniform(0.3, self.T_MAX - 0.3, size=20)
        t2 = rng.uniform(0.3, self.T_MAX - 0.3, size=20)

        # Sample 3-D unit vectors uniformly on S^2.
        u = rng.normal(size=(20, 3))
        v = rng.normal(size=(20, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        v /= np.linalg.norm(v, axis=1, keepdims=True)

        got = np.empty(20)
        for k in range(20):
            got[k] = cache.C_at_batch(
                np.array([t1[k]]), np.array([t2[k]]),
                u[k], v[k],
            )[0, 0]

        expected = np.array([
            C_closed_form(t1[k], t2[k], lam=self.LAM)
            * 0.5 * (1.0 + _rotation_cos(u[k], v[k]))
            for k in range(20)
        ])
        rel = np.abs(got - expected) / (np.abs(expected) + 1e-15)
        # 1.5e-2 matches the headroom used by the other linear-
        # interpolation regression tests in this class (S2/S3/S7);
        # what we are checking is that the d=3 input path doesn't
        # go off the rails, not the spline accuracy itself. Higher
        # truncation error than S6 because random 3-D points more
        # readily probe near the cos=±1 extremes.
        assert rel.max() < 1.5e-2, (
            f"3-D rotation spline error {rel.max():.2e} exceeds 1.5e-2"
        )

        # Sanity: cos values for these random unit vectors must
        # span (-1, 1), proving the test isn't accidentally probing
        # only one corner of the spline.
        cos_vals = np.array([_rotation_cos(u[k], v[k]) for k in range(20)])
        assert cos_vals.min() < -0.2 and cos_vals.max() > 0.2, (
            f"S^2 sample didn't cover both signs of cos: "
            f"min={cos_vals.min():.2f}, max={cos_vals.max():.2f}"
        )

    # ----- S7: lazy vs full-grid translation agreement + memoization - #

    def test_S7_lazy_translation_matches_full_and_memoizes(
        self, cache_translation_lazy, cache_translation_full, order2_dts,
    ):
        """Lazy translation mode (no r-grid) must produce the same
        per-diagram integrated value as the full-grid mode for the
        bubble-scaling test, AND the lazy cache must build exactly
        one 2-D time spline per distinct r value encountered (proving
        the memoization contract: no wasted builds).
        """
        dts, F_MSR = order2_dts
        r = 0.5
        cache_lazy = cache_translation_lazy
        cache_full = cache_translation_full

        # Start with a fresh lazy cache to count spline builds.
        # (We rebuild here rather than reusing the shared fixture so
        #  the memo count is deterministic.)
        fresh_lazy = self._make_fast_cache()
        fresh_lazy.precompute_C_table_translation(
            t_max=self.T_MAX, n_grid_t=25,
        )

        # Evaluate the first diagram (double tadpole: 0 cross-group C)
        # on BOTH caches at positions={x:0, y:r}.  The double tadpole
        # only queries r=0 regardless of the external y offset.
        dbl = [dt for dt in dts
               if all(p.spatial_left == p.spatial_right
                      for p in dt.propagators if p.kind == "C")][0]
        ig_dbl = dbl.build_integrand({"F": F_MSR})
        v_full, _ = ig_dbl.integrate_moment_qmc_vectorized(
            lambda_f=self.T_MAX, cache=cache_full, n_samples=2**12, seed=1,
            positions={"x": 0.0, "y": r},
        )
        v_lazy, _ = ig_dbl.integrate_moment_qmc_vectorized(
            lambda_f=self.T_MAX, cache=fresh_lazy, n_samples=2**12, seed=1,
            positions={"x": 0.0, "y": r},
        )
        rel_dbl = abs(v_full - v_lazy) / abs(v_full)
        # Two different linear-interpolation paths (3-D table vs
        # per-r 2-D table) → small discrepancy expected. Linear was
        # chosen over cubic to avoid steep-tail sign flips; see
        # test_evaluate_interpolation_accuracy.py. Empirically ~1.5e-2
        # on this well-sampled t-grid; 5e-2 leaves comfortable
        # randomness headroom while still catching dispatching bugs.
        assert rel_dbl < 5e-2, (
            f"S7 double-tadpole: full-grid {v_full:.6e} vs lazy "
            f"{v_lazy:.6e} (rel {rel_dbl:.2e})"
        )
        # Double tadpole queries only r=0 (self-loops) — exactly ONE
        # key in the lazy cache.
        assert len(fresh_lazy._lazy_translation._splines_by_key) == 1, (
            f"double tadpole should trigger 1 lazy build, got "
            f"{len(fresh_lazy._lazy_translation._splines_by_key)}"
        )

        # Now evaluate a bubble diagram (2 cross-group C's) to prove
        # the lazy cache handles distinct-r queries correctly.
        bubble = next(
            dt for dt in dts if self._count_cross_group_c(dt) > 0
        )
        ig_b = bubble.build_integrand({"F": F_MSR})
        v_b_full, _ = ig_b.integrate_moment_qmc_vectorized(
            lambda_f=self.T_MAX, cache=cache_full, n_samples=2**13, seed=1,
            positions={"x": 0.0, "y": r},
        )
        v_b_lazy, _ = ig_b.integrate_moment_qmc_vectorized(
            lambda_f=self.T_MAX, cache=fresh_lazy, n_samples=2**13, seed=1,
            positions={"x": 0.0, "y": r},
        )
        rel_b = abs(v_b_full - v_b_lazy) / abs(v_b_full)
        # Same linear-interpolation tolerance argument as above
        # (5e-2). Bubble has 2 cross-group C edges so error compounds;
        # observed values typically ~1-3e-2 on this seed/grid.
        assert rel_b < 5e-2, (
            f"S7 bubble: full {v_b_full:.6e} vs lazy {v_b_lazy:.6e} "
            f"(rel {rel_b:.2e})"
        )
        # Bubble has 2 cross-group C's at r=0.5 plus may or may not
        # have a self-loop at r=0; lazy cache should now hold at most
        # 2 distinct keys (r=0 from the tadpole test + r=0.5 from the
        # bubble).  Definitely < 5 (no wasteful rebuilds).
        n_keys = len(fresh_lazy._lazy_translation._splines_by_key)
        assert n_keys <= 3, (
            f"lazy cache accumulated {n_keys} keys — should be ≤ 3 "
            f"(one per unique r across both diagrams)"
        )


# --------------------------------------------------------------------- #
# Phase 6 — White-noise (δ-correlated) source-field contribution (W1–W3)
# --------------------------------------------------------------------- #


class TestWhiteNoiseComponent:
    """Phase-6 deductive tests: the ``κ²(t₁, t₂) = κ²_smooth + δ(t₁−t₂)·σ²``
    extension.

    When the source-field correlation contains a δ-correlated piece,
    the usual 2-D dblquad for C collapses one integration via the
    δ, yielding an additive ``C_white`` built from a 1-D integral::

        C_white(t₁, t₂; x₁, x₂) =
            ∫_{t_min}^{min(t₁,t₂)} R(t₁, τ) σ²(τ; x₁, x₂) R(t₂, τ) dτ.

    This class verifies:

    =========  ================================================================
    Test ID     Claim
    =========  ================================================================
    W1          Pure white noise (``κ²_smooth = 0``, constant σ²) matches
                the closed form ``C(t₁,t₂) = σ² · exp(−γ|Δt|) · (1 − exp(−2γ
                min(t₁,t₂))) / (2γ)`` to quad precision.
    W2          Additivity: C with (smooth + white) equals the sum of
                C(smooth only) and C(white only) — no spurious coupling
                across the two channels.
    W3          The white-noise contribution is **automatically absorbed**
                into the Phase-5 translation-mode spline, so an end-to-end
                bubble diagram at r > 0 still obeys the cross-group
                scaling ``value(r) / value(0) = exp(−r/σ_x)^(n_cross_C)``
                when σ² carries the same spatial kernel
                ``exp(−|x₁−x₂|/σ_x)``.
    =========  ================================================================
    """

    LAM = 0.05
    SIGMA_T = 0.3
    SIGMA_X = 1.0
    GAMMA = 1.0
    SIGMA2_AMPL = 0.3
    T_MAX = 3.0

    def _R_time(self, t1, t2):
        return float(np.exp(-self.GAMMA * (t1 - t2))) if t1 >= t2 else 0.0

    def _kappa2_smooth(self, n1, t1, n2, t2):
        """Same κ²_smooth as Phase-5: OU in time, exponential in |Δx|."""
        r = abs(float(np.asarray(n1).sum()) - float(np.asarray(n2).sum()))
        kt = self.LAM * np.exp(-abs(t1 - t2) / self.SIGMA_T)
        kx = np.exp(-r / self.SIGMA_X)
        return kt * kx * np.eye(1)

    def _sigma2_const(self, n1, t, n2):
        """Position-independent white-noise amplitude (W1, W2)."""
        return np.array([[self.SIGMA2_AMPL]])

    def _sigma2_translation(self, n1, t, n2):
        """Spatially translation-invariant white-noise kernel (W3)."""
        r = abs(float(np.asarray(n1).sum()) - float(np.asarray(n2).sum()))
        return np.array([[self.SIGMA2_AMPL * np.exp(-r / self.SIGMA_X)]])

    def _closed_form_white(self, t1, t2, s2=None):
        """Closed form for OU R + constant σ² white noise (zero smooth)."""
        s2 = self.SIGMA2_AMPL if s2 is None else s2
        g = self.GAMMA
        tl = min(t1, t2)
        if tl <= 0:
            return 0.0
        return s2 * np.exp(-g * abs(t1 - t2)) * (1 - np.exp(-2 * g * tl)) / (2 * g)

    # --------------- W1 --------------- #

    def test_W1_pure_white_noise_matches_closed_form(self):
        """With zero ``κ²_smooth`` and constant ``σ²``, C reduces to
        a 1-D integral with an elementary closed form.  Verifies the
        white-noise machinery in isolation."""
        model = PropagatorModel(
            R_time=self._R_time,
            kappa2=lambda n1, t1, n2, t2: np.zeros((1, 1)),
            sigma2=self._sigma2_const,
            n_components=1, iso_R=True, diag_C=True, t_min=0.0,
        )
        cache = PropagatorCache(model)

        for t1, t2 in [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (0.5, 1.5)]:
            got = cache._C_value_direct(
                np.asarray(0.0), t1, np.asarray(0.0), t2,
            )[0, 0]
            want = self._closed_form_white(t1, t2)
            rel = abs(got - want) / abs(want) if want != 0 else abs(got)
            assert rel < 1e-10, (
                f"W1 at (t1={t1}, t2={t2}): got {got:.8e} vs closed "
                f"form {want:.8e}, rel={rel:.2e}"
            )

    # --------------- W2 --------------- #

    def test_W2_additivity_smooth_plus_white(self):
        """C(smooth + white) must equal C(smooth only) + C(white only).

        Tests that the two channels don't interfere numerically and
        that the δ really does collapse one time integral (no spurious
        cross-terms).
        """
        # Model 1: smooth only
        model_s = PropagatorModel(
            R_time=self._R_time, kappa2=self._kappa2_smooth,
            n_components=1, iso_R=True, diag_C=True, t_min=0.0,
        )
        # Model 2: white only
        model_w = PropagatorModel(
            R_time=self._R_time,
            kappa2=lambda n1, t1, n2, t2: np.zeros((1, 1)),
            sigma2=self._sigma2_const,
            n_components=1, iso_R=True, diag_C=True, t_min=0.0,
        )
        # Model 3: both
        model_sw = PropagatorModel(
            R_time=self._R_time, kappa2=self._kappa2_smooth,
            sigma2=self._sigma2_const,
            n_components=1, iso_R=True, diag_C=True, t_min=0.0,
        )

        cache_s = PropagatorCache(model_s)
        cache_w = PropagatorCache(model_w)
        cache_sw = PropagatorCache(model_sw)

        for t1, t2, r in [(1.0, 1.5, 0.0), (0.8, 1.2, 0.4),
                           (2.0, 2.0, 1.0)]:
            C_s = cache_s._C_value_direct(
                np.asarray(0.0), t1, np.asarray(r), t2,
            )[0, 0]
            C_w = cache_w._C_value_direct(
                np.asarray(0.0), t1, np.asarray(r), t2,
            )[0, 0]
            C_sw = cache_sw._C_value_direct(
                np.asarray(0.0), t1, np.asarray(r), t2,
            )[0, 0]
            want = C_s + C_w
            rel = abs(C_sw - want) / abs(want) if want != 0 else 0
            assert rel < 1e-10, (
                f"W2 additivity at (t1={t1}, t2={t2}, r={r}): "
                f"C_smooth={C_s:.6e}, C_white={C_w:.6e}, "
                f"C_both={C_sw:.6e}, sum={want:.6e}, rel={rel:.2e}"
            )

    # --------------- W3 --------------- #

    def test_W3_white_noise_absorbed_into_translation_spline(self):
        """Build a Phase-5 translation-mode lazy cache from a model
        that has BOTH ``κ²_smooth`` and a translation-invariant
        white-noise ``σ²``.  The resulting bubble diagram at r > 0
        must still obey the cross-group scaling
        ``value(r) / value(0) = exp(-r/σ_x)^(n_cross_C)``, because
        *both* contributions carry the same spatial kernel — the
        white-noise piece is silently absorbed into
        ``_C_value_direct`` and therefore into the 2-D time splines
        built per r on demand.
        """
        model = PropagatorModel(
            R_time=self._R_time, kappa2=self._kappa2_smooth,
            sigma2=self._sigma2_translation,
            n_components=1, iso_R=True, diag_C=True, t_min=0.0,
        )
        cache = PropagatorCache(model)
        cache.precompute_C_table_translation(
            t_max=self.T_MAX, n_grid_t=15,  # lazy mode — trimmed from 25
                                            # (costs come from lazy
                                            # per-r dblquad grid build,
                                            # not from QMC sampling)
        )

        # Order-2 bubble diagram setup (same as Phase-5 S5)
        F_MSR = np.array(-1j)
        reset_uid_counter()
        phi = Field("phi", "physical", n_components=1)
        psi = Field("psi", "response", n_components=1)
        action = Action(vertices=[Vertex([psi, phi, phi], coupling="F")])
        obs = [phi("x"), phi("y")]
        result = compute_moment(
            obs, action, order=2, response_phase=True, ito=True,
            diag_R=True, diag_C=True, iso_R=True, iso_C=True,
        )
        dts = result.diagram_terms(2)

        def _count_cross(dt):
            sp = dt.analyze_spatial()
            return sum(
                1 for p in dt.propagators if p.kind == "C"
                and sp.direction_map[p.spatial_left]
                    != sp.direction_map[p.spatial_right]
            )

        r = 0.5
        factor_unit = np.exp(-r / self.SIGMA_X)

        # Pick a bubble-type (n_cross >= 2) diagram
        bubble = next(dt for dt in dts if _count_cross(dt) >= 2)
        n_cross = _count_cross(bubble)
        ig = bubble.build_integrand({"F": F_MSR})

        v0, _ = ig.integrate_moment_qmc_vectorized(
            lambda_f=self.T_MAX, cache=cache, n_samples=2**11, seed=3,
            positions={"x": 0.0, "y": 0.0},
        )
        vr, _ = ig.integrate_moment_qmc_vectorized(
            lambda_f=self.T_MAX, cache=cache, n_samples=2**11, seed=3,
            positions={"x": 0.0, "y": r},
        )
        predicted = factor_unit ** n_cross
        actual = vr / v0
        rel = abs(actual - predicted) / abs(predicted)
        assert rel < 5e-3, (
            f"W3 bubble with white noise: n_cross={n_cross}, "
            f"expected ratio {predicted:.6f}, got {actual:.6f} "
            f"(rel {rel:.2e})"
        )
