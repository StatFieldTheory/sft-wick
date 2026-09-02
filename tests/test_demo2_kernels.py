"""Demo-2 (non-Gaussian noise, ``alpha``) kernels and channels.

Nothing in ``tests/`` used to pin a demo-2 number.
``tests/test_R_contracted_vertex.py`` exercises the
``already_R_contracted`` machinery only on a CONSTANT ``κ³`` -- the one
case where the R-contraction factorises exactly and the two routes must
agree for trivial reasons -- and the ``validate_k*_R.py`` scripts in
``examples/paper_assets/demo2_kappa4/`` are loose scripts that nothing
runs.  So the two hand-written kernels the demo2 results rest on, and
the channel values the paper quotes, were unprotected.

This module pins:

* **DK1** ``k3_R`` against cusp-aware 3-D adaptive quadrature of the raw
  leg integral, at coincident / FK-type / split / short / long partner
  times and with spatial factors.  Tolerances are the MEASURED errors
  (see ``examples/demo2/k3_R_coupling.py``'s accuracy table) with
  headroom, not round numbers.
* **DK2** ``k4_R`` against randomised-Sobol QMC of the raw 4-leg
  integral, with the reference's own seed scatter quoted.
* **DK3** the boundary test ``already_R_contracted`` never had on a
  NON-constant kernel: the FK channel through the raw ``κ³`` (4-D
  tensor-product rule) must converge ONTO the R-contracted answer as
  the raw rule is refined.  Constant-``κ³`` agreement (test_R_contracted
  _vertex.py) cannot see a leg-aliasing error that depends on the
  kernel's time structure; this can.
* **DK4/DK5** one pinned FK value and one pinned order-0 value for the
  demo2 system, so a numerical regression in either shows up as drift
  rather than as a changed figure nobody re-reads.
* **DK6** the ``already_R_contracted`` contract's silent assumptions --
  see the module docstring of ``examples/demo2/k3_R_coupling.py``.  The
  runtime does not check any of them; the callable must.

The kernels live under ``examples/`` rather than in the package, so
they are loaded by path.
"""
from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

import numpy as np
import pytest

import sft_wick as sw

_ROOT = Path(__file__).resolve().parents[1]
_DEMO2 = _ROOT / "examples" / "demo2"
_ASSETS = _ROOT / "examples" / "paper_assets" / "demo2_kappa4"

LAM, SIGMA_T, SIGMA_X, GAMMA, ALPHA, N = 0.05, 0.3, 1.0, 1.0, 0.6, 2


def _load(path: Path, name: str):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def k3R():
    return _load(_DEMO2 / "k3_R_coupling.py", "demo2_k3_R_coupling")


@pytest.fixture(scope="module")
def k3raw():
    return _load(_DEMO2 / "k3_coupling.py", "demo2_k3_coupling")


@pytest.fixture(scope="module")
def k4R():
    _load(_ASSETS / "k4_coupling.py", "demo2_k4_coupling")
    return _load(_ASSETS / "k4_R_contracted.py", "demo2_k4_R_contracted")


@pytest.fixture(scope="module")
def k4raw():
    return _load(_ASSETS / "k4_coupling.py", "demo2_k4_coupling")


# ---------------------------------------------------------------------
# DK1 -- k3_R against the raw leg integral
# ---------------------------------------------------------------------

def _brute_k3(mod, t1, t2, t3, x=(0.0, 0.0, 0.0)):
    """``∫∫∫ du R(t1',u1) R(t2',u2) R(t3',u3) κ³(u; x)`` by nested
    adaptive quadrature, with explicit break points at the OUTER legs'
    times so each 1-D ``quad`` sees the ``|u_i − u_j|`` kinks as panel
    edges instead of trying to resolve them adaptively.  Without the
    break points the reference is itself only ~1e-6 accurate and the
    comparison measures the reference, not the kernel.
    """
    from scipy.integrate import nquad
    g = mod.GAMMA

    def f(u3, u2, u1):
        R = np.exp(-g * ((t1 - u1) + (t2 - u2) + (t3 - u3)))
        return R * mod.kappa3_raw(u1, u2, u3, *x)

    opt = dict(epsabs=1e-15, epsrel=1e-11, limit=400)
    opts = [
        lambda u2, u1: dict(points=sorted({p for p in (u1, u2) if 0 < p < t3}), **opt),
        lambda u1: dict(points=sorted({p for p in (u1,) if 0 < p < t2}), **opt),
        dict(**opt),
    ]
    val, err = nquad(f, [[0, t3], [0, t2], [0, t1]], opts=opts)
    return val, err


@pytest.mark.parametrize("t1,t2,t3,tol,note", [
    # tol = the measured relative error, rounded up ~2x.  See the
    # accuracy table in examples/demo2/k3_R_coupling.py.
    (3.0, 1.5, 1.5, 5e-6, "FK-type (t', s, s) -- what the kernel was tuned on"),
    (1.0, 1.0, 1.0, 3e-4, "coincident, moderate"),
    (20.0, 2.0, 0.2, 3e-3, "EXTREME split; reference itself only ~5e-3 here"),
    (0.05, 0.03, 0.04, 1.5e-3, "all VERY SHORT"),
])
def test_DK1_k3R_matches_raw_leg_quadrature(k3R, t1, t2, t3, tol, note):
    ref, _err = _brute_k3(k3R, t1, t2, t3)
    got = float(k3R.k3_R([t1], [t2], [t3], [1.0], [1.0], [1.0])[0])
    rel = abs(got - ref) / abs(ref)
    assert np.isfinite(got)
    assert rel < tol, f"{note}: k3_R {got:.8e} vs reference {ref:.8e} (rel {rel:.2e})"


def test_DK1_k3R_matches_raw_leg_quadrature_with_spatial(k3R):
    """The spatial factors enter as constants pulled out of the leg
    integral, so a transposed / mismatched (s12, s13, s23) argument
    order is invisible at coincident positions.  Three DISTINCT
    separations catch it."""
    x = (0.0, 0.5, 1.5)
    tt = (3.0, 1.5, 1.5)
    ref, _ = _brute_k3(k3R, *tt, x=x)
    s12 = np.exp(-abs(x[0] - x[1]) / SIGMA_X)
    s13 = np.exp(-abs(x[0] - x[2]) / SIGMA_X)
    s23 = np.exp(-abs(x[1] - x[2]) / SIGMA_X)
    got = float(k3R.k3_R([tt[0]], [tt[1]], [tt[2]], [s12], [s13], [s23])[0])
    assert abs(got - ref) / abs(ref) < 1e-4


# ---------------------------------------------------------------------
# DK6 -- the already_R_contracted contract
# ---------------------------------------------------------------------

def test_DK6_k3R_enforces_leg_causality_and_t_min_zero(k3R):
    """``already_R_contracted=True`` makes the CALLABLE responsible for
    the leg integral's limits: the runtime only aliases each absorbed
    leg's time onto its partner's and skips the R factor.  Two limits
    the callable must impose, neither of which the package checks:

    * ``u_i <= t_i'`` (leg causality) -- so a vanishing partner time
      leaves no support and the kernel must be exactly zero;
    * ``t_min = 0`` -- the lower limit is hard-wired, so the kernel must
      be non-decreasing in every partner time (κ³ > 0 here: widening a
      leg window can only add mass).
    """
    assert float(k3R.k3_R([0.0], [1.0], [1.0], [1.0], [1.0], [1.0])[0]) == 0.0
    assert float(k3R.k3_R([1.0], [0.0], [1.0], [1.0], [1.0], [1.0])[0]) == 0.0
    assert float(k3R.k3_R([1.0], [1.0], [0.0], [1.0], [1.0], [1.0])[0]) == 0.0

    ts = np.array([0.1, 0.3, 0.6, 1.0, 2.0, 5.0, 15.0, 50.0])
    ones = np.ones_like(ts)
    vals = k3R.k3_R(ts, ts, ts, ones, ones, ones)
    assert np.all(vals > 0)
    assert np.all(np.diff(vals) > 0), f"not monotone in t': {vals}"

    # Raising ONE partner time alone is a different story, and worth
    # pinning because it is easy to get backwards: the leg window grows
    # but its R weight exp(-gamma (t1' - u1)) decays faster, so K_R
    # rises to a peak near t1' ~ t2' and then falls off as
    # exp(-gamma t1').  Anything that mishandled the causal upper limit
    # (say by clamping u1 to t3' as well) would flatten this tail.
    t_lag = np.array([15.0, 50.0])
    two = np.full_like(t_lag, 2.0)
    tail = k3R.k3_R(t_lag, two, two, np.ones(2), np.ones(2), np.ones(2))
    decay = tail[1] / tail[0]
    assert decay == pytest.approx(np.exp(-GAMMA * 35.0), rel=2e-2), (
        f"lagging-leg tail decays as {decay:.3e}, expected "
        f"exp(-gamma dt) = {np.exp(-GAMMA * 35.0):.3e}"
    )


def test_DK6_already_R_contracted_rejects_equal_time():
    """The two absorption mechanisms both rewrite leg times and cannot
    be combined; the spec rejects it at construction rather than
    silently producing a wrong alias."""
    with pytest.raises((ValueError, TypeError)):
        sw.NonLocalVertex("K", order=3, coupling=np.zeros((N,) * 3),
                          already_R_contracted=True, equal_time=True)


# ---------------------------------------------------------------------
# DK3 -- raw kappa^3 vs already_R_contracted on a NON-constant kernel
# ---------------------------------------------------------------------

def _demo2_system(coupling_fn, *, r_contracted: bool):
    """demo2's F + kappa^3 action.  ``lam`` is the BARE variance: the
    alpha shift is carried explicitly by the K vertex."""
    F = np.zeros((N, N, N))
    F[0, 1, 1] = 1.0
    F[1, 0, 1] = F[1, 1, 0] = 0.5
    return sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=sw.DiagonalA(gamma=[GAMMA, GAMMA]),
        vertices=[sw.LocalVertex("F", coupling=F)],
        nonlocal_vertices=[sw.NonLocalVertex(
            "K", 3, coupling=coupling_fn, coupling_vectorized=True,
            already_R_contracted=r_contracted)],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=LAM, sigma_t=SIGMA_T),
            spatial=sw.ExponentialSpatial(sigma_x=SIGMA_X))),
    )


def _fk_value(system, t_final, n_gauss, t_max=5.0):
    props = system.propagators(
        t_max=t_max, n_grid_t=40, c_closed_form="auto",
        c_closed_form_only=True, c_closed_form_vectorized=True, progress=False,
    )
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2], progress=False)
    return expansion.evaluate(
        props, positions={"x": 0.0, "y": 0.0}, t_final=t_final,
        component_pair=(0, 1), orders=[2], vertex_types=["FK"],
        method="gauss_legendre", n_gauss=n_gauss, n_jobs=1,
    ).total


def test_DK3_raw_kappa3_converges_onto_the_R_contracted_answer(k3raw, k3R):
    """The boundary test the ``already_R_contracted`` feature never had.

    ``tests/test_R_contracted_vertex.py`` compares the two routes on a
    CONSTANT kappa^3, where the leg integral factorises into three
    independent 1-D integrals and any leg-aliasing bug that depends on
    the kernel's time structure cancels.  Here the kernel is demo2's
    real, narrow-in-relative-time kappa^3, so the routes agree only if
    the aliasing is right.

    The comparison has to respect which side is converged.  The raw
    route integrates a 4-D integrand with a kernel of width sigma_t on
    a simplex of size t_f, so its tensor-product rule converges slowly
    -- that is the whole reason the R-contracted route exists.  We
    therefore run at a SMALL t_f (t_f = 0.5, where the narrow kernel
    still covers a decent fraction of the simplex) and assert that the
    raw answer approaches the R-contracted one MONOTONICALLY as the raw
    rule is refined.  Convergence onto the value, rather than equality
    at one node count, is what distinguishes "the two routes agree" from
    "the two routes happen to be close".
    """
    sw.reset_uid_counter()
    rc = _fk_value(_demo2_system(k3R.coupling_fn_vectorized, r_contracted=True),
                   t_final=0.5, n_gauss=24)
    sw.reset_uid_counter()
    raw_sys = _demo2_system(k3raw.coupling_fn_vectorized, r_contracted=False)
    raw = {n: _fk_value(raw_sys, t_final=0.5, n_gauss=n) for n in (10, 16, 22)}

    rel = {n: abs(v - rc) / abs(rc) for n, v in raw.items()}
    assert rel[10] > rel[16] > rel[22], (
        f"raw rule does not converge onto the R-contracted answer: {rel}"
    )
    assert rel[22] < 6e-4, (
        f"raw(GL22) {raw[22]:.8e} vs R-contracted {rc:.8e} "
        f"(rel {rel[22]:.2e}); measured 4.1e-4 on 2026-09-02"
    )
    # ... and the residual is the RAW rule's own error, not a
    # disagreement: its node-to-node change is the same size.
    assert abs(raw[22] - raw[16]) / abs(rc) > 0.3 * rel[22]


# ---------------------------------------------------------------------
# DK2 -- k4_R against the raw 4-leg integral
# ---------------------------------------------------------------------

def _qmc_k4(mod_R, mod_raw, tp, n=2 ** 18, n_seed=4):
    """Randomised-Sobol reference for the raw 4-leg integral, with its
    OWN uncertainty from the scrambling scatter.  4-D adaptive
    quadrature of a four-cusp integrand is not practical, so the
    reference is stochastic and the test must check that it is precise
    enough to be a reference at all -- hence returning the error."""
    from scipy.stats import qmc as _qmc
    tp = np.asarray(tp, float)
    vals = []
    for seed in range(n_seed):
        u = _qmc.Sobol(4, scramble=True, seed=seed).random(n) * tp
        f = np.exp(-mod_R.GAMMA * np.sum(tp[None, :] - u, axis=1)) \
            * mod_raw.kappa4_amplitude(np.zeros((4, u.shape[0])), u.T)
        vals.append(f.mean() * np.prod(tp))
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(n_seed))


@pytest.mark.parametrize("tp,tol,note", [
    ((1.0, 1.0, 1.0, 1.0), 1e-3, "coincident, moderate"),
    ((3.0, 1.5, 1.5, 1.5), 3e-3, "FFK4-type"),
    ((0.2, 0.2, 0.2, 0.2), 2e-2, "VERY SHORT -- the composite grid's worst corner"),
])
def test_DK2_k4R_matches_raw_leg_quadrature(k4R, k4raw, tp, tol, note):
    ref, ref_err = _qmc_k4(k4R, k4raw, tp)
    s = {(i, j): np.ones(1) for i in range(4) for j in range(4) if i != j}
    got = float(k4R.k4_R(np.asarray(tp, float)[:, None], s)[0])
    rel = abs(got - ref) / abs(ref)
    # The reference must be precise enough that the comparison measures
    # the kernel and not the Sobol scatter.
    assert ref_err / abs(ref) < tol / 3, (
        f"{note}: QMC reference too noisy ({ref_err / abs(ref):.1e}) to test at {tol:.0e}"
    )
    assert rel < tol, f"{note}: k4_R {got:.7e} vs QMC {ref:.7e} (rel {rel:.2e})"


def test_DK2_k4R_saturates_at_late_partner_times(k4R):
    """Beyond a few 1/gamma the leg integrals stop growing: R has
    already decayed over the kernel's width, so K_R is t'-independent.
    Randomised QMC cannot check this -- at t' = 50 a Sobol rule on the
    50^4 box has 80 % seed scatter, because the integrand lives in a
    sigma_t-sized corner -- but the kernel must still show it, and a
    normalisation that leaked a factor of t' would not.
    """
    s = {(i, j): np.ones(1) for i in range(4) for j in range(4) if i != j}
    v15 = float(k4R.k4_R(np.full((4, 1), 15.0), s)[0])
    v50 = float(k4R.k4_R(np.full((4, 1), 50.0), s)[0])
    assert v15 == pytest.approx(v50, rel=1e-10)
    assert 3.0e-5 < v15 < 5.0e-5


# ---------------------------------------------------------------------
# DK7 -- the single-site cumulant ladder
# ---------------------------------------------------------------------

def _cumulant_closed_form(n: int) -> float:
    """Exact ``kappa_n`` of ``X = eta + alpha (eta^2 - lam)`` with
    ``eta ~ N(0, lam)``.

    ``E[exp(a eta + b eta^2)] = (1 - 2 b lam)^(-1/2)
    exp(a^2 lam / (2 (1 - 2 b lam)))`` gives, with ``a = s``,
    ``b = s alpha`` and ``u = 2 alpha lam``,

        K(s) = -s alpha lam - (1/2) log(1 - s u) + s^2 lam / (2 (1 - s u))

    whose Taylor coefficients are

        kappa_n = n! [ u^n / (2n) + (lam / 2) u^(n-2) ],   n >= 2.

    This reproduces the hand-derived ``lam + 2 a^2 l^2``,
    ``6 a l^2 + 8 a^3 l^3`` and ``48 a^2 l^3 + 48 a^4 l^4`` that demo2
    uses, and continues the ladder past them.
    """
    import math
    u = 2.0 * ALPHA * LAM
    return math.factorial(n) * (u ** n / (2 * n) + 0.5 * LAM * u ** (n - 2))


def test_DK7_demo2_cumulant_formulas_match_the_closed_form(k4raw):
    """``k4_coupling.single_site_cumulants`` writes kappa2/3/4 as
    hand-expanded polynomials in (alpha, lam).  A dropped term there is
    exactly the class of error the 0.3.0 revision found (kappa^3 was
    missing its ``8 alpha^3 lam^3``), so pin all three against the
    generating function rather than against themselves.
    """
    k2, k3, k4 = k4raw.single_site_cumulants()
    assert k2 == pytest.approx(_cumulant_closed_form(2), rel=1e-14)
    assert k3 == pytest.approx(_cumulant_closed_form(3), rel=1e-14)
    assert k4 == pytest.approx(_cumulant_closed_form(4), rel=1e-14)
    # ... and the hand-written forms themselves.
    assert k2 == pytest.approx(LAM + 2 * ALPHA ** 2 * LAM ** 2, rel=1e-14)
    assert k3 == pytest.approx(6 * ALPHA * LAM ** 2 + 8 * ALPHA ** 3 * LAM ** 3, rel=1e-14)
    assert k4 == pytest.approx(48 * ALPHA ** 2 * LAM ** 3 + 48 * ALPHA ** 4 * LAM ** 4, rel=1e-14)


def test_DK7_cumulant_ladder_does_not_terminate_at_four():
    """demo2's action carries kappa^2, kappa^3 and kappa^4 and stops.
    That is a TRUNCATION, not an exact statement about the model: the
    quadratic deformation has every cumulant.  kappa^5 is 7 % of
    kappa^3 and its lowest channel (F^3 kappa^5) sits at the SAME
    perturbative order as F^3 kappa^3 -- so it is inside the residual
    and inside the s^3 coefficient of the amplitude-scaling fit.  This
    test exists so that the truncation is a recorded fact with a size
    attached, not an unexamined assumption; see
    ``examples/demo2/INTERPRETATION.md``.
    """
    k3 = _cumulant_closed_form(3)
    ratios = {n: _cumulant_closed_form(n) / k3 for n in (4, 5, 6, 7)}
    assert ratios[4] == pytest.approx(0.2386, abs=1e-4)
    assert ratios[5] == pytest.approx(0.0713, abs=1e-4)
    assert ratios[6] == pytest.approx(0.0256, abs=1e-4)
    # Strictly decreasing, i.e. the series is at least well ordered.
    assert all(ratios[n] > ratios[n + 1] for n in (4, 5, 6))


# ---------------------------------------------------------------------
# DK4 / DK5 -- pinned demo2 channel values
# ---------------------------------------------------------------------

def _c_eff_exact():
    """demo2's EXACT effective covariance,
    ``kappa2_eff = lam k + 2 alpha^2 lam^2 k^2``.

    Both pieces are separable exponentials, so both have the built-in
    closed form; their sum is what order 0 and FF must be evaluated
    with.  The single-kernel ``lam_eff = lam (1 + 2 alpha^2 lam)``
    approximation the demo used before 0.3.0 gets the amplitude right
    and the CORRELATION LENGTHS wrong, and is worth +1.8e-4 at
    ``xi_00(t = 15)`` -- 1.5 % -- which is 13 sigma of the simulation.
    """
    from sft_wick.workflow.closed_forms import ClosedFormC
    a = ClosedFormC(gamma=(GAMMA, GAMMA), lam=LAM, sigma_t=SIGMA_T,
                    spatial=sw.ExponentialSpatial(sigma_x=SIGMA_X))
    b = ClosedFormC(gamma=(GAMMA, GAMMA), lam=2 * ALPHA ** 2 * LAM ** 2,
                    sigma_t=SIGMA_T / 2,
                    spatial=sw.ExponentialSpatial(sigma_x=SIGMA_X / 2))
    return lambda n1, t1, n2, t2: a(n1, t1, n2, t2) + b(n1, t1, n2, t2)


def test_DK4_pinned_FK_value(k3R):
    """One pinned number for the order-2 F x kappa^3 channel.

    This is the channel the demo2 headline rests on and the one the
    0.3.0 revision changed twice (the missing ``8 alpha^3 lam^3`` term
    in kappa^3, and the un-converged 4-D tensor rule).  Pinning it means
    a third change shows up as drift here rather than as a quietly
    different figure.  Cross-checked against
    ``examples/paper_assets/demo2_kappa4/budget.npz``.
    """
    sw.reset_uid_counter()
    system = _demo2_system(k3R.coupling_fn_vectorized, r_contracted=True)
    val = _fk_value(system, t_final=1.0, n_gauss=32, t_max=50.0)
    assert val == pytest.approx(1.5841e-04, rel=2e-4)


def test_DK5_pinned_order0_value():
    """One pinned number for order 0 with the exact two-kernel C_eff.

    Order 0 is 96 % of ``xi_00``, so an error here dwarfs every
    perturbative channel; it is also the only place the exact C_eff
    enters on its own, un-mixed with a vertex.
    """
    F = np.zeros((N, N, N))
    F[0, 1, 1] = 1.0
    F[1, 0, 1] = F[1, 1, 0] = 0.5
    sw.reset_uid_counter()
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=sw.DiagonalA(gamma=[GAMMA, GAMMA]),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=LAM, sigma_t=SIGMA_T),
            spatial=sw.ExponentialSpatial(sigma_x=SIGMA_X))),
    )
    props = system.propagators(
        t_max=50.0, n_grid_t=60, c_closed_form=_c_eff_exact(),
        c_closed_form_only=True, progress=False,
    )
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0], progress=False)
    val = expansion.evaluate(
        props, positions={"x": 0.0, "y": 0.0}, t_final=1.0,
        component_pair=(0, 0), orders=[0], n_jobs=1,
    ).total
    assert val == pytest.approx(8.9600e-03, rel=2e-4)
    # The cross-component value must vanish exactly: the KroneckerDelta
    # the diagonal-simplification pass retains is what makes it do so.
    off = expansion.evaluate(
        props, positions={"x": 0.0, "y": 0.0}, t_final=1.0,
        component_pair=(0, 1), orders=[0], n_jobs=1,
    ).total
    assert off == 0.0
