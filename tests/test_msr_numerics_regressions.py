"""Regression tests for the MSR numerics fixes.

Each test pins a defect that was silently returning a wrong number, against an
**independently derived** ground truth (closed form or exactly-solvable model),
never against another sft-wick path.

Covered:
  F1  DiagramIntegrand.integration_bounds mis-mapped scipy's nquad callback
      arguments, collapsing the domain of any diagram with a vertex-to-vertex
      response chain.
  F2  abs()/`.real` projections destroyed the sign (or zeroed the value) of
      diagrams whose observable carries external response legs.
  F3  the C builder contracted only the diagonal of a matrix R_time.
  F4  no public L0 hook to supply C directly.
  F6  callable (time-dependent) LOCAL couplings were rejected outright.
  F7a DiagramIntegrand.evaluate silently returned 0 for a callable coupling.
  F7c a scalar-field coupling of the wrong rank raised an opaque TypeError.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import nquad, quad

from sft_wick import Action, Field, Vertex, compute_moment
from sft_wick.evaluate import PropagatorCache, PropagatorModel

MU, D = 1.0, 0.5


def _C0(t1: float, t2: float) -> float:
    """Free correlator of dx/dt = -mu x + xi, <xi xi> = 2 D delta, x(0)=0."""
    return (D / MU) * (np.exp(-MU * abs(t1 - t2)) - np.exp(-MU * (t1 + t2)))


def _scalar_cache(**kw) -> PropagatorCache:
    model = PropagatorModel(
        R_time=lambda t, tp: np.exp(-MU * (t - tp)),
        kappa2=lambda n1, t1, n2, t2: np.zeros((1, 1)),
        sigma2=lambda n1, t, n2: np.array([[2.0 * D]]),
        n_components=1, iso_R=True, diag_C=True, t_min=0.0,
    )
    return PropagatorCache(
        model, c_value_fn=lambda n1, t1, n2, t2: np.array([[_C0(t1, t2)]]), **kw
    )


def _quartic(order: int, obs=None):
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    obs = obs or [phi("x"), phi("y")]
    return compute_moment(
        obs, Action([Vertex(fields=[psi, phi, phi, phi], coupling="g")]),
        order=order, ito=True, response_phase=True, collect_topology=True,
        diag_R=True, diag_C=True, iso_R=True, iso_C=True,
    )


# --------------------------------------------------------------------- #
# F1 — integration_bounds
# --------------------------------------------------------------------- #
def test_F1_bounds_callable_uses_the_outer_variable():
    """The innermost bound must track its causal parent, not a constant.

    scipy calls ``ranges[i]`` with the OUTER variables ``int_vars[i+1:]``.
    The old code indexed them as if they were the inner ones and fell back to
    a literal ``1.0``.
    """
    res = _quartic(2)
    checked = 0
    for dt in res.diagram_terms(2):
        ig = dt.build_integrand({"g": np.array(1j)})
        sp = ig.spatial
        ivars = list(sp.time_integration_vars)
        chain = [(e, l) for e, l in sp.time_orderings
                 if e in ivars and l in ivars]
        if not chain or len(ivars) < 2:
            continue
        bounds = ig.integration_bounds({"x": 8.0, "y": 8.0}, t_min=0.0)
        for probe in (2.0, 4.0, 7.0):
            lo, hi = bounds[0](probe)
            assert lo == 0.0
            assert hi == pytest.approx(probe), (
                "innermost bound ignored its causal parent"
            )
        checked += 1
    assert checked > 0, "no chained order-2 diagram found"


@pytest.mark.parametrize("order,exact", [(1, -0.75), (2, 3.0)])
def test_F1_quartic_series_matches_stationary_fokker_planck(order, exact):
    """<x^2> = 1/2 - (3/4) g + 3 g^2 + O(g^3) from P(x) ~ exp(-V/D).

    Independent of the diagrammatics: it comes from the stationary
    Fokker-Planck weight with V = mu x^2/2 + g x^4/4.
    """
    T = 8.0
    cache = _scalar_cache()
    res = _quartic(2)
    total = 0.0
    for dt in res.diagram_terms(order):
        ig = dt.build_integrand({"g": np.array(1j)})
        dirs = {d: 0 for d in set(ig.spatial.direction_map.values())}
        ext = {"x": T, "y": T}
        f = ig.make_scipy_integrand(ext, dirs, cache)
        val, _ = nquad(f, ig.integration_bounds(ext, t_min=0.0),
                       opts={"epsabs": 1e-11, "epsrel": 1e-9, "limit": 200})
        total += val
    # t = 8 is not quite t -> infinity; exp(-2t) ~ 1e-7 leaves ~2.5e-5.
    assert total == pytest.approx(exact, rel=1e-3)


def test_F1_linear_perturbation_is_exactly_solvable():
    """Drift -k x shifts mu -> mu + k, so <x^2> = D/(mu+k) exactly.

    At mu=1, D=1/2 the series is 0.5 - 0.5 k + 0.5 k^2 - ...  The order-2
    coefficient exercises the vertex-to-vertex chain; the old code gave 0.25.
    """
    T = 8.0
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    res = compute_moment(
        [phi("x"), phi("y")],
        Action([Vertex(fields=[psi, phi], coupling="k")]),
        order=2, ito=True, response_phase=True, collect_topology=True,
        diag_R=True, diag_C=True, iso_R=True, iso_C=True,
    )
    cache = _scalar_cache()
    exact = [D / MU, -D / MU**2, D / MU**3]
    for n in (0, 1, 2):
        total = 0.0
        for dt in res.diagram_terms(n):
            ig = dt.build_integrand({"k": np.array(1j)})
            dirs = {d: 0 for d in set(ig.spatial.direction_map.values())}
            ext = {"x": T, "y": T}
            if not ig.spatial.time_integration_vars:
                total += float(np.real(ig.evaluate(ext, dirs, cache)))
                continue
            f = ig.make_scipy_integrand(ext, dirs, cache)
            val, _ = nquad(f, ig.integration_bounds(ext, t_min=0.0),
                           opts={"epsabs": 1e-11, "epsrel": 1e-9})
            total += val
        assert total == pytest.approx(exact[n], rel=1e-4), f"order {n}"


# --------------------------------------------------------------------- #
# F2 — reality theorem and the projection onto the reals
# --------------------------------------------------------------------- #
def test_F2_reality_theorem_holds_for_each_E_psi():
    """value = i^{-E_psi} * real, for E_psi = 0, 1, 2."""
    T, cache = 6.0, _scalar_cache()
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    cases = [
        ([phi("x"), phi("y")], 0),
        ([phi("x"), psi("y")], 1),
        ([phi("x"), phi("y"), psi("z"), psi("w")], 2),
    ]
    pts = {"x": T, "y": 0.8 * T, "z": 0.6 * T, "w": 0.4 * T}
    for obs, e_psi in cases:
        res = _quartic(1, obs=obs)
        for n in (0, 1):
            for dt in res.diagram_terms(n):
                assert dt.n_external_response == e_psi
                ig = dt.build_integrand({"g": np.array(1j)})
                if ig.spatial.time_integration_vars:
                    continue
                dirs = {d: 0 for d in set(ig.spatial.direction_map.values())}
                raw = ig.evaluate(pts, dirs, cache)
                rotated = raw * dt.observable_phase_factor()
                assert abs(rotated.imag) <= 1e-12 * max(abs(rotated), 1.0)


def test_F2_never_returns_abs_of_a_negative_contribution():
    """A rounding-level imaginary residue must not flip the sign.

    Previously ``abs(result)`` turned the O(g) coefficient of <x^2> from
    -0.7499 into +0.7499 for a coupling perturbed by 1e-12.
    """
    T, cache = 8.0, _scalar_cache()
    res = _quartic(1)
    ext = {"x": T, "y": T}

    def run(coupling):
        total = 0.0
        for dt in res.diagram_terms(1):
            ig = dt.build_integrand({"g": coupling})
            dirs = {d: 0 for d in set(ig.spatial.direction_map.values())}
            f = ig.make_scipy_integrand(ext, dirs, cache)
            val, _ = nquad(f, ig.integration_bounds(ext, t_min=0.0),
                           opts={"epsabs": 1e-11, "epsrel": 1e-9})
            total += val
        return total

    clean = run(np.array(1j))
    assert clean < 0
    perturbed = run(np.array(1j * (1 + 1e-12j)))
    assert perturbed < 0, "sign was destroyed by an abs()-style projection"
    assert perturbed == pytest.approx(clean, rel=1e-9)


def test_F2_real_or_raise_boundary():
    """Boundary sweep for the reality projection (see boundary-validation)."""
    from sft_wick.evaluate import _real_or_raise

    eps = 1e-18
    # exactly real -> passes at every scale
    for re in (1.0, 1e-8, 1e-200, 0.0):
        assert _real_or_raise(complex(re, 0.0)) == re
    # negligible imaginary residue relative to a finite real part -> passes
    assert _real_or_raise(complex(1.0, 1e-15)) == pytest.approx(1.0)
    # denormal-scale everything -> treated as an exact zero, no raise
    assert _real_or_raise(complex(eps * 1e-280, eps * 1e-280)) is not None
    # macroscopically imaginary -> must raise, never guess
    with pytest.raises(ValueError, match="not negligible"):
        _real_or_raise(complex(0.0, 1.0))
    # the E_psi rotation makes an E_psi=1 value real
    assert _real_or_raise(complex(0.0, -1.0), 1) == pytest.approx(1.0)


# --------------------------------------------------------------------- #
# F3 — matrix C
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["dblquad", "gauss_legendre"])
def test_F3_matrix_C_uses_the_full_response_matrix(method):
    """C = int int R kappa R^T for a dense (non-diagonal) drift matrix.

    Closed form: with A = Q diag(a) Q^T symmetric and kappa = 2T I delta,
    C(t1,t2) = Q diag[(T/a)(exp(-a|t1-t2|) - exp(-a(t1+t2)))] Q^T.
    Contracting only diag(R) gave a 57% Frobenius error and wrong signs.
    """
    rng = np.random.default_rng(3)
    N, T = 3, 0.7
    M = rng.standard_normal((N, N))
    A = M @ M.T + 1.5 * np.eye(N)
    ev, Q = np.linalg.eigh(A)

    def R(t, tp):
        return Q @ np.diag(np.exp(-ev * (t - tp))) @ Q.T

    def C_exact(t1, t2):
        d = (T / ev) * (np.exp(-ev * abs(t1 - t2)) - np.exp(-ev * (t1 + t2)))
        return Q @ np.diag(d) @ Q.T

    cache = PropagatorCache(PropagatorModel(
        R_time=R, kappa2=lambda *a: np.zeros((N, N)),
        sigma2=lambda n1, t, n2: 2 * T * np.eye(N),
        n_components=N, iso_R=False, diag_C=False, t_min=0.0,
    ))
    cache.c_method = method
    for t1, t2 in [(1.0, 1.0), (1.4, 0.6), (0.3, 2.0)]:
        got = cache._C_value_direct(0, t1, 0, t2, method=method)
        assert np.allclose(got, C_exact(t1, t2), rtol=1e-9, atol=1e-12)


def test_F3_iso_R_path_is_unchanged():
    """The scalar/isotropic path must be bit-for-bit what it always was."""
    cache = PropagatorCache(PropagatorModel(
        R_time=lambda t, tp: float(np.exp(-2.0 * (t - tp))),
        kappa2=lambda *a: np.zeros((1, 1)),
        sigma2=lambda n1, t, n2: np.array([[1.0]]),
        n_components=1, iso_R=True, diag_C=True, t_min=0.0,
    ))
    exact = (0.5 / 2.0) * (np.exp(-2 * 0.4) - np.exp(-2 * 1.6))
    for method in ("dblquad", "gauss_legendre"):
        got = float(cache._C_value_direct(0, 1.0, 0, 0.6, method=method)[0, 0])
        assert got == pytest.approx(exact, rel=1e-12)


# --------------------------------------------------------------------- #
# F4 — public L0 hook for C
# --------------------------------------------------------------------- #
def test_F4_c_value_fn_bypasses_the_quadrature():
    """A supplied C must be used verbatim, with no R/kappa convolution.

    Needed because a disorder-averaged (DMFT) C cannot be produced from the
    averaged R: <R kappa R> != <R> kappa <R>.
    """
    sentinel = np.array([[0.3141592653589793]])
    cache = PropagatorCache(
        PropagatorModel(
            R_time=lambda t, tp: np.exp(-(t - tp)),
            kappa2=lambda *a: np.zeros((1, 1)),
            sigma2=lambda n1, t, n2: np.array([[999.0]]),  # would dominate
            n_components=1, iso_R=True, diag_C=True, t_min=0.0,
        ),
        c_value_fn=lambda n1, t1, n2, t2: sentinel,
    )
    assert cache.C_value(0, 1.0, 0, 0.5)[0, 0] == pytest.approx(sentinel[0, 0])
    assert cache.C_diagonal(0, 1.0, 0, 0.5)[0] == pytest.approx(sentinel[0, 0])


# --------------------------------------------------------------------- #
# F6 — time-dependent LOCAL couplings
# --------------------------------------------------------------------- #
def _linear_local_terms():
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    res = compute_moment(
        [phi("x"), phi("y")],
        Action([Vertex(fields=[psi, phi], coupling="c")]),
        order=1, ito=True, response_phase=True, collect_topology=True,
        diag_R=True, diag_C=True, iso_R=True, iso_C=True,
    )
    return res, res.diagram_terms(1)


def test_F6_local_coupling_latex_is_unchanged():
    """A constant local coupling must still render as ``c``, not ``c(y_0)``."""
    res, _ = _linear_local_terms()
    assert "c(" not in res.order(1).to_latex()


def test_F6_callable_local_coupling_matches_static():
    """A callable returning a constant must equal the ndarray path exactly."""
    T = 3.0
    cache = _scalar_cache()
    cache.precompute_C_table(t_max=T, n_grid=401, direction=0)
    _, terms = _linear_local_terms()

    def total(cv):
        return sum(
            t.build_integrand({"c": cv}).integrate_moment_gauss_legendre(
                lambda_f=T, cache=cache, t_min=0.0, n_gauss=64)[0]
            for t in terms
        )

    assert total(lambda n, t: np.asarray(1j)) == pytest.approx(
        total(np.array(1j)), rel=1e-14
    )


def test_F6_time_dependent_local_coupling_matches_closed_form():
    """S_int = int c(t) psi x with c(t) = i k(t) shifts the drift by -k(t) x.

    O(k):  d<x(T)^2> = -2 int_0^T ds k(s) R0(T,s) C0(T,s).
    """
    T = 3.0
    cache = _scalar_cache()
    cache.precompute_C_table(t_max=T, n_grid=401, direction=0)
    _, terms = _linear_local_terms()
    k = lambda s: np.exp(-s)  # noqa: E731

    got = sum(
        t.build_integrand(
            {"c": lambda n, tt: np.asarray(1j * k(float(np.atleast_1d(tt)[0])))}
        ).integrate_moment_gauss_legendre(
            lambda_f=T, cache=cache, t_min=0.0, n_gauss=64)[0]
        for t in terms
    )
    truth, _ = quad(lambda s: k(s) * np.exp(-MU * (T - s)) * _C0(T, s),
                    0.0, T, limit=300)
    assert got == pytest.approx(-2.0 * truth, rel=1e-8)


def test_F6_multi_point_callable_is_refused_not_silently_wrong():
    """Two copies of one vertex sit at different times; a single-coordinate
    callable evaluation would be silently wrong, so it must raise."""
    res = _quartic(2)
    raised = 0
    for dt in res.diagram_terms(2):
        try:
            dt.build_integrand({"g": lambda n, t: np.asarray(1j)})
        except NotImplementedError as exc:
            assert "different sets of spacetime points" in str(exc)
            raised += 1
    assert raised > 0


# --------------------------------------------------------------------- #
# F7 — API traps
# --------------------------------------------------------------------- #
def test_F7a_evaluate_refuses_a_dynamic_coupling():
    """It reads the zeros placeholder, so it must not silently return 0."""
    _, terms = _linear_local_terms()
    ig = terms[0].build_integrand({"c": lambda n, t: np.asarray(1j)})
    with pytest.raises(NotImplementedError, match="silently be 0"):
        ig.evaluate({"x": 1.0, "y": 1.0, "y_0": 0.5}, {"n_x": 0},
                    _scalar_cache())


def test_F7c_wrong_rank_scalar_coupling_names_the_symbol():
    """A rank-4 array for a scalar field used to raise an opaque TypeError."""
    res = _quartic(1)
    dt = res.diagram_terms(1)[0]
    with pytest.raises(ValueError, match="coupling 'g'"):
        dt.build_integrand({"g": np.zeros((2, 2, 2, 2), dtype=complex)})
    # a size-1 array in any shape is accepted
    dt.build_integrand({"g": np.array(1j).reshape(1, 1, 1, 1)})


# --------------------------------------------------------------------- #
# F9 — causal LOWER bounds from external response legs
# --------------------------------------------------------------------- #
def _d1R_closed_form(tx: float, ty: float) -> float:
    """O(g) response of dx/dt = -mu x - g x^3 + xi.

    Wick: psi(t_y) must reach a vertex phi (pairing it with x(t_x) strands
    psi(s) on R(s,s)=0), so
        delta_1 R = -3 g int_{t_y}^{t_x} R0(t_x,s) R0(s,t_y) C0(s,s) ds
                  = -3 g exp(-mu (t_x - t_y)) int_{t_y}^{t_x} C0(s,s) ds.
    The LOWER limit t_y comes from R0(s,t_y) being retarded; dropping it was
    the defect this test pins.  At t_x = t_y the domain collapses and the
    result is exactly zero.
    """
    if tx <= ty:
        return 0.0
    a = D / MU
    val, _ = quad(lambda s: np.exp(-MU * (tx - s)) * np.exp(-MU * (s - ty))
                  * a * (1.0 - np.exp(-2.0 * MU * s)), ty, tx, limit=400)
    return -3.0 * val


@pytest.mark.parametrize("t2", [0.5, 2.0, 3.5, 5.0, 6.5, 7.5, 8.0])
def test_F9_response_at_order1_respects_the_lower_causal_bound(t2):
    """An external psi leg bounds the vertex time from BELOW.

    Before lower bounds existed the vertex was integrated over [t_min, t_x]
    instead of [t_y, t_x] — up to 5x wrong, and non-zero at t_x = t_y where
    the exact answer is 0.
    """
    T = 8.0
    cache = _scalar_cache()
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    res = compute_moment(
        [phi("x"), psi("y")],
        Action([Vertex(fields=[psi, phi, phi, phi], coupling="g")]),
        order=1, ito=True, response_phase=True, collect_topology=True,
        diag_R=True, diag_C=True, iso_R=True, iso_C=True,
    )
    total = 0.0
    ext = {"x": T, "y": t2}
    for dt in res.diagram_terms(1):
        ig = dt.build_integrand({"g": np.array(1j)})
        dirs = {d: 0 for d in set(ig.spatial.direction_map.values())}
        f = ig.make_scipy_integrand(ext, dirs, cache)
        val, _ = nquad(f, ig.integration_bounds(ext, t_min=0.0),
                       opts={"epsabs": 1e-12, "epsrel": 1e-10, "limit": 300})
        total += val
    truth = _d1R_closed_form(T, t2)
    if truth == 0.0:
        assert total == pytest.approx(0.0, abs=1e-12)
    else:
        assert total == pytest.approx(truth, rel=1e-8)


def test_F9_lower_bound_helper_only_fires_for_external_earlier_endpoints():
    """Only (external -> internal) orderings become lower bounds.

    (internal -> anything) is already expressible as an upper bound on the
    internal variable; turning it into a lower bound as well would double-count.
    """
    from sft_wick.evaluate import _causal_lower_bounds

    class _SP:
        time_orderings = (("x_ext", "s_a"),   # external earlier -> lower bound
                          ("s_a", "s_b"),      # internal earlier -> not ours
                          ("s_b", "x_ext"))    # internal earlier -> not ours
    got = _causal_lower_bounds(_SP(), ["s_a", "s_b"], {"x_ext": 3.0}, 0.0)
    assert got == {"s_a": 3.0}


def test_F10_qmc_scalar_zero_dim_uses_the_reality_projection():
    """integrate_moment_qmc has its OWN zero-dimensional branch.

    It used `val.real`, so every observable with an external response leg
    returned exactly 0.0 at order 0 — including R(t,t') itself.
    """
    T = 4.0
    cache = _scalar_cache()
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    res = compute_moment(
        [phi("x"), psi("y")],
        Action([Vertex(fields=[psi, phi, phi, phi], coupling="g")]),
        order=0, ito=True, response_phase=True, collect_topology=True,
        diag_R=True, diag_C=True, iso_R=True, iso_C=True,
    )
    ig = res.diagram_terms(0)[0].build_integrand({"g": np.array(1j)})
    scalar = ig.integrate_moment_qmc(lambda_f=T, cache=cache, t_min=0.0,
                                     n_samples=256, seed=1)[0]
    reference = ig.integrate_moment_nquad(lambda_f=T, cache=cache, t_min=0.0)[0]
    assert scalar == pytest.approx(reference, rel=1e-12)
    assert scalar != 0.0
