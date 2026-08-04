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
  F9  causal LOWER bounds from external response legs were dropped by every
      bound-builder, and were not transitively closed along internal edges.
  F10 the zero-dimensional QMC branch projected with `.real`.
  F11 R propagators between two fixed externals had no Theta.
  F12 integrate_two_point_qmc was a sixth bound-builder and was missed.
  F13 a SWEPT external's lower bound is variable-valued; dropping it costs
      gauss_legendre its spectral convergence (22% at the default n_gauss=8).
  F14 integrate_two_point_qmc dropped the spatial factor at ni == 0, so the
      order-0 correlator ignored separation entirely.
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


def _scalar_model() -> PropagatorModel:
    return PropagatorModel(
        R_time=lambda t, tp: np.exp(-MU * (t - tp)),
        kappa2=lambda n1, t1, n2, t2: np.zeros((1, 1)),
        sigma2=lambda n1, t, n2: np.array([[2.0 * D]]),
        n_components=1, iso_R=True, diag_C=True, t_min=0.0,
    )


def _scalar_cache(**kw) -> PropagatorCache:
    return PropagatorCache(
        _scalar_model(),
        c_value_fn=lambda n1, t1, n2, t2: np.array([[_C0(t1, t2)]]), **kw
    )


class _ScalarBatchCache(PropagatorCache):
    """``_scalar_cache`` plus a closed-form **batch** C.

    ``gauss_legendre`` and ``qmc_vectorized`` reject a cache without batch C,
    and would otherwise need ``precompute_C_table`` — whose derivative kink on
    the ``t1 == t2`` diagonal costs ~0.05% on its own, larger than the effect
    some of these tests resolve.  Overriding the scalar accessors too keeps
    every backend on the same machine-precision propagator values.
    """

    def __init__(self):
        super().__init__(
            _scalar_model(),
            c_value_fn=lambda n1, t1, n2, t2: np.array([[_C0(t1, t2)]]),
        )
        self._c_splines = True  # sentinel: "batch C is available"

    def C_value(self, n1, t1, n2, t2):
        return np.array([[_C0(t1, t2)]])

    def C_diagonal(self, n, t1, n_prime=None, t2=None):
        return np.array([_C0(t1, t1 if t2 is None else t2)])

    def C_diagonal_batch(self, t1, t2):
        return np.asarray(_C0(t1, t2), dtype=float)[:, None]


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
    # The E_psi rotation, over the full period.  This is the ONLY place the
    # projection is directly observable: end to end, every zero-dimensional
    # diagram with E_psi >= 1 is identically 0 while all externals share a
    # time (Theta kills the R joining them), so `_real_or_raise` and a bare
    # `.real` cannot be told apart there.  Mutation-verified: reverting a
    # call site to `.real` does NOT fail the end-to-end tests, only this one.
    for e_psi, raw, want in [
        (0, complex(2.5, 0.0), 2.5),      # i^0 = 1
        (1, complex(0.0, -1.5), 1.5),     # i^1 rotates -1.5i -> 1.5
        (2, complex(-3.0, 0.0), 3.0),     # i^2 = -1
        (3, complex(0.0, 4.0), 4.0),      # i^3 = -i
    ]:
        assert _real_or_raise(raw, e_psi) == pytest.approx(want), (
            f"E_psi={e_psi}: {raw} should rotate onto {want}"
        )
        # ... and the WRONG phase must raise rather than silently truncate.
        wrong = (e_psi + 1) % 4
        if abs(raw) > 0:
            with pytest.raises(ValueError, match="not negligible"):
                _real_or_raise(raw, wrong)


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


def test_F9_lower_bounds_seed_externally_then_propagate():
    """Lower bounds are SEEDED by (external -> internal) orderings only, then
    propagated transitively along internal edges.

    An (internal -> internal) ordering never *seeds* a lower bound — it is
    already carried as an upper bound on the earlier variable — but it does
    *propagate* one: ``x_ext -> s_a -> s_b`` implies ``t_s_b >= t_x_ext``.
    An internal variable with no external ancestor keeps ``t_min``.
    """
    from sft_wick.evaluate import _causal_lower_bounds

    class _Chain:
        time_orderings = (("x_ext", "s_a"),   # seeds s_a
                          ("s_a", "s_b"),      # propagates to s_b
                          ("s_b", "x_ext"))
    assert _causal_lower_bounds(
        _Chain(), ["s_a", "s_b"], {"x_ext": 3.0}, 0.0
    ) == {"s_a": 3.0, "s_b": 3.0}

    class _NoExternalAncestor:
        time_orderings = (("s_a", "s_b"), ("s_b", "x_ext"))
    assert _causal_lower_bounds(
        _NoExternalAncestor(), ["s_a", "s_b"], {"x_ext": 3.0}, 0.0
    ) == {}


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
    # NOTE: both sides of the next assert are 0 by construction -- all
    # externals are pinned at lambda_f, so Theta kills R(T,T).  It is a
    # cross-backend consistency check, NOT coverage of the projection: it
    # survives reverting the call site to `.real`.  The projection itself is
    # pinned directly in test_F2_real_or_raise_boundary, and the causal,
    # non-zero configuration below is what carries this test.
    scalar = ig.integrate_moment_qmc(lambda_f=T, cache=cache, t_min=0.0,
                                     n_samples=256, seed=1)[0]
    reference = ig.integrate_moment_nquad(lambda_f=T, cache=cache, t_min=0.0)[0]
    assert scalar == pytest.approx(reference, abs=1e-15)

    dirs = {d: 0 for d in set(ig.spatial.direction_map.values())}
    causal = float(np.real(ig.evaluate({"x": T, "y": 0.6 * T}, dirs, cache)
                           * res.diagram_terms(0)[0].observable_phase_factor()))
    assert causal == pytest.approx(np.exp(-MU * (T - 0.6 * T)), rel=1e-12)
    assert causal != 0.0


@pytest.mark.parametrize("tx,ty,expected", [
    (8.0, 6.0, np.exp(-2.0)),   # causal
    (8.0, 8.0, 0.0),            # equal times -> 0 under Ito
    (6.0, 8.0, 0.0),            # acausal -> 0, not exp(+mu(ty-tx))
    (3.0, 5.0, 0.0),
])
def test_F11_response_propagator_is_retarded(tx, ty, expected):
    """R must vanish for t_x <= t_y even with no integration domain.

    ``PropagatorCache.R_time`` deliberately does not apply Theta ("the
    integration domain handles causality"), but an R joining two FIXED
    external points has no domain — an order-0 response diagram is exactly
    that.  Without Theta it returned the unbounded acausal
    ``exp(+mu (t_y - t_x))``, and 1 instead of 0 at equal times.
    """
    cache = _scalar_cache()
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    res = compute_moment(
        [phi("x"), psi("y")],
        Action([Vertex(fields=[psi, phi, phi, phi], coupling="g")]),
        order=0, ito=True, response_phase=True, collect_topology=True,
        diag_R=True, diag_C=True, iso_R=True, iso_C=True,
    )
    dt = res.diagram_terms(0)[0]
    ig = dt.build_integrand({"g": np.array(1j)})
    dirs = {d: 0 for d in set(ig.spatial.direction_map.values())}
    got = float(np.real(ig.evaluate({"x": tx, "y": ty}, dirs, cache)
                        * dt.observable_phase_factor()))
    assert got == pytest.approx(expected, abs=1e-14)


def test_F9_lower_bounds_are_transitively_closed():
    """A chain ext -> v1 -> v2 induces t_v2 >= t_ext, not just t_v1 >= t_ext.

    Bounding only v1 lets v2 range below t_ext, at which point v1's interval
    [t_ext, t_v2] inverts.  Measured 13.5% low (44 sigma) on order-2 <phi psi>
    before the closure was added.
    """
    from sft_wick.evaluate import _causal_lower_bounds

    class _SP:
        time_orderings = (("y", "y_0"), ("y_0", "y_1"), ("y_1", "x"))

    got = _causal_lower_bounds(_SP(), ["y_0", "y_1"], {"y": 1.5, "x": 4.0}, 0.0)
    assert got == {"y_0": 1.5, "y_1": 1.5}


@pytest.mark.parametrize("order", [1, 2])
def test_F9_acausal_external_times_give_exactly_zero(order):
    """R is retarded: <phi(t_x) psi(t_y)> must vanish for t_x < t_y.

    The domain is empty, so the integral is 0 -- never a negative volume from
    scipy integrating an inverted interval backwards.
    """
    cache = _scalar_cache()
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    res = compute_moment(
        [phi("x"), psi("y")],
        Action([Vertex(fields=[psi, phi, phi, phi], coupling="g")]),
        order=order, ito=True, response_phase=True, collect_topology=True,
        diag_R=True, diag_C=True, iso_R=True, iso_C=True,
    )
    ext = {"x": 1.0, "y": 3.0}          # acausal: t_x < t_y
    total = 0.0
    for dt in res.diagram_terms(order):
        ig = dt.build_integrand({"g": np.array(1j)})
        dirs = {d: 0 for d in set(ig.spatial.direction_map.values())}
        bounds = ig.integration_bounds(ext, t_min=0.0)
        for b in bounds:
            if not callable(b):
                assert b[1] >= b[0], f"inverted constant range {b}"
        f = ig.make_scipy_integrand(ext, dirs, cache)
        val, _ = nquad(f, bounds, opts={"epsabs": 1e-12, "epsrel": 1e-10})
        total += val
    assert total == pytest.approx(0.0, abs=1e-12)


def test_F12_two_point_qmc_agrees_with_the_other_backends():
    """integrate_two_point_qmc is a 6th bound-builder and was missed.

    It had no lower bounds at all and its zero-dimensional branch used
    `val.real` (a 15th unprojected site), so for a response observable it
    disagreed with all four other backends by 100%.

    NOTE: since Theta landed, BOTH sides of the assert are 0.0 at both
    orders and it is structurally impossible for them to be anything else
    (every external sits at T, so the R joining them is killed).  This is
    a cross-backend consistency check, not coverage: it cannot detect any
    change to the ni == 0 branch.  That branch is covered by F14 (spatial
    factor) and F17 (delegation), and the projection itself by
    test_F2_real_or_raise_boundary.
    """
    from sft_wick.evaluate import integrate_two_point_qmc

    T = 3.0
    cache = _scalar_cache()
    cache.precompute_C_table(t_max=4.0, n_grid=201, direction=0)
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    for order in (0, 1):
        res = compute_moment(
            [phi("x"), psi("y")],
            Action([Vertex(fields=[psi, phi, phi, phi], coupling="g")]),
            order=order, ito=True, response_phase=True, collect_topology=True,
            diag_R=True, diag_C=True, iso_R=True, iso_C=True,
        )
        igs = [dt.build_integrand({"g": np.array(1j)})
               for dt in res.diagram_terms(order)]
        tp, _ = integrate_two_point_qmc(
            igs, T, {"x": 0.0, "y": 0.0}, cache, t_min=0.0,
            n_samples=2 ** 14, seed=1,
        )
        ref = sum(ig.integrate_moment_nquad(lambda_f=T, cache=cache, t_min=0.0)[0]
                  for ig in igs)
        assert tp == pytest.approx(ref, abs=1e-8), f"order {order}"


# --------------------------------------------------------------------- #
# F13: a SWEPT external that lower-bounds a vertex
# --------------------------------------------------------------------- #
#
# ``integrate_over=['y']`` sweeps the psi leg, so the ordering (y -> vertex)
# is a lower bound whose value changes sample to sample.  Every backend
# dropped it.  Two corrections to the original diagnosis, both measured:
#
#   * It is no longer a *correctness* defect.  Since Theta is applied
#     pointwise at every R site, the integrand already vanishes wherever
#     s < y, so the too-large domain only wastes evaluations.  Measured at
#     HEAD~ with the bound dropped: nquad 0.022%, qmc 0.012%,
#     qmc_vectorized 0.144% (2^14) -> 0.002% (2^20).  The 205%/749% figures
#     recorded earlier were measured before Theta landed.
#   * It IS a quadrature defect, and only for a fixed-node smooth rule.
#     Theta puts a jump inside the domain, which costs gauss_legendre its
#     spectral convergence: 22.454% at the library default n_gauss=8, then
#     6.035 / 1.544 / 0.766% at n=32/128/256 — error ~ 1/n.
#
# The bound is expressible in every backend (integrated externals are drawn
# before all internals; nquad places them outside every internal variable),
# so it is now applied rather than refused.  With it, n_gauss=8 gives 0.006%.

_F13_LAMBDA_F = 3.0


def _f13_reference() -> float:
    """int_0^lambda_f dy  delta_1 R(lambda_f, y), from the closed form."""
    val, _ = quad(lambda y: _d1R_closed_form(_F13_LAMBDA_F, y),
                  0.0, _F13_LAMBDA_F, limit=200)
    return val


@pytest.mark.parametrize("method,kw,tol,batch", [
    # `batch` picks the cache: _ScalarBatchCache advertises batch C, which
    # makes the `qmc` DISPATCHER resolve to qmc_vectorized.  The scalar
    # cells therefore use the plain cache, so that "qmc" and
    # "qmc_vectorized" are genuinely different code paths and not the same
    # call counted twice.
    ("nquad", {}, 1e-6, False),
    ("qmc", {"n_samples": 2 ** 14, "seed": 7}, 1e-5, False),
    ("qmc_scalar", {"n_samples": 2 ** 14, "seed": 7}, 1e-5, False),
    ("qmc_vectorized", {"n_samples": 2 ** 14, "seed": 7}, 1e-5, True),
    ("gauss_legendre", {"n_gauss": 8}, 2e-3, True),
    ("gauss_legendre", {"n_gauss": 32}, 2e-4, True),
])
def test_F13_swept_external_lower_bound_is_applied(method, kw, tol, batch):
    """Every backend must carry the variable lower bound, not drop it.

    Tolerances bound what each backend achieves with the bound *and* stay
    below what it achieves without -- the QMC cells are the tight ones
    (2.8e-7 achieved vs ~1e-3 under mutation); nquad and gauss_legendre
    clear theirs by a wide margin, so for those the tolerance is bounded
    from above by the mutation, not from below by the quadrature.
    Verified by mutation: blanking the variable sources fails all six.
    """
    from sft_wick.evaluate import integrate_moment

    cache = _ScalarBatchCache() if batch else _scalar_cache()
    res = _quartic(1, obs=[Field("phi", "physical")("x"),
                           Field("psi", "response")("y")])
    total = 0.0
    for dt in res.diagram_terms(1):
        v, _ = integrate_moment(
            dt.build_integrand({"g": np.array(1j)}), _F13_LAMBDA_F, cache,
            method=method, t_min=0.0, integrate_over=["y"], **kw
        )
        total += v
    ref = _f13_reference()
    assert total == pytest.approx(ref, rel=tol), (
        f"{method} {kw}: {total:.8f} vs closed form {ref:.8f} "
        f"(rel {abs(total - ref) / abs(ref):.2e})"
    )


def test_F13_gauss_legendre_recovers_fast_convergence():
    """The bound removes the interior Theta jump, so GL stops being O(1/n).

    Dropping it left gauss_legendre at 22% for the default n_gauss=8 and
    first-order convergence thereafter.  Pinning the *default* is what
    matters: a user who never tunes n_gauss must still get a usable number.
    """
    from sft_wick.evaluate import integrate_moment

    cache = _ScalarBatchCache()
    res = _quartic(1, obs=[Field("phi", "physical")("x"),
                           Field("psi", "response")("y")])
    igs = [dt.build_integrand({"g": np.array(1j)})
           for dt in res.diagram_terms(1)]
    ref = _f13_reference()

    errs = {}
    for n_gauss in (8, 32):
        total = sum(
            integrate_moment(ig, _F13_LAMBDA_F, cache,
                             method="gauss_legendre", t_min=0.0,
                             integrate_over=["y"], n_gauss=n_gauss)[0]
            for ig in igs
        )
        errs[n_gauss] = abs(total - ref) / abs(ref)

    assert errs[8] < 1e-3, (
        f"n_gauss=8 rel err {errs[8]:.2e} — was 2.2e-1 with the bound dropped"
    )
    # Secondary, and deliberately weak: it also holds under the mutation
    # (0.060 < 0.225), so `errs[8] < 1e-3` above is the actual detector.
    assert errs[32] < errs[8], f"not converging: {errs}"


def test_F13_variable_lower_bound_sources_are_transitively_closed():
    """A chain ``swept-external -> v1 -> v2`` must bound v2 as well.

    The constant part of the closure was already covered by F9; the variable
    part propagates along the same internal edges and for the same reason.
    """
    from sft_wick.evaluate import _causal_lower_bound_sources

    class _Sp:
        time_orderings = (("y", "v1"), ("v1", "v2"), ("v2", "x"))

    const, srcs = _causal_lower_bound_sources(
        _Sp(), ["v1", "v2"], {"x": 5.0}, 0.0, swept=("y",)
    )
    assert srcs.get("v1") == ("y",)
    assert srcs.get("v2") == ("y",), (
        f"y must propagate to v2 through v1; got {srcs}"
    )
    assert "v1" not in const and "v2" not in const


# --------------------------------------------------------------------- #
# F14: integrate_two_point_qmc dropped the spatial factor at ni == 0
# --------------------------------------------------------------------- #
#
# The zero-integration-variable branch delegated to ``DiagramIntegrand.
# evaluate``, whose C lookup short-circuits to the *direction-agnostic*
# legacy spline table.  So the order-0 two-point function came back at its
# coincident-point value for **every** separation, while the ni >= 1 branch
# in the same function applied ``c_spatial_factors`` correctly.  One call,
# two spatial conventions.
#
# Ground truth (independent of sft-wick): for a separable kernel
#
#     kappa_ab(n1, t1; n2, t2) = delta_ab * exp(-|n1 - n2| / sigma_x)
#
# and R(t, t') = exp(-mu (t - t')), the order-0 correlator is exactly
#
#     C_aa(x, T; y, T) = exp(-|x-y| / sigma_x) * ((1 - exp(-mu T)) / mu)^2
#
# and a diagram with n_cross C-propagators spanning the two external points
# carries exp(-n_cross |x-y| / sigma_x).

_F14_SIGMA_X = 1.0
_F14_T = 4.0
_F14_NCOMP = 2


def _f14_setup(order: int):
    """Return (integrands, cache) for the separable two-point model."""
    def kappa2(n1, t1, n2, t2):
        a1 = np.atleast_1d(np.asarray(n1, dtype=float))
        a2 = np.atleast_1d(np.asarray(n2, dtype=float))
        r = float(np.abs(a1[0] - a2[0]))
        return np.eye(_F14_NCOMP) * np.exp(-r / _F14_SIGMA_X)

    phi = Field("phi", "physical", n_components=_F14_NCOMP)
    psi = Field("psi", "response", n_components=_F14_NCOMP)
    res = compute_moment(
        [phi("a", "x"), phi("b", "y")],
        Action([Vertex(fields=[psi, phi, phi], coupling="F")]),
        order=order, ito=True, response_phase=True, collect_topology=True,
        diag_R=True, diag_C=True, iso_R=True,
    )
    F = np.zeros((_F14_NCOMP,) * 3)
    F[0, 0, 0] = F[1, 1, 1] = -0.5
    model = PropagatorModel(
        R_time=lambda t, tp: np.exp(-MU * (t - tp)), kappa2=kappa2,
        n_components=_F14_NCOMP, iso_R=True, diag_C=True, t_min=0.0,
    )
    cache = PropagatorCache(model)
    cache.precompute_C_table(t_max=_F14_T, n_grid=60)
    igs = [dt.build_integrand({"F": -1j * F}, fixed_indices={"a": 0, "b": 0})
           for dt in res.diagram_terms(order)]
    return igs, cache


def _f14_n_cross(ig, positions):
    """C-propagators whose two ends sit at different external positions."""
    sp = ig.spatial
    pos_of = {}
    for pt in sp.external_points:
        dvar = sp.direction_map.get(pt)
        if dvar is not None:
            pos_of[dvar] = positions.get(pt, 0.0)
    return sum(
        1 for sl, sr, _il, _ir in sp.c_propagators
        if abs(pos_of.get(sp.direction_map[sl], 0.0)
               - pos_of.get(sp.direction_map[sr], 0.0)) > 1e-15
    )


@pytest.mark.parametrize("r", [0.0, 0.5, 1.0, 2.0])
def test_F14_two_point_qmc_order0_carries_the_spatial_factor(r):
    """ni == 0 must fall off with separation, not return the r=0 value."""
    from sft_wick.evaluate import integrate_two_point_qmc

    igs, cache = _f14_setup(order=0)
    assert all(len(ig.spatial.time_integration_vars) == 0 for ig in igs)

    val, _ = integrate_two_point_qmc(
        igs, _F14_T, {"x": 0.0, "y": r}, cache, n_samples=2 ** 12, seed=0,
    )
    exact = (np.exp(-r / _F14_SIGMA_X)
             * ((1.0 - np.exp(-MU * _F14_T)) / MU) ** 2)
    assert val == pytest.approx(exact, rel=1e-6), (
        f"r={r}: got {val:.10f}, closed form {exact:.10f}"
    )


@pytest.mark.parametrize("order", [0, 2])
def test_F14_cross_propagator_count_sets_the_separation_scaling(order):
    """Both branches must obey exp(-n_cross r / sigma_x), exactly.

    Grouping by ``n_cross`` is what makes this a sharp test: the *sum* over
    an order's diagrams mixes several exponentials, so an ungrouped ratio
    check is blunt enough to hide the ni == 0 defect behind quadrature noise.
    """
    from sft_wick.evaluate import integrate_two_point_qmc

    igs, cache = _f14_setup(order=order)
    groups: dict[int, list] = {}
    for ig in igs:
        groups.setdefault(_f14_n_cross(ig, {"x": 0.0, "y": 1.0}), []).append(ig)
    assert groups, f"order {order} produced no diagrams"

    for n_cross, gigs in sorted(groups.items()):
        base = None
        for r in (0.0, 0.5, 1.0, 2.0):
            val, _ = integrate_two_point_qmc(
                gigs, _F14_T, {"x": 0.0, "y": r}, cache,
                n_samples=2 ** 12, seed=0,
            )
            if base is None:
                base = val
                assert abs(base) > 1e-12, f"n_cross={n_cross} vanishes at r=0"
                continue
            want = np.exp(-n_cross * r / _F14_SIGMA_X)
            assert val / base == pytest.approx(want, rel=1e-12), (
                f"order {order}, n_cross={n_cross}, r={r}: "
                f"ratio {val / base:.12f} != exp(-{n_cross} r/sx) = {want:.12f}"
            )


# --------------------------------------------------------------------- #
# F15: ito=False is a SYMBOLIC switch, not a numerical one
# --------------------------------------------------------------------- #
#
# It was flagged as "a documented public flag silently inert" after the
# Theta commit collapsed its numbers onto the ito=True ones, with the
# proposed fix "use Theta(0)=1/2 when ito=False".  That prescription is
# WRONG, and this test exists to stop anyone acting on it.
#
# The MSRJD Theta(0) ambiguity is not independent of the functional
# Jacobian.  For dx/dt = F(x) + xi the Stratonovich discretisation carries
# -1/2 int ds dF/dphi, which is exactly what cancels the Theta(0)=1/2
# equal-point R terms; Ito sets both to zero.  sft-wick's ito=False keeps
# the equal-point R terms symbolically but never emits the Jacobian, so
# Theta(0)=0 is the only self-consistent numerical choice available -- and
# it happens to give the right answer, since for ADDITIVE noise the Ito
# and Stratonovich results coincide.
#
# Measured for the linear vertex k psi phi (exact: <x^2> = D/(mu+k), whose
# order-1 coefficient is -D/mu^2 = -0.5, independent of T):
#
#     T      ito=True        terms   ito=False       terms
#     4.0   -0.498490418       2    -0.498490418       3
#     8.0   -0.499999043       2    -0.499999043       3
#    16.0   -0.500000000       2    -0.500000000       3
#
# The extra term Theta(0)=1/2 would have contributed is -C0(T,T)*T/2 =
# -1.00 / -2.00 / -4.00 at those T -- i.e. 200% / 400% / 800% of the exact
# answer, growing without bound.


@pytest.mark.parametrize("T", [4.0, 8.0, 16.0])
def test_F15_ito_false_changes_the_expression_not_the_number(T):
    """ito=False must add diagram terms yet leave the value untouched."""
    phi = Field("phi", "physical")
    psi = Field("psi", "response")

    def _order1(ito):
        res = compute_moment(
            [phi("x"), phi("y")],
            Action([Vertex(fields=[psi, phi], coupling="k")]),
            order=1, ito=ito, response_phase=True, collect_topology=True,
            diag_R=True, diag_C=True, iso_R=True, iso_C=True,
        )
        cache, ext, total = _scalar_cache(), {"x": T, "y": T}, 0.0
        terms = res.diagram_terms(1)
        for dt in terms:
            ig = dt.build_integrand({"k": np.array(1j)})
            bounds = ig.integration_bounds(ext, t_min=0.0)
            if not bounds:
                continue
            dirs = {d: 0 for d in set(ig.spatial.direction_map.values())}
            val, _ = nquad(ig.make_scipy_integrand(ext, dirs, cache), bounds,
                           opts={"epsabs": 1e-11, "epsrel": 1e-9})
            total += val
        return total, len(terms)

    v_ito, n_ito = _order1(True)
    v_str, n_str = _order1(False)

    # (a) the switch really does change the symbolic expansion ...
    assert n_str > n_ito, (
        f"ito=False produced {n_str} terms, ito=True {n_ito} — the extra "
        f"equal-point R term is missing, so this test proves nothing"
    )
    # (b) ... but not the number ...
    assert v_str == pytest.approx(v_ito, rel=1e-12)
    # (c) ... and that number is the stationary one.  The tolerance
    # TIGHTENS with T, because the finite-time correction is
    # O(T exp(-2 mu T)) while the error a Theta(0)=1/2 "fix" would inject
    # GROWS with T.  A single loose tolerance would admit the latter --
    # which is the whole failure mode this test guards.
    stationary_tol = {4.0: 1e-2, 8.0: 1e-4, 16.0: 1e-6}[T]
    assert v_str == pytest.approx(-D / MU ** 2, rel=stationary_tol)


def test_F15_theta_half_would_grow_without_bound():
    """Pin WHY Theta(0)=1/2 is not the fix: the spurious term scales with T.

    Guards the reasoning, not just the outcome — if someone later adds the
    Jacobian counter-term and legitimately enables Theta(0)=1/2, this test
    documents the size of what must cancel.
    """
    spurious = {T: -_C0(T, T) * T / 2 for T in (4.0, 8.0, 16.0)}
    exact = -D / MU ** 2
    # Grows linearly in T, so it cannot be part of a T-independent answer.
    assert spurious[16.0] == pytest.approx(2 * spurious[8.0], rel=1e-3)
    assert abs(spurious[16.0] / exact) > 7.0


def test_F16_theta_sites_never_call_R_time_on_acausal_pairs():
    """All three Θ sites must be semantically equivalent, not just numerically.

    ``R_product`` and ``_evaluate_r_product_general`` short-circuit *before*
    calling the model's ``R_time``; ``R_time_batch`` used to evaluate every
    pair and mask afterwards.  A model whose ``R_time`` raises or overflows
    on acausal input therefore behaved differently depending on which
    backend was chosen — same number through nquad, exception through
    gauss_legendre.
    """
    calls: list[tuple[float, float]] = []

    def picky_R(t, tp):
        calls.append((float(t), float(tp)))
        if t <= tp:  # a model author entitled to assume retardation
            raise AssertionError(f"R_time called acausally: {t} <= {tp}")
        return np.exp(-MU * (t - tp))

    model = PropagatorModel(
        R_time=picky_R, kappa2=lambda n1, t1, n2, t2: np.zeros((1, 1)),
        sigma2=lambda n1, t, n2: np.array([[2.0 * D]]),
        n_components=1, iso_R=True, diag_C=True, t_min=0.0,
    )
    cache = PropagatorCache(model)

    t1 = np.array([3.0, 1.0, 2.0, 2.0])
    t2 = np.array([1.0, 3.0, 2.0, 0.5])   # causal, acausal, equal, causal
    out = cache.R_time_batch(t1, t2)

    assert out == pytest.approx(
        [np.exp(-2.0), 0.0, 0.0, np.exp(-1.5)], rel=1e-12, abs=1e-15
    )
    assert all(a > b for a, b in calls), (
        f"R_time was called on a non-retarded pair: {calls}"
    )
    # R_product agrees, and also never calls acausally.
    assert cache.R_product((("l", "r"),), {"l": 1.0, "r": 3.0}) == 0.0
    assert cache.R_product((("l", "r"),), {"l": 2.0, "r": 2.0}) == 0.0
    assert all(a > b for a, b in calls)


def test_F16_R_time_batch_handles_a_fully_acausal_batch():
    """No causal pair at all must not blow up on an empty vectorize call."""
    cache = _scalar_cache()
    out = cache.R_time_batch(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert out.shape == (2,)
    assert np.all(out == 0.0)


# --------------------------------------------------------------------- #
# F17: the ni == 0 branch must delegate whenever `evaluate` sees positions
# --------------------------------------------------------------------- #
#
# The first attempt at F14 routed *every* ni == 0 diagram through the
# vectorised legacy-table-times-kappa2-ratio path.  That was a regression:
# `C_value` is position-blind ONLY on the legacy time-only spline table.
# With a spatial table it takes the exact `C_at_batch` fast path, and with
# no table at all it falls through to dblquad / c_value_fn -- both exact.
# Adversarial review measured 4.9% -> 34.3% degradation (r = 0.5 -> 4) for a
# kernel whose correlation length grows with time, plus a hard RuntimeError
# for a table-less cache.  These tests pin all three configurations.

_F17_T = 4.0
_F17_N = 2


def _f17_igs(kappa2, diag_C=True, n_comp=_F17_N, fixed=None):
    phi = Field("phi", "physical", n_components=n_comp)
    psi = Field("psi", "response", n_components=n_comp)
    res = compute_moment(
        [phi("a", "x"), phi("b", "y")],
        Action([Vertex(fields=[psi, phi, phi], coupling="F")]),
        order=0, response_phase=True, diag_R=True, diag_C=diag_C, iso_R=True,
    )
    model = PropagatorModel(
        R_time=lambda t, tp: np.exp(-MU * (t - tp)), kappa2=kappa2,
        n_components=n_comp, iso_R=True, diag_C=diag_C, t_min=0.0,
    )
    igs = [dt.build_integrand({"F": -1j * np.zeros((n_comp,) * 3)},
                              fixed_indices=fixed or {"a": 0, "b": 0})
           for dt in res.diagram_terms(0)]
    assert all(len(ig.spatial.time_integration_vars) == 0 for ig in igs)
    return igs, model


def _f17_sep(n1, n2):
    a = np.atleast_1d(np.asarray(n1, dtype=float)).ravel()
    b = np.atleast_1d(np.asarray(n2, dtype=float)).ravel()
    return float(abs(a[0] - b[0]))


@pytest.mark.parametrize("r", [0.5, 2.0, 4.0])
def test_F17_spatial_table_stays_exact_for_a_nonseparable_kernel(r):
    """With a spatial table the kappa2 ratio must NOT be substituted.

    The ratio is taken at a single t_ref, so it is exact only for a
    separable kernel.  Here the correlation length grows with time, and the
    exact answer is a genuine double time integral.
    """
    from sft_wick.evaluate import integrate_two_point_qmc
    from scipy.integrate import dblquad

    def kappa2(n1, t1, n2, t2):
        ell = 1.0 + 0.25 * (t1 + t2)
        return np.eye(_F17_N) * np.exp(-_f17_sep(n1, n2) / ell)

    igs, model = _f17_igs(kappa2)
    cache = PropagatorCache(model)
    cache.precompute_C_table_translation(t_max=_F17_T, n_grid_t=40)
    cache.precompute_C_table(t_max=_F17_T, n_grid=40)  # legacy present too

    got, _ = integrate_two_point_qmc(igs, _F17_T, {"x": 0.0, "y": r}, cache,
                                     n_samples=2 ** 8, seed=0)
    exact, _e = dblquad(
        lambda l2, l1: (np.exp(-MU * (_F17_T - l1))
                        * np.exp(-r / (1.0 + 0.25 * (l1 + l2)))
                        * np.exp(-MU * (_F17_T - l2))),
        0.0, _F17_T, lambda _: 0.0, lambda _: _F17_T, epsabs=1e-11,
    )
    assert got == pytest.approx(exact, rel=1e-3), (
        f"r={r}: {got:.9f} vs exact {exact:.9f} — the kappa2 ratio was "
        f"substituted for an exact spatial C"
    )


def test_F17_cache_without_a_legacy_table_still_works_at_order_0():
    """A table-less cache must not turn into a RuntimeError."""
    from sft_wick.evaluate import integrate_two_point_qmc

    sigma_x = 1.0

    def kappa2(n1, t1, n2, t2):
        return np.eye(_F17_N) * np.exp(-_f17_sep(n1, n2) / sigma_x)

    igs, model = _f17_igs(kappa2)
    cache = PropagatorCache(model)  # no table of any kind

    for r in (0.0, 1.0, 2.0):
        got, _ = integrate_two_point_qmc(igs, _F17_T, {"x": 0.0, "y": r},
                                         cache, n_samples=2 ** 8, seed=0)
        exact = (np.exp(-r / sigma_x)
                 * ((1.0 - np.exp(-MU * _F17_T)) / MU) ** 2)
        assert got == pytest.approx(exact, rel=1e-6), f"r={r}"


def test_F17_off_diagonal_C_uses_both_propagator_indices():
    """diag_C=False must keep C_{01}, not collapse onto C_{00}.

    `C_diagonal_batch` can only ever return diagonals, and the vectorised
    branch resolves the left leg index only -- so a diag_C=False cache must
    be delegated, not batched.
    """
    from sft_wick.evaluate import integrate_two_point_qmc

    M = np.array([[1.0, 0.7], [0.7, 1.0]])
    sigma_x = 1.0

    def kappa2(n1, t1, n2, t2):
        return M * np.exp(-_f17_sep(n1, n2) / sigma_x)

    r = 1.0
    scale = np.exp(-r / sigma_x) * ((1.0 - np.exp(-MU * _F17_T)) / MU) ** 2
    for (a, b), want in [((0, 0), M[0, 0] * scale), ((0, 1), M[0, 1] * scale)]:
        igs, model = _f17_igs(kappa2, diag_C=False, fixed={"a": a, "b": b})
        cache = PropagatorCache(model)
        got, _ = integrate_two_point_qmc(igs, _F17_T, {"x": 0.0, "y": r},
                                         cache, n_samples=2 ** 8, seed=0)
        assert got == pytest.approx(want, rel=1e-6), (
            f"<phi_{a} phi_{b}>: {got:.9f} vs {want:.9f} "
            f"(M[{a},{b}]={M[a, b]})"
        )


def test_F13_two_swept_externals_pick_the_right_column():
    """With one swept external the column index is always 0.

    `ext_integrated.index(src_v)` would then be untestable, so sweep TWO and
    make the lower-bound source the SECOND one.  Reference is the closed
    form double-integrated over both swept times.
    """
    from sft_wick.evaluate import integrate_moment
    from scipy.integrate import dblquad

    L = 3.0
    cache = _ScalarBatchCache()
    res = _quartic(1, obs=[Field("phi", "physical")("x"),
                           Field("psi", "response")("y")])
    igs = [dt.build_integrand({"g": np.array(1j)})
           for dt in res.diagram_terms(1)]

    ref, _e = dblquad(lambda ty, tx: _d1R_closed_form(tx, ty),
                      0.0, L, lambda _: 0.0, lambda _: L, epsabs=1e-10)

    for method, kw, tol in [("nquad", {}, 1e-5),
                            ("qmc_vectorized",
                             {"n_samples": 2 ** 14, "seed": 3}, 5e-3)]:
        total = sum(
            integrate_moment(ig, L, cache, method=method, t_min=0.0,
                             integrate_over=["x", "y"], **kw)[0]
            for ig in igs
        )
        assert total == pytest.approx(ref, rel=tol), (
            f"{method}: {total:.8f} vs {ref:.8f}"
        )


# --------------------------------------------------------------------- #
# F18 (7c): integrate_two_point_qmc must use the spatial table at ni >= 1
# --------------------------------------------------------------------- #
#
# F17 made the ni == 0 branch delegate, so order 0 is spatially EXACT once a
# translation/rotation/general table exists.  Orders >= 1 still hard-coded
# `C_diagonal_batch x c_spatial_factors` -- the single-t_ref kappa2 ratio --
# so the expansion was exact in its leading term and approximate above it.
# `integrate_moment_qmc_vectorized` already had the right shape (`_lookup_C`
# dispatching on `_cache_has_spatial_table`); this mirrors it.
#
# The sharp test is the boundary-validation one: for a SEPARABLE kernel the
# ratio is exact by construction, so both evaluation modes are valid and must
# agree.  Building the spatial table must therefore not move the answer at
# ANY order.  That catches both directions of error -- forgetting to route
# through C_at_batch, and applying the ratio on top of it (double-counting).

_F18_T = 3.0
_F18_SX = 1.5


def _f18_kappa_separable(n1, t1, n2, t2):
    return np.eye(1) * np.exp(-_f17_sep(n1, n2) / _F18_SX)


def _f18_setup(order, kappa2):
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    res = compute_moment(
        [phi("x"), phi("y")],
        Action([Vertex(fields=[psi, phi, phi, phi], coupling="g")]),
        order=order, ito=True, response_phase=True, collect_topology=True,
        diag_R=True, diag_C=True, iso_R=True, iso_C=True,
    )
    model = PropagatorModel(
        R_time=lambda t, tp: np.exp(-MU * (t - tp)), kappa2=kappa2,
        n_components=1, iso_R=True, diag_C=True, t_min=0.0,
    )
    igs = [dt.build_integrand({"g": np.array(1j)})
           for dt in res.diagram_terms(order)]
    return igs, model


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("r", [0.0, 1.0, 2.5])
def test_F18_spatial_table_does_not_move_a_separable_result(order, r):
    """Both C evaluation modes are valid here, so they must agree.

    Boundary validation: the kappa2 ratio is exact for a separable kernel,
    and so is `C_at_batch`.  Adding a spatial table must therefore leave the
    value unchanged at every order and separation.
    """
    from sft_wick.evaluate import integrate_two_point_qmc

    igs_a, model_a = _f18_setup(order, _f18_kappa_separable)
    if not igs_a:
        pytest.skip(f"order {order} has no diagrams")
    legacy = PropagatorCache(model_a)
    legacy.precompute_C_table(t_max=_F18_T, n_grid=60)

    igs_b, model_b = _f18_setup(order, _f18_kappa_separable)
    spatial = PropagatorCache(model_b)
    spatial.precompute_C_table_translation(t_max=_F18_T, n_grid_t=60)
    spatial.precompute_C_table(t_max=_F18_T, n_grid=60)

    kw = dict(n_samples=2 ** 12, seed=5)
    v_legacy, _ = integrate_two_point_qmc(igs_a, _F18_T, {"x": 0.0, "y": r},
                                          legacy, **kw)
    v_spatial, _ = integrate_two_point_qmc(igs_b, _F18_T, {"x": 0.0, "y": r},
                                           spatial, **kw)
    assert v_spatial == pytest.approx(v_legacy, rel=2e-3), (
        f"order {order}, r={r}: spatial table changed a separable result "
        f"({v_spatial:.10f} vs {v_legacy:.10f}) — the ratio was either "
        f"dropped or double-counted"
    )


def test_F18_nonseparable_agrees_with_the_spatially_aware_integrator():
    """Cross-path: `integrate_moment` already routes through `C_at_batch`.

    NOT an independent ground truth — both sides are sft-wick.  It is the
    check that the two entry points use the same C, which is exactly what
    7c was about; the absolute order-0 value is pinned against dblquad by
    F17.  Measured 8.17% apart before this fix.
    """
    from sft_wick.evaluate import integrate_two_point_qmc, integrate_moment

    def kappa2(n1, t1, n2, t2):
        ell = 1.0 + 0.25 * (t1 + t2)
        return np.eye(1) * np.exp(-_f17_sep(n1, n2) / ell)

    r, order = 2.0, 2
    igs, model = _f18_setup(order, kappa2)
    cache = PropagatorCache(model)
    cache.precompute_C_table_translation(t_max=_F18_T, n_grid_t=50)
    cache.precompute_C_table(t_max=_F18_T, n_grid=50)

    tp, _ = integrate_two_point_qmc(igs, _F18_T, {"x": 0.0, "y": r}, cache,
                                    n_samples=2 ** 13, seed=11)
    ref = sum(
        integrate_moment(ig, _F18_T, cache, method="qmc_vectorized",
                         t_min=0.0, n_samples=2 ** 13, seed=11,
                         positions={"x": 0.0, "y": r})[0]
        for ig in igs
    )
    assert tp == pytest.approx(ref, rel=1e-3), (
        f"two_point_qmc {tp:.8f} vs spatially-aware integrate_moment "
        f"{ref:.8f} (rel {abs(tp - ref) / abs(ref):.2%})"
    )


# --------------------------------------------------------------------- #
# F19 (7d): an ordering between TWO SWEPT externals
# --------------------------------------------------------------------- #
#
# `_causal_lower_bound_sources` emits a bound only when the LATER endpoint is
# an internal vertex; `parent_map` only when the EARLIER one is.  An ordering
# with swept externals on BOTH ends was emitted by neither.
#
# Ground truth, independent of sft-wick: <phi(x) psi(y)> at order 0 is the
# retarded R(x,y) = exp(-mu (x-y)) Theta(x-y), so sweeping BOTH externals over
# [0, L] gives
#
#     M = int_0^L dx int_0^L dy R(x,y) = L/mu - (1 - exp(-mu L)) / mu^2.
#
# Theta already made the value right; what the missing bound cost was
# quadrature accuracy, exactly as in F13 -- a jump inside the domain, which a
# fixed-node rule cannot resolve.

_F19_L = 3.0


def _f19_reference() -> float:
    return _F19_L / MU - (1.0 - np.exp(-MU * _F19_L)) / MU ** 2


def _f19_integrands():
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    res = compute_moment(
        [phi("x"), psi("y")],
        Action([Vertex(fields=[psi, phi, phi, phi], coupling="g")]),
        order=0, ito=True, response_phase=True, collect_topology=True,
        diag_R=True, diag_C=True, iso_R=True, iso_C=True,
    )
    return [dt.build_integrand({"g": np.array(1j)})
            for dt in res.diagram_terms(0)]


@pytest.mark.parametrize("method,kw,tol", [
    # Achieved WITH the ordering vs. what the mutation (ordering removed)
    # produces, so every cell can actually fail:
    #   gauss_legendre n=8    0.0     vs 2.93e-1   (29.3% at the library default)
    #   gauss_legendre n=32   0.0     vs 8.16e-2
    #   qmc_vectorized 2^14   7.5e-8  vs 1.07e-3
    # On the restricted domain the integrand is a smooth exp(-(x-y)), so GL
    # is exact; the 29% was entirely the Theta jump sitting inside the box.
    ("gauss_legendre", {"n_gauss": 8}, 1e-12),
    ("gauss_legendre", {"n_gauss": 32}, 1e-12),
    ("qmc_vectorized", {"n_samples": 2 ** 14, "seed": 4}, 1e-6),
    ("qmc_scalar", {"n_samples": 2 ** 14, "seed": 4}, 1e-6),
    # nquad carries it too, now that `all_vars` places a swept external's
    # causal predecessors at HIGHER indices -- scipy calls ranges[i] with
    # all_vars[i+1:], so only an outer variable can bound an inner one.
    # 3.4e-6 (domain unrestricted, adaptivity absorbing the jump) -> exact.
    ("nquad", {}, 1e-12),
])
def test_F19_swept_to_swept_ordering_is_carried(method, kw, tol):
    """Sweeping both ends of an R propagator must respect its retardation."""
    from sft_wick.evaluate import integrate_moment

    cache = _ScalarBatchCache()
    total = sum(
        integrate_moment(ig, _F19_L, cache, method=method, t_min=0.0,
                         integrate_over=["x", "y"], **kw)[0]
        for ig in _f19_integrands()
    )
    ref = _f19_reference()
    assert total == pytest.approx(ref, rel=tol), (
        f"{method} {kw}: {total:.8f} vs closed form {ref:.8f} "
        f"(rel {abs(total - ref) / abs(ref):.2e})"
    )


def test_F19_helper_orders_and_is_a_no_op_without_swept_edges():
    """The ordering must be a strict no-op when no swept-to-swept edge exists.

    That is what makes the change bit-identical for every existing call.
    """
    from sft_wick.evaluate import _swept_external_order

    class _Sp:
        time_orderings = (("y", "x"), ("y", "v0"), ("v0", "x"))

    order, lowers, lo_c, hi_c = _swept_external_order(_Sp(), ["x", "y"])
    assert lowers == {"x": ["y"]}
    assert order.index("y") < order.index("x"), order

    # only x swept -> the (y, x) edge has a non-swept end -> no swept source,
    # but y is now a FIXED external, so it must become a CONSTANT lower bound.
    order2, lowers2, lo2, hi2 = _swept_external_order(
        _Sp(), ["x"], fixed_times={"y": 1.25}, lambda_f=5.0, t_min=0.0,
    )
    assert (order2, lowers2) == (["x"], {})
    assert lo2.get("x") == 1.25, lo2

    # ... and the mirror case: a swept point that PRECEDES a fixed one is
    # bounded from above by it.
    class _Fwd:
        time_orderings = (("x", "y"),)

    _, _, lo3, hi3 = _swept_external_order(
        _Fwd(), ["x"], fixed_times={"y": 2.5}, lambda_f=9.0, t_min=0.0,
    )
    assert hi3.get("x") == 2.5, hi3

    # swept -> internal -> swept must close transitively; the raw edge list
    # has no swept-to-swept edge at all.
    class _Chain:
        time_orderings = (("a", "v0"), ("v0", "b"))

    order4, lowers4, _, _ = _swept_external_order(_Chain(), ["a", "b"])
    assert lowers4 == {"b": ["a"]}, lowers4
    assert order4.index("a") < order4.index("b"), order4

    # a cycle must fall back, never drop a variable
    class _Cyc:
        time_orderings = (("a", "b"), ("b", "a"))

    order5, lowers5, _, _ = _swept_external_order(_Cyc(), ["a", "b"])
    assert (order5, lowers5) == (["a", "b"], {})


# --------------------------------------------------------------------- #
# F20 (7a): unequal FIXED external times in the production integrators
# --------------------------------------------------------------------- #
#
# Every production integrator took a single `lambda_f` and pinned ALL fixed
# externals there, so `R(t, t')` and `C(t, t')` -- the DMFT order parameters --
# were unreachable through any supported path.  Worse, with every external at
# one time, Theta kills the R joining them, so every observable carrying an
# external psi leg was identically 0 at every order through all five backends.
# That 0 is the correct Ito value and completely useless.
#
# `lambda_f` played two roles and only one generalises: the sweep limit (stays)
# and "the time of a fixed external point" (now per point, via
# `external_times`).  Ground truth is closed form at both orders:
#
#     order 0:  R(tx, ty) = exp(-mu (tx - ty)) Theta(tx - ty)
#     order 1:  _d1R_closed_form(tx, ty)  (already validated by F9)

_F20_MTHDS = ["nquad", "qmc", "qmc_scalar", "qmc_vectorized", "gauss_legendre"]


def _f20_response(order):
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    res = compute_moment(
        [phi("x"), psi("y")],
        Action([Vertex(fields=[psi, phi, phi, phi], coupling="g")]),
        order=order, ito=True, response_phase=True, collect_topology=True,
        diag_R=True, diag_C=True, iso_R=True, iso_C=True,
    )
    return [dt.build_integrand({"g": np.array(1j)})
            for dt in res.diagram_terms(order)]


def _f20_eval(order, tx, ty, method, **kw):
    from sft_wick.evaluate import integrate_moment
    cache = _ScalarBatchCache()
    return sum(
        integrate_moment(ig, tx, cache, method=method, t_min=0.0,
                         external_times={"x": tx, "y": ty}, **kw)[0]
        for ig in _f20_response(order)
    )


@pytest.mark.parametrize("method", _F20_MTHDS)
@pytest.mark.parametrize("ty", [1.0, 2.0, 3.5])
def test_F20_order0_response_at_unequal_times(method, ty):
    """R(T, t') must come out, not the identically-zero equal-time value."""
    T = 4.0
    kw = {"n_gauss": 16} if method == "gauss_legendre" else {}
    got = _f20_eval(0, T, ty, method, **kw)
    assert got == pytest.approx(np.exp(-MU * (T - ty)), rel=1e-9)
    assert got != 0.0


@pytest.mark.parametrize("method", _F20_MTHDS)
@pytest.mark.parametrize("ty", [1.0, 3.0])
def test_F20_order1_response_matches_the_closed_form(method, ty):
    """The O(g) response at unequal external times, vs the closed form."""
    T = 4.0
    kw = ({"n_samples": 2 ** 14, "seed": 2} if method.startswith("qmc")
          else {"n_gauss": 24} if method == "gauss_legendre" else {})
    got = _f20_eval(1, T, ty, method, **kw)
    want = _d1R_closed_form(T, ty)
    assert got == pytest.approx(want, rel=2e-3), (
        f"{method} ty={ty}: {got:.8f} vs closed form {want:.8f}"
    )


@pytest.mark.parametrize("method", _F20_MTHDS)
def test_F20_acausal_and_equal_times_are_exactly_zero(method):
    """Theta must still hold: t_y >= t_x gives exactly 0, not a huge number."""
    kw = {"n_gauss": 16} if method == "gauss_legendre" else {}
    for tx, ty in [(2.0, 2.0), (2.0, 3.5)]:
        assert _f20_eval(0, tx, ty, method, **kw) == pytest.approx(0.0,
                                                                   abs=1e-14)


@pytest.mark.parametrize("method", _F20_MTHDS)
@pytest.mark.parametrize("order", [0, 1, 2])
def test_F20_default_is_bit_identical(method, order):
    """`external_times=None` must reproduce the old numbers EXACTLY.

    Not approximately: this is the guarantee that adding the feature moved
    nothing.  Equal explicit times must also match the default bit for bit.
    """
    from sft_wick.evaluate import integrate_moment
    T = 3.0
    kw = ({"n_samples": 2 ** 12, "seed": 8} if method.startswith("qmc")
          else {"n_gauss": 8} if method == "gauss_legendre" else {})
    igs = [dt.build_integrand({"g": np.array(1j)})
           for dt in _quartic(order).diagram_terms(order)]
    cache = _ScalarBatchCache()
    for ig in igs:
        base = integrate_moment(ig, T, cache, method=method, t_min=0.0, **kw)[0]
        same = integrate_moment(ig, T, cache, method=method, t_min=0.0,
                                external_times={"x": T, "y": T}, **kw)[0]
        assert same == base, (
            f"{method} order {order}: explicit equal times {same!r} != "
            f"default {base!r}"
        )


def test_F20_validation_rejects_bad_names_and_conflicts():
    from sft_wick.evaluate import integrate_moment
    ig = _f20_response(1)[0]
    cache = _ScalarBatchCache()
    with pytest.raises(ValueError, match="unknown external point"):
        integrate_moment(ig, 3.0, cache, method="nquad", t_min=0.0,
                         external_times={"nope": 1.0})
    with pytest.raises(ValueError, match="BOTH external_times and"):
        integrate_moment(ig, 3.0, cache, method="nquad", t_min=0.0,
                         integrate_over=["y"], external_times={"y": 1.0})


@pytest.mark.parametrize("method", _F20_MTHDS)
def test_F20_lambda_f_must_not_stand_in_for_an_external_time(method):
    """Decouple `lambda_f` from every external time.

    The two roles coincide in every other test (`lambda_f == t_x`), which is
    exactly why the conflation survived: a vertex bounded above by external
    `x` reads `lambda_f` instead of `t_x`, and with them equal nothing shows.
    Here `lambda_f = 6` while `x` sits at 4, so a backend that still uses
    `lambda_f` integrates the vertex over `[t_y, 6]` instead of `[t_y, 4]`.

    Mutation-verified for `qmc`, `qmc_vectorized`, `gauss_legendre` and (via
    its own `times` seed) `qmc_scalar`.  It does NOT fail for `nquad`: there
    the over-wide bound only enlarges a region where Theta already zeroes
    R(x, s), so the value stays right and only quadrature effort is wasted --
    the same pattern as F13 and F19.
    """

    from sft_wick.evaluate import integrate_moment

    tx, ty, lam = 4.0, 1.0, 6.0
    kw = ({"n_samples": 2 ** 14, "seed": 2} if method.startswith("qmc")
          else {"n_gauss": 24} if method == "gauss_legendre" else {})
    cache = _ScalarBatchCache()
    got = sum(
        integrate_moment(ig, lam, cache, method=method, t_min=0.0,
                         external_times={"x": tx, "y": ty}, **kw)[0]
        for ig in _f20_response(1)
    )
    want = _d1R_closed_form(tx, ty)
    assert got == pytest.approx(want, rel=2e-3), (
        f"{method}: {got:.8f} vs closed form {want:.8f} — lambda_f={lam} "
        f"leaked in as the time of external 'x' (t_x={tx})"
    )


# --------------------------------------------------------------------- #
# F21 (7b): the diagonal kink in precompute_C_table
# --------------------------------------------------------------------- #
#
# C(t1,t2) = int_0^{min(t1,t2)} R(t1,l) sigma2(l) R(t2,l) dl.  The min() puts a
# derivative discontinuity of exactly -sigma2(t) on t1 == t2: approaching from
# t1 < t2 the moving upper limit contributes an extra R(t1,t1) sigma2(t1)
# R(t2,t1), absent from the other side.
#
# `RectBivariateSpline` is C^2 by construction and cannot represent that, so
# the table stopped converging ON the diagonal while staying clean O(h^4) off
# it.  Every tadpole evaluates C(s,s), exactly on the kink.  Measured relative
# error of C_diagonal(t,t) at mid-cell t:
#
#     n_grid      before        after
#         41    2.227e-01    3.525e-04
#         81    2.178e-01    4.773e-05
#        161    2.152e-01    6.210e-06
#        321    2.139e-01    7.920e-07
#     exponent p   0.009        2.97
#
# i.e. before, refining the grid 8x moved the error from 22.3% to 21.4%.
#
# To be precise about WHERE that lives: the ABSOLUTE error on the diagonal is
# clean O(h) (p = 1.00 measured), versus O(h^4) off it.  The *relative* max
# stalls only because it is attained as t -> 0, where C(t,t) is itself O(h),
# so numerator and denominator shrink together.  At an ordinary time the
# relative error does fall, but only linearly: 1.42% / 0.71% / 0.36% / 0.18%
# at t = 2 for n_grid = 41 / 81 / 161 / 321.  So the practical cost is an
# accuracy FLOOR that refining the time grid cannot remove.
#
# The fix harvests the i == j grid entries -- already computed -- into a 1-D
# CubicSpline, which IS smooth along the diagonal.

_F21_TMAX = 4.0


def _f21_cache(n_grid):
    """Cache whose grid is the EXACT C sampled on nodes.

    `c_value_fn` removes the quadrature entirely, so what these tests measure
    is interpolation error and nothing else.
    """
    cache = _scalar_cache()
    cache.precompute_C_table(t_max=_F21_TMAX, n_grid=n_grid)
    return cache


def _f21_max_rel_err(cache, n_grid, offset=0.0):
    ts = np.linspace(0.0, _F21_TMAX, n_grid)
    mids = 0.5 * (ts[:-1] + ts[1:])          # worst case: between nodes
    worst = 0.0
    for t in mids:
        t2 = t + offset
        if t2 > _F21_TMAX:
            continue
        got = float(cache.C_diagonal(0, t, 0, t2)[0])
        exact = _C0(t, t2)
        worst = max(worst, abs(got - exact) / max(abs(exact), 1e-30))
    return worst


def test_F21_diagonal_converges_instead_of_stalling():
    """On the diagonal the table must converge, not sit at ~22%."""
    errs = {n: _f21_max_rel_err(_f21_cache(n), n) for n in (41, 161)}
    assert errs[41] < 1e-3, (
        f"mid-cell C(t,t) rel err {errs[41]:.3e} at n_grid=41 "
        f"(was 2.227e-01 with the 2-D spline alone)"
    )
    p = np.log(errs[41] / errs[161]) / np.log(4.0)
    assert p > 2.5, f"diagonal convergence exponent {p:.3f} (was 0.009)"


def test_F21_off_diagonal_is_untouched():
    """The working path must keep its O(h^4) -- this is a control."""
    # n=41 is still pre-asymptotic here (its own step measures p ~ 7.6), so
    # take the exponent in the asymptotic range.
    errs = {n: _f21_max_rel_err(_f21_cache(n), n, offset=1.0)
            for n in (161, 321)}
    p = np.log(errs[161] / errs[321]) / np.log(2.0)
    assert 3.5 < p < 4.5, f"off-diagonal exponent {p:.3f}, expected ~4"
    assert errs[321] < 1e-8


def test_F21_the_kink_is_exactly_minus_sigma2():
    """Pin WHY a tensor-product spline cannot do this, not just that it can't.

    Independent of sft-wick: differentiate the closed form across t1 == t2.
    """
    t, eps = 2.0, 1e-6
    d_above = (_C0(t + eps, t) - _C0(t, t)) / eps
    d_below = (_C0(t, t) - _C0(t - eps, t)) / eps
    assert (d_above - d_below) == pytest.approx(-2.0 * D, rel=1e-5), (
        "the derivative jump across the diagonal is not -sigma2"
    )


def test_F21_tadpole_coefficient_is_accurate():
    """The physically meaningful payoff.

    <x^2> at O(g) is -3 int_0^T ds R0(T,s)^2 C0(s,s) -- every term ON the
    diagonal.  Before: 7.285e-03 at n_grid=41, falling only as O(h)
    (3.651e-03 / 1.826e-03 / 9.130e-04 at 81/161/321).
    """
    from scipy.integrate import quad as _quad

    T = 3.0
    exact, _ = _quad(lambda s: np.exp(-2 * MU * (T - s)) * _C0(s, s), 0.0, T,
                     limit=400)
    exact *= -3.0
    cache = _f21_cache(41)
    ss = np.linspace(0.0, T, 2001)
    vals = np.array([float(cache.C_diagonal(0, s, 0, s)[0]) for s in ss])
    got = -3.0 * np.trapezoid(np.exp(-2 * MU * (T - ss)) * vals, ss)
    assert got == pytest.approx(exact, rel=1e-5), (
        f"tadpole c1 {got:.10f} vs exact {exact:.10f} "
        f"(was 7.3e-03 relative at this grid)"
    )


def test_F21_batch_and_scalar_accessors_agree_on_the_diagonal():
    """All three table accessors must route identically.

    `_C_diagonal_from_table`, `_C_value_from_table` and `C_diagonal_batch`
    are three entry points to the same table; letting only some of them
    take the diagonal spline is exactly the two-conventions failure mode.
    """
    cache = _f21_cache(41)
    ts = np.array([0.35, 1.15, 2.75, 3.9])
    batch = cache.C_diagonal_batch(ts, ts)
    for k, t in enumerate(ts):
        scalar = float(cache.C_diagonal(0, t, 0, t)[0])
        matrix = float(cache.C_value(0, t, 0, t)[0, 0])
        assert batch[k, 0] == pytest.approx(scalar, rel=1e-14)
        assert matrix == pytest.approx(scalar, rel=1e-14)
        assert scalar == pytest.approx(_C0(t, t), rel=1e-3)


def test_F21_lazy_spatial_path_has_the_same_diagonal_fix():
    """The LAZY spatial builder carried the identical kink.

    It builds its own `RectBivariateSpline(ts, ts, grid)` per parameter value,
    so it reproduced the legacy table's numbers bit-for-bit: 2.227e-01 at
    n_grid_t=41, 2.178e-01 at 81.  This is the path `examples/demo1` uses --
    the earlier note that the defect "touches zero shipped demo output" was
    wrong, and fixing it moves demo1's rows by up to 6.1e-3 relative.
    """
    t_max = 4.0
    for n in (41, 81):
        cache = _scalar_cache()
        cache.precompute_C_table_translation(t_max=t_max, n_grid_t=n)
        ts = np.linspace(0.0, t_max, n)
        z = np.array([0.0])
        worst = max(
            abs(float(cache.C_at_batch(np.array([t]), np.array([t]), z, z)[0, 0])
                - _C0(t, t)) / max(abs(_C0(t, t)), 1e-30)
            for t in 0.5 * (ts[:-1] + ts[1:])
        )
        assert worst < 1e-3, (
            f"lazy spatial path, n_grid_t={n}: mid-cell C(t,t) rel err "
            f"{worst:.3e} (was 2.2e-01)"
        )


# --------------------------------------------------------------------- #
# F22: the three carve-outs left by F13/F19/F20/F21
# --------------------------------------------------------------------- #


def test_F22_full_spatial_grid_diagonal_is_fixed_too():
    """The FULL (non-lazy) spatial grid had the ridge as well -- worse.

    `RegularGridInterpolator` with method 'linear' blends across the ridge
    inside any cell that straddles it, and a bilinear cut is harsher than a
    smooth spline's: 52.4% relative error at n_grid_t=41, barely converging
    (51.2% / 50.6% at 81 / 161).  Harvesting `grid[i, i, ...]` into an
    interpolator over `(t, *extra)` gives 4.76e-02 / 2.44e-02 / 1.24e-02.
    """
    sigma_x = 1.0

    def kappa2(n1, t1, n2, t2):
        return np.eye(1) * np.exp(-_f17_sep(n1, n2) / sigma_x)

    t_max = 4.0
    prev = None
    for n in (41, 81):
        model = PropagatorModel(
            R_time=lambda t, tp: np.exp(-MU * (t - tp)), kappa2=kappa2,
            sigma2=lambda n1, t, n2: np.array([[2.0 * D]]),
            n_components=1, iso_R=True, diag_C=True, t_min=0.0,
        )
        cache = PropagatorCache(
            model,
            c_value_fn=lambda n1, t1, n2, t2: np.array([[_C0(t1, t2)]]),
        )
        cache.precompute_C_table_translation(
            t_max=t_max, n_grid_t=n, r_max=3.0, n_grid_r=13,
        )
        ts = np.linspace(0.0, t_max, n)
        z = np.array([0.0])
        worst = max(
            abs(float(cache.C_at_batch(np.array([t]), np.array([t]), z, z)[0, 0])
                - _C0(t, t)) / max(abs(_C0(t, t)), 1e-30)
            for t in 0.5 * (ts[:-1] + ts[1:])
        )
        assert worst < 0.1, (
            f"full spatial grid, n_grid_t={n}: diagonal rel err {worst:.3e} "
            f"(was 5.2e-01)"
        )
        if prev is not None:
            assert worst < 0.7 * prev, (
                f"not converging on the diagonal: {prev:.3e} -> {worst:.3e}"
            )
        prev = worst


def test_F22_two_point_qmc_takes_external_times():
    """`integrate_two_point_qmc` was the last single-time entry point.

    Order 0 of `<phi(x) phi(y)>` at unequal times has the closed form
    C0(t1, t2) * exp(-r/sigma_x) for a separable kernel.
    """
    from sft_wick.evaluate import integrate_two_point_qmc

    igs, cache = _f14_setup(order=0)
    tx, ty, r = 4.0, 1.5, 1.0
    got, _ = integrate_two_point_qmc(
        igs, tx, {"x": 0.0, "y": r}, cache, n_samples=2 ** 10, seed=0,
        external_times={"x": tx, "y": ty},
    )
    # Same model as F14: R = exp(-mu (t-t')), kappa separable exponential.
    exact = (np.exp(-r / _F14_SIGMA_X)
             * (1.0 - np.exp(-MU * tx)) * (1.0 - np.exp(-MU * ty)) / MU ** 2)
    assert got == pytest.approx(exact, rel=1e-5), (
        f"two-point at unequal times: {got:.10f} vs closed form {exact:.10f}"
    )

    # Default and explicit-equal must be bit-identical to the old behaviour.
    base, _ = integrate_two_point_qmc(igs, tx, {"x": 0.0, "y": r}, cache,
                                      n_samples=2 ** 10, seed=0)
    same, _ = integrate_two_point_qmc(
        igs, tx, {"x": 0.0, "y": r}, cache, n_samples=2 ** 10, seed=0,
        external_times={"x": tx, "y": tx},
    )
    assert same == base

    with pytest.raises(ValueError, match="unknown external point"):
        integrate_two_point_qmc(igs, tx, {"x": 0.0, "y": r}, cache,
                                n_samples=2 ** 8, seed=0,
                                external_times={"zzz": 1.0})


@pytest.mark.parametrize("method,kw,tol", [
    ("nquad", {}, 1e-12),
    ("gauss_legendre", {"n_gauss": 8}, 1e-12),
    ("qmc_vectorized", {"n_samples": 2 ** 14, "seed": 4}, 1e-6),
])
def test_F22_nquad_orders_swept_externals_outermost(method, kw, tol):
    """The swept-external ORDER matters, not just the bound.

    scipy calls `ranges[i]` with `all_vars[i+1:]`, so a causal predecessor
    must sit at a HIGHER index.  `external_points` is canonicalised to sorted
    order, so whether that happens by luck depends on the propagator's
    orientation:

      * `<phi(x) psi(y)>` gives R(x,y), ordering (y -> x): the predecessor y
        already sorts last, and the natural order works.  That is the F19
        case, and the reordering is a no-op there -- it passes either way.
      * `<phi(y) psi(x)>` gives R(y,x), ordering (x -> y): the predecessor x
        sorts FIRST, i.e. innermost, and its value is not yet bound when the
        range callable for y fires.  Without the reordering the bound is
        silently inert -- measured 3.412e-06 with an IntegrationWarning.

    So this pins the second orientation.  Reference is the same closed form:
    int_0^L dx int_0^L dy R(y,x) = L/mu - (1 - exp(-mu L))/mu^2.
    """
    from sft_wick.evaluate import integrate_moment

    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    res = compute_moment(
        [phi("y"), psi("x")],
        Action([Vertex(fields=[psi, phi, phi, phi], coupling="g")]),
        order=0, ito=True, response_phase=True, collect_topology=True,
        diag_R=True, diag_C=True, iso_R=True, iso_C=True,
    )
    igs = [dt.build_integrand({"g": np.array(1j)})
           for dt in res.diagram_terms(0)]
    assert ("x", "y") in tuple(igs[0].spatial.time_orderings), (
        "expected the R(y,x) orientation, i.e. ordering (x -> y)"
    )
    cache = _ScalarBatchCache()
    total = sum(
        integrate_moment(ig, _F19_L, cache, method=method, t_min=0.0,
                         integrate_over=["x", "y"], **kw)[0]
        for ig in igs
    )
    assert total == pytest.approx(_f19_reference(), rel=tol)


# --------------------------------------------------------------------- #
# F23: the diagonal-ridge fix must not cost picklability
# --------------------------------------------------------------------- #
#
# `RectBivariateSpline` is picklable; `CubicSpline` is NOT (scipy 1.18:
# "cannot pickle 'module' object").  So naively storing a CubicSpline for the
# diagonal made every table-carrying PropagatorCache unserialisable -- it could
# no longer be sent through joblib or saved.  Found by the final branch review.
#
# The obvious swap is worse, not better: `RegularGridInterpolator(method=
# 'cubic')` IS picklable but DIVERGES here as the grid refines -- mid-cell
# relative error 4.0e-04 at n_grid=41 against 8.2e-03 at n_grid=321.  That
# trades a serialisation bug for a numerical one, which is why `_DiagLineSpline`
# keeps the CubicSpline and serialises only its nodes and values.


@pytest.mark.parametrize("kind", ["none", "legacy", "lazy", "full"])
def test_F23_propagator_cache_survives_a_pickle_round_trip(kind):
    """Every table flavour must round-trip, with identical values."""
    import pickle
    import tests._pickle_model as pm  # noqa: F401  (module-level = picklable)

    model = PropagatorModel(
        R_time=pm.R_time, kappa2=pm.kappa2, sigma2=pm.sigma2,
        n_components=1, iso_R=True, diag_C=True, t_min=0.0,
    )
    cache = PropagatorCache(model, c_value_fn=pm.c_value_fn)
    if kind == "legacy":
        cache.precompute_C_table(t_max=4.0, n_grid=21)
    elif kind == "lazy":
        cache.precompute_C_table_translation(t_max=4.0, n_grid_t=21)
    elif kind == "full":
        cache.precompute_C_table_translation(
            t_max=4.0, n_grid_t=21, r_max=2.0, n_grid_r=7,
        )

    z = np.array([0.0])

    def probe(c):
        if kind in ("lazy", "full"):
            return float(c.C_at_batch(np.array([2.0]), np.array([2.0]),
                                      z, z)[0, 0])
        return float(c.C_diagonal(0, 2.0, 0, 2.0)[0])

    before = probe(cache)
    after = probe(pickle.loads(pickle.dumps(cache)))
    assert after == before, f"{kind}: {before!r} -> {after!r} across pickle"
    # and it is still the right number (t=2.0 is a node of this grid)
    assert before == pytest.approx(_C0(2.0, 2.0), rel=1e-9)


def test_F23_diag_line_spline_keeps_cubic_accuracy():
    """Guard the trade-off: picklable must not mean less accurate.

    Pins that the diagonal interpolator still converges -- the divergent
    RegularGridInterpolator('cubic') alternative would fail this at n=161.
    """
    from sft_wick.evaluate import _diag_line_interp, _diag_line_eval

    errs = {}
    for n in (41, 161):
        ts = np.linspace(0.0, 4.0, n)
        itp = _diag_line_interp(ts, np.array([_C0(t, t) for t in ts]))
        mids = 0.5 * (ts[:-1] + ts[1:])
        errs[n] = max(
            abs(float(_diag_line_eval(itp, t)[0]) - _C0(t, t))
            / max(abs(_C0(t, t)), 1e-30)
            for t in mids
        )
    assert errs[161] < errs[41], f"not converging: {errs}"
    assert errs[161] < 1e-5, errs
