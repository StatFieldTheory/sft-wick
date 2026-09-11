"""A DiagramTerm's R-propagator indices must match the cache's R type.

``compute_moment`` writes an R propagator with two component indices
(``R_{a i}``, the default), one repeated index (``diag_R=True``) or none
(``iso_R=True``).  A propagator cache holds a scalar R (``model.iso_R``,
meaning ``R_ab = delta_ab R``) or a matrix R.  The evaluators read R's indices
only for a matrix R, and skip the factor of an R absorbed into an
``already_R_contracted`` vertex.  Up to 0.4.2 three pairings returned a wrong
number without an error; they now raise ``ValueError``:

* two indices, scalar-R cache: the delta was dropped and the leg components
  were summed;
* no index, matrix-R cache: R was evaluated as its trace, N = 3 times too
  large per R;
* two indices on an absorbed R, either cache: the leg components were summed.

The system is order-1 ``<phi_a(x1) phi_b(x2) phi_c(z)>`` with N = 3, one
non-local psi-psi-psi vertex with the static coupling ``K = (i/6) A`` (A a
random (3, 3, 3) tensor), R = Theta (``DiagonalA`` with gamma = 0, or the
identity matrix) and every external at t = 1.  Each R factor is 1 on the
integration domain, whose volume is 1, so every backend returns the coupling
contraction to rounding:

    HAND = (1/6) sum over permutations beta of A[beta(0), beta(1), beta(2)]

The mismatched pairings returned ``A.sum()`` and ``27 * HAND``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.evaluate import (
    integrate_diagrams,
    integrate_moment,
    integrate_two_point_qmc,
)

N = 3
A = np.random.default_rng(20260911).normal(size=(N, N, N))
COUPLING = {"K": (1j / 6.0) * A}
HAND = sum(A[beta] for beta in permutations(range(N))) / 6.0
FIXED = {"a": 0, "b": 1, "c": 2}
BACKENDS = ("qmc_scalar", "gauss_legendre", "qmc_vectorized")
N_SAMPLES = 2**5


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


N_A, N_B = _unit([0.3, -0.2, 0.93]), _unit([0.8, 0.5, -0.33])
POSITIONS = {"x1": N_A, "x2": N_A, "z": N_B}


@dataclass(frozen=True)
class _IdentityR:
    """Matrix R = Theta(t1 - t2) times the n x n identity."""

    n: int

    def __call__(self, t1, t2):
        return np.eye(self.n) if t1 > t2 else np.zeros((self.n, self.n))


def _system(r_type: str, n: int = N) -> sw.System:
    linear = (sw.DiagonalA(gamma=[0.0] * n) if r_type == "scalar"
              else sw.ExplicitR(R_time=_IdentityR(n), iso_R=False))
    noise = sw.GaussianNoise(kappa2=sw.SeparableRotation(
        temporal=sw.ExponentialTemporal(lam=1.0, sigma_t=0.5),
        angular=sw.LegendreAngular(coeffs=[1.0])))
    coupling = A if n == N else np.zeros((n, n, n))
    return sw.System(
        field=sw.FieldSpec("phi", n_components=n), linear=linear, noise=noise,
        nonlocal_vertices=[sw.NonLocalVertex("K", order=3, coupling=coupling)],
    )


@lru_cache(maxsize=None)
def _props(r_type: str, n: int = N):
    return _system(r_type, n).propagators(t_max=2.0, n_grid_t=8)


def _cache(r_type: str, n: int = N):
    return _props(r_type, n).cache


def _terms(n: int = N, *, absorbed: bool = False, **flags):
    """Order-1 L0 DiagramTerms of <phi_a(x1) phi_b(x2) phi_c(z)>."""
    sw.reset_uid_counter()
    phi = sw.Field("phi", "physical", n_components=n)
    psi = sw.Field("psi", "response", n_components=n)
    if n == 1:
        ops = [phi("x1"), phi("x2"), phi("z")]
    else:
        ops = [phi("a", "x1"), phi("b", "x2"), phi("c", "z")]
    vertex = sw.Vertex(fields=[psi] * 3, coupling="K", local=False,
                       already_R_contracted=absorbed)
    return sw.compute_moment(ops, sw.Action(vertices=[vertex]), order=1,
                             **flags).diagram_terms(1)


def _integrate(terms, r_type, method, *, n=N, coupling=COUPLING, fixed=FIXED):
    total, _ = integrate_diagrams(
        terms, coupling, lambda_f=1.0, cache=_cache(r_type, n), method=method,
        n_samples=N_SAMPLES, seed=1, fixed_indices=fixed, positions=POSITIONS,
    )
    return total


def _assert_hand(value, hand=HAND):
    assert abs(value - hand) <= 1e-12 * abs(hand), f"{value!r} != {hand!r}"


# ---------------------------------------------------------------------------
# Controls: terms that match the cache give the hand value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", BACKENDS)
@pytest.mark.parametrize("flag", ["diag_R", "iso_R"])
def test_scalar_R_cache_with_the_delta_applied(flag, method):
    _assert_hand(_integrate(_terms(**{flag: True}), "scalar", method))


@pytest.mark.parametrize("flags", [{}, {"diag_R": True}], ids=["default", "diag_R"])
def test_matrix_R_cache_with_indexed_R(flags):
    _assert_hand(_integrate(_terms(**flags), "matrix", "qmc_scalar"))


@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized"])
@pytest.mark.parametrize("flags", [{}, {"diag_R": True}], ids=["default", "diag_R"])
def test_matrix_R_cache_keeps_its_batched_backend_refusal(flags, method):
    """Terms that match a matrix R still get the batched backends' existing
    NotImplementedError; the ValueError is only for terms that contradict R."""
    with pytest.raises(NotImplementedError, match="matrix-valued R"):
        _integrate(_terms(**flags), "matrix", method)


# ---------------------------------------------------------------------------
# The two reported mismatches, through integrate_diagrams
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", BACKENDS + ("nquad",))
def test_two_index_R_with_scalar_R_cache_raises(method):
    """Returned A.sum() on every backend: each R's delta_{a i} was dropped."""
    with pytest.raises(ValueError, match="R is a scalar"):
        _integrate(_terms(), "scalar", method)


@pytest.mark.parametrize("method", BACKENDS + ("nquad",))
def test_index_free_R_with_matrix_R_cache_raises(method):
    """Returned 27 * HAND on qmc_scalar: trace(I_3) for each of the three R."""
    with pytest.raises(ValueError, match="R is a matrix"):
        _integrate(_terms(iso_R=True), "matrix", method)


# ---------------------------------------------------------------------------
# Every public route from a DiagramIntegrand to a number is guarded
# ---------------------------------------------------------------------------

def _evaluate(ig, cache):
    """``DiagramIntegrand.evaluate`` at one point inside the domain."""
    sp = ig.spatial
    times = dict.fromkeys(sp.external_points, 1.0)
    times.update(dict.fromkeys(sp.time_integration_vars, 0.5))
    directions = dict.fromkeys(set(sp.direction_map.values()), N_A)
    return (ig.evaluate(times, directions, cache).real, 0.0)


_ENTRY_POINTS = {
    "integrate_moment": lambda ig, c: integrate_moment(
        ig, 1.0, c, method="qmc", n_samples=N_SAMPLES, seed=1,
        positions=POSITIONS),
    "integrate_moment_qmc": lambda ig, c: ig.integrate_moment_qmc(
        1.0, c, n_samples=N_SAMPLES, seed=1, positions=POSITIONS),
    "integrate_moment_qmc_vectorized": lambda ig, c: (
        ig.integrate_moment_qmc_vectorized(
            1.0, c, n_samples=N_SAMPLES, seed=1, positions=POSITIONS)),
    "integrate_moment_gauss_legendre": lambda ig, c: (
        ig.integrate_moment_gauss_legendre(1.0, c, positions=POSITIONS)),
    "integrate_moment_nquad": lambda ig, c: ig.integrate_moment_nquad(
        1.0, c, positions=POSITIONS),
    "evaluate": _evaluate,
    "integrate_two_point_qmc": lambda ig, c: integrate_two_point_qmc(
        [ig], 1.0, POSITIONS, c, n_samples=N_SAMPLES, seed=1),
}


def _integrand(**flags):
    (term,) = _terms(**flags)
    return term.build_integrand(COUPLING, FIXED)


@pytest.mark.parametrize("entry", list(_ENTRY_POINTS))
def test_every_entry_point_evaluates_matching_terms(entry):
    value, _ = _ENTRY_POINTS[entry](_integrand(iso_R=True), _cache("scalar"))
    _assert_hand(value)


@pytest.mark.parametrize("entry", list(_ENTRY_POINTS))
@pytest.mark.parametrize("r_type, flags, match", [
    ("scalar", {}, "R is a scalar"),
    ("matrix", {"iso_R": True}, "R is a matrix"),
], ids=["two_index_R-scalar_cache", "index_free_R-matrix_cache"])
def test_every_entry_point_refuses_mismatched_terms(entry, r_type, flags, match):
    with pytest.raises(ValueError, match=match):
        _ENTRY_POINTS[entry](_integrand(**flags), _cache(r_type))


def test_parallel_integrate_diagrams_refuses_before_starting_workers(monkeypatch):
    """integrate_diagrams checks every term up front, so with n_jobs != 1 the
    ValueError comes from the caller's process, not from inside a worker."""
    joblib = pytest.importorskip("joblib")

    def no_workers(*args, **kwargs):
        raise AssertionError("joblib.Parallel was started")

    monkeypatch.setattr(joblib, "Parallel", no_workers)
    # n_jobs != 1 is honoured only for more than two terms.
    with pytest.raises(ValueError, match="R is a scalar"):
        integrate_diagrams(
            _terms() * 3, COUPLING, lambda_f=1.0, cache=_cache("scalar"),
            method="qmc_scalar", n_jobs=2, fixed_indices=FIXED,
            positions=POSITIONS,
        )


# ---------------------------------------------------------------------------
# An R absorbed into an already_R_contracted vertex
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", BACKENDS)
@pytest.mark.parametrize("r_type", ["scalar", "matrix"])
def test_absorbed_R_with_two_indices_raises(r_type, method):
    """Returned A.sum() with either cache: the absorbed R's factor is skipped,
    and the Kronecker delta that stands in for it was never applied."""
    with pytest.raises(ValueError, match="already_R_contracted"):
        _integrate(_terms(absorbed=True), r_type, method)


@pytest.mark.parametrize("method", BACKENDS)
@pytest.mark.parametrize("r_type", ["scalar", "matrix"])
@pytest.mark.parametrize("flag", ["diag_R", "iso_R"])
def test_absorbed_R_with_the_delta_applied(flag, r_type, method):
    _assert_hand(_integrate(_terms(absorbed=True, **{flag: True}), r_type, method))


# ---------------------------------------------------------------------------
# External-external R: <phi_a(x, 1) psi_b(y, 1/2)> = R_ab = delta_ab here
# ---------------------------------------------------------------------------

def _response(r_type, a, b, **flags):
    sw.reset_uid_counter()
    phi = sw.Field("phi", "physical", n_components=N)
    psi = sw.Field("psi", "response", n_components=N)
    terms = sw.compute_moment([phi("a", "x"), psi("b", "y")],
                              sw.Action(vertices=[]), order=0,
                              **flags).diagram_terms(0)
    total, _ = integrate_diagrams(
        terms, {}, lambda_f=1.0, cache=_cache(r_type), method="qmc_scalar",
        fixed_indices={"a": a, "b": b}, external_times={"x": 1.0, "y": 0.5},
        positions={"x": N_A, "y": N_A},
    )
    return total


@pytest.mark.parametrize("a, b", [(0, 0), (0, 1), (2, 1)])
@pytest.mark.parametrize("r_type, flags", [
    ("scalar", {"diag_R": True}),
    ("scalar", {"iso_R": True}),
    ("matrix", {}),
], ids=["scalar-diag_R", "scalar-iso_R", "matrix-default"])
def test_order_zero_response_is_the_kronecker_delta(r_type, flags, a, b):
    assert abs(_response(r_type, a, b, **flags) - float(a == b)) <= 1e-12


def test_order_zero_response_mismatches_raise():
    """Were R_01 = 1 instead of 0, and R_00 = 3 instead of 1."""
    with pytest.raises(ValueError, match="R is a scalar"):
        _response("scalar", 0, 1)
    with pytest.raises(ValueError, match="R is a matrix"):
        _response("matrix", 0, 0, iso_R=True)


# ---------------------------------------------------------------------------
# One component: no index to lose, nothing to refuse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r_type, method", [
    *(("scalar", m) for m in BACKENDS),
    ("matrix", "qmc_scalar"),
])
@pytest.mark.parametrize("flags", [{}, {"iso_R": True}], ids=["default", "iso_R"])
def test_one_component_is_not_refused(flags, r_type, method):
    """Scalar fields carry no component index, and for N = 1 the delta is 1
    and trace(R) is R's only entry, so every pairing is exact."""
    a1 = np.array([[[0.7]]])
    value = _integrate(_terms(1, **flags), r_type, method, n=1,
                       coupling={"K": (1j / 6.0) * a1}, fixed={})
    _assert_hand(value, hand=0.7)


# ---------------------------------------------------------------------------
# L1: System.expand flags that contradict the system's own R
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r_type, overrides, match", [
    ("scalar", {"iso_R": False, "diag_R": False}, "System.expand"),
    ("matrix", {"iso_R": True}, "R is a matrix"),
], ids=["scalar_R-iso_R_False_diag_R_False", "matrix_R-iso_R_True"])
def test_system_expand_overrides_that_contradict_R_raise(r_type, overrides, match):
    system = _system(r_type)
    obs = ("phi_a(x1)", "phi_b(x2)", "phi_c(z)")
    kw = dict(t_final=1.0, positions=POSITIONS, component_pair=(0, 1, 2),
              method="qmc_scalar", n_samples=N_SAMPLES, seed=1)
    _assert_hand(system.expand(obs, orders=[1]).evaluate(_props(r_type), **kw).total)
    with pytest.raises(ValueError, match=match):
        system.expand(obs, orders=[1], **overrides).evaluate(_props(r_type), **kw)
