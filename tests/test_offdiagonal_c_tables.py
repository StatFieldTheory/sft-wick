"""An off-diagonal C by quadrature: the full N x N propagator tables.

``diag_C=False`` used to require a user closed form.  The quadrature
tables held ``C_aa`` only, so ``Propagators.build`` refused the
combination, and at L0, where nothing refused it, every lookup of
``C_ab`` with ``a != b`` came back as zero.  The tables now hold every
entry.  Checked here:

* the table against an exact C at ``a != b``, at ``r != 0``, at both
  orders of the two times, and converging with the grid;
* that exact C -- the package's own closed form -- against the Markov
  embedding of the same noise, a Lyapunov equation solved with scipy,
  which shares no code with the package;
* each of the three sources of an off-diagonal C (a dense R, a
  component-mixing ``κ²``, a matrix ``σ²``) through
  ``System.propagators`` with no closed form at all;
* every C lookup: the scalar loop, the batched QMC and Gauss-Legendre
  products, ``integrate_over``, ``external_times``, the
  zero-dimensional path, ``integrate_two_point_qmc``, and the legacy
  time-only table;
* the transposition ``C_ab(t1, t2) = C_ba(t2, t1)`` that fills half of
  each table, including a kernel that does not satisfy it;
* that the off-diagonal entries change the answer, by the margin
  between the full C and the same C with its off-diagonal entries
  dropped -- which is what every lookup returned before.

Nothing here is symmetric that the physics does not force: the decay
rates differ by component, ``σ²`` has a negative off-diagonal entry,
the drift of the dense case is non-normal, the points and the times are
distinct, and ``t_min != 0``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.integrate import dblquad, solve_ivp
from scipy.linalg import expm

import sft_wick as sw
from sft_wick.evaluate import (PropagatorCache, _time_table_C_batch,
                               integrate_two_point_qmc)
from sft_wick.workflow.closed_forms import builtin_closed_form_for
from sft_wick.workflow.specs import (ConstantImpulse, CustomImpulse,
                                     GeneralKappa2)

N = 2
T_MIN = 0.4
SPAN = 1.9
GAMMA = (0.9, 1.6)          # component dependent: R is a diagonal matrix
GAMMA_ISO = 1.15            # equal rates: R is scalar (every backend)
LAM, SIGMA_T, SIGMA_X = 0.35, 0.55, 0.8
S_MIX = np.array([[0.6, -0.25], [-0.25, 0.4]])
A_DENSE = np.array([[-1.0, 2.5], [0.0, -1.7]])      # non-normal
F_TENSOR = np.array([[[0.30, -0.45], [0.20, 0.35]],
                     [[-0.40, 0.25], [0.50, -0.15]]])
X, Y = 0.0, 0.7
POS = {"x": X, "y": Y}
#: Times (offsets from ``t_min``) at which the tables are compared, in both
#: orders and including one pair on the time diagonal.
TIME_PAIRS = ((1.7, 0.3), (0.3, 1.7), (1.9, 1.1), (1.1, 1.9), (0.85, 0.85))


# =====================================================================
# The systems
# =====================================================================


def _noise(sigma2=None, lam=LAM):
    return sw.GaussianNoise(
        kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=lam, sigma_t=SIGMA_T),
            spatial=sw.GaussianSpatial(sigma_x=SIGMA_X)),
        sigma2=sigma2)


def mixing_system(gamma=GAMMA, vertices=(), sigma2=None):
    """Matrix ``σ²`` on a diagonal drift: C is dense, R is diagonal."""
    return sw.System(
        field=sw.FieldSpec("phi", N),
        linear=sw.DiagonalA(gamma=list(gamma)),
        vertices=list(vertices),
        noise=_noise(ConstantImpulse(S_MIX) if sigma2 is None else sigma2),
        t_min=T_MIN)


@dataclass(frozen=True)
class DenseR:
    """``R(t, s) = exp(A (t − s))`` for the non-normal ``A_DENSE``.

    Written out rather than called through ``scipy.linalg.expm``: the
    table build evaluates R at every quadrature node.
    """

    def __call__(self, t, s):
        if t < s:
            return np.zeros((N, N))
        a, b, c = -A_DENSE[0, 0], -A_DENSE[1, 1], A_DENSE[0, 1]
        ea, eb = np.exp(-a * (t - s)), np.exp(-b * (t - s))
        return np.array([[ea, c * (ea - eb) / (b - a)], [0.0, eb]])


def dense_system():
    """A non-normal drift: C is dense even though κ² and σ² are not."""
    return sw.System(
        field=sw.FieldSpec("phi", N),
        linear=sw.ExplicitR(R_time=DenseR(), iso_R=False),
        noise=_noise(ConstantImpulse(np.diag([0.6, 0.4]))),
        t_min=T_MIN)


@dataclass(frozen=True)
class MixingKappa2:
    """A component-mixing κ², the second source of an off-diagonal C."""

    def __call__(self, n1, t1, n2, t2):
        d = np.asarray(n1, dtype=float) - np.asarray(n2, dtype=float)
        r = float(abs(d)) if d.ndim == 0 else float(np.linalg.norm(d))
        return (LAM * np.exp(-abs(t1 - t2) / SIGMA_T
                             - r * r / (2.0 * SIGMA_X ** 2))
                * np.array([[1.0, 0.45], [0.45, 0.8]]))


@dataclass(frozen=True)
class MixingImpulse:
    """``S_MIX`` as a callable: a ``CustomImpulse``, the third source."""

    def __call__(self, n1, t, n2):
        return S_MIX


# =====================================================================
# The reference: the Markov embedding of the same noise (scipy only)
# =====================================================================


class Embedding:
    """Exact C at a set of points, from the Markov embedding of the noise.

    State ``z = (φ_{a,i}, η_{a,i})`` over components ``a`` and points ``i``:

        dφ_{a,i} = (A φ_{·,i})_a dt + η_{a,i} dt + dW_{a,i},
        dη_{a,i} = −η_{a,i}/σ_t dt + dB_{a,i},

    with ``⟨dW_a dW_b⟩ = S_ab dt`` -- the same white noise at every point,
    as a ``ConstantImpulse`` is position independent -- and
    ``⟨dB_{a,i} dB_{b,j}⟩ = δ_ab (2λ/σ_t) K(x_i − x_j) dt``, started in its
    stationary law ``⟨η_{a,i} η_{b,j}⟩ = δ_ab λ K(x_i − x_j)``; ``φ`` starts
    at 0, which is the package's ``φ(t_min) = 0``.  Then ``η`` is the
    package's stationary exponential noise and

        P(t) = E[z zᵀ] solves  Ṗ = M P + P Mᵀ + BB,
        E[z(t1) z(t2)ᵀ] = e^{M (t1 − t2)} P(t2)   for t1 ≥ t2,

    whose ``φφ`` block is C.  Imports no sft-wick code.
    """

    def __init__(self, positions, A, sigma2=None):
        self.pos = [float(x) for x in positions]
        P = len(self.pos)
        self.D = D = 2 * N * P
        self.phi = {(a, i): a * P + i for a in range(N) for i in range(P)}
        self.eta = {(a, i): N * P + a * P + i
                    for a in range(N) for i in range(P)}
        x = np.asarray(self.pos)
        K = np.exp(-(x[:, None] - x[None, :]) ** 2 / (2.0 * SIGMA_X ** 2))
        M = np.zeros((D, D))
        BB = np.zeros((D, D))
        P0 = np.zeros((D, D))
        A = np.asarray(A, dtype=float)
        for (a, i), k in self.phi.items():
            for b in range(N):
                M[k, self.phi[(b, i)]] = A[a, b]
            M[k, self.eta[(a, i)]] = 1.0
        for k in self.eta.values():
            M[k, k] = -1.0 / SIGMA_T
        if sigma2 is not None:
            s = np.asarray(sigma2, dtype=float)
            for (a, _i), k in self.phi.items():
                for (b, _j), l in self.phi.items():
                    BB[k, l] = s[a, b]
        for (a, i), k in self.eta.items():
            for (b, j), l in self.eta.items():
                if a == b:
                    BB[k, l] = (2.0 * LAM / SIGMA_T) * K[i, j]
                    P0[k, l] = LAM * K[i, j]
        self.M, self.BB, self.P0 = M, BB, P0
        self._P_cache: dict = {}

    def _P(self, u: float) -> np.ndarray:
        if u <= 0.0:
            return self.P0
        key = round(float(u), 12)
        if key not in self._P_cache:
            D = self.D
            sol = solve_ivp(
                lambda _s, y: (self.M @ y.reshape(D, D)
                               + y.reshape(D, D) @ self.M.T + self.BB).ravel(),
                (0.0, float(u)), self.P0.ravel(), rtol=1e-12, atol=1e-14)
            self._P_cache[key] = sol.y[:, -1].reshape(D, D)
        return self._P_cache[key]

    def C(self, i: int, u1: float, j: int, u2: float) -> np.ndarray:
        """``(N, N)`` block ``C_ab(x_i, t_min + u1; x_j, t_min + u2)``."""
        if u1 >= u2:
            Z = expm(self.M * (u1 - u2)) @ self._P(u2)
        else:
            Z = self._P(u1) @ expm(self.M * (u2 - u1)).T
        return np.array([[Z[self.phi[(a, i)], self.phi[(b, j)]]
                          for b in range(N)] for a in range(N)])


def _rel(got, ref) -> float:
    return float(np.max(np.abs(np.asarray(got) - np.asarray(ref)))
                 / np.max(np.abs(np.asarray(ref))))


# =====================================================================
# The references check out
# =====================================================================


def test_dense_R_matches_scipy_expm():
    R = DenseR()
    for t, s in [(1.3, 0.2), (2.0, 2.0), (0.7, 0.5)]:
        assert R(t, s) == pytest.approx(expm(A_DENSE * (t - s)),
                                        rel=1e-12, abs=1e-14)
    assert np.all(R(0.4, 1.1) == 0.0)


def test_embedding_matches_the_builtin_closed_form():
    """The reference against the package's own analytic C, which the
    table comparisons below use as the dense comparand."""
    cf = builtin_closed_form_for(mixing_system())
    emb = Embedding([X, Y], np.diag([-g for g in GAMMA]), S_MIX)
    for u1, u2 in TIME_PAIRS:
        for i, j in ((0, 0), (0, 1), (1, 0)):
            got = cf(np.asarray(emb.pos[i]), T_MIN + u1,
                     np.asarray(emb.pos[j]), T_MIN + u2)
            assert _rel(got, emb.C(i, u1, j, u2)) < 1e-8, (u1, u2, i, j)


def test_the_off_diagonal_entries_are_not_incidental():
    """The comparisons below would pass on a diagonal C if C_01 were
    small or if C_01 and C_10 agreed at every time pair."""
    emb = Embedding([X, Y], np.diag([-g for g in GAMMA]), S_MIX)
    for u1, u2 in TIME_PAIRS:
        C = emb.C(0, u1, 1, u2)
        assert abs(C[0, 1]) > 0.05 * np.max(np.abs(C)), (u1, u2)
    C = emb.C(0, 1.7, 1, 0.3)
    assert abs(C[0, 1] - C[1, 0]) > 0.1 * abs(C[0, 1])


# =====================================================================
# The tables
# =====================================================================


def _table(system, n_grid_t=41, n_gauss=12, diag_C=False, **kw):
    return system.propagators(
        t_max=T_MIN + SPAN, n_grid_t=n_grid_t, c_closed_form=None,
        diag_C=diag_C, c_method="gauss_legendre", c_n_gauss=n_gauss,
        progress=False, **kw)


def _table_C(props, u1, u2, r):
    """``C(0, t_min + u1; r, t_min + u2)`` from the table, ``(N, N)``."""
    return props.cache.C_at_batch(np.array([T_MIN + u1]),
                                  np.array([T_MIN + u2]),
                                  np.zeros(1), np.full(1, r))[0]


@pytest.fixture(scope="module")
def mixing_table():
    return _table(mixing_system()), builtin_closed_form_for(mixing_system())


@pytest.mark.parametrize("r", [0.0, Y])
def test_full_table_matches_the_closed_form(mixing_table, r):
    """Every entry, both orders of the two times, on and off the time
    diagonal.  The white noise kinks C across ``t1 = t2``; a tensor-product
    spline cannot represent that, so the pairs within one grid spacing of
    the diagonal are held to the looser tolerance -- a diagonal table has
    exactly the same error there."""
    props, cf = mixing_table
    h = SPAN / 40.0
    for u1, u2 in TIME_PAIRS:
        got = _table_C(props, u1, u2, r)
        ref = cf(np.asarray(0.0), T_MIN + u1, np.asarray(r), T_MIN + u2)
        tol = 1e-4 if abs(u1 - u2) > h else 1e-3
        assert got.shape == (N, N)
        assert _rel(got, ref) < tol, (u1, u2, r, _rel(got, ref))


def test_full_table_converges_with_the_grid():
    """Away from the kink each halving of the spacing cuts the error by
    more than 4 (measured 3.3e-05, 1.8e-07, 1.6e-08 at 11, 21 and 41
    points; the last step is already near the 12-node cell quadrature's
    own 1e-09)."""
    system = mixing_system()
    cf = builtin_closed_form_for(system)
    errs = []
    for n_grid_t in (11, 21, 41):
        props = _table(system, n_grid_t=n_grid_t)
        worst = 0.0
        for u1, u2 in ((1.7, 0.3), (0.3, 1.7), (1.9, 1.1)):
            ref = cf(np.asarray(0.0), T_MIN + u1, np.asarray(Y), T_MIN + u2)
            worst = max(worst, _rel(_table_C(props, u1, u2, Y), ref))
        errs.append(worst)
    assert errs[0] > 1e-6, f"the coarse table is already exact: {errs}"
    for coarse, fine in zip(errs, errs[1:]):
        assert fine < coarse / 4.0, errs


def test_table_agrees_with_direct_quadrature_at_both_time_orders():
    """The mirrored half of the table is the transpose of the computed
    half, so the cell at ``(t2, t1)`` must equal the direct quadrature
    there, not the value at ``(t1, t2)``."""
    props = _table(mixing_system(), n_grid_t=21)
    cache = props.cache
    ts = cache._lazy_translation.ts
    for i, j in ((3, 14), (14, 3), (9, 9)):
        direct = cache._C_value_direct(np.asarray(0.0), ts[i],
                                       np.asarray(Y), ts[j],
                                       method="gauss_legendre", n_gauss=12)
        table = cache.C_at_batch(np.array([ts[i]]), np.array([ts[j]]),
                                 np.zeros(1), np.full(1, Y))[0]
        assert _rel(table, direct) < 1e-9, (i, j)


def test_dense_drift_table_matches_the_embedding():
    """A dense R with a diagonal κ² and σ²: C is dense because R mixes the
    components, which no entrywise κ² ratio could reproduce."""
    props = _table(dense_system(), n_grid_t=31)
    emb = Embedding([X, Y], A_DENSE, np.diag([0.6, 0.4]))
    h = SPAN / 30.0
    for u1, u2 in TIME_PAIRS:
        for r, j in ((0.0, 0), (Y, 1)):
            got = _table_C(props, u1, u2, r)
            ref = emb.C(0, u1, j, u2)
            tol = 1e-3 if abs(u1 - u2) > h else 1e-2
            assert _rel(got, ref) < tol, (u1, u2, r, _rel(got, ref))
    ref = emb.C(0, 1.7, 1, 0.3)
    assert abs(ref[0, 1]) > 0.1 * np.max(np.abs(ref))
    assert abs(ref[0, 1] - ref[1, 0]) > 0.1 * abs(ref[0, 1])


def test_component_mixing_kappa2_table_and_its_swapped_key():
    """``GeneralKappa2`` puts the cache in general mode, where the table is
    keyed by the ordered pair ``(x1, x2)``.  The table at ``(y, x)`` is the
    one at ``(x, y)`` transposed, and is filled without a second build."""
    system = sw.System(
        field=sw.FieldSpec("phi", N), linear=sw.DiagonalA(gamma=list(GAMMA)),
        noise=sw.GaussianNoise(kappa2=GeneralKappa2(fn=MixingKappa2())),
        t_min=T_MIN)
    props = _table(system, n_grid_t=21)
    assert props.homogeneity == "general"
    lazy = props.cache._lazy_general
    cache = props.cache

    def _at(x1, u1, x2, u2):
        return cache.C_at_batch(np.array([T_MIN + u1]), np.array([T_MIN + u2]),
                                np.full(1, x1), np.full(1, x2))[0]

    _at(X, 1.7, Y, 0.3)
    assert lazy.n_grid_builds == 1
    assert len(lazy._splines_by_key) == 2, "the swapped key was not filled"
    for u1, u2 in ((1.7, 0.3), (0.3, 1.7)):
        forward = _at(X, u1, Y, u2)
        backward = _at(Y, u2, X, u1)
        assert lazy.n_grid_builds == 1
        assert _rel(backward, forward.T) < 1e-12, (u1, u2)
        direct = cache._C_value_direct(np.asarray(X), T_MIN + u1,
                                       np.asarray(Y), T_MIN + u2,
                                       method="gauss_legendre", n_gauss=12)
        assert _rel(forward, direct) < 1e-3, (u1, u2)
        assert abs(direct[0, 1]) > 0.05 * np.max(np.abs(direct))


def test_custom_impulse_matrix_table_matches_the_closed_form():
    """A ``CustomImpulse`` matrix is the third source, and no closed form
    is detected for it -- the table is pure quadrature."""
    system = mixing_system(sigma2=CustomImpulse(fn=MixingImpulse()))
    assert builtin_closed_form_for(system) is None
    props = _table(system, n_grid_t=41)
    cf = builtin_closed_form_for(mixing_system())     # same physics
    for u1, u2 in ((1.7, 0.3), (0.3, 1.7)):
        got = _table_C(props, u1, u2, Y)
        ref = cf(np.asarray(0.0), T_MIN + u1, np.asarray(Y), T_MIN + u2)
        assert _rel(got, ref) < 1e-4, (u1, u2)


def rotation_system():
    """The same noise on the sphere: the rotation builder's kernel."""
    return sw.System(
        field=sw.FieldSpec("phi", N), linear=sw.DiagonalA(gamma=list(GAMMA)),
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableRotation(
                temporal=sw.ExponentialTemporal(lam=LAM, sigma_t=SIGMA_T),
                angular=sw.LegendreAngular(coeffs=[0.6, 0.3, 0.1])),
            sigma2=ConstantImpulse(S_MIX)),
        t_min=T_MIN)


def _rotation_pair(cos_val):
    """The two unit vectors ``precompute_C_table_rotation`` evaluates at."""
    c = float(np.clip(cos_val, -1.0, 1.0))
    return np.array([1.0, 0.0]), np.array([c, float(np.sqrt(1.0 - c * c))])


FULL_GRID = {
    "translation": (mixing_system, dict(r_max=Y, n_grid_r=3)),
    "rotation": (rotation_system, dict(n_grid_cos=3)),
    "general": (mixing_system, dict(x_max=Y, n_grid_x=3)),
}


@pytest.mark.parametrize("mode", sorted(FULL_GRID))
def test_full_grid_builders_hold_every_entry(mode):
    """The pre-allocated-grid builders, not only the lazy caches.  A
    node of the grid is interpolated exactly, so the table there must be
    the direct quadrature -- for the cells the transposition filled as
    much as for the cells that were evaluated."""
    n_t = 7
    make, grid = FULL_GRID[mode]
    props = make().propagators(
        t_max=T_MIN + SPAN, n_grid_t=n_t, homogeneity=mode,
        c_closed_form=None, diag_C=False, c_method="gauss_legendre",
        c_n_gauss=12, progress=False, **grid)
    cache = props.cache
    ts = np.linspace(T_MIN, T_MIN + SPAN, n_t)
    if mode == "translation":
        pairs = [(np.asarray(0.0), np.asarray(r))
                 for r in (0.0, Y / 2.0, Y)]
    elif mode == "rotation":
        pairs = [_rotation_pair(c) for c in (-1.0, 0.0, 1.0)]
    else:
        xs = np.linspace(-Y, Y, 3)
        pairs = [(np.asarray(xs[p]), np.asarray(xs[q]))
                 for p, q in ((0, 2), (2, 0), (1, 1))]

    assert cache._c_transpose_ok(pairs[0][0], pairs[0][1], T_MIN + SPAN,
                                 swap=(mode == "general")), (
        f"{mode}: the transposition was refused, so every cell was "
        f"evaluated and this check says less than it looks")

    seen_off = False
    for x1, x2 in pairs:
        for i, j in ((1, 5), (5, 1), (3, 3)):
            direct = cache._C_value_direct(x1, ts[i], x2, ts[j],
                                           method="gauss_legendre",
                                           n_gauss=12)
            table = cache.C_at_batch(np.array([ts[i]]), np.array([ts[j]]),
                                     x1, x2)[0]
            assert table.shape == (N, N)
            assert _rel(table, direct) < 1e-9, (mode, x1, x2, i, j)
            seen_off = seen_off or (
                abs(direct[0, 1]) > 0.02 * np.max(np.abs(direct)))
    assert seen_off, f"{mode}: no off-diagonal entry to speak of"


# =====================================================================
# The transposition that fills half of each table
# =====================================================================


@dataclass(frozen=True)
class OneSidedTemporal:
    """``κ_t(Δt) = λ e^{−Δt/σ_t}`` for ``Δt > 0`` and ``λ e^{3 Δt/σ_t}``
    below: not even, so ``C_ab(t1, t2) != C_ba(t2, t1)`` and no cell may be
    mirrored.  (Not a covariance; the package accepts it.)"""

    def __call__(self, dt: float) -> float:
        d = float(dt)
        return LAM * (np.exp(-d / SIGMA_T) if d > 0
                      else np.exp(3.0 * d / SIGMA_T))


def _asymmetric_system():
    return sw.System(
        field=sw.FieldSpec("phi", N), linear=sw.DiagonalA(gamma=list(GAMMA)),
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.CustomKernel(fn=OneSidedTemporal()),
                spatial=sw.GaussianSpatial(sigma_x=SIGMA_X)),
            sigma2=ConstantImpulse(S_MIX)),
        t_min=T_MIN)


def test_the_half_grid_decision_follows_the_kernel():
    zero, r = np.asarray(0.0), np.asarray(Y)
    t_max = T_MIN + SPAN
    assert PropagatorCache(
        mixing_system().build_propagator_model(diag_C=False),
    )._c_half_grid(zero, r, t_max) is True
    assert PropagatorCache(
        _asymmetric_system().build_propagator_model(diag_C=False),
    )._c_half_grid(zero, r, t_max) is False


def test_an_asymmetric_kernel_gets_every_cell():
    """No mirroring, and the table still reproduces the direct quadrature
    at both orders of the two times -- where a wrongly mirrored table would
    return the other order's value."""
    props = _table(_asymmetric_system(), n_grid_t=21)
    cache = props.cache
    ts = cache._lazy_translation.ts
    direct = [cache._C_value_direct(np.asarray(0.0), ts[i], np.asarray(Y),
                                    ts[j], method="gauss_legendre",
                                    n_gauss=12)
              for i, j in ((4, 15), (15, 4))]
    assert _rel(direct[0], direct[1].T) > 1e-2, "the probe case is vacuous"
    for (i, j), ref in zip(((4, 15), (15, 4)), direct):
        table = cache.C_at_batch(np.array([ts[i]]), np.array([ts[j]]),
                                 np.zeros(1), np.full(1, Y))[0]
        assert _rel(table, ref) < 1e-9, (i, j)


# =====================================================================
# Every lookup honours the component indices
# =====================================================================


@dataclass(frozen=True)
class DiagonalOnlyC:
    """The closed form with its off-diagonal entries dropped: what every
    lookup returned before the tables held them."""

    cf: object

    def __call__(self, n1, t1, n2, t2):
        C = np.asarray(self.cf(n1, t1, n2, t2), dtype=float)
        keep = np.zeros_like(C)
        idx = np.arange(C.shape[-1])
        if C.ndim == 2:
            keep[idx, idx] = C[idx, idx]
        else:
            keep[:, idx, idx] = C[:, idx, idx]
        return keep


def _iso_system():
    """Equal decay rates, so R is scalar and every backend accepts it."""
    return mixing_system(gamma=(GAMMA_ISO, GAMMA_ISO),
                         vertices=[sw.LocalVertex("F", coupling=F_TENSOR)])


@pytest.fixture(scope="module")
def iso_props():
    """The quadrature table, the exact closed form, and the closed form
    with its off-diagonal entries dropped, for one system."""
    system = _iso_system()
    cf = builtin_closed_form_for(system)
    exact = system.propagators(t_max=T_MIN + SPAN, c_closed_form=cf,
                               c_closed_form_only=True,
                               c_closed_form_vectorized=True, diag_C=False,
                               progress=False)
    dropped = system.propagators(t_max=T_MIN + SPAN,
                                 c_closed_form=DiagonalOnlyC(cf),
                                 c_closed_form_only=True,
                                 c_closed_form_vectorized=True, diag_C=False,
                                 progress=False)
    return system, _table(system, n_grid_t=41), exact, dropped


@pytest.mark.parametrize("method,kw", [
    ("gauss_legendre", dict(n_gauss=12)),
    ("qmc_vectorized", dict(n_samples=2 ** 10, seed=5)),
    ("qmc_scalar", dict(n_samples=2 ** 9, seed=5)),
    ("qmc", dict(n_samples=2 ** 9, seed=5)),
    ("nquad", {}),
])
def test_every_backend_reads_the_off_diagonal_entries(iso_props, method, kw):
    """The order-1 tadpole ``⟨φ_a(x)⟩ = F_abc ∫ R C_bc`` reads C at
    ``b != c``.  Each backend is compared with the same expansion against a
    machine-precision closed-form C -- the same quadrature nodes, so the
    difference is the table's -- and must differ from the value the
    dropped-off-diagonal C gives."""
    system, table, exact, dropped = iso_props
    exp = system.expand(("phi_a(x)",), orders=[1], diag_C=False)
    for a in range(N):
        args = dict(positions=POS, t_final=T_MIN + SPAN, component_pair=(a,),
                    orders=[1], method=method, **kw)
        got = exp.evaluate(table, **args).total
        ref = exp.evaluate(exact, **args).total
        blind = exp.evaluate(dropped, **args).total
        assert got == pytest.approx(ref, rel=2e-3, abs=0.0), (method, a)
        assert abs(blind - ref) > 0.05 * abs(ref), (method, a, "vacuous")


@pytest.mark.parametrize("method,kw", [
    ("gauss_legendre", dict(n_gauss=10)),
    ("qmc_vectorized", dict(n_samples=2 ** 10, seed=5)),
    ("qmc_scalar", dict(n_samples=2 ** 8, seed=5)),
])
@pytest.mark.parametrize("ab", [(0, 1), (1, 1)])
def test_order_two_through_the_table(iso_props, method, kw, ab):
    system, table, exact, dropped = iso_props
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2], diag_C=False)
    args = dict(positions=POS, t_final=T_MIN + SPAN, component_pair=ab,
                orders=[2], method=method, **kw)
    got = exp.evaluate(table, **args).total
    ref = exp.evaluate(exact, **args).total
    blind = exp.evaluate(dropped, **args).total
    assert got == pytest.approx(ref, rel=5e-3, abs=0.0), (method, ab)
    assert abs(blind - ref) > 0.05 * abs(ref), (method, ab, "vacuous")


def test_order_zero_at_unequal_external_times(mixing_table):
    """The zero-dimensional path at ``t_x != t_y``, for both orders of the
    component pair.  With component-dependent decay rates ``C_01(t1, t2)``
    and ``C_10(t1, t2)`` differ by 63 %, so the mirrored half of the table
    is only right if the component pair is transposed with the times."""
    props, cf = mixing_table
    exp = mixing_system().expand(("phi_a(x)", "phi_b(y)"), orders=[0],
                                 diag_C=False)
    times = {"x": T_MIN + 1.7, "y": T_MIN + 0.3}
    ref = cf(np.asarray(X), times["x"], np.asarray(Y), times["y"])
    for ab in ((0, 1), (1, 0), (1, 1)):
        got = exp.evaluate(props, positions=POS, t_final=T_MIN + SPAN,
                           external_times=times, component_pair=ab,
                           orders=[0], method="gauss_legendre").total
        assert got == pytest.approx(ref[ab], rel=1e-4, abs=0.0), ab
    assert abs(ref[0, 1] - ref[1, 0]) > 0.1 * abs(ref[0, 1])


@pytest.mark.parametrize("integrate_over", ["all", ["x"]])
def test_integrate_over_sweeps_both_sides_of_the_diagonal(mixing_table,
                                                          integrate_over):
    """An integrated external time makes the lookups sweep the square,
    both sides of the time diagonal.  The table is compared with the same
    expansion against a machine-precision C at the same nodes -- that
    difference is the table's -- and, for the one-sided integral, with the
    1-D quadrature of the exact C."""
    from scipy.integrate import quad

    props, cf = mixing_table
    system = mixing_system()
    exact = system.propagators(t_max=T_MIN + SPAN, c_closed_form=cf,
                               c_closed_form_only=True,
                               c_closed_form_vectorized=True, diag_C=False,
                               progress=False)
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0], diag_C=False)
    t_f = T_MIN + SPAN
    refs = {}
    for ab in ((0, 1), (1, 0)):
        args = dict(positions=POS, t_final=t_f, component_pair=ab,
                    orders=[0], integrate_over=integrate_over,
                    method="gauss_legendre", n_gauss=24)
        got = exp.evaluate(props, **args).total
        assert got == pytest.approx(exp.evaluate(exact, **args).total,
                                    rel=1e-3, abs=0.0), ab
        if integrate_over == "all":
            # Both external times integrated: C_01 and C_10 have the same
            # integral over the symmetric square, so only the value is
            # checked here.  The white-noise kink then sits inside the
            # domain and the tensor-product rule converges algebraically on
            # it (2.6e-03 at 24 nodes against the 2-D adaptive quadrature,
            # 3.8e-04 at 64), which is why this one is not compared with
            # dblquad.
            continue
        refs[ab], _err = quad(
            lambda t, ab=ab: float(cf(np.asarray(X), t, np.asarray(Y),
                                      t_f)[ab]),
            T_MIN, t_f, epsabs=1e-12)
        assert got == pytest.approx(refs[ab], rel=1e-3, abs=0.0), ab
    if refs:
        assert abs(refs[(0, 1)] - refs[(1, 0)]) > 0.1 * abs(refs[(0, 1)])


def test_two_point_qmc_reads_the_off_diagonal_entries(iso_props):
    """``integrate_two_point_qmc`` is the one entry point with its own C
    lookup."""
    system, table, _exact, _dropped = iso_props
    cf = builtin_closed_form_for(system)
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0], diag_C=False)
    cv = system.build_coupling_values()
    t_f = T_MIN + SPAN
    ref = cf(np.asarray(X), t_f, np.asarray(Y), t_f)
    for ab in ((0, 1), (1, 1)):
        igs = [dt.build_integrand(cv, {"a": ab[0], "b": ab[1]})
               for dt in exp.diagrams(0)]
        got, _err = integrate_two_point_qmc(igs, t_f, POS, table.cache,
                                            t_min=T_MIN, n_samples=2 ** 8,
                                            seed=1)
        assert got == pytest.approx(ref[ab], rel=1e-4, abs=0.0), ab


# =====================================================================
# The legacy time-only table
# =====================================================================


def _legacy_cache(diag_C=False, n_grid=21):
    cache = PropagatorCache(
        mixing_system().build_propagator_model(diag_C=diag_C),
        c_method="gauss_legendre", n_gauss=12)
    cache.precompute_C_table(t_max=T_MIN + SPAN, n_grid=n_grid, direction=0.0)
    return cache


def test_legacy_time_table_holds_every_entry():
    """``precompute_C_table`` stores every ``C_ab`` under ``diag_C=False``,
    and ``C_matrix_batch`` is what the integrators read it through."""
    cache = _legacy_cache()
    cf = builtin_closed_form_for(mixing_system())
    t1 = np.array([T_MIN + 1.7, T_MIN + 0.3])
    t2 = np.array([T_MIN + 0.3, T_MIN + 1.7])
    got = _time_table_C_batch(cache, t1, t2)
    assert got.shape == (2, N, N)
    for k in range(2):
        ref = cf(np.asarray(0.0), t1[k], np.asarray(0.0), t2[k])
        assert _rel(got[k], ref) < 1e-3, k
    diag = cache.C_diagonal_batch(t1, t2)
    assert diag.shape == (2, N)
    assert diag == pytest.approx(np.einsum("iaa->ia", got), rel=1e-12,
                                 abs=0.0)
    assert _rel(cache.C_value(0.0, t1[0], 0.0, t2[0]), got[0]) < 1e-12


def test_a_diagonal_only_batch_cache_is_refused():
    """A cache whose model keeps the off-diagonal entries but which can
    only return diagonals must say so, not drop them."""

    class DiagOnly:
        model = mixing_system().build_propagator_model(diag_C=False)

        def C_diagonal_batch(self, t1, t2):
            return np.zeros((len(t1), N))

    with pytest.raises(ValueError, match="C_matrix_batch"):
        _time_table_C_batch(DiagOnly(), np.array([1.0]), np.array([1.0]))


def test_two_point_qmc_refuses_a_time_only_table_at_distinct_positions():
    """The kappa2 ratio that carries a time-only table to a separation is
    defined per diagonal component; with an off-diagonal C it is refused
    rather than applied to the diagonal alone."""
    system = _iso_system()
    cache = PropagatorCache(system.build_propagator_model(diag_C=False),
                            c_method="gauss_legendre", n_gauss=12)
    cache.precompute_C_table(t_max=T_MIN + SPAN, n_grid=15, direction=0.0)
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2], diag_C=False)
    cv = system.build_coupling_values()
    igs = [dt.build_integrand(cv, {"a": 0, "b": 1}) for dt in exp.diagrams(2)]
    with pytest.raises(ValueError, match="no spatial table"):
        integrate_two_point_qmc(igs, T_MIN + SPAN, POS, cache, t_min=T_MIN,
                                n_samples=2 ** 6, seed=1)


# =====================================================================
# The L1 guards
# =====================================================================


SOURCES = {
    "dense R": dense_system,
    "component-mixing kappa2": lambda: sw.System(
        field=sw.FieldSpec("phi", N), linear=sw.DiagonalA(gamma=list(GAMMA)),
        noise=sw.GaussianNoise(kappa2=GeneralKappa2(fn=MixingKappa2())),
        t_min=T_MIN),
    "ConstantImpulse matrix": mixing_system,
    "CustomImpulse matrix": lambda: mixing_system(
        sigma2=CustomImpulse(fn=MixingImpulse())),
}


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_propagators_need_no_closed_form(source):
    """Each source of an off-diagonal C builds its tables by quadrature,
    and they carry the off-diagonal entries."""
    system = SOURCES[source]()
    props = _table(system, n_grid_t=9)
    assert props.c_source.startswith("quadrature")
    C = _table_C(props, 1.6, 0.5, Y)
    assert C.shape == (N, N)
    assert abs(C[0, 1]) > 0.02 * np.max(np.abs(C)), source


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_diag_C_true_is_still_refused(source):
    system = SOURCES[source]()
    with pytest.raises(ValueError, match="diag_C=False"):
        system.propagators(t_max=T_MIN + SPAN, n_grid_t=5, progress=False)
    with pytest.raises(ValueError, match="diag_C=False"):
        system.expand(("phi_a(x)", "phi_b(y)"), orders=[0], diag_R=False)


def test_closed_form_only_still_bypasses_the_tables():
    """The route the refusal used to point to still works, and agrees with
    the tables it no longer requires."""
    system = mixing_system()
    cf = builtin_closed_form_for(system)
    props = system.propagators(t_max=T_MIN + SPAN, c_closed_form=cf,
                               c_closed_form_only=True, diag_C=False,
                               progress=False)
    assert props.c_source == "closed_form:user"
    for u1, u2 in ((1.7, 0.3), (0.3, 1.7)):
        got = _table_C(props, u1, u2, Y)
        ref = cf(np.asarray(0.0), T_MIN + u1, np.asarray(Y), T_MIN + u2)
        assert _rel(got, ref) < 1e-12, (u1, u2)
