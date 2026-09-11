r"""Matrix-valued R on the batched integrators; an absorbed R with a matrix R.

``gauss_legendre`` and ``qmc_vectorized`` raised ``NotImplementedError`` for a
matrix-valued R (``DiagonalA`` with component-dependent rates,
``ExplicitR(iso_R=False)``); only the scalar loops (``qmc_scalar``, ``qmc``)
and ``nquad`` evaluated one.  They now select R's component entries per index
assignment over the whole batch of samples, as they do for C.
``integrate_two_point_qmc`` raised ``ValueError: setting an array element
with a sequence`` for a matrix R and returned 0 for a callable coupling.

An R absorbed into an ``already_R_contracted`` vertex stands in for the
Kronecker delta between its partner's component and its leg's.  Only
``diag_R=True`` / ``iso_R=True`` applied that delta, and they apply it to
every R, which is wrong for a non-diagonal R; without them the R-cache guard
refused the terms.  ``compute_moment`` now applies it to the absorbed R alone.

* **MB0** ``PropagatorCache.R_matrix_batch``: strict Θ, one ``R_time`` call
  per distinct causal pair, shape check.
* **MB1** L1, ``DiagonalA`` with rates (0.6, 1.6), a local F with no index
  symmetry, coloured and white noise, ``t_min = 0.5``, distinct points,
  orders 0-2 and off-diagonal component tuples, against the Itô moment
  hierarchy of the Markov embedding.
* **MB2** L1, a non-normal dense drift (``ExplicitR``; every entry of R is
  non-zero and R is not symmetric), white noise, ``t_min = 0.4``:
  Gauss-Legendre against the hierarchy; ``qmc_vectorized`` against
  ``qmc_scalar`` on the same Sobol points, also with unequal external times
  and ``integrate_over``; Gauss-Legendre against ``nquad``.
* **MB3** a static κ³ with the dense R, the order-1 three-point function at
  unequal times: raw and ``already_R_contracted`` against the closed form
  ``Σ κ_ijk M_ai(t_x) M_bj(t_y) M_ck(t_z)``, ``M(t) = A⁻¹(1 − e^{−A(t −
  t_min)})``, on every backend, and with ``integrate_over='all'``.
* **MB4** the FK channel of ``<φ_a φ_b>`` (one F, the static κ³) with the
  dense R: raw and ``already_R_contracted`` against the hierarchy with a
  constant random source ξ whose third cumulant is κ³.  The L0 route that
  ran ``already_R_contracted`` before (``diag_R=True``) is wrong for this R.
* **MB5** a callable κ³ with no index symmetry and leg-dependent time
  dependence (the leg-order symbols ``K@0 …``, a prop-indexed dynamic
  coupling), both callable contracts, against a scipy hand contraction.
* **MB6** an ``equal_time`` κ³ against a scipy hand contraction.
* **MB7** ``integrate_two_point_qmc`` with a matrix R and with a callable.
* **MB8** ``integrate_moment(method='qmc')`` routes a matrix R with a
  batch-capable cache to the batched integrator.

The references (``examples/reference/ito_moments.py``, the closed form,
``scipy.linalg.expm`` + ``scipy.integrate.quad_vec``) share no code with the
package.  On the code before this change every batched call on a matrix R
raises ``NotImplementedError``, every ``already_R_contracted`` term built with
the default flags raises the R-cache guard's ``ValueError``,
``integrate_two_point_qmc`` raises for a matrix R and returns 0 for a
callable, and ``method='qmc'`` takes the scalar loop.
"""
from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.linalg import expm

import sft_wick as sw
from sft_wick.evaluate import (
    DiagramIntegrand,
    PropagatorCache,
    PropagatorModel,
    integrate_diagrams,
    integrate_moment,
    integrate_two_point_qmc,
)
from sft_wick.workflow.expansion import _collect_symbol_names
from sft_wick.workflow.specs import ConstantImpulse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "examples" / "reference"))
import ito_moments as im  # noqa: E402

N = 2
T = 1.3                          # observation time above t_min
F_TENSOR = np.array([[[0.25, -0.40], [0.15, 0.30]],
                     [[-0.20, 0.10], [0.35, -0.45]]])   # no index symmetry
S_WHITE = np.array([[0.50, 0.20], [0.20, 0.35]])        # white noise, mixing
POS = {"x": 0.0, "y": 0.7, "z": -0.4}
PAIRS = [(0, 1), (1, 0), (1, 1)]
OBS2 = ("phi_a(x)", "phi_b(y)")
OBS3 = ("phi_a(x)", "phi_b(y)", "phi_c(z)")

# MB1: DiagonalA with distinct rates, coloured + white noise.
GAMMAS = (0.6, 1.6)
LAM_C, SIGMA_T, SIGMA_X = 0.4, 0.6, 0.9
T_MIN_RATES = 0.5

# MB2-MB8: dφ = −A φ dt + ...; A non-normal, every entry non-zero, real
# distinct eigenvalues (0.256, 1.644).
A_DENSE = np.array([[1.3, -0.9], [-0.4, 0.6]])
T_MIN = 0.4
T_F = T_MIN + T
EXT_TIMES = {"x": T_F, "y": T_F - 0.35, "z": T_F - 0.8}
_EIG, _VEC = np.linalg.eig(A_DENSE)
assert not np.iscomplexobj(_EIG)
_VEC_INV = np.linalg.inv(_VEC)
_W = _VEC_INV @ S_WHITE @ _VEC_INV.T
_LSUM = _EIG[:, None] + _EIG[None, :]

#: A static third cumulant: symmetric in its indices, entries of both signs.
_RAW3 = np.random.default_rng(20260911).normal(size=(N, N, N))
KAPPA3 = sum(_RAW3.transpose(p) for p in itertools.permutations(range(3))) / 6
#: MB5's callable κ³: no index symmetry, a different time slope per leg.
B3 = np.random.default_rng(7).normal(size=(N, N, N))
LEG_SLOPES = np.array([0.3, -0.2, 0.15])


# --------------------------------------------------------------------------- #
# The dense system's inputs (fed to the package; eigenbasis of A)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DenseR:
    """``R(t, s) = e^{−A (t − s)}`` for ``t ≥ s``, 0 otherwise."""

    def __call__(self, t, s):
        if t < s:
            return np.zeros((N, N))
        return (_VEC * np.exp(-_EIG * (t - s))) @ _VEC_INV


@dataclass(frozen=True)
class DenseWhiteC:
    """``C(t1, t2) = ∫_{t_min}^{min(t1,t2)} R(t1 − s) S R(t2 − s)ᵀ ds``, the
    exact C of ``dφ = −Aφ dt + dW``, ``⟨dW dWᵀ⟩ = S dt``, ``φ(t_min) = 0``;
    the vectorised ``(n1, t1, n2, t2)`` contract, positions unused."""

    t_min: float = T_MIN

    def __call__(self, n1, t1, n2, t2):
        t1a, t2a = np.asarray(t1, dtype=float), np.asarray(t2, dtype=float)
        scalar = t1a.ndim == 0 and t2a.ndim == 0
        t1v, t2v = np.broadcast_arrays(np.atleast_1d(t1a), np.atleast_1d(t2a))
        m = np.minimum(t1v, t2v)
        tau = np.maximum(m - self.t_min, 0.0)
        e1 = np.exp(-np.multiply.outer(t1v - m, _EIG))              # (n, N)
        e2 = np.exp(-np.multiply.outer(t2v - m, _EIG))
        g = -np.expm1(-_LSUM[None] * tau[:, None, None]) / _LSUM[None]
        inner = _W[None] * g * e1[:, :, None] * e2[:, None, :]
        C = np.einsum("ai,nij,bj->nab", _VEC, inner, _VEC)
        return C[0] if scalar else C


def _M_eig(t) -> np.ndarray:
    """``M(t) = A⁻¹(1 − e^{−A(t − t_min)})`` at an array of times,
    ``(n, N, N)``."""
    tau = np.asarray(t, dtype=float) - T_MIN
    d = -np.expm1(-np.multiply.outer(tau, _EIG)) / _EIG
    return np.einsum("ai,ni,ib->nab", _VEC, d, _VEC_INV)


@dataclass(frozen=True)
class KappaR:
    """The static κ³ contracted with the dense R, in the partners'
    components: ``K^R_{abc}(t0, t1, t2) = Σ κ_ijk M_ai(t0) M_bj(t1) M_ck(t2)``
    (``already_R_contracted``, per-sample contract)."""

    def __call__(self, n_list, t_list):
        Ms = _M_eig(t_list)
        return np.einsum("ijk,ai,bj,ck->abc", KAPPA3, Ms[0], Ms[1], Ms[2])


@dataclass(frozen=True)
class KappaT:
    """``κ_ijk(t0, t1, t2) = B_ijk Π_l (1 + c_l t_l)``, per-sample contract."""

    def __call__(self, n_list, t_list):
        return B3 * float(np.prod(1.0 + LEG_SLOPES * np.asarray(t_list)))


@dataclass(frozen=True)
class KappaTBatch:
    """:class:`KappaT` under the batched contract (``coupling_vectorized``)."""

    def __call__(self, n_arr, t_arr):
        w = np.prod(1.0 + LEG_SLOPES[:, None] * np.asarray(t_arr), axis=0)
        return w[:, None, None, None] * B3[None]


# --------------------------------------------------------------------------- #
# References (no sft-wick code)
# --------------------------------------------------------------------------- #
def _M_ref(t: float) -> np.ndarray:
    return np.linalg.solve(A_DENSE, np.eye(N) - expm(-A_DENSE * (t - T_MIN)))


def _M_int_ref(t_top: float) -> np.ndarray:
    """``∫_{t_min}^{t_top} M(t) dt = A⁻¹((t_top − t_min) 1 − M(t_top))``."""
    return np.linalg.solve(A_DENSE, (t_top - T_MIN) * np.eye(N) - _M_ref(t_top))


def _three_point_exact(comps, Ms) -> float:
    """``Σ κ_ijk M⁽ˣ⁾_ai M⁽ʸ⁾_bj M⁽ᶻ⁾_ck``: the third cumulant of φ = M ξ."""
    a, b, c = comps
    return float(np.einsum("ijk,i,j,k->", KAPPA3, Ms[0][a], Ms[1][b],
                           Ms[2][c]))


@lru_cache(maxsize=None)
def _dense_moment(comps: tuple, order: int) -> float:
    """``E[Π φ]`` at tag ``F^order`` for ``dφ = (−Aφ + F φφ) dt + dW``."""
    sde = im.PolySDE(D=N, n_tags=1)
    sde.add_linear_drift(-A_DENSE)
    sde.add_quadratic_drift(F_TENSOR, tag=(1,))
    sde.add_constant_diffusion(S_WHITE)
    mono = im.unit(N, *comps)
    return float(im.solve(sde, [(mono, (order,))], [T])[(mono, (order,))][0])


@lru_cache(maxsize=None)
def _fk_moment(comps: tuple) -> float:
    """``E[φ_a φ_b]`` at tag ``(F, κ³)`` for ``dφ = (−Aφ + ξ + F φφ) dt`` with
    ξ a constant random vector whose only cumulant is κ³ (the FK channel
    sees no other)."""
    D = 2 * N
    sde = im.PolySDE(D=D, n_tags=2)
    drift = np.zeros((D, D))
    drift[:N, :N] = -A_DENSE
    drift[:N, N:] = np.eye(N)
    sde.add_linear_drift(drift)
    Q = np.zeros((D, D, D))
    Q[:N, :N, :N] = F_TENSOR
    sde.add_quadratic_drift(Q, tag=(1, 0))

    def initial(alpha, tag):
        if any(alpha[:N]):                      # φ(t_min) = 0
            return 0.0
        xi = alpha[N:]
        if tag == (0, 0):
            return 0.0 if any(xi) else 1.0
        if tag == (0, 1) and sum(xi) == 3:
            return float(KAPPA3[tuple(i for i in range(N)
                                      for _ in range(xi[i]))])
        return 0.0

    mono = im.unit(D, *comps)
    return float(im.solve(sde, [(mono, (1, 1))], [T],
                          initial=initial)[(mono, (1, 1))][0])


class _RatesHierarchy:
    """MB1's system at the points ``positions``: ``φ_{a,i}`` and the
    stationary Ornstein-Uhlenbeck noise ``η_{a,i}`` (Markov embedding), with
    the white noise shared by every point."""

    def __init__(self, positions):
        x = np.asarray(positions, dtype=float)
        P = len(x)
        self.D = D = 2 * N * P
        self.phi = {(a, i): a * P + i for a in range(N) for i in range(P)}
        self.eta = {(a, i): N * P + a * P + i
                    for a in range(N) for i in range(P)}
        self.K = np.exp(-(x[:, None] - x[None, :]) ** 2 / (2 * SIGMA_X ** 2))
        A, S = np.zeros((D, D)), np.zeros((D, D))
        Q = np.zeros((D, D, D))
        for (a, i), k in self.phi.items():
            A[k, k] = -GAMMAS[a]
            A[k, self.eta[(a, i)]] = 1.0
            for (b, _j), l in self.phi.items():
                S[k, l] = S_WHITE[a, b]
        for (a, i), k in self.eta.items():
            A[k, k] = -1.0 / SIGMA_T
            for (b, j), l in self.eta.items():
                if a == b:
                    S[k, l] = 2.0 * LAM_C / SIGMA_T * self.K[i, j]
        for i in range(P):
            for a, b, c in itertools.product(range(N), repeat=3):
                Q[self.phi[(a, i)], self.phi[(b, i)],
                  self.phi[(c, i)]] += F_TENSOR[a, b, c]
        self.sde = im.PolySDE(D=D, n_tags=1)
        self.sde.add_linear_drift(A)
        self.sde.add_constant_diffusion(S)
        self.sde.add_quadratic_drift(Q, tag=(1,))

    def _initial(self, alpha, tag) -> float:
        if tag != (0,) or any(alpha[k] for k in self.phi.values()):
            return 0.0
        legs = [key for key, k in self.eta.items() for _ in range(alpha[k])]

        def cumulant(idx):
            if len(idx) != 2:
                return 0.0
            (a, i), (b, j) = legs[idx[0]], legs[idx[1]]
            return LAM_C * self.K[i, j] if a == b else 0.0

        return im.moments_from_cumulants(cumulant, list(range(len(legs))))

    def moment(self, legs, order: int) -> float:
        mono = im.monomial_of(self.D, [self.phi[leg] for leg in legs])
        out = im.solve(self.sde, [(mono, (order,))], [T],
                       initial=self._initial)
        return float(out[(mono, (order,))][0])


def _Q_ref(t_e: float, slope: float) -> np.ndarray:
    """``∫_{t_min}^{t_e} e^{−A(t_e − s)} (1 + c s) ds``."""
    val, _ = quad_vec(lambda s: expm(-A_DENSE * (t_e - s)) * (1.0 + slope * s),
                      T_MIN, t_e, epsabs=1e-15, epsrel=1e-13)
    return val


def _mb5_hand(comps, times) -> float:
    """Order-1 ``<φφφ>`` of :class:`KappaT`: leg ``l`` pairs with external
    ``β(l)`` for every bijection β, ``<φ_c(e) ψ_i(u)> = −i R_ci(t_e, t_u)``,
    overall factor ``(−1)(−i)³(i/6) = 1/6``."""
    ext = [(comps[0], times["x"]), (comps[1], times["y"]),
           (comps[2], times["z"])]
    total = 0.0
    for beta in itertools.permutations(range(3)):
        rows = [_Q_ref(ext[beta[l]][1], LEG_SLOPES[l])[ext[beta[l]][0]]
                for l in range(3)]
        total += float(np.einsum("ijk,i,j,k->", B3, *rows))
    return total / 6.0


def _mb6_hand(comps, times) -> float:
    """``equal_time`` κ³: ``Σ κ_ijk ∫_{t_min}^{min t} R_ai(t_x, s) R_bj(t_y,
    s) R_ck(t_z, s) ds``."""
    a, b, c = comps

    def f(s):
        return np.einsum("ijk,i,j,k->", KAPPA3,
                         expm(-A_DENSE * (times["x"] - s))[a],
                         expm(-A_DENSE * (times["y"] - s))[b],
                         expm(-A_DENSE * (times["z"] - s))[c])

    val, _ = quad_vec(f, T_MIN, min(times.values()), epsabs=1e-15,
                      epsrel=1e-13)
    return float(val)


# --------------------------------------------------------------------------- #
# Systems
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _rates():
    system = sw.System(
        field=sw.FieldSpec("phi", N), linear=sw.DiagonalA(gamma=list(GAMMAS)),
        vertices=[sw.LocalVertex("F", coupling=F_TENSOR)],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=LAM_C, sigma_t=SIGMA_T),
                spatial=sw.GaussianSpatial(sigma_x=SIGMA_X)),
            sigma2=ConstantImpulse(S_WHITE)),
        t_min=T_MIN_RATES)
    props = system.propagators(
        t_max=T_MIN_RATES + T + 0.3, c_closed_form="auto",
        c_closed_form_only=True, c_closed_form_vectorized=True,
        diag_C=False, progress=False)
    return system, props


@lru_cache(maxsize=None)
def _rates_expansion(obs, order):
    return _rates()[0].expand(obs, orders=[order], diag_C=False)


_K_VERTICES = {
    "raw": lambda: sw.NonLocalVertex("K", order=3, coupling=KAPPA3),
    "absorbed": lambda: sw.NonLocalVertex(
        "K", order=3, coupling=KappaR(), already_R_contracted=True),
    "equal_time": lambda: sw.NonLocalVertex(
        "K", order=3, coupling=KAPPA3, equal_time=True),
    "callable": lambda: sw.NonLocalVertex("K", order=3, coupling=KappaT()),
    "callable_batched": lambda: sw.NonLocalVertex(
        "K", order=3, coupling=KappaTBatch(), coupling_vectorized=True),
}


@lru_cache(maxsize=None)
def _dense(key: str):
    """The dense-R system; ``key`` lists its vertices, e.g. ``"F+raw"``."""
    parts = key.split("+")
    system = sw.System(
        field=sw.FieldSpec("phi", N),
        linear=sw.ExplicitR(R_time=DenseR(), iso_R=False),
        vertices=([sw.LocalVertex("F", coupling=F_TENSOR)]
                  if "F" in parts else []),
        nonlocal_vertices=[_K_VERTICES[p]() for p in parts if p != "F"],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.0, sigma_t=1.0),
                spatial=sw.ExponentialSpatial(sigma_x=1.0)),
            sigma2=ConstantImpulse(S_WHITE)),
        t_min=T_MIN)
    props = system.propagators(
        t_max=T_F + 0.2, c_closed_form=DenseWhiteC(), c_closed_form_only=True,
        c_closed_form_vectorized=True, diag_C=False, progress=False)
    return system, props


@lru_cache(maxsize=None)
def _dense_expansion(key: str, obs: tuple, order: int):
    return _dense(key)[0].expand(obs, orders=[order], diag_R=False,
                                 diag_C=False)


def _dense_eval(key, obs, order, comps, method, **kw):
    kw.setdefault("positions", POS)
    return _dense_expansion(key, obs, order).evaluate(
        _dense(key)[1], t_final=T_F, component_pair=comps, orders=[order],
        method=method, **kw)


def _assert_per_diagram(got, want, rel):
    """Per-diagram agreement, relative to the largest diagram."""
    g = [d["value"] for d in got.per_diagram]
    w = [d["value"] for d in want.per_diagram]
    assert len(g) == len(w) > 0
    scale = max(abs(v) for v in w)
    assert scale > 0.0
    worst = max(abs(x - y) for x, y in zip(g, w)) / scale
    assert worst <= rel, f"worst per-diagram difference {worst:.2e}"


# --------------------------------------------------------------------------- #
# MB0: R_matrix_batch
# --------------------------------------------------------------------------- #
class _CountingR:
    def __init__(self):
        self.calls = []

    def __call__(self, t, s):
        self.calls.append((t, s))
        return np.array([[t, -s], [0.5 * t * s, 1.0 + t]])


def _bare_cache(R_time) -> PropagatorCache:
    return PropagatorCache(PropagatorModel(
        R_time=R_time, kappa2=lambda n1, t1, n2, t2: np.eye(N),
        n_components=N, iso_R=False))


def test_MB0_R_matrix_batch_theta_and_distinct_pairs():
    R = _CountingR()
    t1 = np.array([1.0, 1.0, 0.5, 0.7, 1.0, 0.8])
    t2 = np.array([0.2, 0.2, 0.5, 0.9, 0.3, 0.2])
    out = _bare_cache(R).R_matrix_batch(t1, t2)
    assert out.shape == (6, N, N)
    # Each distinct causal pair once; equal and acausal times never.
    assert sorted(R.calls) == [(0.8, 0.2), (1.0, 0.2), (1.0, 0.3)]
    ref = _CountingR()
    for s, want in [(0, ref(1.0, 0.2)), (1, ref(1.0, 0.2)),
                    (4, ref(1.0, 0.3)), (5, ref(0.8, 0.2))]:
        np.testing.assert_array_equal(out[s], want)
    assert not out[2].any() and not out[3].any()


def test_MB0_R_matrix_batch_refuses_a_scalar_R_time():
    with pytest.raises(ValueError, match=r"\(2, 2\) array"):
        _bare_cache(lambda t, s: 1.0).R_matrix_batch(np.array([1.0]),
                                                    np.array([0.0]))


# --------------------------------------------------------------------------- #
# MB1: DiagonalA with distinct rates against the hierarchy
# --------------------------------------------------------------------------- #
MB1_CASES = (
    [(OBS2, 0, c) for c in PAIRS] + [(OBS2, 2, c) for c in PAIRS]
    + [(("phi_a(x)",), 1, (a,)) for a in range(N)]
    + [(OBS3, 1, (0, 1, 1))]
)


def _labels(obs):
    return [o[o.index("(") + 1:-1] for o in obs]


@pytest.mark.parametrize("obs, order, comps", MB1_CASES)
def test_MB1_distinct_rates_gauss_legendre_matches_hierarchy(obs, order,
                                                             comps):
    system, props = _rates()
    assert props.cache.model.iso_R is False
    ref = _RatesHierarchy([POS[lab] for lab in _labels(obs)]).moment(
        [(c, i) for i, c in enumerate(comps)], order)
    got = _rates_expansion(obs, order).evaluate(
        props, positions=POS, t_final=T_MIN_RATES + T, component_pair=comps,
        orders=[order], method="gauss_legendre", n_gauss=12).total
    assert got == pytest.approx(ref, rel=1e-10, abs=0.0)


def test_MB1_distinct_rates_qmc_vectorized():
    """The QMC error at 2^13 samples, and the scalar loop on the same Sobol
    points to round-off."""
    _, props = _rates()
    exp = _rates_expansion(OBS2, 2)
    kw = dict(positions=POS, t_final=T_MIN_RATES + T, component_pair=(0, 1),
              orders=[2], seed=11)
    ref = _RatesHierarchy([POS["x"], POS["y"]]).moment([(0, 0), (1, 1)], 2)
    big = exp.evaluate(props, method="qmc_vectorized", n_samples=2 ** 13,
                       **kw).total
    assert big == pytest.approx(ref, rel=2e-3, abs=0.0)
    _assert_per_diagram(
        exp.evaluate(props, method="qmc_vectorized", n_samples=2 ** 7, **kw),
        exp.evaluate(props, method="qmc_scalar", n_samples=2 ** 7, **kw),
        rel=1e-12)


# --------------------------------------------------------------------------- #
# MB2: a non-normal dense R
# --------------------------------------------------------------------------- #
MB2_CASES = (
    [(OBS2, k, c) for k in (0, 2) for c in PAIRS]
    + [(("phi_a(x)",), 1, (a,)) for a in range(N)]
    + [(OBS3, 1, (1, 0, 1))]
)


@pytest.mark.parametrize("obs, order, comps", MB2_CASES)
def test_MB2_dense_R_gauss_legendre_matches_hierarchy(obs, order, comps):
    got = _dense_eval("F", obs, order, comps, "gauss_legendre",
                      n_gauss=12).total
    assert got == pytest.approx(_dense_moment(comps, order), rel=1e-10,
                                abs=0.0)


@pytest.mark.parametrize("kw", [
    {},
    {"external_times": {"x": T_F, "y": T_F - 0.3}},
    {"integrate_over": {"x"}},
], ids=["equal_times", "external_times", "integrate_over_x"])
def test_MB2_dense_R_batched_equals_scalar_loop(kw):
    """Same Sobol points: every order-2 diagram to round-off."""
    common = dict(n_samples=2 ** 6, seed=3, **kw)
    _assert_per_diagram(
        _dense_eval("F", OBS2, 2, (0, 1), "qmc_vectorized", **common),
        _dense_eval("F", OBS2, 2, (0, 1), "qmc_scalar", **common), rel=1e-12)


@pytest.mark.parametrize("obs, comps, kw", [
    (("phi_a(x)",), (1,), {}),
    (("phi_a(x)",), (0,), {"integrate_over": "all"}),
    (OBS3, (1, 0, 1), {}),
], ids=["tadpole", "tadpole-integrate_over_all", "three_point"])
def test_MB2_dense_R_gauss_legendre_equals_nquad(obs, comps, kw):
    _assert_per_diagram(
        _dense_eval("F", obs, 1, comps, "gauss_legendre", n_gauss=16, **kw),
        _dense_eval("F", obs, 1, comps, "nquad", **kw), rel=1e-12)


# --------------------------------------------------------------------------- #
# MB3: static κ³, order-1 three-point function, raw and already_R_contracted
# --------------------------------------------------------------------------- #
TRIPLES = list(itertools.product(range(N), repeat=3))
M_EXT = [_M_ref(EXT_TIMES[lab]) for lab in ("x", "y", "z")]


@pytest.mark.parametrize("kind, method, kw, rel", [
    ("absorbed", "gauss_legendre", {}, 1e-12),
    ("absorbed", "qmc_vectorized", {"n_samples": 2 ** 4}, 1e-12),
    ("absorbed", "qmc_scalar", {"n_samples": 2 ** 4}, 1e-12),
    ("absorbed", "nquad", {}, 1e-12),
    ("raw", "gauss_legendre", {"n_gauss": 10}, 1e-11),
])
def test_MB3_three_point_matches_closed_form(kind, method, kw, rel):
    """Every component triple at unequal times.  ``absorbed`` is
    zero-dimensional (each leg sits at a fixed partner); before the change
    its default-flag terms were refused, and ``raw`` on Gauss-Legendre
    raised NotImplementedError."""
    got = {c: _dense_eval(kind, OBS3, 1, c, method, external_times=EXT_TIMES,
                          **kw).total for c in TRIPLES}
    want = {c: _three_point_exact(c, M_EXT) for c in TRIPLES}
    scale = max(abs(v) for v in want.values())
    worst = max(abs(got[c] - want[c]) for c in TRIPLES) / scale
    assert worst < rel, worst
    assert got[(0, 1, 1)] == pytest.approx(want[(0, 1, 1)], rel=rel, abs=0.0)


def test_MB3_raw_three_point_nquad_and_scalar_loop():
    for comps in [(0, 1, 1), (1, 0, 0)]:
        got = _dense_eval("raw", OBS3, 1, comps, "nquad",
                          external_times=EXT_TIMES).total
        assert got == pytest.approx(_three_point_exact(comps, M_EXT),
                                    rel=1e-10, abs=0.0)
    kw = dict(external_times=EXT_TIMES, n_samples=2 ** 7, seed=2)
    _assert_per_diagram(
        _dense_eval("raw", OBS3, 1, (0, 1, 1), "qmc_vectorized", **kw),
        _dense_eval("raw", OBS3, 1, (0, 1, 1), "qmc_scalar", **kw),
        rel=1e-12)


@pytest.mark.parametrize("comps", [(0, 1, 1), (1, 1, 0), (0, 0, 0)])
def test_MB3_absorbed_integrate_over_all(comps):
    """Every external time integrated over [t_min, t_f]: the callable is
    evaluated at swept partner times; exact ``Σ κ (∫M)(∫M)(∫M)``."""
    Mi = _M_int_ref(T_F)
    want = _three_point_exact(comps, [Mi, Mi, Mi])
    got = _dense_eval("absorbed", OBS3, 1, comps, "gauss_legendre",
                      n_gauss=12, integrate_over="all").total
    assert got == pytest.approx(want, rel=1e-12, abs=0.0)
    kw = dict(integrate_over="all", n_samples=2 ** 6, seed=4)
    _assert_per_diagram(
        _dense_eval("absorbed", OBS3, 1, comps, "qmc_vectorized", **kw),
        _dense_eval("absorbed", OBS3, 1, comps, "qmc_scalar", **kw),
        rel=1e-12)


def test_MB3_absorbed_legs_carry_their_partners_index():
    """The premise: with diag_R=False every kept R keeps two indices and
    every absorbed R carries its partner's index twice."""
    dts = _dense_expansion("absorbed", OBS3, 1).dts_by_order[1]
    assert dts
    for dt in dts:
        absorbed = set(dt.r_absorbed_pairs)
        assert len(absorbed) == 3
        for p in dt.propagators:
            if (p.spatial_left, p.spatial_right) in absorbed:
                assert p.index_left == p.index_right in ("a", "b", "c")


# --------------------------------------------------------------------------- #
# MB4: the FK channel against the hierarchy with a constant random source
# --------------------------------------------------------------------------- #
def _fk(kind, comps, method, **kw):
    return _dense_eval(f"F+{kind}", OBS2, 2, comps, method,
                       vertex_types={"FK"}, **kw)


@pytest.mark.parametrize("comps", PAIRS)
@pytest.mark.parametrize("kind, n_gauss", [("absorbed", 12), ("raw", 8)])
def test_MB4_fk_channel_matches_hierarchy(kind, n_gauss, comps):
    got = _fk(kind, comps, "gauss_legendre", n_gauss=n_gauss).total
    assert got == pytest.approx(_fk_moment(comps), rel=1e-10, abs=0.0)


@pytest.mark.parametrize("kind", ["raw", "absorbed"])
def test_MB4_fk_batched_equals_scalar_loop(kind):
    kw = dict(n_samples=2 ** 6, seed=5)
    _assert_per_diagram(_fk(kind, (0, 1), "qmc_vectorized", **kw),
                        _fk(kind, (0, 1), "qmc_scalar", **kw), rel=1e-12)


def _l0_fk_terms(diag_R: bool) -> list:
    """The FK terms of ``<φ_a(x) φ_b(y)>`` at L0, absorbed K."""
    sw.reset_uid_counter()
    phi = sw.Field("phi", "physical", n_components=N)
    psi = sw.Field("psi", "response", n_components=N)
    action = sw.Action(vertices=[
        sw.Vertex(fields=[psi, phi, phi], coupling="F", local=True),
        sw.Vertex(fields=[psi, psi, psi], coupling="K", local=False,
                  already_R_contracted=True),
    ])
    res = sw.compute_moment([phi("a", "x"), phi("b", "y")], action, order=2,
                            diag_R=diag_R)
    return [dt for dt in res.diagram_terms(2)
            if set(_collect_symbol_names(dt.coupling_sum)) == {"F", "K"}]


def test_MB4_l0_absorbed_default_flags_right_diag_R_wrong():
    """At L0 the default flags now give the exact FK channel.  The route
    that ran ``already_R_contracted`` before, ``diag_R=True``, applies the
    absorbed R's delta but also keeps only R_ii of every other R, which for
    this dense R is far off."""
    kappa_r = KappaR()
    cv = {"F": -1j * F_TENSOR,
          "K": lambda n_list, t_list: (1j / 6.0) * kappa_r(n_list, t_list)}
    kw = dict(lambda_f=T_F, t_min=T_MIN, cache=_dense("F+absorbed")[1].cache,
              method="gauss_legendre", n_gauss=12,
              fixed_indices={"a": 0, "b": 1}, positions={"x": 0.0, "y": 0.7})
    want = _fk_moment((0, 1))
    right, _ = integrate_diagrams(_l0_fk_terms(False), cv, **kw)
    assert right == pytest.approx(want, rel=1e-10, abs=0.0)
    wrong, _ = integrate_diagrams(_l0_fk_terms(True), cv, **kw)
    assert abs(wrong - want) > 0.3 * abs(want)


# --------------------------------------------------------------------------- #
# MB5 / MB6: callable and equal_time κ³ against scipy hand contractions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["callable", "callable_batched"])
def test_MB5_callable_kappa_matches_hand(kind):
    got = {c: _dense_eval(kind, OBS3, 1, c, "gauss_legendre", n_gauss=8,
                          external_times=EXT_TIMES).total for c in TRIPLES}
    want = {c: _mb5_hand(c, EXT_TIMES) for c in TRIPLES}
    scale = max(abs(v) for v in want.values())
    assert max(abs(got[c] - want[c]) for c in TRIPLES) / scale < 1e-11
    kw = dict(external_times=EXT_TIMES, n_samples=2 ** 6, seed=6)
    _assert_per_diagram(
        _dense_eval(kind, OBS3, 1, (1, 0, 1), "qmc_vectorized", **kw),
        _dense_eval(kind, OBS3, 1, (1, 0, 1), "qmc_scalar", **kw), rel=1e-12)


@pytest.mark.parametrize("method, kw, rel", [
    ("gauss_legendre", {"n_gauss": 12}, 1e-12),
    ("nquad", {}, 1e-10),
])
def test_MB6_equal_time_kappa_matches_hand(method, kw, rel):
    for comps in [(0, 1, 1), (1, 0, 0), (1, 1, 1)]:
        got = _dense_eval("equal_time", OBS3, 1, comps, method,
                          external_times=EXT_TIMES, **kw).total
        assert got == pytest.approx(_mb6_hand(comps, EXT_TIMES), rel=rel,
                                    abs=0.0)
    kw = dict(external_times=EXT_TIMES, n_samples=2 ** 6, seed=8)
    _assert_per_diagram(
        _dense_eval("equal_time", OBS3, 1, (0, 1, 1), "qmc_vectorized", **kw),
        _dense_eval("equal_time", OBS3, 1, (0, 1, 1), "qmc_scalar", **kw),
        rel=1e-12)


# --------------------------------------------------------------------------- #
# MB7: integrate_two_point_qmc
# --------------------------------------------------------------------------- #
def test_MB7_two_point_qmc_with_a_matrix_R():
    """Raised 'setting an array element with a sequence'; now equals the
    scalar loop on the same Sobol points."""
    system, props = _dense("F")
    cv = system.build_coupling_values()
    igs = [dt.build_integrand(cv, {"a": 0, "b": 1})
           for dt in _dense_expansion("F", OBS2, 2).dts_by_order[2]]
    pos = {"x": 0.0, "y": 0.7}
    kw = dict(t_min=T_MIN, n_samples=2 ** 6, seed=3)
    two, _ = integrate_two_point_qmc(igs, T_F, pos, props.cache, **kw)
    scalar = sum(integrate_moment(ig, T_F, props.cache, method="qmc_scalar",
                                  positions=pos, **kw)[0] for ig in igs)
    assert two == pytest.approx(scalar, rel=1e-12, abs=0.0)
    assert abs(scalar) > 1e-4


def test_MB7_two_point_qmc_with_a_callable_coupling():
    """Returned exactly 0 for a callable coupling: it read the zeros
    placeholder of the static coupling array."""
    _, props = _dense("raw")
    dts = _dense_expansion("raw", OBS3, 1).dts_by_order[1]
    fixed = {"a": 0, "b": 1, "c": 1}
    msr = (1j / 6.0) * KAPPA3
    static = [dt.build_integrand({"K": msr}, fixed) for dt in dts]
    dynamic = [dt.build_integrand({"K": lambda n_list, t_list: msr}, fixed)
               for dt in dts]
    assert all(ig.dynamic_coupling is not None for ig in dynamic)
    kw = dict(t_min=T_MIN, n_samples=2 ** 7, seed=9)
    want, _ = integrate_two_point_qmc(static, T_F, POS, props.cache, **kw)
    got, _ = integrate_two_point_qmc(dynamic, T_F, POS, props.cache, **kw)
    assert abs(want) > 1e-3
    assert got == pytest.approx(want, rel=1e-12, abs=0.0)


# --------------------------------------------------------------------------- #
# MB8: method='qmc' dispatch
# --------------------------------------------------------------------------- #
class _BatchCapable:
    """A cache that reports batch C support (``_c_splines``)."""

    _c_splines = True

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_MB8_qmc_dispatch_takes_the_batched_path_for_a_matrix_R(monkeypatch):
    """``method='qmc'`` required ``cache.model.iso_R`` for the batched path,
    so a matrix R always took the scalar loop."""
    system, props = _dense("F")
    ig = _dense_expansion("F", ("phi_a(x)",), 1).dts_by_order[1][0] \
        .build_integrand(system.build_coupling_values(), {"a": 1})
    kw = dict(t_min=T_MIN, n_samples=2 ** 6, seed=3, positions={"x": 0.0})
    want = ig.integrate_moment_qmc_vectorized(T_F, props.cache, **kw)

    def no_scalar_loop(*args, **kwargs):
        raise AssertionError("method='qmc' took the scalar loop")

    monkeypatch.setattr(DiagramIntegrand, "integrate_moment_qmc",
                        no_scalar_loop)
    got = integrate_moment(ig, T_F, _BatchCapable(props.cache), method="qmc",
                           **kw)
    assert got == want
