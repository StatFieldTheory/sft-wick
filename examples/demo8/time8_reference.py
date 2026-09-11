r"""Demo 8's exact reference: moment hierarchies with a time-dependent generator.

The field at the observation points and the auxiliary states of its noise
form a finite polynomial Itô SDE whose linear drift and diffusion
coefficients may depend on time.  Its generator is a sum

.. math::

    L(t) = Σ_k c_k(t)\, L_k ,

each ``L_k`` a constant-coefficient generator
(:class:`ito_moments.PolySDE`, whose monomial action this module reuses)
and ``c_k`` a scalar function of time (``None`` for the constant part).
Moment coefficients ``m_α^{(K)}`` (``K`` counts powers of the coupling
tensors) obey ``ṁ = Σ_k c_k(t) M_k m``, a finite linear system collected by
breadth-first search from the targets as in :func:`ito_moments.solve`, and
integrated with ``scipy.integrate.solve_ivp`` (DOP853, ``rtol = 1e-12``).

Two-time moments ``E[X(t_1)^β X(t_2)^α]`` with ``t_1 ≤ t_2``: for
``t ≥ t_1`` the vector ``E[X(t_1)^β X(t)^{α'}]`` obeys the same linear
system in ``α'`` (``X(t_1)`` is fixed after ``t_1``), started from
``E[X(t_1)^{β+α'}]``.

Also here, for the free field ``C``:

* :class:`ExactC`, the defining integrals of ``C`` by Gauss-Legendre
  quadrature, split at the kernel's diagonal (about 1e-15 for these smooth
  pieces; checked against the hierarchy's order 0);
* :func:`gaussian_kernel_C`, ``C`` for a Gaussian temporal kernel and an
  exponential ``R`` in closed form (``erf``);
* :class:`HandContraction`, the order-1 tadpole and the order-2 two-point
  function of a quadratic drift, written out from the perturbative solution
  of the SDE and Wick's theorem and integrated by Gauss-Legendre.  It needs
  only ``R`` and ``C``, so it applies to kernels that have no Markov
  embedding.

Imports no sft-wick code.
"""
from __future__ import annotations

import itertools
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
from scipy.special import erf, erfcx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import ito_moments as im  # noqa: E402

import time8_params as pm  # noqa: E402

__all__ = [
    "RTOL", "ATOL", "Piece", "TDGenerator", "Model",
    "rate_model", "oscillator_model", "kernel_model", "white_model",
    "ExactC", "gaussian_kernel_C", "HandContraction",
]

RTOL = 1e-12
ATOL = 1e-18


# ---------------------------------------------------------------------------
# The time-dependent hierarchy
# ---------------------------------------------------------------------------

@dataclass
class Piece:
    """``c(t) · L`` for one part of the generator (``c = None`` means 1)."""

    coefficient: Callable[[float], float] | None
    sde: im.PolySDE


class TDGenerator:
    """``L(t) = Σ_k c_k(t) L_k`` acting on moment coefficients."""

    def __init__(self, pieces: Sequence[Piece]):
        self.pieces = list(pieces)

    def system(self, targets):
        """``(index, [M_k])``: node ``(α, K) ↦ row`` and one sparse matrix
        per piece, over every node the targets need."""
        index: dict = {}
        rows: list[list] = [[] for _ in self.pieces]
        queue = list(dict.fromkeys(targets))
        while queue:
            node = queue.pop()
            if node in index:
                continue
            r = index[node] = len(index)
            alpha, tag = node
            for k, piece in enumerate(self.pieces):
                for beta, c, dtag in piece.sde.generator(alpha):
                    src_tag = im._tag_sub(tag, dtag)
                    if src_tag is None:
                        continue
                    src = (beta, src_tag)
                    rows[k].append((r, src, c))
                    if src not in index:
                        queue.append(src)
        n = len(index)
        mats = []
        for entries in rows:
            if not entries:
                mats.append(csr_matrix((n, n)))
                continue
            ri, srcs, cs = zip(*entries)
            ci = [index[s] for s in srcs]
            mats.append(csr_matrix((cs, (ri, ci)), shape=(n, n)))
        return index, mats

    def integrate(self, mats, y0, t0: float, times) -> np.ndarray:
        """``y(t)`` at ``times`` (``≥ t0``, increasing) from ``y(t0) = y0``."""
        times = np.asarray(times, dtype=float)
        if times[0] < t0 or np.any(np.diff(times) < 0):
            raise ValueError("times must be increasing and >= t0")
        n = len(y0)
        const = csr_matrix((n, n))
        timed = []
        for piece, M in zip(self.pieces, mats):
            if piece.coefficient is None:
                const = const + M
            elif M.nnz:
                timed.append((piece.coefficient, M))

        def rhs(t, y):
            out = const @ y
            for fn, M in timed:
                out = out + fn(t) * (M @ y)
            return out

        if times[-1] == t0:
            return np.repeat(np.asarray(y0, float)[:, None], len(times), axis=1)
        sol = solve_ivp(rhs, (t0, times[-1]), y0, method="DOP853",
                        t_eval=times, rtol=RTOL, atol=ATOL)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y

    def solve(self, targets, t0: float, times, initial) -> dict:
        """``{(α, K): m_α^{(K)}(times)}``; ``initial(α, K)`` at ``t0``."""
        index, mats = self.system(targets)
        nodes = sorted(index, key=index.get)
        y0 = np.array([initial(*node) for node in nodes], dtype=float)
        Y = self.integrate(mats, y0, t0, times)
        return {node: Y[index[node]] for node in targets}

    def solve_two_time(self, early, late, tags, t0: float, t1: float,
                       t2: float, initial) -> dict:
        """``{K: E[X(t1)^early X(t2)^late]^{(K)}}`` for ``t0 ≤ t1 ≤ t2``."""
        if not t0 <= t1 <= t2:
            raise ValueError("need t0 <= t1 <= t2")
        index, mats = self.system([(late, tuple(K)) for K in tags])
        nodes = sorted(index, key=index.get)
        stage1 = [(tuple(a + b for a, b in zip(alpha, early)), K)
                  for alpha, K in nodes]
        m1 = self.solve(stage1, t0, [t1], initial)
        y0 = np.array([m1[node][0] for node in stage1])
        Y = self.integrate(mats, y0, t1, [t2])
        return {tuple(K): float(Y[index[(late, tuple(K))], 0]) for K in tags}


class Model:
    """A polynomial Itô SDE with time-dependent linear coefficients and a
    zero-mean Gaussian initial law for its auxiliary variables.

    Variables are named by hashable keys, e.g. ``("phi", a, i)`` for
    component ``a`` of the field at observation point ``i``.
    """

    def __init__(self, n_tags: int):
        self.n_tags = n_tags
        self.keys: list = []
        self._index: dict = {}
        self._lin: dict = defaultdict(lambda: defaultdict(float))
        self._diff: dict = defaultdict(lambda: defaultdict(float))
        self._poly: list = []
        self._cov: dict = {}
        self._gen: TDGenerator | None = None
        self._gauss_cache: dict = {}

    # -- construction --------------------------------------------------------

    def var(self, key) -> int:
        if key not in self._index:
            if self._gen is not None:
                raise RuntimeError("model already built")
            self._index[key] = len(self.keys)
            self.keys.append(key)
        return self._index[key]

    def drift(self, target, source, c: float, fn=None) -> None:
        """``dX_target ⊃ fn(t) · c · X_source dt``."""
        self._lin[fn][(self.var(target), self.var(source))] += float(c)

    def noise(self, k1, k2, c: float, fn=None) -> None:
        """Covariance rate ``⟨dX_k1 dX_k2⟩ ⊃ fn(t) · c · dt``."""
        i, j = self.var(k1), self.var(k2)
        self._diff[fn][(i, j)] += float(c)
        if i != j:
            self._diff[fn][(j, i)] += float(c)

    def poly(self, target, c: float, sources, tag) -> None:
        """``dX_target ⊃ c Π X_sources dt``, counted by ``tag``."""
        self._poly.append((self.var(target), float(c),
                           tuple(self.var(s) for s in sources), tuple(tag)))

    def initial_cov(self, k1, k2, c: float) -> None:
        i, j = self.var(k1), self.var(k2)
        self._cov[(i, j)] = self._cov[(j, i)] = float(c)

    # -- solving -------------------------------------------------------------

    @property
    def D(self) -> int:
        return len(self.keys)

    def generator(self) -> TDGenerator:
        if self._gen is None:
            D, zero = self.D, (0,) * self.n_tags
            pieces = []
            for fn in dict.fromkeys([None, *self._lin, *self._diff]):
                sde = im.PolySDE(D=D, n_tags=self.n_tags)
                for (i, j), c in self._lin.get(fn, {}).items():
                    if c:
                        sde.drift.append((i, c, im.unit(D, j), zero))
                for (i, j), c in self._diff.get(fn, {}).items():
                    if c:
                        sde.diffusion.append((i, j, c, (0,) * D, zero))
                if fn is None:
                    for i, c, src, tag in self._poly:
                        sde.drift.append((i, c, im.unit(D, *src), tag))
                pieces.append(Piece(fn, sde))
            self._gen = TDGenerator(pieces)
        return self._gen

    def initial(self, alpha, tag) -> float:
        """Moments at ``t0``: fields at 0, auxiliary states Gaussian."""
        if any(tag):
            return 0.0
        legs = tuple(i for i, a in enumerate(alpha) for _ in range(a))
        return self._gauss(legs)

    def _gauss(self, legs: tuple) -> float:
        if not legs:
            return 1.0
        if len(legs) % 2:
            return 0.0
        key = tuple(sorted(legs))
        hit = self._gauss_cache.get(key)
        if hit is not None:
            return hit
        first, rest = key[0], key[1:]
        total = 0.0
        for k, other in enumerate(rest):
            c = self._cov.get((first, other), 0.0)
            if c:
                total += c * self._gauss(rest[:k] + rest[k + 1:])
        self._gauss_cache[key] = total
        return total

    def monomial(self, keys) -> im.Monomial:
        return im.unit(self.D, *[self._index[k] for k in keys])

    def moments(self, keys, tags, t0: float, T: float) -> dict:
        """``{K: E[Π X_keys](T)^{(K)}}``."""
        mono = self.monomial(keys)
        tags = [tuple(K) for K in tags]
        out = self.generator().solve([(mono, K) for K in tags], t0, [T],
                                     self.initial)
        return {K: float(out[(mono, K)][0]) for K in tags}

    def two_time(self, early_keys, t1: float, late_keys, t2: float, tags,
                 t0: float) -> dict:
        """``{K: E[Π X_early(t1) Π X_late(t2)]^{(K)}}``, either order."""
        if t1 > t2:
            early_keys, late_keys, t1, t2 = late_keys, early_keys, t2, t1
        return self.generator().solve_two_time(
            self.monomial(early_keys), self.monomial(late_keys), tags, t0,
            t1, t2, self.initial)


# ---------------------------------------------------------------------------
# The four models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Component:
    """``t ↦ fn(t)[a]`` as a float (a hashable piece coefficient)."""

    fn: object
    a: int
    method: str = "__call__"

    def __call__(self, t: float) -> float:
        return float(np.asarray(getattr(self.fn, self.method)(t))[self.a])


def _distances(positions) -> np.ndarray:
    x = np.asarray(positions, dtype=float)
    return np.abs(x[:, None] - x[None, :])


def _add_drift_tensor(m: Model, tensor, target: str, field: str,
                      P: int, tag) -> None:
    """``dX_(target,a,i) ⊃ Σ tensor[a, b, …] Π X_(field,b,i) dt``."""
    for i in range(P):
        for idx in itertools.product(range(pm.N), repeat=tensor.ndim):
            c = float(tensor[idx])
            if c:
                m.poly((target, idx[0], i), c,
                       [(field, b, i) for b in idx[1:]], tag)


def _ou_noise(m: Model, name: str, a: int, P: int, lam: float,
              sigma_t: float, K: np.ndarray) -> None:
    """Stationary OU states ``(name, a, i)``, ``⟨η_i η_j⟩ = λ K_ij``."""
    for i in range(P):
        m.drift((name, a, i), (name, a, i), -1.0 / sigma_t)
    for i in range(P):
        for j in range(i, P):
            m.noise((name, a, i), (name, a, j), 2.0 * lam / sigma_t * K[i, j])
            m.initial_cov((name, a, i), (name, a, j), lam * K[i, j])


def rate_model(p: pm.RateParams, positions) -> Model:
    """Model (a): ``dφ = (−γ(t) φ + η + F φφ) dt``, OU ``η``."""
    P = len(positions)
    K = pm.GaussianEnvelope(p.sigma_x)(_distances(positions))
    m = Model(n_tags=1)
    for a in range(pm.N):
        g = _Component(p.rate, a)
        for i in range(P):
            m.drift(("phi", a, i), ("phi", a, i), -1.0, fn=g)
            m.drift(("phi", a, i), ("eta", a, i), 1.0)
        _ou_noise(m, "eta", a, P, p.lam, p.sigma_t, K)
    _add_drift_tensor(m, pm.F_QUAD, "phi", "phi", P, (1,))
    return m


def oscillator_model(p: pm.OscillatorParams, positions) -> Model:
    """Model (b): the ``(x, v)`` embedding, tags ``(F, G)``."""
    P = len(positions)
    Kw = p.white.spatial(_distances(positions))
    om, ze = p.R.omega, p.R.zeta
    m = Model(n_tags=2)
    for a in range(pm.N):
        for i in range(P):
            x, v = ("x", a, i), ("v", a, i)
            m.drift(x, v, 1.0)
            m.drift(v, x, -om * om)
            m.drift(v, v, -2.0 * ze * om)
        constant = all(mm == 0.0 for mm in p.white.m)
        fn = None if constant else _Component(p.white, a, "s")
        amp = p.white.S[a] if constant else 1.0
        for i in range(P):
            for j in range(i, P):
                m.noise(("v", a, i), ("v", a, j), amp * Kw[i, j], fn=fn)
    _add_drift_tensor(m, pm.F_QUAD, "v", "x", P, (1, 0))
    _add_drift_tensor(m, pm.G_CUBIC, "v", "x", P, (0, 1))
    return m


def kernel_model(p: pm.KernelParams, positions) -> Model:
    """Model (c) for a kernel with a finite embedding."""
    P = len(positions)
    K = pm.GaussianEnvelope(p.sigma_x)(_distances(positions))
    k = p.kernel
    m = Model(n_tags=1)
    for a in range(pm.N):
        for i in range(P):
            m.drift(("phi", a, i), ("phi", a, i), -p.gamma[a])
            m.drift(("phi", a, i), ("eta", a, i), 1.0)
        if isinstance(k, pm.ExponentialKernel):
            _ou_noise(m, "eta", a, P, k.lam, k.sigma, K)
        elif isinstance(k, pm.Matern32):
            for i in range(P):
                m.drift(("eta", a, i), ("xi", a, i), 1.0)
                m.drift(("xi", a, i), ("eta", a, i), -k.a ** 2)
                m.drift(("xi", a, i), ("xi", a, i), -2.0 * k.a)
            for i in range(P):
                for j in range(i, P):
                    m.noise(("xi", a, i), ("xi", a, j),
                            4.0 * k.a ** 3 * k.lam * K[i, j])
                    m.initial_cov(("eta", a, i), ("eta", a, j), k.lam * K[i, j])
                    m.initial_cov(("xi", a, i), ("xi", a, j),
                                  k.a ** 2 * k.lam * K[i, j])
        elif isinstance(k, pm.DampedCosine):
            for i in range(P):
                z1, z2 = ("eta", a, i), ("zeta", a, i)
                m.drift(z1, z1, -k.a)
                m.drift(z1, z2, -k.omega)
                m.drift(z2, z1, k.omega)
                m.drift(z2, z2, -k.a)
            for i in range(P):
                for j in range(i, P):
                    for name in ("eta", "zeta"):
                        m.noise((name, a, i), (name, a, j),
                                2.0 * k.a * k.lam * K[i, j])
                        m.initial_cov((name, a, i), (name, a, j),
                                      k.lam * K[i, j])
        else:
            raise TypeError(f"{type(k).__name__} has no finite embedding")
    _add_drift_tensor(m, pm.F_QUAD, "phi", "phi", P, (1,))
    return m


def white_model(p: pm.WhiteParams, positions) -> Model:
    """Model (d): OU ``η`` plus white noise of amplitude ``s_a(t)``."""
    P = len(positions)
    d = _distances(positions)
    K = pm.GaussianEnvelope(p.sigma_x)(d)
    Kw = p.white.spatial(d)
    m = Model(n_tags=1)
    for a in range(pm.N):
        fn = _Component(p.white, a, "s")
        for i in range(P):
            m.drift(("phi", a, i), ("phi", a, i), -p.gamma[a])
            m.drift(("phi", a, i), ("eta", a, i), 1.0)
            for j in range(i, P):
                m.noise(("phi", a, i), ("phi", a, j), Kw[i, j], fn=fn)
        _ou_noise(m, "eta", a, P, p.lam, p.sigma_t, K)
    _add_drift_tensor(m, pm.F_QUAD, "phi", "phi", P, (1,))
    return m


# ---------------------------------------------------------------------------
# The free C by quadrature of its definition
# ---------------------------------------------------------------------------

def _gl01(n: int):
    x, w = leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


class ExactC:
    """``C_ab(x1, t1; x2, t2) = δ_ab c_a``, with

    .. math::

        c_a = K(r) ∫∫ R_a(t_1, s_1)\\, k(s_1 − s_2)\\, R_a(t_2, s_2)
              + K_w(r) ∫ R_a(t_1, s)\\, w_a(s)\\, R_a(t_2, s)\\, ds ,

    over ``[t0, t1] × [t0, t2]`` and ``[t0, min(t1, t2)]``, ``r = |x1 − x2|``.
    The square is split at ``s_1 = s_2`` (where an OU-type kernel has a cusp)
    and each piece integrated with an ``n × n`` Gauss-Legendre rule.

    Called as the package's ``c_closed_form``: scalar times give ``(N, N)``,
    ``(n,)`` arrays give ``(n, N, N)``.

    Args:
        R: ``R(a, t, s)`` for ``t ≥ s``, vectorised.
        kernel: ``k(τ)``, vectorised and even, or ``None``.
        envelope: ``K(r)``.
        t0: start time.
        white: ``w(s) -> (N, …)`` white-noise amplitudes, or ``None``.
        white_envelope: ``K_w(r)``.
        has_diagonal_kink: attribute read by the package's Gauss-Legendre
            integrator (split the domain where two ends of a C cross).
    """

    def __init__(self, R, kernel, envelope, t0: float, *, white=None,
                 white_envelope=None, n: int = 32, has_diagonal_kink=None,
                 chunk: int = 256):
        self.R, self.kernel, self.envelope = R, kernel, envelope
        self.white, self.white_envelope = white, white_envelope
        self.t0 = float(t0)
        self.n, self.chunk = int(n), int(chunk)
        self._u, self._w = _gl01(self.n)
        self.has_diagonal_kink = (white is not None if has_diagonal_kink is None
                                  else bool(has_diagonal_kink))

    def __repr__(self) -> str:
        return (f"ExactC(R={self.R!r}, kernel={self.kernel!r}, t0={self.t0}, "
                f"white={self.white!r}, n={self.n})")

    # -- pieces ------------------------------------------------------------

    def _smooth(self, a: int, t1, t2) -> np.ndarray:
        """``∫∫ R_a k R_a`` for ``(m,)`` times ``t1, t2 > t0``."""
        t0, u, w = self.t0, self._u, self._w
        tl, th = np.minimum(t1, t2), np.maximum(t1, t2)
        U, V = u[None, :, None], u[None, None, :]
        W = (w[:, None] * w[None, :])[None]
        L = (tl - t0)[:, None, None]
        tl_, th_ = tl[:, None, None], th[:, None, None]
        t1_, t2_ = t1[:, None, None], t2[:, None, None]
        # square [t0, tl]^2, lower triangle s2 < s1 and upper s1 < s2
        s_hi = t0 + L * U
        s_lo = t0 + (s_hi - t0) * V
        jac = L * (s_hi - t0)
        k = self.kernel(s_hi - s_lo)
        lower = self.R(a, t1_, s_hi) * k * self.R(a, t2_, s_lo)
        upper = self.R(a, t1_, s_lo) * k * self.R(a, t2_, s_hi)
        out = np.sum(W * jac * (lower + upper), axis=(1, 2))
        # strip: the variable of the later time in [tl, th], the other in [t0, tl]
        sh = tl_ + (th_ - tl_) * U
        sl = t0 + L * V
        strip = self.R(a, th_, sh) * self.kernel(sh - sl) * self.R(a, tl_, sl)
        out = out + np.sum(W * (th_ - tl_) * L * strip, axis=(1, 2))
        return out

    def _white(self, a: int, t1, t2) -> np.ndarray:
        t0, u, w = self.t0, self._u, self._w
        tl = np.minimum(t1, t2)
        L = (tl - t0)[:, None]
        s = t0 + L * u[None, :]
        amp = np.asarray(self.white(s))[a]
        vals = self.R(a, t1[:, None], s) * amp * self.R(a, t2[:, None], s)
        return np.sum(w[None, :] * L * vals, axis=1)

    def diagonal(self, t1, t2, r) -> np.ndarray:
        """``(m, N)``: ``c_a`` for arrays ``t1, t2, r``."""
        t1 = np.asarray(t1, float)
        t2 = np.asarray(t2, float)
        r = np.asarray(r, float)
        out = np.zeros(t1.shape + (pm.N,))
        live = (t1 > self.t0) & (t2 > self.t0)
        idx = np.nonzero(live)[0]
        for start in range(0, len(idx), self.chunk):
            sel = idx[start:start + self.chunk]
            for a in range(pm.N):
                val = np.zeros(len(sel))
                if self.kernel is not None:
                    val += self.envelope(r[sel]) * self._smooth(a, t1[sel], t2[sel])
                if self.white is not None:
                    val += (self.white_envelope(r[sel])
                            * self._white(a, t1[sel], t2[sel]))
                out[sel, a] = val
        return out

    def __call__(self, n1, t1, n2, t2):
        t1a = np.asarray(t1, float)
        t2a = np.asarray(t2, float)
        scalar = t1a.ndim == 0 and t2a.ndim == 0
        t1v, t2v = np.broadcast_arrays(np.atleast_1d(t1a), np.atleast_1d(t2a))
        m = t1v.shape[0]
        r = np.broadcast_to(np.abs(np.asarray(n1, float) - np.asarray(n2, float)),
                            (m,))
        diag = self.diagonal(np.array(t1v), np.array(t2v), np.array(r))
        C = np.zeros((m, pm.N, pm.N))
        C[:, range(pm.N), range(pm.N)] = diag
        return C[0] if scalar else C


# ---------------------------------------------------------------------------
# Gaussian temporal kernel: C in closed form, and the hand contraction
# ---------------------------------------------------------------------------

def gaussian_kernel_C(gamma: float, lam: float, sigma: float, t1, t2):
    """``λ ∫_0^{t1}∫_0^{t2} e^{−γ(t1−s1)} e^{−(s1−s2)²/(2σ²)} e^{−γ(t2−s2)}``
    in closed form (``erf``), times measured from ``t0``; zero where either
    time is ``≤ 0``.

    With ``μ = γσ²``, ``b = σ√2``, ``d = |t1 − t2|``, ``t_h = max``,
    ``t_l = min``::

        C = λσ√(π/2) e^{γ²σ²/2} / (2γ) · {
              e^{γd} [erfc((d+μ)/b) − erfc((t_h+μ)/b)]
            + e^{−γd} [erf((d−μ)/b) + erf((t_l+μ)/b)]
            − e^{−γ(t1+t2)} [erf((t1−μ)/b) + erf((t2−μ)/b) + 2 erf(μ/b)] }

    ``e^{γd} erfc(z)`` is evaluated as ``erfcx(z) e^{γd − z²}``.
    """
    t1 = np.asarray(t1, float)
    t2 = np.asarray(t2, float)
    th, tl = np.maximum(t1, t2), np.minimum(t1, t2)
    d = th - tl
    mu, b = gamma * sigma ** 2, sigma * np.sqrt(2.0)
    z1, z2 = (d + mu) / b, (th + mu) / b
    first = (erfcx(z1) * np.exp(gamma * d - z1 * z1)
             - erfcx(z2) * np.exp(gamma * d - z2 * z2))
    second = np.exp(-gamma * d) * (erf((d - mu) / b) + erf((tl + mu) / b))
    third = np.exp(-gamma * (t1 + t2)) * (erf((t1 - mu) / b) + erf((t2 - mu) / b)
                                          + 2.0 * erf(mu / b))
    pref = lam * sigma * np.sqrt(np.pi / 2.0) * np.exp(0.5 * (gamma * sigma) ** 2)
    out = pref / (2.0 * gamma) * (first + second - third)
    return np.where(tl > 0.0, out, 0.0)


class HandContraction:
    """Low orders of ``dφ_a = (−A φ + F_abc φ_b φ_c + η)_a dt`` written out.

    With ``φ = φ⁰ + φ¹ + φ² + …``, ``φ⁰ = R ∗ η`` Gaussian with covariance
    ``C``, ``φ¹_a = ∫R_a F_abc φ⁰_b φ⁰_c`` and
    ``φ²_a = ∫R_a F_abc (φ⁰_b φ¹_c + φ¹_b φ⁰_c)``:

    * order 1, ``⟨φ¹_a(x, T)⟩ = ∫ R_a(T, s) τ_a(s) ds`` with the tadpole
      ``τ_a(s) = Σ_b F_abb c_b(s, s; 0)``;
    * order 2, ``⟨φ_a(x,T) φ_d(y,T)⟩ = ⟨φ¹_a φ¹_d⟩ + ⟨φ²_a φ⁰_d⟩ + ⟨φ⁰_a φ²_d⟩``,
      each by Wick's theorem (docstrings of the methods).

    Args:
        R: ``R(a, t, s)``, vectorised.
        C: ``C(b, t1, t2, r)``, the component-``b`` diagonal of the free
            covariance, vectorised.
        t0: start time.
        F: the quadratic drift tensor.
        n: Gauss-Legendre nodes per dimension.
    """

    def __init__(self, R, C, t0: float, F=pm.F_QUAD, n: int = 40):
        self.R, self.C, self.t0, self.F = R, C, float(t0), np.asarray(F)
        self.u, self.w = _gl01(n)

    def _tad(self, a: int, s) -> np.ndarray:
        return sum(self.F[a, b, b] * self.C(b, s, s, 0.0) for b in range(pm.N))

    def tadpole(self, a: int, T: float) -> float:
        L = T - self.t0
        s = self.t0 + L * self.u
        return float(np.sum(self.w * L * self.R(a, T, s) * self._tad(a, s)))

    def _one_one(self, a: int, d: int, T: float, r: float) -> float:
        """``⟨φ¹_a(x,T) φ¹_d(y,T)⟩ = ∫∫ R_a(T,s1) R_d(T,s2) [τ_a(s1) τ_d(s2)
        + Σ_bc F_abc (F_dbc + F_dcb) c_b(s1,s2;r) c_c(s1,s2;r)]``, the square
        split at ``s1 = s2``."""
        F, t0, L = self.F, self.t0, T - self.t0
        U, V = self.u[:, None], self.u[None, :]
        W = self.w[:, None] * self.w[None, :]
        total = 0.0
        hi = t0 + L * U
        lo = t0 + (hi - t0) * V
        jac = L * (hi - t0)
        for s1, s2 in ((hi, lo), (lo, hi)):
            val = self._tad(a, s1) * self._tad(d, s2)
            for b, c in itertools.product(range(pm.N), repeat=2):
                coef = F[a, b, c] * (F[d, b, c] + F[d, c, b])
                if coef:
                    val = val + coef * self.C(b, s1, s2, r) * self.C(c, s1, s2, r)
            total += float(np.sum(W * jac * self.R(a, T, s1) * self.R(d, T, s2)
                                  * val))
        return total

    def _two_zero(self, a: int, d: int, T: float, r: float) -> float:
        """``⟨φ²_a(x,T) φ⁰_d(y,T)⟩ = ∫_{t0}^T ds R_a(T,s) Σ_bc F_abc
        [J(c, b) + J(b, c)]`` with
        ``J(c, b) = ∫_{t0}^s du R_c(s,u) [δ_db c_d(T,s;r) τ_c(u)
        + (F_cdb + F_cbd) c_d(T,u;r) c_b(s,u;0)]``."""
        F, t0, L = self.F, self.t0, T - self.t0
        U, V = self.u[:, None], self.u[None, :]
        W = self.w[:, None] * self.w[None, :]
        s = t0 + L * U
        u = t0 + (s - t0) * V
        jac = L * (s - t0)
        Cd_Ts = self.C(d, T, s, r)
        Cd_Tu = self.C(d, T, u, r)

        def J(c, b):
            val = (F[c, d, b] + F[c, b, d]) * Cd_Tu * self.C(b, s, u, 0.0)
            if d == b:
                val = val + Cd_Ts * self._tad(c, u)
            return self.R(c, s, u) * val

        inner = 0.0
        for b, c in itertools.product(range(pm.N), repeat=2):
            if F[a, b, c]:
                inner = inner + F[a, b, c] * (J(c, b) + J(b, c))
        return float(np.sum(W * jac * self.R(a, T, s) * inner))

    def two_point_order2(self, a: int, d: int, T: float, r: float) -> float:
        return (self._one_one(a, d, T, r) + self._two_zero(a, d, T, r)
                + self._two_zero(d, a, T, r))

    def two_point_order1_cubic(self, a: int, d: int, t_a: float, t_d: float,
                               r: float, G, *, split: bool = True,
                               n: int | None = None) -> float:
        r"""``⟨φ_a(x, t_a) φ_d(y, t_d)⟩`` at one cubic vertex ``G_{ibce}``,
        the two external legs at different times.

        Wick's theorem on ``⟨φ⁰_b φ⁰_c φ⁰_e (s) φ⁰_j(t_j)⟩`` leaves one
        ``s`` integral per assignment of the vertex to an external leg::

            Σ_{(i,j)} ∫_{t0}^{t_i} ds R_i(t_i, s)
                      Σ_b [G_{ibbj} + G_{ibjb} + G_{ijbb}]
                      c_b(s, s; 0)\, c_j(s, t_j; r) ,   (i, j) = (a, d), (d, a).

        ``c_j(s, t_j; r)`` is kinked at ``s = t_j`` whenever C is (white
        noise), and that point lies inside the domain of the assignment whose
        ``t_i`` is the later time.  With ``split`` the integral is cut there
        and Gauss-Legendre converges exponentially; without it, algebraically
        -- which is what the package's Gauss-Legendre does at unequal
        external times (see the demo README).
        """
        G = np.asarray(G)
        u, w = (self.u, self.w) if n is None else _gl01(n)
        total = 0.0
        for i, ti, j, tj in ((a, t_a, d, t_d), (d, t_d, a, t_a)):
            cuts = [self.t0, ti]
            if split and self.t0 < tj < ti:
                cuts = [self.t0, tj, ti]
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                s = lo + (hi - lo) * u
                acc = np.zeros_like(s)
                for b in range(pm.N):
                    coef = G[i, b, b, j] + G[i, b, j, b] + G[i, j, b, b]
                    if coef:
                        acc = acc + coef * self.C(b, s, s, 0.0) * self.C(j, s, tj, r)
                total += float(np.sum(w * (hi - lo) * self.R(i, ti, s) * acc))
        return total

    def two_point_order0(self, a: int, d: int, T: float, r: float) -> float:
        return float(self.C(a, T, T, r)) if a == d else 0.0
