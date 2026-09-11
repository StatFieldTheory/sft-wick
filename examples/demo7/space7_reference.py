r"""Demo 7's exact reference: the moment hierarchy of the Markov embedding at
the observation points.

``R`` is local in space, ``R(x, t; z, s) = δ(x − z) R(t − s)``, and every
local vertex acts at one point, so an n-point function at observation points
``x_1 … x_P`` involves the noise only at those points.  The fields there obey
a finite-dimensional Itô SDE,

.. math::

    dφ_{a,i} = (−γ_a φ_{a,i} + η_{a,i} + F_{abc} φ_{b,i} φ_{c,i}
                + G_{abcd} φ_{b,i} φ_{c,i} φ_{d,i})\,dt + dW_a ,

    dη_{a,i} = −ρ_a η_{a,i}\,dt + dB_{a,i},\qquad
    ⟨dB_{a,i}\,dB_{b,j}⟩ = δ_{ab}\,2ρ_a Σ^a_{ij}\,dt .

``η_{a,i}`` is the coloured noise at ``x_i``: an Ornstein-Uhlenbeck process
with rate ``ρ_a = 1/σ_{t,a}``, started in its stationary law ``N(0, Σ^a)``,
so that ``⟨η_a(x_i, t) η_b(x_j, t')⟩ = δ_ab Σ^a_ij e^{−ρ_a |t − t'|}``.  The
spatial structure of the noise enters only through ``Σ^a``, a ``P × P``
matrix per component: ``λ κ_x(|x_i − x_j|)`` for translation-invariant
noise, ``λ κ_Ω(cos θ_ij)`` for rotation-invariant noise, any kernel
``K_a(x_i, x_j)`` otherwise, in any dimension.  ``dW`` is optional white
noise shared by every point, ``⟨dW_a dW_b⟩ = S_ab dt``.  ``φ(0) = 0``; the
time axis starts at 0 here and at ``t_min`` in the package.

Three kinds of query:

* :meth:`Hierarchy.moments`: equal-time moments ``E[Π X(T)]``;
* :meth:`Hierarchy.two_time`: ``E[X^α(t) X^β(t')]`` for ``t ≥ t'``.  The
  conditional expectation ``E[X^α(t) | X(t')]`` is a polynomial in
  ``X(t')`` whose coefficients are a row of ``exp((t − t') M)``, ``M`` the
  generator on the monomials reachable from ``X^α``; the second factor is
  an equal-time moment at ``t'``;
* integrated fields ``I_{a,i}(t) = ∫_0^t φ_{a,i}``: extra states with
  ``dI = φ dt``, for the package's ``integrate_over``.

Tags ``(k_F, k_G)`` count powers of ``F`` and ``G``: a coefficient is the sum
of the package's diagrams with ``k_F`` F vertices and ``k_G`` G vertices.

The kernels are written out here with numpy (Legendre polynomials by their
explicit formulas), so the reference shares no code with the package.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import ito_moments as im  # noqa: E402

__all__ = ["Hierarchy", "reachable_generator", "legendre_series",
           "exponential_kernel", "gaussian_kernel", "damped_cosine_kernel",
           "inhomogeneous_kernel", "covariance"]


# ---------------------------------------------------------------------------
# Spatial kernels, written out independently of the package
# ---------------------------------------------------------------------------

def legendre_series(coeffs, cos_theta) -> np.ndarray:
    """``Σ_ℓ C_ℓ P_ℓ(cos θ)`` for ``ℓ ≤ 3``, from the explicit polynomials."""
    c = np.asarray(cos_theta, dtype=float)
    polys = [np.ones_like(c), c, 0.5 * (3.0 * c ** 2 - 1.0),
             0.5 * (5.0 * c ** 3 - 3.0 * c)]
    if len(coeffs) > len(polys):
        raise ValueError("legendre_series is written out up to l = 3")
    return sum(cl * pl for cl, pl in zip(coeffs, polys))


def exponential_kernel(sigma_x: float, r) -> np.ndarray:
    return np.exp(-np.asarray(r, dtype=float) / sigma_x)


def gaussian_kernel(sigma_x: float, r) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return np.exp(-r ** 2 / (2.0 * sigma_x ** 2))


def damped_cosine_kernel(ell: float, k: float, r) -> np.ndarray:
    """``e^{−r/ℓ} cos(k r)``, positive definite on the line (its spectrum is
    a sum of two Lorentzians) and negative for ``π/2 < k r < 3π/2``."""
    r = np.asarray(r, dtype=float)
    return np.exp(-r / ell) * np.cos(k * r)


def inhomogeneous_kernel(s: float, centre: float, length: float, x1, x2):
    """``g(x1) g(x2) e^{−(x1 − x2)²/(2 s²)}`` with ``g(x) = e^{−(x − c)²/(2 L²)}``:
    positive definite, and a function of ``x1`` and ``x2`` separately."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    g1 = np.exp(-(x1 - centre) ** 2 / (2.0 * length ** 2))
    g2 = np.exp(-(x2 - centre) ** 2 / (2.0 * length ** 2))
    return g1 * g2 * np.exp(-(x1 - x2) ** 2 / (2.0 * s ** 2))


def covariance(points, kernel) -> np.ndarray:
    """``[kernel(x_i, x_j)]_{ij}`` over the observation points."""
    P = len(points)
    out = np.empty((P, P))
    for i, j in itertools.product(range(P), repeat=2):
        out[i, j] = float(kernel(points[i], points[j]))
    return out


# ---------------------------------------------------------------------------
# The generator on the reachable monomials
# ---------------------------------------------------------------------------

def reachable_generator(sde: im.PolySDE, root) -> tuple[dict, csr_matrix]:
    """``(index, M)``: the ``(monomial, tag)`` nodes reachable from ``root``
    and the generator on them, ``ṁ = M m``, collected as
    :func:`ito_moments.solve` collects it."""
    index: dict = {}
    rows: list = []
    queue = [root]
    while queue:
        node = queue.pop()
        if node in index:
            continue
        index[node] = len(rows)
        alpha, tag = node
        row = []
        for beta, c, dtag in sde.generator(alpha):
            src_tag = tuple(x - y for x, y in zip(tag, dtag))
            if min(src_tag, default=0) < 0:
                continue
            src = (beta, src_tag)
            row.append((src, c))
            if src not in index:
                queue.append(src)
        rows.append(row)
    n = len(rows)
    data, ri, ci = [], [], []
    for r, row in enumerate(rows):
        for src, c in row:
            data.append(c)
            ri.append(r)
            ci.append(index[src])
    return index, csr_matrix((data, (ri, ci)), shape=(n, n))


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# The hierarchy
# ---------------------------------------------------------------------------

class Hierarchy:
    """Moment hierarchy of the embedding at ``P`` observation points.

    Args:
        n_points: ``P``.
        gamma: decay rates ``γ_a``, shape ``(N,)``.
        rho: noise rates ``ρ_a = 1/σ_{t,a}``, shape ``(N,)``.
        sigma: stationary covariances ``Σ^a_ij`` of ``η`` at the points,
            shape ``(N, P, P)``.
        F: ``(N, N, N)``, ``dφ_a ⊃ F_abc φ_b φ_c dt``; tag ``(1, 0)``.
        G: ``(N, N, N, N)`` or ``None``, ``dφ_a ⊃ G_abcd φ_b φ_c φ_d dt``;
            tag ``(0, 1)``.
        S: white-noise covariance rate ``(N, N)``, shared by every point, or
            ``None``.
        integrated: ``(a, i)`` pairs whose time integral ``I_{a,i}`` is a
            state.

    A leg is ``(a, i)`` for ``φ_{a,i}`` or ``("I", a, i)`` for ``I_{a,i}``.
    """

    def __init__(self, *, n_points: int, gamma, rho, sigma, F, G=None,
                 S=None, integrated=()):
        gamma = np.asarray(gamma, dtype=float)
        rho = np.asarray(rho, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        N, P = gamma.shape[0], int(n_points)
        if sigma.shape != (N, P, P):
            raise ValueError(f"sigma must have shape {(N, P, P)}, "
                             f"got {sigma.shape}")
        self.N, self.P = N, P
        self._phi = {(a, i): a * P + i for a in range(N) for i in range(P)}
        self._eta = {(a, i): N * P + a * P + i
                     for a in range(N) for i in range(P)}
        self._int = {tuple(key): 2 * N * P + k
                     for k, key in enumerate(integrated)}
        self.D = 2 * N * P + len(self._int)
        D = self.D

        sde = im.PolySDE(D=D, n_tags=2)
        A = np.zeros((D, D))
        for (a, i), k in self._phi.items():
            A[k, k] = -gamma[a]
            A[k, self._eta[(a, i)]] = 1.0
        for (a, i), k in self._eta.items():
            A[k, k] = -rho[a]
        for (a, i), k in self._int.items():
            A[k, self._phi[(a, i)]] = 1.0
        sde.add_linear_drift(A)

        Sd = np.zeros((D, D))
        for (a, i), k in self._eta.items():
            for (b, j), q in self._eta.items():
                if a == b:
                    Sd[k, q] = 2.0 * rho[a] * sigma[a, i, j]
        if S is not None:
            S = np.asarray(S, dtype=float)
            for (a, i), k in self._phi.items():
                for (b, j), q in self._phi.items():
                    Sd[k, q] = S[a, b]
        sde.add_constant_diffusion(Sd)

        Q = np.zeros((D, D, D))
        for i in range(P):
            for a, b, c in itertools.product(range(N), repeat=3):
                Q[self._phi[(a, i)], self._phi[(b, i)],
                  self._phi[(c, i)]] += F[a, b, c]
        sde.add_quadratic_drift(Q, tag=(1, 0))
        if G is not None:
            for i in range(P):
                for a, b, c, d in itertools.product(range(N), repeat=4):
                    if G[a, b, c, d] == 0.0:
                        continue
                    sde.drift.append((
                        self._phi[(a, i)], float(G[a, b, c, d]),
                        im.unit(D, self._phi[(b, i)], self._phi[(c, i)],
                                self._phi[(d, i)]),
                        (0, 1)))
        self.sde = sde

        # Stationary covariance of the η states, for the initial law.
        self._cov = np.zeros((D, D))
        for (a, i), k in self._eta.items():
            for (b, j), q in self._eta.items():
                if a == b:
                    self._cov[k, q] = sigma[a, i, j]
        self._eta_set = frozenset(self._eta.values())
        self._gauss_memo: dict = {(): 1.0}

    # -- the initial law ------------------------------------------------------

    def _gauss(self, legs: tuple) -> float:
        """``E[Π η]`` over the sorted state indices ``legs`` (Isserlis)."""
        hit = self._gauss_memo.get(legs)
        if hit is not None:
            return hit
        if len(legs) % 2:
            val = 0.0
        else:
            first, rest = legs[0], legs[1:]
            val = 0.0
            for j, other in enumerate(rest):
                c = self._cov[first, other]
                if c != 0.0:
                    val += c * self._gauss(rest[:j] + rest[j + 1:])
        self._gauss_memo[legs] = val
        return val

    def _initial(self, alpha, tag) -> float:
        if any(tag):
            return 0.0
        legs = []
        for k, n in enumerate(alpha):
            if n == 0:
                continue
            if k not in self._eta_set:
                return 0.0
            legs += [k] * n
        return self._gauss(tuple(legs))

    # -- queries ----------------------------------------------------------------

    def _state(self, leg) -> int:
        if len(leg) == 3:
            if leg[0] != "I":
                raise ValueError(f"unknown leg {leg!r}")
            return self._int[(leg[1], leg[2])]
        return self._phi[tuple(leg)]

    def monomial(self, legs) -> tuple:
        return im.monomial_of(self.D, [self._state(leg) for leg in legs])

    def moments(self, legs, tags, T: float) -> dict:
        """``{tag: E[Π legs](T)}`` at the tags ``(k_F, k_G)``."""
        mono = self.monomial(legs)
        out = im.solve(self.sde, [(mono, tuple(t)) for t in tags], [T],
                       initial=self._initial)
        return {tuple(t): float(out[(mono, tuple(t))][0]) for t in tags}

    def two_time(self, late_legs, early_legs, tags, t_late: float,
                 t_early: float) -> dict:
        """``{tag: E[Π late_legs(t_late) · Π early_legs(t_early)]}`` for
        ``t_late ≥ t_early``."""
        s = float(t_late) - float(t_early)
        if s < 0.0:
            raise ValueError("two_time needs t_late >= t_early")
        alpha = self.monomial(late_legs)
        beta = self.monomial(early_legs)
        out = {}
        for tag in tags:
            tag = tuple(tag)
            index, M = reachable_generator(self.sde, (alpha, tag))
            e = np.zeros(len(index))
            e[index[(alpha, tag)]] = 1.0
            row = expm_multiply(M.T.tocsr() * s, e) if s > 0.0 else e
            nodes = [(node, row[k]) for node, k in index.items()
                     if row[k] != 0.0]
            targets = [(_add(mono, beta), ntag) for (mono, ntag), _ in nodes]
            early = im.solve(self.sde, targets, [t_early],
                             initial=self._initial)
            out[tag] = float(sum(
                w * early[(_add(mono, beta), ntag)][0]
                for (mono, ntag), w in nodes))
        return out
