"""The R-contracted third cumulant of demo2's deformed noise.

``NonLocalVertex(already_R_contracted=True)`` takes the vertex with its
leg integrals already done,

    K_R(t1', t2', t3'; x') = ∫ du1 du2 du3  Π_i R(t_i', u_i)  κ^(3)(u; x'),

where ``t_i'`` are the PARTNER (outer) times of the three ψ legs and
``R(t, u) = Θ(t − u) exp(−γ (t − u))``.  For demo2

    κ^(3)(u) = 2 α λ² [k13 k23 + k12 k23 + k12 k13] + 8 α³ λ³ k12 k23 k13,
    k_ij = exp(−|u_i − u_j| / σ_t) · exp(−|x_i' − x_j'| / σ_x)

(the ``α³`` term is the connected three-point function of ``η² − λ``;
it is 2.4 % of the coincident value and was missing from the original
demo2 module).  Because the kernel is narrow in the RELATIVE times, the
tensor-product Gauss-Legendre rule the L2 config used on the raw 4-D
integral stops converging beyond ``t ≈ 10`` (n = 8/12/16/20 give
4.95/4.23/3.86/3.71e-4 for ξ01 at t = 15); contracting the legs first
removes the problem entirely -- the outer integral of an FK diagram is
one-dimensional.

Method.  With ``u1 = u + v1``, ``u2 = u + v2``, ``u3 = u`` the kernel
depends on ``(v1, v2)`` only and the three retarded factors give

    ∫ du e^{3γu} over [u_lo, u_hi] = (E(u_hi) − E(u_lo)) / (3γ),
    E(u) = exp(−γ(t1' − v1 − u) − γ(t2' − v2 − u) − γ(t3' − u)),
    u_lo = max(0, −v1, −v2),  u_hi = min(t3', t1' − v1, t2' − v2),

so ``K_R`` is a two-dimensional integral of a product of exponentials
with cusps on ``v1 = 0``, ``v2 = 0`` and ``v1 = v2``.  Each of the four
terms is integrated in coordinates that put ITS cusps on the axes
(``(v1, v2)``, ``(v1 − v2, v2)``, ``(v1 − v2, v1)``; the small ``α³`` term
keeps one unaligned cusp) with a composite Gauss-Legendre rule whose
panels are graded towards the peak.  Accuracy ~1e-6 relative; validated
against 3-D adaptive quadrature of the raw integral in
``validate_k3_R.py``.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

LAM, SIGMA_T, SIGMA_X, GAMMA, ALPHA, N_COMP = 0.05, 0.3, 1.0, 1.0, 0.6, 2

# Graded composite Gauss-Legendre grid on [-L, L], symmetric about 0.
_EDGES = SIGMA_T * np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 14.0])
_N_GL = 6


def _composite_grid():
    x, w = leggauss(_N_GL)
    nodes, weights = [], []
    edges = np.concatenate([-_EDGES[::-1], _EDGES[1:]])
    for lo, hi in zip(edges[:-1], edges[1:]):
        half = 0.5 * (hi - lo)
        nodes.append(lo + half * (x + 1.0))
        weights.append(half * w)
    return np.concatenate(nodes), np.concatenate(weights)


_G1, _W1 = _composite_grid()
_A, _B = np.meshgrid(_G1, _G1, indexing="ij")
_A, _B = _A.ravel(), _B.ravel()
_W2 = np.outer(_W1, _W1).ravel()


def _G_factor(v1, v2, t1, t2, t3, gamma=GAMMA):
    """``(E(u_hi) − E(u_lo)) / (3γ)`` with all exponents non-positive."""
    u_lo = np.maximum(0.0, np.maximum(-v1, -v2))
    u_hi = np.minimum(t3, np.minimum(t1 - v1, t2 - v2))
    ok = u_hi > u_lo
    u_hi_s = np.where(ok, u_hi, 0.0)
    u_lo_s = np.where(ok, u_lo, 0.0)
    a = -gamma * ((t1 - v1 - u_hi_s) + (t2 - v2 - u_hi_s) + (t3 - u_hi_s))
    b = -gamma * ((t1 - v1 - u_lo_s) + (t2 - v2 - u_lo_s) + (t3 - u_lo_s))
    a = np.minimum(a, 0.0)
    b = np.minimum(b, 0.0)
    return np.where(ok, (np.exp(a) - np.exp(b)) / (3.0 * gamma), 0.0)


def k3_R(t1, t2, t3, s12, s13, s23, *, lam=LAM, sigma_t=SIGMA_T, alpha=ALPHA):
    """``K_R`` for arrays of partner times ``(n,)`` and spatial factors ``(n,)``."""
    t1 = np.asarray(t1, float)[:, None]
    t2 = np.asarray(t2, float)[:, None]
    t3 = np.asarray(t3, float)[:, None]
    s12 = np.asarray(s12, float)[:, None]
    s13 = np.asarray(s13, float)[:, None]
    s23 = np.asarray(s23, float)[:, None]
    A = _A[None, :]
    B = _B[None, :]
    W = _W2[None, :]
    c2 = 2.0 * alpha * lam ** 2
    c3 = 8.0 * alpha ** 3 * lam ** 3
    ea = np.exp(-np.abs(A) / sigma_t)
    eb = np.exp(-np.abs(B) / sigma_t)
    # T1: k13 k23 -- cusps on v1 = 0, v2 = 0: coordinates (v1, v2) = (A, B)
    t_1 = c2 * s13 * s23 * ea * eb * _G_factor(A, B, t1, t2, t3)
    # T2: k12 k23 -- cusps on v1 - v2 = 0, v2 = 0: (w, v2) = (A, B), v1 = A + B
    t_2 = c2 * s12 * s23 * ea * eb * _G_factor(A + B, B, t1, t2, t3)
    # T3: k12 k13 -- cusps on v1 - v2 = 0, v1 = 0: (w, v1) = (A, B), v2 = B - A
    t_3 = c2 * s12 * s13 * ea * eb * _G_factor(B, B - A, t1, t2, t3)
    # T4 (alpha^3): k12 k23 k13 in (v1, v2); the diagonal cusp is unaligned.
    t_4 = c3 * s12 * s23 * s13 * ea * eb * np.exp(-np.abs(A - B) / sigma_t) \
        * _G_factor(A, B, t1, t2, t3)
    return np.sum(W * (t_1 + t_2 + t_3 + t_4), axis=1)


def _spatial(x_i, x_j, sigma_x=SIGMA_X):
    return np.exp(-np.abs(np.asarray(x_i, float) - np.asarray(x_j, float)) / sigma_x)


def coupling_fn_vectorized(n_2d, t_2d):
    """Batched contract: ``(3, n)`` partner positions / times → ``(n, N, N, N)``."""
    n = np.asarray(n_2d, float)
    t = np.asarray(t_2d, float)
    amp = k3_R(t[0], t[1], t[2], _spatial(n[0], n[1]), _spatial(n[0], n[2]), _spatial(n[1], n[2]))
    K = np.zeros((amp.shape[0],) + (N_COMP,) * 3)
    for a in range(N_COMP):
        K[:, a, a, a] = amp
    return K


def coupling_fn(n_list, t_list):
    n = np.asarray(n_list, float)[:, None]
    t = np.asarray(t_list, float)[:, None]
    return coupling_fn_vectorized(n, t)[0]


def kappa3_raw(u1, u2, u3, x1=0.0, x2=0.0, x3=0.0, *, lam=LAM, sigma_t=SIGMA_T,
               sigma_x=SIGMA_X, alpha=ALPHA):
    """The raw (un-contracted) κ^(3) amplitude, for validation."""
    def k(a, b, xa, xb):
        return np.exp(-abs(a - b) / sigma_t) * np.exp(-abs(xa - xb) / sigma_x)
    k12, k13, k23 = k(u1, u2, x1, x2), k(u1, u3, x1, x3), k(u2, u3, x2, x3)
    return 2 * alpha * lam ** 2 * (k13 * k23 + k12 * k23 + k12 * k13) \
        + 8 * alpha ** 3 * lam ** 3 * k12 * k23 * k13
