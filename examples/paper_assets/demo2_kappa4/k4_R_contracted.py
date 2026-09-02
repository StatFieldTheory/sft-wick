"""The R-contracted fourth cumulant of demo2's deformed noise.

Same construction as :mod:`k3_R_contracted` one order up.  With
``u_i = u + v_i`` (i = 1, 2, 3) and ``u_4 = u``, the four retarded factors
integrate to ``(E(u_hi) − E(u_lo)) / (4γ)`` with

    u_lo = max(0, −v1, −v2, −v3),   u_hi = min(t4', t1' − v1, t2' − v2, t3' − v3),
    E(u) = exp(−γ Σ_i (t_i' − v_i − u) − γ (t4' − u)),

and ``κ^(4)(v)`` (see ``k4_coupling.py``) is a sum of 12 Hamiltonian-path
terms ``k_ab k_bc k_cd`` and 3 Hamiltonian-cycle terms.  Each path term is
integrated in the coordinates ``(v_a − v_b, v_b − v_c, v_c − v_d)`` -- a
unimodular change of variables that puts all three of its cusps on the
coordinate planes -- with a graded composite Gauss-Legendre rule; a
cycle term keeps its fourth cusp unaligned (the cycle terms are
``α² λ = 1.8 %`` of the amplitude).  Accuracy ~1e-3 relative (validated
against QMC in ``validate_k4_R.py``), ample for a channel whose size is
below the Monte-Carlo error of the simulation it is compared with.
"""
from __future__ import annotations

import itertools

import numpy as np
from numpy.polynomial.legendre import leggauss

LAM, SIGMA_T, SIGMA_X, GAMMA, ALPHA, N_COMP = 0.05, 0.3, 1.0, 1.0, 0.6, 2

_EDGES = SIGMA_T * np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 14.0])
_N_GL = 4


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
_P, _Q, _R = (a.ravel() for a in np.meshgrid(_G1, _G1, _G1, indexing="ij"))
_W3 = np.einsum("i,j,k->ijk", _W1, _W1, _W1).ravel()
_EXP = np.exp(-(np.abs(_P) + np.abs(_Q) + np.abs(_R)) / SIGMA_T)

_PATHS = sorted({
    tuple(p) if p[0] < p[-1] else tuple(reversed(p))
    for p in itertools.permutations(range(4))
})
_CYCLES = [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3)]


def _v_from_path(path, p, q, r):
    """``(v0, v1, v2, v3)`` with ``v3 = 0`` from the path's edge differences."""
    a, b, c, d = path
    v = [None] * 4
    if d == 3:
        v[c], v[b], v[a] = r, q + r, p + q + r
    elif c == 3:
        v[d], v[b], v[a] = -r, q, p + q
    elif b == 3:
        v[c], v[d], v[a] = -q, -q - r, p
    else:  # a == 3
        v[b], v[c], v[d] = -p, -p - q, -p - q - r
    v[3] = np.zeros_like(p)
    return v


def _G_factor(v1, v2, v3, t1, t2, t3, t4, gamma=GAMMA):
    u_lo = np.maximum(0.0, np.maximum(-v1, np.maximum(-v2, -v3)))
    u_hi = np.minimum(t4, np.minimum(t1 - v1, np.minimum(t2 - v2, t3 - v3)))
    ok = u_hi > u_lo
    uh = np.where(ok, u_hi, 0.0)
    ul = np.where(ok, u_lo, 0.0)
    a = -gamma * ((t1 - v1 - uh) + (t2 - v2 - uh) + (t3 - v3 - uh) + (t4 - uh))
    b = -gamma * ((t1 - v1 - ul) + (t2 - v2 - ul) + (t3 - v3 - ul) + (t4 - ul))
    a = np.minimum(a, 0.0)
    b = np.minimum(b, 0.0)
    return np.where(ok, (np.exp(a) - np.exp(b)) / (4.0 * gamma), 0.0)


def k4_R(t, s, *, lam=LAM, sigma_t=SIGMA_T, alpha=ALPHA, chunk=64):
    """``K_R`` for partner times ``t`` (4, n) and spatial factors ``s[(i,j)]`` (n,)."""
    t = np.asarray(t, float)
    n = t.shape[1]
    out = np.zeros(n)
    c2 = 4.0 * alpha ** 2 * lam ** 3
    c4 = 16.0 * alpha ** 4 * lam ** 4
    for lo in range(0, n, chunk):
        sl = slice(lo, lo + chunk)
        tt = [t[i, sl][:, None] for i in range(4)]
        acc = np.zeros(tt[0].shape[0])
        for path in _PATHS:
            a, b, c, d = path
            sp = (s[(a, b)][sl] * s[(b, c)][sl] * s[(c, d)][sl])[:, None]
            v = _v_from_path(path, _P[None, :], _Q[None, :], _R[None, :])
            g = _G_factor(v[0], v[1], v[2], *tt)
            acc += c2 * np.sum(_W3[None, :] * _EXP[None, :] * sp * g, axis=1)
        for cyc in _CYCLES:
            a, b, c, d = cyc
            sp = (s[(a, b)][sl] * s[(b, c)][sl] * s[(c, d)][sl] * s[(d, a)][sl])[:, None]
            v = _v_from_path(cyc, _P[None, :], _Q[None, :], _R[None, :])
            extra = np.exp(-np.abs(v[d] - v[a]) / sigma_t)
            g = _G_factor(v[0], v[1], v[2], *tt)
            acc += c4 * np.sum(_W3[None, :] * _EXP[None, :] * extra * sp * g, axis=1)
        out[sl] = acc
    return out


def _spatial_factors(n_2d, sigma_x=SIGMA_X):
    n = np.asarray(n_2d, float)
    s = {}
    for i in range(4):
        for j in range(4):
            if i != j:
                s[(i, j)] = np.exp(-np.abs(n[i] - n[j]) / sigma_x)
    return s


def coupling_fn_vectorized(n_2d, t_2d):
    """Batched contract: ``(4, n)`` partner positions / times → ``(n, N, N, N, N)``."""
    amp = k4_R(np.asarray(t_2d, float), _spatial_factors(n_2d))
    K = np.zeros((amp.shape[0],) + (N_COMP,) * 4)
    for a in range(N_COMP):
        K[:, a, a, a, a] = amp
    return K


def coupling_fn(n_list, t_list):
    n = np.asarray(n_list, float)[:, None]
    t = np.asarray(t_list, float)[:, None]
    return coupling_fn_vectorized(n, t)[0]
