"""Exact R-contracted third cumulant of demo2's deformed OU noise.

The raw cumulant is a sum of three two-edge trees (2 alpha lambda**2)
and one triangle (8 alpha**3 lambda**3).  For each graph, partition the
three raw leg times into their six orderings.  Every absolute value then
has a fixed sign, so all three integrals are elementary exponentials.
The upper limits are the PARTNER times supplied by already_R_contracted.

This removes the former composite quadrature's ~1e-4 typical and ~2.6e-3
early-time relative errors, and its 7056 inner quadrature nodes per sample.
The remaining outer diagram integral still needs a convergence check;
R contraction does not guarantee that GL8 or GL10 resolves t_final=50.

Contract: gamma=1, t_min=0, all three R legs absorbed.  Partner-time
coincidences change analytic branches; the callables declare them so the
package can split the outer integration domain there.
"""
from __future__ import annotations

from itertools import permutations
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ordered_exponentials import ordered_integral

LAM, SIGMA_T, SIGMA_X, GAMMA, ALPHA, N_COMP = 0.05, 0.3, 1.0, 1.0, 0.6, 2
_PERMUTATIONS = tuple(permutations(range(3)))
_GRAPHS = (((0, 1), (0, 2)), ((0, 1), (1, 2)),
           ((0, 2), (1, 2)), ((0, 1), (0, 2), (1, 2)))


def k3_R(t1, t2, t3, s12, s13, s23, *, lam=LAM, sigma_t=SIGMA_T, alpha=ALPHA):
    """K_R for broadcastable arrays of partner times and spatial factors."""
    if not np.isfinite(sigma_t) or sigma_t <= 0:
        raise ValueError("sigma_t must be positive and finite")
    arrays = np.broadcast_arrays(t1, t2, t3, s12, s13, s23)
    shape = arrays[0].shape
    t = np.asarray(arrays[:3], dtype=np.longdouble).reshape(3, -1)
    s12, s13, s23 = (np.asarray(v).ravel() for v in arrays[3:])
    spatial = (s12*s13, s12*s23, s13*s23, s12*s13*s23)
    valid = np.all(t > 0, axis=0)
    result = np.zeros(t.shape[1], dtype=np.longdouble)
    if not np.any(valid):
        return np.asarray(result, float).reshape(shape)
    t = t[:, valid]
    for graph, sp in zip(_GRAPHS, spatial):
        total = np.zeros(t.shape[1], dtype=np.longdouble)
        for perm in _PERMUTATIONS:
            rank = {leg: k for k, leg in enumerate(perm)}
            rates = np.full(3, GAMMA, dtype=np.longdouble)
            for i, j in graph:
                earlier, later = sorted((rank[i], rank[j]))
                rates[earlier] += 1 / sigma_t
                rates[later] -= 1 / sigma_t
            total += ordered_integral(t[list(perm)], rates, -GAMMA*np.sum(t, axis=0))
        factor = 8*alpha**3*lam**3 if len(graph) == 3 else 2*alpha*lam**2
        result[valid] += factor * sp[valid] * total
    return np.asarray(result, float).reshape(shape)


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


# The outer integrators use this existing callable contract.
coupling_fn.has_coincident_time_kinks = True
coupling_fn_vectorized.has_coincident_time_kinks = True
