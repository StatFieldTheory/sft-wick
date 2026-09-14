"""Exact R-contracted fourth cumulant of demo2's deformed OU noise.

The twelve Hamiltonian paths and three cycles of the raw cumulant are
integrated over all 24 orderings of their four raw leg times.  Within an
ordering the kernel is an exponential; ordered_exponentials evaluates
its capped simplex integral without inner quadrature.  This replaces the
former 110592-node composite rule and its ~1e-3 typical/~1e-2 short-time
relative errors.  Partner-time coincidences are declared for outer cuts.

As for the third cumulant: gamma=1, t_min=0, all response legs absorbed.
The remaining two-dimensional FFK4 integral still needs refinement.
"""
from __future__ import annotations

import itertools
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "demo2"))
from ordered_exponentials import ordered_integral

LAM, SIGMA_T, SIGMA_X, GAMMA, ALPHA, N_COMP = 0.05, 0.3, 1.0, 1.0, 0.6, 2
_PATHS = sorted({tuple(p) if p[0] < p[-1] else tuple(reversed(p))
                 for p in itertools.permutations(range(4))})
_CYCLES = [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3)]
_PERMUTATIONS = tuple(itertools.permutations(range(4)))
_GRAPHS = ([tuple(zip(path[:-1], path[1:])) for path in _PATHS]
           + [tuple(zip(cycle, cycle[1:]+cycle[:1])) for cycle in _CYCLES])


def k4_R(t, s, *, lam=LAM, sigma_t=SIGMA_T, alpha=ALPHA, chunk=64):
    """K_R for times (4,n) and spatial factors s[(i,j)] (n,).

    chunk is retained for compatibility; the analytic implementation
    has no inner quadrature tensor to chunk.
    """
    t = np.asarray(t, float)
    if not np.isfinite(sigma_t) or sigma_t <= 0:
        raise ValueError("sigma_t must be positive and finite")
    valid = np.all(t > 0, axis=0)
    out = np.zeros(t.shape[1])
    if not np.any(valid):
        return out
    tt = t[:, valid]
    offset = -GAMMA*np.sum(tt, axis=0)
    for graph in _GRAPHS:
        spatial = np.ones(np.count_nonzero(valid))
        for i, j in graph:
            spatial *= np.broadcast_to(s[i, j], (t.shape[1],))[valid]
        total = np.zeros_like(spatial)
        for perm in _PERMUTATIONS:
            rank = {leg: i for i, leg in enumerate(perm)}
            rates = np.full(4, GAMMA)
            for i, j in graph:
                earlier, later = sorted((rank[i], rank[j]))
                rates[earlier] += 1/sigma_t
                rates[later] -= 1/sigma_t
            total += ordered_integral(tt[list(perm)], rates, offset)
        factor = 4*alpha**2*lam**3 if len(graph) == 3 else 16*alpha**4*lam**4
        out[valid] += factor*spatial*total
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


coupling_fn.has_coincident_time_kinks = True
coupling_fn_vectorized.has_coincident_time_kinks = True
