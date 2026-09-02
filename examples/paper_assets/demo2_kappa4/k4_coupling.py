"""The fourth cumulant ``kappa^(4)`` of demo2's deformed noise.

With ``eta`` Gaussian, ``<eta_i eta_j> = lam k_ij``,
``k_ij = exp(-|t_i - t_j|/sigma_t) exp(-|x_i - x_j|/sigma_x)``, and
``eta_tilde = eta + alpha (eta^2 - lam)``, the connected four-point
function of ``eta_tilde`` is (Wick on Gaussian ``eta``; only even powers
of ``alpha`` survive because odd ones leave an odd number of ``eta`` legs):

    kappa^(4)(1,2,3,4) = 4 alpha^2 lam^3  sum over the 12 Hamiltonian
                                              paths a-b-c-d of K4 of
                                              k_ab k_bc k_cd
                       + 16 alpha^4 lam^4 sum over the 3 Hamiltonian
                                              cycles a-b-c-d-a of
                                              k_ab k_bc k_cd k_da

The ``alpha^2`` term: the two ``delta = eta^2 - lam`` factors sit at the
interior vertices of the path (2 x 2 leg choices = 4 pairings per path);
the ``alpha^4`` term: every vertex carries a ``delta``, and a connected
pairing of four 2-leg vertices is a 4-cycle (2 leg choices per vertex =
16 pairings per cycle).  Single-site check (all ``k = 1``): the cumulant
of ``eta^2 = lam chi^2_1`` is ``48 lam^4``, and ``3 cycles x 16 = 48``.

Component structure: ``delta_ab delta_bc delta_cd`` (one component per
site), exactly as for ``kappa^(3)`` in ``examples/demo2/k3_coupling.py``.
This module exposes the batched contract
(``coupling_vectorized: true``): ``fn((4, n), (4, n)) -> (n, N, N, N, N)``.
"""
from __future__ import annotations

import itertools

import numpy as np

LAM = 0.05
SIGMA_T = 0.3
SIGMA_X = 1.0
ALPHA = 0.6
N_COMP = 2

# The 12 undirected Hamiltonian paths and 3 Hamiltonian cycles of K4.
_PATHS = sorted({
    tuple(p) if p[0] < p[-1] else tuple(reversed(p))
    for p in itertools.permutations(range(4))
})
_CYCLES = [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3)]
assert len(_PATHS) == 12


def kappa4_amplitude(n_2d, t_2d, *, lam=LAM, sigma_t=SIGMA_T, sigma_x=SIGMA_X,
                     alpha=ALPHA):
    """``kappa^(4)`` at four spacetime points, batched: inputs ``(4, n)``."""
    n = np.asarray(n_2d, dtype=float)
    t = np.asarray(t_2d, dtype=float)

    def k(i, j):
        return np.exp(-np.abs(t[i] - t[j]) / sigma_t) * np.exp(-np.abs(n[i] - n[j]) / sigma_x)

    paths = sum(k(a, b) * k(b, c) * k(c, d) for a, b, c, d in _PATHS)
    cycles = sum(k(a, b) * k(b, c) * k(c, d) * k(d, a) for a, b, c, d in _CYCLES)
    return 4.0 * alpha ** 2 * lam ** 3 * paths + 16.0 * alpha ** 4 * lam ** 4 * cycles


def coupling_fn_vectorized(n_2d, t_2d):
    """Batched contract for ``NonLocalVertex(order=4, coupling_vectorized=True)``."""
    amp = kappa4_amplitude(n_2d, t_2d)                      # (n,)
    K = np.zeros((amp.shape[0],) + (N_COMP,) * 4)
    for a in range(N_COMP):
        K[:, a, a, a, a] = amp
    return K


def coupling_fn(n_list, t_list):
    """Per-sample contract (length-4 inputs) -> ``(N, N, N, N)``."""
    n = np.asarray(n_list, dtype=float)[:, None]
    t = np.asarray(t_list, dtype=float)[:, None]
    return coupling_fn_vectorized(n, t)[0]


def single_site_cumulants(alpha=ALPHA, lam=LAM):
    """Analytic ``(kappa2, kappa3, kappa4)`` of ``eta_tilde`` at one point."""
    return (lam + 2 * alpha ** 2 * lam ** 2,
            6 * alpha * lam ** 2 + 8 * alpha ** 3 * lam ** 3,
            48 * alpha ** 2 * lam ** 3 + 48 * alpha ** 4 * lam ** 4)
