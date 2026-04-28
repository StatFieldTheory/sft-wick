"""Closed-form C propagator for demo2.

Two flavours, both vectorised (return ``(n, N, N)`` for batched
``(t1, t2, x1, x2)`` inputs):

* :func:`C_fn_eff` -- noise variance ``lambda_eff = lambda *
  (1 + 2 * alpha^2 * lambda)``. Used by the **FF channel** to
  absorb the leading O(alpha^2) variance shift into a renormalised
  Gaussian kernel.
* :func:`C_fn_bare` -- bare noise variance ``lambda``. Used by
  the **FK channel** because the alpha factor is already carried by
  the explicit non-local ``kappa^{(3)}`` vertex in K, so the C
  propagators inside the diagram must use the unshifted kernel.

Mirrors the notebook's ``cache_eff`` / ``cache`` split. Both
functions use the standard demo1 OU integral.
"""
from __future__ import annotations

import numpy as np

LAM = 0.05
SIGMA_T = 0.3
SIGMA_X = 1.0
GAMMA = 1.0
N_COMP = 2
ALPHA = 0.6
LAM_EFF = LAM * (1.0 + 2.0 * ALPHA**2 * LAM)


def _C_t_batch(t1: np.ndarray, t2: np.ndarray, lam: float) -> np.ndarray:
    """Vectorised analytical OU C(t1, t2; lam)."""
    g = GAMMA
    a = 1.0 / SIGMA_T
    t_lo = np.minimum(t1, t2)
    t_hi = np.maximum(t1, t2)
    gpa = g + a
    gma = g - a
    pos = t_lo > 0
    safe_lo = np.where(pos, t_lo, 1.0)
    E1 = np.expm1(2 * g * safe_lo) / (2 * g)
    if abs(gma) < 1e-14:
        E2 = safe_lo
    else:
        E2 = np.expm1(gma * safe_lo) / gma
    E3 = np.expm1(gpa * safe_lo) / gpa
    E4 = np.exp(gma * t_hi)
    I = E1 / gpa - E2 / gpa + E4 * E3 / gma - E1 / gma
    val = lam * np.exp(-g * (t1 + t2)) * I
    return np.where(pos, val, 0.0)


def _C_fn_template(n1, t1, n2, t2, lam: float) -> np.ndarray:
    t1 = np.atleast_1d(np.asarray(t1, dtype=float))
    t2 = np.atleast_1d(np.asarray(t2, dtype=float))
    n1_arr = np.broadcast_to(np.asarray(n1, dtype=float), t1.shape)
    n2_arr = np.broadcast_to(np.asarray(n2, dtype=float), t1.shape)
    r = np.abs(n1_arr - n2_arr)
    c_t = _C_t_batch(t1, t2, lam)
    spatial = np.exp(-r / SIGMA_X)
    diag = c_t * spatial
    return diag[:, None, None] * np.eye(N_COMP)[None, :, :]


def C_fn_eff(n1, t1, n2, t2):
    """C with noise variance ``LAM_EFF`` (FF channel)."""
    return _C_fn_template(n1, t1, n2, t2, LAM_EFF)


def C_fn_bare(n1, t1, n2, t2):
    """C with bare noise variance ``LAM`` (FK channel)."""
    return _C_fn_template(n1, t1, n2, t2, LAM)
