"""Closed-form C propagators for demo2.

Three flavours, all vectorised (return ``(n, N, N)`` for batched
``(t1, t2, x1, x2)`` inputs):

* :func:`C_fn_eff_exact` -- the EXACT effective covariance of the
  deformed noise, ``kappa2_eff = lam k + 2 alpha^2 lam^2 k^2`` with
  ``k = exp(-|dt|/sigma_t) exp(-|dx|/sigma_x)``.  Its second piece has
  half the correlation time and length, and being separable-exponential
  it has the package's built-in closed form too, so ``C_eff`` is the sum
  of two :class:`~sft_wick.workflow.closed_forms.ClosedFormC` objects.
  Used by ``config_FF.yaml`` (the **FF channel** and order 0).
* :func:`C_fn_eff` -- the single-kernel approximation
  ``lambda_eff = lambda (1 + 2 alpha^2 lambda)`` the FF channel used
  until the CPC referee revision.  Exact at coincident points only; it
  over-counts the second piece's contribution to C by 70 %, i.e.
  +1.8e-4 on xi_00 at large t (1.5 % of the signal).  Kept for
  comparison.
* :func:`C_fn_bare` -- bare noise variance ``lambda``.  The **FK
  channel** has no C propagator, so this is only used by scripts that
  want the un-deformed C.
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


def _exact_pieces():
    from sft_wick.workflow.closed_forms import ClosedFormC
    from sft_wick.workflow.specs import ExponentialSpatial
    a = ClosedFormC(gamma=(GAMMA,) * N_COMP, lam=LAM, sigma_t=SIGMA_T,
                    spatial=ExponentialSpatial(sigma_x=SIGMA_X))
    b = ClosedFormC(gamma=(GAMMA,) * N_COMP, lam=2.0 * ALPHA ** 2 * LAM ** 2,
                    sigma_t=SIGMA_T / 2.0, spatial=ExponentialSpatial(sigma_x=SIGMA_X / 2.0))
    return a, b


_EXACT = None


def C_fn_eff_exact(n1, t1, n2, t2):
    """C with the exact two-kernel effective covariance of the deformed noise."""
    global _EXACT
    if _EXACT is None:
        _EXACT = _exact_pieces()
    a, b = _EXACT
    out = a(n1, t1, n2, t2) + b(n1, t1, n2, t2)
    return out if np.ndim(t1) else out[None]
