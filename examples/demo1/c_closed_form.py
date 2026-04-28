"""Closed-form C propagator for demo1's separable OU kernel.

Referenced by ``examples/demo1_config.yaml`` via
``propagators.c_closed_form_module``; the workflow loader imports
``C_fn`` from this module and plugs it into the cache-build path.
"""

from __future__ import annotations

import numpy as np

LAM = 0.05
SIGMA_T = 0.3
SIGMA_X = 1.0
GAMMA = 1.0
N_COMP = 2


def _C_t(t1: float, t2: float) -> float:
    a = 1.0 / SIGMA_T
    tl, th = (t1, t2) if t1 <= t2 else (t2, t1)
    if tl <= 0:
        return 0.0
    gpa, gma = GAMMA + a, GAMMA - a
    E1 = np.expm1(2 * GAMMA * tl) / (2 * GAMMA)
    E2 = tl if abs(gma) < 1e-14 else np.expm1(gma * tl) / gma
    E3 = np.expm1(gpa * tl) / gpa
    E4 = np.exp(gma * th)
    I = E1 / gpa - E2 / gpa + E4 * E3 / gma - E1 / gma
    return LAM * np.exp(-GAMMA * (t1 + t2)) * I


def C_fn(n1, t1, n2, t2):
    """Per-sample contract: scalar ``t1, t2``; scalar / 1-D ``n1, n2``.
    Returns ``(N, N)`` matrix at that single spacetime pair.

    Suitable for the spline-build path (``c_closed_form_module``
    + ``c_closed_form_attr: C_fn``).
    """
    r = abs(float(np.asarray(n1).sum()) - float(np.asarray(n2).sum()))
    return _C_t(t1, t2) * np.exp(-r / SIGMA_X) * np.eye(N_COMP)


def _C_t_batch(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Vectorised analogue of :func:`_C_t`. Accepts ``(n,)`` arrays
    of times, returns ``(n,)`` array of scalar C(t1, t2) values.
    """
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
    val = LAM * np.exp(-g * (t1 + t2)) * I
    return np.where(pos, val, 0.0)


def C_fn_vec(n1, t1, n2, t2):
    """Vectorised counterpart of :func:`C_fn`.

    Batched contract: ``t1, t2`` are ``(n,)``-shape arrays;
    ``n1, n2`` may be scalars, ``(n,)`` arrays, or broadcast-
    compatible. Returns ``(n, N, N)`` -- the full C matrix for
    each sample. Required when wiring this module via
    ``c_closed_form_only: true`` + ``c_closed_form_vectorized: true``
    (the no-spline path).

    For demo1's diagonal C the only non-zero entries are
    ``C[i, a, a] = c_t * spatial[i]``; off-diagonal entries are
    identically zero, so the result is
    ``diag[:, None, None] * np.eye(N)[None, :, :]``.
    """
    t1 = np.atleast_1d(np.asarray(t1, dtype=float))
    t2 = np.atleast_1d(np.asarray(t2, dtype=float))
    n1_arr = np.broadcast_to(np.asarray(n1, dtype=float), t1.shape)
    n2_arr = np.broadcast_to(np.asarray(n2, dtype=float), t1.shape)
    r = np.abs(n1_arr - n2_arr)
    c_t = _C_t_batch(t1, t2)
    spatial = np.exp(-r / SIGMA_X)
    diag = c_t * spatial
    return diag[:, None, None] * np.eye(N_COMP)[None, :, :]
