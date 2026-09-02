"""``κ^{(3)}`` third cumulant for demo2 as a spacetime-dependent
callable.  Referenced by ``examples/demo2_config.yaml`` via
``nonlocal_vertices.coupling_module``.

Derived analytically from ``η̃ = η + α(η² − λ)`` with Gaussian η::

    κ^{(3)}_{abc}(1,2,3) = δ_{ab}δ_{bc}
        · { 2αλ² [κ(1,3)κ(2,3) + κ(1,2)κ(2,3) + κ(1,2)κ(1,3)]
            + 8α³λ³ κ(1,2)κ(2,3)κ(1,3) }

    κ(i, j) = exp(-|t_i-t_j|/σ_t) · exp(-|x_i-x_j|/σ_x)

The ``α³`` term is the connected three-point function of ``η² − λ``
(``<δ1 δ2 δ3> = 8 λ³ κ12 κ23 κ13`` for Gaussian η); at coincident points
it is ``8α³λ³ / 6αλ² = (4/3) α² λ = 2.4 %`` of the leading term.  It was
missing from this module (and from the ``6αλ²`` line of the κ³
cross-check figure) until the CPC referee revision; the simulated third
moment of ``η̃`` is ``9.21e-3``, which is ``6αλ² + 8α³λ³``, not
``6αλ² = 9.00e-3``.

For the FK channel prefer the R-contracted form in
``k3_R_coupling.py`` (``already_R_contracted: true``): the raw kernel is
narrow in the relative leg times, so a tensor-product quadrature on the
raw 4-D integral stops converging beyond ``t ≈ 10``.

This module exposes both supported contracts:

* :func:`coupling_fn` -- per-sample contract; expects length-m
  ``n_list`` / ``t_list`` and returns ``(N, N, N)``.  Reference
  this from YAML with ``coupling_attr: coupling_fn`` and **no**
  ``coupling_vectorized`` (or ``false``).
* :func:`coupling_fn_vectorized` -- batched contract; expects
  ``(m, n_samples)`` ``n_list`` / ``t_list`` and returns
  ``(n_samples, N, N, N)``.  Reference this with
  ``coupling_attr: coupling_fn_vectorized`` *and*
  ``coupling_vectorized: true`` so the workflow knows to call it
  with batched arguments.

Both produce identical numerics; pick whichever fits the cost
profile.  See ``docs/user_guide/workflow.rst`` for the contract
trade-off.
"""

from __future__ import annotations

import numpy as np

LAM = 0.05
SIGMA_T = 0.3
SIGMA_X = 1.0
ALPHA = 0.6
N_COMP = 2


def coupling_fn(n_list, t_list):
    """``κ^{(3)}_{abc}(1,2,3)`` at the 3 ψ-leg spacetime points
    (per-sample contract)."""
    n = np.asarray(n_list, dtype=float)
    t = np.asarray(t_list, dtype=float)

    def kappa(i, j):
        return (
            np.exp(-abs(t[i] - t[j]) / SIGMA_T)
            * np.exp(-abs(n[i] - n[j]) / SIGMA_X)
        )

    bracket = (
        kappa(0, 2) * kappa(1, 2)
        + kappa(0, 1) * kappa(1, 2)
        + kappa(0, 1) * kappa(0, 2)
    )
    amplitude = (2.0 * ALPHA * LAM ** 2 * bracket
                 + 8.0 * ALPHA ** 3 * LAM ** 3 * kappa(0, 1) * kappa(1, 2) * kappa(0, 2))

    # Component structure δ_{ab}δ_{bc} — non-zero only on the
    # diagonal (a=b=c).
    K = np.zeros((N_COMP, N_COMP, N_COMP), dtype=float)
    for a in range(N_COMP):
        K[a, a, a] = amplitude
    return K


def coupling_fn_vectorized(n_2d, t_2d):
    """Batched contract: same physics as :func:`coupling_fn`, but
    accepts ``(m, n_samples)`` shaped inputs and returns
    ``(n_samples, N, N, N)``.

    Vectorising the per-sample bracket via ufuncs cuts the per-call
    cost roughly in half on demo2's modest workload; the win scales
    with how heavy the user kernel becomes.
    """
    n = np.asarray(n_2d, dtype=float)  # (3, n_samples)
    t = np.asarray(t_2d, dtype=float)  # (3, n_samples)

    def kappa(i, j):
        return (
            np.exp(-np.abs(t[i] - t[j]) / SIGMA_T)
            * np.exp(-np.abs(n[i] - n[j]) / SIGMA_X)
        )

    bracket = (
        kappa(0, 2) * kappa(1, 2)
        + kappa(0, 1) * kappa(1, 2)
        + kappa(0, 1) * kappa(0, 2)
    )  # (n_samples,)
    amplitude = (2.0 * ALPHA * LAM ** 2 * bracket
                 + 8.0 * ALPHA ** 3 * LAM ** 3 * kappa(0, 1) * kappa(1, 2) * kappa(0, 2))

    # Build (n_samples, N, N, N) with the same δ_{ab}δ_{bc} sparsity.
    K = np.zeros((amplitude.shape[0], N_COMP, N_COMP, N_COMP), dtype=float)
    for a in range(N_COMP):
        K[:, a, a, a] = amplitude
    return K
