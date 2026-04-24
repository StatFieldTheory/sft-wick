"""``κ^{(3)}`` third cumulant for demo2 as a spacetime-dependent
callable.  Referenced by ``examples/demo2_config.yaml`` via
``nonlocal_vertices.coupling_module``.

Derived analytically from ``η̃ = η + α(η² − λ)`` with Gaussian η::

    κ^{(3)}_{abc}(1,2,3) = 2αλ² δ_{ab}δ_{bc}
                            · [κ(1,3)κ(2,3) + κ(1,2)κ(2,3)
                               + κ(1,2)κ(1,3)]

    κ(i, j) = exp(-|t_i-t_j|/σ_t) · exp(-|x_i-x_j|/σ_x)

The workflow loader imports ``coupling_fn`` as the non-local
vertex's callable; it gets invoked per QMC sample with the 3 ψ-leg
spacetime coordinates extracted from the diagram.
"""

from __future__ import annotations

import numpy as np

LAM = 0.05
SIGMA_T = 0.3
SIGMA_X = 1.0
ALPHA = 0.6
N_COMP = 2


def coupling_fn(n_list, t_list):
    """``κ^{(3)}_{abc}(1,2,3)`` at the 3 ψ-leg spacetime points."""
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
    amplitude = 2.0 * ALPHA * LAM ** 2 * bracket

    # Component structure δ_{ab}δ_{bc} — non-zero only on the
    # diagonal (a=b=c).
    K = np.zeros((N_COMP, N_COMP, N_COMP), dtype=float)
    for a in range(N_COMP):
        K[a, a, a] = amplitude
    return K
