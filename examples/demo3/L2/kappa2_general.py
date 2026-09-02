r"""``κ²`` for the L2 layer, as a general (n1, t1, n2, t2) callable.

``κ²(z₁, z₂) = ν h² (σ_t/2) e^{−|Δt|/σ_t} · X₂(r) · I_N``.

It *is* separable and translation invariant; it is declared through the
general escape hatch only because the YAML schema cannot express the
non-exponential spatial envelope (see :mod:`c_closed_form`).  ``C`` is
supplied in closed form alongside, so this callable is never integrated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shot_noise import PARAMS, kappa2_lam, kappa2_spatial  # noqa: E402

N_COMP = PARAMS.n_components


def kappa2(n1, t1, n2, t2):
    """``(N, N)`` matrix at one pair of spacetime points."""
    temporal = kappa2_lam(PARAMS) * np.exp(-abs(float(t1) - float(t2))
                                           / PARAMS.sigma_t)
    spatial = float(kappa2_spatial(float(n1) - float(n2), PARAMS))
    return temporal * spatial * np.eye(N_COMP)
