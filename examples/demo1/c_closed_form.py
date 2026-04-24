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
    """``PropagatorCache._C_value_direct`` drop-in for the separable OU
    kernel used by demo1 and demo2's FF channel."""
    r = abs(float(np.asarray(n1).sum()) - float(np.asarray(n2).sum()))
    return _C_t(t1, t2) * np.exp(-r / SIGMA_X) * np.eye(N_COMP)
