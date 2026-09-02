r"""Closed-form ``C`` propagator for demo 3, for the L2 (YAML) layer.

``C(z₁, z₂) = ν h² X₂(|x₁−x₂|) · T̃₂(t₁, t₂) · I_N`` --- i.e. exactly
``K_R`` at ``m = 2``, because with ``F = 0`` the two-point function *is*
the R-contracted second cumulant.  Machine precision; no quadrature.

**Why this module exists at all.**  Demo 3's ``κ²`` is genuinely
``SeparableTranslation`` with an exponential temporal kernel, which is
precisely the family the package's built-in closed form recognises --- so
in the L1 Python API (``examples/demo3/system.py``) nothing extra is
needed and ``Propagators.c_source`` reports ``closed_form:builtin``.  But
the L2 YAML schema's spatial-kernel builder
(``config.py::_build_kernel``, ``axis="space"``) accepts only
``exponential`` and ``gaussian``, and demo 3's envelope

    ``X₂(r) = σ_x (1 + r/σ_x) e^{−r/σ_x}``

is neither: convolving two exponential pulses leaves the extra linear
factor.  There is no ``callable_module`` escape hatch at the *kernel*
level (only for the whole ``κ²``, which would lower to
``GeneralKappa2`` and give up translation invariance).  So the YAML
declares ``κ²`` through ``kappa2_general.py`` and supplies ``C`` here,
with ``c_closed_form_only: true``.  See ``INTERPRETATION.md``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shot_noise import PARAMS, K_R  # noqa: E402

N_COMP = PARAMS.n_components


def C_fn(n1, t1, n2, t2):
    """Vectorised ``C``: scalars or ``(n,)`` arrays → ``(N, N)`` or ``(n, N, N)``."""
    n1a, t1a, n2a, t2a = (np.atleast_1d(np.asarray(v, dtype=float))
                          for v in (n1, t1, n2, t2))
    n1a, t1a, n2a, t2a = np.broadcast_arrays(n1a, t1a, n2a, t2a)
    amp = K_R(np.stack([n1a, n2a]), np.stack([t1a, t2a]), PARAMS)
    out = amp[:, None, None] * np.eye(N_COMP)[None, :, :]
    return out if np.ndim(n1) or np.ndim(t1) else out[0]
