r"""Exact moments for a quadratic drift that decays in time.

The reference behind ``tests/test_local_callable_coupling.py``, which checks
a callable (time-dependent) local coupling.  The field obeys, for
``t ≥ t_min`` from ``φ(t_min) = 0``,

.. math::

    dφ_a = \bigl(−γ_a φ_a + F_{abc}(x, t)\, φ_b φ_c + η_a\bigr)\,dt + dW_a,
    \qquad F_{abc}(x, t) = F^0_{abc}\, (1 + β x)\, e^{−λ t},

with the noise of demo 5: ``η`` coloured,
``⟨η_a(x,t) η_b(y,t')⟩ = δ_ab λ_η e^{−|t−t'|/σ_t} e^{−(x−y)^2/(2σ_x^2)}``
and stationary; ``dW`` white, ``⟨dW_a dW_b⟩ = S_ab dt``, the same at every
point.

A deterministic state ``u`` with ``du = −λ u dt`` and ``u(t_min) =
e^{−λ t_min}`` equals ``e^{−λ t}``, so the drift ``u F^0 (1 + β x) φ φ`` is a
polynomial in the state ``(φ_{a,i}, η_{a,i}, u)`` at the observation points
``x_1 … x_P``, with

* ``dφ_{a,i} = (−γ_a φ_{a,i} + η_{a,i} + ε (1 + β x_i) u F^0_{abc} φ_{b,i}
  φ_{c,i}) dt + dW_a``;
* ``dη_{a,i} = −η_{a,i}/σ_t dt + dB_{a,i}``,
  ``⟨dB_{a,i} dB_{b,j}⟩ = δ_ab (2λ_η/σ_t) e^{−(x_i−x_j)^2/(2σ_x^2)} dt``,
  started in its stationary Gaussian law;
* ``du = −λ u dt``.

Its moment hierarchy (:mod:`ito_moments`), solved order by order in ``ε``,
gives the coefficient of ``ε^k`` of every moment, which is the package's
order ``k``.  The time axis starts at 0 here and at ``t_min`` in the
package.  Imports no sft-wick code.
"""
from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ito_moments as im  # noqa: E402

__all__ = ["DecayParams", "PARAMS", "PARAMS_MATRIX_R", "F0_TENSOR",
           "DecayHierarchy"]

#: ``F^0_{abc}``: no index symmetry, not even in the last two indices,
#: and both signs.
F0_TENSOR = ((( 0.35, -0.20), ( 0.15,  0.40)),
             ((-0.30,  0.25), ( 0.45, -0.10)))


@dataclass(frozen=True)
class DecayParams:
    """Model parameters.

    Args:
        gamma: linear decay rate of each component.  Equal rates give the
            package a scalar R; distinct rates a diagonal matrix R.
        decay: ``λ``, the decay rate of the drift.
        beta: ``β``, the slope of the drift's position dependence.
        lam, sigma_t, sigma_x: amplitude, correlation time and spatial width
            of the coloured noise.
        S: white-noise covariance rate (mixes the components).
        t_min: start time.
        f0: ``F^0``.
    """

    gamma: tuple = (1.0, 1.0)
    decay: float = 0.7
    beta: float = 0.0
    lam: float = 0.4
    sigma_t: float = 0.6
    sigma_x: float = 0.9
    S: tuple = ((0.50, 0.20), (0.20, 0.35))
    t_min: float = 0.5
    f0: tuple = F0_TENSOR

    @property
    def n_components(self) -> int:
        return len(self.gamma)

    @property
    def F0(self) -> np.ndarray:
        return np.asarray(self.f0, dtype=float)

    def drift_factor(self, x: float, t: float) -> float:
        """``(1 + β x) e^{−λ t}``, the factor multiplying ``F^0``."""
        return (1.0 + self.beta * x) * np.exp(-self.decay * t)


#: Equal rates (scalar R): every integrator of the package applies.
PARAMS = DecayParams()
#: Distinct rates (diagonal matrix R): the scalar loops of the package.
PARAMS_MATRIX_R = replace(PARAMS, gamma=(0.9, 1.4))


class DecayHierarchy:
    """Moment hierarchy of the embedding at ``positions`` (one per point)."""

    def __init__(self, p: DecayParams, positions):
        self.p = p
        self.positions = tuple(float(x) for x in positions)
        N, P = p.n_components, len(self.positions)
        self.N, self.P = N, P
        self.D = 2 * N * P + 1
        self._phi = {(a, i): a * P + i for a in range(N) for i in range(P)}
        self._eta = {(a, i): N * P + a * P + i
                     for a in range(N) for i in range(P)}
        self._u = 2 * N * P
        x = np.asarray(self.positions)
        self._K = np.exp(-(x[:, None] - x[None, :]) ** 2
                         / (2.0 * p.sigma_x ** 2))

        sde = im.PolySDE(D=self.D, n_tags=1)
        A = np.zeros((self.D, self.D))
        for (a, i), k in self._phi.items():
            A[k, k] = -p.gamma[a]
            A[k, self._eta[(a, i)]] = 1.0
        for k in self._eta.values():
            A[k, k] = -1.0 / p.sigma_t
        A[self._u, self._u] = -p.decay
        sde.add_linear_drift(A)

        S = np.zeros((self.D, self.D))
        Smat = np.asarray(p.S, dtype=float)
        for (a, _i), k in self._phi.items():
            for (b, _j), m in self._phi.items():
                S[k, m] = Smat[a, b]
        for (a, i), k in self._eta.items():
            for (b, j), m in self._eta.items():
                if a == b:
                    S[k, m] = 2.0 * p.lam / p.sigma_t * self._K[i, j]
        sde.add_constant_diffusion(S)

        F0 = p.F0
        for i in range(P):
            g_i = 1.0 + p.beta * self.positions[i]
            for a, b, c in itertools.product(range(N), repeat=3):
                if F0[a, b, c] == 0.0:
                    continue
                mono = im.unit(self.D, self._u, self._phi[(b, i)],
                               self._phi[(c, i)])
                sde.drift.append((self._phi[(a, i)], float(F0[a, b, c] * g_i),
                                  mono, (1,)))
        self.sde = sde
        self._u0 = float(np.exp(-p.decay * p.t_min))

    def _initial(self, alpha, tag) -> float:
        if tag != (0,):
            return 0.0
        if any(alpha[k] for k in self._phi.values()):
            return 0.0
        legs = []
        for key, k in self._eta.items():
            legs += [key] * alpha[k]
        return (self._u0 ** alpha[self._u]
                * _gaussian_moment(tuple(legs), self.p.lam,
                                   tuple(map(tuple, self._K))))

    def moments(self, legs, orders, T: float) -> dict:
        """``{k: E[Π φ]^{(ε^k)}}`` at package time ``t_min + T``, for legs
        ``[(a, i), …]``."""
        mono = im.monomial_of(self.D, [self._phi[leg] for leg in legs])
        out = im.solve(self.sde, [(mono, (k,)) for k in orders], [T],
                       initial=self._initial)
        return {k: float(out[(mono, (k,))][0]) for k in orders}


@lru_cache(maxsize=None)
def _gaussian_moment(legs, lam, K) -> float:
    """``E[Π η]`` for the stationary Gaussian ``η``: Wick's theorem with
    ``⟨η_{a,i} η_{b,j}⟩ = δ_ab λ K_ij``."""
    def cumulant(idx):
        if len(idx) != 2:
            return 0.0
        (a, i), (b, j) = legs[idx[0]], legs[idx[1]]
        return lam * K[i][j] if a == b else 0.0

    return im.moments_from_cumulants(cumulant, list(range(len(legs))))
