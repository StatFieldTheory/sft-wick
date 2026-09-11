r"""Demo 5's exact reference: the moment hierarchy of the Markov embedding.

At observation points ``x_1 … x_P`` the state is ``(φ_{a,i}, η_{a,i})``:

* ``dφ_{a,i} = (−γ φ_{a,i} + η_{a,i} + ε F_abc φ_{b,i} φ_{c,i}) dt + dW_a``,
  the white noise shared by every point: ``⟨dW_a dW_b⟩ = S_ab dt``;
* ``dη_{a,i} = −η_{a,i}/σ_t dt + dB_{a,i}`` with
  ``⟨dB_{a,i} dB_{b,j}⟩ = δ_ab (2λ/σ_t) e^{−(x_i−x_j)^2/(2σ_x^2)} dt``,
  which makes ``η`` the stationary exponential noise; it starts in its
  stationary Gaussian law, ``φ`` at 0.

The time axis starts at 0 here and at ``t_min`` in the package.  Imports no
sft-wick code; it takes the parameters as an argument.
"""
from __future__ import annotations

import itertools
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import ito_moments as im  # noqa: E402

__all__ = ["Hierarchy"]


class Hierarchy:
    def __init__(self, p, positions, f_tensor):
        self.p = p
        self.positions = tuple(float(x) for x in positions)
        N, P = p.n_components, len(self.positions)
        self.N, self.P = N, P
        self.D = 2 * N * P
        self._phi = {(a, i): a * P + i for a in range(N) for i in range(P)}
        self._eta = {(a, i): N * P + a * P + i
                     for a in range(N) for i in range(P)}
        K = self._spatial()
        sde = im.PolySDE(D=self.D, n_tags=1)
        A = np.zeros((self.D, self.D))
        for key, k in self._phi.items():
            A[k, k] = -p.gamma
            A[k, self._eta[key]] = 1.0
        for k in self._eta.values():
            A[k, k] = -1.0 / p.sigma_t
        sde.add_linear_drift(A)
        S = np.zeros((self.D, self.D))
        Smat = np.asarray(p.S, dtype=float)
        for (a, i), k in self._phi.items():
            for (b, j), l in self._phi.items():
                S[k, l] = Smat[a, b]
        for (a, i), k in self._eta.items():
            for (b, j), l in self._eta.items():
                if a == b:
                    S[k, l] = 2.0 * p.lam / p.sigma_t * K[i, j]
        sde.add_constant_diffusion(S)
        Q = np.zeros((self.D,) * 3)
        for i in range(P):
            for a, b, c in itertools.product(range(N), repeat=3):
                Q[self._phi[(a, i)], self._phi[(b, i)],
                  self._phi[(c, i)]] += f_tensor[a, b, c]
        sde.add_quadratic_drift(Q, tag=(1,))
        self.sde = sde
        self._K = K

    def _spatial(self) -> np.ndarray:
        x = np.asarray(self.positions)
        return np.exp(-(x[:, None] - x[None, :]) ** 2
                      / (2.0 * self.p.sigma_x ** 2))

    def _initial(self, alpha, tag) -> float:
        zero = (0,) * self.D
        if tag != (0,):
            return 0.0
        if alpha == zero:
            return 1.0
        if any(alpha[k] for k in self._phi.values()):
            return 0.0
        legs = []
        for key, k in self._eta.items():
            legs += [key] * alpha[k]
        return _gaussian_moment(tuple(legs), self.p.lam,
                                tuple(map(tuple, self._K)))

    def moments(self, legs, orders, T: float) -> dict:
        """``{k: E[Π φ]^{(ε^k)}(T)}`` for legs ``[(a, i), …]``."""
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
