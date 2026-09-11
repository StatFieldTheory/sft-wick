r"""Demo 5, part C: the exact reference for multiplicative white noise.

The state is ``(φ_0, φ_1, η_0, η_1)``:

* ``dφ_a = (−γ_a φ_a + η_a + ε F_abc φ_b φ_c) dt + Σ_k g_ak(φ) (∘) dW_k``,
  ``g_ak(φ) = g0_ak + μ g1_akb φ_b``;
* ``dη_a = −η_a/σ_t dt + √(2λ/σ_t) dB_a``, which makes ``η`` the stationary
  exponential noise ``⟨η_a(t) η_b(t')⟩ = δ_ab λ e^{−|t−t'|/σ_t}``; it starts
  in its stationary Gaussian law, ``φ`` at 0.

The generator is built from ``f`` and the columns ``g_k`` as written
(``examples/reference/hormander_moments.py``): for the Stratonovich reading
it is ``f·∇ + ½ Σ_k (g_k·∇)(g_k·∇)``, so the noise-induced drift is never
formed.  Tags ``(k_F, k_g)`` count powers of ``ε`` and ``μ``.  Imports no
sft-wick code; the time axis starts at 0 here and at ``t_min`` in the
package.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import hormander_moments as hm  # noqa: E402
import ito_moments as im  # noqa: E402

__all__ = ["MultiplicativeHierarchy"]


class MultiplicativeHierarchy:
    """Moments of ``φ`` order by order in ``(ε, μ)``.

    Args:
        gammas: ``(γ_0, γ_1)``.
        f_tensor: ``F``, shape ``(N, N, N)``.
        g0, g1: shapes ``(N, M)`` and ``(N, M, N)``, ``g1[a, k, b]``.
        lam, sigma_t: the coloured noise.
        interpretation: ``'ito'`` or ``'stratonovich'``.
    """

    def __init__(self, gammas, f_tensor, g0, g1, lam, sigma_t,
                 interpretation):
        g0, g1 = np.asarray(g0, float), np.asarray(g1, float)
        N, M = g0.shape
        self.N, self.D = N, 2 * N
        self.lam = float(lam)
        phi = list(range(N))
        sde = hm.VectorFieldSDE(D=2 * N, n_tags=2,
                                interpretation=interpretation)
        A = np.zeros((2 * N, 2 * N))
        for a in range(N):
            A[a, a] = -float(gammas[a])
            A[a, N + a] = 1.0
            A[N + a, N + a] = -1.0 / sigma_t
        sde.add_linear_drift(A)
        sde.add_quadratic_drift(np.asarray(f_tensor, float), tag=(1, 0),
                                coords=phi)
        for k in range(M):
            sde.columns.append(hm.affine_column(2 * N, 2, phi, g0[:, k],
                                                g1[:, k, :], (0, 1)))
        amp = np.sqrt(2.0 * lam / sigma_t)
        for a in range(N):
            sde.columns.append(hm.constant_column(2 * N, 2, {N + a: amp}))
        self.sde = sde

    def _initial(self, alpha, tag) -> float:
        """``φ(0) = 0``, ``η(0)`` stationary: only η-moments at tag 0."""
        if tag != (0, 0) or any(alpha[: self.N]):
            return 0.0
        legs = []
        for a in range(self.N):
            legs += [a] * alpha[self.N + a]
        return _gaussian_moment(tuple(legs), self.lam)

    def moments(self, comps, tags, T: float) -> dict:
        """``{tag: E[Π_i φ_{comps[i]}](T)}``."""
        mono = im.unit(self.D, *comps)
        out = hm.solve(self.sde, [(mono, tuple(t)) for t in tags], [T],
                       initial=self._initial)
        return {tuple(t): float(out[(mono, tuple(t))][0]) for t in tags}

    def two_time(self, a: int, b: int, tags, t1: float, t2: float) -> dict:
        """``{tag: E[φ_a(t1) φ_b(t2)]}``, ``t2 ≥ t1``."""
        return hm.two_time(self.sde, im.unit(self.D, a), im.unit(self.D, b),
                           [tuple(t) for t in tags], t1, t2,
                           initial=self._initial)


@lru_cache(maxsize=None)
def _gaussian_moment(legs, lam) -> float:
    """``E[Π η_a]`` for independent ``η_a ~ N(0, λ)`` (Wick)."""
    def cumulant(idx):
        if len(idx) != 2:
            return 0.0
        return lam if legs[idx[0]] == legs[idx[1]] else 0.0
    return im.moments_from_cumulants(cumulant, list(range(len(legs))))
