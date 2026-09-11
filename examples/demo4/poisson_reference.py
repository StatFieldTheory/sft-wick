r"""Demo 4's exact reference: the moment hierarchy at the observation points.

The drift has no spatial coupling, so the fields at a finite set of points
``x_1 … x_P`` form a closed Markov process driven by the events:

* white pulses: the state is ``φ_{a,i}``; an event at ``z`` moves it by
  ``h_a w_a(x_i − z)``;
* exponential pulses: the state is ``(φ_{a,i}, η_{a,i})`` with
  ``dη_{a,i} = −η_{a,i}/τ_a dt`` plus the jumps and
  ``dφ_{a,i} = (−γ φ_{a,i} + η_{a,i}) dt``; ``η`` starts in its stationary
  law, whose cumulants are Campbell's.

Both carry ``ε F_abc φ_{b,i} φ_{c,i}`` in the drift.  Tags ``(k, j)``
count ``ε^k`` and ``μ^j``, where a jump cumulant of order ``m`` carries
``μ^{m−2}``, so a moment coefficient ``(k, j)`` is the sum of the package's
diagrams with ``k`` F vertices and non-local vertices ``K{m_i}`` with
``Σ (m_i − 2) = j``.  Imports no sft-wick code.
"""
from __future__ import annotations

import itertools
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import ito_moments as im  # noqa: E402

import poisson_noise as nz  # noqa: E402

__all__ = ["Hierarchy"]


class Hierarchy:
    """Moment hierarchy of demo 4 at ``positions`` (one per point)."""

    def __init__(self, p: nz.Params, positions, f_tensor=nz.F_TENSOR,
                 max_jump_order: int = 6):
        self.p = p
        self.positions = tuple(float(x) for x in positions)
        N, P = p.n_components, len(self.positions)
        self.N, self.P = N, P
        colored = p.pulse == "exponential"
        self.D = (2 if colored else 1) * N * P
        self._phi = {(a, i): a * P + i for a in range(N) for i in range(P)}
        self._eta = ({(a, i): N * P + a * P + i
                      for a in range(N) for i in range(P)} if colored else {})
        sde = im.PolySDE(D=self.D, n_tags=2)
        A = np.zeros((self.D, self.D))
        for (a, i), k in self._phi.items():
            A[k, k] = -p.gamma
            if colored:
                A[k, self._eta[(a, i)]] = 1.0
        for (a, i), k in self._eta.items():
            A[k, k] = -p.rates[a]
        sde.add_linear_drift(A)
        Q = np.zeros((self.D,) * 3)
        for i in range(P):
            for a, b, c in itertools.product(range(N), repeat=3):
                Q[self._phi[(a, i)], self._phi[(b, i)],
                  self._phi[(c, i)]] += f_tensor[a, b, c]
        sde.add_quadratic_drift(Q, tag=(1, 0))
        jumped = self._eta if colored else self._phi
        self._jumped_index = {k: key for key, k in jumped.items()}
        sde.jump_moment = self._jump_moment
        sde.max_jump_order = max_jump_order
        sde.jump_tag = lambda m: (0, max(m - 2, 0))
        self.sde = sde

    # -- inputs ------------------------------------------------------------

    def _jump_moment(self, gamma) -> float:
        counts = {}
        for k, g in enumerate(gamma):
            if g == 0:
                continue
            key = self._jumped_index.get(k)
            if key is None:          # a φ variable under coloured pulses
                return 0.0
            counts[key] = g
        return nz.jump_moment(counts, self.positions, self.p)

    def _initial(self, alpha, tag) -> float:
        zero = (0,) * self.D
        if tag[0] != 0:
            return 0.0
        if alpha == zero:
            return 1.0 if tag == (0, 0) else 0.0
        if not self._eta or any(alpha[k] for k in self._phi.values()):
            return 0.0
        legs = []
        for key, k in self._eta.items():
            legs += [key] * alpha[k]
        return _stationary_moment(tuple(legs), tag[1], self.positions, self.p)

    # -- queries -------------------------------------------------------------

    def monomial(self, legs) -> im.Monomial:
        """``Π φ_{a,i}`` for legs ``[(a, i), …]``."""
        return im.monomial_of(self.D, [self._phi[leg] for leg in legs])

    def moments(self, legs, tags, T: float) -> dict:
        """``{tag: E[Π φ]^{(tag)}(T)}``."""
        mono = self.monomial(legs)
        out = im.solve(self.sde, [(mono, tuple(t)) for t in tags], [T],
                       initial=self._initial)
        return {tuple(t): float(out[(mono, tuple(t))][0]) for t in tags}


@lru_cache(maxsize=None)
def _stationary_moment(legs, j, positions, p) -> float:
    """``E[Π η]`` at tag ``μ^j`` from the stationary cumulants: the sum over
    set partitions into blocks of size ≥ 2 with ``Σ (|b| − 2) = j``."""
    total = 0.0
    for partition in im._set_partitions(list(range(len(legs)))):
        if any(len(b) < 2 for b in partition):
            continue
        if sum(len(b) - 2 for b in partition) != j:
            continue
        term = 1.0
        for block in partition:
            term *= nz.stationary_eta_cumulant([legs[k] for k in block],
                                               positions, p)
        total += term
    return total
