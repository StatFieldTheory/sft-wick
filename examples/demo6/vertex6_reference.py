r"""Demo 6's exact reference: the moment hierarchy of the Markov embedding,
observed at several times.

At observation points ``x_1 … x_P`` the state is ``φ_{a,i}``, plus the
static force ``X_a`` when it has cumulants.  The generator (see
``examples/reference/ito_moments.py``) carries

* drift ``dφ_{a,i} = (−γ_a φ_{a,i} + X_a + F_abc φ_{b,i} φ_{c,i}
  + G_abcd φ_{b,i} φ_{c,i} φ_{d,i}) dt`` and ``dX = 0``;
* Gaussian white noise ``σ²_ab(x_i, x_j)`` between ``φ_{a,i}`` and
  ``φ_{b,j}`` (the package's C), and a second one, ``W_ab`` at every pair of
  points (a white noise entered as an ``m = 2`` vertex);
* jumps: one event moves every ``φ_{a,i}`` by ``J_a``, with rate-weighted
  moments ``ν E[J^{⊗m}]``;
* the initial law ``φ = 0`` and ``X`` with cumulants ``κ^(m)``, whose
  moments follow from the set-partition formula.

Each term carries a tag (:data:`TAG_NAMES`): one count per ``F``, ``G``,
``W`` insertion and per jump of order ``m`` (``J3 … J5``), and one per
cumulant block of ``X`` of order ``m`` (``X2 … X5``).  A moment coefficient
at a tag is then the sum of the package's diagrams with those vertices.

Unequal observation times: the variables of point ``i`` are frozen (their
drift, noise and jumps removed) after ``t_i``, so ``E[Π_i φ(t_i)]`` is a
moment of the frozen process at the last time.  :func:`solve_multitime`
does this for any :class:`ito_moments.PolySDE`, interval by interval.

Imports no sft-wick code.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
import ito_moments as im  # noqa: E402

__all__ = ["TAG_NAMES", "tag", "frozen_copy", "closure", "solve_multitime",
           "Hierarchy"]

TAG_NAMES = ("F", "G", "X2", "X3", "X4", "X5", "W2", "J3", "J4", "J5")


def tag(**counts: int) -> tuple:
    """The tag with the given counts, e.g. ``tag(F=1, X3=1)``."""
    unknown = set(counts) - set(TAG_NAMES)
    if unknown:
        raise ValueError(f"unknown tag names {sorted(unknown)}")
    return tuple(int(counts.get(name, 0)) for name in TAG_NAMES)


# ---------------------------------------------------------------------------
# Several observation times, for any PolySDE
# ---------------------------------------------------------------------------

class _FrozenJump:
    """Jump moments of the process with ``frozen`` variables held fixed."""

    def __init__(self, base: Callable, frozen: frozenset) -> None:
        self.base = base
        self.frozen = frozen

    def __call__(self, gamma) -> float:
        if any(gamma[k] for k in self.frozen):
            return 0.0
        return self.base(gamma)


def frozen_copy(sde: im.PolySDE, frozen: Iterable[int]) -> im.PolySDE:
    """``sde`` with the variables ``frozen`` held at their current values.

    Raises if a variable left free has a drift or a diffusion that depends
    on a frozen one: freezing is only exact when nothing that is still
    evolving reads the frozen variables.
    """
    frozen = frozenset(frozen)
    if not frozen:
        return sde
    drift = [t for t in sde.drift if t[0] not in frozen]
    diffusion = [t for t in sde.diffusion
                 if t[0] not in frozen and t[1] not in frozen]
    for term in drift:
        if any(term[2][k] for k in frozen):
            raise ValueError("a free variable's drift reads a frozen one")
    for term in diffusion:
        if any(term[3][k] for k in frozen):
            raise ValueError("a free variable's diffusion reads a frozen one")
    return im.PolySDE(
        D=sde.D, n_tags=sde.n_tags, drift=drift, diffusion=diffusion,
        jump_moment=(None if sde.jump_moment is None
                     else _FrozenJump(sde.jump_moment, frozen)),
        max_jump_order=sde.max_jump_order, jump_tag=sde.jump_tag)


def closure(sde: im.PolySDE, targets: Iterable) -> set:
    """The ``(α, tag)`` nodes :func:`ito_moments.solve` reaches from
    ``targets`` (the same breadth-first search)."""
    seen: set = set()
    queue = list(targets)
    while queue:
        node = queue.pop()
        if node in seen:
            continue
        seen.add(node)
        alpha, tg = node
        for beta, _c, dtag in sde.generator(alpha):
            src = im._tag_sub(tg, dtag)
            if src is not None and (beta, src) not in seen:
                queue.append((beta, src))
    return seen


def solve_multitime(sde: im.PolySDE, targets: Sequence,
                    freeze_times: Mapping[int, float],
                    initial: Callable) -> dict:
    """Moment coefficients of the process in which variable ``v`` stops at
    ``freeze_times[v]`` (time measured from the start), at the last of those
    times.  Variables not in ``freeze_times`` never stop.

    Returns ``{(α, tag): value}`` for the targets.
    """
    taus = sorted({float(t) for t in freeze_times.values()})
    if not taus or taus[0] < 0.0:
        raise ValueError("freeze times must be non-negative")
    bounds = [0.0] + taus

    def values_at(k: int, nodes: list) -> dict:
        if k == 0:
            return {n: float(initial(*n)) for n in nodes}
        frozen = [v for v, t in freeze_times.items() if t <= bounds[k - 1]]
        sde_k = frozen_copy(sde, frozen)
        start = values_at(k - 1, sorted(closure(sde_k, nodes)))
        out = im.solve(sde_k, nodes, [bounds[k] - bounds[k - 1]],
                       initial=lambda a, tg: start[(a, tg)])
        return {n: float(out[n][0]) for n in nodes}

    return values_at(len(bounds) - 1, list(dict.fromkeys(targets)))


# ---------------------------------------------------------------------------
# Demo 6's process
# ---------------------------------------------------------------------------

class Hierarchy:
    """Moment hierarchy of demo 6's model at ``positions``.

    Args:
        p: parameters with ``rates``, ``t_min``, ``Q`` and ``ell``
            (:class:`vertex6_model.Params`).
        positions: one position per observation point.
        sigma2: ``(a, x, b, y) ↦ σ²_ab(x, y)`` of the white noise that
            gives C; ``None`` for no such noise.
        f_tensor, g_tensor: quadratic and cubic drift, or ``None``.
        x_cumulants: ``{m: κ^(m)}`` of the static force (``m ≥ 2``).
        jump_cumulants: ``{m: ν E[J^{⊗m}]}`` of the white jumps
            (``3 ≤ m ≤ 5``).
        white_vertex: covariance rate ``W`` of the white noise entered as
            a vertex, or ``None``.
    """

    def __init__(self, p, positions: Sequence[float], *,
                 sigma2: Callable | None = None, f_tensor=None,
                 g_tensor=None, x_cumulants: Mapping | None = None,
                 jump_cumulants: Mapping | None = None,
                 white_vertex=None) -> None:
        self.p = p
        self.positions = tuple(float(x) for x in positions)
        N, P = len(p.rates), len(self.positions)
        self.N, self.P = N, P
        self.x_cumulants = dict(x_cumulants or {})
        self.jump_cumulants = dict(jump_cumulants or {})
        if any(m < 3 or m > 5 for m in self.jump_cumulants):
            raise ValueError("jump cumulants of order 3-5 only; order 2 is "
                             "white_vertex")
        if any(m < 2 or f"X{m}" not in TAG_NAMES for m in self.x_cumulants):
            raise ValueError("static cumulants of order 2-5 only")
        off = N if self.x_cumulants else 0
        self._x = {a: a for a in range(off)}
        self._phi = {(a, i): off + a * P + i
                     for a in range(N) for i in range(P)}
        self._phi_key = {k: key for key, k in self._phi.items()}
        self.D = off + N * P
        sde = im.PolySDE(D=self.D, n_tags=len(TAG_NAMES))

        A = np.zeros((self.D, self.D))
        for (a, i), k in self._phi.items():
            A[k, k] = -p.rates[a]
            if self._x:
                A[k, self._x[a]] = 1.0
        sde.add_linear_drift(A)
        if f_tensor is not None:
            Q = np.zeros((self.D,) * 3)
            for i in range(P):
                for a, b, c in itertools.product(range(N), repeat=3):
                    Q[self._phi[(a, i)], self._phi[(b, i)],
                      self._phi[(c, i)]] += f_tensor[a, b, c]
            sde.add_quadratic_drift(Q, tag=tag(F=1))
        if g_tensor is not None:
            for i in range(P):
                for a, b, c, d in itertools.product(range(N), repeat=4):
                    coeff = float(g_tensor[a, b, c, d])
                    if coeff == 0.0:
                        continue
                    sde.drift.append((
                        self._phi[(a, i)], coeff,
                        im.unit(self.D, self._phi[(b, i)], self._phi[(c, i)],
                                self._phi[(d, i)]),
                        tag(G=1)))
        if sigma2 is not None:
            S = np.zeros((self.D, self.D))
            for (a, i), k in self._phi.items():
                for (b, j), l in self._phi.items():
                    S[k, l] = sigma2(a, self.positions[i], b,
                                     self.positions[j])
            sde.add_constant_diffusion(S)
        if white_vertex is not None:
            W = np.asarray(white_vertex, dtype=float)
            S2 = np.zeros((self.D, self.D))
            for (a, i), k in self._phi.items():
                for (b, j), l in self._phi.items():
                    S2[k, l] = W[a, b]
            sde.add_constant_diffusion(S2, tag=tag(W2=1))
        if self.jump_cumulants:
            sde.jump_moment = self._jump_moment
            sde.max_jump_order = max(self.jump_cumulants)
            sde.jump_tag = lambda m: tag(**{f"J{m}": 1})
        self.sde = sde
        self._x_cache: dict = {}

    # -- inputs --------------------------------------------------------------

    def _jump_moment(self, gamma) -> float:
        comps: list[int] = []
        for k, g in enumerate(gamma):
            if g == 0:
                continue
            key = self._phi_key.get(k)
            if key is None:          # X does not jump
                return 0.0
            comps += [key[0]] * g
        tensor = self.jump_cumulants.get(len(comps))
        return 0.0 if tensor is None else float(tensor[tuple(sorted(comps))])

    def _initial(self, alpha, tg) -> float:
        counts = dict(zip(TAG_NAMES, tg))
        if any(counts[n] for n in ("F", "G", "W2", "J3", "J4", "J5")):
            return 0.0
        if any(alpha[k] for k in self._phi.values()):
            return 0.0
        legs: list[int] = []
        for a, k in self._x.items():
            legs += [a] * alpha[k]
        want = tuple(counts[f"X{m}"] for m in (2, 3, 4, 5))
        return self._x_moment(tuple(legs), want)

    def _x_moment(self, legs: tuple, want: tuple) -> float:
        """``E[Π X]`` at the tag with ``want[m−2]`` blocks of order ``m``."""
        key = (legs, want)
        if key in self._x_cache:
            return self._x_cache[key]
        if not legs:
            val = 1.0 if not any(want) else 0.0
        else:
            val = 0.0
            for part in im._set_partitions(list(range(len(legs)))):
                sizes = [len(b) for b in part]
                if any(s not in self.x_cumulants for s in sizes):
                    continue
                if tuple(sizes.count(m) for m in (2, 3, 4, 5)) != want:
                    continue
                term = 1.0
                for block in part:
                    comps = tuple(sorted(legs[j] for j in block))
                    term *= float(self.x_cumulants[len(block)][comps])
                val += term
        self._x_cache[key] = val
        return val

    # -- queries -------------------------------------------------------------

    def moments(self, legs, tags, times) -> dict:
        """``{tag: E[Π_i φ_{a_i, i}(t_i)]^{(tag)}}``.

        Args:
            legs: ``[(a, i), …]``, a component and a point index per factor.
            tags: tags from :func:`tag`.
            times: the package's time of each point, or one time for all.
        """
        if np.ndim(times) == 0:
            times = [float(times)] * self.P
        rel = [float(t) - self.p.t_min for t in times]
        mono = im.monomial_of(self.D, [self._phi[leg] for leg in legs])
        freeze = {k: rel[i] for (a, i), k in self._phi.items()}
        targets = [(mono, tuple(tg)) for tg in tags]
        out = solve_multitime(self.sde, targets, freeze, self._initial)
        return {tuple(tg): out[(mono, tuple(tg))] for tg in tags}
