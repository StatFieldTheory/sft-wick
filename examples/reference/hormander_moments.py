r"""Exact perturbative moments of a polynomial SDE given by its vector fields.

An independent reference for sft-wick's multiplicative-noise route.  Like
``ito_moments.py`` it imports nothing from the package and uses no
diagrams, no response field and no propagators.  It builds the generator of
the process from the drift ``f`` and the noise columns ``g_k`` as they are
written, and hands it to :func:`ito_moments.solve`, which integrates the
moment hierarchy exactly.

The process ``X ∈ R^D`` obeys

.. math::

    dX = f(X)\,dt + \sum_k g_k(X) \circ dW_k \quad\text{(Stratonovich)},
    \qquad
    dX = f(X)\,dt + \sum_k g_k(X)\, dW_k \quad\text{(Itô)},

with ``⟨dW_k dW_l⟩ = δ_kl dt`` and polynomial ``f`` and ``g_k``.  On a
polynomial ``P`` the generator is

.. math::

    \text{Stratonovich:}\quad L P = f\cdot\nabla P
        + \tfrac12 \sum_k V_k(V_k P), \qquad V_k = g_k\cdot\nabla ,

    \text{Itô:}\quad L P = f\cdot\nabla P
        + \tfrac12 \sum_k \sum_{ij} g_{ik}\, g_{jk}\, \partial_i\partial_j P .

The Stratonovich generator is used in Hörmander form: ``V_k`` is applied
twice by polynomial algebra.  The noise-induced drift
``½ Σ_k (g_k·∇) g_k``, which the Itô form of a Stratonovich SDE carries and
which sft-wick adds as vertices, is never written down here.

Every drift term and every coefficient of ``g_k`` carries a *tag*, a vector
of non-negative integers counting powers of bookkeeping parameters, exactly
as in ``ito_moments.py``; the tags of a product add.  :func:`two_time` gives
two-time moments ``E[X^α(t₁) X^β(t₂)]`` through the Markov property.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ito_moments as im  # noqa: E402

Monomial = tuple[int, ...]
Tag = tuple[int, ...]
#: A polynomial: ``{(monomial, tag): coefficient}``.
Poly = dict

__all__ = ["VectorFieldSDE", "affine_column", "constant_column", "two_time",
           "solve"]


def _add_into(out: Poly, key, value: float) -> None:
    if value == 0.0:
        return
    out[key] = out.get(key, 0.0) + value


def _plus(a: tuple, b: tuple) -> tuple:
    return tuple(x + y for x, y in zip(a, b))


def _derivative(p: Poly, i: int) -> Poly:
    """``∂P/∂X_i``."""
    out: Poly = {}
    for (mono, tag), c in p.items():
        if mono[i]:
            lowered = list(mono)
            lowered[i] -= 1
            _add_into(out, (tuple(lowered), tag), c * mono[i])
    return out


def _product(p: Poly, q: Poly) -> Poly:
    out: Poly = {}
    for (m1, t1), c1 in p.items():
        for (m2, t2), c2 in q.items():
            _add_into(out, (_plus(m1, m2), _plus(t1, t2)), c1 * c2)
    return out


def _apply_vector_field(column: dict[int, Poly], p: Poly) -> Poly:
    """``(g·∇) P = Σ_i g_i ∂_i P`` for a column ``{i: g_i}``."""
    out: Poly = {}
    for i, g_i in column.items():
        for key, c in _product(g_i, _derivative(p, i)).items():
            _add_into(out, key, c)
    return out


def constant_column(D: int, n_tags: int, entries: dict[int, float],
                    tag: Tag | None = None) -> dict[int, Poly]:
    """A column ``g_k`` with constant entries ``{i: value}``."""
    tag = (0,) * n_tags if tag is None else tuple(tag)
    zero = (0,) * D
    return {i: {(zero, tag): float(v)} for i, v in entries.items() if v != 0.0}


def affine_column(D: int, n_tags: int, coords: Sequence[int],
                  g0: np.ndarray, g1: np.ndarray, g1_tag: Tag) -> dict[int, Poly]:
    """``g_{coords[a]}(X) = g0[a] + Σ_b g1[a, b] X_{coords[b]}``.

    ``g0`` has shape ``(n,)`` and ``g1`` shape ``(n, n)`` for the ``n``
    coordinates listed in ``coords``; the ``g1`` terms carry ``g1_tag``.
    """
    zero_tag = (0,) * n_tags
    zero = (0,) * D
    column: dict[int, Poly] = {}
    for a, i in enumerate(coords):
        poly: Poly = {}
        _add_into(poly, (zero, zero_tag), float(g0[a]))
        for b, j in enumerate(coords):
            _add_into(poly, (im.unit(D, j), tuple(g1_tag)), float(g1[a, b]))
        if poly:
            column[i] = poly
    return column


@dataclass
class VectorFieldSDE:
    """A polynomial SDE in terms of its drift and noise columns.

    Args:
        D: dimension of ``X``.
        n_tags: number of bookkeeping parameters.
        drift: terms ``(i, c, p, tag)``: ``f_i(X) ⊃ c X^p`` (the format of
            ``ito_moments.PolySDE.drift``).
        columns: the noise columns ``g_k``, each ``{i: polynomial}``.
        interpretation: ``'stratonovich'`` or ``'ito'``.
    """

    D: int
    n_tags: int
    drift: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    interpretation: str = "stratonovich"

    def __post_init__(self) -> None:
        if self.interpretation not in ("ito", "stratonovich"):
            raise ValueError(f"interpretation {self.interpretation!r}")
        self._cache: dict[Monomial, list] = {}

    def zero_tag(self) -> Tag:
        return (0,) * self.n_tags

    def add_linear_drift(self, A: np.ndarray, coords: Sequence[int] | None = None,
                         tag: Tag | None = None) -> None:
        """``f_{coords[a]} ⊃ Σ_b A[a, b] X_{coords[b]}``."""
        coords = list(range(A.shape[0])) if coords is None else list(coords)
        tag = self.zero_tag() if tag is None else tuple(tag)
        for a, b in zip(*np.nonzero(A)):
            self.drift.append((coords[a], float(A[a, b]),
                               im.unit(self.D, coords[b]), tag))

    def add_quadratic_drift(self, Q: np.ndarray, tag: Tag,
                            coords: Sequence[int] | None = None) -> None:
        """``f_{coords[a]} ⊃ Σ_bc Q[a, b, c] X_{coords[b]} X_{coords[c]}``."""
        coords = list(range(Q.shape[0])) if coords is None else list(coords)
        for a, b, c in zip(*np.nonzero(Q)):
            self.drift.append((coords[a], float(Q[a, b, c]),
                               im.unit(self.D, coords[b], coords[c]),
                               tuple(tag)))

    def generator(self, alpha: Monomial) -> list[tuple[Monomial, float, Tag]]:
        """``L X^α`` as ``(β, c, tag)`` terms."""
        hit = self._cache.get(alpha)
        if hit is not None:
            return hit
        zero_tag = self.zero_tag()
        P: Poly = {(tuple(alpha), zero_tag): 1.0}
        out: Poly = {}
        for i, c, p, tag in self.drift:
            if alpha[i]:
                lowered = list(alpha)
                lowered[i] -= 1
                _add_into(out, (_plus(tuple(lowered), p), tuple(tag)),
                          c * alpha[i])
        for column in self.columns:
            if self.interpretation == "stratonovich":
                noise = _apply_vector_field(
                    column, _apply_vector_field(column, P))
            else:
                noise = {}
                for i, g_i in column.items():
                    d_i = _derivative(P, i)
                    for j, g_j in column.items():
                        term = _product(_product(g_i, g_j), _derivative(d_i, j))
                        for key, c in term.items():
                            _add_into(noise, key, c)
            for key, c in noise.items():
                _add_into(out, key, 0.5 * c)
        terms = [(mono, c, tag) for (mono, tag), c in out.items() if c != 0.0]
        self._cache[alpha] = terms
        return terms


def solve(sde, targets, times, initial=None):
    """:func:`ito_moments.solve` on this module's generator."""
    return im.solve(sde, targets, times, initial=initial)


class _FrozenFactor:
    """``(X, Y)`` with ``dY = 0``: the generator of ``X`` acting on
    ``X^β Y^j``, so that ``E[Y X^β(t₂)]`` with ``Y = X^α(t₁)`` is a
    two-time moment."""

    def __init__(self, sde) -> None:
        self.sde = sde
        self.D = sde.D + 1
        self.n_tags = sde.n_tags

    def zero_tag(self) -> Tag:
        return self.sde.zero_tag()

    def generator(self, alpha: Monomial):
        j = alpha[-1]
        return [(beta + (j,), c, tag)
                for beta, c, tag in self.sde.generator(alpha[:-1])]


def two_time(sde, first: Monomial, second: Monomial, tags: Sequence[Tag],
             t1: float, t2: float,
             initial: Callable | None = None) -> dict[Tag, float]:
    """``E[X^first(t₁) X^second(t₂)]`` at each bookkeeping tag, ``t₂ ≥ t₁``.

    By the Markov property the moment is ``E[Y X^second(t₂)]`` for the
    process ``(X, Y)`` started at ``t₁`` with ``Y = X^first(t₁)`` frozen.
    The moments at ``t₁`` that this needs are solved first, from ``t = 0``
    with ``initial`` (default ``X(0) = 0``).
    """
    if t2 < t1:
        raise ValueError("two_time needs t2 >= t1")
    aug = _FrozenFactor(sde)
    targets = [(tuple(second) + (1,), tuple(tag)) for tag in tags]
    needed: set = set()

    def record(alpha, tag):
        needed.add((tuple(alpha), tuple(tag)))
        return 0.0

    im.solve(aug, targets, [0.0], initial=record)
    at_t1 = {}
    stage1 = []
    for alpha, tag in needed:
        beta, j = alpha[:-1], alpha[-1]
        key = (tuple(b + j * f for b, f in zip(beta, first)), tag)
        at_t1[(alpha, tag)] = key
        stage1.append(key)
    values = im.solve(sde, sorted(set(stage1)), [t1], initial=initial)

    def start(alpha, tag):
        return float(values[at_t1[(tuple(alpha), tuple(tag))]][0])

    out = im.solve(aug, targets, [t2 - t1], initial=start)
    return {tuple(tag): float(out[target][0])
            for tag, target in zip(tags, targets)}
