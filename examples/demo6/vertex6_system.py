"""Demo 6's sft-wick side: the ``System``, the closed-form C, and the
vertex composition of a diagram.

Vertex names: ``F`` (quadratic drift), ``G`` (cubic drift), ``X{m}`` (the
static force's cumulant, an ndarray coupling), ``J{m}`` (the white jumps'
cumulant, ``equal_time=True``) and ``W2`` (a white noise entered as an
``m = 2`` vertex, ``equal_time=True``).  They match the tag names of
:mod:`vertex6_reference`, so a diagram's composition is its tag.

The white noise that gives C mixes the components and depends on the points,
so C comes from the closed form (``c_closed_form_only=True``,
``diag_C=False``).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

import sft_wick as sw
from sft_wick.expressions import Product, Sum, Symbol
from sft_wick.workflow.specs import (CustomImpulse, ExponentialTemporal,
                                     GaussianSpatial, SeparableTranslation)

import vertex6_model as md

__all__ = ["WhiteSigma2", "ClosedFormC", "make_system", "propagators_for",
           "composition", "channel"]


@dataclass(frozen=True)
class WhiteSigma2:
    """``σ²(n1, t, n2) → (N, N)`` of the white noise (``CustomImpulse``)."""

    p: md.Params

    def __call__(self, n1, t, n2):
        N = self.p.n_components
        return np.array([[float(md.sigma2(a, n1, b, n2, self.p))
                          for b in range(N)] for a in range(N)])


@dataclass(frozen=True)
class ClosedFormC:
    """The exact C, ``(n1, t1, n2, t2) → (N, N)`` or batched ``(n, N, N)``.

    White noise kinks C on its time diagonal; ``has_diagonal_kink`` makes
    Gauss-Legendre split the domain there.
    """

    p: md.Params
    has_diagonal_kink = True

    def __call__(self, n1, t1, n2, t2):
        scalar = np.ndim(t1) == 0 and np.ndim(t2) == 0
        C = md.C_matrix(n1, t1, n2, t2, self.p)
        return C[0] if scalar else C


def noise_spec(p: md.Params) -> sw.GaussianNoise:
    return sw.GaussianNoise(
        kappa2=SeparableTranslation(
            temporal=ExponentialTemporal(lam=0.0, sigma_t=1.0),
            spatial=GaussianSpatial(sigma_x=1.0)),
        sigma2=CustomImpulse(fn=WhiteSigma2(p)))


def make_system(p: md.Params, *, f_amplitude: float = 0.0,
                g_amplitude: float = 0.0, static: tuple = (),
                equal_time: tuple = ()) -> sw.System:
    """Demo 6's system.

    Args:
        p: parameters (``PARAMS``: distinct rates, matrix R;
            ``PARAMS_COMMON``: scalar R).
        f_amplitude, g_amplitude: scales of ``F_TENSOR`` and ``G_TENSOR``.
        static: orders ``m`` of the static-force vertices ``X{m}``.
        equal_time: orders ``m`` of the white vertices (``W2``, ``J{m}``).
    """
    vertices = []
    if f_amplitude:
        vertices.append(sw.LocalVertex("F", coupling=f_amplitude
                                       * md.F_TENSOR))
    if g_amplitude:
        vertices.append(sw.LocalVertex("G", coupling=g_amplitude
                                       * md.G_TENSOR))
    nonlocal_vertices = [
        sw.NonLocalVertex(f"X{m}", order=m, coupling=md.X_CUMULANTS[m])
        for m in static
    ] + [
        sw.NonLocalVertex("W2" if m == 2 else f"J{m}", order=m,
                          coupling=md.JUMP_CUMULANTS[m], equal_time=True)
        for m in equal_time
    ]
    return sw.System(
        field=sw.FieldSpec("phi", p.n_components),
        linear=sw.DiagonalA(gamma=list(p.rates)),
        vertices=vertices, nonlocal_vertices=nonlocal_vertices,
        noise=noise_spec(p), t_min=p.t_min)


def propagators_for(system: sw.System, p: md.Params, t_max: float):
    """Propagators with the exact C and the full ``(N, N)`` matrix."""
    return system.propagators(
        t_max=t_max, c_closed_form=ClosedFormC(p), c_closed_form_only=True,
        c_closed_form_vectorized=True, diag_C=False, progress=False)


def _symbols(expr) -> list:
    if isinstance(expr, Symbol):
        return [expr.name]
    if isinstance(expr, Product):
        return [n for f in expr.factors for n in _symbols(f)]
    return []


def composition(dt) -> Counter:
    """``{vertex name: count}`` of one diagram (from one coupling term)."""
    expr = dt.coupling_sum
    first = expr.terms[0] if isinstance(expr, Sum) else expr
    return Counter(_symbols(first))


def channel(expansion, order: int, label: str) -> Counter:
    """The composition shared by every diagram of ``order`` whose vertex
    label is ``label``; raises if they differ (the label is a set, not a
    multiset, so ``order`` has to pin the counts)."""
    comps = {tuple(sorted(composition(dt).items()))
             for dt in expansion.by_vertex_type(order).get(label, [])}
    if len(comps) != 1:
        raise ValueError(f"order {order}, label {label!r}: compositions "
                         f"{sorted(comps)}")
    return Counter(dict(comps.pop()))


def restrict(expansion, order: int, counts: dict):
    """The expansion reduced to the diagrams of ``order`` whose vertex
    composition is ``counts`` (e.g. ``{"F": 2, "X3": 2}``); raises if there
    are none."""
    want = Counter({k: v for k, v in counts.items() if v})
    keep = [dt for dt in expansion.diagrams(order) if composition(dt) == want]
    if not keep:
        raise ValueError(f"no diagram of order {order} with {dict(want)}")
    return replace(expansion, dts_by_order={order: keep}, orders=(order,))
