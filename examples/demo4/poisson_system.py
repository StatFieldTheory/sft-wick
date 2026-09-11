"""Demo 4's sft-wick side: picklable coupling callables and the ``System``.

Every callable is a frozen dataclass, for two reasons the other demos give:
a stable ``repr`` keeps the expansion cache working, and joblib can pickle
it.  The C propagator comes from the closed form in :mod:`poisson_noise`
(``c_closed_form_only=True``, ``diag_C=False``): the noise mixes the
components, so C has off-diagonal entries, which the quadrature tables do
not hold.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

import sft_wick as sw
from sft_wick.workflow.specs import (CustomImpulse, ExponentialTemporal,
                                     GaussianSpatial, GeneralKappa2,
                                     SeparableTranslation)

import poisson_noise as nz

__all__ = ["RawKappa", "RContractedKappa", "ClosedFormC", "make_system",
           "propagators_for"]


def _tensor(fn, m: int, n_comp: int, xs, ts) -> np.ndarray:
    """``(n,) + (N,)*m`` tensor of ``fn(comps, xs, ts)`` over all component
    tuples; axis ``l`` belongs to leg ``l``."""
    xs = nz._legs(xs)
    ts = nz._legs(ts)
    out = np.empty((xs.shape[1],) + (n_comp,) * m)
    for comps in itertools.product(range(n_comp), repeat=m):
        out[(slice(None),) + comps] = fn(comps, xs, ts)
    return out


@dataclass(frozen=True)
class RawKappa:
    """Bare ``κ^(m)`` at the vertex legs (vectorised contract).

    ``has_coincident_time_kinks`` tells Gauss-Legendre and nquad that the
    kernel is kinked where two of its time arguments cross: for exponential
    pulses ``G_a(t)`` depends on the smallest leg time.  They then split
    the time domain at every pair of leg times the diagram leaves
    unordered.  For white pulses the raw vertex is ``equal_time``, its legs
    share one time, and the declaration changes nothing.
    """

    m: int
    p: nz.Params
    has_coincident_time_kinks = True

    def __call__(self, n_2d, t_2d):
        return _tensor(lambda c, x, t: nz.kappa_raw(c, x, t, self.p),
                       self.m, self.p.n_components, n_2d, t_2d)


@dataclass(frozen=True)
class RContractedKappa:
    """``already_R_contracted`` ``κ^(m)``: ``K_R`` at the partner points.

    ``T̃_a(t')`` depends on the smallest partner time, so ``K_R`` is kinked
    where two partner times cross; ``has_coincident_time_kinks`` makes
    Gauss-Legendre and nquad split there (two F-vertex partners in FFK4).
    """

    m: int
    p: nz.Params
    has_coincident_time_kinks = True

    def __call__(self, n_2d, t_2d):
        return _tensor(lambda c, x, t: nz.K_R(c, x, t, self.p),
                       self.m, self.p.n_components, n_2d, t_2d)


@dataclass(frozen=True)
class ClosedFormC:
    """The exact C, ``(n1, t1, n2, t2) → (N, N)`` or batched ``(n, N, N)``.

    ``has_diagonal_kink`` tells the Gauss-Legendre integrator that C is not
    smooth across ``t1 = t2``: a derivative jump for white pulses, a jump in
    a higher derivative for exponential ones (where ``J(t, s)`` has a kink
    at ``t = s``).  It then splits the time domain there.
    """

    p: nz.Params
    has_diagonal_kink = True

    def __call__(self, n1, t1, n2, t2):
        scalar = np.ndim(t1) == 0 and np.ndim(t2) == 0
        C = nz.C_matrix(n1, t1, n2, t2, self.p)
        return C[0] if scalar else C


@dataclass(frozen=True)
class Kappa2Colored:
    """κ² of the exponential pulses, for the record (C comes from
    :class:`ClosedFormC`)."""

    p: nz.Params

    def __call__(self, n1, t1, n2, t2):
        N = self.p.n_components
        xs = np.array([[float(n1)], [float(n2)]])
        ts = np.array([[float(t1)], [float(t2)]])
        return np.array([[nz.kappa_raw((a, b), xs, ts, self.p)[0]
                          for b in range(N)] for a in range(N)])


@dataclass(frozen=True)
class Sigma2White:
    """White-noise amplitude of the instantaneous pulses, for the record."""

    p: nz.Params

    def __call__(self, n1, t, n2):
        N = self.p.n_components
        xs = np.array([[float(n1)], [float(n2)]])
        return np.array([[nz.kappa_raw((a, b), xs, xs, self.p)[0]
                          for b in range(N)] for a in range(N)])


def noise_spec(p: nz.Params) -> sw.GaussianNoise:
    if p.pulse == "white":
        return sw.GaussianNoise(
            kappa2=SeparableTranslation(
                temporal=ExponentialTemporal(lam=0.0, sigma_t=1.0),
                spatial=GaussianSpatial(sigma_x=1.0)),
            sigma2=CustomImpulse(fn=Sigma2White(p)))
    return sw.GaussianNoise(kappa2=GeneralKappa2(fn=Kappa2Colored(p)))


def make_system(p: nz.Params, *, f_amplitude: float = 0.0,
                cumulants: tuple = (3,),
                r_contracted: bool = True) -> sw.System:
    """Demo 4's system.

    Args:
        p: noise parameters.
        f_amplitude: scale of :data:`noise.F_TENSOR`; 0 gives the free field.
        cumulants: orders ``m`` of the non-local vertices ``K{m}``.
        r_contracted: ``already_R_contracted=True`` callables (``K_R``) or
            the raw ``κ^(m)``, whose leg integrals the package then does
            itself; for white pulses the raw vertex is ``equal_time``.
    """
    vertices = ([sw.LocalVertex("F", coupling=f_amplitude * nz.F_TENSOR)]
                if f_amplitude else [])
    cls = RContractedKappa if r_contracted else RawKappa
    nonlocal_vertices = [
        sw.NonLocalVertex(
            f"K{m}", order=m, coupling=cls(m=m, p=p),
            coupling_vectorized=True,
            already_R_contracted=r_contracted,
            equal_time=(p.pulse == "white" and not r_contracted))
        for m in cumulants
    ]
    return sw.System(
        field=sw.FieldSpec("phi", p.n_components),
        linear=sw.DiagonalA(gamma=[p.gamma] * p.n_components),
        vertices=vertices, nonlocal_vertices=nonlocal_vertices,
        noise=noise_spec(p))


def propagators_for(system: sw.System, p: nz.Params, t_max: float):
    """Propagators with the exact C and the full ``(N, N)`` matrix."""
    return system.propagators(
        t_max=t_max, c_closed_form=ClosedFormC(p), c_closed_form_only=True,
        c_closed_form_vectorized=True, diag_C=False, progress=False)
