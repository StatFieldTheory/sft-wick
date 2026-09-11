"""Demo 7, item (d): the sft-wick side of the 3-D shot noise.

Every callable is a frozen dataclass (stable ``repr`` for the caches,
picklable for joblib).  The callables receive 3-D leg positions: under the
vectorised contract ``n_2d`` has shape ``(m, n_samples, 3)``.  The C
propagator comes from the closed form in :mod:`space7_shot`
(``c_closed_form_only=True``, ``diag_C=False``): the events drive both
components, so C has off-diagonal entries.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

import sft_wick as sw
from sft_wick.workflow.specs import CustomImpulse, GeneralKappa2

import space7_shot as sh

__all__ = ["RawKappa3D", "RContractedKappa3D", "ClosedFormC3D",
           "make_system", "propagators_for"]


def _tensor(fn, m: int, n_comp: int, xs, ts) -> np.ndarray:
    """``(n,) + (N,)*m`` tensor of ``fn(comps, xs, ts)``; axis ``l`` is leg
    ``l``."""
    xs = sh._legs_x(xs)
    ts = sh._legs_t(ts)
    out = np.empty((ts.shape[1],) + (n_comp,) * m)
    for comps in itertools.product(range(n_comp), repeat=m):
        out[(slice(None),) + comps] = fn(comps, xs, ts)
    return out


@dataclass(frozen=True)
class RawKappa3D:
    """Bare ``κ^(m)`` of white pulses at the vertex legs: the amplitude of
    the equal-time vertex (vectorised contract)."""

    m: int
    p: sh.ShotParams

    def __call__(self, n_2d, t_2d):
        return _tensor(lambda c, x, t: sh.kappa_raw(c, x, t, self.p),
                       self.m, self.p.n_components, n_2d, t_2d)


@dataclass(frozen=True)
class RContractedKappa3D:
    """``already_R_contracted`` ``κ^(m)``: ``K_R`` at the partner points."""

    m: int
    p: sh.ShotParams

    def __call__(self, n_2d, t_2d):
        return _tensor(lambda c, x, t: sh.K_R(c, x, t, self.p),
                       self.m, self.p.n_components, n_2d, t_2d)


@dataclass(frozen=True)
class ClosedFormC3D:
    """The exact C: ``(n1, t1, n2, t2) → (N, N)``, or ``(n, N, N)`` for
    batched times; positions are 3-vectors.  ``has_diagonal_kink`` makes
    Gauss-Legendre split the domain where two times of a C cross."""

    p: sh.ShotParams
    has_diagonal_kink = True

    def __call__(self, n1, t1, n2, t2):
        scalar = np.ndim(t1) == 0 and np.ndim(t2) == 0
        C = sh.C_matrix(n1, t1, n2, t2, self.p)
        return C[0] if scalar else C


@dataclass(frozen=True)
class Kappa2Record:
    """κ² of exponential pulses, for the record (C is :class:`ClosedFormC3D`)."""

    p: sh.ShotParams

    def __call__(self, n1, t1, n2, t2):
        N = self.p.n_components
        xs = np.stack([np.atleast_1d(np.asarray(n1, float)),
                       np.atleast_1d(np.asarray(n2, float))])
        dt = abs(float(t1) - float(t2))
        # The later leg's pulse has decayed for |t1 − t2| longer.
        return np.array([[self.p.nu * self.p.h[a] * self.p.h[b]
                          * sh.spatial_overlap((a, b), xs, self.p)[0]
                          * np.exp(-dt * self.p.rates[a if t1 > t2 else b])
                          / (self.p.rates[a] + self.p.rates[b])
                          for b in range(N)] for a in range(N)])


@dataclass(frozen=True)
class Sigma2Record:
    """Covariance rate of white pulses, for the record."""

    p: sh.ShotParams

    def __call__(self, n1, t, n2):
        N = self.p.n_components
        xs = np.stack([np.atleast_1d(np.asarray(n1, float)),
                       np.atleast_1d(np.asarray(n2, float))])
        return np.array([[self.p.nu * self.p.h[a] * self.p.h[b]
                          * sh.spatial_overlap((a, b), xs, self.p)[0]
                          for b in range(N)] for a in range(N)])


def _noise(p: sh.ShotParams) -> sw.GaussianNoise:
    if p.pulse == "white":
        return sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.0, sigma_t=1.0),
                spatial=sw.GaussianSpatial(sigma_x=1.0)),
            sigma2=CustomImpulse(fn=Sigma2Record(p)))
    return sw.GaussianNoise(kappa2=GeneralKappa2(fn=Kappa2Record(p)))


def make_system(p: sh.ShotParams, *, f_amplitude: float = 1.0,
                cumulants: tuple = (3,)) -> sw.System:
    """The 3-D shot-noise system.  White pulses use the raw ``equal_time``
    vertex (the package does the leg-time integral); exponential pulses
    the ``already_R_contracted`` ``K_R``, whose raw kernel is kinked where
    two leg times cross."""
    raw = p.pulse == "white"
    cls = RawKappa3D if raw else RContractedKappa3D
    vertices = ([sw.LocalVertex("F", coupling=f_amplitude * sh.F_SHOT)]
                if f_amplitude else [])
    nonlocal_vertices = [
        sw.NonLocalVertex(f"K{m}", order=m, coupling=cls(m=m, p=p),
                          coupling_vectorized=True, equal_time=raw,
                          already_R_contracted=not raw)
        for m in cumulants]
    return sw.System(
        field=sw.FieldSpec("phi", p.n_components),
        linear=sw.DiagonalA(gamma=list(p.gamma)),
        vertices=vertices, nonlocal_vertices=nonlocal_vertices,
        noise=_noise(p), t_min=p.t_min)


def propagators_for(system: sw.System, p: sh.ShotParams, t_max: float):
    """The exact, full ``(N, N)`` C."""
    return system.propagators(
        t_max=t_max, c_closed_form=ClosedFormC3D(p), c_closed_form_only=True,
        c_closed_form_vectorized=True, diag_C=False, progress=False)
