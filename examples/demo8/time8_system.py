r"""Demo 8: the four models as sft-wick ``System`` objects, and their C.

Two ways to give the package its C propagator:

* **table**: the package's own route, ``c_closed_form=None`` (or the
  default ``'auto'``, which finds no built-in closed form for any model
  here): ``C = ∫∫ R κ² R (+ ∫ R σ² R)`` by quadrature on an
  ``n_grid_t × n_grid_t`` grid, looked up by cubic-spline interpolation;
* **exact**: ``c_closed_form=`` :class:`time8_reference.ExactC`, the
  defining integrals evaluated at every point the integrators ask for, so
  that a comparison measures the diagrams and R alone.

The integrators take a matrix R (component-dependent rates) only on the
scalar loops today; ``SCALAR_ONLY`` lists what the others raise.
"""
from __future__ import annotations

import numpy as np

import sft_wick as sw
from sft_wick.workflow.specs import CustomImpulse, CustomKernel

import time8_params as pm
import time8_reference as ref

__all__ = ["rate_system", "oscillator_system", "kernel_system",
           "white_system", "exact_C", "table_propagators",
           "exact_propagators", "METHOD_KW", "matrix_R"]

#: Integrator keyword arguments used throughout the demo.
METHOD_KW = {
    "gauss_legendre": dict(n_gauss=12),
    "qmc_vectorized": dict(n_samples=2 ** 14, seed=5),
    "qmc_scalar": dict(n_samples=2 ** 11, seed=5),
    "qmc": dict(n_samples=2 ** 11, seed=5),
    "nquad": {},
}


def _ou(lam: float, sigma_t: float, sigma_x: float):
    return sw.SeparableTranslation(
        temporal=sw.ExponentialTemporal(lam=lam, sigma_t=sigma_t),
        spatial=sw.GaussianSpatial(sigma_x=sigma_x))


def rate_system(p: pm.RateParams, *, n_grid_cache: int | None = None,
                t_max_cache: float | None = None) -> sw.System:
    """Model (a).  The rate spline grid spans ``[0, t_max_cache]``;
    ``System`` extends it down to a negative ``t_min`` at the same spacing."""
    t_max_cache = p.t_final + 1.0 if t_max_cache is None else t_max_cache
    lin = (sw.DiagonalA(gamma=p.rate, t_max_cache=t_max_cache)
           if n_grid_cache is None else
           sw.DiagonalA(gamma=p.rate, t_max_cache=t_max_cache,
                        n_grid_cache=n_grid_cache))
    return sw.System(
        field=sw.FieldSpec("phi", pm.N), linear=lin,
        vertices=[sw.LocalVertex("F", coupling=pm.F_QUAD)],
        noise=sw.GaussianNoise(kappa2=_ou(p.lam, p.sigma_t, p.sigma_x)),
        t_min=p.t_min)


def oscillator_system(p: pm.OscillatorParams) -> sw.System:
    """Model (b): ``ExplicitR`` with the oscillator's response, a quadratic
    vertex ``F`` and a cubic vertex ``G`` acting on ``x``, white force
    (``CustomImpulse``) and no coloured noise."""
    return sw.System(
        field=sw.FieldSpec("phi", pm.N),
        linear=sw.ExplicitR(R_time=p.R, iso_R=True),
        vertices=[sw.LocalVertex("F", coupling=pm.F_QUAD),
                  sw.LocalVertex("G", coupling=pm.G_CUBIC)],
        noise=sw.GaussianNoise(kappa2=_ou(0.0, 1.0, 1.0),
                               sigma2=CustomImpulse(fn=p.white)),
        t_min=p.t_min)


def kernel_system(p: pm.KernelParams) -> sw.System:
    """Model (c): ``CustomKernel`` for the Matérn and damped-cosine
    kernels, the built-in ``GaussianTemporal`` for the Gaussian one."""
    k = p.kernel
    temporal = (sw.GaussianTemporal(lam=k.lam, sigma_t=k.sigma)
                if isinstance(k, pm.GaussianKernel) else CustomKernel(fn=k))
    return sw.System(
        field=sw.FieldSpec("phi", pm.N),
        linear=sw.DiagonalA(gamma=list(p.gamma)),
        vertices=[sw.LocalVertex("F", coupling=pm.F_QUAD)],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=temporal, spatial=sw.GaussianSpatial(sigma_x=p.sigma_x))),
        t_min=p.t_min)


def white_system(p: pm.WhiteParams) -> sw.System:
    """Model (d): OU noise plus white noise with ``s_a(t)`` (``CustomImpulse``)."""
    return sw.System(
        field=sw.FieldSpec("phi", pm.N),
        linear=sw.DiagonalA(gamma=list(p.gamma)),
        vertices=[sw.LocalVertex("F", coupling=pm.F_QUAD)],
        noise=sw.GaussianNoise(kappa2=_ou(p.lam, p.sigma_t, p.sigma_x),
                               sigma2=CustomImpulse(fn=p.white)),
        t_min=p.t_min)


def matrix_R(system: sw.System) -> bool:
    return not system.iso_R


# ---------------------------------------------------------------------------
# C
# ---------------------------------------------------------------------------

def _constant_R(gamma):
    g = np.asarray(gamma, dtype=float)

    def R(a, t, s):
        return np.exp(-g[a] * (np.asarray(t) - np.asarray(s)))
    return R


def exact_C(p, *, n: int = 32, has_diagonal_kink: bool | None = None):
    """:class:`time8_reference.ExactC` for any of the four parameter sets."""
    kw = dict(n=n, has_diagonal_kink=has_diagonal_kink)
    if isinstance(p, pm.RateParams):
        return ref.ExactC(p.rate.R, pm.ExponentialKernel(p.lam, p.sigma_t),
                          pm.GaussianEnvelope(p.sigma_x), p.t_min, **kw)
    if isinstance(p, pm.OscillatorParams):
        return ref.ExactC(lambda a, t, s: p.R.tau(np.asarray(t) - np.asarray(s)),
                          None, None, p.t_min, white=p.white.s,
                          white_envelope=p.white.spatial, **kw)
    if isinstance(p, pm.KernelParams):
        return ref.ExactC(_constant_R(p.gamma), p.kernel,
                          pm.GaussianEnvelope(p.sigma_x), p.t_min, **kw)
    if isinstance(p, pm.WhiteParams):
        return ref.ExactC(_constant_R(p.gamma),
                          pm.ExponentialKernel(p.lam, p.sigma_t),
                          pm.GaussianEnvelope(p.sigma_x), p.t_min,
                          white=p.white.s, white_envelope=p.white.spatial, **kw)
    raise TypeError(type(p).__name__)


def table_propagators(system: sw.System, t_max: float, n_grid_t: int, *,
                      c_method: str = "auto", c_n_gauss: int = 20):
    """The package's own C: quadrature table, spline lookup."""
    return system.propagators(t_max=t_max, n_grid_t=n_grid_t,
                              c_closed_form=None, c_method=c_method,
                              c_n_gauss=c_n_gauss, progress=False)


def exact_propagators(system: sw.System, c_fn, t_max: float):
    return system.propagators(t_max=t_max, c_closed_form=c_fn,
                              c_closed_form_only=True,
                              c_closed_form_vectorized=True, progress=False)
