r"""Demo 3's :class:`~sft_wick.workflow.System` builders.

Level A (``F = 0``) and level B (the interacting ``Z₂`` system) share the
same noise sector, so both are built here.

The ``F`` structure is deliberately demo 2's::

    F[0,1,1] = 1,   F[1,0,1] = F[1,1,0] = 1/2

i.e. ``dφ₀/dt = −γφ₀ + φ₁² + η₀`` and ``dφ₁/dt = −γφ₁ + φ₀φ₁ + η₁``.
The drift is invariant under ``φ₁ → −φ₁``.  If the noise law were also
invariant under ``η₁ → −η₁`` the cross-correlator
``ξ₀₁ = ⟨φ₀(x,t) φ₁(y,t)⟩`` would vanish identically.  Shot noise is
*skewed*, so the symmetry is broken by exactly the **odd** cumulants of
``η₁``: every diagram carrying an even number of ``η₁`` legs --- order 0,
FF, FFFF, and the whole ``κ⁴`` channel --- cancels, and ``ξ₀₁`` is driven
by ``κ³`` alone.  It is the cleanest available non-Gaussian observable.
"""
from __future__ import annotations

import numpy as np

from sft_wick.workflow import (DiagonalA, ExponentialTemporal, FieldSpec,
                               GaussianNoise, LocalVertex, NonLocalVertex,
                               SeparableTranslation, System)
from sft_wick.workflow.specs import CustomKernel

from shot_noise import (PARAMS, ShotNoise, coupling_k3_raw_vectorized,
                        coupling_vectorized_for, kappa2_lam, kappa2_spatial)

__all__ = ["F_TENSOR", "noise_for", "make_system", "spatial_kernel_for"]


def F_TENSOR(amplitude: float = 1.0, n_comp: int = 2) -> np.ndarray:
    """The ``Z₂`` drift tensor, scaled by ``amplitude``.

    ``F`` enters the observable as ``F^k``, so scaling it is the clean
    handle for confirming a channel's order by its scaling exponent.
    """
    F = np.zeros((n_comp,) * 3)
    F[0, 1, 1] = 1.0
    F[1, 0, 1] = F[1, 1, 0] = 0.5
    return amplitude * F


def spatial_kernel_for(p: ShotNoise) -> CustomKernel:
    """``CustomKernel`` wrapping ``X₂(r)`` bound to ``p``."""
    return CustomKernel(fn=lambda r, _p=p: float(kappa2_spatial(r, _p)))


def noise_for(p: ShotNoise) -> GaussianNoise:
    """κ² as ``SeparableTranslation(ExponentialTemporal, CustomKernel)``.

    This is exactly the family ``builtin_closed_form_for`` recognises, so
    ``Propagators.c_source`` reports ``closed_form:builtin`` and **no C
    quadrature ever runs** --- the propagators are machine precision.
    """
    return GaussianNoise(kappa2=SeparableTranslation(
        temporal=ExponentialTemporal(lam=kappa2_lam(p), sigma_t=p.sigma_t),
        spatial=spatial_kernel_for(p)))


def make_system(p: ShotNoise = PARAMS, *, f_amplitude: float = 0.0,
                cumulants: tuple[int, ...] = (3,),
                r_contracted: bool = True) -> System:
    """Build the demo-3 system.

    Args:
        p: model parameters.
        f_amplitude: ``0.0`` for level A (free field, the exact test),
            non-zero for level B.
        cumulants: which higher cumulants to include as non-local
            vertices --- ``(3,)``, ``(4,)`` or ``(3, 4)``.
        r_contracted: use ``already_R_contracted=True`` callables (the
            fast, exact path).  ``False`` installs the raw ``κ^(m)``
            instead, so the runtime does the ``m`` leg integrals itself;
            the two must agree, which is what validates the feature on a
            *non-constant* kernel.
    """
    vertices = []
    if f_amplitude != 0.0:
        vertices.append(LocalVertex(name="F", coupling=F_TENSOR(f_amplitude,
                                                                p.n_components)))
    nonlocal_vertices = []
    for m in cumulants:
        if not r_contracted:
            if m != 3:
                raise NotImplementedError(
                    "the raw-vertex comparison is wired up for m = 3 only")
            fn = coupling_k3_raw_vectorized
        else:
            fn = coupling_vectorized_for(m)
        nonlocal_vertices.append(NonLocalVertex(
            name=f"K{m}", order=m,
            coupling=lambda n, t, _f=fn, _p=p: _f(n, t, _p),
            coupling_vectorized=True,
            already_R_contracted=r_contracted))
    return System(
        field=FieldSpec(name="phi", n_components=p.n_components),
        linear=DiagonalA(gamma=[p.gamma] * p.n_components),
        vertices=vertices,
        nonlocal_vertices=nonlocal_vertices,
        noise=noise_for(p),
    )
