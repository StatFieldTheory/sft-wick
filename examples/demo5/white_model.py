r"""Demo 5: white noise through the package's own machinery.

The field obeys, for ``t ≥ t_min`` from ``φ(t_min) = 0``,

.. math::

    dφ_a = (−γ φ_a + F_{abc} φ_b φ_c + η_a)\,dt + dW_a ,

with two Gaussian noises:

* ``η_a(x, t)``: coloured, ``⟨η_a(x,t) η_b(y,t')⟩ = δ_ab λ e^{−|t−t'|/σ_t}
  e^{−(x−y)^2/(2σ_x^2)}``, stationary;
* ``dW_a``: white, ``⟨dW_a dW_b⟩ = S_ab dt``, the same at every point
  (``ConstantImpulse``).  ``S`` mixes the components in the default
  variant.

White noise makes ``C = ∫R S R`` kinked on its time diagonal and the
coloured part keeps distinct points distinct.  Three variants of ``S`` and
the flags exercise three code paths:

============  =====================  =============================
variant       ``S``                  flags
============  =====================  =============================
mixing        dense ``S``            ``diag_C=False``
diagonal      ``diag(S)``            ``diag_C=True`` (diag-C fast path)
isotropic     ``s · 1``              ``diag_C=True``, ``iso_C=True``
============  =====================  =============================

In every variant the propagators come from the built-in closed form
(``c_closed_form='auto'``), which covers ``DiagonalA`` +
``SeparableTranslation(ExponentialTemporal)`` + ``ConstantImpulse``.
``t_min = 0.5``, so the time translation to the reference's ``t = 0``
is tested as well.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

import sft_wick as sw
from sft_wick.workflow.specs import ConstantImpulse

__all__ = ["Params", "F_TENSOR", "VARIANTS", "variant", "make_system",
           "propagators_for", "expand_flags"]

#: ``dφ_a/dt ⊃ F_abc φ_b φ_c``; no index symmetry.
F_TENSOR = np.array([[[0.30, -0.25], [0.10, 0.20]],
                     [[-0.15, 0.35], [0.40, -0.20]]])


@dataclass(frozen=True)
class Params:
    gamma: float = 1.0
    lam: float = 0.4            # coloured amplitude
    sigma_t: float = 0.6        # coloured correlation time
    sigma_x: float = 0.9        # coloured spatial width
    S: tuple = ((0.50, 0.20), (0.20, 0.35))    # white covariance rate
    t_min: float = 0.5
    variant: str = "mixing"

    @property
    def S_matrix(self) -> np.ndarray:
        return np.asarray(self.S, dtype=float)

    @property
    def n_components(self) -> int:
        return self.S_matrix.shape[0]


VARIANTS = ("mixing", "diagonal", "isotropic")


def variant(name: str, base: Params = Params()) -> Params:
    """The three noise variants of the module docstring."""
    S = base.S_matrix
    if name == "mixing":
        S_v = S
    elif name == "diagonal":
        S_v = np.diag(np.diag(S))
    elif name == "isotropic":
        S_v = float(np.mean(np.diag(S))) * np.eye(S.shape[0])
    else:
        raise ValueError(f"unknown variant {name!r}")
    return replace(base, S=tuple(map(tuple, S_v)), variant=name)


def make_system(p: Params, f_amplitude: float = 1.0) -> sw.System:
    vertices = ([sw.LocalVertex("F", coupling=f_amplitude * F_TENSOR)]
                if f_amplitude else [])
    return sw.System(
        field=sw.FieldSpec("phi", p.n_components),
        linear=sw.DiagonalA(gamma=[p.gamma] * p.n_components),
        vertices=vertices,
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=p.lam,
                                                sigma_t=p.sigma_t),
                spatial=sw.GaussianSpatial(sigma_x=p.sigma_x)),
            sigma2=ConstantImpulse(p.S_matrix)),
        t_min=p.t_min)


def expand_flags(p: Params) -> dict:
    """``System.expand`` flags for the variant."""
    if p.variant == "mixing":
        return dict(diag_C=False)
    if p.variant == "isotropic":
        return dict(diag_C=True, iso_C=True)
    return dict(diag_C=True)


def propagators_for(system: sw.System, p: Params, t_max: float):
    return system.propagators(
        t_max=t_max, c_closed_form="auto", c_closed_form_only=True,
        c_closed_form_vectorized=True, diag_C=(p.variant != "mixing"),
        progress=False)


def quadrature_propagators_for(system: sw.System, p: Params, t_max: float,
                               n_grid_t: int = 31, n_gauss: int = 16):
    """The same propagators from quadrature tables instead of the closed form.

    ``c_closed_form=None`` forces the quadrature.  Under ``diag_C=False``
    the tables hold every ``C_ab``, which is what the mixing variant needs
    and what the propagator builder used to refuse without a closed form.
    The accuracy is then the table's, not machine precision: with
    ``n_grid_t=31`` the mixing variant lands within 1.2e-08 (order 0),
    8.8e-08 (order 1) and 2.1e-05 / 1.1e-03 (order 2, the two component
    pairs) of the hierarchy.  Reached by ``run.py --c-quadrature``.
    """
    return system.propagators(
        t_max=t_max, n_grid_t=n_grid_t, c_closed_form=None,
        diag_C=(p.variant != "mixing"), c_method="gauss_legendre",
        c_n_gauss=n_gauss, progress=False)
