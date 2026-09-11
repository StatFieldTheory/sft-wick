r"""Demo 7's Gaussian model: the package side, and the covariances the
reference needs.

.. math::

    \frac{dφ_a}{dt} = −γ_a φ_a + F_{abc} φ_b φ_c + G_{abcd} φ_b φ_c φ_d
                      + η_a(x, t) + ξ_a(t),
    \qquad t ≥ t_{\min},\ φ(t_{\min}) = 0,

with coloured Gaussian noise ``η`` whose temporal kernel is exponential,
``λ e^{−|t−t'|/σ_t}``, and whose spatial kernel is one of

=============  =====================================  ==========================
homogeneity    spatial kernel                          package spec
=============  =====================================  ==========================
translation    ``e^{−r/σ_x}``, ``e^{−r²/2σ_x²}``,       ``SeparableTranslation``
               ``e^{−r/ℓ} cos(k r)``                    (``ExponentialSpatial``,
                                                        ``GaussianSpatial``,
                                                        ``CustomKernel``)
rotation       ``Σ_ℓ C_ℓ P_ℓ(cos θ)``, four ``C_ℓ``     ``SeparableRotation``
                                                        (``LegendreAngular``)
general        ``λ_a e^{−|Δt|/σ_{t,a}} g_a(x) g_a(x')   ``GeneralKappa2``
               e^{−(x−x')²/2s_a²}``, per component
=============  =====================================  ==========================

and optional white noise ``ξ`` with a dense covariance ``S``
(``ConstantImpulse``), the same at every point.  ``F`` and ``G`` have no
index symmetry.  Every kernel callable is a frozen dataclass (a stable
``repr`` for the caches, picklable for joblib).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

import sft_wick as sw
from sft_wick.workflow.specs import ConstantImpulse, CustomKernel, GeneralKappa2

import space7_reference as ref

__all__ = ["F_TENSOR", "G_TENSOR", "S_MIX", "DampedCosine",
           "InhomogeneousKappa2", "Config", "channel_tag", "CONFIGS"]

#: ``dφ_a/dt ⊃ F_abc φ_b φ_c``; no index symmetry.
F_TENSOR = np.array([[[0.30, -0.45], [0.20, 0.15]],
                     [[-0.25, 0.40], [0.35, -0.20]]])

#: ``dφ_a/dt ⊃ G_abcd φ_b φ_c φ_d``; no index symmetry.  Chosen so that no
#: component pair's order-1 channel cancels: ``Σ_c (G_accb + G_acbc +
#: G_abcc)`` is between 0.69 and 0.94 in magnitude for every ``(a, b)``.
G_TENSOR = np.array([[[[0.12, -0.21], [0.17, 0.09]],
                      [[-0.14, 0.23], [0.06, -0.18]]],
                     [[[-0.11, 0.19], [0.08, -0.16]],
                      [[0.22, -0.07], [-0.13, 0.15]]]])

#: White-noise covariance rate of the mixing variant (dense).
S_MIX = ((0.30, 0.12), (0.12, 0.22))


@dataclass(frozen=True)
class DampedCosine:
    """``κ_x(r) = e^{−r/ℓ} cos(k r)`` for ``CustomKernel``: positive definite
    on the line, negative for ``π/2 < k r < 3π/2``."""

    ell: float = 1.5
    k: float = 1.6

    def __call__(self, r) -> float:
        r = abs(float(r))
        return float(np.exp(-r / self.ell) * np.cos(self.k * r))


@dataclass(frozen=True)
class InhomogeneousKappa2:
    r"""``κ²_ab(x, t; x', t') = δ_ab λ_a e^{−|t−t'|/σ_{t,a}} g_a(x) g_a(x')
    e^{−(x−x')²/(2 s_a²)}``, ``g_a(x) = e^{−(x − c_a)²/(2 L_a²)}``: diagonal in
    the components, every parameter component-dependent, and a function of
    ``x`` and ``x'`` separately (no spatial symmetry)."""

    lam: tuple = (0.5, 0.35)
    sigma_t: tuple = (0.6, 1.1)
    s: tuple = (0.9, 1.4)
    centre: tuple = (0.8, -0.4)
    length: tuple = (0.9, 1.3)

    def spatial(self, a: int, x1: float, x2: float) -> float:
        g1 = np.exp(-(x1 - self.centre[a]) ** 2 / (2.0 * self.length[a] ** 2))
        g2 = np.exp(-(x2 - self.centre[a]) ** 2 / (2.0 * self.length[a] ** 2))
        return float(g1 * g2 * np.exp(-(x1 - x2) ** 2 / (2.0 * self.s[a] ** 2)))

    def __call__(self, n1, t1, n2, t2) -> np.ndarray:
        x1 = float(np.asarray(n1, dtype=float).reshape(-1)[0])
        x2 = float(np.asarray(n2, dtype=float).reshape(-1)[0])
        N = len(self.lam)
        out = np.zeros((N, N))
        for a in range(N):
            out[a, a] = (self.lam[a] * np.exp(-abs(t1 - t2) / self.sigma_t[a])
                         * self.spatial(a, x1, x2))
        return out


def channel_tag(label: str, order: int) -> tuple:
    """The hierarchy tag ``(k_F, k_G)`` of the package's channel ``label``
    (``Expansion.by_vertex_type`` keys) at ``order ≤ 2``."""
    if order > 2:
        raise ValueError("the label fixes (k_F, k_G) only up to order 2")
    return {"": (0, 0), "F": (order, 0), "G": (0, order),
            "FG": (1, 1)}[label]


@dataclass(frozen=True)
class Config:
    """One noise model, its package propagators and its reference.

    Args:
        name: label in the results.
        homogeneity: ``"translation"``, ``"rotation"`` or ``"general"``.
        gamma: decay rates; unequal rates make R a (diagonal) matrix.
        spatial: translation kernel, ``"exponential"``, ``"gaussian"`` or
            ``"custom"`` (:class:`DampedCosine`).
        white: dense white-noise covariance rate ``S`` or ``None``.
        c_source: ``"closed_form"`` (the built-in closed form; translation
            only) or ``"quadrature"`` (the package's C tables).
        n_grid_t, c_method, c_n_gauss: quadrature-table settings.
    """

    name: str
    homogeneity: str = "translation"
    gamma: tuple = (1.0, 1.0)
    lam: float = 0.5
    sigma_t: float = 0.6
    spatial: str = "exponential"
    sigma_x: float = 0.9
    custom: DampedCosine = DampedCosine()
    coeffs: tuple = (0.25, 0.6, 0.3, 0.15)
    general: InhomogeneousKappa2 = InhomogeneousKappa2()
    white: tuple | None = None
    c_source: str = "closed_form"
    n_grid_t: int = 40
    c_method: str = "auto"
    c_n_gauss: int = 20
    t_min: float = 0.4

    # -- package side -------------------------------------------------------

    @property
    def matrix_R(self) -> bool:
        return not np.allclose(self.gamma, self.gamma[0])

    @property
    def diag_C(self) -> bool:
        return self.white is None

    def _kappa2(self):
        temporal = sw.ExponentialTemporal(lam=self.lam, sigma_t=self.sigma_t)
        if self.homogeneity == "translation":
            spatial = {"exponential": sw.ExponentialSpatial(self.sigma_x),
                       "gaussian": sw.GaussianSpatial(self.sigma_x),
                       "custom": CustomKernel(fn=self.custom)}[self.spatial]
            return sw.SeparableTranslation(temporal=temporal, spatial=spatial)
        if self.homogeneity == "rotation":
            return sw.SeparableRotation(
                temporal=temporal,
                angular=sw.LegendreAngular(coeffs=list(self.coeffs)))
        if self.homogeneity == "general":
            return GeneralKappa2(fn=self.general)
        raise ValueError(f"unknown homogeneity {self.homogeneity!r}")

    def system(self) -> sw.System:
        sigma2 = (None if self.white is None
                  else ConstantImpulse(np.asarray(self.white, dtype=float)))
        return sw.System(
            field=sw.FieldSpec("phi", 2),
            linear=sw.DiagonalA(gamma=list(self.gamma)),
            vertices=[sw.LocalVertex("F", coupling=F_TENSOR),
                      sw.LocalVertex("G", coupling=G_TENSOR)],
            noise=sw.GaussianNoise(kappa2=self._kappa2(), sigma2=sigma2),
            t_min=self.t_min)

    def expand_flags(self) -> dict:
        return dict(diag_C=self.diag_C)

    def propagators(self, system: sw.System, t_max: float):
        if self.c_source == "closed_form":
            return system.propagators(
                t_max=t_max, c_closed_form="auto", c_closed_form_only=True,
                c_closed_form_vectorized=True, diag_C=self.diag_C,
                progress=False)
        if not self.diag_C:
            raise ValueError("the quadrature tables hold C_aa only")
        return system.propagators(
            t_max=t_max, n_grid_t=self.n_grid_t, c_closed_form=None,
            c_method=self.c_method, c_n_gauss=self.c_n_gauss, progress=False)

    # -- reference side -----------------------------------------------------

    @property
    def rho(self) -> tuple:
        if self.homogeneity == "general":
            return tuple(1.0 / s for s in self.general.sigma_t)
        return (1.0 / self.sigma_t,) * len(self.gamma)

    def sigma(self, points) -> np.ndarray:
        """``Σ^a_ij``, the stationary covariance of ``η_a`` at the points,
        from the kernels written out in :mod:`space7_reference`."""
        N = len(self.gamma)
        if self.homogeneity == "general":
            g = self.general
            return np.stack([
                g.lam[a] * ref.covariance(points, lambda x1, x2, a=a:
                                          ref.inhomogeneous_kernel(
                                              g.s[a], g.centre[a],
                                              g.length[a], x1, x2))
                for a in range(N)])
        if self.homogeneity == "rotation":
            def kern(n1, n2):
                n1 = np.asarray(n1, float)
                n2 = np.asarray(n2, float)
                cos = float(n1 @ n2 / np.sqrt((n1 @ n1) * (n2 @ n2)))
                return ref.legendre_series(self.coeffs, cos)
        else:
            base = {"exponential": lambda r: ref.exponential_kernel(
                        self.sigma_x, r),
                    "gaussian": lambda r: ref.gaussian_kernel(
                        self.sigma_x, r),
                    "custom": lambda r: ref.damped_cosine_kernel(
                        self.custom.ell, self.custom.k, r)}[self.spatial]

            def kern(x1, x2):
                d = np.atleast_1d(np.asarray(x1, float) - np.asarray(x2, float))
                return base(float(np.sqrt(d @ d)))
        K = ref.covariance(points, kern)
        return np.stack([self.lam * K] * N)

    def hierarchy(self, points, integrated=()) -> ref.Hierarchy:
        return ref.Hierarchy(
            n_points=len(points), gamma=self.gamma, rho=self.rho,
            sigma=self.sigma(points), F=F_TENSOR, G=G_TENSOR,
            S=self.white, integrated=integrated)

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)


#: The configurations of the demo, by name.
CONFIGS = {
    c.name: c for c in [
        # (a), (e): translation, exponential kernel, exact C.
        Config("translation-exp"),
        Config("translation-exp-matrixR", gamma=(0.7, 1.4)),
        Config("translation-exp-white-matrixR", gamma=(0.7, 1.4),
               white=S_MIX),
        # (b): rotation, four Legendre coefficients, quadrature tables.
        Config("rotation-legendre", homogeneity="rotation",
               c_source="quadrature", n_grid_t=48),
        Config("rotation-legendre-matrixR", homogeneity="rotation",
               gamma=(0.7, 1.4), c_source="quadrature", n_grid_t=32),
        # (c): the package's quadrature tables for three more kernels.
        Config("translation-gauss-quad", spatial="gaussian", sigma_x=0.8,
               c_source="quadrature", n_grid_t=48),
        Config("translation-custom-quad", spatial="custom",
               c_source="quadrature", n_grid_t=48),
        Config("general-quad", homogeneity="general", c_source="quadrature",
               c_method="gauss_legendre", c_n_gauss=12, n_grid_t=40),
        Config("general-quad-matrixR", homogeneity="general",
               gamma=(0.7, 1.4), c_source="quadrature",
               c_method="gauss_legendre", c_n_gauss=12, n_grid_t=30),
    ]
}
