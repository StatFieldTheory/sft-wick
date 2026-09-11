r"""Demo 8: parameters and model functions (numpy only).

Four models of a two-component field ``φ_a`` observed at the points
``x, y, z``, each started from ``φ(t_min) = 0``:

(a) **rate**: ``dφ_a = (−γ_a(t) φ_a + F_abc φ_b φ_c + η_a) dt`` with
    ``γ(t) = [1 + 0.5 sin t, 0.6 + 0.3 cos 2t]`` and an Ornstein-Uhlenbeck
    noise ``⟨η_a(x,t) η_b(y,t')⟩ = δ_ab λ e^{−|t−t'|/σ_t} K(x−y)``;
(b) **oscillator**: ``ẍ_a + 2ζω ẋ_a + ω² x_a = F_abc x_b x_c +
    G_abcd x_b x_c x_d + ξ_a`` with white ``ξ``,
    ``⟨ξ_a(x,t) ξ_b(y,t')⟩ = δ_ab s_a K_w(x−y) δ(t−t')``, so
    ``R(τ) = e^{−ζωτ} sin(ω_d τ)/ω_d``;
(c) **kernels**: ``dφ_a = (−γ_a φ_a + F_abc φ_b φ_c + η_a) dt`` with a
    Matérn-3/2, damped-cosine or Gaussian temporal covariance of ``η``;
(d) **white**: model (c) with an exponential (OU) kernel, plus white noise
    ``dW_a``, ``⟨dW_a dW_b⟩ = δ_ab s_a(t) K_w(x−y) dt``, whose amplitude
    ``s_a`` varies in time.

The callables below are frozen dataclasses (stable ``repr``, picklable);
the package receives them as ``DiagonalA.gamma``, ``ExplicitR.R_time``,
``CustomKernel.fn`` and ``CustomImpulse.fn``, and the reference evaluates
the same functions.  This module imports nothing from sft-wick.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "N", "POSITIONS", "F_QUAD", "G_CUBIC",
    "SinusoidalRate", "UNEQUAL_RATE", "EQUAL_RATE",
    "OscillatorR", "Matern32", "DampedCosine", "ExponentialKernel",
    "GaussianKernel",
    "GaussianEnvelope", "WhiteSigma2",
    "RateParams", "OscillatorParams", "KernelParams", "WhiteParams",
]

N = 2
POSITIONS = {"x": 0.0, "y": 0.7, "z": -0.4}

#: ``dφ_a/dt ⊃ F_abc φ_b φ_c``; no index symmetry, both signs, and the two
#: tadpole traces ``Σ_b F_abb`` (0.35, −0.25) differ in magnitude, so a
#: swapped component index cannot pass as a sign.
F_QUAD = np.array([[[0.25, -0.40], [0.15, 0.10]],
                   [[-0.20, 0.30], [0.45, -0.05]]])

#: ``ẍ_a ⊃ G_abcd x_b x_c x_d`` (model b); no index symmetry.
G_CUBIC = np.array([[[[0.12, -0.08], [0.05, 0.10]],
                     [[-0.06, 0.09], [0.11, -0.04]]],
                    [[[0.07, 0.03], [-0.10, 0.06]],
                     [[0.08, -0.12], [0.04, 0.09]]]])


# ---------------------------------------------------------------------------
# (a) a decay rate varying in time
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SinusoidalRate:
    """``γ_a(t) = c_a + A_a sin(w_a t + p_a)``, one entry per component."""

    c: tuple = (1.0, 0.6)
    A: tuple = (0.5, 0.3)
    w: tuple = (1.0, 2.0)
    p: tuple = (0.0, np.pi / 2)      # 0.6 + 0.3 cos 2t

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        return np.array([c + A * np.sin(w * t + p)
                         for c, A, w, p in zip(self.c, self.A, self.w, self.p)])

    def Gamma(self, t):
        """``Γ_a(t) = ∫^t γ_a``, the exact antiderivative (any origin)."""
        t = np.asarray(t, dtype=float)
        return np.array([c * t - (A / w) * np.cos(w * t + p)
                         for c, A, w, p in zip(self.c, self.A, self.w, self.p)])

    def R(self, a: int, t, s):
        """``R_aa(t, s) = exp(−(Γ_a(t) − Γ_a(s)))`` for ``t ≥ s``."""
        return np.exp(-(self.Gamma(t)[a] - self.Gamma(s)[a]))


UNEQUAL_RATE = SinusoidalRate()
#: Both components at component 0's rate: a scalar R, which every
#: integrator accepts today.
EQUAL_RATE = SinusoidalRate(c=(1.0, 1.0), A=(0.5, 0.5), w=(1.0, 1.0),
                            p=(0.0, 0.0))


# ---------------------------------------------------------------------------
# (b) a damped oscillator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OscillatorR:
    """``R(t, s) = Θ(t − s) e^{−ζω(t−s)} sin(ω_d (t−s)) / ω_d``: the
    response of ``ẍ + 2ζω ẋ + ω² x`` to a unit impulse of force."""

    omega: float = 1.7
    zeta: float = 0.3

    @property
    def omega_d(self) -> float:
        return self.omega * np.sqrt(1.0 - self.zeta ** 2)

    def tau(self, tau):
        tau = np.asarray(tau, dtype=float)
        out = (np.exp(-self.zeta * self.omega * tau)
               * np.sin(self.omega_d * tau) / self.omega_d)
        return np.where(tau > 0.0, out, 0.0)

    def __call__(self, t1, t2):
        if t1 <= t2:
            return 0.0
        return float(self.tau(t1 - t2))


# ---------------------------------------------------------------------------
# (c) temporal kernels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Matern32:
    """``λ (1 + a|τ|) e^{−a|τ|}``: the stationary covariance of ``η`` in
    ``η̇ = ξ``, ``ξ̇ = −a² η − 2a ξ + √(4a³λ) w``."""

    lam: float = 0.5
    a: float = 1.6

    def __call__(self, tau):
        u = self.a * np.abs(np.asarray(tau, dtype=float))
        return self.lam * (1.0 + u) * np.exp(-u)


@dataclass(frozen=True)
class DampedCosine:
    """``λ e^{−a|τ|} cos(ω τ)``: the stationary covariance of ``z_1`` in
    ``ż = [[−a, −ω], [ω, −a]] z + √(2aλ) w``."""

    lam: float = 0.5
    a: float = 1.2
    omega: float = 2.5

    def __call__(self, tau):
        tau = np.asarray(tau, dtype=float)
        return self.lam * np.exp(-self.a * np.abs(tau)) * np.cos(self.omega * tau)


@dataclass(frozen=True)
class ExponentialKernel:
    """``λ e^{−|τ|/σ}``, the kernel of ``sw.ExponentialTemporal`` (an OU
    process); used to check the hand contraction against the hierarchy."""

    lam: float = 0.5
    sigma: float = 0.6

    def __call__(self, tau):
        return self.lam * np.exp(-np.abs(np.asarray(tau, dtype=float)) / self.sigma)


@dataclass(frozen=True)
class GaussianKernel:
    """``λ e^{−τ²/(2σ²)}``, the kernel of ``sw.GaussianTemporal``.  No
    finite linear SDE has it as a stationary covariance."""

    lam: float = 0.5
    sigma: float = 0.45

    def __call__(self, tau):
        tau = np.asarray(tau, dtype=float)
        return self.lam * np.exp(-tau * tau / (2.0 * self.sigma ** 2))


@dataclass(frozen=True)
class GaussianEnvelope:
    """``K(r) = e^{−r²/(2ℓ²)}``, the spatial factor of every noise here."""

    ell: float = 0.9

    def __call__(self, r):
        r = np.asarray(r, dtype=float)
        return np.exp(-r * r / (2.0 * self.ell ** 2))


# ---------------------------------------------------------------------------
# (b), (d) white noise, possibly varying in time
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WhiteSigma2:
    """``σ²_ab(t; x, y) = δ_ab s_a(t) K_w(x − y)`` with
    ``s_a(t) = S_a (1 + m_a sin(w_a t + p_a))``.

    Called by the package as ``σ²(n1, t, n2) -> (N, N)``.
    """

    S: tuple = (0.5, 0.35)
    m: tuple = (0.6, 0.5)
    w: tuple = (1.7, 2.3)
    p: tuple = (0.3, np.pi / 2)
    ell: float = 1.1

    def s(self, t):
        t = np.asarray(t, dtype=float)
        return np.array([S * (1.0 + m * np.sin(w * t + p))
                         for S, m, w, p in zip(self.S, self.m, self.w, self.p)])

    def spatial(self, r):
        r = np.asarray(r, dtype=float)
        return np.exp(-r * r / (2.0 * self.ell ** 2))

    def __call__(self, n1, t, n2):
        r = float(np.linalg.norm(np.atleast_1d(np.asarray(n1, float)
                                              - np.asarray(n2, float))))
        return np.diag(self.s(float(t))) * float(self.spatial(r))


# ---------------------------------------------------------------------------
# Parameter sets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RateParams:
    """Model (a)."""

    rate: SinusoidalRate = UNEQUAL_RATE
    lam: float = 0.5
    sigma_t: float = 0.6
    sigma_x: float = 0.9
    t_min: float = -1.3
    span: float = 1.9

    @property
    def t_final(self) -> float:
        return self.t_min + self.span


@dataclass(frozen=True)
class OscillatorParams:
    """Model (b).  ``span`` exceeds ``π/ω_d = 1.94``, where R changes sign."""

    R: OscillatorR = OscillatorR()
    white: WhiteSigma2 = WhiteSigma2(S=(0.6, 0.35), m=(0.0, 0.0), ell=0.8)
    t_min: float = 0.4
    span: float = 3.1

    @property
    def t_final(self) -> float:
        return self.t_min + self.span


@dataclass(frozen=True)
class KernelParams:
    """Model (c)."""

    kernel: object = Matern32()
    gamma: tuple = (0.8, 1.3)
    sigma_x: float = 0.9
    t_min: float = -0.5
    span: float = 2.0

    @property
    def t_final(self) -> float:
        return self.t_min + self.span


@dataclass(frozen=True)
class WhiteParams:
    """Model (d)."""

    gamma: tuple = (0.9, 1.4)
    lam: float = 0.3
    sigma_t: float = 0.5
    sigma_x: float = 0.9
    white: WhiteSigma2 = field(default_factory=WhiteSigma2)
    t_min: float = 0.4
    span: float = 1.7

    @property
    def t_final(self) -> float:
        return self.t_min + self.span
