r"""Built-in closed-form C propagators.

The built-in ``C = ∫∫ R κ² R`` uses the following structure:

* diagonal, constant drift, :class:`~sft_wick.workflow.specs.DiagonalA`
  with a constant ``gamma`` (scalar or per component), so
  ``R_aa(t, s) = Θ(t − s) exp(−γ_a (t − s))``;
* separable, translation-invariant noise with any spatial envelope and an
  exponential, Gaussian, Matérn-3/2 or damped-cosine temporal kernel;
* optionally a constant white-noise impulse
  :class:`~sft_wick.workflow.specs.ConstantImpulse`.

For the exponential kernel, with ``a = 1/σ_t`` and times measured from
``t_min``,

.. math::

   C_{aa}(t_1, t_2; r) = K_x(r)\,\lambda\,\Phi(\gamma_a, a; t_1, t_2)
                         + \sigma^2_{aa}\,\Psi(\gamma_a, \gamma_a; t_1, t_2)

where, for ``t_1 ≤ t_2``, ``d = t_2 − t_1`` and
``D(p, q; T) = (e^{-pT} − e^{-qT})/(q − p)``,

.. math::

   (\gamma + a)\,\Phi = e^{-\gamma d}\frac{1 - e^{-2\gamma t_1}}{\gamma}
       - e^{-\gamma t_2} D(\gamma, a; t_1)
       + D(\gamma, a; d)\,\bigl(1 - e^{-(\gamma+a) t_1}\bigr)
       - e^{-\gamma d}\,D(2\gamma, \gamma + a; t_1),

   \Psi(\gamma_a, \gamma_b; t_1, t_2) =
       e^{-\gamma_a (t_1 - t_l) - \gamma_b (t_2 - t_l)}
       \frac{1 - e^{-(\gamma_a + \gamma_b) t_l}}{\gamma_a + \gamma_b},
       \qquad t_l = \min(t_1, t_2).

Every exponential above has a non-positive argument, so the expression
is overflow-free at any ``t`` (the textbook form multiplies
``exp(+2γ t)`` by ``exp(−γ(t₁ + t₂))`` and overflows near ``γ t ≈ 350``),
and the removable singularities at ``a = γ`` and ``γ = 0`` are handled
through ``D`` and ``(1 − e^{-x})/x``, which are evaluated with ``expm1``.

The object returned by :func:`builtin_closed_form_for` is a callable with
the same ``(n1, t1, n2, t2)`` contract as a user ``c_closed_form_module``:
scalar times give an ``(N, N)`` matrix, batched ``(n,)`` times give
``(n, N, N)``.  It plugs into every existing path, including
``c_closed_form_only=True`` with the vectorised lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial, isfinite
from typing import Any, Callable

import numpy as np

from ._stationary_c import damped_cosine_phi, gaussian_phi, matern32_phi
from .specs import (
    ConstantImpulse,
    DampedCosineTemporal,
    DiagonalA,
    ExponentialTemporal,
    GaussianTemporal,
    Matern32Temporal,
    MultiplicativeImpulse,
    SeparableTranslation,
)


# --------------------------------------------------------------------- #
# Numerically stable primitives                                          #
# --------------------------------------------------------------------- #


#: Below this ``(γ + a) t`` the closed form of ``Φ`` loses it to cancellation
#: (:func:`ou_exponential_phi`).
_PHI_SMALL_RATE = 1.0

#: Taylor coefficients of ``ψ(x) = (1 − e^{−x}) / x``.  Truncating here leaves
#: less than ``2e−19`` over ``0 ≤ x ≤ 2`` (and over its divided differences),
#: the range :func:`_phi_at_small_rates` reads.
_PSI_TAYLOR = tuple((-1.0) ** k / factorial(k + 1) for k in range(25))


# The CSE rule uses a divided difference of psi below a rate-time
# product of 0.5, where sixteen Taylor terms suffice in binary64.
_OU_CSE_TAYLOR = _PSI_TAYLOR[:16]
_OU_CSE_SMALL_ARGUMENT = 0.5
_OU_CSE_MIN_NORMAL = np.finfo(float).tiny
_OU_CSE_TINY_RATE = np.sqrt(_OU_CSE_MIN_NORMAL)


def _psi(x: np.ndarray) -> np.ndarray:
    """``(1 − e^{−x}) / x`` with ``ψ(0) = 1``; accurate for all ``x ≥ −1``."""
    x = np.asarray(x, dtype=float)
    zero = x == 0.0
    xs = np.where(zero, 1.0, x)
    return np.where(zero, 1.0, -np.expm1(-xs) / xs)


def _dexp(p: Any, q: Any, T: Any) -> np.ndarray:
    """``D(p, q; T) = (e^{−pT} − e^{−qT}) / (q − p)`` for ``p, q, T ≥ 0``.

    Equals ``T e^{−pT}`` at ``q == p``.  Near that point (``|q − p| T < 0.5``)
    the difference is formed through ``expm1`` instead of by cancellation.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    T = np.asarray(T, dtype=float)
    if not p.shape == q.shape == T.shape:
        p, q, T = np.broadcast_arrays(p, q, T)
    x = (q - p) * T
    near = np.abs(x) < 0.5
    e_p = np.exp(-p * T)
    out = np.asarray((e_p - np.exp(-q * T)) / np.where(near, 1.0, q - p))
    if near.any():
        out[near] = T[near] * e_p[near] * _psi(x[near])
    return out


def _phi_at_small_rates(
    gamma: np.ndarray, a: np.ndarray, tl: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """``Φ`` at ``t₁ = tl + d``, ``t₂ = tl`` as a sum of positive terms, for
    ``(γ + a)(tl + d) < 1``.

    The strip of the later time past ``tl`` gives
    ``D(γ, a; d) tl ψ((γ + a) tl)`` and the square gives
    ``2 e^{−γd} ∫₀^{tl} D(γ + a, 2γ; v) dv``.  With ``x₁ = (γ + a) tl`` and
    ``x₂ = 2γ tl`` that integral is ``tl²`` times minus the divided difference
    ``(ψ(x₂) − ψ(x₁)) / (x₂ − x₁)``, which has no ``0 / 0`` at ``γ = a``.  One
    Horner recurrence on the Taylor series gives ``ψ(x₁)``, and a second
    gives the divided difference from the first one's partial values.
    """
    x1 = (gamma + a) * tl
    x2 = 2.0 * gamma * tl
    psi_x1 = np.full(x1.shape, _PSI_TAYLOR[-1])
    slope = np.zeros(x1.shape)
    for coefficient in reversed(_PSI_TAYLOR[:-1]):
        slope = slope * x2 + psi_x1
        psi_x1 = psi_x1 * x1 + coefficient
    square = -2.0 * np.exp(-gamma * d) * tl * tl * slope
    return _dexp(gamma, a, d) * tl * psi_x1 + square


def _phi_closed(
    gamma: np.ndarray, a: np.ndarray, tl: np.ndarray, th: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """``Φ`` from its closed form, for ``(γ + a) th ≥ 1`` (module docstring)."""
    egd = np.exp(-gamma * d)
    term = (
        egd * 2.0 * tl * _psi(2.0 * gamma * tl)
        - np.exp(-gamma * th) * _dexp(gamma, a, tl)
        + _dexp(gamma, a, d) * (-np.expm1(-(gamma + a) * tl))
        - egd * _dexp(2.0 * gamma, gamma + a, tl)
    )
    return term / (gamma + a)


def ou_exponential_phi(gamma: Any, a: Any, t1: Any, t2: Any) -> np.ndarray:
    """``Φ(γ, a; t₁, t₂) = ∫₀^{t₁}∫₀^{t₂} e^{−γ(t₁−λ₁)} e^{−a|λ₁−λ₂|} e^{−γ(t₂−λ₂)} dλ₁ dλ₂``.

    All arguments broadcast; ``gamma ≥ 0``, ``a ≥ 0``.  Zero when either
    time is ``≤ 0`` (empty domain).

    The closed form divides by ``γ + a`` a sum that cancels to ``(γ + a) t²``,
    so it loses ``1 / ((γ + a) t)`` of its digits, and is ``0 / 0`` at
    ``γ = a = 0``.  Below ``(γ + a) t = 1`` :func:`_phi_at_small_rates` holds
    instead.
    """
    gamma, a, t1, t2 = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (gamma, a, t1, t2))
    )
    tl = np.minimum(t1, t2)
    th = np.maximum(t1, t2)
    tl_pos = np.where(tl > 0, tl, 0.0)
    d = th - tl_pos
    small = (gamma + a) * th < _PHI_SMALL_RATE
    if small.all():
        out = _phi_at_small_rates(gamma, a, tl_pos, d)
    elif not small.any():
        out = _phi_closed(gamma, a, tl_pos, th, d)
    else:
        # Each path on its own elements: _psi overflows far outside the small
        # branch, and the closed form is 0 / 0 at γ = a = 0.
        large = ~small
        out = np.empty(small.shape)
        out[small] = _phi_at_small_rates(
            gamma[small], a[small], tl_pos[small], d[small]
        )
        out[large] = _phi_closed(
            gamma[large], a[large], tl_pos[large], th[large], d[large]
        )
    return np.where(tl > 0, out, 0.0)


def _ou_integrated_exp(rate: float, duration: np.ndarray) -> np.ndarray:
    """Integral of exp(-rate * u) from zero to duration, at a fixed rate."""
    if rate == 0.0:
        return duration
    return np.expm1((-rate) * duration) * (-1.0 / rate)


def _ou_cse_small_square(
    rate_min: float, rate_max: float, duration: np.ndarray,
) -> np.ndarray:
    """Two equal triangles in the square where both times are early."""
    x = rate_min * duration
    y = rate_max * duration
    value = _OU_CSE_TAYLOR[-1]
    slope = 0.0
    for coefficient in reversed(_OU_CSE_TAYLOR[:-1]):
        slope = slope * y + value
        value = value * x + coefficient
    return (-2.0 * duration * duration) * slope


def _ou_cse_square(
    rate_min: float, rate_max: float, rate_delta: float, duration: np.ndarray,
) -> np.ndarray:
    """Twice the integral of D(rate_min, rate_max; u) along the early square."""
    threshold = _OU_CSE_SMALL_ARGUMENT / rate_max
    if np.max(duration, initial=0.0) < threshold:
        return _ou_cse_small_square(rate_min, rate_max, duration)
    small = duration < threshold
    by_delta = _ou_integrated_exp(rate_delta, duration)
    if rate_min == 0.0:
        square = duration - by_delta
    else:
        exp_m1 = np.expm1((-rate_min) * duration)
        # Reuse expm1 in the integrated exponential and in the small
        # exponential correction. If expm1 rounds to -1, the discarded
        # correction is at most x exp(-x)/(1-exp(-x)) times the integrated
        # term, where x = rate_min * duration is already above 37.
        square = exp_m1 * (-1.0 / rate_min) - (1.0 + exp_m1) * by_delta
    square *= 2.0 / rate_max
    if np.any(small):
        square[small] = _ou_cse_small_square(rate_min, rate_max, duration[small])
    return square


def _ou_cse_underflow(duration: np.ndarray, min_duration: float) -> bool:
    """Check positive times where a multiplication can lose its mantissa."""
    return (
        np.min(duration, initial=np.inf) < min_duration
        and np.any((duration > 0.0) & (duration < min_duration))
    )


def _ou_exponential_phi_cse(
    gamma: float, a: float, t1: Any, t2: Any,
) -> np.ndarray:
    """OU/exponential covariance at fixed rates using a square and a strip.

    Use the dimensionless reference path when tiny rates can underflow
    before division. Each argument time is measured from the lower bound.
    """
    sum_rate = gamma + a
    twice_gamma = 2.0 * gamma
    delta_rate = abs(gamma - a)
    if sum_rate == 0 or not isfinite(max(sum_rate, twice_gamma)):
        return ou_exponential_phi(gamma, a, t1, t2)
    min_rate = min(rate for rate in (sum_rate, twice_gamma, delta_rate) if rate > 0)
    if min_rate < _OU_CSE_TINY_RATE:
        return ou_exponential_phi(gamma, a, t1, t2)

    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    early = np.asarray(np.minimum(t1, t2))
    np.maximum(early, 0.0, out=early)
    delay = np.asarray(np.maximum(t1, t2))
    delay -= early
    np.maximum(delay, 0.0, out=delay)
    min_duration = _OU_CSE_MIN_NORMAL / min_rate
    if _ou_cse_underflow(early, min_duration) or _ou_cse_underflow(delay, min_duration):
        return ou_exponential_phi(gamma, a, t1, t2)

    square = _ou_cse_square(
        min(sum_rate, twice_gamma), max(sum_rate, twice_gamma), delta_rate, early,
    )
    exp_gamma = np.exp(-gamma * delay)
    integrated_early = _ou_integrated_exp(sum_rate, early)
    if delta_rate == 0.0:
        # At equal rates this integrated delay is the delay itself.
        # Apply its damping before forming a potentially very large product.
        strip = (exp_gamma * delay) * integrated_early
        return np.asarray(exp_gamma * square + strip)
    strip = _ou_integrated_exp(delta_rate, delay) * integrated_early
    if gamma <= a:
        return np.asarray(exp_gamma * (square + strip))
    return np.asarray(exp_gamma * square + np.exp(-a * delay) * strip)


def ou_white_psi(gamma_a: Any, gamma_b: Any, t1: Any, t2: Any) -> np.ndarray:
    """``Ψ = ∫₀^{min(t₁,t₂)} e^{−γ_a(t₁−τ)} e^{−γ_b(t₂−τ)} dτ`` (broadcasting)."""
    ga = np.asarray(gamma_a, dtype=float)
    gb = np.asarray(gamma_b, dtype=float)
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    tl = np.minimum(t1, t2)
    tl_pos = np.where(tl > 0, tl, 0.0)
    s = ga + gb
    pref = np.exp(-ga * (t1 - tl_pos) - gb * (t2 - tl_pos))
    out = pref * tl_pos * _psi(s * tl_pos)
    return np.where(tl > 0, out, 0.0)


# --------------------------------------------------------------------- #
# The callable                                                           #
# --------------------------------------------------------------------- #


def _resolve_positions(x: Any, n: int) -> np.ndarray:
    """Scalar / ``(n,)`` / ``(d,)`` / ``(n, d)`` → ``(n,)`` or ``(n, d)``.

    Same convention as :meth:`PropagatorCache.C_at_batch`.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.ndim == 1:
        if arr.shape[0] == n:
            return arr
        return np.tile(arr[None, :], (n, 1))
    return arr


def _separation(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    # A one-dimensional position array has one scalar per sample. When
    # paired with vectors, broadcast that scalar across its own row.
    if x1.ndim == 1 and x2.ndim == 2:
        x1 = x1[:, None]
    if x2.ndim == 1 and x1.ndim == 2:
        x2 = x2[:, None]
    diff = x1 - x2
    if diff.ndim == 1:
        return np.abs(diff)
    return np.linalg.norm(diff, axis=-1)


@dataclass(frozen=True)
class ClosedFormC:
    """Finite-start ``C`` for constant diagonal drift and a stationary kernel.

    Instances are produced by :func:`builtin_closed_form_for`; the fields
    are the *lowered* parameters, kept as plain tuples so the object has a
    deterministic ``repr`` (it is part of the on-disk propagator-cache key)
    and pickles cleanly into joblib workers.

    Attributes:
        gamma: per-component decay rates ``(γ_0, …, γ_{N−1})``.
        lam: temporal-kernel amplitude ``λ``.
        sigma_t: temporal correlation time ``σ_t``.
        spatial: the spatial envelope ``K_x(r)``; any scalar callable.
        sigma2: white-noise amplitude matrix as nested tuples, or ``None``.
        t_min: lower time bound of the propagator integrals.
        temporal_kind: built-in temporal kernel name.
        omega: angular frequency for the damped cosine.
        ou_cse: use the fixed-rate OU formula for an exponential kernel.
    """

    gamma: tuple[float, ...]
    lam: float
    sigma_t: float
    spatial: Callable[[float], float]
    sigma2: tuple[tuple[float, ...], ...] | None = None
    t_min: float = 0.0

    #: Advertises the factorisation ``C(r; t₁, t₂) = K_x(r) · C(0; t₁, t₂)``
    #: (white noise apart) to the lazy spline cache.
    separable_translation: bool = True

    #: C is kinked on ``t₁ = t₂``, so the Gauss-Legendre and nquad
    #: integrators split the time domain there
    #: (``evaluate._c_has_diagonal_kink``). The Gaussian part is smooth;
    #: white noise and the other kernels benefit from a diagonal split.
    has_diagonal_kink: bool = True
    temporal_kind: str = "exponential"
    omega: float = 0.0
    ou_cse: bool = False

    @property
    def n_components(self) -> int:
        return len(self.gamma)

    # -- pieces ---------------------------------------------------------

    def spatial_factor(self, r: Any) -> np.ndarray:
        """``K_x(r)`` for scalar or ``(n,)`` ``r`` (evaluated once per distinct value)."""
        r_arr = np.atleast_1d(np.asarray(r, dtype=float))
        uniq, inv = np.unique(r_arr, return_inverse=True)
        vals = np.array([float(self.spatial(float(u))) for u in uniq])
        out = vals[np.asarray(inv).reshape(r_arr.shape)]
        return out if np.ndim(r) else float(out[0])

    def temporal(self, t1: Any, t2: Any) -> np.ndarray:
        """Temporal part of C, with output shape ``t.shape + (N,)``."""
        t1s = np.asarray(t1, dtype=float) - self.t_min
        t2s = np.asarray(t2, dtype=float) - self.t_min
        if self.temporal_kind == "exponential" and self.ou_cse:
            a = 1.0 / self.sigma_t
            out = np.empty((*np.broadcast_shapes(t1s.shape, t2s.shape), self.n_components))
            for component, rate in enumerate(self.gamma):
                out[..., component] = _ou_exponential_phi_cse(rate, a, t1s, t2s)
            out *= self.lam
            return out
        t1s, t2s = t1s[..., None], t2s[..., None]
        g = np.asarray(self.gamma, dtype=float)
        if self.temporal_kind == "exponential":
            phi = ou_exponential_phi(g, 1.0 / self.sigma_t, t1s, t2s)
        elif self.temporal_kind == "gaussian":
            phi = gaussian_phi(g, g, self.sigma_t, t1s, t2s)
        elif self.temporal_kind == "matern32":
            phi = matern32_phi(g, g, 3.0 ** 0.5 / self.sigma_t, t1s, t2s)
        elif self.temporal_kind == "damped_cosine":
            phi = damped_cosine_phi(g, g, 1.0 / self.sigma_t, self.omega, t1s, t2s)
        else:
            raise ValueError(f"Unsupported temporal kernel: {self.temporal_kind!r}")
        return self.lam * phi

    def white(self, t1: Any, t2: Any) -> np.ndarray | None:
        """White-noise part ``σ²_ab Ψ(γ_a, γ_b; …)`` → ``t.shape + (N, N)``, or ``None``."""
        if self.sigma2 is None:
            return None
        t1s = np.asarray(t1, dtype=float) - self.t_min
        t2s = np.asarray(t2, dtype=float) - self.t_min
        g = np.asarray(self.gamma, dtype=float)
        psi = ou_white_psi(
            g[:, None], g[None, :], t1s[..., None, None], t2s[..., None, None],
        )
        return np.asarray(self.sigma2, dtype=float) * psi

    # -- the (n1, t1, n2, t2) contract ------------------------------------

    def __call__(self, n1: Any, t1: Any, n2: Any, t2: Any) -> np.ndarray:
        t1a = np.asarray(t1, dtype=float)
        t2a = np.asarray(t2, dtype=float)
        scalar = t1a.ndim == 0 and t2a.ndim == 0
        t1v = np.atleast_1d(t1a)
        t2v = np.atleast_1d(t2a)
        n = max(t1v.shape[0], t2v.shape[0])
        t1v = np.broadcast_to(t1v, (n,))
        t2v = np.broadcast_to(t2v, (n,))
        r = _separation(_resolve_positions(n1, n), _resolve_positions(n2, n))
        N = self.n_components
        diag = self.spatial_factor(r)[:, None] * self.temporal(t1v, t2v)  # (n, N)
        C = np.zeros((n, N, N))
        idx = np.arange(N)
        C[:, idx, idx] = diag
        white = self.white(t1v, t2v)
        if white is not None:
            C = C + white
        return C[0] if scalar else C


# --------------------------------------------------------------------- #
# Detection                                                              #
# --------------------------------------------------------------------- #


def builtin_closed_form_for(system: Any, *, ou_cse: bool = False) -> ClosedFormC | None:
    """Return the built-in closed form for ``system`` if its kernel family
    admits one, else ``None``.

    The conditions are exactly those listed in the module docstring; any
    escape hatch (``explicit_R``, :class:`ExplicitR`, a callable ``gamma``,
    a custom temporal kernel, a :class:`CustomImpulse`) disables
    the closed form and the caller falls back to quadrature.
    """
    if getattr(system, "explicit_R", None) is not None:
        return None
    linear = system.linear
    if not isinstance(linear, DiagonalA) or callable(linear.gamma):
        return None
    gamma = np.atleast_1d(np.asarray(linear.gamma, dtype=float))
    if gamma.ndim != 1 or not np.all(np.isfinite(gamma)) or np.any(gamma < 0):
        return None
    N = int(system.n_components)
    # Mirror the model that ``System.build_propagator_model`` builds: when
    # ``DiagonalA`` judges the rates equal it lowers to a scalar R with
    # ``gamma[0]`` for EVERY component.  The closed form must describe that
    # model, not the raw list.
    if linear.is_iso_R:
        gamma = np.full(N, float(gamma[0]))
    kappa2 = system.noise.kappa2
    if not isinstance(kappa2, SeparableTranslation):
        return None
    temporal = kappa2.temporal
    kinds = (
        (ExponentialTemporal, "exponential"),
        (GaussianTemporal, "gaussian"),
        (Matern32Temporal, "matern32"),
        (DampedCosineTemporal, "damped_cosine"),
    )
    kind = next((name for cls, name in kinds if type(temporal) is cls), None)
    if kind is None or (ou_cse and kind != "exponential"):
        return None
    if not np.isfinite(temporal.lam):
        return None

    if gamma.shape[0] == 1 and N > 1:
        gamma = np.repeat(gamma, N)
    if gamma.shape[0] != N:
        return None

    sigma2 = system.noise.sigma2
    sigma2_tuple: tuple | None
    if sigma2 is None:
        sigma2_tuple = None
    elif isinstance(sigma2, (ConstantImpulse, MultiplicativeImpulse)):
        # A MultiplicativeImpulse enters C through D0 = g0 g0ᵀ only; its
        # φ-dependent part is vertices.
        amp = np.asarray(sigma2.amplitude, dtype=float)
        mat = float(amp) * np.eye(N) if amp.ndim == 0 else amp
        sigma2_tuple = tuple(tuple(float(v) for v in row) for row in mat)
    else:
        return None

    return ClosedFormC(
        gamma=tuple(float(g) for g in gamma),
        lam=float(temporal.lam),
        sigma_t=float(temporal.sigma_t),
        spatial=kappa2.spatial,
        sigma2=sigma2_tuple,
        t_min=float(system.t_min),
        has_diagonal_kink=kind != "gaussian" or sigma2_tuple is not None,
        temporal_kind=kind,
        omega=float(temporal.omega) if kind == "damped_cosine" else 0.0,
        ou_cse=ou_cse,
    )


# Preserve definitions before the lazy disk-cache identity helper is imported.
from ._native_protocol import capture_native_protocol as _capture_native_protocol

_NATIVE_CALLABLE_PROTOCOL = _capture_native_protocol(globals(), ("ClosedFormC",))
