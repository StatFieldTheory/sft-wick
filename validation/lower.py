r"""Lowering of a :class:`~validation.modelspec.ModelSpec` onto the current
sft-wick release (the L1 API: ``System``, ``Expansion``, ``Propagators``).

This module, and ``lower_v042.py`` for the 0.4.2 release once it is added,
are the only modules of ``validation/`` that import sft_wick;
``validation/worker.py`` runs either in its own process.  A model becomes

* ``DiagonalA(gamma)`` for the response;
* ``LocalVertex`` ``h`` (rank 1), ``F`` and ``G`` for the drift;
* ``NonLocalVertex`` ``S{m}`` (static) and ``W{m}`` (``equal_time``) for the
  cumulants, and ``P{m}`` for each kept Poisson order, whose coupling is the
  raw Campbell cumulant (``equal_time`` for white pulses, a callable kinked
  where two leg times cross for exponential pulses);
* a ``MultiplicativeImpulse`` whose vertices are renamed ``D``, ``E``
  (``ψψφ``, ``ψψφφ``) and ``B``, ``L`` (the Stratonovich drift), so that no
  name clashes with the cubic drift ``G``; white noise beside it is folded
  into extra ``g0`` columns;
* the noise covariance for the record (the package probes it for the
  diagonal flags), and C from one of three sources: ``'oracle'``, the
  closed form written out here for every model; ``'auto'``, the package's
  built-in closed form, for the models it covers; ``'table'``, the package's
  quadrature tables.

The closed form of C integrates ``R κ² R`` exactly.  Every Gaussian part of
the noise has a covariance ``c e^{−p(s−u)}`` for ``s ≥ u`` and ``c
e^{−q(u−s)}`` for ``u > s`` (an OU block has ``p = q``; exponential Poisson
pulses have ``p = r_a``, ``q = r_b``) or is white, so C is a sum of the two
integrals of :func:`coloured_integral` and :func:`white_integral`.  The
coloured integral splits into triangles ``∫∫_{a ≤ b} e^{−xa − yb}``: a
difference of two exponential integrals where it does not cancel, and a
series in ``x`` where it would (rates that meet, or ``|x|`` small beside a
large ``y``), whose moments come from one incomplete gamma function and a
recurrence (:func:`_triangle`, :func:`_moments`).  Each exponential factor
is formed on its own, so a factor whose exponent passes 709.78 makes the
result inf or NaN even where C is finite; C is then above about 1e255.  An
empty interval (a time at ``t_min``) gives exactly 0.

:func:`package_moments` evaluates an observable and sums the package's
diagrams by their tag over ``model.channels()``, read from the vertex
multiset of each diagram.

Not lowered: response (``psi``) legs, and white Poisson pulses beside
multiplicative noise (both raise ``NotImplementedError``).  The package
refuses multiplicative noise with legs at different positions.
"""
from __future__ import annotations

import itertools
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import gamma, gammainc

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

import sft_wick as sw  # noqa: E402
from sft_wick.expressions import Product, Sum, Symbol  # noqa: E402
from sft_wick.workflow.specs import (ConstantImpulse,  # noqa: E402
                                     CustomImpulse, CustomKernel,
                                     GeneralKappa2, MultiplicativeImpulse)

from .modelspec import ModelSpec, Observable, as_array  # noqa: E402
from .stages import stage as _stage  # noqa: E402

__all__ = ["to_system", "closed_form_c", "propagators", "Lowered",
           "package_moments", "coloured_integral", "white_integral",
           "MULTIPLICATIVE_NAMES"]

#: ``(vertex_names, drift_names)`` given to the MultiplicativeImpulse.
MULTIPLICATIVE_NAMES = (("D", "E"), ("B", "L"))
C_SOURCES = ("oracle", "auto", "table")


# ---------------------------------------------------------------------------
# Closed-form integrals
# ---------------------------------------------------------------------------

def _e(k: np.ndarray, T: np.ndarray) -> np.ndarray:
    """``∫_0^T e^{−k t} dt``, accurate for every real ``k`` while ``−kT``
    stays below the overflow of ``exp`` (709.78)."""
    z = -np.asarray(k, dtype=float) * T
    small = z == 0.0
    safe = np.where(small, 1.0, z)
    return T * np.where(small, 1.0, np.expm1(safe) / safe)


def _two_exp(a: np.ndarray, b: np.ndarray, D: np.ndarray) -> np.ndarray:
    """``∫_0^D e^{−a(D−v) − b v} dv``."""
    return np.exp(-np.minimum(a, b) * D) * _e(np.abs(a - b), D)


#: ``|x T|`` below which :func:`_triangle` takes the Taylor series in ``x``.
TRIANGLE_SWITCH = 0.5
#: The series is taken as well where ``y > 0`` and ``|x| ≤ y/RATE_RATIO``:
#: there ``(E(y) − E(x + y))/x`` cancels at large ``yT``, and the terms of
#: the series are bounded by ``(|x|/y)^n``.
RATE_RATIO = 8.0
_SERIES_TERMS = 20
_SERIES_EPS = 1e-17


def _power_series(k: np.ndarray, z: np.ndarray) -> np.ndarray:
    """``M_k(z)`` for the orders ``k`` (shape ``(K, 1)``) from the power
    series in ``z``: for ``z ≤ 1``, where for ``z < 0`` every term is
    positive."""
    term = np.ones(z.shape)
    total = np.repeat(1.0 / (k + 1), z.shape[0], axis=1)
    if z.size == 0:
        return total
    for j in range(1, 60 + int(3.0 * max(0.0, -float(z.min())))):
        term = term * (-z) / j
        add = term[None, :] / (k + j + 1)
        total += add
        if np.all(np.abs(add) <= 1e-18 * np.abs(total)):
            break
    return total


def _moments_series(K: int, z: np.ndarray) -> np.ndarray:
    """``M_k(z) = ∫_0^1 s^k e^{−z s} ds`` for ``k = 1, …, K``, shape ``(K,
    n)``, from the power series in ``z``."""
    return _power_series(np.arange(1, K + 1, dtype=float)[:, None],
                         np.asarray(z, dtype=float))


def _moment_gamma(K: int, z: np.ndarray) -> np.ndarray:
    """``M_K(z)`` from the regularised incomplete gamma function, for
    ``z > 0``."""
    z = np.asarray(z, dtype=float)
    return gamma(K + 1) * gammainc(K + 1, z) / z ** (K + 1)


def _moments_upward(K: int, z: np.ndarray) -> np.ndarray:
    """``M_1, …, M_K`` upwards from ``M_0 = (1 − e^{−z})/z`` by ``z M_k =
    k M_{k−1} − e^{−z}``: a relative error stays bounded while ``k ≤ z``."""
    z = np.asarray(z, dtype=float)
    out = np.empty((K, z.shape[0]))
    ez = np.exp(-z)
    m = -np.expm1(-z) / z
    for k in range(1, K + 1):
        m = (k * m - ez) / z
        out[k - 1] = m
    return out


def _moments_downward(K: int, z: np.ndarray) -> np.ndarray:
    """``M_K, …, M_1`` downwards by ``M_{k−1} = (z M_k + e^{−z})/k`` from
    ``M_K`` (the series for ``z ≤ 1``, else the incomplete gamma function):
    for ``z ≥ 0`` a relative error shrinks at every step."""
    z = np.asarray(z, dtype=float)
    out = np.empty((K, z.shape[0]))
    ez = np.exp(-z)
    m = np.empty(z.shape)
    big = z > 1.0
    if np.any(big):
        m[big] = _moment_gamma(K, z[big])
    if np.any(~big):
        m[~big] = _power_series(np.array([[float(K)]]), z[~big])[0]
    out[K - 1] = m
    for k in range(K, 1, -1):
        m = (z * m + ez) / k
        out[k - 2] = m
    return out


def _moments(K: int, z: np.ndarray) -> np.ndarray:
    """``M_k(z)`` for ``k = 1, …, K``, shape ``(K, n)``: upwards where ``z ≥
    K`` (no gamma function), downwards for ``0 ≤ z < K``, and the series for
    ``z < 0``, where both recurrences cancel."""
    z = np.asarray(z, dtype=float)
    out = np.empty((K, z.shape[0]))
    for mask, method in ((z < 0.0, _moments_series),
                         (z >= K, _moments_upward),
                         ((z >= 0.0) & (z < K), _moments_downward)):
        if np.any(mask):
            out[:, mask] = method(K, z[mask])
    return out


def _series_terms(h: np.ndarray, ratio: np.ndarray) -> int:
    """Terms of the triangle series that reach :data:`_SERIES_EPS`: term
    ``n`` is at most ``|h|^n/(n+1)!`` of the sum where ``|h| < 1/2``, and
    at most ``ratio^n`` where ``|x| ≤ y/RATE_RATIO`` and ``yT ≥ 4``."""
    hmax = float(np.max(np.abs(h), initial=0.0))
    rmax = float(np.max(ratio, initial=0.0))
    bound_h, bound_r = 1.0, 1.0
    for n in range(1, _SERIES_TERMS + 1):
        bound_h *= hmax / (n + 1)
        bound_r *= rmax
        if max(bound_h, bound_r) <= _SERIES_EPS:
            return n
    return _SERIES_TERMS


def _triangle_far(x, y, T, s):
    """The triangle as ``(E(y) − E(s))/x``, ``s = x + y``."""
    return (_e(y, T) - _e(s, T)) / x


def _triangle_series(x, y, T, K: int = _SERIES_TERMS):
    """The triangle as ``T² Σ_{n<K} (−xT)^n/(n+1)! M_{n+1}(yT)``."""
    x, y, T = (np.asarray(a, dtype=float) for a in (x, y, T))
    h = x * T
    M = _moments(K, y * T)
    coeff = np.empty((K, h.shape[0]))
    coeff[0] = 1.0
    for n in range(1, K):
        coeff[n] = coeff[n - 1] * (-h) / (n + 1)
    return T * T * np.sum(coeff * M, axis=0)


def _triangle(x: np.ndarray, y: np.ndarray, T: np.ndarray,
              s: np.ndarray | None = None) -> np.ndarray:
    """``∫∫_{0 ≤ a ≤ b ≤ T} e^{−x a − y b}``; ``s = x + y`` when the caller
    holds it exactly (a rounded ``x + y`` is amplified by ``T`` in
    ``E(s)``).

    Where ``|xT| < TRIANGLE_SWITCH``, or ``y > 0`` and ``|x| ≤ y/RATE_RATIO``,
    it is :func:`_triangle_series`, with as many terms as the samples need;
    elsewhere :func:`_triangle_far`, whose difference then loses at most a
    factor 9.
    """
    x, y, T = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float),
                                  np.asarray(T, float))
    s = x + y if s is None else np.broadcast_to(np.asarray(s, float), x.shape)
    x, y, T, s = (a.ravel() for a in (x, y, T, s))
    out = np.empty(x.shape)
    h = x * T
    classic = np.abs(h) < TRIANGLE_SWITCH
    near = classic | ((y > 0) & (RATE_RATIO * np.abs(x) <= y))
    far = ~near
    if np.any(far):
        out[far] = _triangle_far(x[far], y[far], T[far], s[far])
    if np.any(near):
        # The h bound holds relative to the sum wherever |h| < 1/2; the ratio
        # bound only where yT ≥ 4, which every other near sample has.
        c = classic[near]
        ratio = np.where(c, 0.0, np.abs(x[near]) / np.where(c, 1.0, y[near]))
        K = _series_terms(np.where(c, h[near], 0.0), ratio)
        out[near] = _triangle_series(x[near], y[near], T[near], K)
    return out


def _coloured_ordered(al, be, p, q, T1, T2):
    """The coloured integral for ``T1 ≥ T2``.  Both triangles have ``x + y =
    α + β``, passed exactly."""
    D = T1 - T2
    late = _two_exp(al, p, D) * _e(be + p, T2)
    s = al + be
    square = (_triangle(al - p, be + p, T2, s)
              + _triangle(be - q, al + q, T2, s))
    return late + np.exp(-al * D) * square


def coloured_integral(alpha: float, beta: float, p: float, q: float,
                      T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    r"""``∫_0^{T1} ds ∫_0^{T2} du e^{−α(T1−s) − β(T2−u)} k(s − u)`` with
    ``k(Δ) = e^{−pΔ}`` for ``Δ ≥ 0`` and ``e^{qΔ}`` for ``Δ < 0``."""
    T1, T2 = np.broadcast_arrays(np.asarray(T1, float), np.asarray(T2, float))
    out = np.zeros(T1.shape)
    live = np.minimum(T1, T2) > 0.0
    first = live & (T1 >= T2)
    second = live & (T1 < T2)
    if np.any(first):
        out[first] = _coloured_ordered(alpha, beta, p, q, T1[first],
                                       T2[first])
    if np.any(second):
        out[second] = _coloured_ordered(beta, alpha, q, p, T2[second],
                                        T1[second])
    return out


def white_integral(alpha: float, beta: float, T1: np.ndarray,
                   T2: np.ndarray) -> np.ndarray:
    """``∫_0^{min(T1, T2)} e^{−α(T1−s) − β(T2−s)} ds``."""
    T1, T2 = np.broadcast_arrays(np.asarray(T1, float), np.asarray(T2, float))
    m = np.minimum(T1, T2)
    out = np.zeros(T1.shape)
    live = m > 0.0
    T1, T2, m = T1[live], T2[live], m[live]
    out[live] = (np.exp(-alpha * (T1 - m) - beta * (T2 - m))
                 * _e(alpha + beta, m))
    return out


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def _per_sample(x: Any, n: int, scalar_call: bool) -> np.ndarray:
    """A position argument as ``(n, d)``."""
    a = np.asarray(x, dtype=float)
    if scalar_call:
        return np.atleast_1d(a)[None, :]
    if a.ndim == 0:
        return np.full((n, 1), float(a))
    if a.ndim == 1:
        return a[:, None] if a.shape[0] == n else np.tile(a, (n, 1))
    return a


def _package_position(pos: tuple) -> Any:
    return pos[0] if len(pos) == 1 else np.asarray(pos, dtype=float)


# ---------------------------------------------------------------------------
# Noise callables (frozen dataclasses: a stable repr and picklable)
# ---------------------------------------------------------------------------

def _white_matrix(model: ModelSpec) -> np.ndarray:
    """Position-independent white noise: ``S`` plus ``g0 g0ᵀ``."""
    N = model.n_components
    S = np.zeros((N, N))
    if model.white is not None:
        S += as_array(model.white)
    if model.multiplicative is not None:
        g0 = as_array(model.multiplicative.g0)
        S += g0 @ g0.T
    return S


def c_matrix(model: ModelSpec, x1: np.ndarray, T1: np.ndarray,
             x2: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """``C_ab`` at ``n`` pairs ``(x1, t_min + T1)``, ``(x2, t_min + T2)``;
    ``x1``, ``x2`` of shape ``(n, d)``.  Returns ``(n, N, N)``."""
    N, g = model.n_components, as_array(model.gamma)
    C = np.zeros((T1.shape[0], N, N))
    seen: dict = {}

    def coloured(a: int, b: int, p: float, q: float) -> np.ndarray:
        key = (g[a], g[b], p, q)
        if key not in seen:
            seen[key] = coloured_integral(g[a], g[b], p, q, T1, T2)
        return seen[key]

    for blk in model.ou:
        K = blk.spatial.values(x1, x2)
        rate = 1.0 / blk.sigma_t
        for a, b in itertools.product(range(N), repeat=2):
            amp = blk.lam * blk.loading[a] * blk.loading[b]
            if amp != 0.0:
                C[:, a, b] += amp * K * coloured(a, b, rate, rate)
    S = _white_matrix(model)
    for a, b in itertools.product(range(N), repeat=2):
        if S[a, b] != 0.0:
            C[:, a, b] += S[a, b] * white_integral(g[a], g[b], T1, T2)
    p = model.poisson
    if p is not None:
        xs = np.stack([x1, x2])
        for a, b in itertools.product(range(N), repeat=2):
            amp = p.nu * p.amplitude[a] * p.amplitude[b] * p.overlap((a, b), xs)
            if p.pulse == "white":
                C[:, a, b] += amp * white_integral(g[a], g[b], T1, T2)
            else:
                ra, rb = 1.0 / p.tau[a], 1.0 / p.tau[b]
                C[:, a, b] += amp / (ra + rb) * coloured(a, b, ra, rb)
    return C


@dataclass(frozen=True)
class ClosedFormC:
    """The exact C of a model, ``(n1, t1, n2, t2) → (N, N)``, or ``(n, N,
    N)`` on arrays.  White parts kink C on its diagonal.  Before ``t_min`` no
    noise has accumulated and C is 0, as in the package."""

    model: ModelSpec
    has_diagonal_kink = True

    def __call__(self, n1, t1, n2, t2):
        scalar = np.ndim(t1) == 0 and np.ndim(t2) == 0
        t_min = self.model.t_min
        T1 = np.maximum(np.atleast_1d(np.asarray(t1, dtype=float)) - t_min, 0.0)
        T2 = np.maximum(np.atleast_1d(np.asarray(t2, dtype=float)) - t_min, 0.0)
        n = max(T1.shape[0], T2.shape[0])
        T1, T2 = np.broadcast_to(T1, (n,)), np.broadcast_to(T2, (n,))
        x1 = _per_sample(n1, n, scalar)
        x1 = np.broadcast_to(x1, (n, x1.shape[1]))
        x2 = np.broadcast_to(_per_sample(n2, n, scalar), x1.shape)
        C = c_matrix(self.model, x1, T1, x2, T2)
        return C[0] if scalar else C


def closed_form_c(model: ModelSpec) -> ClosedFormC:
    return ClosedFormC(model)


@dataclass(frozen=True)
class Kappa2Of:
    """``κ²_ab(n1, t1; n2, t2)`` of the coloured noise, ``(N, N)``."""

    model: ModelSpec

    def __call__(self, n1, t1, n2, t2):
        m, N = self.model, self.model.n_components
        x1 = np.atleast_1d(np.asarray(n1, dtype=float))[None, :]
        x2 = np.atleast_1d(np.asarray(n2, dtype=float))[None, :]
        out = np.zeros((N, N))
        for blk in m.ou:
            L = np.asarray(blk.loading)
            out += (blk.lam * np.exp(-abs(t1 - t2) / blk.sigma_t)
                    * blk.spatial.values(x1, x2)[0] * np.outer(L, L))
        p = m.poisson
        if p is not None and p.pulse == "exponential":
            xs = np.stack([x1, x2])
            for a, b in itertools.product(range(N), repeat=2):
                ra, rb = 1.0 / p.tau[a], 1.0 / p.tau[b]
                decay = ra * (t1 - t2) if t1 >= t2 else rb * (t2 - t1)
                out[a, b] += (p.nu * p.amplitude[a] * p.amplitude[b]
                              * p.overlap((a, b), xs)[0]
                              * np.exp(-decay) / (ra + rb))
        return out


@dataclass(frozen=True)
class Sigma2Of:
    """White-noise covariance rate at two positions, ``(N, N)``."""

    model: ModelSpec

    def __call__(self, n1, t, n2):
        out = _white_matrix(self.model)
        p = self.model.poisson
        if p is not None and p.pulse == "white":
            xs = np.stack([np.atleast_1d(np.asarray(n1, float))[None, :],
                           np.atleast_1d(np.asarray(n2, float))[None, :]])
            N = self.model.n_components
            for a, b in itertools.product(range(N), repeat=2):
                out[a, b] += (p.nu * p.amplitude[a] * p.amplitude[b]
                              * p.overlap((a, b), xs)[0])
        return out


@dataclass(frozen=True)
class PoissonCumulant:
    """The raw Campbell cumulant ``κ^(m)`` at the vertex legs (vectorised
    contract ``(m, n[, d]), (m, n) → (n,) + (N,)*m``).  For exponential
    pulses it depends on the smallest leg time, so it is kinked where two
    leg times cross."""

    model: ModelSpec
    m: int
    has_coincident_time_kinks = True

    def __call__(self, n_2d, t_2d):
        p, N = self.model.poisson, self.model.n_components
        xs = np.asarray(n_2d, dtype=float)
        ts = np.asarray(t_2d, dtype=float)
        if xs.ndim == 2:
            xs = xs[:, :, None]
        n = ts.shape[1]
        if p.pulse == "exponential":
            T = ts.min(axis=0)
        out = np.empty((n,) + (N,) * self.m)
        for comps in itertools.product(range(N), repeat=self.m):
            val = (p.nu * np.prod([p.amplitude[a] for a in comps])
                   * p.overlap(comps, xs))
            if p.pulse == "exponential":
                r = np.array([1.0 / p.tau[a] for a in comps])[:, None]
                val = val * np.exp(-(r * (ts - T)).sum(axis=0)) / r.sum()
            out[(slice(None),) + comps] = val
        return out


@dataclass(frozen=True)
class _Unit:
    """``κ_x(r) = 1`` (a spatially constant kernel)."""

    def __call__(self, r) -> float:
        return 1.0


@dataclass(frozen=True)
class _DampedCosine:
    ell: float
    k: float

    def __call__(self, r) -> float:
        r = abs(float(r))
        return float(np.exp(-r / self.ell) * np.cos(self.k * r))


# ---------------------------------------------------------------------------
# The System
# ---------------------------------------------------------------------------

def _structured_kappa2(model: ModelSpec):
    """``SeparableTranslation``/``SeparableRotation`` when the coloured
    noise is ``δ_ab λ e^{−|Δt|/σ} K``, else ``None``."""
    N = model.n_components
    if len(model.ou) != N or model.poisson is not None:
        return None
    first = model.ou[0]
    for a, blk in enumerate(model.ou):
        if (blk.lam, blk.sigma_t, blk.spatial) != (first.lam, first.sigma_t,
                                                   first.spatial):
            return None
        if tuple(blk.loading) != tuple(np.eye(N)[a]):
            return None
    temporal = sw.ExponentialTemporal(lam=first.lam, sigma_t=first.sigma_t)
    kind, params = first.spatial.kind, first.spatial.params
    if kind == "legendre":
        return sw.SeparableRotation(temporal=temporal,
                                    angular=sw.LegendreAngular(
                                        coeffs=list(params)))
    spatial = {"exponential": lambda: sw.ExponentialSpatial(params[0]),
               "gaussian": lambda: sw.GaussianSpatial(params[0]),
               "damped_cosine": lambda: CustomKernel(
                   fn=_DampedCosine(*params)),
               "constant": lambda: CustomKernel(fn=_Unit())}.get(kind)
    if spatial is None:
        return None
    return sw.SeparableTranslation(temporal=temporal, spatial=spatial())


def _noise(model: ModelSpec, structured: bool) -> sw.GaussianNoise:
    kappa2 = _structured_kappa2(model) if structured else None
    if kappa2 is None:
        coloured = model.ou or (model.poisson is not None
                                and model.poisson.pulse == "exponential")
        kappa2 = (GeneralKappa2(fn=Kappa2Of(model)) if coloured else
                  sw.SeparableTranslation(
                      temporal=sw.ExponentialTemporal(lam=0.0, sigma_t=1.0),
                      spatial=sw.GaussianSpatial(sigma_x=1.0)))
    mult, p = model.multiplicative, model.poisson
    white_poisson = p is not None and p.pulse == "white"
    if mult is not None:
        if white_poisson:
            raise NotImplementedError(
                "white Poisson pulses beside multiplicative noise do not fit "
                "the one sigma2 of a System")
        g0, g1 = as_array(mult.g0), as_array(mult.g1)
        if model.white is not None:
            vals, vecs = np.linalg.eigh(as_array(model.white))
            keep = vals > 1e-14 * max(1.0, float(vals.max()))
            extra = vecs[:, keep] * np.sqrt(vals[keep])
            g0 = np.concatenate([g0, extra], axis=1)
            g1 = np.concatenate([g1, np.zeros((g1.shape[0], extra.shape[1],
                                               g1.shape[2]))], axis=1)
        sigma2 = MultiplicativeImpulse(
            g0=g0, g1=g1, interpretation=mult.interpretation,
            vertex_names=MULTIPLICATIVE_NAMES[0],
            drift_names=MULTIPLICATIVE_NAMES[1])
    elif white_poisson:
        sigma2 = CustomImpulse(fn=Sigma2Of(model))
    elif model.white is not None:
        sigma2 = ConstantImpulse(as_array(model.white))
    else:
        sigma2 = None
    return sw.GaussianNoise(kappa2=kappa2, sigma2=sigma2)


def to_system(model: ModelSpec, structured: bool = False) -> sw.System:
    """The ``System`` of ``model``.  ``structured`` declares the coloured
    noise with the package's separable specs where it can, which the
    built-in closed form and the tables need."""
    vertices = []
    for name, value in (("h", model.source), ("F", model.quadratic),
                        ("G", model.cubic)):
        if value is not None:
            vertices.append(sw.LocalVertex(name, coupling=as_array(value)))
    nonlocal_vertices = [
        sw.NonLocalVertex(c.channel, order=c.order,
                          coupling=as_array(c.tensor),
                          equal_time=(c.time == "white"))
        for c in model.cumulants]
    p = model.poisson
    if p is not None:
        nonlocal_vertices += [
            sw.NonLocalVertex(f"P{m}", order=m, coupling=PoissonCumulant(
                model, m), coupling_vectorized=True,
                equal_time=(p.pulse == "white"))
            for m in p.orders]
    return sw.System(field=sw.FieldSpec("phi", model.n_components),
                     linear=sw.DiagonalA(gamma=list(model.gamma)),
                     vertices=vertices, nonlocal_vertices=nonlocal_vertices,
                     noise=_noise(model, structured), t_min=model.t_min)


def diagonal_c(model: ModelSpec) -> bool:
    """Whether C has no off-diagonal entry, so ``diag_C=True`` is allowed."""
    if model.poisson is not None and model.n_components > 1:
        return False
    if np.any(np.abs(np.triu(_white_matrix(model), 1)) > 0.0):
        return False
    return all(np.count_nonzero(blk.loading) <= 1 for blk in model.ou)


def propagators(system: sw.System, model: ModelSpec, t_max: float,
                c_source: str = "oracle", diag_C: bool = False,
                n_grid_t: int = 41):
    """The propagators of ``system`` with C from ``c_source``."""
    if c_source == "oracle":
        return system.propagators(
            t_max=t_max, c_closed_form=closed_form_c(model),
            c_closed_form_only=True, c_closed_form_vectorized=True,
            diag_C=diag_C, progress=False)
    if c_source == "auto":
        return system.propagators(
            t_max=t_max, c_closed_form="auto", c_closed_form_only=True,
            c_closed_form_vectorized=True, diag_C=diag_C, progress=False)
    if c_source == "table":
        return system.propagators(
            t_max=t_max, n_grid_t=n_grid_t, c_closed_form=None,
            c_method="gauss_legendre", diag_C=diag_C, progress=False)
    raise ValueError(f"c_source must be one of {C_SOURCES}, got {c_source!r}")


# ---------------------------------------------------------------------------
# Evaluation by tag
# ---------------------------------------------------------------------------

def _vertex_counts(expr) -> dict:
    """The vertex multiset of a diagram: coupling symbols of one term of its
    coupling sum (every term has the same vertices).  A callable coupling
    appears once per leg order as ``name@k``."""
    counts: dict = defaultdict(int)

    def walk(e):
        if isinstance(e, Symbol):
            counts[e.name.split("@")[0]] += 1
        elif isinstance(e, Product):
            for f in e.factors:
                walk(f)
        elif isinstance(e, Sum):
            walk(e.terms[0])

    walk(expr)
    return counts


def channel_weights(model: ModelSpec, system: sw.System) -> dict:
    """Vertex name → its tag over ``model.channels()``."""
    channels = model.channels()

    def unit(name: str, power: int = 1) -> tuple:
        return tuple(power if c == name else 0 for c in channels)

    weights = {name: unit(name) for name in channels if name != "g"}
    for v in system.multiplicative_vertices:
        weights[v.name] = unit("g", v.g1_power)
    return weights


def _observable_strings(obs: Observable, n_components: int) -> tuple:
    """``phi_a(x)``, …; a one-component field is scalar and takes no
    component index, ``phi(x)``."""
    if n_components == 1:
        return tuple(f"{leg.op}({leg.label})" for leg in obs.legs)
    letters = "abcdefghijklmnopqrstuvwxyz"
    return tuple(f"{leg.op}_{letters[k]}({leg.label})"
                 for k, leg in enumerate(obs.legs))


def _check_times(model: ModelSpec, obs: Observable) -> None:
    """The oracle's refusals: no time before ``t_min``."""
    if obs.t_final < model.t_min:
        raise ValueError("t_final is before t_min")
    for leg in obs.legs:
        if obs.time(leg.label) < model.t_min:
            raise ValueError(f"time of {leg.label!r} is before t_min")


class Lowered:
    """A model and an observable on the package: the expansion is built
    once and evaluated by any integrator.

    Args:
        model, obs: the pair to evaluate.
        max_weight: largest tag weight compared; every order up to it is
            expanded (a vertex carries weight ≥ 1).
        flags: ``System.expand`` flags; ``diag_C`` defaults to
            :func:`diagonal_c`, ``iso_C`` to False.
        c_source: ``'oracle'``, ``'auto'`` or ``'table'``.
    """

    def __init__(self, model: ModelSpec, obs: Observable, max_weight: int,
                 flags: dict | None = None, c_source: str = "oracle",
                 n_grid_t: int = 41):
        with _stage("lower"):
            if any(leg.op != "phi" for leg in obs.legs):
                raise NotImplementedError("response legs are not compared yet")
            _check_times(model, obs)
            self.model, self.obs = model, obs
            self.max_weight = int(max_weight)
            self.flags = dict(diag_C=diagonal_c(model), iso_C=False)
            self.flags.update(flags or {})
            self.system = to_system(model, structured=(c_source != "oracle"))
        with _stage("expand"):
            self.expansion = self.system.expand(
                _observable_strings(obs, model.n_components),
                orders=list(range(self.max_weight + 1)), target=obs.target,
                progress=False, **self.flags)
        with _stage("propagators"):
            self.props = propagators(self.system, model, obs.t_final + 0.25,
                                     c_source, self.flags["diag_C"], n_grid_t)
        self.weights = channel_weights(model, self.system)
        self.c_source = c_source

    def tag_of(self, order: int, idx: int) -> tuple:
        dt = self.expansion.diagrams(order)[idx]
        tag = [0] * len(self.model.channels())
        counts = _vertex_counts(dt.coupling_sum)
        if sum(counts.values()) != order:
            raise RuntimeError(f"diagram {order}/{idx} has vertices {counts}")
        for name, k in counts.items():
            for c, w in enumerate(self.weights[name]):
                tag[c] += k * w
        return tuple(tag)

    def rows(self, method: str, orders=None, **settings) -> list:
        """``(tag, value, error)`` of every diagram of ``orders`` (default
        every order up to ``max_weight``) whose tag has weight ``≤
        max_weight``.

        A tag of weight ``k`` is complete in the diagrams of order ``k`` when
        every vertex carries weight 1.  A multiplicative vertex carries the
        power of ``g1`` (``E`` and ``L`` carry 2), so one tag collects
        diagrams of several orders, and ``orders`` must then be every order.
        """
        obs = self.obs
        over = set(obs.integrate_over) or None
        every = list(range(self.max_weight + 1))
        orders = every if orders is None else sorted(set(orders))
        if orders != every and "g" in self.model.channels():
            raise ValueError("a subset of orders gives incomplete tags of the "
                             "g channel; evaluate every order")
        with _stage("evaluate"):
            res = self.expansion.evaluate(
                self.props,
                positions={lab: _package_position(p)
                           for lab, p in obs.positions},
                t_final=obs.t_final,
                component_pair=tuple(leg.comp for leg in obs.legs),
                orders=orders,
                external_times=dict(obs.times) or None, integrate_over=over,
                method=method, **settings)
        out = []
        for row in res.per_diagram:
            tag = self.tag_of(row["order"], row["diagram_idx"])
            if sum(tag) <= self.max_weight:
                out.append((tag, float(np.real(row["value"])),
                            float(row.get("error") or 0.0)))
        return out

    def evaluate(self, method: str, orders=None, **settings) -> dict:
        """``{tag: (value, error)}``: :meth:`rows` summed by tag, the errors
        in quadrature."""
        value: dict = defaultdict(float)
        var: dict = defaultdict(float)
        for tag, v, e in self.rows(method, orders, **settings):
            value[tag] += v
            var[tag] += e ** 2
        return {t: (value[t], float(np.sqrt(var[t]))) for t in value}


def package_moments(model: ModelSpec, obs: Observable, max_weight: int,
                    method: str = "gauss_legendre", flags: dict | None = None,
                    c_source: str = "oracle", **settings) -> dict:
    """``{tag: (value, error)}``: the package's diagrams summed by tag."""
    return Lowered(model, obs, max_weight, flags, c_source).evaluate(
        method, **settings)
