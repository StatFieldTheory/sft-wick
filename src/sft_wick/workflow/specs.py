"""Physical specification objects for the high-level ``System`` API.

Each object is a frozen dataclass that cleanly separates **what** the
user wants to express (linear operator, interaction vertices, noise
covariance) from **how** it is lowered to the raw package primitives
(``PropagatorModel``, ``Field``, ``Vertex``, ``Action``,
``PropagatorCache.precompute_C_table_*``).

Design choices:

- **Discriminated union via subclasses**: e.g. ``LinearOp`` has
  ``DiagonalA``, ``ExplicitR``, ... subclasses.  Each subclass knows
  how to lower itself to a raw callable via ``build_R_callable()``.
- **Escape hatches everywhere**: any object can be bypassed by
  supplying a raw callable (``ExplicitR``, ``GeneralKappa2``, etc.).
- **Everything immutable**: frozen dataclasses, safe to share between
  parallel workers and to hash for caching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


# =========================================================================
# Field
# =========================================================================


@dataclass(frozen=True)
class FieldSpec:
    """Specification of the physical field ``φ`` of the theory.

    The response field ``ψ`` is auto-generated under the hood and
    shares ``n_components``; users should never need to reference it
    explicitly.

    Args:
        name: Physical field name (e.g. ``"phi"``).  Used only for
            symbolic rendering and as a convention.
        n_components: Number of field components N.  Use ``1`` for a
            scalar theory.
    """

    name: str = "phi"
    n_components: int = 1

    @property
    def response_name(self) -> str:
        """The conventional response-field name (``"psi"``)."""
        return "psi"


# =========================================================================
# Module-level callable wrappers
# =========================================================================
#
# Concrete R / kappa^2 / sigma^2 / coupling callables are defined as
# module-level classes (not closures inside ``build_*`` methods) so that
# the resulting objects can be serialised with the standard protocol.
# This is required for:
#
# 1. ``Propagators.cache_path`` -- ``joblib.dump`` uses the standard
#    serialisation protocol and cannot persist local functions or
#    closures defined inside another function body.
# 2. ``loky`` (joblib's parallel backend) when distributing tasks across
#    worker processes -- cloudpickle is more forgiving, but module-level
#    classes work everywhere.
#
# Each class stores its parameters as instance attributes and implements
# ``__call__`` with the same signature the closure version had.


class _StaticIsoR:
    """Static (constant-gamma) iso-R: ``R(t1, t2) = exp(-gamma (t1 - t2))``
    for ``t1 >= t2`` else 0."""

    __slots__ = ("gamma",)

    def __init__(self, gamma: float):
        self.gamma = float(gamma)

    def __call__(self, t1: float, t2: float) -> float:
        if t1 < t2:
            return 0.0
        return float(np.exp(-self.gamma * (t1 - t2)))


class _StaticMatR:
    """Static diagonal R: ``R_{aa}(t1, t2) = exp(-gamma_a (t1 - t2))``."""

    __slots__ = ("gamma_arr", "_n")

    def __init__(self, gamma_arr: np.ndarray):
        self.gamma_arr = np.asarray(gamma_arr, dtype=float)
        self._n = self.gamma_arr.shape[0]

    def __call__(self, t1: float, t2: float) -> np.ndarray:
        if t1 < t2:
            return np.zeros((self._n, self._n))
        return np.diag(np.exp(-self.gamma_arr * (t1 - t2)))


class _TimeDepIsoR:
    """Time-dependent iso-R: ``R(t1, t2) = exp(-(Gamma(t1) - Gamma(t2)))``
    using a ``CubicSpline`` for the cumulative integral Gamma."""

    __slots__ = ("gamma_spline",)

    def __init__(self, gamma_spline):
        self.gamma_spline = gamma_spline

    def __call__(self, t1: float, t2: float) -> float:
        if t1 < t2:
            return 0.0
        return float(np.exp(-(self.gamma_spline(t1) - self.gamma_spline(t2))))


class _TimeDepMatR:
    """Time-dependent diagonal R from per-component cumulative-Gamma splines."""

    __slots__ = ("splines", "_n")

    def __init__(self, splines):
        self.splines = list(splines)
        self._n = len(self.splines)

    def __call__(self, t1: float, t2: float) -> np.ndarray:
        if t1 < t2:
            return np.zeros((self._n, self._n))
        diag = np.array([
            np.exp(-(s(t1) - s(t2))) for s in self.splines
        ])
        return np.diag(diag)


class _SeparableTranslationKappa2:
    """``kappa^2(n1, t1, n2, t2) = kappa_t(t1-t2) * kappa_x(|n1-n2|) * I_N``
    for ``SeparableTranslation``."""

    __slots__ = ("temporal", "spatial", "_n")

    def __init__(self, temporal: Callable, spatial: Callable, n_components: int):
        self.temporal = temporal
        self.spatial = spatial
        self._n = int(n_components)

    def __call__(self, n1, t1, n2, t2):
        # Note: collapsing via ``.sum()`` on n1/n2 is a 1-D legacy
        # convention preserved by the upstream wrapper; see the
        # d-dim spatial todo (T1) for the planned fix.
        dr = float(
            abs(np.asarray(n1).sum() - np.asarray(n2).sum())
        )
        return self.temporal(t1 - t2) * self.spatial(dr) * np.eye(self._n)


class _SeparableRotationKappa2:
    """``kappa^2(n1, t1, n2, t2) = kappa_t(t1-t2) * kappa_Omega(cos theta) * I_N``
    for ``SeparableRotation``."""

    __slots__ = ("temporal", "angular", "_n")

    def __init__(self, temporal: Callable, angular: Callable, n_components: int):
        self.temporal = temporal
        self.angular = angular
        self._n = int(n_components)

    def __call__(self, n1, t1, n2, t2):
        from sft_wick.evaluate import _rotation_cos
        cos_val = _rotation_cos(n1, n2)
        return self.temporal(t1 - t2) * self.angular(cos_val) * np.eye(self._n)


class _ConstantImpulseIso:
    """Time- and position-independent isotropic white-noise sigma^2."""

    __slots__ = ("amp", "_n")

    def __init__(self, amplitude: float, n_components: int):
        self.amp = float(amplitude)
        self._n = int(n_components)

    def __call__(self, n1, t, n2):  # noqa: ARG002
        return self.amp * np.eye(self._n)


class _ConstantImpulseMat:
    """Time- and position-independent matrix-valued white-noise sigma^2."""

    __slots__ = ("mat",)

    def __init__(self, matrix: np.ndarray):
        self.mat = np.asarray(matrix)

    def __call__(self, n1, t, n2):  # noqa: ARG002
        return self.mat


class _MSRWrappedCoupling:
    """Wraps a user-supplied callable coupling so it returns
    ``factor * np.asarray(bare(*args, **kwargs))``.

    Replaces the ``_wrapped`` closure that
    :attr:`NonLocalVertex.msr_coupling` used to return -- a closure
    cannot be persisted via the standard serialisation protocol, but
    a module-level class with explicit attributes can.
    """

    __slots__ = ("factor", "bare", "__wrapped__")

    def __init__(self, factor, bare: Callable):
        self.factor = factor
        self.bare = bare
        # Mirror the ``__wrapped__`` attribute the closure version
        # set, so any introspection (e.g. ``inspect.unwrap``) keeps
        # working.
        self.__wrapped__ = bare

    def __call__(self, *args, **kwargs):
        return self.factor * np.asarray(self.bare(*args, **kwargs))


# =========================================================================
# Linear operator (defines R-propagator)
# =========================================================================


@dataclass(frozen=True)
class LinearOp:
    """Base class — do not instantiate.  See subclasses."""

    def build_R_callable(self) -> Callable:
        """Return an ``R_time(t1, t2) -> float | (N, N)`` callable
        suitable for ``PropagatorModel.R_time``."""
        raise NotImplementedError

    @property
    def is_iso_R(self) -> bool:
        """True when R is scalar (no component structure)."""
        raise NotImplementedError


@dataclass(frozen=True)
class DiagonalA(LinearOp):
    """Diagonal linear operator ``A_{ab} = −γ_a δ_{ab}``.

    Two calling conventions for ``gamma``:

    **Static** (constant decay rates):
        ``gamma`` is a length-N sequence of floats.

        ``R_{aa}(t_1, t_2) = Θ(t_1 - t_2) · exp(-γ_a (t_1 - t_2))``

    **Time-dependent** (callable):
        ``gamma`` is a callable ``γ(t) -> np.ndarray(shape=(N,))``
        returning the per-component instantaneous decay rate.  In
        this case the wrapper pre-computes
        ``Γ_a(t) = ∫_0^t γ_a(τ) dτ`` on a time grid, caches it as a
        cubic spline, and evaluates R via::

            R_{aa}(t_1, t_2) = Θ(t_1 - t_2) · exp(-(Γ_a(t_1) − Γ_a(t_2)))

        The spline build cost is a one-time ``n_grid_cache`` calls to
        ``γ(t)`` + a cumulative trapezoidal integral.  For the
        full-matrix (non-diagonal) time-dependent case — which
        requires a time-ordered matrix exponential — use
        :class:`ExplicitR` with your own R callable.

    Args:
        gamma: length-N sequence OR a callable ``γ(t) -> array(N)``.
        t_max_cache: Upper bound of the Γ-spline grid, **only used
            in the callable case**.  Queries beyond this bound
            extrapolate the spline (may be inaccurate — set this
            ≥ your maximum ``lambda_f``).
        n_grid_cache: Number of grid points for the Γ-spline build.
    """

    gamma: Any
    t_max_cache: float = 100.0
    n_grid_cache: int = 200

    def build_R_callable(self) -> Callable:
        if callable(self.gamma):
            return self._build_time_dependent_R()
        return self._build_static_R()

    def _build_static_R(self) -> Callable:
        gamma_arr = np.asarray(self.gamma, dtype=float)
        if self._iso_from_array(gamma_arr):
            return _StaticIsoR(gamma=float(gamma_arr[0]))
        return _StaticMatR(gamma_arr=gamma_arr)

    def _build_time_dependent_R(self) -> Callable:
        """Pre-compute Γ_a(t) spline, return fast O(1)-per-query R."""
        from scipy.integrate import cumulative_trapezoid
        from scipy.interpolate import CubicSpline

        gamma_fn = self.gamma
        probe = np.atleast_1d(np.asarray(gamma_fn(0.0), dtype=float))
        N = probe.shape[0]

        t_grid = np.linspace(0.0, self.t_max_cache, self.n_grid_cache)
        gamma_vals = np.empty((self.n_grid_cache, N))
        for i, t in enumerate(t_grid):
            gamma_vals[i] = np.atleast_1d(
                np.asarray(gamma_fn(t), dtype=float)
            )
        # Γ_a(t) = ∫_0^t γ_a(τ) dτ  (per-component cumulative integral)
        Gamma_grid = cumulative_trapezoid(
            gamma_vals, t_grid, axis=0, initial=0.0,
        )
        splines = [
            CubicSpline(t_grid, Gamma_grid[:, a], extrapolate=True)
            for a in range(N)
        ]

        if self._iso_probe_time_dependent(gamma_fn):
            return _TimeDepIsoR(gamma_spline=splines[0])
        return _TimeDepMatR(splines=splines)

    @property
    def is_iso_R(self) -> bool:
        if callable(self.gamma):
            return self._iso_probe_time_dependent(self.gamma)
        return self._iso_from_array(np.asarray(self.gamma, dtype=float))

    @staticmethod
    def _iso_from_array(arr: np.ndarray) -> bool:
        arr = np.atleast_1d(arr)
        return bool(arr.shape[0] == 1 or np.allclose(arr, arr[0]))

    @staticmethod
    def _iso_probe_time_dependent(gamma_fn: Callable) -> bool:
        """Iso detection for callable γ: probe at two points (0 and
        a mid-range value).  Returns True iff all components agree
        at both points to within numerical tolerance.
        """
        a0 = np.atleast_1d(np.asarray(gamma_fn(0.0), dtype=float))
        a1 = np.atleast_1d(np.asarray(gamma_fn(1.0), dtype=float))
        if a0.shape[0] != a1.shape[0]:
            return False
        return bool(
            a0.shape[0] == 1
            or (np.allclose(a0, a0[0]) and np.allclose(a1, a1[0]))
        )


@dataclass(frozen=True)
class ExplicitR(LinearOp):
    """Escape hatch: user provides R directly, bypassing A.

    Args:
        R_time: ``(t1, t2) -> float | (N, N)`` callable.  Must enforce
            causality (return 0 when ``t1 < t2``).
        iso_R: True if the callable returns a scalar, False if matrix.
    """

    R_time: Callable
    iso_R: bool = True

    def build_R_callable(self) -> Callable:
        return self.R_time

    @property
    def is_iso_R(self) -> bool:
        return self.iso_R


# =========================================================================
# Temporal / spatial / angular kernel helpers
# =========================================================================


@dataclass(frozen=True)
class ExponentialTemporal:
    """``κ²_t(Δt) = λ · exp(−|Δt|/σ_t)`` (OU kernel)."""

    lam: float
    sigma_t: float

    def __call__(self, dt: float) -> float:
        return self.lam * float(np.exp(-abs(dt) / self.sigma_t))


@dataclass(frozen=True)
class GaussianTemporal:
    """``κ²_t(Δt) = λ · exp(−Δt² / (2 σ_t²))`` (Gaussian kernel)."""

    lam: float
    sigma_t: float

    def __call__(self, dt: float) -> float:
        return self.lam * float(np.exp(-(dt * dt) / (2.0 * self.sigma_t ** 2)))


@dataclass(frozen=True)
class ExponentialSpatial:
    """``κ²_x(|Δx|) = exp(−|Δx|/σ_x)`` (exponential spatial envelope)."""

    sigma_x: float

    def __call__(self, dr: float) -> float:
        return float(np.exp(-abs(dr) / self.sigma_x))


@dataclass(frozen=True)
class GaussianSpatial:
    """``κ²_x(|Δx|) = exp(−Δx² / (2 σ_x²))``."""

    sigma_x: float

    def __call__(self, dr: float) -> float:
        return float(np.exp(-(dr * dr) / (2.0 * self.sigma_x ** 2)))


@dataclass(frozen=True)
class LegendreAngular:
    """Angular kernel on the sphere: ``κ²_x(cos θ) = Σ_ℓ C_ℓ P_ℓ(cos θ)``.

    Args:
        coeffs: ``[C_0, C_1, …, C_L]`` — ℓ-th entry is the Legendre
            coefficient for order ℓ.
    """

    coeffs: Any  # sequence of floats

    def __call__(self, cos_theta: float) -> float:
        from numpy.polynomial.legendre import legval

        return float(legval(cos_theta, np.asarray(self.coeffs)))


@dataclass(frozen=True)
class CustomKernel:
    """Escape hatch for a user-supplied 1-D kernel callable."""

    fn: Callable

    def __call__(self, x: float) -> float:
        return float(self.fn(x))


# =========================================================================
# κ² (Gaussian two-point cumulant of the driving field)
# =========================================================================


@dataclass(frozen=True)
class Kappa2:
    """Base class — do not instantiate.  See subclasses."""

    def build_callable(self, n_components: int) -> Callable:
        """Lower to a ``kappa2(n1, t1, n2, t2) -> (N, N)`` callable."""
        raise NotImplementedError

    @property
    def homogeneity(self) -> str:
        """One of ``'translation' | 'rotation' | 'general'``."""
        raise NotImplementedError


@dataclass(frozen=True)
class SeparableTranslation(Kappa2):
    """Translation-invariant + separable:
    ``κ²(n1, t1, n2, t2) = κ²_t(t1 − t2) · κ²_x(|n1 − n2|) · I_N``.

    Args:
        temporal: callable accepting Δt, returning scalar.
        spatial: callable accepting |Δx|, returning scalar.
    """

    temporal: Callable
    spatial: Callable

    def build_callable(self, n_components: int) -> Callable:
        return _SeparableTranslationKappa2(
            temporal=self.temporal,
            spatial=self.spatial,
            n_components=n_components,
        )

    @property
    def homogeneity(self) -> str:
        return "translation"


@dataclass(frozen=True)
class SeparableRotation(Kappa2):
    """Rotation-invariant + separable:
    ``κ²(n1, t1, n2, t2) = κ²_t(t1 − t2) · κ²_Ω(cos θ) · I_N``
    where ``cos θ = n1·n2/(|n1||n2|)`` and ``n1, n2`` are direction
    vectors (typically on S²).

    Args:
        temporal: callable Δt → scalar.
        angular: callable cos θ ∈ [−1, 1] → scalar.
    """

    temporal: Callable
    angular: Callable

    def build_callable(self, n_components: int) -> Callable:
        return _SeparableRotationKappa2(
            temporal=self.temporal,
            angular=self.angular,
            n_components=n_components,
        )

    @property
    def homogeneity(self) -> str:
        return "rotation"


@dataclass(frozen=True)
class GeneralKappa2(Kappa2):
    """Escape hatch: user-supplied κ² callable without symmetry
    assumptions.  Use this when κ² is not translation- or
    rotation-invariant and you want the full 4-D ``(t1, t2, x1, x2)``
    spline build.

    Args:
        fn: callable ``(n1, t1, n2, t2) -> (N, N)``.
    """

    fn: Callable

    def build_callable(self, n_components: int) -> Callable:
        return self.fn

    @property
    def homogeneity(self) -> str:
        return "general"


# =========================================================================
# σ² (white-noise impulse in κ²)
# =========================================================================


@dataclass(frozen=True)
class Sigma2:
    """Base — white-noise ``δ(t1 − t2)·σ²(t; n1, n2)`` component."""

    def build_callable(self, n_components: int) -> Callable:
        """Lower to a ``sigma2(n1, t, n2) -> (N, N)`` callable."""
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantImpulse(Sigma2):
    """Time- and position-independent white-noise amplitude.

    Args:
        amplitude: scalar (isotropic) or ``(N, N)`` matrix.
    """

    amplitude: Any  # float or array

    def build_callable(self, n_components: int) -> Callable:
        amp = self.amplitude
        if np.ndim(amp) == 0:
            return _ConstantImpulseIso(
                amplitude=float(amp), n_components=n_components,
            )
        return _ConstantImpulseMat(matrix=np.asarray(amp))


@dataclass(frozen=True)
class CustomImpulse(Sigma2):
    """Escape hatch: arbitrary ``σ²(t; n1, n2) -> (N, N)`` callable."""

    fn: Callable

    def build_callable(self, n_components: int) -> Callable:
        return self.fn


# =========================================================================
# GaussianNoise (wraps κ² + optional σ²)
# =========================================================================


@dataclass(frozen=True)
class GaussianNoise:
    """Gaussian driving: ``W_G = ½ ∫ψ ⊗ κ² ⊗ ψ`` with an optional
    white-noise impulse part.

    Args:
        kappa2: the smooth (time-continuous) κ² cumulant.
        sigma2: optional white-noise ``δ(t-t')·σ²`` part; ``None`` by
            default.
    """

    kappa2: Kappa2
    sigma2: Sigma2 | None = None


# =========================================================================
# Interaction vertices
# =========================================================================


@dataclass(frozen=True)
class LocalVertex:
    """A **local** MSR interaction vertex from the deterministic
    nonlinearity ``F^(n)``: one ψ leg and (n − 1) φ legs at a single
    spacetime point.

    Args:
        name: symbolic coupling name (e.g. ``"F"``).  Must be unique
            per system (used as a dict key in ``coupling_values``).
        coupling: the **bare** ``F^(n)`` tensor — the coefficient as
            it appears in the deterministic equation of motion
            (``dφ_a/dt = … + F^(n)_{a b_1 … b_{n−1}} φ_{b_1} … φ_{b_{n−1}}
            + …``).  Shape ``(N,)*n``; the first axis is the ψ leg.
            The wrapper multiplies by the MSR factor ``-i``
            internally (so demo1's ``F_MSR = -1j * F_bare`` is
            automated).

    Notes:
        Use :attr:`msr_coupling` to retrieve the MSR-factor-applied
        tensor — that is what is forwarded to the raw
        ``compute_moment`` / ``DiagramTerm.evaluate_coupling`` layer.
    """

    name: str
    coupling: Any  # np.ndarray — bare F^(n)

    @property
    def msr_coupling(self) -> np.ndarray:
        """Bare F multiplied by the MSR factor ``-i``."""
        return (-1j) * np.asarray(self.coupling)


@dataclass(frozen=True)
class NonLocalVertex:
    """A **non-local** MSR vertex from a higher driving-field cumulant
    ``κ^(m)`` (m ≥ 3): ``m`` ψ legs at ``m`` distinct spacetime points.

    Args:
        name: symbolic coupling name (e.g. ``"K"``).
        order: ``m``, the number of ψ legs (= rank of κ^(m)).
        coupling: the **bare** ``κ^(m)`` tensor — either a numeric
            tensor of shape ``(N,)*m`` (when κ^(m) is
            spacetime-independent) or a callable with signature
            ``fn(n_list, t_list) -> np.ndarray(shape=(N,)*m)`` where
            ``n_list`` / ``t_list`` are length-m sequences of the
            spatial / time coordinates.  The wrapper multiplies by
            the MSR factor ``-(i^m) / m!`` internally (so demo2's
            ``K = (i/6) * κ^(3)`` is automated).

    Notes:
        The MSR factor values:

        =====  =========  =====================
         m      factor     simplified
        =====  =========  =====================
         1      −i         −i (mean drift)
         2      −i²/2!     +½ (Gaussian kernel)
         3      −i³/3!     +i/6 (demo2's K)
         4      −i⁴/4!     −1/24
        =====  =========  =====================

        Use :attr:`msr_coupling` to retrieve the MSR-factor-applied
        tensor or callable — that is what is forwarded to the raw
        coupling-values dict.
    """

    name: str
    order: int
    coupling: Any  # np.ndarray or callable — bare κ^(m)

    @property
    def msr_factor(self) -> complex:
        """The ``−(i^m) / m!`` prefactor applied to the bare
        ``κ^(m)``."""
        import math

        return -(1j ** self.order) / math.factorial(self.order)

    @property
    def msr_coupling(self) -> Any:
        """Bare κ^(m) multiplied by :attr:`msr_factor`.

        If ``coupling`` is a callable, returns a wrapped callable
        that applies the factor at evaluation time — preserving the
        original signature.
        """
        factor = self.msr_factor
        bare = self.coupling
        if callable(bare):
            return _MSRWrappedCoupling(factor=factor, bare=bare)
        return factor * np.asarray(bare)
