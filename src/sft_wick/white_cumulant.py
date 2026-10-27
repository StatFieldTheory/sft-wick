"""Analytical response contraction of a declared static white cumulant.

The tensor is the bare amplitude of an equal-time driving cumulant; no MSR
phase is included. This adapter does not infer white noise from a callback.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Integral
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .workflow.specs import NonLocalVertex
    from .workflow.system import System

__all__ = ["OUWhiteCumulant"]

_FLOAT_MAX = sys.float_info.max
_FLOAT_TINY = sys.float_info.min
_NEAR_MAX = _FLOAT_MAX * (1 - 1e-11)
_LOG_NEAR_MAX = math.log(_FLOAT_MAX) - 1e-11


def _real_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    original = np.asarray(value)
    if original.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(original, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite float64 values")
    return result


def _integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _snapshot(array: np.ndarray) -> np.ndarray:
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _positive_difference(
    high: NDArray[np.float64], low: float | NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A nonnegative difference and its log, including opposite-sign overflow."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        difference = high - low
        logarithm = np.log(difference)
    overflow = np.isposinf(difference)
    if np.any(overflow):
        # With finite high >= low, subtraction only overflows across zero.
        low_array = np.broadcast_to(np.asarray(low), high.shape)
        logarithm[overflow] = np.logaddexp(
            np.log(np.abs(high[overflow])), np.log(np.abs(low_array[overflow]))
        )
    return difference, logarithm


def _restore(
    amplitude: np.ndarray,
    gamma: np.ndarray,
    t_min: float,
    chunk_size: int,
    max_output_bytes: int,
) -> OUWhiteCumulant:
    return OUWhiteCumulant(
        amplitude,
        gamma,
        t_min=t_min,
        chunk_size=chunk_size,
        max_output_bytes=max_output_bytes,
    )


def _system_parameters(system: System) -> tuple[np.ndarray, float]:
    """Inspect the exact typed drift lowering, rather than probe a callable."""
    from .workflow.specs import DiagonalA
    from .workflow.system import System

    if type(system) is not System:
        raise ValueError("system must be an unmodified System specification")
    linear = system.linear
    if system.explicit_R is not None or type(linear) is not DiagonalA:
        raise ValueError("OUWhiteCumulant requires a constant DiagonalA system")
    if callable(linear.gamma):
        raise ValueError("OUWhiteCumulant requires constant, non-callable rates")
    rates = _real_array(linear.gamma, "system gamma")
    n = system.n_components
    if rates.ndim != 1 or rates.shape not in ((1,), (n,)):
        raise ValueError("system gamma must have length 1 or n_components")
    if np.any(rates < 0):
        raise ValueError("system gamma must be nonnegative")
    # DiagonalA collapses allclose rates to component zero's scalar R. The
    # adapter must integrate that actual lowered response, not the raw list.
    effective = np.full(n, rates[0]) if linear.is_iso_R else rates
    lower = _real_array(system.t_min, "system t_min")
    if lower.ndim != 0:
        raise ValueError("system t_min must be a scalar")
    return effective, float(lower)


@dataclass(frozen=True, eq=False, slots=True, init=False)
class OUWhiteCumulant:
    r"""Bare white-noise cumulant contracted with constant diagonal OU R.

    For a static bare tensor ``A[a_1,...,a_m]`` and responses
    ``R_ab(t,s)=delta_ab*Theta(t-s)*exp(-gamma_a*(t-s))``, return

    ``A * exp(-sum(gamma_ai*(t_i-T))) * Delta * psi(G*Delta)``,

    where ``T=min(t_i)``, ``Delta=T-t_min``, ``G=sum(gamma_ai)``, and
    ``psi(x)=-expm1(-x)/x`` with ``psi(0)=1``. The answer is zero when
    ``T <= t_min``. Thus one common time integral of a white cumulant is
    removed. Spatial coordinates pass through unchanged because the
    amplitude is spatially constant and the response local in space.

    Args:
        amplitude: finite real/complex static tensor of shape ``(N,)*m``,
            with ``m>=2`` and ``N>=1``. The order must leave room for a sample
            axis within NumPy's dimension limit. No symmetry or MSR phase
            is assumed.
        gamma: finite nonnegative scalar (broadcast to N) or shape ``(N,)``.
        t_min: finite common lower time bound.
        chunk_size: maximum samples handled in one work block (default 128).
        max_output_bytes: hard limit on one returned tensor/batch (default
            64 MiB). Work arrays are ``O(chunk_size*N**m)``; this output
            limit is not a limit on total process/native memory. Validation
            additionally snapshots the ``m*S`` supplied times.

    Call with ``t_list.shape==(m,)`` for ``(N,)*m`` or ``(m,S)`` for
    ``(S,)+(N,)*m``. ``n_list`` must have m legs; their values are ignored.
    Finite represented results survive intermediate rate/time overflow and
    temporal underflow by logarithmic scaling. Detected final component
    overflow raises ``OverflowError``; genuine float64 underflow is allowed.
    These are floating-point evaluations, not interval error certificates.

    :meth:`as_vertex` validates an explicit System's actual constant diagonal
    response and time bound. It sets the existing R-absorbed and vectorized
    flags and leaves the MSR phase to NonLocalVertex. General matrix/time-
    dependent responses, spatial kernels and automatic callback recognition
    are outside this adapter. The ordinary NonLocalVertex API is unchanged.
    Complex amplitudes are supported here; the workflow's existing requirement
    that a physical diagram becomes real after its MSR rotation still applies.
    """

    amplitude: np.ndarray
    gamma: NDArray[np.float64]
    t_min: float
    chunk_size: int
    max_output_bytes: int
    _rate_sum: NDArray[np.float64]
    _log_rate_sum: NDArray[np.float64]
    _log_amplitude_real: NDArray[np.float64]
    _log_amplitude_imag: NDArray[np.float64] | None
    _finite_rate_sum: bool
    _zero_rate_slots: bool
    _zero_amplitude: bool
    _uniform_gamma: bool
    _ordinary_time_bound: float
    _ordinary_origin_in_bounds: bool

    has_coincident_time_kinks = True

    def __init__(
        self,
        amplitude: ArrayLike,
        gamma: ArrayLike,
        *,
        t_min: float = 0.0,
        chunk_size: int = 128,
        max_output_bytes: int = 64 * 1024**2,
    ) -> None:
        chunk = _integer(chunk_size, "chunk_size")
        budget = _integer(max_output_bytes, "max_output_bytes")
        original = np.asarray(amplitude)
        if original.dtype.kind not in "iufc":
            raise ValueError("amplitude must contain real or complex numeric values")
        dtype = np.complex128 if np.iscomplexobj(original) else np.float64
        if original.size * np.dtype(dtype).itemsize > budget:
            raise MemoryError("amplitude tensor exceeds max_output_bytes")
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.array(original, dtype=dtype, copy=True)
        if array.ndim < 2 or any(size != array.shape[0] for size in array.shape):
            raise ValueError("amplitude must have shape (N,)*m with m >= 2")
        try:
            np.empty(1).reshape((1,) * (array.ndim + 1))
        except ValueError as error:
            raise ValueError(
                "amplitude order must leave room for NumPy's batch dimension"
            ) from error
        if array.shape[0] < 1 or not np.all(np.isfinite(array)):
            raise ValueError("amplitude must be nonempty and contain finite values")
        rate = _real_array(gamma, "gamma")
        if rate.ndim == 0:
            rate = np.full(array.shape[0], float(rate))
        if rate.shape != (array.shape[0],) or np.any(rate < 0):
            raise ValueError("gamma must be nonnegative with shape (N,) or scalar")
        lower = _real_array(t_min, "t_min")
        if lower.ndim != 0:
            raise ValueError("t_min must be a finite scalar")
        rate_sum = np.zeros(array.shape)
        log_sum = np.full(array.shape, -np.inf)
        with np.errstate(divide="ignore"):
            log_rate = np.log(rate)
        for leg in range(array.ndim):
            shape = [1] * array.ndim
            shape[leg] = rate.size
            with np.errstate(over="ignore"):
                rate_sum += rate.reshape(shape)
            with np.errstate(under="ignore"):
                log_sum = np.logaddexp(log_sum, log_rate.reshape(shape))
        with np.errstate(divide="ignore"):
            log_real = np.log(np.abs(array.real))
            log_imag = np.log(np.abs(array.imag)) if np.iscomplexobj(array) else None
        for name, value in (
            ("amplitude", array),
            ("gamma", rate),
            ("_rate_sum", rate_sum),
            ("_log_rate_sum", log_sum),
            ("_log_amplitude_real", log_real),
        ):
            object.__setattr__(self, name, _snapshot(value))
        object.__setattr__(
            self,
            "_log_amplitude_imag",
            None if log_imag is None else _snapshot(log_imag),
        )
        object.__setattr__(self, "t_min", float(lower))
        object.__setattr__(self, "chunk_size", chunk)
        object.__setattr__(self, "max_output_bytes", budget)
        object.__setattr__(
            self, "_finite_rate_sum", bool(np.all(np.isfinite(rate_sum)))
        )
        object.__setattr__(self, "_zero_rate_slots", bool(np.any(rate_sum == 0)))
        object.__setattr__(self, "_zero_amplitude", not bool(np.any(array)))
        # Equal values alone would admit mixed signed zeros. Exact bit equality
        # certifies that every component follows the same ordered arithmetic.
        rate_bits = rate.view(np.uint64)
        object.__setattr__(
            self,
            "_uniform_gamma",
            rate.size > 1 and bool(np.all(rate_bits == rate_bits[0])),
        )
        maximum_rate = float(rate.max())
        bound = (
            min(_FLOAT_MAX / 4, (_FLOAT_MAX / (4 * array.ndim)) / maximum_rate)
            if maximum_rate
            else _FLOAT_MAX / 4
        )
        object.__setattr__(self, "_ordinary_time_bound", bound)
        object.__setattr__(
            self, "_ordinary_origin_in_bounds", abs(float(lower)) <= bound
        )

    @property
    def order(self) -> int:
        return self.amplitude.ndim

    @property
    def n_components(self) -> int:
        return self.amplitude.shape[0]

    def __copy__(self) -> OUWhiteCumulant:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> OUWhiteCumulant:
        memo[id(self)] = self
        return self

    def __reduce__(self):
        return _restore, (
            self.amplitude,
            self.gamma,
            self.t_min,
            self.chunk_size,
            self.max_output_bytes,
        )

    @classmethod
    def from_system(
        cls,
        amplitude: ArrayLike,
        system: System,
        *,
        chunk_size: int = 128,
        max_output_bytes: int = 64 * 1024**2,
    ) -> OUWhiteCumulant:
        """Snapshot the constant diagonal R actually lowered by a typed System."""
        rates, lower = _system_parameters(system)
        return cls(
            amplitude,
            rates,
            t_min=lower,
            chunk_size=chunk_size,
            max_output_bytes=max_output_bytes,
        )

    def validate_system(self, system: System) -> None:
        """Refuse a different component count, lowered R, or lower time bound."""
        rates, lower = _system_parameters(system)
        if not np.array_equal(rates, self.gamma):
            raise ValueError(
                "OUWhiteCumulant gamma differs from the system's lowered R"
            )
        if lower != self.t_min:
            raise ValueError("OUWhiteCumulant t_min differs from the system")

    def as_vertex(self, name: str, *, system: System) -> NonLocalVertex:
        """Build an explicit guarded bare vertex; the wrapper applies its phase."""
        from .workflow.specs import NonLocalVertex

        self.validate_system(system)
        return NonLocalVertex(
            name,
            order=self.order,
            coupling=self,
            coupling_vectorized=True,
            already_R_contracted=True,
        )

    def _integral(
        self, delta: NDArray[np.float64], log_delta: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        shape = (-1,) + (1,) * self.order
        delta = delta.reshape(shape)
        log_delta = log_delta.reshape(shape)
        log_x = self._log_rate_sum + log_delta
        # log(psi(x)) has a cancellation-free series around the removable pole.
        with np.errstate(under="ignore", over="ignore", invalid="ignore"):
            x_log = np.exp(np.minimum(log_x, math.log(745.0)))
            log_psi = -x_log / 2 + x_log**2 / 24 - x_log**4 / 2880
            middle = (log_x >= math.log(0.01)) & (log_x <= math.log(50.0))
            log_psi[middle] = np.log(-np.expm1(-x_log[middle])) - log_x[middle]
            log_integral = log_delta + log_psi
            large = log_x > math.log(50.0)
            log_integral[large] = np.broadcast_to(-self._log_rate_sum, log_x.shape)[
                large
            ]
            x = self._rate_sum * delta
            psi = np.ones(x.shape)
            small = (x > 0) & (x <= 1)
            psi[small] = -np.expm1(-x[small]) / x[small]
            integral = np.broadcast_to(delta, x.shape).copy() * psi
            saturated = x > 1
            rates = np.broadcast_to(self._rate_sum, x.shape)
            integral[saturated] = -np.expm1(-x[saturated]) / rates[saturated]
            exceptional = ~np.isfinite(delta) | ~np.isfinite(self._rate_sum)
            integral = np.where(exceptional, np.exp(log_integral), integral)
        return integral, log_integral

    @staticmethod
    def _scale_component(
        amplitude: NDArray[np.float64],
        log_amplitude: NDArray[np.float64],
        factor: NDArray[np.float64],
        log_factor: NDArray[np.float64],
        normal_intermediates: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = amplitude * factor
        direct = (factor >= _FLOAT_TINY) & np.isfinite(factor) & normal_intermediates
        if np.any(direct & ~np.isfinite(result)):
            raise OverflowError(
                "OU white-cumulant result exceeds float64 range; rescale"
            )
        fallback = (~direct) & np.broadcast_to(amplitude != 0, factor.shape)
        if np.any(fallback):
            combined = log_amplitude + log_factor
            with np.errstate(over="ignore", under="ignore"):
                values = np.exp(combined[fallback])
            if np.any(~np.isfinite(values)):
                raise OverflowError(
                    "OU white-cumulant result exceeds float64 range; rescale"
                )
            signs = np.broadcast_to(np.sign(amplitude), factor.shape)
            result[fallback] = signs[fallback] * values
        # Zero amplitudes remain zero even when the positive time factor overflows.
        result[np.broadcast_to(amplitude == 0, factor.shape)] = 0.0
        return result

    def _restore_zero_rate_slots(
        self,
        values: np.ndarray,
        outer_minimum: NDArray[np.float64],
        exceptional: NDArray[np.bool_],
    ) -> None:
        """Evaluate the zero-rate overflowing width as an exact binary product.

        No exponential remains on a zero-rate slot. Overflowing Delta or a
        final component near the range boundary needs this scalar fallback.
        Rational work is bounded by the supplied binary floats and output size.
        """
        for entry in zip(*np.nonzero(exceptional)):
            sample, tensor_slot = int(entry[0]), entry[1:]
            width = Fraction(float(outer_minimum[sample])) - Fraction(self.t_min)
            amplitude = self.amplitude[tensor_slot]
            real_exact = width * Fraction(float(amplitude.real))
            imaginary_exact = width * Fraction(float(amplitude.imag))
            maximum = Fraction(_FLOAT_MAX)
            if abs(real_exact) > maximum or abs(imaginary_exact) > maximum:
                raise OverflowError(
                    "OU white-cumulant result exceeds float64 range; rescale"
                )
            try:
                real = float(real_exact)
                imaginary = float(imaginary_exact)
            except OverflowError as error:
                raise OverflowError(
                    "OU white-cumulant result exceeds float64 range; rescale"
                ) from error
            values[entry] = (
                complex(real, imaginary) if np.iscomplexobj(values) else real
            )

    def _ordinary_block(
        self, outer: NDArray[np.float64], lower: NDArray[np.float64]
    ) -> np.ndarray | None:
        """Use ordinary ufuncs at an admitted finite scale with normal time factors.

        A failed admission returns to the full scaled path; it neither rounds
        away exceptional components nor changes their diagnostic behavior.
        """
        if not self._finite_rate_sum or not self._ordinary_origin_in_bounds:
            return None
        if float(np.max(np.abs(outer))) > self._ordinary_time_bound:
            return None
        # For B=min(MAX/4, MAX/(4*m*gamma_max)) and every |time|<=B,
        # Delta/differences<=2B<=MAX/2, G*Delta<=MAX/2 and summed
        # response decay<=MAX/2. This conservative admission avoids repeated
        # tensor scans while retaining a factor-two overflow margin.
        sample_shape = (-1,) + (1,) * self.order
        if self._uniform_gamma:
            # Keep the original ordered rate sum instead of recomputing m*gamma.
            # Only time-factor scratch is compressed; the output stays full.
            rate_sum = self._rate_sum[(slice(0, 1),) * self.order]
            rates = self.gamma[:1]
        else:
            rate_sum = self._rate_sum
            rates = self.gamma
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            delta = lower - self.t_min
            differences = outer - lower
            x = delta.reshape(sample_shape) * rate_sum
            psi = np.negative(x)
            np.expm1(psi, out=psi)
            np.negative(psi, out=psi)
            np.divide(psi, x, out=psi, where=x != 0)
            psi[x == 0] = 1.0
            del x
            np.multiply(delta.reshape(sample_shape), psi, out=psi)
            integral = psi
            decay = np.zeros(integral.shape)
            for leg in range(self.order):
                shape = [1] * (self.order + 1)
                shape[leg + 1] = rates.size
                decay += rates.reshape(shape) * differences[leg].reshape(sample_shape)
        with np.errstate(under="ignore"):
            np.negative(decay, out=decay)
            np.exp(decay, out=decay)
            attenuation = decay
            factor = integral * attenuation
        if not np.all(
            (integral >= _FLOAT_TINY)
            & np.isfinite(integral)
            & (attenuation >= _FLOAT_TINY)
            & (factor >= _FLOAT_TINY)
            & np.isfinite(factor)
        ):
            return None
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = (
                np.empty(
                    (outer.shape[1],) + self.amplitude.shape,
                    dtype=self.amplitude.dtype,
                )
                if self._uniform_gamma
                else np.empty_like(factor, dtype=self.amplitude.dtype)
            )
            np.multiply(self.amplitude.real, factor, out=result.real)
            if self._log_amplitude_imag is not None:
                np.multiply(self.amplitude.imag, factor, out=result.imag)
        if not np.all(np.isfinite(result)):
            return None
        if self._zero_rate_slots:
            near = np.abs(result.real) >= _NEAR_MAX
            if self._log_amplitude_imag is not None:
                near |= np.abs(result.imag) >= _NEAR_MAX
            if np.any(near & (self._rate_sum == 0)):
                return None
        return result

    def _block(self, times: NDArray[np.float64]) -> np.ndarray:
        minimum = times.min(axis=0)
        positive = minimum > self.t_min
        if not np.any(positive) or self._zero_amplitude:
            return np.zeros(
                (times.shape[1],) + self.amplitude.shape, dtype=self.amplitude.dtype
            )
        all_positive = bool(np.all(positive))
        outer = times if all_positive else times[:, positive]
        lower = minimum if all_positive else minimum[positive]
        ordinary = self._ordinary_block(outer, lower)
        if ordinary is not None:
            if all_positive:
                return ordinary
            result = np.zeros(
                (times.shape[1],) + self.amplitude.shape, dtype=self.amplitude.dtype
            )
            result[positive] = ordinary
            return result
        result = np.zeros(
            (times.shape[1],) + self.amplitude.shape, dtype=self.amplitude.dtype
        )
        delta, log_delta = _positive_difference(lower, self.t_min)
        integral, log_integral = self._integral(delta, log_delta)
        decay = np.zeros(integral.shape)
        for leg in range(self.order):
            difference, log_difference = _positive_difference(outer[leg], lower)
            shape = [1] * (self.order + 1)
            shape[leg + 1] = self.n_components
            rates = self.gamma.reshape(shape)
            time_shape = (-1,) + (1,) * self.order
            with np.errstate(
                over="ignore", under="ignore", invalid="ignore", divide="ignore"
            ):
                product = rates * difference.reshape(time_shape)
                if np.any(np.isposinf(difference)):
                    stable = np.exp(np.log(rates) + log_difference.reshape(time_shape))
                    product = np.where(
                        np.isposinf(difference).reshape(time_shape), stable, product
                    )
                decay += product
        with np.errstate(under="ignore", invalid="ignore"):
            attenuation = np.exp(-decay)
            factor = integral * attenuation
        normal = (integral >= _FLOAT_TINY) & np.isfinite(integral)
        normal &= attenuation >= _FLOAT_TINY
        log_factor = log_integral - decay
        sample_shape = (-1,) + (1,) * self.order
        near_boundary = (
            self._log_amplitude_real + log_delta.reshape(sample_shape) >= _LOG_NEAR_MAX
        )
        if self._log_amplitude_imag is not None:
            near_boundary |= (
                self._log_amplitude_imag + log_delta.reshape(sample_shape)
                >= _LOG_NEAR_MAX
            )
        exact_slots = (self._rate_sum == 0) & (
            np.isposinf(delta).reshape(sample_shape) | near_boundary
        )
        if np.any(exact_slots):
            # Decide this algebraic boundary exactly, before a rounded log can
            # either lose a real overflow or refuse a representable maximum.
            factor = np.where(exact_slots, 0.0, factor)
            log_factor = np.where(exact_slots, -np.inf, log_factor)
        real = self._scale_component(
            self.amplitude.real, self._log_amplitude_real, factor, log_factor, normal
        )
        evaluated = np.empty_like(integral, dtype=self.amplitude.dtype)
        if self._log_amplitude_imag is None:
            evaluated[:] = real
        else:
            imaginary = self._scale_component(
                self.amplitude.imag,
                self._log_amplitude_imag,
                factor,
                log_factor,
                normal,
            )
            evaluated.real[:] = real
            evaluated.imag[:] = imaginary
        self._restore_zero_rate_slots(evaluated, lower, exact_slots)
        result[positive] = evaluated
        return result

    def __call__(self, n_list: Any, t_list: ArrayLike) -> np.ndarray:
        times = _real_array(t_list, "t_list")
        if times.ndim not in (1, 2) or times.shape[0] != self.order:
            raise ValueError("t_list must have shape (m,) or (m, n_samples)")
        try:
            n_legs = len(n_list)
        except TypeError as error:
            raise ValueError("n_list must contain m spatial legs") from error
        if n_legs != self.order:
            raise ValueError("n_list must contain m spatial legs")
        scalar = times.ndim == 1
        if scalar:
            times = times[:, None]
        output_bytes = times.shape[1] * self.amplitude.nbytes
        if output_bytes > self.max_output_bytes:
            raise MemoryError(
                "requested OU white-cumulant batch exceeds max_output_bytes"
            )
        if times.shape[1] == 0:
            return np.empty((0,) + self.amplitude.shape, dtype=self.amplitude.dtype)
        if times.shape[1] <= self.chunk_size:
            result = self._block(times)
            return result[0] if scalar else result
        result = np.empty(
            (times.shape[1],) + self.amplitude.shape, dtype=self.amplitude.dtype
        )
        for start in range(0, times.shape[1], self.chunk_size):
            result[start : start + self.chunk_size] = self._block(
                times[:, start : start + self.chunk_size]
            )
        return result[0] if scalar else result
