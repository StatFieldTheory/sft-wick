"""Explicit finite real Gaussian quadratic observables and their selected CGFs.

These laws sum Gaussian rings and chains of the declared observable. They do
not infer a quadratic observable from an action or complete nonlinear dynamics.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain as itertools_chain, permutations
import math
from numbers import Integral
from typing import Any, Iterator, Sequence, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._mixed_contraction import RoundingCertificate, admits, rounding_certificate
from ._mixed_rounding import correctly_rounded_statistic

RealArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
SourceResult: TypeAlias = float | complex | RealArray | ComplexArray
ComplexResult: TypeAlias = complex | ComplexArray

__all__ = [
    "GaussianQuadratic",
    "GaussianQuadraticFamily",
    "GaussianQuadraticStatistics",
    "GaussianSpectralTail",
]

_StatisticKey: TypeAlias = tuple[int, tuple[int, ...]]
_StatisticOutcome: TypeAlias = int | tuple[type[ArithmeticError], str]
_MAX_PREPARED_STATISTICS = 1024
_MAX_PREPARED_KEY_ITEMS = 65_536
_MAX_SUBMITTED_QUERIES = 65_536
_MIXED_PREFIX_MIN_RANK = 256
_MIXED_PREFIX_MIN_DISTINCT = 3
_MIXED_PREFIX_MAX_BYTES = 8 * 1024 * 1024


def _logsumexp(values: Sequence[float] | RealArray) -> float:
    """Sum positive logarithmic terms without importing more NumPy consumers.

    Only one-dimensional real logarithms are needed by the spectral statistics
    and contraction bounds. Empty/all-negative-infinite sums are zero in
    ordinary space; a positive-infinite term dominates. Max shifting keeps
    every exponential bounded by one, and fsum preserves small contributions.
    """
    terms = tuple(float(value) for value in values)
    largest = max(terms, default=-math.inf)
    if not math.isfinite(largest):
        return largest
    return largest + math.log(math.fsum(math.exp(value - largest) for value in terms))


def _real_array(value, name: str, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = np.array(array, dtype=np.float64, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _scalar(value, name: str) -> float:
    array = _real_array(value, name, ndim=0)
    return float(array)


def _snapshot(array: np.ndarray) -> np.ndarray:
    """Read-only bytes backing also prevents callers re-enabling WRITEABLE."""
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _exact_signed_sum(values) -> Fraction:
    return sum((Fraction(float(value)) for value in values), Fraction(0))


def _finite_fraction(total: Fraction, name: str) -> float:
    try:
        return float(total)
    except OverflowError as error:
        raise OverflowError(
            f"{name} exceeds float64 range; rescale the observable"
        ) from error


def _finite_signed_sum(values, name: str) -> float:
    """Sum prepared binary floats exactly before one final float conversion."""
    return _finite_fraction(_exact_signed_sum(values), name)


def _scaled_exact_scalar(value: Fraction, source: np.ndarray) -> np.ndarray:
    """Scale an accurate scalar sum, admitting a finite product of a large sum.

    The ordinary path is one rounded scalar times the source vector. Only a
    sum outside float64 requires exact multiplication before conversion.
    """
    try:
        rounded = float(value)
    except OverflowError:
        result = np.empty_like(source)
        for index, item in enumerate(source):
            if np.iscomplexobj(source):
                result[index] = complex(
                    _finite_fraction(
                        value * Fraction(float(item.real)), "source-scaled first moment"
                    ),
                    _finite_fraction(
                        value * Fraction(float(item.imag)), "source-scaled first moment"
                    ),
                )
            else:
                result[index] = _finite_fraction(
                    value * Fraction(float(item)), "source-scaled first moment"
                )
        return result
    return source * rounded


def _covariance_factor(covariance) -> np.ndarray:
    covariance = _real_array(covariance, "covariance", ndim=2)
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if not np.array_equal(covariance, covariance.T):
        raise ValueError("covariance must be exactly symmetric; symmetrize explicitly")
    eigenvalues, basis = np.linalg.eigh(covariance)
    if np.any(eigenvalues < 0):
        raise ValueError(
            "covariance must be positive semidefinite; use an explicit factor for singular data"
        )
    positive = eigenvalues > 0
    return basis[:, positive] * np.sqrt(eigenvalues[positive])


def _observable_data(factor, quadratic, linear, constant, mean):
    factor = _real_array(factor, "factor", ndim=2)
    dimension = factor.shape[0]
    quadratic = _real_array(quadratic, "quadratic", ndim=2)
    if quadratic.shape != (dimension, dimension):
        raise ValueError("quadratic must be square with the factor's row dimension")
    # The antisymmetric part makes identically zero contribution to z.T A z.
    quadratic = quadratic / 2 + quadratic.T / 2
    linear = (
        np.zeros(dimension) if linear is None else _real_array(linear, "linear", ndim=1)
    )
    mean = np.zeros(dimension) if mean is None else _real_array(mean, "mean", ndim=1)
    if linear.shape != (dimension,) or mean.shape != (dimension,):
        raise ValueError("linear and mean must match the factor's row dimension")
    constant = _scalar(constant, "constant")
    with np.errstate(over="raise", invalid="raise"):
        try:
            shifted = constant + linear @ mean + mean @ quadratic @ mean
            endpoint = factor.T @ (linear + 2 * (quadratic @ mean))
            matrix = factor.T @ quadratic @ factor
        except FloatingPointError as error:
            raise ValueError("whitened observable exceeds float64 range") from error
    # Matrix products can differ by roundoff between the two triangles.
    matrix = matrix / 2 + matrix.T / 2
    if not (
        np.isfinite(shifted)
        and np.all(np.isfinite(endpoint))
        and np.all(np.isfinite(matrix))
    ):
        raise ValueError("whitened observable exceeds float64 range")
    return matrix, endpoint, float(shifted), dimension


def _source_array(value, *, real_only: bool = False) -> np.ndarray:
    array = np.asarray(value)
    allowed = "iuf" if real_only else "iufc"
    if array.dtype.kind not in allowed:
        raise ValueError(
            "sources must contain real numeric values"
            if real_only
            else "sources must contain numeric values"
        )
    array = np.array(
        array, dtype=np.complex128 if np.iscomplexobj(array) else np.float64, copy=True
    )
    if not np.all(np.isfinite(array)):
        raise ValueError("sources must be finite")
    return array


def _log1p_minus_argument(argument: np.ndarray) -> np.ndarray:
    """Accurate log(1+x)-x near zero, with the same analytic branch."""
    result = np.log1p(argument) - argument
    small = np.abs(argument) < 0.01
    if np.any(small):
        x = argument[small]
        value, power = -x * x / 2, x * x * x
        for order in range(3, 13):
            value += (-1) ** (order + 1) * power / order
            power *= x
        result[small] = value
    return result


def _domain(eigenvalues: np.ndarray) -> tuple[float, float]:
    negative = eigenvalues[eigenvalues < 0]
    positive = eigenvalues[eigenvalues > 0]
    with np.errstate(over="ignore", divide="ignore"):
        lower = float(0.5 / negative.min()) if negative.size else -math.inf
        upper = float(0.5 / positive.max()) if positive.size else math.inf
    return lower, upper


def _exact_cancellation_coefficient(terms, order: int, *, cumulant: bool) -> float:
    """Targeted binary-rational fallback when distinct log groups nearly cancel.

    A work bound prevents an adversarial large integer order from allocating
    unbounded integer powers. The usual noncancelling calculation stays in log
    space. This fallback is exact for the prepared floating-point spectral data,
    not a certificate of its eigensolver or physical model.
    """
    power = order - 2
    estimated_bits = 0.0
    for scaled, weight in terms:
        estimated_bits += power * (
            scaled.numerator.bit_length() - 1 + scaled.denominator.bit_length() - 1
        )
        estimated_bits += (
            abs(weight.numerator).bit_length() + weight.denominator.bit_length()
        )
    if cumulant:
        estimated_bits += math.lgamma(order) / math.log(2.0)
    if estimated_bits > 8_000_000:
        raise ArithmeticError(
            "severe spectral cancellation exceeds the exact fallback work bound; use higher precision"
        )
    total = Fraction(0)
    for scaled, weight in terms:
        total += weight * scaled**power
    exact = total * math.factorial(order - 1) / 2 if cumulant else total / (2 * order)
    try:
        return float(exact)
    except OverflowError as error:
        raise OverflowError(
            "spectral statistic exceeds float64 range; rescale the observable"
        ) from error


def _spectral_coefficient(
    eigenvalues: np.ndarray,
    linear_squares: np.ndarray,
    order: int,
    *,
    cumulant: bool,
    tail_variance: float = 0.0,
) -> float:
    """Combine signed terms in log space before deciding float64 range.

    Ring terms are Gamma(n)*(2*lambda)**n/2 for cumulants, and
    (2*lambda)**n/(2*n) for Taylor coefficients. Chain terms are
    Gamma(n+1)*v**2*(2*lambda)**(n-2)/2 or
    v**2*(2*lambda)**(n-2)/2, respectively. Keeping the powers scaled
    avoids separately overflowing 2**n or factorial(n) and underflowing
    lambda**n when the final result is representable.
    """
    log_two = math.log(2.0)
    groups: dict[float, tuple[int, Fraction]] = {}
    for eigenvalue, weight in zip(eigenvalues, linear_squares):
        magnitude = abs(float(eigenvalue))
        if magnitude == 0 and order > 2:
            continue
        group_sign = (-1 if eigenvalue < 0 else 1) if order % 2 else 1
        count, linear = groups.get(magnitude, (0, Fraction(0)))
        groups[magnitude] = count + group_sign, linear + group_sign * Fraction(
            float(weight)
        )
    logs, signs, exact_terms = [], [], []
    for magnitude, (count, linear) in groups.items():
        # Combine each exactly equal-|lambda| group before log scaling.
        # Binary-rational accumulation keeps an exactly cancelling dominant
        # group from swallowing a much smaller finite group in logsumexp.
        weight = 4 * Fraction(magnitude) ** 2 * count + order * linear
        if weight == 0:
            continue
        log_weight = math.log(abs(weight.numerator)) - math.log(weight.denominator)
        power = (order - 2) * (math.log(magnitude) + log_two) if order > 2 else 0.0
        prefactor = (
            math.lgamma(order) - log_two if cumulant else -log_two - math.log(order)
        )
        logs.append(prefactor + power + log_weight)
        signs.append(1.0 if weight > 0 else -1.0)
        exact_terms.append((2 * Fraction(magnitude), weight))
    if order == 2 and tail_variance:
        logs.append(math.log(tail_variance) - (0.0 if cumulant else log_two))
        signs.append(1.0)
    if not logs:
        return 0.0
    values, weights = np.array(logs), np.array(signs)
    positive = _logsumexp(values[weights > 0])
    negative = _logsumexp(values[weights < 0])
    # Equality of rounded logarithms is not equality of distinct magnitudes.
    # Near cancellation needs the exact prepared binary spectral data, so a
    # finite residual is not mistaken for zero or lost below the log scale.
    if np.isfinite(positive) and np.isfinite(negative):
        gap = abs(positive - negative)
        if gap <= max(
            1e-4,
            32
            * float(np.finfo(np.float64).eps)
            * max(abs(positive), abs(negative), 1.0),
        ):
            return _exact_cancellation_coefficient(
                exact_terms, order, cumulant=cumulant
            )
    sign = 1.0 if positive > negative else -1.0
    larger, smaller = max(positive, negative), min(positive, negative)
    log_magnitude = larger + math.log(-math.expm1(smaller - larger))
    if not np.isfinite(log_magnitude) or log_magnitude > math.log(
        np.finfo(np.float64).max
    ):
        raise OverflowError(
            "spectral statistic exceeds float64 range; rescale the observable"
        )
    # math.exp returns zero for a true final underflow; no intermediate power
    # or factorial has been converted to a float before the combined result.
    return float(sign) * math.exp(float(log_magnitude))


def _log_frobenius_norm(array: np.ndarray) -> float:
    largest = float(np.max(np.abs(array), initial=0.0))
    if largest == 0:
        return -math.inf
    with np.errstate(under="ignore"):
        scaled = array / largest
        return math.log(largest) + 0.5 * math.log(float(np.sum(scaled**2)))


def _frobenius_cancellation(
    matrices: np.ndarray,
    endpoints: np.ndarray,
    slots: tuple[int, ...],
    divisor: int,
    result: Fraction,
) -> bool:
    """True when ``|result|`` is at most 1e-10 of a Frobenius-norm bound.

    This is the cancellation test that version 0.7.0 used to send a float64
    value to its exact evaluation, evaluated in the same floating-point steps.
    Values it flags are evaluated exactly even when the rounding certificate
    admits them, so every value 0.7.0 returned is returned with the same bits
    or as the correctly rounded exact value.
    """
    order = len(slots)
    matrix_norms = {index: _log_frobenius_norm(matrices[index]) for index in set(slots)}
    endpoint_norms = {
        index: _log_frobenius_norm(endpoints[index]) for index in set(slots)
    }
    # |tr(A1...An)| <= product(||Ai||_F), and an endpoint chain is bounded by
    # its endpoint norms times its intermediate matrix norms.
    chain_bounds = [
        endpoint_norms[word[0]]
        + endpoint_norms[word[-1]]
        + sum(matrix_norms[index] for index in word[1:-1])
        for word in permutations(slots)
    ]
    ring_bound = (
        math.lgamma(order + 1)
        + (order - 1) * math.log(2.0)
        - math.log(order)
        + sum(matrix_norms[index] for index in slots)
    )
    chain_bound = _logsumexp(chain_bounds) + (order - 3) * math.log(2.0)
    bound = _logsumexp([ring_bound, chain_bound]) - math.log(divisor)
    log_result = (
        -math.inf
        if not result
        else math.log(abs(result.numerator)) - math.log(result.denominator)
    )
    return bool(np.isfinite(bound) and log_result <= bound + math.log(1e-10))


def _magnitude_estimate(
    value: Fraction | None, certificate: RoundingCertificate
) -> Fraction | None:
    """Lower bound on the exact magnitude when certified, else ``|value|``."""
    if not value:
        return None
    size = abs(value)
    if certificate.error is not None and certificate.error < size:
        return size - certificate.error
    return size


def _exact_mixed_coefficient(
    matrices: np.ndarray,
    endpoints: np.ndarray,
    slots: tuple[int, ...],
    divisor: int,
    magnitude: Fraction | None = None,
) -> float:
    """Correctly rounded exact mixed statistic of the prepared binary data."""
    return _finite_fraction(
        correctly_rounded_statistic(matrices, endpoints, slots, divisor, magnitude),
        "mixed coefficient" if divisor > 1 else "mixed cumulant",
    )


@dataclass(frozen=True, slots=True)
class GaussianSpectralTail:
    """Conditional finite-model tail summaries, not interval certificates.

    An omitted tail is replaced by an independent Gaussian matching its
    first two cumulants. ``mean`` is a rounded diagnostic; the compressed law
    separately preserves its total first moment from the prepared binary
    spectrum. ``variance`` describes the Gaussian tail. The CGF bound uses
    ``delta``, ``trace_square`` and ``linear_square`` computed from that spectrum.
    """

    modes: int
    mean: float
    variance: float
    delta: float
    trace_square: float
    linear_square: float


@dataclass(frozen=True, slots=True, init=False, eq=False)
class GaussianQuadratic:
    """Prepared law of ``b + ell.T z + z.T A z`` for real Gaussian ``z``.

    All inputs are copied. Covariance is rejected unless exactly symmetric
    with no negative computed eigenvalues. ``from_factor`` accepts an explicit
    real factor ``L`` defining the PSD covariance ``L @ L.T`` without that
    eigensolver test. No eigenvalue clipping or regularization is implicit.

    Preparation is dense/cubic; a prepared source evaluation is linear in
    latent dimension, with temporary memory bounded by ``chunk_size`` times
    the retained dimension. Characteristic functions concern a real-valued
    observable; complex random inputs and MSR response blocks are unsupported.
    """

    _eigenvalues: np.ndarray
    _linear_squares: np.ndarray
    _constant: float
    _ambient_dimension: int
    _latent_dimension: int
    _tail: GaussianSpectralTail | None
    _reference_domain: tuple[float, float]
    _mean_exact: Fraction
    _shift_exact: Fraction

    def __init__(
        self,
        covariance: ArrayLike,
        quadratic: ArrayLike,
        linear: ArrayLike | None = None,
        *,
        constant: float = 0.0,
        mean: ArrayLike | None = None,
    ) -> None:
        prepared = type(self).from_covariance(
            covariance, quadratic, linear, constant=constant, mean=mean
        )
        for name in self.__slots__:
            object.__setattr__(self, name, getattr(prepared, name))

    @classmethod
    def from_covariance(
        cls,
        covariance: ArrayLike,
        quadratic: ArrayLike,
        linear: ArrayLike | None = None,
        *,
        constant: float = 0.0,
        mean: ArrayLike | None = None,
    ) -> GaussianQuadratic:
        """Prepare a strict symmetric PSD covariance; never clip eigenvalues."""
        return cls.from_factor(
            _covariance_factor(covariance),
            quadratic,
            linear,
            constant=constant,
            mean=mean,
        )

    @classmethod
    def from_factor(
        cls,
        factor: ArrayLike,
        quadratic: ArrayLike,
        linear: ArrayLike | None = None,
        *,
        constant: float = 0.0,
        mean: ArrayLike | None = None,
    ) -> GaussianQuadratic:
        """Prepare with an explicit finite real ``(N,r)`` covariance factor."""
        matrix, endpoint, shifted, dimension = _observable_data(
            factor, quadratic, linear, constant, mean
        )
        return cls._from_whitened(matrix, endpoint, shifted, dimension)

    @classmethod
    def _from_whitened(cls, matrix, endpoint, shifted, dimension) -> GaussianQuadratic:
        eigenvalues, basis = np.linalg.eigh(matrix)
        if not (np.all(np.isfinite(eigenvalues)) and np.all(np.isfinite(basis))):
            raise ValueError("quadratic eigensystem exceeds float64 range")
        with np.errstate(over="raise", invalid="raise"):
            try:
                rotated = basis.T @ endpoint
                linear_squares = rotated**2
            except FloatingPointError as error:
                raise ValueError(
                    "rotated linear observable exceeds float64 range"
                ) from error
        if np.any((rotated != 0) & (linear_squares < np.finfo(np.float64).tiny)):
            raise ValueError(
                "nonzero rotated linear square is subnormal or underflows float64 storage; rescale the observable"
            )
        return cls._from_spectrum(
            eigenvalues,
            linear_squares,
            shifted,
            dimension,
            len(eigenvalues),
            None,
            _domain(eigenvalues),
        )

    @classmethod
    def _from_spectrum(
        cls,
        eigenvalues,
        linear_squares,
        shifted,
        dimension,
        latent_dimension,
        tail,
        reference_domain,
        mean_exact: Fraction | None = None,
    ) -> GaussianQuadratic:
        result = object.__new__(cls)
        spectral_mean = _exact_signed_sum(eigenvalues)
        if mean_exact is None:
            mean_exact = Fraction(float(shifted)) + spectral_mean
        for name, value in (
            ("_eigenvalues", _snapshot(eigenvalues)),
            ("_linear_squares", _snapshot(linear_squares)),
            ("_constant", float(shifted)),
            ("_ambient_dimension", dimension),
            ("_latent_dimension", latent_dimension),
            ("_tail", tail),
            ("_reference_domain", reference_domain),
            ("_mean_exact", mean_exact),
            ("_shift_exact", mean_exact - spectral_mean),
        ):
            object.__setattr__(result, name, value)
        return result

    def __copy__(self) -> GaussianQuadratic:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> GaussianQuadratic:
        memo[id(self)] = self
        return self

    def __reduce__(self):
        return (
            type(self)._from_spectrum,
            (
                self._eigenvalues,
                self._linear_squares,
                self._constant,
                self._ambient_dimension,
                self._latent_dimension,
                self._tail,
                self._reference_domain,
                self._mean_exact,
            ),
        )

    @classmethod
    def from_weighted_grid(
        cls,
        covariance_kernel: ArrayLike,
        quadratic_kernel: ArrayLike,
        weights: ArrayLike,
        linear_kernel: ArrayLike | None = None,
        *,
        constant: float = 0.0,
        mean: ArrayLike | None = None,
    ) -> GaussianQuadratic:
        """Use sampled colored covariance and positive quadrature weights.

        The nodal observable is ``b + ell.T W z + z.T W A W z``. The
        covariance kernel is nodal covariance, not white-noise intensity.
        """
        weights = _real_array(weights, "weights", ndim=1)
        if np.any(weights <= 0):
            raise ValueError("weights must be strictly positive")
        matrix = _real_array(quadratic_kernel, "quadratic_kernel", ndim=2)
        if matrix.shape != (len(weights), len(weights)):
            raise ValueError("quadratic_kernel and weights have incompatible shapes")
        endpoint = (
            np.zeros(len(weights))
            if linear_kernel is None
            else _real_array(linear_kernel, "linear_kernel", ndim=1)
        )
        if endpoint.shape != weights.shape:
            raise ValueError("linear_kernel and weights have incompatible shapes")
        return cls.from_covariance(
            covariance_kernel,
            weights[:, None] * matrix * weights[None, :],
            weights * endpoint,
            constant=constant,
            mean=mean,
        )

    @classmethod
    def from_white_noise(
        cls,
        intensity: ArrayLike,
        quadratic_kernel: ArrayLike,
        weights: ArrayLike,
        linear_kernel: ArrayLike | None = None,
        *,
        constant: float = 0.0,
        mean: ArrayLike | None = None,
    ) -> GaussianQuadratic:
        """Use white-noise intensity, whose nodal covariance is diag(c/w)."""
        weights = _real_array(weights, "weights", ndim=1)
        intensity = _real_array(intensity, "intensity", ndim=1)
        if intensity.shape != weights.shape or np.any(intensity < 0):
            raise ValueError("intensity must match weights and be nonnegative")
        if np.any(weights <= 0):
            raise ValueError("weights must be strictly positive")
        matrix = _real_array(quadratic_kernel, "quadratic_kernel", ndim=2)
        if matrix.shape != (len(weights), len(weights)):
            raise ValueError("quadratic_kernel and weights have incompatible shapes")
        endpoint = (
            np.zeros(len(weights))
            if linear_kernel is None
            else _real_array(linear_kernel, "linear_kernel", ndim=1)
        )
        if endpoint.shape != weights.shape:
            raise ValueError("linear_kernel and weights have incompatible shapes")
        # x=Wz has covariance diag(w*c), so its factor can be formed without
        # underflowing the intermediate nodal ratio c/w or weighted kernel.
        with np.errstate(over="raise", divide="raise", invalid="raise", under="raise"):
            try:
                scale = np.sqrt(weights) * np.sqrt(intensity)
                weighted_mean = (
                    None
                    if mean is None
                    else weights * _real_array(mean, "mean", ndim=1)
                )
            except FloatingPointError as error:
                raise ValueError(
                    "white-noise weighted factor or mean is not representable in float64"
                ) from error
        if np.any((intensity > 0) & (scale < np.finfo(np.float64).tiny)):
            raise ValueError(
                "positive white-noise weighted factor is subnormal or underflows float64 storage"
            )
        if weighted_mean is not None and weighted_mean.shape != weights.shape:
            raise ValueError("mean and weights have incompatible shapes")
        return cls.from_factor(
            np.diag(scale), matrix, endpoint, constant=constant, mean=weighted_mean
        )

    @property
    def eigenvalues(self) -> np.ndarray:
        """Retained quadratic spectrum, backed by immutable bytes."""
        return self._eigenvalues

    @property
    def ambient_dimension(self) -> int:
        return self._ambient_dimension

    @property
    def latent_dimension(self) -> int:
        """Number of original covariance-factor columns, not a claimed rank."""
        return self._latent_dimension

    @property
    def retained_dimension(self) -> int:
        return len(self._eigenvalues)

    @property
    def spectral_tail(self) -> GaussianSpectralTail | None:
        return self._tail

    @property
    def real_domain(self) -> tuple[float, float]:
        """Open MGF interval of this law (the approximate law if compressed)."""
        return _domain(self._eigenvalues)

    @property
    def reference_real_domain(self) -> tuple[float, float]:
        """Original prepared law's open interval, retained after compression."""
        return self._reference_domain

    @property
    def mean(self) -> float:
        return _finite_fraction(self._mean_exact, "mean")

    @property
    def variance(self) -> float:
        return _spectral_coefficient(
            self._eigenvalues,
            self._linear_squares,
            2,
            cumulant=True,
            tail_variance=self._tail.variance if self._tail else 0.0,
        )

    def _evaluate_chunk(self, source: np.ndarray, *, centered: bool) -> np.ndarray:
        argument = -2 * (source[:, None] * self._eigenvalues)
        if np.any(1 + argument.real <= 0):
            raise ValueError("source real part is outside the open Gaussian MGF domain")
        if centered:
            ring = -0.5 * np.sum(_log1p_minus_argument(argument), axis=1)
        else:
            # Near zero raw log terms can lose the quadratic remainder before
            # signed first-order terms cancel. Use the accurate centered ring
            # plus the total mean there. Large-argument rows retain raw logs,
            # avoiding cancellation of a large artificial centering term.
            small = np.all(np.abs(argument) < 0.01, axis=1)
            ring = np.empty_like(source)
            ring[small] = -0.5 * np.sum(_log1p_minus_argument(argument[small]), axis=1)
            ring[small] += _scaled_exact_scalar(self._mean_exact, source[small])
            ring[~small] = -0.5 * np.sum(np.log1p(argument[~small]), axis=1)
            ring[~small] += _scaled_exact_scalar(self._shift_exact, source[~small])
        if np.any(self._linear_squares):
            weighted_source = source[:, None] * np.sqrt(self._linear_squares)
            chain = 0.5 * np.sum(
                weighted_source * (weighted_source / (1 + argument)), axis=1
            )
        else:
            chain = np.zeros_like(source)
        tail_variance = self._tail.variance if self._tail else 0.0
        result = ring + chain
        if tail_variance:
            result += 0.5 * (source * math.sqrt(tail_variance)) ** 2
        return result

    def cgf(
        self, source: ArrayLike, *, centered: bool = False, chunk_size: int = 256
    ) -> SourceResult:
        """Evaluate log E exp(tY), with real parts in this law's open domain.

        Scalar input returns a scalar; array input preserves shape. At most
        ``chunk_size`` source rows are broadcast against the spectrum at once.
        Floating-point overflow raises instead of returning a nonfinite CGF.
        """
        if not isinstance(centered, (bool, np.bool_)):
            raise ValueError("centered must be boolean")
        chunk_size = _integer(chunk_size, "chunk_size", minimum=1)
        array = _source_array(source)
        flattened = array.reshape(-1)
        result = np.empty_like(flattened)
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            try:
                for start in range(0, len(flattened), chunk_size):
                    result[start : start + chunk_size] = self._evaluate_chunk(
                        flattened[start : start + chunk_size], centered=bool(centered)
                    )
            except FloatingPointError as error:
                raise ValueError(
                    "source evaluation exceeds float64 range or reaches a pole"
                ) from error
        if not np.all(np.isfinite(result)):
            raise ValueError("source evaluation produced a nonfinite CGF")
        result = result.reshape(array.shape)
        return result.item() if result.ndim == 0 else result

    def log_characteristic(
        self, frequency: ArrayLike, *, centered: bool = False, chunk_size: int = 256
    ) -> ComplexResult:
        """Continuous log E exp(i omega Y), anchored to zero at omega=0."""
        return cast(
            ComplexResult,
            self.cgf(
                1j * _source_array(frequency, real_only=True),
                centered=centered,
                chunk_size=chunk_size,
            ),
        )

    def characteristic(
        self, frequency: ArrayLike, *, centered: bool = False, chunk_size: int = 256
    ) -> ComplexResult:
        """Characteristic function; very small magnitudes may underflow to zero."""
        return cast(
            ComplexResult,
            np.exp(
                self.log_characteristic(
                    frequency, centered=centered, chunk_size=chunk_size
                )
            ),
        )

    def cumulant(self, order: int, *, centered: bool = False) -> float:
        """Selected-law cumulant of positive integer order."""
        order = _integer(order, "order", minimum=1)
        if not isinstance(centered, (bool, np.bool_)):
            raise ValueError("centered must be boolean")
        if order == 1:
            return 0.0 if centered else self.mean
        if order == 2:
            return self.variance
        return _spectral_coefficient(
            self._eigenvalues, self._linear_squares, order, cumulant=True
        )

    def taylor_coefficients(
        self, max_order: int, *, centered: bool = False
    ) -> np.ndarray:
        """Return coefficients of t**n, n=0..max_order; constant coefficient is zero."""
        max_order = _integer(max_order, "max_order")
        if not isinstance(centered, (bool, np.bool_)):
            raise ValueError("centered must be boolean")
        coefficients = np.zeros(max_order + 1)
        if max_order:
            coefficients[1] = 0.0 if centered else self.mean
        if max_order >= 2:
            coefficients[2] = self.taylor_coefficient(2)
        for order in range(3, max_order + 1):
            coefficients[order] = self.taylor_coefficient(order)
        if not np.all(np.isfinite(coefficients)):
            raise OverflowError("Taylor coefficient exceeds float64 range")
        return coefficients

    def taylor_coefficient(self, order: int, *, centered: bool = False) -> float:
        """One CGF Taylor coefficient, without forming any lower-order terms.

        This can return a finite odd coefficient when lower even coefficients
        exceed float64 range. ``taylor_coefficients`` requires every returned
        array entry to be representable.
        """
        order = _integer(order, "order")
        if not isinstance(centered, (bool, np.bool_)):
            raise ValueError("centered must be boolean")
        if order == 0:
            return 0.0
        if order == 1:
            return 0.0 if centered else self.mean
        return _spectral_coefficient(
            self._eigenvalues,
            self._linear_squares,
            order,
            cumulant=False,
            tail_variance=self._tail.variance if self._tail else 0.0,
        )

    def compress(self, retain_modes: int) -> GaussianQuadratic:
        """Explicitly approximate omitted eigenmodes by a matching Gaussian.

        Keep the largest absolute quadratic eigenvalues. Tail linear variance
        is also preserved, including zero-eigenvalue directions. Preparation
        still uses the complete dense spectrum; this saves repeated source
        evaluation, not the initial factorization. Further compression of an
        already approximate law is refused to keep the reference unambiguous.
        """
        retain_modes = _integer(retain_modes, "retain_modes")
        if self._tail is not None:
            raise ValueError(
                "compress the original prepared law, not an already compressed law"
            )
        if retain_modes > self.retained_dimension:
            raise ValueError("retain_modes exceeds the latent dimension")
        if retain_modes == self.retained_dimension:
            return self
        order = np.argsort(-np.abs(self._eigenvalues), kind="stable")
        retained, omitted = order[:retain_modes], order[retain_modes:]
        spectrum, linear = self._eigenvalues[omitted], self._linear_squares[omitted]
        with np.errstate(over="raise", invalid="raise", under="raise"):
            try:
                trace_square = float(np.sum(spectrum**2))
                linear_square = float(np.sum(linear))
                tail_mean = _finite_signed_sum(spectrum, "spectral tail mean")
                variance = 2 * trace_square + linear_square
                delta = float(np.max(np.abs(spectrum), initial=0.0))
            except FloatingPointError as error:
                raise ValueError(
                    "spectral tail summaries are not representable in float64; compression refused"
                ) from error
        if not all(
            np.isfinite(value)
            for value in (trace_square, linear_square, tail_mean, variance, delta)
        ):
            raise ValueError(
                "spectral tail summaries are not representable in float64; compression refused"
            )
        tail = GaussianSpectralTail(
            len(omitted), tail_mean, variance, delta, trace_square, linear_square
        )
        return type(self)._from_spectrum(
            self._eigenvalues[retained],
            self._linear_squares[retained],
            self._constant,
            self.ambient_dimension,
            self.latent_dimension,
            tail,
            self.reference_real_domain,
            self._mean_exact,
        )

    def cgf_error_bound(self, source: ArrayLike) -> float | RealArray:
        """Conditional CGF truncation bound when 2*abs(t)*tail.delta < 1.

        This uses floating-point spectral summaries, not interval arithmetic.
        It bounds only compression of the selected finite model, not matrix
        preprocessing, continuous quadrature, nonlinear closure or PDF errors.
        """
        if self._tail is None:
            raise ValueError("cgf_error_bound requires an explicitly compressed law")
        array = _source_array(source)
        # Check retained poles/strip as well as the stronger omitted-tail disk.
        lower, upper = self.real_domain
        if np.any(array.real <= lower) or np.any(array.real >= upper):
            raise ValueError(
                "source real part is outside the approximate law's open domain"
            )
        magnitude = np.abs(array)
        if self._tail.delta == 0:
            result = np.zeros_like(magnitude)
            return result.item() if result.ndim == 0 else result
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                scaled_source = magnitude * self._tail.delta
                q = 2 * scaled_source
                if np.any(q >= 1):
                    raise ValueError("tail bound requires 2*abs(source)*delta < 1")
                scaled_trace = (magnitude * math.sqrt(self._tail.trace_square)) ** 2
                scaled_linear = (magnitude * math.sqrt(self._tail.linear_square)) ** 2
                result = (
                    scaled_source * (4 * scaled_trace / 3 + scaled_linear) / (1 - q)
                )
            except FloatingPointError as error:
                raise ValueError(
                    "tail bound evaluation exceeds float64 range"
                ) from error
        return result.item() if result.ndim == 0 else result


@dataclass(frozen=True, slots=True, init=False, eq=False)
class GaussianQuadraticFamily:
    """Explicit correlated quadratic observables of one finite Gaussian input.

    Fixed directions can be projected to a prepared ``GaussianQuadratic`` law.
    Arbitrary real joint CGFs and characteristic sources require a fresh dense
    eigensystem per source; no simultaneous diagonalization is assumed. Mixed
    cumulants and Taylor monomial coefficients are supported through order four.
    """

    _matrices: np.ndarray
    _endpoints: np.ndarray
    _constants: np.ndarray
    _ambient_dimension: int

    def __init__(
        self,
        covariance: ArrayLike,
        quadratic: ArrayLike,
        linear: ArrayLike | None = None,
        *,
        constant: ArrayLike | None = None,
        mean: ArrayLike | None = None,
    ) -> None:
        prepared = type(self).from_factor(
            _covariance_factor(covariance),
            quadratic,
            linear,
            constant=constant,
            mean=mean,
        )
        for name in self.__slots__:
            object.__setattr__(self, name, getattr(prepared, name))

    @classmethod
    def from_factor(
        cls,
        factor: ArrayLike,
        quadratic: ArrayLike,
        linear: ArrayLike | None = None,
        *,
        constant: ArrayLike | None = None,
        mean: ArrayLike | None = None,
    ) -> GaussianQuadraticFamily:
        factor = _real_array(factor, "factor", ndim=2)
        matrices = _real_array(quadratic, "quadratic", ndim=3)
        count, dimension, other = matrices.shape
        if count < 1 or dimension != other or dimension != factor.shape[0]:
            raise ValueError(
                "quadratic must have shape (observables, N, N) with at least one observable"
            )
        linear = (
            np.zeros((count, dimension))
            if linear is None
            else _real_array(linear, "linear", ndim=2)
        )
        constant = (
            np.zeros(count)
            if constant is None
            else _real_array(constant, "constant", ndim=1)
        )
        if linear.shape != (count, dimension) or constant.shape != (count,):
            raise ValueError(
                "linear and constant must match the observable count and input dimension"
            )
        data = [
            _observable_data(factor, matrices[i], linear[i], constant[i], mean)
            for i in range(count)
        ]
        return cls._from_prepared(
            np.array([item[0] for item in data]),
            np.array([item[1] for item in data]),
            np.array([item[2] for item in data]),
            dimension,
        )

    @classmethod
    def _from_prepared(
        cls, matrices, endpoints, constants, dimension
    ) -> GaussianQuadraticFamily:
        result = object.__new__(cls)
        object.__setattr__(result, "_matrices", _snapshot(matrices))
        object.__setattr__(result, "_endpoints", _snapshot(endpoints))
        object.__setattr__(result, "_constants", _snapshot(constants))
        object.__setattr__(result, "_ambient_dimension", dimension)
        return result

    def __copy__(self) -> GaussianQuadraticFamily:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> GaussianQuadraticFamily:
        memo[id(self)] = self
        return self

    def __reduce__(self):
        return (
            type(self)._from_prepared,
            (self._matrices, self._endpoints, self._constants, self._ambient_dimension),
        )

    @property
    def observable_count(self) -> int:
        return len(self._constants)

    def project(self, direction: ArrayLike) -> GaussianQuadratic:
        """Prepare the real linear combination sum(direction[a] * Y[a])."""
        direction = _real_array(direction, "direction", ndim=1)
        if direction.shape != (self.observable_count,):
            raise ValueError("direction must match the observable count")
        matrix = np.tensordot(direction, self._matrices, axes=1)
        endpoint = direction @ self._endpoints
        shifted = _finite_fraction(
            sum(
                (
                    Fraction(float(weight)) * Fraction(float(value))
                    for weight, value in zip(direction, self._constants)
                ),
                Fraction(0),
            ),
            "projected constant",
        )
        if not (
            np.all(np.isfinite(matrix))
            and np.all(np.isfinite(endpoint))
            and np.isfinite(shifted)
        ):
            raise ValueError("projected observable exceeds float64 range")
        return GaussianQuadratic._from_whitened(
            matrix, endpoint, shifted, self._ambient_dimension
        )

    def cgf(self, source: ArrayLike) -> float:
        """Real joint CGF at one source vector; projection is refactored here."""
        return cast(float, self.project(source).cgf(1.0))

    def log_characteristic(self, frequency: ArrayLike) -> complex:
        """Joint characteristic log at one real frequency vector."""
        return cast(complex, self.project(frequency).cgf(1j))

    def characteristic(self, frequency: ArrayLike) -> complex:
        return complex(np.exp(self.log_characteristic(frequency)))

    def mixed_cumulant(self, indices: Sequence[int]) -> float:
        """Exact ordered-slot symmetrization for one to four observable indices.

        The float64 contraction is returned only when its rounding error is
        certified to be at most 1e-12 of its magnitude, it raised no
        floating-point exception and the cancellation test of version 0.7.0
        does not flag it. Otherwise the exact value of the prepared matrices
        and endpoints is returned correctly rounded to float64, and a proved
        zero returns 0.0. ``ArithmeticError`` is raised when the exact
        evaluation that the rounding needs exceeds its work or memory bound
        and the float64 value is not certified.
        """
        slots = tuple(_integer(index, "observable index") for index in indices)
        return self._mixed_statistic(slots, 1)

    def _native_storage(self) -> bool:
        """True for the native float64 ndarray storage the certificate models."""
        return (
            type(self._matrices) is np.ndarray
            and type(self._endpoints) is np.ndarray
            and self._matrices.dtype == np.dtype(np.float64)
            and self._endpoints.dtype == np.dtype(np.float64)
        )

    def _mixed_statistic(self, slots: tuple[int, ...], divisor: int) -> float:
        order = len(slots)
        if not 1 <= order <= 4:
            raise ValueError("mixed cumulants support orders one through four")
        if max(slots) >= self.observable_count:
            raise ValueError("observable index is out of range")
        if order == 1:
            return _finite_signed_sum(
                itertools_chain(
                    (self._constants[slots[0]],), np.diag(self._matrices[slots[0]])
                ),
                "mixed mean",
            )
        name = "mixed coefficient" if divisor > 1 else "mixed cumulant"
        # Bound, from the absolute matrices, the rounding error of the float64
        # contraction (see _mixed_contraction). A proved zero needs no signed
        # contraction.
        certificate = rounding_certificate(
            self._matrices, self._endpoints, slots, divisor
        )
        if certificate.zero:
            return 0.0
        native = self._native_storage()
        raised = False
        try:
            value = self._mixed_float_statistic(slots, divisor, native, "raise")
        except FloatingPointError:
            # Overflow and invalid operations raise again; an underflow alone
            # is inside the certificate's allowance.
            raised = True
            try:
                value = self._mixed_float_statistic(slots, divisor, native, "ignore")
            except FloatingPointError:
                value = None
        admitted = native and value is not None and admits(certificate, value)
        # Keep the float64 value only where 0.7.0 kept it as well (no
        # floating-point exception, no Frobenius cancellation flag) and the
        # certificate admits it; everything else is correctly rounded.
        if (
            admitted
            and value is not None
            and not raised
            and not _frobenius_cancellation(
                self._matrices, self._endpoints, slots, divisor, value
            )
        ):
            return _finite_fraction(value, name)
        try:
            return _exact_mixed_coefficient(
                self._matrices,
                self._endpoints,
                slots,
                divisor,
                _magnitude_estimate(value, certificate),
            )
        except ArithmeticError as error:
            # Above the exact work or memory bound (a plain ArithmeticError) a
            # certified float64 value is still returned. Out-of-range exact
            # values raise OverflowError; a floating-point exception or any
            # other ArithmeticError subclass is not a refusal and propagates.
            if type(error) is not ArithmeticError or not admitted or value is None:
                raise
            return _finite_fraction(value, name)

    def _mixed_float_statistic(
        self, slots: tuple[int, ...], divisor: int, native: bool, underflow: str
    ) -> Fraction:
        """Float64 contraction of all slot permutations with exact final sums.

        Raises ``FloatingPointError`` on overflow, an invalid operation, a
        nonfinite product, or (with ``underflow="raise"``) an underflow.
        """
        order = len(slots)
        ring_terms: list[float] = []
        chain_terms: list[float] = []
        # Equal ordered words perform the same native contractions. Keep the
        # original slot permutation expansion, including every scalar entry:
        # weighted sums or cyclic/reversed words would change floating-point
        # operation order and therefore the admitted float64 bits.
        distinct_slots = len(set(slots))
        repeated = distinct_slots != order
        word_cache: (
            dict[tuple[int, ...], tuple[tuple[float, ...], tuple[float, ...]]] | None
        ) = ({} if repeated else None)
        prefix_enabled = False
        if (
            order == 4
            and distinct_slots >= _MIXED_PREFIX_MIN_DISTINCT
            and type(self) is GaussianQuadraticFamily
            and native
        ):
            rank = self._matrices.shape[-1]
            prefix_enabled = (
                rank >= _MIXED_PREFIX_MIN_RANK
                and rank * rank * 8 <= _MIXED_PREFIX_MAX_BYTES
            )
        prefix_cache: tuple[tuple[int, int], RealArray] | None = None
        with np.errstate(over="raise", invalid="raise", under=underflow):
            for permutation in permutations(slots):
                if word_cache is not None and permutation in word_cache:
                    diagonal, endpoint_terms = word_cache[permutation]
                    ring_terms.extend(diagonal)
                    chain_terms.extend(endpoint_terms)
                    continue
                if prefix_enabled:
                    pair = (permutation[0], permutation[1])
                    if prefix_cache is not None and prefix_cache[0] == pair:
                        product = prefix_cache[1]
                    else:
                        product = self._matrices[pair[0]]
                        prefix_cache = None
                        product = product @ self._matrices[pair[1]]
                        prefix_cache = (pair, product)
                    for index in permutation[2:]:
                        product = product @ self._matrices[index]
                else:
                    product = self._matrices[permutation[0]]
                    for index in permutation[1:]:
                        product = product @ self._matrices[index]
                vector = self._endpoints[permutation[-1]]
                for index in reversed(permutation[1:-1]):
                    vector = self._matrices[index] @ vector
                terms = self._endpoints[permutation[0]] * vector
                if not (np.all(np.isfinite(product)) and np.all(np.isfinite(terms))):
                    raise FloatingPointError("nonfinite mixed contraction")
                if word_cache is None:
                    ring_terms.extend(np.diag(product))
                    chain_terms.extend(terms)
                else:
                    # Copies of scalar values do not retain a diagonal
                    # view of an O(r**2) dense product in the local cache.
                    diagonal = tuple(float(value) for value in np.diag(product))
                    endpoint_terms = tuple(float(value) for value in terms)
                    word_cache[permutation] = (diagonal, endpoint_terms)
                    ring_terms.extend(diagonal)
                    chain_terms.extend(endpoint_terms)
        prefix_cache = None
        ring, chain = _exact_signed_sum(ring_terms), _exact_signed_sum(chain_terms)
        return (
            Fraction(2 ** (order - 1), order) * ring
            + Fraction(2) ** (order - 3) * chain
        ) / divisor

    def joint_coefficient(self, powers: Sequence[int]) -> float:
        """Coefficient of prod(J[a]**powers[a]) in K(J), total degree <= 4.

        Certified or correctly rounded exactly as ``mixed_cumulant``.
        """
        powers = tuple(_integer(power, "power") for power in powers)
        if len(powers) != self.observable_count or sum(powers) > 4:
            raise ValueError(
                "powers must match the observable count and total degree must be at most four"
            )
        if not sum(powers):
            return 0.0
        indices = tuple(
            index for index, power in enumerate(powers) for _ in range(power)
        )
        return self._mixed_statistic(
            indices, math.prod(math.factorial(power) for power in powers)
        )

    def prepare_statistics(
        self,
        *,
        indices: Sequence[Sequence[int]] = (),
        coefficients: Sequence[Sequence[int]] = (),
        max_entries: int = _MAX_PREPARED_STATISTICS,
    ) -> GaussianQuadraticStatistics:
        """Prepare only explicitly selected existing statistic queries.

        Ordered mixed-cumulant tuples and Taylor-power tuples are separate
        keys. Every unique query invokes its existing public method once;
        coefficients are never reconstructed from rounded cumulants. Numeric
        ``OverflowError`` and ``ArithmeticError`` outcomes are recorded and
        re-raised on lookup with their original type and message. Invalid
        queries fail before any contraction. Unselected queries raise
        ``KeyError`` rather than triggering additional work.

        ``max_entries`` lies in [0, 1024], after exact-key deduplication. Input
        collections and their rows must be finite sequences whose actual
        lengths match their declarations. A 65,536-query submission limit and
        separate 65,536-integer key budget bound preparation and key storage.
        The immutable result retains scalar values and outcomes, not this
        family's matrices or endpoints; preparation time must be included for
        one-shot queries.
        Subclasses are refused because overridden queries need not describe
        the immutable native family or have deterministic outcomes.
        """
        if type(self) is not GaussianQuadraticFamily:
            raise TypeError(
                "statistic preparation requires the native GaussianQuadraticFamily"
            )
        max_entries = _integer(max_entries, "max_entries")
        if max_entries > _MAX_PREPARED_STATISTICS:
            raise ValueError("max_entries must be at most 1024")
        lengths = (
            _finite_sequence_length(indices, "indices"),
            _finite_sequence_length(coefficients, "coefficients"),
        )
        if sum(lengths) > _MAX_SUBMITTED_QUERIES:
            raise ValueError("selected statistics exceed the submitted query budget")
        keys: list[_StatisticKey] = []
        seen: set[_StatisticKey] = set()
        key_items = 0
        for kind, queries in ((0, indices), (1, coefficients)):
            for query in _bounded_sequence_items(
                queries, "indices" if kind == 0 else "coefficients", lengths[kind]
            ):
                key = (
                    kind,
                    _statistic_query_key(query, kind, self.observable_count),
                )
                if key in seen:
                    continue
                seen.add(key)
                keys.append(key)
                key_items += len(key[1])
                if len(keys) > max_entries:
                    raise ValueError("selected statistics exceed max_entries")
                if key_items > _MAX_PREPARED_KEY_ITEMS:
                    raise ValueError(
                        "selected statistics exceed the key storage budget"
                    )
        values: list[float] = []
        outcomes: list[_StatisticOutcome] = []
        for kind, query in keys:
            try:
                value = (
                    self.mixed_cumulant(query)
                    if kind == 0
                    else self.joint_coefficient(query)
                )
            except ArithmeticError as error:
                # Only the two explicit numeric admission outcomes are part
                # of this API. A subclass or unrelated error is not cached.
                if type(error) not in (OverflowError, ArithmeticError):
                    raise
                if len(error.args) != 1 or not isinstance(error.args[0], str):
                    raise
                outcomes.append((type(error), error.args[0]))
            else:
                if not math.isfinite(value):
                    raise ArithmeticError(
                        "selected statistic returned a nonfinite value"
                    )
                outcomes.append(len(values))
                values.append(value)
        ordered = sorted(zip(keys, outcomes))
        return GaussianQuadraticStatistics._from_prepared(
            self.observable_count,
            tuple(key for key, _ in ordered),
            tuple(outcome for _, outcome in ordered),
            np.array(values, dtype=np.float64),
        )


def _finite_sequence_length(value, name: str) -> int:
    try:
        return len(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a finite sequence") from error


def _bounded_sequence_items(value, name: str, length: int) -> Iterator[Any]:
    """Read the declared finite length, then check at most one extra item."""
    iterator = iter(value)
    for _ in range(length):
        try:
            item = next(iterator)
        except StopIteration as error:
            raise ValueError(
                f"{name} length differs from its declared sequence length"
            ) from error
        yield item
    try:
        next(iterator)
    except StopIteration:
        return
    raise ValueError(f"{name} length differs from its declared sequence length")


def _statistic_query_key(
    query: Sequence[int], kind: int, observable_count: int
) -> tuple[int, ...]:
    length = _finite_sequence_length(query, "statistic query")
    if length > _MAX_PREPARED_KEY_ITEMS:
        raise ValueError("selected statistics exceed the key storage budget")
    if kind == 0:
        if not 1 <= length <= 4:
            raise ValueError("mixed cumulants support orders one through four")
        key = tuple(
            _integer(index, "observable index")
            for index in _bounded_sequence_items(query, "statistic query", length)
        )
        if max(key) >= observable_count:
            raise ValueError("observable index is out of range")
    else:
        if length != observable_count:
            raise ValueError(
                "powers must match the observable count and total degree must be at most four"
            )
        key = tuple(
            _integer(power, "power")
            for power in _bounded_sequence_items(query, "statistic query", length)
        )
        if sum(key) > 4:
            raise ValueError(
                "powers must match the observable count and total degree must be at most four"
            )
    return key


@dataclass(frozen=True, slots=True, init=False, eq=False)
class GaussianQuadraticStatistics:
    """Immutable selected mixed cumulants and joint Taylor coefficients.

    Obtain an instance with ``GaussianQuadraticFamily.prepare_statistics``.
    It preserves the existing individual queries, including their order and
    rounded float64 results. It supplies no directional polynomial or CGF
    reconstruction and retains no parent family or dense contractions.
    """

    _observable_count: int
    _keys: tuple[_StatisticKey, ...]
    _outcomes: tuple[_StatisticOutcome, ...]
    _values: RealArray

    def __init__(self) -> None:
        raise TypeError("use GaussianQuadraticFamily.prepare_statistics")

    @classmethod
    def _from_prepared(
        cls,
        observable_count: int,
        keys: tuple[_StatisticKey, ...],
        outcomes: tuple[_StatisticOutcome, ...],
        values: RealArray,
    ) -> GaussianQuadraticStatistics:
        result = object.__new__(cls)
        object.__setattr__(result, "_observable_count", observable_count)
        object.__setattr__(result, "_keys", tuple(keys))
        object.__setattr__(result, "_outcomes", tuple(outcomes))
        object.__setattr__(result, "_values", _snapshot(values))
        return result

    def __copy__(self) -> GaussianQuadraticStatistics:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> GaussianQuadraticStatistics:
        memo[id(self)] = self
        return self

    def __reduce__(self):
        return (
            type(self)._from_prepared,
            (self._observable_count, self._keys, self._outcomes, self._values),
        )

    @property
    def observable_count(self) -> int:
        return self._observable_count

    @property
    def entry_count(self) -> int:
        return len(self._keys)

    @property
    def indices(self) -> tuple[tuple[int, ...], ...]:
        """Prepared ordered mixed-cumulant keys, in lexicographic order."""
        return tuple(key for kind, key in self._keys if kind == 0)

    @property
    def coefficients(self) -> tuple[tuple[int, ...], ...]:
        """Prepared Taylor-power keys, in lexicographic order."""
        return tuple(key for kind, key in self._keys if kind == 1)

    def _lookup(self, query: Sequence[int], kind: int) -> float:
        key = (kind, _statistic_query_key(query, kind, self.observable_count))
        index = bisect_left(self._keys, key)
        if index == len(self._keys) or self._keys[index] != key:
            raise KeyError(f"statistic query was not prepared: {key}")
        outcome = self._outcomes[index]
        if isinstance(outcome, int):
            return float(self._values[outcome])
        error_type, message = outcome
        raise error_type(message)

    def mixed_cumulant(self, indices: Sequence[int]) -> float:
        """Return one selected original ordered mixed-cumulant query."""
        return self._lookup(indices, 0)

    def joint_coefficient(self, powers: Sequence[int]) -> float:
        """Return one selected original Taylor-power query."""
        return self._lookup(powers, 1)
