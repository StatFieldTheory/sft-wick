"""Campbell functionals of finite Poisson event measures."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Context, Decimal, localcontext
from functools import lru_cache
from fractions import Fraction
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

from ._util import require_bool
from .quadratic import _real_array, _snapshot

__all__ = ["CampbellDriver", "ExponentialMark", "UnitMark"]


def _sum_products(weights, columns, moment):
    """Sum weight times one mark moment times selected response columns."""
    values = np.broadcast_to(moment, weights.shape)
    try:
        with np.errstate(under="raise", over="raise", invalid="raise"):
            terms = weights * values
            for column in columns:
                terms *= column
            result = np.sum(terms)
        if np.isfinite(result):
            return result
    except (FloatingPointError, OverflowError, TypeError):
        pass

    # Binary input values are exact rationals. In the uncommon extreme
    # case combine the rate, mark moment and all legs before rounding.
    def exact(parts):
        total = Fraction(0)
        for i, part in enumerate(parts):
            term = Fraction(weights[i]) * Fraction(part)
            for column in columns:
                term *= Fraction(column[i])
            total += term
        return float(total)

    if np.iscomplexobj(values):
        return complex(exact(values.real), exact(values.imag))
    return exact(values)


@lru_cache(maxsize=3)
def _two_pi(precision):
    # Gauss--Legendre doubles the number of correct digits each iteration.
    with localcontext(Context(prec=precision + 12)):
        a = Decimal(1)
        b = (a / 2).sqrt()
        t = a / 4
        multiplier = 1
        for _ in range(precision.bit_length() + 1):
            mid = (a + b) / 2
            t -= multiplier * (a - mid) ** 2
            a, b = mid, (a * b).sqrt()
            multiplier *= 2
        return (a + b) ** 2 / (2 * t)


def _exact_rotation(angle):
    """Rotate by an exact binary angle, retaining the low bits of a product."""
    try:
        value = float(angle)
    except OverflowError:
        # A characteristic function may exist beyond the range of its
        # unwrapped log phase. Reduce the angle with enough digits for it.
        bits = angle.numerator.bit_length() - angle.denominator.bit_length()
        precision = 1 << (int(max(bits, 0) * math.log10(2)) + 40).bit_length()
        with localcontext(Context(prec=precision)):
            value = float(
                (Decimal(angle.numerator) / Decimal(angle.denominator))
                % _two_pi(precision)
            )
        return np.exp(1j * value)
    result = np.exp(1j * value)
    residual = angle - Fraction(value)
    while residual:
        piece = float(residual)
        if piece == 0:
            break
        result *= np.exp(1j * piece)
        residual -= Fraction(piece)
    return result


def _logabs_phase(value):
    """Signed/complex logarithmic magnitude with an exact zero phase at zero."""
    magnitude = np.abs(value)
    with np.errstate(divide="ignore", invalid="ignore"):
        phase = np.divide(
            value,
            magnitude,
            out=np.zeros_like(value, dtype=np.result_type(value, float)),
            where=magnitude != 0,
        )
        return np.log(magnitude), phase


def _log_quadratic(logweight, columns, first, second):
    """Combine the signed same-event and distinct-event sums in log magnitude."""
    from scipy.special import logsumexp

    (yl, ys), (zl, zs) = columns
    m1, s1 = first
    m2, s2 = second
    pair, sp = logsumexp(
        logweight + yl + zl + m2, b=ys * zs * s2, axis=-1, return_sign=True
    )
    y, sy = logsumexp(logweight + yl + m1, b=ys * s1, axis=-1, return_sign=True)
    z, sz = logsumexp(logweight + zl + m1, b=zs * s1, axis=-1, return_sign=True)
    combined, sign = logsumexp(
        np.stack((pair, y + z)), b=np.stack((sp, sy * sz)), axis=0, return_sign=True
    )
    return combined, sign, np.maximum(pair, y + z)


def _restore(logabs, phase):
    with np.errstate(over="ignore", invalid="ignore"):
        value = phase * np.exp(logabs)
    if not np.isfinite(value):
        raise ArithmeticError("Campbell readout exceeds numerical range")
    return value


class UnitMark:
    """A deterministic unit mark. General discrete marks fit in the event rows."""

    nonnegative = True

    @staticmethod
    def increment(z, *, compensated):
        raw = np.expm1(z)
        if not compensated:
            return raw
        # Preserve the leading quadratic even when expm1(z) rounds to z.
        result = raw - z
        small = np.abs(z) < 0.01
        t = z[small]
        result[small] = (
            t
            * t
            * (
                1 / 2
                + t * (1 / 6 + t * (1 / 24 + t * (1 / 120 + t * (1 / 720 + t / 5040))))
            )
        )
        return result

    @staticmethod
    def tilted_moment(order, z):
        return np.exp(z)

    @staticmethod
    def tilted_first_increment(z):
        return np.expm1(z)


class ExponentialMark:
    """A shared nonnegative exponential mark of mean one for each event."""

    nonnegative = True

    @staticmethod
    def increment(z, *, compensated):
        raw = z / (1.0 - z)
        return raw * z if compensated else raw

    @staticmethod
    def tilted_moment(order, z):
        return math.factorial(order) * (1.0 / (1.0 - z)) ** (order + 1)

    @staticmethod
    def tilted_first_increment(z):
        raw = z / (1.0 - z)
        return raw * (1.0 + 1.0 / (1.0 - z))


@dataclass(frozen=True, eq=False, init=False)
class CampbellDriver:
    """Joint law of linear responses to independent marked Poisson events.

    Row e has expected event count weights[e] and a vector responses[e] of
    filtered amplitudes per unit mark. Each individual event contributes the
    whole row times a single iid mark. Continuous event position, time and
    mark laws can be supplied as weighted quadrature rows. Compensation
    subtracts the deterministic mean of every component.
    """

    weights: np.ndarray
    responses: np.ndarray
    mark: object
    compensated: bool
    _wide: bool

    def __init__(
        self, weights: ArrayLike, responses: ArrayLike, *, mark=None, compensated=False
    ):
        weights = _real_array(weights, "weights", ndim=1)
        responses = _real_array(responses, "responses", ndim=2)
        if len(weights) != len(responses) or responses.shape[1] == 0:
            raise ValueError("responses must give one nonempty vector per event weight")
        if np.any(weights < 0):
            raise ValueError("Poisson event weights must be nonnegative")
        keep = weights > 0
        mark = UnitMark() if mark is None else mark
        if not callable(getattr(mark, "increment", None)):
            raise TypeError("mark must supply increment(z, *, compensated)")
        object.__setattr__(self, "weights", _snapshot(weights[keep]))
        object.__setattr__(self, "responses", _snapshot(responses[keep]))
        object.__setattr__(self, "mark", mark)
        object.__setattr__(
            self,
            "compensated",
            require_bool(compensated, "compensated", "CampbellDriver"),
        )

        # Four physical factors can meet before a third is divided out by
        # the Laplace integration. Reserve range for their common product.
        with np.errstate(divide="ignore"):
            extreme = any(
                np.any(np.abs(np.log(np.abs(a[a != 0]))) > 150)
                for a in (self.weights, self.responses)
            )
        object.__setattr__(self, "_wide", extreme)

    def _component(self, i):
        if isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral):
            raise ValueError("component must be an integer index")
        if not 0 <= i < self.responses.shape[1]:
            raise ValueError("component index outside the response vector")
        return int(i)

    def _argument(self, frequency):
        k = _real_array(frequency, "frequency")
        if k.ndim == 0 and self.responses.shape[1] == 1:
            k = k.reshape(1)
        if k.shape != (self.responses.shape[1],):
            raise ValueError("frequency must give one value per component")
        with np.errstate(over="ignore", invalid="ignore"):
            angles = self.responses @ k
        if (
            type(self.mark) not in (UnitMark, ExponentialMark)
            and not np.isfinite(angles).all()
        ):
            raise ArithmeticError(
                "custom mark Fourier argument exceeds numerical range"
            )
        z = np.zeros(len(angles), dtype=complex)
        z.imag = angles
        return z

    def _exponent(self, z, weights=None, compensated=None):
        weights = self.weights if weights is None else weights
        compensated = self.compensated if compensated is None else compensated
        # The per-event quadratic term can underflow before a large event
        # rate multiplies it. Combine the weight with one small factor first.
        if type(self.mark) in (UnitMark, ExponentialMark):
            tiny = (z != 0) & (np.abs(z) < 1e-150)
        else:
            tiny = np.zeros(len(z), dtype=bool)
        if np.any(tiny):
            small = z[tiny]
            weighted = weights[tiny] * small
            divisor = 2 if type(self.mark) is UnitMark else 1
            value = np.sum(weighted * small / divisor)
            if not compensated:
                value += np.sum(weighted)
            value += weights[~tiny] @ self.mark.increment(
                z[~tiny], compensated=compensated
            )
        else:
            increment = np.broadcast_to(
                self.mark.increment(z, compensated=compensated), z.shape
            )
            value = weights @ increment
        if not np.isfinite(value):
            raise ArithmeticError("Campbell exponent exceeds numerical range")
        if not np.iscomplexobj(z):
            if np.imag(value) != 0:
                raise ValueError(
                    "a real Laplace argument requires a real mark transform"
                )
            return np.real(value)
        return value

    def _fourier_factor(self, frequency, z):
        # A large projected angle or deterministically weighted phase must
        # retain the low binary bits of its products. Expand near zero in
        # compensated form; at large angles integrate the uncentred mark
        # transform before applying the deterministic centering rotation.
        k = np.atleast_1d(frequency)
        with np.errstate(over="ignore", invalid="ignore"):
            scale = np.abs(self.responses) @ np.abs(k)
            risky = (scale > 1e3) | (self.weights * scale > 1e3)
        if type(self.mark) not in (UnitMark, ExponentialMark) or not np.any(risky):
            exponent = self._exponent(z)
            return exponent, Fraction(0), None, {}
        exact_k = tuple(Fraction(float(x)) for x in k)
        rest = self._exponent(z[~risky], self.weights[~risky])
        center = Fraction(0)
        log_angles = {}
        increments = (
            np.expm1(z, out=np.zeros_like(z), where=np.isfinite(z.imag))
            if type(self.mark) is UnitMark
            else None
        )
        for i in np.flatnonzero(risky):
            theta = sum(Fraction(a) * b for a, b in zip(self.responses[i], exact_k))
            weight = self.weights[i]
            try:
                z[i] = complex(0, float(theta))
            except OverflowError:
                sign = 1 if theta > 0 else -1
                z[i] = complex(0.0, sign * np.inf)
                log_angles[i] = (
                    math.log(abs(theta.numerator)) - math.log(theta.denominator),
                    sign,
                )
                if increments is None:
                    # iθ/(1-iθ) = -1 + i/θ + O(θ^-2). Combine the weight
                    # before reciprocation to retain a subnormal imaginary part.
                    rest += complex(-weight, float(Fraction(weight) / theta))
                    if self.compensated:
                        center -= Fraction(weight) * theta
                    continue
            if abs(theta) < 0.25:
                rest += self._exponent(z[i : i + 1], self.weights[i : i + 1], True)
                if not self.compensated:
                    center += Fraction(weight) * theta
                if increments is not None:
                    increments[i] = np.expm1(z[i])
            else:
                if increments is None:
                    increment = z[i] / (1 - z[i])
                else:
                    half = _exact_rotation(theta / 2)
                    increment = complex(-2 * half.imag**2, 2 * half.real * half.imag)
                    increments[i] = increment
                rest += weight * increment
                if self.compensated:
                    center -= Fraction(weight) * theta
        return rest, center, increments, log_angles

    def log_characteristic(self, frequency: ArrayLike) -> complex:
        """Return log E exp(i k.X); an unwrapped phase is rounded to binary64."""
        rest, center, _, _ = self._fourier_factor(frequency, self._argument(frequency))
        return complex(rest.real, float(Fraction(rest.imag) + center))

    def characteristic(self, frequency: ArrayLike) -> complex:
        """Return E exp(i k.X) including any zero-event atom."""
        rest, center, _, _ = self._fourier_factor(frequency, self._argument(frequency))
        return complex(
            _restore(rest.real, np.exp(1j * rest.imag) * _exact_rotation(center))
        )

    def fourier_expectation(
        self, frequencies: ArrayLike, coefficients: ArrayLike
    ) -> complex:
        """Average a Fourier sum sum_l c_l exp(i k_l.X), including its zero mode."""
        modes = _real_array(frequencies, "frequencies", ndim=2)
        coefficients = np.asarray(coefficients)
        if modes.shape[1] != self.responses.shape[1] or coefficients.shape != (
            len(modes),
        ):
            raise ValueError(
                "supply one coefficient per mode and one frequency per component"
            )
        if coefficients.dtype.kind not in "iufc" or not np.all(
            np.isfinite(coefficients)
        ):
            raise ValueError("Fourier coefficients must be finite numbers")
        return complex(
            sum(c * self.characteristic(k) for c, k in zip(coefficients, modes))
        )

    def _moment(self, order, z):
        moment = getattr(self.mark, "tilted_moment", None)
        if moment is None:
            raise ValueError("this readout requires the mark's tilted_moment(order, z)")
        values = np.broadcast_to(moment(order, z), z.shape)
        if not np.all(np.isfinite(values)):
            raise ValueError("the requested tilted mark moment must be finite")
        return values

    def mixed_cumulant(self, indices) -> float:
        """Connected joint cumulant for the specified component list."""
        indices = tuple(self._component(i) for i in indices)
        if not indices:
            raise ValueError("at least one component is required")
        if self.compensated and len(indices) == 1:
            # Check the mean exists for user-supplied laws.
            self._moment(1, np.zeros_like(self.weights))
            return 0.0
        order = len(indices)
        if type(self.mark) is ExponentialMark and order > 170:
            # The mark moment itself can be too large to convert to float.
            moment = math.factorial(order)
        else:
            moment = self._moment(order, np.zeros_like(self.weights))
        columns = tuple(self.responses[:, i] for i in indices)
        result = _sum_products(self.weights, columns, moment)
        return float(result)

    def _wide_fourier_quadratic(
        self, z, left, right, exponent, rotation, increments, log_angles
    ):
        if type(self.mark) is UnitMark:
            if increments is None:
                increments = np.expm1(z)
            second = (z.real, 1 + increments)
            first = _logabs_phase(increments) if self.compensated else second
        elif type(self.mark) is ExponentialMark:
            finite_z = np.where(np.isfinite(z.imag), z, 0.0)
            denominator = 1.0 - finite_z
            logden, phase = _logabs_phase(denominator)
            phase = phase.conj()
            for i, (log_angle, sign) in log_angles.items():
                # |1-iθ| is |θ| to binary64 precision. Keep its log even
                # when θ itself overflows and inverse powers underflow.
                logden[i], phase[i] = log_angle, complex(0.0, sign)
            shifted_first = self.mark.tilted_first_increment(finite_z)
            for i in log_angles:
                shifted_first[i] = -1
            second = (math.log(2.0) - 3 * logden, phase**3)
            first = (
                _logabs_phase(shifted_first)
                if self.compensated
                else (-2 * logden, phase**2)
            )
        else:
            first = _logabs_phase(self._tilted_first(z))
            second = _logabs_phase(self._moment(2, z))
        columns = tuple(_logabs_phase(self.responses[:, i]) for i in (left, right))
        logabs, phase, largest = _log_quadratic(
            np.log(self.weights), columns, first, second
        )
        if (
            np.isfinite(largest)
            and logabs < largest - 20
            and type(self.mark) is UnitMark
        ):
            return self._cancelled_fourier_quadratic(
                z, left, right, increments, exponent, rotation
            )
        return _restore(exponent + logabs, rotation * phase)

    def _cancelled_fourier_quadratic(
        self, z, left, right, increments, exponent, rotation
    ):
        # Log magnitudes lose the low bits when a connected contribution
        # nearly cancels a product of means, e.g. near a zero of the second
        # derivative of a Gaussian CF. Contract the binary moment values
        # and responses exactly before applying the common damping.
        increment = np.expm1(z) if increments is None else increments
        second = 1 + increment
        first = increment if self.compensated else second

        def terms(indices, moments):
            re, im = Fraction(0), Fraction(0)
            for weight, row, m in zip(self.weights, self.responses, moments):
                coefficient = Fraction(weight)
                for i in indices:
                    coefficient *= Fraction(row[i])
                re += coefficient * Fraction(float(np.real(m)))
                im += coefficient * Fraction(float(np.imag(m)))
            return re, im

        x, y = terms((left,), first)
        u, v = terms((right,), first)
        re, im = terms((left, right), second)
        re += x * u - y * v
        im += x * v + y * u
        scale = max(abs(re), abs(im))
        if not scale:
            return 0j
        power = scale.numerator.bit_length() - scale.denominator.bit_length()
        unit = Fraction(2) ** power
        return _restore(
            exponent + power * math.log(2),
            rotation * complex(float(re / unit), float(im / unit)),
        )

    def _tilted_first(self, z):
        if not self.compensated:
            return self._moment(1, z)
        shifted = getattr(self.mark, "tilted_first_increment", None)
        return (
            shifted(z)
            if shifted is not None
            else self._moment(1, z) - self._moment(1, np.zeros_like(z))
        )

    def _quadratic(self, z, left, right, frequency=None):
        # Twice differentiate the functional itself. Derivatives of log L
        # give the same-event term; the product of first derivatives gives
        # the contribution of two independent events.
        increments = None
        if frequency is None:
            exponent, rotation = self._exponent(z), 1.0
        else:
            rest, center, increments, log_angles = self._fourier_factor(frequency, z)
            exponent, rotation = rest.real, np.exp(1j * rest.imag) * _exact_rotation(
                center
            )
            if self._wide or exponent < -600 or np.max(np.abs(z), initial=0) > 1e75:
                return self._wide_fourier_quadratic(
                    z, left, right, exponent, rotation, increments, log_angles
                )
        if increments is None:
            first, second = self._tilted_first(z), self._moment(2, z)
        else:
            second = 1 + increments
            first = increments if self.compensated else second
        left_h, right_h = self.responses[:, left], self.responses[:, right]
        pair = _sum_products(self.weights, (left_h, right_h), second)
        left_first = _sum_products(self.weights, (left_h,), first)
        right_first = _sum_products(self.weights, (right_h,), first)
        result = rotation * np.exp(exponent) * (pair + left_first * right_first)
        if not np.isfinite(result):
            raise ArithmeticError("tilted quadratic moment exceeds numerical range")
        return result

    def quadratic_characteristic(self, frequency, left, right) -> complex:
        """Return E[X_left X_right exp(i k.X)] for a quadratic drift insertion."""
        z = self._argument(frequency)
        return complex(
            self._quadratic(z, self._component(left), self._component(right), frequency)
        )

    def _wide_positive_readout(self, h, pair):
        from scipy.special import logsumexp, roots_legendre

        with np.errstate(divide="ignore"):
            lh, lw = np.log(h), np.log(self.weights)
        # Both a single jump and the whole population set Laplace scales.
        # Keep log(q) as the integration variable even when q itself is
        # smaller than binary64. The quadratic kernel then includes q².
        lower = -40 - max(0.0, np.max(lh, initial=-np.inf), logsumexp(lw + lh))
        upper = math.log(4096.0)
        edges = np.linspace(lower, upper, math.ceil((upper - lower) / 4) + 1)
        if pair is not None:
            columns = tuple(_logabs_phase(self.responses[:, i]) for i in pair)

        def kernel(u):
            logarg = u[:, None] + lh
            with np.errstate(over="ignore", under="ignore", divide="ignore"):
                z = -np.exp(logarg)
                if type(self.mark) is UnitMark:
                    loss = np.where(logarg < -36, logarg, np.log(-np.expm1(z)))
                    first = second = (z, 1.0)
                elif type(self.mark) is ExponentialMark:
                    loss = -np.logaddexp(0.0, -logarg)
                    logden = np.logaddexp(0.0, logarg)
                    first = (-2 * logden, 1.0)
                    second = (math.log(2.0) - 3 * logden, 1.0)
                else:
                    if not np.isfinite(z).all():
                        raise ArithmeticError(
                            "custom mark Laplace argument exceeds numerical range"
                        )
                    loss = np.log(-self.mark.increment(z, compensated=False))
                    if pair is not None:
                        first = _logabs_phase(self._moment(1, z))
                        second = _logabs_phase(self._moment(2, z))
                logloss = logsumexp(lw + loss, axis=-1)
                exponent = -np.exp(logloss)
                q = np.exp(u)
                if pair is None:
                    readout = np.where(
                        logloss < -36, logloss, np.log(-np.expm1(exponent))
                    )
                    return u - q + readout, np.ones_like(u)
                moment, phase, _ = _log_quadratic(lw, columns, first, second)
                return 2 * u - q + exponent + moment, phase

        previous = None
        for order in (12, 24, 48, 96):
            nodes, weights = roots_legendre(order)
            widths = np.diff(edges) / 2
            u = (edges[:-1, None] + widths[:, None] * (nodes + 1)).ravel()
            rule = (widths[:, None] * weights).ravel()
            logs, signs = [], []
            batch = max(1, 32768 // max(1, len(h)))
            for i in range(0, len(u), batch):
                term, sign = kernel(u[i : i + batch])
                logs.append(term)
                signs.append(sign)
            logtotal, sign = logsumexp(
                np.concatenate(logs), b=rule * np.concatenate(signs), return_sign=True
            )
            value = float(_restore(logtotal, sign))
            if previous is not None and abs(value - previous) <= 2e-12 * abs(value):
                return value
            previous = value
        raise ArithmeticError("Campbell readout quadrature did not converge")

    def _positive_readout(self, output, integrand, pair=None):
        from scipy.integrate import quad

        h = self.responses[:, self._component(output)]
        if (
            self.compensated
            or not getattr(self.mark, "nonnegative", False)
            or np.any(h < 0)
        ):
            raise ValueError(
                "this readout requires an uncompensated nonnegative output"
            )
        if self._wide:
            return self._wide_positive_readout(h, pair)
        # The Laplace tilt begins to suppress an event when q*h is order
        # one. Resolve this part for profiles spanning many orders of magnitude.
        peak = np.max(h, initial=1.0)
        decades = max(0, math.ceil(math.log10(peak)))
        cuts = (
            np.geomspace(1 / peak, 1.0, 1 + math.ceil(decades / 2)) if decades else []
        )
        edges = list(cuts)
        # Many weak events can suppress the Laplace transform well before
        # even the strongest individual event is tilted. Locate that layer
        # using the whole event law, which does not require a finite mean.
        scale = 1.0 / peak
        while scale > np.finfo(float).tiny and self._exponent(-scale * h) < -1.0:
            edges.append(scale)
            scale *= 0.01
        edges = np.unique([0.0, scale, *edges, 1.0, np.inf])
        value = 0.0
        for a, b in zip(edges, edges[1:]):
            integral, _, _, *failure = quad(
                lambda q: integrand(q, -q * h) * math.exp(-q),
                a,
                b,
                epsabs=2e-13 / (len(edges) - 1),
                epsrel=2e-12,
                full_output=True,
            )
            if failure or not math.isfinite(integral):
                raise ArithmeticError("Campbell readout quadrature did not converge")
            value += integral
        return value

    def saturating_mean(self, output=0) -> float:
        """Return E[X/(1+X)] for a nonnegative output X."""
        return self._positive_readout(output, lambda q, z: -np.expm1(self._exponent(z)))

    def saturating_quadratic_response(self, left, right, *, output=0) -> float:
        """Return E[X_left X_right/(1+X_output)^2] for a nonnegative output.

        Integrate this coefficient with the retarded response and quadratic
        coupling over the insertion time to obtain the first drift correction.
        """
        left, right = self._component(left), self._component(right)
        return self._positive_readout(
            output, lambda q, z: q * self._quadratic(z, left, right), (left, right)
        )
