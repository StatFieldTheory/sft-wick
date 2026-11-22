"""Finite-start temporal covariances for stationary colored-noise kernels.

For times measured from the initial time, the functions evaluate

    integral_0^t1 ds1 integral_0^t2 ds2
      exp(-gamma_a * (t1 - s1) - gamma_b * (t2 - s2)) K(s1 - s2).

The result excludes the overall amplitude and spatial factor. Gamma rates
are nonnegative. All arguments broadcast, and an empty integration interval
gives zero. The Gaussian uses a stationary erfcx primitive with a positive
finite-lag integral when stationary subtraction is ill-conditioned.
"""

from __future__ import annotations

from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import erfc, erfcx

from ._native_protocol import capture_native_protocol as _capture_native_protocol


def _unit_nodes(order):
    nodes, weights = leggauss(order)
    return (1 + nodes) / 2, weights / 2


_SHORT_NODES, _SHORT_WEIGHTS = _unit_nodes(8)
_GAUSS_NODES, _GAUSS_WEIGHTS = _unit_nodes(32)
_PSI1_TAYLOR = tuple((-1.) ** n / (factorial(n) * (n + 2)) for n in range(19))
_GAUSSIAN_CANCELLATION = 64.
_GAUSSIAN_SIGMA_CUTOFF = 10.
_GAUSSIAN_EXP_CUTOFF = 45.
_GAUSSIAN_BATCH = 8192
_CONSTANT_GAUSSIAN = np.sqrt(np.finfo(float).eps)
_TIME_SQUARE_LIMIT = np.sqrt(np.finfo(float).max) / 2
_NARROW_DECAY = float(2**55)


def _psi(x):
    """Integral of exp(-x z) for z in [0, 1], for Re(x) >= 0."""
    return -np.expm1(-x) / np.where(x == 0, 1., x) + (x == 0)


def _rate_integral(rate, span):
    """Integral of exp(-rate*s) from 0 to span for a nonnegative real rate."""
    with np.errstate(over="ignore"):
        scaled = rate * span
    return np.divide(
        -np.expm1(-scaled), rate, where=scaled >= np.finfo(float).tiny,
        out=np.array(span, copy=True),
    )


def _narrow_mask(ga, gb, earlier, a):
    # Both the finite-interval edge correction and decay over one lag
    # width are below 2**-55, before factors of order unity.
    safe_a = np.where(a > 0.0, a, 1.0)
    with np.errstate(over="ignore"):
        return ((a > _NARROW_DECAY / earlier)
                & ((ga / safe_a + gb / safe_a) < 1.0 / _NARROW_DECAY))


def _white_overlap(ga, gb, t1, t2):
    earlier = np.minimum(t1, t2)
    with np.errstate(over="ignore"):
        decay = np.exp(-ga * (t1 - earlier) - gb * (t2 - earlier))
    return decay * _rate_integral(ga + gb, earlier)


def _psi1(x):
    """Integral of z exp(-x z) for z in [0, 1]."""
    short = np.abs(x) < 1
    out = np.empty_like(x)
    regular = x[~short]
    out[~short] = (_psi(regular) - np.exp(-regular)) / regular
    if np.any(short):
        z = x[short]
        value = _PSI1_TAYLOR[-1]
        for coefficient in reversed(_PSI1_TAYLOR[:-1]):
            value = value * z + coefficient
        out[short] = value
    return out


def _lag_interval(left, right, first_moment):
    """Integrals of (1, z) exp(-left (1-z) - right z) on [0, 1]."""
    reverse = np.real(left) > np.real(right)
    least = np.where(reverse, right, left)
    delta = np.where(reverse, left - right, right - left)
    norm = np.exp(-least)
    mean = norm * _psi(delta)
    if not first_moment:
        return mean, None
    moment = norm * _psi1(delta)
    return mean, np.where(reverse, mean - moment, moment)


def _lag_triangle(x, y, first_moment):
    """Integrals of (1, z) exp(-x z - y v) over z + v <= 1."""
    q0 = np.empty_like(x)
    q1 = np.empty_like(x) if first_moment else None
    # Integrate in v first when its rate y avoids cancellation.
    normal = y >= 0.5
    x0, y0 = x[normal], y[normal]
    b0, b1 = _lag_interval(y0, x0, first_moment)
    q0[normal] = (_psi(x0) - b0) / y0
    if first_moment:
        q1[normal] = (_psi1(x0) - b1) / y0
    # When only y is small, integrating in z first gives another form.
    reverse = ~normal & (np.abs(x) >= 1)
    x1, y1 = x[reverse], y[reverse]
    b0, b1 = _lag_interval(y1, x1, first_moment)
    q0[reverse] = (_psi(y1) - b0) / x1
    if first_moment:
        q1[reverse] = (q0[reverse] - b1) / x1
    # Both rates small. The integrand is analytic on the unit panel and
    # positive for a real decay. GL8 resolves it at binary64 precision.
    short = ~normal & ~reverse
    if np.any(short):
        nodes = _SHORT_NODES
        remaining = 1. - nodes
        values = (np.exp(-x[short, None] * nodes) * remaining
                  * _psi(y[short, None] * remaining))
        q0[short] = values @ _SHORT_WEIGHTS
        if first_moment:
            q1[short] = values @ (_SHORT_WEIGHTS * nodes)
    return q0, q1


def _exponential_positive_lag(ga, gb, t1, t2, decay, polynomial):
    """The part with s1 >= s2, for K(z) = (1 + polynomial*z) exp(-decay*z)."""
    u = np.minimum(t1, t2)
    d = np.maximum(t1 - t2, 0.)
    other = np.maximum(t2 - t1, 0.)
    sum_rate = ga + gb
    # Integrating over the earlier time gives a rectangle for lags below
    # d and a triangle for the remaining lags of length u.
    b0, b1 = _lag_interval(ga * d, decay * d, polynomial is not None)
    if polynomial is not None:
        b0 = b0 + polynomial * d * b1
    a = gb + decay
    with np.errstate(over="ignore"):
        y = sum_rate * u
        # Truncating each real axis beyond half the duration bounds the
        # omitted triangle tail. Extend that margin as oscillation grows.
        tail = 128.0 + 2.0 * np.log1p(abs(a.imag) / np.maximum(a.real, np.finfo(float).tiny))
        steady = (u > _TIME_SQUARE_LIMIT) & (y > tail) & (a.real * u > tail)
    integrated_rate = _rate_integral(sum_rate, u)
    strip = d * b0 * integrated_rate

    triangle = np.empty_like(a)
    regular = ~steady
    ur, dr = u[regular], d[regular]
    ar, yr = a[regular], y[regular]
    q0, q1 = _lag_triangle(ar * ur, yr, polynomial is not None)
    if polynomial is not None:
        pr = polynomial[regular]
        q0 = (1.0 + pr * dr) * q0 + pr * ur * q1
    decay_term = np.exp(-gb * other - decay * d)
    triangle[regular] = ur * ((ur * q0) * decay_term[regular])

    steady_a = a[steady]
    steady_base = (1.0 / steady_a) / sum_rate[steady]
    if polynomial is not None:
        with np.errstate(over="ignore"):
            lag_scaled = polynomial[steady] * d[steady]
        lag_prefactor = lag_scaled * np.exp(-np.where(np.isfinite(lag_scaled), lag_scaled, 0.0))
        lag_prefactor[~np.isfinite(lag_scaled)] = 0.0
        triangle[steady] = steady_base * (
            decay_term[steady] * (1.0 + polynomial[steady] / steady_a)
            + np.exp(-gb[steady] * other[steady]) * lag_prefactor
        )
    else:
        triangle[steady] = steady_base * decay_term[steady]
    return strip + triangle


def _exponential_family(ga, gb, t1, t2, decay, polynomial):
    # The two lag halves share only a zero-measure diagonal.
    pos = _exponential_positive_lag(ga, gb, t1, t2, decay, polynomial)
    neg = _exponential_positive_lag(gb, ga, t2, t1, decay, polynomial)
    return np.real(pos + neg)


def _apply(fn, t1, t2, *args):
    t1, t2, *args = np.broadcast_arrays(
        np.asarray(t1, dtype=float), np.asarray(t2, dtype=float),
        *(np.asarray(v, dtype=float) for v in args),
    )
    active = (t1 > 0.) & (t2 > 0.)
    out = np.zeros_like(t1)
    if np.any(active):
        out[active] = fn(t1[active], t2[active], *(v[active] for v in args))
    return out


def matern32_phi(gamma_a, gamma_b, a, t1, t2):
    """Normalized Matérn-3/2 K(z) = (1 + a |z|) exp(-a |z|)."""
    def temporal(t1, t2, ga, gb, a):
        narrow = _narrow_mask(ga, gb, np.minimum(t1, t2), a)
        out = np.empty_like(t1)
        normal = ~narrow
        out[normal] = _exponential_family(
            ga[normal], gb[normal], t1[normal], t2[normal], a[normal], a[normal],
        )
        out[narrow] = (4.0 / a[narrow]) * _white_overlap(
            ga[narrow], gb[narrow], t1[narrow], t2[narrow],
        )
        return out

    return _apply(temporal, t1, t2, gamma_a, gamma_b, a)


def damped_cosine_phi(gamma_a, gamma_b, a, omega, t1, t2):
    """Normalized K(z) = exp(-a |z|) cos(omega z)."""
    def temporal(t1, t2, ga, gb, a, omega):
        narrow = _narrow_mask(ga, gb, np.minimum(t1, t2), a) & (abs(omega) <= a)
        out = np.empty_like(t1)
        normal = ~narrow
        out[normal] = _exponential_family(
            ga[normal], gb[normal], t1[normal], t2[normal],
            a[normal] - 1j * omega[normal], None,
        )
        an = a[narrow]
        area = (2.0 / an) / (1.0 + (omega[narrow] / an)**2)
        out[narrow] = area * _white_overlap(ga[narrow], gb[narrow], t1[narrow], t2[narrow])
        return out

    return _apply(temporal, t1, t2, gamma_a, gamma_b, a, omega)


def _gaussian_laplace(gamma, sigma, tau):
    """Gaussian convolved with exp(-gamma*w) on 0 <= w < infinity."""
    with np.errstate(over="ignore"):
        z = tau / sigma
        q = gamma * sigma
    finite = np.isfinite(q)
    x = np.zeros_like(z)
    x[finite] = (q[finite] - z[finite]) / np.sqrt(2.)
    upper = finite & (x >= 0.)
    result = np.empty_like(z)
    # Mask each branch. erfcx of a very negative argument overflows.
    with np.errstate(over="ignore"):
        gaussian = np.exp(-0.5 * z[upper] ** 2)
        result[upper] = sigma[upper] * (np.sqrt(np.pi / 2) * gaussian * erfcx(x[upper]))
        # An unrepresentable gamma*sigma is deep in the large-q limit.
        result[~finite] = np.exp(-0.5 * z[~finite] ** 2) / gamma[~finite]
    lower = finite & ~upper
    # Form gamma*tau in physical units. z can overflow while gamma*tau
    # is representable, especially for narrow kernels and small drift.
    shift = 0.5 * q[lower] * sigma[lower] - tau[lower]
    result[lower] = sigma[lower] * (
        np.sqrt(np.pi / 2) * np.exp(gamma[lower] * shift) * erfc(x[lower])
    )
    return result


def _gaussian_lag_panels(ga, gb, t1, t2, sigma):
    """Positive finite-lag quadrature of the s1 >= s2 half."""
    u = np.minimum(t1, t2)
    d = np.maximum(t1 - t2, 0.)
    other = np.maximum(t2 - t1, 0.)
    rate = ga + gb
    # The strip integrand is a Gaussian with mode ga * sigma**2
    # truncated to [0,d]. Drop tails beyond 10 widths or 45 e-folds.
    with np.errstate(over="ignore"):
        q = ga * sigma
        end = d / sigma
    mode = np.minimum(q, end)
    left_slope = np.maximum(q - end, 0.)
    left = np.minimum(mode, _GAUSSIAN_SIGMA_CUTOFF)
    left = np.minimum(left, _GAUSSIAN_EXP_CUTOFF / np.maximum(1., left_slope))
    right = np.minimum(end - mode, _GAUSSIAN_SIGMA_CUTOFF)
    strip_width = left + right
    offset = -left[:, None] + strip_width[:, None] * _GAUSS_NODES
    lag = mode[:, None] + offset
    # Keep the small distances of the nodes from the peak before adding
    # the peak position. Subtracting two large physical lags can lose them.
    gap = np.zeros_like(d)
    interior = mode < end
    finite = interior & np.isfinite(end)
    gap[finite] = sigma[finite] * (end[finite] - mode[finite])
    unbounded = interior & ~np.isfinite(end)
    gap[unbounded] = d[unbounded] - sigma[unbounded] * mode[unbounded]
    time_to_end = np.maximum(gap[:, None] - sigma[:, None] * offset, 0.)
    strip_log = -ga[:, None] * time_to_end - 0.5 * lag * lag
    integrated = _rate_integral(rate, u)
    strip = sigma * strip_width * integrated * (np.exp(strip_log) @ _GAUSS_WEIGHTS)
    broad = ~np.isfinite(q)
    if np.any(broad):
        with np.errstate(over="ignore"):
            envelope = np.exp(-0.5 * end[broad] ** 2)
        strip[broad] = envelope * _rate_integral(ga[broad], d[broad]) * integrated[broad]
    # On the triangle the integrand falls at least as quickly as
    # exp(-(gb*sigma + end)*w - w*w/2) in normalized lag w.
    slope = gb * sigma + end
    with np.errstate(over="ignore"):
        width = np.minimum(u / sigma, _GAUSSIAN_SIGMA_CUTOFF)
    width = np.minimum(width, _GAUSSIAN_EXP_CUTOFF / np.maximum(1., slope))
    # At an infinitely large normalized lag, this half underflows to zero.
    # Skip its zero-width panel before multiplying slope by the nodes.
    active = width > 0.0
    w = width[active, None] * _GAUSS_NODES
    scales = sigma[active, None]
    remaining = np.maximum(u[active, None] - scales * w, 0.)
    with np.errstate(over="ignore"):
        tri_log = (-gb[active, None] * other[active, None] - 0.5 * end[active, None] ** 2
                   - slope[active, None] * w - 0.5 * w * w)
    values = np.exp(tri_log) * _rate_integral(rate[active, None], remaining)
    triangle = np.zeros_like(strip)
    triangle[active] = sigma[active] * width[active] * (values @ _GAUSS_WEIGHTS)
    return strip + triangle


def _gaussian_temporal(t1, t2, ga, gb, sigma):
    swap = (t1 < t2) | ((t1 == t2) & (ga < gb))
    if np.any(swap):
        t1, t2 = np.where(swap, t2, t1), np.where(swap, t1, t2)
        ga, gb = np.where(swap, gb, ga), np.where(swap, ga, gb)

    def infinite(tau):
        return _gaussian_laplace(ga, sigma, tau) + _gaussian_laplace(gb, sigma, -tau)

    with np.errstate(over="ignore"):
        e1, e2 = np.exp(-ga * t1), np.exp(-gb * t2)
    diagonal = infinite(t1 - t2)
    cross_a = e1 * infinite(-t2)
    cross_b = e2 * infinite(t1)
    starting = e1 * e2 * infinite(np.zeros_like(t1))
    numerator = (diagonal - cross_a) - (cross_b - starting)
    total = diagonal + cross_a + cross_b + starting
    rate = ga + gb
    out = np.divide(numerator, rate, out=np.zeros_like(numerator), where=rate > 0.)
    # A condition check triggers the short-domain form even for markedly
    # asymmetric time spans or correlation scales.
    unstable = np.flatnonzero((rate == 0.) | (_GAUSSIAN_CANCELLATION * numerator <= total))
    for start in range(0, unstable.size, _GAUSSIAN_BATCH):
        indices = unstable[start:start + _GAUSSIAN_BATCH]
        a, b, x, y, s = (v[indices] for v in (ga, gb, t1, t2, sigma))
        out[indices] = _gaussian_lag_panels(a, b, x, y, s) + _gaussian_lag_panels(b, a, y, x, s)
    return out


def gaussian_phi(gamma_a, gamma_b, sigma_t, t1, t2):
    """Normalized Gaussian K(z) = exp(-z**2 / (2 sigma_t**2))."""
    def temporal(t1, t2, ga, gb, sigma):
        # The kernel differs from one by at most half a machine epsilon on
        # this rectangle, including when the width is infinite.
        broad = np.maximum(t1, t2) <= _CONSTANT_GAUSSIAN * sigma
        out = np.empty_like(t1)
        ta, tb = t1[broad], t2[broad]
        out[broad] = _rate_integral(ga[broad], ta) * _rate_integral(gb[broad], tb)
        resolved = ~broad
        out[resolved] = _gaussian_temporal(
            t1[resolved], t2[resolved], ga[resolved], gb[resolved], sigma[resolved],
        )
        return out

    return _apply(temporal, t1, t2, gamma_a, gamma_b, sigma_t)


_NATIVE_CALLABLE_PROTOCOL = _capture_native_protocol(globals(), ())
