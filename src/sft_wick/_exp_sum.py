"""Exponential-polynomial integration on an ordered time simplex.

As in the ordered-exponential evaluator used in the examples, nested
exponential integrals are entries of a bidiagonal matrix exponential.
After changing to time gaps, polynomial factors repeat its diagonal nodes.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb, exp, factorial, frexp, fsum, ldexp, lgamma, log, prod

import numpy as np


@lru_cache(maxsize=1024)
def _taylor_size(span_bucket: int, precision: int) -> int:
    """Number of terms needed for a positive divided-difference series.

    After factoring out 1/degree!, the kth term is bounded by span**k/k!.
    The bucket is a power-of-two upper bound for span. Beyond twice that
    bound, the rest of the series is less than twice its first omitted term.
    """
    upper = ldexp(1.0, span_bucket)
    log_upper = log(upper)
    log_target = log(0.5) - ((precision or 16) + 2) * log(10)
    order = max(1, int(2 * upper))
    # Start beyond the maximum term. A direct estimate can overflow
    # for long polynomials even though the tail eventually decays.
    log_bound = order * log_upper - lgamma(order + 1)
    while log_bound > log_target:
        order += 1
        log_bound += log_upper - log(order)
    return order


@lru_cache(maxsize=65536)
def _exp_difference(nodes, precision=0):
    """Divided difference of exp, with confluent nodes and a Taylor branch.

    The nodes are sorted and nonpositive. On a short interval, use the
    complete homogeneous polynomials of the node offsets in the exponential
    Taylor series. The coefficients are nonnegative, including at a collision.
    """
    degree = len(nodes) - 1
    lo, hi = nodes[0], nodes[-1]
    span = hi - lo
    exponential = lo.exp if precision else lambda: exp(lo)
    if span == 0:
        return exponential() / factorial(degree)
    if span <= max(2, degree):
        zero = lo * 0
        _, span_bucket = frexp(float(span))
        terms_count = _taylor_size(span_bucket, precision)
        series = [zero + 1] + [zero] * (terms_count - 1)
        for node in nodes[1:]:
            delta = node - lo
            for order in range(1, len(series)):
                series[order] += delta * series[order - 1] / (degree + order)
        normalized = sum(series) if precision else fsum(series)
        return exponential() * normalized / factorial(degree)
    return (
        _exp_difference(nodes[1:], precision) - _exp_difference(nodes[:-1], precision)
    ) / span


@lru_cache(maxsize=16384)
def _gap_powers(powers: tuple[int, ...]) -> dict[tuple[int, ...], int]:
    """Expand products of powers of cumulative time gaps."""
    count = len(powers)
    terms = {(0,) * count: 1}
    for leg, power in enumerate(powers):
        for _ in range(power):
            expanded: dict[tuple[int, ...], int] = {}
            for degrees, coefficient in terms.items():
                for gap in range(leg + 1):
                    new = (*degrees[:gap], degrees[gap] + 1, *degrees[gap + 1 :])
                    expanded[new] = expanded.get(new, 0) + coefficient
            terms = expanded
    return terms


@lru_cache(maxsize=131072)
def simplex(width, rates, powers: tuple[int, ...], precision=0):
    """Integrate a monomial times an exponential over one ordered simplex.

    Return a mantissa and a log scale so that damping from fixed times can be
    applied before exponentiating. With gap rates p_j=sum(rates[j:]), the
    integral is a divided difference at the p_j (including the final zero).
    A gap raised to k repeats its node k+1 times and contributes k!.
    """
    size = len(rates)
    if not size:
        return width * 0 + 1, width * 0
    summation = sum if precision else fsum
    nodes = tuple(summation(rates[j:]) * width for j in range(size)) + (width * 0,)
    shift = max(nodes)
    nodes = tuple(node - shift for node in nodes)
    if not any(powers):
        return width**size * _exp_difference(tuple(sorted(nodes)), precision), shift
    factors = []
    for degrees, coefficient in _gap_powers(powers).items():
        repeated = [
            node
            for node, degree in zip(nodes, (*degrees, 0))
            for _ in range(degree + 1)
        ]
        factor = coefficient * prod(factorial(degree) for degree in degrees)
        factors.append(factor * _exp_difference(tuple(sorted(repeated)), precision))
    return width ** (size + sum(powers)) * summation(factors), shift


def shifted_power(power: int, offset: float) -> tuple[tuple[int, float], ...]:
    """Terms of (t + offset)**power, indexed by the degree of t."""
    if offset == 0:
        return ((power, 1),)
    return tuple((k, comb(power, k) * offset ** (power - k)) for k in range(power + 1))


@lru_cache(maxsize=4096)
def stationary_simplex(rates, powers, precision=0):
    """Integrate the same monomial on the unbounded ordered lookback cone."""
    summation = sum if precision else fsum
    decays = tuple(-summation(rates[j:]) for j in range(len(rates)))
    if any(rate <= 0 for rate in decays):
        raise ValueError(
            "method='exact' requires a convergent stationary time integral"
        )
    return summation(
        coefficient
        * prod(
            factorial(degree) / rate ** (degree + 1)
            for rate, degree in zip(decays, degrees)
        )
        for degrees, coefficient in _gap_powers(powers).items()
    )


@lru_cache(maxsize=4096)
def _ideal_schedule(predecessors, powers):
    """States and their nonzero transitions, independent of the rates."""
    full = (1 << len(powers)) - 1
    available = set()
    steps = []
    for mask in range(full + 1):
        present = tuple(j for j in range(len(powers)) if mask & (1 << j))
        if any(predecessors[j] & ~mask for j in present):
            continue
        remaining = tuple(j for j in range(len(powers)) if not mask & (1 << j))
        for q in range(sum(powers[j] for j in remaining) + 1):
            sources = [(q, (mask, q - 1))] if q else []
            sources.extend((1, (mask ^ (1 << j), q + powers[j])) for j in present)
            key = (mask, q)
            steps.append(
                (
                    key,
                    remaining,
                    tuple(
                        (scale, prev) for scale, prev in sources if prev in available
                    ),
                )
            )
            available.add(key)
    return tuple(steps)


@lru_cache(maxsize=8192)
def _pole_paths(rates, powers, predecessors, precision):
    """Lift the gated bidiagonal chain to the ideals of a time order.

    With I a prefix ideal, let z_I(s) be the partial integral times the
    exponential of the remaining rates. Adding a vertex v feeds z_I
    by s**powers[v] * z_{I-v}. A polynomial state s**q * z_I also
    receives q * s**(q-1) * z_I. Each feed appends the destination pole
    in the Laplace transform. Sum paths with the same poles before
    evaluating the divided differences.
    """
    summation = sum if precision else fsum
    states = {}
    for key, remaining, sources in _ideal_schedule(predecessors, powers):
        pole = summation(rates[j] for j in remaining)
        paths = {}
        for multiplier, source in sources:
            for poles, weight in states[source].items():
                combined = tuple(sorted((*poles, pole)))
                paths[combined] = paths.get(combined, 0) + multiplier * weight
        if key == (0, 0):
            paths[(pole,)] = 1
        states[key] = paths
    return tuple(states[(1 << len(rates)) - 1, 0].items())


@lru_cache(maxsize=1024)
def _chain_order(predecessors):
    chain = tuple(
        sorted(range(len(predecessors)), key=lambda i: predecessors[i].bit_count())
    )
    if all(predecessors[i].bit_count() == j for j, i in enumerate(chain)):
        return chain
    return None


@lru_cache(maxsize=32768)
def poset_integral(width, rates, powers, predecessors, precision=0):
    """Integrate one rate monomial over a causal partial order."""
    chain = _chain_order(predecessors)
    if chain is not None:
        chain_rates = tuple(rates[i] for i in chain)
        chain_powers = tuple(powers[i] for i in chain)
        if width == float("inf"):
            return stationary_simplex(chain_rates, chain_powers, precision), 0
        return simplex(width, chain_rates, chain_powers, precision)
    paths = _pole_paths(rates, powers, predecessors, precision)
    summation = sum if precision else fsum
    if width == float("inf"):
        factors = []
        for poles, weight in paths:
            if poles[-1] != 0 or poles[-2] >= 0:
                raise ValueError(
                    "method='exact' requires a convergent stationary time integral"
                )
            factors.append(weight / prod(-rate for rate in poles[:-1]))
        return summation(factors), 0

    peak = max(poles[-1] for poles, _ in paths)
    shift = peak * width
    terms = (
        weight * _exp_difference(tuple((p - peak) * width for p in poles), precision)
        for poles, weight in paths
    )
    return width ** (len(rates) + sum(powers)) * summation(terms), shift


def _batch_exp_difference(nodes, widths):
    """Vectorised version of the same divided difference for capped chains."""
    degree = len(nodes) - 1
    lo, hi = nodes[0], nodes[-1]
    span = (hi - lo) * widths
    if hi == lo:
        return widths**degree * np.exp(lo * widths) / factorial(degree)
    values = np.empty_like(widths)
    small = span <= 2
    if np.any(small):
        step = widths[small]
        terms = np.zeros((23, len(step)))
        terms[0] = 1
        for node in nodes[1:]:
            delta = (node - lo) * step
            for k in range(1, 23):
                terms[k] += delta * terms[k - 1]
        denominators = np.asarray(
            [factorial(degree + k) for k in range(23)], dtype=float
        )
        values[small] = (
            step**degree
            * np.exp(lo * step)
            * (terms / denominators[:, None]).sum(axis=0)
        )
    if np.any(~small):
        step = widths[~small]
        values[~small] = (
            _batch_exp_difference(nodes[1:], step)
            - _batch_exp_difference(nodes[:-1], step)
        ) / (hi - lo)
    return values


@lru_cache(maxsize=256)
def _coefficients(rates, first):
    p = np.r_[np.cumsum(np.asarray(rates)[::-1])[::-1], 0.0][first:]
    shift = max(p)
    q = p - shift
    n = len(q)
    differences = np.abs(q[:, None] - q[None, :])
    np.fill_diagonal(differences, np.inf)
    if differences.min() < 1e-4 * max(np.max(np.abs(q)), 1):
        value = q, shift, None
    else:
        terms = []
        for i in range(n):
            for j in range(i + 1):
                terms.append(
                    (
                        i,
                        j,
                        tuple(
                            (
                                k,
                                1
                                / np.prod(
                                    [
                                        q[k] - q[node]
                                        for node in range(j, i + 1)
                                        if node != k
                                    ]
                                ),
                            )
                            for k in range(j, i + 1)
                        ),
                    )
                )
        value = q, shift, tuple(terms)
    return value


def ordered_integral(times, rates, offset):
    """Batched times (m,n), fixed rates (m,), and log prefactor (n,)."""
    times = np.asarray(times, dtype=float)
    rates = tuple(float(rate) for rate in rates)
    m, n = times.shape
    if len(rates) != m:
        raise ValueError("one rate per leg required")
    bounds = np.minimum.accumulate(times[::-1], axis=0)[::-1]
    state = np.zeros((m + 1, n))
    state[0] = 1
    start = np.zeros(n)
    log_scale = np.broadcast_to(np.asarray(offset, dtype=float), (n,)).copy()
    for first, end in enumerate(bounds):
        width = np.maximum(end - start, 0)
        q, shift, terms = _coefficients(rates, first)
        current = state[first:]
        following = np.zeros_like(current)
        small = width * max(np.max(np.abs(q)), 1) < 2
        big = ~small
        if np.any(big):
            if terms is None:
                widths = width[big]
                for i in range(len(q)):
                    for j in range(i + 1):
                        entry = _batch_exp_difference(
                            tuple(sorted(q[j : i + 1])), widths
                        )
                        following[i, big] += entry * current[j, big]
            else:
                exponentials = np.exp(q[:, None] * width[big])
                for i, j, coefficients in terms:
                    weighted = np.asarray(
                        [c * exponentials[k] for k, c in coefficients]
                    )
                    entry = weighted.sum(axis=0)
                    # Entrywise conditioning detects clusters smaller than
                    # the eigenvalue spacing check can reliably classify.
                    repair = abs(entry) < 0.02 * abs(weighted).sum(axis=0)
                    if np.any(repair):
                        entry[repair] = _batch_exp_difference(
                            tuple(sorted(q[j : i + 1])), width[big][repair]
                        )
                    following[i, big] += entry * current[j, big]
        if np.any(small):
            term = current[:, small].copy()
            value = term.copy()
            for degree in range(1, 29):
                derivative = q[:, None] * term
                derivative[1:] += term[:-1]
                term = derivative * (width[small] / degree)
                value += term
            following[:, small] = value
        state[first:] = following
        state[first] = 0  # closed gate: this state can no longer feed the next
        log_scale += shift * width
        start = end
    return state[-1] * np.exp(log_scale)
