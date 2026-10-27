"""Independent references for the mixed-statistic tests.

``fraction_statistic``, ``float_statistic`` and ``integer_statistic`` share no
code with ``sft_wick``: rational loops over all slot permutations, the
left-associated float64 contraction with exact final sums, and Python-int
matrix products. The family helper only constructs test inputs.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations
import math

import numpy as np
import pytest

from sft_wick.quadratic import GaussianQuadraticFamily

RTOL = Fraction(1, 10**12)
UNIT = Fraction(1, 2**53)
# Final rounding of a value certified to RTOL before rounding.
ADMITTED = (RTOL + UNIT) / (1 - RTOL)


def symmetric(random, latent):
    raw = random.normal(size=(latent, latent))
    return raw + raw.T


def make_family(matrices, endpoints=None):
    matrices = np.asarray(matrices, dtype=np.float64)
    count, latent = matrices.shape[0], matrices.shape[-1]
    endpoints = np.zeros((count, latent)) if endpoints is None else endpoints
    family = GaussianQuadraticFamily.from_factor(np.eye(latent), matrices, endpoints)
    # The identity factor and symmetric inputs keep every prepared bit, so a
    # closed form of the inputs is a closed form of the prepared data.
    assert np.array_equal(family._matrices, matrices)
    assert np.array_equal(family._endpoints, np.asarray(endpoints, dtype=np.float64))
    return family


def assert_certified(value: float, exact: Fraction) -> None:
    """Correctly rounded, or within the certified relative error."""
    try:
        rounded = float(exact)
    except OverflowError:
        pytest.fail(f"returned {value!r} for a statistic outside float64 range")
    if value == rounded and math.copysign(1.0, value) == math.copysign(1.0, rounded):
        return
    assert exact != 0, f"exact zero returned as {value!r}"
    error = abs(Fraction(value) - exact) / abs(exact)
    assert error <= ADMITTED, f"relative error {float(error):.3e} exceeds 1e-12"


def assert_positive_zero(value: float) -> None:
    assert value == 0.0 and not math.copysign(1.0, value) < 0


def fraction_statistic(matrices, endpoints, slots, divisor=1) -> Fraction:
    """Independent exact statistic: rational loops over all slot permutations."""
    order, latent = len(slots), matrices.shape[-1]
    exact = {
        index: [[Fraction(float(value)) for value in row] for row in matrices[index]]
        for index in set(slots)
    }
    ends = {
        index: [Fraction(float(value)) for value in endpoints[index]]
        for index in set(slots)
    }
    span = range(latent)

    def product(left, right):
        return [
            [sum((left[i][k] * right[k][j] for k in span), Fraction(0)) for j in span]
            for i in span
        ]

    ring = chain = Fraction(0)
    for word in permutations(slots):
        current = exact[word[0]]
        for index in word[1:]:
            current = product(current, exact[index])
        ring += sum((current[i][i] for i in span), Fraction(0))
        vector = ends[word[-1]]
        for index in reversed(word[1:-1]):
            vector = [
                sum((exact[index][i][k] * vector[k] for k in span), Fraction(0))
                for i in span
            ]
        chain += sum((a * b for a, b in zip(ends[word[0]], vector)), Fraction(0))
    return (
        Fraction(2 ** (order - 1), order) * ring + Fraction(2) ** (order - 3) * chain
    ) / divisor


def float_statistic(matrices, endpoints, slots, divisor=1) -> Fraction:
    """Left-associated float64 contraction with exact final sums."""
    order = len(slots)
    ring = chain = Fraction(0)
    for word in permutations(slots):
        product = matrices[word[0]]
        for index in word[1:]:
            product = product @ matrices[index]
        ring += sum((Fraction(float(v)) for v in np.diag(product)), Fraction(0))
        vector = endpoints[word[-1]]
        for index in reversed(word[1:-1]):
            vector = matrices[index] @ vector
        chain += sum(
            (Fraction(float(v)) for v in endpoints[word[0]] * vector), Fraction(0)
        )
    return (
        Fraction(2 ** (order - 1), order) * ring + Fraction(2) ** (order - 3) * chain
    ) / divisor


def integer_array(array: np.ndarray) -> tuple[np.ndarray, int]:
    """Python-int object array ``values`` and ``exponent`` with array = values * 2**exponent."""
    ratios = [Fraction(float(value)) for value in array.reshape(-1)]
    denominator = max(ratio.denominator for ratio in ratios)
    values = np.array(
        [ratio.numerator * (denominator // ratio.denominator) for ratio in ratios],
        dtype=object,
    ).reshape(array.shape)
    return values, -(denominator.bit_length() - 1)


def integer_statistic(matrices, endpoints, slots) -> Fraction:
    """Independent exact statistic from Python-int matrix products.

    Ring words of order four are traced as tr((M_a M_b)(M_c M_d)) with the
    pair products memoised; chains apply exact matrix-vector products.
    """
    order = len(slots)
    ints = {index: integer_array(matrices[index]) for index in set(slots)}
    ends = {index: integer_array(endpoints[index]) for index in set(slots)}
    pairs: dict = {}

    def pair(left, right):
        if (left, right) not in pairs:
            (a, ea), (b, eb) = ints[left], ints[right]
            pairs[(left, right)] = (np.dot(a, b), ea + eb)
        return pairs[(left, right)]

    ring = chain = Fraction(0)
    for word in permutations(slots):
        if order == 2:
            (a, ea), (b, eb) = ints[word[0]], ints[word[1]]
            value, exponent = int(np.sum(a * b.T)), ea + eb
        elif order == 3:
            (p, ep), (c, ec) = pair(word[0], word[1]), ints[word[2]]
            value, exponent = int(np.sum(p * c.T)), ep + ec
        else:
            (p, ep), (q, eq) = pair(word[0], word[1]), pair(word[2], word[3])
            value, exponent = int(np.sum(p * q.T)), ep + eq
        ring += value * Fraction(2) ** exponent
        vector, exponent = ends[word[-1]]
        for index in reversed(word[1:-1]):
            matrix, shift = ints[index]
            vector, exponent = np.dot(matrix, vector), exponent + shift
        first, shift = ends[word[0]]
        chain += int(np.dot(first, vector)) * Fraction(2) ** (exponent + shift)
    return Fraction(2 ** (order - 1), order) * ring + Fraction(2) ** (order - 3) * chain


def endpoint_tuned(matrices, endpoints, slots, index, level):
    """Endpoints with ``index`` replaced so the statistic is ``level`` of its size.

    ``index`` occurs once in ``slots``, so its endpoint sits at one end of a
    chain word or not at all and the statistic is affine in it; the offset
    and gradient come from the float contraction of this file.
    """
    assert slots.count(index) == 1

    def total(vector):
        trial = endpoints.copy()
        trial[index] = vector
        return float(float_statistic(matrices, trial, slots))

    base = endpoints[index]
    offset = total(np.zeros_like(base))
    gradient = np.array([total(unit) - offset for unit in np.eye(base.size)])
    natural = total(base)
    tuned = endpoints.copy()
    tuned[index] = base - ((1 - level) * natural / (gradient @ gradient)) * gradient
    return tuned


def cancelled_family(latent, order, level, seed, with_endpoints):
    """Second observable tuned so tr(A**(order-1) B) is ``level`` of its
    uncancelled value; diagonal dominance makes level 1 well conditioned."""
    random = np.random.default_rng(seed)
    noise = 0.1 / math.sqrt(latent)
    a = np.diag(random.uniform(0.5, 2.0, latent)) + noise * symmetric(random, latent)
    base = np.diag(random.uniform(-2.0, 2.0, latent))
    base += noise * symmetric(random, latent)
    power = np.linalg.matrix_power(a, order - 1)
    natural = np.sum(power * base)
    b = base - ((1 - level) * natural / np.sum(power * power)) * power
    b = b / 2 + b.T / 2
    endpoints = None
    if with_endpoints:
        endpoints = random.normal(size=(2, latent)) * np.array([[1.3], [-0.4]])
    return make_family(np.array([a, b]), endpoints)


SWEEP = [10.0 ** (-k / 2) for k in range(0, 17)] + [0.0, 1e-12, -1e-9]


def dense_family_arrays(latent, count, seed):
    """Dense symmetric indefinite matrices with full mantissas, and endpoints."""
    random = np.random.default_rng(seed)
    raw = random.normal(size=(count, latent, latent))
    matrices = (raw + raw.swapaxes(-1, -2)) / math.sqrt(latent)
    return matrices, random.normal(size=(count, latent))
