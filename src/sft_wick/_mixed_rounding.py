"""Correctly rounded mixed statistics from truncated exact evaluations.

``correctly_rounded_statistic`` returns a rational whose float64 rounding is
the float64 rounding of the exact statistic of ``_mixed_contraction``, for the
stored binary matrices and endpoints. It first evaluates exactly the statistic
of the arrays cut to their leading bits, bounds what the cut dropped, and keeps
that value when the two ends of the bound round to the same float64 (Ziv's
rounding test; rounding to nearest is monotone, so every number between the
ends rounds to that float64 as well). Otherwise it retries with more bits, and
finally evaluates the full arrays.

Cut arrays
----------
``_mixed_exact`` writes every array on a power-of-two grid of ``w``-bit digit
slices. Keeping the offsets at and above ``c`` writes ``A = H + L`` where ``L``
holds the bits of each entry below ``eps = 2**(base + c w)``, so
``|L_ij| < eps``, and ``H`` holds the rest. Both parts have the sign of the
entry, so ``|H| <= |A|`` elementwise.

Bound on the dropped part
-------------------------
For a product of ``n`` factors, by telescoping,

    A_1 ... A_n - H_1 ... H_n = sum_j H_1 ... H_{j-1} L_j A_{j+1} ... A_n.

Let ``nu_i`` bound the largest absolute row sum and the largest absolute column
sum of ``A_i``, and ``||g||_1`` for an endpoint. For a ring word,
``|tr(H_1 ... L_j ... A_n)| <= eps_j 1^T P 1`` with ``P`` the cyclic product of
the other ``n - 1`` absolute matrices, and ``1^T P 1 <= N ||P||_1 <= N
prod_{i != j} nu_i``. For a chain word ``g_1^T A_2 ... A_{n-1} g_n`` the
endpoints are factors too: a dropped interior part contributes at most
``eps_j (|g_1|^T |A_2| ... |A_{j-1}| 1)(1^T |A_{j+1}| ... |g_n|)``, a dropped
endpoint part at most ``eps`` times ``1^T`` applied to the absolute product of
the other factors, and both are at most ``eps_j prod_{i != j} nu_i``. Summed
over the ``n!`` slot permutations with the weights of the statistic, the
dropped part is at most ``sum_a eps_a K_a + sum_a eps'_a K'_a`` over the
distinct matrices and endpoints, with exact rational weights ``K``. In a ring
word every permutation has the same factors, so the ring weight is ``n! N``
times one word's.

The norms are float64 sums of nonnegative terms. A sum of ``N`` such terms, in
any order, is at least ``1 - gamma_{N-1}`` times the exact sum, and a sum of
floats whose result is subnormal is exact, so twice the computed value bounds
the exact one for ``N < 2**50``. A nonfinite norm withholds the cut, and the
arrays are then evaluated in full.

Choice of the cut
-----------------
Given a magnitude estimate ``sigma`` of the statistic and ``guard`` bits, each
array keeps the bits within ``k`` binades of its largest entry, with ``k`` the
smallest nonnegative integer such that ``2**-k sum_a mu_a K_a`` is at most
``2**-guard sigma``, where ``mu_a`` is the largest absolute entry. The bound is
then recomputed from the actual cut points. An estimate below the true
magnitude only costs digits; one above it can make the rounding test fail, and
the next attempt uses ``|S_H| - bound``, a lower bound on the magnitude. An
array whose cut drops nothing is evaluated in full, and when no array drops
anything the attempt is the exact evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
import math

import numpy as np

from ._mixed_contraction import Word
from ._mixed_exact import ExactQuery, SliceGrid, evaluate_query, exact_query

# Bits of the magnitude estimate below which the dropped part is kept, in
# successive attempts. With 64 bits the rounding test fails for about one
# value in 2**10, when the exact value lies that close to a rounding boundary.
_GUARD_BITS = (64, 128)
# Doubling a computed sum of nonnegative floats bounds the exact sum while the
# count is below 2**50.
_NORM_INFLATION = 2


@dataclass(frozen=True, slots=True)
class _Sizes:
    """Largest absolute entries and norm bounds of the distinct arrays."""

    largest: Mapping[int, float]
    norms: Mapping[int, Fraction]
    end_largest: Mapping[int, float]
    end_norms: Mapping[int, Fraction]


def _norm_bound(total: float) -> Fraction | None:
    """Upper bound of an exact sum of nonnegative terms from its float64 value."""
    if not math.isfinite(total):
        return None
    return _NORM_INFLATION * Fraction(total)


def _sizes(query: ExactQuery) -> _Sizes | None:
    """Largest entries and norm bounds, or None when a norm is not finite."""
    largest, norms, end_largest, end_norms = {}, {}, {}, {}
    with np.errstate(over="ignore"):
        for index, array in query.dense.items():
            absolute = np.abs(array)
            largest[index] = float(np.max(absolute, initial=0.0))
            rows = float(np.max(absolute.sum(axis=1), initial=0.0))
            columns = float(np.max(absolute.sum(axis=0), initial=0.0))
            norm = _norm_bound(max(rows, columns))
            if norm is None:
                return None
            norms[index] = norm
        for index, vector in query.ends.items():
            absolute = np.abs(vector)
            end_largest[index] = float(np.max(absolute, initial=0.0))
            norm = _norm_bound(float(absolute.sum()))
            if norm is None:
                return None
            end_norms[index] = norm
    return _Sizes(largest, norms, end_largest, end_norms)


def _weights(
    slots: Word, latent: int, sizes: _Sizes, divisor: int
) -> tuple[dict[int, Fraction], dict[int, Fraction]]:
    """Weights ``K_a`` and ``K'_a`` of the cut sizes in the bound on the dropped part."""
    order = len(slots)
    ring_weight = Fraction(2 ** (order - 1), order) * math.factorial(order) * latent
    chain_weight = Fraction(2) ** (order - 3)
    matrix: dict[int, Fraction] = {index: Fraction(0) for index in sizes.norms}
    endpoint: dict[int, Fraction] = {index: Fraction(0) for index in sizes.end_norms}
    for position, index in enumerate(slots):
        others = math.prod(
            (
                sizes.norms[other]
                for place, other in enumerate(slots)
                if place != position
            ),
            start=Fraction(1),
        )
        matrix[index] += ring_weight * others
    for word in permutations(slots):
        factors = [
            sizes.end_norms[word[0]],
            *(sizes.norms[index] for index in word[1:-1]),
            sizes.end_norms[word[-1]],
        ]
        for position, index in enumerate(word):
            others = math.prod(
                (factor for place, factor in enumerate(factors) if place != position),
                start=Fraction(1),
            )
            target = endpoint if position in (0, order - 1) else matrix
            target[index] += chain_weight * others
    return (
        {index: weight / divisor for index, weight in matrix.items()},
        {index: weight / divisor for index, weight in endpoint.items()},
    )


def _ceil_log2(value: Fraction) -> int:
    """Smallest integer ``k`` with ``2**k >= value`` for a positive rational."""
    numerator, denominator = value.numerator, value.denominator
    power = numerator.bit_length() - denominator.bit_length()

    def at_least(exponent: int) -> bool:
        if exponent >= 0:
            return denominator << exponent >= numerator
        return denominator >= numerator << -exponent

    while not at_least(power):
        power += 1
    while at_least(power - 1):
        power -= 1
    return power


def _cut(
    grid: SliceGrid, largest: float, binades: int, width: int
) -> tuple[SliceGrid, Fraction]:
    """Grid that keeps the bits within ``binades`` of ``largest``, and its ``eps``.

    ``eps = 2**(base + c w) <= 2**(e - binades) <= 2**-binades * largest`` with
    ``e = floor(log2 largest)``. A cut that drops nothing returns ``eps = 0``.
    """
    if not grid.offsets or largest == 0:
        return grid, Fraction(0)
    exponent = math.frexp(largest)[1] - 1
    lowest = (exponent - binades - grid.base) // width
    if lowest <= grid.offsets[0]:
        return grid, Fraction(0)
    unit = grid.base + lowest * width
    size = Fraction(2) ** unit
    return grid.from_offset(lowest), size


@dataclass(frozen=True, slots=True)
class _Truncation:
    grids: Mapping[int, SliceGrid]
    vector_grids: Mapping[int, SliceGrid]
    bound: Fraction


def _truncation(
    query: ExactQuery,
    sizes: _Sizes,
    weights: tuple[dict[int, Fraction], dict[int, Fraction]],
    magnitude: Fraction,
    guard: int,
) -> _Truncation | None:
    """Cut grids for one attempt, or None when the cut would drop nothing."""
    matrix_weights, endpoint_weights = weights
    scale = sum(
        (
            Fraction(sizes.largest[index]) * weight
            for index, weight in matrix_weights.items()
        ),
        Fraction(0),
    ) + sum(
        (
            Fraction(sizes.end_largest[index]) * weight
            for index, weight in endpoint_weights.items()
        ),
        Fraction(0),
    )
    if not scale:
        return None
    binades = max(0, _ceil_log2(scale / magnitude) + guard)
    grids, vector_grids, bound = {}, {}, Fraction(0)
    for index, grid in query.grids.items():
        grids[index], size = _cut(grid, sizes.largest[index], binades, query.width)
        bound += size * matrix_weights[index]
    for index, grid in query.vector_grids.items():
        vector_grids[index], size = _cut(
            grid, sizes.end_largest[index], binades, query.width
        )
        bound += size * endpoint_weights[index]
    if not bound:
        return None
    return _Truncation(grids, vector_grids, bound)


def _rounded(value: Fraction) -> float:
    """Float64 rounding of ``value``, infinite beyond the float64 range."""
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _same_rounding(low: Fraction, high: Fraction) -> bool:
    """True when both ends round to the same float64, sign of zero included."""
    first, second = _rounded(low), _rounded(high)
    return first == second and math.copysign(1.0, first) == math.copysign(1.0, second)


def correctly_rounded_statistic(
    matrices: np.ndarray,
    endpoints: np.ndarray,
    slots: Word,
    divisor: int,
    magnitude: Fraction | None = None,
) -> Fraction:
    """A rational that rounds to float64 as the exact mixed statistic does.

    ``magnitude`` is an estimate of the absolute value of the statistic; it
    only steers how many bits the truncated attempts keep, and without it the
    full arrays are evaluated. Raises ``ArithmeticError`` when an evaluation
    that is still needed exceeds the exact work or memory bound.
    """
    query = exact_query(matrices, endpoints, slots)
    sizes = _sizes(query) if magnitude else None
    if sizes is not None and magnitude is not None:
        weights = _weights(slots, query.latent, sizes, divisor)
        for guard in _GUARD_BITS:
            if magnitude <= 0:
                break
            truncation = _truncation(query, sizes, weights, magnitude, guard)
            if truncation is None:
                break
            value = evaluate_query(
                query, truncation.grids, truncation.vector_grids, divisor
            )
            bound = truncation.bound
            if _same_rounding(value - bound, value + bound):
                return value
            magnitude = abs(value) - bound
    return evaluate_query(query, query.grids, query.vector_grids, divisor)
