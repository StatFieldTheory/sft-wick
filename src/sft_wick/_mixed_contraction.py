"""Certified float64 admission and exact evaluation of mixed quadratic contractions.

For prepared real matrices ``M_a`` and endpoint vectors ``g_a`` the mixed
statistic of ordered slots ``(a_1, ..., a_n)`` is

    (2**(n-1)/n * sum_w tr(M_w1 ... M_wn)
     + 2**(n-3) * sum_w g_w1^T M_w2 ... M_w(n-1) g_wn) / divisor

where ``w`` runs over all ``n!`` slot permutations, repeated ones included.
``GaussianQuadraticFamily`` first evaluates it in float64. ``rounding_certificate``
bounds the rounding error of that evaluation; ``admits`` keeps the float64
value only when the bound is at most ``MIXED_STATISTIC_RTOL`` times the value.
Otherwise ``exact_mixed_statistic`` returns the exact rational value of the
stored binary data, which the caller rounds once to float64.

Both functions concern the prepared matrices and endpoints. Errors committed
while whitening, projecting or diagonalising a user's model are not covered.

Rounding model of the certificate
---------------------------------
IEEE binary64 with round to nearest, unit roundoff ``u = 2**-53``. A matrix
product or matrix-vector product computes every output entry as a sum of its
``N`` products, in any order, with or without fused multiply-add (the classical
algorithm used by OpenBLAS, MKL, Accelerate and NumPy's fallback loops). Each
multiply, add or fused multiply-add commits a relative error of at most ``u``
and, only when its result is below the normal range, an absolute error of at
most ``eta = 2**-1022``. That ``eta`` also covers flush-to-zero results and
subnormal inputs read as zero.

With ``gamma_k = k u / (1 - k u)`` an inner product of ``N`` terms satisfies
``|fl(x.y) - x.y| <= gamma_N |x|.|y| + 2 N eta`` (Higham, *Accuracy and
Stability of Numerical Algorithms*, 2nd ed., chapter 3, with an absolute
underflow term added for each operation). For the left-associated
ring ``X_1 = A_1``, ``X_{j+1} = fl(X_j A_{j+1})``, induction on ``j`` gives

    |X_n - A_1 ... A_n| <= ((1 + gamma_N)**(n-1) - 1) |A_1| ... |A_n| + R

where ``R`` collects the absolute terms carried through the later factors.
The diagonal of ``X_n`` is summed exactly, so one ring word is in error by at
most ``((1 + gamma_N)**(n-1) - 1) tr(|A_1| ... |A_n|) + tr(R)``. The endpoint
chain applies ``n - 2`` matrix-vector products and one elementwise product
(relative error ``u``) before an exact sum, so one chain word is in error by
at most ``((1 + gamma_N)**(n-2) (1 + u) - 1) |g_1|^T |A_2| ... |g_n| + R'``.

The absolute traces and chains are computed in float64 as well. All their
terms are nonnegative, so each computed value is at least
``(1 - u)**((n-1) N + 1)`` times the exact one, less absolute underflow terms;
``(1 - u)**-k <= 1 + gamma_k`` restores an upper bound. Each product step adds
at most ``2 N eta`` to an entry (plus ``eta`` times an absolute row or column
sum for a subnormal input read as zero); later factors carry it by at most
their absolute row and column sums, and a trace adds ``N`` entries. Summed
over the at most ``n - 1`` steps of the signed and of the absolute evaluation,
this gives the per-word allowance ``64 n N**2 eta prod_m s_m`` covering
``tr(R)``, ``R'`` and the underflow of the absolute values, with
``s_m = max(1, 2 ||A_m||)`` over the slots, ``||A||`` the larger of the maximum
absolute row and column sums, and two endpoint factors
``max(1, 2 max_a ||g_a||_1)``; the factor 2 covers the rounding of the computed
norms. All coefficients are exact rationals and
the admission test ``bound <= RTOL |value|`` is decided in exact rational
arithmetic.

Overflow: every entry and partial sum of every signed or absolute product,
and every absolute trace, is at most ``N prod_m s_m / 2``. The certificate is
withheld unless ``N prod_m s_m <= 2**1020``, so neither contraction can have
overflowed even if a threaded BLAS did not report floating-point exceptions.

A computed absolute trace or chain of zero does not by itself prove a zero:
a positive term may have underflowed. A zero is certified only when every
term is at least ``2**-1000`` (the product of the smallest nonzero entries of
the factors) or when the integer zero patterns of the factors show that no
term exists.

Exact evaluation
----------------
Every float64 entry is ``sign * m * 2**e`` with an integer ``m < 2**53``. Each
array is cut into integer digit slices of ``w`` bits on a common power-of-two
grid, ``A = sum_s 2**(base + s w) D_s`` with ``|D_s| < 2**w``. ``w`` is chosen
so that ``N (2**w - 1)**2 < 2**63``; then every partial sum of a product of
two digit slices fits in int64, and NumPy's integer ``einsum`` (its own loops,
not BLAS) is exact for any summation order. Products are carried back to
``w``-bit digits with integer carries, and traces and dot products accumulate
row sums of digit products in Python integers. The result is one exact
``Fraction``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
import math

import numpy as np

# The float64 value of a mixed statistic is kept only when its certified
# rounding error is at most this fraction of its magnitude.
MIXED_STATISTIC_RTOL = Fraction(1, 10**12)

# Refusal limits of the exact evaluation, checked before any product is formed:
# integer multiply-adds, and bytes of int64 digit slices held at once.
EXACT_WORK_LIMIT = 10**11
EXACT_MEMORY_LIMIT = 2**31

_PRECISION = 53
_UNIT_ROUNDOFF = Fraction(1, 2**_PRECISION)
_UNDERFLOW_ERROR = Fraction(1, 2**1022)
_OVERFLOW_LOG2 = 1020.0
_UNDERFLOW_LOG2 = -1000.0
_MAX_CERTIFIED_ROUNDINGS = 2**40

Word = tuple[int, ...]


def word_classes(slots: tuple[int, ...], *, cyclic: bool, reversal: bool) -> dict:
    """Count the slot permutations in each class of words with equal value.

    Every one of the ``n!`` permutations, repeated ones included, is counted.
    A class representative is the smallest member, so it is itself a word.
    """
    classes: Counter = Counter()
    for word in permutations(slots):
        variants = (
            [word[shift:] + word[:shift] for shift in range(len(word))]
            if cyclic
            else [word]
        )
        if reversal:
            variants += [variant[::-1] for variant in variants]
        classes[min(variants)] += 1
    return dict(sorted(classes.items()))


@dataclass(frozen=True, slots=True)
class RoundingCertificate:
    """Outcome of ``rounding_certificate``.

    ``zero`` is true when the exact statistic is proved to be zero. ``error``
    bounds the distance between the float64 evaluation (summed exactly) and
    the exact statistic, or is ``None`` when no bound could be established.
    """

    zero: bool
    error: Fraction | None


def admits(certificate: RoundingCertificate, value: Fraction) -> bool:
    """True when the certified error is at most RTOL times ``|value|``."""
    return (
        certificate.error is not None
        and certificate.error <= MIXED_STATISTIC_RTOL * abs(value)
    )


def rounding_certificate(
    matrices: np.ndarray, endpoints: np.ndarray, slots: tuple[int, ...], divisor: int
) -> RoundingCertificate:
    """Bound the float64 rounding error of the mixed statistic of ``slots``.

    The work is one absolute contraction per class of words with equal
    absolute trace (cyclic shifts, and reversals for symmetric matrices),
    which is a small fraction of the float64 evaluation itself.
    """
    order, latent = len(slots), int(matrices.shape[-1])
    if latent == 0:
        return RoundingCertificate(True, Fraction(0))
    distinct = sorted(set(slots))
    absolute = {
        index: np.abs(np.asarray(matrices[index], dtype=np.float64))
        for index in distinct
    }
    vectors = {
        index: np.abs(np.asarray(endpoints[index], dtype=np.float64))
        for index in distinct
    }
    with np.errstate(all="ignore"):
        scales = _norm_scales(absolute, vectors, slots, latent)
        if scales is None or (order - 1) * latent + 1 > _MAX_CERTIFIED_ROUNDINGS:
            return RoundingCertificate(False, None)
        symmetric = all(np.array_equal(array, array.T) for array in absolute.values())
        rings = word_classes(slots, cyclic=True, reversal=symmetric)
        chains = word_classes(slots, cyclic=False, reversal=symmetric)
        ring_values = _absolute_ring_traces(rings, absolute, symmetric)
        chain_values = _absolute_chain_values(chains, absolute, vectors)
    values = [*ring_values.values(), *chain_values.values()]
    if not all(math.isfinite(value) for value in values):
        return RoundingCertificate(False, None)
    if not any(values) and (
        _underflow_free(absolute, vectors, slots)
        or _structurally_zero(rings, chains, absolute, vectors)
    ):
        return RoundingCertificate(True, Fraction(0))
    restore = 1 + _gamma((order - 1) * latent + 1)
    growth = 1 + _gamma(latent)
    ring_coefficient = (growth ** (order - 1) - 1) * restore
    chain_coefficient = (growth ** (order - 2) * (1 + _UNIT_ROUNDOFF) - 1) * restore
    allowance = (
        math.factorial(order)
        * 64
        * order
        * latent**2
        * _UNDERFLOW_ERROR
        * math.prod(Fraction(scale) for scale in scales)
    )
    ring = ring_coefficient * sum(
        count * Fraction(ring_values[word]) for word, count in rings.items()
    )
    chain = chain_coefficient * sum(
        count * Fraction(chain_values[word]) for word, count in chains.items()
    )
    error = (
        Fraction(2 ** (order - 1), order) * (ring + allowance)
        + Fraction(2) ** (order - 3) * (chain + allowance)
    ) / divisor
    return RoundingCertificate(False, error)


def _gamma(count: int) -> Fraction:
    return Fraction(count, 2**_PRECISION - count)


def _absolute_norm(array: np.ndarray) -> float:
    """Larger of the maximum row and column sums; the 1-norm of a vector."""
    if array.size == 0:
        return 0.0
    if array.ndim == 1:
        return float(np.sum(array))
    return float(max(np.max(np.sum(array, axis=0)), np.max(np.sum(array, axis=1))))


def _norm_scales(absolute, vectors, slots, latent: int) -> list[float] | None:
    """Per-factor scales ``max(1, 2 ||.||)``, or None when overflow is possible.

    Every entry and partial sum of the signed and absolute contractions, and
    every absolute trace, is at most ``latent * prod(scales) / 2``; the guard
    keeps that below ``2**1021``.
    """
    norms = {index: _absolute_norm(array) for index, array in absolute.items()}
    endpoint = max(_absolute_norm(vector) for vector in vectors.values())
    if not (math.isfinite(endpoint) and all(map(math.isfinite, norms.values()))):
        return None
    scales = [max(1.0, 2.0 * norms[index]) for index in slots]
    scales += [max(1.0, 2.0 * endpoint)] * 2
    if not all(map(math.isfinite, scales)):
        return None
    logs = math.fsum(math.log2(scale) for scale in scales) + math.log2(latent)
    if logs > _OVERFLOW_LOG2:
        return None
    return scales


def _absolute_ring_traces(rings, absolute, symmetric) -> dict:
    pairs: dict[tuple[int, int], np.ndarray] = {}

    def pair(left: int, right: int) -> np.ndarray:
        if symmetric and (right, left) in pairs:
            return pairs[(right, left)].T
        if (left, right) not in pairs:
            pairs[(left, right)] = absolute[left] @ absolute[right]
        return pairs[(left, right)]

    traces = {}
    for word in rings:
        if len(word) == 2:
            left, right = absolute[word[0]], absolute[word[1]]
        elif len(word) == 3:
            left, right = pair(word[0], word[1]), absolute[word[2]]
        else:
            left, right = pair(word[0], word[1]), pair(word[2], word[3])
        traces[word] = math.fsum(np.einsum("ij,ji->i", left, right))
    return traces


def _absolute_chain_values(chains, absolute, vectors) -> dict:
    values = {}
    for word in chains:
        vector = vectors[word[-1]]
        for index in reversed(word[1:-1]):
            vector = absolute[index] @ vector
        values[word] = math.fsum(vectors[word[0]] * vector)
    return values


def _smallest_nonzero(array: np.ndarray) -> float:
    positive = array[array > 0]
    return float(positive.min()) if positive.size else 1.0


def _underflow_free(absolute, vectors, slots) -> bool:
    """True when every term of every absolute contraction is at least 2**-1000."""
    logs = [math.log2(min(1.0, _smallest_nonzero(absolute[index]))) for index in slots]
    endpoint = min(_smallest_nonzero(vector) for vector in vectors.values())
    logs += [math.log2(min(1.0, endpoint))] * 2
    return math.fsum(logs) >= _UNDERFLOW_LOG2


def _structurally_zero(rings, chains, absolute, vectors) -> bool:
    """Decide from integer zero patterns whether any contraction term exists."""
    pattern = {
        index: (array != 0).astype(np.int64) for index, array in absolute.items()
    }
    support = {
        index: (vector != 0).astype(np.int64) for index, vector in vectors.items()
    }
    for word in rings:
        reach = pattern[word[0]]
        for index in word[1:]:
            reach = np.minimum(reach @ pattern[index], 1)
        if np.trace(reach):
            return False
    for word in chains:
        reach = support[word[-1]]
        for index in reversed(word[1:-1]):
            reach = np.minimum(pattern[index] @ reach, 1)
        if support[word[0]] @ reach:
            return False
    return True


# ---------------------------------------------------------------------------
# Exact evaluation
# ---------------------------------------------------------------------------


def digit_width(latent: int) -> int:
    """``w = (63 - ceil(log2 latent)) // 2``, so ``latent * (2**w - 1)**2 < 2**63``."""
    return (63 - max(latent - 1, 0).bit_length()) // 2


@dataclass(frozen=True, slots=True, eq=False)
class _Limbs:
    """Exact array ``sum_l digits[l] * 2**(base + offsets[l] * width)``."""

    base: int
    offsets: tuple[int, ...]
    digits: np.ndarray
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True, eq=False)
class _SplitPlan:
    """Integer mantissas and slice ranges of the nonzero entries of one array."""

    shape: tuple[int, ...]
    where: np.ndarray
    sign: np.ndarray
    magnitude: np.ndarray
    remainder: np.ndarray
    first: np.ndarray
    last: np.ndarray
    base: int
    occupied: np.ndarray


def _split_plan(values: np.ndarray, width: int) -> _SplitPlan:
    flat = values.reshape(-1)
    where = np.flatnonzero(flat)
    empty = np.zeros(0, dtype=np.int64)
    if not where.size:
        return _SplitPlan(values.shape, where, *([empty] * 5), 0, empty)
    fraction, exponent = np.frexp(flat[where])
    mantissa = np.ldexp(fraction, _PRECISION).astype(np.int64)
    magnitude = np.abs(mantissa)
    lowest_bit = magnitude & -magnitude
    trailing = np.frexp(lowest_bit.astype(np.float64))[1].astype(np.int64) - 1
    magnitude = magnitude >> trailing
    exponent = exponent.astype(np.int64) - _PRECISION + trailing
    bits = np.frexp(magnitude.astype(np.float64))[1].astype(np.int64)
    base = int(exponent.min())
    position = exponent - base
    first = position // width
    last = (position + bits - 1) // width
    size = int(last.max()) + 2
    coverage = np.cumsum(
        np.bincount(first, minlength=size) - np.bincount(last + 1, minlength=size)
    )
    return _SplitPlan(
        values.shape,
        where,
        np.sign(mantissa),
        magnitude,
        position - first * width,
        first,
        last,
        base,
        np.flatnonzero(coverage[:-1] > 0),
    )


def _plan_limbs(plan: _SplitPlan, width: int) -> _Limbs:
    """Signed ``width``-bit digits of each entry on the plan's slice grid."""
    mask = (1 << width) - 1
    size = math.prod(plan.shape)
    offsets, digits = [], []
    for index in plan.occupied.tolist():
        selected = (plan.first <= index) & (plan.last >= index)
        level = index - plan.first[selected]
        shift = plan.remainder[selected]
        magnitude = plan.magnitude[selected]
        # Digit ``level`` of magnitude * 2**shift, computed without forming
        # the shifted value, which can exceed int64.
        lowest = (magnitude & ((np.int64(1) << (width - shift)) - 1)) << shift
        higher = (magnitude >> np.clip(level * width - shift, 0, 63)) & mask
        digit = np.where(level == 0, lowest, higher) * plan.sign[selected]
        if not np.any(digit):
            continue
        flat = np.zeros(size, dtype=np.int64)
        flat[plan.where[selected]] = digit
        offsets.append(index)
        digits.append(flat.reshape(plan.shape))
    return _Limbs(plan.base, tuple(offsets), _stack(digits, plan.shape), plan.shape)


def _stack(digits: list, shape: tuple[int, ...]) -> np.ndarray:
    return np.stack(digits) if digits else np.zeros((0, *shape), dtype=np.int64)


def _split_int64(values: np.ndarray, width: int) -> list[np.ndarray]:
    """Digits in [0, 2**width) and a top digit in {-1, 0}, low digit first."""
    mask = (1 << width) - 1
    pieces = []
    while True:
        pieces.append(values & mask)
        values = values >> width
        if np.all((values == 0) | (values == -1)):
            if np.any(values):
                pieces.append(values)
            return pieces


def _carry_normalized(accumulator: dict, base: int, shape, width: int) -> _Limbs:
    if not accumulator:
        return _Limbs(base, (), _stack([], shape), shape)
    mask = (1 << width) - 1
    offset, top = min(accumulator), max(accumulator)
    carry = np.zeros(shape, dtype=np.int64)
    offsets, digits = [], []
    while True:
        total = carry + accumulator[offset] if offset in accumulator else carry
        if offset > top and np.all((total == 0) | (total == -1)):
            if np.any(total):
                offsets.append(offset)
                digits.append(total)
            break
        digit = total & mask
        carry = total >> width
        if np.any(digit):
            offsets.append(offset)
            digits.append(digit)
        offset += 1
    return _Limbs(base, tuple(offsets), _stack(digits, shape), shape)


def _exact_product(left: _Limbs, right: _Limbs, width: int) -> _Limbs:
    """Exact matrix or matrix-vector product from int64 digit products.

    ``einsum`` on int64 runs NumPy's own integer loops (no BLAS) and is
    faster than integer ``matmul`` for large dimensions; every partial sum is
    below ``2**63`` by the choice of ``width``, so any order is exact.
    """
    shape = left.shape[:-1] + right.shape[1:]
    subscripts = "ij,j->i" if len(right.shape) == 1 else "ij,jk->ik"
    accumulator: dict[int, np.ndarray] = {}
    for left_offset, left_digits in zip(left.offsets, left.digits):
        for right_offset, right_digits in zip(right.offsets, right.digits):
            product = np.einsum(subscripts, left_digits, right_digits)
            for level, piece in enumerate(_split_int64(product, width)):
                key = left_offset + right_offset + level
                previous = accumulator.get(key)
                accumulator[key] = piece if previous is None else previous + piece
    return _carry_normalized(accumulator, left.base + right.base, shape, width)


def _transposed(limbs: _Limbs) -> _Limbs:
    return _Limbs(
        limbs.base, limbs.offsets, np.swapaxes(limbs.digits, -1, -2), limbs.shape[::-1]
    )


def _exact_int_sum(values: np.ndarray) -> int:
    """Exact sum of int64 values whose count is below 2**31."""
    high = values >> 31
    low = values & ((1 << 31) - 1)
    return (int(np.sum(high, dtype=np.int64)) << 31) + int(np.sum(low, dtype=np.int64))


def _power_of_two(numerator: int, exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(numerator << exponent)
    return Fraction(numerator, 1 << -exponent)


def _combine(terms: list[tuple[int, int]], base: int, width: int) -> Fraction:
    if not terms:
        return Fraction(0)
    lowest = min(offset for offset, _ in terms)
    total = sum(value << ((offset - lowest) * width) for offset, value in terms)
    return _power_of_two(total, base + lowest * width)


def _exact_pair_trace(left: _Limbs, right: _Limbs, width: int) -> Fraction:
    """Exact ``sum_ij left_ij right_ji``."""
    transposed = np.ascontiguousarray(np.swapaxes(right.digits, -1, -2))
    terms = []
    for left_offset, left_digits in zip(left.offsets, left.digits):
        for right_offset, right_digits in zip(right.offsets, transposed):
            rows = np.einsum("ij,ij->i", left_digits, right_digits)
            terms.append((left_offset + right_offset, _exact_int_sum(rows)))
    return _combine(terms, left.base + right.base, width)


def _exact_dot(left: _Limbs, right: _Limbs, width: int) -> Fraction:
    terms = [
        (left_offset + right_offset, int(np.einsum("i,i->", left_digits, right_digits)))
        for left_offset, left_digits in zip(left.offsets, left.digits)
        for right_offset, right_digits in zip(right.offsets, right.digits)
    ]
    return _combine(terms, left.base + right.base, width)


class _PairProducts:
    """Exact two-matrix products, each formed once and released after last use."""

    def __init__(self, rings, limbs, width: int, symmetric: bool) -> None:
        self._limbs, self._width, self._symmetric = limbs, width, symmetric
        self._uses: Counter = Counter()
        for word in rings:
            for left, right in _ring_pairs(word):
                self._uses[self._key(left, right)] += 1
        self._cache: dict[tuple[int, int], _Limbs] = {}

    def _key(self, left: int, right: int) -> tuple[int, int]:
        if self._symmetric and right < left:
            return (right, left)
        return (left, right)

    def take(self, left: int, right: int) -> _Limbs:
        key = self._key(left, right)
        product = self._cache.get(key)
        if product is None:
            product = _exact_product(
                self._limbs[key[0]], self._limbs[key[1]], self._width
            )
        self._uses[key] -= 1
        if self._uses[key] > 0:
            self._cache[key] = product
        else:
            self._cache.pop(key, None)
        return product if key == (left, right) else _transposed(product)


def _ring_pairs(word: Word) -> list[tuple[int, int]]:
    """Two-matrix products an exact ring word needs."""
    if len(word) == 3:
        return [(word[0], word[1])]
    if len(word) == 4:
        return [(word[0], word[1]), (word[2], word[3])]
    return []


def _exact_ring_sum(rings, limbs, width: int, symmetric: bool) -> Fraction:
    pairs = _PairProducts(rings, limbs, width, symmetric)
    total = Fraction(0)
    for word, count in rings.items():
        if len(word) == 2:
            value = _exact_pair_trace(limbs[word[0]], limbs[word[1]], width)
        elif len(word) == 3:
            value = _exact_pair_trace(
                pairs.take(word[0], word[1]), limbs[word[2]], width
            )
        else:
            left = pairs.take(word[0], word[1])
            value = _exact_pair_trace(left, pairs.take(word[2], word[3]), width)
        total += count * value
    return total


def _exact_chain_sum(chains, limbs, vectors, width: int) -> Fraction:
    applied: dict[Word, _Limbs] = {}
    total = Fraction(0)
    for word, count in chains.items():
        vector, key = vectors[word[-1]], (word[-1],)
        for index in reversed(word[1:-1]):
            key = (index, *key)
            if key not in applied:
                applied[key] = _exact_product(limbs[index], vector, width)
            vector = applied[key]
        total += count * _exact_dot(vectors[word[0]], vector, width)
    return total


def _check_budget(
    order, latent, rings, chains, matrix_plans, vector_plans, width, symmetric
):
    """Refuse before any product when the exact evaluation is too large."""
    counts = {index: len(plan.occupied) for index, plan in matrix_plans.items()}
    vector_counts = {index: len(plan.occupied) for index, plan in vector_plans.items()}

    def span(plan: _SplitPlan) -> int:
        occupied = plan.occupied
        return int(occupied[-1] - occupied[0] + 1) if occupied.size else 0

    spans = {index: span(plan) for index, plan in matrix_plans.items()}
    vector_spans = {index: span(plan) for index, plan in vector_plans.items()}
    # A product of two digit slices splits into at most this many digits.
    pieces = -(-63 // width) + 1

    def product_limbs(left_span: int, right_span: int) -> int:
        return left_span + right_span + pieces + 2 if left_span and right_span else 0

    square, cube = latent**2, latent**3
    work, pairs, widest = 0, set(), 0
    for word in rings:
        if order == 2:
            work += counts[word[0]] * counts[word[1]] * square
            continue
        for left, right in _ring_pairs(word):
            pairs.add(
                (min(left, right), max(left, right)) if symmetric else (left, right)
            )
            widest = max(widest, product_limbs(spans[left], spans[right]))
        first = product_limbs(spans[word[0]], spans[word[1]])
        last = (
            counts[word[2]]
            if order == 3
            else product_limbs(spans[word[2]], spans[word[3]])
        )
        work += first * last * square
    work += sum(counts[left] * counts[right] * cube for left, right in pairs)
    for word in chains:
        limbs, current = vector_counts[word[-1]], vector_spans[word[-1]]
        for index in reversed(word[1:-1]):
            work += counts[index] * limbs * square
            current = product_limbs(spans[index], current)
            limbs = current
        work += vector_counts[word[0]] * limbs * latent
    memory = 8 * square * (sum(counts.values()) + 3 * widest + pieces)
    if work > EXACT_WORK_LIMIT or memory > EXACT_MEMORY_LIMIT:
        raise ArithmeticError(
            "uncertified mixed statistic exceeds the exact contraction work bound: "
            f"about {work:.2e} integer multiply-adds and {memory / 2**20:.0f} MiB "
            f"(limits {EXACT_WORK_LIMIT:.0e} and {EXACT_MEMORY_LIMIT / 2**20:.0f} MiB); "
            "reduce the latent dimension or the dynamic range of the prepared matrices"
        )


def exact_mixed_statistic(
    matrices: np.ndarray, endpoints: np.ndarray, slots: tuple[int, ...], divisor: int
) -> Fraction:
    """Exact rational mixed statistic of the stored binary matrices and endpoints.

    Raises ``ArithmeticError`` before forming any product when the estimated
    integer work or digit-slice memory exceeds ``EXACT_WORK_LIMIT`` or
    ``EXACT_MEMORY_LIMIT``.
    """
    order, latent = len(slots), int(matrices.shape[-1])
    distinct = sorted(set(slots))
    dense = {index: np.asarray(matrices[index], dtype=np.float64) for index in distinct}
    ends = {index: np.asarray(endpoints[index], dtype=np.float64) for index in distinct}
    if not all(
        np.all(np.isfinite(array)) for array in (*dense.values(), *ends.values())
    ):
        raise ValueError("mixed contraction requires finite prepared matrices")
    if latent == 0:
        return Fraction(0)
    width = digit_width(latent)
    symmetric = all(np.array_equal(array, array.T) for array in dense.values())
    rings = word_classes(slots, cyclic=True, reversal=symmetric)
    chains = word_classes(slots, cyclic=False, reversal=symmetric)
    matrix_plans = {index: _split_plan(dense[index], width) for index in distinct}
    vector_plans = {index: _split_plan(ends[index], width) for index in distinct}
    _check_budget(
        order, latent, rings, chains, matrix_plans, vector_plans, width, symmetric
    )
    limbs = {index: _plan_limbs(plan, width) for index, plan in matrix_plans.items()}
    vectors = {index: _plan_limbs(plan, width) for index, plan in vector_plans.items()}
    ring = _exact_ring_sum(rings, limbs, width, symmetric)
    chain = _exact_chain_sum(chains, limbs, vectors, width)
    return (
        Fraction(2 ** (order - 1), order) * ring + Fraction(2) ** (order - 3) * chain
    ) / divisor
