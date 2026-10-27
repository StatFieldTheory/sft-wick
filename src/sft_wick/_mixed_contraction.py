"""Certified float64 admission of mixed quadratic contractions.

For prepared real matrices ``M_a`` and endpoint vectors ``g_a`` the mixed
statistic of ordered slots ``(a_1, ..., a_n)`` is

    (2**(n-1)/n * sum_w tr(M_w1 ... M_wn)
     + 2**(n-3) * sum_w g_w1^T M_w2 ... M_w(n-1) g_wn) / divisor

where ``w`` runs over all ``n!`` slot permutations, repeated ones included.
``GaussianQuadraticFamily`` first evaluates it in float64. ``rounding_certificate``
bounds the rounding error of that evaluation; ``admits`` keeps the float64
value only when the bound is at most ``MIXED_STATISTIC_RTOL`` times the value.
Otherwise ``_mixed_exact.exact_mixed_statistic`` returns the exact rational
value of the stored binary data, which the caller rounds once to float64.

Both concern the prepared matrices and endpoints. Errors committed while
whitening, projecting or diagonalising a user's model are not covered.

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
terms are nonnegative, and every term passes through at most
``(n-1) N + 1`` roundings (the products and sums of the absolute pair
products, the final elementwise product and row sum, and the correctly
rounded ``math.fsum``), so each computed value is at least
``(1 - u)**((n-1) N + 1)`` times the exact one, less absolute underflow terms;
``(1 - u)**-k <= 1 + gamma_k`` restores an upper bound. Each product step adds
at most ``2 N eta`` to an entry (plus ``eta`` times an absolute row or column
sum for a subnormal input read as zero); later factors carry it by at most
their absolute row and column sums, and a trace adds ``N`` entries. Summed
over the at most ``n - 1`` steps of the signed and of the absolute evaluation,
this gives the per-word allowance ``64 n N**2 eta prod_m s_m`` covering
``tr(R)``, ``R'``, the underflow of the elementwise endpoint product and the
underflow of the absolute values, with ``s_m = max(1, 2 ||A_m||)`` over the
slots, ``||A||`` the larger of the maximum absolute row and column sums, and
two endpoint factors ``max(1, 2 max_a ||g_a||_1)``; the factor 2 covers the
rounding of the computed norms. All coefficients are exact rationals and the
admission test ``bound <= RTOL |value|`` is decided in exact rational
arithmetic. The certificate therefore does not depend on floating-point
exception flags: an underflow that a threaded BLAS does not report is inside
the allowance.

Overflow: every entry and partial sum of every signed or absolute product,
and every absolute trace, is at most ``N prod_m s_m / 2``. The certificate is
withheld unless ``N prod_m s_m <= 2**1020``, so neither contraction can have
overflowed even if a threaded BLAS did not report floating-point exceptions.

A computed absolute trace or chain of zero does not by itself prove a zero:
a positive term may have underflowed. A zero is certified only when every
term is at least ``2**-1000`` (the product of the smallest nonzero entries of
the factors) or when the zero patterns of the factors show that no term
exists. The pattern test multiplies 0/1 matrices in float64; every partial
sum is an integer of magnitude at most ``N < 2**53``, so it is exact. It runs
also when the overflow guard or a nonfinite absolute value withholds the
error bound, because it does not depend on the magnitudes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import permutations
import math
from typing import Generic, TypeVar

import numpy as np

# The float64 value of a mixed statistic is kept only when its certified
# rounding error is at most this fraction of its magnitude.
MIXED_STATISTIC_RTOL = Fraction(1, 10**12)

_PRECISION = 53
_UNIT_ROUNDOFF = Fraction(1, 2**_PRECISION)
_UNDERFLOW_ERROR = Fraction(1, 2**1022)
_OVERFLOW_LOG2 = 1020.0
_UNDERFLOW_LOG2 = -1000.0

Word = tuple[int, ...]
Product = TypeVar("Product")


@lru_cache(maxsize=512)
def _class_items(
    slots: Word, cyclic: bool, reversal: bool
) -> tuple[tuple[Word, int], ...]:
    classes: Counter[Word] = Counter()
    for word in permutations(slots):
        variants = (
            [word[shift:] + word[:shift] for shift in range(len(word))]
            if cyclic
            else [word]
        )
        if reversal:
            variants += [variant[::-1] for variant in variants]
        classes[min(variants)] += 1
    return tuple(sorted(classes.items()))


def word_classes(slots: Word, *, cyclic: bool, reversal: bool) -> dict[Word, int]:
    """Count the slot permutations in each class of words with equal value.

    Every one of the ``n!`` permutations, repeated ones included, is counted.
    A class representative is the smallest member, so it is itself a word.
    """
    return dict(_class_items(tuple(slots), cyclic, reversal))


def ring_pairs(word: Word) -> list[tuple[int, int]]:
    """Two-matrix products that one ring word of order three or four needs."""
    if len(word) == 3:
        return [(word[0], word[1])]
    if len(word) == 4:
        return [(word[0], word[1]), (word[2], word[3])]
    return []


class PairCache(Generic[Product]):
    """Two-matrix products of ring words, each formed once, released after last use.

    With ``symmetric`` true, the product for ``(b, a)`` is the transpose of the
    product for ``(a, b)``, so only one of them is formed.
    """

    def __init__(
        self,
        words: Mapping[Word, int],
        symmetric: bool,
        multiply: Callable[[int, int], Product],
        transpose: Callable[[Product], Product],
    ) -> None:
        self._symmetric = symmetric
        self._multiply, self._transpose = multiply, transpose
        self._uses: Counter[tuple[int, int]] = Counter(
            self._key(left, right) for word in words for left, right in ring_pairs(word)
        )
        self._cache: dict[tuple[int, int], Product] = {}

    def _key(self, left: int, right: int) -> tuple[int, int]:
        if self._symmetric and right < left:
            return (right, left)
        return (left, right)

    def take(self, left: int, right: int) -> Product:
        key = self._key(left, right)
        product = self._cache.pop(key, None)
        if product is None:
            product = self._multiply(*key)
        self._uses[key] -= 1
        if self._uses[key] > 0:
            self._cache[key] = product
        return product if key == (left, right) else self._transpose(product)


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


_PROVED_ZERO = RoundingCertificate(True, Fraction(0))
_NO_BOUND = RoundingCertificate(False, None)


def rounding_certificate(
    matrices: np.ndarray, endpoints: np.ndarray, slots: Word, divisor: int
) -> RoundingCertificate:
    """Bound the float64 rounding error of the mixed statistic of ``slots``.

    The work is one absolute contraction per class of words with equal
    absolute value: cyclic shifts, and from order three reversals when the
    absolute matrices are symmetric. It holds the absolute values of the
    distinct matrices and up to two absolute pair products at a time.
    """
    order, latent = len(slots), int(matrices.shape[-1])
    if latent == 0:
        return _PROVED_ZERO
    distinct = sorted(set(slots))
    absolute = {
        index: np.abs(np.asarray(matrices[index], dtype=np.float64))
        for index in distinct
    }
    vectors = {
        index: np.abs(np.asarray(endpoints[index], dtype=np.float64))
        for index in distinct
    }
    # An order-two ring is a cyclic shift of its reversal, and order-two
    # chains are dot products, so symmetry only matters from order three.
    symmetric = order > 2 and all(
        np.array_equal(array, array.T) for array in absolute.values()
    )
    rings = word_classes(slots, cyclic=True, reversal=symmetric)
    chains = word_classes(slots, cyclic=False, reversal=symmetric)
    with np.errstate(all="ignore"):
        scales = _norm_scales(absolute, vectors, slots, latent, symmetric)
        if scales is None:
            return _pattern_outcome(rings, chains, absolute, vectors)
        ring_values = _absolute_ring_traces(rings, absolute, symmetric)
        chain_values = _absolute_chain_values(chains, absolute, vectors, symmetric)
    values = [*ring_values.values(), *chain_values.values()]
    if not all(math.isfinite(value) for value in values):
        return _pattern_outcome(rings, chains, absolute, vectors)
    if not any(values) and (
        _underflow_free(absolute, vectors, slots)
        or _structurally_zero(rings, chains, absolute, vectors)
    ):
        return _PROVED_ZERO
    ring_factor, chain_factor, allowance_factor = _coefficients(order, latent)
    ring = _weighted_sum((count, ring_values[word]) for word, count in rings.items())
    chain = _weighted_sum((count, chain_values[word]) for word, count in chains.items())
    allowance = _product(scales)
    error = ring_factor * ring + chain_factor * chain + allowance_factor * allowance
    return RoundingCertificate(False, error / divisor)


def _weighted_sum(terms: Iterable[tuple[int, float]]) -> Fraction:
    """Exact ``sum(count * value)`` of finite floats, with one reduction."""
    ratios = [(count, value.as_integer_ratio()) for count, value in terms]
    common = max(denominator for _, (_, denominator) in ratios)
    total = sum(
        count * numerator * (common // denominator)
        for count, (numerator, denominator) in ratios
    )
    return Fraction(total, common)


def _product(values: Sequence[float]) -> Fraction:
    """Exact product of finite floats."""
    numerator = denominator = 1
    for value in values:
        top, bottom = value.as_integer_ratio()
        numerator *= top
        denominator *= bottom
    return Fraction(numerator, denominator)


def _gamma(count: int) -> Fraction:
    # count < 2**53 always: count is at most 3 N + 1 for an N x N matrix.
    return Fraction(count, 2**_PRECISION - count)


@lru_cache(maxsize=64)
def _coefficients(order: int, latent: int) -> tuple[Fraction, Fraction, Fraction]:
    """Factors of the class sums and of the scale product in the error bound.

    ``error * divisor = L c_r S_r + E c_c S_c + (L + E) A prod(scales)`` with
    ``L = 2**(n-1)/n`` and ``E = 2**(n-3)``, the ring and chain coefficients
    ``c_r = ((1 + gamma_N)**(n-1) - 1)(1 + gamma_{(n-1)N+1})`` and
    ``c_c = ((1 + gamma_N)**(n-2)(1 + u) - 1)(1 + gamma_{(n-1)N+1})``, the
    class-weighted absolute sums ``S_r`` and ``S_c``, and the allowance
    factor ``A = n! 64 n N**2 eta``.
    """
    restore = 1 + _gamma((order - 1) * latent + 1)
    growth = 1 + _gamma(latent)
    ring = (growth ** (order - 1) - 1) * restore
    chain = (growth ** (order - 2) * (1 + _UNIT_ROUNDOFF) - 1) * restore
    allowance = math.factorial(order) * 64 * order * latent**2 * _UNDERFLOW_ERROR
    leading, endpoint = Fraction(2 ** (order - 1), order), Fraction(2) ** (order - 3)
    return leading * ring, endpoint * chain, (leading + endpoint) * allowance


def _pattern_outcome(
    rings: Mapping[Word, int],
    chains: Mapping[Word, int],
    absolute: Mapping[int, np.ndarray],
    vectors: Mapping[int, np.ndarray],
) -> RoundingCertificate:
    """Proved zero when no term exists, otherwise no bound."""
    if _structurally_zero(rings, chains, absolute, vectors):
        return _PROVED_ZERO
    return _NO_BOUND


def _absolute_norm(array: np.ndarray, symmetric: bool = False) -> float:
    """Larger of the maximum row and column sums; the 1-norm of a vector."""
    if array.size == 0:
        return 0.0
    if array.ndim == 1:
        return float(array.sum())
    rows = array.sum(axis=1).max()
    # Row and column sums of a symmetric matrix agree up to the summation
    # order, whose rounding the factor 2 in the scales covers.
    return float(rows if symmetric else max(rows, array.sum(axis=0).max()))


def _norm_scales(
    absolute: Mapping[int, np.ndarray],
    vectors: Mapping[int, np.ndarray],
    slots: Word,
    latent: int,
    symmetric: bool,
) -> list[float] | None:
    """Per-factor scales ``max(1, 2 ||.||)``, or None when overflow is possible.

    Every entry and partial sum of the signed and absolute contractions, and
    every absolute trace, is at most ``latent * prod(scales) / 2``; the guard
    keeps that below ``2**1021``.
    """
    norms = {
        index: _absolute_norm(array, symmetric) for index, array in absolute.items()
    }
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


def _absolute_ring_traces(
    rings: Mapping[Word, int], absolute: Mapping[int, np.ndarray], symmetric: bool
) -> dict[Word, float]:
    pairs: PairCache[np.ndarray] = PairCache(
        rings,
        symmetric,
        lambda left, right: absolute[left] @ absolute[right],
        lambda product: product.T,
    )
    traces = {}
    for word in rings:
        if len(word) == 2:
            left, right = absolute[word[0]], absolute[word[1]]
        elif len(word) == 3:
            left, right = pairs.take(word[0], word[1]), absolute[word[2]]
        else:
            left = pairs.take(word[0], word[1])
            right = pairs.take(word[2], word[3])
        traces[word] = math.fsum(np.einsum("ij,ji->i", left, right))
    return traces


def _absolute_chain_values(
    chains: Mapping[Word, int],
    absolute: Mapping[int, np.ndarray],
    vectors: Mapping[int, np.ndarray],
    symmetric: bool,
) -> dict[Word, float]:
    """Absolute chains, ``|g_a|^T |M_b| |M_c| |g_d| = (|M_b|^T |g_a|).(|M_c| |g_d|)``.

    Each matrix multiplies all distinct endpoints at once; every term still
    passes through at most ``2 N + 2`` roundings.
    """
    order = len(next(iter(chains)))
    if order == 2:
        return {word: math.fsum(vectors[word[0]] * vectors[word[1]]) for word in chains}
    column = {index: position for position, index in enumerate(vectors)}
    stacked = np.stack(list(vectors.values()), axis=1)
    right = {index: array @ stacked for index, array in absolute.items()}
    left = (
        right
        if symmetric or order == 3
        else {index: array.T @ stacked for index, array in absolute.items()}
    )
    values = {}
    for word in chains:
        if order == 3:
            first = vectors[word[0]]
            second = right[word[1]][:, column[word[2]]]
        else:
            first = left[word[1]][:, column[word[0]]]
            second = right[word[2]][:, column[word[3]]]
        values[word] = math.fsum(first * second)
    return values


def _smallest_nonzero(array: np.ndarray) -> float:
    positive = array[array > 0]
    return float(positive.min()) if positive.size else 1.0


def _underflow_free(
    absolute: Mapping[int, np.ndarray], vectors: Mapping[int, np.ndarray], slots: Word
) -> bool:
    """True when every term of every absolute contraction is at least 2**-1000."""
    logs = [math.log2(min(1.0, _smallest_nonzero(absolute[index]))) for index in slots]
    endpoint = min(_smallest_nonzero(vector) for vector in vectors.values())
    logs += [math.log2(min(1.0, endpoint))] * 2
    return math.fsum(logs) >= _UNDERFLOW_LOG2


def _structurally_zero(
    rings: Mapping[Word, int],
    chains: Mapping[Word, int],
    absolute: Mapping[int, np.ndarray],
    vectors: Mapping[int, np.ndarray],
) -> bool:
    """Decide from the zero patterns whether any contraction term exists.

    The 0/1 patterns are multiplied in float64 and clamped to 0/1 after each
    product, so every partial sum is an integer of magnitude at most
    ``N < 2**53`` and the products are exact in any summation order.
    """
    pattern = {
        index: (array != 0).astype(np.float64) for index, array in absolute.items()
    }
    support = {
        index: (vector != 0).astype(np.float64) for index, vector in vectors.items()
    }
    for word in rings:
        reach = pattern[word[0]]
        for index in word[1:]:
            reach = np.minimum(reach @ pattern[index], 1.0)
        if np.trace(reach):
            return False
    for word in chains:
        reach = support[word[-1]]
        for index in reversed(word[1:-1]):
            reach = np.minimum(pattern[index] @ reach, 1.0)
        if support[word[0]] @ reach:
            return False
    return True
