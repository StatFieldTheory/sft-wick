"""Exact rational evaluation of mixed quadratic contractions.

``exact_mixed_statistic`` returns the exact value, for the stored binary
matrices and endpoints, of the statistic defined in ``_mixed_contraction``.
``evaluate_query`` evaluates it from any subset of the upper slices of each
array, which ``_mixed_rounding`` uses for its cut attempts.

Digit slices
------------
Every nonzero float64 entry is ``sign * m * 2**e`` with an odd integer
``m < 2**53``. An array is written on one power-of-two grid as
``A = sum_s 2**(base + s w) D_s``, with ``base`` the smallest ``e`` among its
entries and integer slices ``D_s`` with ``|D_s| <= 2**w - 1`` and the sign of
the entry. The width ``w = (53 - ceil(log2 N)) // 2`` gives
``N (2**w - 1)**2 < 2**53``. Slices are extracted from the top down: with
``r = A`` and the unit ``U = 2**(base + s w)`` of an occupied offset ``s``,
``D_s = trunc(r / U)`` and ``r = r - D_s U``. Each step is exact: scaling
by a power of two is exact when its result is normal, and a scaled value
below the normal range has ``|r / U| < 1`` and digit zero (the underflow it
signals is ignored, whatever the caller's NumPy error state); ``D_s U`` is an
integer below ``2**w`` times a power of two no smaller than ``2**-1074``; and
``r - D_s U`` keeps a subset of the bits of ``r``. No bits lie between
occupied offsets, so ``|r| < 2**w U`` holds at every step.

Exactness of float64 slice products
-----------------------------------
A product of two digits is an integer of magnitude at most ``(2**w - 1)**2``.
A sum of at most ``N`` such products, in any order and grouping, is an
integer of magnitude below ``2**53``, and every such integer is a float64.
So every multiply, add and fused multiply-add of a float64 matrix product of
digit slices is exact, and the computed product is the integer product. This
holds for any product that forms each entry as a sum of products of input
entries, whatever its blocking, threading, summation order or use of fused
multiply-add: the classical model under which the rounding certificate is
derived. The ``N**3`` slice products use BLAS through ``matmul``. Each
product of two sliced matrices is checked: with a fixed vector ``x`` of odd
64-bit weights, ``sum_{s,t} P_st x`` must equal ``(sum_s L_s)(sum_t R_t x)``
modulo ``2**64``, evaluated in NumPy's unsigned integer loops. A mismatch,
which a product outside the classical model (Strassen-like or reduced
precision) would produce, issues a ``RuntimeWarning`` and the product is
recomputed with NumPy's int64 ``einsum`` loops, which are exact for the same
digits. Traces, matrix-vector products and dot products of digit slices use
``einsum`` (NumPy's own loops, not BLAS) and are exact by the same argument.

Accumulation
------------
Slice products with equal digit offset ``s + t`` are summed in int64 (at most
``_MAX_SLICES`` of them, each below ``2**53``) and carried into ``w``-bit
digits from the lowest offset up. A product then has digits in ``[0, 2**w)``
and a top digit in ``{-1, 0}``, so it can enter a trace. Row sums of trace
products are summed exactly in int64 halves and combined in Python integers.
An endpoint chain ``g_a^T M_b M_c g_d`` is the dot product of the exact
vectors ``M_b^T g_a`` and ``M_c g_d``. The result is one ``Fraction``.

Work and memory
---------------
Before any product, the slice counts of every array give the BLAS
multiply-adds, the multiply-adds in NumPy's own loops (traces,
matrix-vector products, checksums) and the peak number of dimension-square
arrays alive at once, following the evaluation order. The evaluation is
refused with ``ArithmeticError`` when the weighted work exceeds
``EXACT_WORK_LIMIT`` or the peak memory exceeds ``EXACT_MEMORY_LIMIT``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
import warnings

import numpy as np

from ._mixed_contraction import PairCache, Word, ring_pairs, word_classes

# Refusal limits, checked before any product is formed. Work counts BLAS
# multiply-adds plus _LOOP_WEIGHT per multiply-add in NumPy's own loops,
# which take about that much longer per operation than a single-core BLAS
# product. Memory counts the arrays alive at the peak of the evaluation.
EXACT_WORK_LIMIT = 10**13
EXACT_MEMORY_LIMIT = 4 * 2**30
_LOOP_WEIGHT = 64

_PRECISION = 53
# At most this many slices per array, so the products summed at one digit
# offset (at most the smaller slice count, each below 2**53), plus the
# incoming carry, stay below 2**63.
_MAX_SLICES = 511
# Entries handled at once while slicing an array; the temporary arrays of one
# block take at most about 128 bytes per entry.
_BLOCK_ENTRIES = 2**18
_BLOCK_SCRATCH_BYTES = 128 * _BLOCK_ENTRIES
_CHECKSUM_SEED = 0x6D69786564
# Normal powers of two used to scale by 2**k in exact steps.
_SCALE_STEP = 1000
# Dimension-square float64-sized scratch arrays alive while a product of two
# sliced matrices is accumulated (see exact_cost).
_PRODUCT_SCRATCH = 7


def digit_width(latent: int) -> int:
    """``w = (53 - ceil(log2 latent)) // 2``, so ``latent * (2**w - 1)**2 < 2**53``."""
    return (_PRECISION - max(latent - 1, 0).bit_length()) // 2


@dataclass(frozen=True, slots=True)
class SliceGrid:
    """Power-of-two grid of one array: ``base`` and the occupied slice offsets."""

    base: int
    offsets: tuple[int, ...]

    @property
    def span(self) -> int:
        return self.offsets[-1] - self.offsets[0] + 1 if self.offsets else 0

    def from_offset(self, lowest: int) -> SliceGrid:
        """The same grid without the offsets below ``lowest``.

        Slicing an array on it keeps each entry's bits at and above
        ``2**(base + lowest * width)`` and drops the rest, which is smaller in
        magnitude than that power of two and has the sign of the entry.
        """
        return SliceGrid(
            self.base, tuple(offset for offset in self.offsets if offset >= lowest)
        )


@dataclass(frozen=True, slots=True, eq=False)
class _Limbs:
    """Exact array ``sum_l digits[l] * 2**(base + offsets[l] * width)``.

    Digits are int32 arrays holding integers of magnitude below ``2**width``.
    """

    base: int
    offsets: tuple[int, ...]
    digits: tuple[np.ndarray, ...]
    shape: tuple[int, ...]


def _transposed(limbs: _Limbs) -> _Limbs:
    return _Limbs(
        limbs.base,
        limbs.offsets,
        tuple(digit.T for digit in limbs.digits),
        limbs.shape[::-1],
    )


def _blocks(values: np.ndarray) -> Iterator[tuple[slice, np.ndarray]]:
    """Consecutive row blocks of a vector or matrix, as (rows, view)."""
    rows = 1 if values.ndim == 1 else max(1, _BLOCK_ENTRIES // values.shape[1])
    if values.ndim == 1:
        yield slice(None), values
        return
    for start in range(0, values.shape[0], rows):
        span = slice(start, start + rows)
        yield span, values[span]


def _grid(values: np.ndarray, width: int) -> SliceGrid:
    """Smallest bit exponent and the slice offsets that hold any nonzero bit."""
    lows, highs = [], []
    for _, block in _blocks(values):
        nonzero = block[block != 0]
        if not nonzero.size:
            continue
        fraction, exponent = np.frexp(nonzero)
        magnitude = np.abs(np.ldexp(fraction, _PRECISION).astype(np.int64))
        lowest_bit = (magnitude & -magnitude).astype(np.float64)
        trailing = np.frexp(lowest_bit)[1].astype(np.int64) - 1
        lows.append(exponent.astype(np.int64) - _PRECISION + trailing)
        highs.append(exponent.astype(np.int64) - 1)
    if not lows:
        return SliceGrid(0, ())
    base = min(int(low.min()) for low in lows)
    # Binary exponents span at most 1074 + 1024 positions.
    covered = np.zeros(2200 // width + 3, dtype=np.int64)
    for low, high in zip(lows, highs):
        covered += np.bincount((low - base) // width, minlength=covered.size)
        covered -= np.bincount((high - base) // width + 1, minlength=covered.size)
    occupied = np.flatnonzero(np.cumsum(covered) > 0)
    return SliceGrid(base, tuple(int(offset) for offset in occupied))


def _times_power_of_two(values: np.ndarray, exponent: int) -> np.ndarray:
    """``values * 2**exponent`` by multiplications with normal powers of two."""
    while exponent > _SCALE_STEP:
        values = values * 2.0**_SCALE_STEP
        exponent -= _SCALE_STEP
    while exponent < -_SCALE_STEP:
        values = values * 2.0**-_SCALE_STEP
        exponent += _SCALE_STEP
    return values * 2.0**exponent


def _limbs(values: np.ndarray, grid: SliceGrid, width: int) -> _Limbs:
    """Signed ``width``-bit int32 digits of each entry on the array's grid.

    A remainder scaled below the normal range has digit zero, and the
    underflow that scaling signals is ignored whatever the caller's error
    state: the remainder itself is not changed by it.
    """
    slices = {offset: np.zeros(values.shape, dtype=np.int32) for offset in grid.offsets}
    units = [(offset, grid.base + offset * width) for offset in grid.offsets[::-1]]
    with np.errstate(under="ignore"):
        for rows, block in _blocks(values):
            remainder = np.array(block, dtype=np.float64)
            for offset, exponent in units:
                digit = np.trunc(_times_power_of_two(remainder, -exponent))
                remainder -= _times_power_of_two(digit, exponent)
                slices[offset][rows] = digit
    offsets = tuple(offset for offset in grid.offsets if np.any(slices[offset]))
    digits = tuple(slices[offset] for offset in offsets)
    return _Limbs(grid.base, offsets, digits, values.shape)


@lru_cache(maxsize=16)
def _checksum_weights(size: int) -> np.ndarray:
    random = np.random.default_rng(_CHECKSUM_SEED)
    weights = random.integers(0, 2**64, size=size, dtype=np.uint64)
    weights |= np.uint64(1)
    weights.flags.writeable = False
    return weights


def _unsigned(digits: np.ndarray) -> np.ndarray:
    return digits.astype(np.int64).view(np.uint64)


def _blas_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Float64 product of two digit slices; exact under the classical model."""
    product = np.matmul(left.astype(np.float64), right.astype(np.float64))
    return product.astype(np.int64)


def _integer_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Int64 product of two digit slices in NumPy's own loops."""
    return np.einsum("ij,jk->ik", left.astype(np.int64), right.astype(np.int64))


def _matrix_vector(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Int64 matrix-vector product of two digit slices in NumPy's own loops."""
    return np.einsum("ij,j->i", matrix, vector.astype(np.int64))


def _accumulate(
    left: _Limbs,
    right: _Limbs,
    width: int,
    multiply: Callable[[np.ndarray, np.ndarray], np.ndarray],
    observe: Callable[[np.ndarray], None] | None = None,
) -> _Limbs:
    """Exact product of two digit-slice arrays, carried to int32 digits."""
    shape = left.shape[:-1] + right.shape[1:]
    by_offset: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for left_offset, left_digits in zip(left.offsets, left.digits):
        for right_offset, right_digits in zip(right.offsets, right.digits):
            by_offset.setdefault(left_offset + right_offset, []).append(
                (left_digits, right_digits)
            )
    base = left.base + right.base
    if not by_offset:
        return _Limbs(base, (), (), shape)
    mask = (1 << width) - 1
    offset, top = min(by_offset), max(by_offset)
    carry: np.ndarray | None = None
    offsets, digits = [], []
    while True:
        total = carry
        for left_digits, right_digits in by_offset.get(offset, ()):
            product = multiply(left_digits, right_digits)
            if observe is not None:
                observe(product)
            if total is None:
                total = product
            else:
                total += product
        if total is None:
            offset += 1
            continue
        if offset > top and not np.any((total != 0) & (total != -1)):
            if np.any(total):
                offsets.append(offset)
                digits.append(total.astype(np.int32))
            break
        digit = total & mask
        carry = total >> width
        if np.any(digit):
            offsets.append(offset)
            digits.append(digit.astype(np.int32))
        offset += 1
    return _Limbs(base, tuple(offsets), tuple(digits), shape)


def _exact_matrix_product(left: _Limbs, right: _Limbs, width: int) -> _Limbs:
    """Exact matrix product: BLAS on digit slices, checked modulo ``2**64``."""
    if not (left.digits and right.digits):
        return _accumulate(left, right, width, _blas_product)
    weights = _checksum_weights(right.shape[1])
    left_sum = np.zeros(left.shape, dtype=np.uint64)
    for digits in left.digits:
        left_sum += _unsigned(digits)
    right_check = np.zeros(right.shape[0], dtype=np.uint64)
    for digits in right.digits:
        right_check += np.einsum("ij,j->i", _unsigned(digits), weights)
    expected = np.einsum("ij,j->i", left_sum, right_check)
    del left_sum
    observed = np.zeros(left.shape[0], dtype=np.uint64)

    def observe(product: np.ndarray) -> None:
        observed[...] += np.einsum("ij,j->i", product.view(np.uint64), weights)

    result = _accumulate(left, right, width, _blas_product, observe)
    if np.array_equal(observed, expected):
        return result
    del result
    warnings.warn(
        "a float64 matrix product of integer digit slices was not exact; the "
        "exact mixed contraction recomputes it in NumPy's int64 loops",
        RuntimeWarning,
        stacklevel=2,
    )
    return _accumulate(left, right, width, _integer_product)


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


def _exact_trace(left: _Limbs, right: _Limbs, width: int) -> Fraction:
    """Exact ``tr(left right) = sum_ij left_ij right_ji``."""
    # sum_ij x_ij y_ij with x = left and y = right^T, or with both transposed
    # when that makes x contiguous; y is copied to contiguous float64 one digit
    # at a time. Every row sum is an integer below 2**53, exact in float64.
    first, second = left.digits, tuple(digit.T for digit in right.digits)
    if first and not first[0].flags.c_contiguous:
        first = tuple(digit.T for digit in first)
        second = tuple(digit.T for digit in second)
    terms = []
    for right_offset, digits in zip(right.offsets, second):
        contiguous = np.ascontiguousarray(digits, dtype=np.float64)
        for left_offset, left_digits in zip(left.offsets, first):
            rows = np.einsum("ij,ij->i", contiguous, left_digits).astype(np.int64)
            terms.append((left_offset + right_offset, _exact_int_sum(rows)))
    return _combine(terms, left.base + right.base, width)


def _integer_dot(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.einsum("i,i->", left.astype(np.int64), right.astype(np.int64)))


def _exact_dot(left: _Limbs, right: _Limbs, width: int) -> Fraction:
    terms = [
        (left_offset + right_offset, _integer_dot(left_digits, right_digits))
        for left_offset, left_digits in zip(left.offsets, left.digits)
        for right_offset, right_digits in zip(right.offsets, right.digits)
    ]
    return _combine(terms, left.base + right.base, width)


def _chain_vectors(word: Word, symmetric: bool) -> tuple[tuple, tuple]:
    """Keys of the two exact vectors whose dot product is a chain word.

    ``g_a^T g_b`` is ``(a,) . (b,)``; ``g_a^T M_b g_c`` is ``(a,) . (b, c)``;
    ``g_a^T M_b M_c g_d`` is ``(b, a, transposed) . (c, d)``.
    """
    if len(word) == 2:
        return (word[0],), (word[1],)
    if len(word) == 3:
        return (word[0],), (word[1], word[2], False)
    return (word[1], word[0], not symmetric), (word[2], word[3], False)


class _Contraction:
    """Exact ring and chain sums of one query from int32 matrix slices."""

    def __init__(
        self,
        limbs: Mapping[int, _Limbs],
        vectors: Mapping[int, _Limbs],
        width: int,
        symmetric: bool,
    ) -> None:
        self._limbs, self._vectors = limbs, vectors
        self._width, self._symmetric = width, symmetric

    def _pair(self, left: int, right: int) -> _Limbs:
        return _exact_matrix_product(self._limbs[left], self._limbs[right], self._width)

    def rings(self, rings: Mapping[Word, int]) -> Fraction:
        pairs: PairCache[_Limbs] = PairCache(
            rings, self._symmetric, self._pair, _transposed
        )
        total = Fraction(0)
        for word, count in rings.items():
            if len(word) == 2:
                left, right = self._limbs[word[0]], self._limbs[word[1]]
            elif len(word) == 3:
                left, right = pairs.take(word[0], word[1]), self._limbs[word[2]]
            else:
                left = pairs.take(word[0], word[1])
                right = pairs.take(word[2], word[3])
            total += count * _exact_trace(left, right, self._width)
            del left, right
        return total

    def _vector(self, key: tuple, applied: dict[tuple, _Limbs]) -> _Limbs:
        if len(key) == 1:
            return self._vectors[key[0]]
        if key not in applied:
            matrix, endpoint, transposed = key
            limbs = self._limbs[matrix]
            applied[key] = _accumulate(
                _transposed(limbs) if transposed else limbs,
                self._vectors[endpoint],
                self._width,
                _matrix_vector,
            )
        return applied[key]

    def chains(self, chains: Mapping[Word, int]) -> Fraction:
        applied: dict[tuple, _Limbs] = {}
        total = Fraction(0)
        for word, count in chains.items():
            left, right = _chain_vectors(word, self._symmetric)
            value = _exact_dot(
                self._vector(left, applied), self._vector(right, applied), self._width
            )
            total += count * value
        return total


@dataclass(frozen=True, slots=True)
class ExactCost:
    """Estimated cost of one exact evaluation, before any product is formed.

    ``blas`` and ``loops`` count multiply-adds in BLAS products and in NumPy's
    own loops; ``memory`` bounds the bytes of arrays alive at the peak.
    """

    blas: int
    loops: int
    memory: int

    @property
    def work(self) -> int:
        return self.blas + _LOOP_WEIGHT * self.loops


def _carry_digits(width: int) -> int:
    """Digits a carried offset total can add above the top offset."""
    return -(-64 // width) + 1


def _product_digits(left: SliceGrid, right: SliceGrid, width: int) -> int:
    if not (left.offsets and right.offsets):
        return 0
    return left.span + right.span - 1 + _carry_digits(width)


def exact_cost(
    latent: int,
    rings: Mapping[Word, int],
    chains: Mapping[Word, int],
    grids: Mapping[int, SliceGrid],
    vector_grids: Mapping[int, SliceGrid],
    width: int,
    symmetric: bool,
) -> ExactCost:
    """Work and peak memory of ``exact_mixed_statistic`` from its slice grids.

    Memory is counted in dimension-square float64 arrays, an int32 digit
    counting one half: the slices of every distinct matrix are held
    throughout; a matrix product adds its int32 output digits and
    ``_PRODUCT_SCRATCH`` arrays (float64 copies of the two digits in use, the
    BLAS result, its int64 copy, the offset total, the carry and the digit);
    a trace adds one contiguous float64 digit. Products are released after
    their last use, as in the evaluation.
    """
    held = sum(len(grid.offsets) for grid in grids.values()) / 2
    blas, ring_loops, peak = _ring_cost(latent, rings, grids, width, symmetric, held)
    chain_loops = _chain_cost(latent, chains, grids, vector_grids, width, symmetric)
    memory = math.ceil(8 * latent * latent * peak) + _BLOCK_SCRATCH_BYTES
    return ExactCost(blas, ring_loops + chain_loops, memory)


def _ring_cost(
    latent: int,
    rings: Mapping[Word, int],
    grids: Mapping[int, SliceGrid],
    width: int,
    symmetric: bool,
    held: float,
) -> tuple[int, int, float]:
    """BLAS and loop multiply-adds and peak arrays of the ring words."""
    square = latent * latent
    slices = {index: len(grid.offsets) for index, grid in grids.items()}
    blas = loops = 0
    peak = held
    alive: dict[tuple[int, int], int] = {}

    def key(left: int, right: int) -> tuple[int, int]:
        return (right, left) if symmetric and right < left else (left, right)

    uses = Counter(key(*pair) for word in rings for pair in ring_pairs(word))
    for word in rings:
        operands = []
        for left, right in ring_pairs(word):
            pair = key(left, right)
            if pair not in alive:
                first, second = slices[pair[0]], slices[pair[1]]
                blas += first * second * latent * square
                # Checksum: one matrix-vector product per slice and per pair.
                loops += (first + second + first * second) * square
                digits = _product_digits(grids[pair[0]], grids[pair[1]], width)
                building = digits / 2 + _PRODUCT_SCRATCH
                peak = max(peak, held + sum(alive.values()) / 2 + building)
                alive[pair] = digits
            uses[pair] -= 1
            operands.append(alive[pair])
        operands += [slices[index] for index in word[2 * len(operands) :]]
        loops += operands[0] * operands[1] * square
        peak = max(peak, held + sum(alive.values()) / 2 + 2)
        for pair in [pair for pair, count in uses.items() if count == 0]:
            alive.pop(pair, None)
    return blas, loops, peak


def _chain_cost(
    latent: int,
    chains: Mapping[Word, int],
    grids: Mapping[int, SliceGrid],
    vector_grids: Mapping[int, SliceGrid],
    width: int,
    symmetric: bool,
) -> int:
    """Loop multiply-adds of the chain words: matrix-vector and dot products."""
    loops = 0
    digits: dict[tuple, int] = {}
    for word in chains:
        vectors = _chain_vectors(word, symmetric)
        for vector in vectors:
            if len(vector) == 1:
                digits[vector] = len(vector_grids[vector[0]].offsets)
            elif vector not in digits:
                matrix, endpoint = grids[vector[0]], vector_grids[vector[1]]
                loops += len(matrix.offsets) * len(endpoint.offsets) * latent**2
                digits[vector] = _product_digits(matrix, endpoint, width)
        loops += digits[vectors[0]] * digits[vectors[1]] * latent
    return loops


def _check_cost(cost: ExactCost, grids: list[SliceGrid]) -> None:
    """Refuse before any product when the exact evaluation is too large."""
    widest = max((len(grid.offsets) for grid in grids), default=0)
    if (
        cost.work > EXACT_WORK_LIMIT
        or cost.memory > EXACT_MEMORY_LIMIT
        or widest > _MAX_SLICES
    ):
        raise ArithmeticError(
            "uncertified mixed statistic exceeds the exact contraction work bound: "
            f"about {cost.work:.2e} weighted multiply-adds and "
            f"{cost.memory / 2**20:.0f} MiB at the peak (limits "
            f"{EXACT_WORK_LIMIT:.0e} and {EXACT_MEMORY_LIMIT / 2**20:.0f} MiB); "
            "reduce the latent dimension or the dynamic range of the prepared "
            "matrices"
        )


@dataclass(frozen=True, slots=True, eq=False)
class ExactQuery:
    """Arrays, word classes and full slice grids of one exact evaluation."""

    order: int
    latent: int
    dense: Mapping[int, np.ndarray]
    ends: Mapping[int, np.ndarray]
    width: int
    symmetric: bool
    rings: Mapping[Word, int]
    chains: Mapping[Word, int]
    grids: Mapping[int, SliceGrid]
    vector_grids: Mapping[int, SliceGrid]


def exact_query(matrices: np.ndarray, endpoints: np.ndarray, slots: Word) -> ExactQuery:
    """Collect the distinct arrays of ``slots`` and their full slice grids."""
    order, latent = len(slots), int(matrices.shape[-1])
    distinct = sorted(set(slots))
    dense = {index: np.asarray(matrices[index], dtype=np.float64) for index in distinct}
    ends = {index: np.asarray(endpoints[index], dtype=np.float64) for index in distinct}
    if not all(
        np.all(np.isfinite(array)) for array in (*dense.values(), *ends.values())
    ):
        raise ValueError("mixed contraction requires finite prepared matrices")
    width = digit_width(latent)
    symmetric = all(np.array_equal(array, array.T) for array in dense.values())
    return ExactQuery(
        order,
        latent,
        dense,
        ends,
        width,
        symmetric,
        word_classes(slots, cyclic=True, reversal=symmetric),
        word_classes(slots, cyclic=False, reversal=symmetric),
        {index: _grid(dense[index], width) for index in distinct},
        {index: _grid(ends[index], width) for index in distinct},
    )


def evaluate_query(
    query: ExactQuery,
    grids: Mapping[int, SliceGrid],
    vector_grids: Mapping[int, SliceGrid],
    divisor: int,
) -> Fraction:
    """Exact statistic of the arrays restricted to the slices of ``grids``.

    With the full grids of ``query`` this is the exact statistic. A grid that
    keeps only the upper offsets of a full grid drops the bits of each entry
    below its lowest kept offset, rounding the entry toward zero. Raises
    ``ArithmeticError`` before forming any product when the estimated work or
    peak memory exceeds ``EXACT_WORK_LIMIT`` or ``EXACT_MEMORY_LIMIT``.
    """
    if query.latent == 0:
        return Fraction(0)
    width, symmetric = query.width, query.symmetric
    cost = exact_cost(
        query.latent, query.rings, query.chains, grids, vector_grids, width, symmetric
    )
    _check_cost(cost, [*grids.values(), *vector_grids.values()])
    limbs = {
        index: _limbs(array, grids[index], width)
        for index, array in query.dense.items()
    }
    vectors = {
        index: _limbs(vector, vector_grids[index], width)
        for index, vector in query.ends.items()
    }
    contraction = _Contraction(limbs, vectors, width, symmetric)
    ring = contraction.rings(query.rings)
    chain = contraction.chains(query.chains)
    order = query.order
    return (
        Fraction(2 ** (order - 1), order) * ring + Fraction(2) ** (order - 3) * chain
    ) / divisor


def exact_mixed_statistic(
    matrices: np.ndarray, endpoints: np.ndarray, slots: Word, divisor: int
) -> Fraction:
    """Exact rational mixed statistic of the stored binary matrices and endpoints.

    Raises ``ArithmeticError`` before forming any product when the estimated
    work or peak memory exceeds ``EXACT_WORK_LIMIT`` or ``EXACT_MEMORY_LIMIT``.
    """
    query = exact_query(matrices, endpoints, slots)
    return evaluate_query(query, query.grids, query.vector_grids, divisor)
