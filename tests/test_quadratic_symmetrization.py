"""Symmetrization of declared quadratic matrices, subnormal entries included.

Preparation replaces a quadratic matrix by its symmetric part. These tests
compare the prepared matrices, endpoints and constants, and the cumulants of
the declared observables, with references that share no code with the
package: ``Fraction`` arithmetic on the declared arrays, the joint cumulant
generating function of Gaussian quadratic forms, and a frozen copy of the
earlier preparation for inputs whose halves are exact in float64, in C order,
Fortran order and as transposed views. A declared matrix whose symmetric part
has an entry below the normal range that float64 cannot hold is refused when
every entry of the matrix is below 2**-1022, and rounded once otherwise: W A W
for a Gaussian kernel over some 38 correlation lengths on a Gauss-Legendre,
Simpson or trapezoid grid, and user products such as K D K, are prepared and
checked against cumulants computed with NumPy from the matrix formed
symmetrically.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
from itertools import permutations
from pathlib import Path
import struct
import sys
import tarfile
import tracemalloc
import types

import numpy as np
import pytest

from sft_wick import quadratic
from sft_wick.quadratic import GaussianQuadratic, GaussianQuadraticFamily

from .mixed_reference import assert_certified

U = 2.0**-1074  # smallest positive subnormal
MIN_NORMAL = 2.0**-1022
EXACT_HALVING = 2.0**-1021  # x / 2 is exact for every |x| at least this
LARGEST = float(np.finfo(np.float64).max)


def _bits(array) -> bytes:
    return np.ascontiguousarray(array, dtype=np.float64).tobytes()


def _exact(array):
    return np.vectorize(lambda value: Fraction(float(value)), otypes=[object])(
        np.asarray(array, dtype=np.float64)
    )


def _exact_symmetric_part(matrix):
    exact = _exact(matrix)
    return (exact + exact.T) / 2


def _rounded(exact) -> np.ndarray:
    # CPython converts a Fraction by correctly rounded integer division,
    # subnormal results included.
    return np.vectorize(float, otypes=[np.float64])(exact)


def _exact_cumulant(matrices, endpoints, slots) -> Fraction:
    """Joint cumulant of ``Y_a = l_a.w + w.A_a w`` for standard Gaussian ``w``.

    The joint CGF is ``-1/2 log det(I - 2 A_t) + 1/2 l_t.(I - 2 A_t)^-1 l_t``
    with ``A_t = sum t_a A_a``. Its slot derivative sums every ordering of
    the slots: rings ``2**(n-1) / n tr(A...A)`` and chains
    ``2**(n-3) l.A...A.l``. A matrix may be given as a ``Fraction`` array,
    such as the exact symmetric part of a declared matrix.
    """
    exact = [
        matrix if matrix.dtype == object else _exact(matrix) for matrix in matrices
    ]
    lines = [_exact(endpoint) for endpoint in endpoints]
    dimension = exact[0].shape[0]
    identity = np.array(
        [[Fraction(int(i == j)) for j in range(dimension)] for i in range(dimension)],
        dtype=object,
    )
    ring = chain = Fraction(0)
    for order in permutations(slots):
        product = identity
        for slot in order:
            product = product.dot(exact[slot])
        ring += sum(product[i, i] for i in range(dimension))
        middle = identity
        for slot in order[1:-1]:
            middle = middle.dot(exact[slot])
        chain += lines[order[0]].dot(middle).dot(lines[order[-1]])
    count = len(slots)
    return Fraction(2 ** (count - 1), count) * ring + Fraction(2**count, 8) * chain


def _with_pair(first, second, size=3):
    """Generic normal-range matrix with ``first`` at (0, 1), ``second`` at (1, 0)."""
    matrix = np.array(
        [[1.375, 0.0, -2.25], [0.0, -0.625, 0.5], [-2.25, 0.5, 3.0]]
    )[:size, :size]
    matrix[0, 1], matrix[1, 0] = first, second
    return matrix


# -- exactly symmetric matrices are prepared unchanged --------------------

ODD_SMALL = [
    U,
    3 * U,
    5 * U,
    (2**52 - 1) * U,  # largest subnormal
    (2**52 + 1) * U,  # odd entry of the lowest normal binade
    EXACT_HALVING - U,
]


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("value", ODD_SMALL)
def test_exactly_symmetric_small_entries_are_prepared_unchanged(value, sign):
    first = _with_pair(sign * value, sign * value)
    first[1, 2] = first[2, 1] = -3 * U
    second = _with_pair(-7 * U, -7 * U)
    matrices = np.array([first, second])
    family = GaussianQuadraticFamily.from_factor(np.eye(3), matrices)
    assert _bits(family._matrices) == _bits(matrices)
    covariance = GaussianQuadraticFamily(np.eye(3), matrices)
    assert _bits(covariance._matrices) == _bits(matrices)


def test_exact_whitening_keeps_odd_subnormal_entries():
    # Even subnormal declared entries; one power-of-two entry per factor
    # column makes every whitened entry a single exact product.
    declared = np.array(
        [[0.75, 6 * U, -2 * U], [6 * U, -1.5, 10 * U], [-2 * U, 10 * U, 2.5]]
    )
    factor = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, -2.0, 0.0]])
    family = GaussianQuadraticFamily.from_factor(factor, declared[None])
    exact_factor = _exact(factor)
    whitened = exact_factor.T.dot(_exact(declared)).dot(exact_factor)
    expected = _rounded(whitened)
    assert np.all(_exact(expected) == whitened)
    assert expected[0, 2] == 3 * U  # an odd subnormal unit
    assert _bits(family._matrices[0]) == _bits(expected)


def test_mean_shift_uses_the_declared_subnormal_entries():
    declared = np.array([[0.0, 3 * U, 0.0], [3 * U, 0.0, -5 * U], [0.0, -5 * U, 0.0]])
    mean = np.array([2.0**70, -(2.0**71), 2.0**69])
    constant = 2.0**-930
    family = GaussianQuadraticFamily.from_factor(
        np.eye(3), declared[None], constant=[constant], mean=mean
    )
    exact_matrix, exact_mean = _exact(declared), _exact(mean)
    endpoint = 2 * exact_matrix.dot(exact_mean)
    shifted = Fraction(constant) + exact_mean.dot(exact_matrix).dot(exact_mean)
    # Every product and partial sum is exact in float64.
    assert np.all(_exact(family._endpoints[0]) == endpoint)
    assert Fraction(float(family._constants[0])) == shifted == 7 * Fraction(2) ** -933


# -- cumulants of the declared observables ---------------------------------


def _audit_family():
    """Reproducer of the audit: one subnormal coupling, the rest normal."""
    first = np.zeros((2, 2))
    first[0, 0] = 1e300
    second = np.array([[0.0, U], [U, 1.0]])
    return np.array([first, second]), np.zeros((2, 2))


def _generic_family():
    big = 2.0**1000
    first = np.array([[1.25, 3 * U, 0.0], [3 * U, 0.0, -7 * U], [0.0, -7 * U, 0.0]])
    second = np.array(
        [[0.0, -1.5 * big, 0.0], [-1.5 * big, 0.0, 0.0], [0.0, 0.0, 0.75]]
    )
    third = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.625 * 2.0**1010], [0.0, 0.625 * 2.0**1010, -2.5]]
    )
    endpoints = np.array([[0.4, 0.0, 0.0], [0.0, 0.0, -0.3], [0.0, 0.0, 0.0]])
    return np.array([first, second, third]), endpoints


def _cross_family(units):
    # Cov(2 s z1 z2, 2 T z1 z2) = 4 s T exactly.
    tall = 2.0**1000
    first = np.array([[0.0, units * U], [units * U, 0.0]])
    second = np.array([[0.0, tall], [tall, 0.0]])
    return np.array([first, second]), np.zeros((2, 2))


CASES = {
    "audit": (_audit_family, [(0, 1, 0, 1)]),
    "generic": (
        _generic_family,
        [(0, 1), (0, 2), (1, 2), (0, 1, 2), (0, 0, 1), (0, 0, 2), (0, 2, 0, 2)],
    ),
    "cross_1": (lambda: _cross_family(1), [(0, 1)]),
    "cross_3": (lambda: _cross_family(3), [(0, 1)]),
    "cross_5": (lambda: _cross_family(5), [(0, 1)]),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_declared_observable_cumulants_match_exact_reference(case):
    build, queries = CASES[case]
    matrices, endpoints = build()
    family = GaussianQuadraticFamily.from_factor(
        np.eye(matrices.shape[-1]), matrices, endpoints
    )
    for slots in queries:
        exact = _exact_cumulant(matrices, endpoints, slots)
        assert exact != 0
        assert_certified(family.mixed_cumulant(slots), exact)
        powers = [slots.count(index) for index in range(len(matrices))]
        divisor = 1
        for power in powers:
            for factor in range(2, power + 1):
                divisor *= factor
        assert_certified(family.joint_coefficient(powers), exact / divisor)


# -- matrices that are not symmetric ---------------------------------------

SCALED_BOUND = 2.0**-900
PAIRS = [
    (3 * U, -U),  # exact U; halving first gave 2U
    (5 * U, U),  # exact 3U
    (-3 * U, U),  # exact -U
    (U, EXACT_HALVING + 2.0**-1073),  # 2**-1022 + 1.5U, ties to + 2U
    (-U, EXACT_HALVING + 2.0**-1073),  # 2**-1022 + 0.5U, ties to 2**-1022
    (2 * U, 0.0),  # exact U
    (-6 * U, -0.0),  # exact -3U
    ((2**52 - 1) * U, (2**52 + 1) * U),
    (EXACT_HALVING - U, U),  # exact 2**-1022
    (EXACT_HALVING - U, 2 * U),  # 2**-1022 + 0.5U, ties to 2**-1022
    (-(EXACT_HALVING - U), -2 * U),
    (EXACT_HALVING, 0.0),  # exact 2**-1022, halves exact
    (EXACT_HALVING, -EXACT_HALVING + 2 * U),
    (-0.0, 0.0),
    (3 * U, SCALED_BOUND * (1 - 2.0**-53)),
    (3 * U, SCALED_BOUND),
    (-3 * U, SCALED_BOUND * (1 + 2.0**-52)),
    (U, LARGEST),
    (-U, -LARGEST),
    (LARGEST, LARGEST * (1 - 2.0**-52)),
    (1.0, 2.0**-60),
]


@pytest.mark.parametrize("pair", PAIRS)
def test_symmetric_part_of_a_nonsymmetric_matrix_is_rounded_once(pair):
    matrix = _with_pair(*pair)
    matrix[2, 1] = -0.1  # one more nonsymmetric normal pair with (1, 2)
    family = GaussianQuadraticFamily.from_factor(np.eye(3), matrix[None])
    expected = _rounded(_exact_symmetric_part(matrix))
    assert _bits(family._matrices[0]) == _bits(expected)


# Exact averages below 2**-1022 that are not float64 values.
UNREPRESENTABLE_PAIRS = [
    (U, 0.0),  # 0.5U, ties to +0.0: a proved zero of a nonzero observable
    (-U, -0.0),
    (3 * U, 0.0),  # 1.5U, ties to 2U
    (5 * U, 0.0),  # 2.5U, ties to 2U
    (U, 2 * U),
    (-5 * U, 2 * U),
    (U, MIN_NORMAL),  # 2**-1023 + 0.5U
    (EXACT_HALVING - U, 0.0),  # 2**-1022 - 0.5U, ties up to 2**-1022
    (-(EXACT_HALVING - U), -0.0),
    ((2**52 - 1) * U, 2 * U),  # largest subnormal paired with an even unit
]

CONSTRUCTORS = {
    "single_factor": lambda matrix: GaussianQuadratic.from_factor(np.eye(3), matrix),
    "single_covariance": lambda matrix: GaussianQuadratic(np.eye(3), matrix),
    "single_white_noise": lambda matrix: GaussianQuadratic.from_white_noise(
        np.full(3, 4.0), matrix, np.full(3, 0.25), mean=np.array([0.5, -1.0, 2.0])
    ),
    # Unit weights form W A W = A exactly.
    "single_weighted_grid": lambda matrix: GaussianQuadratic.from_weighted_grid(
        np.diag([1.5, 0.25, 2.0]), matrix, np.ones(3)
    ),
    "family_factor": lambda matrix: GaussianQuadraticFamily.from_factor(
        np.eye(3), np.array([np.eye(3), matrix])
    ),
    "family_covariance": lambda matrix: GaussianQuadraticFamily(
        np.diag([2.0, 0.5, 3.0]), np.array([matrix, np.eye(3)])
    ),
}


#: The argument each constructor names in its refusal.
REFUSAL_NAMES = {
    "single_factor": "quadratic",
    "single_covariance": "quadratic",
    "single_white_noise": "quadratic_kernel",
    "single_weighted_grid": "W A W formed from quadratic_kernel and weights",
    "family_factor": r"quadratic\[1\]",
    "family_covariance": r"quadratic\[0\]",
}


def _subnormal_pair(first, second):
    """The pair at (0, 1) and (1, 0) of an otherwise zero 3 x 3 matrix."""
    matrix = np.zeros((3, 3))
    matrix[0, 1], matrix[1, 0] = first, second
    return matrix


@pytest.mark.parametrize("constructor", sorted(CONSTRUCTORS))
@pytest.mark.parametrize(
    "pair", [p for p in UNREPRESENTABLE_PAIRS if max(map(abs, p)) < MIN_NORMAL]
)
def test_unrepresentable_subnormal_symmetric_part_is_refused(pair, constructor):
    # Every entry is below 2**-1022, so rounding can change the entry by up
    # to its whole value relative to the largest one.
    matrix = _subnormal_pair(*pair)
    exact = _exact_symmetric_part(matrix)[0, 1]
    assert abs(exact) < MIN_NORMAL and Fraction(float(exact)) != exact
    match = f"symmetric part of {REFUSAL_NAMES[constructor]} has"
    with pytest.raises(ValueError, match=match):
        CONSTRUCTORS[constructor](matrix)
    # The transposed declaration is the same observable and is refused too.
    with pytest.raises(ValueError, match=match):
        CONSTRUCTORS[constructor](matrix.T.copy())


@pytest.mark.parametrize("constructor", sorted(CONSTRUCTORS))
@pytest.mark.parametrize("pair", UNREPRESENTABLE_PAIRS)
def test_unrepresentable_entry_beside_a_normal_one_is_rounded(pair, constructor):
    """With an entry of at least 2**-1022 in the matrix the rounding, at
    most 2**-1075, is at most 2**-53 times that entry, the bound of the
    normal range; the entry is rounded once, correctly."""
    matrix = _with_pair(*pair)
    CONSTRUCTORS[constructor](matrix)
    CONSTRUCTORS[constructor](matrix.T.copy())
    # Compared by value: the whitening F.T A F adds +0.0 to a -0.0 entry.
    family = GaussianQuadraticFamily.from_factor(np.eye(3), matrix[None])
    assert np.array_equal(family._matrices[0],
                          _rounded(_exact_symmetric_part(matrix)))


@pytest.mark.parametrize("largest, refused", [
    (MIN_NORMAL, False),
    (MIN_NORMAL - U, True),  # largest subnormal
    (-MIN_NORMAL, False),
    (-(MIN_NORMAL - U), True),
], ids=["2**-1022", "below", "-2**-1022", "-below"])
def test_refusal_threshold_is_the_largest_entry(largest, refused):
    """Both sides of the threshold, on the diagonal (an exact entry) and
    off it, with an odd pair whose average 0.5 U float64 cannot hold."""
    for where in ((2, 2), (0, 2)):
        matrix = _subnormal_pair(U, 0.0)
        matrix[where] = largest
        if refused:
            with pytest.raises(ValueError, match="every entry of quadratic"):
                GaussianQuadratic.from_factor(np.eye(3), matrix)
            continue
        family = GaussianQuadraticFamily.from_factor(np.eye(3), matrix[None])
        assert np.array_equal(family._matrices[0],
                              _rounded(_exact_symmetric_part(matrix)))


def _grid(rule, n, lo, hi):
    if rule == "gauss_legendre":
        nodes, weights = np.polynomial.legendre.leggauss(n)
        half = 0.5 * (hi - lo)
        return lo + (nodes + 1) * half, weights * half
    x = np.linspace(lo, hi, n)
    h = x[1] - x[0]
    if rule == "simpson":
        w = np.full(n, 2.0)
        w[1::2] = 4.0
        w[0] = w[-1] = 1.0
        return x, w * h / 3
    w = np.full(n, h)
    w[0] = w[-1] = h / 2
    return x, w


def _cumulants_reference(covariance, matrix, orders=(1, 2, 3)):
    """``κ_n = 2**(n-1) (n-1)! tr((C B)^n)`` of ``z.T B z``, ``z ~ N(0, C)``."""
    product = covariance @ matrix
    power = np.eye(len(matrix))
    out = []
    for n in orders:
        power = power @ product
        out.append(2.0 ** (n - 1) * float(np.prod(range(1, n))) * np.trace(power))
    return out


def _law_cumulants(law, orders=(1, 2, 3)):
    return [law.mean if n == 1 else law.cumulant(n) for n in orders]


# A Gaussian kernel over about 38 or more correlation lengths has subnormal
# entries; W A W rounds them differently in its two triangles.
KERNEL_GRIDS = [
    ("gauss_legendre", 120, 0.7, 45.3, 1.1),
    ("simpson", 301, 0.5, 60.5, 1.5),
    ("trapezoid", 400, 0.25, 80.25, 2.0),
]


@pytest.mark.parametrize("rule, n, lo, hi, ell", KERNEL_GRIDS,
                         ids=[g[0] for g in KERNEL_GRIDS])
def test_weighted_grid_of_a_wide_gaussian_kernel_is_prepared(rule, n, lo, hi,
                                                             ell):
    x, w = _grid(rule, n, lo, hi)
    d = x[:, None] - x[None, :]
    kernel = np.exp(-d**2 / (2 * ell**2)) * np.cos(0.37 * d)
    assert np.array_equal(kernel, kernel.T)
    covariance = 1.3 * np.exp(-np.abs(d) / 2.3) + 0.05 * np.eye(n)
    product = w[:, None] * kernel * w[None, :]
    assert np.any((product != 0) & (np.abs(product) < MIN_NORMAL))
    law = GaussianQuadratic.from_weighted_grid(covariance, kernel, w)
    # The same observable with its matrix formed symmetrically.
    symmetric = (w[:, None] * w[None, :]) * kernel
    assert np.array_equal(symmetric, symmetric.T)
    np.testing.assert_allclose(_law_cumulants(law),
                               _cumulants_reference(covariance, symmetric),
                               rtol=1e-12)


def test_matrix_products_with_subnormal_entries_are_prepared():
    """User-formed products: K D K (a matmul whose triangles differ), S K S
    elementwise, and a family holding K and S K S."""
    n = 300
    x = np.linspace(0.0, 60.0, n)
    d = x[:, None] - x[None, :]
    kernel = np.exp(-d**2 / 2)
    covariance = np.exp(-np.abs(d) / 3.0) + 0.1 * np.eye(n)
    diagonal = np.linspace(0.5, 1.5, n)
    kdk = kernel @ np.diag(diagonal) @ kernel
    scale = np.linspace(0.8, 1.2, n)
    sks = scale[:, None] * kernel * scale[None, :]
    assert not np.array_equal(kdk, kdk.T)
    for matrix in (kdk, sks):
        assert np.any((matrix != 0) & (np.abs(matrix) < MIN_NORMAL))
        law = GaussianQuadratic(covariance, matrix)
        # Reference: the float64 average, whose subnormal entries differ
        # from the prepared ones by at most 2**-1074.
        reference = _cumulants_reference(covariance, (matrix + matrix.T) / 2)
        np.testing.assert_allclose(_law_cumulants(law), reference, rtol=1e-11)
    family = GaussianQuadraticFamily(covariance, np.array([kernel, sks]))
    covariance_ref = 2 * np.trace(covariance @ kernel @ covariance @ sks)
    np.testing.assert_allclose(family.mixed_cumulant((0, 1)), covariance_ref,
                               rtol=1e-11)


@pytest.mark.parametrize("units", [1, 2, 3, 4, 5])
def test_one_sided_subnormal_coupling_is_refused_or_exact(units):
    # Y1 = k U z0 z1 declared in one triangle, Y2 = 2 T z0 z1.
    tall = 2.0**1000
    declared = np.array([[0.0, units * U], [0.0, 0.0]])
    partner = np.array([[0.0, tall], [tall, 0.0]])
    matrices = np.array([declared, partner])
    if units % 2:
        with pytest.raises(ValueError, match="symmetric part of quadratic"):
            GaussianQuadraticFamily.from_factor(np.eye(2), matrices)
        return
    family = GaussianQuadraticFamily.from_factor(np.eye(2), matrices)
    exact = _exact_cumulant(
        [_exact_symmetric_part(declared), partner], np.zeros((2, 2)), (0, 1)
    )
    assert exact == 2 * units * Fraction(U) * Fraction(tall)
    assert_certified(family.mixed_cumulant((0, 1)), exact)


def test_whitened_matrix_is_rounded_without_refusal():
    # Every whitened entry is one product: (F.T A F)[1, 0] = 3U * 0.25 rounds
    # to U and (F.T A F)[0, 1] = 0, an average of 0.5U that float64 cannot
    # hold. Only a declared matrix is refused; the whitening is float64
    # preprocessing that the guarantee does not cover.
    declared = np.array([[U, 0.0], [0.0, 0.0]])
    factor = np.array([[0.25, 3.0], [0.0, 0.0]])
    whitened = np.array(
        [[0.0, 0.0], [float(Fraction(3 * U) * Fraction(0.25)), 9 * U]]
    )
    assert whitened[1, 0] == U
    expected = _rounded(_exact_symmetric_part(whitened))
    family = GaussianQuadraticFamily.from_factor(factor, declared[None])
    single = GaussianQuadratic.from_factor(factor, declared)
    assert _bits(family._matrices[0]) == _bits(expected)
    assert single.retained_dimension == 2


def test_preparation_ignores_a_raising_numpy_error_state():
    symmetric = _with_pair(3 * U, 3 * U)
    nonsymmetric = _with_pair(5 * U, -U)
    arrays = np.array([symmetric, nonsymmetric])
    plain = GaussianQuadraticFamily.from_factor(np.eye(3), arrays)
    with np.errstate(all="raise"):
        trapped = GaussianQuadraticFamily.from_factor(np.eye(3), arrays)
        single = GaussianQuadratic.from_factor(np.eye(3), nonsymmetric)
    assert _bits(trapped._matrices) == _bits(plain._matrices)
    assert _bits(plain._matrices[0]) == _bits(symmetric)
    assert _bits(plain._matrices[1]) == _bits(
        _rounded(_exact_symmetric_part(nonsymmetric))
    )
    assert np.isfinite(single.cumulant(2))


# -- the two averaging methods agree where the preparation switches -------


def _halved_sum(first, second):
    with np.errstate(under="ignore"):
        return np.float64(first) / 2 + np.float64(second) / 2


def _scaled_sum(first, second):
    with np.errstate(under="ignore", over="ignore"):
        up = np.float64(2.0**200)
        return (np.float64(first) * up / 2 + np.float64(second) * up / 2) * 2.0**-200


@pytest.mark.parametrize(
    "large",
    [
        SCALED_BOUND * (1 - 2.0**-53),
        SCALED_BOUND,
        SCALED_BOUND * (1 + 2.0**-52),
        2.0**-899,
        2.0**-901,
        -SCALED_BOUND,
        -(2.0**-967),
        2.0**822,
    ],
)
@pytest.mark.parametrize("small", [U, -U, 3 * U, -5 * U, (2**52 - 1) * U, 0.0])
def test_averaging_methods_agree_at_the_scaling_bound(large, small):
    exact = float(Fraction(large) / 2 + Fraction(small) / 2)
    assert _halved_sum(large, small) == exact
    assert _scaled_sum(large, small) == exact
    family = GaussianQuadraticFamily.from_factor(
        np.eye(3), _with_pair(large, small)[None]
    )
    assert family._matrices[0][0, 1] == exact == family._matrices[0][1, 0]


def test_each_averaging_method_fails_where_the_other_is_used():
    # Below the bound halving first rounds twice; far above it, scaling
    # overflows. The prepared entry is correctly rounded in both cases.
    low = (U, EXACT_HALVING + 2.0**-1073)
    high = (U, LARGEST)
    for pair, wrong in ((low, _halved_sum), (high, _scaled_sum)):
        exact = float(Fraction(pair[0]) / 2 + Fraction(pair[1]) / 2)
        assert wrong(*pair) != exact
        family = GaussianQuadraticFamily.from_factor(np.eye(3), _with_pair(*pair)[None])
        assert family._matrices[0][0, 1] == exact


# -- every input whose halves are exact keeps the earlier bits ------------


@pytest.fixture(scope="module")
def frozen_quadratic():
    archive = (
        Path(__file__).resolve().parents[1]
        / "tests/fixtures/quadratic_white_reference_source.tar.gz"
    )
    with tarfile.open(archive) as stream:
        member = stream.extractfile("src/sft_wick/quadratic.py")
        assert member is not None
        source = member.read()
    assert (
        hashlib.sha256(source).hexdigest()
        == "eee500133e818e60cdda1c0ea268a8cf6f61a69d2ea4b0b6e7423187eaf7ed89"
    )
    name = "frozen_symmetrization_quadratic"
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, f"<{name}>", "exec"), module.__dict__)
    return module


def _prepared_bytes(module, factor, matrix, linear, constant, mean):
    matrix_out, endpoint, shifted, dimension = module._observable_data(
        factor, matrix, linear, constant, mean
    )
    return _bits(matrix_out), _bits(endpoint), struct.pack("!d", shifted), dimension


def _normal_range_cases():
    random = np.random.default_rng(90210)
    for size, rank in ((1, 1), (4, 4), (6, 3), (9, 12)):
        for scale in (1.0, 1e300, 1e-280):
            for symmetric in (True, False):
                raw = random.normal(size=(size, size)) * scale
                matrix = raw + raw.T if symmetric else raw
                factor = random.normal(size=(size, rank)) * np.exp2(
                    random.integers(-3, 4, size=(size, rank))
                )
                linear = random.normal(size=size)
                mean = random.normal(size=size) * 0.7
                yield factor, matrix, linear, -1.25, mean
    # Structured extremes with an exact identity whitening: overflow-sized
    # pairs, the exact-halving floor, rounded averages and signed zeros.
    edge = np.array(
        [
            [LARGEST, -LARGEST, EXACT_HALVING, -0.0],
            [LARGEST * (1 - 2.0**-52), -0.0, -EXACT_HALVING, 0.0],
            [3 * EXACT_HALVING + 2 * U, EXACT_HALVING, 1.0, 2.0**-60],
            [-0.0, -0.0, 1.0 + 2.0**-52, -3.5],
        ]
    )
    yield np.eye(4), edge, None, 0.0, None
    symmetric = edge.copy()
    lower = np.tril_indices(4, -1)
    symmetric[lower] = edge.T[lower]
    yield np.eye(4), symmetric, None, 0.0, None


def test_inputs_with_exact_halves_keep_the_earlier_prepared_bits(frozen_quadratic):
    count = 0
    for factor, matrix, linear, constant, mean in _normal_range_cases():
        nonzero = np.abs(matrix[matrix != 0])
        assert nonzero.size == 0 or nonzero.min() >= EXACT_HALVING
        current = _prepared_bytes(quadratic, factor, matrix, linear, constant, mean)
        earlier = _prepared_bytes(frozen_quadratic, factor, matrix, linear, constant, mean)
        assert current == earlier
        whitened = np.frombuffer(current[0])
        small = np.abs(whitened[whitened != 0])
        assert small.size == 0 or small.min() >= EXACT_HALVING
        count += 1
    assert count == 26


def test_family_preparation_keeps_the_earlier_bits(frozen_quadratic):
    random = np.random.default_rng(7)
    raw = random.normal(size=(3, 5, 5)) * np.array([1.0, -1e200, 1e-200])[:, None, None]
    factor = random.normal(size=(5, 4))
    linear = random.normal(size=(3, 5))
    constants = np.array([0.5, -2.0, 1e-3])
    mean = random.normal(size=5)
    for matrices in (raw, raw + raw.swapaxes(-1, -2)):
        current = GaussianQuadraticFamily.from_factor(
            factor, matrices, linear, constant=constants, mean=mean
        )
        earlier = frozen_quadratic.GaussianQuadraticFamily.from_factor(
            factor, matrices, linear, constant=constants, mean=mean
        )
        for name in ("_matrices", "_endpoints", "_constants"):
            assert _bits(getattr(current, name)) == _bits(getattr(earlier, name))


# -- memory layout of exactly symmetric inputs ----------------------------

# NumPy lays out A / 2 + A.T / 2 for a Fortran-ordered A in C order up to
# N = 181 and in Fortran order from N = 182, so both sides are checked.
LAYOUT_SIZES = [5, 40, 181, 182]


def _layouts(matrix):
    """One exactly symmetric matrix in C order, Fortran order and as a view."""
    contiguous = np.ascontiguousarray(matrix)
    return {
        "C": contiguous,
        "fortran": np.asfortranarray(contiguous),
        "transposed_view": contiguous.T,
    }


def _generic_symmetric(random, size, scale):
    raw = random.normal(size=(size, size)) * scale
    return raw + raw.T


def _law_bytes(law):
    return (
        _bits(law.eigenvalues),
        _bits(law._linear_squares),
        struct.pack("!2d", law._constant, law.mean),
        struct.pack("!3d", *(law.cumulant(order) for order in (2, 3, 4))),
    )


def _prepared_layout(module, factor, matrix, linear, constant, mean):
    matrix_out, endpoint, shifted, _ = module._observable_data(
        factor, matrix, linear, constant, mean
    )
    flags = (matrix_out.flags.c_contiguous, matrix_out.flags.f_contiguous)
    return _bits(matrix_out), flags, _bits(endpoint), struct.pack("!d", shifted)


@pytest.mark.parametrize("size", LAYOUT_SIZES)
def test_symmetric_input_layout_keeps_the_earlier_single_law_bits(
    frozen_quadratic, size
):
    random = np.random.default_rng(3100 + size)
    matrix = _generic_symmetric(random, size, 1.75)
    factor = random.normal(size=(size, size - 1))
    linear = random.normal(size=size)
    mean = random.normal(size=size) * 0.6
    gram = factor @ factor.T + np.eye(size)
    covariance = np.triu(gram) + np.triu(gram, 1).T
    for label, declared in _layouts(matrix).items():
        assert np.array_equal(declared, matrix)
        for build in ("from_factor", "covariance"):
            laws = []
            for module in (quadratic, frozen_quadratic):
                if build == "from_factor":
                    law = module.GaussianQuadratic.from_factor(
                        factor, declared, linear, constant=0.3, mean=mean
                    )
                else:
                    law = module.GaussianQuadratic(
                        covariance, declared, linear, constant=0.3, mean=mean
                    )
                laws.append(_law_bytes(law))
            assert laws[0] == laws[1], (label, build)
        assert _prepared_layout(
            quadratic, factor, declared, linear, 0.3, mean
        ) == _prepared_layout(frozen_quadratic, factor, declared, linear, 0.3, mean)


@pytest.mark.parametrize("size", [6, 182])
def test_symmetric_kernel_layout_keeps_the_earlier_white_noise_bits(
    frozen_quadratic, size
):
    random = np.random.default_rng(3300 + size)
    kernel = _generic_symmetric(random, size, -0.8)
    intensity = random.uniform(0.5, 2.0, size)
    weights = random.uniform(0.1, 1.0, size)
    linear = random.normal(size=size)
    mean = random.normal(size=size)
    for label, declared in _layouts(kernel).items():
        laws = [
            _law_bytes(
                module.GaussianQuadratic.from_white_noise(
                    intensity, declared, weights, linear, constant=-0.7, mean=mean
                )
            )
            for module in (quadratic, frozen_quadratic)
        ]
        assert laws[0] == laws[1], label


@pytest.mark.parametrize("size", [4, 12, 40, 182])
def test_symmetric_input_layout_keeps_the_earlier_family_bits(frozen_quadratic, size):
    random = np.random.default_rng(3500 + size)
    scales = np.array([1.0, -3e10, 2e-9])[:, None, None]
    raw = random.normal(size=(3, size, size)) * scales
    matrices = raw + raw.swapaxes(1, 2)
    factor = random.normal(size=(size, size - 1))
    linear = random.normal(size=(3, size))
    constants = random.normal(size=3)
    mean = random.normal(size=size)
    directions = np.array([1.0, -0.5, 2.0])
    for label, declared in {
        "C": matrices,
        "fortran": np.asfortranarray(matrices),
        "swapaxes": matrices.swapaxes(1, 2),
    }.items():
        assert np.array_equal(declared, matrices)
        current = GaussianQuadraticFamily.from_factor(
            factor, declared, linear, constant=constants, mean=mean
        )
        earlier = frozen_quadratic.GaussianQuadraticFamily.from_factor(
            factor, declared, linear, constant=constants, mean=mean
        )
        for name in ("_matrices", "_endpoints", "_constants"):
            assert _bits(getattr(current, name)) == _bits(getattr(earlier, name)), (
                label,
                name,
            )
        # Mixed cumulants of the earlier prepared bytes, by the current code.
        reference = GaussianQuadraticFamily._from_prepared(
            earlier._matrices, earlier._endpoints, earlier._constants, size
        )
        for slots in ((0, 1), (0, 1, 1), (0, 2, 1, 2)):
            assert struct.pack("!d", current.mixed_cumulant(slots)) == struct.pack(
                "!d", reference.mixed_cumulant(slots)
            ), (label, slots)
        assert _law_bytes(current.project(directions)) == _law_bytes(
            earlier.project(directions)
        ), label


# -- temporary memory of a triangular declaration --------------------------


def test_matrix_with_zero_entries_needs_no_extra_temporary():
    if tracemalloc.is_tracing():
        pytest.skip("tracemalloc is already tracing")
    size = 512
    upper = np.triu(np.random.default_rng(5).normal(size=(size, size)))

    def peak(function):
        tracemalloc.start()
        try:
            result = function(upper)
            return result, tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    earlier, earlier_peak = peak(lambda matrix: matrix / 2 + matrix.T / 2)
    current, current_peak = peak(quadratic._symmetric_part)
    assert _bits(current) == _bits(earlier)
    # Half a matrix of slack holds the boolean masks; one more float64
    # matrix, as when the transposed mask and the maximum were built, does not.
    assert current_peak <= earlier_peak + upper.nbytes // 2
