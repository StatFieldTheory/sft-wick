"""Symmetrization of declared quadratic matrices, subnormal entries included.

Preparation replaces a quadratic matrix by its symmetric part. These tests
compare the prepared matrices, endpoints and constants, and the cumulants of
the declared observables, with references that share no code with the
package: ``Fraction`` arithmetic on the declared arrays, the joint cumulant
generating function of Gaussian quadratic forms, and a frozen copy of the
earlier preparation for inputs whose halves are exact in float64.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
from itertools import permutations
from pathlib import Path
import struct
import sys
import tarfile
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
    ``2**(n-3) l.A...A.l``.
    """
    exact = [_exact(matrix) for matrix in matrices]
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
    (U, 0.0),  # 0.5U is not a float64: ties to +0.0
    (3 * U, 0.0),
    ((2**52 - 1) * U, (2**52 + 1) * U),
    (EXACT_HALVING - U, U),
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
