"""Certified mixed cumulants: exact zeros, certified admission and exact rounding.

Every value returned by ``GaussianQuadraticFamily.mixed_cumulant`` and
``joint_coefficient`` is either within a relative error of 1e-12 of the exact
statistic of the prepared matrices (before its final rounding) or the
correctly rounded exact value. References here share no code with the
package: plain ``Fraction`` contractions, closed forms of block, rank-one and
reversal-parity structures, and exact integer sums.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations
import math
import time

import numpy as np
import pytest

from sft_wick.quadratic import GaussianQuadraticFamily

RTOL = Fraction(1, 10**12)
UNIT = Fraction(1, 2**53)
# Final rounding of a value certified to RTOL before rounding.
ADMITTED = (RTOL + UNIT) / (1 - RTOL)


def _fraction_statistic(matrices, endpoints, slots, divisor=1) -> Fraction:
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


def _assert_certified(value: float, exact: Fraction) -> None:
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


def _assert_positive_zero(value: float) -> None:
    assert value == 0.0 and not math.copysign(1.0, value) < 0


def _family(matrices, endpoints=None):
    matrices = np.asarray(matrices, dtype=np.float64)
    count, latent = matrices.shape[0], matrices.shape[-1]
    endpoints = np.zeros((count, latent)) if endpoints is None else endpoints
    family = GaussianQuadraticFamily.from_factor(np.eye(latent), matrices, endpoints)
    # The identity factor and symmetric inputs keep every prepared bit, so a
    # closed form of the inputs is a closed form of the prepared data.
    assert np.array_equal(family._matrices, matrices)
    assert np.array_equal(family._endpoints, np.asarray(endpoints, dtype=np.float64))
    return family


def _symmetric(random, latent):
    raw = random.normal(size=(latent, latent))
    return raw + raw.T


# ---------------------------------------------------------------------------
# Structural zeros: independent observables on disjoint blocks
# ---------------------------------------------------------------------------


def _block_family(latent, seed):
    random = np.random.default_rng(seed)
    matrices = np.zeros((3, latent, latent))
    endpoints = np.zeros((3, latent))
    if latent == 1:
        # Y0 = a z**2 + 0 z, Y1 = 0 z**2 + b z, Y2 = c z**2; only cumulants
        # odd in Y1 are zero, and those are zero term by term.
        matrices[0, 0, 0], matrices[2, 0, 0] = -1.7, 0.4
        endpoints[1, 0] = 2.3
        return _family(matrices, endpoints)
    cut = max(1, latent // 3)
    first, second = slice(0, cut), slice(cut, latent)
    for index, block in ((0, first), (1, second), (2, first)):
        width = block.stop - block.start
        matrices[index][block, block] = _symmetric(random, width) * (1.5 - index)
        endpoints[index][block] = random.normal(size=width) * (index - 0.7)
    return _family(matrices, endpoints)


_BLOCK_ZERO_SLOTS = [
    (0, 1),
    (1, 2),
    (0, 0, 1),
    (2, 0, 1),
    (0, 0, 0, 1),
    (0, 2, 0, 1),
    (1, 0, 2, 0),
]
_ODD_IN_ONE_SLOTS = [(0, 1), (0, 0, 1), (2, 0, 1), (0, 0, 0, 1), (1, 1, 1, 0)]


@pytest.mark.parametrize("latent", [1, 2, 100, 500])
def test_block_disjoint_mixed_cumulants_are_exact_positive_zeros(latent):
    family = _block_family(latent, seed=4100 + latent)
    cases = _ODD_IN_ONE_SLOTS if latent == 1 else _BLOCK_ZERO_SLOTS
    started = time.perf_counter()
    for slots in cases:
        _assert_positive_zero(family.mixed_cumulant(slots))
    powers = [(1, 1, 0), (2, 1, 0), (1, 1, 2)] if latent > 1 else [(1, 1, 0)]
    for power in powers:
        _assert_positive_zero(family.joint_coefficient(power))
    # A certified structural zero needs no signed dense contraction.
    assert time.perf_counter() - started < 5.0


def test_block_zero_survives_statistic_preparation():
    family = _block_family(100, seed=4199)
    selected = family.prepare_statistics(
        indices=[(0, 1), (0, 2, 0, 1)], coefficients=[(2, 1, 0)]
    )
    _assert_positive_zero(selected.mixed_cumulant((0, 2, 0, 1)))
    _assert_positive_zero(selected.joint_coefficient((2, 1, 0)))


def test_correlated_block_pair_stays_nonzero_beside_the_zero():
    family = _block_family(6, seed=4142)
    for slots in [(0, 2), (2, 0, 2), (0, 0, 2, 2)]:
        exact = _fraction_statistic(family._matrices, family._endpoints, slots)
        assert exact != 0
        _assert_certified(family.mixed_cumulant(slots), exact)


# ---------------------------------------------------------------------------
# Symmetry zeros: index reversal parity, dense and not structural
# ---------------------------------------------------------------------------


def _parity_family(latent, seed):
    random = np.random.default_rng(seed)

    def even(matrix):
        return matrix + matrix[::-1, ::-1]

    def odd(matrix):
        return matrix - matrix[::-1, ::-1]

    matrices = np.array(
        [
            even(_symmetric(random, latent)),
            odd(_symmetric(random, latent) * 0.3),
            even(_symmetric(random, latent) * -2.1),
        ]
    )
    raw = random.normal(size=(3, latent))
    endpoints = np.array(
        [raw[0] + raw[0][::-1], raw[1] - raw[1][::-1], (raw[2] + raw[2][::-1]) * 5]
    )
    family = _family(matrices, endpoints)
    for index, sign in ((0, 1), (1, -1), (2, 1)):
        assert np.array_equal(
            family._matrices[index][::-1, ::-1], sign * family._matrices[index]
        )
        assert np.array_equal(
            family._endpoints[index][::-1], sign * family._endpoints[index]
        )
    return family


@pytest.mark.parametrize(
    "latent, slot_sets",
    [
        (2, [(0, 1), (0, 0, 1), (2, 0, 1), (1, 1, 1), (0, 0, 0, 1), (0, 1, 1, 1)]),
        (3, [(0, 1), (0, 0, 1), (2, 0, 1), (1, 1, 1), (0, 2, 0, 1), (0, 1, 1, 1)]),
        (100, [(0, 1), (0, 0, 1), (2, 0, 1), (1, 1, 1), (0, 2, 0, 1), (0, 1, 1, 1)]),
        (500, [(0, 1), (0, 0, 1), (0, 1, 1, 1)]),
    ],
)
def test_reversal_parity_zeros_are_exact_positive_zeros(latent, slot_sets):
    family = _parity_family(latent, seed=4300 + latent)
    for slots in slot_sets:
        _assert_positive_zero(family.mixed_cumulant(slots))


def test_parity_even_cumulants_are_nonzero_and_certified():
    family = _parity_family(5, seed=4305)
    for slots in [(1, 1), (0, 1, 1), (2, 0, 1, 1), (1, 1, 1, 1)]:
        exact = _fraction_statistic(family._matrices, family._endpoints, slots)
        assert exact != 0
        _assert_certified(family.mixed_cumulant(slots), exact)


# ---------------------------------------------------------------------------
# Near zeros: one overlapping entry between otherwise disjoint blocks
# ---------------------------------------------------------------------------


def _overlap_case(latent, overlap, seed):
    random = np.random.default_rng(seed)
    cut = latent // 2
    a = _symmetric(random, cut)
    b = _symmetric(random, latent - cut)
    matrices = np.zeros((2, latent, latent))
    matrices[0][:cut, :cut] = a
    matrices[1][cut:, cut:] = b
    matrices[1][0, 0] = overlap
    family = _family(matrices)
    row = [Fraction(float(value)) for value in a[0]]
    exact_a = [[Fraction(float(value)) for value in line] for line in a]
    a00, eps = row[0], Fraction(float(overlap))
    a2_00 = sum((value * value for value in row), Fraction(0))
    # (A**3)_00 = a_0^T A a_0 with a_0 the first row of the symmetric block.
    a3_00 = sum(
        (row[k] * exact_a[k][m] * row[m] for k in range(cut) for m in range(cut)),
        Fraction(0),
    )
    # Only words through the shared entry (0, 0) survive; each count is the
    # number of the n! slot permutations in a cyclic class times its weight.
    closed = {
        (0, 1): 2 * a00 * eps,
        (0, 0, 1): 8 * a2_00 * eps,
        (0, 1, 1): 8 * a00 * eps**2,
        (0, 1, 0, 1): 2 * (8 * eps**2 * a00**2 + 16 * eps**2 * a2_00),
        (0, 0, 0, 1): 48 * a3_00 * eps,
    }
    return family, closed


@pytest.mark.parametrize("latent", [4, 100, 500])
# 1e-323 is two subnormal units, the smallest overlap that the family's
# halving symmetrisation preserves.
@pytest.mark.parametrize("overlap", [1e-9, -3e-30, 1e-150, 1e-323])
def test_near_zero_overlap_matches_closed_form(latent, overlap):
    family, closed = _overlap_case(latent, overlap, seed=4400 + latent)
    for slots, exact in closed.items():
        _assert_certified(family.mixed_cumulant(slots), exact)
        if latent <= 100:
            powers = [slots.count(0), slots.count(1)]
            divisor = math.prod(math.factorial(power) for power in powers)
            _assert_certified(family.joint_coefficient(powers), exact / divisor)


def test_overlap_closed_form_matches_independent_fraction_reference():
    family, closed = _overlap_case(5, 1e-9, seed=4405)
    for slots, exact in closed.items():
        assert _fraction_statistic(family._matrices, family._endpoints, slots) == exact


# ---------------------------------------------------------------------------
# Dense cancellation swept across the admission threshold
# ---------------------------------------------------------------------------


def _cancelled_family(latent, order, level, seed, with_endpoints):
    """Second observable tuned so tr(A**(order-1) B) is ``level`` of its
    uncancelled value; diagonal dominance makes level 1 well conditioned."""
    random = np.random.default_rng(seed)
    noise = 0.1 / math.sqrt(latent)
    a = np.diag(random.uniform(0.5, 2.0, latent)) + noise * _symmetric(random, latent)
    base = np.diag(random.uniform(-2.0, 2.0, latent))
    base += noise * _symmetric(random, latent)
    power = np.linalg.matrix_power(a, order - 1)
    natural = np.sum(power * base)
    b = base - ((1 - level) * natural / np.sum(power * power)) * power
    b = b / 2 + b.T / 2
    endpoints = None
    if with_endpoints:
        endpoints = random.normal(size=(2, latent)) * np.array([[1.3], [-0.4]])
    return _family(np.array([a, b]), endpoints)


_SWEEP = [10.0 ** (-k / 2) for k in range(0, 17)] + [0.0, 1e-12, -1e-9]


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("with_endpoints", [False, True])
def test_cancellation_sweep_is_certified_or_correctly_rounded(order, with_endpoints):
    slots = (0,) * (order - 1) + (1,)
    for step, level in enumerate(_SWEEP):
        family = _cancelled_family(
            7, order, level, 4500 + 10 * order + step, with_endpoints
        )
        exact = _fraction_statistic(family._matrices, family._endpoints, slots)
        _assert_certified(family.mixed_cumulant(slots), exact)
        coefficient = family.joint_coefficient([order - 1, 1])
        _assert_certified(coefficient, exact / math.factorial(order - 1))


def test_chain_cancellation_is_certified_or_correctly_rounded():
    random = np.random.default_rng(4590)
    latent = 9
    first = random.normal(size=latent)
    for level in _SWEEP:
        base = random.normal(size=latent)
        second = base - ((1 - level) * (first @ base) / (first @ first)) * first
        family = _family(np.zeros((2, latent, latent)), np.array([first, second]))
        exact = _fraction_statistic(family._matrices, family._endpoints, (0, 1))
        _assert_certified(family.mixed_cumulant((0, 1)), exact)


# ---------------------------------------------------------------------------
# Extreme scales
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "order, exponent",
    [(2, 450), (2, -480), (3, 300), (3, -320), (4, 250), (4, -255), (4, 262)],
)
def test_extreme_scales_round_correctly_or_overflow(order, exponent):
    random = np.random.default_rng(4600 + order + exponent)
    latent = 4
    matrices = np.array([_symmetric(random, latent) for _ in range(3)]) * 2.0**exponent
    endpoints = random.normal(size=(3, latent)) * 2.0 ** (exponent * order / 2)
    family = _family(matrices, endpoints)
    for slots in [(0, 1, 2, 0)[:order], (1,) * order, (2, 0, 2, 1)[:order]]:
        exact = _fraction_statistic(family._matrices, family._endpoints, slots)
        try:
            float(exact)
        except OverflowError:
            with pytest.raises(OverflowError):
                family.mixed_cumulant(slots)
            continue
        _assert_certified(family.mixed_cumulant(slots), exact)


def test_subnormal_and_mixed_magnitude_entries():
    random = np.random.default_rng(4700)
    latent = 5
    matrices = np.array([_symmetric(random, latent) for _ in range(3)])
    matrices[0][1, 3] = matrices[0][3, 1] = 3e-320
    matrices[1] *= 1e150
    matrices[2] *= 1e-150
    # Two subnormal units: preparation halves and re-adds it exactly.
    matrices[2][0, 0] = -1e-323
    endpoints = random.normal(size=(3, latent))
    endpoints[0, 2] = 2e-310
    family = _family(matrices, endpoints)
    for slots in [(0, 1), (1, 2), (0, 1, 2), (2, 2, 1), (0, 1, 2, 2), (1, 2, 1, 2)]:
        exact = _fraction_statistic(family._matrices, family._endpoints, slots)
        _assert_certified(family.mixed_cumulant(slots), exact)


@pytest.mark.parametrize("latent", [1, 3, 100])
def test_underflowed_absolute_contraction_is_not_taken_for_zero(latent):
    # |M0||M1| underflows to zero in float64, yet every word of (0, 1, 2)
    # runs through the (0, 0) entries: kappa = (4/3) * 6 * 1e-200**2 * 1e250.
    matrices = np.zeros((3, latent, latent))
    matrices[0, 0, 0], matrices[1, 0, 0], matrices[2, 0, 0] = 1e-200, 1e-200, 1e250
    matrices[0, 1:, 1:] = np.eye(latent - 1)
    matrices[2, 1:, 1:] = -np.eye(latent - 1) * 0.5
    family = _family(matrices)
    exact = 8 * Fraction(1e-200) ** 2 * Fraction(1e250)
    if latent <= 3:
        assert exact == _fraction_statistic(matrices, family._endpoints, (0, 1, 2))
    value = family.mixed_cumulant((0, 1, 2))
    assert value != 0.0
    _assert_certified(value, exact)


# ---------------------------------------------------------------------------
# Internals: certificate validity and the exact evaluator, bypassing dispatch
# ---------------------------------------------------------------------------


def _float_statistic(matrices, endpoints, slots, divisor=1) -> Fraction:
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


@pytest.mark.parametrize("latent", [3, 7])
@pytest.mark.parametrize(
    "style", ["plain", "wide", "cancel", "tiny", "huge", "nonsymmetric"]
)
def test_exact_evaluator_equals_fraction_reference(latent, style):
    mixed = pytest.importorskip("sft_wick._mixed_contraction")
    random = np.random.default_rng(4800 + latent + len(style))
    matrices = np.array([_symmetric(random, latent) for _ in range(3)])
    endpoints = random.normal(size=(3, latent))
    if style == "wide":
        matrices *= np.exp2(random.integers(-400, 400, size=matrices.shape))
        matrices += matrices.swapaxes(-1, -2)
        endpoints *= np.exp2(random.integers(-400, 400, size=endpoints.shape))
    elif style == "cancel":
        matrices[1] = matrices[0] * (1 + 2.0**-40) - matrices[0]
    elif style == "tiny":
        matrices *= 1e-300
        endpoints *= 1e-310
    elif style == "huge":
        matrices *= 1e300
    elif style == "nonsymmetric":
        matrices += random.normal(size=matrices.shape)
    for slots in [(0, 1), (2, 2), (0, 1, 2), (1, 0, 1), (0, 1, 2, 0), (2, 1, 1, 2)]:
        expected = _fraction_statistic(matrices, endpoints, slots)
        for divisor in (1, 6):
            value = mixed.exact_mixed_statistic(matrices, endpoints, slots, divisor)
            assert value == expected / divisor


def _signed_unit_family(latent, seed, all_positive):
    """M_a = c_a s_a s_a^T and g_a = d_a t_a with sign vectors s_a, t_a."""
    random = np.random.default_rng(seed)
    signs = (
        np.ones((3, latent))
        if all_positive
        else random.choice([-1.0, 1.0], size=(3, latent))
    )
    ends = random.choice([-1.0, 1.0], size=(3, latent))
    # Every mantissa bit set, so every digit slice holds its largest digits.
    full = float(2**53 - 1)
    scales = [full * 2.0**-55, -full * 2.0**-60, full * 2.0**-53]
    end_scales = [full * 2.0**-54, -full * 2.0**-57, full * 2.0**-52]
    matrices = np.array([c * np.outer(s, s) for c, s in zip(scales, signs)])
    endpoints = np.array([d * t for d, t in zip(end_scales, ends)])
    return matrices, endpoints, scales, signs, end_scales, ends


def _rank_one_statistic(scales, signs, end_scales, ends, slots):
    """Closed form for M_a = c_a s_a s_a^T and g_a = d_a t_a (integer dots)."""
    order = len(slots)

    def dot(x, y):
        return int(np.dot(x.astype(np.int64), y.astype(np.int64)))

    ring = chain = Fraction(0)
    for word in permutations(slots):
        value = Fraction(1)
        for position, index in enumerate(word):
            following = word[(position + 1) % order]
            value *= Fraction(scales[index]) * dot(signs[index], signs[following])
        ring += value
        if order == 2:
            link = Fraction(dot(ends[word[0]], ends[word[1]]))
        else:
            link = Fraction(dot(ends[word[0]], signs[word[1]]))
            for left, right in zip(word[1:-2], word[2:-1]):
                link *= dot(signs[left], signs[right])
            link *= dot(signs[word[-2]], ends[word[-1]])
            for index in word[1:-1]:
                link *= Fraction(scales[index])
        ends_product = Fraction(end_scales[word[0]]) * Fraction(end_scales[word[-1]])
        chain += ends_product * link
    return Fraction(2 ** (order - 1), order) * ring + Fraction(2) ** (order - 3) * chain


@pytest.mark.parametrize(
    "latent, slot_sets",
    [
        (256, [(0, 1), (0, 1, 2), (0, 1, 0, 1)]),
        (257, [(0, 1), (0, 1, 2), (0, 1, 0, 1)]),
        (512, [(0, 1), (2, 0, 1)]),
        (513, [(0, 1), (2, 0, 1)]),
    ],
)
@pytest.mark.parametrize("all_positive", [True, False])
def test_exact_evaluator_at_digit_width_boundaries(latent, slot_sets, all_positive):
    mixed = pytest.importorskip("sft_wick._mixed_contraction")
    width = mixed.digit_width(latent)
    # Widest digits whose int64 partial sums cannot overflow; at N = 512,
    # 512 * (2**27 - 1)**2 = 2**63 - 2**37 + 512 is the tightest case.
    assert latent * (2**width - 1) ** 2 < 2**63
    assert latent * (2 ** (width + 1) - 1) ** 2 >= 2**63
    matrices, endpoints, scales, signs, end_scales, ends = _signed_unit_family(
        latent, 4900 + latent, all_positive
    )
    for slots in slot_sets:
        expected = _rank_one_statistic(scales, signs, end_scales, ends, slots)
        assert mixed.exact_mixed_statistic(matrices, endpoints, slots, 1) == expected
        family = _family(matrices, endpoints)
        _assert_certified(family.mixed_cumulant(slots), expected)


@pytest.mark.parametrize("latent", [2, 9, 40])
@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("with_endpoints", [False, True])
def test_certificate_bounds_the_float_error_across_the_threshold(
    latent, order, with_endpoints
):
    mixed = pytest.importorskip("sft_wick._mixed_contraction")
    slots = (0,) * (order - 1) + (1,)
    outcomes = set()
    for step, level in enumerate(_SWEEP):
        family = _cancelled_family(latent, order, level, 5000 + step, with_endpoints)
        matrices, endpoints = family._matrices, family._endpoints
        certificate = mixed.rounding_certificate(matrices, endpoints, slots, 1)
        exact = mixed.exact_mixed_statistic(matrices, endpoints, slots, 1)
        # Bypass the dispatcher: evaluate the float64 method and the exact
        # method separately, at and on both sides of the admission threshold.
        approximate = _float_statistic(matrices, endpoints, slots)
        assert not certificate.zero
        assert abs(approximate - exact) <= certificate.error
        outcomes.add(mixed.admits(certificate, approximate))
    if not with_endpoints:
        # The ring sweep crosses the admission threshold in both directions.
        assert outcomes == {True, False}


def test_exact_evaluator_matches_fraction_on_generic_repeated_slots():
    mixed = pytest.importorskip("sft_wick._mixed_contraction")
    random = np.random.default_rng(5100)
    latent = 6
    matrices = np.array([_symmetric(random, latent) * 10.0**k for k in (-3, 0, 4)])
    endpoints = random.normal(size=(3, latent)) * np.array([[1e-5], [1.0], [1e5]])
    for slots in [(0, 0), (1, 1, 1), (0, 0, 2, 2), (2, 1, 2, 1), (0, 0, 0, 0)]:
        expected = _fraction_statistic(matrices, endpoints, slots)
        for divisor in (1, 2, 4, 24):
            value = mixed.exact_mixed_statistic(matrices, endpoints, slots, divisor)
            assert value == expected / divisor


def test_large_dimension_dispatch_agrees_with_exact_evaluator():
    mixed = pytest.importorskip("sft_wick._mixed_contraction")
    random = np.random.default_rng(5200)
    latent = 300
    data = random.normal(size=(3, latent, latent))
    matrices = data @ data.swapaxes(-1, -2) / latent
    matrices = matrices / 2 + matrices.swapaxes(-1, -2) / 2
    endpoints = random.normal(size=(3, latent)) / 3
    family = _family(matrices, endpoints)
    for slots in [(0, 1), (0, 0, 1), (0, 1, 2), (2, 0, 1, 0)]:
        exact = mixed.exact_mixed_statistic(matrices, endpoints, slots, 1)
        _assert_certified(family.mixed_cumulant(slots), exact)


# ---------------------------------------------------------------------------
# Refusal above the exact work bound
# ---------------------------------------------------------------------------


def test_exact_work_bound_refuses_before_any_product():
    random = np.random.default_rng(5300)
    latent = 512
    raw = random.normal(size=(latent, latent))
    raw *= np.exp2(random.integers(-1000, 1000, size=raw.shape).astype(np.float64))
    wide = raw + raw.T
    family = GaussianQuadraticFamily.from_factor(
        np.eye(latent), np.array([wide, wide[::-1, ::-1].copy()])
    )
    started = time.perf_counter()
    with pytest.raises(ArithmeticError, match="exact contraction work bound"):
        family.mixed_cumulant((0, 0, 1))
    assert time.perf_counter() - started < 10.0


def test_exact_work_limit_is_explicit_and_recorded(monkeypatch):
    mixed = pytest.importorskip("sft_wick._mixed_contraction")
    family = _parity_family(8, seed=5400)
    assert family.mixed_cumulant((0, 0, 1)) == 0.0
    monkeypatch.setattr(mixed, "EXACT_WORK_LIMIT", 10)
    with pytest.raises(ArithmeticError, match="exact contraction work bound") as error:
        family.mixed_cumulant((0, 0, 1))
    selected = family.prepare_statistics(indices=[(0, 0, 1)])
    assert selected._outcomes == ((ArithmeticError, str(error.value)),)
    # A certified structural zero never reaches the exact evaluator.
    _assert_positive_zero(_block_family(30, seed=5401).mixed_cumulant((0, 1, 0)))
