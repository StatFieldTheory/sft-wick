"""Certified mixed cumulants through the public dispatcher.

Every value returned by ``GaussianQuadraticFamily.mixed_cumulant`` and
``joint_coefficient`` is either within a relative error of 1e-12 of the exact
statistic of the prepared matrices (before its final rounding) or the
correctly rounded exact value, and a proved zero is ``+0.0``. References share
no code with the package: plain ``Fraction`` contractions (``mixed_reference``),
closed forms of block, rank-one and reversal-parity structures, and exact
integer sums.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations
import math
import time

import numpy as np
import pytest

from sft_wick import _mixed_contraction as certificates
from sft_wick import _mixed_exact as exact_evaluation
from sft_wick import _mixed_rounding as rounding
from sft_wick import quadratic

from .mixed_reference import (
    SWEEP,
    assert_certified,
    assert_positive_zero,
    cancelled_family,
    dense_family_arrays,
    endpoint_tuned,
    make_family,
    float_statistic,
    fraction_statistic,
    integer_statistic,
    symmetric,
)


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
        return make_family(matrices, endpoints)
    cut = max(1, latent // 3)
    first, second = slice(0, cut), slice(cut, latent)
    for index, block in ((0, first), (1, second), (2, first)):
        width = block.stop - block.start
        matrices[index][block, block] = symmetric(random, width) * (1.5 - index)
        endpoints[index][block] = random.normal(size=width) * (index - 0.7)
    return make_family(matrices, endpoints)


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
        assert_positive_zero(family.mixed_cumulant(slots))
    powers = [(1, 1, 0), (2, 1, 0), (1, 1, 2)] if latent > 1 else [(1, 1, 0)]
    for power in powers:
        assert_positive_zero(family.joint_coefficient(power))
    # A certified structural zero needs no signed dense contraction.
    assert time.perf_counter() - started < 5.0


def test_block_zero_survives_statistic_preparation():
    family = _block_family(100, seed=4199)
    selected = family.prepare_statistics(
        indices=[(0, 1), (0, 2, 0, 1)], coefficients=[(2, 1, 0)]
    )
    assert_positive_zero(selected.mixed_cumulant((0, 2, 0, 1)))
    assert_positive_zero(selected.joint_coefficient((2, 1, 0)))


def test_correlated_block_pair_stays_nonzero_beside_the_zero():
    family = _block_family(6, seed=4142)
    for slots in [(0, 2), (2, 0, 2), (0, 0, 2, 2)]:
        exact = fraction_statistic(family._matrices, family._endpoints, slots)
        assert exact != 0
        assert_certified(family.mixed_cumulant(slots), exact)


# ---------------------------------------------------------------------------
# Structural zeros whose magnitudes withhold the error bound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "latent, scale, spread, slots",
    [
        (40, 1e200, 0, (0, 0, 1, 1)),
        (600, 1e80, 400, (0, 0, 1)),
        (600, 1e-90, 0, (0, 1, 0, 1)),
        (300, 1e80, 300, (0, 0, 0, 1)),
    ],
)
def test_block_zero_is_proved_when_magnitudes_withhold_the_bound(
    latent, scale, spread, slots
):
    # Huge or tiny entries make the overflow guard or the underflow check
    # fail; the zero patterns alone still prove that no term exists.
    random = np.random.default_rng(5600 + latent)
    cut = latent // 2
    matrices = np.zeros((2, latent, latent))
    for index, block in ((0, slice(0, cut)), (1, slice(cut, latent))):
        size = block.stop - block.start
        raw = random.normal(size=(size, size)) * scale
        if spread:
            raw *= np.exp2(random.integers(-spread, spread, size=raw.shape))
        matrices[index][block, block] = raw + raw.T
    family = make_family(matrices)
    certificate = certificates.rounding_certificate(
        family._matrices, family._endpoints, slots, 1
    )
    assert certificate.zero
    assert_positive_zero(family.mixed_cumulant(slots))


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
            even(symmetric(random, latent)),
            odd(symmetric(random, latent) * 0.3),
            even(symmetric(random, latent) * -2.1),
        ]
    )
    raw = random.normal(size=(3, latent))
    endpoints = np.array(
        [raw[0] + raw[0][::-1], raw[1] - raw[1][::-1], (raw[2] + raw[2][::-1]) * 5]
    )
    family = make_family(matrices, endpoints)
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
        assert_positive_zero(family.mixed_cumulant(slots))


def test_parity_even_cumulants_are_nonzero_and_certified():
    family = _parity_family(5, seed=4305)
    for slots in [(1, 1), (0, 1, 1), (2, 0, 1, 1), (1, 1, 1, 1)]:
        exact = fraction_statistic(family._matrices, family._endpoints, slots)
        assert exact != 0
        assert_certified(family.mixed_cumulant(slots), exact)


# ---------------------------------------------------------------------------
# Near zeros: one overlapping entry between otherwise disjoint blocks
# ---------------------------------------------------------------------------


def _overlap_case(latent, overlap, seed):
    random = np.random.default_rng(seed)
    cut = latent // 2
    a = symmetric(random, cut)
    b = symmetric(random, latent - cut)
    matrices = np.zeros((2, latent, latent))
    matrices[0][:cut, :cut] = a
    matrices[1][cut:, cut:] = b
    matrices[1][0, 0] = overlap
    family = make_family(matrices)
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
        assert_certified(family.mixed_cumulant(slots), exact)
        if latent <= 100:
            powers = [slots.count(0), slots.count(1)]
            divisor = math.prod(math.factorial(power) for power in powers)
            assert_certified(family.joint_coefficient(powers), exact / divisor)


def test_overlap_closed_form_matches_independent_fraction_reference():
    family, closed = _overlap_case(5, 1e-9, seed=4405)
    for slots, exact in closed.items():
        assert fraction_statistic(family._matrices, family._endpoints, slots) == exact


# ---------------------------------------------------------------------------
# Dense cancellation swept across the admission threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("with_endpoints", [False, True])
def test_cancellation_sweep_is_certified_or_correctly_rounded(order, with_endpoints):
    slots = (0,) * (order - 1) + (1,)
    for step, level in enumerate(SWEEP):
        family = cancelled_family(
            7, order, level, 4500 + 10 * order + step, with_endpoints
        )
        exact = fraction_statistic(family._matrices, family._endpoints, slots)
        assert_certified(family.mixed_cumulant(slots), exact)
        coefficient = family.joint_coefficient([order - 1, 1])
        assert_certified(coefficient, exact / math.factorial(order - 1))


def test_chain_cancellation_is_certified_or_correctly_rounded():
    random = np.random.default_rng(4590)
    latent = 9
    first = random.normal(size=latent)
    for level in SWEEP:
        base = random.normal(size=latent)
        second = base - ((1 - level) * (first @ base) / (first @ first)) * first
        family = make_family(np.zeros((2, latent, latent)), np.array([first, second]))
        exact = fraction_statistic(family._matrices, family._endpoints, (0, 1))
        assert_certified(family.mixed_cumulant((0, 1)), exact)


def _chain_cancelled_family(latent, order, level, seed):
    """Endpoint g1 tuned so the statistic of (0,)*(order-1) + (1,) cancels."""
    random = np.random.default_rng(seed)
    noise = 0.3 / math.sqrt(latent)
    a = np.diag(random.uniform(0.5, 2.0, latent)) + noise * symmetric(random, latent)
    a = a / 2 + a.T / 2
    b = 1e-3 * symmetric(random, latent)
    endpoints = np.array(
        [1.7 * random.normal(size=latent), -0.6 * random.normal(size=latent)]
    )
    slots = (0,) * (order - 1) + (1,)
    matrices = np.array([a, b])
    return make_family(matrices, endpoint_tuned(matrices, endpoints, slots, 1, level))


@pytest.mark.parametrize("order", [3, 4])
def test_endpoint_chain_cancellation_is_certified_or_correctly_rounded(
    monkeypatch, order
):
    from sft_wick import quadratic

    exact_calls = []
    rounding = quadratic.correctly_rounded_statistic
    evaluate = exact_evaluation.exact_mixed_statistic

    def counted(*arguments):
        exact_calls.append(arguments[2])
        return rounding(*arguments)

    monkeypatch.setattr(quadratic, "correctly_rounded_statistic", counted)
    slots = (0,) * (order - 1) + (1,)
    outcomes = set()
    for latent in (3, 7, 12):
        for step, level in enumerate(SWEEP):
            family = _chain_cancelled_family(
                latent, order, level, 9000 + 100 * latent + 10 * order + step
            )
            matrices, endpoints = family._matrices, family._endpoints
            if latent < 12:
                exact = fraction_statistic(matrices, endpoints, slots)
            else:
                exact = evaluate(matrices, endpoints, slots, 1)
            # Both methods, bypassing the dispatcher.
            certificate = certificates.rounding_certificate(
                matrices, endpoints, slots, 1
            )
            approximate = float_statistic(matrices, endpoints, slots)
            assert abs(approximate - exact) <= certificate.error
            exact_calls.clear()
            assert_certified(family.mixed_cumulant(slots), exact)
            outcomes.add(bool(exact_calls))
            divisor = math.factorial(order - 1)
            assert_certified(family.joint_coefficient([order - 1, 1]), exact / divisor)
    # Both admitted float64 values and exact evaluations occur.
    assert outcomes == {True, False}


# ---------------------------------------------------------------------------
# Extreme scales and underflow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "order, exponent",
    [(2, 450), (2, -480), (3, 300), (3, -320), (4, 250), (4, -255), (4, 262)],
)
def test_extreme_scales_round_correctly_or_overflow(order, exponent):
    random = np.random.default_rng(4600 + order + exponent)
    latent = 4
    matrices = np.array([symmetric(random, latent) for _ in range(3)]) * 2.0**exponent
    endpoints = random.normal(size=(3, latent)) * 2.0 ** (exponent * order / 2)
    family = make_family(matrices, endpoints)
    for slots in [(0, 1, 2, 0)[:order], (1,) * order, (2, 0, 2, 1)[:order]]:
        exact = fraction_statistic(family._matrices, family._endpoints, slots)
        try:
            float(exact)
        except OverflowError:
            with pytest.raises(OverflowError):
                family.mixed_cumulant(slots)
            continue
        assert_certified(family.mixed_cumulant(slots), exact)


def test_subnormal_and_mixed_magnitude_entries():
    random = np.random.default_rng(4700)
    latent = 5
    matrices = np.array([symmetric(random, latent) for _ in range(3)])
    matrices[0][1, 3] = matrices[0][3, 1] = 3e-320
    matrices[1] *= 1e150
    matrices[2] *= 1e-150
    # Two subnormal units: preparation halves and re-adds it exactly.
    matrices[2][0, 0] = -1e-323
    endpoints = random.normal(size=(3, latent))
    endpoints[0, 2] = 2e-310
    family = make_family(matrices, endpoints)
    for slots in [(0, 1), (1, 2), (0, 1, 2), (2, 2, 1), (0, 1, 2, 2), (1, 2, 1, 2)]:
        exact = fraction_statistic(family._matrices, family._endpoints, slots)
        assert_certified(family.mixed_cumulant(slots), exact)


@pytest.mark.parametrize("latent", [1, 3, 100])
def test_underflowed_absolute_contraction_is_not_taken_for_zero(latent):
    # |M0||M1| underflows to zero in float64, yet every word of (0, 1, 2)
    # runs through the (0, 0) entries: kappa = (4/3) * 6 * 1e-200**2 * 1e250.
    matrices = np.zeros((3, latent, latent))
    matrices[0, 0, 0], matrices[1, 0, 0], matrices[2, 0, 0] = 1e-200, 1e-200, 1e250
    matrices[0, 1:, 1:] = np.eye(latent - 1)
    matrices[2, 1:, 1:] = -np.eye(latent - 1) * 0.5
    family = make_family(matrices)
    exact = 8 * Fraction(1e-200) ** 2 * Fraction(1e250)
    if latent <= 3:
        assert exact == fraction_statistic(matrices, family._endpoints, (0, 1, 2))
    value = family.mixed_cumulant((0, 1, 2))
    assert value != 0.0
    assert_certified(value, exact)


def _isolated_endpoint_family(latent, tiny, seed):
    random = np.random.default_rng(seed)
    matrices = []
    for _ in range(3):
        matrix = np.diag(random.uniform(0.5, 2.0, latent))
        matrix += 0.01 * symmetric(random, latent)
        matrix[0, 1:] = matrix[1:, 0] = 0.0
        matrices.append(matrix)
    endpoints = random.normal(size=(3, latent))
    endpoints[:, 0] = tiny * np.array([1.0, -0.5, 3.0])
    return make_family(np.array(matrices), endpoints)


def test_endpoint_underflow_rounds_exactly_or_keeps_the_certified_value(monkeypatch):
    family = _isolated_endpoint_family(6, 1e-170, seed=5700)
    matrices, endpoints = family._matrices, family._endpoints
    slots = (0, 1, 2)
    # The elementwise endpoint product underflows on coordinate 0.
    with np.errstate(under="raise"), pytest.raises(FloatingPointError):
        endpoints[0] * (matrices[1] @ endpoints[2])
    exact = fraction_statistic(matrices, endpoints, slots)
    # Within the exact work bound the value is correctly rounded, as before.
    assert family.mixed_cumulant(slots) == float(exact)
    # Above it, an underflow alone does not discard the certified value.
    with np.errstate(under="ignore"):
        approximate = float_statistic(matrices, endpoints, slots)
    certificate = certificates.rounding_certificate(matrices, endpoints, slots, 1)
    assert certificates.admits(certificate, approximate)
    monkeypatch.setattr(exact_evaluation, "EXACT_WORK_LIMIT", 0)
    value = family.mixed_cumulant(slots)
    assert value == float(approximate)
    assert_certified(value, exact)
    # Without an error bound (here the overflow guard) the refusal stands.
    huge = make_family(family._matrices * 1e154, family._endpoints)
    with pytest.raises(ArithmeticError):
        huge.mixed_cumulant(slots)


def test_only_the_exact_work_refusal_returns_the_certified_value(monkeypatch):
    # The query above reaches the exact evaluation with a certified float64
    # value. A floating-point exception raised there is not the work or
    # memory refusal, so it propagates instead of returning that value.
    family = _isolated_endpoint_family(6, 1e-170, seed=5700)
    slots = (0, 1, 2)

    def trapped(*args):
        raise FloatingPointError("underflow encountered in multiply")

    monkeypatch.setattr(rounding, "evaluate_query", trapped)
    with pytest.raises(FloatingPointError):
        family.mixed_cumulant(slots)
    with pytest.raises(FloatingPointError):
        family.prepare_statistics(indices=[slots])


def _subnormal_tail_family(latent, seed):
    """Entries of 2**20 to 2**60 of both signs, a subnormal last row and column."""
    random = np.random.default_rng(seed)
    matrices = random.normal(size=(3, latent, latent))
    matrices *= 2.0**60 * np.exp2(random.integers(-40, 1, size=matrices.shape))
    matrices = np.triu(matrices) + np.triu(matrices, 1).swapaxes(-1, -2)
    tail = 3e-320 * random.choice([-1.0, 1.0], size=(3, latent))
    matrices[:, -1, :] = tail
    matrices[:, :, -1] = tail
    return make_family(matrices, random.normal(size=(3, latent)))


def _cancelling_subnormal_family(latent, seed):
    """Entries near 1e9 and one subnormal coupling; tr(A0 A1) cancels."""
    random = np.random.default_rng(seed)
    first = symmetric(random, latent) * 1e9
    first[1, -1] = first[-1, 1] = 7.1e-319
    second = symmetric(random, latent) * 0.83
    second -= (np.sum(first * second) / np.sum(first * first)) * first
    return make_family(np.array([first, (second + second.T) / 2]))


@pytest.mark.parametrize("state", [{"under": "raise"}, {"all": "raise"}])
def test_callers_underflow_trap_leaves_every_value_unchanged(monkeypatch, state):
    # Slicing these arrays for the exact evaluation underflows by design. Up to
    # now a caller's np.errstate(under="raise") made the exact path raise
    # FloatingPointError, or return the float64 value of a contraction that
    # underflowed, which version 0.7.0 never kept.
    tail = _subnormal_tail_family(5, seed=2024)
    cancelling = _cancelling_subnormal_family(5, seed=424242)
    queries = [
        (tail, (0, 1)),
        (tail, (0, 1, 2)),
        (tail, (0, 0, 1)),
        (tail, (0, 1, 2, 0)),
        (tail, (2, 2, 2, 2)),
        (cancelling, (0, 1)),
    ]
    original = quadratic._exact_mixed_coefficient
    exact_calls = []

    def counted(*args):
        exact_calls.append(args[2])
        return original(*args)

    monkeypatch.setattr(quadratic, "_exact_mixed_coefficient", counted)
    default = [family.mixed_cumulant(slots) for family, slots in queries]
    joint = tail.joint_coefficient((2, 1, 1))
    assert len(exact_calls) >= 2
    with np.errstate(**state):
        trapped = [family.mixed_cumulant(slots) for family, slots in queries]
        assert tail.joint_coefficient((2, 1, 1)) == joint
        selected = tail.prepare_statistics(
            indices=[slots for family, slots in queries if family is tail],
            coefficients=[(2, 1, 1)],
        )
    assert trapped == default
    for (family, slots), value in zip(queries, default):
        exact = integer_statistic(family._matrices, family._endpoints, slots)
        assert_certified(value, exact)
        if family is tail:
            assert selected.mixed_cumulant(slots) == value
    assert selected.joint_coefficient((2, 1, 1)) == joint
    exact = integer_statistic(cancelling._matrices, cancelling._endpoints, (0, 1))
    assert default[-1] == float(exact)


# ---------------------------------------------------------------------------
# Large dimensions
# ---------------------------------------------------------------------------


def test_large_dimension_dispatch_agrees_with_exact_evaluator():
    random = np.random.default_rng(5200)
    latent = 300
    data = random.normal(size=(3, latent, latent))
    matrices = data @ data.swapaxes(-1, -2) / latent
    matrices = matrices / 2 + matrices.swapaxes(-1, -2) / 2
    endpoints = random.normal(size=(3, latent)) / 3
    family = make_family(matrices, endpoints)
    for slots in [(0, 1), (0, 0, 1), (0, 1, 2), (2, 0, 1, 0)]:
        exact = exact_evaluation.exact_mixed_statistic(matrices, endpoints, slots, 1)
        assert_certified(family.mixed_cumulant(slots), exact)


def test_dense_fourth_order_query_at_large_dimension_returns_a_value():
    # The first exact evaluator refused this query (about 1.2e11 integer
    # multiply-adds against a 1e11 limit); main returned a float64 value.
    latent = 1500
    matrices, endpoints = dense_family_arrays(latent, 4, seed=5810)
    family = make_family(matrices, endpoints)
    value = family.mixed_cumulant((0, 1, 2, 3))
    pairs = {
        (a, b): matrices[a] @ matrices[b] for a in range(4) for b in range(4) if a != b
    }
    ring = sum(
        np.sum(pairs[w[0], w[1]] * pairs[w[2], w[3]].T) for w in permutations(range(4))
    )
    chain = sum(
        endpoints[w[0]] @ pairs[w[1], w[2]] @ endpoints[w[3]]
        for w in permutations(range(4))
    )
    # (2**3 / 4) * ring + 2**1 * chain at order four.
    assert value == pytest.approx(2 * ring + 2 * chain, rel=1e-9, abs=0.0)


# ---------------------------------------------------------------------------
# Decaying couplings: entries over 160 decades
# ---------------------------------------------------------------------------


def _decaying_family(latent, seed):
    """Random-sign matrices times exp(-|i - j| / length), entries down to 1e-160."""
    random = np.random.default_rng(seed)
    index = np.arange(latent)
    decay = np.exp(-np.abs(index[:, None] - index[None, :]) * (370.0 / latent))
    raw = random.normal(size=(4, latent, latent))
    matrices = (raw + raw.swapaxes(-1, -2)) * decay
    matrices *= np.array([1.0, -0.3, 2.5, 0.7])[:, None, None]
    endpoints = random.normal(size=(4, latent)) * decay[0]
    return make_family(matrices, endpoints)


_DECAYING_SLOTS = [(0, 1), (0, 1, 2), (0, 0, 1, 1), (3, 1, 0, 2), (2, 0, 2, 1)]


def test_decaying_couplings_round_correctly_below_the_full_exact_cost(monkeypatch):
    # The slice count of a full exact evaluation follows the binary exponent
    # range of the entries, here at least 20 slices per matrix. With admission
    # switched off every query takes the exact route, and with the work limit
    # at a quarter of its own full evaluation each query is still returned
    # correctly rounded: only the leading bits of each matrix are evaluated
    # and the rest is bounded. Before, every query was refused here.
    latent = 40
    family = _decaying_family(latent, seed=6500)
    matrices, endpoints = family._matrices, family._endpoints
    assert np.min(np.abs(matrices[matrices != 0])) < 1e-150
    width = exact_evaluation.digit_width(latent)
    monkeypatch.setattr(certificates, "MIXED_STATISTIC_RTOL", Fraction(0))
    for slots in _DECAYING_SLOTS:
        # The matrices are symmetric, so reversed words share a class.
        rings = certificates.word_classes(slots, cyclic=True, reversal=True)
        chains = certificates.word_classes(slots, cyclic=False, reversal=True)
        grids = {
            index: exact_evaluation._grid(matrices[index], width) for index in slots
        }
        vectors = {
            index: exact_evaluation._grid(endpoints[index], width) for index in slots
        }
        assert min(len(grid.offsets) for grid in grids.values()) >= 20
        full = exact_evaluation.exact_cost(
            latent, rings, chains, grids, vectors, width, True
        )
        monkeypatch.setattr(exact_evaluation, "EXACT_WORK_LIMIT", full.work // 4)
        with pytest.raises(ArithmeticError, match="exact contraction work bound"):
            exact_evaluation.exact_mixed_statistic(matrices, endpoints, slots, 1)
        exact = integer_statistic(matrices, endpoints, slots)
        assert family.mixed_cumulant(slots) == float(exact)
        powers = tuple(slots.count(index) for index in range(4))
        divisor = math.prod(math.factorial(power) for power in powers)
        assert family.joint_coefficient(powers) == float(exact / divisor)
        selected = family.prepare_statistics(indices=[slots], coefficients=[powers])
        assert selected.mixed_cumulant(slots) == float(exact)
        assert selected.joint_coefficient(powers) == float(exact / divisor)


# ---------------------------------------------------------------------------
# Refusal above the exact work bound
# ---------------------------------------------------------------------------


def test_exact_work_limit_is_explicit_and_recorded(monkeypatch):
    family = _parity_family(8, seed=5400)
    assert family.mixed_cumulant((0, 0, 1)) == 0.0
    monkeypatch.setattr(exact_evaluation, "EXACT_WORK_LIMIT", 10)
    with pytest.raises(ArithmeticError, match="exact contraction work bound") as error:
        family.mixed_cumulant((0, 0, 1))
    selected = family.prepare_statistics(indices=[(0, 0, 1)])
    assert selected._outcomes == ((ArithmeticError, str(error.value)),)
    # A certified structural zero never reaches the exact evaluator.
    assert_positive_zero(_block_family(30, seed=5401).mixed_cumulant((0, 1, 0)))
