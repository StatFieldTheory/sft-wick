"""Exact evaluator and rounding certificate of mixed quadratic contractions.

Both methods are evaluated directly, bypassing the dispatcher: the exact
evaluator against independent ``Fraction`` and Python-int references
(``mixed_reference``) and rank-one closed forms, at the float64 digit-width
boundaries, with narrow digits and with a corrupted BLAS product; the
certificate against the formula of its docstring and against the actual
float64 error on both sides of the admission threshold.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations
import math

import numpy as np
import pytest

from sft_wick import _mixed_contraction as certificates
from sft_wick import _mixed_exact as exact_evaluation
from sft_wick import _mixed_rounding as rounding

from .mixed_reference import (
    SWEEP,
    UNIT,
    assert_certified,
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
# Exact evaluator against independent references
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("latent", [3, 7])
@pytest.mark.parametrize(
    "style", ["plain", "wide", "cancel", "tiny", "huge", "nonsymmetric"]
)
def test_exact_evaluator_equals_fraction_reference(latent, style):
    random = np.random.default_rng(4800 + latent + len(style))
    matrices = np.array([symmetric(random, latent) for _ in range(3)])
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
        expected = fraction_statistic(matrices, endpoints, slots)
        for divisor in (1, 6):
            value = exact_evaluation.exact_mixed_statistic(
                matrices, endpoints, slots, divisor
            )
            assert value == expected / divisor


def test_exact_evaluator_matches_fraction_on_generic_repeated_slots():
    random = np.random.default_rng(5100)
    latent = 6
    matrices = np.array([symmetric(random, latent) * 10.0**k for k in (-3, 0, 4)])
    endpoints = random.normal(size=(3, latent)) * np.array([[1e-5], [1.0], [1e5]])
    for slots in [(0, 0), (1, 1, 1), (0, 0, 2, 2), (2, 1, 2, 1), (0, 0, 0, 0)]:
        expected = fraction_statistic(matrices, endpoints, slots)
        for divisor in (1, 2, 4, 24):
            value = exact_evaluation.exact_mixed_statistic(
                matrices, endpoints, slots, divisor
            )
            assert value == expected / divisor


def test_integer_reference_agrees_with_fraction_reference():
    random = np.random.default_rng(5250)
    matrices = np.array([symmetric(random, 4) for _ in range(3)])
    matrices *= np.exp2(random.integers(-20, 20, size=matrices.shape))
    endpoints = random.normal(size=(3, 4))
    for slots in [(0, 1), (2, 0, 1), (2, 0, 1, 0)]:
        expected = fraction_statistic(matrices, endpoints, slots)
        assert integer_statistic(matrices, endpoints, slots) == expected


@pytest.mark.parametrize("slots", [(2, 0, 1, 0), (2, 0, 1, 1), (2, 1, 0), (2, 0)])
def test_uncertified_dense_query_matches_an_independent_integer_reference(slots):
    # Dense indefinite matrices with entries spread over 40 binades (several
    # digit slices per matrix and carries between them), and endpoint 2 tuned
    # so the statistic cancels to 1e-7 of its size: the query is not
    # certified, the dispatcher evaluates it exactly, and the reference
    # shares no code with that evaluation.
    random = np.random.default_rng(5260)
    latent = 48
    raw = random.normal(size=(3, latent, latent))
    raw *= np.exp2(random.integers(-20, 20, size=raw.shape))
    matrices = (raw + raw.swapaxes(-1, -2)) / math.sqrt(latent)
    endpoints = random.normal(size=(3, latent)) * np.array([[1e-3], [1.0], [7.0]])
    endpoints = endpoint_tuned(matrices, endpoints, slots, 2, 1e-7)
    family = make_family(matrices, endpoints)
    certificate = certificates.rounding_certificate(matrices, endpoints, slots, 1)
    approximate = float_statistic(matrices, endpoints, slots)
    assert not certificates.admits(certificate, approximate)
    expected = integer_statistic(matrices, endpoints, slots)
    assert (
        exact_evaluation.exact_mixed_statistic(matrices, endpoints, slots, 1)
        == expected
    )
    assert family.mixed_cumulant(slots) == float(expected)


# ---------------------------------------------------------------------------
# The caller's floating-point error state
# ---------------------------------------------------------------------------


def _wide_range_arrays(kind, latent, seed):
    """Symmetric arrays whose slicing scales small entries below float64 range.

    ``subnormal_tail``: entries of 2**20 to 2**60 of both signs with a
    subnormal last row and column. ``large_and_subnormal``: entries near 1e10
    and one subnormal coupling. ``wide``: entries near 1e60 and 1e-300 in one
    matrix. Endpoints are generic, with one subnormal coordinate.
    """
    random = np.random.default_rng(seed)
    matrices = random.normal(size=(3, latent, latent))
    if kind == "subnormal_tail":
        matrices *= 2.0**60 * np.exp2(random.integers(-40, 1, size=matrices.shape))
    elif kind == "large_and_subnormal":
        matrices *= 1e10
    else:
        matrices *= np.where(random.random(matrices.shape) < 0.5, 1e60, 1e-300)
    matrices = np.triu(matrices) + np.triu(matrices, 1).swapaxes(-1, -2)
    if kind == "subnormal_tail":
        tail = 3e-320 * random.choice([-1.0, 1.0], size=(3, latent))
        matrices[:, -1, :] = tail
        matrices[:, :, -1] = tail
    elif kind == "large_and_subnormal":
        matrices[:, 1, -1] = matrices[:, -1, 1] = [7.1e-319, -3e-320, 2.2e-310]
    endpoints = random.normal(size=(3, latent)) * np.array([[1.9], [-0.6], [2.4]])
    endpoints[:, 0] = [5e-322, -1.3e-310, 2e-315]
    return matrices, endpoints


def _slicing_underflows(array, width):
    """True when slicing ``array`` scales an entry to an inexact subnormal value.

    The first slicing step scales every entry by the top slice unit, so the
    smallest entry underflows there.
    """
    grid = exact_evaluation._grid(array, width)
    top = grid.base + grid.offsets[-1] * width
    smallest = Fraction(float(np.min(np.abs(array[array != 0]))))
    scaled = smallest / Fraction(2) ** top
    return scaled < Fraction(2) ** -1022 and Fraction(float(scaled)) != scaled


@pytest.mark.parametrize("kind", ["subnormal_tail", "large_and_subnormal", "wide"])
@pytest.mark.parametrize("state", [{"under": "raise"}, {"all": "raise"}])
def test_exact_evaluator_is_independent_of_the_callers_error_state(kind, state):
    # Slicing scales every remainder by its slice unit, and a remainder far
    # below the unit underflows to the digit zero by design. Up to now a
    # caller's np.errstate(under="raise") turned that into FloatingPointError.
    latent = 5
    matrices, endpoints = _wide_range_arrays(kind, latent, seed=5270 + len(kind))
    width = exact_evaluation.digit_width(latent)
    assert all(_slicing_underflows(matrix, width) for matrix in matrices)
    for slots in [(0, 1), (0, 1, 2), (1, 1, 2), (0, 1, 2, 0), (2, 0, 1, 1)]:
        expected = integer_statistic(matrices, endpoints, slots)
        with np.errstate(**state):
            value = exact_evaluation.exact_mixed_statistic(
                matrices, endpoints, slots, 2
            )
            rounded = rounding.correctly_rounded_statistic(
                matrices, endpoints, slots, 2, abs(expected) / 2 or None
            )
        assert value == expected / 2
        assert float(rounded) == float(expected / 2)


# ---------------------------------------------------------------------------
# Digit slices: width boundaries, narrow widths, BLAS checksum
# ---------------------------------------------------------------------------


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
        (128, [(0, 1), (0, 1, 2), (0, 1, 0, 1), (2, 0, 1, 0)]),
        (129, [(0, 1), (0, 1, 2), (0, 1, 0, 1)]),
        (512, [(0, 1), (2, 0, 1), (0, 1, 0, 1)]),
        (513, [(0, 1), (2, 0, 1)]),
    ],
)
@pytest.mark.parametrize("all_positive", [True, False])
# The widest digits must make every BLAS slice product exact: a checksum
# mismatch would only warn and recompute, so it fails this test.
@pytest.mark.filterwarnings("error:a float64 matrix product:RuntimeWarning")
def test_exact_evaluator_at_digit_width_boundaries(latent, slot_sets, all_positive):
    width = exact_evaluation.digit_width(latent)
    # Widest digits whose float64 partial sums are exact integers; at N = 512,
    # 512 * (2**22 - 1)**2 = 2**53 - 2**32 + 512, and at N = 128,
    # 128 * (2**23 - 1)**2 = 2**53 - 2**31 + 128, are the tightest cases.
    assert latent * (2**width - 1) ** 2 < 2**53
    assert latent * (2 ** (width + 1) - 1) ** 2 >= 2**53
    matrices, endpoints, scales, signs, end_scales, ends = _signed_unit_family(
        latent, 4900 + latent, all_positive
    )
    for slots in slot_sets:
        expected = _rank_one_statistic(scales, signs, end_scales, ends, slots)
        assert (
            exact_evaluation.exact_mixed_statistic(matrices, endpoints, slots, 1)
            == expected
        )
        family = make_family(matrices, endpoints)
        assert_certified(family.mixed_cumulant(slots), expected)


@pytest.mark.parametrize("width", [2, 3, 5, 8, 13, 21, 25])
def test_exact_evaluator_with_narrow_digit_slices(monkeypatch, width):
    # Narrow digits put every entry, with exponents spread over 300 binades,
    # across many slices; the result must not depend on the width.
    random = np.random.default_rng(5270 + width)
    latent = 5
    assert latent * (2**width - 1) ** 2 < 2**53
    # Elementwise exponents make matrices 0 and 2 nonsymmetric; matrix 1 is
    # symmetrised again, so both evaluation routes are exercised.
    matrices = np.array([symmetric(random, latent) for _ in range(3)])
    matrices *= np.exp2(random.integers(-30, 30, size=matrices.shape))
    matrices[1] = matrices[1] + matrices[1].T
    endpoints = random.normal(size=(3, latent))
    endpoints *= np.exp2(random.integers(-30, 30, size=endpoints.shape))
    monkeypatch.setattr(exact_evaluation, "digit_width", lambda _: width)
    for slots in [(0, 1), (1, 1), (0, 1, 2), (2, 2, 0), (0, 1, 2, 1), (1, 0, 0, 1)]:
        expected = fraction_statistic(matrices, endpoints, slots)
        assert (
            exact_evaluation.exact_mixed_statistic(matrices, endpoints, slots, 1)
            == expected
        )


def test_inexact_blas_slice_product_is_detected_and_recomputed(monkeypatch):
    random = np.random.default_rng(5280)
    latent = 6
    matrices = np.array([symmetric(random, latent) for _ in range(3)])
    endpoints = random.normal(size=(3, latent))
    slots = (0, 1, 2, 0)
    expected = fraction_statistic(matrices, endpoints, slots)
    original = exact_evaluation._blas_product
    calls = []

    def perturbed(left, right):
        product = original(left, right)
        calls.append(product.shape)
        product[1, 2] += 2
        return product

    monkeypatch.setattr(exact_evaluation, "_blas_product", perturbed)
    with pytest.warns(RuntimeWarning, match="was not exact"):
        value = exact_evaluation.exact_mixed_statistic(matrices, endpoints, slots, 1)
    assert calls
    assert value == expected


# ---------------------------------------------------------------------------
# Rounding certificate: the documented formula, and validity at the threshold
# ---------------------------------------------------------------------------


def _gamma(count: int) -> Fraction:
    return Fraction(count) * UNIT / (1 - count * UNIT)


def _documented_error(matrices, endpoints, slots, divisor) -> Fraction:
    """The bound of _mixed_contraction's docstring, summed over all n! words."""
    order, latent = len(slots), matrices.shape[-1]
    absolute = {index: np.abs(matrices[index]) for index in set(slots)}
    vectors = {index: np.abs(endpoints[index]) for index in set(slots)}
    ring = chain = Fraction(0)
    for word in permutations(slots):
        product = np.eye(latent)
        for index in word:
            product = product @ absolute[index]
        ring += Fraction(float(np.trace(product)))
        vector = vectors[word[-1]]
        for index in reversed(word[1:-1]):
            vector = absolute[index] @ vector
        chain += Fraction(float(vectors[word[0]] @ vector))

    def norm(array):
        return max(array.sum(axis=0).max(), array.sum(axis=1).max())

    scales = [max(Fraction(1), 2 * Fraction(float(norm(absolute[i])))) for i in slots]
    endpoint = max(Fraction(float(vector.sum())) for vector in vectors.values())
    scales += [max(Fraction(1), 2 * endpoint)] * 2
    restore = 1 + _gamma((order - 1) * latent + 1)
    growth = 1 + _gamma(latent)
    ring_coefficient = (growth ** (order - 1) - 1) * restore
    chain_coefficient = (growth ** (order - 2) * (1 + UNIT) - 1) * restore
    allowance = (
        math.factorial(order)
        * 64
        * order
        * latent**2
        * Fraction(1, 2**1022)
        * math.prod(scales)
    )
    leading, endpoint_weight = (
        Fraction(2 ** (order - 1), order),
        Fraction(2) ** (order - 3),
    )
    return (
        leading * (ring_coefficient * ring + allowance)
        + endpoint_weight * (chain_coefficient * chain + allowance)
    ) / divisor


@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize(
    "slots, divisor",
    [
        ((0, 1), 1),
        ((1, 1), 2),
        ((0, 1, 2), 1),
        ((0, 0, 1), 2),
        ((2, 0, 1, 3), 1),
        ((0, 0, 1, 1), 4),
        ((2, 0, 1, 0), 2),
        ((3, 3, 3, 3), 24),
    ],
)
def test_certificate_equals_the_documented_bound(symmetric, slots, divisor):
    # Small integers keep every absolute product, trace and norm exact in
    # float64, so the certificate must equal the formula exactly; this pins
    # gamma_N, the (1 - u)**-k restoration, the class weights (here summed
    # over all n! words without classes) and the underflow allowance.
    random = np.random.default_rng(5500 + len(slots) + divisor + symmetric)
    matrices = random.integers(-4, 5, size=(4, 3, 3)).astype(np.float64)
    if symmetric:
        matrices = matrices + matrices.swapaxes(-1, -2)
    endpoints = random.integers(-3, 4, size=(4, 3)).astype(np.float64)
    certificate = certificates.rounding_certificate(matrices, endpoints, slots, divisor)
    assert not certificate.zero
    assert certificate.error == _documented_error(matrices, endpoints, slots, divisor)


@pytest.mark.parametrize("latent", [2, 9, 40])
@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("with_endpoints", [False, True])
def test_certificate_bounds_the_float_error_across_the_threshold(
    latent, order, with_endpoints
):
    slots = (0,) * (order - 1) + (1,)
    outcomes = set()
    for step, level in enumerate(SWEEP):
        family = cancelled_family(latent, order, level, 5000 + step, with_endpoints)
        matrices, endpoints = family._matrices, family._endpoints
        certificate = certificates.rounding_certificate(matrices, endpoints, slots, 1)
        exact = exact_evaluation.exact_mixed_statistic(matrices, endpoints, slots, 1)
        # Bypass the dispatcher: evaluate the float64 method and the exact
        # method separately, at and on both sides of the admission threshold.
        approximate = float_statistic(matrices, endpoints, slots)
        assert not certificate.zero
        assert abs(approximate - exact) <= certificate.error
        outcomes.add(certificates.admits(certificate, approximate))
    if not with_endpoints:
        # The ring sweep crosses the admission threshold in both directions.
        assert outcomes == {True, False}


# ---------------------------------------------------------------------------
# Exact work and memory estimate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("latent, slots", [(1600, (0, 1, 2, 3)), (2600, (0, 1, 2))])
def test_dense_exact_cost_is_within_the_limits(latent, slots):
    # The estimate depends on the slice grids only; every slot gets the grid
    # of one dense matrix with full 53-bit mantissas.
    matrices, endpoints = dense_family_arrays(latent, 1, seed=5800)
    width = exact_evaluation.digit_width(latent)
    rings = certificates.word_classes(slots, cyclic=True, reversal=True)
    chains = certificates.word_classes(slots, cyclic=False, reversal=True)
    grid = exact_evaluation._grid(matrices[0], width)
    vector = exact_evaluation._grid(endpoints[0], width)
    assert len(grid.offsets) >= 3
    grids = {index: grid for index in set(slots)}
    vectors = {index: vector for index in set(slots)}
    cost = exact_evaluation.exact_cost(
        latent, rings, chains, grids, vectors, width, True
    )
    assert cost.work <= exact_evaluation.EXACT_WORK_LIMIT
    assert cost.memory <= exact_evaluation.EXACT_MEMORY_LIMIT
