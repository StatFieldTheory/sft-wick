"""Correct rounding of mixed statistics from arrays cut to their leading bits.

``_mixed_rounding`` evaluates exactly the statistic of arrays cut to their
leading digit slices, bounds the dropped part, and keeps the value when both
ends of the bound round to the same float64. These tests bypass the dispatcher:
the dropped part against its bound with an independent Python-int reference
(``mixed_reference.integer_statistic``) on graded, signed, decaying and
extreme-scale arrays; the cut placement at its slice boundary; the bound
against a closed form; the fall-back chain at a rounding tie; wrong magnitude
estimates; and a work limit that refuses the full evaluation but not the cut
one.
"""

from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

from sft_wick import _mixed_exact as exact_evaluation
from sft_wick import _mixed_rounding as rounding

from .mixed_reference import float_statistic, integer_statistic


def _graded_arrays(kind, latent, seed):
    random = np.random.default_rng(seed)
    if kind == "positive":
        matrices = random.random((3, latent, latent))
        endpoints = random.random((3, latent))
    else:
        matrices = random.normal(size=(3, latent, latent))
        endpoints = random.normal(size=(3, latent))
    index = np.arange(latent)
    if kind == "decaying":
        distance = np.abs(index[:, None] - index[None, :])
        matrices *= np.exp(-distance * random.uniform(20.0, 60.0, size=(3, 1, 1)))
        endpoints *= np.exp(-index * 30.0)
    elif kind == "row_graded":
        matrices *= np.exp2(random.integers(-200, 200, size=(3, latent, 1)))
    elif kind in ("huge_with_subnormal_tail", "subnormal_tail"):
        scale = 1e300 if kind == "huge_with_subnormal_tail" else 2.0**60
        matrices *= scale * np.exp2(random.integers(-40, 1, size=matrices.shape))
        endpoints *= np.exp2(random.integers(-1070, -900, size=endpoints.shape))
        endpoints[:, 0] = random.normal(size=3)
    else:
        matrices *= np.exp2(random.integers(-80, 1, size=matrices.shape))
        matrices *= np.exp2(random.integers(-300, 300, size=(3, 1, 1)))
        endpoints *= np.exp2(random.integers(-300, 300, size=(3, 1)))
    if kind != "row_graded":
        matrices = np.triu(matrices) + np.triu(matrices, 1).swapaxes(-1, -2)
    if kind in ("huge_with_subnormal_tail", "subnormal_tail") and latent > 1:
        # Subnormal entries in the last row and column, of every sign.
        tail = 3e-320 * random.choice([-1.0, 1.0], size=(3, latent))
        matrices[:, -1, :] = tail
        matrices[:, :, -1] = tail
    return matrices, endpoints


_KINDS = [
    "positive",
    "signed",
    "decaying",
    "row_graded",
    "huge_with_subnormal_tail",
    "subnormal_tail",
]
_SLOTS = [(0, 1), (0, 1, 2), (1, 1, 2), (0, 1, 2, 0), (2, 2, 2, 2)]


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("latent", [1, 2, 5, 13])
def test_dropped_part_is_within_its_bound(kind, latent):
    # Magnitude estimates far above the statistic and small guards cut deep,
    # so the dropped part is a visible fraction of the statistic.
    random = np.random.default_rng(6100 + latent)
    cuts = 0
    for seed in range(4):
        matrices, endpoints = _graded_arrays(kind, latent, 6000 + 10 * latent + seed)
        for slots in _SLOTS:
            divisor = int(random.choice([1, 2, 6]))
            query = exact_evaluation.exact_query(matrices, endpoints, slots)
            sizes = rounding._sizes(query)
            assert sizes is not None
            weights = rounding._weights(slots, latent, sizes, divisor)
            exact = integer_statistic(matrices, endpoints, slots) / divisor
            for exponent, guard in [(0, 0), (40, -10), (-60, 30), (900, 5)]:
                magnitude = (abs(exact) or Fraction(1)) * Fraction(2) ** exponent
                truncation = rounding._truncation(
                    query, sizes, weights, magnitude, guard
                )
                if truncation is None:
                    continue
                cuts += 1
                kept = exact_evaluation.evaluate_query(
                    query, truncation.grids, truncation.vector_grids, divisor
                )
                assert 0 < truncation.bound
                assert abs(exact - kept) <= truncation.bound
    assert cuts


def _reconstructed(array, grid, width):
    """Exact entries kept by slicing ``array`` on ``grid``, as Fractions."""
    limbs = exact_evaluation._limbs(array, grid, width)
    kept = np.full(array.shape, Fraction(0), dtype=object)
    for offset, digits in zip(limbs.offsets, limbs.digits):
        unit = Fraction(2) ** (limbs.base + offset * width)
        kept = kept + digits.astype(object) * unit
    return kept


@pytest.mark.parametrize("shift", [-1, 0, 1, 2])
def test_cut_keeps_the_bits_above_its_unit_at_the_slice_boundary(shift):
    # Entries over 300 binades, one of them subnormal; the cut is placed at,
    # just below and just above the lowest occupied offset.
    random = np.random.default_rng(6200)
    array = random.normal(size=(4, 4)) * np.exp2(random.integers(-150, 150, (4, 4)))
    array[3, 2] = -5e-324 * 7
    width = exact_evaluation.digit_width(4)
    grid = exact_evaluation._grid(array, width)
    largest = float(np.max(np.abs(array)))
    exponent = math.frexp(largest)[1] - 1
    lowest = grid.offsets[0] + shift
    binades = exponent - grid.base - lowest * width
    cut, size = rounding._cut(grid, largest, binades, width)
    if shift <= 0:
        assert cut == grid and size == 0
        return
    assert cut.offsets == tuple(offset for offset in grid.offsets if offset >= lowest)
    assert size == Fraction(2) ** (grid.base + lowest * width)
    assert size <= Fraction(largest) * Fraction(2) ** -binades
    kept = _reconstructed(array, cut, width)
    for value, part in zip(array.reshape(-1), kept.reshape(-1)):
        dropped = Fraction(float(value)) - part
        assert abs(dropped) < size
        assert dropped * Fraction(float(value)) >= 0
        assert abs(part) <= abs(Fraction(float(value)))


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("latent", [1, 3, 8])
def test_bound_matches_closed_form_for_coherent_dropped_parts(order, latent):
    # M = (1 + t) J with t = 2**-15 - 2**-40 just below the cut unit, so every
    # dropped part is nearly eps and has the sign of its entry. With zero
    # endpoints the statistic is 2**(n-1) (n-1)! N**n (1 + t)**n, the cut
    # keeps 2**(n-1) (n-1)! N**n, and the module's bound is
    # 2**(n-1) (n-1)! n N eps (2 N (1 + t))**(n-1), with 2 N (1 + t) the
    # doubled row sum.
    tail = 2.0**-15 - 2.0**-40
    matrices = np.full((1, latent, latent), 1.0 + tail)
    endpoints = np.zeros((1, latent))
    slots = (0,) * order
    query = exact_evaluation.exact_query(matrices, endpoints, slots)
    sizes = rounding._sizes(query)
    weights = rounding._weights(slots, latent, sizes, 1)
    # A magnitude 2**-10 of the scale puts the cut ten binades below the
    # largest entry, which drops the lowest slice and with it t.
    magnitude = Fraction(1.0 + tail) * weights[0][0] * Fraction(2) ** -10
    truncation = rounding._truncation(query, sizes, weights, magnitude, 0)
    assert truncation is not None
    assert query.grids[0].base == -40
    eps = Fraction(2) ** (-40 + query.width)
    assert Fraction(tail) < eps < Fraction(tail) * 2**12
    one = Fraction(1)
    leading = 2 ** (order - 1) * math.factorial(order - 1)
    exact = leading * latent**order * (one + Fraction(tail)) ** order
    assert exact == integer_statistic(matrices, endpoints, slots)
    kept = exact_evaluation.evaluate_query(
        query, truncation.grids, truncation.vector_grids, 1
    )
    assert kept == leading * latent**order
    norm = 2 * latent * (one + Fraction(tail))
    assert truncation.bound == leading * order * latent * eps * norm ** (order - 1)
    # The drop is at least n t times the kept part, so the bound exceeds it
    # by at most the norm doubling 2**(n-1), eps / t and (1 + t)**(n-1).
    drop = exact - kept
    excess = (
        2 ** (order - 1)
        * (eps / Fraction(tail))
        * (one + Fraction(tail)) ** (order - 1)
    )
    assert drop <= truncation.bound <= excess * drop


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_rounding_tie_falls_back_to_the_full_evaluation(monkeypatch, sign):
    # tr(A B) = 1 + 2**-52 + 2**-53 + sign * 2**-400, and twice it is within
    # 2**-399 of the midpoint between two float64 values. Both cut attempts
    # drop the 2**-200 entries that carry the tail, so neither can decide the
    # rounding; the full arrays round up for the positive tail and down for
    # the negative one.
    matrices = np.zeros((2, 3, 3))
    matrices[0] = np.diag([1.0, 2.0**-26, 2.0**-200])
    matrices[1] = np.diag([1.0 + 2.0**-52, 2.0**-27, sign * 2.0**-200])
    endpoints = np.zeros((2, 3))
    half = Fraction(2) ** -52 + Fraction(2) ** -53
    exact = 2 * (1 + half + int(sign) * Fraction(2) ** -400)
    assert exact == integer_statistic(matrices, endpoints, (0, 1))
    calls = []
    original = rounding.evaluate_query

    def counted(query, grids, vector_grids, divisor):
        calls.append(sum(len(grid.offsets) for grid in grids.values()))
        return original(query, grids, vector_grids, divisor)

    monkeypatch.setattr(rounding, "evaluate_query", counted)
    value = rounding.correctly_rounded_statistic(
        matrices, endpoints, (0, 1), 1, Fraction(2)
    )
    assert float(value) == float(exact)
    assert float(exact) == (2 + 2.0**-50 if sign > 0 else 2 + 2.0**-51)
    assert len(calls) == 3 and calls[0] <= calls[1] < calls[2]


@pytest.mark.parametrize("kind", ["decaying", "signed", "subnormal_tail"])
@pytest.mark.parametrize("estimate", ["exact", "far_above", "far_below", "none"])
def test_wrong_magnitude_estimates_still_round_correctly(kind, estimate):
    matrices, endpoints = _graded_arrays(kind, 9, seed=6300)
    for slots in _SLOTS:
        exact = integer_statistic(matrices, endpoints, slots)
        magnitude = {
            "exact": abs(exact),
            "far_above": abs(exact) * Fraction(2) ** 200,
            "far_below": abs(exact) * Fraction(2) ** -200,
            "none": None,
        }[estimate]
        value = rounding.correctly_rounded_statistic(
            matrices, endpoints, slots, 1, magnitude
        )
        rounded, expected = float(value), float(exact)
        assert rounded == expected
        assert math.copysign(1.0, rounded) == math.copysign(1.0, expected)


def _decaying(latent, seed):
    """Random-sign matrices times exp(-|i - j| / length), entries down to 1e-160."""
    random = np.random.default_rng(seed)
    index = np.arange(latent)
    decay = np.exp(-np.abs(index[:, None] - index[None, :]) * (370.0 / latent))
    raw = random.normal(size=(4, latent, latent))
    matrices = (raw + raw.swapaxes(-1, -2)) * decay
    matrices *= np.array([1.0, -0.3, 2.5, 0.7])[:, None, None]
    endpoints = random.normal(size=(4, latent)) * decay[0]
    return matrices, endpoints


@pytest.mark.parametrize("slots", [(0, 1), (0, 1, 2), (0, 0, 1, 1), (3, 1, 0, 2)])
def test_cut_evaluation_runs_where_the_full_evaluation_is_refused(monkeypatch, slots):
    latent = 40
    matrices, endpoints = _decaying(latent, seed=6400)
    assert np.min(np.abs(matrices[matrices != 0])) < 1e-150
    query = exact_evaluation.exact_query(matrices, endpoints, slots)
    approximate = float_statistic(matrices, endpoints, slots)
    sizes = rounding._sizes(query)
    weights = rounding._weights(slots, latent, sizes, 1)
    truncation = rounding._truncation(
        query, sizes, weights, abs(approximate), rounding._GUARD_BITS[0]
    )

    def work(grids, vector_grids):
        return exact_evaluation.exact_cost(
            latent,
            query.rings,
            query.chains,
            grids,
            vector_grids,
            query.width,
            query.symmetric,
        ).work

    full = work(query.grids, query.vector_grids)
    cut = work(truncation.grids, truncation.vector_grids)
    assert 10 * cut < full
    monkeypatch.setattr(exact_evaluation, "EXACT_WORK_LIMIT", 2 * cut)
    with pytest.raises(ArithmeticError, match="exact contraction work bound"):
        exact_evaluation.exact_mixed_statistic(matrices, endpoints, slots, 1)
    value = rounding.correctly_rounded_statistic(
        matrices, endpoints, slots, 1, abs(approximate)
    )
    assert float(value) == float(integer_statistic(matrices, endpoints, slots))


def test_rounding_test_compares_bits_and_range():
    tiny = Fraction(1, 2**1100)
    assert rounding._same_rounding(tiny, 2 * tiny)
    assert not rounding._same_rounding(-tiny, tiny)
    assert not rounding._same_rounding(Fraction(0), -tiny)
    largest = Fraction(np.finfo(float).max)
    beyond = Fraction(2) ** 1024
    assert rounding._same_rounding(beyond, 2 * beyond)
    assert not rounding._same_rounding(largest, beyond)
    assert not rounding._same_rounding(-beyond, beyond)
