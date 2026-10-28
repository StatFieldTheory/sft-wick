"""Admission rules of the opt-in contraction backends, checked one at a time.

Each test isolates one rule of :mod:`sft_wick._dense_contraction` that the
error-bound tests reach only together with others:

* the smallest nonzero magnitude of an array holding both signs (the
  bit-pattern path of ``finite_range``), against a reference that sorts
  the magnitudes in Python, and the underflow declines that depend on it
  when that entry is negative;
* the headroom of the sum over assignments on both sides of the overflow
  limit, and an ordered sum that overflows under a strict error policy;
* the structural zeros (a diagonal C table between two different fixed
  components, and two pins of one index that disagree), which make both
  backends decline so the call returns the ordered zeros;
* the factor-sign proof for monomials of opposite sign whose tensors each
  hold one sign, below and beyond the work limit of the per-assignment
  check.
"""

from dataclasses import replace

import numpy as np
import pytest

import sft_wick._dense_contraction as dense
import sft_wick._factorized_coupling as factorized
from sft_wick._dense_contraction import finite_range, try_static_contraction
from sft_wick._factorized_coupling import factorized_state, try_build_factorized
from sft_wick.evaluate import analyze_spatial
from sft_wick.expressions import Product, Propagator, Rational, Symbol
from sft_wick.perturbation import DiagramTerm

from . import _contraction_exact as exact
from .test_contraction_error_bound import _uniform_sum

LIMIT = 64 * 1024**2
SAMPLES = 6


@pytest.fixture
def admit_all(monkeypatch):
    """Run the backend on small fixtures, whatever the cost rule estimates."""
    monkeypatch.setattr(
        dense, "backend_is_cheaper", lambda backend, ordered: True, raising=False
    )


def _signed(rng, shape, low=0.2, high=1.3):
    """Values of both signs, bounded away from zero, no symmetry."""
    return rng.uniform(low, high, shape) * rng.choice((-1.0, 1.0), shape)


def _einsum(ig, cb, rp, jac, rb):
    return try_static_contraction(
        ig.coupling_array,
        ig.diagram_term.propagator_indices,
        ig.fixed_indices,
        cb,
        rp,
        jac,
        rb,
        ig._e_psi,
        LIMIT,
    )


def _ordered(ig, cb, rp, jac, rb):
    return replace(ig, _contraction_backend="ordered")._static_values(
        cb, rp, jac, r_batches=rb
    )


def _outcome(compute):
    """The value, or the type and message of the floating-point error."""
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            return compute()
        except FloatingPointError as error:
            return type(error), str(error)


@pytest.mark.parametrize(
    "smallest",
    [-(2.0**-1000), 2.0**-1000, -5e-324, 5e-324, -np.finfo(float).tiny, -0.7],
)
@pytest.mark.parametrize("zeros", [False, True])
def test_finite_range_of_an_array_with_both_signs(smallest, zeros):
    """``(max |x|, min |x| over nonzero x)`` when the array holds both signs,
    with the smallest magnitude negative or positive, normal or subnormal,
    next to zeros and ``-0.0``."""
    rng = np.random.default_rng(9900)
    array = _signed(rng, (5, 4, 3), 0.75, 1.3)
    array[0, 0, 0], array[4, 3, 2] = 1.0, -1.0
    array[2, 1, 1] = smallest
    if zeros:
        array[1, 2, 0] = 0.0
        array[3, 0, 2] = -0.0
    magnitudes = [abs(float(value)) for value in array.ravel()]
    reference = (max(magnitudes), min(value for value in magnitudes if value))
    assert finite_range(array) == reference
    assert finite_range(np.ascontiguousarray(array[:, ::-1])) == reference


def _single_index_term(n):
    """``F(i) C(i, a)`` with ``a = 1``, summed over ``i``."""
    term = DiagramTerm(
        (Propagator("C", "i", "a", "x", "y"),),
        Product((Rational(1), Symbol("F", ("i",)))),
        Rational(1),
        (),
        (("i", n),),
        0,
    )
    return term, {"a": 1}


@pytest.mark.parametrize("where", ["coefficient", "kernel"])
def test_a_negative_smallest_magnitude_decides_the_underflow_rule(admit_all, where):
    """The array holds both signs and its smallest magnitude, 2**-1000, is a
    negative entry; the other factor is near 2**-30, so a product can fall
    below 2**-1021 and both backends decline.  Positive entries alone would
    give a floor of 0 and admit the call."""
    rng = np.random.default_rng(9910)
    n = 5
    term, fixed = _single_index_term(n)
    values = {"F": _signed(rng, n, 0.6, 0.9)}
    kernel = _signed(rng, (SAMPLES, n, n), 0.6, 0.9)
    if where == "coefficient":
        values["F"][3] = -(2.0**-1000)
        kernel *= 2.0**-30
    else:
        kernel[2, 3, 1] = -(2.0**-1000)
        values["F"] *= 2.0**-30
    cb = [(kernel, "i", "a")]
    rp, jac = _signed(rng, SAMPLES, 0.6, 0.9), _signed(rng, SAMPLES, 0.6, 0.9)
    baseline = term.build_integrand(values, fixed)
    assert _einsum(baseline, cb, rp, jac, []) is None
    candidate = try_build_factorized(term, analyze_spatial(term), values, fixed, LIMIT)
    fused = candidate._static_values(cb, rp, jac, r_batches=[])
    assert factorized_state(candidate)[1] is None, "the factorized call contracted"
    np.testing.assert_array_equal(fused, _ordered(baseline, cb, rp, jac, []))


@pytest.mark.parametrize(
    "exponent, admitted, overflows",
    [(1004, True, False), (1005, False, False), (1010, False, True)],
)
def test_sum_headroom_at_the_overflow_limit(admit_all, exponent, admitted, overflows):
    """``F(i) C(i, a)`` over N = 64 assignments of one sign: the exponent
    bounds are ``exponent`` (F), 9 (C), 0 (rp, jac), 1 (the phase) and 6
    bits of headroom for 64 terms, so 1004 reaches the limit 1020 and 1005
    exceeds it.  At 1010 every product is below 2**1020 but their sum
    overflows; the ordered fold then raises under ``over="raise"``, and the
    einsum request declines and raises the same error."""
    rng = np.random.default_rng(9920)
    n = 64
    term, fixed = _single_index_term(n)
    values = {"F": rng.uniform(0.99, 1.0, n) * 2.0**exponent}
    cb = [(rng.uniform(0.99, 1.0, (SAMPLES, n, n)) * 2.0**9, "i", "a")]
    rp, jac = rng.uniform(0.99, 1.0, SAMPLES), rng.uniform(0.99, 1.0, SAMPLES)
    baseline = term.build_integrand(values, fixed)
    expected = _outcome(lambda: _ordered(baseline, cb, rp, jac, []))
    backend = _einsum(baseline, cb, rp, jac, [])
    assert (backend is not None) == admitted
    requested = replace(baseline, _contraction_backend="einsum")
    got = _outcome(lambda: requested._static_values(cb, rp, jac, r_batches=[]))
    if overflows:
        assert type(expected) is tuple and expected[0] is FloatingPointError
        assert got == expected
        return
    table = exact.dense_coefficients(baseline.coupling_array, 0)
    reference, scale = exact.exact_sum(table, term.propagator_indices, fixed, cb, rp, jac, [])
    # The coefficient, C, rp and jac; the coefficient is float64.
    for result in (got, expected):
        exact.assert_within_bound(result, reference, scale, 4, n, inexact=0)
    if not admitted:
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("layout", ["distinct_fixed", "conflicting_pins"])
def test_structural_zeros_decline_to_the_ordered_zeros(admit_all, layout):
    """``"distinct_fixed"``: a diagonal C table between the fixed components
    ``a = 0`` and ``b = 1``.  ``"conflicting_pins"``: diagonal tables
    ``C(i, a)`` and ``C(i, b)`` pin ``i`` to 0 and to 1.  Every term of the
    sum is zero; both backends decline and the call returns the ordered
    zeros."""
    rng = np.random.default_rng(9930)
    n = 3
    full = (_signed(rng, (SAMPLES, n, n)), "i", "j")
    if layout == "distinct_fixed":
        cb = [full, (_signed(rng, (SAMPLES, n)), "a", "b")]
    else:
        cb = [full, (_signed(rng, (SAMPLES, n)), "i", "a"), (_signed(rng, (SAMPLES, n)), "i", "b")]
    term = DiagramTerm(
        tuple(Propagator("C", left, right, "x", "y") for _, left, right in cb),
        Product((Rational(2, 3), Symbol("F", ("i", "j")))),
        Rational(1),
        (),
        (("i", n), ("j", n)),
        0,
    )
    values = {"F": _signed(rng, (n, n))}
    fixed = {"a": 0, "b": 1}
    rp, jac = _signed(rng, SAMPLES), _signed(rng, SAMPLES)
    baseline = term.build_integrand(values, fixed)
    ordered = _ordered(baseline, cb, rp, jac, [])
    assert np.all(ordered == 0.0)
    assert _einsum(baseline, cb, rp, jac, []) is None
    requested = replace(baseline, _contraction_backend="einsum")
    np.testing.assert_array_equal(requested._static_values(cb, rp, jac, r_batches=[]), ordered)
    candidate = try_build_factorized(term, analyze_spatial(term), values, fixed, LIMIT)
    fused = candidate._static_values(cb, rp, jac, r_batches=[])
    assert factorized_state(candidate)[1] is None, "the factorized call contracted"
    np.testing.assert_array_equal(fused, ordered)


@pytest.mark.parametrize("n", [3, 17])
def test_factor_sign_proof_rejects_monomials_of_opposite_sign(monkeypatch, n):
    """``F G H - (2/3) P Q W`` with every tensor of one sign and ``P`` negated,
    so the first monomial is nonpositive and the second nonnegative.  At
    N = 3 the work limit is set to 0, so the factor-sign proof decides
    instead of the per-assignment check: the sum is not prepared and the
    factorized request returns the ordered value.  At N = 17 the proof runs
    at the default limit and the dense fallback, above the budget, raises
    before it is allocated."""
    rng = np.random.default_rng(9940 + n)
    term, values = _uniform_sum(n, True, rng)
    values["P"] = -values["P"]
    if n == 17:
        with pytest.raises(ValueError, match="factorized coefficient materialization"):
            try_build_factorized(term, analyze_spatial(term), values, {}, LIMIT)
        return
    monkeypatch.setattr(factorized, "_MAX_SIGN_WORK", 0)
    assert try_build_factorized(term, analyze_spatial(term), values, {}, LIMIT) is None
    pairs = (("i", "j"), ("k", "l"), ("m", "n"))
    cb = [(_signed(rng, (SAMPLES, n, n)), left, right) for left, right in pairs]
    rp, jac = _signed(rng, SAMPLES), _signed(rng, SAMPLES)
    requested = term.build_integrand(values, {}, contraction_backend="factorized")
    ordered = term.build_integrand(values, {})
    np.testing.assert_array_equal(
        requested._static_values(cb, rp, jac, r_batches=[]),
        ordered._static_values(cb, rp, jac, r_batches=[]),
    )
