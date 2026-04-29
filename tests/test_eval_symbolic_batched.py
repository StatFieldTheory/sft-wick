"""Unit tests for the batched symbolic evaluator.

Locks the per-node behaviour of
:func:`sft_wick.perturbation._eval_symbolic_batched` and the
public ``DiagramTerm.evaluate_coupling_batched`` against the
existing scalar :func:`_eval_symbolic` /
:meth:`DiagramTerm.evaluate_coupling` paths at element-wise
``rtol=1e-13``.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from sft_wick.expressions import (
    ImaginaryUnit,
    KroneckerDelta,
    Product,
    Propagator,
    Rational,
    Sum,
    Symbol,
)
from sft_wick.perturbation import (
    DiagramTerm,
    _eval_symbolic,
    _eval_symbolic_batched,
)


N_SAMPLES = 7
RTOL = 1e-13


def _scalar_loop(expr, sym_vals_per_sample, idx_map):
    """Reference: call the scalar evaluator once per sample."""
    out = []
    for sample_cv in sym_vals_per_sample:
        out.append(_eval_symbolic(expr, sample_cv, idx_map))
    return np.array(out)


# -----------------------------------------------------------------
# Atom / leaf nodes
# -----------------------------------------------------------------

def test_rational_returns_scalar_value():
    r = Rational(3, 4)
    out = _eval_symbolic_batched(r, {}, {}, n_samples=N_SAMPLES)
    assert out == 0.75


def test_imaginary_unit_returns_1j():
    out = _eval_symbolic_batched(ImaginaryUnit(), {}, {}, n_samples=N_SAMPLES)
    assert out == 1j


def test_kronecker_delta_resolves_via_index_map():
    kd = KroneckerDelta("a", "b")
    out_eq = _eval_symbolic_batched(
        kd, {}, {"a": 1, "b": 1}, n_samples=N_SAMPLES,
    )
    out_neq = _eval_symbolic_batched(
        kd, {}, {"a": 0, "b": 1}, n_samples=N_SAMPLES,
    )
    assert out_eq == 1.0
    assert out_neq == 0.0


def test_kronecker_delta_literal_index():
    kd = KroneckerDelta("0", "0")
    out = _eval_symbolic_batched(kd, {}, {}, n_samples=N_SAMPLES)
    assert out == 1.0


# -----------------------------------------------------------------
# Symbol -- the core "indexed access into per-sample tensor" case
# -----------------------------------------------------------------

def test_symbol_static_array_broadcasts():
    """When a Symbol's array has rank == len(indices) it is treated as
    static and the same value is broadcast across all samples."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    sym = Symbol("F", indices=("i", "j"))
    out = _eval_symbolic_batched(
        sym, {"F": arr}, {"i": 0, "j": 1}, n_samples=N_SAMPLES,
    )
    assert out == pytest.approx(2.0)


def test_symbol_batched_array_per_sample_slice():
    rng = np.random.default_rng(0)
    arr_batched = rng.standard_normal((N_SAMPLES, 2, 2))
    sym = Symbol("F", indices=("i", "j"))
    out = _eval_symbolic_batched(
        sym, {"F": arr_batched}, {"i": 1, "j": 0}, n_samples=N_SAMPLES,
    )
    np.testing.assert_allclose(out, arr_batched[:, 1, 0], rtol=RTOL)


def test_symbol_rank3_batched_matches_scalar_loop():
    rng = np.random.default_rng(1)
    arr_batched = rng.standard_normal((N_SAMPLES, 2, 2, 2))
    sym = Symbol("K", indices=("i", "j", "k"))
    idx_map = {"i": 0, "j": 1, "k": 1}
    sym_vals_per_sample = [{"K": arr_batched[s]} for s in range(N_SAMPLES)]
    expected = _scalar_loop(sym, sym_vals_per_sample, idx_map)
    actual = _eval_symbolic_batched(
        sym, {"K": arr_batched}, idx_map, n_samples=N_SAMPLES,
    )
    np.testing.assert_allclose(actual, expected, rtol=RTOL)


def test_symbol_literal_observable_index():
    """Literal '1' → 0 (1-indexed convention)."""
    arr = np.array([10.0, 20.0])
    sym = Symbol("v", indices=("1",))
    out = _eval_symbolic_batched(
        sym, {"v": arr}, {}, n_samples=N_SAMPLES,
    )
    assert out == pytest.approx(10.0)


# -----------------------------------------------------------------
# Composite nodes -- Product / Sum / nesting
# -----------------------------------------------------------------

def test_product_of_two_batched_symbols():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((N_SAMPLES, 2, 2))
    B = rng.standard_normal((N_SAMPLES, 2, 2))
    expr = Product(
        (Symbol("A", indices=("i", "j")), Symbol("B", indices=("j", "k")))
    )
    idx_map = {"i": 0, "j": 1, "k": 0}
    sym_vals_per_sample = [
        {"A": A[s], "B": B[s]} for s in range(N_SAMPLES)
    ]
    expected = _scalar_loop(expr, sym_vals_per_sample, idx_map)
    actual = _eval_symbolic_batched(
        expr, {"A": A, "B": B}, idx_map, n_samples=N_SAMPLES,
    )
    np.testing.assert_allclose(actual, expected, rtol=RTOL)


def test_product_with_rational_and_imaginary_unit():
    """Mixed scalar + batched factors -- result should be batched."""
    rng = np.random.default_rng(3)
    A = rng.standard_normal((N_SAMPLES, 2))
    expr = Product(
        (Rational(1, 6), ImaginaryUnit(), Symbol("v", indices=("i",)))
    )
    idx_map = {"i": 1}
    sym_vals_per_sample = [{"v": A[s]} for s in range(N_SAMPLES)]
    expected = _scalar_loop(expr, sym_vals_per_sample, idx_map)
    actual = _eval_symbolic_batched(
        expr, {"v": A}, idx_map, n_samples=N_SAMPLES,
    )
    np.testing.assert_allclose(actual, expected, rtol=RTOL)


def test_sum_of_two_terms():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((N_SAMPLES, 2))
    B = rng.standard_normal((N_SAMPLES, 2))
    expr = Sum(
        (Symbol("a", indices=("i",)), Symbol("b", indices=("i",)))
    )
    idx_map = {"i": 0}
    sym_vals_per_sample = [
        {"a": A[s], "b": B[s]} for s in range(N_SAMPLES)
    ]
    expected = _scalar_loop(expr, sym_vals_per_sample, idx_map)
    actual = _eval_symbolic_batched(
        expr, {"a": A, "b": B}, idx_map, n_samples=N_SAMPLES,
    )
    np.testing.assert_allclose(actual, expected, rtol=RTOL)


def test_nested_product_inside_sum():
    """Sum over (Product, Product) with KroneckerDelta and Rational
    coefficients — the typical shape of a coupling_sum."""
    rng = np.random.default_rng(5)
    K = rng.standard_normal((N_SAMPLES, 2, 2, 2))
    expr = Sum(
        (
            Product(
                (
                    Rational(1, 2),
                    Symbol("K", indices=("i", "j", "k")),
                    KroneckerDelta("j", "k"),
                )
            ),
            Product(
                (
                    Rational(-1, 3),
                    Symbol("K", indices=("k", "j", "i")),
                )
            ),
        )
    )
    idx_map = {"i": 0, "j": 1, "k": 1}
    sym_vals_per_sample = [{"K": K[s]} for s in range(N_SAMPLES)]
    expected = _scalar_loop(expr, sym_vals_per_sample, idx_map)
    actual = _eval_symbolic_batched(
        expr, {"K": K}, idx_map, n_samples=N_SAMPLES,
    )
    np.testing.assert_allclose(actual, expected, rtol=RTOL)


def test_kronecker_delta_zero_kills_product():
    """delta_{ij} with i != j collapses the whole Product to 0."""
    rng = np.random.default_rng(6)
    A = rng.standard_normal((N_SAMPLES, 2))
    expr = Product(
        (Symbol("a", indices=("i",)), KroneckerDelta("i", "j"))
    )
    out = _eval_symbolic_batched(
        expr, {"a": A}, {"i": 0, "j": 1}, n_samples=N_SAMPLES,
    )
    # Could be a scalar 0 or a (n_samples,) of zeros -- accept both.
    assert np.all(np.asarray(out) == 0.0)


# -----------------------------------------------------------------
# Fallback: unsupported node types raise NotImplementedError
# -----------------------------------------------------------------

def test_propagator_node_falls_through_to_not_implemented():
    """Propagator inside a coupling_sum is non-physical but the
    batched evaluator must raise NotImplementedError so the caller's
    safety net falls back to the scalar loop."""
    expr = Propagator("C", "i", "j", "x_0", "x_1")
    with pytest.raises(NotImplementedError):
        _eval_symbolic_batched(
            expr, {}, {"i": 0, "j": 1}, n_samples=N_SAMPLES,
        )


# -----------------------------------------------------------------
# DiagramTerm.evaluate_coupling_batched -- integration with Diagram
# -----------------------------------------------------------------

def _make_simple_term():
    """A degenerate DiagramTerm whose coupling_sum is a rank-3 K
    contracted on a single summation index. No propagators -- the
    full-tree numerical pipeline isn't exercised, only the
    coupling-substitution path."""
    expr = Sum(
        (
            Product(
                (
                    Symbol("K", indices=("a", "i", "i")),
                )
            ),
        )
    )
    return DiagramTerm(
        propagators=(),
        coupling_sum=expr,
        rational_prefactor=Rational(1, 2),
        integration_vars=(),
        summation_indices=(("i", 2),),
        n_response=0,
    )


def test_evaluate_coupling_batched_matches_loop_over_evaluate_coupling():
    rng = np.random.default_rng(42)
    K_static = rng.standard_normal((2, 2, 2))
    term = _make_simple_term()

    # Static path: array shape (N, N, N) — broadcast across samples.
    expected_scalar = term.evaluate_coupling(
        {"K": K_static}, {"a": 0},
    )
    out = term.evaluate_coupling_batched(
        {"K": K_static}, n_samples=N_SAMPLES, fixed_indices={"a": 0},
    )
    assert out.shape == (N_SAMPLES,)
    np.testing.assert_allclose(
        out, np.full(N_SAMPLES, expected_scalar), rtol=RTOL,
    )

    # Batched path: array shape (n_samples, N, N, N) — per-sample
    # numerical match against scalar evaluate_coupling.
    K_batched = rng.standard_normal((N_SAMPLES, 2, 2, 2))
    expected = np.array(
        [
            complex(
                term.evaluate_coupling({"K": K_batched[s]}, {"a": 0})
            )
            for s in range(N_SAMPLES)
        ]
    )
    out_b = term.evaluate_coupling_batched(
        {"K": K_batched}, n_samples=N_SAMPLES, fixed_indices={"a": 0},
    )
    np.testing.assert_allclose(out_b, expected, rtol=RTOL)
