"""Tests for expressions module."""

from sft_wick import (
    Rational, Propagator, Sum, Product, Symbol, ZERO, ONE, I, ImaginaryUnit,
)
from sft_wick.expressions import MINUS_ONE, apply_response_phase


def test_rational_normalization():
    r = Rational(2, 4)
    assert r.numerator == 1
    assert r.denominator == 2


def test_rational_negative_denominator():
    r = Rational(3, -6)
    assert r.numerator == -1
    assert r.denominator == 2


def test_rational_arithmetic():
    a = Rational(1, 2)
    b = Rational(1, 3)
    result = a * b
    assert result == Rational(1, 6)


def test_rational_addition():
    a = Rational(1, 2)
    b = Rational(1, 3)
    result = a + b
    assert result == Rational(5, 6)


def test_rational_is_zero():
    assert ZERO.is_zero
    assert not ONE.is_zero


def test_rational_latex():
    assert Rational(3, 1).to_latex() == "3"
    assert Rational(1, 2).to_latex() == r"\frac{1}{2}"
    assert Rational(-1, 2).to_latex() == r"-\frac{1}{2}"


def test_propagator_equality():
    p1 = Propagator("C", "a", "b", "x", "y")
    p2 = Propagator("C", "a", "b", "x", "y")
    assert p1 == p2
    assert hash(p1) == hash(p2)


def test_propagator_latex():
    p = Propagator("C", "a", "b", "x", "y")
    assert p.to_latex() == "C_{ab}(x, y)"


def test_propagator_scalar_latex():
    p = Propagator("R", None, None, "x", "y")
    assert p.to_latex() == "R(x, y)"


def test_sum_flattening():
    a = Propagator("C", "a", "b", "x", "y")
    b = Propagator("R", "c", "d", "x", "y")
    c = Propagator("C", "e", "f", "x", "y")
    s1 = Sum((a, b))
    s2 = Sum((s1, c))
    assert len(s2.terms) == 3


def test_product_flattening():
    a = Propagator("C", "a", "b", "x", "y")
    b = Propagator("R", "c", "d", "x", "y")
    c = Propagator("C", "e", "f", "x", "y")
    p1 = Product((a, b))
    p2 = Product((p1, c))
    assert len(p2.factors) == 3


def test_symbol_latex():
    s = Symbol("F", ("i", "j", "k"))
    assert s.to_latex() == "F_{ijk}"

    s2 = Symbol("K", ("i", "j"), ("y_0", "y_1"))
    assert s2.to_latex() == "K_{ij}(y_{0}, y_{1})"


# --- ImaginaryUnit tests ---


def test_imaginary_unit_latex():
    assert I.to_latex() == r"\mathrm{i}"


def test_imaginary_unit_equality():
    assert I == ImaginaryUnit()
    assert hash(I) == hash(ImaginaryUnit())


def test_imaginary_unit_in_product():
    r = Propagator("R", None, None, "x", "y")
    p = Product((MINUS_ONE, I, r))
    assert len(p.factors) == 3
    latex = p.to_latex()
    assert r"\mathrm{i}" in latex


# --- apply_response_phase tests ---


def test_response_phase_bare_r():
    """A bare R propagator gets multiplied by -i."""
    r = Propagator("R", None, None, "x", "y")
    result = apply_response_phase(r)
    assert isinstance(result, Product)
    assert MINUS_ONE in result.factors
    assert I in result.factors
    assert r in result.factors


def test_response_phase_bare_c():
    """A bare C propagator is unaffected (0 R's → factor 1)."""
    c = Propagator("C", None, None, "x", "y")
    result = apply_response_phase(c)
    assert result is c


def test_response_phase_product_two_r():
    """Product with 2 R's gets (-i)^2 = -1."""
    r1 = Propagator("R", "a", "b", "x", "y")
    r2 = Propagator("R", "c", "d", "z", "w")
    prod = Product((r1, r2))
    result = apply_response_phase(prod)
    assert isinstance(result, Product)
    assert MINUS_ONE in result.factors


def test_response_phase_product_one_r_one_c():
    """Product with 1 R and 1 C gets (-i)^1 = -i."""
    r = Propagator("R", "a", "b", "x", "y")
    c = Propagator("C", "c", "d", "z", "w")
    prod = Product((r, c))
    result = apply_response_phase(prod)
    assert isinstance(result, Product)
    assert MINUS_ONE in result.factors
    assert I in result.factors


def test_response_phase_product_no_r():
    """Product with only C's is unaffected."""
    c1 = Propagator("C", "a", "b", "x", "y")
    c2 = Propagator("C", "c", "d", "z", "w")
    prod = Product((c1, c2))
    result = apply_response_phase(prod)
    assert result is prod


def test_response_phase_sum():
    """Phase is applied independently to each term of a Sum."""
    r = Propagator("R", None, None, "x", "y")
    c = Propagator("C", None, None, "x", "y")
    s = Sum((r, c))
    result = apply_response_phase(s)
    assert isinstance(result, Sum)
    # First term (R) should have -i; second term (C) should be unchanged
    assert isinstance(result.terms[0], Product)  # -i * R
    assert result.terms[1] is c  # C unchanged
