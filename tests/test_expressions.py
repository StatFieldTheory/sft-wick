"""Tests for expressions module."""

from sft_wick import Rational, Propagator, Sum, Product, Symbol, ZERO, ONE


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
    assert s2.to_latex() == "K_{ij}(y_0, y_1)"
