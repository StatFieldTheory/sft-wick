"""Expression simplification for Wick contraction results.

Pipeline:
1. Flatten nested Sum/Product
2. Absorb rational prefactors in products
3. Eliminate zeros
4. Canonical ordering of propagators
5. Term collection (combine like terms)
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

from .expressions import (
    ZERO,
    ONE,
    Expr,
    IntegralOver,
    Product,
    Propagator,
    Rational,
    Sum,
    SumOverIndex,
    Symbol,
)


def simplify(expr: Expr) -> Expr:
    """Main simplification entry point."""
    expr = _flatten(expr)
    expr = _absorb_rationals(expr)
    expr = _eliminate_zeros(expr)
    expr = _collect_terms(expr)
    return expr


def _flatten(expr: Expr) -> Expr:
    """Recursively flatten nested Sum and Product."""
    if isinstance(expr, Sum):
        flat_terms: list[Expr] = []
        for t in expr.terms:
            t = _flatten(t)
            if isinstance(t, Sum):
                flat_terms.extend(t.terms)
            else:
                flat_terms.append(t)
        return Sum(tuple(flat_terms))

    if isinstance(expr, Product):
        flat_factors: list[Expr] = []
        for f in expr.factors:
            f = _flatten(f)
            if isinstance(f, Product):
                flat_factors.extend(f.factors)
            else:
                flat_factors.append(f)
        return Product(tuple(flat_factors))

    if isinstance(expr, IntegralOver):
        return IntegralOver(expr.variable, _flatten(expr.body))

    if isinstance(expr, SumOverIndex):
        return SumOverIndex(expr.index_name, expr.dimension, _flatten(expr.body))

    return expr


def _absorb_rationals(expr: Expr) -> Expr:
    """In a Product, multiply all Rational factors into a single prefactor."""
    if isinstance(expr, Product):
        coeff = Fraction(1)
        other_factors: list[Expr] = []
        for f in expr.factors:
            f = _absorb_rationals(f)
            if isinstance(f, Rational):
                coeff *= f.to_fraction()
            else:
                other_factors.append(f)

        if coeff == 0:
            return ZERO
        if not other_factors:
            return Rational(coeff.numerator, coeff.denominator)
        if coeff == 1:
            if len(other_factors) == 1:
                return other_factors[0]
            return Product(tuple(other_factors))
        return Product(
            (Rational(coeff.numerator, coeff.denominator), *other_factors)
        )

    if isinstance(expr, Sum):
        return Sum(tuple(_absorb_rationals(t) for t in expr.terms))

    if isinstance(expr, IntegralOver):
        return IntegralOver(expr.variable, _absorb_rationals(expr.body))

    if isinstance(expr, SumOverIndex):
        return SumOverIndex(
            expr.index_name, expr.dimension, _absorb_rationals(expr.body)
        )

    return expr


def _eliminate_zeros(expr: Expr) -> Expr:
    """Remove zero terms from sums; collapse products containing zero."""
    if isinstance(expr, Sum):
        terms = [_eliminate_zeros(t) for t in expr.terms]
        terms = [t for t in terms if not _is_zero(t)]
        if not terms:
            return ZERO
        if len(terms) == 1:
            return terms[0]
        return Sum(tuple(terms))

    if isinstance(expr, Product):
        factors = [_eliminate_zeros(f) for f in expr.factors]
        if any(_is_zero(f) for f in factors):
            return ZERO
        factors = [f for f in factors if not _is_one(f)]
        if not factors:
            return ONE
        if len(factors) == 1:
            return factors[0]
        return Product(tuple(factors))

    if isinstance(expr, IntegralOver):
        body = _eliminate_zeros(expr.body)
        if _is_zero(body):
            return ZERO
        return IntegralOver(expr.variable, body)

    if isinstance(expr, SumOverIndex):
        body = _eliminate_zeros(expr.body)
        if _is_zero(body):
            return ZERO
        return SumOverIndex(expr.index_name, expr.dimension, body)

    return expr


def _is_zero(expr: Expr) -> bool:
    return isinstance(expr, Rational) and expr.is_zero


def _is_one(expr: Expr) -> bool:
    return isinstance(expr, Rational) and expr.is_one


def _propagator_key(p: Propagator) -> tuple:
    """Canonical sort key for a propagator."""
    return (p.kind, p.spatial_left, p.spatial_right, p.index_left or "", p.index_right or "")


def _term_signature(expr: Expr) -> tuple | None:
    """Extract the non-coefficient part of a term for grouping.

    A term is either:
      - A single Propagator
      - A Product of (optional Rational) * Propagators * Symbols * IntegralOvers * SumOverIndices
    Returns a hashable signature, or None if it can't be grouped.
    """
    if isinstance(expr, Propagator):
        return (("prop", _propagator_key(expr)),)

    if isinstance(expr, Product):
        parts: list[tuple] = []
        for f in expr.factors:
            if isinstance(f, Rational):
                continue  # skip coefficient
            if isinstance(f, Propagator):
                parts.append(("prop", _propagator_key(f)))
            elif isinstance(f, Symbol):
                parts.append(("sym", f.name, f.indices, f.spatial_args))
            elif isinstance(f, IntegralOver):
                parts.append(("int", f.variable))
            elif isinstance(f, SumOverIndex):
                parts.append(("sum", f.index_name, f.dimension))
            else:
                return None  # can't group
        return tuple(sorted(parts))

    return None


def _get_coefficient(expr: Expr) -> Fraction:
    """Extract the rational coefficient from a term."""
    if isinstance(expr, Rational):
        return expr.to_fraction()
    if isinstance(expr, Product):
        coeff = Fraction(1)
        for f in expr.factors:
            if isinstance(f, Rational):
                coeff *= f.to_fraction()
        return coeff
    return Fraction(1)


def _set_coefficient(expr: Expr, coeff: Fraction) -> Expr:
    """Replace the rational coefficient of a term."""
    if isinstance(expr, Rational):
        return Rational(coeff.numerator, coeff.denominator)

    if isinstance(expr, Product):
        non_rational = [f for f in expr.factors if not isinstance(f, Rational)]
        if coeff == 1:
            if len(non_rational) == 1:
                return non_rational[0]
            return Product(tuple(non_rational))
        return Product(
            (Rational(coeff.numerator, coeff.denominator), *non_rational)
        )

    # expr has implicit coefficient 1
    if coeff == 1:
        return expr
    return Product((Rational(coeff.numerator, coeff.denominator), expr))


def _collect_terms(expr: Expr) -> Expr:
    """Combine terms with identical propagator structures."""
    if not isinstance(expr, Sum):
        return expr

    groups: defaultdict[tuple | None, list[Expr]] = defaultdict(list)
    ungroupable: list[Expr] = []

    for term in expr.terms:
        sig = _term_signature(term)
        if sig is not None:
            groups[sig].append(term)
        else:
            ungroupable.append(term)

    collected: list[Expr] = []
    for sig, terms in groups.items():
        total_coeff = sum((_get_coefficient(t) for t in terms), Fraction(0))
        if total_coeff != 0:
            representative = terms[0]
            collected.append(_set_coefficient(representative, total_coeff))

    collected.extend(ungroupable)

    if not collected:
        return ZERO
    if len(collected) == 1:
        return collected[0]
    return Sum(tuple(collected))
