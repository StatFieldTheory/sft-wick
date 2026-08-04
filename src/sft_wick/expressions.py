"""Custom symbolic expression tree for Wick contraction results.

All expression types are frozen dataclasses (immutable + hashable).
Uses exact rational arithmetic via fractions.Fraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Sequence


class Expr(ABC):
    """Base class for all symbolic expressions."""

    @abstractmethod
    def to_latex(self) -> str:
        ...

    def __add__(self, other: Expr) -> Sum:
        return Sum(_flatten_sum([self, other]))

    def __radd__(self, other: object) -> Expr:
        if isinstance(other, int) and other == 0:
            return self
        return NotImplemented

    def __mul__(self, other: Expr) -> Product:
        return Product(_flatten_product([self, other]))

    def __rmul__(self, other: object) -> Expr:
        if isinstance(other, (int, Fraction)):
            return Product((Rational.from_number(other), self))
        return NotImplemented

    def __neg__(self) -> Product:
        return Product((Rational(-1, 1), self))

    def __sub__(self, other: Expr) -> Sum:
        return self + (-other)

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        ...

    @abstractmethod
    def __hash__(self) -> int:
        ...

    def __repr__(self) -> str:
        return self.to_latex()


# ---------------------------------------------------------------------------
# Scalar / numeric expressions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rational(Expr):
    """Exact rational number."""

    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        if self.denominator == 0:
            raise ZeroDivisionError("Rational denominator cannot be zero")
        # Normalize: keep denominator positive, reduce
        d = gcd(abs(self.numerator), abs(self.denominator))
        sign = 1 if self.denominator > 0 else -1
        object.__setattr__(self, "numerator", sign * self.numerator // d)
        object.__setattr__(self, "denominator", sign * self.denominator // d)

    @classmethod
    def from_number(cls, n: int | Fraction) -> Rational:
        if isinstance(n, Fraction):
            return cls(n.numerator, n.denominator)
        return cls(n, 1)

    @property
    def is_zero(self) -> bool:
        return self.numerator == 0

    @property
    def is_one(self) -> bool:
        return self.numerator == 1 and self.denominator == 1

    def to_fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def to_latex(self) -> str:
        if self.denominator == 1:
            return str(self.numerator)
        if self.numerator < 0:
            return rf"-\frac{{{-self.numerator}}}{{{self.denominator}}}"
        return rf"\frac{{{self.numerator}}}{{{self.denominator}}}"

    def __mul__(self, other: object) -> Expr:
        if isinstance(other, Rational):
            f = self.to_fraction() * other.to_fraction()
            return Rational(f.numerator, f.denominator)
        if isinstance(other, Expr):
            return Product(_flatten_product([self, other]))
        return NotImplemented

    def __add__(self, other: object) -> Expr:
        if isinstance(other, Rational):
            f = self.to_fraction() + other.to_fraction()
            return Rational(f.numerator, f.denominator)
        if isinstance(other, Expr):
            return Sum(_flatten_sum([self, other]))
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Rational):
            return self.numerator == other.numerator and self.denominator == other.denominator
        return NotImplemented

    def __hash__(self) -> int:
        return hash((Rational, self.numerator, self.denominator))


ZERO = Rational(0, 1)
ONE = Rational(1, 1)
MINUS_ONE = Rational(-1, 1)


# ---------------------------------------------------------------------------
# Named symbols (coupling constants, etc.)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Symbol(Expr):
    """A named symbol, possibly with indices.

    Examples:
        Symbol('F', ('i', 'j', 'k'))  ->  F_{ijk}
        Symbol('K', ('i', 'j'), ('y_0', 'y_1'))  ->  K_{ij}(y_0, y_1)

    ``local`` marks a coupling belonging to a *local* vertex, whose legs all
    sit at one spacetime point.  Such a symbol still carries that single point
    in :attr:`spatial_args` — which is what makes two copies of the same vertex
    distinguishable, and what lets a callable (time-dependent) coupling be
    evaluated at the right place — but the point is suppressed when rendering,
    so ``to_latex()`` is unchanged for the overwhelmingly common case of a
    constant local coupling.  Equality and hashing always include it.
    """

    name: str
    indices: tuple[str, ...] = ()
    spatial_args: tuple[str, ...] = ()
    local: bool = False

    def to_latex(self) -> str:
        s = self.name
        if self.indices:
            s += "_{" + "".join(_latex_index(i) for i in self.indices) + "}"
        if self.spatial_args and not self.local:
            s += "(" + ", ".join(_latex_index(a) for a in self.spatial_args) + ")"
        return s

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Symbol):
            return (self.name == other.name
                    and self.indices == other.indices
                    and self.spatial_args == other.spatial_args)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((Symbol, self.name, self.indices, self.spatial_args))


# ---------------------------------------------------------------------------
# Propagators
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Propagator(Expr):
    """A two-point function C_{ij}(x, x') or R_{ij}(x, x').

    For scalar fields, index_left and index_right are None.

    Convention for R: the physical field's index/position is always on the left.
    R_{ij}(x, x') means <phi_i(x) psi_j(x')>_{S_0}.
    """

    kind: str  # 'C' or 'R'
    index_left: str | None
    index_right: str | None
    spatial_left: str
    spatial_right: str

    def to_latex(self) -> str:
        s = self.kind
        if self.index_left is not None and self.index_right is not None:
            s += "_{" + _latex_index(self.index_left) + _latex_index(self.index_right) + "}"
        s += "(" + _latex_index(self.spatial_left) + ", " + _latex_index(self.spatial_right) + ")"
        return s

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Propagator):
            return (self.kind == other.kind
                    and self.index_left == other.index_left
                    and self.index_right == other.index_right
                    and self.spatial_left == other.spatial_left
                    and self.spatial_right == other.spatial_right)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((Propagator, self.kind, self.index_left, self.index_right,
                      self.spatial_left, self.spatial_right))


# ---------------------------------------------------------------------------
# Composite expressions
# ---------------------------------------------------------------------------

@dataclass(frozen=True, init=False)
class Sum(Expr):
    """Sum of expressions."""

    terms: tuple[Expr, ...]

    def __init__(self, terms: Sequence[Expr]) -> None:
        object.__setattr__(self, "terms", _flatten_sum(terms))

    def to_latex(self) -> str:
        if not self.terms:
            return "0"
        parts: list[str] = []
        for i, t in enumerate(self.terms):
            s = t.to_latex()
            if i > 0 and not s.startswith("-"):
                parts.append("+ " + s)
            else:
                parts.append(s)
        return " ".join(parts)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sum):
            return self.terms == other.terms
        return NotImplemented

    def __hash__(self) -> int:
        return hash((Sum, self.terms))

    def __add__(self, other: Expr) -> Sum:
        if isinstance(other, Sum):
            return Sum(self.terms + other.terms)
        return Sum(self.terms + (other,))


@dataclass(frozen=True, init=False)
class Product(Expr):
    """Product of expressions."""

    factors: tuple[Expr, ...]

    def __init__(self, factors: Sequence[Expr]) -> None:
        object.__setattr__(self, "factors", _flatten_product(factors))

    def to_latex(self) -> str:
        if not self.factors:
            return "1"
        parts = []
        for f in self.factors:
            s = f.to_latex()
            if isinstance(f, Sum) and len(f.terms) > 1:
                s = r"\left(" + s + r"\right)"
            parts.append(s)
        return " ".join(parts)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Product):
            return self.factors == other.factors
        return NotImplemented

    def __hash__(self) -> int:
        return hash((Product, self.factors))

    def __mul__(self, other: Expr) -> Product:
        if isinstance(other, Product):
            return Product(self.factors + other.factors)
        return Product(self.factors + (other,))


# ---------------------------------------------------------------------------
# Index/spatial wrappers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SumOverIndex(Expr):
    """Summation over a component index: sum_{i=1}^{N} body."""

    index_name: str
    dimension: int
    body: Expr

    def to_latex(self) -> str:
        return rf"\sum_{{{_latex_index(self.index_name)}=1}}^{{{self.dimension}}} {self.body.to_latex()}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SumOverIndex):
            return (self.index_name == other.index_name
                    and self.dimension == other.dimension
                    and self.body == other.body)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((SumOverIndex, self.index_name, self.dimension, self.body))


@dataclass(frozen=True)
class IntegralOver(Expr):
    """Integration over a spatial variable: integral d(var) body."""

    variable: str
    body: Expr

    def to_latex(self) -> str:
        return rf"\int \mathrm{{d}}{_latex_index(self.variable)}\, {self.body.to_latex()}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IntegralOver):
            return self.variable == other.variable and self.body == other.body
        return NotImplemented

    def __hash__(self) -> int:
        return hash((IntegralOver, self.variable, self.body))


# ---------------------------------------------------------------------------
# Delta functions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KroneckerDelta(Expr):
    """delta_{ij} for component indices."""

    index1: str
    index2: str

    def to_latex(self) -> str:
        return rf"\delta_{{{_latex_index(self.index1)}{_latex_index(self.index2)}}}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, KroneckerDelta):
            return {self.index1, self.index2} == {other.index1, other.index2}
        return NotImplemented

    def __hash__(self) -> int:
        return hash((KroneckerDelta, frozenset({self.index1, self.index2})))


@dataclass(frozen=True)
class DiracDelta(Expr):
    """delta(x - y) for spatial arguments."""

    arg1: str
    arg2: str

    def to_latex(self) -> str:
        return rf"\delta({self.arg1} - {self.arg2})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DiracDelta):
            return {self.arg1, self.arg2} == {other.arg1, other.arg2}
        return NotImplemented

    def __hash__(self) -> int:
        return hash((DiracDelta, frozenset({self.arg1, self.arg2})))


# ---------------------------------------------------------------------------
# Imaginary unit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImaginaryUnit(Expr):
    r"""The imaginary unit :math:`\mathrm{i}`.

    Used to represent the phase factor :math:`(-\mathrm{i})^n` that arises
    from the MSR convention :math:`\langle\phi\,\psi\rangle \propto -\mathrm{i}\,R`.
    """

    def to_latex(self) -> str:
        return r"\mathrm{i}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ImaginaryUnit):
            return True
        return NotImplemented

    def __hash__(self) -> int:
        return hash(ImaginaryUnit)


I = ImaginaryUnit()
"""Module-level constant for the imaginary unit."""


# ---------------------------------------------------------------------------
# Response phase utility
# ---------------------------------------------------------------------------

def _count_r_deep(expr: Expr) -> int:
    """Count R propagators recursively through the expression tree.

    For a **Sum**, returns the count from the first term (within a
    single vertex combination all surviving pairings have the same R
    count).  Returns -1 if terms disagree.
    """
    if isinstance(expr, Propagator):
        return 1 if expr.kind == "R" else 0
    if isinstance(expr, Product):
        return sum(_count_r_deep(f) for f in expr.factors)
    if isinstance(expr, Sum) and expr.terms:
        first = _count_r_deep(expr.terms[0])
        if all(_count_r_deep(t) == first for t in expr.terms[1:]):
            return first
        return -1  # terms disagree
    if isinstance(expr, (IntegralOver, SumOverIndex)):
        return _count_r_deep(expr.body)
    return 0


def apply_response_phase(expr: Expr) -> Expr:
    r"""Multiply each term by :math:`(-\mathrm{i})^n` where *n* is the
    number of response propagators *R* in that term.

    This implements the MSR convention
    :math:`\langle\phi(a)\,\psi(b)\rangle = -\mathrm{i}\,R(a,b)`.

    The phase is **absorbed into the existing rational coefficient**
    so that the factored form of the expression is preserved.  For
    example, :math:`(-\mathrm{i})^3 = \mathrm{i}` applied to a term
    with prefactor :math:`-\tfrac{1}{6}` yields
    :math:`-\tfrac{\mathrm{i}}{6}`.
    """
    if isinstance(expr, Sum):
        return Sum(tuple(apply_response_phase(t) for t in expr.terms))

    if isinstance(expr, Product):
        n_r = _count_r_deep(expr)
        if n_r <= 0:
            return expr  # 0 R's or disagreeing terms
        r = n_r % 4
        if r == 0:
            return expr

        factors = list(expr.factors)

        # Find the existing Rational coefficient (if any)
        rat_idx = next(
            (i for i, f in enumerate(factors) if isinstance(f, Rational)),
            None,
        )

        # (-i)^n phase rules applied to existing coefficient c:
        #   r=1: (-i)   → coeff becomes -c, insert i
        #   r=2: (-1)   → coeff becomes -c
        #   r=3: (+i)   → coeff stays  c, insert i
        if r in (1, 2):
            # Negate the rational coefficient
            if rat_idx is not None:
                old = factors[rat_idx]
                factors[rat_idx] = Rational(-old.numerator, old.denominator)
            else:
                factors.insert(0, Rational(-1, 1))
                rat_idx = 0

        if r in (1, 3):
            # Insert imaginary unit right after the rational
            insert_pos = (rat_idx + 1) if rat_idx is not None else 0
            factors.insert(insert_pos, ImaginaryUnit())

        # Drop Rational(1) — it's the identity and just adds clutter.
        factors = [f for f in factors
                   if not (isinstance(f, Rational) and f.is_one)]
        if not factors:
            return ONE

        return Product(tuple(factors))

    if isinstance(expr, Propagator) and expr.kind == "R":
        return Product((Rational(-1, 1), ImaginaryUnit(), expr))

    if isinstance(expr, IntegralOver):
        return IntegralOver(expr.variable, apply_response_phase(expr.body))

    if isinstance(expr, SumOverIndex):
        return SumOverIndex(
            expr.index_name, expr.dimension, apply_response_phase(expr.body)
        )

    return expr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latex_index(name: str) -> str:
    """Escape an index name for LaTeX.

    Converts ``i_0`` to ``i_{0}``, ``i_10`` to ``i_{10}``, ``y_0`` to
    ``y_{0}``, etc.  Single-character names (``a``, ``b``) are
    returned unchanged.
    """
    if "_" in name:
        prefix, suffix = name.split("_", 1)
        return prefix + "_{" + suffix + "}"
    return name


def _flatten_sum(terms: Sequence[Expr]) -> tuple[Expr, ...]:
    """Recursively flatten nested Sums."""
    result: list[Expr] = []
    for t in terms:
        if isinstance(t, Sum):
            result.extend(t.terms)
        else:
            result.append(t)
    return tuple(result)


def _flatten_product(factors: Sequence[Expr]) -> tuple[Expr, ...]:
    """Recursively flatten nested Products."""
    result: list[Expr] = []
    for f in factors:
        if isinstance(f, Product):
            result.extend(f.factors)
        else:
            result.append(f)
    return tuple(result)
