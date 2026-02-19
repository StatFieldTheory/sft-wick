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
    """

    name: str
    indices: tuple[str, ...] = ()
    spatial_args: tuple[str, ...] = ()

    def to_latex(self) -> str:
        s = self.name
        if self.indices:
            s += "_{" + "".join(self.indices) + "}"
        if self.spatial_args:
            s += "(" + ", ".join(self.spatial_args) + ")"
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
            s += "_{" + self.index_left + self.index_right + "}"
        s += "(" + self.spatial_left + ", " + self.spatial_right + ")"
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
        return rf"\sum_{{{self.index_name}=1}}^{{{self.dimension}}} {self.body.to_latex()}"

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
        return rf"\int \mathrm{{d}}{self.variable}\, {self.body.to_latex()}"

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
        return rf"\delta_{{{self.index1}{self.index2}}}"

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
# Helpers
# ---------------------------------------------------------------------------

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
