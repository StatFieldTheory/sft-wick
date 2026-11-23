"""Optional SymPy translation for Wick expressions and diagram integrands."""

from __future__ import annotations

from collections.abc import Mapping
from operator import index as integer_index
from typing import TYPE_CHECKING

from .expressions import (
    DiracDelta,
    Expr,
    ImaginaryUnit,
    IntegralOver,
    KroneckerDelta,
    Product,
    Propagator,
    Rational,
    Sum,
    SumOverIndex,
    Symbol,
)
from .indices import LITERAL_COMPONENT_BASE, pinned_component
from .perturbation import DiagramTerm

if TYPE_CHECKING:
    from sympy import Expr as SymPyExpr


def _sympy():
    try:
        import sympy
    except ImportError as exc:
        raise ImportError("SymPy export requires the sft-wick[cas] extra.") from exc
    return sympy


def _integrand(term: DiagramTerm) -> Expr:
    # Use the numerical phase even when the original term hides it in LaTeX.
    phase = term.n_response % 4
    coefficient = term.rational_prefactor
    if phase in (1, 2):
        coefficient = Rational(-coefficient.numerator, coefficient.denominator)
    factors = [coefficient]
    if phase % 2:
        factors.append(ImaginaryUnit())
    factors.append(term.coupling_sum)
    absorbed = set(term.r_absorbed_pairs)
    factors.extend(
        p for p in term.propagators
        if p.kind != "R" or (p.spatial_left, p.spatial_right) not in absorbed
    )
    expr = Product(factors)
    for index, dimension in reversed(term.summation_indices):
        expr = SumOverIndex(index, dimension, expr)
    return expr


def to_sympy(
    value: Expr | DiagramTerm, *, fixed_indices: Mapping[str, int] | None = None,
) -> SymPyExpr:
    """Emit SymPy algebra with named tensor entries and 1-based component sums.

    For a diagram, export the raw component-summed integrand, before any
    time integration, with its rational prefactor and numerical MSR phase.
    A response factor absorbed into the coupling is omitted. Couplings
    stand for the MSR-scaled values accepted by ``evaluate_coupling``.

    ``WickLocal`` and ``WickNonlocal`` hold the coupling name, component
    tuple and spatial tuple. ``R`` and ``C`` each take two endpoint tuples
    ``(component, point)``, in the propagator's original order. An empty
    tuple stands for an absent component. The ``index:``, ``point:`` and
    ``coupling:`` namespaces keep equal labels in different roles distinct.

    Component symbols retain their exact label spelling by default. Pass
    ``fixed_indices`` to bind free component labels to zero-based values.
    This also resolves any remaining free numeric labels with the native
    one-based literal rule, so an empty mapping resolves just those labels.
    Summed indices keep their own binding.
    """
    sp = _sympy()
    coupling = sp.Function("WickLocal"), sp.Function("WickNonlocal")
    propagator = {kind: sp.Function(kind) for kind in ("R", "C")}

    def component(name, bound):
        if name is None:
            return sp.Tuple()
        if fixed_indices is not None and name not in bound:
            resolved = pinned_component(name, fixed_indices)
            if resolved is not None:
                return sp.Integer(integer_index(resolved) + LITERAL_COMPONENT_BASE)
        return sp.Symbol("index:" + name)

    def point(name):
        return sp.Symbol("point:" + name)

    def encode(expr, bound):
        match expr:
            case Rational(numerator, denominator):
                return sp.Rational(numerator, denominator)
            case ImaginaryUnit():
                return sp.I
            case Symbol(name, indices, spatial, local):
                leaf = coupling[0 if local else 1]
                return leaf(sp.Symbol("coupling:" + name),
                            sp.Tuple(*(component(i, bound) for i in indices)),
                            sp.Tuple(*map(point, spatial)))
            case Propagator(kind, left, right, p, q):
                return propagator[kind](sp.Tuple(component(left, bound), point(p)),
                                        sp.Tuple(component(right, bound), point(q)))
            case KroneckerDelta(a, b):
                return sp.KroneckerDelta(component(a, bound), component(b, bound), evaluate=False)
            case DiracDelta(a, b):
                difference = sp.Add(point(a), -point(b), evaluate=False)
                return sp.DiracDelta(difference, evaluate=False)
            case Sum(terms):
                return sp.Add(*(encode(term, bound) for term in terms), evaluate=False)
            case Product(factors):
                return sp.Mul(*(encode(factor, bound) for factor in factors), evaluate=False)
            case SumOverIndex(name, dimension, body):
                inner = bound | {name}
                return sp.Sum(encode(body, inner), (component(name, inner), 1, dimension))
            case IntegralOver(name, body):
                return sp.Integral(encode(body, bound), point(name))
            case _:
                raise TypeError(f"Cannot export {type(expr).__name__} to SymPy.")

    return encode(_integrand(value) if isinstance(value, DiagramTerm) else value, frozenset())


def from_sympy(value: SymPyExpr) -> Expr:
    """Read the Wick expression subset emitted by :func:`to_sympy`.

    Import an expression object directly. Component labels, propagation
    direction, and ordered coupling arguments survive the round trip.
    SymPy may collapse empty or single-child sums and products to their
    equivalent algebraic expressions. Component symbols preserve their
    spelling. A SymPy integer used as a component decodes to a one-based
    numeric label in standard integer spelling.
    """
    sp = _sympy()
    coupling = {sp.Function("WickLocal"): True, sp.Function("WickNonlocal"): False}
    propagator = {sp.Function(kind): kind for kind in ("R", "C")}

    def label(atom, prefix):
        if isinstance(atom, sp.Symbol) and atom.name.startswith(prefix):
            return atom.name[len(prefix):]
        raise ValueError(f"Expected a SymPy symbol in the {prefix} namespace.")

    def index(atom):
        return str(atom) if isinstance(atom, sp.Integer) else label(atom, "index:")

    def point(atom):
        return label(atom, "point:")

    def endpoint(pair):
        component, spatial = pair
        return (None if component == sp.Tuple() else index(component), point(spatial))

    def decode(expr):
        if not isinstance(expr, sp.Basic):
            raise TypeError(f"Expected a SymPy expression, got {type(expr).__name__}.")
        if isinstance(expr, sp.Rational):
            return Rational(int(expr.p), int(expr.q))
        if expr == sp.I:
            return ImaginaryUnit()
        if expr.func in coupling:
            name, indices, spatial = expr.args
            return Symbol(label(name, "coupling:"), tuple(map(index, indices)),
                          tuple(map(point, spatial)), coupling[expr.func])
        if expr.func in propagator:
            (left, p), (right, q) = map(endpoint, expr.args)
            return Propagator(propagator[expr.func], left, right, p, q)
        if expr.func == sp.KroneckerDelta and len(expr.args) == 2:
            return KroneckerDelta(*map(index, expr.args))
        if expr.func == sp.DiracDelta and len(expr.args) == 1:
            terms = expr.args[0].args
            if len(terms) == 2:
                a, minus_b = terms
                if isinstance(minus_b, sp.Mul) and minus_b.args[0] == -1:
                    return DiracDelta(point(a), point(-minus_b))
        if expr.func == sp.Add:
            return Sum(tuple(map(decode, expr.args)))
        if expr.func == sp.Mul:
            return Product(tuple(map(decode, expr.args)))
        if expr.func == sp.Sum:
            body = decode(expr.function)
            for name, start, stop in expr.limits:
                if start != 1 or not isinstance(stop, sp.Integer) or stop < 1:
                    raise ValueError("A Wick component sum needs fixed bounds 1 to N.")
                body = SumOverIndex(index(name), int(stop), body)
            return body
        if expr.func == sp.Integral:
            body = decode(expr.function)
            for limit in expr.limits:
                if len(limit) != 1:
                    raise ValueError("A Wick coordinate integral has no explicit bounds.")
                body = IntegralOver(point(limit[0]), body)
            return body
        raise TypeError(f"Cannot import SymPy node {expr.func.__name__} as a Wick expression.")

    return decode(value)
