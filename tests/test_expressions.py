"""Tests for expressions module."""

from itertools import product
from math import fsum
from pathlib import Path

import numpy as np
import pytest

from sft_wick import (
    Action, Field, Rational, Propagator, Sum, Product, Symbol, ZERO, ONE, I,
    ImaginaryUnit, compute_moment,
)
from sft_wick.ir import from_sympy, to_sympy
from sft_wick.expressions import (
    DiracDelta, IntegralOver, KroneckerDelta, MINUS_ONE, SumOverIndex,
    apply_response_phase,
)
from sft_wick.perturbation import DiagramTerm
from sft_wick.workflow.config import build_system, expand_kwargs, load_workflow_config


def test_rational_normalization():
    r = Rational(2, 4)
    assert r.numerator == 1
    assert r.denominator == 2


def test_sympy_round_trip_preserves_argument_order_and_display_metadata():
    sp = pytest.importorskip("sympy")
    expr = IntegralOver("x", IntegralOver("y", SumOverIndex("x", 2, Product((
        Rational(-1, 3), I,
        Symbol("F", ("3", "x", "b"), ("x",), local=True),
        Symbol("K", ("b", "2"), ("y", "x")),
        KroneckerDelta("x", "b"), DiracDelta("y", "x"),
        Propagator("R", "a", None, "y", "x"),
        Sum((Propagator("C", None, "b", "x", "y"), ONE)),
    )))))
    exported = to_sympy(expr)
    restored = from_sympy(exported)
    assert restored == expr
    assert restored.to_latex() == expr.to_latex()
    r, = exported.atoms(sp.Function("R"))
    assert r.args == (sp.Tuple(sp.Symbol("index:a"), sp.Symbol("point:y")),
                      sp.Tuple(sp.Tuple(), sp.Symbol("point:x")))
    assert exported.has(sp.Symbol("index:x"), sp.Symbol("point:x"))


def test_sympy_literal_component_agrees_with_native_delta():
    sp = pytest.importorskip("sympy")
    delta = KroneckerDelta("01", "+1")
    diagram = DiagramTerm((), delta, Rational(1), (), (), 0)
    assert from_sympy(to_sympy(delta)) == delta
    assert float(to_sympy(delta, fixed_indices={}).doit()) == diagram.evaluate_coupling({}).item() == 1
    pins = {"01": 0, "+1": 1}
    assert float(to_sympy(delta, fixed_indices=pins).doit()) == diagram.evaluate_coupling({}, pins).item() == 0
    old_delta = sp.KroneckerDelta(1, 2, evaluate=False)
    assert from_sympy(old_delta) == KroneckerDelta("1", "2")


def test_sympy_round_trip_preserves_pinned_literal_in_native_diagram():
    sp = pytest.importorskip("sympy")
    phi = Field("phi", "physical", n_components=2)
    result = compute_moment([phi("1", "x"), phi("b", "y")], Action([]), order=0, diag_C=True)
    term, = result.diagram_terms(0)
    restored = from_sympy(to_sympy(term))
    for pins, expected in [({"b": 1}, 0), ({"b": 1, "1": 1}, 1)]:
        formula = to_sympy(restored, fixed_indices=pins)
        edge, = formula.atoms(sp.Function("C"))
        assert formula.xreplace({edge: sp.Integer(1)}).doit() == expected
        assert term.evaluate_coupling({}, pins).item() == expected


def test_sympy_component_sum_shadows_literal_pin():
    sp = pytest.importorskip("sympy")
    expr = SumOverIndex("1", 2, KroneckerDelta("1", "b"))
    formula = to_sympy(expr, fixed_indices={"1": 0, "b": 1})
    assert formula.variables == [sp.Symbol("index:1")]
    assert formula.doit() == 1


@pytest.fixture(scope="module")
def _cas_quickstart():
    cfg = load_workflow_config(Path(__file__).resolve().parents[1] / "examples/quickstart.yaml")
    return build_system(cfg.system), expand_kwargs(cfg)


def _contract_sympy(sp, expr, coupling, matrices, fixed):
    """Substitute independent tensor entries into the emitted CAS expression."""
    limits = expr.limits if isinstance(expr, sp.Sum) else ()
    body = expr.function if limits else expr
    tensors = body.atoms(sp.Function("R"), sp.Function("C"), sp.Function("WickLocal"))
    deltas = body.atoms(sp.KroneckerDelta)
    fixed = {sp.Symbol("index:" + name): index + 1 for name, index in fixed.items()}
    values = []
    for assignment in product(*(range(int(lo), int(hi) + 1) for _, lo, hi in limits)):
        lookup = fixed | dict(zip((index for index, _, _ in limits), assignment))

        def component(index):
            if index == sp.Tuple():
                return 0
            return int(lookup.get(index, index)) - 1

        replacements = {
            delta: int(component(delta.args[0]) == component(delta.args[1]))
            for delta in deltas
        }
        for tensor in tensors:
            kind = tensor.func.__name__
            if kind == "WickLocal":
                name, indices, _ = tensor.args
                val = coupling[name.name.removeprefix("coupling:")][tuple(map(component, indices))]
            else:
                (i, x), (j, y) = tensor.args
                val = matrices[kind, x.name.removeprefix("point:"), y.name.removeprefix("point:")][
                    component(i), component(j)
                ]
            replacements[tensor] = sp.sympify(complex(val))
        values.append(complex(body.xreplace(replacements)))
    return complex(fsum(v.real for v in values), fsum(v.imag for v in values))


def _contract_diagram(diagram, coupling, matrices, fixed):
    idx = diagram.propagator_indices
    evaluated = diagram.evaluate_coupling(coupling, fixed)
    values = []
    for choice in np.ndindex(*(dimension for _, dimension in idx)):
        assignment = fixed | dict(zip((name for name, _ in idx), choice))
        value = evaluated[choice]
        for p in diagram.propagators:
            value *= matrices[p.kind, p.spatial_left, p.spatial_right][
                assignment.get(p.index_left, 0), assignment.get(p.index_right, 0),
            ]
        values.append(value)
    return complex(fsum(v.real for v in values), fsum(v.imag for v in values))


@pytest.mark.parametrize("order", [2, 4])
def test_sympy_diagram_round_trip_matches_dense_contraction(_cas_quickstart, order):
    sp = pytest.importorskip("sympy")
    system, expansion_kwargs = _cas_quickstart
    terms = system.expand(**(expansion_kwargs | {
        "orders": (order,), "diag_R": order == 4, "diag_C": False, "iso_R": False,
    })).diagrams(order)
    rng = np.random.default_rng(412)
    coupling = {"F": -1j * rng.uniform(0.12, 0.7, size=(2,) * 3)}
    fixed = {"a": 0, "b": 1}
    for d in (terms[0], terms[len(terms) // 2], terms[-1]):
        points = sorted({p for prop in d.propagators
                         for p in (prop.spatial_left, prop.spatial_right)})
        matrices = {key: rng.uniform(0.12, 0.7, size=(2, 2))
                    for key in product(("R", "C"), points, points)}
        formula = to_sympy(d)
        restored = to_sympy(from_sympy(formula))
        assert restored == formula
        expected = _contract_diagram(d, coupling, matrices, fixed)
        assert abs(expected) > 1e-8
        assert abs(_contract_sympy(sp, restored, coupling, matrices, fixed) - expected) <= 1e-14


def test_sympy_off_diagonal_value_requires_kronecker_delta(_cas_quickstart):
    sp = pytest.importorskip("sympy")
    system, expansion_kwargs = _cas_quickstart
    d, = system.expand(**(expansion_kwargs | {"orders": (0,)})).diagrams(0)
    matrices = {("C", "x", "y"): np.diag([0.9, 0.8])}
    fixed = {"a": 0, "b": 1}
    formula = to_sympy(d)
    assert _contract_diagram(d, {}, matrices, fixed) == 0
    assert _contract_sympy(sp, formula, {}, matrices, fixed) == 0
    delta, = formula.atoms(sp.KroneckerDelta)
    assert abs(_contract_sympy(sp, formula.xreplace({delta: 1}), {}, matrices, fixed)) > 0.8


def test_sympy_diagram_retains_phase_of_absorbed_response():
    sp = pytest.importorskip("sympy")
    term = DiagramTerm(
        (Propagator("R", None, None, "x", "y"),), I, ONE, ("y",), (), 1,
        equal_time_aliases=(("y", "x"),), r_absorbed_pairs=(("x", "y"),),
        n_external_response=0, response_phase=False,
    )
    raw = to_sympy(term)
    assert not raw.has(sp.Function("R"))
    assert complex(raw.doit()) == term.evaluate_coupling({}).item() == 1


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


@pytest.mark.parametrize("left, right", [("j", None), (None, "j")])
def test_propagator_one_index_latex(left, right):
    assert Propagator("C", left, right, "u", "x").to_latex() == "C_{j}(u, x)"


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


def test_response_phase_refuses_a_product_over_a_mixed_sum():
    """No single power of -i multiplies R + C."""
    r = Propagator("R", None, None, "x", "y")
    c = Propagator("C", None, None, "x", "y")
    with pytest.raises(ValueError, match="no single response phase"):
        apply_response_phase(Product((Rational(1, 2), Sum((r, c)))))
