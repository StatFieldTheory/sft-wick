"""Transport and structural identity for grouped and shifted measures."""

import json
from dataclasses import replace

import pytest

from sft_wick.expressions import Propagator, Rational, Symbol
from sft_wick.ir import diagram_ids, from_ir, to_ir
from sft_wick.perturbation import DiagramTerm


def delay(left, right, lag):
    return DiagramTerm(
        (Propagator("R", None, None, "x", left),
         Propagator("C", None, None, right, right)),
        Symbol("F", (), (left, right)), Rational(1),
        (left, right), (), 1,
        group_templates=(("F", (left, right),
                          ("response", "physical", "physical"), (0, 1, 1), "spacetime"),),
        time_lags=((right, left, lag),),
    )


def test_delay_wire_keeps_lag_orientation_and_group_incidence():
    term = delay("a", "b", 0.25)
    wire = json.loads(json.dumps(to_ir(term)))
    assert wire["schema"] == "wickir/v1.1"
    assert from_ir(wire) == term
    ids = diagram_ids(term)
    assert diagram_ids(from_ir(wire)) == ids == diagram_ids(delay("u", "v", 0.25))
    assert ids != diagram_ids(delay("a", "b", 0.5))
    assert ids != diagram_ids(replace(term, time_lags=(("a", "b", 0.25),)))
    assert ids != diagram_ids(replace(term, group_templates=((
        "F", ("a", "b"), ("response", "physical", "physical"), (0, 0, 1), "spacetime",
    ),)))


def test_time_only_groups_round_trip_and_preserve_distinct_positions():
    term = DiagramTerm(
        (Propagator("R", None, None, "x", "a"), Propagator("R", None, None, "y", "b")),
        Symbol("K", (), ("a", "b")), Rational(1), ("a", "b"), (), 2,
        equal_time_aliases=(("b", "a"),),
        group_templates=(("K", ("a", "b"), ("response", "response"), (0, 0), "time"),),
    )
    wire = to_ir(term, positions={"x": 1, "y": 2})
    assert wire["evaluation"]["rejection"] is None
    assert from_ir(wire) == term
    assert diagram_ids(term) == diagram_ids(replace(term, equal_time_aliases=(("a", "b"),)))
    assert diagram_ids(term) != diagram_ids(replace(term, group_templates=(), equal_time_aliases=()))
    wire["diagram"]["equal_time_aliases"] = []
    with pytest.raises(ValueError, match="share one clock"):
        from_ir(wire)
    wire["diagram"]["equal_time_aliases"] = [["a", "b"]]
    assert diagram_ids(from_ir(wire)) == diagram_ids(term)
    wire["diagram"]["group_templates"][0][4] = "spacetime"
    with pytest.raises(ValueError, match="coordinates"):
        from_ir(wire)


def test_sympy_diagram_retains_oriented_fixed_delay():
    sp = pytest.importorskip("sympy")
    from sft_wick.expressions import TimeDelta
    from sft_wick.ir import from_sympy, to_sympy

    formulas = []
    for lag in (0.21, 0.37):
        formula = to_sympy(delay("a", "b", lag))
        constraint, = formula.atoms(sp.DiracDelta)
        assert from_sympy(constraint) == TimeDelta("b", "a", lag)
        formulas.append(formula)
    assert formulas[0] != formulas[1]


def test_sympy_time_group_retains_clock_without_reintroducing_absorbed_response():
    sp = pytest.importorskip("sympy")
    from sft_wick.expressions import TimeDelta
    from sft_wick.ir import from_sympy, to_sympy

    term = DiagramTerm(
        (Propagator("R", None, None, "x", "a"), Propagator("R", None, None, "b", "c")),
        Symbol("K", (), ("a", "b")), Rational(1), ("a", "b", "c"), (), 2,
        equal_time_aliases=(("b", "a"), ("c", "b")),
        r_absorbed_pairs=(("b", "c"),),
        group_templates=(("K", ("a", "b"), ("response", "physical"), (0, 0), "time"),),
    )
    formula = to_sympy(term)
    constraint, = formula.atoms(sp.DiracDelta)
    assert from_sympy(constraint) == TimeDelta("b", "a")
    response, = [p for p in formula.atoms(sp.Function) if p.func == sp.Function("R")]
    assert from_sympy(response) == Propagator("R", None, None, "x", "a")


@pytest.mark.parametrize("lag", [0.0, 0.31])
def test_time_only_delta_in_a_symbolic_expression_round_trips(lag):
    sp = pytest.importorskip("sympy")
    from sft_wick.expressions import TimeDelta
    from sft_wick.ir import from_sympy, to_sympy

    delta = TimeDelta("left", "right", lag)
    symbolic = to_sympy(delta)
    assert symbolic.func is sp.DiracDelta
    assert {p.name for p in symbolic.free_symbols} == {"time:left", "time:right"}
    assert from_sympy(symbolic) == delta

    term = DiagramTerm((), delta, Rational(1), ("left","right"), (), 0)
    wire = to_ir(term)
    assert wire["schema"] == "wickir/v1.1"
    assert from_ir(wire) == term
    with pytest.raises(ValueError, match="lag|finite|invalid WickIR"):
        wire["diagram"]["coupling_sum"]["lag"] = float("inf")
        from_ir(wire)
