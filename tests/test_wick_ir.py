"""Transport of evaluator-ready Wick diagrams and their conventions."""

import json
from pathlib import Path

import numpy as np
import pytest

from sft_wick import Action, Field, PropagatorCache, PropagatorModel, compute_moment, integrate_moment
from sft_wick.expressions import (
    I,
    DiracDelta,
    IntegralOver,
    KroneckerDelta,
    Product,
    Propagator,
    Rational,
    Sum,
    SumOverIndex,
    Symbol,
)
from sft_wick.ir import from_ir, ir_digest, to_ir
from sft_wick.perturbation import DiagramTerm
from sft_wick.workflow.config import build_system, expand_kwargs, load_workflow_config


def test_quickstart_round_trip():
    cfg = load_workflow_config(Path(__file__).parents[1] / "examples/quickstart.yaml")
    system = build_system(cfg.system)
    kwargs = expand_kwargs(cfg) | {"orders": [0, 2, 4], "cache_path": None}
    expansion = system.expand(**kwargs)
    assert [len(expansion.diagrams(n)) for n in expansion.orders] == [1, 6, 64]
    settings = {"t_min": system.t_min, "iso_R": system.iso_R, "iso_C": kwargs["iso_C"]}
    for order in expansion.orders:
        for diagram in expansion.diagrams(order):
            wire = to_ir(diagram, **settings)
            decoded = from_ir(json.loads(json.dumps(wire)))
            assert decoded == diagram
            assert to_ir(decoded, **settings) == wire
            assert decoded.to_latex() == diagram.to_latex()
            assert ir_digest(wire) == ir_digest(to_ir(decoded, **settings))


@pytest.mark.parametrize("dimension", [np.int64(2), 2.0])
def test_integer_like_summation_dimensions_encode_as_json_integers(dimension):
    diagram = DiagramTerm((), Rational(1), Rational(1), (), (("i", dimension),), 0)
    wire = to_ir(diagram)
    assert type(wire["diagram"]["summation_indices"][0][1]) is int
    assert from_ir(wire) == diagram


def test_round_trip_preserves_phase_delta_and_absorbed_response():
    coupling = Sum((
        Product((I, KroneckerDelta("a", "b"), Symbol("F", ("a",), ("x",), local=True))),
        IntegralOver("z", SumOverIndex("j", 2, DiracDelta("x", "z"))),
    ))
    diagram = DiagramTerm(
        (Propagator("R", "a", "a", "x", "y"), Propagator("C", None, None, "x", "u")),
        coupling, Rational(1, 3), ("y",), (("i", 2),), 1,
        equal_time_aliases=(("y", "x"),), r_absorbed_pairs=(("x", "y"),),
        n_external_response=1, response_phase=False,
    )
    wire = to_ir(diagram, t_min=0.125, target="connected", iso_C=True)
    assert wire["spatial_structure"]["time_integration_vars"] == []
    assert wire["spatial_structure"]["time_orderings"] == []
    assert ["x", "y"] in wire["spatial_structure"]["direction_groups"]
    assert wire["conventions"]["response_phase"] is False
    assert wire["conventions"]["observable_time"]["t_min"] == 0.125
    assert wire["conventions"]["iso_C_scalar"] == "common_diagonal"
    decoded = from_ir(wire)
    assert decoded.response_phase_factor() == -1j
    assert decoded.observable_phase_factor() == 1j
    assert decoded == diagram
    assert decoded.to_latex() == diagram.to_latex()
    assert to_ir(decoded, t_min=0.125, target="connected", iso_C=True) == wire


@pytest.mark.parametrize(
    ("settings", "rejection"),
    [
        ({"positions": {"x": [1, 2], "y": [1, 3]}}, "r_group_position_conflict"),
        ({"integrate_over": ["u"]}, "invalid_integrated_points"),
        ({"integrate_over": "all", "external_times": {"y": 0.8}}, "invalid_external_times"),
        ({"external_times": {"u": 0.8}}, "invalid_external_times"),
        ({"positions": {"x": np.array([1, 2]), "y": [1, 2], "z": [3, 4]},
          "external_times": {"y": 0.8}, "t_final": 1.0}, None),
    ],
)
def test_evaluation_context_records_refusals(settings, rejection):
    diagram = DiagramTerm(
        (Propagator("R", None, None, "x", "y"),), Rational(1), Rational(1),
        (), (), 1, n_external_response=1,
    )
    wire = to_ir(diagram, **settings)
    error = wire["evaluation"]["rejection"]
    assert (error["code"] if error else None) == rejection
    assert from_ir(wire) == diagram


def test_external_sweep_ceiling_does_not_follow_later_pinned_time():
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    result = compute_moment([phi("x"), psi("y")], Action([]), order=0)
    diagram, = result.diagram_terms(0)
    context = dict(t_min=0.2, external_times={"x": 2.0}, integrate_over={"y"})
    wire = to_ir(diagram, t_final=1.0, **context)
    time = wire["conventions"]["observable_time"]
    assert time["swept_external_ceiling"] == "t_final"
    assert time["integration_ceiling"] == "max(t_final, *external_times.values())"
    cache = PropagatorCache(PropagatorModel(
        R_time=lambda left, right: np.exp(-0.8 * (left - right)),
        kappa2=lambda *args: np.zeros((1, 1)), n_components=1, iso_R=True, t_min=0.2,
    ))
    value, _ = integrate_moment(
        from_ir(wire).build_integrand({}), 1.0, cache, method="gauss_legendre", **context,
    )
    assert value == pytest.approx((np.exp(-0.8) - np.exp(-1.44)) / 0.8, rel=0, abs=1e-14)

    # The earlier v1 record omitted the external cap. Reading it preserves
    # both the diagram and the old record's digest.
    del time["swept_external_ceiling"]
    old_digest = ir_digest(wire)
    assert from_ir(wire) == diagram
    assert ir_digest(wire) == old_digest
    time["swept_external_ceiling"] = "integration_ceiling"
    with pytest.raises(ValueError, match="unsupported WickIR v1 conventions"):
        from_ir(wire)


def test_single_component_cache_accepts_distinct_absorbed_r_labels():
    diagram = DiagramTerm(
        (Propagator("R", "1", "+1", "x", "y"),), Rational(1), Rational(1),
        ("y",), (), 1, equal_time_aliases=(("y", "x"),),
        r_absorbed_pairs=(("x", "y"),), n_external_response=1,
    )
    wire = to_ir(diagram, n_components=np.int64(1))
    assert wire["evaluation"]["n_components"] == 1
    assert wire["evaluation"]["rejection"] is None
    assert "conditional_rejection" not in wire["evaluation"]
    cache = PropagatorCache(PropagatorModel(
        R_time=lambda *args: 1.0, kappa2=lambda *args: np.zeros((1, 1)),
        n_components=1, iso_R=True,
    ))
    value, _ = integrate_moment(
        from_ir(wire).build_integrand({}), 1.0, cache, method="gauss_legendre",
    )
    assert value == 1.0


def test_n_components_must_be_a_positive_integer():
    diagram = DiagramTerm((), Rational(1), Rational(1), (), (), 0)
    for size in (0, -1, True, np.bool_(True), 2.0):
        with pytest.raises(ValueError, match="n_components must be a positive integer"):
            to_ir(diagram, n_components=size)


def test_unknown_nodes_schema_and_wrong_ito_are_refused():
    diagram = DiagramTerm((), Rational(1), Rational(1), (), (), 0)
    with pytest.raises(ValueError, match="external point names as strings"):
        to_ir(diagram, integrate_over=["x", 1])
    with pytest.raises(ValueError, match="positions needs external point names as strings"):
        to_ir(diagram, positions={1: 0.0})
    with pytest.raises(ValueError, match="external_times needs external point names as strings"):
        to_ir(diagram, external_times={1: 0.5})
    wire = to_ir(diagram)
    with pytest.raises(ValueError, match="schema"):
        from_ir(wire | {"schema": "wickir/v2"})
    wire["conventions"]["theta_at_zero"] = 1
    with pytest.raises(ValueError, match="theta_at_zero"):
        from_ir(wire)
    wire["conventions"]["theta_at_zero"] = 0
    wire["diagram"]["coupling_sum"]["type"] = "unknown"
    with pytest.raises(ValueError, match="expression type"):
        from_ir(wire)
