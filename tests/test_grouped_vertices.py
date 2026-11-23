"""Space and time measures of ordered mixed-field interaction vertices."""
import pytest

from sft_wick import Action, Field, Vertex, compute_moment, compute_moment_numerical
from sft_wick.indices import IndexContext
from sft_wick.vertices import VertexInstance


def test_interleaved_slots_share_spacetime_or_time_only():
    phi, psi = Field("phi", "physical", 2), Field("psi", "response", 2)
    slots = (phi, psi, phi, psi)
    for mode in ("spacetime", "time"):
        vertex = Vertex(slots, "K", local=False, leg_groups=(1, 0, 0, 1),
                        group_mode=mode, group_lags=((1, 0, 0.25),))
        vi = VertexInstance.instantiate(vertex, IndexContext(), 0)
        points = vi.spatial_variables
        assert vi.coupling_symbol.indices == tuple(op.component_index for op in vi.field_operators)
        assert list(vi.coupling_symbol.spatial_args) == points
        if mode == "spacetime":
            assert [op.spatial_arg for op in vi.field_operators] == [points[i] for i in (1,0,0,1)]
            assert not vi.equal_time_aliases
            assert vi.time_lags == ((points[1], points[0], 0.25),)
        else:
            assert len(set(points)) == 4
            assert set(vi.equal_time_aliases) == {(points[3], points[0]), (points[2], points[1])}
            assert vi.time_lags == ((points[0], points[1], 0.25),)


def test_same_time_response_inside_one_group_disappears_from_both_engines():
    phi, psi = Field("phi", "physical"), Field("psi", "response")
    vertex = Vertex((phi, psi), "K", local=False, leg_groups=(0, 0), group_mode="time")
    obs = [phi("x"), phi("z")]
    action = Action(vertices=[vertex])
    for result in (compute_moment(obs, action, 1, certify=True).diagram_terms_by_order,
                   compute_moment_numerical(obs, action, 1, certify=True)):
        terms = result[1]
        for term in terms:
            aliases = dict(term.equal_time_aliases)
            assert aliases
            for edge in term.propagators:
                if edge.kind == "R":
                    assert aliases.get(edge.spatial_left, edge.spatial_left) != aliases.get(
                        edge.spatial_right, edge.spatial_right)


@pytest.mark.parametrize("groups,lags", [
    ((0,), ()), ((0, 2), ()), ((False, 0), ()),
    ((0, 1), ((0, 1, -1),)), ((0, 1), ((0, 1, float("nan")),)),
    ((0, 1), ((0, 1, 1), (1, 0, 1))), ((0, 1), ((0, 1, 1), (0, 1, 2))),
])
def test_invalid_group_and_lag_definitions_fail_at_construction(groups, lags):
    phi = Field("phi", "physical")
    with pytest.raises((TypeError, ValueError), match="group"):
        Vertex((phi, phi), "G", local=False, leg_groups=groups, group_lags=lags)


@pytest.mark.parametrize("collect", [True, False])
def test_lagged_measure_is_preserved_in_the_symbolic_expression(collect):
    phi, psi = Field("phi", "physical"), Field("psi", "response")
    action = Action([Vertex((phi, phi, psi), "D", False, leg_groups=(1, 1, 0),
                            group_lags=((1, 0, 0.31),))])
    result = compute_moment([phi("end")], action, 1, collect_topology=collect, certify=True)
    assert "0.31" in result.order_terms[1].to_latex()
    if collect:
        assert all("0.31" in dt.to_latex() for dt in result.diagram_terms(1))


def test_collected_lagged_copies_retain_their_own_ordered_kernel_arguments():
    phi, psi = Field("phi", "physical"), Field("psi", "response")
    action = Action([Vertex((phi, phi, psi), "D", False, leg_groups=(1, 1, 0),
                            group_lags=((1, 0, 0.31),))])
    obs = [phi("a"), phi("b")]
    for routing in ("enumerate", "multiplicity"):
        result = compute_moment(obs, action, 2, target="connected",
                                routing=routing, certify=True)
        assert result.diagram_terms(2)
        for term in result.diagram_terms(2):
            assert len(term.group_templates) == 2
            shifts = []
            for name, (parent, child), roles, groups, mode in term.group_templates:
                assert (name, roles, groups, mode) == (
                    "D", ("physical", "physical", "response"), (1, 1, 0), "spacetime")
                shifts.append((child, parent, 0.31))
            assert set(term.time_lags) == set(shifts)


def test_chained_and_zero_group_delays_share_exactly_the_specified_clocks():
    phi = Field("phi", "physical")
    vertex = Vertex([phi] * 4, "K", False, leg_groups=(0, 1, 2, 3),
                    group_lags=((1, 2, 0), (2, 0, 0.25), (3, 1, 0.5)))
    instance = VertexInstance.instantiate(vertex, IndexContext(), 0)
    a, b, c, d = instance.spatial_variables
    assert instance.equal_time_aliases == ((c, b),)
    assert instance.time_lags == ((b, a, 0.25), (d, a, 0.75))


@pytest.mark.parametrize("lags", [
    ((1, 0, 1.0), (2, 1, 1e-17)),
    ((1, 0, 1e-17), (2, 1, 1.0)),
    ((1, 0, 0.1), (2, 1, 0.2), (3, 0, 0.1 + 0.2)),
])
def test_flattening_must_preserve_distinct_clocks(lags):
    phi = Field("phi", "physical")
    with pytest.raises(ValueError, match="group_lags.*precision"):
        Vertex([phi] * 4, "K", False, leg_groups=(0, 1, 2, 3), group_lags=lags)
