"""Tests for perturbative expansion."""

from sft_wick import (
    Field, Vertex, Action, compute_moment, Rational, reset_uid_counter
)
from sft_wick.expressions import ZERO
import pytest


@pytest.fixture(autouse=True)
def _reset_uid():
    reset_uid_counter()


def test_zeroth_order_two_phi():
    """<phi_a(x) phi_b(y)>_S at order 0 = C_{ab}(x, y)."""
    phi = Field("phi", "physical", n_components=3)
    psi = Field("psi", "response", n_components=3)

    v = Vertex(fields=[phi, phi, psi], coupling="F")
    action = Action(vertices=[v])

    obs = [phi("a", "x"), phi("b", "y")]
    result = compute_moment(obs, action, order=0)

    zeroth = result.order(0)
    assert zeroth.to_latex() == "C_{ab}(x, y)"


def test_zeroth_order_scalar_psi_phi_phi_phi():
    """<psi(x) phi(x) phi(x) phi(x)>_S at order 0 = 3 R(x,x) C(x,x) without Itô."""
    phi = Field("phi", "physical")
    psi = Field("psi", "response")

    v = Vertex(fields=[phi, phi, psi], coupling="F")
    action = Action(vertices=[v])

    obs = [psi("x"), phi("x"), phi("x"), phi("x")]
    result = compute_moment(obs, action, order=0, ito=False, response_phase=False)

    zeroth = result.order(0)
    latex = zeroth.to_latex()
    # After simplification, we should have 3 identical terms combined
    # The exact format depends on simplification, but there should be 3 terms
    # or a coefficient of 3
    assert "R(x, x)" in latex
    assert "C(x, x)" in latex


def test_first_order_odd_total_vanishes():
    """When observable + vertex fields give odd total, result is zero.

    Observable: <phi(x) phi(y)> (2 fields)
    Vertex: phi phi psi (3 fields)
    Total: 5 (odd) -> zeroth order is C(x,y), first order is 0.
    """
    phi = Field("phi", "physical")
    psi = Field("psi", "response")

    v = Vertex(fields=[phi, phi, psi], coupling="F")
    action = Action(vertices=[v])

    obs = [phi("x"), phi("y")]
    result = compute_moment(obs, action, order=1)

    first = result.order(1)
    assert isinstance(first, Rational) and first.is_zero


def test_first_order_even_total():
    """Observable + vertex fields give even total -> non-trivial first order.

    Observable: <phi(x) phi(y)> (2 fields)
    Vertex: phi psi (2 fields) with coupling 'g'
    Total at order 1: 4 fields -> non-zero
    """
    phi = Field("phi", "physical")
    psi = Field("psi", "response")

    v = Vertex(fields=[phi, psi], coupling="g")
    action = Action(vertices=[v])

    obs = [phi("x"), phi("y")]
    result = compute_moment(obs, action, order=1, response_phase=False)

    first = result.order(1)
    # Should not be zero
    assert not (isinstance(first, Rational) and first.is_zero)

    latex = result.to_latex()
    assert "R" in latex or "C" in latex


def test_multinomial_coefficients():
    """With two vertices at order 2, verify correct combinations."""
    phi = Field("phi", "physical")
    psi = Field("psi", "response")

    v1 = Vertex(fields=[phi, psi], coupling="g")
    v2 = Vertex(fields=[phi, psi], coupling="h")
    action = Action(vertices=[v1, v2])

    combos = list(action.all_vertex_combinations(2))
    # (v1,v1), (v1,v2), (v2,v2) -> coefficients 1, 2, 1
    assert len(combos) == 3
    coeffs = [c for _, c in combos]
    assert sorted(coeffs) == [1, 1, 2]


def test_result_has_diagrams():
    """Verify that diagrams are generated."""
    phi = Field("phi", "physical")
    psi = Field("psi", "response")

    v = Vertex(fields=[phi, psi], coupling="g")
    action = Action(vertices=[v])

    obs = [phi("x"), phi("y")]
    result = compute_moment(obs, action, order=1)

    # Zeroth order should have 1 diagram
    assert len(result.diagrams_by_order[0]) == 1


# --- Itô prescription tests ---


def test_ito_zeroth_order_equal_point_vanishes():
    """<psi(x) phi(x) phi(x) phi(x)>_S at order 0 = 0 under Itô.

    All R(x,x) contractions vanish under Theta(0) = 0.
    """
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    v = Vertex(fields=[phi, phi, psi], coupling="F")
    action = Action(vertices=[v])

    obs = [psi("x"), phi("x"), phi("x"), phi("x")]

    # Without Itô: 3 R(x,x) C(x,x)
    result_no_ito = compute_moment(obs, action, order=0, ito=False, response_phase=False)
    assert "R(x, x)" in result_no_ito.order(0).to_latex()

    # With Itô (default): all pairings produce R(x,x) which vanishes
    result_ito = compute_moment(obs, action, order=0)
    assert isinstance(result_ito.order(0), Rational) and result_ito.order(0).is_zero


def test_ito_first_order_eliminates_intravertex_R():
    """Itô eliminates intra-vertex R contractions in local vertices.

    For <phi(x) phi(y)> with S_int = int g phi(z) psi(z) dz:
    At order 1 we have operators [phi(x), phi(y), phi(z), psi(z)].
    psi(z) can pair with phi(x)->R(x,z), phi(y)->R(y,z), or phi(z)->R(z,z).
    Under Itô, R(z,z) = 0, so fewer diagrams survive.
    """
    phi = Field("phi", "physical")
    psi = Field("psi", "response")

    v = Vertex(fields=[phi, psi], coupling="g")
    action = Action(vertices=[v])

    obs = [phi("x"), phi("y")]

    result_no_ito = compute_moment(obs, action, order=1, ito=False, response_phase=False)
    result_ito = compute_moment(obs, action, order=1, response_phase=False)

    n_no_ito = len(result_no_ito.diagrams_by_order.get(1, []))
    n_ito = len(result_ito.diagrams_by_order.get(1, []))

    # Itô should eliminate the R(z,z) diagram(s)
    assert n_ito < n_no_ito


def test_ito_zeroth_order_different_points_unaffected():
    """<phi(x) phi(y)>_S at order 0 = C(x,y) — unaffected by Itô.

    No R propagators at all, so ito flag has no effect.
    """
    phi = Field("phi", "physical", n_components=3)
    psi = Field("psi", "response", n_components=3)
    v = Vertex(fields=[phi, phi, psi], coupling="F")
    action = Action(vertices=[v])

    obs = [phi("a", "x"), phi("b", "y")]

    result_ito = compute_moment(obs, action, order=0, ito=True)
    assert result_ito.order(0).to_latex() == "C_{ab}(x, y)"


# --- Response phase tests ---


def test_response_phase_pure_c_unaffected():
    """<phi(x) phi(y)> at order 0 = C(x,y) — no R, so phase has no effect."""
    phi = Field("phi", "physical")
    action = Action(vertices=[])

    obs = [phi("x"), phi("y")]

    result = compute_moment(obs, action, order=0, response_phase=True)
    assert result.order(0).to_latex() == "C(x, y)"


def test_response_phase_with_r():
    """response_phase=True adds (-i) factors per R propagator."""
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    action = Action(vertices=[])

    obs = [phi("x"), psi("y")]

    # With response_phase (default)
    result = compute_moment(obs, action, order=0, ito=False)
    latex = result.order(0).to_latex()
    assert r"\mathrm{i}" in latex

    # Without response_phase
    result_raw = compute_moment(obs, action, order=0, ito=False, response_phase=False)
    latex_raw = result_raw.order(0).to_latex()
    assert r"\mathrm{i}" not in latex_raw
    assert "R(x, y)" in latex_raw


def test_response_phase_disabled():
    """response_phase=False gives raw R propagators without -i factors."""
    phi = Field("phi", "physical")
    psi = Field("psi", "response")

    v = Vertex(fields=[phi, psi], coupling="g")
    action = Action(vertices=[v])

    obs = [phi("x"), phi("y")]

    result = compute_moment(obs, action, order=1, response_phase=False)
    latex = result.order(1).to_latex()
    assert r"\mathrm{i}" not in latex


# --- Diagram-based collection tests ---


def test_collect_diagram_coupling_permutation_order1():
    """At order 1 with multi-component cubic vertex, coupling indices
    should be correctly permuted in the collected output.

    With obs=[phi_a(x)] and vertex F_{ijk} phi_i phi_j psi_k at order 1:
    operators = [phi_a(x), phi_{i0}(y_0), phi_{i1}(y_0), psi_{i2}(y_0)]
    Only one pairing survives (ψ pairs with external φ):
        psi_{i2} <-> phi_a -> R_{a,i2}(x,y_0)
        phi_{i0} <-> phi_{i1} -> C_{i0,i1}(y_0,y_0)
    So there's only one topology — coupling is F_{i0,i1,i2} as-is.
    """
    phi = Field("phi", "physical", n_components=3)
    psi = Field("psi", "response", n_components=3)

    v = Vertex(fields=[phi, phi, psi], coupling="F")
    action = Action(vertices=[v])

    obs = [phi("a", "x")]
    result = compute_moment(obs, action, order=1, response_phase=False)

    first = result.order(1)
    latex = first.to_latex()
    assert "R" in latex
    assert "C" in latex
    assert "F" in latex


def test_collect_diagram_multiple_pairings():
    """At order 1 with quadratic vertex g_{ij} phi_i psi_j,
    observable [phi_a(x), phi_b(y), phi_c(z), phi_d(w)]:
    Multiple pairings arise — test that the collection produces
    correct (non-zero) output.
    """
    phi = Field("phi", "physical", n_components=3)
    psi = Field("psi", "response", n_components=3)

    v = Vertex(fields=[phi, psi], coupling="g")
    action = Action(vertices=[v])

    obs = [phi("a", "x"), phi("b", "y"), phi("c", "z"), phi("d", "w")]
    result = compute_moment(obs, action, order=1, response_phase=False)

    first = result.order(1)
    # Should have non-trivial result
    assert not (isinstance(first, Rational) and first.is_zero)
    latex = first.to_latex()
    assert "R" in latex
    assert "C" in latex


def test_collect_diagram_scalar_degeneracy():
    """For scalar fields (no component indices), topology collection
    should produce integer degeneracy factors.
    """
    phi = Field("phi", "physical")
    psi = Field("psi", "response")

    v = Vertex(fields=[phi, psi], coupling="g")
    action = Action(vertices=[v])

    obs = [phi("x"), phi("y")]
    result = compute_moment(obs, action, order=1, response_phase=False)
    first = result.order(1)
    assert not (isinstance(first, Rational) and first.is_zero)


def test_collect_diagram_order2_same_vertex():
    """At order 2 with same vertex type, some pairings are related by
    swapping vertex copies (y_0 <-> y_1). The new collect_by_diagram
    should group these, reducing the number of distinct topology terms.
    """
    phi = Field("phi", "physical")
    psi = Field("psi", "response")

    v = Vertex(fields=[phi, psi], coupling="g")
    action = Action(vertices=[v])

    obs = [phi("x"), phi("y")]

    # With collection
    result_on = compute_moment(obs, action, order=2,
                               response_phase=False, collect_topology=True)
    # Without collection
    reset_uid_counter()
    result_off = compute_moment(obs, action, order=2,
                                response_phase=False, collect_topology=False)

    # Both should be non-zero at order 2
    expr_on = result_on.order(2)
    expr_off = result_off.order(2)
    assert not (isinstance(expr_on, Rational) and expr_on.is_zero)
    assert not (isinstance(expr_off, Rational) and expr_off.is_zero)


def test_collect_diagram_backward_compat_alias():
    """collect_by_topology is an alias for collect_by_diagram."""
    from sft_wick.simplify import collect_by_topology, collect_by_diagram
    assert collect_by_topology is collect_by_diagram


# --- DiagramTerm tests ---


def test_diagram_term_zeroth_order():
    """DiagramTerm is populated at order 0."""
    phi = Field("phi", "physical", n_components=3)
    obs = [phi("a", "x"), phi("b", "y")]
    result = compute_moment(obs, Action(vertices=[]), order=0)

    dts = result.diagram_terms(0)
    assert len(dts) == 1
    dt = dts[0]
    assert len(dt.propagators) == 1
    assert dt.propagators[0].kind == "C"
    assert dt.rational_prefactor == Rational(1)
    assert dt.integration_vars == ()
    assert dt.summation_indices == ()
    assert dt.n_response == 0
    assert dt.response_phase_factor() == 1.0


def test_diagram_term_order1_scalar():
    """DiagramTerm at order 1 with scalar phi-psi vertex."""
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    v = Vertex(fields=[phi, psi], coupling="g")
    action = Action(vertices=[v])

    obs = [phi("x"), phi("y")]
    result = compute_moment(obs, action, order=1, response_phase=False)

    dts = result.diagram_terms(1)
    assert len(dts) == 2  # Two distinct spatial topologies
    for dt in dts:
        assert len(dt.propagators) == 2
        assert dt.rational_prefactor == Rational(-1)
        assert dt.integration_vars == ("y_0",)
        assert dt.summation_indices == ()
        assert dt.n_response == 1
        assert dt.response_phase_factor() == -1j
        # Coupling is just 'g' (scalar, no index permutation)
        assert "g" in dt.coupling_sum.to_latex()


def test_diagram_term_order1_multicomponent():
    """DiagramTerm at order 1 with multi-component cubic vertex."""
    phi = Field("phi", "physical", n_components=3)
    psi = Field("psi", "response", n_components=3)
    v = Vertex(fields=[phi, phi, psi], coupling="F")
    action = Action(vertices=[v])

    obs = [phi("a", "x")]
    result = compute_moment(obs, action, order=1, response_phase=False)

    dts = result.diagram_terms(1)
    assert len(dts) == 1
    dt = dts[0]
    assert len(dt.propagators) == 2
    assert dt.rational_prefactor == Rational(-1)
    assert dt.integration_vars == ("y_0",)
    assert len(dt.summation_indices) == 3
    assert all(dim == 3 for _, dim in dt.summation_indices)
    assert dt.n_response == 1
    # Check spatial topology
    topo = dt.spatial_topology()
    kinds = sorted(k for k, _, _ in topo)
    assert kinds == ["C", "R"]


def test_diagram_term_evaluate_coupling():
    """evaluate_coupling substitutes numeric values correctly."""
    import numpy as np

    phi = Field("phi", "physical", n_components=3)
    psi = Field("psi", "response", n_components=3)
    v = Vertex(fields=[phi, phi, psi], coupling="F")
    action = Action(vertices=[v])

    obs = [phi("a", "x")]
    result = compute_moment(obs, action, order=1)

    dt = result.diagram_terms(1)[0]

    # F tensor with a single nonzero entry
    F = np.zeros((3, 3, 3))
    F[0, 1, 2] = 6.0

    arr = dt.evaluate_coupling({"F": F})
    # rational_prefactor = -1, n_response = 1 → phase = (-i)^1 = -i
    # full prefactor = (-1) × (-i) = i, so result[0,1,2] = 6j
    assert arr.shape == (3, 3, 3)
    assert arr.dtype == complex
    assert arr[0, 1, 2] == pytest.approx(6j)
    # Total nonzero entries: only (0,1,2) since coupling is F_{i0,i1,i2}
    assert np.count_nonzero(arr) == 1


def test_diagram_term_order2_has_coupling_sums():
    """At order 2, some diagrams should have coupling permutation sums."""
    phi = Field("phi", "physical", n_components=3)
    psi = Field("psi", "response", n_components=3)
    v = Vertex(fields=[phi, phi, psi], coupling="F")
    action = Action(vertices=[v])

    obs = [phi("a", "x"), phi("b", "y")]
    result = compute_moment(obs, action, order=2, response_phase=False)

    dts = result.diagram_terms(2)
    assert len(dts) > 0

    # All should have 4 propagators (2 vertices × 2 edges each, minus vertex pairing)
    for dt in dts:
        assert len(dt.propagators) == 4
        assert dt.rational_prefactor == Rational(1, 2)
        assert len(dt.integration_vars) == 2
        assert len(dt.summation_indices) == 6
        assert dt.n_response == 2

    # At least one diagram should have a coupling sum (Sum expression)
    from sft_wick.expressions import Sum
    has_sum = any(isinstance(dt.coupling_sum, Sum) for dt in dts)
    assert has_sum, "Expected at least one diagram with a coupling permutation sum"


def test_diagram_term_empty_for_operator_level():
    """collect_topology=False does not populate diagram_terms."""
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    v = Vertex(fields=[phi, psi], coupling="g")
    action = Action(vertices=[v])

    obs = [phi("x"), phi("y")]
    result = compute_moment(obs, action, order=1,
                            response_phase=False, collect_topology=False)

    # Order 0 still has diagram terms (uses wick_contract directly)
    assert len(result.diagram_terms(0)) == 1
    # Order 1 with collect_topology=False does NOT populate diagram_terms
    assert len(result.diagram_terms(1)) == 0
