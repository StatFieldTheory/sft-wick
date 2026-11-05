"""Tests for perturbative expansion."""

from sft_wick import (
    Field, Vertex, Action, compute_moment, compute_moment_numerical, Rational,
    reset_uid_counter, wick_contract,
)
from sft_wick.propagators import contract_pair
from sft_wick.expressions import (
    ZERO, ImaginaryUnit, IntegralOver, Product, Propagator, Sum, SumOverIndex,
    Symbol,
)
import math
import numpy as np
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

    # All pairings produce R(x,x), which vanishes
    result = compute_moment(obs, action, order=0)
    assert isinstance(result.order(0), Rational) and result.order(0).is_zero


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

    result = compute_moment(obs, action, order=1, response_phase=False)

    # R(x,z) C(y,z) and R(y,z) C(x,z) survive; R(z,z) C(x,y) does not.
    assert len(result.diagrams_by_order[1]) == 2


def test_zeroth_order_at_different_points_is_one_C():
    """<phi(x) phi(y)>_S at order 0 = C(x,y), with no R propagator."""
    phi = Field("phi", "physical", n_components=3)
    psi = Field("psi", "response", n_components=3)
    v = Vertex(fields=[phi, phi, psi], coupling="F")
    action = Action(vertices=[v])

    obs = [phi("a", "x"), phi("b", "y")]

    result = compute_moment(obs, action, order=0)
    assert result.order(0).to_latex() == "C_{ab}(x, y)"


def test_flags_are_keyword_only():
    """The fourth argument was ``ito``; a boolean there must not be read as
    ``response_phase``."""
    phi = Field("phi", "physical")
    with pytest.raises(TypeError):
        compute_moment([phi("x"), phi("y")], Action(vertices=[]), 0, True)


@pytest.mark.parametrize("call", [
    lambda obs, a: compute_moment(obs, a, 0, ito=True),
    lambda obs, a: compute_moment(obs, a, 0, ito=False),
    lambda obs, a: compute_moment_numerical(obs, a, 0, ito=False),
    # The sixth positional was ``ito``, and would now be read as a flag.
    lambda obs, a: compute_moment_numerical(obs, a, 0, {}, None, True),
    lambda obs, a: wick_contract(obs, ito=False),
    # ``keep`` took the second slot, where a positional ``ito`` was.
    lambda obs, a: wick_contract(
        [obs[0], Field("psi", "response")("x")], False),
    lambda obs, a: contract_pair(obs[0], obs[1], ito=False),
])
def test_the_ito_switch_is_gone(call):
    """A silently ignored ``ito=False`` would look like a Stratonovich run."""
    phi = Field("phi", "physical")
    with pytest.raises(TypeError):
        call([phi("x"), phi("y")], Action(vertices=[]))


def test_iso_flags_strip_a_term_with_nothing_to_merge():
    """The term kept ``R_{aa}`` where the expression of the same contraction
    had an index-free R, and skipped the refusal of an R that is no multiple
    of the identity."""
    phi = Field("phi", "physical", n_components=2)
    psi = Field("psi", "response", n_components=2)
    result = compute_moment(
        [phi("a", "x"), psi("a", "y")], Action(vertices=[]), 0, iso_R=True)
    term = result.diagram_terms(0)[0]
    assert (term.propagators[0].index_left, term.propagators[0].index_right) == (
        None, None)


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
    result = compute_moment(obs, action, order=0)
    latex = result.order(0).to_latex()
    assert r"\mathrm{i}" in latex

    # Without response_phase
    result_raw = compute_moment(obs, action, order=0, response_phase=False)
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


def test_diagram_terms_refuses_an_order_the_operator_level_built_none_for():
    """collect_topology=False builds no diagrams above order 0, and an empty
    list would integrate to 0 where the expression is not."""
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    v = Vertex(fields=[phi, psi], coupling="g")
    cubic = Vertex(fields=[phi, phi, psi], coupling="F")

    obs = [phi("x"), phi("y")]
    result = compute_moment(obs, Action(vertices=[v]), order=1,
                            response_phase=False, collect_topology=False)

    # Order 0 still has diagram terms (uses wick_contract directly)
    assert len(result.diagram_terms(0)) == 1
    with pytest.raises(ValueError, match="no numerical evaluation"):
        result.diagram_terms(1)
    assert 1 not in result.diagram_terms_by_order

    # An order that is 0 has no diagrams to miss.
    zero = compute_moment(obs, Action(vertices=[cubic]), order=1,
                          response_phase=False, collect_topology=False)
    assert zero.diagram_terms(1) == []
    assert result.diagram_terms(7) == []


def _on_points(expr, C, R, couplings, points):
    """Value of a one-component expression on a finite set of points: each
    integral is a sum over the rows of the tables."""
    if isinstance(expr, Rational):
        return expr.numerator / expr.denominator
    if isinstance(expr, ImaginaryUnit):
        return 1j
    if isinstance(expr, Symbol):
        return couplings[expr.name]
    if isinstance(expr, Propagator):
        table = C if expr.kind == "C" else R
        return table[points[expr.spatial_left], points[expr.spatial_right]]
    if isinstance(expr, Sum):
        return sum(_on_points(t, C, R, couplings, points) for t in expr.terms)
    if isinstance(expr, Product):
        return math.prod(
            _on_points(f, C, R, couplings, points) for f in expr.factors
        )
    if isinstance(expr, SumOverIndex):
        return expr.dimension * _on_points(expr.body, C, R, couplings, points)
    if isinstance(expr, IntegralOver):
        return sum(
            _on_points(expr.body, C, R, couplings, {**points, expr.variable: z})
            for z in range(len(C))
        )
    raise TypeError(type(expr))


def _linear_series(C, G, g, order, x, w):
    """Truncated Dyson series of <phi(x) phi(w)> under S_int = g sum phi psi,
    with G = <phi psi>: order n is (-g)^n sum_{a+b=n} G^a C (G^T)^b."""
    powers = [np.linalg.matrix_power(G, k) for k in range(order + 1)]
    return sum(
        (-g) ** (a + b) * (powers[a] @ C @ powers[b].T)[x, w]
        for a in range(order + 1)
        for b in range(order + 1 - a)
    )


@pytest.mark.parametrize("response_phase", [True, False])
@pytest.mark.parametrize("g", [1e-8, 0.7, 1e8])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_total_is_the_series_with_each_phase_applied_once(order, g, response_phase):
    """Orders up to 4 hold terms with 1 to 4 R, so a phase applied twice shows
    in three of the four residues of n_R mod 4.  The tables have no symmetry
    and the reference is matrix algebra, with no contraction in it."""
    rng = np.random.default_rng(23)
    C = rng.normal(size=(4, 4))
    C = C + C.T
    R = rng.normal(size=(4, 4))
    phi, psi = Field("phi", "physical"), Field("psi", "response")
    action = Action(vertices=[Vertex(fields=[phi, psi], coupling="g")])

    result = compute_moment(
        [phi("x"), phi("w")], action, order=order, response_phase=response_phase
    )

    points = {"x": 0, "w": 2}
    G = -1j * R if response_phase else R
    expected = _linear_series(C, G, g, order, 0, 2)
    assert _on_points(result.total, C, R, {"g": g}, points) == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )


def test_total_of_a_single_nonzero_order_keeps_its_phase():
    """One nonzero order takes the branch that never reapplied the phase."""
    R = np.array([[0.0, 3.0], [5.0, 0.0]])
    phi, psi = Field("phi", "physical"), Field("psi", "response")

    result = compute_moment([phi("x"), psi("w")], Action(vertices=[]), 0)

    value = _on_points(result.total, 0 * R, R, {}, {"x": 0, "w": 1})
    assert value == pytest.approx(-3j, rel=1e-15, abs=0.0)


# =====================================================================
# Regression test: diag_C must keep delta_{a, b} for external pairs
# =====================================================================


def test_apply_diagonal_keeps_delta_for_external_indices():
    """Order-0 ``<phi_a(x) phi_b(y)>`` under ``diag_C=True`` must
    evaluate to 0 when the observable component pair is off-diagonal
    (a != b), and to ``C(t1, t2; x, y)`` (encoded as 1 in the
    coupling sum) when a == b.

    Regression-locks the fix to ``DiagramTerm.apply_diagonal``: the
    delta_{a, b} factor that pins the C propagator to its diagonal
    entries must be retained as a ``KroneckerDelta`` factor in the
    coupling expression for index pairs that are NOT summation
    indices (i.e. observable / external labels). Before the fix the
    union-find rename collapsed b -> a and dropped the delta, which
    only happens to be correct when the caller pins a == b.

    Before the fix, a fresh order-0 expansion at component pair
    (a=0, b=1) returned the diagonal C value (~0.012 in demo2)
    instead of 0, masking the bug end-to-end on diagonal pairs and
    requiring downstream code (analysis.ipynb's xi_0 helper) to
    hard-code the delta itself.
    """
    import sft_wick as sw

    sw.reset_uid_counter()
    phi = sw.Field("phi", "physical", n_components=2)
    psi = sw.Field("psi", "response", n_components=2)
    obs = [phi("a", "x"), phi("b", "y")]
    action = sw.Action(vertices=[])

    result = sw.compute_moment(
        obs, action, order=0,
        diag_R=True, diag_C=True, iso_R=True, iso_C=True,
    )
    dts = result.diagram_terms(0)
    assert len(dts) == 1
    dt = dts[0]

    # Diagonal pair: should evaluate to 1 (the C propagator
    # symbol; numerical C value is supplied by the cache later).
    val_00 = float(dt.evaluate_coupling({}, fixed_indices={"a": 0, "b": 0}))
    val_11 = float(dt.evaluate_coupling({}, fixed_indices={"a": 1, "b": 1}))
    assert val_00 == 1.0
    assert val_11 == 1.0

    # Cross pair: must be 0 (delta_{0, 1} = 0).
    val_01 = float(dt.evaluate_coupling({}, fixed_indices={"a": 0, "b": 1}))
    val_10 = float(dt.evaluate_coupling({}, fixed_indices={"a": 1, "b": 0}))
    assert val_01 == 0.0, (
        f"order-0 cross-pair (a=0, b=1) must vanish under diag_C=True; "
        f"got {val_01}. The delta_{{a, b}} from C[a, b] = delta_{{a, b}} "
        f"C[a, a] is being dropped during diagonal simplification."
    )
    assert val_10 == 0.0


# ---------------------------------------------------------------------------
# One rule per vertex copy, in both engines
# ---------------------------------------------------------------------------


def _unit_weight(expr) -> float:
    """``expr`` with every coupling set to 1."""
    if isinstance(expr, Sum):
        return sum(_unit_weight(t) for t in expr.terms)
    if isinstance(expr, Product):
        return math.prod(_unit_weight(f) for f in expr.factors)
    if isinstance(expr, Rational):
        return expr.numerator / expr.denominator
    return 1.0


def _weight_by_flagged_legs(terms) -> dict:
    """Unit-coupling weight of the terms, by the legs that carry an alias."""
    weights = {}
    for dt in terms:
        aliased = len(dt.equal_time_aliases)
        prefactor = dt.rational_prefactor
        weights[aliased] = weights.get(aliased, 0.0) + _unit_weight(
            dt.coupling_sum
        ) * prefactor.numerator / prefactor.denominator
    return weights


@pytest.mark.parametrize("flag, legs_per_flagged_copy", [
    ("equal_time", 1),
    ("already_R_contracted", 2),
])
def test_a_flagged_and_a_plain_vertex_may_share_a_coupling_name(
    flag, legs_per_flagged_copy
):
    """Four externals on two two-leg vertices give 4! routings.  Hand count of
    the prefactors of S_int^2 = KK + 2 KQ + QQ, each over 2!: 12 with two
    flagged copies, 24 with one, 12 with none.  The name used to mark the
    plain copy as flagged, which put all 36 on the two-copy measure."""
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    observable = [phi(s) for s in ("a1", "a2", "a3", "a4")]
    action = Action([
        Vertex([psi, psi], "K", local=False, **{flag: True}),
        Vertex([psi, psi], "K", local=False),
    ])
    want = {
        2 * legs_per_flagged_copy: 12.0,
        legs_per_flagged_copy: 24.0,
        0: 12.0,
    }
    terms = compute_moment(observable, action, 2).diagram_terms(2)
    assert _weight_by_flagged_legs(terms) == want
    assert _weight_by_flagged_legs(
        compute_moment_numerical(observable, action, 2)[2]
    ) == want


def test_numerical_terms_evaluate_a_callable_coupling_at_its_legs():
    r"""<phi(x) phi(y) phi(z)> at order 1 on one three-leg vertex, with
    kappa(s) = exp(c . s), R(t, s) = exp(-g (t - s)) and three distinct
    external times, is a sum over the 3! routings of
    prod_k (e^(c_k T_k) - e^(-g T_k)) / (c_k + g).  The factor i/6 cancels the
    (-1) (-i)^3 of the expansion and the 3! of the sum."""
    from sft_wick.evaluate import (
        PropagatorCache,
        PropagatorModel,
        integrate_moment,
    )
    from itertools import permutations

    gamma, rates = 0.7, (-0.4, 0.3, -1.1)
    times = {"x": 0.02, "y": 1.3, "z": 6.0}

    def leg(rate, t):
        return (np.exp(rate * t) - np.exp(-gamma * t)) / (rate + gamma)

    want = sum(
        math.prod(leg(c, t) for c, t in zip(rates, order))
        for order in permutations(times.values())
    ) / 6.0

    def kappa(_points, leg_times):
        return 1j / 6.0 * np.exp(np.dot(rates, leg_times))

    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    action = Action([Vertex([psi, psi, psi], "K", local=False)])
    observable = [phi(s) for s in times]
    cache = PropagatorCache(PropagatorModel(
        R_time=lambda t1, t2: np.exp(-gamma * (t1 - t2)),
        kappa2=lambda n1, t1, n2, t2: 0.0,
        n_components=1,
    ))
    terms = compute_moment_numerical(observable, action, 1)[1]
    assert len(terms) == 6
    total = sum(
        integrate_moment(
            term.build_integrand({"K": kappa}),
            6.0,
            cache,
            method="gauss_legendre",
            n_gauss=48,
            external_times=times,
        )[0]
        for term in terms
    )
    assert total == pytest.approx(want, rel=1e-10, abs=0.0)


@pytest.mark.parametrize("shown", [True, False])
def test_response_phase_flag_changes_the_display_and_no_number(shown):
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    action = Action([Vertex([phi, psi], "L")])
    result = compute_moment(
        [phi("x"), phi("w")], action, 1, response_phase=shown
    )
    ((term, *_)) = result.diagram_terms(1)
    values = {"L": np.array(2.0)}
    assert (r"\mathrm{i}" in result.order(1).to_latex()) is shown
    assert (r"\mathrm{i}" in result.to_latex()) is shown
    assert (r"\mathrm{i}" in term.to_latex()) is shown
    assert ("j" in term.to_latex_evaluated(values)) is shown
    assert term.evaluate_coupling(values) == 2j


def test_an_imaginary_factor_survives_a_real_coupling_array():
    """A hand-built ``I K`` with a real K: the batched entry returned zeros
    with a ComplexWarning, and the scalar entry raised TypeError."""
    from sft_wick.expressions import I, Product, Rational, Symbol
    from sft_wick.perturbation import DiagramTerm

    term = DiagramTerm(
        propagators=(), coupling_sum=Product((I, Symbol("K"))),
        rational_prefactor=Rational(1), integration_vars=(), summation_indices=(),
        n_response=0,
    )
    assert term.evaluate_coupling({"K": 2.0}) == 2j
    np.testing.assert_array_equal(
        term.evaluate_coupling_batched({"K": np.array([2.0, 3.0])}, 2), [2j, 3j]
    )
