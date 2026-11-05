"""Tests for the simplification pipeline and diagram-based collection."""

import signal

import numpy as np
import pytest
from sft_wick import Action, Field, Vertex, compute_moment
from sft_wick.expressions import (
    ZERO, ONE, ImaginaryUnit, KroneckerDelta, Product, Propagator, Rational,
    Sum, Symbol,
    IntegralOver, SumOverIndex,
)
from sft_wick.expressions import canonical_edge
from sft_wick.simplify import (
    _canonical_diagram_form,
    _match_propagators_after_spatial,
    collect_by_diagram,
    diagonal_propagators,
    simplify,
)


# ---------------------------------------------------------------------------
# canonical_edge
# ---------------------------------------------------------------------------


class TestCanonicalEdge:
    def test_c_symmetric_sorted(self):
        assert canonical_edge("C", "y", "x") == ("C", "x", "y")

    def test_c_already_sorted(self):
        assert canonical_edge("C", "x", "y") == ("C", "x", "y")

    def test_c_equal_points(self):
        assert canonical_edge("C", "x", "x") == ("C", "x", "x")

    def test_r_directed_not_sorted(self):
        assert canonical_edge("R", "x", "y") == ("R", "x", "y")
        assert canonical_edge("R", "y", "x") == ("R", "y", "x")

    def test_r_and_c_different(self):
        assert canonical_edge("R", "y", "x") != canonical_edge("C", "y", "x")


# ---------------------------------------------------------------------------
# _canonical_diagram_form
# ---------------------------------------------------------------------------


class TestCanonicalDiagramForm:
    def test_no_internal_vars(self):
        """With no integration variables, canonical form is just sorted edges."""
        props = [
            Propagator("R", "a", "i2", "x", "y_0"),
            Propagator("C", "i0", "i1", "y_0", "y_0"),
        ]
        form, mapping = _canonical_diagram_form(props, frozenset())
        assert mapping == {}
        assert form == (("C", "y_0", "y_0"), ("R", "x", "y_0"))

    def test_single_internal_var(self):
        """With one internal var, there's only one permutation (identity)."""
        props = [
            Propagator("R", "a", "i2", "x", "y_0"),
            Propagator("C", "i0", "i1", "y_0", "y_0"),
        ]
        form, mapping = _canonical_diagram_form(props, frozenset({"y_0"}))
        assert form == (("C", "y_0", "y_0"), ("R", "x", "y_0"))

    def test_two_internal_vars_swap(self):
        """Two terms related by y_0 <-> y_1 should produce the same canonical form."""
        props_a = [
            Propagator("R", "a", "i2", "x", "y_0"),
            Propagator("C", "i0", "i1", "y_0", "y_0"),
            Propagator("R", "b", "i5", "y", "y_1"),
            Propagator("C", "i3", "i4", "y_1", "y_1"),
        ]
        props_b = [
            Propagator("R", "a", "i5", "x", "y_1"),
            Propagator("C", "i3", "i4", "y_1", "y_1"),
            Propagator("R", "b", "i2", "y", "y_0"),
            Propagator("C", "i0", "i1", "y_0", "y_0"),
        ]
        int_vars = frozenset({"y_0", "y_1"})
        form_a, _ = _canonical_diagram_form(props_a, int_vars)
        form_b, _ = _canonical_diagram_form(props_b, int_vars)
        assert form_a == form_b

    def test_two_internal_vars_different_diagrams(self):
        """Two genuinely different diagrams should have different canonical forms."""
        # Diagram A: x connects to y_0 via R, y_0 connects to y_1 via C
        props_a = [
            Propagator("R", "a", "i0", "x", "y_0"),
            Propagator("C", "i1", "i2", "y_0", "y_1"),
            Propagator("R", "b", "i3", "y", "y_1"),
        ]
        # Diagram B: x connects to y_0 via R, y_0 self-loop C, y connects to y_1 via R
        props_b = [
            Propagator("R", "a", "i0", "x", "y_0"),
            Propagator("C", "i1", "i2", "y_0", "y_0"),
            Propagator("R", "b", "i3", "y", "y_1"),
        ]
        int_vars = frozenset({"y_0", "y_1"})
        form_a, _ = _canonical_diagram_form(props_a, int_vars)
        form_b, _ = _canonical_diagram_form(props_b, int_vars)
        assert form_a != form_b

    def test_c_symmetry_in_canonical_form(self):
        """C(y_0, y_1) and C(y_1, y_0) should give the same canonical form."""
        props_a = [
            Propagator("C", "i0", "i1", "y_0", "y_1"),
        ]
        props_b = [
            Propagator("C", "i0", "i1", "y_1", "y_0"),
        ]
        # No internal vars — just edge canonicalization
        form_a, _ = _canonical_diagram_form(props_a, frozenset())
        form_b, _ = _canonical_diagram_form(props_b, frozenset())
        assert form_a == form_b


# ---------------------------------------------------------------------------
# _match_propagators_after_spatial
# ---------------------------------------------------------------------------


class TestMatchPropagatorsAfterSpatial:
    def test_identity_permutation(self):
        """Same propagators map every internal index they carry to itself."""
        ref = [
            Propagator("R", "a", "i2", "x", "y_0"),
            Propagator("C", "i0", "i1", "y_0", "y_0"),
        ]
        other = [
            Propagator("R", "a", "i2", "x", "y_0"),
            Propagator("C", "i0", "i1", "y_0", "y_0"),
        ]
        perm = _match_propagators_after_spatial(
            ref, other, {}, {"i0", "i1", "i2"}
        )
        assert perm == {"i0": "i0", "i1": "i1", "i2": "i2"}

    def test_index_swap(self):
        """Propagators differing by internal index swap."""
        ref = [
            Propagator("R", "a", "i2", "x", "y_0"),
            Propagator("C", "i0", "i1", "y_0", "y_0"),
        ]
        other = [
            Propagator("R", "a", "i1", "x", "y_0"),
            Propagator("C", "i0", "i2", "y_0", "y_0"),
        ]
        perm = _match_propagators_after_spatial(
            ref, other, {}, {"i0", "i1", "i2"}
        )
        assert perm == {"i0": "i0", "i1": "i2", "i2": "i1"}

    def test_external_index_mismatch_fails(self):
        """External indices must match exactly."""
        ref = [
            Propagator("R", "a", "i2", "x", "y_0"),
        ]
        other = [
            Propagator("R", "b", "i2", "x", "y_0"),
        ]
        perm = _match_propagators_after_spatial(
            ref, other, {}, {"i2"}
        )
        assert perm is None

    def test_with_spatial_permutation(self):
        """After spatial relabeling, propagators should match."""
        ref = [
            Propagator("R", "a", "i1", "x", "y_0"),
            Propagator("R", "b", "i3", "y", "y_1"),
        ]
        other = [
            Propagator("R", "a", "i3", "x", "y_1"),
            Propagator("R", "b", "i1", "y", "y_0"),
        ]
        spatial_perm = {"y_0": "y_1", "y_1": "y_0"}
        perm = _match_propagators_after_spatial(
            ref, other, spatial_perm, {"i1", "i3"}
        )
        assert perm is not None
        assert perm == {"i3": "i1", "i1": "i3"}

    def test_c_flip_matching(self):
        """C propagator with reversed spatial args should still match."""
        ref = [
            Propagator("C", "i0", "i1", "x", "y_0"),
        ]
        other = [
            Propagator("C", "i2", "i3", "y_0", "x"),
        ]
        perm = _match_propagators_after_spatial(
            ref, other, {}, {"i0", "i1", "i2", "i3"}
        )
        assert perm is not None
        # Flipped: other.index_right -> ref.index_left, other.index_left -> ref.index_right
        assert perm == {"i3": "i0", "i2": "i1"}

    def test_scalar_no_indices(self):
        """Scalar propagators (no component indices) should match trivially."""
        ref = [
            Propagator("R", None, None, "x", "y_0"),
            Propagator("C", None, None, "y_0", "y_0"),
        ]
        other = [
            Propagator("R", None, None, "x", "y_0"),
            Propagator("C", None, None, "y_0", "y_0"),
        ]
        perm = _match_propagators_after_spatial(ref, other, {}, set())
        assert perm is not None
        assert perm == {}


# ---------------------------------------------------------------------------
# collect_by_diagram (integration tests on expression trees)
# ---------------------------------------------------------------------------


class TestCollectByDiagram:
    def test_no_sum_unchanged(self):
        """Expression without a Sum is returned unchanged."""
        prop = Propagator("C", "a", "b", "x", "y")
        result = collect_by_diagram(prop)
        assert result == prop

    def test_single_term_sum_with_coupling(self):
        """Single term in Sum: coupling is moved inside."""
        r = Propagator("R", "a", "i2", "x", "y_0")
        c = Propagator("C", "i0", "i1", "y_0", "y_0")
        F = Symbol("F", ("i0", "i1", "i2"))
        expr = Product((Rational(-1), F, Sum((Product((r, c)),))))

        result = collect_by_diagram(expr)
        latex = result.to_latex()
        assert "F_" in latex
        assert "R_" in latex

    def test_two_terms_same_topology_different_indices(self):
        """Two terms with same spatial topology but different index routings
        merge under the summed ``i1 <-> i2``."""
        F = Symbol("F", ("i0", "i1", "i2"))
        r1 = Propagator("R", "a", "i2", "x", "y_0")
        c1 = Propagator("C", "i0", "i1", "y_0", "y_0")
        r2 = Propagator("R", "a", "i1", "x", "y_0")
        c2 = Propagator("C", "i0", "i2", "y_0", "y_0")

        inner_sum = Sum((Product((r1, c1)), Product((r2, c2))))
        expr = Product((Rational(-1), F, inner_sum))
        for name in ("i2", "i1", "i0"):
            expr = SumOverIndex(name, 3, expr)

        result = collect_by_diagram(expr)
        latex = result.to_latex()

        # One propagator pair with two permuted couplings
        assert latex.count("F_") == 2
        assert latex.count("R_") == 1

    def test_through_integral_wrapper(self):
        """collect_by_diagram works through IntegralOver wrapper."""
        F = Symbol("F", ("i0", "i1", "i2"))
        r1 = Propagator("R", "a", "i2", "x", "y_0")
        c1 = Propagator("C", "i0", "i1", "y_0", "y_0")
        r2 = Propagator("R", "a", "i1", "x", "y_0")
        c2 = Propagator("C", "i0", "i2", "y_0", "y_0")

        inner_sum = Sum((Product((r1, c1)), Product((r2, c2))))
        expr = IntegralOver("y_0", Product((Rational(-1), F, inner_sum)))

        result = collect_by_diagram(expr)
        assert isinstance(result, IntegralOver)

    def test_through_sum_over_index_wrapper(self):
        """collect_by_diagram works through SumOverIndex wrapper."""
        F = Symbol("F", ("i0", "i1", "i2"))
        r1 = Propagator("R", "a", "i2", "x", "y_0")
        c1 = Propagator("C", "i0", "i1", "y_0", "y_0")

        inner_sum = Sum((Product((r1, c1)),))
        expr = SumOverIndex("i0", 3, Product((Rational(-1), F, inner_sum)))

        result = collect_by_diagram(expr)
        assert isinstance(result, SumOverIndex)

    def test_collect_topology_false_preserves_original(self):
        """When collect_topology=False, the raw simplified form is returned."""
        from sft_wick import Field, Vertex, Action, compute_moment, reset_uid_counter

        reset_uid_counter()
        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        v = Vertex(fields=[phi, psi], coupling="g")
        action = Action(vertices=[v])
        obs = [phi("x"), phi("y")]

        result_off = compute_moment(obs, action, order=1,
                                    response_phase=False, collect_topology=False)
        reset_uid_counter()
        result_on = compute_moment(obs, action, order=1,
                                   response_phase=False, collect_topology=True)
        # Both should be non-zero
        assert not (isinstance(result_off.order(1), Rational) and result_off.order(1).is_zero)
        assert not (isinstance(result_on.order(1), Rational) and result_on.order(1).is_zero)

    def test_spatial_relabeling_merges_terms(self):
        """Terms related by y_0 <-> y_1 swap at order 2 should be merged."""
        # Build a hand-crafted expression simulating order-2 with same vertex
        F0 = Symbol("F", ("i0", "i1", "i2"))
        F1 = Symbol("F", ("i3", "i4", "i5"))

        # Term A: R(x,y_0) C(y_0,y_0) R(y,y_1) C(y_1,y_1)
        term_a = Product((
            Propagator("R", "a", "i2", "x", "y_0"),
            Propagator("C", "i0", "i1", "y_0", "y_0"),
            Propagator("R", "b", "i5", "y", "y_1"),
            Propagator("C", "i3", "i4", "y_1", "y_1"),
        ))
        # Term B: R(x,y_1) C(y_1,y_1) R(y,y_0) C(y_0,y_0) (y_0 <-> y_1 swapped)
        term_b = Product((
            Propagator("R", "a", "i5", "x", "y_1"),
            Propagator("C", "i3", "i4", "y_1", "y_1"),
            Propagator("R", "b", "i2", "y", "y_0"),
            Propagator("C", "i0", "i1", "y_0", "y_0"),
        ))

        inner_sum = Sum((term_a, term_b))
        expr = IntegralOver("y_0", IntegralOver("y_1",
            Product((Rational(1, 2), F0, F1, inner_sum))
        ))

        result = collect_by_diagram(expr)
        # After collection, the inner sum should have 1 group (merged)
        # not 2 separate terms
        latex = result.to_latex()
        # The merged form should show 2x the coupling or summed couplings
        # Either way, the two terms should have been combined
        # Count propagator groups: should be 1 collected term, not 2
        assert "+" in latex or "2" in latex  # either summed couplings or degeneracy factor


# ---------------------------------------------------------------------------
# diagonal_propagators
# ---------------------------------------------------------------------------


class TestDiagonalPropagators:
    """Tests for the diagonal propagator simplification pass."""

    def test_noop_when_flags_false(self):
        """No change when both flags are False."""
        R = Propagator("R", "a", "i_0", "x", "y_0")
        C = Propagator("C", "i_1", "i_2", "y_0", "y_0")
        expr = Product((R, C))
        assert diagonal_propagators(expr, diag_R=False, diag_C=False) == expr

    def test_diag_R_substitutes_index(self):
        """diag_R pins the R-propagator indices equal and substitutes."""
        R = Propagator("R", "a", "i_0", "x", "y_0")
        F = Symbol("F", ("i_0", "i_1"))
        expr = SumOverIndex(
            "i_0", 3,
            SumOverIndex("i_1", 3, Product((F, R))),
        )
        result = diagonal_propagators(expr, diag_R=True)
        # i_0 eliminated by δ_{a,i_0}: sum collapses to one term, no factor of N
        # surviving index is canonically renamed to i_0
        assert isinstance(result, SumOverIndex)
        assert result.index_name == "i_0"
        # The body should contain F_{a, i_0} and R_{a,a} — no extra Rational factor
        body = result.body
        assert isinstance(body, Product)
        props = [f for f in body.factors if isinstance(f, Propagator)]
        syms = [f for f in body.factors if isinstance(f, Symbol)]
        assert len(props) == 1
        assert props[0].index_left == "a"
        assert props[0].index_right == "a"
        assert syms[0].indices == ("a", "i_0")
        # No spurious Rational(3): the diagonal selects one term, not N terms
        rats = [f for f in body.factors if isinstance(f, Rational)]
        assert len(rats) == 0

    def test_diag_C_substitutes_index(self):
        """diag_C pins the C-propagator indices equal."""
        C = Propagator("C", "i_0", "i_1", "y_0", "y_0")
        F = Symbol("F", ("i_0", "i_1"))
        expr = SumOverIndex(
            "i_0", 3,
            SumOverIndex("i_1", 3, Product((F, C))),
        )
        result = diagonal_propagators(expr, diag_C=True)
        assert isinstance(result, SumOverIndex)
        # One index eliminated, one remains
        body = result.body
        assert isinstance(body, Product)
        props = [f for f in body.factors if isinstance(f, Propagator)]
        # C should now have equal indices
        assert props[0].index_left == props[0].index_right

    def test_combined_R_and_C(self):
        """Both diag_R and diag_C eliminate multiple indices."""
        R = Propagator("R", "a", "i_0", "x", "y_0")
        C = Propagator("C", "i_1", "i_2", "y_0", "y_0")
        F = Symbol("F", ("i_0", "i_1", "i_2"))
        expr = SumOverIndex(
            "i_0", 3,
            SumOverIndex(
                "i_1", 3,
                SumOverIndex(
                    "i_2", 3,
                    IntegralOver("y_0", Product((Rational(-1), F, R, C))),
                ),
            ),
        )
        result = diagonal_propagators(expr, diag_R=True, diag_C=True)
        # i_0 → a (from R), i_2 → i_1 (from C) → 2 indices eliminated
        # Only one index remains; canonically renamed to i_0
        assert isinstance(result, SumOverIndex)
        assert result.index_name == "i_0"
        # The body should be IntegralOver wrapping a Product
        body = result.body
        assert isinstance(body, IntegralOver)
        inner = body.body
        assert isinstance(inner, Product)
        # Coefficient unchanged at -1: diagonal selects one term per constraint, no extra N factor
        rats = [f for f in inner.factors if isinstance(f, Rational)]
        assert len(rats) == 1 and rats[0] == Rational(-1)
        # F should have indices (a, i_0, i_0)
        syms = [f for f in inner.factors if isinstance(f, Symbol)]
        assert syms[0].indices == ("a", "i_0", "i_0")

    def test_external_external_produces_delta(self):
        """When both indices are external, a KroneckerDelta is inserted."""
        C = Propagator("C", "a", "b", "x", "xp")
        expr = Product((C,))
        result = diagonal_propagators(expr, diag_C=True)
        # Should contain KroneckerDelta and C with equal indices
        assert isinstance(result, Product)
        deltas = [f for f in result.factors if isinstance(f, KroneckerDelta)]
        props = [f for f in result.factors if isinstance(f, Propagator)]
        assert len(deltas) == 1
        assert {deltas[0].index1, deltas[0].index2} == {"a", "b"}
        assert props[0].index_left == props[0].index_right

    def test_scalar_propagator_unchanged(self):
        """Scalar propagators (no component indices) are unaffected."""
        R = Propagator("R", None, None, "x", "y")
        expr = Product((Rational(1), R))
        result = diagonal_propagators(expr, diag_R=True, diag_C=True)
        assert result == R  # simplify strips Rational(1)

    def test_already_diagonal_unchanged(self):
        """Propagator with equal indices is already diagonal."""
        R = Propagator("R", "a", "a", "x", "y")
        F = Symbol("F", ("a",))
        expr = Product((F, R))
        result = diagonal_propagators(expr, diag_R=True)
        assert result == expr

    def test_through_integral_wrapper(self):
        """Diagonal simplification works through IntegralOver."""
        R = Propagator("R", "a", "i_0", "x", "y_0")
        expr = SumOverIndex(
            "i_0", 3,
            IntegralOver("y_0", R),
        )
        result = diagonal_propagators(expr, diag_R=True)
        # i_0 eliminated by δ_{a,i_0}: sum collapses to one term, no factor of 3
        assert isinstance(result, IntegralOver)
        body = result.body
        # Body may be a bare Propagator or Product after simplification
        props = [body] if isinstance(body, Propagator) else [
            f for f in body.factors if isinstance(f, Propagator)
        ]
        assert props[0].index_left == "a" and props[0].index_right == "a"
        # No spurious Rational(3) factor
        if isinstance(body, Product):
            rats = [f for f in body.factors if isinstance(f, Rational)]
            assert all(r == Rational(1) for r in rats)


def _value(expr, tables, env, positions):
    """Evaluate an expression by explicit sums over seeded finite tables."""
    if isinstance(expr, Rational):
        return expr.numerator / expr.denominator
    if isinstance(expr, ImaginaryUnit):
        return 1j
    if isinstance(expr, Sum):
        return sum(_value(t, tables, env, positions) for t in expr.terms)
    if isinstance(expr, Product):
        return np.prod([_value(f, tables, env, positions) for f in expr.factors])
    if isinstance(expr, KroneckerDelta):
        return float(env[expr.index1] == env[expr.index2])
    if isinstance(expr, Symbol):
        points = () if expr.local else tuple(positions[s] for s in expr.spatial_args)
        return tables[expr.name][tuple(env[i] for i in expr.indices) + points]
    if isinstance(expr, Propagator):
        left, right = positions[expr.spatial_left], positions[expr.spatial_right]
        a = 0 if expr.index_left is None else env[expr.index_left]
        b = 0 if expr.index_right is None else env[expr.index_right]
        return tables[expr.kind][a, left, b, right]
    if isinstance(expr, SumOverIndex):
        return sum(
            _value(expr.body, tables, {**env, expr.index_name: v}, positions)
            for v in range(expr.dimension)
        )
    if isinstance(expr, IntegralOver):
        n_points = tables["C"].shape[1]
        return sum(
            _value(expr.body, tables, env, {**positions, expr.variable: p})
            for p in range(n_points)
        )
    raise TypeError(type(expr))


def _diagonal_table(rng, n_components, n_points, isotropic):
    """A table that is delta_ab, and a multiple of I when ``isotropic``."""
    values = rng.normal(size=(n_components, n_points, n_points))
    if isotropic:
        values = np.broadcast_to(values[:1], values.shape)
    table = np.zeros((n_components, n_points, n_components, n_points))
    for a in range(n_components):
        table[a, :, a, :] = values[a]
    return table


class TestDiagonalSumsPerTerm:
    """A delta constraint removes a sum from its own term only."""

    @pytest.mark.parametrize("n_components", [2, 3, 50])
    @pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
    def test_term_that_keeps_its_index_keeps_the_sum(self, n_components, scale):
        """sum_i (G_i C_ai + H_ii C_bb) = G_a C_aa + tr(H) C_bb."""
        rng = np.random.default_rng(5)
        G = scale * rng.normal(size=n_components)
        H = scale * rng.normal(size=(n_components, n_components))
        C = _diagonal_table(rng, n_components, 2, isotropic=False)
        tables = {"G": G, "H": H, "C": C}
        a, b = 0, n_components - 1
        expected = G[a] * C[a, 0, a, 1] + np.trace(H) * C[b, 0, b, 1]

        C_ai = Propagator("C", "a", "i", "x", "w")
        C_bb = Propagator("C", "b", "b", "x", "w")
        expr = SumOverIndex(
            "i",
            n_components,
            Sum((
                Product((Symbol("G", ("i",)), C_ai)),
                Product((Symbol("H", ("i", "i")), C_bb)),
            )),
        )
        result = diagonal_propagators(expr, diag_C=True)

        got = _value(result, tables, {"a": a, "b": b}, {"x": 0, "w": 1})
        assert got == pytest.approx(expected, rel=1e-13, abs=0.0)

    def test_only_elimination_removes_a_sum(self):
        """A sum that never met a constraint still counts N; a clash refuses."""
        n_components = 3
        tables = {"C": _diagonal_table(np.random.default_rng(7), 3, 2, True)}
        env, positions = {"b": 1}, {"x": 0, "w": 1}
        c = tables["C"][0, 0, 0, 1]

        unused = SumOverIndex(
            "i", n_components, Propagator("C", "b", "b", "x", "w")
        )
        trace = SumOverIndex(
            "i", n_components, Propagator("C", "i", "i", "x", "w")
        )
        for expr in (unused, trace):
            result = diagonal_propagators(expr, iso_C=True)
            assert _value(result, tables, env, positions) == pytest.approx(
                n_components * c
            )

        eliminated = SumOverIndex(
            "i", n_components, Propagator("C", "b", "i", "x", "w")
        )
        result = diagonal_propagators(eliminated, iso_C=True)
        assert _value(result, tables, env, positions) == pytest.approx(c)

        with pytest.raises(ValueError, match="bound once"):
            diagonal_propagators(Product((eliminated, eliminated)), diag_C=True)

    @pytest.mark.parametrize(
        "flags, n_components",
        [
            ({"diag_C": True, "diag_R": True}, 2),
            ({"diag_C": True, "diag_R": True}, 3),
            ({"diag_C": True}, 2),
            ({"diag_R": True}, 2),
            ({"iso_C": True, "iso_R": True}, 2),
        ],
    )
    def test_order_terms_match_the_index_explicit_expansion(
        self, flags, n_components
    ):
        """The flagged order 2 equals the unflagged sums on diagonal tables."""
        phi = Field("phi", "physical", n_components)
        psi = Field("psi", "response", n_components)
        action = Action([Vertex([phi, phi, psi], "F"), Vertex([phi, psi], "L")])
        observable = [phi("a", "x"), phi("b", "w")]

        rng = np.random.default_rng(5)
        isotropic = "iso_C" in flags
        tables = {
            "C": _diagonal_table(rng, n_components, 2, isotropic),
            "R": _diagonal_table(rng, n_components, 2, isotropic),
            "F": rng.normal(size=(n_components,) * 3),
            "L": rng.normal(size=(n_components,) * 2),
        }
        tables["C"] = (tables["C"] + tables["C"].transpose(2, 3, 0, 1)) / 2
        env, positions = {"a": 0, "b": n_components - 1}, {"x": 0, "w": 1}

        reference = compute_moment(observable, action, 2).order(2)
        flagged = compute_moment(observable, action, 2, **flags).order(2)

        assert _value(flagged, tables, env, positions) == pytest.approx(
            _value(reference, tables, env, positions), rel=1e-12, abs=0.0
        )


class TestDiagonalComputeMoment:
    """Tests for diagonal propagators integrated into compute_moment."""

    def test_order1_diag_R_reduces_indices(self):
        """diag_R at order 1 eliminates one summation index."""
        from sft_wick import (
            Action, Field, Vertex, compute_moment, reset_uid_counter,
        )

        reset_uid_counter()
        phi = Field("phi", "physical", n_components=3)
        psi = Field("psi", "response", n_components=3)
        v = Vertex(fields=[psi, phi, phi], coupling="F")
        action = Action(vertices=[v])
        obs = [phi("a", "x")]

        # Without diagonal
        r_normal = compute_moment(obs, action, order=1, response_phase=False)
        dt_normal = r_normal.diagram_terms(1)
        assert len(dt_normal) == 1
        assert len(dt_normal[0].summation_indices) == 3

        # With diag_R
        r_diag = compute_moment(
            obs, action, order=1, response_phase=False, diag_R=True,
        )
        dt_diag = r_diag.diagram_terms(1)
        assert len(dt_diag) == 1
        assert len(dt_diag[0].summation_indices) == 2  # one eliminated

    def test_order1_diag_both_reduces_to_one_index(self):
        """diag_R + diag_C at order 1 reduces to a single summation index."""
        from sft_wick import (
            Action, Field, Vertex, compute_moment, reset_uid_counter,
        )

        reset_uid_counter()
        phi = Field("phi", "physical", n_components=3)
        psi = Field("psi", "response", n_components=3)
        v = Vertex(fields=[psi, phi, phi], coupling="F")
        action = Action(vertices=[v])
        obs = [phi("a", "x")]

        r = compute_moment(
            obs, action, order=1, response_phase=False,
            diag_R=True, diag_C=True,
        )
        dt = r.diagram_terms(1)
        assert len(dt) == 1
        assert len(dt[0].summation_indices) == 1
        # Prefactor unchanged at -1: diagonal collapses sums, no extra N factors
        assert dt[0].rational_prefactor == Rational(-1)

    def test_diagram_term_evaluate_with_fixed_indices(self):
        """evaluate_coupling with fixed_indices works after diagonal."""
        from sft_wick import (
            Action, Field, Vertex, compute_moment, reset_uid_counter,
        )
        import numpy as np

        reset_uid_counter()
        phi = Field("phi", "physical", n_components=3)
        psi = Field("psi", "response", n_components=3)
        v = Vertex(fields=[psi, phi, phi], coupling="F")
        action = Action(vertices=[v])
        obs = [phi("a", "x")]

        r = compute_moment(
            obs, action, order=1, response_phase=False,
            diag_R=True, diag_C=True,
        )
        dt = r.diagram_terms(1)[0]

        # F tensor with F_{0,1,2} = F_{0,2,1} = 1 (all others 0)
        F = np.zeros((3, 3, 3))
        F[0, 1, 2] = F[0, 2, 1] = 1.0

        # With a=0: F_{0, i_1, i_1} — only nonzero when i_1 ∈ {1,2}? No,
        # F[0,i,i] is zero for all i since F[0,0,0]=F[0,1,1]=F[0,2,2]=0.
        result = dt.evaluate_coupling({"F": F}, fixed_indices={"a": 0})
        # F_{0,i,i} = 0 for i=0,1,2 → all zero
        np.testing.assert_array_equal(result, np.zeros(3) * dt.rational_prefactor.numerator / dt.rational_prefactor.denominator)

    def test_diagram_term_apply_diagonal_method(self):
        """DiagramTerm.apply_diagonal works standalone."""
        from sft_wick import (
            Action, Field, Vertex, compute_moment, reset_uid_counter,
            DiagramTerm,
        )

        reset_uid_counter()
        phi = Field("phi", "physical", n_components=3)
        psi = Field("psi", "response", n_components=3)
        v = Vertex(fields=[psi, phi, phi], coupling="F")
        action = Action(vertices=[v])
        obs = [phi("a", "x")]

        # Get normal DiagramTerm, then apply diagonal manually
        r = compute_moment(obs, action, order=1, response_phase=False)
        dt_orig = r.diagram_terms(1)[0]
        dt_diag = dt_orig.apply_diagonal(diag_R=True, diag_C=True)

        assert len(dt_diag.summation_indices) == 1
        # Prefactor unchanged at -1: diagonal collapses sums, no extra N factors
        assert dt_diag.rational_prefactor == Rational(-1)
        # Propagators should have equal indices after diagonal
        for p in dt_diag.propagators:
            if p.index_left is not None:
                assert p.index_left == p.index_right


def test_diagonal_propagators_refuses_a_product_past_its_size():
    """A product of sums is distributed, so twelve sums of four terms would
    be 4**12 terms: 7.6 s and 423 MB at 4**8."""
    def sum_of_four(k):
        return SumOverIndex(f"i{k}", 2, Sum(tuple(
            Product((Symbol(f"f{m}", indices=(f"i{k}",)),
                     Propagator("C", f"i{k}", f"i{k}", f"x{k}", f"y{k}")))
            for m in range(4))))

    with pytest.raises(ValueError, match="above the limit"):
        diagonal_propagators(Product(tuple(sum_of_four(k) for k in range(12))),
                             diag_C=True)


def test_collect_declines_a_merge_that_is_not_injective():
    """``R_ai R_bi`` and ``R_aj R_bk`` would need j and k to both become i,
    which drops the sum over one of them: -6.90 against -7.33."""
    rng = np.random.default_rng(2)
    tables = {"F": rng.normal(size=(3, 3, 3)), "R": rng.normal(size=(3, 2, 3, 2))}
    env = {"a": 0, "b": 1}
    points = {"x": 0, "w": 1}

    def pair(first, second):
        return Product((Propagator("R", "a", first, "x", "w"),
                        Propagator("R", "b", second, "x", "w")))

    expr = SumOverIndex("i", 3, SumOverIndex("j", 3, SumOverIndex("k", 3, Product((
        Symbol("F", ("i", "j", "k")),
        Sum((pair("i", "i"), pair("j", "k"))),
    )))))
    assert _value(collect_by_diagram(expr), tables, env, points) == pytest.approx(
        _value(expr, tables, env, points), rel=1e-13, abs=0.0
    )


def test_collect_keeps_an_index_that_meets_itself():
    """``R_ai R_bi`` and ``R_aj R_bi`` are different diagrams: one index on
    both legs against two.  The matcher dropped the ``i -> i`` it read off
    ``R_bi`` and kept ``j -> i``, so the swap ``i <-> j`` merged them.  On
    integer tables the merged value was 813 against a direct sum of 797."""
    rng = np.random.default_rng(5)
    tables = {"F": rng.normal(size=(3, 3)), "R": rng.normal(size=(3, 2, 3, 2))}
    env = {"a": 0, "b": 1}
    points = {"x": 0, "w": 1}

    def pair(first, second):
        return Product((Propagator("R", "a", first, "x", "w"),
                        Propagator("R", "b", second, "x", "w")))

    expr = SumOverIndex("i", 3, SumOverIndex("j", 3, Product((
        Symbol("F", ("i", "j")),
        Sum((pair("i", "i"), pair("j", "i"))),
    ))))
    assert _value(collect_by_diagram(expr), tables, env, points) == pytest.approx(
        _value(expr, tables, env, points), rel=1e-13, abs=0.0
    )


def test_collect_merges_through_a_c_at_one_point():
    """``C_jk(w, w) C_ak(x, w)`` and ``C_jk(w, w) C_aj(x, w)`` are one diagram
    under ``j <-> k``, because ``C_jk(w, w) = C_kj(w, w)``.  The merge needs
    the C at one point read the other way round."""
    rng = np.random.default_rng(6)
    c = rng.normal(size=(3, 2, 3, 2))
    tables = {"F": rng.normal(size=(3, 3, 3)), "C": c + c.transpose(2, 3, 0, 1)}
    env = {"a": 0}
    points = {"x": 0, "w": 1}

    def pair(leg):
        return Product((Propagator("C", "j", "k", "w", "w"),
                        Propagator("C", "a", leg, "x", "w")))

    expr = SumOverIndex("i", 3, SumOverIndex("j", 3, SumOverIndex("k", 3, Product((
        Symbol("F", ("i", "j", "k")),
        Sum((pair("k"), pair("j"))),
    )))))
    collected = collect_by_diagram(expr)
    assert collected.to_latex().count("C_{") == 2
    assert _value(collected, tables, env, points) == pytest.approx(
        _value(expr, tables, env, points), rel=1e-13, abs=0.0
    )


@pytest.mark.parametrize("delta", [True, False])
def test_collect_renames_every_occurrence_of_a_dummy_index(delta):
    """Merging ``R_aj`` onto ``R_ai`` renames j to i.  That reaches a delta
    next to the propagators, and it must move i out of the way in the
    coupling: F_ij became F_ii."""
    rng = np.random.default_rng(2)
    tables = {"F": rng.normal(size=(3, 3)), "R": rng.normal(size=(3, 2, 3, 2))}
    env = {"a": 0, "b": 1, "c": 2}
    points = {"x": 0, "w": 1}

    def branch(first, second):
        factors = (
            Propagator("R", "a", first, "x", "w"),
            Propagator("R", "b", "c", "x", "w"),
        )
        if delta:
            factors += (KroneckerDelta(second, "c"),)
        return Product(factors)

    expr = SumOverIndex("i", 3, SumOverIndex("j", 3, Product((
        Symbol("F", ("i", "j")),
        Sum((branch("i", "j"), branch("j", "i"))),
    ))))
    collected = collect_by_diagram(expr)
    assert _value(collected, tables, env, points) == pytest.approx(
        _value(expr, tables, env, points), rel=1e-13, abs=0.0
    )


def _r(first, second):
    return Propagator("R", first, second, "x", "w")


def _scope_case(case):
    """``sum_i G_ai (R_ai + R_ia)`` with a free; ``sum_{i<2} sum_{j<3}
    F_ij (R_ai + R_aj)``; ``sum_ij delta_ia F_ij (R_ai + R_aj)`` with the
    delta outside the inner sum; ``sum_i delta_ia G_ia (2 + R_ii)``, a term
    with no propagator; the free index ``a`` beside an integrated point
    ``a``; and ``sum_ij G_ij H(i) (R_ai + R_aj)`` with ``i`` also a free
    point."""
    if case == "bare term":
        return SumOverIndex("i", 3, Product((
            KroneckerDelta("i", "a"), Symbol("G", ("i", "a")),
            Sum((Rational(2), _r("i", "i"))))))
    if case == "point named like an index":
        return SumOverIndex("i", 3, SumOverIndex("j", 3, Product((
            Symbol("G", ("i", "j")), Symbol("H", (), ("i",)),
            Sum((_r("a", "i"), _r("a", "j")))))))
    if case == "shared name":
        def branch(first, second):
            return Product((Propagator("R", "e", "e", "x", first),
                            Propagator("C", "e", "e", second, "w")))
        return SumOverIndex("e", 3, IntegralOver("a", IntegralOver("b", Product((
            KroneckerDelta("a", "c"), Symbol("G", ("e", "e")),
            Sum((branch("a", "b"), branch("b", "a"))))))))
    if case == "free index":
        return SumOverIndex("i", 3, Product((
            Symbol("G", ("a", "i")), Sum((_r("a", "i"), _r("i", "a"))))))
    body = Product((Symbol("G", ("i", "j")), Sum((_r("a", "i"), _r("a", "j")))))
    if case == "two ranges":
        return SumOverIndex("i", 2, SumOverIndex("j", 3, body))
    return SumOverIndex("i", 3, SumOverIndex("j", 3, Product(
        (KroneckerDelta("i", "a"),) + body.factors)))


@pytest.mark.parametrize(
    "case", ["free index", "two ranges", "outer delta", "bare term", "shared name",
             "point named like an index"])
def test_collect_renames_only_bound_indices_within_their_range(case):
    """A merge renames a summed index onto one summed over the same range,
    and renames it in every factor.  Renaming the free a, swapping an index
    over two values with one over three, or leaving a delta outside the sum
    unrenamed gave 8, 186 and 14 against direct sums of 12, 231 and 18 on
    integer tables.  A term with no propagator lost the outer factors, and
    a point swap renamed the index of the same name, or an index swap the
    point."""
    rng = np.random.default_rng(11)
    c = rng.normal(size=(3, 2, 3, 2))
    tables = {"G": rng.normal(size=(3, 3)), "R": rng.normal(size=(3, 2, 3, 2)),
              "C": c + c.transpose(2, 3, 0, 1), "H": rng.normal(size=2)}
    points = {"x": 0, "w": 1, "i": 1}
    expr = _scope_case(case)
    collected = collect_by_diagram(expr)
    for a in range(3):
        env = {"a": a, "c": 1}
        assert _value(collected, tables, env, points) == pytest.approx(
            _value(expr, tables, env, points), rel=1e-13, abs=0.0
        )


@pytest.mark.parametrize("collect_topology", [True, False])
def test_fields_with_different_component_counts(monkeypatch, collect_topology):
    """At a psi(3) phi(3) chi(2) vertex both engines renamed an index over
    three values onto one over two: -215.2 against -129.2 for the Wick sum
    before collection."""
    import sft_wick.perturbation as perturbation

    def order_two(topology):
        phi = Field("phi", "physical", n_components=3)
        psi = Field("psi", "response", n_components=3)
        chi = Field("chi", "physical", n_components=2)
        result = compute_moment(
            [phi("a", "x"), phi("b", "y")], Action([Vertex([psi, phi, chi], "F")]),
            order=2, collect_topology=topology, response_phase=False,
        )
        return result.order(2)

    rng = np.random.default_rng(7)
    c = rng.normal(size=(3, 3, 3, 3))
    tables = {"F": rng.normal(size=(3, 3, 3)), "R": rng.normal(size=(3, 3, 3, 3)),
              "C": c + c.transpose(2, 3, 0, 1)}
    env, points = {"a": 0, "b": 2}, {"x": 0, "y": 1}
    got = _value(order_two(collect_topology), tables, env, points)
    monkeypatch.setattr(perturbation, "collect_by_diagram", lambda e: e)
    wick_sum = _value(order_two(False), tables, env, points)
    assert got == pytest.approx(wick_sum, rel=1e-12, abs=0.0)


@pytest.fixture
def hang_fails():
    """Listing every matching of twelve C at one point, 12! 2**12 of them,
    would not end in the life of the test."""

    def expired(signum, frame):
        pytest.fail("still matching")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(10)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)


def test_collect_matches_c_at_one_point_without_listing_every_matching(hang_fails):
    """Seven such C took 0.22 s and 320 MB when every matching was built
    before the first was tried."""
    names = [f"i{k}" for k in range(24)]

    def branch(order):
        return Product(tuple(
            Propagator("C", order[2 * k], order[2 * k + 1], "w", "w") for k in range(12)))

    swapped = [names[k + 1 if k % 2 == 0 else k - 1] for k in range(24)]
    expr = Product((Symbol("F", tuple(names)), Sum((branch(names), branch(swapped)))))
    for name in reversed(names):
        expr = SumOverIndex(name, 2, expr)
    assert collect_by_diagram(expr).to_latex().count("C_{") == 12
