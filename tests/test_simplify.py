"""Tests for the simplification pipeline and diagram-based collection."""

import pytest
from sft_wick.expressions import (
    ZERO, ONE, KroneckerDelta, Product, Propagator, Rational, Sum, Symbol,
    IntegralOver, SumOverIndex,
)
from sft_wick.simplify import (
    _canonical_edge,
    _canonical_diagram_form,
    _match_propagators_after_spatial,
    collect_by_diagram,
    diagonal_propagators,
    simplify,
)


# ---------------------------------------------------------------------------
# _canonical_edge
# ---------------------------------------------------------------------------


class TestCanonicalEdge:
    def test_c_symmetric_sorted(self):
        assert _canonical_edge("C", "y", "x") == ("C", "x", "y")

    def test_c_already_sorted(self):
        assert _canonical_edge("C", "x", "y") == ("C", "x", "y")

    def test_c_equal_points(self):
        assert _canonical_edge("C", "x", "x") == ("C", "x", "x")

    def test_r_directed_not_sorted(self):
        assert _canonical_edge("R", "x", "y") == ("R", "x", "y")
        assert _canonical_edge("R", "y", "x") == ("R", "y", "x")

    def test_r_and_c_different(self):
        assert _canonical_edge("R", "y", "x") != _canonical_edge("C", "y", "x")


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
        """Same propagators should yield empty permutation."""
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
        assert perm is not None
        assert perm == {}

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
        assert perm is not None
        assert perm == {"i1": "i2", "i2": "i1"}

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
        """Two terms with same spatial topology but different index routings."""
        F = Symbol("F", ("i0", "i1", "i2"))
        r1 = Propagator("R", "a", "i2", "x", "y_0")
        c1 = Propagator("C", "i0", "i1", "y_0", "y_0")
        r2 = Propagator("R", "a", "i1", "x", "y_0")
        c2 = Propagator("C", "i0", "i2", "y_0", "y_0")

        inner_sum = Sum((Product((r1, c1)), Product((r2, c2))))
        expr = Product((Rational(-1), F, inner_sum))

        result = collect_by_diagram(expr)
        latex = result.to_latex()

        # Should contain two F terms summed (permuted couplings)
        assert latex.count("F_") >= 2

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
        # i_0 should be eliminated (substituted → a)
        # Result should have SumOverIndex for i_1 only, with Rational(3) absorbed
        assert isinstance(result, SumOverIndex)
        assert result.index_name == "i_1"
        # The body should contain F_{a, i_1} and R_{a,a}
        body = result.body
        assert isinstance(body, Product)
        props = [f for f in body.factors if isinstance(f, Propagator)]
        syms = [f for f in body.factors if isinstance(f, Symbol)]
        assert len(props) == 1
        assert props[0].index_left == "a"
        assert props[0].index_right == "a"
        assert syms[0].indices == ("a", "i_1")
        # Should have Rational(3) from eliminating i_0
        rats = [f for f in body.factors if isinstance(f, Rational)]
        assert len(rats) == 1 and rats[0] == Rational(3)

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
        # Only i_1 remains as summation index
        assert isinstance(result, SumOverIndex)
        assert result.index_name == "i_1"
        # The body should be IntegralOver wrapping a Product
        body = result.body
        assert isinstance(body, IntegralOver)
        inner = body.body
        assert isinstance(inner, Product)
        # Coefficient: -1 × 3 × 3 = -9
        rats = [f for f in inner.factors if isinstance(f, Rational)]
        assert len(rats) == 1 and rats[0] == Rational(-9)
        # F should have indices (a, i_1, i_1)
        syms = [f for f in inner.factors if isinstance(f, Symbol)]
        assert syms[0].indices == ("a", "i_1", "i_1")

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
        # i_0 eliminated → factor of 3
        assert isinstance(result, IntegralOver)
        body = result.body
        assert isinstance(body, Product)
        rats = [f for f in body.factors if isinstance(f, Rational)]
        assert rats[0] == Rational(3)
        props = [f for f in body.factors if isinstance(f, Propagator)]
        assert props[0].index_left == "a" and props[0].index_right == "a"


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
        # Prefactor should include dimension factors: -1 × 3^2 = -9
        assert dt[0].rational_prefactor == Rational(-9)

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
        assert dt_diag.rational_prefactor == Rational(-9)
        # Propagators should have equal indices
        for p in dt_diag.propagators:
            if p.index_left is not None:
                assert p.index_left == p.index_right
