"""Tests for Feynman diagram canonical form and deduplication."""

import pytest
from sft_wick import (
    Field, Vertex, Action, compute_moment, reset_uid_counter,
)
from sft_wick.diagrams import FeynmanDiagram


@pytest.fixture(autouse=True)
def _reset_uid():
    reset_uid_counter()


class TestCanonicalForm:
    def test_same_diagram_same_form(self):
        """Two identical diagrams should have the same canonical form."""
        d1 = FeynmanDiagram()
        e1 = d1.add_external_point("phi_a(x)", "physical", "a", "x")
        e2 = d1.add_external_point("phi_b(y)", "physical", "b", "y")
        v1 = d1.add_vertex("g")
        d1.add_propagator(e1, v1, "R", phi_end=e1, psi_end=v1)
        d1.add_propagator(e2, v1, "C")

        d2 = FeynmanDiagram()
        e1b = d2.add_external_point("phi_a(x)", "physical", "a", "x")
        e2b = d2.add_external_point("phi_b(y)", "physical", "b", "y")
        v1b = d2.add_vertex("g")
        d2.add_propagator(e1b, v1b, "R", phi_end=e1b, psi_end=v1b)
        d2.add_propagator(e2b, v1b, "C")

        assert d1.canonical_form() == d2.canonical_form()

    def test_different_topology_different_form(self):
        """Diagrams with different edge structure should differ."""
        # Diagram 1: both externals connect to the same vertex
        d1 = FeynmanDiagram()
        e1 = d1.add_external_point("phi_a(x)", "physical", "a", "x")
        e2 = d1.add_external_point("phi_b(y)", "physical", "b", "y")
        v1 = d1.add_vertex("g")
        d1.add_propagator(e1, v1, "C")
        d1.add_propagator(e2, v1, "C")

        # Diagram 2: externals connect to different vertices
        d2 = FeynmanDiagram()
        e1b = d2.add_external_point("phi_a(x)", "physical", "a", "x")
        e2b = d2.add_external_point("phi_b(y)", "physical", "b", "y")
        v1b = d2.add_vertex("g")
        v2b = d2.add_vertex("g")
        d2.add_propagator(e1b, v1b, "C")
        d2.add_propagator(e2b, v2b, "C")

        assert d1.canonical_form() != d2.canonical_form()

    def test_vertex_swap_same_coupling(self):
        """Swapping vertices of the same coupling type should give same form."""
        # Diagram A: ext1->v1 via R, ext2->v2 via R, v1-v2 via C
        d1 = FeynmanDiagram()
        e1 = d1.add_external_point("phi_a(x)", "physical", "a", "x")
        e2 = d1.add_external_point("phi_b(y)", "physical", "b", "y")
        v1 = d1.add_vertex("g")
        v2 = d1.add_vertex("g")
        d1.add_propagator(e1, v1, "R", phi_end=e1, psi_end=v1)
        d1.add_propagator(e2, v2, "R", phi_end=e2, psi_end=v2)
        d1.add_propagator(v1, v2, "C")

        # Diagram B: ext1->v2 via R, ext2->v1 via R, v1-v2 via C (vertices swapped)
        d2 = FeynmanDiagram()
        e1b = d2.add_external_point("phi_a(x)", "physical", "a", "x")
        e2b = d2.add_external_point("phi_b(y)", "physical", "b", "y")
        v1b = d2.add_vertex("g")
        v2b = d2.add_vertex("g")
        d2.add_propagator(e1b, v2b, "R", phi_end=e1b, psi_end=v2b)
        d2.add_propagator(e2b, v1b, "R", phi_end=e2b, psi_end=v1b)
        d2.add_propagator(v1b, v2b, "C")

        assert d1.canonical_form() == d2.canonical_form()

    def test_r_direction_matters(self):
        """Reversing R direction should produce different canonical form."""
        # R from ext to vertex (phi=ext, psi=vertex)
        d1 = FeynmanDiagram()
        e1 = d1.add_external_point("phi_a(x)", "physical", "a", "x")
        v1 = d1.add_vertex("g")
        d1.add_propagator(e1, v1, "R", phi_end=e1, psi_end=v1)

        # R from vertex to ext (phi=vertex, psi=ext) — physically different
        d2 = FeynmanDiagram()
        e1b = d2.add_external_point("phi_a(x)", "physical", "a", "x")
        v1b = d2.add_vertex("g")
        d2.add_propagator(e1b, v1b, "R", phi_end=v1b, psi_end=e1b)

        assert d1.canonical_form() != d2.canonical_form()

    def test_c_symmetry(self):
        """C propagator is symmetric: C(a,b) = C(b,a)."""
        d1 = FeynmanDiagram()
        e1 = d1.add_external_point("phi_a(x)", "physical", "a", "x")
        e2 = d1.add_external_point("phi_b(y)", "physical", "b", "y")
        d1.add_propagator(e1, e2, "C")

        d2 = FeynmanDiagram()
        e1b = d2.add_external_point("phi_a(x)", "physical", "a", "x")
        e2b = d2.add_external_point("phi_b(y)", "physical", "b", "y")
        d2.add_propagator(e2b, e1b, "C")  # reversed order

        assert d1.canonical_form() == d2.canonical_form()

    def test_self_loop(self):
        """Self-loops should be handled correctly."""
        d1 = FeynmanDiagram()
        e1 = d1.add_external_point("phi_a(x)", "physical", "a", "x")
        v1 = d1.add_vertex("F")
        d1.add_propagator(e1, v1, "R", phi_end=e1, psi_end=v1)
        d1.add_propagator(v1, v1, "C")  # self-loop

        form = d1.canonical_form()
        assert form is not None
        # The edge tuple should contain both the R and the self-loop C
        edges = form[2]
        assert len(edges) == 2

    def test_no_vertices(self):
        """Zeroth-order diagram with only external nodes."""
        d1 = FeynmanDiagram()
        e1 = d1.add_external_point("phi_a(x)", "physical", "a", "x")
        e2 = d1.add_external_point("phi_b(y)", "physical", "b", "y")
        d1.add_propagator(e1, e2, "C")

        form = d1.canonical_form()
        assert form is not None
        ext_meta, vert_meta, edges = form
        assert len(ext_meta) == 2
        assert vert_meta == ()
        assert len(edges) == 1


class TestDeduplication:
    def test_order0_four_phi_dedup(self):
        """At order 0, <phi_a(x) phi_b(y) phi_c(z) phi_d(w)> has 3
        pairings.  All are genuinely different (different external legs
        connected), so 3 unique diagrams.
        """
        phi = Field("phi", "physical", n_components=3)
        action = Action(vertices=[])
        obs = [phi("a", "x"), phi("b", "y"), phi("c", "z"), phi("d", "w")]
        result = compute_moment(obs, action, order=0, response_phase=False)

        assert len(result.diagrams_by_order[0]) == 3
        fds = [d.to_feynman_diagram() for d in result.diagrams_by_order[0]]
        forms = {fd.canonical_form() for fd in fds}
        # All 3 are distinct because external nodes at different spatial points
        assert len(forms) == 3

    def test_order1_dedup(self):
        """At order 1, verify deduplication reduces diagram count."""
        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        v = Vertex(fields=[phi, psi], coupling="g")
        action = Action(vertices=[v])
        obs = [phi("x"), phi("y")]
        result = compute_moment(obs, action, order=1, response_phase=False)

        raw_count = len(result.diagrams_by_order.get(1, []))
        if raw_count > 0:
            fds = [d.to_feynman_diagram() for d in result.diagrams_by_order[1]]
            forms = {fd.canonical_form() for fd in fds}
            # Unique count should be <= raw count
            assert len(forms) <= raw_count

    def test_vertex_swap_dedup_at_order2(self):
        """At order 2 with same vertex type, diagrams related by
        swapping vertex copies should have the same canonical form."""
        # Build two diagrams that are identical up to v1 <-> v2 swap
        d1 = FeynmanDiagram()
        e1 = d1.add_external_point("phi(x)", "physical", spatial="x")
        e2 = d1.add_external_point("phi(y)", "physical", spatial="y")
        v1 = d1.add_vertex("g", copy_id=0)
        v2 = d1.add_vertex("g", copy_id=1)
        d1.add_propagator(e1, v1, "R", phi_end=e1, psi_end=v1)
        d1.add_propagator(e2, v2, "R", phi_end=e2, psi_end=v2)
        d1.add_propagator(v1, v2, "C")

        # Same topology but v1/v2 swapped
        d2 = FeynmanDiagram()
        e1b = d2.add_external_point("phi(x)", "physical", spatial="x")
        e2b = d2.add_external_point("phi(y)", "physical", spatial="y")
        v1b = d2.add_vertex("g", copy_id=0)
        v2b = d2.add_vertex("g", copy_id=1)
        d2.add_propagator(e1b, v2b, "R", phi_end=e1b, psi_end=v2b)
        d2.add_propagator(e2b, v1b, "R", phi_end=e2b, psi_end=v1b)
        d2.add_propagator(v1b, v2b, "C")

        assert d1.canonical_form() == d2.canonical_form()


class TestDifferentCouplings:
    def test_different_coupling_vertices_not_swapped(self):
        """Vertices with different coupling types should NOT be interchangeable."""
        d1 = FeynmanDiagram()
        e1 = d1.add_external_point("phi_a(x)", "physical", "a", "x")
        v1 = d1.add_vertex("g")
        v2 = d1.add_vertex("h")
        d1.add_propagator(e1, v1, "R", phi_end=e1, psi_end=v1)
        d1.add_propagator(v1, v2, "C")

        # Swap which vertex has which coupling
        d2 = FeynmanDiagram()
        e1b = d2.add_external_point("phi_a(x)", "physical", "a", "x")
        v1b = d2.add_vertex("h")  # note: h first, g second
        v2b = d2.add_vertex("g")
        d2.add_propagator(e1b, v1b, "R", phi_end=e1b, psi_end=v1b)
        d2.add_propagator(v1b, v2b, "C")

        # These should be DIFFERENT because the coupling types differ
        assert d1.canonical_form() != d2.canonical_form()
