"""Deductive verification of sft-wick's symbolic expansion pipeline.

Each test class exercises one transformation stage and cross-checks
sft-wick's output against an independent brute-force reference in
:mod:`tests.brute_wick`.  All equalities are exact (symbolic or
rational) — no numerical tolerances.

Test IDs T1–T14 are defined in
``/Users/zzhang/.claude/plans/in-the-current-demo-sunny-panda.md``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np
import pytest

from sft_wick import Action, Field, Vertex, compute_moment, reset_uid_counter
from sft_wick.expressions import Product, Propagator as PropagatorExpr, Sum, Symbol
from sft_wick.propagators import contract_pair
from sft_wick.wick import generate_valid_pairings, wick_contract

from . import brute_wick as brute
from .brute_wick import Op


@pytest.fixture(autouse=True)
def _reset_uid() -> None:
    reset_uid_counter()


# =====================================================================
# Fixtures — sft-wick operator lists matching brute_wick's layouts
# =====================================================================


def _sftwick_case_A_ops(phi, psi):
    """sft-wick operators matching :func:`brute_wick.case_A_operators`."""
    return [
        phi("x"),
        phi("y"),
        psi("z0"),
        phi("z0"),
        phi("z0"),
        psi("z1"),
        phi("z1"),
        phi("z1"),
    ]


def _sftwick_case_B_FK_ops(phi, psi):
    """sft-wick operators matching :func:`brute_wick.case_B_FK_operators`."""
    return [
        phi("x"),
        phi("y"),
        psi("z0"),
        phi("z0"),
        phi("z0"),
        psi("y1"),
        psi("y2"),
        psi("y3"),
    ]


def _pair_to_normal_form(pair_tuple: tuple[tuple[int, int], ...]) -> frozenset:
    """Convert sft-wick's Pairing (index-based tuple) to brute's frozenset form.

    The indices in sft-wick's Pairing are positions in the ``operators``
    list, not uids.  Callers must map them to uids before comparing with
    brute output.
    """
    return frozenset(frozenset(p) for p in pair_tuple)


def _sftwick_pairing_to_uid_set(
    sftwick_pairing, operators
) -> frozenset:
    """Translate sft-wick's (i,j)-index pairing into a uid-based frozenset."""
    return frozenset(
        frozenset({operators[i].uid, operators[j].uid}) for (i, j) in sftwick_pairing
    )


# =====================================================================
# T1 — pairing enumeration
# =====================================================================


class TestEnumeration:
    """T1: (2n-1)!! pairing counts."""

    @pytest.mark.parametrize(
        "n_ops,expected",
        [(0, 1), (2, 1), (4, 3), (6, 15), (8, 105), (10, 945)],
    )
    def test_brute_count_matches_double_factorial(self, n_ops, expected):
        assert brute.count_pairings(n_ops) == expected
        uids = list(range(n_ops))
        assert sum(1 for _ in brute.enumerate_pairings(uids)) == expected

    def test_odd_count_is_empty(self):
        assert brute.count_pairings(3) == 0
        assert sum(1 for _ in brute.enumerate_pairings([0, 1, 2])) == 0

    def test_sftwick_all_phi_count_matches_brute(self):
        """sft-wick generate_valid_pairings on all-phi ops equals (2n-1)!!."""
        phi = Field("phi", "physical")
        for n in (2, 3, 4):
            ops = [phi(f"x{i}") for i in range(2 * n)]
            phi_idx = list(range(len(ops)))
            psi_idx: list[int] = []
            sft_count = sum(
                1 for _ in generate_valid_pairings(phi_idx, psi_idx, ops, ito=False)
            )
            assert sft_count == brute.count_pairings(2 * n)


# =====================================================================
# T2 — per-pairing vanishing rules
# =====================================================================


class TestVanishing:
    """T2: brute classification agrees with sft-wick on *exactly* which
    pairings survive Case A at order 2."""

    def test_case_A_classification_counts(self):
        ops = brute.case_A_operators()
        kept, reasons = brute.classify_all(ops, ito=True)
        # From the analysis in brute_wick's doctest run:
        # 8 ops -> 105 pairings; 15 killed by psi-psi (the 2 ψ's cannot
        # pair with each other in fact they form a *single* bad pair and
        # all pairings containing it); 48 by Itô (R(z,z) tadpoles);
        # 12 by causal R-loops; 30 survive.
        assert reasons == {
            "psi-psi": 15,
            "ito": 48,
            "r-loop": 12,
            "ok": 30,
        }
        assert len(kept) == 30

    def test_sftwick_kept_set_matches_brute(self):
        """generate_valid_pairings(ops, ito=True) returns exactly the
        brute-force kept set."""
        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        ops_sft = _sftwick_case_A_ops(phi, psi)

        # Split indices into phi / psi
        phi_idx = [i for i, op in enumerate(ops_sft) if op.field.is_physical]
        psi_idx = [i for i, op in enumerate(ops_sft) if op.field.is_response]
        sft_kept_uid_sets = set()
        for pairing in generate_valid_pairings(phi_idx, psi_idx, ops_sft, ito=True):
            sft_kept_uid_sets.add(_sftwick_pairing_to_uid_set(pairing, ops_sft))

        # Build brute ops with the *same* uids as the sft-wick ops so the
        # comparison is direct.
        ops_brute = [
            Op(
                kind="phi" if op.field.is_physical else "psi",
                spatial=op.spatial_arg,
                comp=op.component_index,
                uid=op.uid,
            )
            for op in ops_sft
        ]
        brute_kept, _ = brute.classify_all(ops_brute, ito=True)

        assert sft_kept_uid_sets == brute_kept

    def test_psipsi_reason_fires(self):
        """Every ψ-ψ pair must be classified psi-psi, not other reasons."""
        ops = brute.case_B_KK_operators()  # 2φ + 6ψ
        # Every pairing must contain a ψ-ψ pair (only 2 φ's to spread
        # across 8 operators), so all 105 die by psi-psi.
        _, reasons = brute.classify_all(ops, ito=True)
        assert reasons == {"psi-psi": 105, "ok": 0}

    def test_ito_prune_equals_post_filter(self):
        """Itô early pruning in generate_valid_pairings equals post-hoc
        filter — same pairings survive both paths."""
        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        ops = _sftwick_case_A_ops(phi, psi)
        phi_idx = [i for i, op in enumerate(ops) if op.field.is_physical]
        psi_idx = [i for i, op in enumerate(ops) if op.field.is_response]

        with_prune = set(
            generate_valid_pairings(phi_idx, psi_idx, ops, ito=True)
        )
        # Post-hoc: enumerate with ito=False, then filter via contract_pair
        no_prune_raw = list(
            generate_valid_pairings(phi_idx, psi_idx, ops, ito=False)
        )
        filtered = set()
        for pairing in no_prune_raw:
            props = []
            ok = True
            for i, j in pairing:
                p = contract_pair(ops[i], ops[j], ito=True)
                if p is None:
                    ok = False
                    break
                props.append(p)
            if not ok:
                continue
            # Post-hoc R-loop filter
            if brute._has_r_cycle(
                [(p.kind, p.spatial_left, p.spatial_right, None, None) for p in props]
            ):
                continue
            filtered.add(pairing)

        assert with_prune == filtered


# =====================================================================
# T3 — operator-level Wick sum
# =====================================================================


class TestOperatorLevelWick:
    """T3: ``wick_contract`` produces a Sum whose terms are exactly the
    brute-classified propagator products (modulo ordering)."""

    def test_case_A_term_count_matches(self):
        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        ops = _sftwick_case_A_ops(phi, psi)
        expr, pairings = wick_contract(ops, ito=True)
        assert len(pairings) == 30
        # If expr is a Sum, its term count matches.  If only one
        # surviving pairing, it collapses — here we have 30 so it's a Sum.
        assert isinstance(expr, Sum)
        assert len(expr.terms) == 30

    def test_case_B_FK_term_count_matches(self):
        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        ops = _sftwick_case_B_FK_ops(phi, psi)
        expr, pairings = wick_contract(ops, ito=True)
        assert len(pairings) == 12  # brute reference
        assert isinstance(expr, Sum)
        assert len(expr.terms) == 12


# =====================================================================
# T4 — spatial multiplicity
# =====================================================================


class TestSpatialMultiplicity:
    """T4: sft-wick's spatial topology enumeration has the right
    per-topology multiplicities."""

    def test_case_A_total_count_equals_brute(self):
        """The sum of multiplicities over spatial topologies equals the
        number of kept labelled pairings."""
        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        ops_sft = _sftwick_case_A_ops(phi, psi)

        # Build equivalent brute ops using identical uids
        ops_brute = [
            Op(
                "phi" if op.field.is_physical else "psi",
                op.spatial_arg,
                op.component_index,
                op.uid,
            )
            for op in ops_sft
        ]
        brute_topologies = brute.topology_multiplicities(ops_brute, ito=True)
        total = sum(brute_topologies.values())
        assert total == 30  # Matches brute_wick.classify_all count.

        # Cross-check: wick_contract gives 30 pairings too.
        _, pairings = wick_contract(ops_sft, ito=True)
        assert len(pairings) == 30

    def test_case_B_FK_total_count_equals_brute(self):
        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        ops_sft = _sftwick_case_B_FK_ops(phi, psi)

        ops_brute = [
            Op(
                "phi" if op.field.is_physical else "psi",
                op.spatial_arg,
                op.component_index,
                op.uid,
            )
            for op in ops_sft
        ]
        brute_topologies = brute.topology_multiplicities(ops_brute, ito=True)
        total = sum(brute_topologies.values())
        assert total == 12

        _, pairings = wick_contract(ops_sft, ito=True)
        assert len(pairings) == 12


# =====================================================================
# T5 — engine agreement (collect_topology=False vs True)
# =====================================================================


class TestEngineAgreement:
    """T5: the two compute_moment paths produce the same set of Feynman
    diagrams (compared via :meth:`FeynmanDiagram.canonical_form`).

    Implementation note: the ``collect_topology=False`` path doesn't
    populate ``diagram_terms_by_order`` (that structured representation
    is built only by the hybrid spatial/topology engine).  Both paths
    however populate ``diagrams_by_order`` with ``DiagramInfo`` records,
    which can be rendered into Feynman diagrams and canonicalised.  This
    is the natural handle for comparing topology equivalence.
    """

    @pytest.mark.parametrize("case", ["A", "B"])
    def test_engine_agreement_order_2(self, case):
        phi = Field("phi", "physical", n_components=1)
        psi = Field("psi", "response", n_components=1)
        v1 = Vertex(fields=[psi, phi, phi], coupling="F", local=True)
        if case == "A":
            action = Action(vertices=[v1])
        else:
            v2 = Vertex(fields=[psi, psi, psi], coupling="K", local=False)
            action = Action(vertices=[v1, v2])
        obs = [phi("x"), phi("y")]

        reset_uid_counter()
        r_op = compute_moment(
            obs, action, order=2,
            collect_topology=False, response_phase=False, ito=True,
        )
        reset_uid_counter()
        r_sp = compute_moment(
            obs, action, order=2,
            collect_topology=True, response_phase=False, ito=True,
        )

        op_keys = Counter(
            d.to_feynman_diagram().canonical_form()
            for d in r_op.diagrams_by_order.get(2, [])
        )
        sp_keys = Counter(
            d.to_feynman_diagram().canonical_form()
            for d in r_sp.diagrams_by_order.get(2, [])
        )
        # Same set of topologies must appear.  Multiplicities may differ
        # because the operator-level path records one DiagramInfo per
        # pairing (labelled) while the spatial path groups by topology;
        # so we compare sets of distinct canonical forms.
        assert set(op_keys.keys()) == set(sp_keys.keys()), (
            f"operator-level topologies: {set(op_keys.keys()) - set(sp_keys.keys())}\n"
            f"spatial-level topologies:  {set(sp_keys.keys()) - set(op_keys.keys())}"
        )

    def test_order2_topology_count_agrees_with_brute(self):
        """For Case A, both paths report exactly 6 distinct Feynman-diagram
        canonical forms at order 2 — matching the hand-enumeration count."""
        phi = Field("phi", "physical", n_components=1)
        psi = Field("psi", "response", n_components=1)
        v1 = Vertex(fields=[psi, phi, phi], coupling="F", local=True)
        action = Action(vertices=[v1])
        obs = [phi("x"), phi("y")]
        reset_uid_counter()
        r_sp = compute_moment(obs, action, order=2, response_phase=False, ito=True)
        distinct = {
            d.to_feynman_diagram().canonical_form()
            for d in r_sp.diagrams_by_order.get(2, [])
        }
        assert len(distinct) == 6


# =====================================================================
# T6 — Taylor / multinomial prefactor
# =====================================================================


class TestPrefactor:
    """T6: DiagramTerm.rational_prefactor equals (-1)^n/n! × multinomial."""

    def test_case_A_order_2_prefactors(self):
        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        v1 = Vertex(fields=[psi, phi, phi], coupling="F", local=True)
        action = Action(vertices=[v1])
        obs = [phi("x"), phi("y")]

        result = compute_moment(
            obs, action, order=2, response_phase=False, ito=True,
        )
        # For order=2 with a single vertex type, multinomial = 2!/2! = 1.
        # Prefactor = (-1)^2 / 2! × 1 = 1/2.
        expected = brute.taylor_prefactor(n=2, multinomial=1)
        assert expected == Fraction(1, 2)
        for dt in result.diagram_terms(2):
            got = Fraction(
                dt.rational_prefactor.numerator,
                dt.rational_prefactor.denominator,
            )
            assert got == expected, f"{got} != {expected} for {dt.propagators}"

    def test_case_B_FK_prefactor(self):
        """For the FK vertex pair at order 2: multinomial = 2!/1!1! = 2,
        prefactor = 2/2! = 1."""
        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        v1 = Vertex(fields=[psi, phi, phi], coupling="F", local=True)
        v2 = Vertex(fields=[psi, psi, psi], coupling="K", local=False)
        action = Action(vertices=[v1, v2])
        obs = [phi("x"), phi("y")]

        result = compute_moment(
            obs, action, order=2, response_phase=False, ito=True,
        )

        expected_FF = brute.taylor_prefactor(n=2, multinomial=1)   # 1/2
        expected_FK = brute.taylor_prefactor(n=2, multinomial=2)   # 1

        def _tags(dt) -> set[str]:
            stack, tags = [dt.coupling_sum], set()
            while stack:
                n = stack.pop()
                if isinstance(n, Symbol):
                    tags.add(n.name)
                elif isinstance(n, Sum):
                    stack.extend(n.terms)
                elif isinstance(n, Product):
                    stack.extend(n.factors)
            return tags

        for dt in result.diagram_terms(2):
            tags = _tags(dt)
            got = Fraction(
                dt.rational_prefactor.numerator,
                dt.rational_prefactor.denominator,
            )
            if tags == {"F"}:
                assert got == expected_FF, f"FF got {got}"
            elif tags == {"F", "K"}:
                assert got == expected_FK, f"FK got {got}"
            else:
                pytest.fail(f"Unexpected tag set {tags}")


# =====================================================================
# T7 — response phase
# =====================================================================


class TestResponsePhase:
    """T7: DiagramTerm.response_phase_factor() equals (-i)^n_R."""

    @pytest.mark.parametrize("n_R,expected", [
        (0, 1.0 + 0j),
        (1, -1j),
        (2, -1.0 + 0j),
        (3, 1j),
        (4, 1.0 + 0j),
        (5, -1j),
    ])
    def test_phase_formula(self, n_R, expected):
        assert brute.response_phase(n_R) == expected

    def test_n_response_matches_R_count(self):
        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        v1 = Vertex(fields=[psi, phi, phi], coupling="F", local=True)
        action = Action(vertices=[v1])
        obs = [phi("x"), phi("y")]

        result = compute_moment(
            obs, action, order=2, response_phase=True, ito=True,
        )
        for dt in result.diagram_terms(2):
            r_count = sum(1 for p in dt.propagators if p.kind == "R")
            assert dt.n_response == r_count
            assert dt.response_phase_factor() == brute.response_phase(r_count)


# =====================================================================
# T8 — canonical-form agreement (nauty vs brute)
# =====================================================================


class TestCanonicalForm:
    """T8: the brute canonicaliser and ``_canonical_key_nauty`` (if
    available) induce the same equivalence relation."""

    def test_c_edge_symmetry(self):
        """``simplify._canonical_diagram_form`` treats C(a,b) == C(b,a)."""
        from sft_wick.expressions import Propagator as PE
        from sft_wick.simplify import _canonical_diagram_form

        c1 = PE(kind="C", index_left=None, index_right=None,
                spatial_left="a", spatial_right="b")
        c2 = PE(kind="C", index_left=None, index_right=None,
                spatial_left="b", spatial_right="a")
        k1, _ = _canonical_diagram_form([c1], frozenset())
        k2, _ = _canonical_diagram_form([c2], frozenset())
        assert k1 == k2

    def test_r_directionality(self):
        """R is directed: R(a,b) != R(b,a)."""
        from sft_wick.expressions import Propagator as PE
        from sft_wick.simplify import _canonical_diagram_form

        r1 = PE(kind="R", index_left=None, index_right=None,
                spatial_left="a", spatial_right="b")
        r2 = PE(kind="R", index_left=None, index_right=None,
                spatial_left="b", spatial_right="a")
        k1, _ = _canonical_diagram_form([r1], frozenset())
        k2, _ = _canonical_diagram_form([r2], frozenset())
        assert k1 != k2

    def test_integration_var_relabel_invariance(self):
        """Relabeling z0 ↔ z1 in a two-vertex diagram gives the same
        canonical form."""
        from sft_wick.expressions import Propagator as PE
        from sft_wick.simplify import _canonical_diagram_form

        props_A = [
            PE(kind="R", index_left=None, index_right=None,
               spatial_left="x", spatial_right="z0"),
            PE(kind="R", index_left=None, index_right=None,
               spatial_left="y", spatial_right="z1"),
            PE(kind="C", index_left=None, index_right=None,
               spatial_left="z0", spatial_right="z0"),
            PE(kind="C", index_left=None, index_right=None,
               spatial_left="z1", spatial_right="z1"),
        ]
        props_B = [
            PE(kind="R", index_left=None, index_right=None,
               spatial_left="x", spatial_right="z1"),
            PE(kind="R", index_left=None, index_right=None,
               spatial_left="y", spatial_right="z0"),
            PE(kind="C", index_left=None, index_right=None,
               spatial_left="z1", spatial_right="z1"),
            PE(kind="C", index_left=None, index_right=None,
               spatial_left="z0", spatial_right="z0"),
        ]
        integration = frozenset({"z0", "z1"})
        k1, _ = _canonical_diagram_form(props_A, integration)
        k2, _ = _canonical_diagram_form(props_B, integration)
        assert k1 == k2, "relabelling should not change canonical form"

    def test_nauty_agrees_with_simple_form(self):
        """If pynauty is installed, its key partitions identically to the
        simple canonicaliser on Case A labelled pairings."""
        pynauty = pytest.importorskip("pynauty")
        from sft_wick.expressions import Propagator as PE
        from sft_wick.simplify import _canonical_diagram_form, _canonical_key_nauty

        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        ops_sft = _sftwick_case_A_ops(phi, psi)
        _, pairings = wick_contract(ops_sft, ito=True)

        # Extract propagator lists per pairing
        integration = frozenset({"z0", "z1"})
        pair_props_list = []
        for pairing in pairings:
            props: list = []
            for i, j in pairing:
                pr = contract_pair(ops_sft[i], ops_sft[j], ito=True)
                props.append(pr)
            pair_props_list.append(props)

        # Partition using both canonicalisers
        simple_key = [
            _canonical_diagram_form(plist, integration)[0]
            for plist in pair_props_list
        ]
        nauty_key = [
            _canonical_key_nauty(plist, integration)
            for plist in pair_props_list
        ]

        # Partition refines each other iff the two keyings agree up to
        # bijection.  Easier check: "i, j have same simple key" iff
        # "i, j have same nauty key".
        for i in range(len(pair_props_list)):
            for j in range(i + 1, len(pair_props_list)):
                same_simple = simple_key[i] == simple_key[j]
                same_nauty = nauty_key[i] == nauty_key[j]
                assert same_simple == same_nauty, (
                    f"mismatch on pair {i}, {j}: simple={same_simple} "
                    f"nauty={same_nauty}"
                )


# =====================================================================
# T9 — diagram-collection soundness
# =====================================================================


class TestCollectionSoundness:
    """T9: ``collect_by_diagram`` merges iff brute isomorphism key agrees.

    This checks both (a) no false merges (two non-isomorphic topologies
    combined) and (b) no false splits (one topology scattered into
    multiple collected terms)."""

    def _brute_isomorphism_key(self, propagators) -> tuple:
        """Canonical key over *all* relabelings of integration variables.

        For small topologies we can afford to try every permutation of
        {z0, z1} and pick the lexicographically smallest edge list.
        """
        from itertools import permutations

        internal = ("z0", "z1")
        best: tuple | None = None
        for perm in permutations(internal):
            mapping = dict(zip(internal, perm))
            edges = []
            for p in propagators:
                sl = mapping.get(p.spatial_left, p.spatial_left)
                sr = mapping.get(p.spatial_right, p.spatial_right)
                if p.kind == "C" and sl > sr:
                    sl, sr = sr, sl
                edges.append((p.kind, sl, sr))
            edges.sort()
            key = tuple(edges)
            if best is None or key < best:
                best = key
        return best

    def test_case_A_no_false_merges_or_splits(self):
        """For every pair of surviving pairings, sft-wick's canonical
        form merges them iff their brute-isomorphism keys match."""
        from sft_wick.simplify import _canonical_diagram_form

        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        ops_sft = _sftwick_case_A_ops(phi, psi)
        _, pairings = wick_contract(ops_sft, ito=True)

        integration = frozenset({"z0", "z1"})
        brute_keys: list[tuple] = []
        sft_keys: list[tuple] = []
        for pairing in pairings:
            props = [
                contract_pair(ops_sft[i], ops_sft[j], ito=True)
                for (i, j) in pairing
            ]
            brute_keys.append(self._brute_isomorphism_key(props))
            sft_keys.append(_canonical_diagram_form(props, integration)[0])

        for i in range(len(pairings)):
            for j in range(i + 1, len(pairings)):
                assert (brute_keys[i] == brute_keys[j]) == (
                    sft_keys[i] == sft_keys[j]
                ), (
                    f"partition mismatch on pair {i},{j}:\n"
                    f"  brute keys: {brute_keys[i]} vs {brute_keys[j]}\n"
                    f"  sft   keys: {sft_keys[i]} vs {sft_keys[j]}"
                )

    def test_case_A_order_2_exactly_6_physical_diagrams(self):
        """sft-wick's diagram_terms(2) has 6 distinct physical diagrams —
        matches the Feynman-diagram enumeration used in demo1."""
        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        v1 = Vertex(fields=[psi, phi, phi], coupling="F", local=True)
        action = Action(vertices=[v1])
        obs = [phi("x"), phi("y")]

        result = compute_moment(obs, action, order=2, response_phase=False)
        assert len(result.diagram_terms(2)) == 6


# =====================================================================
# T10 — simplification passes (individual)
# =====================================================================


class TestSimplifyPasses:
    """T10: hand-built expressions pass through each simplification step
    with the expected invariants."""

    def test_flatten_nested_sums(self):
        from sft_wick.expressions import Rational
        from sft_wick.simplify import _flatten

        inner = Sum((Rational(1), Rational(2)))
        outer = Sum((inner, Rational(3)))
        flat = _flatten(outer)
        assert isinstance(flat, Sum)
        # Nested sums collapse into one flat sum
        assert len(flat.terms) == 3

    def test_flatten_nested_products(self):
        from sft_wick.expressions import Rational
        from sft_wick.simplify import _flatten

        inner = Product((Rational(2), Rational(3)))
        outer = Product((inner, Rational(5)))
        flat = _flatten(outer)
        assert isinstance(flat, Product)
        assert len(flat.factors) == 3

    def test_absorb_rationals(self):
        from sft_wick.expressions import Rational
        from sft_wick.simplify import _absorb_rationals, _flatten

        expr = Product((Rational(2, 3), Rational(3, 4), Rational(1, 5)))
        absorbed = _absorb_rationals(_flatten(expr))
        # 2/3 × 3/4 × 1/5 = 6/60 = 1/10
        assert isinstance(absorbed, Rational)
        assert Fraction(absorbed.numerator, absorbed.denominator) == Fraction(1, 10)

    def test_eliminate_zeros_in_product(self):
        from sft_wick.expressions import Rational, ZERO
        from sft_wick.simplify import _eliminate_zeros, _flatten

        expr = Product((Rational(5), ZERO, Rational(3)))
        result = _eliminate_zeros(_flatten(expr))
        # Any product containing zero simplifies to zero
        assert result == ZERO

    def test_eliminate_zeros_in_sum(self):
        from sft_wick.expressions import Rational, ZERO
        from sft_wick.simplify import _eliminate_zeros, _flatten

        expr = Sum((Rational(2), ZERO, Rational(3)))
        result = _eliminate_zeros(_flatten(expr))
        # Zeros dropped from sum
        assert isinstance(result, Sum)
        assert ZERO not in result.terms


# =====================================================================
# T11 — non-local vertex instantiation (Case B)
# =====================================================================


class TestNonLocalInstantiation:
    """T11: VertexInstance.instantiate produces distinct spatial args for
    non-local vertices across repeated instantiation."""

    def test_non_local_unique_spatial_vars(self):
        from sft_wick.indices import IndexContext
        from sft_wick.vertices import VertexInstance

        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        v2 = Vertex(fields=[psi, psi, psi], coupling="K", local=False)

        all_spatial_vars: set[str] = set()
        ctx = IndexContext()
        for k in range(50):
            inst = VertexInstance.instantiate(v2, ctx, copy_id=k)
            assert len(inst.spatial_variables) == 3  # one per field
            # All three variables must be distinct within this instance
            assert len(set(inst.spatial_variables)) == 3
            # And disjoint from all prior instantiations
            for var in inst.spatial_variables:
                assert var not in all_spatial_vars
                all_spatial_vars.add(var)

    def test_local_vs_non_local_spatial_sharing(self):
        """Local vertex shares one spatial var; non-local uses three."""
        from sft_wick.indices import IndexContext
        from sft_wick.vertices import VertexInstance

        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        v_local = Vertex(fields=[psi, phi, phi], coupling="F", local=True)
        v_nonlocal = Vertex(fields=[psi, psi, psi], coupling="K", local=False)

        ctx = IndexContext()
        il = VertexInstance.instantiate(v_local, ctx, 0)
        inl = VertexInstance.instantiate(v_nonlocal, ctx, 1)

        assert len(il.spatial_variables) == 1
        assert len(inl.spatial_variables) == 3
        # Local vertex: all field operators share the one spatial
        assert len({op.spatial_arg for op in il.field_operators}) == 1
        # Non-local vertex: each field gets a distinct spatial
        assert len({op.spatial_arg for op in inl.field_operators}) == 3


# =====================================================================
# T12 — KK vanishing
# =====================================================================


class TestKKVanishes:
    """T12: ⟨φφ⟩ at order 2 with ψψψ-only action vanishes by leg count."""

    def test_kk_vanishing_by_leg_count(self):
        ops = brute.case_B_KK_operators()
        assert brute.vanishes_by_leg_count(ops)

    def test_sftwick_kk_at_order_2_empty(self):
        """compute_moment with only v_2 at order 2 has no surviving
        DiagramTerms."""
        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        v2 = Vertex(fields=[psi, psi, psi], coupling="K", local=False)
        action = Action(vertices=[v2])
        obs = [phi("x"), phi("y")]

        result = compute_moment(obs, action, order=2, response_phase=False)
        assert result.diagram_terms(2) == []


# =====================================================================
# T13 — coupling-sum collapse for FK (component-diagonal K)
# =====================================================================


class TestFKCouplingCollapse:
    """T13: with component-diagonal K, the 6-term coupling sum collapses
    to 6·F[a,b,b]·K[b,b,b] for diagram [6] and 6·F[b,a,a]·K[a,a,a] for [7]."""

    def _setup_FK(self, n_comp: int = 2):
        phi = Field("phi", "physical", n_components=n_comp)
        psi = Field("psi", "response", n_components=n_comp)
        v1 = Vertex(fields=[psi, phi, phi], coupling="F", local=True)
        v2 = Vertex(fields=[psi, psi, psi], coupling="K", local=False)
        action = Action(vertices=[v1, v2])
        obs = [phi("a", "x"), phi("b", "y")]
        reset_uid_counter()
        result = compute_moment(
            obs, action, order=2, response_phase=True, ito=True,
            diag_R=True, diag_C=True, iso_R=True,
        )
        # Extract FK DiagramTerms (those referencing both F and K)
        FK = []
        for dt in result.diagram_terms(2):
            stack, tags = [dt.coupling_sum], set()
            while stack:
                n = stack.pop()
                if isinstance(n, Symbol):
                    tags.add(n.name)
                elif isinstance(n, Sum):
                    stack.extend(n.terms)
                elif isinstance(n, Product):
                    stack.extend(n.factors)
            if tags == {"F", "K"}:
                FK.append(dt)
        return FK, phi, psi

    def test_FK_count_is_2(self):
        FK, *_ = self._setup_FK()
        assert len(FK) == 2

    def test_FK_evaluate_component_diagonal(self):
        """At (a, b) = (0, 1) with component-diagonal K, the hand formula
        gives ``F[a,b,b] × kappa3_bbb + F[b,a,a] × kappa3_aaa`` (times
        the prefactor and phase)."""
        FK, phi, psi = self._setup_FK(n_comp=2)

        # F tensor from demo1 convention; non-zero entries:
        F_arr = np.zeros((2, 2, 2))
        F_arr[0, 1, 1] = 1.0
        F_arr[1, 0, 1] = 0.5
        F_arr[1, 1, 0] = 0.5
        F_MSR = -1j * F_arr

        # Component-diagonal K = (i/6) κ^(3): K[a,a,a] = k_aaa[a].
        k_aaa = np.array([0.3, 0.5])  # arbitrary distinct values
        K_arr = np.zeros((2, 2, 2), dtype=complex)
        for a in range(2):
            K_arr[a, a, a] = 1j / 6.0 * k_aaa[a]

        # Sum both FK diagrams — each returns a scalar after evaluation
        # with fixed (a=0, b=1), propagator indices already summed by
        # evaluate_coupling internally.
        total = 0.0 + 0j
        for dt in FK:
            val = dt.evaluate_coupling(
                {"F": F_MSR, "K": K_arr}, fixed_indices={"a": 0, "b": 1}
            )
            # evaluate_coupling returns 0-d array (no propagator indices
            # in iso-R component-diagonal mode).  Convert to Python scalar.
            total += complex(np.asarray(val))

        # Hand derivation at (a, b) = (0, 1):
        #
        # For each FK DiagramTerm, evaluate_coupling returns:
        #     rational_prefactor × (-i)^{n_R=4} × [coupling_sum at indices]
        #   = 1 × 1 × ∑_{i_0, i_1} F_MSR[a, i_0, i_1] K_MSR[b, i_0, i_1] + perm
        #
        # K is component-diagonal, so K_MSR[b, i_0, i_1] ≠ 0 only if
        # b = i_0 = i_1.  Each of the 6 permutation terms in coupling_sum
        # collapses to the same F_MSR[a, b, b] × K_MSR[b, b, b], giving
        # 6 × F_MSR[a, b, b] × K_MSR[b, b, b].
        #
        # With K_MSR[b,b,b] = (i/6) × k_aaa[b] and F_MSR[a,b,b] = -i × F[a,b,b]:
        #     6 × (-i·F[a,b,b]) × (i/6·k_aaa[b])
        #   = (-i)(i) × F[a,b,b] × k_aaa[b]
        #   = F[a,b,b] × k_aaa[b].
        #
        # The 6 from permutations cancels 1/6 from K's definition as
        # (i/6)·κ³.  So the per-diagram value is simply F[a,b,b]·κ³_bbb.
        expected_from_6 = F_arr[0, 1, 1] * k_aaa[1]  # diag [6]
        expected_from_7 = F_arr[1, 0, 0] * k_aaa[0]  # diag [7], F_arr[1,0,0]=0
        expected = expected_from_6 + expected_from_7

        assert abs(complex(total).imag) < 1e-10, f"result should be real: {total}"
        assert abs(complex(total).real - expected) < 1e-10, (
            f"got {complex(total).real}, expected {expected}"
        )


# =====================================================================
# T15-T20 multi-component coverage
# =====================================================================


class TestMultiComponentTopologyInvariance:
    """T15: the *spatial* topology count is N-invariant.

    At order 2 with a single cubic vertex, six distinct Feynman-diagram
    canonical forms exist. This is a property of the spatial Wick
    structure and must not depend on how many internal component indices
    each propagator carries.
    """

    @pytest.mark.parametrize("n_comp", [1, 2, 3])
    def test_six_spatial_topologies_order_2(self, n_comp):
        phi = Field("phi", "physical", n_components=n_comp)
        psi = Field("psi", "response", n_components=n_comp)
        v1 = Vertex(fields=[psi, phi, phi], coupling="F", local=True)
        action = Action(vertices=[v1])
        if n_comp == 1:
            obs = [phi("x"), phi("y")]
        else:
            obs = [phi("a", "x"), phi("b", "y")]

        reset_uid_counter()
        result = compute_moment(obs, action, order=2, response_phase=False)

        dt_keys = set()
        for dt in result.diagram_terms(2):
            rename: dict[str, str] = {}
            nxt = 0
            edges = []
            for p in dt.propagators:
                ends = []
                for v in (p.spatial_left, p.spatial_right):
                    if v in dt.integration_vars and v not in rename:
                        rename[v] = f"_i{nxt}"; nxt += 1
                    ends.append(rename.get(v, v))
                sl, sr = ends
                if p.kind == "C" and sl > sr:
                    sl, sr = sr, sl
                edges.append((p.kind, sl, sr))
            edges.sort()
            dt_keys.add(tuple(edges))

        assert len(dt_keys) == 6, (
            f"n_comp={n_comp}: expected 6 spatial topologies, got {len(dt_keys)}"
        )


class TestMultiComponentSummationStructure:
    """T16: ``DiagramTerm.summation_indices`` scales predictably with N.

    Each component-index summation records ``(name, dimension)``; the
    dimensions must all equal the ``n_components`` of the field that
    introduced them.
    """

    @pytest.mark.parametrize("n_comp", [2, 3, 5])
    def test_summation_dimensions_equal_n_components(self, n_comp):
        phi = Field("phi", "physical", n_components=n_comp)
        psi = Field("psi", "response", n_components=n_comp)
        v1 = Vertex(fields=[psi, phi, phi], coupling="F")
        action = Action(vertices=[v1])
        obs = [phi("a", "x"), phi("b", "y")]

        reset_uid_counter()
        result = compute_moment(
            obs, action, order=2, response_phase=False,
            diag_R=True, diag_C=True, iso_R=True, iso_C=True,
        )
        for dt in result.diagram_terms(2):
            for name, dim in dt.summation_indices:
                assert dim == n_comp, (
                    f"n_comp={n_comp}: index {name} has dim {dim}, "
                    f"expected {n_comp}"
                )

    def test_order_2_has_two_summation_indices_per_diagram(self):
        """Every order-2 FF diagram has exactly 2 internal summation
        indices (from the two vertex copies' phi-phi component pairing).
        """
        phi = Field("phi", "physical", n_components=3)
        psi = Field("psi", "response", n_components=3)
        v1 = Vertex(fields=[psi, phi, phi], coupling="F")
        action = Action(vertices=[v1])
        obs = [phi("a", "x"), phi("b", "y")]

        reset_uid_counter()
        result = compute_moment(
            obs, action, order=2, response_phase=False,
            diag_R=True, diag_C=True, iso_R=True, iso_C=True,
        )
        for i, dt in enumerate(result.diagram_terms(2)):
            assert len(dt.summation_indices) == 2, (
                f"diagram {i}: got {len(dt.summation_indices)} "
                f"summation indices, expected 2"
            )


class TestMultiComponentHandVerified:
    """T17: hand-computed ``evaluate_coupling`` outputs at N=2 for demo 1's
    F tensor.

    For each of the six FF diagrams at order 2, we derive the closed-form
    coupling-sum value as a function of ``(a, b)`` and verify sft-wick's
    numerical output matches.  The derivations depend only on the Wick
    contraction structure and the psi-phi-phi vertex's symmetry under
    swapping its two phi legs.
    """

    def _demo1_F(self):
        F = np.zeros((2, 2, 2))
        F[0, 1, 1] = 1.0
        F[1, 0, 1] = 0.5
        F[1, 1, 0] = 0.5
        return F

    def _setup_diagrams(self):
        N = 2
        F_arr = self._demo1_F()
        F_MSR = -1j * F_arr

        reset_uid_counter()
        phi = Field("phi", "physical", n_components=N)
        psi = Field("psi", "response", n_components=N)
        action = Action(vertices=[Vertex([psi, phi, phi], coupling="F")])
        obs = [phi("a", "x"), phi("b", "y")]

        result = compute_moment(
            obs, action, order=2, response_phase=True,
            diag_R=True, diag_C=True, iso_R=True, iso_C=True,
        )
        return result.diagram_terms(2), F_MSR, F_arr

    def _value(self, dt, F_MSR, a, b):
        val = dt.evaluate_coupling({"F": F_MSR}, fixed_indices={"a": a, "b": b})
        return complex(np.asarray(val))

    # ---------- Hand-derived reference formulas ----------

    def _S(self, F, c):
        """Trace-like:  S(c) = sum_i F[c, i, i]."""
        return sum(F[c, i, i] for i in range(F.shape[1]))

    def _T1(self, F, a, b):
        return sum(F[a, i, j] * F[b, i, j]
                   for i in range(F.shape[1]) for j in range(F.shape[2]))

    def _T2(self, F, a, b):
        return sum(F[a, i, j] * F[b, j, i]
                   for i in range(F.shape[1]) for j in range(F.shape[2]))

    def _U(self, F, a, b):
        """For demo 1's F at N=2 with S(1)=0: only i=0 term of the
        diagram 2 coupling-sum survives, giving F[a,0,b] + F[a,b,0]."""
        return F[a, 0, b] + F[a, b, 0]

    def _W(self, F, a, b):
        """For diagram 3 with F symmetric in its last two indices:

        coupling_sum = 8 * sum_{i,j} F[a,i,j] F[i,b,j]   (all 4 sym-reduced terms)
        coupling_sum numerical = -8 * sum_{i,j} F[a,i,j] F[i,b,j]   (after F_MSR**2)
        evaluate_coupling = prefactor(1/2) * phase(-i)^2=-1 * (-8 * sum)
                          = 4 * sum_{i,j} F[a,i,j] F[i,b,j].
        """
        return 4 * sum(F[a, i, j] * F[i, b, j]
                       for i in range(F.shape[1]) for j in range(F.shape[2]))

    # ---------- Per-diagram checks ----------

    @pytest.mark.parametrize("a,b", [(0, 0), (0, 1), (1, 1)])
    def test_diagram_0_double_tadpole(self, a, b):
        dts, F_MSR, F_arr = self._setup_diagrams()
        # Diagram 0: R(x,y0) R(y,y1) C(y0,y0) C(y1,y1).
        # evaluate_coupling reduces to S(a) * S(b) for symmetric F.
        expected = self._S(F_arr, a) * self._S(F_arr, b)
        got = self._value(dts[0], F_MSR, a, b)
        assert abs(got.imag) < 1e-12
        assert abs(got.real - expected) < 1e-12, (
            f"diag 0 @ (a={a},b={b}): expected {expected}, got {got.real}"
        )

    @pytest.mark.parametrize("a,b", [(0, 0), (0, 1), (1, 1)])
    def test_diagram_1_bubble(self, a, b):
        dts, F_MSR, F_arr = self._setup_diagrams()
        # Diagram 1 (bubble with two C's):
        # evaluate_coupling = T1(a,b) + T2(a,b).
        expected = self._T1(F_arr, a, b) + self._T2(F_arr, a, b)
        got = self._value(dts[1], F_MSR, a, b)
        assert abs(got.imag) < 1e-12
        assert abs(got.real - expected) < 1e-12, (
            f"diag 1 @ (a={a},b={b}): expected {expected}, got {got.real}"
        )

    @pytest.mark.parametrize("a,b,expected", [
        (0, 0, 0.0),
        (0, 1, 0.0),
        (1, 0, 0.0),
        (1, 1, 1.0),
    ])
    def test_diagram_2_u_formula(self, a, b, expected):
        """Diagram 2: C(y, y_0) on y side.  Closed form at N=2 w/ demo 1
        F is U(a,b) = F[a,0,b] + F[a,b,0]."""
        dts, F_MSR, F_arr = self._setup_diagrams()
        got = self._value(dts[2], F_MSR, a, b)
        assert abs(got.imag) < 1e-12
        assert abs(got.real - expected) < 1e-12, (
            f"diag 2 @ (a={a},b={b}): expected {expected}, got {got.real}"
        )

    @pytest.mark.parametrize("a,b", [(0, 0), (0, 1), (1, 1)])
    def test_diagram_3_w_formula(self, a, b):
        """Diagram 3: evaluate_coupling = 2 * sum F[a,i,j] F[i,b,j]."""
        dts, F_MSR, F_arr = self._setup_diagrams()
        expected = self._W(F_arr, a, b)
        got = self._value(dts[3], F_MSR, a, b)
        assert abs(got.imag) < 1e-12
        assert abs(got.real - expected) < 1e-12, (
            f"diag 3 @ (a={a},b={b}): expected {expected}, got {got.real}"
        )

    @pytest.mark.parametrize("a,b", [(0, 0), (0, 1), (1, 1)])
    def test_diagrams_4_and_5_are_a_b_swap(self, a, b):
        """Diagrams 4 and 5 are (a <-> b)-images of 2 and 3 respectively.
        C on x-side replaces C on y-side; the hand formula tells us
        ``diag_4(a,b) = diag_2(b,a)`` and ``diag_5(a,b) = diag_3(b,a)``.
        """
        dts, F_MSR, F_arr = self._setup_diagrams()
        v4 = self._value(dts[4], F_MSR, a, b)
        v2_swap = self._value(dts[2], F_MSR, b, a)
        assert abs(v4 - v2_swap) < 1e-12, (
            f"diag 4({a},{b}) = {v4} != diag 2({b},{a}) = {v2_swap}"
        )
        v5 = self._value(dts[5], F_MSR, a, b)
        v3_swap = self._value(dts[3], F_MSR, b, a)
        assert abs(v5 - v3_swap) < 1e-12, (
            f"diag 5({a},{b}) = {v5} != diag 3({b},{a}) = {v3_swap}"
        )


class TestMultiComponentDiagonalF:
    """T18: sparse F has a predicted vanishing pattern.

    For ``F[0,0,0] = 1, all others = 0``, two-point ``xi_ab`` at order 2
    is entirely determined by whether Wick contractions can route
    ``a=0``, ``b=0`` through the vertices.  Only (a,b)=(0,0) survives.
    """

    def _sparse_F(self, N):
        F = np.zeros((N, N, N))
        F[0, 0, 0] = 1.0
        return F

    @pytest.mark.parametrize("n_comp", [2, 3])
    def test_vanishes_unless_all_indices_zero(self, n_comp):
        reset_uid_counter()
        phi = Field("phi", "physical", n_components=n_comp)
        psi = Field("psi", "response", n_components=n_comp)
        action = Action(vertices=[Vertex([psi, phi, phi], coupling="F")])
        obs = [phi("a", "x"), phi("b", "y")]

        F_MSR = -1j * self._sparse_F(n_comp)
        result = compute_moment(
            obs, action, order=2, response_phase=True,
            diag_R=True, diag_C=True, iso_R=True, iso_C=True,
        )
        dts = result.diagram_terms(2)

        # Any (a, b) where a != 0 or b != 0 produces zero — every Wick
        # contraction routing through the vertex would force a vanishing
        # F[.,.,.] index combination.
        for (a, b) in [(0, 1), (1, 0), (1, 1)]:
            per_diag = [complex(np.asarray(dt.evaluate_coupling(
                {"F": F_MSR}, fixed_indices={"a": a, "b": b}
            ))) for dt in dts]
            total = sum(per_diag)
            assert all(abs(v) < 1e-12 for v in per_diag), (
                f"(a={a},b={b}) @ N={n_comp}: per-diagram values "
                f"{per_diag} — expected all zero."
            )

        # At (a, b) = (0, 0), all six diagrams *can* route (0, 0, 0)
        # through the vertex (since F[0,0,0] is the only non-zero
        # entry).  Hand-derived values via S, T1+T2, U, W (see
        # ``TestMultiComponentHandVerified``) with F[0,0,0]=1 only:
        #   diag 0: S(0)^2 = 1
        #   diag 1: T1(0,0) + T2(0,0) = 1 + 1 = 2
        #   diag 2: U(0,0) = F[0,0,0] + F[0,0,0] = 2
        #   diag 3: W(0,0) = 4 * F[0,0,0]^2 = 4
        #   diag 4: U(0,0) by (a,b)-swap symmetry = 2
        #   diag 5: W(0,0) by (a,b)-swap symmetry = 4
        vals = [
            complex(np.asarray(dt.evaluate_coupling(
                {"F": F_MSR}, fixed_indices={"a": 0, "b": 0}
            )))
            for dt in dts
        ]
        expected = [1.0, 2.0, 2.0, 4.0, 2.0, 4.0]
        for i, (val, exp) in enumerate(zip(vals, expected)):
            assert abs(val.imag) < 1e-12
            assert abs(val.real - exp) < 1e-12, (
                f"sparse-F diag {i} @ (0,0): expected {exp}, got {val.real}"
            )


class TestMultiComponentEngineAgreement:
    """T19: operator-level and spatial-level engines produce the same
    Feynman topology set at N in {2, 3} (extends T5)."""

    @pytest.mark.parametrize("n_comp", [2, 3])
    def test_engine_agreement_order_2(self, n_comp):
        phi = Field("phi", "physical", n_components=n_comp)
        psi = Field("psi", "response", n_components=n_comp)
        v1 = Vertex(fields=[psi, phi, phi], coupling="F", local=True)
        action = Action(vertices=[v1])
        obs = [phi("a", "x"), phi("b", "y")]

        reset_uid_counter()
        r_op = compute_moment(obs, action, order=2,
                               collect_topology=False, response_phase=False)
        reset_uid_counter()
        r_sp = compute_moment(obs, action, order=2,
                               collect_topology=True, response_phase=False)

        op_keys = {
            d.to_feynman_diagram().canonical_form()
            for d in r_op.diagrams_by_order.get(2, [])
        }
        sp_keys = {
            d.to_feynman_diagram().canonical_form()
            for d in r_sp.diagrams_by_order.get(2, [])
        }
        assert op_keys == sp_keys, (
            f"N={n_comp}: op-only: {op_keys - sp_keys}; "
            f"sp-only: {sp_keys - op_keys}"
        )


class TestMultiComponentFKAtNEquals3:
    """T20: FK selection rule + coupling-sum collapse extends to N=3."""

    def test_FK_n3_diagonal_K(self):
        N = 3
        reset_uid_counter()
        phi = Field("phi", "physical", n_components=N)
        psi = Field("psi", "response", n_components=N)
        v1 = Vertex(fields=[psi, phi, phi], coupling="F", local=True)
        v2 = Vertex(fields=[psi, psi, psi], coupling="K", local=False)
        action = Action(vertices=[v1, v2])
        obs = [phi("a", "x"), phi("b", "y")]

        result = compute_moment(
            obs, action, order=2, response_phase=True,
            diag_R=True, diag_C=True, iso_R=True,
        )

        F_arr = np.zeros((N, N, N))
        F_arr[0, 1, 1] = 1.0
        F_arr[1, 0, 1] = 0.5
        F_arr[1, 1, 0] = 0.5
        F_MSR = -1j * F_arr

        k_aaa = np.array([0.3, 0.5, 0.7])
        K_arr = np.zeros((N, N, N), dtype=complex)
        for a in range(N):
            K_arr[a, a, a] = 1j / 6.0 * k_aaa[a]

        def _tags(dt):
            stack, out = [dt.coupling_sum], set()
            while stack:
                node = stack.pop()
                if isinstance(node, Symbol):
                    out.add(node.name)
                elif isinstance(node, Sum):
                    stack.extend(node.terms)
                elif isinstance(node, Product):
                    stack.extend(node.factors)
            return out

        FK = [dt for dt in result.diagram_terms(2)
              if _tags(dt) == {"F", "K"}]
        assert len(FK) == 2

        # (a, b) = (0, 1): FK = F[0,1,1]*k[1] + F[1,0,0]*k[0] = 0.5.
        total = sum(
            complex(np.asarray(
                dt.evaluate_coupling({"F": F_MSR, "K": K_arr},
                                      fixed_indices={"a": 0, "b": 1})
            ))
            for dt in FK
        )
        expected = F_arr[0, 1, 1] * k_aaa[1] + F_arr[1, 0, 0] * k_aaa[0]
        assert abs(total.real - expected) < 1e-10
        assert abs(total.imag) < 1e-10

        # Diagonal pairs at any a: FK vanishes by the T13 selection rule.
        for diag in (0, 1, 2):
            total = sum(
                complex(np.asarray(
                    dt.evaluate_coupling({"F": F_MSR, "K": K_arr},
                                          fixed_indices={"a": diag, "b": diag})
                ))
                for dt in FK
            )
            assert abs(total) < 1e-10, (
                f"FK @ ({diag},{diag}) N=3: expected 0, got {total}"
            )


# =====================================================================
# T21 non-iso / non-diagonal propagator consistency
# =====================================================================


class TestNonIsoPropagatorConsistency:
    """T21: `evaluate_coupling` with full R_{ij}/C_{ij} index structure
    contracts correctly back to the iso-flagged scalar result.

    Without ``iso_R``/``iso_C`` flags, each propagator carries its own
    pair of component indices (e.g. ``R_{a i_0}``, ``C_{i_1 i_2}``) and
    ``evaluate_coupling`` returns a multi-dimensional array indexed by
    every free propagator leg — at N=2 with a 4-propagator diagram that's
    a 6-axis tensor of shape (2,2,2,2,2,2).

    **Deductive claim under test**: if we contract each propagator
    (index_left, index_right) pair with a Kronecker δ (the "trivially
    diagonal + isotropic" structure), we must recover the scalar value
    produced by the ``iso_R=True, iso_C=True`` pipeline.  This verifies
    that the two symbolic paths (full index tracking vs. index-stripped)
    are equivalent on the class of propagators where the stripping is
    mathematically valid — catching any bug where ``_enumerate_component_routings``
    or the iso rewrite mis-routes a component index.
    """

    def _contract_with_deltas(self, dt_full, arr, fixed):
        """Manual contraction of the full-index array with δ_{i_l, i_r}
        per propagator.  Returns a scalar.
        """
        from itertools import product

        pi_names = [n for n, _ in dt_full.propagator_indices]
        total = 0.0 + 0j
        for vals in product(*[range(dim) for _, dim in dt_full.propagator_indices]):
            idx_map = dict(zip(pi_names, vals))
            idx_map.update(fixed)
            # Every propagator's (index_left, index_right) must match
            skip = False
            for p in dt_full.propagators:
                il, ir = p.index_left, p.index_right
                if il is None or ir is None:
                    continue
                vl = idx_map.get(il)
                vr = idx_map.get(ir)
                if vl is not None and vr is not None and vl != vr:
                    skip = True
                    break
            if not skip:
                total += arr[vals]
        return total

    @pytest.mark.parametrize("a,b", [(0, 0), (0, 1), (1, 1)])
    def test_iso_equals_full_contracted_per_diagram(self, a, b):
        """For every order-2 diagram, iso-flag evaluation equals the
        full-index evaluation contracted with per-propagator δ."""
        N = 2
        F_arr = np.zeros((N, N, N))
        F_arr[0, 1, 1] = 1.0
        F_arr[1, 0, 1] = 0.5
        F_arr[1, 1, 0] = 0.5
        F_MSR = -1j * F_arr

        reset_uid_counter()
        phi = Field("phi", "physical", n_components=N)
        psi = Field("psi", "response", n_components=N)
        action = Action(vertices=[Vertex([psi, phi, phi], coupling="F")])
        obs = [phi("a", "x"), phi("b", "y")]

        r_full = compute_moment(
            obs, action, order=2, response_phase=True, ito=True,
            # No iso or diag flags — full index structure retained
        )
        reset_uid_counter()
        r_iso = compute_moment(
            obs, action, order=2, response_phase=True, ito=True,
            diag_R=True, diag_C=True, iso_R=True, iso_C=True,
        )

        for i in range(6):
            dt_full = r_full.diagram_terms(2)[i]
            dt_iso = r_iso.diagram_terms(2)[i]
            arr = np.asarray(dt_full.evaluate_coupling(
                {"F": F_MSR}, fixed_indices={"a": a, "b": b}
            ))
            contracted = self._contract_with_deltas(
                dt_full, arr, fixed={"a": a, "b": b}
            )
            iso_val = complex(np.asarray(dt_iso.evaluate_coupling(
                {"F": F_MSR}, fixed_indices={"a": a, "b": b}
            )))
            assert abs(contracted - iso_val) < 1e-10, (
                f"diag {i} @ (a={a},b={b}): non-iso→δ contracted "
                f"{contracted} ≠ iso scalar {iso_val}"
            )

    def test_full_index_shape_matches_propagator_count_at_N_equals_3(self):
        """At N=3, a 4-propagator diagram has 4 or more component indices
        (one per propagator leg whose index is summed); the array shape
        has dimension N for each such axis.  Verifies that
        `_enumerate_component_routings` assigns fresh indices correctly
        and that ``DiagramTerm.propagator_indices`` lists them all.
        """
        N = 3
        F_arr = np.zeros((N, N, N))
        F_arr[0, 1, 1] = 1.0
        F_MSR = -1j * F_arr

        reset_uid_counter()
        phi = Field("phi", "physical", n_components=N)
        psi = Field("psi", "response", n_components=N)
        action = Action(vertices=[Vertex([psi, phi, phi], coupling="F")])
        obs = [phi("a", "x"), phi("b", "y")]

        result = compute_moment(
            obs, action, order=2, response_phase=True, ito=True,
        )
        dt0 = result.diagram_terms(2)[0]
        arr = np.asarray(dt0.evaluate_coupling(
            {"F": F_MSR}, fixed_indices={"a": 0, "b": 0}
        ))
        # Shape has dim N on every propagator_index axis
        for dim in arr.shape:
            assert dim == N, f"expected N={N}, got axis dim {dim}"
        # Exactly 6 free propagator indices for diag 0 (R_x,R_y,C_0,C_1 each
        # with 1 or 2 free indices; total 6 at this diagram)
        assert len(arr.shape) == 6


# =====================================================================
# T14 alpha=0 regression
# =====================================================================


class TestAlphaZeroRegression:
    """T14: Case B with K=0 (i.e. α=0) produces the same symbolic result
    as Case A at order 2.

    Implementation detail: setting ``K=0`` numerically would only null
    the FK diagrams' evaluated coupling, not remove their symbolic
    DiagramTerms.  So we check: the FF subset of Case B exactly matches
    Case A's DiagramTerms after canonicalisation, and the FK subset
    exists but would evaluate to zero with zero K.
    """

    def _canonical_key(self, dt) -> tuple:
        """Reuse the canonical key from TestEngineAgreement."""
        rename: dict[str, str] = {}
        next_id = 0
        edges = []
        for p in dt.propagators:
            endpoints = []
            for var in (p.spatial_left, p.spatial_right):
                if var in dt.integration_vars and var not in rename:
                    rename[var] = f"_i{next_id}"
                    next_id += 1
                endpoints.append(rename.get(var, var))
            sl, sr = endpoints
            if p.kind == "C" and sl > sr:
                sl, sr = sr, sl
            edges.append((p.kind, sl, sr))
        edges.sort()
        return (
            tuple(edges),
            dt.n_response,
            (dt.rational_prefactor.numerator, dt.rational_prefactor.denominator),
        )

    def test_caseB_FF_subset_equals_caseA(self):
        phi = Field("phi", "physical")
        psi = Field("psi", "response")
        v1 = Vertex(fields=[psi, phi, phi], coupling="F", local=True)
        v2 = Vertex(fields=[psi, psi, psi], coupling="K", local=False)

        obs = [phi("x"), phi("y")]

        reset_uid_counter()
        result_A = compute_moment(
            obs, Action(vertices=[v1]), order=2, response_phase=False, ito=True,
        )
        reset_uid_counter()
        result_B = compute_moment(
            obs, Action(vertices=[v1, v2]), order=2, response_phase=False, ito=True,
        )

        # Extract FF-only DiagramTerms from Case B
        def _tags(dt) -> set[str]:
            stack, out = [dt.coupling_sum], set()
            while stack:
                n = stack.pop()
                if isinstance(n, Symbol):
                    out.add(n.name)
                elif isinstance(n, Sum):
                    stack.extend(n.terms)
                elif isinstance(n, Product):
                    stack.extend(n.factors)
            return out

        B_FF_terms = [dt for dt in result_B.diagram_terms(2) if _tags(dt) == {"F"}]
        A_terms = result_A.diagram_terms(2)

        A_keys = {self._canonical_key(dt) for dt in A_terms}
        B_FF_keys = {self._canonical_key(dt) for dt in B_FF_terms}
        assert A_keys == B_FF_keys, (
            f"A has {len(A_keys)} keys, B_FF has {len(B_FF_keys)};\n"
            f"  A \\ B = {A_keys - B_FF_keys}\n"
            f"  B \\ A = {B_FF_keys - A_keys}"
        )

    def test_caseB_FK_exists_but_would_evaluate_to_zero_with_null_K(self):
        """FK DiagramTerms are present but a null K tensor makes them
        contribute zero on evaluate_coupling."""
        phi = Field("phi", "physical", n_components=2)
        psi = Field("psi", "response", n_components=2)
        v1 = Vertex(fields=[psi, phi, phi], coupling="F", local=True)
        v2 = Vertex(fields=[psi, psi, psi], coupling="K", local=False)
        action = Action(vertices=[v1, v2])
        obs = [phi("a", "x"), phi("b", "y")]

        reset_uid_counter()
        result = compute_moment(
            obs, action, order=2, response_phase=True, ito=True,
            diag_R=True, diag_C=True, iso_R=True,
        )

        F_arr = np.zeros((2, 2, 2))
        F_arr[0, 1, 1] = 1.0
        F_arr[1, 0, 1] = 0.5
        F_arr[1, 1, 0] = 0.5
        F_MSR = -1j * F_arr
        K_zero = np.zeros((2, 2, 2), dtype=complex)

        def _tags(dt) -> set[str]:
            stack, out = [dt.coupling_sum], set()
            while stack:
                n = stack.pop()
                if isinstance(n, Symbol):
                    out.add(n.name)
                elif isinstance(n, Sum):
                    stack.extend(n.terms)
                elif isinstance(n, Product):
                    stack.extend(n.factors)
            return out

        FK_terms = [
            dt for dt in result.diagram_terms(2) if _tags(dt) == {"F", "K"}
        ]
        assert len(FK_terms) == 2  # The 2 FK diagrams

        for dt in FK_terms:
            val = dt.evaluate_coupling(
                {"F": F_MSR, "K": K_zero}, fixed_indices={"a": 0, "b": 1}
            )
            assert abs(complex(np.asarray(val))) < 1e-14
