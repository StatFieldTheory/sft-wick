"""Tests for the core Wick contraction engine."""

import sys

from sft_wick import Field, Propagator, Sum, reset_uid_counter
from sft_wick.perturbation import _enumerate_component_routings
from sft_wick import wick
from sft_wick.wick import (
    r_class_pairings,
    evaluate_pairing,
    generate_all_pairings,
    generate_valid_pairings,
    r_vanishes,
    wick_contract,
    wick_contract_spatial,
)
from sft_wick._util import double_factorial
from sft_wick.propagators import contract_pair
import pytest


@pytest.fixture(autouse=True)
def _reset_uid():
    reset_uid_counter()


# --- Pairing count tests ---

def test_pairing_count_2():
    pairings = list(generate_all_pairings([0, 1]))
    assert len(pairings) == 1  # (2-1)!! = 1


def test_pairing_count_4():
    pairings = list(generate_all_pairings([0, 1, 2, 3]))
    assert len(pairings) == 3  # 3!! = 3


def test_pairing_count_6():
    pairings = list(generate_all_pairings([0, 1, 2, 3, 4, 5]))
    assert len(pairings) == 15  # 5!! = 15


def test_pairing_count_8():
    pairings = list(generate_all_pairings(list(range(8))))
    assert len(pairings) == double_factorial(7)  # 7!! = 105


def test_pairing_odd_returns_empty():
    pairings = list(generate_all_pairings([0, 1, 2]))
    assert len(pairings) == 0


def test_pairing_empty():
    pairings = list(generate_all_pairings([]))
    assert len(pairings) == 1  # one empty pairing


# --- Wick contraction tests ---

def test_wick_two_phi():
    """<phi_a(x) phi_b(y)>_0 = C_{ab}(x, y)."""
    phi = Field("phi", "physical", n_components=3)
    ops = [phi("a", "x"), phi("b", "y")]
    result, pairings = wick_contract(ops)
    assert isinstance(result, Propagator)
    assert result.kind == "C"
    assert result.index_left == "a"
    assert result.index_right == "b"
    assert len(pairings) == 1


def test_wick_phi_psi():
    """<phi_a(x) psi_b(y)>_0 = R_{ab}(x, y)."""
    phi = Field("phi", "physical", n_components=3)
    psi = Field("psi", "response", n_components=3)
    ops = [phi("a", "x"), psi("b", "y")]
    result, pairings = wick_contract(ops)
    assert isinstance(result, Propagator)
    assert result.kind == "R"
    assert len(pairings) == 1


def test_wick_two_psi_vanishes():
    """<psi_a(x) psi_b(y)>_0 = 0."""
    psi = Field("psi", "response", n_components=3)
    ops = [psi("a", "x"), psi("b", "y")]
    result, pairings = wick_contract(ops)
    assert result.to_latex() == "0"
    assert len(pairings) == 0


def test_wick_odd_vanishes():
    """<phi phi phi>_0 = 0 (odd number)."""
    phi = Field("phi", "physical")
    ops = [phi("x"), phi("y"), phi("z")]
    result, pairings = wick_contract(ops)
    assert result.to_latex() == "0"


def test_wick_four_phi():
    """<phi_a(x) phi_b(y) phi_c(z) phi_d(w)>_0 = 3 terms (one for each pairing)."""
    phi = Field("phi", "physical", n_components=3)
    ops = [phi("a", "x"), phi("b", "y"), phi("c", "z"), phi("d", "w")]
    result, pairings = wick_contract(ops)
    assert len(pairings) == 3
    # Result should be a Sum of 3 Products
    assert isinstance(result, Sum)
    assert len(result.terms) == 3


def test_wick_mixed_phi_psi_four():
    """<phi_a(x) psi_b(y) phi_c(z) psi_d(w)>_0.

    Pairings: (0,1)(2,3): R*R -- valid
              (0,2)(1,3): C * <psi psi> = 0
              (0,3)(1,2): R * R -- valid (swap order)
    Result: 2 surviving terms.
    """
    phi = Field("phi", "physical", n_components=3)
    psi = Field("psi", "response", n_components=3)
    ops = [phi("a", "x"), psi("b", "y"), phi("c", "z"), psi("d", "w")]
    result, pairings = wick_contract(ops)
    assert len(pairings) == 2


def test_wick_three_psi_one_phi_vanishes():
    """<psi psi psi phi> = 0 (not enough phi's to absorb all psi's)."""
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    ops = [psi("x"), psi("y"), psi("z"), phi("w")]
    result, pairings = wick_contract(ops)
    assert result.to_latex() == "0"
    assert len(pairings) == 0


def _normalize_pairing(pairing):
    """Normalize a pairing for comparison: sort each pair, then sort pairs."""
    return tuple(sorted(tuple(sorted(p)) for p in pairing))


def test_valid_pairings_equals_filtered_all():
    """generate_valid_pairings produces same results as filtering generate_all_pairings."""
    phi = Field("phi", "physical", n_components=3)
    psi = Field("psi", "response", n_components=3)
    ops = [phi("a", "x"), psi("b", "y"), phi("c", "z"), phi("d", "w")]

    phi_idx = [i for i, op in enumerate(ops) if op.is_physical]
    psi_idx = [i for i, op in enumerate(ops) if op.is_response]

    valid = {_normalize_pairing(p) for p in generate_valid_pairings(phi_idx, psi_idx, ops)}

    # Compare with brute force
    from sft_wick.wick import evaluate_pairing
    all_p = list(generate_all_pairings(list(range(len(ops)))))
    brute_force_valid = set()
    for p in all_p:
        if evaluate_pairing(ops, p) is not None:
            brute_force_valid.add(_normalize_pairing(p))

    assert valid == brute_force_valid


# --- Itô prescription tests ---


def test_wick_ito_all_equal_point_psi_phi_vanishes():
    """<psi(x) phi(x) phi(x) phi(x)>_0 = 0 under Itô.

    All operators are at 'x', so every R(x,x) vanishes, killing all pairings.
    """
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    ops = [psi("x"), phi("x"), phi("x"), phi("x")]
    result, pairings = wick_contract(ops)
    assert result.to_latex() == "0"
    assert len(pairings) == 0


def test_wick_ito_different_points_survives():
    """<phi(x) psi(y)>_0 = R(x,y) survives under Itô when x != y."""
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    ops = [phi("x"), psi("y")]
    result, pairings = wick_contract(ops)
    assert isinstance(result, Propagator)
    assert result.kind == "R"
    assert len(pairings) == 1


def test_wick_pure_phi_at_one_point_is_one_C():
    """<phi(x) phi(x)>_0 = C(x,x), with no R propagator."""
    phi = Field("phi", "physical")
    ops = [phi("x"), phi("x")]
    result, pairings = wick_contract(ops)
    assert isinstance(result, Propagator)
    assert result.kind == "C"
    assert len(pairings) == 1


def test_wick_ito_mixed_points_partial_elimination():
    """<psi(x) phi(x) phi(y) phi(y)>_0 under Itô.

    psi(x) paired with phi(x) -> R(x,x) = 0  (eliminated)
    psi(x) paired with phi(y) -> R(y,x) != 0  (2 choices, each with C(x,y) or C(y,y))
    So 2 pairings survive.
    """
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    ops = [psi("x"), phi("x"), phi("y"), phi("y")]

    # psi(x)-phi(x) pairs are killed
    _, pairings = wick_contract(ops)
    assert len(pairings) == 2


# --- Causal R-loop vanishing tests ---


def test_causal_r_loop_2_cycle():
    """R(x,y) R(y,x) = 0 by causality (Theta(t) Theta(-t) = 0).

    <phi(x) psi(y) phi(y) psi(x)> has pairings including one where
    phi(x)-psi(y) → R(x,y) and phi(y)-psi(x) → R(y,x), which should vanish.
    """
    from sft_wick.wick import evaluate_pairing
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    ops = [phi("x"), psi("y"), phi("y"), psi("x")]

    # Pairing (0,1)(2,3): phi(x)-psi(y)→R(x,y), phi(y)-psi(x)→R(y,x) → 2-cycle!
    result = evaluate_pairing(ops, ((0, 1), (2, 3)))
    assert result is None  # killed by causal vanishing

    # Pairing (0,3)(2,1): phi(x)-psi(x)→R(x,x)=0 by Itô
    result2 = evaluate_pairing(ops, ((0, 3), (2, 1)))
    assert result2 is None  # killed by Itô equal-point rule


def test_causal_c_propagators_ignored():
    """C propagators don't participate in R-cycle detection."""
    phi = Field("phi", "physical")
    ops = [phi("x"), phi("y"), phi("y"), phi("x")]
    assert evaluate_pairing(ops, ((0, 1), (2, 3))) is not None


@pytest.mark.parametrize("edges, vanishes", [
    ([("x", "x")], True),
    ([("x", "y"), ("y", "x")], True),
    ([("x", "y"), ("y", "z"), ("z", "x")], True),
    ([("x", "y"), ("y", "y")], True),
    ([("x", "y"), ("x", "y")], False),
    ([("x", "y"), ("y", "z")], False),
    ([], False),
])
def test_r_vanishes_at_an_equal_point_or_on_a_loop(edges, vanishes):
    assert r_vanishes(edges) is vanishes


def test_generate_valid_pairings_checks_the_R_of_a_pairing_when_it_reaches_it(
    monkeypatch,
):
    """Eight phi and eight psi have 8! assignments.  A caller that stops at
    the first pairing pays for one branch of them."""
    phi, psi = Field("phi", "physical"), Field("psi", "response")
    ops = [op for i in range(8) for op in (phi(f"p{i}"), psi(f"q{i}"))]
    calls = []
    vanishes = wick.r_vanishes

    def counted(edges):
        calls.append(1)
        return vanishes(edges)

    monkeypatch.setattr(wick, "r_vanishes", counted)
    pairings = generate_valid_pairings(
        list(range(0, 16, 2)), list(range(1, 16, 2)), ops
    )
    assert next(pairings) == tuple((i, i + 1) for i in range(0, 16, 2))
    assert len(calls) <= 16  # of 8! assignments


REPEATED_LABELS = [
    # (fields, labels): labels unsorted and repeated, on both sides.
    ("frpfrf", "bbaabb"),
    ("ffrrffrf", "cabbacab"),
    ("rrffffff", "bbaabbcc"),
    ("frfrffff", "abbcabca"),
]


def _operators(fields, labels):
    phi, psi = Field("phi", "physical"), Field("psi", "response")
    return [(psi if kind == "r" else phi)(s) for kind, s in zip(fields, labels)]


@pytest.mark.parametrize("fields, labels", REPEATED_LABELS)
def test_every_valid_pairing_comes_once_with_repeated_labels(fields, labels):
    ops = _operators(fields, labels)
    phi_idx = [i for i, op in enumerate(ops) if op.is_physical]
    psi_idx = [i for i, op in enumerate(ops) if op.is_response]
    valid = [_normalize_pairing(p)
             for p in generate_valid_pairings(phi_idx, psi_idx, ops)]
    expected = {_normalize_pairing(p)
                for p in generate_all_pairings(list(range(len(ops))))
                if evaluate_pairing(ops, p) is not None}
    assert expected
    assert len(valid) == len(set(valid))
    assert set(valid) == expected


@pytest.mark.parametrize("fields, labels", [
    ("ff", (1, "x")),
    ("ff", (None, "x")),
    ("frfr", (2, 1, "x", 1)),
])
def test_labels_that_do_not_compare_still_contract(fields, labels):
    """``wick_contract`` reads labels as keys, so it takes what a dict takes.
    The last case repeats a psi label, which reaches the ordering of a run."""
    ops = _operators(fields, labels)
    _, pairings = wick_contract(ops)
    expected = {_normalize_pairing(p)
                for p in generate_all_pairings(list(range(len(ops))))
                if evaluate_pairing(ops, p) is not None}
    assert expected
    assert {_normalize_pairing(p) for p in pairings} == expected


@pytest.mark.parametrize("phi_label, psi_label", [
    (lambda i: f"p{i}", lambda i: f"q{i}"),
    (lambda i: "a", lambda i: "b"),
], ids=["distinct", "two_labels"])
def test_the_first_pairing_does_not_depend_on_the_recursion_limit(
    phi_label, psi_label
):
    n = sys.getrecursionlimit() + 100
    phi, psi = Field("phi", "physical"), Field("psi", "response")
    ops = [phi(phi_label(i)) for i in range(n)]
    ops += [psi(psi_label(i)) for i in range(n)]
    pairing = next(
        generate_valid_pairings(list(range(n)), list(range(n, 2 * n)), ops)
    )
    assert sorted(j for _, j in pairing) == list(range(n, 2 * n))
    # The search of a cycle may start anywhere on the chain.
    assert not r_vanishes((f"n{i}", f"n{i + 1}") for i in range(10 * n))


@pytest.mark.parametrize("points", [("x", "y"), ("x", "y", "z")])
def test_both_engines_keep_no_R_term_of_a_permutation(points):
    """One phi and one psi at each of ``points``.  Each R assignment is a
    permutation of the points, so an equal point or a causal loop closes in
    all of them."""
    phi, psi = Field("phi", "physical"), Field("psi", "response")
    ops = [op for p in points for op in (phi(p), psi(p))]
    assert wick_contract(ops)[1] == []
    assert wick_contract_spatial(ops) == {}


def test_a_topology_reached_by_two_C_enumerations_is_one_topology():
    """x carries three phi and y one: the C pairing (x,x)(x,y) is enumerated
    twice, and the routing count, not the enumeration, gives its 3 pairings."""
    phi = Field("phi", "physical")
    ops = [phi("x"), phi("x"), phi("x"), phi("y")]
    spatial = wick_contract_spatial(ops)
    assert len(spatial) == 1
    ((props, pairing),) = spatial.values()
    routings = _enumerate_component_routings(props, pairing, ops, frozenset("xy"))
    assert len(routings) == len(wick_contract(ops)[1]) == 3


@pytest.mark.parametrize("fields, points", [
    ("pr", "xy"),
    ("prpr", "xyzw"),
    ("prpr", "xyyx"),
    ("pppr", "xxyz"),
    ("pprrpp", "xyzwuv"),
    ("pprrpp", "xxyyzz"),
])
def test_r_class_pairings_hold_every_R_of_the_whole_contraction(fields, points):
    """The R parts of the contraction, without its C pairings."""
    field = {"p": Field("phi", "physical"), "r": Field("psi", "response")}
    ops = [field[f](s) for f, s in zip(fields, points)]

    def r_edges(propagators):
        return tuple(sorted(
            (p.spatial_left, p.spatial_right) for p in propagators if p.kind == "R"
        ))

    from_whole = {
        r_edges(contract_pair(ops[i], ops[j]) for i, j in pairing)
        for pairing in wick_contract(ops)[1]
    }
    assert from_whole == {r_edges(props) for props, _ in r_class_pairings(ops)}
