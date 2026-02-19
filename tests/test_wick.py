"""Tests for the core Wick contraction engine."""

from sft_wick import Field, Propagator, Product, Sum, reset_uid_counter
from sft_wick.wick import generate_all_pairings, generate_valid_pairings, wick_contract
from sft_wick._util import double_factorial
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


def test_wick_scalar_psi_phi_phi_phi():
    """<psi(x) phi(x) phi(x) phi(x)>_0 = 3 R(x,x) C(x,x).

    This is the user's example. 4 operators with 1 psi and 3 phi.
    The psi must pair with one of the 3 phi's (R), the other 2 phi's pair (C).
    There are 3 choices -> 3 identical terms -> 3 R(x,x) C(x,x).
    """
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    ops = [psi("x"), phi("x"), phi("x"), phi("x")]
    result, pairings = wick_contract(ops)
    assert len(pairings) == 3

    # All 3 pairings produce the same propagator structure: R(x,x) * C(x,x)
    # (since all fields are at the same point and scalar)
    assert isinstance(result, Sum)
    assert len(result.terms) == 3

    # Each term should contain R(x,x) and C(x,x)
    for term in result.terms:
        assert isinstance(term, Product)
        kinds = set()
        for factor in term.factors:
            assert isinstance(factor, Propagator)
            kinds.add(factor.kind)
        assert kinds == {"C", "R"}


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

    valid = {_normalize_pairing(p) for p in generate_valid_pairings(phi_idx, psi_idx)}

    # Compare with brute force
    from sft_wick.wick import evaluate_pairing
    all_p = list(generate_all_pairings(list(range(len(ops)))))
    brute_force_valid = set()
    for p in all_p:
        if evaluate_pairing(ops, p) is not None:
            brute_force_valid.add(_normalize_pairing(p))

    assert valid == brute_force_valid
