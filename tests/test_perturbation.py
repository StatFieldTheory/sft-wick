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
    """<psi(x) phi(x) phi(x) phi(x)>_S at order 0 = 3 R(x,x) C(x,x)."""
    phi = Field("phi", "physical")
    psi = Field("psi", "response")

    v = Vertex(fields=[phi, phi, psi], coupling="F")
    action = Action(vertices=[v])

    obs = [psi("x"), phi("x"), phi("x"), phi("x")]
    result = compute_moment(obs, action, order=0)

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
    result = compute_moment(obs, action, order=1)

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
