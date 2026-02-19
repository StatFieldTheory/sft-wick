"""Tests for propagator contraction rules."""

from sft_wick import Field, contract_pair, reset_uid_counter
import pytest


@pytest.fixture(autouse=True)
def _reset_uid():
    reset_uid_counter()


def test_phi_phi_gives_C():
    phi = Field("phi", "physical", n_components=3)
    op1 = phi("a", "x")
    op2 = phi("b", "y")
    prop = contract_pair(op1, op2)
    assert prop is not None
    assert prop.kind == "C"
    assert prop.index_left == "a"
    assert prop.index_right == "b"
    assert prop.spatial_left == "x"
    assert prop.spatial_right == "y"


def test_phi_psi_gives_R():
    phi = Field("phi", "physical", n_components=3)
    psi = Field("psi", "response", n_components=3)
    op1 = phi("a", "x")
    op2 = psi("b", "y")
    prop = contract_pair(op1, op2)
    assert prop is not None
    assert prop.kind == "R"
    assert prop.index_left == "a"
    assert prop.index_right == "b"
    assert prop.spatial_left == "x"
    assert prop.spatial_right == "y"


def test_psi_phi_gives_R_canonical():
    """<psi_j(y) phi_i(x)> = R_{ij}(x, y) -- phi first in canonical form."""
    phi = Field("phi", "physical", n_components=3)
    psi = Field("psi", "response", n_components=3)
    op1 = psi("j", "y")
    op2 = phi("i", "x")
    prop = contract_pair(op1, op2)
    assert prop is not None
    assert prop.kind == "R"
    assert prop.index_left == "i"
    assert prop.index_right == "j"
    assert prop.spatial_left == "x"
    assert prop.spatial_right == "y"


def test_psi_psi_vanishes():
    psi = Field("psi", "response", n_components=3)
    op1 = psi("a", "x")
    op2 = psi("b", "y")
    assert contract_pair(op1, op2) is None


def test_scalar_phi_phi():
    phi = Field("phi", "physical")
    op1 = phi("x")
    op2 = phi("y")
    prop = contract_pair(op1, op2)
    assert prop is not None
    assert prop.kind == "C"
    assert prop.index_left is None
    assert prop.index_right is None
