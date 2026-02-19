"""Tests for fields module."""

from sft_wick import Field, FieldOperator, FieldType, reset_uid_counter
import pytest


@pytest.fixture(autouse=True)
def _reset_uid():
    reset_uid_counter()


def test_field_creation_with_enum():
    phi = Field("phi", FieldType.PHYSICAL, n_components=3)
    assert phi.name == "phi"
    assert phi.field_type == FieldType.PHYSICAL
    assert phi.n_components == 3
    assert not phi.is_scalar


def test_field_creation_with_string():
    psi = Field("psi", "response")
    assert psi.field_type == FieldType.RESPONSE
    assert psi.is_scalar
    assert psi.n_components == 1


def test_scalar_field_call():
    phi = Field("phi", "physical")
    op = phi("x")
    assert op.field is phi
    assert op.component_index is None
    assert op.spatial_arg == "x"
    assert op.is_physical


def test_multicomponent_field_call():
    phi = Field("phi", "physical", n_components=3)
    op = phi("a", "x")
    assert op.component_index == "a"
    assert op.spatial_arg == "x"


def test_scalar_field_rejects_two_args():
    phi = Field("phi", "physical")
    with pytest.raises(ValueError):
        phi("a", "x")


def test_multicomponent_field_rejects_one_arg():
    phi = Field("phi", "physical", n_components=3)
    with pytest.raises(ValueError):
        phi("x")


def test_uid_uniqueness():
    phi = Field("phi", "physical")
    op1 = phi("x")
    op2 = phi("x")
    assert op1.uid != op2.uid


def test_field_operator_repr():
    phi = Field("phi", "physical", n_components=3)
    op = phi("a", "x")
    assert repr(op) == "phi_a(x)"

    psi = Field("psi", "response")
    op2 = psi("y")
    assert repr(op2) == "psi(y)"
