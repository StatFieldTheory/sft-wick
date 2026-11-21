"""Synthetic value checks for the public Wick expansion entry points."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

from sft_wick import Action, CertificateError, Field, Vertex, compute_moment, compute_moment_numerical
from sft_wick import perturbation
from sft_wick._value_certificate import ValueFixture


def _cubic(width):
    phi = Field("phi", "physical", width)
    psi = Field("psi", "response", width)
    return [phi("1", "x"), phi("2", "y")], Action([Vertex([psi, phi, phi], "F")])


@pytest.mark.parametrize("engine,width,target", [
    (compute_moment, 3, "connected"),
    (compute_moment_numerical, 2, "moment"),
])
def test_VC1_dense_cubic_terms_match_independent_pairings(engine, width, target):
    obs, action = _cubic(width)
    certificate = engine(obs, action, 2, certify=True, target=target).certificate
    row, = certificate.order(2).sequences
    probes = ("roles", "points") if engine is compute_moment_numerical else ("roles",)
    assert tuple(check.probe for check in row.value_checks) == probes
    assert all(check.passed for check in row.value_checks)
    assert all(check.reference != (0.0, 0.0) for check in row.value_checks)
    assert not row.value_skip
    json.dumps(asdict(certificate), allow_nan=False)


def test_VC2_preserving_the_pairing_count_can_change_the_value(monkeypatch):
    obs, action = _cubic(2)
    routing = perturbation._enumerate_component_routings

    def repeat_representative(*args, **kwargs):
        records = routing(*args, **kwargs)
        if len(records) < 2:
            return records
        representative, _ = records[0]
        return [(representative, pairing) for _, pairing in records]

    monkeypatch.setattr(perturbation, "_enumerate_component_routings", repeat_representative)
    with pytest.raises(CertificateError, match="value certificate") as caught:
        compute_moment_numerical(obs, action, 2, certify=True, canonical="builtin")
    error = caught.value
    row, = error.certificate.order(2).sequences
    assert row.term_weight == row.counter == 30
    assert error.failures and all("value certificate" in message for message in error.failures)
    assert any(not check.passed for check in row.value_checks)


@pytest.mark.heavy
def test_VC3_fourth_order_cubic_fits_the_pairing_budget():
    obs, action = _cubic(2)
    certificate = compute_moment_numerical(
        obs, action, 4, certify=True, canonical="builtin"
    ).certificate
    for order in range(5):
        row, = certificate.order(order).sequences
        assert not row.value_skip
        assert row.value_checks and all(check.passed for check in row.value_checks)
    assert certificate.order(4).pairings == 15120


def test_VC4_value_check_exposes_count_only_rows():
    obs, action = _cubic(2)
    cert = compute_moment(obs, action, 2, certify=True, collect_topology=False).certificate
    zero, = cert.order(0).sequences
    two, = cert.order(2).sequences
    assert zero.value_checks
    assert two.counter == 30 and not two.value_checks
    assert "no numeric terms" in two.value_skip

    phi = Field("p", "physical")
    psi = Field("s", "response")
    odd_obs = [phi("x")]
    action = Action([Vertex([phi, psi], "M")])
    cert = compute_moment_numerical(odd_obs, action, 7, certify=True).certificate
    row, = cert.order(7).sequences
    assert row.counter == 0 and not row.value_checks
    assert "more than 14 operators" in row.value_skip

    response = Field("s", "response", 2)
    absorbed = Vertex([response] * 2, "K", local=False, already_R_contracted=True)
    cert = compute_moment([phi("x"), phi("y")], Action([absorbed]), 1, certify=True).certificate
    row, = cert.order(1).sequences
    assert row.counter == 2 and not row.value_checks
    assert "unequal component dimensions" in row.value_skip

    # An index-free C endpoint is not component zero of a vector matrix.
    vector = Field("v", "physical", 2)
    cert = compute_moment([phi("x"), vector("2", "y")], Action([]), 0,
                          certify=True).certificate
    row, = cert.order(0).sequences
    assert row.counter == 1 and not row.value_checks
    assert "scalar and vector" in row.value_skip


def test_VC5_response_phase_and_literal_matrix_entry(monkeypatch):
    # An asymmetric complex response with a wide range and nearly proportional
    # rows keeps the phase and the endpoint orientation independently observable.
    matrix = np.array([[1e-6, 1e-6 + 2e-6j, 3e-6],
                       [1e6, 1e6 + 2e6j + 1e-5, 3e6],
                       [2, 5j, 9]])
    original = ValueFixture.matrix

    def fixed_response(self, probe, kind, a, b, absorbed=False, roles=None):
        if kind == "R" and not absorbed:
            return matrix
        return original(self, probe, kind, a, b, absorbed, roles)

    monkeypatch.setattr(ValueFixture, "matrix", fixed_response)
    phi = Field("phi", "physical", 3)
    psi = Field("psi", "response", 3)
    cert = compute_moment([psi("2", "a"), phi("1", "b")], Action([]), 0,
                          certify=True, response_phase=False).certificate
    row, = cert.order(0).sequences
    check, = row.value_checks
    expected = -1j * matrix[0, 1]
    assert check.reference == check.emitted == (expected.real, expected.imag)


@pytest.mark.parametrize("engine", [compute_moment, compute_moment_numerical])
def test_VC6_mixed_role_sequence_with_absorbed_response(engine):
    obs, cubic = _cubic(2)
    psi = Field("psi", "response", 2)
    absorbed = Vertex([psi] * 3, "K", local=False, already_R_contracted=True)
    cert = engine(obs, Action([cubic.vertices[0], absorbed]), 2,
                  certify=True, target="connected").certificate
    row, = [row for row in cert.order(2).sequences if row.vertices == ("F", "K")]
    assert row.value_checks and not row.value_skip
    assert all(check.passed and check.reference != (0.0, 0.0) for check in row.value_checks)


def test_VC7_nonfinite_emitted_value_fails_against_zero(monkeypatch):
    monkeypatch.setattr(ValueFixture, "emitted", lambda *args: complex(float("nan")))
    phi = Field("p", "physical")
    with pytest.raises(CertificateError, match="value certificate"):
        compute_moment([phi("x")], Action([]), 0, certify=True)


def test_VC8_missing_absorbed_component_pins_change_the_value(monkeypatch):
    obs, _ = _cubic(2)
    psi = Field("psi", "response", 2)
    vertex = Vertex([psi] * 2, "K", local=False, already_R_contracted=True)
    monkeypatch.setattr(perturbation, "_pin_absorbed_leg_indices", lambda term: term)
    with pytest.raises(CertificateError, match="value certificate") as caught:
        compute_moment_numerical(obs, Action([vertex]), 1, certify=True)
    row, = caught.value.certificate.order(1).sequences
    assert row.counter == row.term_weight == 2
    assert any(not check.passed for check in row.value_checks)
