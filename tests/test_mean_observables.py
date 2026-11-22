"""Reconstruction of original field moments from centered Wick diagrams."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from sft_wick import ExplicitR, FieldSpec, GaussianNoise, System
from sft_wick.action import Action
from sft_wick.evaluate import integrate_diagrams
from sft_wick.vertices import Vertex
from sft_wick.workflow._mean_observables import expand_with_mean, observable_subsets
from sft_wick.workflow.specs import (
    DiagonalA,
    ExponentialSpatial,
    ExponentialTemporal,
    SeparableTranslation,
)


START, LEFT, RIGHT = 0.3, 0.54, 1.2
RATES = np.array([0.7, 0.8])
MEAN = np.array([0.32, -0.41])
SOURCE = np.array([-0.13, 0.22])


def _system():
    return System(
        FieldSpec("phi", 2),
        DiagonalA(gamma=RATES),
        GaussianNoise(SeparableTranslation(
            ExponentialTemporal(0.0, 1.0), ExponentialSpatial(1.0),
        )),
        t_min=START,
    )


def _mean_coupling(_positions, times):
    return 1j * np.exp(-(times[0, :, None] - START) * RATES) * MEAN


_mean_coupling.vectorized = True


def _evaluate(result, cache, order, *, left=LEFT, integrate=False):
    pins = {"y_0": left}
    if not integrate:
        pins["z"] = RIGHT
    return integrate_diagrams(
        result.diagram_terms(order),
        {"F": -1j * SOURCE, "mean": _mean_coupling},
        cache=cache,
        lambda_f=RIGHT,
        t_min=START,
        method="gauss_legendre",
        n_gauss=24,
        positions={"y_0": 0.0, "z": 0.2},
        external_times=pins,
        integrate_over=["z"] if integrate else None,
        fixed_indices={"i_0": 0, "b": 1},
    )[0]


@pytest.mark.parametrize("routing", ["enumerate", "multiplicity"])
def test_external_mean_recovers_unequal_and_integrated_times_with_dummy_collisions(routing):
    system = _system()
    phi, psi = system.build_fields()
    observable = [phi("i_0", "y_0"), phi("b", "z")]
    action = Action([Vertex((psi,), "F")])
    shifted = expand_with_mean(
        observable, action, 2, mean_symbol="mean", diag_R=True, diag_C=True,
        routing=routing,
    )
    cache = system.propagators(RIGHT).cache

    def reference(left, integrate):
        a = np.exp(-RATES[0] * (left - START))
        b = np.exp(-RATES[1] * (RIGHT - START))
        first_mean = MEAN[0] * a
        first_source = SOURCE[0] * (1 - a) / RATES[0]
        if integrate:
            area = (1 - b) / RATES[1]
            second_mean = MEAN[1] * area
            second_source = SOURCE[1] * (RIGHT - START - area) / RATES[1]
        else:
            second_mean = MEAN[1] * b
            second_source = SOURCE[1] * (1 - b) / RATES[1]
        return (
            first_mean * second_mean,
            first_mean * second_source + first_source * second_mean,
            first_source * second_source,
        )

    for left in (START, LEFT):
        for integrate in (False, True):
            actual = [
                _evaluate(shifted, cache, order, left=left, integrate=integrate)
                for order in range(3)
            ]
            np.testing.assert_allclose(actual, reference(left, integrate), rtol=0, atol=1e-15)
    assert all(
        "y_0" not in dt.integration_vars and "i_0" not in dict(dt.summation_indices)
        for order in range(3)
        for dt in shifted.diagram_terms(order)
    )
    assert all(
        len(info.to_feynman_diagram().external_nodes) == 2
        for infos in shifted.diagrams_by_order.values()
        for info in infos
    )
    assert "mean" in shifted.order(0).to_latex()


def test_connected_mean_only_translates_a_single_physical_insertion():
    system = _system()
    phi, psi = system.build_fields()
    one = phi("i_0", "y_0")
    two = phi("b", "z")
    action = Action([])
    connected = expand_with_mean(
        [one, two], action, 0, target="connected", mean_symbol="mean",
        diag_R=True, diag_C=True, certify=True,
    )
    assert connected.certificate is not None
    assert _evaluate(connected, system.propagators(RIGHT).cache, 0) == 0.0

    single = expand_with_mean([one], action, 0, target="connected", mean_symbol="mean")
    assert len(single.diagram_terms(0)) == 1
    assert single.diagram_terms(0)[0].analyze_spatial().external_points == ("y_0",)

    pair = (one, psi("b", "z"))
    assert list(observable_subsets(pair, "moment")) == [(pair, ()), ((pair[1],), (one,))]
    assert list(observable_subsets(pair, "connected")) == [(pair, ())]
    with pytest.raises(ValueError, match="cannot certify reconstructed mean factors"):
        expand_with_mean([one], action, 0, target="connected", mean_symbol="mean", certify=True)


def test_external_mean_preserves_response_phase_with_dense_response():
    drift = np.array([[-0.8, 0.43], [-0.71, -1.3]])

    def response(late, early):
        lag = np.asarray(late) - np.asarray(early)
        return expm(drift * lag[..., None, None])

    def mean(_positions, times):
        return 1j * (response(times[0], START) @ MEAN)

    mean.vectorized = True
    system = System(
        FieldSpec("phi", 2), ExplicitR(response, iso_R=False, vectorized=True),
        _system().noise, t_min=START,
    )
    phi, psi = system.build_fields()
    result = expand_with_mean(
        [phi("a", "x"), phi("b", "y"), psi("c", "z")], Action([]), 0,
        mean_symbol="mean", diag_R=False, diag_C=False,
    )
    times = {"x": LEFT, "y": RIGHT, "z": 0.42}
    terms = result.diagram_terms(0)
    expected = (
        (response(LEFT, START) @ MEAN)[0] * response(RIGHT, times["z"])[1, 0]
        + (response(RIGHT, START) @ MEAN)[1] * response(LEFT, times["z"])[0, 0]
    )
    actual, _details = integrate_diagrams(
        terms, {"mean": mean}, lambda_f=RIGHT,
        cache=system.propagators(RIGHT, diag_C=False).cache, t_min=START,
        positions={"x": 0.0, "y": 0.0, "z": 0.0},
        external_times=times, fixed_indices={"a": 0, "b": 1, "c": 0},
        method="gauss_legendre",
    )
    assert actual == pytest.approx(expected, abs=1e-15, rel=0)
