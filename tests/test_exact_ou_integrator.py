"""Ordered exponential integrals and normal-path analytic OU integration."""

from math import exp, expm1, factorial, inf

import numpy as np
import pytest

from sft_wick._exp_sum import simplex
from sft_wick.workflow import (
    DiagonalA,
    ExponentialSpatial,
    ExponentialTemporal,
    FieldSpec,
    GaussianNoise,
    GaussianTemporal,
    LocalVertex,
    SeparableTranslation,
    System,
)


@pytest.mark.parametrize("rate", [0.00009999999, 0.00010000001, 1.0])
@pytest.mark.parametrize("time", [1.95, 2.0, 100.0])
def test_close_ordered_rates(rate, time):
    value, shift = simplex(time, (rate,) * 4, (0,) * 4)
    reference = (expm1(rate * time) / rate) ** 4 / factorial(4)
    assert value * exp(shift) == pytest.approx(reference, rel=1e-13, abs=0)


def _scalar_case(gamma, sigma_t, time, *, temporal=ExponentialTemporal):
    system = System(
        field=FieldSpec("phi", 1),
        linear=DiagonalA([gamma]),
        noise=GaussianNoise(
            SeparableTranslation(
                temporal=temporal(lam=1.0, sigma_t=sigma_t),
                spatial=ExponentialSpatial(sigma_x=1.0),
            )
        ),
        vertices=(LocalVertex("F", coupling=np.ones((1, 1, 1))),),
    )
    expansion = system.expand(("phi(x)",), orders=[1], progress=False)
    props = system.propagators(t_max=time, progress=False)
    return expansion, props


@pytest.mark.parametrize("gamma,sigma_t,time", [(1e-8, 1e8, 1.0), (1.0, 0.3, 1e-6)])
def test_conditioned_finite_start(gamma, sigma_t, time):
    expansion, props = _scalar_case(gamma, sigma_t, time)
    query = dict(positions={"x": 0.0}, t_final=time)
    reference = expansion.evaluate(props, **query, method="gauss_legendre", n_gauss=24)
    exact = expansion.evaluate(props, **query, method="exact")
    assert reference.total > 0
    assert exact.total == pytest.approx(reference.total, rel=1e-13, abs=0)


def test_stationary_scalar_mean():
    expansion, props = _scalar_case(1.0, 0.5, 100.0)
    exact = expansion.evaluate(props, positions={"x": 0.0}, t_final=inf, method="exact")
    # Stationary variance is 1/[gamma*(gamma+1/sigma_t)].
    # The time integral of the single response gives another 1/gamma.
    assert exact.total == pytest.approx(1 / 3, rel=1e-13, abs=0)


def test_exact_refuses_closed_form_gaussian():
    expansion, props = _scalar_case(1.0, 0.5, 1.0, temporal=GaussianTemporal)
    with pytest.raises(ValueError, match="direct built-in OU/white"):
        expansion.evaluate(
            props,
            positions={"x": 0.0},
            t_final=1.0,
            component_pair=(0,),
            method="exact",
        )


def test_branching_polynomial_time_order():
    from sft_wick._exp_sum import poset_integral

    time = 2.0
    value, shift = poset_integral(time, (0.0, 0.0, 0.0), (1, 2, 3), (0, 0, 3))
    # Integrate the independent leaf powers up to t2, then t2 from 0 to time.
    assert value * exp(shift) == pytest.approx(time**9 / 54, rel=1e-13, abs=0)


def test_capped_integral_near_clustered_rates():
    from sft_wick._exp_sum import ordered_integral

    rate, time = 0.00010000001, 2.0
    value = ordered_integral(np.full((4, 1), time), [rate] * 4, 0.0)[0]
    reference = (expm1(rate * time) / rate) ** 4 / factorial(4)
    assert value == pytest.approx(reference, rel=1e-13, abs=0)


def test_confluent_high_degree_divided_difference():
    # Over 0 < x < h, integrate x**4 * (h-x)**4 * exp(-x).
    from scipy.integrate import quad

    from sft_wick._exp_sum import _exp_difference

    width = 2.01
    value = width**9 * factorial(4) ** 2 * _exp_difference((-width,) * 5 + (0.0,) * 5)
    expected = quad(lambda x: x**4 * (width - x) ** 4 * exp(-x), 0.0, width)[0]
    assert value == pytest.approx(expected, rel=1e-13, abs=0)


@pytest.mark.parametrize("gamma", [1e-160, 1e155])
def test_stationary_fixed_covariance_at_extreme_rates(gamma):
    system = System(
        field=FieldSpec("phi", 1),
        linear=DiagonalA([gamma]),
        noise=GaussianNoise(
            SeparableTranslation(
                temporal=ExponentialTemporal(lam=gamma, sigma_t=1 / gamma),
                spatial=ExponentialSpatial(sigma_x=1.0),
            )
        ),
    )
    expansion = system.expand(("phi(x)", "phi(y)"), orders=[0], progress=False)
    props = system.propagators(t_max=1.0, progress=False)
    result = expansion.evaluate(
        props, positions={"x": 0.0, "y": 0.0}, t_final=inf, method="exact"
    )
    assert result.total == pytest.approx(0.5 / gamma, rel=1e-13, abs=0)


def test_workflow_exact_refuses_selected_callable_couplings():
    from pathlib import Path

    from sft_wick.workflow.config import (
        ConfigError,
        check_before_run,
        load_workflow_config,
    )

    path = Path(__file__).resolve().parents[1] / "examples" / "demo2_config.yaml"
    selected = load_workflow_config(path, {"sweep.method": "exact"})
    with pytest.raises(ConfigError, match="static couplings"):
        check_before_run(selected)

    free_only = load_workflow_config(
        path, {"sweep.method": "exact", "sweep.orders": [0]}
    )
    check_before_run(free_only)


def test_many_colored_factors_on_a_short_horizon():
    expansion, props = _scalar_case(1.0, 0.3, 1e-3)
    expansion = expansion.system.expand(
        ("phi(x)", "phi(y)"), orders=[2], progress=False
    )
    query = dict(positions={"x": 0.0, "y": 0.3}, t_final=1e-3)
    reference = expansion.evaluate(props, **query, method="gauss_legendre", n_gauss=20)
    exact = expansion.evaluate(props, **query, method="exact")
    assert reference.total > 0
    assert exact.total == pytest.approx(reference.total, rel=1e-13, abs=0)
