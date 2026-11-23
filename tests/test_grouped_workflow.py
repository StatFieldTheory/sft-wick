"""Ordered tensor axes and MSR normalization of a grouped interaction."""
import numpy as np
import pytest

import sft_wick as sw


def _system(vertex, n=1, **kwargs):
    return sw.System(
        field=sw.FieldSpec("phi", n),
        linear=sw.DiagonalA([0.7] * n),
        noise=sw.GaussianNoise(
            sw.SeparableTranslation(sw.ExponentialTemporal(0., .4), sw.GaussianSpatial(1.)),
            sw.ConstantImpulse(np.eye(n)),
        ),
        grouped_vertices=(vertex,), **kwargs,
    )


def test_averaged_rate_preserves_asymmetric_matrix_axes_and_one_half():
    g = np.array([[.2, 3.], [-2., .7]])
    def kernel(positions, times):
        return np.exp(-abs(times[0] - times[1]))
    vertex = sw.ColouredMultiplicativeNoise(kernel, g, name="M")
    system = _system(vertex, 2)
    raw, = system.build_action().vertices
    assert [f.is_response for f in raw.fields] == [True, False, True, False]
    assert raw.leg_groups == (0, 0, 1, 1)
    value = system.build_coupling_values()["M"]([4., 9.], [.3, .7])
    np.testing.assert_allclose(value, .5 * np.exp(-.4) * np.multiply.outer(g, g))
    with pytest.raises(ValueError, match="rate matrix"):
        _system(vertex, 1)
    times = np.array([[.3, .5], [.7, .6]])
    vectors = sw.ColouredMultiplicativeNoise(kernel, g, coupling_vectorized=True)
    batches = _system(vectors, 2).build_coupling_values()[vectors.name]([[0, 0], [1, 1]], times)
    np.testing.assert_allclose(batches[1], .5 * np.exp(-.1) * np.multiply.outer(g, g))


def test_delayed_quadratic_vertex_preserves_order_and_oriented_time_difference():
    vertex = sw.GroupedVertex(
        "F", ("phi", "psi", "phi"), np.ones((1, 1, 1)), (1, 0, 1),
        group_lags=((1, 0, .31),),
    )
    system = _system(vertex)
    raw, = system.build_action().vertices
    assert [f.is_response for f in raw.fields] == [False, True, False]
    assert raw.group_lags == ((1, 0, .31),)
    assert system.build_coupling_values()["F"].item() == -1j
    with pytest.raises(ValueError, match="mean"):
        _system(vertex, initial_state=sw.InitialState(mean=[2.]))


def test_time_group_keeps_two_positions_in_ordered_component_kernel():
    def kernel(positions, times):
        assert times[0] == times[1]
        return (positions[1] - positions[0]) * np.array([[0., 1.], [-1., 0.]])

    vertex = sw.GroupedVertex("K", ("psi", "psi"), kernel, (0, 0), group_mode="time")
    system = _system(vertex, 2)
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[1])
    result = expansion.evaluate(
        system.propagators(t_max=2., progress=False),
        positions={"x": .2, "y": 1.1}, external_times={"x": .8, "y": 1.3},
        t_final=1.3, component_pair=(0, 1), method="gauss_legendre", n_gauss=12,
    )
    scale = np.exp(-.7 * (1.3 - .8)) * -np.expm1(-2 * .7 * .8) / (2 * .7)
    assert result.total == pytest.approx(.9 * scale, abs=1e-12)


def test_yaml_grouped_vertices_and_invalid_schema_input(tmp_path):
    from sft_wick.workflow.config import ConfigError, load_workflow_config, run_workflow

    import yaml

    doc = {
        "system": {
            "field": {"n_components": 1}, "linear": {"type": "diagonal", "gamma": [.7]},
            "noise": {"kappa2": {"type": "separable_translation",
                                 "temporal": {"type": "exponential", "lam": 0, "sigma_t": .5},
                                 "spatial": {"type": "gaussian", "sigma_x": 1}},
                      "sigma2": {"type": "constant", "amplitude": 1.}},
            "grouped_vertices": [{"name": "F", "fields": ["psi", "phi", "phi"],
                                  "coupling": [[[1.]]], "leg_groups": [0, 1, 1],
                                  "group_lags": [[1, 0, .31]]}],
        },
        "expand": {"observable": ["phi(x)"], "orders": [1]},
        "propagators": {"t_max": 1.5},
        "sweep": {"positions_grid": {"x": [0.]}, "t_final_grid": [1.5],
                  "component_tuples": [[0]], "method": "gauss_legendre", "n_gauss": 12},
    }
    path = tmp_path / "delayed.yaml"
    path.write_text(yaml.safe_dump(doc))
    _, totals = run_workflow(load_workflow_config(path), progress=False)
    # Integrate the retarded response times the linear variance from tau to T.
    expected_mean = np.expm1(-.7 * (1.5 - .31))**2 / (2 * .7**2)
    np.testing.assert_allclose(totals["value"], [expected_mean], rtol=1e-11)
    doc["system"]["grouped_vertices"][0]["group_lags"] = [[0, 2, .3]]
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ConfigError, match="group"):
        load_workflow_config(path)


def test_unequal_time_gaussian_rate_at_two_insertions():
    from math import erf, pi, sqrt

    from scipy.integrate import quad

    gamma, sigma, width = .68, .79, .33
    start, t, s = .2, .89, 1.38

    def kernel(positions, times):
        return np.exp(-.5 * ((times[0] - times[1]) / width)**2)

    system = sw.System(
        sw.FieldSpec("phi", 1), sw.DiagonalA([gamma]),
        sw.GaussianNoise(
            sw.SeparableTranslation(sw.ExponentialTemporal(0., 1.), sw.ExponentialSpatial(1.)),
            sw.ConstantImpulse(sigma**2),
        ),
        t_min=start,
        grouped_vertices=[sw.ColouredMultiplicativeNoise(kernel, coupling_vectorized=True)],
    )
    expansion = system.expand(("phi(x)", "phi(y)"), orders=[2], progress=False)
    actual = expansion.evaluate(
        system.propagators(t_max=s, progress=False), positions={"x": 0., "y": 0.},
        external_times={"x": t, "y": s}, t_final=s, method="gauss_legendre", n_gauss=8,
    ).by_order[2]

    # Conditioning on the rate gives exp(epsilon**2 * Var / 2). Each
    # vertex carries two rate powers, so two insertions select Var**2 / 8.
    def variance(length):
        z = length / (sqrt(2) * width)
        return 2 * width * (length * sqrt(pi / 2) * erf(z) + width * np.expm1(-z*z))

    def fourth_order(u):
        v = 2 * variance(t - u) + 2 * variance(s - u) - variance(s - t)
        return sigma**2 * np.exp(-gamma * (t + s - 2*u)) * v**2 / 8

    expected = quad(fourth_order, start, t)[0]
    assert actual == pytest.approx(expected, abs=1e-9)
