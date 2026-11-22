"""Initial means, covariance, and cumulants at unequal times."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm

import sft_wick as sw


T0 = 0.3
T1 = 0.59
T2 = 1.1
A = np.array([[-0.8, 0.43], [-0.71, -1.3]])
B0 = np.array([[0.72, -0.21], [-0.21, 0.39]])
K3 = np.array([0.19, -0.14, 0.32, -0.08])[np.indices((2, 2, 2)).sum(axis=0)]


def _noise(lam):
    return sw.GaussianNoise(kappa2=sw.SeparableTranslation(
        temporal=sw.ExponentialTemporal(lam=lam, sigma_t=0.54),
        spatial=sw.ExponentialSpatial(sigma_x=0.71),
    ))


def _response(t1, t2):
    return expm(A * (t1 - t2))


def _zero_drive(x1, t1, x2, t2):
    return np.zeros(np.shape(t1) + (2, 2))


@pytest.mark.parametrize("routing", ["enumerate", "multiplicity"])
def test_dense_initial_cumulant_uses_full_forward_response(routing):
    system = sw.System(
        sw.FieldSpec("phi", 2), sw.ExplicitR(_response, iso_R=False), _noise(0),
        t_min=T0, initial_state=sw.InitialState(covariance=B0, cumulants=(K3,)),
    )
    props = system.propagators(
        t_max=T2, c_closed_form=_zero_drive, c_closed_form_only=True,
        c_closed_form_vectorized=True, diag_C=False,
    )
    expected_covariance = _response(T2, T0) @ B0
    np.testing.assert_allclose(
        props.cache.C_value(0, T2, 0, T0), expected_covariance, rtol=0, atol=1e-15
    )
    expansion = system.expand(
        ("phi_a(x)", "phi_b(y)", "phi_c(z)"), orders=(1,),
        diag_C=False, diag_R=False, routing=routing,
    )
    times = (T0, T1, T2)
    g0, g1, g2 = (_response(t, T0) for t in times)
    expected = np.einsum("ijk,ai,bj,ck->abc", K3, g0, g1, g2)
    for components in product(range(2), repeat=3):
        actual = expansion.evaluate(
            props, positions={"x": 0.0, "y": 0.0, "z": 0.0},
            external_times=dict(zip("xyz", times)), t_final=T2,
            component_pair=components, method="gauss_legendre", n_gauss=20,
        ).total
        assert actual == pytest.approx(expected[components], rel=0, abs=1e-15)


@pytest.mark.parametrize("closed_form", ["auto", None])
def test_initial_covariance_adds_to_the_drive_at_two_separations(closed_form):
    gamma = 0.8
    system = sw.System(
        sw.FieldSpec(), sw.DiagonalA(gamma=[gamma]), _noise(0.12), t_min=T0,
        initial_state=sw.InitialState(covariance=[[0.4]]),
    )
    kwargs = {"c_closed_form": closed_form, "c_method": "gauss_legendre",
              "c_n_gauss": 10, "n_grid_t": 5, "table_rtol": None}
    cache = system.propagators(t_max=T2, **kwargs).cache
    drive = replace(system, initial_state=None).propagators(t_max=T2, **kwargs).cache
    time_left, time_right = T0, (T0 + T2) / 2
    expected = 0.4 * np.exp(-gamma * (time_left + time_right - 2 * T0))
    for distance in (0.0, 0.67):
        value = cache.C_value(0.0, time_left, distance, time_right)
        base = drive.C_value(0.0, time_left, distance, time_right)
        np.testing.assert_allclose(value - base, [[expected]], rtol=0, atol=5e-15)


def test_vectorized_full_grid_preserves_covariance_at_initial_time():
    state = sw.InitialState(covariance=B0)
    rates = np.array([0.8, 1.3])
    system = sw.System(
        sw.FieldSpec("phi", 2), sw.DiagonalA(gamma=rates), _noise(0.12),
        t_min=T0, initial_state=state,
    )
    props = system.propagators(
        t_max=T2, diag_C=False, n_grid_t=12,
        r_max=1.0, n_grid_r=6, table_rtol=1e-2,
    )
    np.testing.assert_allclose(
        props.cache.C_value(0.0, T2, 1.0, T0),
        np.exp(-rates * (T2 - T0))[:, None] * B0, rtol=0, atol=1e-15,
    )


def test_exact_backend_refuses_initial_covariance():
    system = sw.System(
        sw.FieldSpec(), sw.DiagonalA(gamma=[0.8]), _noise(0.12),
        t_min=T0, initial_state=sw.InitialState(covariance=[[0.4]]),
    )
    props = system.propagators(t_max=T2)
    expansion = system.expand(("phi(x)", "phi(y)"), orders=(0,))
    with pytest.raises(ValueError, match="exact.*initial covariance"):
        expansion.evaluate(
            props, positions={"x": 0.0, "y": 0.0}, t_final=T2,
            component_pair=(0, 0), method="exact",
        )


def test_uncertified_matrix_response_requires_full_initial_covariance():
    def shear(time):
        return 0.5 * float(0.52 < time < 0.60)

    def transition(t, s):
        return np.array([[1.0, shear(t) - shear(s)], [0.0, 1.0]])

    system = sw.System(
        sw.FieldSpec("phi", 2), sw.ExplicitR(transition, iso_R=False), _noise(0),
        t_min=T0, initial_state=sw.InitialState(covariance=np.eye(2)),
    )
    with pytest.raises(ValueError, match="diag_C=False"):
        system.propagators(t_max=T2)


def test_initial_covariance_rejects_negative_variance_at_any_scale():
    with pytest.raises(ValueError, match="positive semidefinite"):
        sw.InitialState(covariance=[[-1.0, 0.0], [0.0, 1e30]])


def test_initial_covariance_does_not_broadcast_a_malformed_closed_form():
    system = sw.System(
        sw.FieldSpec("phi", 2), sw.DiagonalA(gamma=[0.8, 1.3]), _noise(0),
        t_min=T0, initial_state=sw.InitialState(covariance=B0),
    )
    props = system.propagators(
        t_max=T2, c_closed_form=lambda *args: np.zeros((2, 2)),
        c_closed_form_vectorized=True, diag_C=False,
    )
    times = np.array([T1, T2])
    with pytest.raises(ValueError, match="driven covariance must return shape"):
        props.cache.C_at_batch(times, times, np.zeros(2), np.zeros(2))


@pytest.mark.parametrize("mean", [None, 0.34])
def test_initial_state_in_yaml_reaches_the_normal_order_zero_workflow(tmp_path, mean):
    import yaml
    from sft_wick.workflow.config import load_workflow_config, run_workflow
    from sft_wick.workflow.schema import resolved_config

    path = tmp_path / "initial.yaml"
    path.write_text("""
system:
  t_min: 0.3
  linear: {type: diagonal, gamma: 0.8}
  noise:
    kappa2:
      type: separable_translation
      temporal: {type: exponential, lam: 0.0, sigma_t: 1.0}
      spatial: {type: exponential, sigma_x: 1.0}
  initial_state:
    covariance: [[0.4]]
    cumulants: [[[[0.2]]]]
expand:
  observable: [phi(x), phi(y)]
  orders: [0]
propagators: {t_max: 1.1}
sweep:
  positions_grid: {x: [0.0], y: [0.0]}
  t_final_grid: [1.1]
  component_tuples: [[0, 0]]
  method: gauss_legendre
""")
    if mean is not None:
        config = yaml.safe_load(path.read_text())
        config["system"]["initial_state"]["mean"] = [mean]
        path.write_text(yaml.safe_dump(config))
    cfg = load_workflow_config(path)
    assert resolved_config(cfg)["system"]["initial_state"]["cumulants"] == [[[[0.2]]]]
    initial_variance = 0.4 + (0.0 if mean is None else mean**2)
    _, totals = run_workflow(cfg)
    assert totals["value"].iloc[0] == pytest.approx(
        initial_variance * np.exp(-2 * 0.8 * (1.1 - 0.3)), abs=1e-15, rel=0
    )


def test_dense_raw_mean_and_covariance_at_an_integrated_external_time():
    mean = np.array([0.27, -0.43])
    system = sw.System(
        sw.FieldSpec("phi", 2), sw.ExplicitR(_response, iso_R=False), _noise(0),
        t_min=T0, initial_state=sw.InitialState(covariance=B0, mean=mean),
    )
    props = system.propagators(
        T2, c_closed_form=_zero_drive, c_closed_form_vectorized=True, diag_C=False,
    )
    for target, initial_second in (("moment", B0 + np.outer(mean, mean)), ("connected", B0)):
        expansion = system.expand(
            ("phi_a(x)", "phi_b(y)"), (0,), target=target, diag_R=False, diag_C=False,
        )
        result = expansion.evaluate(
            props, positions={"x": 0.0, "y": 0.2}, t_final=T2, external_times={"x": T1},
            integrate_over=("y",), component_pair=(0, 1), method="gauss_legendre",
            n_gauss=20, gl_error=True,
        )
        expected = quad(
            lambda t: (_response(T1, T0) @ initial_second @ _response(t, T0).T)[0, 1],
            T0, T2, epsabs=1e-14,
        )[0]
        assert result.total == pytest.approx(expected, abs=1e-15, rel=0)
        assert set(result.by_vertex_type) == {""}


@pytest.mark.parametrize("vectorized", [False, True])
def test_translated_callable_quadratic_drift_keeps_original_tag(vectorized):
    rates = np.array([0.8, 1.3])
    mean = np.array([0.27, -0.43])
    tensor = np.array([[[0.42, -0.14], [0.06, 0.12]], [[0.29, -0.3], [0.4, 0.25]]])

    def coupling(positions, times):
        time = times[0]
        return (1 + np.asarray(time)[..., None, None, None]) * tensor

    system = sw.System(
        sw.FieldSpec("phi", 2), sw.DiagonalA(gamma=rates), _noise(0.0),
        vertices=(sw.LocalVertex("F", coupling, rank=3, coupling_vectorized=vectorized),),
        t_min=T0, initial_state=sw.InitialState(mean=mean),
    )
    exp = system.expand(("phi_a(x)",), (1,), routing="multiplicity")
    props = system.propagators(T2)
    assert set(exp.by_vertex_type(1)) == {"F"}
    for component in range(2):
        def integrand(time):
            background = np.exp(-rates * (time - T0)) * mean
            source = (1 + time) * np.einsum("ij,i,j->", tensor[component], background, background)
            return np.exp(-rates[component] * (T2 - time)) * source

        expected = quad(integrand, T0, T2, epsabs=1e-14)[0]
        result = exp.evaluate(
            props, positions={"x": 0.0}, t_final=T2, component_pair=(component,),
            vertex_types="F", method="gauss_legendre", n_gauss=20,
        )
        assert result.by_order[1] == pytest.approx(expected, abs=1e-15, rel=0)


def test_mean_cache_stores_structure_and_binds_current_amplitude(tmp_path):
    system = sw.System(sw.FieldSpec(), sw.DiagonalA(gamma=[0.8]), _noise(0.0), t_min=T0)
    props = system.propagators(T2)
    for mean in (0.5, -0.2, 0.0):
        current = replace(system, initial_state=sw.InitialState(mean=[mean]))
        exp = current.expand(("phi(x)",), (0,), cache_path=tmp_path)
        result = exp.evaluate(
            props, positions={"x": 0.0}, t_final=T2, component_pair=(0,),
            method="gauss_legendre",
        )
        assert result.total == pytest.approx(mean * np.exp(-0.8 * (T2 - T0)), abs=1e-15)


def test_scalar_field_operator_with_a_component_label_can_carry_mean():
    system = sw.System(
        sw.FieldSpec(), sw.DiagonalA(gamma=[0.8]), _noise(0.0),
        t_min=T0, initial_state=sw.InitialState(mean=[0.5]),
    )
    phi, _ = system.build_fields()
    observable = replace(phi("x"), component_index="a")
    result = system.expand([observable], (0,)).evaluate(
        system.propagators(T2), positions={"x": 0.0}, t_final=T2,
        component_pair=(0,), method="gauss_legendre",
    )
    assert result.total == pytest.approx(0.5 * np.exp(-0.8 * (T2 - T0)), abs=1e-15)
