"""Finite Markov representations of stochastic forcing, memory and delays."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.integrate import quad, quad_vec
from scipy.linalg import expm

import sft_wick as sw
from sft_wick.workflow._markov_embedding import (
    EmbeddedProcess,
    ErlangDelay,
    PronyMemory,
    compile_markov_embedding,
)
from sft_wick.workflow.config import load_workflow_config, run_workflow
from sft_wick.workflow.specs import (
    ConstantImpulse,
    DampedCosineTemporal,
    ExponentialTemporal,
    Matern32Temporal,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "reference"))
import ito_moments as im  # noqa: E402
from hormander_moments import two_time  # noqa: E402


@pytest.mark.parametrize("kernel", [
    ExponentialTemporal(0.31, 0.78),
    Matern32Temporal(0.31, 0.78),
    DampedCosineTemporal(0.31, 0.78, 2.08),
])
def test_random_rate_stationary_hidden_covariance_and_tagged_drift(kernel):
    spec = EmbeddedProcess(kernel, mode="random_rate", coupling=-0.74, start="stationary")
    model = compile_markov_embedding([[-0.72]], [[0.86**2]], spec)
    assert model.physical_components == 1
    assert model.vertex is not None
    assert model.vertex.coupling[0, 1, 0] == -0.37
    assert model.vertex.coupling[0, 0, 1] == -0.37
    assert np.count_nonzero(model.vertex.coupling) == 2
    assert model.drift[0, 1] == 0
    assert model.initial_covariance[0, 0] == 0
    for lag in (0.0, 0.13, 0.83, 1.47):
        out = model.response(0.3 + lag, 0.3) @ model.initial_covariance
        np.testing.assert_allclose(out[1, 1], kernel(lag), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(
        (model.drift @ model.initial_covariance
         + model.initial_covariance @ model.drift.T + model.diffusion)[1:, 1:],
        0.0, atol=1e-14,
    )


@pytest.mark.parametrize("kernel, reference", [
    (PronyMemory((1.13, 1.92), (0.25, 0.44), temperature=0.41),
     lambda t: 0.25 * np.exp(-1.13*t) + 0.44 * np.exp(-1.92*t)),
    (ErlangDelay(0.94, stages=3), lambda t: (3/0.94)**3 * t*t/2 * np.exp(-3*t/0.94)),
])
def test_memory_feedback_and_joint_stationary_covariance(kernel, reference):
    spec = EmbeddedProcess(kernel, source=1, target=0, coupling=-0.21, start="stationary")
    drift = np.diag([-0.72, -0.6])
    diffusion = np.diag([0.86**2, 0.44**2])
    model = compile_markov_embedding(drift, diffusion, spec)
    assert model.vertex is None
    assert model.initial_covariance[0, 0] > 0
    for lag in (0.0, 0.13, 0.83, 1.47):
        hidden_response = expm(model.drift[2:, 2:] * lag)
        lowered = (model.drift[0, 2:] @ hidden_response @ model.drift[2:, 1])
        np.testing.assert_allclose(lowered, -0.21 * reference(lag), atol=1e-15)
    np.testing.assert_allclose(
        model.drift @ model.initial_covariance
        + model.initial_covariance @ model.drift.T + model.diffusion,
        0.0, atol=1e-14,
    )


@pytest.mark.parametrize("kernel", [
    Matern32Temporal(0.31, 0.78), DampedCosineTemporal(0.31, 0.78, 2.08),
    PronyMemory((1.13, 1.92), (0.25, 0.44), temperature=0.41), ErlangDelay(0.94),
])
def test_driven_covariance_at_unequal_times_matches_defining_integral(kernel):
    spec = EmbeddedProcess(kernel, coupling=-0.21, start="stationary")
    model = compile_markov_embedding([[-0.72]], [[0.86**2]], spec)
    t0 = 0.3
    t1 = np.array([0.3, 0.3000000001, 0.83, 1.47, 6.0])
    t2 = np.array([0.3, 1.47, 1.47, 0.83, 7.0])
    closed = model.driven_covariance(t0)
    batch = closed(None, t1, None, t2)
    assert closed.has_diagonal_kink
    for x, y, got in zip(t1, t2, batch):
        def integrand(s):
            return expm(model.drift * (x - s)) @ model.diffusion @ expm(model.drift.T * (y - s))

        ref = quad_vec(integrand, t0, min(x, y), epsabs=1e-13, epsrel=1e-13)[0]
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(got, closed(None, x, None, y), rtol=1e-13, atol=1e-16)
    np.testing.assert_array_equal(batch[0], np.zeros_like(model.diffusion))
    early = 0.0000000001
    scalar = closed(None, t0 + early, None, t0 + early)
    np.testing.assert_allclose(scalar / (t0 + early - t0), model.diffusion,
                               rtol=1e-8, atol=1e-8)


def test_driven_covariance_avoids_the_antidamped_exponential():
    model = compile_markov_embedding(
        [[-800.0]], [[1600.0]], EmbeddedProcess(ExponentialTemporal(1.0, 0.001))
    )
    val = model.driven_covariance(0.3)(0.0, 2.0, 0.0, 2.0)
    assert np.all(np.isfinite(val))
    assert val[0, 0] == pytest.approx(1 + 1/(800 * (800 + 1000)), abs=1e-14)


def test_covariance_keeps_unrelated_slow_and_fast_scales():
    spec = EmbeddedProcess(ErlangDelay(1.0), coupling=0.0, start="stationary")
    model = compile_markov_embedding([[-1e-20]], [[1e-20]], spec)
    assert model.initial_covariance[0, 0] == pytest.approx(0.5, abs=1e-13)
    result = model.driven_covariance(0.0)(0.0, 1e20, 0.0, 1e20)
    assert result[0, 0] == pytest.approx(-0.5 * np.expm1(-2.0), abs=1e-13)


def test_large_finite_diffusion_gives_large_finite_covariance():
    model = compile_markov_embedding(
        [[-1.0]], [[1e300]], EmbeddedProcess(ExponentialTemporal(0.0, 1.0), coupling=0.0)
    )
    value = model.driven_covariance(0.0)(0.0, 1.0, 0.0, 1.0)
    assert value[0, 0] == pytest.approx(-0.5e300 * np.expm1(-2.0), rel=1e-14, abs=0.0)


def test_invalid_parameters_and_unstable_stationary_memory():
    with pytest.raises(ValueError, match="same length"):
        PronyMemory((1.0, 2.0), (0.4,))
    with pytest.raises(ValueError, match="non-negative"):
        PronyMemory((1.0,), (-0.4,), temperature=1.0)
    with pytest.raises(ValueError, match="positive integer"):
        ErlangDelay(1.0, stages=True)
    with pytest.raises(ValueError, match="rate modes"):
        EmbeddedProcess(ErlangDelay(1.0), mode="random_rate")
    with pytest.raises(ValueError, match="source"):
        EmbeddedProcess(Matern32Temporal(0.31, 0.78), source=0)
    with pytest.raises(ValueError, match="variance"):
        EmbeddedProcess(Matern32Temporal(-0.31, 0.78))
    with pytest.raises(ValueError, match="physical component"):
        compile_markov_embedding([[-0.72]], [[1.0]],
                                 EmbeddedProcess(ErlangDelay(0.5), target=1))
    with pytest.raises(ValueError, match="positive semidefinite"):
        compile_markov_embedding([[-0.72]], [[-1.0]], EmbeddedProcess(ErlangDelay(0.5)))
    with pytest.raises(ValueError, match="positive semidefinite"):
        compile_markov_embedding(np.diag([-0.72, -1.0]), np.diag([1e12, -0.5]),
                                 EmbeddedProcess(ErlangDelay(0.5)))
    with pytest.raises(ValueError, match="positive semidefinite"):
        compile_markov_embedding(np.diag([-0.72, -1.0]), [[1e12, 2.0], [2.0, 1e-12]],
                                 EmbeddedProcess(ErlangDelay(0.5)))
    with pytest.raises(ValueError, match="stable"):
        compile_markov_embedding([[-0.72]], [[1.0]],
                                 EmbeddedProcess(ErlangDelay(0.5), coupling=0.9,
                                                 start="stationary"))


def test_embedding_checks_augmented_tensor_size_before_padding():
    physical = sw.System(
        sw.FieldSpec(), sw.DiagonalA(gamma=[0.8]),
        sw.GaussianNoise(sw.SeparableTranslation(
            sw.ExponentialTemporal(0.0, 1.0), sw.ExponentialSpatial(1.0),
        )),
        vertices=(sw.LocalVertex("F", np.ones((1,) * 6)),),
    )
    with pytest.raises(ValueError, match="augmented tensor memory limit"):
        sw.embed_system(physical, EmbeddedProcess(ErlangDelay(0.94, stages=20)))


T0, T1, T2 = 0.3, 0.83, 1.47
GAMMA, SIGMA, RATE, VAR, OMEGA = 0.72, 0.86, 1.23, 0.31, 2.08


def _reference(name, stationary):
    # The hand-written Matérn generator uses the unscaled derivative, ξ′.
    if name == "matern":
        drift = [[-GAMMA, 0, 0], [0, 0, 1], [0, -RATE**2, -2*RATE]]
        diffusion = np.diag([SIGMA**2, 0, 4*RATE**3*VAR])
        covariance = np.diag([0, VAR, RATE**2*VAR])
    elif name == "damped":
        drift = [[-GAMMA, 0, 0], [0, -RATE, -OMEGA], [0, OMEGA, -RATE]]
        diffusion = np.diag([SIGMA**2, 2*RATE*VAR, 2*RATE*VAR])
        covariance = np.diag([0, VAR, VAR])
    elif name == "prony":
        drift = [[-GAMMA, -1, -1], [0.25, -1.13, 0], [0.44, 0, -1.92]]
        diffusion = 2 * 0.41 * np.diag([GAMMA, 1.13*0.25, 1.92*0.44])
        covariance = 0.41 * np.diag([1, 0.25, 0.44])  # thermal law
    else:
        rate = 3/0.94
        drift = np.array([[-GAMMA, 0, 0, 0.21], [rate, -rate, 0, 0],
                          [0, rate, -rate, 0], [0, 0, rate, -rate]])
        diffusion = np.diag([SIGMA**2, 0, 0, 0])

        def past_driving(s):
            response = expm(drift * s)
            return response @ diffusion @ response.T

        covariance = quad_vec(past_driving, 0, np.inf, epsabs=1e-13, epsrel=1e-13)[0]
    n = len(drift)
    sde = im.PolySDE(D=n, n_tags=1)
    sde.add_linear_drift(np.asarray(drift))
    sde.add_constant_diffusion(diffusion)
    if name in ("matern", "damped"):
        sde.drift.append((0, -1.0, im.unit(n, 0, 1), (1,)))

    def wick_moment(legs):
        if not legs:
            return 1.0
        first, *tail = legs
        return sum(covariance[first, k] * wick_moment(tail[:i] + tail[i+1:])
                   for i, k in enumerate(tail))

    def initial(alpha, tag):
        if any(tag) or not stationary:
            return float(not any(alpha) and not any(tag))
        return wick_moment([i for i, count in enumerate(alpha) for _ in range(count)])

    orders = [0, 2] if name in ("matern", "damped") else [0]
    moments = two_time(sde, im.unit(n, 0), im.unit(n, 0), [(o,) for o in orders],
                       T1-T0, T2-T0, initial=initial)
    return {key[0]: val for key, val in moments.items()}


def _conditional_random_rate(name, stationary):
    # For t >= u, the conditional stochastic convolution has weight
    # exp[-γ(t-u) - ε∫_u^t ξ]. Wick averaging the common physical white
    # driving gives a one-dimensional integral over u <= min(T1,T2).
    if name == "matern":
        def kernel(lag):
            return VAR * (1 + RATE*lag) * np.exp(-RATE*lag)

        def primitive(t):
            return np.array([-t-2/RATE, -t-1/RATE]) * np.exp(-RATE*t)
    else:
        def kernel(lag):
            return VAR * np.cos(OMEGA*lag) * np.exp(-RATE*lag)

        def primitive(t):
            val = np.exp((-RATE+1j*OMEGA)*t) / (-RATE+1j*OMEGA)
            return np.array([val.real, -val.imag])

    def variance(duration):
        return 2*quad(lambda lag: (duration-lag)*kernel(lag), 0, duration,
                      epsabs=1e-13, epsrel=1e-13)[0]

    def integrand(u):
        var = 2*variance(T1-u) + 2*variance(T2-u) - variance(T2-T1)
        if not stationary:
            # Subtract the stationary prehistory term propagated to t0.
            row = primitive(T1-T0) + primitive(T2-T0) - 2*primitive(u-T0)
            var -= VAR * (row @ row)
        return 0.5*SIGMA**2 * var * np.exp(-GAMMA*(T1+T2-2*u))

    return quad(integrand, T0, T1, epsabs=1e-13, epsrel=1e-13)[0]


def _existing_workflow(model, orders, nodes):
    # This explicitly wires the compiler's L1 building blocks. The separate
    # declarative integration test exercises the System initial-state hook.
    system = sw.System(
        field=sw.FieldSpec(n_components=len(model.drift)),
        linear=sw.ExplicitR(model.response, iso_R=False, vectorized=True),
        noise=sw.GaussianNoise(
            sw.SeparableTranslation(sw.ExponentialTemporal(0.0, 1.0), sw.ExponentialSpatial(1.0)),
            sigma2=ConstantImpulse(model.diffusion),
        ),
        vertices=() if model.vertex is None else (model.vertex,),
        t_min=T0,
    )
    driven = model.driven_covariance(T0)

    def full_covariance(x, t, y, s):
        return (
            driven(x, t, y, s)
            + model.response(t, T0) @ model.initial_covariance
            @ model.response(s, T0).swapaxes(-1, -2)
        )

    props = system.propagators(
        T2+0.1, c_closed_form=full_covariance, c_closed_form_vectorized=True,
        diag_C=False, progress=False,
    )
    expansion = system.expand(
        ("phi_a(x)", "phi_b(y)"), orders, diag_R=False, diag_C=False, progress=False
    )
    return [expansion.evaluate(
        props, positions={"x": 0.0, "y": 0.0}, t_final=T2, external_times={"x": T1},
        component_pair=(0, 0), orders=orders, method="gauss_legendre", n_gauss=n,
    ).by_order for n in nodes]


@pytest.mark.parametrize("name, kernel, coupling, mode, white", [
    ("matern", Matern32Temporal(VAR, np.sqrt(3)/RATE), -1.0, "random_rate", SIGMA**2),
    ("damped", DampedCosineTemporal(VAR, 1/RATE, OMEGA), -1.0, "random_rate", SIGMA**2),
    ("prony", PronyMemory((1.13, 1.92), (0.25, 0.44), 0.41), -1.0, "additive", 2*0.41*GAMMA),
    ("erlang", ErlangDelay(0.94), 0.21, "additive", SIGMA**2),
])
@pytest.mark.parametrize("start", ["zero", "stationary"])
def test_embedded_moments_match_independent_generator(
    name, kernel, coupling, mode, white, start, tmp_path,
):
    stationary = start == "stationary"
    ref = _reference(name, stationary)
    orders = sorted(ref)
    kernel_yaml = {
        "matern": lambda: {"type": "matern32", "lam": VAR, "sigma_t": float(kernel.sigma_t)},
        "damped": lambda: {"type": "damped_cosine", "lam": VAR, "sigma_t": kernel.sigma_t,
                           "omega": kernel.omega},
        "prony": lambda: {"type": "prony", "rates": list(kernel.rates),
                          "weights": list(kernel.weights), "temperature": kernel.temperature},
        "erlang": lambda: {"type": "erlang", "mean_delay": kernel.mean_delay,
                           "stages": kernel.stages},
    }[name]()
    doc = {
        "system": {
            "t_min": T0,
            "linear": {"type": "diagonal", "gamma": GAMMA},
            "noise": {
                "kappa2": {
                    "type": "separable_translation",
                    "temporal": {"type": "exponential", "lam": 0, "sigma_t": 1},
                    "spatial": {"type": "exponential", "sigma_x": 1},
                },
                "sigma2": {"type": "constant", "amplitude": white},
            },
            "embedding": {"kernel": kernel_yaml, "coupling": coupling, "mode": mode,
                          "start": start},
        },
        "expand": {"observable": ["phi(x)", "phi(y)"], "orders": orders, "diag_R": False},
        "propagators": {"t_max": T2, "diag_C": False},
        "sweep": {"positions_grid": {"x": [0], "y": [0]}, "t_final_grid": [T2],
                  "external_times_grid": {"x": [T1]}, "component_tuples": [[0, 0]],
                  "method": "gauss_legendre"},
    }
    path = tmp_path / "embedding.yaml"
    for nodes in (20, 28, 36):
        doc["sweep"]["n_gauss"] = nodes
        path.write_text(yaml.safe_dump(doc))
        _, table = run_workflow(load_workflow_config(path))
        actual = dict(zip(table["order"], table["value"]))
        np.testing.assert_allclose([actual[o] for o in orders], [ref[o] for o in orders],
                                   rtol=0.0, atol=1e-10)
    if mode == "random_rate":
        assert ref[2] == pytest.approx(_conditional_random_rate(name, stationary),
                                      rel=0.0, abs=1e-12)


def test_random_rate_detects_wrong_matern_diffusion_and_omitted_hidden_initial_state():
    process = EmbeddedProcess(Matern32Temporal(VAR, np.sqrt(3)/RATE), coupling=-1.0,
                              mode="random_rate", start="stationary")
    model = compile_markov_embedding([[-GAMMA]], [[SIGMA**2]], process)
    wrong_diffusion = model.diffusion.copy()
    wrong_diffusion[2, 2] *= 0.5
    ref = _reference("matern", True)[2]
    for wrong in (replace(model, diffusion=wrong_diffusion),
                  replace(model, initial_covariance=np.zeros_like(model.initial_covariance))):
        result = _existing_workflow(wrong, [2], [28])[0][2]
        assert abs(result - ref) > 1e-4


def test_shared_random_rate_matrix_compiles_without_transposing():
    coupling = np.array([[0.3, -0.41], [0.19, 0.2]])
    process = EmbeddedProcess(ExponentialTemporal(2.5, 0.2), mode="random_rate",
                              start="stationary", coupling=coupling)
    model = compile_markov_embedding(-np.eye(2), np.eye(2), process)
    x = np.array([0.79, -0.16, 1.4])
    force = np.einsum("abc,b,c->a", model.vertex.coupling, x, x)
    np.testing.assert_allclose(force[:2], x[-1] * (coupling @ x[:2]))
    assert force[-1] == 0
    assert model.initial_covariance[-1, -1] == 2.5


def test_matrix_rate_refuses_ambiguous_or_incompatible_inputs():
    kw = dict(kernel=ExponentialTemporal(1.0, 0.3), coupling=[[1.0, 0.1], [0.4, 1.0]])
    for extra in ({}, {"mode": "random_rate", "target": 1}):
        with pytest.raises(ValueError, match="matrix coupling"):
            EmbeddedProcess(**kw, **extra)
    with pytest.raises(ValueError, match="physical"):
        compile_markov_embedding([[-1.0]], [[1.0]], EmbeddedProcess(**kw, mode="random_rate"))
    for bad in ([[0.2, 1.0]], [[True]], [[np.inf]], [["0.1"]]):
        with pytest.raises(ValueError, match="matrix coupling"):
            EmbeddedProcess(ExponentialTemporal(1.0, 0.3), coupling=bad, mode="random_rate")

def test_colored_rate_example_covariance_matches_conditioned_gaussian():
    import jsonschema

    from sft_wick.workflow.schema import workflow_json_schema

    path = Path(__file__).resolve().parents[1] / "examples" / "demo5" / "colored_rate.yaml"
    jsonschema.validate(yaml.safe_load(path.read_text()), workflow_json_schema())
    _, table = run_workflow(load_workflow_config(path))
    # At fixed OU history the additive impulse propagates by
    # exp(-gamma*u) exp(G * integral_0^u eta). Gaussian averaging yields
    # V(u)/2 * (G. + .G^T)^2 Q as the order-two integrand.
    g = np.array([[0.35, -0.42], [0.27, 0.18]])
    q = np.array([[0.72, 0.16], [0.16, 0.43]])
    kernel = quad(lambda u: np.exp(-1.2 * u) * (u + 0.2 * np.expm1(-u / 0.2)),
                  0, 1.5)[0]
    first = g @ q + q @ g.T
    expected = 0.5 * kernel * (g @ first + first @ g.T)
    values = table.loc[table["order"] == 2]
    assert len(values) == 3
    for row in values.itertuples():
        assert row.value == pytest.approx(expected[row.a, row.b], rel=1e-9, abs=0)
