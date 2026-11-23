"""Matrix Ornstein–Uhlenbeck propagation and node-local polynomial drift."""

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.linalg import expm

from sft_wick.network import NetworkSystem


@pytest.mark.parametrize("start", ["zero", "stationary"])
def test_directed_chain_against_white_noise_integral(start):
    # Directed feed-forward chain with a defective drift. Reversing either
    # response leg changes the unequal-time covariance.
    laplacian = [[0, 0, 0], [-1, 1, 0], [0, -1, 1]]
    noise = np.array([[1.0, 0.25, 0], [0.25, 0.8, 0.1], [0, 0.1, 0.55]])
    model = NetworkSystem(laplacian, 0.7, 0.6, noise, t_min=0.3, start=start)
    drift = -0.7 * np.eye(3) - 0.6 * np.asarray(laplacian)
    later, earlier = 1.04, 0.68
    # Propagating an impulse on the root changes the sink. An impulse on
    # the sink never changes the root.
    response = model.system.linear.R_time(later, earlier)
    assert response[2, 0] > 0.01
    assert response[0, 2] == 0
    np.testing.assert_allclose(response, expm((later - earlier) * drift), atol=1e-15)
    start_time = -40 if start == "stationary" else model.t_min
    expected = quad_vec(
        lambda u: expm((later - u) * drift) @ noise @ expm((earlier - u) * drift.T),
        start_time,
        earlier,
        epsabs=1e-13,
    )[0]
    cache = model.propagators(later).cache
    for t, s, target in [(later, earlier, expected), (earlier, later, expected.T)]:
        np.testing.assert_allclose(
            cache.C_value(0.0, t, 0.0, s), target, rtol=0, atol=2e-13
        )


def test_undamped_zero_mode_and_correlated_noise():
    laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
    q = np.array([[1.0, 0.3], [0.3, 0.5]])
    model = NetworkSystem(laplacian, 0, 0.85, q, t_min=-0.5)
    t, s = 1.1, 0.3
    drift = -0.85 * laplacian
    expected = quad_vec(
        lambda u: expm((t - u) * drift) @ q @ expm((s - u) * drift.T), -0.5, s
    )[0]
    np.testing.assert_allclose(
        model.propagators(t).cache.C_value(0.0, t, 0.0, s), expected, rtol=0, atol=2e-14
    )


def test_graph_diffusion_conserves_each_component_at_long_times():
    eye = np.eye(8)
    ring = 2 * eye - np.roll(eye, 1, axis=1) - np.roll(eye, -1, axis=1)
    model = NetworkSystem(ring, 0, 1, 1)
    stop = 1e16
    # The ring's uniform mode has zero decay and unit noise per site.
    R = model.system.linear.R_time(stop, 0.0)
    C = model.propagators(stop).cache.C_value(0.0, stop, 0.0, stop)
    np.testing.assert_allclose(R.sum(axis=1), 1.0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(C.sum(axis=1) / stop, 1.0, atol=1e-14, rtol=0)


def test_cubic_ring_at_distinct_times_against_tagged_ito_moments():
    sites = np.arange(8)
    eye = np.eye(8)
    laplacian = 2 * eye - np.roll(eye, 1, axis=1) - np.roll(eye, -1, axis=1)
    model = NetworkSystem(
        laplacian,
        0.68,
        0.34,
        0.9 + 0.3 * np.cos(0.7 * sites),
        local_drift={3: 0.31 * np.cos(0.9 * sites) - 0.05},
        t_min=0.3,
    )
    expansion = model.expand(["phi_a(x)", "phi_b(y)"], orders=[0, 1, 2])
    props = model.propagators(0.92)
    # The independent Itô hierarchy freezes site 2 at the earlier time
    # and evolves the other endpoint. The two sites are graph distance 2.
    expected = [0.004678961497300946, 0.00031444125971130557, 0.000057276372903088235]
    for order, ref in enumerate(expected):
        value = expansion.evaluate(
            props,
            positions={"x": 0.0, "y": 0.0},
            external_times={"x": 0.92, "y": 0.77},
            t_final=0.92,
            component_pair=(0, 2),
            orders=[order],
            method="gauss_legendre",
            n_gauss=8,
            contraction_backend="einsum",
        ).total
        assert value == pytest.approx(ref, rel=0, abs=1e-12)
        assert all(len(d.summation_indices) == order for d in expansion.diagrams(order))


@pytest.mark.parametrize(
    "kwargs,problem",
    [
        ({"laplacian": [[1, -1], [-1, 0]]}, "row sums"),
        ({"laplacian": [[-1, 1], [0, 0]]}, "off-diagonals"),
        ({"noise": [[1.0, 2.0], [2.0, 1.0]]}, "semidefinite"),
        ({"gamma": 0, "start": "stationary"}, "gamma"),
        ({"local_drift": {2: [1, 2, 3]}}, "coefficient"),
    ],
)
def test_network_rejects_inconsistent_inputs(kwargs, problem):
    params = dict(laplacian=[[1, -1], [-1, 1]], gamma=1, diffusion=0.2)
    with pytest.raises(ValueError, match=problem):
        NetworkSystem(**{**params, **kwargs})


@pytest.mark.parametrize("gamma,stop", [(1e-14, 1e14), (0.0, 1e16)])
def test_directed_long_horizon_preserves_the_uniform_mode(gamma, stop):
    graph = [[1.0, -1.0], [-2.0, 2.0]]
    q = np.ones((2, 2))
    model = NetworkSystem(graph, gamma, 1.0, q)
    r = model.system.linear.R_time(stop, 0.0)
    decay = np.exp(-gamma * stop)
    np.testing.assert_allclose(
        r, decay * np.array([[2 / 3, 1 / 3], [2 / 3, 1 / 3]]), rtol=2e-13
    )
    variance = -np.expm1(-2 * gamma * stop) / (2 * gamma) if gamma else stop
    np.testing.assert_allclose(
        model.propagators(stop).cache.C_value(0, stop, 0, stop),
        variance * q,
        rtol=2e-13,
    )
    if gamma:
        steady = NetworkSystem(graph, gamma, 1.0, q, start="stationary")
        np.testing.assert_allclose(
            steady.system.initial_state.covariance, q / (2 * gamma), rtol=2e-13
        )
    # The signed noise vector is the decaying right eigenvector. Its
    # stationary mean-mode forcing is zero, even over a very long horizon.
    q = np.outer([1.0, -2.0], [1.0, -2.0])
    signed = NetworkSystem(graph, gamma, 1.0, q)
    np.testing.assert_allclose(
        signed.propagators(stop).cache.C_value(0, stop, 0, stop),
        q / (2 * (3 + gamma)),
        rtol=2e-13,
    )


@pytest.mark.parametrize("diffusion", [0.0, 1e-12])
def test_stationary_covariance_from_one_forced_site(diffusion):
    eye = np.eye(4)
    laplacian = 2 * eye - np.roll(eye, 1, axis=1) - np.roll(eye, -1, axis=1)
    model = NetworkSystem(
        laplacian, 1.0, diffusion, [1.0, 0.0, 0.0, 0.0], start="stationary"
    )
    # Integrate each entry of R Q R^T in physical coordinates. Scale out
    # powers of diffusion so the reference resolves the weakest sites.
    distance = np.array([0, 1, 2, 1])
    scale = diffusion ** distance if diffusion else np.ones(4)
    drift = -eye - diffusion * laplacian

    def integrand(t):
        col = expm(t * drift)[:, 0] / scale
        return np.outer(col, col)

    expected = quad_vec(integrand, 0.0, 45.0, epsabs=1e-12)[0]
    q0 = model.system.initial_state.covariance
    np.testing.assert_allclose(
        q0 / np.outer(scale, scale), expected, rtol=2e-9, atol=1e-14
    )


def test_reducible_directed_graph_has_two_independent_conservation_laws():
    # The two absorbing source sites drive independent noises. A third
    # site tracks their weighted mean with decay 3.
    L = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, -2.0, 3.0]]
    model = NetworkSystem(L, 0.0, 1.0, [1.0, 1.0, 0.0])
    time = 1e16
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1 / 3, 2 / 3, 0.0]])
    np.testing.assert_allclose(
        model.system.linear.R_time(time, 0), expected, atol=1e-15
    )
    # Relative corrections to the Brownian covariance/time are O(1/time).
    cov = model.propagators(time).cache.C_value(0.0, time, 0.0, time)
    np.testing.assert_allclose(
        cov / time, expected @ expected.T, atol=1e-15, rtol=1e-14
    )


def test_symmetric_response_is_independent_of_which_sites_are_forced():
    eye = np.eye(4)
    laplacian = 2 * eye - np.roll(eye, 1, axis=1) - np.roll(eye, -1, axis=1)
    for noise in (1.0, [1.0, 0.0, 0.0, 0.0]):
        model = NetworkSystem(laplacian, 0.0, 1.0, noise)
        np.testing.assert_allclose(
            model.system.linear.R_time(1e16, 0.0), 0.25, atol=1e-14
        )


@pytest.mark.parametrize("weak", [2.0**-46, 1e-9])
def test_weak_bridge_resolves_consensus_and_cluster_relaxation(weak):
    strong = 3.0
    laplacian = np.array(
        [[strong + weak, -strong, -weak], [-strong, strong, 0], [-weak, 0, weak]]
    )
    # The two nonzero roots have sum 2(a+b) and product 3ab. Use their
    # spectral projectors so the exact uniform mode never mixes with them.
    fast = strong + weak + np.sqrt(strong**2 - strong * weak + weak**2)
    slow = 3 * strong * weak / fast
    uniform = np.ones((3, 3)) / 3
    contrast = np.eye(3) - uniform
    fast_proj = (laplacian - slow * contrast) / (fast - slow)
    slow_proj = contrast - fast_proj
    model = NetworkSystem(laplacian, 0.0, 1.0)
    cache = model.propagators(100 / weak).cache
    for time in (1 / weak, 100 / weak):
        response = uniform + np.exp(-slow * time) * slow_proj
        covariance = uniform * time
        for rate, projector in ((slow, slow_proj), (fast, fast_proj)):
            covariance += (-np.expm1(-2 * rate * time) / (2 * rate)) * projector
        actual_r = model.system.linear.R_time(time, 0.0)
        actual_c = cache.C_value(0.0, time, 0.0, time)
        np.testing.assert_allclose(actual_r, response, rtol=2e-13, atol=0)
        np.testing.assert_allclose(actual_c / time, covariance / time, rtol=2e-13)
        np.testing.assert_allclose(actual_r.sum(axis=1), 1.0, rtol=2e-13)
        np.testing.assert_allclose(actual_c.sum(axis=1) / time, 1.0, rtol=2e-13)

    gamma = weak / 10
    stationary = NetworkSystem(laplacian, gamma, 1.0, start="stationary")
    expected = (
        uniform / (2 * gamma)
        + slow_proj / (2 * (gamma + slow))
        + fast_proj / (2 * (gamma + fast))
    )
    np.testing.assert_allclose(
        stationary.system.initial_state.covariance, expected, rtol=2e-13
    )


@pytest.mark.parametrize("start", ["zero", "stationary"])
def test_signed_noise_unequal_times_keep_the_decaying_mode(start):
    gamma = 1e-14
    rate = 3 + gamma
    q = np.outer([1.0, -2.0], [1.0, -2.0])
    model = NetworkSystem([[1, -1], [-2, 2]], gamma, 1.0, q, start=start)
    cache = model.propagators(151.0).cache
    c0 = (
        q / (2 * rate)
        if start == "stationary"
        else q * (-np.expm1(-2 * rate) / (2 * rate))
    )
    for lag in (20.0, 150.0):
        expected = np.exp(-rate * lag) * c0
        got = cache.C_value(0.0, 1.0 + lag, 0.0, 1.0)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)
