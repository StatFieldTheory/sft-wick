"""Independent common-time integrals and guarded workflow absorption."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace
import itertools
import math
import pickle
import warnings

import numpy as np
import pytest
from scipy.integrate import quad

import sft_wick as sw
from sft_wick.white_cumulant import OUWhiteCumulant


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("complex_amplitude", [False, True])
def test_scalar_matches_independent_common_time_quadrature(order, complex_amplitude):
    amplitude = np.arange(1, 2**order + 1, dtype=float).reshape((2,) * order) / 7
    amplitude.flat[1] = -0.7
    if complex_amplitude:
        amplitude = amplitude + 1j * amplitude[::-1] / 3
    gamma = np.array([0.0, 1.3])
    times = np.linspace(0.4, 1.2, order)
    lower = -0.2
    law = OUWhiteCumulant(amplitude, gamma, t_min=lower)
    values = law([0] * order, times)
    for indices in itertools.product(range(2), repeat=order):
        rates = gamma[list(indices)]
        integral = quad(
            lambda tau: np.exp(-np.sum(rates * (times - tau))),
            lower,
            times.min(),
            epsabs=1e-14,
            epsrel=1e-13,
        )[0]
        assert values[indices] == pytest.approx(
            amplitude[indices] * integral, rel=1e-13, abs=0.0
        )


@pytest.mark.parametrize("chunk", [1, 2, 128])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_vectorized_scalar_parity_and_zero_time_support(chunk, order):
    rng = np.random.default_rng(241 + order)
    amplitude = rng.normal(size=(2,) * order) + 1j * rng.normal(size=(2,) * order)
    gamma = [0.0, 2.1]
    times = rng.uniform(-0.2, 2, size=(order, 13))
    times[:, 0] = 0.2
    times[0, 1] = 0.1
    law = OUWhiteCumulant(amplitude, gamma, t_min=0.2, chunk_size=chunk)
    batch = law(np.zeros(times.shape), times)
    scalar = np.stack([law([0] * order, times[:, k]) for k in range(times.shape[1])])
    np.testing.assert_array_equal(batch, scalar)
    np.testing.assert_array_equal(batch[:2], np.zeros_like(batch[:2]))
    assert (
        law(np.empty((order, 0)), np.empty((order, 0))).shape == (0,) + amplitude.shape
    )


def _high_precision_value(amplitude, gamma, times, lower):
    mp = pytest.importorskip("mpmath")
    with mp.workdps(180):
        ts = [mp.mpf(float(value)) for value in times]
        gs = [mp.mpf(float(value)) for value in gamma]
        delta = min(ts) - mp.mpf(float(lower))
        if delta <= 0:
            return 0.0
        decay = sum(rate * (time - min(ts)) for rate, time in zip(gs, ts))
        total_rate = sum(gs)
        integral = (
            delta if not total_rate else -mp.expm1(-total_rate * delta) / total_rate
        )
        return float(mp.mpf(float(amplitude)) * mp.exp(-decay) * integral)


_EXTREME_CASES = [
    # G overflows, but amplitude/G is comfortably finite.
    (1e308, [1e308] * 3, [1.0] * 3, 0.0),
    # The positive time difference overflows before multiplication by a tiny rate.
    (1e-308, [1e-308] * 2, [-1e308, 1e308], -1.5e308),
    # The time integral itself exceeds float64; the final product does not.
    (1e-308, [0.0] * 3, [1e308] * 3, -1e308),
    (1e-308, [np.nextafter(0.0, 1.0)] * 3, [1e308] * 3, -1e308),
    # exp(-decay) underflows before multiplication by a large amplitude.
    (1e308, [1.0] * 3, [1000.0, 1.0, 1.0], 0.0),
    # Both a tiny rate and its tiny argument must keep psi(0)=1.
    (1e308, [1e-308] * 2, [1e-308, 1e-308], 0.0),
    (1.0, [0.0] * 2, [1e-300, 1e-300], 0.0),
    # A positive subnormal integral/factor has lost relative accuracy before
    # the large amplitude restores a normal-sized final result.
    (1e308, [1e308] * 2, [1e-307, 4.5e-307], 0.0),
    # Subnormal exp(-decay), compensated by a large normal integral, must
    # also use logs even when the product factor is normal and positive.
    (1.0, [1e-305] * 2, [1e305, 7.41e307], 0.0),
    # A genuine unrepresentable small result rounds to zero.
    (1e-300, [1.0] * 2, [1000.0, 1.0], 0.0),
]


@pytest.mark.parametrize("amplitude,gamma,times,lower", _EXTREME_CASES)
def test_extreme_scales_match_180_digit_reference(amplitude, gamma, times, lower):
    order = len(times)
    # Two components exercise tensor broadcasting and independently tiny rates.
    tensor = np.full((2,) * order, amplitude)
    rates = np.array([gamma[0], gamma[0]])
    law = OUWhiteCumulant(tensor, rates, t_min=lower)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = law([0] * order, times)
    expected = _high_precision_value(amplitude, gamma, times, lower)
    if expected == 0:
        np.testing.assert_array_equal(actual, np.zeros_like(actual))
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0.0)


def test_extreme_mixed_rates_keep_small_component_rate():
    tiny = np.nextafter(0.0, 1.0)
    law = OUWhiteCumulant(np.full((2,) * 3, 1e-308), [tiny, 1e308], t_min=-1e308)
    actual = law([0] * 3, [1e308] * 3)
    expected = _high_precision_value(1e-308, [tiny] * 3, [1e308] * 3, -1e308)
    assert actual[0, 0, 0] == pytest.approx(expected, rel=2e-12, abs=0.0)
    assert actual[1, 1, 1] == 0


@pytest.mark.parametrize("under", ["warn", "raise"])
def test_finite_ordinary_integral_handles_subnormal_psi(under):
    bound = np.finfo(float).max / 8
    law = OUWhiteCumulant(np.ones((1, 1)), 1.0, t_min=-bound)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(under=under):
            actual = law([0, 0], [bound, bound])
    # The saturated common-time integral is 1/(gamma_0+gamma_0).
    assert actual[0, 0] == pytest.approx(0.5, rel=2e-15, abs=0.0)


def test_ordinary_time_bound_keeps_decay_finite_for_rates_below_one():
    law = OUWhiteCumulant(np.ones((1,) * 3), 0.9)
    with np.errstate(over="raise", invalid="raise"):
        largest_decay = law._rate_sum * (2 * law._ordinary_time_bound)
    assert np.all(np.isfinite(largest_decay))
    assert np.max(largest_decay) < np.finfo(float).max


def test_constructor_keeps_mixed_extreme_rate_logs_stable_under_strict_flags():
    tiny = np.nextafter(0.0, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            law = OUWhiteCumulant(np.ones((2,) * 3), [tiny, 1e308])
            actual = law([0] * 3, [1.0] * 3)
    assert actual[0, 0, 0] == pytest.approx(1.0, rel=2e-15, abs=0.0)
    assert np.all(np.isfinite(actual))


def test_order_must_allow_a_numpy_batch_axis():
    one = np.ones(1)
    dimension = 1
    while True:
        try:
            one.reshape((1,) * (dimension + 1))
        except ValueError:
            break
        dimension += 1
    with pytest.raises(ValueError, match="batch dimension"):
        OUWhiteCumulant(one.reshape((1,) * dimension), 0.0)
    supported = dimension - 1
    law = OUWhiteCumulant(one.reshape((1,) * supported), 0.0)
    actual = law([0] * supported, [1.0] * supported)
    assert actual.shape == (1,) * supported
    assert actual.reshape(-1)[0] == 1.0


@pytest.mark.parametrize("magnitude", [1e308, np.finfo(float).max])
@pytest.mark.parametrize("width", [0.5, 1.0])
def test_complex_components_do_not_require_finite_magnitude(magnitude, width):
    law = OUWhiteCumulant(np.full((1, 1), complex(magnitude, magnitude)), 0.0)
    actual = law([0, 0], [width, width])
    assert np.isfinite(actual).all()
    assert actual[0, 0] == complex(magnitude * width, magnitude * width)


@pytest.mark.parametrize("amplitude", [1.0, -1.0, 1j])
def test_genuine_final_overflow_is_explicit(amplitude):
    law = OUWhiteCumulant(np.full((1, 1), amplitude), 0.0, t_min=-1e308)
    with pytest.raises(OverflowError, match="float64"):
        law([0, 0], [1e308, 1e308])


def test_near_float64_max_overflow_is_not_hidden_by_log_rounding():
    law = OUWhiteCumulant(np.full((1, 1), np.finfo(float).max), 0.0)
    with pytest.raises(OverflowError, match="float64"):
        law([0, 0], [np.nextafter(1.0, 2.0)] * 2)


def test_exact_zero_rate_difference_overflow_detects_maximum_boundary():
    maximum = np.finfo(float).max
    safe = OUWhiteCumulant(np.full((1, 1), 0.5), 0.0, t_min=-maximum)
    assert safe([0, 0], [maximum, maximum])[0, 0] == maximum
    unsafe = OUWhiteCumulant(
        np.full((1, 1), np.nextafter(0.5, 1.0)), 0.0, t_min=-maximum
    )
    with pytest.raises(OverflowError, match="float64"):
        unsafe([0, 0], [maximum, maximum])


@pytest.mark.parametrize("lower", [-1.0, -1e292])
def test_zero_rate_range_check_uses_exact_width_not_rounded_log(lower):
    maximum = np.finfo(float).max
    law = OUWhiteCumulant(np.ones((1, 1)), 0.0, t_min=lower)
    with pytest.raises(OverflowError, match="float64"):
        law([0, 0], [maximum, maximum])


def test_zero_rate_boundary_preserves_just_below_maximum():
    maximum = np.finfo(float).max
    amplitude = np.nextafter(0.5, 0.0)
    law = OUWhiteCumulant(np.full((1, 1), amplitude), 0.0, t_min=-maximum)
    expected = _high_precision_value(amplitude, [0, 0], [maximum, maximum], -maximum)
    assert law([0, 0], [maximum, maximum])[0, 0] == expected


def test_zero_amplitude_shortcut_is_exact_even_for_extreme_time_scales():
    law = OUWhiteCumulant(np.zeros((2,) * 3, dtype=complex), [0.0, 1e308], t_min=-1e308)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = law([0] * 3, [1e308] * 3)
    np.testing.assert_array_equal(actual, np.zeros_like(actual))


def test_constructor_snapshots_and_copy_pickle_preserve_byte_backing():
    amplitude = np.arange(8).reshape(2, 2, 2).astype(complex)
    gamma = np.array([0.2, 0.7])
    law = OUWhiteCumulant(
        amplitude, gamma, t_min=0.3, chunk_size=3, max_output_bytes=4096
    )
    before = law([0] * 3, [0.8, 1.3, 0.9])
    amplitude[:] = 100
    gamma[:] = 100
    assert copy.copy(law) is law
    assert copy.deepcopy(law) is law
    restored = pickle.loads(pickle.dumps(law))
    for target in (law, restored):
        np.testing.assert_array_equal(target([0] * 3, [0.8, 1.3, 0.9]), before)
        assert target.chunk_size == 3 and target.max_output_bytes == 4096
        for values in (
            target.amplitude,
            target.gamma,
            target._rate_sum,
            target._log_rate_sum,
            target._log_amplitude_real,
            target._log_amplitude_imag,
        ):
            with pytest.raises(ValueError):
                values.setflags(write=True)
        with pytest.raises(FrozenInstanceError):
            target.t_min = 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"amplitude": [1, 2]},
        {"amplitude": np.ones((2, 3))},
        {"amplitude": np.empty((0, 0))},
        {"amplitude": [[np.nan]]},
        {"amplitude": [[1 + np.inf * 1j]]},
        {"amplitude": [[True]]},
        {"gamma": -1},
        {"gamma": [0, 1]},
        {"gamma": [np.nan]},
        {"gamma": 1j},
        {"gamma": True},
        {"t_min": np.inf},
        {"t_min": [0]},
        {"chunk_size": True},
        {"chunk_size": 0},
        {"chunk_size": 1.5},
        {"max_output_bytes": 0},
        {"max_output_bytes": False},
    ],
)
def test_invalid_constructor_contracts(kwargs):
    parameters = {"amplitude": np.ones((1, 1)), "gamma": 0.7}
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        OUWhiteCumulant(**parameters)


@pytest.mark.parametrize(
    "times",
    [
        [0.1],
        np.ones((3, 1)),
        np.ones((2, 1, 1)),
        [np.nan, 0.1],
        [np.inf, 0.1],
        [0.1j, 0.1],
        [True, True],
    ],
)
def test_invalid_time_contracts(times):
    law = OUWhiteCumulant(np.ones((1, 1)), 0.7)
    with pytest.raises(ValueError, match="t_list"):
        law([0, 0], times)


@pytest.mark.parametrize("positions", [None, [0], [0, 0, 0]])
def test_spatial_leg_count_is_validated(positions):
    law = OUWhiteCumulant(np.ones((1, 1)), 0.7)
    with pytest.raises(ValueError, match="n_list"):
        law(positions, [0.3, 0.4])


def test_spatial_values_are_static_and_may_be_vector_positions():
    law = OUWhiteCumulant(np.ones((1, 1)), 0.7)
    np.testing.assert_array_equal(
        law([[0, 1, 2], [3, 4, 5]], [0.3, 0.4]), law([0, 0], [0.3, 0.4])
    )


def test_explicit_output_budget_and_chunk_determinism():
    with pytest.raises(MemoryError, match="amplitude"):
        OUWhiteCumulant(np.ones((2, 2)), 0.7, max_output_bytes=16)
    law = OUWhiteCumulant(np.ones((2, 2)), 0.7, max_output_bytes=32)
    assert law([0, 0], [0.3, 0.4]).nbytes == 32
    with pytest.raises(MemoryError, match="batch"):
        law(np.zeros((2, 2)), np.full((2, 2), 0.5))


def _base_system(gamma=(0.7, 1.3), lower=0.2, *, field_size=2):
    return sw.System(
        field=sw.FieldSpec("phi", field_size),
        linear=sw.DiagonalA(gamma=list(gamma)),
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.2, sigma_t=0.3),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            )
        ),
        t_min=lower,
    )


def test_factory_uses_actual_allclose_collapsed_rates_and_bare_phase():
    system = _base_system(gamma=(1.0, 1.0 + 1e-7))
    amplitude = np.arange(8).reshape((2,) * 3).astype(float)
    law = OUWhiteCumulant.from_system(amplitude, system)
    np.testing.assert_array_equal(law.gamma, [1.0, 1.0])
    vertex = law.as_vertex("K", system=system)
    assert vertex.order == 3 and vertex.already_R_contracted
    assert vertex.coupling_vectorized and not vertex.equal_time
    assert vertex.coupling is law and law.has_coincident_time_kinks
    times = [1.0] * 3
    np.testing.assert_array_equal(
        vertex.msr_coupling([0] * 3, times), vertex.msr_factor * law([0] * 3, times)
    )
    with pytest.raises(ValueError, match="lowered R"):
        OUWhiteCumulant(amplitude, [1.0, 1.0 + 1e-7], t_min=0.2).as_vertex(
            "K", system=system
        )


@pytest.mark.parametrize(
    "modification", ["gamma", "t_min", "components", "explicit", "callable", "matrix"]
)
def test_factory_rejects_mismatched_or_unsupported_system(modification):
    system = _base_system()
    law = OUWhiteCumulant.from_system(np.ones((2,) * 3), system)
    if modification == "gamma":
        system = replace(system, linear=sw.DiagonalA([0.8, 1.3]))
    elif modification == "t_min":
        system = replace(system, t_min=0.3)
    elif modification == "components":
        system = _base_system(gamma=(0.7,), field_size=1)
    elif modification == "explicit":
        system = replace(system, explicit_R=lambda t, s: np.exp(s - t))
    elif modification == "callable":
        system = replace(system, linear=sw.DiagonalA(lambda t: np.array([0.7, 1.3])))
    else:
        system = replace(
            system, linear=sw.ExplicitR(lambda t, s: np.eye(2), iso_R=False)
        )
    with pytest.raises(ValueError):
        law.as_vertex("K", system=system)


def test_factory_rejects_overridden_typed_lowering_and_malformed_rates():
    class CustomDiagonal(sw.DiagonalA):
        def build_R_callable(self):
            return lambda t, s: np.eye(2)

    for linear in (
        CustomDiagonal([0.7, 1.3]),
        sw.DiagonalA([0.7, 1.3, 2.0]),
        sw.DiagonalA([np.inf, np.inf]),
        sw.DiagonalA([-0.7, 1.3]),
    ):
        with pytest.raises(ValueError):
            OUWhiteCumulant.from_system(
                np.ones((2,) * 3), replace(_base_system(), linear=linear)
            )


def _paired_systems(amplitude, gamma=(0.7, 1.3), lower=0.2, local=False):
    base = _base_system(gamma=gamma, lower=lower)
    if local:
        tensor = np.array([[[0.2, -0.1], [0.3, 0.4]], [[-0.3, 0.2], [0.5, -0.1]]])
        base = replace(base, vertices=(sw.LocalVertex("F", coupling=tensor),))
    analytic = OUWhiteCumulant.from_system(amplitude, base)
    raw = replace(
        base,
        nonlocal_vertices=(
            sw.NonLocalVertex(
                "K", order=amplitude.ndim, coupling=amplitude, equal_time=True
            ),
        ),
    )
    absorbed = replace(base, nonlocal_vertices=(analytic.as_vertex("K", system=base),))
    return raw, absorbed


def _workflow_value(
    system,
    observable,
    components,
    order,
    vertex_type,
    *,
    method,
    times=None,
    n_samples=2**13,
):
    expansion = system.expand(observable, orders=[order])
    props = system.propagators(
        t_max=1.6,
        n_grid_t=8,
        c_closed_form="auto",
        c_closed_form_only=True,
        progress=False,
    )
    positions = {spec.split("(")[1].rstrip(")"): 0.0 for spec in observable}
    result = expansion.evaluate(
        props,
        positions=positions,
        t_final=1.1,
        component_pair=components,
        orders=[order],
        vertex_types=[vertex_type],
        method=method,
        n_gauss=12,
        n_samples=n_samples,
        seed=24,
        external_times=times,
    )
    return result.total, expansion


@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized"])
@pytest.mark.parametrize("gamma", [(0.7, 1.3), (0.9, 0.9), (0.0, 0.0)])
def test_actual_k_only_workflow_reduces_one_common_time_dimension(method, gamma):
    amplitude = np.arange(1, 9).reshape((2,) * 3) / 10
    raw, absorbed = _paired_systems(amplitude, gamma=gamma)
    obs = ("phi_a(x)", "phi_b(y)", "phi_c(z)")
    times = {"x": 0.8, "y": 1.1, "z": 0.9}
    raw_value, raw_exp = _workflow_value(
        raw, obs, (0, 1, 1), 1, "K", method=method, times=times
    )
    exact_value, exact_exp = _workflow_value(
        absorbed, obs, (0, 1, 1), 1, "K", method=method, times=times
    )
    assert abs(exact_value) > 0.01
    tolerance = 2e-8 if method == "qmc_vectorized" else 2e-12
    assert raw_value == pytest.approx(exact_value, rel=tolerance, abs=0.0)
    for diagram in raw_exp.diagrams(1):
        assert diagram.r_absorbed_pairs == ()
    for diagram in exact_exp.diagrams(1):
        assert len(diagram.r_absorbed_pairs) == 3
    # Dimension reduction is one per WHITE vertex, not three independent times.
    from sft_wick.evaluate import analyze_spatial

    raw_dimensions = [
        len(analyze_spatial(dt).time_integration_vars) for dt in raw_exp.diagrams(1)
    ]
    exact_dimensions = [
        len(analyze_spatial(dt).time_integration_vars) for dt in exact_exp.diagrams(1)
    ]
    assert set(raw_dimensions) == {1} and set(exact_dimensions) == {0}


@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized"])
def test_actual_mixed_f_k_diagrams_match_raw_equal_time_reference(method):
    amplitude = np.arange(1, 9).reshape((2,) * 3) / 7
    raw, absorbed = _paired_systems(amplitude, local=True)
    obs = ("phi_a(x)", "phi_b(y)")
    raw_value, _ = _workflow_value(raw, obs, (0, 1), 2, "FK", method=method)
    exact_value, _ = _workflow_value(absorbed, obs, (0, 1), 2, "FK", method=method)
    assert abs(exact_value) > 0.001
    tolerance = 3e-4 if method == "qmc_vectorized" else 2e-10
    assert raw_value == pytest.approx(exact_value, rel=tolerance, abs=0.0)


@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized"])
def test_actual_repeated_k_preserves_copy_and_component_routes(method):
    amplitude = np.arange(1, 9).reshape((2,) * 3) / 7
    raw, absorbed = _paired_systems(amplitude)
    obs = tuple(f"phi_{letter}({letter})" for letter in "abcdef")
    times = dict(zip("abcdef", [0.6, 0.9, 1.0, 0.7, 1.1, 0.8]))
    raw_value, raw_exp = _workflow_value(
        raw, obs, (0, 1, 1, 0, 0, 1), 2, "K", method=method, times=times
    )
    exact_value, exact_exp = _workflow_value(
        absorbed, obs, (0, 1, 1, 0, 0, 1), 2, "K", method=method, times=times
    )
    assert abs(exact_value) > 0.01
    tolerance = 3e-5 if method == "qmc_vectorized" else 2e-12
    assert raw_value == pytest.approx(exact_value, rel=tolerance, abs=0.0)
    # The absorbed graph may merge drawings while preserving each copy's
    # routed coupling tuple. Compare the nonzero sums, not the drawing count.
    assert len(raw_exp.diagrams(2)) == 10
    assert len(exact_exp.diagrams(2)) >= 1
    assert all(len(diagram.r_absorbed_pairs) == 6 for diagram in exact_exp.diagrams(2))


@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized", "nquad"])
def test_actual_m2_m4_phase_and_routes_match_raw_white_integral(order, method):
    amplitude = np.arange(1, 2**order + 1).reshape((2,) * order) / 7
    raw, absorbed = _paired_systems(amplitude)
    legs = "abcd"[:order]
    observable = tuple(f"phi_{letter}({letter})" for letter in legs)
    components = (0, 1, 1, 0)[:order]
    times = dict(zip(legs, [0.8, 1.1, 0.9, 1.0][:order]))
    raw_value, _ = _workflow_value(
        raw, observable, components, 1, "K", method=method, times=times
    )
    exact_value, _ = _workflow_value(
        absorbed, observable, components, 1, "K", method=method, times=times
    )
    assert exact_value > 0.01
    tolerance = 2e-8 if method == "qmc_vectorized" else 2e-12
    assert raw_value == pytest.approx(exact_value, rel=tolerance, abs=0.0)
    # A static bare tensor is summed over its occurrence assignments: this
    # independently computes the normalized permutation average and temporal
    # integral without using any diagram counts or implementation helpers.
    bare_average = sum(
        amplitude[tuple(permutation)]
        for permutation in itertools.permutations(components)
    ) / math.factorial(order)
    gamma = [raw.linear.gamma[component] for component in components]
    expected = _high_precision_value(
        bare_average, gamma, list(times.values()), raw.t_min
    )
    assert exact_value == pytest.approx(expected, rel=1e-13, abs=0.0)


@pytest.mark.parametrize("method", ["nquad", "qmc_scalar"])
def test_adaptive_and_scalar_qmc_dispatch_pure_k_zero_dim(method):
    amplitude = np.arange(1, 9).reshape((2,) * 3) / 7
    raw, absorbed = _paired_systems(amplitude)
    observable = ("phi_a(x)", "phi_b(y)", "phi_c(z)")
    times = {"x": 0.8, "y": 1.1, "z": 0.9}
    raw_value, _ = _workflow_value(
        raw, observable, (0, 1, 1), 1, "K", method=method, times=times, n_samples=2**11
    )
    exact_value, _ = _workflow_value(
        absorbed,
        observable,
        (0, 1, 1),
        1,
        "K",
        method=method,
        times=times,
        n_samples=2**11,
    )
    assert exact_value > 0.01
    tolerance = 2e-7 if method == "qmc_scalar" else 2e-12
    assert raw_value == pytest.approx(exact_value, rel=tolerance, abs=0.0)


@pytest.mark.parametrize("method", ["nquad", "qmc_scalar"])
def test_adaptive_and_scalar_qmc_dispatch_nonzero_mixed_f_k(method):
    amplitude = np.arange(1, 9).reshape((2,) * 3) / 7
    raw, absorbed = _paired_systems(amplitude, local=True)
    observable = ("phi_a(x)", "phi_b(y)")
    raw_value, _ = _workflow_value(
        raw, observable, (0, 1), 2, "FK", method=method, n_samples=2**11
    )
    exact_value, _ = _workflow_value(
        absorbed, observable, (0, 1), 2, "FK", method=method, n_samples=2**11
    )
    assert abs(exact_value) > 0.001
    tolerance = 1e-3 if method == "qmc_scalar" else 2e-10
    assert raw_value == pytest.approx(exact_value, rel=tolerance, abs=0.0)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_actual_single_component_factory_is_supported(order):
    base = _base_system(gamma=(0.7,), field_size=1)
    law = OUWhiteCumulant.from_system(np.full((1,) * order, 1.2), base)
    system = replace(base, nonlocal_vertices=(law.as_vertex("K", system=base),))
    observable = tuple(f"phi({letter})" for letter in "abcd"[:order])
    value, _ = _workflow_value(
        system, observable, None, 1, "K", method="gauss_legendre"
    )
    expected = _high_precision_value(1.2, [0.7] * order, [1.1] * order, 0.2)
    assert value == pytest.approx(expected, rel=1e-13, abs=0.0)


def test_factory_must_revalidate_mutated_system_rate_inputs():
    rates = np.array([0.7, 1.3])
    base = replace(_base_system(), linear=sw.DiagonalA(rates))
    law = OUWhiteCumulant.from_system(np.ones((2,) * 3), base)
    rates[1] = 2.0
    np.testing.assert_array_equal(law.gamma, [0.7, 1.3])
    with pytest.raises(ValueError, match="lowered R"):
        law.as_vertex("K", system=base)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_large_finite_products_near_coarse_admission_bound_match_mp(order):
    amplitude, rate = 1e308, 1e307
    bound = np.finfo(float).max / rate / (4 * order)
    lower, upper = -0.9 * bound, 0.9 * bound
    law = OUWhiteCumulant(np.full((2,) * order, amplitude), rate, t_min=lower)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = law([0] * order, [upper] * order)
    expected = _high_precision_value(amplitude, [rate] * order, [upper] * order, lower)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0.0)


def test_mixed_normal_and_extreme_batches_preserve_scalar_accuracy_across_chunks():
    amplitude = np.arange(1, 9).reshape((2,) * 3) / 7
    times = np.array([[0.8, 1e308], [1.0, 1e308], [0.9, 1e308]])
    for chunk in (1, 128):
        law = OUWhiteCumulant(amplitude, [0.7, 1.3], t_min=0.2, chunk_size=chunk)
        batch = law(np.zeros_like(times), times)
        scalar = np.stack([law([0] * 3, times[:, sample]) for sample in range(2)])
        np.testing.assert_allclose(batch, scalar, rtol=2e-12, atol=0.0)
