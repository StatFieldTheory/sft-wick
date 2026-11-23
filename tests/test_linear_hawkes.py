"""Connected count increments of a finite-history marked Hawkes network."""

from itertools import combinations

import numpy as np
import pytest

from sft_wick.hawkes import LinearHawkes


def test_distinct_count_windows_against_backward_branching_solution():
    model = LinearHawkes(
        baseline=[0.49, 0.73],
        excitation=[[0.15, 0.38], [0.26, 0.14]],
        decay=[1.31, 0.82],
        mark_moments=[1.0, 2.0, 6.0],
        t_min=0.5,
    )
    windows = [(0, 0.70, 1.33), (1, 0.98, 1.80), (0, 1.51, 2.14)]
    # From the independent backward cluster generating function for the same
    # three windows and mean-one exponential mark on both excitation jumps.
    values = [model.count_cumulant(pair) for pair in combinations(windows, 2)]
    values.append(model.count_cumulant(windows))
    np.testing.assert_allclose(
        values,
        [
            0.09985685836834430,
            0.04251311174802172,
            0.14316810868576349,
            0.03930498331105078,
        ],
        rtol=0,
        atol=1e-10,
    )


def test_poisson_counts_in_overlapping_repeated_and_empty_windows():
    model = LinearHawkes(baseline=2.3, excitation=0, decay=0.8, t_min=1.0)
    a, b = (0, 1.0, 2.1), (0, 1.4, 2.4)
    for windows, width in [
        ([a], 1.1),
        ([a, a], 1.1),
        ([a, a, b], 0.7),
        ([a, (0, 2.0, 2.0)], 0.0),
    ]:
        assert model.count_cumulant(windows) == pytest.approx(2.3 * width, abs=1e-12)


def test_poisson_count_checks_final_result_range():
    wide = LinearHawkes(1e-308, 0.0, 1.0, t_min=-1e308)
    assert wide.count_cumulant([(0, -1e308, 1e308)]) == pytest.approx(2.0)

    model = LinearHawkes(1e308, 0.0, 1.0)
    with pytest.raises(ArithmeticError, match="floating-point range"):
        model.count_cumulant([(0, 0.0, 2.0)])


def test_supercritical_node_without_immigration_has_zero_count():
    model = LinearHawkes(0.0, 1.0, 0.1)
    window = (0, 0.0, 1000.0)
    for rank in (1, 2, 3):
        assert model.count_cumulant([window] * rank) == 0.0


@pytest.mark.parametrize(
    "other_baseline, downstream_excitation", [(0.0, 0.0), (0.0, 0.2), (0.1, 0.0)]
)
def test_poisson_readout_ignores_supercritical_nodes_without_a_path_back(
    other_baseline,
    downstream_excitation,
):
    model = LinearHawkes(
        [other_baseline, 0.4],
        [[1.0, downstream_excitation], [0.0, 0.0]],
        [0.1, 1.0],
    )
    window = (1, 0.0, 1000.0)
    for rank in (3, 2, 1):
        assert model.count_cumulant([window] * rank) == 400.0
    if not downstream_excitation:
        independent = (0, 0.0, 1000.0)
        assert model.count_cumulant([window, independent]) == 0.0
        assert model.count_cumulant([window, independent, independent]) == 0.0


def test_count_projection_keeps_noncontiguous_immigrant_ancestor():
    from scipy.integrate import quad

    mu_child, mu_parent, excitation, decay = 0.27, 0.58, 0.63, 0.81
    model = LinearHawkes(
        [mu_child, 0.4, mu_parent],
        [[0.0, 0.0, excitation], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        [decay, 0.1, 1.05],
        t_min=0.2,
    )
    start, stop = 0.2, 1.35

    # An immigrant at the parent creates a Poisson cluster of child events.
    # The raw moments of its count enter each cumulant after Poisson mixing.
    def offspring_moments(time):
        mean = -excitation * np.expm1(-decay * (stop - time)) / decay
        return mean, mean + mean**2, mean + 3 * mean**2 + mean**3

    for rank in (1, 2, 3):
        cluster, _ = quad(lambda time: offspring_moments(time)[rank - 1], start, stop)
        expected = mu_child * (stop - start) + mu_parent * cluster
        value = model.count_cumulant([(0, start, stop)] * rank)
        assert value == pytest.approx(expected, rel=0, abs=1e-10)


def test_joint_cumulant_only_uses_immigrants_that_can_reach_both_counters():
    mu_parent, excitation, decay, stop = 0.58, 0.63, 0.81, 1000.0
    model = LinearHawkes(
        [0.27, 0.1, mu_parent],
        [[0.0, 0.5, excitation], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        [decay, 0.1, 1.05],
    )
    # Only immigrants at node 2 contribute to both this count and node 0.
    expected = mu_parent * excitation / decay * (stop + np.expm1(-decay * stop) / decay)
    assert model.count_cumulant([(0, 0.0, stop), (2, 0.0, stop)]) == pytest.approx(
        expected, rel=0, abs=1e-10
    )


def test_empty_history_covariance_before_and_at_origin():
    model = LinearHawkes(0.7, 0.4, 1.1, t_min=0.5)
    cache = model.propagators(1.0).cache
    pairs = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.5, 0.75), (0.75, 0.75)]
    expected = np.zeros((len(pairs), 2, 2))
    reference = LinearHawkes(0.7, 0.4, 1.1).propagators(0.5).cache
    expected[-1] = reference.C_value(0.0, 0.25, 0.0, 0.25)

    scalar = np.array([cache.C_value(0.0, t, 0.0, s) for t, s in pairs])
    t, s = np.asarray(pairs).T
    batched = cache.C_at_batch(t, s, 0.0, 0.0)
    np.testing.assert_array_equal(scalar, expected)
    np.testing.assert_array_equal(batched, expected)


def test_count_cumulants_are_invariant_under_time_translation():
    windows = [(0, 0.125, 0.625), (0, 0.25, 0.75), (0, 0.125, 0.375)]
    reference = LinearHawkes(0.7, 0.6, 0.6)
    origin = 1e10
    shifted = LinearHawkes(0.7, 0.6, 0.6, t_min=origin)
    shifted_windows = [
        (node, start + origin, stop + origin) for node, start, stop in windows
    ]
    for rank in (1, 2, 3):
        assert shifted.count_cumulant(shifted_windows[:rank]) == pytest.approx(
            reference.count_cumulant(windows[:rank]),
            abs=1e-13,
        )
    assert shifted.count_cumulant(shifted_windows, n_gauss=12) == pytest.approx(
        reference.count_cumulant(windows, n_gauss=12),
        abs=1e-13,
    )
    long_history = LinearHawkes(0.7, 0.6, 0.6, t_min=-1e20)
    with pytest.raises(ArithmeticError, match="elapsed count windows.*precision"):
        long_history.count_cumulant([(0, 0.0, 32768.0), (0, 0.25, 65536.0)])


def test_high_order_jump_tensor_uses_marks_only_on_excitation_axes():
    model = LinearHawkes(0.8, 0.4, 1.2, mark_moments=[2, 8, 48, 384])
    vertices = {vertex.name: vertex for vertex in model.system.km_vertices}
    # A two-excitation, two-count jump samples only E[M²]; counts jump by one.
    linear = np.asarray(vertices["K4rate"].coupling)
    assert linear[0, 1, 0, 1, 0] == pytest.approx(0.4**2 * 8)
    assert linear[0, 0, 0, 0, 0] == pytest.approx(0.4**4 * 384)
    assert np.all(linear[..., 1] == 0)
    constant = vertices["K4mean"].coupling([0.0], [np.array([0.0])])
    assert constant[0, 0, 1, 0, 1] == pytest.approx(0.8 * 0.4**2 * 8)


def test_mean_against_marked_univariate_closed_form():
    # E[lambda(t)] solves l' = (a E[M] - beta) l + beta mu.
    mu, beta, a, mean_mark = 0.7, 1.1, 0.3, 2.0
    model = LinearHawkes(mu, a, beta, [mean_mark, 8.0, 48.0])
    low, high = 0.17, 1.24
    k = a * mean_mark - beta
    stationary = -beta * mu / k
    mean = (
        stationary * (high - low)
        + (mu - stationary) * (np.exp(k * high) - np.exp(k * low)) / k
    )
    assert model.count_cumulant([(0, low, high)]) == pytest.approx(mean, abs=1e-13)


@pytest.mark.parametrize(
    "baseline, stop", [(1.0, 40.0), (1.0, 400.0), (2.0**-600, 800.0)]
)
def test_supercritical_mean_and_jump_rate_remain_finite(baseline, stop):
    model = LinearHawkes(baseline, 2.0, 1.0)
    # lambda' = lambda + baseline, starting from lambda(0) = baseline.
    # Scale before the second half of the growth so the reference stays finite.
    growth = (2 * baseline * np.exp(stop / 2)) * np.exp(stop / 2)
    count = growth - baseline * (2 + stop)
    rate = growth - baseline
    assert model.count_cumulant([(0, 0.0, stop)]) == pytest.approx(
        count, rel=1e-11, abs=0.0
    )
    impulse = model.system.noise.sigma2.fn(0.0, stop, 0.0)
    np.testing.assert_allclose(impulse, rate * np.array([[4, 2], [2, 1]]), rtol=1e-11)


def test_inactive_growth_leaves_the_public_count_impulse_zero():
    model = LinearHawkes([0.0, 0.4], [[2.0, 0.0], [0.0, 0.0]], 1.0)
    impulse = model.system.noise.sigma2.fn(0.0, 1000.0, 0.0)
    expected = np.zeros((4, 4))
    expected[3, 3] = 0.4
    np.testing.assert_allclose(impulse, expected, atol=0.0, rtol=1e-12)


def test_fast_source_decay_preserves_slow_downstream_growth():
    # Source 0 stays Poisson. Source 1 obeys rate' = rate + 2.
    model = LinearHawkes([1, 1], [[0, 0], [1, 2]], [1e15, 1])
    count = model.count_cumulant([(1, 0.0, 1.0)])
    assert count == pytest.approx(3 * np.expm1(1) - 2, rel=1e-12, abs=0.0)


def test_finite_critical_window_above_the_cumulative_float_range():
    from decimal import FloatOperation, Inexact, localcontext

    # The rate is mu*(1+t). Both cumulative counts exceed float64.
    model = LinearHawkes(1e304, 1.0, 1.0)
    with localcontext() as ctx:
        ctx.prec = 3
        ctx.Emax = 12
        ctx.traps[Inexact] = ctx.traps[FloatOperation] = True
        ctx.clear_flags()
        count = model.count_cumulant([(0, 399.0, 400.0)])
        assert not any(ctx.flags.values())
    assert count == pytest.approx(1e304 * 400.5, rel=1e-12, abs=0.0)


def test_tiny_transfer_from_a_large_poisson_source():
    model = LinearHawkes([1e308, 0], [[0, 0], [1e-200, 0]], 1.0)
    # The child count is mu*A*(t + expm1(-t)). At this t its leading
    # quadratic term gives the answer to better than binary64 precision.
    count = model.count_cumulant([(1, 0.0, 1e-100)])
    assert count == pytest.approx(5e-93, rel=1e-12, abs=0.0)


def test_mean_count_at_the_end_of_a_chain_in_a_short_window():
    from math import factorial

    from scipy.integrate import quad

    n, mu, t = 8, 1e100, 0.001
    model = LinearHawkes([mu] + [0] * (n - 1), np.diag(np.ones(n - 1), k=-1), 1.0)
    # The n-1 successive exponential kernels convolve to an Erlang density.
    integral, _ = quad(lambda u: (1 - u) * u ** (n - 2) * np.exp(-t * u), 0, 1)
    expected = mu * t**n * integral / factorial(n - 2)
    assert model.count_cumulant([(n - 1, 0, t)]) == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )


def test_model_and_windows_reject_invalid_values():
    with pytest.raises(ValueError, match="excitation"):
        LinearHawkes([1, 1], [[1, -1], [0, 1]], [1, 1])
    with pytest.raises(ValueError, match="decay"):
        LinearHawkes(1, 1, [1, 2])
    model = LinearHawkes(1.0, 0.3, 1.0, [1.0, 1.0])
    for window in ((True, 0, 1), (0.5, 0, 1), (1, 0, 1), (0, -1, 1), (0, 2, 1)):
        with pytest.raises(ValueError):
            model.count_cumulant([window])
    with pytest.raises(ValueError, match="mark_moments"):
        model.count_cumulant([(0, 0, 1)] * 3)


def test_count_readout_only_builds_the_jump_orders_it_uses():
    # Rank 21 needs more than the default workspace, but its lower-order
    # cumulants still use ordinary small Wick diagrams.
    model = LinearHawkes(1.0, 0.1, 0.6, [1.0] * 21)
    pair = [(0, 0.1, 1.0)] * 2
    limited = LinearHawkes(1.0, 0.1, 0.6)
    assert model.count_cumulant(pair) == limited.count_cumulant(pair)
    with pytest.raises(ValueError, match="dense jump tensors"):
        _ = model.system
    with pytest.raises(ValueError, match="matrix-exponential state"):
        LinearHawkes(1.0, 0.1, 0.6, workspace_max_bytes=1).count_cumulant(pair[:1])


def test_marks_preserve_fractional_deterministic_law_and_reject_impossible_moments():
    for moments in ([1.0, 2.0, 0.0], [1.0, 1.0, 4.0], [1.0, 2.0, 3.0]):
        with pytest.raises(ValueError, match="mark_moments"):
            LinearHawkes(1.0, 0.1, 0.6, moments)
    model = LinearHawkes(1.0, 0.1, 0.6, [0.1, 0.01, 0.001])
    assert model.count_cumulant([(0, 0.1, 0.8)]) > 0


def test_hawkes_sweep_agrees_in_worker_processes():
    model = LinearHawkes(0.5, 0.1, 1.0)
    expansion = model.system.expand(
        ["phi_a(x)", "phi_b(y)", "phi_c(z)"],
        orders=[1],
        target="connected",
        routing="multiplicity",
        diag_R=False,
        diag_C=False,
    )
    props = model.propagators(1.2)
    kwargs = dict(
        positions_grid={"x": [0.0], "y": [0.0], "z": [0.0]},
        t_final_grid=[0.8, 1.0, 1.2],
        component_tuples=[(1, 1, 1)],
        orders=[1],
        method="gauss_legendre",
        n_gauss=6,
    )
    serial = expansion.sweep(props, **kwargs).rows
    assert expansion.sweep(props, n_jobs=2, **kwargs).rows == serial


def test_long_history_refines_the_count_quadrature():
    model = LinearHawkes(
        [0.49, 0.73], [[0.15, 0.38], [0.26, 0.14]], [1.31, 0.82], [1.0, 2.0, 6.0]
    )
    windows = [(0, 100.0, 100.63), (1, 100.28, 101.1), (0, 100.81, 101.44)]
    reference = (
        0.09094557484927679  # independent finite-window forward moment generator
    )
    assert abs(model.count_cumulant(windows, n_gauss=12) - reference) > 0.01
    assert abs(model.count_cumulant(windows) - reference) < 1e-10


def test_late_short_poisson_windows_keep_their_width():
    t = 1e11
    model = LinearHawkes(1.23456, 0.0, 1.0)
    a, b = (0, t, t + 0.125), (0, t + 0.0625, t + 1)
    for windows, width in [([a], 0.125), ([a, a], 0.125), ([a, b, a], 0.0625)]:
        assert model.count_cumulant(windows) == 1.23456 * width
    coupled = LinearHawkes(1.23456, 0.1, 1.0)
    for rank in (1, 2):
        with pytest.raises(ArithmeticError, match="precision"):
            coupled.count_cumulant([a] * rank)
