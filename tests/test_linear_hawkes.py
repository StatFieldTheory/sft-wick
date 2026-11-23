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


@pytest.mark.parametrize(
    ("n", "stop", "variance", "cross"),
    [
        (8, 2000.0, 2223.102739591411, 49.28421407156207),
        (14, 3000.0, 3575.870882803650, 87.68808128931751),
        (8, 1e12, 1111613380455.206, 24656858716.17005),
    ],
)
def test_long_subcritical_network_covariance(n, stop, variance, cross):
    model = LinearHawkes(np.ones(n), np.full((n, n), 0.01), 1.0, [1.0, 1.0])
    # Martingale integral of the independent source responses
    # H_ij(v) = delta_ij + .01 * (1 - exp(-k*v))/k, k = 1 - .01*n.
    w = (0, 0.0, stop)
    assert model.count_cumulant([w, w]) == pytest.approx(variance, rel=1e-12, abs=0.0)
    assert model.count_cumulant([w, (1, 0.0, stop)]) == pytest.approx(
        cross, rel=1e-12, abs=0.0
    )
    if n == 8:
        block = model.propagators(stop).cache.C_value(0.0, stop, 0.0, stop)
        np.testing.assert_allclose(block[n, n : n + 2], [variance, cross], rtol=1e-12)


def test_long_network_with_late_count_windows():
    model = LinearHawkes(np.ones(14), np.full((14, 14), 0.01), 1.0, [1.0, 1.0])
    # Integral of the stationary cross-node density q * exp(-.86 * lag);
    # the empty-history transient at 2900 contributes less than 1e-1000.
    windows = [(0, 2900.0, 3000.0), (1, 2900.0, 3000.0)]
    np.testing.assert_allclose(
        model.count_cumulant(windows), 2.8902676698643765, rtol=1e-12
    )


def test_rare_late_descendants_of_a_closed_window():
    model = LinearHawkes(1e300, 0.1, 1.0, [1.0, 1.0])
    # The remaining response decays as exp(-.9*t), below binary64 at
    # t=1500. The early immigrant population keeps its covariance finite.
    windows = [(0, 0.0, 0.25), (0, 1500.0, 1500.25)]
    np.testing.assert_allclose(
        model.count_cumulant(windows), 3.237215422455588e-289, rtol=1e-12, atol=0
    )


def test_third_count_with_one_distant_window():
    model = LinearHawkes(0.8, 0.2, 1.0)
    pair = [(0, 0.0, 1.0)] * 2
    early = model.count_cumulant([*pair, (0, 1.0, 2.0)])
    # Conditional on the early history, only fluctuations of the rate at
    # t=1 affect the later mean. Their response decays as exp(-.8*lag).
    late = model.count_cumulant([*pair, (0, 50.0, 51.0)])
    np.testing.assert_allclose(late, early * np.exp(-0.8 * 49), rtol=1e-11, atol=0)


def test_stiff_subcritical_count_variance_and_public_covariance():
    beta, stop = 3e15, 0.6
    model = LinearHawkes(1.0, beta / 2, beta)
    # Both the mean intensity and the count response are 2-exp(-beta*t/2).
    # Integrating their martingale variance gives the following expression.
    x = -beta * stop / 2
    expected = 8 * stop + 4 * stop * np.exp(x) + 22 * np.expm1(x) / beta
    pair = [(0, 0.0, stop)] * 2
    assert model.count_cumulant(pair) == pytest.approx(expected, rel=1e-12, abs=0.0)
    cache = model.propagators(stop).cache
    assert cache.C_value(0.0, stop, 0.0, stop)[1, 1] == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )


def test_count_covariance_selects_the_window_before_float_conversion():
    # Critical unit-jump process. Integrating the squared response of the
    # window [24,25] against mu*(1+t) gives mu*371.25, although the
    # cumulative variance at 25 is already too large for binary64.
    model = LinearHawkes(1e304, 1.0, 1.0)
    window = (0, 24.0, 25.0)
    assert model.count_cumulant([window, window]) == pytest.approx(
        3.7125e306, rel=1e-12, abs=0.0
    )


def test_count_covariance_scales_with_independent_immigration():
    windows = [(0, 0.1, 0.8), (0, 0.2, 0.6)]
    reference = LinearHawkes(0.61, 0.8, 1.0).count_cumulant(windows)
    model = LinearHawkes(6.1e279, 0.8, 1.0)
    assert model.count_cumulant(windows) == pytest.approx(
        1e280 * reference, rel=1e-12, abs=0.0
    )


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


def test_count_cumulant_is_invariant_to_mark_units():
    windows = [(0, 0.1, 0.8), (0, 0.2, 0.6), (0, 0.85, 1.4)]
    expected = LinearHawkes(0.61, 0.8, 1).count_cumulant(windows)
    shift = 345
    model = LinearHawkes(
        0.61, np.ldexp(0.8, shift), 1, np.exp2(-shift * np.arange(1, 4))
    )
    assert model.count_cumulant(windows) == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("shift", [-258, 255])
def test_marked_cross_node_jump_tensors_across_unit_scales(shift):
    # A single mark, zero or two with equal probability, drives both targets.
    excitation = np.array([[0.5, 0.0], [2.0**-20, 0.25]])
    orders = np.arange(1, 5)
    raw_moments = np.exp2(orders - 1)
    model = LinearHawkes(
        [0.4, 0.7],
        np.ldexp(excitation, -shift),
        1,
        np.ldexp(raw_moments, shift * orders),
    )
    vertices = {v.name: v for v in model.system.km_vertices}
    for order in (2, 3, 4):
        tensor = np.asarray(vertices[f"K{order}rate"].coupling)
        for node in range(2):
            expected = np.zeros((4,) * order)
            for mark in (0, 2):
                jump = np.r_[mark * excitation[:, node], np.eye(2)[node]]
                product = jump
                for _ in range(order - 1):
                    product = np.multiply.outer(product, jump)
                expected += 0.5 * product
            np.testing.assert_array_equal(tensor[..., node], expected)


def test_mean_jump_tensors_preserve_small_jumps_with_large_immigration():
    impulse = LinearHawkes(1e300, 1e-200, 1, [1, 1]).system.noise.sigma2
    assert impulse.fn(0.0, 0.0, 0.0)[0, 0] == pytest.approx(1e-100, rel=1e-14, abs=0.0)

    model = LinearHawkes(1e290, 1e-130, 1)
    vertex = next(v for v in model.system.km_vertices if v.name == "K3mean")
    value = vertex.coupling([0.0], [np.array([0.0])])[0, 0, 0, 0]
    assert value == pytest.approx(1e-100, rel=1e-14, abs=0.0)


def test_rare_parent_rate_drives_a_finite_downstream_impulse():
    # Node 0 is Poisson; at time t node 1's rate is 1e-400*(1-exp(-t)).
    # Only node 1 can excite node 2, by 1e100 per event.
    model = LinearHawkes(
        [1e-240, 0, 0], [[0, 0, 0], [1e-160, 0, 0], [0, 1e100, 0]], 1, [1, 1]
    )
    times = np.array([0.0, 0.25, 0.5, 0.25])
    impulse = model.system.noise.sigma2.fn(0.0, times, 0.0)
    np.testing.assert_allclose(
        impulse[:, 2, 2], 1e-200 * -np.expm1(-times), rtol=1e-14, atol=0.0
    )


@pytest.mark.parametrize(
    "bounds, order",
    [([(0, 1)] * 3, 4), ([(0.1, 0.8), (0.2, 0.6), (0.85, 1.4)], 1)],
)
def test_joint_third_count_retains_weak_daughter_pairs(bounds, order):
    def fork(rate, edge):
        model = LinearHawkes([rate, 0, 0], [[0.4, 0, 0], [edge, 0, 0], [edge, 0, 0]], 1)
        windows = [(i, lo, hi) for i, (lo, hi) in enumerate(bounds)]
        return model.count_cumulant(windows, n_gauss=order)

    # Independent daughters of each parent cluster imply exact proportionality
    # to immigration times the square of the daughter excitation.
    assert fork(1e300, 1e-170) == pytest.approx(1e-40 * fork(1, 1), rel=1e-12, abs=0.0)


def test_short_chain_preserves_the_parent_parent_grandchild_cumulant():
    model = LinearHawkes([1e250, 0, 0], [[0, 0, 0], [1, 0, 0], [0, 1e-90, 0]], 1)
    # Each cluster contains precisely one immigrant. Repeating the parent
    # window twice leaves the joint cumulant equal to the grandchild mean.
    t = 1e-120
    expected = 1e-200 / 6  # mu*A10*A21*t^3/6, relative corrections O(t)
    assert model.count_cumulant([(i, 0, t) for i in (0, 0, 2)]) == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )


def test_joint_third_count_retains_rare_parent_covariance():
    a = np.zeros((4, 4))
    a[1, 0] = 1e-100
    a[2, 1] = a[3, 1] = 1e60
    model = LinearHawkes([1e-240, 0, 0, 0], a, 1)
    # At leading order in the rare 0->1 jump, the two independent daughters
    # share one intermediate parent. Integrating its possible arrival times
    # gives integral_0^1 dh integral_0^h du exp(-u)*(1-exp(-(h-u)))**2.
    expected = 1e-220 * (-1.5 + 4 / np.e + np.exp(-2) / 2)
    assert model.count_cumulant([(i, 0, 1) for i in (0, 2, 3)]) == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("edge_exp, rate_exp", [(195, 450), (-205, -470)])
def test_third_critical_count_has_consistent_time_and_rate_units(edge_exp, rate_exp):
    a, mu = 2.0**edge_exp, 2.0**rate_exp
    model = LinearHawkes(mu, a, a)
    # At criticality the first three moment equations are polynomials.
    # Changing the clock to a*t leaves unit decay and immigration mu/a.
    assert model.count_cumulant([(0, 0, 1 / a)] * 3) == pytest.approx(
        (659 / 60) * (mu / a), rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize(("edge", "decay", "t"), [(0.02, 1.0, 300.0), (1.0, 1e8, 1.0)])
def test_stable_dense_network_keeps_the_fixed_rule_in_the_default_workspace(
    edge, decay, t
):
    n = 11
    offspring = n * edge / decay
    model = LinearHawkes(np.ones(n), np.full((n, n), edge), decay)
    # The all-to-all process is a single branching process with uniform
    # labels. In the bulk the distinct-label cumulant is the immigrant
    # rate times E[S(S-1)(S-2)]/n**3, for total progeny S. Both GL2 nodes
    # are over 49 relaxation times from either boundary.
    k = 1 - offspring
    expected = t / n**2 * offspring**2 * (9 - 8 * offspring + 2 * offspring**2) / k**5
    assert model.count_cumulant([(i, 0, t) for i in (0, 1, 2)], n_gauss=2) == (
        pytest.approx(expected, rel=1e-10, abs=0.0)
    )


def test_strongly_damped_chain_retains_its_third_count():
    model = LinearHawkes(
        [1e240, 0, 0], [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [1, 1e170, 1e170]
    )
    # A repeated Poisson parent leaves the cumulant equal to the expected
    # number of grandchildren. At the GL1 midpoint both daughter kernels
    # have integrated to their full mass 1/beta, within binary64 precision.
    windows = [(i, 0, 1) for i in (0, 0, 2)]
    assert model.count_cumulant(windows, n_gauss=1) == pytest.approx(
        1e-100, rel=1e-12, abs=0.0
    )


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
    with pytest.raises(ValueError, match="quadrature"):
        limited.count_cumulant([(0, 0.0, 1.0)] * 3, n_gauss=10000)
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
    with pytest.raises(ArithmeticError, match="precision"):
        coupled.count_cumulant([a])
    # The covariance counts only the short interval. After this long past,
    # the stationary covariance density away from zero is q*exp(-k*|lag|).
    k, width = 0.9, 0.125
    rate = 1.23456 / k
    q = rate * 0.1 * (2.0 - 0.1) / (2 * k)
    expected = rate * width + 2 * q * (width * k + np.expm1(-k * width)) / k**2
    assert coupled.count_cumulant([a, a]) == pytest.approx(expected, rel=1e-12, abs=0.0)
