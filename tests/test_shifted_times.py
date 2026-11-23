"""Finite-history integrals on affine time coordinates."""

from types import SimpleNamespace

import numpy as np
import pytest

from sft_wick.evaluate import analyze_spatial
from sft_wick.expressions import Propagator, Rational, Symbol
from sft_wick.perturbation import DiagramTerm


def test_delayed_groups_keep_one_integration_time_and_distinct_points():
    term = DiagramTerm(
        (
            Propagator("R", None, None, spatial_left="x", spatial_right="u"),
            Propagator("C", None, None, spatial_left="d", spatial_right="y"),
        ),
        Symbol("b"),
        Rational(1),
        ("d", "u"),
        (),
        1,
        time_lags=(("d", "u", 0.31),),
    )
    spatial = analyze_spatial(term)
    assert spatial.time_integration_vars == ("u",)
    assert spatial.direction_map["d"] != spatial.direction_map["u"]
    assert spatial.time_lags == (("d", "u", 0.31),)


def _ou_cache(start, final):
    import sft_wick as sw

    system = sw.System(
        sw.FieldSpec("phi", 1),
        sw.DiagonalA(gamma=[0.68]),
        sw.GaussianNoise(
            sw.SeparableTranslation(
                sw.ExponentialTemporal(0.0, 1.0), sw.ExponentialSpatial(1.0)
            ),
            sigma2=sw.ConstantImpulse(0.79**2),
        ),
        t_min=start,
    )
    return system.propagators(t_max=final, progress=False).cache


@pytest.fixture(scope="module")
def ou_cache():
    return _ou_cache(0.2, 1.4)


@pytest.mark.parametrize(
    "method,tol",
    [
        ("gauss_legendre", 1e-12),
        ("nquad", 1e-12),
        ("qmc_scalar", 2e-7),
        ("qmc_vectorized", 2e-7),
    ],
)
def test_delayed_quadratic_drift_mean(method, tol, ou_cache):
    """R(T,u) C(u-lag,u-lag) integrated from start+lag to T."""
    from sft_wick.evaluate import integrate_moment

    term = DiagramTerm(
        (Propagator("R", None, None, "x", "u"), Propagator("C", None, None, "d", "d")),
        Symbol("b", spatial_args=("u", "d")),
        Rational(1),
        ("u", "d"),
        (),
        0,
        time_lags=(("d", "u", 0.31),),
    )

    def coupling(positions, times):
        assert times[0] - times[1] == pytest.approx(0.31)
        return 1.0

    ig = term.build_integrand({"b": coupling})
    amount, _ = integrate_moment(
        ig,
        0.95,
        ou_cache,
        method=method,
        n_gauss=8,
        n_samples=1024,
        seed=417,
    )
    assert amount == pytest.approx(0.04512592085613112, abs=tol)
    empty, _ = integrate_moment(ig, 0.49, ou_cache, method=method, n_samples=32)
    assert empty == 0.0


def test_nested_delay_and_fixed_external_cusp(ou_cache):
    """Independent Isserlis pairing of X2(1.4) X0(0.95), including both routings."""
    from sft_wick.evaluate import integrate_moment

    def factor(kind, a, b):
        return Propagator(kind, None, None, a, b)

    response = (factor("R", "x", "u"), factor("R", "du", "v"))
    routes = [
        (1, (factor("C", "du", "y"), factor("C", "dv", "dv"))),
        (2, (factor("C", "du", "dv"), factor("C", "dv", "y"))),
    ]
    total = 0.0
    for multiplicity, covariance in routes:
        term = DiagramTerm(
            response + covariance,
            Symbol("b"),
            Rational(2 * multiplicity),
            ("u", "du", "v", "dv"),
            (),
            0,
            time_lags=(("du", "u", 0.31), ("dv", "v", 0.31)),
        )
        value, _ = integrate_moment(
            term.build_integrand({"b": np.array(1.0)}),
            1.4,
            ou_cache,
            method="gauss_legendre",
            n_gauss=6,
            external_times={"y": 0.95},
        )
        total += value
    assert total == pytest.approx(0.011871889851399718, abs=1e-12)


@pytest.mark.parametrize("final", [0.95, np.float32(0.95)])
def test_swept_endpoint_and_positive_lag_response(ou_cache, final):
    """A response from a delayed time to its own anchor survives the Itô rule."""
    from sft_wick.evaluate import integrate_moment

    term = DiagramTerm(
        (Propagator("R", None, None, "x", "u"), Propagator("R", None, None, "u", "d")),
        Symbol("b"),
        Rational(1),
        ("d", "u"),
        (),
        0,
        time_lags=(("d", "u", 0.31),),
    )
    ig = term.build_integrand({"b": np.array(1.0)})
    # Integral of exp[-gamma*(x-u+lag)] over start+lag < u < x < end.
    width, gamma = float(final) - 0.2 - 0.31, 0.68
    expected = (
        np.exp(-gamma * 0.31) * (width + np.expm1(-gamma * width) / gamma) / gamma
    )
    for method in ("gauss_legendre", "nquad"):
        value, _ = integrate_moment(
            ig, final, ou_cache, method=method, integrate_over="all"
        )
        assert value == pytest.approx(expected, rel=0, abs=2e-14)
    assert ig.evaluate({"u": 0.3, "x": 0.95}, {}, ou_cache) == 0j
    with pytest.raises(ValueError, match="fixed time lags"):
        integrate_moment(ig, 0.95, ou_cache, method="exact")


@pytest.mark.parametrize(("origin", "lag"), [(0.2, 1e-17), (1e12, 1e-6)])
@pytest.mark.parametrize(
    "method", ["gauss_legendre", "nquad", "qmc_scalar", "qmc_vectorized"]
)
def test_subresolution_delay_refuses_collapsed_response(origin, lag, method):
    """The exact retarded response is nonzero even for very small positive lags."""
    from sft_wick.evaluate import integrate_moment

    term = DiagramTerm(
        (Propagator("R", None, None, "x", "u"), Propagator("R", None, None, "u", "d")),
        Symbol("b"),
        Rational(1),
        ("d", "u"),
        (),
        0,
        time_lags=(("d", "u", lag),),
    )
    ig = term.build_integrand({"b": np.array(1.0)})
    ou_cache = _ou_cache(origin, origin + 0.75)
    with pytest.raises(ValueError, match="time origin"):
        integrate_moment(
            ig,
            origin + 0.75,
            ou_cache,
            t_min=origin,
            method=method,
            integrate_over="all",
            n_samples=64,
        )
    with pytest.raises(ValueError, match="time origin"):
        ig.evaluate({"u": origin + 0.5, "x": origin + 0.75}, {}, ou_cache)


def test_neighboring_delays_must_preserve_retarded_order(ou_cache):
    from sft_wick.evaluate import integrate_moment

    first, second = 0.31, np.nextafter(0.31, np.inf)
    term = DiagramTerm(
        (Propagator("R", None, None, "x", "u"), Propagator("R", None, None, "a", "b")),
        Symbol("g"),
        Rational(1),
        ("u", "a", "b"),
        (),
        0,
        time_lags=(("a", "u", first), ("b", "u", second)),
    )
    ig = term.build_integrand({"g": np.array(1.0)})
    # Distinct offsets are visible near u=.6, but can coincide near the upper
    # integration limit. Refuse the unresolvable part independently of GL order.
    # These particular float results are adjacent, but their spacing exceeds
    # the physical time difference. A point evaluation needs the same guard.
    with pytest.raises(ValueError, match="clock precision"):
        ig.evaluate({"u": 0.9, "x": 0.95}, {}, ou_cache)
    for n_gauss in (6, 10, 16, 32):
        with pytest.raises(ValueError, match="clock precision"):
            integrate_moment(
                ig, 0.95, ou_cache, method="gauss_legendre", n_gauss=n_gauss
            )
    for method in ("nquad", "qmc_scalar", "qmc_vectorized"):
        with pytest.raises(ValueError, match="clock precision"):
            integrate_moment(ig, 0.95, ou_cache, method=method, n_samples=64)
    expected = (
        -np.expm1(-0.68 * (0.72 - 0.2 - second))
        * np.exp(-0.68 * (second - first))
        / 0.68
    )
    value, _ = integrate_moment(ig, 0.72, ou_cache, method="gauss_legendre", n_gauss=10)
    assert value == pytest.approx(expected, rel=0, abs=2e-14)


def test_late_epoch_resolves_binary_lag_and_latex_labels():
    epoch = 1e12
    cache = _ou_cache(epoch, epoch + 1)
    term = DiagramTerm(
        (Propagator("R", None, None, "x", "u"), Propagator("R", None, None, "a", "d")),
        Symbol("b"),
        Rational(1),
        ("d", "u", "a"),
        (),
        0,
        equal_time_aliases=(("a", "u"),),
        time_lags=(("d", "u", 0.125),),
    )
    ig = term.build_integrand({"b": np.array(1.0)})
    value = ig.evaluate({"u": epoch + 0.5, "x": epoch + 0.75}, {}, cache)
    assert value == pytest.approx(np.exp(-0.68 * 0.375), rel=0, abs=2e-15)
    rendered = ig.to_latex()
    assert "t_{a}=t_{u}" in rendered
    assert "t_{d}=t_{u}-0.125" in rendered


@pytest.mark.parametrize(
    "method", ["gauss_legendre", "nquad", "qmc_scalar", "qmc_vectorized"]
)
def test_delay_onset_requires_resolvable_interior_times(method):
    from sft_wick.evaluate import integrate_moment

    epoch, lag = 1e12, 0.125
    spacing = np.spacing(epoch)
    term = DiagramTerm(
        (Propagator("R", None, None, "x", "u"), Propagator("R", None, None, "d", "y")),
        Symbol("b"),
        Rational(1),
        ("d", "u"),
        (),
        0,
        time_lags=(("d", "u", lag),),
    )
    integrand = term.build_integrand({"b": np.array(1.0)})
    late_cache, elapsed_cache = _ou_cache(epoch, epoch + 0.5), _ou_cache(0.0, 0.5)

    def response(origin, width, cache):
        return integrate_moment(
            integrand, origin + lag + width, cache, t_min=origin,
            external_times={"y": origin}, method=method, n_gauss=8,
            n_samples=256, seed=417,
        )[0]

    assert response(epoch, 0.0, late_cache) == 0.0
    for ulps in (1, 2, 3, 8, 16):
        width = ulps * spacing
        with pytest.raises(ValueError, match="clock precision.*time origin"):
            response(epoch, width, late_cache)
        expected = width * np.exp(-0.68 * width)
        assert response(0.0, width, elapsed_cache) == pytest.approx(expected, abs=1e-18)


def test_thin_affine_strip_requires_resolvable_time_differences():
    from sft_wick._shifted_times import ShiftedDomain

    epoch, lag = 1e12, 0.125
    spatial = SimpleNamespace(
        time_lags=(("du", "u", lag), ("dv", "v", lag + 2 * np.spacing(epoch))),
        equal_time_aliases=(), r_absorbed_pairs=(),
        r_propagators=(("v", "u"), ("du", "dv")),
    )

    def domain():
        return ShiftedDomain(spatial, ("u", "v"), (), {}, epoch + 1, epoch + 1, epoch)

    # Each marginal clock spans most of the interval, but causal orderings
    # together constrain 0 < v - u < 2 ULPs.
    with pytest.raises(ValueError, match="time difference.*clock precision"):
        domain()
    # The ordinary triangle has thin slices near an endpoint only. It can
    # still be sampled over a well-resolved range in both coordinates.
    spatial.r_propagators = (("v", "u"),)
    times, weights = domain().bounds().sample(
        np.array([[0.5, 0.5], [1.0, 0.5], [0.5, 0.0]]), {"u": 0, "v": 1}
    )
    assert epoch < times[0, 0] < times[0, 1] < epoch + 1
    assert weights[0] > 0
    assert times[1, 0] == times[1, 1] == epoch + 1 and weights[1] == 0
    assert times[2, 0] == times[2, 1] and weights[2] > 0


def test_clock_projection_uses_real_neighbors_across_power_of_two():
    from sft_wick._shifted_times import ShiftedDomain

    center = 2.0**40
    lower = np.nextafter(center, -np.inf)
    upper = np.nextafter(center, np.inf)
    # Only the power of two itself lies inside. Extending the finer clock
    # grid below it to the upper half would invent another interior time.
    spatial = SimpleNamespace(
        time_lags=(), equal_time_aliases=(),
        r_absorbed_pairs=(), r_propagators=(),
    )
    with pytest.raises(ValueError, match="clock 'u'.*clock precision"):
        ShiftedDomain(spatial, ("u",), (), {}, upper, upper, lower)


def test_delayed_response_requires_distinct_reconstructed_interior_endpoints():
    from sft_wick.evaluate import integrate_moment

    epoch, gamma = 1e12, 0.68
    tick = np.spacing(epoch)
    lag_du = 0.125 + 0.6 * tick
    lag_dv = 0.125 + 0.4 * tick
    lag_fast = lag_dv - 40 * tick
    term = DiagramTerm(
        (Propagator("R", None, None, "du", "dv"),
         Propagator("R", None, None, "dv_fast", "du"),
         Propagator("R", None, None, "y", "u")),
        Symbol("b"), Rational(1), ("du", "dv", "dv_fast", "u", "v"), (), 0,
        time_lags=(("du", "v", lag_du), ("dv", "u", lag_dv), ("dv_fast", "u", lag_fast)),
    )
    integrand = term.build_integrand({"b": np.array(1.0)})

    def response(origin):
        return integrate_moment(
            integrand, origin + 1.0, _ou_cache(origin, origin + 1.0), t_min=origin,
            external_times={"y": origin + 0.5}, method="gauss_legendre", n_gauss=8,
        )[0]

    # The sampled roots are all interior, but binary64 rounds some of the
    # reconstructed delayed endpoints to the same clock at the late epoch.
    with pytest.raises(ValueError, match="shifted response.*clock precision"):
        response(epoch)
    width = lag_dv - lag_fast
    expected = width * np.exp(-gamma * width) * -np.expm1(-gamma * (0.5 - lag_dv)) / gamma
    assert response(0.0) == pytest.approx(expected, rel=0, abs=2e-18)


def test_delayed_covariance_rejects_rounding_to_history_lower_bound():
    from sft_wick.evaluate import integrate_moment

    origin, lag, end = -1e12, 1e12, 1e-5
    term = DiagramTerm(
        (Propagator("R", None, None, "x", "u"), Propagator("C", None, None, "d", "d")),
        Symbol("b"), Rational(1), ("u", "d"), (), 0,
        time_lags=(("d", "u", lag),),
    )
    integrand = term.build_integrand({"b": np.array(1.0)})

    # The exact history length is u > 0. Reconstructing d from the
    # well-resolved root u rounds every actual covariance to d = t_min.
    with pytest.raises(ValueError, match="history.*clock precision"):
        integrate_moment(
            integrand, end, _ou_cache(origin, end), t_min=origin,
            method="gauss_legendre", n_gauss=8,
        )

    # A short lag from zero history has the same analytic mean at the
    # same interval after onset, and its absolute clocks are resolved.
    short_lag = 0.125
    short_term = DiagramTerm(
        term.propagators, Symbol("b"), Rational(1), ("u", "d"), (), 0,
        time_lags=(("d", "u", short_lag),),
    )
    actual, _ = integrate_moment(
        short_term.build_integrand({"b": np.array(1.0)}),
        short_lag + end, _ou_cache(0.0, 0.2), method="gauss_legendre", n_gauss=8,
    )
    expected = 0.5 * (0.79 * np.expm1(-0.68 * end) / 0.68) ** 2
    assert actual == pytest.approx(expected, rel=2e-10, abs=0)


def test_raw_term_rejects_an_unrepresentable_chain():
    term = DiagramTerm(
        (Propagator("R", None, None, "x", "u"), Propagator("R", None, None, "a", "b")),
        Symbol("g"),
        Rational(1),
        ("u", "a", "b"),
        (),
        0,
        time_lags=(("a", "u", 1.0), ("b", "a", 1e-17)),
    )
    with pytest.raises(ValueError, match="floating-point precision"):
        analyze_spatial(term)
