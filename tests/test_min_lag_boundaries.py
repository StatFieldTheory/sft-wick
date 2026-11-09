"""C tables preserve spatial order and the declared physical time domain."""

from dataclasses import dataclass

import numpy as np
import pytest

import sft_wick as sw
import sft_wick.evaluate as ev


BUILDERS = [
    ("translation", {}),
    ("rotation", {}),
    ("general", {}),
    ("translation", {"r_max": 1.0, "n_grid_r": 4}),
    ("rotation", {"n_grid_cos": 4}),
    ("general", {"x_max": 1.0, "n_grid_x": 4}),
]


def covariance(x, t, y, s):
    # phi(x,t) = (t + x*t**2)*Z, Z a unit-variance scalar Gaussian.
    return np.array([[t * s * (1 + float(x) * t) * (1 + float(y) * s)]])


@pytest.mark.parametrize("times", [(0.25, 0.75), (0.75, 0.25)])
def test_general_full_grid_preserves_space_for_both_time_orders(times):
    model = ev.PropagatorModel(
        R_time=lambda t, s: 1.0,
        kappa2=lambda x, t, y, s: np.array([
            [(1 + 2 * float(x) * t) * (1 + 2 * float(y) * s)]]),
        n_components=1, iso_R=True, diag_C=True,
    )
    cache = ev.PropagatorCache(
        model, homogeneity="general", c_value_fn=covariance,
    )
    cache.precompute_C_table_general(
        t_max=1.0, n_grid_t=5, x_max=1.0, n_grid_x=3,
    )
    t, s = times
    got = cache.C_at_batch(
        np.array([t]), np.array([s]), np.array([0.0]), np.array([1.0]),
    )[0, 0]
    # All queried coordinates are grid nodes in both representations.
    assert got == pytest.approx(covariance(0.0, t, 1.0, s)[0, 0],
                                rel=1e-12, abs=0.0)


@pytest.mark.parametrize("mode, options", [
    ("lazy", {}),
    ("translation", {"r_max": 1.0, "n_grid_r": 2}),
    ("rotation", {"n_grid_cos": 2}),
    ("general", {"x_max": 1.0, "n_grid_x": 2}),
])
def test_table_respects_requested_time_horizon(mode, options):
    def bounded_response(t, s):
        if max(t, s) > 1.0 + 1e-12:
            raise ValueError(f"R outside requested [0,1]: ({t}, {s})")
        return float(np.exp(-(t - s)))

    homogeneity = "translation" if mode == "lazy" else mode
    model = ev.PropagatorModel(
        R_time=bounded_response,
        kappa2=lambda x, t, y, s: np.ones((1, 1)),
        n_components=1, iso_R=True, diag_C=True,
    )
    cache = ev.PropagatorCache(
        model, homogeneity=homogeneity, c_method="gauss_legendre", n_gauss=4,
    )
    getattr(cache, "precompute_C_table_" + homogeneity)(
        t_max=1.0, n_grid_t=5, **options,
    )
    x = np.array([[1.0, 0.0]]) if mode == "rotation" else np.array([0.0])
    got = cache.C_at_batch(np.array([0.9]), np.array([0.8]), x, x)
    assert np.isfinite(got).all()
    assert (got > 0.0).all()


@pytest.mark.parametrize("mode, options", BUILDERS)
@pytest.mark.parametrize("diag", [True, False])
def test_bounded_user_covariance_converges_at_the_horizon(mode, options, diag):
    """Both temporal corners and the hypotenuse, including off-grid times.

    A matrix white-noise covariance has unequal decay rates and a negative
    cross-component entry.  The independent exact expression is bounded
    to the requested domain, including a nonzero initial time.  Errors
    are normalized by the largest covariance in the sample, since C
    vanishes at the initial-time corner.
    """
    lo, hi = 0.4, 2.3
    gamma = np.array([0.9, 1.6])
    sigma = np.array([[0.6, -0.25], [-0.25, 0.4]])

    def exact(x, t, y, s):
        assert lo - 1e-12 <= min(t, s) <= max(t, s) <= hi + 1e-12
        earlier = min(t, s)
        rates = gamma[:, None] + gamma[None, :]
        return (sigma * np.exp(-gamma[:, None] * (t - earlier)
                               - gamma[None, :] * (s - earlier))
                * (-np.expm1(-rates * (earlier - lo))) / rates)

    model = ev.PropagatorModel(
        R_time=lambda t, s: np.diag(np.exp(-gamma * (t - s))),
        kappa2=lambda *args: np.zeros((2, 2)),
        n_components=2, iso_R=False, diag_C=diag, t_min=lo,
    )
    x = np.array([[1.0, 0.0]]) if mode == "rotation" else np.array([0.0])
    errors = []
    for n in (11, 21):
        cache = ev.PropagatorCache(model, homogeneity=mode, c_value_fn=exact)
        getattr(cache, "precompute_C_table_" + mode)(
            t_max=hi, n_grid_t=n, **options,
        )
        h = (hi - lo) / (n - 1)
        got, want = [], []
        for offset in (0.1, 0.5, 0.9):
            for earlier in (lo + 0.1 * h, lo + 0.6 * h,
                            lo + 0.3 * (hi - lo), hi - (offset + 0.2) * h):
                later = hi - offset * h
                for t, s in ((later, earlier), (earlier, later)):
                    got.append(cache.C_at_batch(np.array([t]), np.array([s]),
                                                 x, x)[0])
                    ref = exact(x, t, x, s)
                    want.append(np.diag(ref) if diag else ref)
        got, want = np.asarray(got), np.asarray(want)
        assert np.isfinite(got).all()
        errors.append(np.max(np.abs(got - want)) / np.max(np.abs(want)))
    assert errors[1] < errors[0] / 3.0, errors
    assert errors[1] < 2e-3, errors


@pytest.mark.parametrize("schema", ["2", "3"])
def test_new_schema_rejects_the_incorrect_min_lag_cache(tmp_path, schema):
    """Schema 3 held tables with ``c_n_gauss`` nodes under a label naming the
    converged count, and expansions with the response phase applied twice."""
    from joblib import dump, hash as joblib_hash
    from sft_wick.workflow.cache import load_or_compute

    spec = {"t_max": 1.0, "n_grid_t": 5}
    path = tmp_path / "props.joblib"
    dump({"key": joblib_hash((schema, spec))[:12], "value": "incorrect E"}, path)
    assert load_or_compute(path, spec, lambda: "recomputed") == "recomputed"
    assert load_or_compute(path, spec, lambda: "rebuilt again") == "recomputed"


@pytest.mark.parametrize("mode, options", BUILDERS[3:])
@pytest.mark.parametrize("rate", [10.0, 30.0])
def test_linear_table_keeps_a_positive_covariance_positive(mode, options, rate):
    """A decaying rank-one Gaussian field must not acquire negative C.

    These queries lie in the last physical time cell.  A polynomial
    continuation can be negative outside the triangle, but default linear
    interpolation must stay between the physical cell's extremal values.
    """
    model = ev.PropagatorModel(
        R_time=lambda t, s: 1.0, kappa2=lambda *args: np.ones((1, 1)),
        n_components=1, iso_R=True,
    )
    cache = ev.PropagatorCache(
        model, homogeneity=mode,
        c_value_fn=lambda x, t, y, s: np.array([[np.exp(-rate * (t + s))]]),
    )
    getattr(cache, "precompute_C_table_" + mode)(
        t_max=1.0, n_grid_t=6, **options,
    )
    x = np.array([[1.0, 0.0]]) if mode == "rotation" else np.array([0.0])
    for t, s in ((0.99, 0.98), (0.98, 0.99)):
        got = cache.C_at_batch(np.array([t]), np.array([s]), x, x)[0, 0]
        assert np.exp(-2.0 * rate) <= got <= np.exp(-1.6 * rate)


def _bump(r):
    """Zero at the grid values 0, 2 and 4 of ``n_grid_r = 5``, not at 1."""
    return 0.5 * np.sin(np.pi * r / 2.0) if 0.0 < r < 2.0 else 0.0


def _one_sided_kernel(n1, t1, n2, t2):
    """``κ(r, s1, s2)`` that the time swap changes at r = 1 only, so that a
    probe of the first, middle and last ``r`` of the grid passes."""
    r = abs(float(n1) - float(n2))
    return np.eye(2) * np.exp(-abs(t1 - t2)) * (1.0 + _bump(r) * np.tanh(t1 - t2))


@pytest.mark.parametrize("times", [(0.75, 2.25), (1.5, 3.0)])
def test_full_table_keeps_both_time_orders_when_one_grid_value_needs_them(times):
    """The swap was probed at three grid values, and the earlier-first half
    was then derived where it does not hold: 58 % off.  The reference is the
    defining integral, by scipy alone."""
    from scipy.integrate import dblquad

    model = ev.PropagatorModel(
        R_time=lambda t, s: 1.0 if t > s else 0.0,
        kappa2=_one_sided_kernel,
        n_components=2,
        diag_C=False,
    )
    cache = ev.PropagatorCache(model, homogeneity="translation")
    cache.precompute_C_table_translation(
        t_max=3.0, n_grid_t=5, r_max=4.0, n_grid_r=5
    )
    t1, t2 = times
    got = cache.C_at_batch(
        np.array([t1]), np.array([t2]), np.array([0.0]), np.array([1.0])
    )[0, 0, 0]
    ref, _ = dblquad(
        lambda b, a: _one_sided_kernel(0.0, a, 1.0, b)[0, 0], 0.0, t1, 0.0, t2
    )
    assert got == pytest.approx(ref, rel=1e-3, abs=0.0)


@dataclass(frozen=True)
class _ShortOneSided:
    """Linear on ``0 <= d < width`` and zero elsewhere: no reach past ``width``,
    so the kernel reads 0 on both sides at every lag beyond it."""

    width: float

    def __call__(self, d: float) -> float:
        return 1.0 - d / self.width if 0.0 <= d < self.width else 0.0


@dataclass(frozen=True)
class _Triangle:
    width: float

    def __call__(self, r: float) -> float:
        return max(0.0, 1.0 - abs(r) / self.width)


_RATES = (0.3, 0.9)


def _lazy_props(temporal, spatial):
    system = sw.System(
        field=sw.FieldSpec("phi", 2),
        linear=sw.DiagonalA(gamma=list(_RATES)),
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.CustomKernel(fn=temporal),
            spatial=sw.CustomKernel(fn=spatial))))
    return system.propagators(
        t_max=3.0, n_grid_t=33, diag_C=False, c_closed_form=None,
        c_method="gauss_legendre", c_n_gauss=24, progress=False)


def _reference_C00(temporal, t1, t2):
    """``∫∫ e^{-γ(t1-s1)} κ_t(s1-s2) e^{-γ(t2-s2)}`` for ``κ_t`` supported on
    ``0 <= s1 - s2 < width``, by scipy alone."""
    from scipy.integrate import dblquad

    rate = _RATES[0]
    return dblquad(
        lambda s2, s1: temporal(s1 - s2)
        * np.exp(-rate * (t1 - s1) - rate * (t2 - s2)),
        0.0,
        t1,
        lambda s1: max(0.0, s1 - temporal.width),
        lambda s1: min(s1, t2),
        epsabs=1e-11,
        epsrel=1e-11,
    )[0]


@pytest.mark.parametrize("width", [0.5, 1.1, 1.2])
@pytest.mark.heavy
def test_a_table_keeps_both_time_orders_of_a_kernel_of_short_reach(width):
    """Past the last lag the swap was read at the kernel is 0 on both sides,
    so a one-sided kernel took a half table: 22 and 14 percent off, on the
    time order it stored the other's value for."""
    temporal = _ShortOneSided(width)
    cache = _lazy_props(temporal, _Triangle(5.0)).cache
    for t1, t2 in ((2.0, 2.4), (2.4, 2.0)):
        got = cache.C_at_batch(
            np.array([t1]), np.array([t2]), np.zeros(1), np.zeros(1)
        )[0, 0, 0]
        assert got == pytest.approx(
            _reference_C00(temporal, t1, t2), rel=2e-3, abs=0.0)


@pytest.mark.heavy
def test_a_value_does_not_depend_on_the_separation_read_first():
    """The first read decided for every later separation.  At a separation
    where the spatial factor is 0 the swap compares 0 with 0: 0.79377 against
    0.57343, with one table built either way."""
    temporal = _ShortOneSided(1.1)
    spatial = _Triangle(1.5)
    near, far = 0.5, 2.0

    def read(cache, r):
        return cache.C_at_batch(
            np.array([2.0]), np.array([2.4]), np.zeros(1), np.array([r])
        )[0, 0, 0]

    ref = spatial(near) * _reference_C00(temporal, 2.0, 2.4)
    cache = _lazy_props(temporal, spatial).cache
    assert read(cache, far) == 0.0
    assert read(cache, near) == pytest.approx(ref, rel=2e-3, abs=0.0)


def _bump_x(n1, n2):
    """Tells only the interior grid pair (0, 1) of ``-2 .. 2``."""
    a, b = float(n1), float(n2)
    return 0.5 * np.exp(-((a - 0.0) ** 2 + (b - 1.0) ** 2) / 0.02)


def _bump_cos(n1, n2):
    """Tells only ``cos = 0.5``, an interior value of a grid of five."""
    return 0.5 * np.exp(-((float(np.dot(n1, n2)) - 0.5) ** 2) / 0.005)


UNIT = np.array([1.0, 0.0])
AT_60_DEGREES = np.array([0.5, np.sqrt(0.75)])


@pytest.mark.parametrize("mode, bump, x, y, build", [
    pytest.param("general", _bump_x, 0.0, 1.0,
                 lambda c: c.precompute_C_table_general(
                     t_max=3.0, n_grid_t=5, x_max=2.0, n_grid_x=5),
                 marks=pytest.mark.heavy),
    ("general", _bump_x, 0.0, 1.0,
     lambda c: c.precompute_C_table_general(t_max=3.0, n_grid_t=5)),
    ("rotation", _bump_cos, UNIT, AT_60_DEGREES,
     lambda c: c.precompute_C_table_rotation(
         t_max=3.0, n_grid_t=5, n_grid_cos=5)),
], ids=["general", "general_lazy", "rotation"])
def test_a_table_of_any_homogeneity_keeps_a_kernel_one_sided_at_one_cell(
        mode, bump, x, y, build):
    """18.4 percent off on the general table, 36.8 on the rotation one, when
    the swap was not read at the one cell it fails at.  The reference is the
    defining integral, by scipy alone."""
    from scipy.integrate import dblquad

    def kernel(n1, t1, n2, t2):
        return np.eye(2) * np.exp(-abs(t1 - t2)) * (
            1.0 + bump(n1, n2) * np.tanh(t1 - t2))

    model = ev.PropagatorModel(
        R_time=lambda t, s: 1.0 if t > s else 0.0, kappa2=kernel,
        n_components=2, diag_C=False)
    cache = ev.PropagatorCache(model, homogeneity=mode)
    build(cache)
    t1, t2 = 0.75, 2.25
    x_b, y_b = np.array([x]), np.array([y])
    got = cache.C_at_batch(np.array([t1]), np.array([t2]), x_b, y_b)[0, 0, 0]
    ref = dblquad(lambda b, a: kernel(x, a, y, b)[0, 0], 0.0, t1, 0.0, t2,
                  epsabs=1e-9)[0]
    assert got == pytest.approx(ref, rel=2e-3, abs=0.0)
