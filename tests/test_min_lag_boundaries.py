"""C tables preserve spatial order and the declared physical time domain."""

import numpy as np
import pytest

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


def test_new_schema_rejects_the_incorrect_min_lag_cache(tmp_path):
    from joblib import dump, hash as joblib_hash
    from sft_wick.workflow.cache import load_or_compute

    spec = {"t_max": 1.0, "n_grid_t": 5}
    path = tmp_path / "props.joblib"
    dump({"key": joblib_hash(("2", spec))[:12], "value": "incorrect E"}, path)
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
