"""Spectral (disorder-averaged) propagators.

Every ground truth here is derived independently of the module under test:
the Ornstein-Uhlenbeck closed form, an explicit two-rate average done by hand,
and scipy quadrature of the defining integral.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from sft_wick import Action, Field, Vertex, compute_moment
from sft_wick.evaluate import integrate_moment
from sft_wick.spectral import SpectralDensity, spectral_cache

D_NOISE = 0.5


def _ou_R(h, t, tp):
    """R(t,t') for dx/dt = -h x + xi — retarded."""
    return np.exp(-h * (t - tp)) if t > tp else 0.0


def _ou_C(h, t1, t2, D=D_NOISE):
    """C(t1,t2) for the same, with x(0) = 0 and <xi xi> = 2 D delta."""
    return (D / h) * (np.exp(-h * abs(t1 - t2)) - np.exp(-h * (t1 + t2)))


# --------------------------------------------------------------------- #
# SP1 — the single-node reduction.  Boundary validation: where the spectral
# form and the plain OU form are BOTH valid they must agree exactly.
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("h", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("t1,t2", [(3.0, 1.0), (1.0, 3.0), (2.0, 2.0),
                                   (0.0, 0.0), (5.0, 0.25)])
def test_SP1_delta_density_reduces_to_ornstein_uhlenbeck(h, t1, t2):
    cache = spectral_cache(SpectralDensity.delta(h), D_NOISE)

    got_c = float(cache.C_diagonal(0, t1, 0, t2)[0])
    assert got_c == pytest.approx(_ou_C(h, t1, t2), rel=1e-12, abs=1e-15)

    # R goes through the batch accessor, which is where Theta lives.
    got_r = float(cache.R_time_batch(np.array([t1]), np.array([t2]))[0])
    assert got_r == pytest.approx(_ou_R(h, t1, t2), rel=1e-12, abs=1e-15)


def test_SP1_theta_is_the_librarys_not_a_hand_rolled_one():
    """Acausal and equal times give exactly 0 through the batch accessor,
    while the RAW model accessor stays Theta-stripped per the package
    convention (`PropagatorCache.R_time` is the raw hook; Theta is applied at
    diagram evaluation)."""
    cache = spectral_cache(1.0, D_NOISE)
    t1 = np.array([3.0, 2.0, 1.0])
    t2 = np.array([1.0, 2.0, 3.0])          # causal, equal, acausal
    out = cache.R_time_batch(t1, t2)
    assert out[0] == pytest.approx(np.exp(-2.0), rel=1e-12)
    assert out[1] == 0.0 and out[2] == 0.0
    # raw accessor: no Theta, and it is the spectral sum
    assert cache.model.R_time(1.0, 3.0) == pytest.approx(np.exp(2.0), rel=1e-12)


# --------------------------------------------------------------------- #
# SP2 — a genuine superposition, checked against a hand average
# --------------------------------------------------------------------- #

def test_SP2_two_rate_density_is_the_weighted_mean_of_two_OU_answers():
    h1, h2, w1 = 0.5, 3.0, 0.25
    dens = SpectralDensity(np.array([h1, h2]), np.array([w1, 1.0 - w1]))
    cache = spectral_cache(dens, D_NOISE)
    for t1, t2 in [(4.0, 1.0), (2.0, 2.0), (0.5, 3.5)]:
        want_c = w1 * _ou_C(h1, t1, t2) + (1 - w1) * _ou_C(h2, t1, t2)
        assert float(cache.C_diagonal(0, t1, 0, t2)[0]) == pytest.approx(
            want_c, rel=1e-12)
        want_r = w1 * _ou_R(h1, t1, t2) + (1 - w1) * _ou_R(h2, t1, t2)
        assert float(cache.R_time_batch(np.array([t1]), np.array([t2]))[0]) \
            == pytest.approx(want_r, rel=1e-12, abs=1e-15)


def test_SP2_superposition_is_not_a_single_exponential():
    """The point of the construction: R* is non-Markovian.

    If a two-rate R* happened to equal some single-rate OU, the module would
    be testable but useless -- so pin that it does not.
    """
    dens = SpectralDensity(np.array([0.5, 4.0]), np.array([0.5, 0.5]))
    cache = spectral_cache(dens, D_NOISE)
    dts = np.array([0.25, 1.0, 3.0])
    r = cache.R_time_batch(dts, np.zeros_like(dts))
    # An exponential would make log R linear in dt; a superposition is convex.
    slopes = np.diff(np.log(r)) / np.diff(dts)
    assert slopes[1] > slopes[0] + 0.05, (
        f"log R* is too close to linear: slopes {slopes}"
    )


# --------------------------------------------------------------------- #
# SP3 — quadrature reductions converge to the defining integral
# --------------------------------------------------------------------- #

def test_SP3_from_callable_converges_to_the_scipy_integral():
    """rho(h) = 2h on [0,1] (normalised).  Reference: scipy.quad of the
    defining integral, not another path through this module."""
    def rho(h):
        return 2.0 * np.asarray(h, dtype=float)

    t1, t2 = 3.0, 1.0
    ref, _ = quad(lambda h: rho(h) * _ou_C(h, t1, t2), 1e-9, 1.0, limit=300)

    errs = {}
    for n in (8, 32, 128):
        cache = spectral_cache(
            SpectralDensity.from_callable(rho, 1e-9, 1.0, n_nodes=n), D_NOISE)
        got = float(cache.C_diagonal(0, t1, 0, t2)[0])
        errs[n] = abs(got - ref) / abs(ref)
    assert errs[128] < errs[8], errs
    assert errs[128] < 1e-8, errs


def test_SP3_from_samples_reproduces_the_sample_mean():
    """Equal-mass binning must not bias the average.

    Reference: the average over the FULL sample, computed directly.
    """
    rng = np.random.default_rng(0)
    samples = rng.gamma(shape=2.0, scale=1.0, size=20_000) + 0.05
    t1, t2 = 2.5, 0.75
    exact = float(np.mean([_ou_C(h, t1, t2) for h in samples]))
    # Equal-mass binning places each node at its bin mean, so it is exact only
    # for a function that is linear across a bin.  C is convex in h, so the
    # reduction converges from one side -- test the CONVERGENCE, not a single
    # tolerance at a coarse n, which would either be vacuous or arbitrary.
    errs = {}
    for n in (16, 64, 256):
        cache = spectral_cache(
            SpectralDensity.from_samples(samples, n_nodes=n), D_NOISE)
        got = float(cache.C_diagonal(0, t1, 0, t2)[0])
        errs[n] = abs(got - exact) / abs(exact)
    assert errs[64] < errs[16] and errs[256] < errs[64], errs
    # Node-at-bin-mean is the optimal 1-point rule per bin (exact for a linear
    # f), so the residual is the curvature term and the rate should be ~2.
    # Pinning the RATE says more than an absolute bound at one n: observed
    # 1.81e-2 / 1.54e-3 / 1.15e-4 for n = 16 / 64 / 256.
    rate = np.log(errs[16] / errs[256]) / np.log(256 / 16)
    assert 1.7 < rate < 2.6, f"convergence rate {rate:.2f}, errs {errs}"
    assert errs[256] < 5e-4, errs


def test_SP3_shift_adds_to_every_rate():
    """`h + lambda` — a regulariser or weight decay."""
    lam = 0.7
    a = spectral_cache(SpectralDensity.delta(1.0), D_NOISE, shift=lam)
    b = spectral_cache(SpectralDensity.delta(1.0 + lam), D_NOISE)
    for t1, t2 in [(3.0, 1.0), (2.0, 2.0)]:
        assert float(a.C_diagonal(0, t1, 0, t2)[0]) == pytest.approx(
            float(b.C_diagonal(0, t1, 0, t2)[0]), rel=1e-13)


# --------------------------------------------------------------------- #
# SP4 — no diagonal ridge, because nothing is tabulated
# --------------------------------------------------------------------- #

def test_SP4_diagonal_is_exact_not_interpolated():
    """The failure mode this module exists to avoid.

    A tabulated C is O(h) on the diagonal because C has a derivative jump of
    -2D there; evaluating the spectral sum directly is exact at every t, so
    there is no grid to refine and no ridge to resolve.
    """
    cache = spectral_cache(SpectralDensity.delta(1.0), D_NOISE)
    # deliberately off any plausible grid node
    for t in (0.0137, 0.5001, 2.71828, 4.99999):
        assert float(cache.C_diagonal(0, t, 0, t)[0]) == pytest.approx(
            _ou_C(1.0, t, t), rel=1e-13)
    # and the derivative jump is the one the docstring claims
    t, eps = 2.0, 1e-6
    up = (_ou_C(1.0, t + eps, t) - _ou_C(1.0, t, t)) / eps
    dn = (_ou_C(1.0, t, t) - _ou_C(1.0, t - eps, t)) / eps
    assert (up - dn) == pytest.approx(-2.0 * D_NOISE, rel=1e-4)


def test_SP4_batch_and_scalar_accessors_agree():
    """Three entry points to the same value must not drift apart."""
    cache = spectral_cache(
        SpectralDensity(np.array([0.4, 2.0]), np.array([0.3, 0.7])), D_NOISE)
    ts1 = np.array([0.3, 1.7, 3.0, 2.0])
    ts2 = np.array([0.3, 0.4, 1.0, 2.0])
    batch = cache.C_diagonal_batch(ts1, ts2)
    for k in range(len(ts1)):
        scalar = float(cache.C_diagonal(0, ts1[k], 0, ts2[k])[0])
        matrix = float(cache.C_value(0, ts1[k], 0, ts2[k])[0, 0])
        assert batch[k, 0] == pytest.approx(scalar, rel=1e-14)
        assert matrix == pytest.approx(scalar, rel=1e-14)


# --------------------------------------------------------------------- #
# SP5 — it actually drives a diagram
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("method", ["nquad", "qmc_vectorized", "gauss_legendre"])
def test_SP5_order0_through_the_integrators_is_C_star(method):
    dens = SpectralDensity(np.array([0.5, 2.5]), np.array([0.4, 0.6]))
    cache = spectral_cache(dens, D_NOISE)
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    res = compute_moment(
        [phi("x"), phi("y")],
        Action([Vertex(fields=[psi, phi, phi, phi], coupling="g")]),
        order=0, ito=True, response_phase=True, collect_topology=True,
        diag_R=True, diag_C=True, iso_R=True, iso_C=True,
    )
    T = 3.0
    kw = ({"n_samples": 2 ** 12, "seed": 1} if method.startswith("qmc")
          else {"n_gauss": 12} if method == "gauss_legendre" else {})
    total = sum(
        integrate_moment(dt.build_integrand({"g": np.array(1j)}), T, cache,
                         method=method, t_min=0.0, **kw)[0]
        for dt in res.diagram_terms(0)
    )
    want = 0.4 * _ou_C(0.5, T, T) + 0.6 * _ou_C(2.5, T, T)
    assert total == pytest.approx(want, rel=1e-9)


def test_SP5_two_time_response_is_the_spectral_R():
    """R*(T, t') through the integrators, at unequal external times."""
    dens = SpectralDensity(np.array([0.5, 2.5]), np.array([0.4, 0.6]))
    cache = spectral_cache(dens, D_NOISE)
    phi = Field("phi", "physical")
    psi = Field("psi", "response")
    res = compute_moment(
        [phi("x"), psi("y")],
        Action([Vertex(fields=[psi, phi, phi, phi], coupling="g")]),
        order=0, ito=True, response_phase=True, collect_topology=True,
        diag_R=True, diag_C=True, iso_R=True, iso_C=True,
    )
    T, tp = 4.0, 1.5
    total = sum(
        integrate_moment(dt.build_integrand({"g": np.array(1j)}), T, cache,
                         method="nquad", t_min=0.0,
                         external_times={"x": T, "y": tp})[0]
        for dt in res.diagram_terms(0)
    )
    want = 0.4 * _ou_R(0.5, T, tp) + 0.6 * _ou_R(2.5, T, tp)
    assert total == pytest.approx(want, rel=1e-9)
    assert total != 0.0


# --------------------------------------------------------------------- #
# SP6 — validation and serialisability
# --------------------------------------------------------------------- #

def test_SP6_density_validation():
    with pytest.raises(ValueError, match="same shape"):
        SpectralDensity(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="at least one node"):
        SpectralDensity(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="non-negative"):
        SpectralDensity(np.array([1.0]), np.array([-1.0]))
    with pytest.raises(ValueError, match="sum to zero"):
        SpectralDensity(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        SpectralDensity(np.array([1.0, np.inf]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="hi > lo"):
        SpectralDensity.from_callable(lambda h: np.ones_like(h), 1.0, 1.0)
    # weights are normalised on construction
    d = SpectralDensity(np.array([1.0, 2.0]), np.array([3.0, 1.0]))
    assert d.weights.sum() == pytest.approx(1.0)
    assert d.weights[0] == pytest.approx(0.75)


def test_SP6_cache_survives_joblib_round_trip(tmp_path):
    """It must go through `propagators.cache_path` and loky like any cache.

    Uses joblib because that is the production path; a bound-method callable
    on the instance is exactly the shape that breaks naive serialisation.
    """
    joblib = pytest.importorskip("joblib")
    cache = spectral_cache(
        SpectralDensity(np.array([0.5, 2.0]), np.array([0.5, 0.5])), D_NOISE)
    before = float(cache.C_diagonal(0, 2.0, 0, 1.0)[0])
    path = tmp_path / "spectral_cache.joblib"
    joblib.dump(cache, path)
    after = float(joblib.load(path).C_diagonal(0, 2.0, 0, 1.0)[0])
    assert after == before


# --------------------------------------------------------------------- #
# SP7 — the two regimes the first draft got wrong
# --------------------------------------------------------------------- #
#
# Both were found by checking the module against the DEFINING integral
#
#     C(t1,t2) = int_{t_min}^{min(t1,t2)} R(t1,l) 2D R(t2,l) dl
#
# rather than against the closed form the module itself implements.


def _c_defining(h, t1, t2, t_min, D=D_NOISE):
    """C from its definition, by quadrature — independent of the module."""
    m = min(t1, t2)
    if m <= t_min:
        return 0.0
    v, _ = quad(
        lambda l: np.exp(-h * (t1 - l)) * 2 * D * np.exp(-h * (t2 - l)),
        t_min, m, limit=200,
    )
    return v


@pytest.mark.parametrize("t_min", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("t1,t2", [(3.0, 2.0), (2.5, 2.5), (4.0, 1.2)])
def test_SP7_t_min_shifts_the_initial_condition(t_min, t1, t2):
    """`x(t_min) = 0`, not `x(0) = 0`.

    The first draft accepted `t_min` and then hard-coded the initial condition
    at 0 -- wrong by 1.5% at t_min=0.5 and 7.4% at t_min=1.0, silently.
    """
    h = 1.3
    cache = spectral_cache(SpectralDensity.delta(h), D_NOISE, t_min=t_min)
    got = float(cache.C_diagonal(0, t1, 0, t2)[0])
    assert got == pytest.approx(_c_defining(h, t1, t2, t_min), rel=1e-10)


def test_SP7_before_t_min_there_is_no_accumulated_noise():
    cache = spectral_cache(SpectralDensity.delta(1.3), D_NOISE, t_min=2.0)
    assert float(cache.C_diagonal(0, 1.0, 0, 1.0)[0]) == 0.0
    assert float(cache.C_diagonal(0, 2.0, 0, 0.5)[0]) == 0.0


@pytest.mark.parametrize("h", [0.0, 1e-14, 1e-8])
@pytest.mark.parametrize("t_min", [0.0, 1.0])
def test_SP7_zero_rate_is_free_diffusion_not_a_division_by_zero(h, t_min):
    """C carries a `D/h`, but C itself is finite as h -> 0.

    The limit is `2 D (min(t1,t2) - t_min)` -- free diffusion.  Naively the
    prefactor would give inf or nan.
    """
    t1, t2 = 3.0, 2.0
    cache = spectral_cache(SpectralDensity.delta(h), D_NOISE, t_min=t_min)
    got = float(cache.C_diagonal(0, t1, 0, t2)[0])
    assert np.isfinite(got)
    want = 2.0 * D_NOISE * (min(t1, t2) - t_min)
    assert got == pytest.approx(want, rel=1e-6)
    if h > 0:  # and it agrees with the defining integral too
        assert got == pytest.approx(_c_defining(h, t1, t2, t_min), rel=1e-6)


def test_SP7_negative_rates_are_rejected():
    """A negative rate is an unstable mode with no stationary C."""
    with pytest.raises(ValueError, match="must be >= 0"):
        SpectralDensity(np.array([1.0, -0.5]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="must be >= 0"):
        spectral_cache(np.array([-1.0]), D_NOISE)


def test_SP7_small_h_does_not_lose_precision_to_cancellation():
    """The two exponentials nearly cancel for small h, so the difference goes
    through `expm1`.  A naive `exp(-x) - exp(-y)` loses digits here."""
    for h in (1e-6, 1e-4, 1e-2):
        cache = spectral_cache(SpectralDensity.delta(h), D_NOISE)
        got = float(cache.C_diagonal(0, 3.0, 0, 2.0)[0])
        assert got == pytest.approx(_c_defining(h, 3.0, 2.0, 0.0), rel=1e-9)
