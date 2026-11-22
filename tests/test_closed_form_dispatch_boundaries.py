"""Boundary validation of the C-propagator dispatcher.

``Propagators.build`` now chooses between three evaluators of the same
quantity ``C = ∫∫ R κ² R``:

* the built-in closed form (:mod:`sft_wick.workflow.closed_forms`), taken
  automatically for the OU-drift / exponential-temporal family;
* tensor-product Gauss-Legendre with the diagonal split
  (``c_method='gauss_legendre'``, the default quadrature);
* adaptive ``scipy.integrate.dblquad`` (the fallback).

Testing the dispatcher's *output* across its threshold only shows the
function's own continuity.  These tests bypass it and evaluate every
method directly at the same cells -- including the extreme ones where a
formula tends to break: the diagonal ridge ``t1 == t2``, ``t → t_min``
(empty domain), ``t = t_max``, the removable singularity ``γ == 1/σ_t``,
``r = 0`` and ``r = r_max``, per-component ``γ``, a non-zero ``t_min``
and a white-noise impulse -- and require all three to agree.

Tolerances: closed form vs GL(n=20) at ``1e-10`` relative (GL is
converged to machine precision at these ``(γ + 1/σ_t) t_max``), and
closed form vs dblquad at ``1e-8`` (dblquad is run at ``epsrel=1e-10``;
the package splits the rectangle at the ``λ1 = λ2`` cusp, without which
scipy reports roundoff and delivers only ~2e-6).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.evaluate import PropagatorCache, select_gl_node_count
from sft_wick.workflow import closed_forms
from sft_wick.workflow.closed_forms import (
    ClosedFormC,
    builtin_closed_form_for,
    ou_exponential_phi,
)

T_MAX = 10.0
R_MAX = 2.5
GL_TOL = 1e-10
DBLQUAD_TOL = 1e-8
# Relative error floor: cells with ``|C| < FLOOR_FRAC * max|C|`` are judged
# against that fraction of the table scale instead of their own value.
FLOOR_FRAC = 1e-6


def _system(gamma, *, sigma_t=0.5, lam=0.05, t_min=0.0, sigma2=None,
            spatial=None, n_components=2):
    F = np.zeros((n_components,) * 3)
    F[0, 1 % n_components, 1 % n_components] = 1.0
    return sw.System(
        field=sw.FieldSpec("phi", n_components=n_components),
        linear=sw.DiagonalA(gamma=gamma),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=lam, sigma_t=sigma_t),
                spatial=spatial or sw.ExponentialSpatial(sigma_x=1.0),
            ),
            sigma2=sigma2,
        ),
        t_min=t_min,
    )


SYSTEMS = {
    "iso": _system([1.0, 1.0]),
    "per_component_gamma": _system([0.7, 2.0]),
    "gamma_equals_inv_sigma_t": _system([2.0, 2.0]),          # γ = 1/σ_t exactly
    "gamma_near_inv_sigma_t": _system([2.0 + 1e-9, 2.0 - 1e-7]),
    "t_min_nonzero": _system([1.0, 1.5], t_min=0.75),
    "white_noise": _system([1.0, 1.0], sigma2=sw.ConstantImpulse(0.1)),
    "gaussian_spatial": _system([1.0, 1.0], spatial=sw.GaussianSpatial(sigma_x=0.8)),
}


def _cells(t_min: float):
    """Time pairs covering the ridge, both limits and asymmetric corners."""
    h = (T_MAX - t_min) / 59.0        # the default 60-point grid spacing
    mid = 0.5 * (T_MAX + t_min)
    return [
        (t_min, t_min),               # empty domain → exactly 0
        (t_min, T_MAX),               # one empty axis → exactly 0
        (t_min + 1e-9, t_min + 1e-9), # t → t_min⁺ on the ridge
        (t_min + h, t_min + h),       # first grid cell on the ridge
        (mid, mid),                   # ridge, mid table
        (T_MAX, T_MAX),               # ridge, corner
        (T_MAX - h, T_MAX),           # just off the ridge
        (T_MAX, t_min + h),           # extreme aspect ratio (strip region)
        (t_min + h, T_MAX),           # mirror
        (mid, T_MAX),
    ]


RS = [0.0, 0.5, R_MAX]


def _direct(cache: PropagatorCache, r: float, t1: float, t2: float, **kw):
    return np.asarray(cache._C_value_direct(
        np.asarray(0.0), t1, np.asarray(r), t2, **kw,
    ))


def _rel_err(got: np.ndarray, ref: np.ndarray, scale: float) -> float:
    denom = np.maximum(np.abs(ref), FLOOR_FRAC * scale)
    return float(np.max(np.abs(got - ref) / denom))


# --------------------------------------------------------------------- #
# Method-vs-method agreement, bypassing the dispatcher                   #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_closed_form_matches_gauss_legendre_at_every_boundary_cell(name):
    system = SYSTEMS[name]
    cf = builtin_closed_form_for(system)
    assert isinstance(cf, ClosedFormC), f"{name}: closed form not detected"
    model = system.build_propagator_model()
    cache = PropagatorCache(model, c_method="gauss_legendre", n_gauss=20)
    scale = float(np.max(np.abs(cf(0.0, T_MAX, 0.0, T_MAX))))
    assert scale > 0
    worst = 0.0
    for r in RS:
        for t1, t2 in _cells(system.t_min):
            got = np.asarray(cf(0.0, t1, r, t2))
            ref = _direct(cache, r, t1, t2)
            assert got.shape == ref.shape == (2, 2)
            assert np.all(np.isfinite(got))
            err = _rel_err(got, ref, scale)
            worst = max(worst, err)
            assert err < GL_TOL, (
                f"{name}: closed form vs GL(20) at r={r}, t=({t1}, {t2}): "
                f"cf={got.ravel()}, gl={ref.ravel()}, rel={err:.2e}"
            )
    assert worst < GL_TOL


@pytest.mark.parametrize("name", ["iso", "per_component_gamma",
                                  "gamma_equals_inv_sigma_t", "t_min_nonzero",
                                  "white_noise"])
def test_closed_form_matches_dblquad_at_corner_cells(name):
    """dblquad is the slow reference; keep it to the cells that matter."""
    system = SYSTEMS[name]
    cf = builtin_closed_form_for(system)
    model = system.build_propagator_model()
    cache = PropagatorCache(
        model, c_method="dblquad", quad_opts={"epsabs": 1e-13, "epsrel": 1e-10},
    )
    scale = float(np.max(np.abs(cf(0.0, T_MAX, 0.0, T_MAX))))
    t_min = system.t_min
    h = (T_MAX - t_min) / 59.0
    cells = [(t_min + h, t_min + h), (T_MAX, T_MAX), (T_MAX, t_min + h),
             (0.5 * (T_MAX + t_min), T_MAX)]
    for r in (0.0, R_MAX):
        for t1, t2 in cells:
            got = np.asarray(cf(0.0, t1, r, t2))
            ref = _direct(cache, r, t1, t2)
            err = _rel_err(got, ref, scale)
            assert err < DBLQUAD_TOL, (
                f"{name}: closed form vs dblquad at r={r}, t=({t1}, {t2}): "
                f"cf={got.ravel()}, db={ref.ravel()}, rel={err:.2e}"
            )


def test_gl_and_dblquad_agree_when_closed_form_is_bypassed():
    """The two quadratures on a Gaussian temporal kernel agree at extreme cells."""
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=1),
        linear=sw.DiagonalA(gamma=[1.0]),
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.GaussianTemporal(lam=0.05, sigma_t=0.5),
            spatial=sw.ExponentialSpatial(sigma_x=1.0),
        )),
    )
    model = system.build_propagator_model()
    # The node count the 'auto' dispatcher would pick for this horizon
    # (a fixed n=20 is only at ~8e-8 here; the selector escalates).
    n = select_gl_node_count(model, T_MAX)
    assert n is not None and n >= 20
    gl = PropagatorCache(model, c_method="gauss_legendre", n_gauss=n)
    db = PropagatorCache(
        model, c_method="dblquad", quad_opts={"epsabs": 1e-13, "epsrel": 1e-10},
    )
    scale = float(np.max(np.abs(_direct(gl, 0.0, T_MAX, T_MAX))))
    h = T_MAX / 59.0
    for r in (0.0, R_MAX):
        for t1, t2 in [(h, h), (T_MAX, T_MAX), (T_MAX, h), (5.0, T_MAX)]:
            err = _rel_err(_direct(gl, r, t1, t2), _direct(db, r, t1, t2), scale)
            assert err < 1e-7, f"GL(n={n}) vs dblquad at r={r}, t=({t1},{t2}): {err:.2e}"


# --------------------------------------------------------------------- #
# Closed-form self-checks at the analytic limits                         #
# --------------------------------------------------------------------- #


def test_phi_is_symmetric_and_reaches_the_stationary_limit():
    g, a = 1.3, 1.0 / 0.3
    # Φ is manifestly symmetric: the implementation reduces (t₁, t₂) to
    # (min, max) before any arithmetic, so the two orderings are bit-for-bit
    # identical (observed rel deviation exactly 0.0).  `abs=0.0` makes the
    # written rel the one in force: Φ = 5.5e-3 here, so approx's 1e-12
    # default floor had been enforcing 1.8e-10, not 1e-14 -- 4 orders of
    # magnitude looser than the line claimed.
    assert ou_exponential_phi(g, a, 2.0, 5.0) == pytest.approx(
        ou_exponential_phi(g, a, 5.0, 2.0), rel=1e-14, abs=0.0)
    # Stationary variance λ Φ(∞, ∞) = λ / (γ (γ + a)).  At γt = 520 the
    # transient exp(-2γt) has underflowed, so only roundoff is left and the
    # agreement is 1 ULP (rel 1.7e-16).  `abs=0.0`: Φ = 0.166, so the 1e-12
    # floor had been enforcing 6.0e-12 rather than the written 1e-13.
    assert ou_exponential_phi(g, a, 400.0, 400.0) == pytest.approx(
        1.0 / (g * (g + a)), rel=1e-13, abs=0.0)
    # No overflow deep in the domain: the textbook form multiplies
    # exp(+2γt) by exp(-γ(t₁+t₂)), and the first factor is already inf at
    # γt = 1.3e4, whereas every exponent in this one is non-positive.
    assert np.isfinite(ou_exponential_phi(g, a, 1e4, 1e4))
    # ... and it has reached the stationary limit there, where Φ depends on
    # the separation alone -- so Φ(T, T−d) must not depend on T.
    #
    # This replaces an assertion that could not fail: it compared
    # `Φ(1e4, 9.9e3)` against `np.exp(-g*100.0)*0.0 + Φ(1e4, 9.9e3)`, and
    # since that first term is exactly 0.0 both sides were the same
    # expression.  `abs=0.0` is required as well, not decoration: Φ is
    # 9.5e-58 at this separation, far below approx's 1e-12 default floor,
    # so a bare `rel=` would accept any value at all.
    d = 100.0
    assert ou_exponential_phi(g, a, 1e4, 1e4 - d) == pytest.approx(
        ou_exponential_phi(g, a, 1e3, 1e3 - d), rel=1e-13, abs=0.0)
    # Empty domain.
    assert ou_exponential_phi(g, a, 0.0, 3.0) == 0.0
    assert ou_exponential_phi(g, a, -1.0, 3.0) == 0.0


def test_gamma_zero_limit_is_finite_and_continuous():
    """γ = 0 (free diffusion) is a removable singularity of the formula."""
    a = 2.0
    v0 = ou_exponential_phi(0.0, a, 1.5, 2.5)
    v1 = ou_exponential_phi(1e-7, a, 1.5, 2.5)
    assert np.isfinite(v0)
    assert v0 == pytest.approx(v1, rel=1e-5, abs=0.0)


def test_closed_form_batched_and_scalar_contracts_agree():
    cf = builtin_closed_form_for(SYSTEMS["white_noise"])
    t1 = np.array([0.3, 2.0, T_MAX, 4.0])
    t2 = np.array([0.3, 5.0, T_MAX, 1.0])
    x1 = np.zeros(4)
    x2 = np.array([0.0, 0.5, R_MAX, 1.0])
    batched = cf(x1, t1, x2, t2)
    assert batched.shape == (4, 2, 2)
    for i in range(4):
        single = cf(x1[i], float(t1[i]), x2[i], float(t2[i]))
        assert single.shape == (2, 2)
        np.testing.assert_allclose(batched[i], single, rtol=0, atol=0)
    # d-dim positions: the separation is the Euclidean norm.
    vec = cf(np.array([0.0, 0.0, 0.0]), 2.0, np.array([0.3, 0.4, 0.0]), 2.0)
    np.testing.assert_allclose(vec, cf(0.0, 2.0, 0.5, 2.0), rtol=1e-15)


# --------------------------------------------------------------------- #
# Detection: which systems get the closed form                           #
# --------------------------------------------------------------------- #


def test_detection_rejects_every_escape_hatch():
    base = SYSTEMS["iso"]
    assert builtin_closed_form_for(base) is not None

    custom_temporal = sw.System(
        field=base.field, linear=base.linear, vertices=base.vertices,
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.CustomKernel(fn=sw.GaussianTemporal(lam=0.05, sigma_t=0.3)),
            spatial=sw.ExponentialSpatial(sigma_x=1.0))),
    )
    assert builtin_closed_form_for(custom_temporal) is None

    time_dep = sw.System(
        field=base.field, vertices=base.vertices, noise=base.noise,
        linear=sw.DiagonalA(gamma=lambda t: np.array([1.0, 1.0])),
    )
    assert builtin_closed_form_for(time_dep) is None

    explicit = sw.System(
        field=base.field, vertices=base.vertices, noise=base.noise,
        linear=sw.ExplicitR(R_time=lambda t1, t2: np.exp(-(t1 - t2))),
    )
    assert builtin_closed_form_for(explicit) is None

    rotation = sw.System(
        field=base.field, linear=base.linear, vertices=base.vertices,
        noise=sw.GaussianNoise(kappa2=sw.SeparableRotation(
            temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
            angular=sw.LegendreAngular(coeffs=[1.0, 0.5]))),
    )
    assert builtin_closed_form_for(rotation) is None

    custom_impulse = sw.System(
        field=base.field, linear=base.linear, vertices=base.vertices,
        noise=sw.GaussianNoise(kappa2=base.noise.kappa2,
                               sigma2=sw.CustomImpulse(lambda n1, t, n2: np.eye(2))),
    )
    assert builtin_closed_form_for(custom_impulse) is None

    # A general (callable) kappa2 has no structure to exploit.
    general = sw.System(
        field=base.field, linear=base.linear, vertices=base.vertices,
        noise=sw.GaussianNoise(kappa2=sw.GeneralKappa2(
            fn=lambda n1, t1, n2, t2: 0.05 * np.eye(2))),
    )
    assert builtin_closed_form_for(general) is None


@pytest.mark.parametrize("kind", ["Gaussian", "Matern32", "DampedCosine"])
def test_stationary_forms_are_direct_and_match_defining_integral(kind):
    from scipy.integrate import quad

    lam, width, omega = 0.06, 0.5, 2.1
    kernel_type = getattr(sw, kind + "Temporal")
    params = {"lam": lam, "sigma_t": width}
    if kind == "DampedCosine":
        params["omega"] = omega
    kernel = kernel_type(**params)
    base = SYSTEMS["per_component_gamma"]
    rates = np.array([0.8, 1.9])
    spatial = sw.ExponentialSpatial(sigma_x=0.8)
    system = sw.System(
        field=base.field, linear=sw.DiagonalA(gamma=rates), vertices=base.vertices,
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=kernel, spatial=spatial)),
        t_min=0.5,
    )
    props = system.propagators(t_max=4.0, n_grid_t=8, diag_C=False)
    assert props.c_source == "closed_form:builtin"
    assert props.cache._closed_form_only
    t1 = np.array([0.5, 0.50004, 0.9, 2.3])
    t2 = np.array([1.6, 0.5002, 1.6, 0.9])
    r = 0.7
    got = props.cache.C_at_batch(t1, t2, np.zeros(4), np.full(4, r))
    swapped = props.cache.C_at_batch(t2, t1, np.full(4, r), np.zeros(4))
    np.testing.assert_array_equal(got, swapped)

    def ref(gamma, a, b):
        def outer(s):
            points = [s] if 0.5 < s < b else None
            return quad(
                lambda u: np.exp(-gamma * ((a - s) + (b - u))) * kernel(s - u),
                0.5, b, epsabs=1e-27, epsrel=1e-13, points=points,
            )[0]

        points = [b] if 0.5 < b < a else None
        return spatial(r) * quad(outer, 0.5, a, epsabs=1e-27, epsrel=1e-13,
                                 points=points)[0]

    expected = np.array([[ref(g, a, b) for g in rates] for a, b in zip(t1, t2)])
    np.testing.assert_allclose(np.diagonal(got, axis1=-2, axis2=-1), expected,
                               rtol=1e-13, atol=0.0)
    np.testing.assert_array_equal(got[:, 0, 1], np.zeros(4))


@pytest.mark.parametrize("gamma", [0.0, 1e-8])
def test_narrow_gaussian_keeps_finite_physical_lag_decay(gamma):
    from sft_wick.workflow._stationary_c import gaussian_phi
    from sft_wick.workflow.closed_forms import ou_white_psi

    width = 1e-305
    t1, t2 = 3000.0, 10000.0
    # With both the overlap and the gap much larger than the Gaussian width,
    # integrating the normalized kernel gives the white impulse times its area.
    expected = np.sqrt(2.0 * np.pi) * width * ou_white_psi(gamma, gamma, t1, t2)
    with np.errstate(over="raise", invalid="raise"):
        got = gaussian_phi(gamma, gamma, width, t1, t2)
    np.testing.assert_allclose(got, expected, rtol=1e-13, atol=0.0)


def test_stationary_kernels_keep_finite_extreme_limits():
    from sft_wick.workflow._stationary_c import (
        damped_cosine_phi, gaussian_phi, matern32_phi,
    )
    from sft_wick.workflow.closed_forms import ou_white_psi

    with np.errstate(over="raise", invalid="raise"):
        gaussian = gaussian_phi(1e120, 1e120, 1e200, 1e194, 1e194)
        broad_gaussian = gaussian_phi(1e120, 1e120, 1e300, 1e200, 1e200)
        narrow_matern = matern32_phi(1.0, 1.0, 1e170, 2.0, 3.0)
        matern = matern32_phi(1.0, 1.0, 1.0, 1e200, 1e200)
        cosine = damped_cosine_phi(1.0, 1.0, 1.0, 10.0, 1e200, 1e200)
        narrow_cosine = damped_cosine_phi(0.0, 0.0, 1e160, 2.0, 1e151, 1e151)
    expected = [1e-240, 1e-240, 4e-170 * ou_white_psi(1.0, 1.0, 2.0, 3.0),
                0.75, 1.0 / 52.0, 2e-9]
    observed = [gaussian, broad_gaussian, narrow_matern, matern, cosine, narrow_cosine]
    np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("earlier,later", [(0.01, 1e4), (1e-7, 1e6), (1e-7, 1e18)])
def test_gaussian_strip_resolves_decay_at_large_lag(earlier, later):
    from scipy.integrate import quad

    from sft_wick.workflow._stationary_c import gaussian_phi

    # Integrate the original rectangle in response-age coordinates.
    # The second response tail past age 60 is bounded by exp(-60).
    def inner(x):
        return quad(
            lambda age: np.exp(-age - 0.5 * ((later - age - x) / later)**2),
            0.0, 60.0, epsabs=1e-15, epsrel=1e-13,
        )[0] * np.exp(-(earlier - x))

    expected = quad(inner, 0.0, earlier, epsabs=1e-21, epsrel=1e-13)[0]
    np.testing.assert_allclose(gaussian_phi(1.0, 1.0, later, earlier, later), expected,
                               rtol=1e-13, atol=0.0)


# --------------------------------------------------------------------- #
# The dispatcher itself                                                  #
# --------------------------------------------------------------------- #


def test_dispatcher_defaults_to_the_builtin_closed_form_and_agrees_with_quadrature():
    """The default and explicit direct paths agree. Tables built from the
    closed form agree with tables built by forced quadrature."""
    system = SYSTEMS["per_component_gamma"]
    auto = system.propagators(t_max=T_MAX, n_grid_t=30)
    assert auto.c_source.startswith("closed_form:builtin"), auto.c_source
    assert not auto.is_lazy
    direct = system.propagators(t_max=T_MAX, n_grid_t=30, c_closed_form_only=True)
    tabulated = system.propagators(
        t_max=T_MAX, n_grid_t=30, c_closed_form_only=False, table_rtol=None,
    )
    assert tabulated.is_lazy

    quad = system.propagators(t_max=T_MAX, n_grid_t=30, c_closed_form=None, table_rtol=None)
    assert quad.c_source.startswith("quadrature:gauss_legendre"), quad.c_source

    t1 = np.array([0.5, 3.0, T_MAX, 7.0, T_MAX])
    t2 = np.array([0.5, 3.0, T_MAX, 2.0, 0.5])
    for r in RS:
        a = auto.cache.C_at_batch(t1, t2, np.zeros(5), np.full(5, r))
        d = direct.cache.C_at_batch(t1, t2, np.zeros(5), np.full(5, r))
        np.testing.assert_array_equal(a, d)
        a = tabulated.cache.C_at_batch(t1, t2, np.zeros(5), np.full(5, r))
        q = quad.cache.C_at_batch(t1, t2, np.zeros(5), np.full(5, r))
        scale = float(np.max(np.abs(a)))
        assert np.max(np.abs(a - q) / np.maximum(np.abs(a), FLOOR_FRAC * scale)) < 1e-9


def test_dispatcher_checks_custom_kernels_for_convergence():
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=1),
        linear=sw.DiagonalA(gamma=[1.0]),
        noise=sw.GaussianNoise(kappa2=sw.GeneralKappa2(
            fn=lambda n1, t1, n2, t2: 0.05 * np.exp(-abs(t1 - t2)) * np.eye(1))),
    )
    props = system.propagators(t_max=2.0, n_grid_t=8)
    assert props.c_source == "quadrature:gauss_legendre(n=20)", props.c_source


def test_explicit_c_method_is_honoured_over_auto():
    system = SYSTEMS["iso"]
    forced = system.propagators(t_max=T_MAX, n_grid_t=12, c_closed_form=None,
                                c_method="dblquad")
    assert forced.c_source == "quadrature:dblquad"
    gl = system.propagators(t_max=T_MAX, n_grid_t=12, c_closed_form=None,
                            c_method="gauss_legendre", c_n_gauss=24)
    assert gl.c_source == "quadrature:gauss_legendre(n=24)"


def test_user_closed_form_module_still_wins_over_builtin():
    system = SYSTEMS["iso"]
    calls = []

    def C_fn(n1, t1, n2, t2):
        calls.append(1)
        return np.zeros((2, 2))

    props = system.propagators(t_max=T_MAX, n_grid_t=8, c_closed_form=C_fn)
    assert props.c_source == "closed_form:user"
    assert props.is_lazy
    props.cache.C_at_batch(np.array([1.0]), np.array([1.0]), np.zeros(1), np.zeros(1))
    assert calls


def test_builtin_default_preserves_a_requested_full_grid():
    system = SYSTEMS["iso"]
    options = dict(t_max=2.0, n_grid_t=8, r_max=1.0, n_grid_r=3, table_rtol=None)
    auto = system.propagators(**options)
    explicit_table = system.propagators(**options, c_closed_form_only=False)
    t = np.array([0.7, 1.1])
    np.testing.assert_array_equal(
        auto.cache.C_at_batch(t, t, 0.0, 0.4),
        explicit_table.cache.C_at_batch(t, t, 0.0, 0.4),
    )
    with pytest.raises(ValueError):
        auto.cache.C_at_batch(t, t, 0.0, 1.1)


@pytest.mark.parametrize("n", [2, 3])
def test_default_direct_broadcasts_scalar_positions_by_sample(n):
    system = SYSTEMS["iso"]
    props = system.propagators(t_max=1.0)
    times = np.full(n, 0.4)
    scalars = np.linspace(0.0, 1.0, n)
    vectors = np.array([[2.0, 3.0], [4.0, 1.0], [1.0, 2.0]])[:n]
    values = props.cache.C_at_batch(times, times, scalars, vectors)
    expected = props.cache.C_at_batch(times, times, scalars[:, None], vectors)
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(
        props.cache.C_at_batch(times, times, vectors, scalars), expected,
    )


def test_vectorized_user_closed_form_defaults_to_direct():
    def c_fn(n1, t1, n2, t2):
        assert n1.shape[0] == n2.shape[0] == len(t1) == len(t2)
        return np.broadcast_to(np.eye(2), (len(t1), 2, 2))

    system = SYSTEMS["iso"]
    props = system.propagators(
        t_max=T_MAX, c_closed_form=c_fn, c_closed_form_vectorized=True,
    )
    assert not props.is_lazy
    values = props.cache.C_at_batch(np.array([1.0]), np.array([1.0]), 0.0, 0.4)
    np.testing.assert_array_equal(values, np.ones((1, 2)))
    expansion = system.expand(("phi_a(x)", "phi_a(y)"), orders=[0])
    for method in ("qmc_vectorized", "qmc_scalar", "gauss_legendre"):
        result = expansion.evaluate(
            props, positions={"x": 0.0, "y": 0.4}, t_final=1.0, method=method,
        )
        assert result.total == 1.0
    with pytest.raises(ValueError, match="c_closed_form_vectorized"):
        system.propagators(
            t_max=T_MAX, c_closed_form=c_fn, c_closed_form_only=False,
            c_closed_form_vectorized=True,
        )


def test_built_in_closed_form_takes_the_batched_read_whatever_the_flag_says():
    """``c_closed_form_only`` without the vectorised flag ran one Python call
    per sample: 23.7 s against 0.47 s at N = 2, order 3, ``n_gauss = 10``."""
    system = _system((0.8, 1.7), sigma2=sw.ConstantImpulse(np.diag([0.3, 0.5])))
    props = system.propagators(t_max=2.0, c_closed_form_only=True, progress=False)
    assert props.c_source == "closed_form:builtin"
    assert props.cache._closed_form_vectorized
    t1, t2 = np.array([0.4, 1.3, 1.9]), np.array([1.1, 1.3, 0.2])
    batched = props.cache.C_at_batch(t1, t2, 0.0, 0.6)
    for k in range(t1.size):
        row = _direct(props.cache, 0.6, t1[k], t2[k])
        np.testing.assert_allclose(batched[k], np.diag(row), rtol=1e-14)


def _phi_mpmath(gamma, a, t1, t2):
    """The defining integral at 40 digits, split at its cusp."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    gamma, a, t1, t2 = (mp.mpf(v) for v in (gamma, a, t1, t2))

    def inner(l1):
        def f(l2):
            return mp.exp(
                -gamma * (t1 - l1) - a * abs(l1 - l2) - gamma * (t2 - l2)
            )

        cut = min(l1, t2)
        return mp.quad(f, [0, cut]) + (mp.quad(f, [cut, t2]) if cut < t2 else 0)

    return mp.quad(inner, sorted({mp.mpf(0), min(t1, t2), t1}))


@pytest.mark.parametrize("gamma, a, t1, t2", [
    # (gamma + a) t on each side of the dispatch, with the two forms apart.
    (0.3, 0.7 - 1e-9, 1.0, 0.6),
    (0.3, 0.7 + 1e-9, 1.0, 0.6),
    (1.0, 1e-9, 1.0 - 1e-9, 0.4),
    (0.0, 1.0 + 1e-9, 1.0, 1.0),
    # Near-free field, near-constant kernel.
    (0.0, 1e-12, 1.0, 0.5),
    (1e-14, 1e-12, 1.0, 1.0),
    (1e-14, 1e-10, 0.1, 0.01),
    (0.0, 1e-13, 2.0, 1.0),
    (0.0, 1e-15, 1.0, 1.0),
    (1e-14, 1e-8, 30.0, 30.0),
    # Both rates zero, a 0 / 0 of the closed form: Phi = t1 t2.
    (0.0, 0.0, 2.0, 1.0),
    # Far from the dispatch.
    (1.0, 1.0, 30.0, 29.0),
    (0.0, 1e3, 1e-2, 1e-3),
])
def test_phi_against_the_defining_integral_at_small_rates(gamma, a, t1, t2):
    """The closed form cancels to (gamma + a) t^2 and lost it all: 0.25 against
    0.5 at gamma = 0, sigma_t = 1e12, and NaN at sigma_t = inf.  The reference
    shares no code with the package."""
    assert float(ou_exponential_phi(gamma, a, t1, t2)) == pytest.approx(
        float(_phi_mpmath(gamma, a, t1, t2)), rel=1e-13, abs=0.0
    )


def test_a_width_whose_reciprocal_overflows_is_refused():
    """The closed form reads the rate 1 / sigma_t.  It was inf here and C came
    out NaN; a 40-node Gauss-Legendre rule gives 0 on this kernel."""
    with pytest.raises(ValueError, match="sigma_t"):
        sw.ExponentialTemporal(lam=1e300, sigma_t=5.56e-309)


def test_the_narrowest_width_taken_gives_the_white_noise_limit():
    """The other side: C -> 2 lam sigma_t (1 - e^{-2 gamma t}) / (2 gamma)."""
    lam, sigma_t, gamma, t = 1e300, 5.57e-309, 1.0, 2.0
    system = sw.System(
        field=sw.FieldSpec("phi", 1), linear=sw.DiagonalA(gamma=[gamma]),
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=lam, sigma_t=sigma_t),
            spatial=sw.GaussianSpatial(sigma_x=1.0))))
    limit = lam * sigma_t * -np.expm1(-2.0 * gamma * t) / gamma
    C = builtin_closed_form_for(system)(0.0, t, 0.0, t)
    assert np.ravel(C)[0] == pytest.approx(limit, rel=1e-12, abs=0.0)


def test_a_constant_kernel_with_no_drift_stays_on_the_closed_form():
    system = sw.System(
        field=sw.FieldSpec("phi", 1), linear=sw.DiagonalA(gamma=[0.0]),
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=1.0, sigma_t=np.inf),
            spatial=sw.GaussianSpatial(sigma_x=1.0))))
    C = builtin_closed_form_for(system)(0.0, 2.0, 0.0, 1.0)
    assert np.ravel(C)[0] == pytest.approx(2.0, rel=1e-14, abs=0.0)


def test_small_rate_branch_reads_only_the_elements_it_is_for():
    """One small time in a batch used to send the whole batch through
    ``_phi_at_small_rates``, whose ``_psi`` overflows at ``(γ − a) v < -710``.
    Each element keeps the value it has alone, with no warning."""
    gamma, a = 0.5, 1e3
    times = np.array([1e-4, 5.0, 20.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        batch = ou_exponential_phi(gamma, a, times, times)
        alone = [float(ou_exponential_phi(gamma, a, t, t)) for t in times]
    np.testing.assert_array_equal(batch, alone)


@pytest.mark.parametrize("gamma, a, th, tl", [
    # (gamma + a) th = 1, either end of gamma / (gamma + a) and of the lag.
    (1.0, 0.0, 1.0, 1.0),
    (0.5, 0.5, 1.0, 1.0),
    (0.0, 1.0, 1.0, 1.0 - 1e-12),
    (0.25, 0.75, 1.0, 1e-12),
    (0.9, 0.1, 1.0, 0.5),
    # One step under it, where the Taylor sum reads its largest arguments.
    (1.0, 0.0, 1.0 - 1e-12, 1.0 - 1e-12),
])
def test_phi_paths_meet_the_defining_integral_at_the_dispatch(gamma, a, th, tl):
    """Both paths, bypassing the dispatch, against the integral."""
    args = (np.array([gamma]), np.array([a]), np.array([tl]))
    d = np.array([th - tl])
    expected = float(_phi_mpmath(gamma, a, th, tl))
    small = closed_forms._phi_at_small_rates(*args, d)[0]
    closed = closed_forms._phi_closed(*args, np.array([th]), d)[0]
    assert small == pytest.approx(expected, rel=2e-15, abs=0.0)
    assert closed == pytest.approx(expected, rel=3e-15, abs=0.0)


def test_phi_batch_reads_each_element_on_one_path(monkeypatch):
    """Each element of a mixed batch takes one path, and a batch of one kind
    takes no mask."""
    sizes = {}

    def counted(name):
        path = getattr(closed_forms, name)

        def wrapper(gamma, *rest):
            sizes[name] = gamma.size
            return path(gamma, *rest)

        return wrapper

    for name in ("_phi_at_small_rates", "_phi_closed"):
        monkeypatch.setattr(closed_forms, name, counted(name))
    times = np.array([0.1, 3.0, 0.2, 5.0, 9.0])
    ou_exponential_phi(1.0, 2.0, times, times)
    assert sizes == {"_phi_at_small_rates": 2, "_phi_closed": 3}
