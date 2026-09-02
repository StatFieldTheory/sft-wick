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

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.evaluate import PropagatorCache, select_gl_node_count
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


def test_gl_and_dblquad_agree_where_no_closed_form_exists():
    """The two quadratures must agree on a kernel the closed form does
    not cover (Gaussian temporal), at the same extreme cells."""
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
    assert ou_exponential_phi(g, a, 2.0, 5.0) == pytest.approx(
        ou_exponential_phi(g, a, 5.0, 2.0), rel=1e-14)
    # Stationary variance λ Φ(∞, ∞) = λ / (γ (γ + a)).
    assert ou_exponential_phi(g, a, 400.0, 400.0) == pytest.approx(
        1.0 / (g * (g + a)), rel=1e-13)
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
    assert v0 == pytest.approx(v1, rel=1e-5)


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

    gaussian = sw.System(
        field=base.field, linear=base.linear, vertices=base.vertices,
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.GaussianTemporal(lam=0.05, sigma_t=0.3),
            spatial=sw.ExponentialSpatial(sigma_x=1.0))),
    )
    assert builtin_closed_form_for(gaussian) is None

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


# --------------------------------------------------------------------- #
# The dispatcher itself                                                  #
# --------------------------------------------------------------------- #


def test_dispatcher_defaults_to_the_builtin_closed_form_and_agrees_with_quadrature():
    """``System.propagators()`` with no closed-form hook must pick the built-in
    closed form, and the table it builds must agree with the table that
    ``c_closed_form=None`` (forced quadrature) builds."""
    system = SYSTEMS["per_component_gamma"]
    auto = system.propagators(t_max=T_MAX, n_grid_t=30)
    assert auto.c_source.startswith("closed_form:builtin"), auto.c_source

    quad = system.propagators(t_max=T_MAX, n_grid_t=30, c_closed_form=None)
    assert quad.c_source.startswith("quadrature:gauss_legendre"), quad.c_source

    t1 = np.array([0.5, 3.0, T_MAX, 7.0, T_MAX])
    t2 = np.array([0.5, 3.0, T_MAX, 2.0, 0.5])
    for r in RS:
        a = auto.cache.C_at_batch(t1, t2, np.zeros(5), np.full(5, r))
        q = quad.cache.C_at_batch(t1, t2, np.zeros(5), np.full(5, r))
        scale = float(np.max(np.abs(a)))
        assert np.max(np.abs(a - q) / np.maximum(np.abs(a), FLOOR_FRAC * scale)) < 1e-9


def test_dispatcher_falls_back_to_dblquad_for_custom_kernels():
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=1),
        linear=sw.DiagonalA(gamma=[1.0]),
        noise=sw.GaussianNoise(kappa2=sw.GeneralKappa2(
            fn=lambda n1, t1, n2, t2: 0.05 * np.exp(-abs(t1 - t2)) * np.eye(1))),
    )
    props = system.propagators(t_max=2.0, n_grid_t=8)
    assert props.c_source == "quadrature:dblquad", props.c_source


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
    props.cache.C_at_batch(np.array([1.0]), np.array([1.0]), np.zeros(1), np.zeros(1))
    assert calls
