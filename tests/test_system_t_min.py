"""``System.t_min`` reaches the integrators.

Up to 0.4.2 ``Expansion.evaluate`` (and so ``sweep`` and the CLI) called
``integrate_diagrams`` without ``t_min``, so every integrator started vertex
times at 0 while the propagators were built from ``t_min``.  With
``t_min = 0.75`` a non-local vertex picked up noise from ``[0, t_min)``, and
tadpoles read ``C(τ, τ)`` below ``t_min`` from an extrapolated spline.

Reference: time-translation invariance.  The noise is stationary and the
drift constant, so the system started at ``t_min = s`` and observed at
``t_final = T + s`` is the system started at 0 and observed at ``T``.  The
built-in closed-form C measures times from ``t_min``, so the shifted and
unshifted integrands agree node for node.
"""
from __future__ import annotations

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.evaluate import integrate_diagrams
from sft_wick.workflow.specs import ConstantImpulse

N = 2
SHIFT = 0.75
T_FINAL = 2.0
POS = {"x": 0.0, "y": 0.4, "z": -0.3}

F = np.array([[[0.2, -0.7], [0.4, 0.1]],
              [[0.9, 0.3], [-0.5, 0.6]]])       # no index symmetry
K3 = np.random.default_rng(11).normal(size=(N, N, N))


def _system(t_min, *, local=True):
    return sw.System(
        field=sw.FieldSpec("phi", N),
        linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        vertices=[sw.LocalVertex("F", coupling=F)] if local else [],
        nonlocal_vertices=[] if local else [
            sw.NonLocalVertex("K", order=3, coupling=K3)],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.6, sigma_t=0.5),
            spatial=sw.ExponentialSpatial(sigma_x=1.0))),
        t_min=t_min)


def _value(t_min, obs, orders, comps, *, local=True, method="gauss_legendre"):
    system = _system(t_min, local=local)
    props = system.propagators(t_max=T_FINAL + t_min + 0.5, n_grid_t=12,
                               c_closed_form="auto", c_closed_form_only=True,
                               progress=False)
    exp = system.expand(obs, orders=orders)
    return exp.evaluate(props, positions=POS, t_final=T_FINAL + t_min,
                        component_pair=comps, orders=orders, method=method,
                        n_gauss=10, n_samples=2**10, seed=3).total


@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized"])
@pytest.mark.parametrize("comps", [(0, 1), (1, 1)])
def test_local_vertex_order_2_is_time_translation_invariant(method, comps):
    obs = ("phi_a(x)", "phi_b(y)")
    base = _value(0.0, obs, [2], comps, method=method)
    shifted = _value(SHIFT, obs, [2], comps, method=method)
    assert base != 0.0
    assert shifted == pytest.approx(base, rel=1e-10, abs=0.0)


@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized"])
@pytest.mark.parametrize("comps", [(0, 1, 1), (1, 0, 1)])
def test_nonlocal_vertex_order_1_is_time_translation_invariant(method, comps):
    obs = ("phi_a(x)", "phi_b(y)", "phi_c(z)")
    base = _value(0.0, obs, [1], comps, local=False, method=method)
    shifted = _value(SHIFT, obs, [1], comps, local=False, method=method)
    assert base != 0.0
    assert shifted == pytest.approx(base, rel=1e-10, abs=0.0)


def test_sweep_uses_t_min():
    system = _system(SHIFT)
    props = system.propagators(t_max=T_FINAL + SHIFT + 0.5, n_grid_t=12,
                               c_closed_form="auto", c_closed_form_only=True,
                               progress=False)
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2])
    sweep = exp.sweep(props, positions_grid={"x": [0.0], "y": [0.4]},
                      t_final_grid=[T_FINAL + SHIFT],
                      component_pairs=[(0, 1)], orders=[2],
                      method="gauss_legendre", n_gauss=10)
    base = _value(0.0, ("phi_a(x)", "phi_b(y)"), [2], (0, 1))
    total = float(sweep.totals()["value"].sum())
    assert total == pytest.approx(base, rel=1e-10, abs=0.0)


def test_a_cache_built_for_another_t_min_is_refused():
    exp = _system(SHIFT).expand(("phi_a(x)", "phi_b(y)"), orders=[0])
    other = _system(0.0).propagators(t_max=3.5, n_grid_t=12,
                                     c_closed_form="auto",
                                     c_closed_form_only=True, progress=False)
    with pytest.raises(ValueError, match="t_min"):
        exp.evaluate(other, positions=POS, t_final=T_FINAL + SHIFT,
                     component_pair=(0, 0), orders=[0])


@pytest.mark.parametrize("t_min", [-5.0, -1.0, 0.25])
def test_l0_integrals_start_at_the_t_min_of_the_model(t_min):
    """L0 defaulted to 0 whatever the model said: 2.8 % low at t_min = -1,
    and a refused table read at t_min = 0.25.

    Order 1 of ``dφ = (-γ φ + g φ + √(2D) ξ) dt`` from ``φ(t_min) = 0``:
    ``2 g ∫_{t_min}^{T} e^{-γ(T-s)} c(s, T) ds``, with the Ornstein-Uhlenbeck
    ``c`` written out and the integral done by scipy alone.
    """
    from scipy.integrate import quad

    gamma, g, diff, t_f = 1.1, 0.4, 0.7, 2.0
    system = sw.System(
        field=sw.FieldSpec("phi", 1),
        linear=sw.DiagonalA(gamma=[gamma]),
        vertices=[sw.LocalVertex("L", coupling=np.array([[g]]))],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.0, sigma_t=1.0),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            ),
            sigma2=ConstantImpulse(np.array([[2.0 * diff]])),
        ),
        t_min=t_min,
    )
    props = system.propagators(t_max=t_f, n_grid_t=64, progress=False)
    terms = system.expand(("phi(x)", "phi(y)"), orders=[1]).diagrams(1)

    def ou(s, t):
        return (
            diff
            / gamma
            * np.exp(-gamma * (t - s))
            * -np.expm1(-2.0 * gamma * (s - t_min))
        )

    ref, _ = quad(
        lambda s: 2.0 * g * np.exp(-gamma * (t_f - s)) * ou(s, t_f),
        t_min,
        t_f,
        epsabs=1e-12,
    )
    got, _ = integrate_diagrams(
        terms,
        system.build_coupling_values(),
        t_f,
        props.cache,
        method="gauss_legendre",
        n_gauss=48,
    )
    assert got == pytest.approx(ref, rel=2e-5, abs=0.0)


def test_an_explicit_t_min_other_than_the_models_is_refused():
    system = _system(0.25)
    props = system.propagators(t_max=2.0, n_grid_t=12, progress=False)
    terms = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0]).diagrams(0)
    with pytest.raises(ValueError, match="t_min"):
        integrate_diagrams(
            terms,
            system.build_coupling_values(),
            1.0,
            props.cache,
            t_min=0.0,
            fixed_indices={"a": 0, "b": 0},
        )
