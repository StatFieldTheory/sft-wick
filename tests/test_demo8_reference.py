"""Demo 8's reference checked against itself and against closed forms.

``examples/demo8/time8_reference.py`` carries three independent pieces, and
each is checked here before any package number is compared with it:

* the moment hierarchy with a time-dependent generator (``solve_ivp``,
  DOP853, ``rtol = 1e-12``) -- against ``ito_moments.solve``
  (``expm_multiply``) where the coefficients are constant;
* :class:`time8_reference.ExactC`, the free ``C`` by Gauss-Legendre
  quadrature of its definition -- against the hierarchy's two-time order-0
  moment, for every model, which also checks that each Markov embedding
  really has the intended kernel as its stationary covariance;
* :func:`time8_reference.gaussian_kernel_C` (``erf``) -- against
  ``scipy.integrate.dblquad``; and :class:`time8_reference.HandContraction`,
  the order-1 and order-2 diagrams written out from the perturbative
  solution -- against the hierarchy for the exponential kernel, which has an
  embedding.

Imports no sft-wick code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import dblquad

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo8"
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(DEMO.parent / "reference"))
import ito_moments as im  # noqa: E402
import time8_params as pm  # noqa: E402
import time8_reference as ref  # noqa: E402

POS = [pm.POSITIONS["x"], pm.POSITIONS["y"]]


def _rel(got, want):
    return abs(got - want) / abs(want) if want else abs(got)


def _constant_R(gamma):
    def R(a, t, s):
        return np.exp(-gamma[a] * (np.asarray(t) - np.asarray(s)))
    return R


def _model_and_C(name):
    """``(params, model, ExactC, R)`` for one of the four models."""
    if name == "rate":
        p = pm.RateParams()
        return (p, ref.rate_model(p, POS),
                ref.ExactC(p.rate.R, pm.ExponentialKernel(p.lam, p.sigma_t),
                           pm.GaussianEnvelope(p.sigma_x), p.t_min, n=32),
                "phi")
    if name == "oscillator":
        p = pm.OscillatorParams()
        return (p, ref.oscillator_model(p, POS),
                ref.ExactC(lambda a, t, s: p.R.tau(np.asarray(t) - np.asarray(s)),
                           None, None, p.t_min, white=p.white.s,
                           white_envelope=p.white.spatial, n=32),
                "x")
    if name == "white":
        p = pm.WhiteParams()
        return (p, ref.white_model(p, POS),
                ref.ExactC(_constant_R(p.gamma),
                           pm.ExponentialKernel(p.lam, p.sigma_t),
                           pm.GaussianEnvelope(p.sigma_x), p.t_min,
                           white=p.white.s, white_envelope=p.white.spatial, n=32),
                "phi")
    kernel = {"matern": pm.Matern32(), "damped_cosine": pm.DampedCosine(),
              "exponential": pm.ExponentialKernel()}[name]
    p = pm.KernelParams(kernel=kernel)
    return (p, ref.kernel_model(p, POS),
            ref.ExactC(_constant_R(p.gamma), kernel,
                       pm.GaussianEnvelope(p.sigma_x), p.t_min, n=32),
            "phi")


# ---------------------------------------------------------------------------
# The time-dependent solver against expm, where both apply
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("legs,order", [
    ([("phi", 0, 0)], 1),                          # the tadpole
    ([("phi", 0, 0), ("phi", 0, 1)], 0),           # the free correlator
    ([("phi", 0, 0), ("phi", 1, 1)], 2),           # both vertices
])
def test_time_dependent_solver_matches_expm_at_constant_coefficients(legs, order):
    p, model, _C, _f = _model_and_C("exponential")
    gen = model.generator()
    assert all(piece.coefficient is None for piece in gen.pieces)
    mono = model.monomial(legs)
    got = gen.solve([(mono, (order,))], p.t_min, [p.t_final],
                    model.initial)[(mono, (order,))][0]
    merged = im.PolySDE(D=model.D, n_tags=1)
    for piece in gen.pieces:
        merged.drift += piece.sde.drift
        merged.diffusion += piece.sde.diffusion
    want = im.solve(merged, [(mono, (order,))], [p.span],
                    initial=model.initial)[(mono, (order,))][0]
    assert want != 0.0
    assert got == pytest.approx(want, rel=1e-11, abs=0.0)


# ---------------------------------------------------------------------------
# The quadrature C against the hierarchy, for every model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["rate", "oscillator", "white", "matern",
                                  "damped_cosine", "exponential"])
def test_quadrature_C_matches_the_two_time_hierarchy(name):
    p, model, C, field = _model_and_C(name)
    r = abs(POS[0] - POS[1])
    for a in range(pm.N):
        for ta, tb in ((p.t_final, p.t_final),
                       (p.t_final, p.t_final - 0.6),
                       (p.t_min + 0.3, p.t_final)):
            got = C.diagonal(np.array([ta]), np.array([tb]), np.array([r]))[0, a]
            zero = (0,) * model.n_tags
            want = model.two_time([(field, a, 0)], ta, [(field, a, 1)], tb,
                                  [zero], p.t_min)[zero]
            assert want != 0.0
            assert got == pytest.approx(want, rel=1e-10, abs=0.0)


def test_the_kernels_are_not_exponential():
    """The point of item (c): a Matérn-3/2 kernel has zero slope at the
    origin and a damped cosine changes sign, neither of which an OU kernel
    does."""
    k32, dc = pm.Matern32(), pm.DampedCosine()
    assert abs(k32(0.02) - k32(0.0)) / k32(0.0) < 1e-3      # flat at 0
    assert abs(pm.ExponentialKernel()(0.02) / pm.ExponentialKernel()(0.0)
               - 1.0) > 1e-2
    assert min(dc(np.linspace(0.0, 2.5, 101))) < -0.1 * dc(0.0)
    osc = pm.OscillatorR()
    assert min(osc.tau(np.linspace(0.0, 3.1, 101))) < -0.05 * max(
        osc.tau(np.linspace(0.0, 3.1, 101)))


# ---------------------------------------------------------------------------
# The Gaussian-kernel closed form, and the hand contraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gamma", [0.8, 1.3])
@pytest.mark.parametrize("t1,t2", [(2.0, 2.0), (2.0, 1.3), (0.4, 1.7),
                                   (3.0, 0.2)])
def test_gaussian_kernel_C_matches_dblquad(gamma, t1, t2):
    k = pm.GaussianKernel()
    want = dblquad(
        lambda s2, s1: (np.exp(-gamma * (t1 - s1)) * k(s1 - s2)
                        * np.exp(-gamma * (t2 - s2))),
        0.0, t1, 0.0, t2, epsabs=1e-15, epsrel=1e-13)[0]
    got = float(ref.gaussian_kernel_C(gamma, k.lam, k.sigma, t1, t2))
    assert got == pytest.approx(want, rel=1e-11, abs=0.0)


@pytest.mark.parametrize("comps", [(0,), (1,), (0, 1), (1, 1)])
def test_hand_contraction_matches_the_hierarchy_for_an_embedded_kernel(comps):
    """The order-1 and order-2 formulas are checked where a hierarchy exists
    (the exponential kernel); item (c) then applies them to the Gaussian
    kernel, which has no embedding."""
    p, model, C, _f = _model_and_C("exponential")

    def Cvec(b, t1, t2, r):
        t1, t2, r = np.broadcast_arrays(np.asarray(t1, float),
                                        np.asarray(t2, float),
                                        np.asarray(r, float))
        return C.diagonal(t1.ravel(), t2.ravel(), r.ravel())[:, b].reshape(t1.shape)

    hand = ref.HandContraction(_constant_R(p.gamma), Cvec, p.t_min, n=24)
    if len(comps) == 1:
        one = ref.kernel_model(p, [POS[0]])
        want = one.moments([("phi", comps[0], 0)], [(1,)], p.t_min,
                           p.t_final)[(1,)]
        got = hand.tadpole(comps[0], p.t_final)
    else:
        want = model.moments([("phi", comps[0], 0), ("phi", comps[1], 1)],
                             [(2,)], p.t_min, p.t_final)[(2,)]
        got = hand.two_point_order2(comps[0], comps[1], p.t_final,
                                    abs(POS[0] - POS[1]))
    assert want != 0.0
    assert got == pytest.approx(want, rel=1e-10, abs=0.0)
