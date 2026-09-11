"""L1 lowering: structure the default flags assume, now checked.

Three L1 inputs returned wrong numbers without an error (found 2026-09-11):

1. ``DiagonalA(gamma=callable)`` chose between a scalar and a matrix R by
   evaluating ``gamma`` at t = 0 and t = 1 only.  A rate whose components
   agree at those two times was lowered to a scalar R that gives every
   component the rate of component 0.
2. Its cumulative-rate spline started at t = 0 whatever ``System.t_min``
   was, so every R with a time below 0 came from extrapolation.
3. ``System.expand`` and ``System.propagators`` default to ``diag_R=True``
   and ``diag_C=True``, which keep only the diagonal of R and C.  A dense R
   (``ExplicitR(iso_R=False)``), a component-mixing κ² (``GeneralKappa2``)
   or a component-mixing white noise (a ``ConstantImpulse`` or
   ``CustomImpulse`` matrix) has off-diagonal entries, and they were
   dropped.

References: ``scipy.integrate.quad`` of the rate for (1)-(2); for (3) the
Lyapunov equation of the Markov embedding of the exponential noise, which
shares no code with sft-wick.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm

import sft_wick as sw
from sft_wick.workflow.specs import (ConstantImpulse, CustomImpulse,
                                     GeneralKappa2)

N = 2


# ---------------------------------------------------------------------------
# 1-2. Callable decay rate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GammaAgreesAtZeroAndOne:
    """Components equal at t = 0 and t = 1, different in between."""

    def __call__(self, t):
        return np.array([1.0, 1.0 + 0.8 * np.sin(np.pi * t)])


@dataclass(frozen=True)
class GammaEqual:
    def __call__(self, t):
        v = 1.0 + 0.3 * np.sin(t)
        return np.array([v, v])


@dataclass(frozen=True)
class GammaScalar:
    def __call__(self, t):
        return np.array([1.0 + 0.8 * np.sin(2.0 * t)])


def _exact_R(gamma, a, t1, t2):
    return np.exp(-quad(lambda s: gamma(s)[a], t2, t1,
                        epsabs=1e-13, epsrel=1e-12)[0])


def _noise():
    return sw.GaussianNoise(kappa2=sw.SeparableTranslation(
        temporal=sw.ExponentialTemporal(lam=1.0, sigma_t=0.5),
        spatial=sw.ExponentialSpatial(sigma_x=1.0)))


def test_callable_gamma_iso_detection_sees_the_whole_grid():
    g = GammaAgreesAtZeroAndOne()
    lin = sw.DiagonalA(gamma=g, t_max_cache=5.0, n_grid_cache=2000)
    assert lin.is_iso_R is False
    R = lin.build_R_callable()
    for t1, t2 in [(1.5, 0.5), (2.5, 0.2), (4.0, 1.3)]:
        got = np.diag(np.asarray(R(t1, t2)))
        exact = [_exact_R(g, a, t1, t2) for a in range(N)]
        assert got == pytest.approx(exact, rel=1e-5, abs=0.0)
    system = sw.System(field=sw.FieldSpec("phi", N), linear=lin, noise=_noise())
    assert system.iso_R is False


def test_callable_gamma_with_equal_components_stays_scalar():
    g = GammaEqual()
    lin = sw.DiagonalA(gamma=g, t_max_cache=5.0, n_grid_cache=2000)
    assert lin.is_iso_R is True
    R = lin.build_R_callable()
    assert float(R(2.5, 0.2)) == pytest.approx(_exact_R(g, 0, 2.5, 0.2),
                                               rel=1e-5, abs=0.0)


def test_callable_gamma_spline_reaches_a_negative_t_min():
    g = GammaScalar()
    system = sw.System(
        field=sw.FieldSpec("phi", 1),
        linear=sw.DiagonalA(gamma=g, t_max_cache=5.0, n_grid_cache=4000),
        noise=_noise(), t_min=-3.0)
    R = system.build_propagator_model().R_time
    for t1, t2 in [(1.0, 0.2), (1.0, -0.5), (0.5, -2.0), (-0.5, -3.0)]:
        assert float(R(t1, t2)) == pytest.approx(_exact_R(g, 0, t1, t2),
                                                 rel=1e-5, abs=0.0)


def test_diagonal_a_t_min_cache_sets_the_spline_start():
    g = GammaScalar()
    lin = sw.DiagonalA(gamma=g, t_max_cache=5.0, n_grid_cache=4000,
                       t_min_cache=-3.0)
    R = lin.build_R_callable()
    assert float(R(0.5, -2.0)) == pytest.approx(_exact_R(g, 0, 0.5, -2.0),
                                                rel=1e-5, abs=0.0)


def test_propagators_beyond_t_max_cache_are_refused():
    system = sw.System(
        field=sw.FieldSpec("phi", 1),
        linear=sw.DiagonalA(gamma=GammaScalar(), t_max_cache=2.0),
        noise=_noise())
    with pytest.raises(ValueError, match="t_max_cache"):
        system.propagators(t_max=3.0, progress=False)


# ---------------------------------------------------------------------------
# 3. Off-diagonal R and C under the default flags
# ---------------------------------------------------------------------------

A_DENSE = ((-1.0, 2.5), (0.0, -1.7))          # non-normal drift
LAM, TAU = 0.7, 0.4
T_OBS = 1.5


@dataclass(frozen=True)
class DenseR:
    A: tuple = A_DENSE

    def __call__(self, t, s):
        if t < s:
            return np.zeros((N, N))
        return expm(np.asarray(self.A) * (t - s))


def _lyapunov_blocks():
    """``z = (φ, η)`` with ``dη = −η/τ dt + √(2λ/τ) dW`` and φ(0) = 0."""
    Z, I = np.zeros((N, N)), np.eye(N)
    M = np.block([[np.asarray(A_DENSE), I], [Z, -I / TAU]])
    BB = np.block([[Z, Z], [Z, (2 * LAM / TAU) * I]])
    P0 = np.block([[Z, Z], [Z, LAM * I]])
    return M, BB, P0


def _P(t):
    M, BB, P0 = _lyapunov_blocks()
    sol = solve_ivp(lambda _s, y: (M @ y.reshape(4, 4)
                                   + y.reshape(4, 4) @ M.T + BB).ravel(),
                    (0.0, t), P0.ravel(), rtol=1e-12, atol=1e-14)
    return sol.y[:, -1].reshape(4, 4)


@dataclass(frozen=True)
class DenseC:
    """Exact C of the dense system (Markov embedding), ``(N, N)``."""

    def __call__(self, n1, t1, n2, t2):
        M, _, _ = _lyapunov_blocks()
        if t1 >= t2:
            return (expm(M * (t1 - t2)) @ _P(t2))[:N, :N]
        return (_P(t1) @ expm(M * (t2 - t1)).T)[:N, :N]


def _dense_system(vertices=()):
    return sw.System(
        field=sw.FieldSpec("phi", N),
        linear=sw.ExplicitR(R_time=DenseR(), iso_R=False),
        vertices=list(vertices), noise=_noise_for_dense())


def _noise_for_dense():
    return sw.GaussianNoise(kappa2=sw.SeparableTranslation(
        temporal=sw.ExponentialTemporal(lam=LAM, sigma_t=TAU),
        spatial=sw.ExponentialSpatial(sigma_x=1.0)))


OBS = ("phi_a(x)", "phi_b(y)")


def test_dense_R_refuses_diag_R():
    with pytest.raises(ValueError, match="diag_R=False"):
        _dense_system().expand(OBS, orders=[0])


def test_dense_R_refuses_diag_C_in_expand():
    with pytest.raises(ValueError, match="diag_C=False"):
        _dense_system().expand(OBS, orders=[0], diag_R=False)


def test_dense_R_refuses_diagonal_propagator_tables():
    with pytest.raises(ValueError, match="diag_C=False"):
        _dense_system().propagators(t_max=1.6, n_grid_t=11, progress=False)


def test_dense_R_with_full_flags_matches_the_lyapunov_reference():
    """The route the refusals point to gives the right numbers."""
    FB = np.array([[0.3, -0.6], [0.9, -0.4]])
    system = _dense_system([sw.LocalVertex("F", coupling=FB)])
    exp = system.expand(OBS, orders=[0], diag_R=False, diag_C=False)
    props = system.propagators(t_max=1.6, c_closed_form=DenseC(),
                               c_closed_form_only=True, diag_C=False,
                               progress=False)
    P = _P(T_OBS)
    for ab in [(0, 1), (1, 1)]:
        r = exp.evaluate(props, positions={"x": 0.0, "y": 0.0},
                         t_final=T_OBS, component_pair=ab, orders=[0],
                         method="qmc", n_samples=16)
        assert r.total == pytest.approx(P[ab], rel=1e-10, abs=0.0)


@dataclass(frozen=True)
class MixingKappa2:
    def __call__(self, n1, t1, n2, t2):
        return np.exp(-abs(t1 - t2)) * np.array([[1.0, 0.3], [0.3, 1.0]])


@dataclass(frozen=True)
class DiagonalKappa2:
    def __call__(self, n1, t1, n2, t2):
        return np.exp(-abs(t1 - t2)) * np.diag([1.0, 0.6])


@dataclass(frozen=True)
class MixingWhite:
    def __call__(self, n1, t, n2):
        return np.array([[1.0, 0.6], [0.6, 0.5]])


@dataclass(frozen=True)
class DiagonalWhite:
    def __call__(self, n1, t, n2):
        return np.diag([1.0, 0.5])


def _system(kappa2=None, sigma2=None, gamma=(1.0, 1.0)):
    kappa2 = kappa2 if kappa2 is not None else sw.SeparableTranslation(
        temporal=sw.ExponentialTemporal(lam=0.5, sigma_t=0.5),
        spatial=sw.ExponentialSpatial(sigma_x=1.0))
    return sw.System(field=sw.FieldSpec("phi", N),
                     linear=sw.DiagonalA(gamma=list(gamma)),
                     noise=sw.GaussianNoise(kappa2=kappa2, sigma2=sigma2))


MIXING = {
    "GeneralKappa2": dict(kappa2=GeneralKappa2(fn=MixingKappa2())),
    "ConstantImpulse matrix": dict(
        sigma2=ConstantImpulse(np.array([[1.0, 0.6], [0.6, 0.5]]))),
    "CustomImpulse": dict(sigma2=CustomImpulse(fn=MixingWhite())),
}

DIAGONAL = {
    "separable": dict(),
    "distinct decay rates": dict(gamma=(1.0, 1.7)),
    "GeneralKappa2 diagonal": dict(kappa2=GeneralKappa2(fn=DiagonalKappa2())),
    "ConstantImpulse scalar": dict(sigma2=ConstantImpulse(0.8)),
    "ConstantImpulse diagonal": dict(sigma2=ConstantImpulse(np.diag([1.0, 0.5]))),
    "CustomImpulse diagonal": dict(sigma2=CustomImpulse(fn=DiagonalWhite())),
}


@pytest.mark.parametrize("case", sorted(MIXING))
def test_component_mixing_noise_refuses_diag_C(case):
    system = _system(**MIXING[case])
    with pytest.raises(ValueError, match="diag_C=False"):
        system.expand(OBS, orders=[0])
    with pytest.raises(ValueError, match="diag_C=False"):
        system.propagators(t_max=1.0, n_grid_t=5, progress=False)
    system.expand(OBS, orders=[0], diag_C=False)   # the stated way out


@pytest.mark.parametrize("case", sorted(DIAGONAL))
def test_diagonal_structure_is_not_refused(case):
    system = _system(**DIAGONAL[case])
    system.expand(OBS, orders=[0])
    system.propagators(t_max=1.0, n_grid_t=5, progress=False)


def test_a_probe_that_cannot_evaluate_is_not_a_refusal():
    """A kernel that rejects the probe's scalar positions says nothing
    about its structure; the guard stays out of the way."""

    @dataclass(frozen=True)
    class VectorOnly:
        def __call__(self, n1, t1, n2, t2):
            if np.ndim(n1) != 1:
                raise TypeError("vector positions only")
            return np.exp(-abs(t1 - t2)) * np.eye(N)

    _system(kappa2=GeneralKappa2(fn=VectorOnly())).expand(OBS, orders=[0])
