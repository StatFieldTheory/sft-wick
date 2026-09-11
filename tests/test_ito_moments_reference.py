"""The Itô moment-hierarchy reference (``examples/reference/ito_moments.py``)
against results known in closed form.

The reference is what demos 4 and 5 are checked against, so it is pinned
here on its own, with no sft-wick code involved: an OU variance, the first
correction from a quadratic drift, Campbell's theorem for shot noise, the
Itô (not Stratonovich) moments of multiplicative noise, and a non-normal
two-dimensional linear system against its Lyapunov equation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "examples" / "reference"))
import ito_moments as im  # noqa: E402

GAMMA, SIGMA, T = 0.8, 0.7, 1.9


def _ou(D=1, n_tags=1):
    sde = im.PolySDE(D=D, n_tags=n_tags)
    sde.add_linear_drift(-GAMMA * np.eye(D))
    return sde


def test_ou_variance():
    sde = _ou()
    sde.add_constant_diffusion(np.array([[SIGMA ** 2]]))
    got = im.solve(sde, [((2,), (0,))], [T])[((2,), (0,))][0]
    exact = SIGMA ** 2 * (1 - np.exp(-2 * GAMMA * T)) / (2 * GAMMA)
    assert got == pytest.approx(exact, rel=1e-13, abs=0.0)


def test_quadratic_drift_first_order_mean():
    """dX = (−γX + εX²)dt + σdW: E[X] = ε σ²(1 − e^{−γT})²/(2γ²) + O(ε²)."""
    sde = _ou()
    sde.add_constant_diffusion(np.array([[SIGMA ** 2]]))
    sde.add_quadratic_drift(np.ones((1, 1, 1)), tag=(1,))
    got = im.solve(sde, [((1,), (1,)), ((1,), (0,))], [T])
    exact = SIGMA ** 2 * (1 - np.exp(-GAMMA * T)) ** 2 / (2 * GAMMA ** 2)
    assert got[((1,), (1,))][0] == pytest.approx(exact, rel=1e-13, abs=0.0)
    assert got[((1,), (0,))][0] == 0.0


NU, H = 1.3, 0.6


def _shot():
    sde = _ou(n_tags=1)
    sde.jump_moment = lambda g: NU * H ** sum(g)
    sde.max_jump_order = 4
    sde.jump_tag = lambda m: (max(m - 2, 0),)
    return sde


def _campbell(m):
    return NU * H ** m * (1 - np.exp(-m * GAMMA * T)) / (m * GAMMA)


def test_shot_noise_third_moment_is_campbells_cumulant():
    got = im.solve(_shot(), [((3,), (1,)), ((3,), (0,))], [T])
    assert got[((3,), (1,))][0] == pytest.approx(_campbell(3), rel=1e-13,
                                                 abs=0.0)
    assert got[((3,), (0,))][0] == 0.0


def test_shot_noise_fourth_moment_splits_by_cumulant_order():
    got = im.solve(_shot(), [((4,), (2,)), ((4,), (0,))], [T])
    assert got[((4,), (2,))][0] == pytest.approx(_campbell(4), rel=1e-13,
                                                 abs=0.0)
    assert got[((4,), (0,))][0] == pytest.approx(3 * _campbell(2) ** 2,
                                                 rel=1e-13, abs=0.0)


def test_multiplicative_noise_follows_ito():
    """dX = −γX dt + (σ0 + σ1 X) dW.  Under Itô ⟨X⟩ = 0 and
    ``⟨X²⟩ = σ0² (1 − e^{−aT})/a`` with ``a = 2γ − σ1²``; the σ1² coefficient
    is ``σ0² [(1 − e^{−2γT}) − 2γT e^{−2γT}] / (2γ)²``.  (Stratonovich
    would add the drift ½σ1(σ0 + σ1X) and a non-zero mean.)"""
    s0 = SIGMA
    sde = _ou(n_tags=1)
    sde.diffusion += [(0, 0, s0 ** 2, (0,), (0,)),
                      (0, 0, 2 * s0, (1,), (1,)),     # × σ1
                      (0, 0, 1.0, (2,), (2,))]        # × σ1²
    got = im.solve(sde, [((2,), (2,)), ((1,), (1,)), ((1,), (2,))], [T])
    a = 2 * GAMMA
    exact = s0 ** 2 * ((1 - np.exp(-a * T)) - a * T * np.exp(-a * T)) / a ** 2
    assert got[((2,), (2,))][0] == pytest.approx(exact, rel=1e-13, abs=0.0)
    assert got[((1,), (1,))][0] == 0.0
    assert got[((1,), (2,))][0] == 0.0


def test_non_normal_linear_system_matches_lyapunov():
    A = np.array([[-1.0, 2.5], [0.0, -1.7]])
    S = np.array([[0.5, 0.2], [0.2, 0.9]])
    sde = im.PolySDE(D=2, n_tags=1)
    sde.add_linear_drift(A)
    sde.add_constant_diffusion(S)
    targets = [((2, 0), (0,)), ((1, 1), (0,)), ((0, 2), (0,))]
    got = im.solve(sde, targets, [T])
    P = solve_ivp(lambda _t, y: (A @ y.reshape(2, 2) + y.reshape(2, 2) @ A.T
                                 + S).ravel(),
                  (0, T), np.zeros(4), rtol=1e-12, atol=1e-14).y[:, -1]
    P = P.reshape(2, 2)
    assert got[targets[0]][0] == pytest.approx(P[0, 0], rel=1e-10, abs=0.0)
    assert got[targets[1]][0] == pytest.approx(P[0, 1], rel=1e-10, abs=0.0)
    assert got[targets[2]][0] == pytest.approx(P[1, 1], rel=1e-10, abs=0.0)


def test_moments_from_cumulants_is_wicks_theorem_for_gaussians():
    cov = np.array([[1.0, 0.3, 0.2, 0.1],
                    [0.3, 2.0, 0.4, 0.5],
                    [0.2, 0.4, 1.5, 0.6],
                    [0.1, 0.5, 0.6, 1.2]])

    def cumulant(idx):
        return cov[idx[0], idx[1]] if len(idx) == 2 else 0.0

    got = im.moments_from_cumulants(cumulant, [0, 1, 2, 3])
    exact = (cov[0, 1] * cov[2, 3] + cov[0, 2] * cov[1, 3]
             + cov[0, 3] * cov[1, 2])
    assert got == pytest.approx(exact, rel=1e-15, abs=0.0)
