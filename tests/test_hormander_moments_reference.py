"""The vector-field (Hörmander-form) moment reference
(``examples/reference/hormander_moments.py``) against results known in
closed form.

That module is what the multiplicative-noise route of sft-wick is checked
against, so it is pinned here on its own, with no sft-wick code involved:
the Stratonovich and Itô moments of affine noise in one and two dimensions
(from the linear moment equations of the Itô form, integrated with
``solve_ivp``), the Itô branch against ``ito_moments.PolySDE``, the tag
expansion against the untagged solution, and two-time moments against the
Markov property.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "examples" / "reference"))
import hormander_moments as hm  # noqa: E402
import ito_moments as im  # noqa: E402

GAMMA, S0, S1, T = 0.8, 0.7, 0.45, 1.9


def _scalar(interpretation, n_tags=0, g1_tag=()):
    sde = hm.VectorFieldSDE(D=1, n_tags=n_tags, interpretation=interpretation)
    sde.add_linear_drift(np.array([[-GAMMA]]))
    sde.columns.append(hm.affine_column(1, n_tags, [0], np.array([S0]),
                                        np.array([[S1]]), g1_tag))
    return sde


def _scalar_moments_ode(b, c, t):
    """``dX = (b + cX) dt + (S0 + S1 X) dW`` (Itô): ``(E X, E X²)``."""
    def rhs(_t, y):
        m, s = y
        return [b + c * m,
                2 * (b * m + c * s) + S0 ** 2 + 2 * S0 * S1 * m + S1 ** 2 * s]
    return solve_ivp(rhs, (0, t), [0.0, 0.0], rtol=1e-12, atol=1e-14).y[:, -1]


@pytest.mark.parametrize("interpretation", ["stratonovich", "ito"])
def test_scalar_affine_noise(interpretation):
    """Stratonovich: the Itô form has the drift ``½ S1 (S0 + S1 X)``."""
    got = hm.solve(_scalar(interpretation), [((1,), ()), ((2,), ())], [T])
    if interpretation == "stratonovich":
        b, c = 0.5 * S1 * S0, 0.5 * S1 ** 2 - GAMMA
    else:
        b, c = 0.0, -GAMMA
    m, s = _scalar_moments_ode(b, c, T)
    if b == 0.0:
        assert got[((1,), ())][0] == 0.0
    else:
        assert got[((1,), ())][0] == pytest.approx(m, rel=1e-10, abs=0.0)
    assert got[((2,), ())][0] == pytest.approx(s, rel=1e-10, abs=0.0)


def test_ito_branch_equals_poly_sde():
    """The Itô generator built from g equals PolySDE with S(X) = g(X)²."""
    tags = [(k,) for k in range(4)]
    hv = hm.solve(_scalar("ito", 1, (1,)),
                  [((2,), t) for t in tags] + [((3,), t) for t in tags], [T])
    sde = im.PolySDE(D=1, n_tags=1)
    sde.add_linear_drift(np.array([[-GAMMA]]))
    sde.diffusion += [(0, 0, S0 ** 2, (0,), (0,)),
                      (0, 0, 2 * S0 * S1, (1,), (1,)),
                      (0, 0, S1 ** 2, (2,), (2,))]
    ref = im.solve(sde, [((2,), t) for t in tags] + [((3,), t) for t in tags],
                   [T])
    for key, val in ref.items():
        assert hv[key][0] == pytest.approx(val[0], rel=1e-13, abs=1e-16), key


def test_tag_expansion_sums_to_the_untagged_moment():
    """Σ_K S1^K m^(K), with S1 moved into the tag, is the untagged moment."""
    sde = hm.VectorFieldSDE(D=1, n_tags=1, interpretation="stratonovich")
    sde.add_linear_drift(np.array([[-GAMMA]]))
    sde.columns.append(hm.affine_column(1, 1, [0], np.array([S0]),
                                        np.array([[1.0]]), (1,)))
    tags = [(k,) for k in range(30)]
    got = hm.solve(sde, [((2,), t) for t in tags], [T])
    series = sum(S1 ** k * got[((2,), (k,))][0] for k in range(30))
    whole = hm.solve(_scalar("stratonovich"), [((2,), ())], [T])[((2,), ())][0]
    # 14 terms leave 1.4e-6, 20 terms 5.5e-10, 30 terms 1.2e-16.
    assert series == pytest.approx(whole, rel=1e-13, abs=0.0)


def test_two_dimensional_stratonovich_matches_the_moment_equations():
    """N = 2, three noise columns, a non-normal drift, no symmetry.

    The Itô form has the drift ``A X + b + L X`` with
    ``b_i = ½ Σ_jk g1_ikj g0_jk`` and ``L_ic = ½ Σ_jk g1_ikj g1_jkc``, and
    the diffusion ``(g0 + g1 X)(g0 + g1 X)ᵀ``; its first and second moments
    obey closed linear equations."""
    A = np.array([[-1.0, 0.6], [-0.3, -1.4]])
    g0 = np.array([[0.5, -0.2, 0.1], [0.15, 0.4, -0.3]])
    g1 = np.array([[[0.2, -0.1], [0.05, 0.25], [-0.15, 0.1]],
                   [[-0.1, 0.3], [0.2, -0.05], [0.1, 0.15]]])
    sde = hm.VectorFieldSDE(D=2, n_tags=0, interpretation="stratonovich")
    sde.add_linear_drift(A)
    for k in range(3):
        sde.columns.append(hm.affine_column(2, 0, [0, 1], g0[:, k],
                                            g1[:, k, :], ()))
    b = 0.5 * np.einsum("ikj,jk->i", g1, g0)
    L = 0.5 * np.einsum("ikj,jkc->ic", g1, g1)
    Aeff = A + L

    def rhs(_t, y):
        m, P = y[:2], y[2:].reshape(2, 2)
        dm = Aeff @ m + b
        noise = (g0 @ g0.T + np.einsum("ak,bke,e->ab", g0, g1, m)
                 + np.einsum("akc,c,bk->ab", g1, m, g0)
                 + np.einsum("akc,bke,ce->ab", g1, g1, P))
        dP = (Aeff @ P + P @ Aeff.T + np.outer(b, m) + np.outer(m, b)
              + noise)
        return np.concatenate([dm, dP.ravel()])

    y = solve_ivp(rhs, (0, T), np.zeros(6), rtol=1e-12, atol=1e-14).y[:, -1]
    targets = [((1, 0), ()), ((0, 1), ()), ((2, 0), ()), ((1, 1), ()),
               ((0, 2), ())]
    got = hm.solve(sde, targets, [T])
    want = [y[0], y[1], y[2], y[3], y[5]]
    for key, w in zip(targets, want):
        assert got[key][0] == pytest.approx(w, rel=1e-9, abs=0.0), key
    assert abs(y[3] - y[4]) < 1e-12          # P is symmetric


def test_two_time_moment_ou():
    sde = hm.VectorFieldSDE(D=1, n_tags=0)
    sde.add_linear_drift(np.array([[-GAMMA]]))
    sde.columns.append(hm.constant_column(1, 0, {0: S0}))
    t1, t2 = 0.7, 1.9
    got = hm.two_time(sde, (1,), (1,), [()], t1, t2)[()]
    var = S0 ** 2 * (1 - np.exp(-2 * GAMMA * t1)) / (2 * GAMMA)
    assert got == pytest.approx(np.exp(-GAMMA * (t2 - t1)) * var,
                                rel=1e-12, abs=0.0)


def test_two_time_moment_affine_stratonovich():
    """``E[X(t1) X(t2)] = e^{cτ} E[X²(t1)] + (b/c)(e^{cτ} − 1) E[X(t1)]``
    for the Itô form ``dX = (b + cX) dt + …``."""
    sde = _scalar("stratonovich")
    t1, t2 = 0.6, 1.7
    b, c = 0.5 * S1 * S0, 0.5 * S1 ** 2 - GAMMA
    m, s = _scalar_moments_ode(b, c, t1)
    tau = t2 - t1
    want = np.exp(c * tau) * s + b / c * (np.exp(c * tau) - 1) * m
    got = hm.two_time(sde, (1,), (1,), [()], t1, t2)[()]
    assert got == pytest.approx(want, rel=1e-10, abs=0.0)
    assert hm.two_time(sde, (1,), (1,), [()], t1, t1)[()] == pytest.approx(
        s, rel=1e-10, abs=0.0)
