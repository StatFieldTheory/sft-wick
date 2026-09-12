"""Gauss-Legendre: the domain is split at C's diagonal kink.

White noise makes ``C(t1, t2) = ∫ R σ² R`` kinked on ``t1 = t2``, and a
κ² with a ``|Δt|`` cusp makes ``C = ∫∫ R κ² R`` kinked there too, in its
third derivative (``∂²C/∂t1∂t2 = R κ² R``).  A C propagator between two
internal times that the diagram leaves unordered puts that kink inside the
integration domain, and tensor-product Gauss-Legendre then converges
algebraically (``n^-2`` under white noise: a 2-D order-2 channel was
7.9e-5 off at 64 nodes).  ``integrate_moment_gauss_legendre`` integrates
each consistent order of such pairs as its own causal sub-simplex.

Reference: the exact Itô moment hierarchy of the same system
(``examples/reference/ito_moments.py``), which shares no code with the
package.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.evaluate import (_c_has_diagonal_kink, _kink_orientations,
                               _kink_pairs)


def _c_kink_pairs(spatial):
    return _kink_pairs(spatial, c_kink=True)
from sft_wick.workflow.specs import ConstantImpulse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "examples" / "reference"))
import ito_moments as im  # noqa: E402


def _spatial(orderings, c_pairs):
    return SimpleNamespace(
        equal_time_aliases=(), time_integration_vars=("a", "b", "c"),
        time_orderings=list(orderings),
        c_propagators=[(l, r, None, None) for l, r in c_pairs])


def test_three_unordered_times_give_the_six_orders():
    sp = _spatial([], [("a", "b"), ("b", "c"), ("a", "c")])
    pairs = _c_kink_pairs(sp)
    assert len(pairs) == 3
    assert len(_kink_orientations(sp, pairs)) == 6


def test_an_existing_order_is_respected():
    """With a < b already imposed, the consistent total orders of three
    times are the 3 linear extensions; pairs already ordered through the
    causal structure are not split."""
    sp = _spatial([("a", "b")], [("a", "b"), ("b", "c"), ("a", "c")])
    pairs = _c_kink_pairs(sp)
    assert ("a", "b") not in pairs and len(pairs) == 2
    assert len(_kink_orientations(sp, pairs)) == 3


def test_a_tadpole_is_not_a_kink_pair():
    sp = _spatial([], [("a", "a"), ("b", "b")])
    assert _c_kink_pairs(sp) == []


def test_two_unordered_parents_are_a_kink_pair_without_white_noise():
    """A vertex with several ψ legs at one time has several parents, and its
    upper bound ``min(parents)`` is kinked where two unordered ones cross.
    That does not depend on the noise."""
    sp = _spatial([("c", "a"), ("c", "b")], [])
    assert _kink_pairs(sp, c_kink=False) == [("a", "b")]
    assert len(_kink_orientations(sp, [("a", "b")])) == 2
    ordered = _spatial([("c", "a"), ("c", "b"), ("b", "a")], [])
    assert _kink_pairs(ordered, c_kink=False) == []


N = 2
GAMMA, T_FINAL = 1.0, 1.8
S2 = np.array([[0.8, 0.3], [0.3, 0.5]])
F = np.array([[[0.25, -0.40], [0.15, 0.30]],
              [[-0.35, 0.20], [0.45, -0.10]]])


def _system(sigma2):
    return sw.System(
        field=sw.FieldSpec("phi", N), linear=sw.DiagonalA(gamma=[GAMMA] * N),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.0, sigma_t=1.0),
                spatial=sw.ExponentialSpatial(sigma_x=1.0)),
            sigma2=sigma2))


@dataclass(frozen=True)
class _Exp:
    """``λ e^{−|Δt|/σ_t}`` as a user callable: cusped, and declaring
    nothing."""

    sigma_t: float

    def __call__(self, dt) -> float:
        return float(np.exp(-abs(dt) / self.sigma_t))


@dataclass(frozen=True)
class _Gauss:
    """``e^{−Δt²/2σ_t²}`` as a user callable: smooth at ``Δt = 0``."""

    sigma_t: float

    def __call__(self, dt) -> float:
        return float(np.exp(-0.5 * (dt / self.sigma_t) ** 2))


def _colored(temporal):
    return sw.System(
        field=sw.FieldSpec("phi", N), linear=sw.DiagonalA(gamma=[GAMMA] * N),
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=temporal,
            spatial=sw.ExponentialSpatial(sigma_x=1.0))))


def test_kink_detection_follows_white_noise_and_the_kernel_cusp():
    """White noise jumps C's first derivative, a ``|Δt|`` cusp in κ² its
    third (``∂²C/∂t1∂t2 = R κ² R``); a Gaussian κ² leaves C smooth."""
    white = _system(ConstantImpulse(S2)).propagators(
        t_max=2.0, c_closed_form="auto", c_closed_form_only=True,
        diag_C=False, progress=False)
    exponential = _colored(sw.ExponentialTemporal(lam=0.5, sigma_t=0.5)
                           ).propagators(t_max=2.0, n_grid_t=8,
                                         c_closed_form=None, progress=False)
    gaussian = _colored(sw.GaussianTemporal(lam=0.5, sigma_t=0.5)
                        ).propagators(t_max=2.0, n_grid_t=8,
                                      c_closed_form=None, progress=False)
    assert _c_has_diagonal_kink(white.cache)
    assert _c_has_diagonal_kink(exponential.cache)
    assert not _c_has_diagonal_kink(gaussian.cache)


def test_the_builtin_closed_form_declares_the_kink():
    """``builtin_closed_form_for`` exists only for the exponential temporal
    kernel, so the closed form it returns carries the cusp."""
    colored = _colored(sw.ExponentialTemporal(lam=0.5, sigma_t=0.5)
                       ).propagators(t_max=2.0, c_closed_form="auto",
                                     c_closed_form_only=True, progress=False)
    assert _c_has_diagonal_kink(colored.cache)


def test_a_custom_kernel_is_probed():
    """A callable κ² declares nothing, so the cusp probe decides: the same
    two kernels, written as ``CustomKernel``."""
    cusped = _colored(sw.CustomKernel(fn=_Exp(0.5))).propagators(
        t_max=2.0, n_grid_t=8, c_closed_form=None, progress=False)
    smooth = _colored(sw.CustomKernel(fn=_Gauss(0.5))).propagators(
        t_max=2.0, n_grid_t=8, c_closed_form=None, progress=False)
    assert _c_has_diagonal_kink(cusped.cache)
    assert not _c_has_diagonal_kink(smooth.cache)


def _hierarchy_ff(a, b):
    """``⟨φ_a φ_b⟩`` at O(F²) for white noise ``S2`` shared by the two
    observation points (``ConstantImpulse`` is position independent, so
    the noise is the same at x and y)."""
    D = 2 * N                                   # (component, point)
    idx = {(c, i): c * 2 + i for c in range(N) for i in range(2)}
    sde = im.PolySDE(D=D, n_tags=1)
    sde.add_linear_drift(-GAMMA * np.eye(D))
    S = np.zeros((D, D))
    Q = np.zeros((D, D, D))
    for (c, i), k in idx.items():
        for (d, j), l in idx.items():
            S[k, l] = S2[c, d]
        for d in range(N):
            for e in range(N):
                Q[k, idx[(d, i)], idx[(e, i)]] += F[c, d, e]
    sde.add_constant_diffusion(S)
    sde.add_quadratic_drift(Q, tag=(1,))
    mono = im.unit(D, idx[(a, 0)], idx[(b, 1)])
    return im.solve(sde, [(mono, (2,))], [T_FINAL])[(mono, (2,))][0]


@pytest.mark.parametrize("ab", [(0, 1), (1, 1)])
def test_white_noise_order_2_converges_exponentially(ab):
    system = _system(ConstantImpulse(S2))
    props = system.propagators(t_max=2.0, c_closed_form="auto",
                               c_closed_form_only=True, diag_C=False,
                               progress=False)
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2], diag_C=False)
    ref = _hierarchy_ff(*ab)
    got = exp.evaluate(props, positions={"x": 0.0, "y": 0.7},
                       t_final=T_FINAL, component_pair=ab, orders=[2],
                       method="gauss_legendre", n_gauss=16).total
    assert got == pytest.approx(ref, rel=1e-11, abs=0.0)
