"""Gauss-Legendre with white noise: the domain is split at C's kink.

White noise makes ``C(t1, t2) = ∫ R σ² R`` kinked on ``t1 = t2``.  A C
propagator between two internal times that the diagram leaves unordered
puts that kink inside the integration domain, and tensor-product
Gauss-Legendre then converges as ``n^-2`` (a 2-D order-2 channel was
7.9e-5 off at 64 nodes).  ``integrate_moment_gauss_legendre`` now
integrates each consistent order of such pairs as its own causal
sub-simplex.

Reference: the exact Itô moment hierarchy of the same system
(``examples/reference/ito_moments.py``), which shares no code with the
package.
"""
from __future__ import annotations

import sys
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


def test_kink_detection_follows_white_noise():
    white = _system(ConstantImpulse(S2)).propagators(
        t_max=2.0, c_closed_form="auto", c_closed_form_only=True,
        diag_C=False, progress=False)
    colored = sw.System(
        field=sw.FieldSpec("phi", N), linear=sw.DiagonalA(gamma=[GAMMA] * N),
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.5, sigma_t=0.5),
            spatial=sw.ExponentialSpatial(sigma_x=1.0)))).propagators(
        t_max=2.0, c_closed_form="auto", c_closed_form_only=True,
        progress=False)
    assert _c_has_diagonal_kink(white.cache)
    assert not _c_has_diagonal_kink(colored.cache)


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
