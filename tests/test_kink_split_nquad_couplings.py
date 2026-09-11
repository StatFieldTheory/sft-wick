"""Kink splitting on nquad, and kinks declared by coupling callables.

``integrate_moment_gauss_legendre`` splits the time domain at kinks of the
integrand that the package can see (white-noise C, unordered parents; see
``tests/test_gl_white_noise_kinks.py``).  This file covers the two
extensions:

* ``integrate_moment_nquad`` splits at the same kinks.  Adaptive
  quadrature stops short of its tolerance along a kink: on the white-noise
  order-2 channel below it was 3e-8 to 1.4e-7 off the exact value, and it
  is now at round-off.
* a coupling callable can declare ``has_coincident_time_kinks = True``:
  kinked wherever two of its time arguments coincide.  Gauss-Legendre and
  nquad then split at the pairs of those arguments the causal structure
  leaves unordered, the leg times of a raw vertex or the partner times of
  an ``already_R_contracted`` one.  Demo 4's kernels declare it;
  ``tests/test_demo4_asymmetric_noise.py`` checks them against Campbell's
  closed form and the moment hierarchy.

Reference for the numerical checks here: the exact Itô moment hierarchy
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
from sft_wick.evaluate import (_coupling_kink_candidates, _declares,
                               _kink_orientations, _kink_pairs)
from sft_wick.workflow.specs import ConstantImpulse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "reference"))
import ito_moments as im  # noqa: E402


# ---------------------------------------------------------------------------
# Which pairs a declared coupling kink contributes
# ---------------------------------------------------------------------------

class _Kinked:
    has_coincident_time_kinks = True

    def __call__(self, n_list, t_list):
        return np.ones((2, 2, 2))


class _Smooth:
    def __call__(self, n_list, t_list):
        return np.ones((2, 2, 2))


def _promise(fn, legs=("y_0", "y_1", "y_2")):
    return SimpleNamespace(dynamic_values={"K@0": fn},
                           spatial_args_by_name={"K@0": legs})


def _spatial(ivars, aliases=(), orderings=()):
    return SimpleNamespace(time_integration_vars=tuple(ivars),
                           equal_time_aliases=tuple(aliases),
                           time_orderings=tuple(orderings),
                           c_propagators=())


def test_a_raw_vertex_contributes_its_leg_times():
    sp = _spatial(["y_0", "y_1", "y_2"])
    cand = _coupling_kink_candidates(sp, _promise(_Kinked()))
    pairs = _kink_pairs(sp, False, coupling_pairs=cand)
    assert sorted(pairs) == [("y_0", "y_1"), ("y_0", "y_2"), ("y_1", "y_2")]
    assert len(_kink_orientations(sp, pairs)) == 6


def test_an_r_contracted_vertex_contributes_its_partner_times():
    """Legs alias onto their partners: two F-vertex times u, v and a fixed
    external x.  Only (u, v) is a pair of integration variables."""
    sp = _spatial(["u", "v"], aliases=[("y_0", "x"), ("y_1", "u"),
                                       ("y_2", "v")])
    cand = _coupling_kink_candidates(sp, _promise(_Kinked()))
    assert set(cand) == {("x", "u"), ("x", "v"), ("u", "v")}
    assert _kink_pairs(sp, False, coupling_pairs=cand) == [("u", "v")]
    ordered = _spatial(["u", "v"], aliases=sp.equal_time_aliases,
                       orderings=[("u", "v")])
    assert _kink_pairs(ordered, False, coupling_pairs=cand) == []


def test_an_equal_time_vertex_and_a_shared_partner_give_no_pair():
    sp = _spatial(["y_0"], aliases=[("y_1", "y_0"), ("y_2", "y_0")])
    cand = _coupling_kink_candidates(sp, _promise(_Kinked()))
    assert _kink_pairs(sp, False, coupling_pairs=cand) == []


def test_only_a_declaring_callable_contributes():
    sp = _spatial(["y_0", "y_1", "y_2"])
    assert _coupling_kink_candidates(sp, _promise(_Smooth())) == ()
    assert _coupling_kink_candidates(sp, None) == ()


def test_the_declaration_survives_the_msr_wrapper():
    """``NonLocalVertex.msr_coupling`` wraps the user callable; the
    attribute is read through ``__wrapped__``."""
    kinked = sw.NonLocalVertex("K", order=3, coupling=_Kinked()).msr_coupling
    smooth = sw.NonLocalVertex("K", order=3, coupling=_Smooth()).msr_coupling
    assert not hasattr(kinked, "has_coincident_time_kinks")
    assert _declares(kinked, "has_coincident_time_kinks")
    assert not _declares(smooth, "has_coincident_time_kinks")


# ---------------------------------------------------------------------------
# nquad on white noise, matrix R, against the moment hierarchy
# ---------------------------------------------------------------------------

N = 2
GAMMAS = np.array([0.7, 1.3])          # distinct: R is a matrix
T_MIN, T = 0.4, 1.5
S2 = np.array([[0.8, 0.3], [0.3, 0.5]])
F = np.array([[[0.25, -0.40], [0.15, 0.30]],
              [[-0.35, 0.20], [0.45, -0.10]]])
POS = {"x": 0.0, "y": 0.6}


def _white_system():
    return sw.System(
        field=sw.FieldSpec("phi", N), linear=sw.DiagonalA(gamma=list(GAMMAS)),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.0, sigma_t=1.0),
                spatial=sw.ExponentialSpatial(sigma_x=1.0)),
            sigma2=ConstantImpulse(S2)),
        t_min=T_MIN)


def _hierarchy(a, b):
    """Order-2 coefficient of ``⟨φ_a(x) φ_b(y)⟩``.  The white noise is the
    same at both points (``ConstantImpulse``); time runs from 0 here and
    from ``T_MIN`` in the package."""
    D = 2 * N
    idx = {(c, i): c * 2 + i for c in range(N) for i in range(2)}
    sde = im.PolySDE(D=D, n_tags=1)
    sde.add_linear_drift(-np.diag(np.repeat(GAMMAS, 2)))   # index 2c + i
    S = np.zeros((D, D))
    Q = np.zeros((D, D, D))
    for (c, i), k in idx.items():
        for (d, _j), l in idx.items():
            S[k, l] = S2[c, d]
        for d in range(N):
            for e in range(N):
                Q[k, idx[(d, i)], idx[(e, i)]] += F[c, d, e]
    sde.add_constant_diffusion(S)
    sde.add_quadratic_drift(Q, tag=(1,))
    mono = im.unit(D, idx[(a, 0)], idx[(b, 1)])
    return im.solve(sde, [(mono, (2,))], [T])[(mono, (2,))][0]


@pytest.fixture(scope="module")
def white():
    system = _white_system()
    props = system.propagators(t_max=T_MIN + T + 0.5, c_closed_form="auto",
                               c_closed_form_only=True, diag_C=False,
                               progress=False)
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2], diag_C=False)
    return props, exp


def test_the_white_noise_system_has_matrix_r_and_a_kink_pair(white):
    props, exp = white
    assert not props.cache.model.iso_R
    cv = exp.system.build_coupling_values()
    igs = [dt.build_integrand(cv, {"a": 0, "b": 1})
           for dt in exp.dts_by_order[2]]
    assert any(ig._kink_pairs_under(props.cache, ig.spatial.time_orderings)
               for ig in igs), "no diagram with a C kink between unordered times"


@pytest.mark.parametrize("ab", [(0, 1), (1, 1)])
def test_nquad_white_noise_order_2_matrix_r(white, ab):
    """Before the split nquad stopped at 3e-8 to 1.4e-7 relative on this
    kind of channel (demo 5); the tolerance separates that from round-off."""
    props, exp = white
    got = exp.evaluate(props, positions=POS, t_final=T_MIN + T,
                       component_pair=ab, orders=[2], method="nquad").total
    assert got == pytest.approx(_hierarchy(*ab), rel=1e-12, abs=0.0)


# ---------------------------------------------------------------------------
# A declared kink on nquad (once nquad accepts callable couplings)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_nquad_splits_at_a_declared_coupling_kink():
    """Demo 4's raw exponential kernel on nquad.  ``integrate_moment_nquad``
    refuses callable couplings on this branch; the split is in place for
    when it accepts them, and this test then checks it."""
    sys.path.insert(0, str(ROOT / "examples" / "demo4"))
    import poisson_noise as nz
    import poisson_system as dsys

    p = nz.PARAMS_EXP
    system = dsys.make_system(p, cumulants=(3,), r_contracted=False)
    props = dsys.propagators_for(system, p, t_max=2.2)
    exp = system.expand(("phi_a(x)", "phi_b(y)", "phi_c(z)"), orders=[1],
                        diag_C=False)
    pos = {"x": 0.0, "y": 0.8, "z": -0.5}
    try:
        got = exp.evaluate(props, positions=pos, t_final=1.7,
                           component_pair=(0, 1, 1), orders=[1],
                           method="nquad").total
    except NotImplementedError:
        pytest.skip("nquad does not accept callable couplings yet")
    ref = nz.K_R((0, 1, 1), np.array([0.0, 0.8, -0.5]), np.full(3, 1.7), p)[0]
    assert got == pytest.approx(ref, rel=1e-10, abs=0.0)
