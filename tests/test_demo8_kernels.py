"""Demo 8 (c): custom temporal kernels inside interacting diagrams.

Three kernels that are not the package's OU default, each with
component-dependent (or deliberately equal) decay rates and a local ``F``
with no index symmetry:

* Matérn-3/2 ``λ(1 + a|τ|)e^{−a|τ|}`` and a damped cosine
  ``λ e^{−a|τ|} cos(ωτ)``, through ``CustomKernel``.  Each is the stationary
  covariance of a 2-D linear SDE, so the moment hierarchy of that embedding
  is an exact reference at every order.
* ``GaussianTemporal``, which has no finite embedding.  Its reference is the
  order-1 and order-2 diagrams written out by hand and integrated with C in
  closed form (``erf``), plus time-translation invariance.

Reference: ``examples/demo8/time8_reference.py`` (no sft-wick code).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo8"
sys.path.insert(0, str(DEMO))
import time8_params as pm  # noqa: E402
import time8_reference as ref  # noqa: E402
import time8_system as ts  # noqa: E402

POS = pm.POSITIONS
ONE = ("phi_a(x)",)
TWO = ("phi_a(x)", "phi_b(y)")
KERNELS = {"matern": pm.Matern32(), "damped_cosine": pm.DampedCosine()}
EQUAL, UNEQUAL = (1.1, 1.1), (0.8, 1.3)


def _params(name, gamma):
    return pm.KernelParams(kernel=KERNELS[name], gamma=gamma)


def _exact(p):
    system = ts.kernel_system(p)
    props = ts.exact_propagators(system, ts.exact_C(p, n=32,
                                                    has_diagonal_kink=True),
                                 t_max=p.t_final + 0.3)
    return system, props


def _hierarchy(p, order, comps):
    model = ref.kernel_model(p, [POS[k] for k in "xy"[:len(comps)]])
    legs = [("phi", c, i) for i, c in enumerate(comps)]
    return model.moments(legs, [(order,)], p.t_min, p.t_final)[(order,)]


def _check(system, props, p, order, comps, method, rel, **kw):
    obs = ONE if len(comps) == 1 else TWO
    want = _hierarchy(p, order, comps)
    assert want != 0.0
    got = system.expand(obs, orders=[order]).evaluate(
        props, positions=POS, t_final=p.t_final, component_pair=comps,
        orders=[order], method=method, **kw).total
    assert got == pytest.approx(want, rel=rel, abs=0.0), (order, comps, method)


@pytest.mark.parametrize("name", sorted(KERNELS))
@pytest.mark.parametrize("order,comps", [(1, (0,)), (0, (1, 1)), (2, (0, 1)),
                                         (2, (1, 1))])
def test_equal_rates_gauss_legendre(name, order, comps):
    p = _params(name, EQUAL)
    system, props = _exact(p)
    _check(system, props, p, order, comps, "gauss_legendre", 1e-10, n_gauss=12)


@pytest.mark.parametrize("name", sorted(KERNELS))
@pytest.mark.parametrize("order,comps", [(1, (1,)), (2, (0, 1))])
def test_unequal_rates_nquad(name, order, comps):
    """Component-dependent rates give a matrix R: the scalar loops."""
    p = _params(name, UNEQUAL)
    system, props = _exact(p)
    _check(system, props, p, order, comps, "nquad", 1e-7)


@pytest.mark.parametrize("name", sorted(KERNELS))
def test_the_package_c_table(name):
    """``CustomKernel`` reaches the package's own C quadrature, whose
    smoothness it cannot know: ``c_method='auto'`` picks dblquad there, and
    the two quadratures agree to their own accuracy."""
    p = _params(name, EQUAL)
    system = ts.kernel_system(p)
    props = ts.table_propagators(system, p.t_final + 0.3, 21,
                                 c_method="gauss_legendre", c_n_gauss=24)
    _check(system, props, p, 2, (0, 1), "gauss_legendre", 1e-3, n_gauss=12)
    _check(system, props, p, 1, (0,), "gauss_legendre", 1e-4, n_gauss=12)


# ---------------------------------------------------------------------------
# GaussianTemporal: no embedding
# ---------------------------------------------------------------------------

def _gaussian_hand(p):
    env = pm.GaussianEnvelope(p.sigma_x)

    def C(b, t1, t2, r):
        return env(r) * ref.gaussian_kernel_C(
            p.gamma[b], p.kernel.lam, p.kernel.sigma,
            np.asarray(t1, float) - p.t_min, np.asarray(t2, float) - p.t_min)

    def R(a, t, s):
        return np.exp(-p.gamma[a] * (np.asarray(t) - np.asarray(s)))

    return ref.HandContraction(R, C, p.t_min, n=48), C


def _gaussian_reference(p, order, comps):
    hand, C = _gaussian_hand(p)
    r = abs(POS["x"] - POS["y"])
    if len(comps) == 1:
        return hand.tadpole(comps[0], p.t_final)
    if order == 0:
        return hand.two_point_order0(comps[0], comps[1], p.t_final, r)
    return hand.two_point_order2(comps[0], comps[1], p.t_final, r)


def test_the_hand_contraction_is_converged():
    """48 and 64 nodes per dimension agree, so the reference below is the
    diagrams and not the quadrature."""
    p = pm.KernelParams(kernel=pm.GaussianKernel(), gamma=EQUAL)
    hand48, C = _gaussian_hand(p)
    hand64 = ref.HandContraction(hand48.R, C, p.t_min, n=64)
    r = abs(POS["x"] - POS["y"])
    for comps in ((0, 1), (1, 1)):
        a, b = comps
        assert hand64.two_point_order2(a, b, p.t_final, r) == pytest.approx(
            hand48.two_point_order2(a, b, p.t_final, r), rel=1e-11, abs=0.0)


@pytest.mark.parametrize("gamma,method,kw", [
    (EQUAL, "gauss_legendre", dict(n_gauss=12)),
    (UNEQUAL, "nquad", {}),
])
@pytest.mark.parametrize("order,comps", [(1, (0,)), (0, (1, 1)), (2, (0, 1)),
                                         (2, (1, 1))])
def test_gaussian_kernel_against_the_hand_contraction(gamma, method, kw, order,
                                                      comps):
    p = pm.KernelParams(kernel=pm.GaussianKernel(), gamma=gamma)
    system = ts.kernel_system(p)
    props = ts.exact_propagators(system, ts.exact_C(p, n=40),
                                 t_max=p.t_final + 0.3)
    want = _gaussian_reference(p, order, comps)
    assert want != 0.0
    obs = ONE if len(comps) == 1 else TWO
    got = system.expand(obs, orders=[order]).evaluate(
        props, positions=POS, t_final=p.t_final, component_pair=comps,
        orders=[order], method=method, **kw).total
    assert got == pytest.approx(want, rel=1e-10, abs=0.0)


def test_gaussian_kernel_c_table():
    p = pm.KernelParams(kernel=pm.GaussianKernel(), gamma=EQUAL)
    system = ts.kernel_system(p)
    props = ts.table_propagators(system, p.t_final + 0.3, 21)
    want = _gaussian_reference(p, 2, (0, 1))
    got = system.expand(TWO, orders=[2]).evaluate(
        props, positions=POS, t_final=p.t_final, component_pair=(0, 1),
        orders=[2], method="gauss_legendre", n_gauss=12).total
    assert got == pytest.approx(want, rel=1e-4, abs=0.0)


def test_time_translation_invariance():
    """The kernel is stationary and the rates constant, so shifting ``t_min``
    and the observation time together changes nothing -- a check that needs
    no reference at all, and the one available for a kernel with no
    embedding."""
    vals: dict[int, list] = {0: [], 2: []}
    for shift in (-1.1, 0.0, 0.9):
        p = pm.KernelParams(kernel=pm.GaussianKernel(), gamma=EQUAL,
                            t_min=shift)
        system = ts.kernel_system(p)
        props = ts.table_propagators(system, p.t_final + 0.3, 15)
        exp = system.expand(TWO, orders=[0, 2])
        for order in (0, 2):
            vals[order].append(exp.evaluate(
                props, positions=POS, t_final=p.t_final, component_pair=(1, 1),
                orders=[order], method="gauss_legendre", n_gauss=12).total)
    for order, got in vals.items():
        assert got[0] != 0.0
        for v in got[1:]:
            assert v == pytest.approx(got[0], rel=1e-12, abs=0.0)
