"""Demo 8 (a): a decay rate that varies in time.

``DiagonalA(gamma=callable)`` with ``γ(t) = [1 + 0.5 sin t, 0.6 + 0.3 cos 2t]``
-- unequal, time-varying components, hence a matrix R -- a local ``F`` with no
index symmetry, ``t_min`` below and above 0, and orders 0-2 of one-, two- and
three-point functions.  Up to 0.5.0 the end-to-end route was checked at order
0 with a constant rate only.

Reference: the moment hierarchy of the Markov embedding ``(φ, η)`` with the
time-dependent drift (``examples/demo8/time8_reference.py``; no sft-wick
code), integrated by ``solve_ivp`` at ``rtol = 1e-12``.  C reaches the
package either as the defining integrals (``exact C``: a row then measures R
and the diagrams alone) or through the package's own quadrature table.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo8"
sys.path.insert(0, str(DEMO))
import time8_params as pm  # noqa: E402
import time8_reference as ref  # noqa: E402
import time8_system as ts  # noqa: E402

POS = pm.POSITIONS
ONE = ("phi_a(x)",)
TWO = ("phi_a(x)", "phi_b(y)")
THREE = ("phi_a(x)", "phi_b(y)", "phi_c(z)")

_UNEQUAL = pm.RateParams()
_EQUAL = replace(pm.RateParams(), rate=pm.EQUAL_RATE)
_SHIFTED = replace(pm.RateParams(), t_min=0.7)


def _hierarchy(p, obs, order, comps):
    model = ref.rate_model(p, [POS[k] for k in "xyz"[:len(obs)]])
    legs = [("phi", c, i) for i, c in enumerate(comps)]
    return model.moments(legs, [(order,)], p.t_min, p.t_final)[(order,)]


@pytest.fixture(scope="module")
def exact():
    """``{params: (system, propagators)}`` with the exact C."""
    out = {}
    for p in (_UNEQUAL, _EQUAL, _SHIFTED):
        system = ts.rate_system(p, t_max_cache=p.t_final + 1.0, n_grid_cache=800)
        cf = ts.exact_C(p, n=24, has_diagonal_kink=True)
        out[p] = (system, ts.exact_propagators(system, cf, t_max=p.t_final + 0.3))
    return out


def _check(system, props, p, obs, order, comps, method, rel, **kw):
    want = _hierarchy(p, obs, order, comps)
    assert want != 0.0
    got = system.expand(obs, orders=[order]).evaluate(
        props, positions=POS, t_final=p.t_final, component_pair=comps,
        orders=[order], method=method, **kw).total
    assert got == pytest.approx(want, rel=rel, abs=0.0), (obs, order, comps,
                                                          method)
    return got


@pytest.mark.parametrize("order,comps", [(1, (0,)), (1, (1,)), (0, (1, 1)),
                                         (2, (0, 1)), (2, (1, 1))])
def test_matrix_R_negative_t_min_nquad(exact, order, comps):
    """Component-dependent time-varying rates give a matrix R, which runs on
    the scalar loops; ``t_min = −1.3`` needs the rate-cache grid extended
    below 0."""
    system, props = exact[_UNEQUAL]
    obs = ONE if len(comps) == 1 else TWO
    _check(system, props, _UNEQUAL, obs, order, comps, "nquad", 1e-7)


@pytest.mark.parametrize("order,comps", [(1, (0,)), (2, (0, 1))])
def test_matrix_R_positive_t_min_nquad(exact, order, comps):
    system, props = exact[_SHIFTED]
    obs = ONE if len(comps) == 1 else TWO
    _check(system, props, _SHIFTED, obs, order, comps, "nquad", 1e-7)


def test_matrix_R_scalar_qmc_agrees_within_its_sampling_error(exact):
    system, props = exact[_UNEQUAL]
    _check(system, props, _UNEQUAL, ONE, 1, (0,), "qmc_scalar", 1e-6,
           n_samples=2 ** 11, seed=5)
    _check(system, props, _UNEQUAL, TWO, 2, (0, 1), "qmc_scalar", 3e-3,
           n_samples=2 ** 10, seed=5)


@pytest.mark.parametrize("obs,order,comps", [
    (ONE, 1, (0,)), (ONE, 1, (1,)), (TWO, 0, (1, 1)), (TWO, 2, (0, 1)),
    (TWO, 2, (1, 1)), (THREE, 1, (0, 1, 1))])
def test_scalar_R_gauss_legendre(exact, obs, order, comps):
    """Equal (still time-varying) components give a scalar R, which every
    integrator accepts."""
    system, props = exact[_EQUAL]
    _check(system, props, _EQUAL, obs, order, comps, "gauss_legendre", 1e-10,
           n_gauss=12)


@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized"])
def test_matrix_R_on_the_batched_integrators_when_they_accept_it(exact, method):
    """These refuse a matrix R today.  The check runs the moment they take
    one."""
    system, props = exact[_UNEQUAL]
    kw = dict(n_gauss=12) if method == "gauss_legendre" else dict(
        n_samples=2 ** 14, seed=5)
    try:
        _check(system, props, _UNEQUAL, TWO, 2, (0, 1), method,
               1e-10 if method == "gauss_legendre" else 1e-4, **kw)
    except NotImplementedError as exc:
        pytest.skip(f"{method} does not take a matrix R yet: "
                    f"{str(exc).splitlines()[0][:60]}")


def test_the_rate_cache_grid_sets_the_accuracy():
    """Γ_a is the integral of the cubic spline of γ_a on the cache grid, so
    the moments converge in its spacing: 5.8e-6 at h = 0.50 (the DiagonalA
    default grid), 4.0e-12 at h = 0.015."""
    p = _EQUAL
    want = _hierarchy(p, TWO, 2, (0, 1))
    cf = ts.exact_C(p, n=24, has_diagonal_kink=True)
    errs = {}
    for t_max_cache in (100.0, p.t_final + 1.0):
        system = ts.rate_system(p, t_max_cache=t_max_cache)
        props = ts.exact_propagators(system, cf, t_max=p.t_final + 0.3)
        got = system.expand(TWO, orders=[2]).evaluate(
            props, positions=POS, t_final=p.t_final, component_pair=(0, 1),
            orders=[2], method="gauss_legendre", n_gauss=12).total
        errs[t_max_cache] = abs(got / want - 1.0)
    assert errs[100.0] < 1e-4
    assert errs[p.t_final + 1.0] < 1e-10
    assert errs[100.0] > 100 * errs[p.t_final + 1.0]


@pytest.fixture(scope="module")
def table(exact):
    """The package's own C: quadrature on a 21-point grid, spline lookup."""
    system, _ = exact[_UNEQUAL]
    return system, ts.table_propagators(system, _UNEQUAL.t_final + 0.3, 21,
                                        c_method="gauss_legendre", c_n_gauss=24)


@pytest.mark.parametrize("order,comps", [(1, (0,)), (2, (0, 1))])
def test_the_package_c_table(table, order, comps):
    system, props = table
    obs = ONE if len(comps) == 1 else TWO
    _check(system, props, _UNEQUAL, obs, order, comps, "nquad", 1e-4)
