"""Demo 8 (d): a white-noise amplitude that varies in time, and the C table.

``CustomImpulse`` with ``σ²_ab(t; x, y) = δ_ab s_a(t) K_w(x−y)``,
``s_a(t) = S_a (1 + m_a sin(w_a t + p_a))``, on top of OU noise, with
constant component-dependent rates and a local ``F`` with no index symmetry.
Orders 0-2 against the moment hierarchy with the time-dependent diffusion
(``examples/demo8/time8_reference.py``; no sft-wick code).

The package's own C table is the second subject.  White noise gives C a
derivative jump on its time diagonal; the table's tensor-product spline
cannot represent it, so away from the diagonal the table error falls with
the grid step rather than at spline order.  The demo README carries the
measured rates.
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
EQUAL = pm.WhiteParams(gamma=(1.2, 1.2))
UNEQUAL = pm.WhiteParams()


def _hierarchy(p, order, comps):
    model = ref.white_model(p, [POS[k] for k in "xy"[:len(comps)]])
    legs = [("phi", c, i) for i, c in enumerate(comps)]
    return model.moments(legs, [(order,)], p.t_min, p.t_final)[(order,)]


@pytest.fixture(scope="module")
def exact():
    out = {}
    for p in (EQUAL, UNEQUAL):
        system = ts.white_system(p)
        out[p] = (system, ts.exact_propagators(system, ts.exact_C(p, n=32),
                                               t_max=p.t_final + 0.3))
    return out


def _evaluate(system, props, p, order, comps, method, **kw):
    obs = ONE if len(comps) == 1 else TWO
    return system.expand(obs, orders=[order]).evaluate(
        props, positions=POS, t_final=p.t_final, component_pair=comps,
        orders=[order], method=method, **kw).total


def test_the_white_amplitude_really_varies():
    s = UNEQUAL.white.s(np.linspace(UNEQUAL.t_min, UNEQUAL.t_final, 101))
    for a in range(pm.N):
        assert s[a].max() / s[a].min() > 2.0
    assert not np.allclose(s[0] / s[0].max(), s[1] / s[1].max())


@pytest.mark.parametrize("order,comps", [(1, (0,)), (1, (1,)), (0, (1, 1)),
                                         (2, (0, 1)), (2, (1, 1))])
def test_equal_rates_gauss_legendre(exact, order, comps):
    system, props = exact[EQUAL]
    want = _hierarchy(EQUAL, order, comps)
    assert want != 0.0
    got = _evaluate(system, props, EQUAL, order, comps, "gauss_legendre",
                    n_gauss=12)
    assert got == pytest.approx(want, rel=1e-10, abs=0.0)


@pytest.mark.parametrize("order,comps", [(1, (0,)), (2, (0, 1))])
def test_unequal_rates_nquad(exact, order, comps):
    system, props = exact[UNEQUAL]
    want = _hierarchy(UNEQUAL, order, comps)
    got = _evaluate(system, props, UNEQUAL, order, comps, "nquad")
    assert got == pytest.approx(want, rel=1e-6, abs=0.0)


@pytest.fixture(scope="module")
def tables(exact):
    system, _ = exact[EQUAL]
    return {n: ts.table_propagators(system, EQUAL.t_final + 0.3, n,
                                    c_method="gauss_legendre", c_n_gauss=24)
            for n in (11, 31)}


def test_the_table_converges_in_its_step(exact, tables):
    """Order 2 reads C in the band around the diagonal, where a tensor
    product over (t1, t2) crossed the kink.  In the (min(t1,t2), |t1-t2|)
    chart the kink is the d = 0 grid edge, so Gauss-Legendre recovers the
    spline's own order: 9.5e-06 at n_grid_t = 11 and 5.9e-08 at 31.  QMC
    sits at its own ~1e-05 noise floor at both, so the deterministic
    route is the one that measures the table."""
    system, _ = exact[EQUAL]
    want = _hierarchy(EQUAL, 2, (0, 1))
    errs = {}
    for n, props in tables.items():
        got = _evaluate(system, props, EQUAL, 2, (0, 1),
                        "gauss_legendre", n_gauss=12)
        errs[n] = abs(got / want - 1.0)
    assert errs[11] < 1e-4
    assert errs[31] < errs[11] / 4


def test_the_tadpole_reads_the_diagonal_and_converges_faster(exact, tables):
    """``C(s, s)`` comes from a spline of the table's own diagonal, which is
    smooth: 1.2e-5 at n_grid_t = 11, 1.5e-8 at 41."""
    system, _ = exact[EQUAL]
    want = _hierarchy(EQUAL, 1, (0,))
    got = _evaluate(system, tables[31], EQUAL, 1, (0,), "gauss_legendre",
                    n_gauss=12)
    assert got == pytest.approx(want, rel=1e-6, abs=0.0)


def test_where_the_table_C_is_wrong(tables):
    """Pointwise: on the diagonal the table is accurate, just off it the
    spline has to cross the kink."""
    cf = ts.exact_C(EQUAL, n=32)
    t1 = np.linspace(EQUAL.t_min + 0.2, EQUAL.t_final, 37)
    r = abs(POS["x"] - POS["y"])
    err = {}
    for n, props in tables.items():
        for delta in (0.0, 1e-2):
            got = props.cache.C_at_batch(t1, t1 - delta, np.zeros_like(t1),
                                         np.full_like(t1, r))
            want = cf.diagonal(t1, t1 - delta, np.full_like(t1, r))
            err[(n, delta)] = float(np.max(np.abs(got - want))
                                    / np.abs(want).max())
    assert err[(31, 0.0)] < 1e-6
    assert err[(31, 1e-2)] < err[(11, 1e-2)]
