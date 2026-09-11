"""Demo 8 (b): a non-exponential R, the response of a damped oscillator.

``ExplicitR`` with ``R(τ) = e^{−ζωτ} sin(ω_d τ)/ω_d`` (``ω = 1.7``,
``ζ = 0.3``), which changes sign inside the observation window, a quadratic
local vertex ``F`` and a cubic one ``G`` acting on ``x``, and white force
with a spatial envelope (``CustomImpulse``).  Channels ``F``, ``FG``, ``G``,
``FF``, ``GG`` at orders 1 and 2, the free correlator at order 0, and the
same two-point function at unequal external times.

Reference: the moment hierarchy of the ``(x, v)`` embedding
(``examples/demo8/time8_reference.py``; no sft-wick code).
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
THREE = ("phi_a(x)", "phi_b(y)", "phi_c(z)")
P = pm.OscillatorParams()


@pytest.fixture(scope="module")
def setup():
    system = ts.oscillator_system(P)
    props = ts.exact_propagators(system, ts.exact_C(P, n=24),
                                 t_max=P.t_final + 0.3)
    models = {n: ref.oscillator_model(P, [POS[k] for k in "xyz"[:n]])
              for n in (1, 2, 3)}
    return system, props, models


def _reference(models, comps, tag):
    legs = [("x", c, i) for i, c in enumerate(comps)]
    return models[len(comps)].moments(legs, [tag], P.t_min, P.t_final)[tag]


@pytest.mark.parametrize("obs,order,note,tag,comps", [
    (ONE, 1, "F", (1, 0), (0,)),
    (ONE, 2, "FG", (1, 1), (1,)),
    (TWO, 0, "", (0, 0), (1, 1)),
    (TWO, 1, "G", (0, 1), (0, 1)),
    (TWO, 2, "F", (2, 0), (0, 1)),
    (TWO, 2, "G", (0, 2), (1, 1)),
    (THREE, 1, "F", (1, 0), (0, 1, 1)),
])
def test_channels_against_the_embedding(setup, obs, order, note, tag, comps):
    system, props, models = setup
    want = _reference(models, comps, tag)
    assert want != 0.0
    got = system.expand(obs, orders=[order]).evaluate(
        props, positions=POS, t_final=P.t_final, component_pair=comps,
        orders=[order], vertex_types=None if note == "" else {note},
        method="gauss_legendre", n_gauss=16).total
    assert got == pytest.approx(want, rel=1e-8, abs=0.0)


def test_the_response_changes_sign_inside_the_window():
    """The point of the item: R is not a decaying exponential."""
    tau = np.linspace(0.0, P.span, 400)
    assert P.R.tau(tau).min() < -0.05 * P.R.tau(tau).max()
    assert np.pi / P.R.omega_d < P.span


@pytest.mark.parametrize("order,note,tag,comps,rel,n_samples", [
    (0, "", (0, 0), (1, 1), 1e-12, 2 ** 10),
    (1, "G", (0, 1), (0, 1), 1e-7, 2 ** 14),
    (1, "G", (0, 1), (1, 1), 1e-7, 2 ** 14),
    (2, "F", (2, 0), (0, 1), 1e-3, 2 ** 14),
])
def test_unequal_external_times(setup, order, note, tag, comps, rel, n_samples):
    """``external_times`` pins the two legs at different times; the reference
    conditions the hierarchy at the earlier one.  Gauss-Legendre converges
    only algebraically here (see the demo README), so the check runs on QMC,
    which resolves these 0- to 2-dimensional integrands to its sampling
    error."""
    system, props, models = setup
    t1, t2 = P.t_final - 0.9, P.t_final
    want = models[2].two_time([("x", comps[0], 0)], t1, [("x", comps[1], 1)],
                              t2, [tag], P.t_min)[tag]
    assert want != 0.0
    got = system.expand(TWO, orders=[order]).evaluate(
        props, positions=POS, t_final=t2, component_pair=comps, orders=[order],
        vertex_types=None if note == "" else {note},
        external_times={"x": t1, "y": t2}, method="qmc_vectorized",
        n_samples=n_samples, seed=3).total
    assert got == pytest.approx(want, rel=rel, abs=0.0)


def test_unequal_external_times_nquad(setup):
    system, props, models = setup
    t1, t2 = P.t_final - 0.9, P.t_final
    want = models[2].two_time([("x", 0, 0)], t1, [("x", 1, 1)], t2,
                              [(0, 1)], P.t_min)[(0, 1)]
    got = system.expand(TWO, orders=[1]).evaluate(
        props, positions=POS, t_final=t2, component_pair=(0, 1), orders=[1],
        vertex_types={"G"}, external_times={"x": t1, "y": t2},
        method="nquad").total
    assert got == pytest.approx(want, rel=1e-6, abs=0.0)


@pytest.fixture(scope="module")
def table(setup):
    """The package's own C for this R: a 1-D quadrature of ``∫ R σ²(τ) R``
    per grid cell, spline lookup."""
    system, _props, _models = setup
    return ts.table_propagators(system, P.t_final + 0.3, 31,
                                c_method="gauss_legendre", c_n_gauss=20)


@pytest.mark.parametrize("order,note,tag,comps", [(0, "", (0, 0), (1, 1)),
                                                  (2, "F", (2, 0), (0, 1))])
def test_the_package_c_table(setup, table, order, note, tag, comps):
    system, _props, models = setup
    props = table
    want = _reference(models, comps, tag)
    got = system.expand(TWO, orders=[order]).evaluate(
        props, positions=POS, t_final=P.t_final, component_pair=comps,
        orders=[order], vertex_types=None if note == "" else {note},
        method="gauss_legendre", n_gauss=12).total
    assert got == pytest.approx(want, rel=1e-4, abs=0.0)
