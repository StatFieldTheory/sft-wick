"""Demo 5: white noise through the package's own machinery.

Part A (``examples/demo5/run.py``): additive white noise (a
``ConstantImpulse`` matrix, shared by every point) plus coloured noise, a
generic quadratic drift and ``t_min = 0.5``, in three variants that take
the ``diag_C=False``, ``diag_C=True`` (diag-C fast path) and ``iso_C=True``
code paths; every integrator.  Part B
(``examples/demo5/multiplicative.py``): multiplicative white noise at L0,
local vertices with two ψ legs, scalar and matrix R.

Reference: the exact Itô moment hierarchy (``examples/demo5/white_reference.py``
and ``examples/reference/ito_moments.py``), which shares no code with the
package.  Each variant would have failed on a defect fixed on 2026-09-11:
``t_min`` and ``iso_C`` (40a9288), the diag-C fast path (5cb9ab4) and the
repeated R pair (63fc842).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo5"
sys.path.insert(0, str(DEMO))
import white_model as m5  # noqa: E402
import multiplicative as mult  # noqa: E402
from white_reference import Hierarchy  # noqa: E402

T = 1.6
POS = {"x": 0.0, "y": 0.7, "z": -0.4}


@pytest.fixture(scope="module", params=m5.VARIANTS)
def setup(request):
    p = m5.variant(request.param)
    system = m5.make_system(p)
    props = m5.propagators_for(system, p, t_max=p.t_min + T + 0.5)
    return p, system, props, m5.expand_flags(p)


def _check(setup, obs, order, comps, method, rel, **kw):
    p, system, props, flags = setup
    labels = ("x", "y", "z")[:len(obs)]
    H = Hierarchy(p, [POS[lab] for lab in labels], m5.F_TENSOR)
    ref = H.moments([(c, i) for i, c in enumerate(comps)], [order], T)[order]
    exp = system.expand(obs, orders=[order], **flags)
    got = exp.evaluate(props, positions=POS, t_final=p.t_min + T,
                       component_pair=comps, orders=[order], method=method,
                       **kw).total
    assert got == pytest.approx(ref, rel=rel, abs=0.0), (obs, order, comps,
                                                        method)
    return got


@pytest.mark.parametrize("comps", [(0, 1), (1, 1)])
@pytest.mark.parametrize("order", [0, 2])
def test_two_point_gauss_legendre(setup, order, comps):
    _check(setup, ("phi_a(x)", "phi_b(y)"), order, comps, "gauss_legendre",
           1e-10, n_gauss=12)


@pytest.mark.parametrize("method,rel,kw", [
    ("gauss_legendre", 1e-10, dict(n_gauss=12)),
    ("qmc_vectorized", 1e-4, dict(n_samples=2 ** 12, seed=2)),
    ("qmc_scalar", 1e-4, dict(n_samples=2 ** 10, seed=2)),
    ("qmc", 1e-4, dict(n_samples=2 ** 10, seed=2)),
    ("nquad", 1e-8, {}),
])
def test_tadpole_every_integrator(setup, method, rel, kw):
    for a in range(2):
        _check(setup, ("phi_a(x)",), 1, (a,), method, rel, **kw)


def test_scalar_loop_equals_the_batched_loop_at_order_2(setup):
    """The diag-C fast path of the scalar loop: same Sobol points, so the
    two loops agree to round-off, and both with the hierarchy to QMC
    precision."""
    p, system, props, flags = setup
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2], **flags)
    kw = dict(positions=POS, t_final=p.t_min + T, component_pair=(0, 1),
              orders=[2], n_samples=2 ** 10, seed=2)
    batched = exp.evaluate(props, method="qmc_vectorized", **kw).total
    scalar = exp.evaluate(props, method="qmc_scalar", **kw).total
    assert scalar == pytest.approx(batched, rel=1e-12, abs=0.0)
    H = Hierarchy(p, [POS["x"], POS["y"]], m5.F_TENSOR)
    ref = H.moments([(0, 0), (1, 1)], [2], T)[2]
    assert batched == pytest.approx(ref, rel=1e-3, abs=0.0)


def test_three_point_order_1(setup):
    _check(setup, ("phi_a(x)", "phi_b(y)", "phi_c(z)"), 1, (0, 1, 1),
           "gauss_legendre", 1e-10, n_gauss=12)


def test_order_4_two_point():
    p = m5.variant("mixing")
    system = m5.make_system(p)
    props = m5.propagators_for(system, p, t_max=p.t_min + T + 0.5)
    _check((p, system, props, m5.expand_flags(p)),
           ("phi_a(x)", "phi_b(y)"), 4, (0, 1), "gauss_legendre", 1e-8,
           n_gauss=10)


# ---------------------------------------------------------------------------
# Part B: multiplicative noise at L0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gammas,method,rel,kw", [
    ((1.0, 1.0), "gauss_legendre", 1e-10, dict(n_gauss=16)),
    ((0.6, 1.6), "gauss_legendre", 1e-10, dict(n_gauss=16)),
    ((0.6, 1.6), "qmc_vectorized", 1e-3, dict(n_samples=2 ** 13, seed=3)),
    ((0.6, 1.6), "qmc_scalar", 1e-3, dict(n_samples=2 ** 13, seed=3)),
], ids=["scalar-R", "matrix-R-gauss_legendre", "matrix-R-qmc_vectorized",
        "matrix-R-qmc_scalar"])
def test_multiplicative_channels(gammas, method, rel, kw):
    """Matrix R (distinct rates) on the batched integrators raised
    NotImplementedError before the matrix-R change; the scalar loop and
    ``nquad`` were the only routes."""
    rows = mult.compare(np.array(gammas), mult.package(gammas, method, **kw))
    for r in rows:
        if r["hierarchy"] == 0.0:
            assert abs(r["package"]) < 1e-12, r
        else:
            assert r["rel"] < rel, r


def test_multiplicative_matrix_R_batched_equals_scalar_loop():
    """Same Sobol points: the batched and the scalar loop agree to
    round-off on every channel and component tuple of part B."""
    kw = dict(n_samples=2 ** 8, seed=5)
    batched = mult.package((0.6, 1.6), "qmc_vectorized", **kw)
    scalar = mult.package((0.6, 1.6), "qmc_scalar", **kw)
    assert len(batched) == len(scalar) > 0
    for b, s in zip(batched, scalar):
        assert (b["channel"], b["comps"]) == (s["channel"], s["comps"])
        assert b["package"] == pytest.approx(s["package"], rel=1e-12,
                                             abs=1e-15), (b, s)


def test_two_psi_vertex_factor_reproduces_C():
    """A local ψψ vertex with coupling ½ S⁰ at order 1 is the C propagator:
    the convention the multiplicative vertices rely on."""
    cache = mult._cache((1.0, 1.0))
    import sft_wick as sw
    from sft_wick.evaluate import integrate_diagrams
    sw.reset_uid_counter()
    phi = sw.Field("phi", "physical", n_components=2)
    psi = sw.Field("psi", "response", n_components=2)
    action = sw.Action(vertices=[sw.Vertex(fields=[psi, psi], coupling="S",
                                           local=True)])
    dts = sw.compute_moment([phi("a", "x"), phi("b", "y")], action, order=1,
                            iso_R=True, diag_R=True,
                            diag_C=False).diagram_terms(1)
    for a, b in itertools.product(range(2), repeat=2):
        got, _ = integrate_diagrams(
            dts, {"S": 0.5 * mult.S0}, lambda_f=mult.T, cache=cache,
            method="gauss_legendre", n_gauss=16,
            fixed_indices={"a": a, "b": b}, positions={"x": 0.0, "y": 0.0})
        ref = np.asarray(cache.C_value(0.0, mult.T, 0.0, mult.T))[a, b]
        assert got == pytest.approx(ref, rel=1e-12, abs=1e-15)
