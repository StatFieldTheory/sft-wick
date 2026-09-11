"""Demo 6, part C: a cubic and a quartic local vertex in one system.

``G_{a;bcd} ψ_a φ_b φ_c φ_d`` had never been built at L1 (every example has
one cubic ``F`` vertex only), and no test mixed two local vertices of
different rank.  ``⟨φ_a(x)⟩`` at orders 2 and 3 and ``⟨φ_a(x) φ_b(y)⟩`` at
orders 2 and 3, channel by channel, against the moment hierarchy of the
Markov embedding with a cubic drift term (``examples/demo6``).

The two-point channels run at equal external times, where Gauss-Legendre
converges exponentially; ``examples/demo6/vertex6_cubic.py`` also runs them
at distinct times, where a C propagator between an internal and a fixed
external time is kinked inside the domain and the Gauss-Legendre kink split
does not reach it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo6"
sys.path.insert(0, str(DEMO))
import vertex6_model as md  # noqa: E402
import vertex6_system as vs  # noqa: E402
from vertex6_reference import Hierarchy, tag  # noqa: E402

POS = {"x": 0.0, "y": 0.8}
T = 1.9


@pytest.fixture(scope="module")
def setup():
    p = md.PARAMS_COMMON
    system = vs.make_system(p, f_amplitude=1.0, g_amplitude=1.0)
    props = vs.propagators_for(system, p, t_max=T + 0.5)
    exps = {1: system.expand(("phi_a(x)",), orders=[2, 3], diag_C=False),
            2: system.expand(("phi_a(x)", "phi_b(y)"), orders=[2, 3],
                             diag_C=False)}
    sig = lambda a, x, b, y: md.sigma2(a, x, b, y, p)          # noqa: E731
    hier = {1: Hierarchy(p, [POS["x"]], f_tensor=md.F_TENSOR,
                         g_tensor=md.G_TENSOR, sigma2=sig),
            2: Hierarchy(p, [POS["x"], POS["y"]], f_tensor=md.F_TENSOR,
                         g_tensor=md.G_TENSOR, sigma2=sig)}
    return props, exps, hier


@pytest.mark.parametrize("n_ext,order,counts,comps", [
    (1, 2, {"F": 1, "G": 1}, (0,)),
    (1, 2, {"F": 1, "G": 1}, (1,)),
    (1, 3, {"F": 3}, (1,)),
    (1, 3, {"F": 1, "G": 2}, (0,)),
    (2, 2, {"F": 2}, (0, 1)),
    (2, 2, {"G": 2}, (0, 1)),
    (2, 2, {"G": 2}, (1, 1)),
])
def test_channel_matches_the_hierarchy(setup, n_ext, order, counts, comps):
    props, exps, hier = setup
    sub = vs.restrict(exps[n_ext], order, counts)
    tg = tag(**counts)
    times = [T] * n_ext
    ext = dict(zip(("x", "y"), times))
    ref = hier[n_ext].moments([(c, i) for i, c in enumerate(comps)], [tg],
                              times)[tg]
    got = sub.evaluate(props, positions=POS, t_final=T, component_pair=comps,
                       orders=[order], external_times=ext,
                       method="gauss_legendre", n_gauss=12).total
    assert got == pytest.approx(ref, rel=1e-10, abs=0.0)


def test_the_quartic_vertex_has_four_legs_and_no_index_symmetry(setup):
    """``G`` enters as ``ψ φ φ φ`` and its tensor is not symmetric, so the
    checks above are not a cubic vertex in disguise."""
    import numpy as np

    props, exps, _hier = setup
    dts = vs.restrict(exps[2], 2, {"G": 2}).diagrams(2)
    for dt in dts:
        assert vs.composition(dt) == {"G": 2}
    G = md.G_TENSOR
    assert not np.allclose(G, np.swapaxes(G, 1, 2))
    assert not np.allclose(G, np.swapaxes(G, 2, 3))


@pytest.mark.slow
def test_order_three_two_point_channels(setup):
    """``F F G`` and ``G G G`` of ``⟨φ_a φ_b⟩``: 68 and 44 diagrams."""
    props, exps, hier = setup
    for counts, comps in [({"F": 2, "G": 1}, (0, 1)), ({"G": 3}, (1, 1))]:
        sub = vs.restrict(exps[2], 3, counts)
        tg = tag(**counts)
        ref = hier[2].moments([(comps[0], 0), (comps[1], 1)], [tg],
                              [T, T])[tg]
        got = sub.evaluate(props, positions=POS, t_final=T,
                           component_pair=comps, orders=[3],
                           external_times={"x": T, "y": T},
                           method="gauss_legendre", n_gauss=10).total
        assert got == pytest.approx(ref, rel=1e-8, abs=0.0), counts
