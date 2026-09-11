"""Demo 4 at order 4: the ``F³κ³`` channel of ``⟨φ_a(x) φ_b(y)⟩``.

``examples/demo4/level_b.py`` stops at order 3.  The next channel is three
``F`` vertices and one ``κ³`` (30 diagrams, one C propagator), which demo 2
could only estimate for its own system; the moment hierarchy's coefficient
of ``ε³ μ¹`` is exactly it.  Driven by
``examples/demo4/poisson_level_b_order4.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo4"
sys.path.insert(0, str(DEMO))
import poisson_noise as nz  # noqa: E402
import poisson_level_b_order4 as lb4  # noqa: E402
from poisson_reference import Hierarchy  # noqa: E402

PULSES = {"white": nz.PARAMS_WHITE, "exponential": nz.PARAMS_EXP}
#: n_gauss and tolerance per pulse: white takes the raw equal-time kernel
#: (smooth, exponential convergence), exponential the R-contracted one
#: (kinked where two partner times cross, algebraic).
SETTINGS = {"white": (8, 1e-9), "exponential": (12, 1e-6)}


@pytest.fixture(scope="module", params=sorted(PULSES))
def channel(request):
    p = PULSES[request.param]
    props, exp, n_diagrams = lb4.setup(p)
    return request.param, p, props, exp, n_diagrams


def test_the_channel_is_three_F_and_one_K3(channel):
    """``lb4.setup`` raises unless every diagram of the order-4 ``FK3``
    label is ``F³κ³``; this pins the count as well."""
    _pulse, _p, _props, _exp, n_diagrams = channel
    assert n_diagrams == 30


@pytest.mark.parametrize("ab", [(0, 1), (1, 1)])
def test_order4_channel_matches_the_hierarchy(channel, ab):
    pulse, p, props, exp, _n = channel
    n_gauss, rel = SETTINGS[pulse]
    H = Hierarchy(p, [lb4.POS["x"], lb4.POS["y"]])
    ref = H.moments([(ab[0], 0), (ab[1], 1)], [lb4.TAG], lb4.T)[lb4.TAG]
    got = exp.evaluate(props, positions=lb4.POS, t_final=lb4.T,
                       component_pair=ab, orders=[4],
                       vertex_types={"FK3"}, method="gauss_legendre",
                       n_gauss=n_gauss).total
    assert got == pytest.approx(ref, rel=rel, abs=0.0)
