"""Demo 6: repeated and static non-local vertices, and an ``m = 2`` vertex.

``examples/demo6`` drives the field with a force that is constant in space
and time (a static ndarray coupling, one vertex per cumulant order) or with
white jumps whose jump vector does not depend on the point (a static
``equal_time`` coupling).  Both were unevaluated: no test had two copies of
one non-local vertex, and none had an ``m = 2`` one beyond its construction.

Checked here (all at distinct points and distinct times, ``t_min = 0.4``,
every component tuple where it is cheap):

* the order-2 six-point function with two copies of one cubic vertex,
  against the closed form (the sum over the ten splits of the six points)
  and the moment hierarchy;
* ``F X3`` and ``F F X4`` channels of ``⟨φ_a φ_b⟩`` against the hierarchy;
* the order-1 contribution of an ``m = 2`` vertex, which is the C of that
  noise exactly, and its ``F K2`` channel.

The equal-time six-point case fails on 7034888 (see
``tests/test_equal_time_nonlocal.py``); the rest passes there.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo6"
sys.path.insert(0, str(DEMO))
import vertex6_model as md  # noqa: E402
import vertex6_system as vs  # noqa: E402
from vertex6_reference import Hierarchy, tag  # noqa: E402

LABELS = ("x1", "x2", "x3", "x4", "x5", "x6")
POS6 = dict(zip(LABELS, (0.0, 0.7, -0.4, 1.1, -0.9, 0.3)))
TIMES6 = dict(zip(LABELS, (1.9, 1.4, 2.3, 0.9, 1.6, 2.1)))
POS2 = {"x": 0.0, "y": 0.8}
TIMES2 = {"x": 1.9, "y": 1.3}
EQUAL2 = {"x": 1.9, "y": 1.9}
TRIPLES6 = [(0, 1, 1, 0, 1, 0), (1, 1, 0, 0, 0, 1)]


def _sigma2(p):
    return lambda a, x, b, y: md.sigma2(a, x, b, y, p)


# ---------------------------------------------------------------------------
# Two copies of one vertex: the six-point function at order 2
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", params=["static", "equal_time"])
def two_copies(request):
    kind = request.param
    p = md.PARAMS_COMMON
    system = vs.make_system(p, static=(3,) if kind == "static" else (),
                            equal_time=(3,) if kind == "equal_time" else ())
    props = vs.propagators_for(system, p, t_max=max(TIMES6.values()) + 0.5)
    obs = tuple(f"phi_{c}({lab})" for c, lab in zip("abcdef", LABELS))
    exp = system.expand(obs, orders=[2], diag_C=False)
    return kind, p, props, exp


def _closed_form_six(kind, p, comps):
    tensors = md.X_CUMULANTS if kind == "static" else md.JUMP_CUMULANTS
    return md.split_sum(comps, [TIMES6[lab] for lab in LABELS], p, [3, 3],
                        equal_time=(kind == "equal_time"), tensors=tensors)


@pytest.mark.parametrize("comps", TRIPLES6)
def test_two_copies_six_point_matches_the_closed_form(two_copies, comps):
    kind, p, props, exp = two_copies
    n_gauss = 8 if kind == "static" else 12
    got = exp.evaluate(props, positions=POS6, t_final=max(TIMES6.values()),
                       component_pair=comps, orders=[2],
                       external_times=TIMES6, method="gauss_legendre",
                       n_gauss=n_gauss).total
    assert got == pytest.approx(_closed_form_six(kind, p, comps), rel=1e-12,
                                abs=0.0)


def test_two_copies_six_point_matches_the_hierarchy(two_copies):
    kind, p, props, exp = two_copies
    tg = tag(**{"X3" if kind == "static" else "J3": 2})
    kw = ({"x_cumulants": {3: md.X_CUMULANTS[3]}} if kind == "static"
          else {"jump_cumulants": {3: md.JUMP_CUMULANTS[3]}})
    H = Hierarchy(p, [POS6[lab] for lab in LABELS], sigma2=_sigma2(p), **kw)
    comps = TRIPLES6[0]
    ref = H.moments([(c, i) for i, c in enumerate(comps)], [tg],
                    [TIMES6[lab] for lab in LABELS])[tg]
    assert ref == pytest.approx(_closed_form_six(kind, p, comps), rel=1e-12,
                                abs=0.0)
    got = exp.evaluate(props, positions=POS6, t_final=max(TIMES6.values()),
                       component_pair=comps, orders=[2],
                       external_times=TIMES6, method="gauss_legendre",
                       n_gauss=8 if kind == "static" else 12).total
    assert got == pytest.approx(ref, rel=1e-12, abs=0.0)


def test_two_equal_time_copies_on_nquad_and_a_matrix_R():
    """``nquad`` and a diagonal matrix R (distinct rates), which
    Gauss-Legendre and ``qmc_vectorized`` refuse today."""
    p = md.PARAMS
    system = vs.make_system(p, equal_time=(3,))
    props = vs.propagators_for(system, p, t_max=max(TIMES6.values()) + 0.5)
    obs = tuple(f"phi_{c}({lab})" for c, lab in zip("abcdef", LABELS))
    exp = system.expand(obs, orders=[2], diag_C=False)
    comps = TRIPLES6[0]
    got = exp.evaluate(props, positions=POS6, t_final=max(TIMES6.values()),
                       component_pair=comps, orders=[2],
                       external_times=TIMES6, method="nquad").total
    assert got == pytest.approx(_closed_form_six("equal_time", p, comps),
                                rel=1e-10, abs=0.0)


# ---------------------------------------------------------------------------
# F with static cumulants
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def interacting():
    p = md.PARAMS_COMMON
    system = vs.make_system(p, f_amplitude=1.0, static=(3, 4))
    props = vs.propagators_for(system, p, t_max=max(TIMES2.values()) + 0.5)
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2, 3],
                        diag_C=False)
    H = Hierarchy(p, [POS2["x"], POS2["y"]], f_tensor=md.F_TENSOR,
                  sigma2=_sigma2(p),
                  x_cumulants={3: md.X_CUMULANTS[3], 4: md.X_CUMULANTS[4]})
    return p, props, exp, H


@pytest.mark.parametrize("order,counts,times,rel", [
    (0, {}, TIMES2, 1e-12),
    (2, {"F": 1, "X3": 1}, TIMES2, 1e-12),
    (3, {"F": 2, "X4": 1}, TIMES2, 1e-12),
    (2, {"F": 2}, EQUAL2, 1e-10),
])
def test_channels_match_the_hierarchy(interacting, order, counts, times, rel):
    _p, props, exp, H = interacting
    sub = exp if order == 0 else vs.restrict(exp, order, counts)
    tg = tag(**counts)
    for ab in [(0, 1), (1, 1)]:
        ref = H.moments([(ab[0], 0), (ab[1], 1)], [tg],
                        [times["x"], times["y"]])[tg]
        got = sub.evaluate(props, positions=POS2, t_final=max(times.values()),
                           component_pair=ab, orders=[order],
                           external_times=times, method="gauss_legendre",
                           n_gauss=8).total
        assert got == pytest.approx(ref, rel=rel, abs=0.0), (counts, ab)


def test_the_channels_are_the_compositions_they_claim(interacting):
    _p, _props, exp, _H = interacting
    assert vs.channel(exp, 2, "FX3") == {"F": 1, "X3": 1}
    assert vs.channel(exp, 3, "FX4") == {"F": 2, "X4": 1}
    assert vs.channel(exp, 2, "F") == {"F": 2}


@pytest.mark.slow
@pytest.mark.parametrize("kind,name,method,kw,rel", [
    ("static", "X3", "gauss_legendre", {"n_gauss": 8}, 1e-12),
    ("equal_time", "J3", "qmc_vectorized",
     {"n_samples": 2 ** 18, "seed": 1}, 1e-3),
])
def test_two_copies_with_an_interaction(kind, name, method, kw, rel):
    """``F X3 X3`` and ``F J3 J3`` of the five-point function at order 3 —
    two copies of one vertex with an interaction.  Slow: the order-3
    expansion of a five-point observable takes about a minute."""
    labels = ("x", "y", "z", "w", "v")
    pos = {"x": 0.0, "y": 0.8, "z": -0.5, "w": 1.2, "v": 0.3}
    times = {"x": 1.9, "y": 1.3, "z": 2.2, "w": 1.6, "v": 2.0}
    comps = (0, 1, 1, 0, 1)
    p = md.PARAMS_COMMON
    system = vs.make_system(p, f_amplitude=1.0,
                            static=(3,) if kind == "static" else (),
                            equal_time=(3,) if kind == "equal_time" else ())
    props = vs.propagators_for(system, p, t_max=max(times.values()) + 0.5)
    obs = tuple(f"phi_{c}({lab})" for c, lab in zip("abcde", labels))
    sub = vs.restrict(system.expand(obs, orders=[3], diag_C=False), 3,
                      {"F": 1, name: 2})
    H = Hierarchy(p, [pos[k] for k in labels], f_tensor=md.F_TENSOR,
                  sigma2=_sigma2(p),
                  **({"x_cumulants": {3: md.X_CUMULANTS[3]}}
                     if kind == "static"
                     else {"jump_cumulants": {3: md.JUMP_CUMULANTS[3]}}))
    tg = tag(**{name: 2, "F": 1})
    ref = H.moments([(c, i) for i, c in enumerate(comps)], [tg],
                    [times[k] for k in labels])[tg]
    got = sub.evaluate(props, positions=pos, t_final=max(times.values()),
                       component_pair=comps, orders=[3], external_times=times,
                       method=method, **kw).total
    assert got == pytest.approx(ref, rel=rel, abs=0.0)


# ---------------------------------------------------------------------------
# The m = 2 vertex
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", ["X2", "W2"])
@pytest.mark.parametrize("method,kw,rel", [
    ("gauss_legendre", {"n_gauss": 8}, 1e-12),
    ("nquad", {}, 1e-12),
])
def test_order_one_m2_vertex_is_the_C_of_that_noise(label, method, kw, rel):
    p = md.PARAMS_COMMON
    system = vs.make_system(p, static=(2,), equal_time=(2,))
    props = vs.propagators_for(system, p, t_max=max(TIMES2.values()) + 0.5)
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[1], diag_C=False)
    for ab in itertools.product(range(2), repeat=2):
        if label == "X2":
            ref = (md.X_CUMULANTS[2][ab] * md.g_factor(ab[0], TIMES2["x"], p)
                   * md.g_factor(ab[1], TIMES2["y"], p))
        else:
            ref = (md.JUMP_CUMULANTS[2][ab]
                   * md.equal_time_block(ab, [TIMES2["x"], TIMES2["y"]], p))
        got = exp.evaluate(props, positions=POS2, t_final=max(TIMES2.values()),
                           component_pair=ab, orders=[1],
                           vertex_types={label}, external_times=TIMES2,
                           method=method, **kw).total
        assert got == pytest.approx(float(ref), rel=rel, abs=0.0), (label, ab)


@pytest.mark.parametrize("counts", [{"F": 1, "X2": 1}, {"F": 1, "W2": 1}])
def test_the_m2_vertex_channel_with_F_matches_the_hierarchy(counts):
    p = md.PARAMS_COMMON
    system = vs.make_system(p, f_amplitude=1.0, static=(2,), equal_time=(2,))
    props = vs.propagators_for(system, p, t_max=TIMES2["x"] + 0.5)
    exp = system.expand(("phi_a(x)",), orders=[2], diag_C=False)
    sub = vs.restrict(exp, 2, counts)
    H = Hierarchy(p, [POS2["x"]], f_tensor=md.F_TENSOR, sigma2=_sigma2(p),
                  x_cumulants={2: md.X_CUMULANTS[2]},
                  white_vertex=md.JUMP_CUMULANTS[2])
    tg = tag(**counts)
    for a in range(2):
        ref = H.moments([(a, 0)], [tg], [TIMES2["x"]])[tg]
        got = sub.evaluate(props, positions=POS2, t_final=TIMES2["x"],
                           component_pair=(a,), orders=[2],
                           external_times={"x": TIMES2["x"]},
                           method="gauss_legendre", n_gauss=12).total
        assert got == pytest.approx(ref, rel=1e-11, abs=0.0), (counts, a)
