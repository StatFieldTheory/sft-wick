"""Demo 6, part D: ``m = 4`` and ``m = 5`` non-local vertices on every route.

Demo 4 runs ``m = 3`` on every route and ``m = 4`` on the R-contracted one.
Here the 4- and 5-point functions of the free field (``F = 0``, one diagram,
so the package's value is the R-contracted cumulant exactly) are checked on
the R-contracted callable, the raw callable, the static ndarray and the
static ``equal_time`` ndarray, at equal and at distinct external times.

References: Campbell's theorem for demo 4's compound-Poisson noise
(``examples/demo4/poisson_noise.py``), the closed form of demo 6's static
force and white jumps, and both demos' moment hierarchies.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(ROOT / "demo6"))
sys.path.insert(0, str(ROOT / "demo4"))
import poisson_noise as nz  # noqa: E402
import poisson_system as dsys  # noqa: E402
from poisson_reference import Hierarchy as PoissonHierarchy  # noqa: E402
import vertex6_model as md  # noqa: E402
import vertex6_system as vs  # noqa: E402
from vertex6_reference import Hierarchy, solve_multitime, tag  # noqa: E402

LABELS = ("x", "y", "z", "w", "v")
POS = {"x": 0.0, "y": 0.8, "z": -0.5, "w": 1.3, "v": 0.4}
T = 1.7
UNEQUAL = {"x": 1.7, "y": 1.2, "z": 0.6, "w": 1.45, "v": 0.95}
PULSES = {"white": nz.PARAMS_WHITE, "exponential": nz.PARAMS_EXP}
QUADS = [(0, 1, 1, 0), (1, 1, 1, 1), (1, 0, 0, 1)]
QUINTS = [(0, 1, 1, 0, 1), (1, 1, 0, 1, 0)]


def _obs(m):
    labels = LABELS[:m]
    return labels, tuple(f"phi_{c}({lab})" for c, lab in zip("abcde", labels))


def _tuples(m):
    return QUADS if m == 4 else QUINTS


@pytest.mark.parametrize("m", [4, 5])
@pytest.mark.parametrize("pulse", sorted(PULSES))
def test_r_contracted_route_matches_campbell(m, pulse):
    p = PULSES[pulse]
    labels, obs = _obs(m)
    xs = np.array([POS[lab] for lab in labels])
    system = dsys.make_system(p, cumulants=(m,), r_contracted=True)
    props = dsys.propagators_for(system, p, t_max=T + 0.5)
    exp = system.expand(obs, orders=[1], diag_C=False)
    for comps in _tuples(m):
        ref = float(nz.K_R(comps, xs, np.full(m, T), p)[0])
        got = exp.evaluate(props, positions=POS, t_final=T,
                           component_pair=comps, orders=[1],
                           method="gauss_legendre", n_gauss=4).total
        assert got == pytest.approx(ref, rel=1e-11, abs=0.0), comps
        ts = np.array([UNEQUAL[lab] for lab in labels])
        ref_u = float(nz.K_R(comps, xs, ts, p)[0])
        got_u = exp.evaluate(props, positions=POS,
                             t_final=max(UNEQUAL.values()),
                             component_pair=comps, orders=[1],
                             external_times={lab: UNEQUAL[lab]
                                             for lab in labels},
                             method="gauss_legendre", n_gauss=4).total
        assert got_u == pytest.approx(ref_u, rel=1e-11, abs=0.0), comps


@pytest.mark.parametrize("m", [4, 5])
def test_raw_white_route_matches_campbell(m):
    """The raw ``equal_time`` kernel: the package does the leg integral."""
    p = nz.PARAMS_WHITE
    labels, obs = _obs(m)
    xs = np.array([POS[lab] for lab in labels])
    system = dsys.make_system(p, cumulants=(m,), r_contracted=False)
    props = dsys.propagators_for(system, p, t_max=T + 0.5)
    exp = system.expand(obs, orders=[1], diag_C=False)
    for comps in _tuples(m):
        ref = float(nz.K_R(comps, xs, np.full(m, T), p)[0])
        got = exp.evaluate(props, positions=POS, t_final=T,
                           component_pair=comps, orders=[1],
                           method="gauss_legendre", n_gauss=16).total
        assert got == pytest.approx(ref, rel=1e-12, abs=0.0), comps


@pytest.mark.slow
@pytest.mark.parametrize("m", [4, 5])
def test_raw_exponential_route_converges(m):
    """The raw exponential kernel depends on the smallest leg time, so its
    integrand is kinked where two leg times cross; QMC at 2^18 samples
    reaches 1e-3 on the slowest component tuple."""
    p = nz.PARAMS_EXP
    labels, obs = _obs(m)
    xs = np.array([POS[lab] for lab in labels])
    system = dsys.make_system(p, cumulants=(m,), r_contracted=False)
    props = dsys.propagators_for(system, p, t_max=T + 0.5)
    exp = system.expand(obs, orders=[1], diag_C=False)
    for comps in _tuples(m)[:1]:
        ref = float(nz.K_R(comps, xs, np.full(m, T), p)[0])
        got = exp.evaluate(props, positions=POS, t_final=T,
                           component_pair=comps, orders=[1],
                           method="qmc_vectorized", n_samples=2 ** 18,
                           seed=1).total
        assert got == pytest.approx(ref, rel=1e-2, abs=0.0), comps


@pytest.mark.parametrize("m", [4, 5])
@pytest.mark.parametrize("pulse", sorted(PULSES))
def test_poisson_hierarchy_matches_campbell(m, pulse):
    """demo 4's hierarchy at equal times and, through the frozen-variable
    solver, at distinct ones."""
    p = PULSES[pulse]
    labels, _obs_ = _obs(m)
    xs = np.array([POS[lab] for lab in labels])
    H = PoissonHierarchy(p, list(xs))
    tg = (0, m - 2)
    comps = _tuples(m)[0]
    legs = [(c, i) for i, c in enumerate(comps)]
    ref = float(nz.K_R(comps, xs, np.full(m, T), p)[0])
    assert H.moments(legs, [tg], T)[tg] == pytest.approx(ref, rel=1e-11,
                                                         abs=0.0)
    ts = np.array([UNEQUAL[lab] for lab in labels])
    ref_u = float(nz.K_R(comps, xs, ts, p)[0])
    mono = H.monomial(legs)
    freeze = {k: UNEQUAL[labels[i]] for (_a, i), k in H._phi.items()}
    out = solve_multitime(H.sde, [(mono, tg)], freeze, H._initial)
    assert out[(mono, tg)] == pytest.approx(ref_u, rel=1e-11, abs=0.0)


@pytest.mark.parametrize("m", [4, 5])
@pytest.mark.parametrize("kind", ["static", "equal_time"])
def test_static_routes_match_the_closed_form_and_the_hierarchy(m, kind):
    p = md.PARAMS_COMMON
    labels, obs = _obs(m)
    ts = [UNEQUAL[lab] + 0.3 for lab in labels]
    ext = dict(zip(labels, ts))
    system = vs.make_system(p, static=(m,) if kind == "static" else (),
                            equal_time=(m,) if kind == "equal_time" else ())
    props = vs.propagators_for(system, p, t_max=max(ts) + 0.5)
    exp = system.expand(obs, orders=[1], diag_C=False)
    tensors = md.X_CUMULANTS if kind == "static" else md.JUMP_CUMULANTS
    H = Hierarchy(p, [POS[lab] for lab in labels],
                  sigma2=lambda a, x, b, y: md.sigma2(a, x, b, y, p),
                  **({"x_cumulants": {m: tensors[m]}} if kind == "static"
                     else {"jump_cumulants": {m: tensors[m]}}))
    tg = tag(**{f"X{m}" if kind == "static" else f"J{m}": 1})
    for comps in _tuples(m):
        ref = md.split_sum(comps, ts, p, [m],
                           equal_time=(kind == "equal_time"),
                           tensors=tensors)
        got = exp.evaluate(props, positions=POS, t_final=max(ts),
                           component_pair=comps, orders=[1],
                           external_times=ext, method="gauss_legendre",
                           n_gauss=8).total
        assert got == pytest.approx(ref, rel=1e-12, abs=0.0), comps
        hier = H.moments([(c, i) for i, c in enumerate(comps)], [tg], ts)[tg]
        assert hier == pytest.approx(ref, rel=1e-12, abs=0.0), comps


@pytest.mark.parametrize("m", [4, 5])
def test_static_equal_time_on_nquad_with_a_matrix_R(m):
    """Distinct rates make R a diagonal matrix, which only the scalar loops
    take today."""
    p = md.PARAMS
    labels, obs = _obs(m)
    ts = [UNEQUAL[lab] + 0.3 for lab in labels]
    ext = dict(zip(labels, ts))
    system = vs.make_system(p, equal_time=(m,))
    props = vs.propagators_for(system, p, t_max=max(ts) + 0.5)
    exp = system.expand(obs, orders=[1], diag_C=False)
    comps = _tuples(m)[0]
    ref = md.split_sum(comps, ts, p, [m], equal_time=True,
                       tensors=md.JUMP_CUMULANTS)
    got = exp.evaluate(props, positions=POS, t_final=max(ts),
                       component_pair=comps, orders=[1], external_times=ext,
                       method="nquad").total
    assert got == pytest.approx(ref, rel=1e-10, abs=0.0)
