"""Demo 4: compound-Poisson noise with component-dependent pulses.

The noise cumulants are symmetric only under joint permutations of
(component, point) pairs, the structure that exposed the leg-order defect
of 0.4.2 (demo 4's level A is 47 % off on that code).  Checked here:

* the closed forms of ``examples/demo4/noise.py`` against direct quadrature;
* that the kernel is really asymmetric in points and in components, so the
  package checks below are not vacuous;
* the moment hierarchy (``examples/demo4/reference.py``) against the closed
  forms at ``F = 0``;
* level A: the package's 3- and 4-point functions, R-contracted and raw,
  against the closed form, at distinct points, for every component tuple;
* level B: the package's channels of ``⟨φ_a(x) φ_b(y)⟩`` against the
  hierarchy.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo4"
sys.path.insert(0, str(DEMO))
import noise as nz  # noqa: E402
import system as dsys  # noqa: E402
from reference import Hierarchy  # noqa: E402

T = 1.7
POS = {"x": 0.0, "y": 0.8, "z": -0.5, "w": 1.3}
PULSES = {"white": nz.PARAMS_WHITE, "exponential": nz.PARAMS_EXP}


def _rel(a, b):
    return abs(a - b) / abs(b)


# ---------------------------------------------------------------------------
# The noise model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("comps,ts", [((0, 1), (1.2, 0.7)),
                                      ((1, 0, 1), (1.7, 1.1, 0.4)),
                                      ((0, 0, 1, 1), (0.9, 1.3, 1.7, 0.6))])
def test_t_tilde_closed_form_matches_quadrature(comps, ts):
    p = nz.PARAMS_EXP

    def J(a, t, s):
        r = p.rates[a]
        lo = max(0.0, s)
        if lo >= t:
            return 0.0
        return quad(lambda u: np.exp(-p.gamma * (t - u) - (u - s) * r),
                    lo, t, epsabs=1e-15, epsrel=1e-13)[0]

    ref = quad(lambda s: np.prod([J(a, t, s) for a, t in zip(comps, ts)]),
               -40.0, min(ts), epsabs=1e-16, epsrel=1e-13, limit=200,
               points=[0.0])[0]
    got = nz.t_tilde(comps, np.array(ts, float), p)[0]
    assert got == pytest.approx(ref, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("pulse", sorted(PULSES))
def test_the_kernel_is_asymmetric_in_points_and_in_components(pulse):
    """Permuting the points at fixed components, or the components at fixed
    points, changes K_R by more than 1e-2: a leg-order or routing error
    cannot hide."""
    p = PULSES[pulse]
    xs = np.array([POS["x"], POS["y"], POS["z"]])
    ts = np.array([1.7, 1.2, 0.6])
    base = nz.K_R((0, 1, 1), xs, ts, p)[0]
    points = nz.K_R((0, 1, 1), xs[[2, 0, 1]], ts[[2, 0, 1]], p)[0]
    comps = nz.K_R((1, 1, 0), xs, ts, p)[0]
    assert _rel(points, base) > 1e-2
    assert _rel(comps, base) > 1e-2
    joint = nz.K_R((1, 0, 1), xs[[2, 0, 1]], ts[[2, 0, 1]], p)[0]
    assert joint == pytest.approx(base, rel=1e-13, abs=0.0)


def test_C_is_symmetric_under_swapping_its_ends():
    p = nz.PARAMS_EXP
    C12 = nz.C_matrix(0.0, 1.3, 0.8, 0.9, p)[0]
    C21 = nz.C_matrix(0.8, 0.9, 0.0, 1.3, p)[0]
    assert C12 == pytest.approx(C21.T, rel=1e-13, abs=0.0)


# ---------------------------------------------------------------------------
# The hierarchy against the closed forms (F = 0)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pulse", sorted(PULSES))
def test_hierarchy_reproduces_C_and_the_three_point_function(pulse):
    p = PULSES[pulse]
    H2 = Hierarchy(p, [POS["x"], POS["y"]])
    for a, b in itertools.product(range(2), repeat=2):
        got = H2.moments([(a, 0), (b, 1)], [(0, 0)], T)[(0, 0)]
        ref = nz.C_matrix(POS["x"], T, POS["y"], T, p)[0, a, b]
        assert got == pytest.approx(ref, rel=1e-12, abs=0.0)
    H3 = Hierarchy(p, [POS["x"], POS["y"], POS["z"]])
    xs = np.array([POS["x"], POS["y"], POS["z"]])
    for comps in [(0, 1, 1), (1, 0, 0)]:
        got = H3.moments([(c, i) for i, c in enumerate(comps)], [(0, 1)],
                         T)[(0, 1)]
        ref = nz.K_R(comps, xs, np.full(3, T), p)[0]
        assert got == pytest.approx(ref, rel=1e-12, abs=0.0)


# ---------------------------------------------------------------------------
# Level A: the package at F = 0
# ---------------------------------------------------------------------------

def _level_a(p, m, rc):
    system = dsys.make_system(p, cumulants=(m,), r_contracted=rc)
    props = dsys.propagators_for(system, p, t_max=T + 0.5)
    labels = ("x", "y", "z", "w")[:m]
    obs = tuple(f"phi_{c}({lab})" for c, lab in zip("abcd", labels))
    return props, system.expand(obs, orders=[1], diag_C=False), labels


ROUTES = [("white", True, dict(method="gauss_legendre", n_gauss=8)),
          ("white", False, dict(method="gauss_legendre", n_gauss=16)),
          ("exponential", True, dict(method="gauss_legendre", n_gauss=8))]


@pytest.mark.parametrize("pulse,rc,kw", ROUTES,
                         ids=["white-Rcontracted", "white-raw",
                              "exponential-Rcontracted"])
def test_level_a_three_point_every_triple(pulse, rc, kw):
    p = PULSES[pulse]
    props, exp, labels = _level_a(p, 3, rc)
    xs = np.array([POS[lab] for lab in labels])
    for comps in itertools.product(range(2), repeat=3):
        got = exp.evaluate(props, positions=POS, t_final=T,
                           component_pair=comps, orders=[1], **kw).total
        ref = nz.K_R(comps, xs, np.full(3, T), p)[0]
        assert got == pytest.approx(ref, rel=1e-12, abs=0.0), comps


def test_level_a_raw_exponential_route_converges():
    """The raw exponential kernel is kinked where two leg times cross, so
    the check is QMC at 2^16 samples (2.9e-5 at 2^18 in level_a.py)."""
    p = nz.PARAMS_EXP
    props, exp, labels = _level_a(p, 3, rc=False)
    xs = np.array([POS[lab] for lab in labels])
    for comps in [(0, 1, 1), (1, 0, 0)]:
        got = exp.evaluate(props, positions=POS, t_final=T,
                           component_pair=comps, orders=[1],
                           method="qmc_vectorized", n_samples=2 ** 16,
                           seed=1).total
        ref = nz.K_R(comps, xs, np.full(3, T), p)[0]
        assert got == pytest.approx(ref, rel=1e-3, abs=0.0), comps


@pytest.mark.parametrize("pulse", sorted(PULSES))
def test_level_a_unequal_external_times(pulse):
    p = PULSES[pulse]
    props, exp, labels = _level_a(p, 3, rc=True)
    times = {"x": 1.7, "y": 1.2, "z": 0.6}
    xs = np.array([POS[lab] for lab in labels])
    ts = np.array([times[lab] for lab in labels])
    for comps in [(0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        got = exp.evaluate(props, positions=POS, t_final=1.7,
                           component_pair=comps, orders=[1],
                           external_times=times, method="gauss_legendre",
                           n_gauss=8).total
        ref = nz.K_R(comps, xs, ts, p)[0]
        assert got == pytest.approx(ref, rel=1e-12, abs=0.0), comps


@pytest.mark.parametrize("pulse", sorted(PULSES))
def test_level_a_connected_four_point(pulse):
    p = PULSES[pulse]
    props, exp, labels = _level_a(p, 4, rc=True)
    xs = np.array([POS[lab] for lab in labels])
    H = Hierarchy(p, list(xs))
    for comps in [(0, 1, 1, 0), (1, 1, 1, 1)]:
        got = exp.evaluate(props, positions=POS, t_final=T,
                           component_pair=comps, orders=[1],
                           method="gauss_legendre", n_gauss=8).total
        ref = nz.K_R(comps, xs, np.full(4, T), p)[0]
        hier = H.moments([(c, i) for i, c in enumerate(comps)], [(0, 2)],
                         T)[(0, 2)]
        assert got == pytest.approx(ref, rel=1e-12, abs=0.0)
        assert hier == pytest.approx(ref, rel=1e-11, abs=0.0)


# ---------------------------------------------------------------------------
# Level B: the package at F ≠ 0 against the hierarchy
# ---------------------------------------------------------------------------

CHANNELS = [("FK3", 2, "FK3", (1, 1)), ("FF", 2, "F", (2, 0)),
            ("FFK4", 3, "FK4", (2, 2))]


@pytest.fixture(scope="module")
def level_b():
    out = {}
    for pulse, p in PULSES.items():
        # The smooth route per pulse (see examples/demo4/level_b.py).
        system = dsys.make_system(p, f_amplitude=1.0, cumulants=(3, 4),
                                  r_contracted=(pulse == "exponential"))
        props = dsys.propagators_for(system, p, t_max=T + 0.5)
        exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2, 3],
                            diag_C=False)
        out[pulse] = (props, exp, Hierarchy(p, [POS["x"], POS["y"]]))
    return out


@pytest.mark.parametrize("pulse", sorted(PULSES))
@pytest.mark.parametrize("name,order,vtype,tag", CHANNELS,
                         ids=[c[0] for c in CHANNELS])
def test_level_b_channels_match_the_hierarchy(level_b, pulse, name, order,
                                              vtype, tag):
    props, exp, H = level_b[pulse]
    for ab in [(0, 1), (1, 1)]:
        ref = H.moments([(ab[0], 0), (ab[1], 1)], [tag], T)[tag]
        got = exp.evaluate(props, positions=POS, t_final=T,
                           component_pair=ab, orders=[order],
                           vertex_types={vtype}, method="gauss_legendre",
                           n_gauss=16).total
        assert got == pytest.approx(ref, rel=1e-7, abs=0.0), (name, ab)
