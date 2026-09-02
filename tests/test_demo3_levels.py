r"""Demo 3 --- the two validation levels, through the package.

**Level A** (``F = 0``) is the exact test.  With no interaction the theory
is a *single* diagram: the ``m`` external ``φ``\ s contracted with the
``m`` ``ψ``\ s of the ``κ^(m)`` vertex, so

    ``⟨φ(z'_1) … φ(z'_m)⟩_c = K_R(z'_1, …, z'_m)``   **exactly**

--- no truncation, no interacting correction, and no other cumulant can
mix in (a ``κ^(m')`` vertex with ``m' ≠ m`` cannot balance the legs).  The
package's answer is therefore a pure test of the non-local-vertex
machinery: enumeration, the MSR ``−i^m/m!`` factor, the response phase,
the R-contraction and the spatial routing.  This is the check demo 2
never had.

**The R-contracted feature** is validated on a *non-constant* kernel by
running the same observable with the raw ``κ^(3)`` vertex, where the
runtime does the three leg integrals itself.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "demo3"))

import shot_noise as sn          # noqa: E402
import system as dsys           # noqa: E402

P = sn.PARAMS
_OBS3 = ("phi_a(x)", "phi_b(y)", "phi_c(z)")
_OBS4 = ("phi_a(w)", "phi_b(x)", "phi_c(y)", "phi_d(z)")


@pytest.fixture(scope="module")
def props():
    system = dsys.make_system(P, cumulants=(3,))
    return system.propagators(t_max=8.0, n_grid_t=40, c_closed_form="auto",
                              c_closed_form_only=True, progress=False)


@pytest.fixture(scope="module")
def exp3():
    return dsys.make_system(P, cumulants=(3,)).expand(_OBS3, orders=[1])


@pytest.fixture(scope="module")
def exp4():
    return dsys.make_system(P, cumulants=(4,)).expand(_OBS4, orders=[1])


def test_propagators_use_the_builtin_closed_form(props):
    """No ``C`` quadrature ever runs: the propagators are machine precision.

    ``κ²`` is ``ExponentialTemporal × CustomKernel``, and the built-in
    closed form only constrains the *temporal* factor --- the
    (non-exponential) ``X₂(r)`` envelope factors straight out.
    """
    assert props.c_source == "closed_form:builtin"


def test_level_a_expansion_is_a_single_diagram(exp3, exp4):
    """With ``F = 0`` there is exactly one diagram at order 1, and nothing
    else can contribute at any order --- which is why level A is exact."""
    assert exp3.summary()[1]["n_diagrams"] == 1
    assert dict(exp3.summary()[1]["by_vertex_type"]) == {"K3": 1}
    assert exp4.summary()[1]["n_diagrams"] == 1
    assert dict(exp4.summary()[1]["by_vertex_type"]) == {"K4": 1}


@pytest.mark.parametrize("t_final,pos", [
    (0.5, (0.0, 0.0, 0.0)),
    (1.5, (0.0, 0.0, 0.0)),
    (5.0, (0.0, 0.0, 0.0)),
    (2.0, (0.0, 0.6, 1.3)),
    (4.0, (-1.0, 0.0, 2.0)),
])
def test_level_a_three_point_is_exact(props, exp3, t_final, pos):
    """``⟨φ³⟩`` from the package equals the closed form to machine precision."""
    result = exp3.evaluate(
        props, positions=dict(zip("xyz", pos)), t_final=t_final,
        component_pair=(0, 0, 0), orders=[1],
        method="gauss_legendre", n_gauss=20)
    exact = float(sn.K_R(np.array(pos, float)[:, None],
                         np.full((3, 1), t_final), P)[0])
    assert result.total == pytest.approx(exact, rel=1e-13)


@pytest.mark.parametrize("t_final", [1.0, 3.0])
def test_level_a_connected_four_point_is_exact(props, exp4, t_final):
    """The connected ``⟨φ⁴⟩`` --- the ``κ⁴`` channel that drives ``ξ_aa``."""
    result = exp4.evaluate(
        props, positions={k: 0.0 for k in "wxyz"}, t_final=t_final,
        component_pair=(0, 0, 0, 0), orders=[1],
        method="gauss_legendre", n_gauss=20)
    exact = float(sn.K_R(np.zeros((4, 1)), np.full((4, 1), t_final), P)[0])
    assert result.total == pytest.approx(exact, rel=1e-13)


def test_level_a_cross_component_vanishes(props, exp3):
    """``κ_m ∝ δ_{a_1…a_m}``: a mixed component triple must give exactly 0."""
    result = exp3.evaluate(
        props, positions={"x": 0.0, "y": 0.0, "z": 0.0}, t_final=1.5,
        component_pair=(0, 0, 1), orders=[1],
        method="gauss_legendre", n_gauss=20)
    assert result.total == 0.0


def test_r_contracted_agrees_with_the_raw_vertex(props):
    """The ``already_R_contracted`` feature on a **non-constant** kernel.

    The raw vertex leaves the runtime a 3-D integral over the causal
    simplex whose integrand kinks on the ``u_i = u_j`` planes (``T_m``
    carries ``t_min``), so tensor Gauss-Legendre converges only at order
    ~2 there; QMC is the honest comparand at this accuracy.  Contracting
    the legs first removes the integral altogether --- the R-contracted
    diagram has *zero* time integration variables.
    """
    raw = dsys.make_system(P, cumulants=(3,), r_contracted=False)
    exp_raw = raw.expand(_OBS3, orders=[1])
    assert len(exp_raw.diagrams(1)[0].analyze_spatial().time_integration_vars) == 3
    got = exp_raw.evaluate(
        props, positions={"x": 0.0, "y": 0.0, "z": 0.0}, t_final=1.5,
        component_pair=(0, 0, 0), orders=[1],
        method="qmc_vectorized", n_samples=2 ** 18, seed=7)
    exact = float(sn.K_R(np.zeros((3, 1)), np.full((3, 1), 1.5), P)[0])
    assert got.total == pytest.approx(exact, rel=2e-4)


def test_r_contracted_diagram_has_no_time_integrals(exp3):
    """Each absorbed leg time aliases onto its partner's, so the level-A
    diagram is evaluated with no quadrature at all."""
    spatial = exp3.diagrams(1)[0].analyze_spatial()
    assert len(spatial.time_integration_vars) == 0
    assert len(spatial.r_absorbed_pairs) == 3


@pytest.mark.xfail(
    reason="package defect (present on base commit ac7f201, independent of "
           "demo 3): repeating a spatial label across external operators "
           "loses pairing multiplicity, silently. With distinct labels at "
           "equal positions the order-1 K3 diagram is correct; with a "
           "repeated label the coupling sum collapses to a single "
           "permutation and the answer comes out 6x too small. Being fixed "
           "by the parallel demo2-hardening session -- this flips to pass "
           "when that lands.",
    strict=True)
def test_coincident_spatial_labels_agree_with_distinct_ones(props):
    """Equal positions must give the same answer however they are spelled."""
    system = dsys.make_system(P, cumulants=(3,))
    distinct = system.expand(_OBS3, orders=[1]).evaluate(
        props, positions={"x": 0.0, "y": 0.0, "z": 0.0}, t_final=1.5,
        component_pair=(0, 0, 0), orders=[1],
        method="gauss_legendre", n_gauss=20)
    repeated = system.expand(("phi_a(x)", "phi_b(x)", "phi_c(x)"),
                             orders=[1]).evaluate(
        props, positions={"x": 0.0}, t_final=1.5,
        component_pair=(0, 0, 0), orders=[1],
        method="gauss_legendre", n_gauss=20)
    assert repeated.total == pytest.approx(distinct.total, rel=1e-12)


def test_distinct_label_spelling_is_the_correct_one(props, exp3):
    """Guard the workaround itself: the *distinct*-label spelling is the one
    that matches the closed form, so demo 3's numbers are unaffected by the
    defect above."""
    result = exp3.evaluate(
        props, positions={"x": 0.0, "y": 0.0, "z": 0.0}, t_final=1.5,
        component_pair=(0, 0, 0), orders=[1],
        method="gauss_legendre", n_gauss=20)
    exact = float(sn.K_R(np.zeros((3, 1)), np.full((3, 1), 1.5), P)[0])
    assert result.total == pytest.approx(exact, rel=1e-13)


def test_single_component_with_callable_coupling_raises():
    """Second known defect on the base commit: ``n_components=1`` plus a
    *callable* coupling crashes in ``_sum_coupling_batched``.  Demo 3 uses
    ``N = 2`` throughout (level B needs it anyway), so this only documents
    the constraint."""
    p1 = sn.ShotNoise(nu=P.nu, h=P.h, sigma_t=P.sigma_t, sigma_x=P.sigma_x,
                      gamma=P.gamma, n_components=1)
    system = dsys.make_system(p1, cumulants=(3,))
    props1 = system.propagators(t_max=3.0, n_grid_t=20, c_closed_form="auto",
                                c_closed_form_only=True, progress=False)
    # a scalar field takes a single (spatial) argument
    expansion = system.expand(("phi(x)", "phi(y)", "phi(z)"), orders=[1])
    with pytest.raises(ValueError, match="axis remapping|more dimensions"):
        expansion.evaluate(props1, positions={"x": 0.0, "y": 0.0, "z": 0.0},
                           t_final=1.0, component_pair=(0, 0, 0), orders=[1],
                           method="gauss_legendre", n_gauss=8)
