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
    assert result.total == pytest.approx(exact, rel=1e-13, abs=0.0)


@pytest.mark.parametrize("t_final", [1.0, 3.0])
def test_level_a_connected_four_point_is_exact(props, exp4, t_final):
    """The connected ``⟨φ⁴⟩`` --- the ``κ⁴`` channel that drives ``ξ_aa``."""
    result = exp4.evaluate(
        props, positions={k: 0.0 for k in "wxyz"}, t_final=t_final,
        component_pair=(0, 0, 0, 0), orders=[1],
        method="gauss_legendre", n_gauss=20)
    exact = float(sn.K_R(np.zeros((4, 1)), np.full((4, 1), t_final), P)[0])
    assert result.total == pytest.approx(exact, rel=1e-13, abs=0.0)


def test_level_a_cross_component_vanishes(props, exp3):
    """``κ_m ∝ δ_{a_1…a_m}``: a mixed component triple must give exactly 0.

    Deliberately an exact comparison, not ``approx``.  The zero is
    *structural*, not a cancellation: the coupling tensor is built with
    ``np.zeros`` and only its diagonal assigned, so every product chain
    contributing to a mixed triple contains an exact ``0.0`` and the sum
    is exactly ``±0.0`` on any IEEE platform.  Loosening this to a
    tolerance would hide a genuine failure of the delta structure.
    """
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
    assert got.total == pytest.approx(exact, rel=2e-4, abs=0.0)


def test_r_contracted_diagram_has_no_time_integrals(exp3):
    """Each absorbed leg time aliases onto its partner's, so the level-A
    diagram is evaluated with no quadrature at all."""
    spatial = exp3.diagrams(1)[0].analyze_spatial()
    assert len(spatial.time_integration_vars) == 0
    assert len(spatial.r_absorbed_pairs) == 3


def test_coincident_spatial_labels_are_not_usable(props):
    """Repeating a spatial label across external operators is a package
    defect, and demo 3 must never rely on that spelling.

    What the collapse loses is not a scalar factor but a *sum over
    external-operator-to-leg assignments*: with distinct labels the
    order-1 K3 coupling sum is ``K_abc + K_acb + K_bac + K_bca + K_cab +
    K_cba``, while the repeated spelling yields ``K_abc`` alone.  Those
    agree up to a factor 6 only when the coupling is symmetric under all
    six index permutations --- true here (``κ_m ∝ δ_{a_1…a_m}``), which is
    exactly why the bug is invisible, but *not* true for a general bare
    tensor.  A blanket multiplicity would therefore be a new silent wrong
    answer, so the parallel demo2-hardening session refuses the spelling
    outright rather than repairing it.

    This test accepts either behaviour, because they are the same
    statement about demo 3: on the base commit ``ac7f201`` the call
    silently returns exactly ``1/6`` of the right answer, and on the
    hardened branch it raises.  Demo 3 uses distinct labels throughout
    (see :func:`level_a.package_npt`), so no published number is affected.
    """
    system = dsys.make_system(P, cumulants=(3,))
    distinct = system.expand(_OBS3, orders=[1]).evaluate(
        props, positions={"x": 0.0, "y": 0.0, "z": 0.0}, t_final=1.5,
        component_pair=(0, 0, 0), orders=[1],
        method="gauss_legendre", n_gauss=20)
    try:
        repeated = system.expand(("phi_a(x)", "phi_b(x)", "phi_c(x)"),
                                 orders=[1]).evaluate(
            props, positions={"x": 0.0}, t_final=1.5,
            component_pair=(0, 0, 0), orders=[1],
            method="gauss_legendre", n_gauss=20)
    except ValueError:
        return                      # hardened branch: the spelling is refused
    assert repeated.total == pytest.approx(distinct.total / 6.0, rel=1e-12, abs=0.0), (
        "expected either a refusal or the known factor-6 collapse")


def test_distinct_label_spelling_is_the_correct_one(props, exp3):
    """Guard the workaround itself: the *distinct*-label spelling is the one
    that matches the closed form, so demo 3's numbers are unaffected by the
    defect above."""
    result = exp3.evaluate(
        props, positions={"x": 0.0, "y": 0.0, "z": 0.0}, t_final=1.5,
        component_pair=(0, 0, 0), orders=[1],
        method="gauss_legendre", n_gauss=20)
    exact = float(sn.K_R(np.zeros((3, 1)), np.full((3, 1), 1.5), P)[0])
    assert result.total == pytest.approx(exact, rel=1e-13, abs=0.0)


def test_single_component_with_callable_coupling(props):
    """``n_components = 1`` with a *callable* coupling.

    On the base commit ``ac7f201`` this raises in
    ``_sum_coupling_batched`` (a ``(1,1,1,1)`` array broadcast against
    ``(n_samples,)``), which is why demo 3 uses ``N = 2`` throughout ---
    level B needs it anyway, so the constraint costs nothing.  The fix
    landed on the ``demo2-hardening`` branch, so this accepts either
    behaviour: refusal is the documented defect, and success must give
    the exact closed form.  No edit needed when demo 3 rebases.
    """
    p1 = sn.ShotNoise(nu=P.nu, h=P.h, sigma_t=P.sigma_t, sigma_x=P.sigma_x,
                      gamma=P.gamma, n_components=1)
    system = dsys.make_system(p1, cumulants=(3,))
    props1 = system.propagators(t_max=3.0, n_grid_t=20, c_closed_form="auto",
                                c_closed_form_only=True, progress=False)
    # a scalar field takes a single (spatial) argument
    expansion = system.expand(("phi(x)", "phi(y)", "phi(z)"), orders=[1])
    try:
        result = expansion.evaluate(
            props1, positions={"x": 0.0, "y": 0.0, "z": 0.0}, t_final=1.0,
            component_pair=(0, 0, 0), orders=[1],
            method="gauss_legendre", n_gauss=8)
    except ValueError as exc:
        assert "axis remapping" in str(exc) or "more dimensions" in str(exc)
        return                      # base commit: the documented defect
    exact = float(sn.K_R(np.zeros((3, 1)), np.full((3, 1), 1.0), p1)[0])
    assert result.total == pytest.approx(exact, rel=1e-13, abs=0.0)


# =====================================================================
# Simulation estimators
# =====================================================================

def test_control_variate_weights_batches_by_size():
    """Unequal batches must be size-weighted, not averaged equally.

    Regression test.  ``simulate_xi`` chunks realisations for memory, and
    a realisation count that does not divide evenly leaves a small
    remainder batch --- ``333333 = 25 x 13333 + 8``.  Averaging per-batch
    means with equal weight gave those eight realisations the same weight
    as thirteen thousand, which moved ``xi_01`` by 20 %.  Every earlier
    run happened to divide evenly, which is why it stayed hidden.
    """
    import simulate_b as sb

    rng = np.random.default_rng(0)
    big, small = rng.normal(1.0, 1.0, 4000), rng.normal(-5.0, 1.0, 8)
    zeros_b, zeros_s = np.zeros_like(big), np.zeros_like(small)
    m_xy = np.array([[big.mean()], [small.mean()]])
    m_z = np.array([[zeros_b.mean()], [zeros_s.mean()]])
    m_xz = np.array([[(big * zeros_b).mean()], [(small * zeros_s).mean()]])
    m_zz = np.array([[(zeros_b ** 2).mean()], [(zeros_s ** 2).mean()]])
    sizes = [big.size, small.size]

    pooled = np.concatenate([big, small]).mean()
    est, err, _ = sb.control_variate_estimate(m_xy, m_z, m_xz, m_zz, sizes)
    assert float(est[0]) == pytest.approx(pooled, rel=1e-12, abs=0.0)

    unweighted, _, _ = sb.control_variate_estimate(m_xy, m_z, m_xz, m_zz)
    assert abs(float(unweighted[0]) - pooled) > 1.0, (
        "the fixture must actually expose the difference")


def test_control_variate_recovers_a_known_mean():
    """With a control variate of known mean zero the estimate is unbiased,
    and the reported variance reduction is real."""
    import simulate_b as sb

    rng = np.random.default_rng(7)
    n_batch, n = 40, 5000
    m_xy, m_z, m_xz, m_zz = [], [], [], []
    for _ in range(n_batch):
        z = rng.normal(0.0, 1.0, n)              # known mean 0
        y = 0.05 + z + rng.normal(0.0, 0.01, n)  # true mean 0.05, tracks z
        m_xy.append([y.mean()]), m_z.append([z.mean()])
        m_xz.append([(y * z).mean()]), m_zz.append([(z * z).mean()])
    est, err, vr = sb.control_variate_estimate(
        np.array(m_xy), np.array(m_z), np.array(m_xz), np.array(m_zz))
    assert float(est[0]) == pytest.approx(0.05, abs=5.0 * float(err[0]))
    assert float(vr[0]) > 10.0, "the control variate must actually reduce variance"


def test_free_field_recursion_matches_a_direct_event_sum():
    """``_draw_free_field``'s geometric recursion is exact, not an
    approximation: compare against summing ``h w(x-x_k) J(t, s_k)`` over
    the same events."""
    import simulate as sim
    import simulate_b as sb

    p = P
    t_edges = np.linspace(0.0, 2.0, 201)
    sites = np.array([0.0, 0.7])
    window = sim.Window.for_times(float(t_edges[-1]), p)
    n_real = 400

    phi = sb._draw_free_field(np.random.default_rng(99), p, sites, window,
                              t_edges, n_real)
    rng2 = np.random.default_rng(99)
    counts = rng2.poisson(p.nu * window.area, n_real)
    total = int(counts.sum())
    xk = rng2.uniform(-window.half_length, window.half_length, total)
    sk = rng2.uniform(window.s_min, window.s_max, total)
    rep = np.repeat(np.arange(n_real), counts)
    for i, x in enumerate(sites):
        for t in (0.5, 1.0, 2.0):
            w = (p.h * np.exp(-np.abs(x - xk) / p.sigma_x)
                 * np.asarray(sn.J(t, sk, p)))
            direct = (np.bincount(rep, weights=w, minlength=n_real)
                      - sim.mean_window([x], [t], window, p, "phi")[0])
            got = phi[:, i, int(round(t / (t_edges[1] - t_edges[0])))]
            assert np.max(np.abs(got - direct)) < 1e-12
