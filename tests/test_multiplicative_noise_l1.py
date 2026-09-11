"""Multiplicative white noise at L1 and the Stratonovich interpretation.

MN1  the spec: ``D(φ) = g(φ) g(φ)ᵀ`` and the noise-induced drift against a
     direct evaluation and a finite difference of ``g``; the shape, name and
     interpretation guards.
MN2  the lowering: field content of the vertices, the MSR factors in
     ``build_coupling_values``, ``D0`` in C, the closed form.
MN3  ``⟨φ_a⟩`` and ``⟨φ_a φ_b⟩`` per bookkeeping tag (weight ≤ 3, i.e. vertex
     orders 0-3) against the Hörmander-form hierarchy of demo 5 part C,
     Itô and Stratonovich, on ``gauss_legendre`` with a scalar R.
MN4  the same with a matrix R (distinct rates) on ``qmc_scalar`` and
     ``nquad``.
MN5  two-time ``⟨φ_a(t₁) φ_b(t₂)⟩``.
MN6  Itô and Stratonovich differ, in the reference and in the package: MN3
     is not vacuous.
MN7  external points at different positions are refused.
MN8  the YAML route (``sigma2.type: multiplicative``) equals the L1 route.
MN9  ``ito=False``: an equal-point R on a vertex with two ψ legs, or between
     two external operators, is refused instead of evaluated as the Itô
     value; on a vertex with one ψ leg it still gives the ``ito=True``
     number on every integrator.

The reference (``examples/demo5/white_hormander_reference.py`` on
``examples/reference/hormander_moments.py``) shares no code with the
package and never forms the noise-induced drift.
"""
from __future__ import annotations

import itertools
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.evaluate import integrate_diagrams

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo5"
sys.path.insert(0, str(DEMO))
import white_l1_multiplicative as m5c  # noqa: E402

TAGS3 = m5c.all_tags(3)
ONE, TWO = m5c.ONE, m5c.TWO


# --------------------------------------------------------------------------
# MN1 -- the spec
# --------------------------------------------------------------------------

def _g(phi):
    return m5c.G0 + np.einsum("akb,b->ak", m5c.G1, phi)


@pytest.mark.parametrize("seed", [0, 1])
def test_MN1_D_is_g_g_transpose(seed):
    """D0 + D1·φ + D2·φφ is g(φ) g(φ)ᵀ at a random φ."""
    imp = sw.MultiplicativeImpulse(g0=m5c.G0, g1=m5c.G1)
    phi = np.random.default_rng(seed).normal(size=m5c.N)
    got = (imp.amplitude + np.einsum("abc,c->ab", imp.D1, phi)
           + np.einsum("abce,c,e->ab", imp.D2, phi, phi))
    want = _g(phi) @ _g(phi).T
    assert np.allclose(got, want, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("seed", [0, 1])
def test_MN1_stratonovich_drift_is_the_vector_field_derivative(seed):
    """b + Lφ = ½ Σ_jk g_jk ∂_j g_ik, by central differences of g."""
    imp = sw.MultiplicativeImpulse(g0=m5c.G0, g1=m5c.G1,
                                   interpretation="stratonovich")
    b, L = imp.noise_induced_drift
    phi = np.random.default_rng(seed).normal(size=m5c.N)
    h = 1e-5
    drift = np.zeros(m5c.N)
    for j in range(m5c.N):
        step = np.zeros(m5c.N)
        step[j] = h
        dg = (_g(phi + step) - _g(phi - step)) / (2 * h)     # ∂_j g_ik
        drift += 0.5 * dg @ _g(phi)[j]
    assert np.allclose(b + L @ phi, drift, rtol=1e-8, atol=1e-10)
    assert np.allclose(
        sw.MultiplicativeImpulse(g0=m5c.G0, g1=m5c.G1).noise_induced_drift[0],
        0.0)


@pytest.mark.parametrize("kwargs, match", [
    (dict(g0=m5c.G0, g1=m5c.G1[:, :, :1]), "shape"),
    (dict(g0=m5c.G0[0], g1=m5c.G1), "shape"),
    (dict(g0=m5c.G0, g1=m5c.G1, interpretation="strat"), "interpretation"),
    (dict(g0=m5c.G0, g1=m5c.G1, vertex_names=("G", "G")), "distinct"),
    (dict(g0=m5c.G0, g1=m5c.G1, drift_names=("B", "G")), "distinct"),
    (dict(g0=m5c.G0 * 1j, g1=m5c.G1), "real"),
])
def test_MN1_rejects_malformed_specs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sw.MultiplicativeImpulse(**kwargs)


def test_MN1_system_rejects_a_component_mismatch_and_a_name_clash():
    imp = sw.MultiplicativeImpulse(g0=m5c.G0, g1=m5c.G1)
    with pytest.raises(ValueError, match="components"):
        sw.System(field=sw.FieldSpec("phi", 3),
                  linear=sw.DiagonalA(gamma=[1.0] * 3),
                  noise=sw.GaussianNoise(kappa2=_KAPPA2, sigma2=imp))
    with pytest.raises(ValueError, match="both"):
        sw.System(field=sw.FieldSpec("phi", 2),
                  linear=sw.DiagonalA(gamma=[1.0, 1.0]),
                  vertices=[sw.LocalVertex("G", coupling=m5c.F_TENSOR)],
                  noise=sw.GaussianNoise(kappa2=_KAPPA2, sigma2=imp))


_KAPPA2 = sw.SeparableTranslation(
    temporal=sw.ExponentialTemporal(lam=0.3, sigma_t=0.7),
    spatial=sw.GaussianSpatial(sigma_x=0.9))


# --------------------------------------------------------------------------
# MN2 -- the lowering
# --------------------------------------------------------------------------

def _field_pattern(vertex):
    return "".join("P" if f.is_physical else "R" for f in vertex.fields)


@pytest.mark.parametrize("interpretation, want", [
    ("ito", {"F": "RPP", "G": "RRP", "H": "RRPP"}),
    ("stratonovich", {"F": "RPP", "G": "RRP", "H": "RRPP", "B": "R",
                      "L": "RP"}),
])
def test_MN2_vertex_field_content(interpretation, want):
    system = m5c.make_system(m5c.GAMMAS["scalar"], interpretation)
    got = {v.coupling: _field_pattern(v) for v in system.build_action().vertices}
    assert got == want


def test_MN2_msr_factors():
    """½ = −i²/2! on the two-ψ vertices of D(φ), −i on the drift vertices."""
    system = m5c.make_system(m5c.GAMMAS["scalar"], "stratonovich")
    imp = system.noise.sigma2
    b, L = imp.noise_induced_drift
    cv = system.build_coupling_values()
    assert np.allclose(cv["G"], 0.5 * imp.D1)
    assert np.allclose(cv["H"], 0.5 * imp.D2)
    assert np.allclose(cv["B"], -1j * b)
    assert np.allclose(cv["L"], -1j * L)
    assert np.allclose(cv["F"], -1j * m5c.F_TENSOR)
    assert set(m5c.make_system(m5c.GAMMAS["scalar"], "ito")
               .build_coupling_values()) == {"F", "G", "H"}


def test_MN2_D0_enters_C_and_the_closed_form_applies():
    system = m5c.make_system(m5c.GAMMAS["scalar"], "ito")
    model = system.build_propagator_model(diag_C=False)
    assert np.allclose(model.sigma2(0.0, 1.0, 0.0),
                       m5c.G0 @ m5c.G0.T)
    props = m5c.propagators_for(system)
    assert props.c_source == "closed_form:builtin"


def test_MN2_a_zero_g1_generates_no_vertices():
    imp = sw.MultiplicativeImpulse(g0=m5c.G0, g1=np.zeros_like(m5c.G1),
                                   interpretation="stratonovich")
    assert imp.vertices() == ()


# --------------------------------------------------------------------------
# MN3 / MN5 / MN6 -- against the exact hierarchy
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scalar_r():
    """``{interpretation: (system, props, hierarchy, {obs: expansions})}``."""
    out = {}
    for interpretation in m5c.INTERPRETATIONS:
        system = m5c.make_system(m5c.GAMMAS["scalar"], interpretation)
        out[interpretation] = (
            system, m5c.propagators_for(system),
            m5c.hierarchy(m5c.GAMMAS["scalar"], interpretation),
            {obs: m5c.expansions(system, obs, 3) for obs in (ONE, TWO)},
        )
    return out


def _check(got, ref, tags, rel=1e-9, floor=1e-13):
    for tag in tags:
        p, h = float(got.get(tag, 0.0)), ref[tag]
        if h == 0.0:
            assert abs(p) < floor, (tag, p)
        else:
            assert p == pytest.approx(h, rel=rel, abs=0.0), (tag, p, h)


@pytest.mark.parametrize("interpretation", m5c.INTERPRETATIONS)
@pytest.mark.parametrize("obs", [ONE, TWO])
def test_MN3_gauss_legendre_matches_the_hierarchy(scalar_r, interpretation,
                                                  obs):
    system, props, H, exps = scalar_r[interpretation]
    for comps in itertools.product(range(m5c.N), repeat=len(obs)):
        got = m5c.package_by_tag(system, props, exps[obs], comps,
                                 "gauss_legendre", dict(n_gauss=12), 3)
        _check(got, H.moments(comps, TAGS3, m5c.T), TAGS3)


@pytest.mark.parametrize("interpretation", m5c.INTERPRETATIONS)
def test_MN5_two_time_matches_the_hierarchy(scalar_r, interpretation):
    """Distinct external times at one point.

    On ``qmc_vectorized`` here and on ``nquad`` in the slow test below.
    Gauss-Legendre converges algebraically on this integrand, because an
    internal time crosses the earlier external time inside the domain and
    C is kinked on its diagonal -- a kink the GL splitter does not split
    (it pairs internal variables).  Measured on the FG channel of
    ``⟨φ_0(t₁) φ_1(t₂)⟩``: 2.0e-3 at 12 nodes, 1.8e-4 at 48, 5.4e-5 at 80,
    against 1.2e-8 for ``nquad`` and 1.5e-7 for ``qmc_vectorized`` at 2^18.
    The same holds for the FF channel, so it is a property of two external
    times with white noise, not of the noise vertices."""
    _two_time(scalar_r[interpretation], "qmc_vectorized",
              dict(n_samples=2 ** 15, seed=3), rel=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize("interpretation", m5c.INTERPRETATIONS)
def test_MN5_two_time_on_nquad(scalar_r, interpretation):
    _two_time(scalar_r[interpretation], "nquad", {}, rel=1e-6)


def _two_time(fixture, method, kw, rel):
    system, props, H, exps = fixture
    for comps in [(0, 1), (1, 1)]:
        got = m5c.package_by_tag(
            system, props, exps[TWO], comps, method, kw, 3,
            external_times={"x": m5c.T_MIN + m5c.T1, "y": m5c.T_MIN + m5c.T})
        _check(got, H.two_time(comps[0], comps[1], TAGS3, m5c.T1, m5c.T),
               TAGS3, rel=rel, floor=1e-9)


def test_MN6_ito_and_stratonovich_differ(scalar_r):
    """MN3 would pass with the interpretations swapped only if they agreed."""
    refs, pkgs = {}, {}
    for interpretation in m5c.INTERPRETATIONS:
        system, props, H, exps = scalar_r[interpretation]
        refs[interpretation] = H.moments((0, 0), TAGS3, m5c.T)
        pkgs[interpretation] = m5c.package_by_tag(
            system, props, exps[TWO], (0, 0), "gauss_legendre",
            dict(n_gauss=12), 3)
    tag = (0, 2)      # ⟨φ_0 φ_0⟩ at g1²: H alone, or H + L + GG + GB + BB
    for table in (refs, pkgs):
        i, s = table["ito"][tag], table["stratonovich"][tag]
        assert abs(s - i) / abs(s) > 0.5, (i, s)
    mean = (1,)       # ⟨φ_0⟩ at g1: zero under Itô, a source under Stratonovich
    i_sys, i_props, i_H, i_exps = scalar_r["ito"]
    s_sys, s_props, s_H, s_exps = scalar_r["stratonovich"]
    assert i_H.moments(mean, [(0, 1)], m5c.T)[(0, 1)] == 0.0
    assert abs(s_H.moments(mean, [(0, 1)], m5c.T)[(0, 1)]) > 1e-2


# --------------------------------------------------------------------------
# MN4 -- matrix R on the scalar loops
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def matrix_r():
    system = m5c.make_system(m5c.GAMMAS["matrix"], "stratonovich")
    return (system, m5c.propagators_for(system),
            m5c.hierarchy(m5c.GAMMAS["matrix"], "stratonovich"),
            {obs: m5c.expansions(system, obs, 2) for obs in (ONE, TWO)})


@pytest.mark.parametrize("method, kw, weight, rel", [
    ("qmc_scalar", dict(n_samples=2 ** 11, seed=2), 2, 5e-3),
    ("nquad", {}, 1, 1e-8),
    ("gauss_legendre", dict(n_gauss=12), 2, 1e-8),
])
def test_MN4_matrix_r_matches_the_hierarchy(matrix_r, method, kw, weight, rel):
    system, props, H, exps = matrix_r
    tags = m5c.all_tags(weight)
    for obs, comps in [(ONE, (1,)), (TWO, (0, 1))]:
        got = m5c.package_by_tag(system, props, exps[obs], comps, method, kw,
                                 weight)
        _check(got, H.moments(comps, tags, m5c.T), tags, rel=rel, floor=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize("interpretation", m5c.INTERPRETATIONS)
@pytest.mark.parametrize("obs, comps, weight", [
    (("phi_a(x)", "phi_b(y)", "phi_c(z)"), (0, 1, 1), 3),
    (TWO, (0, 1), 4),
])
def test_MN10_vertex_orders_3_and_4(interpretation, obs, comps, weight):
    """Vertex orders 3 (three-point function, 864 diagrams) and 4 (two-point,
    1124 diagrams): the tags of odd weight vanish for ``⟨φ φ⟩`` and of even
    weight for ``⟨φ⟩`` and ``⟨φ φ φ⟩``, so each needs its own observable."""
    system = m5c.make_system(m5c.GAMMAS["scalar"], interpretation)
    props = m5c.propagators_for(system)
    H = m5c.hierarchy(m5c.GAMMAS["scalar"], interpretation)
    exps = m5c.expansions(system, obs, weight)
    tags = m5c.all_tags(weight)
    got = m5c.package_by_tag(system, props, exps, comps, "gauss_legendre",
                             dict(n_gauss=12), weight)
    _check(got, H.moments(comps, tags, m5c.T), tags)


# --------------------------------------------------------------------------
# MN7 -- one SDE per point
# --------------------------------------------------------------------------

def test_MN7_external_points_at_different_positions_are_refused(scalar_r):
    system, props, _H, exps = scalar_r["ito"]
    with pytest.raises(ValueError, match="different positions"):
        exps[TWO][1].evaluate(props, positions={"x": 0.0, "y": 0.7},
                              t_final=m5c.T_MIN + m5c.T,
                              component_pair=(0, 1), orders=[1],
                              method="gauss_legendre", n_gauss=8)
    # The same points at one position are fine.
    exps[TWO][1].evaluate(props, positions=m5c.POS,
                          t_final=m5c.T_MIN + m5c.T, component_pair=(0, 1),
                          orders=[1], method="gauss_legendre", n_gauss=8)


def test_MN7_additive_noise_is_not_refused():
    """The guard is about the local two-ψ vertices, not about positions."""
    system = sw.System(
        field=sw.FieldSpec("phi", 2), linear=sw.DiagonalA(gamma=[1.1, 1.1]),
        vertices=[sw.LocalVertex("F", coupling=m5c.F_TENSOR)],
        noise=sw.GaussianNoise(
            kappa2=_KAPPA2,
            sigma2=sw.ConstantImpulse(m5c.G0 @ m5c.G0.T)),
        t_min=m5c.T_MIN)
    props = m5c.propagators_for(system)
    exp = system.expand(TWO, orders=[2], diag_C=False)
    val = exp.evaluate(props, positions={"x": 0.0, "y": 0.7},
                       t_final=m5c.T_MIN + m5c.T, component_pair=(0, 1),
                       orders=[2], method="gauss_legendre", n_gauss=8).total
    assert np.isfinite(val)


# --------------------------------------------------------------------------
# MN8 -- YAML
# --------------------------------------------------------------------------

_YAML = textwrap.dedent("""
    system:
      field: {{name: phi, n_components: 2}}
      linear: {{type: diagonal, gamma: [1.1, 1.1]}}
      t_min: {t_min}
      vertices:
        - name: F
          coupling: {F}
      noise:
        kappa2:
          type: separable_translation
          temporal: {{type: exponential, lam: 0.3, sigma_t: 0.7}}
          spatial:  {{type: gaussian, sigma_x: 0.9}}
        sigma2:
          type: multiplicative
          interpretation: {interpretation}
          g0: {g0}
          g1: {g1}

    expand:
      observable: ["phi_a(x)", "phi_b(y)"]
      orders: [0, 1, 2]

    propagators:
      t_max: {t_max}
      n_grid_t: 12
      c_closed_form: auto
      c_closed_form_only: true
      c_closed_form_vectorized: true
      diag_C: false

    sweep:
      positions_grid: {{x: [0.0], y: [0.0]}}
      t_final_grid: [{t_final}]
      component_pairs: [[0, 1]]
      method: gauss_legendre
      n_gauss: 12
    """)


@pytest.mark.parametrize("interpretation", m5c.INTERPRETATIONS)
def test_MN8_yaml_route_equals_the_l1_route(tmp_path, interpretation):
    from sft_wick.workflow.config import load_workflow_config, run_workflow

    path = tmp_path / "mult.yaml"
    path.write_text(_YAML.format(
        t_min=m5c.T_MIN, t_max=m5c.T_MIN + m5c.T + 0.5,
        t_final=m5c.T_MIN + m5c.T, interpretation=interpretation,
        F=m5c.F_TENSOR.tolist(), g0=m5c.G0.tolist(), g1=m5c.G1.tolist()))
    cfg = load_workflow_config(path)
    sweep, totals = run_workflow(cfg)

    system = m5c.make_system(m5c.GAMMAS["scalar"], interpretation)
    assert (sorted(v.name for v in cfg_system_vertices(cfg))
            == sorted(v.name for v in system.multiplicative_vertices))
    props = m5c.propagators_for(system)
    for order in (0, 1, 2):
        want = system.expand(TWO, orders=[order], diag_C=False).evaluate(
            props, positions=m5c.POS, t_final=m5c.T_MIN + m5c.T,
            component_pair=(0, 1), orders=[order], method="gauss_legendre",
            n_gauss=12).total
        row = totals[totals["order"] == order]
        assert float(row["value"].iloc[0]) == pytest.approx(
            want, rel=1e-10, abs=1e-14), order


def cfg_system_vertices(cfg):
    from sft_wick.workflow.config import build_system
    return build_system(cfg.system).multiplicative_vertices


# --------------------------------------------------------------------------
# MN9 -- ito=False is refused where the interpretation decides the value
# --------------------------------------------------------------------------

_N = 2
_F = m5c.F_TENSOR
_S1 = (np.einsum("ak,bkc->abc", m5c.G0, m5c.G1)
       + np.einsum("akc,bk->abc", m5c.G1, m5c.G0))


def _l0_terms(obs, order, vertices, ito):
    sw.reset_uid_counter()
    phi = sw.Field("phi", "physical", n_components=_N)
    psi = sw.Field("psi", "response", n_components=_N)
    fields = {"F": [psi, phi, phi], "G": [psi, psi, phi]}
    action = sw.Action(vertices=[sw.Vertex(fields=fields[v], coupling=v,
                                           local=True)
                                 for v in sorted(vertices)])
    ops = [phi(c, lab) for c, lab in zip("ab", obs)]
    return sw.compute_moment(ops, action, order=order, iso_R=True,
                             diag_R=True, diag_C=False,
                             ito=ito).diagram_terms(order)


@pytest.fixture(scope="module")
def l0_cache():
    system = m5c.make_system(m5c.GAMMAS["scalar"], "ito")
    return m5c.propagators_for(system).cache


_L0_COUPLINGS = {"F": -1j * _F, "G": 0.5 * _S1}
_METHODS = [("gauss_legendre", dict(n_gauss=12)),
            ("qmc_vectorized", dict(n_samples=2 ** 10, seed=1)),
            ("qmc_scalar", dict(n_samples=2 ** 10, seed=1)),
            ("nquad", {})]


@pytest.mark.parametrize("method, kw", _METHODS)
def test_MN9_two_psi_vertex_self_loop_is_refused(l0_cache, method, kw):
    """⟨φ_a⟩ at order 1 with a ψψφ vertex: the extra ito=False term is the
    Θ(0) ∂_b D_ab drift, which the interpretation fixes."""
    dts = _l0_terms(("x",), 1, {"G"}, ito=False)
    assert len(dts) == 1 and not _l0_terms(("x",), 1, {"G"}, ito=True)
    with pytest.raises(ValueError, match="psi legs"):
        integrate_diagrams(dts, _L0_COUPLINGS, lambda_f=m5c.T_MIN + m5c.T,
                           t_min=m5c.T_MIN, cache=l0_cache, method=method,
                           fixed_indices={"a": 0}, positions={"x": 0.0}, **kw)
    with pytest.raises(ValueError, match="psi legs"):
        dts[0].build_integrand(_L0_COUPLINGS, {"a": 0})


def test_MN9_external_equal_point_r_is_refused():
    """⟨φ(x) ψ(x)⟩ at order 0 under ito=False is Θ(0) itself."""
    sw.reset_uid_counter()
    phi = sw.Field("phi", "physical")
    psi = sw.Field("psi", "response")
    dts = sw.compute_moment([phi("x"), psi("x")], sw.Action(vertices=[]),
                            order=0, ito=False).diagram_terms(0)
    assert len(dts) == 1
    with pytest.raises(ValueError, match="equal-time response"):
        dts[0].build_integrand({})


@pytest.mark.parametrize("method, kw", _METHODS)
@pytest.mark.parametrize("obs, order", [(("x",), 1), (("x", "y"), 2)])
def test_MN9_one_psi_vertex_keeps_the_ito_value(l0_cache, method, kw, obs,
                                                order):
    """An equal-point R on a one-ψ vertex is cancelled by the Jacobian of
    the Stratonovich path integral, which the package does not emit either;
    both omissions are exact, so the number does not move."""
    if method == "nquad" and order > 1:
        pytest.skip("nquad is slow on the kinked order-2 integrand")
    kept = _l0_terms(obs, order, {"F"}, ito=False)
    dropped = _l0_terms(obs, order, {"F"}, ito=True)
    assert len(kept) > len(dropped), "no equal-point R term to evaluate"
    values = []
    for dts in (dropped, kept):
        total, _ = integrate_diagrams(
            dts, _L0_COUPLINGS, lambda_f=m5c.T_MIN + m5c.T, t_min=m5c.T_MIN,
            cache=l0_cache, method=method,
            fixed_indices=dict(zip("ab", (0,) * len(obs))),
            positions={lab: 0.0 for lab in obs}, **kw)
        values.append(total)
    assert values[1] == pytest.approx(values[0], rel=1e-12, abs=1e-15)


def test_MN9_l1_expand_with_ito_false_is_refused(scalar_r):
    system, props, _H, _exps = scalar_r["stratonovich"]
    exp = system.expand(ONE, orders=[1], diag_C=False, ito=False)
    with pytest.raises(ValueError, match="psi legs"):
        exp.evaluate(props, positions={"x": 0.0}, t_final=m5c.T_MIN + m5c.T,
                     component_pair=(0,), orders=[1],
                     method="gauss_legendre", n_gauss=8)
