"""Sweep-local static preparation against independent analytic fixtures.

Parity checks include per-diagram errors and metadata. Analytic checks use
the defining K_ijk M_ai M_bj M_ck contraction and a prescribed absorbed kernel.
"""
from collections import Counter

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.perturbation import DiagramTerm
from sft_wick.workflow.config import _parse_sweep
from tests import test_sweep_component_tuples as nt
from tests import test_matrix_r_batched as mb
from tests import test_group_position_conflict as gp


def _build_spy(monkeypatch):
    seen = []
    original = DiagramTerm.build_integrand

    def spy(self, coupling_values, fixed_indices=None):
        seen.append((id(self), tuple(sorted((fixed_indices or {}).items()))))
        return original(self, coupling_values, fixed_indices)

    monkeypatch.setattr(DiagramTerm, "build_integrand", spy)
    return seen


@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized"])
def test_static_reuse_routes_components_and_unequal_times_to_analytic_reference(method):
    gammas = np.array([1.0, 1.4])
    exp, props, _ = nt._static(gammas)
    kw = dict(positions_grid=nt.STATIC_POS,
              t_final_grid=[1.2, 1.5],
              external_times_grid={"y": [0.75, 1.1]},
              component_tuples=nt.STATIC_TUPLES,
              orders=[1], method=method, n_gauss=10,
              n_samples=2**11, seed=5, progress=False)
    expected_rows = exp.sweep(props, reuse_integrands=False, **kw).rows
    reused = exp.sweep(props, reuse_integrands=True, **kw)
    assert reused.rows == expected_rows  # includes errors and metadata, not only totals
    tol = 1e-12 if method == "gauss_legendre" else 2e-3
    for row in reused.totals().to_dict("records"):
        times = [row["t_final"], row["t_y"], row["t_final"]]
        Ms = [np.diag(-np.expm1(-gammas*(t-nt.T_MIN))/gammas) for t in times]
        comps = tuple(row[c] for c in "abc")
        ref = np.einsum("ijk,ai,bj,ck->abc", nt.K_SYM, *Ms)[comps]
        assert row["value"] == pytest.approx(ref, rel=tol, abs=1e-13)


def test_static_build_once_per_component_and_current_tensors_per_sweep(monkeypatch):
    exp, props, _ = nt._static([1.0, 1.4])
    # Own the mutable bare tensor; do not change the shared reference fixture.
    from dataclasses import replace
    nv = exp.system.nonlocal_vertices[0]
    system = replace(exp.system, nonlocal_vertices=[replace(nv, coupling=nv.coupling.copy())])
    exp = replace(exp, system=system)
    seen = _build_spy(monkeypatch)
    kw = dict(positions_grid=nt.STATIC_POS,
              t_final_grid=[0.75, 1.2, 1.5],
              component_tuples=nt.STATIC_TUPLES,
              orders=[1], method="gauss_legendre", n_gauss=8,
              progress=False)
    fresh = exp.sweep(props, **kw)
    assert len(seen) == len(exp.diagrams(1))*len(nt.STATIC_TUPLES)*3
    seen.clear()
    first = exp.sweep(props, reuse_integrands=True, **kw)
    assert first.rows == fresh.rows
    counts = Counter(seen)
    assert len(counts) == len(exp.diagrams(1))*len(nt.STATIC_TUPLES)
    assert set(counts.values()) == {1}
    # In-place mutation specifically probes stale MSR-converted arrays.
    bare = exp.system.nonlocal_vertices[0].coupling
    saved = bare.copy()
    try:
        bare *= -2.5
        second = exp.sweep(props, reuse_integrands=True, **kw)
        assert len(seen) == 2*len(counts)
        assert [r["value"] for r in second.rows] == pytest.approx(
            [-2.5*r["value"] for r in first.rows], rel=1e-12, abs=1e-14)
    finally:
        bare[...] = saved


def _dynamic():
    bare = nt.K_SYM.copy()

    def kernel(n_list, t_list):
        # Symmetric under simultaneous component/point leg exchange.
        envelope = np.exp(-sum(float(np.asarray(n)**2) for n in n_list))
        return bare * envelope * (1 + sum(t_list))

    system = sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        nonlocal_vertices=[sw.NonLocalVertex(
            "K", order=3, coupling=kernel, already_R_contracted=True)],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.3, sigma_t=0.5),
            spatial=sw.ExponentialSpatial(sigma_x=1.0))),
        t_min=nt.T_MIN)
    exp = system.expand(("phi_a(x)", "phi_b(y)", "phi_c(z)"), orders=[1])
    props = system.propagators(t_max=2.0, n_grid_t=8, progress=False)
    return exp, props, bare


@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized"])
def test_dynamic_fallback_rebuilds_and_observes_each_position_time(monkeypatch, method):
    exp, props, bare = _dynamic()
    seen = _build_spy(monkeypatch)
    kw = dict(positions_grid={"x": [0.0, 0.6], "y": [0.3], "z": [-0.4]},
              t_final_grid=[0.8, 1.2], external_times_grid={"y": [0.5, 0.7]},
              component_tuples=[(0, 1, 1), (1, 0, 0)],
              orders=[1], method=method, n_gauss=4, n_samples=32,
              seed=7, progress=False)
    fresh = exp.sweep(props, reuse_integrands=False, **kw)
    seen.clear()
    reused = exp.sweep(props, reuse_integrands=True, **kw)
    assert reused.rows == fresh.rows
    assert len(seen) == 2*2*2*2*len(exp.diagrams(1))
    for row in reused.totals().to_dict("records"):
        comps = tuple(row[c] for c in "abc")
        ref = bare[comps] * np.exp(-sum(row[c]**2 for c in "xyz"))
        ref *= 1 + 2*row["t_final"] + row["t_y"]
        assert row["value"] == pytest.approx(ref, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("n_jobs,evaluate_n_jobs", [(2, 1), (1, 2)])
@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized"])
def test_parallel_reused_dense_R_and_C_are_identical(n_jobs, evaluate_n_jobs, method):
    system, props = mb._dense("F+raw")
    exp = system.expand(mb.OBS2, orders=[0, 2], diag_R=False, diag_C=False)
    assert sum(len(exp.diagrams(o)) for o in exp.orders) > 2  # triggers inner loky
    kw = dict(positions_grid={"x": [0.0], "y": [0.0, 0.7]},
              t_final_grid=[mb.T_F-0.1, mb.T_F],
              component_pairs=[(0, 1), (1, 0)], orders=[0, 2],
              method=method, n_gauss=4, n_samples=128, seed=7, progress=False)
    serial = exp.sweep(props, reuse_integrands=False, **kw)
    reused = exp.sweep(props, reuse_integrands=True,
                       n_jobs=n_jobs, evaluate_n_jobs=evaluate_n_jobs, **kw)
    assert reused.rows == serial.rows


def test_position_guard_remains_active_after_first_valid_reused_point():
    system = gp._system()
    props = system.propagators(t_max=4.0, n_grid_t=8,
        c_closed_form=gp._C_fn, c_closed_form_only=True, progress=False)
    exp = system.expand(("phi_a(x)", "psi_b(y)"), orders=[2])
    kw = dict(positions_grid={"x": [0.0], "y": [0.0, 0.9]},
              t_final_grid=[gp.T_F], external_times_grid={"y": [gp.T_Y]},
              component_pairs=[(0, 0)], orders=[2],
              method="gauss_legendre", n_gauss=4, progress=False)
    with pytest.raises(ValueError, match=gp.MATCH):
        exp.sweep(props, reuse_integrands=True, **kw)


def test_R_cache_guard_does_not_disappear_when_integrands_are_prepared():
    # Expansion with index-free scalar R against a matrix-R cache.
    exp, _, _ = nt._static([1.0, 1.0])
    _, matrix_props, _ = nt._static([1.0, 1.4])
    with pytest.raises(ValueError, match="R is a matrix"):
        exp.sweep(matrix_props, reuse_integrands=True,
                  positions_grid=nt.STATIC_POS, t_final_grid=[1.2],
                  component_tuples=nt.STATIC_TUPLES, orders=[1],
                  method="gauss_legendre", n_gauss=4, progress=False)


def test_mixed_static_and_dynamic_diagrams_reuse_only_the_active_static_subset(monkeypatch):
    from sft_wick.workflow.expansion import _collect_symbol_names

    system, props = mb._dense("F+callable")
    exp = system.expand(mb.OBS2, orders=[0, 2], diag_R=False, diag_C=False)
    seen = _build_spy(monkeypatch)
    kw = dict(positions_grid={"x": [0.0], "y": [0.0, 0.7]},
              t_final_grid=[mb.T_F-0.1, mb.T_F],
              component_pairs=[(0, 1), (1, 0)], orders=[0, 2],
              method="gauss_legendre", n_gauss=4, progress=False)
    fresh = exp.sweep(props, reuse_integrands=False, **kw)
    seen.clear()
    reused = exp.sweep(props, reuse_integrands=True, **kw)
    assert reused.rows == fresh.rows
    counts = Counter(diagram_id for diagram_id, _indices in seen)
    for order in exp.orders:
        for dt in exp.diagrams(order):
            dynamic = "K" in _collect_symbol_names(dt.coupling_sum)
            assert counts[id(dt)] == (8 if dynamic else 2)
    # An unrelated callable must not disable the free C or pure FF cache.
    assert any("K" not in _collect_symbol_names(dt.coupling_sum)
               for o in exp.orders for dt in exp.diagrams(o))
    assert any("K" in _collect_symbol_names(dt.coupling_sum)
               for o in exp.orders for dt in exp.diagrams(o))


@pytest.mark.parametrize("field,value,match", [
    ("n_components", 3, "n_components=3"),
    ("t_min", nt.T_MIN+0.1, "built with t_min"),
])
def test_prepared_sweep_keeps_system_cache_consistency_guards(monkeypatch, field, value, match):
    from dataclasses import replace

    exp, props, _ = nt._static([1.0, 1.4])
    monkeypatch.setattr(props.cache, "model", replace(props.cache.model, **{field: value}))
    with pytest.raises(ValueError, match=match):
        exp.sweep(props, reuse_integrands=True,
                  positions_grid=nt.STATIC_POS, t_final_grid=[1.2],
                  component_tuples=nt.STATIC_TUPLES, orders=[1],
                  method="gauss_legendre", n_gauss=4, progress=False)


@pytest.mark.parametrize("bad", [None, 0, 1, "false", [], {}])
def test_L1_reuse_flag_requires_boolean(bad):
    exp, props, _ = nt._static([1.0, 1.4])
    with pytest.raises(TypeError, match="reuse_integrands must be a boolean"):
        exp.sweep(props, reuse_integrands=bad,
                  positions_grid=nt.STATIC_POS, t_final_grid=[1.2],
                  component_tuples=nt.STATIC_TUPLES, orders=[1],
                  method="gauss_legendre", n_gauss=4, progress=False)


def test_yaml_default_false_and_true_are_forwarded_to_sweep(tmp_path, monkeypatch):
    from sft_wick.workflow.config import load_workflow_config, run_workflow
    from sft_wick.workflow.expansion import Expansion

    path = nt._demo4_yaml(tmp_path, lambda cfg: None)
    default_cfg = load_workflow_config(path)
    assert default_cfg.sweep.reuse_integrands is False
    calls = []
    original = Expansion.sweep

    def spy(self, propagators, **kw):
        calls.append(kw["reuse_integrands"])
        return original(self, propagators, **kw)

    monkeypatch.setattr(Expansion, "sweep", spy)
    fresh, _ = run_workflow(default_cfg, progress=False)
    path = nt._demo4_yaml(tmp_path,
                         lambda cfg: cfg["sweep"].update(reuse_integrands=True))
    reuse_cfg = load_workflow_config(path)
    assert reuse_cfg.sweep.reuse_integrands is True
    reused, _ = run_workflow(reuse_cfg, progress=False)
    assert calls == [False, True]
    assert reused.rows == fresh.rows


@pytest.mark.parametrize("value,expected", [(True, True), (False, False),
                                           ("true", True), ("false", False)])
def test_yaml_reuse_flag_is_a_boolean(value, expected):
    cfg = _parse_sweep(dict(positions_grid={"x": [0]}, t_final_grid=[1],
                            component_pairs=[[0, 0]], reuse_integrands=value))
    assert cfg.reuse_integrands is expected


def test_yaml_reuse_flag_default_and_invalid_value():
    raw = dict(positions_grid={"x": [0]}, t_final_grid=[1],
               component_pairs=[[0, 0]])
    assert _parse_sweep(raw).reuse_integrands is False
    with pytest.raises(ValueError, match="sweep.reuse_integrands"):
        _parse_sweep({**raw, "reuse_integrands": "perhaps"})


@pytest.mark.parametrize("kind", ["raw", "equal_time", "absorbed"])
@pytest.mark.parametrize("integrate_over", [None, {"x"}, "all"])
def test_static_aliases_and_external_integrals_match_independent_reference(kind,
                                                                          integrate_over):
    from dataclasses import replace
    from scipy.integrate import quad

    gammas = np.array([1.0, 1.4])
    base, props, _ = nt._static(gammas)
    vertex = sw.NonLocalVertex("K", order=3, coupling=nt.K_SYM.copy(),
                               equal_time=(kind == "equal_time"),
                               already_R_contracted=(kind == "absorbed"))
    system = replace(base.system, nonlocal_vertices=[vertex])
    exp = system.expand(("phi_a(x)", "phi_b(y)", "phi_c(z)"), orders=[1])
    kw = dict(positions_grid=nt.STATIC_POS, t_final_grid=[0.8, 1.2],
              component_tuples=[(0, 1, 1), (1, 0, 0)], orders=[1],
              integrate_over=integrate_over, method="gauss_legendre",
              n_gauss=12, progress=False)
    fresh = exp.sweep(props, **kw)
    reused = exp.sweep(props, reuse_integrands=True, **kw)
    assert reused.rows == fresh.rows
    integrated = set("xyz") if integrate_over == "all" else (integrate_over or set())
    for row in reused.totals().to_dict("records"):
        comps = tuple(row[c] for c in "abc")
        delta = row["t_final"] - nt.T_MIN
        rates = gammas[list(comps)]
        if kind == "equal_time":
            def source_integrand(s):
                lag = row["t_final"] - s
                factors = [-np.expm1(-g * lag) / g if label in integrated
                           else np.exp(-g * lag)
                           for label, g in zip("xyz", rates)]
                return np.prod(factors)
            weight = quad(source_integrand, nt.T_MIN, row["t_final"],
                          epsabs=1e-13, epsrel=1e-13)[0]
        elif kind == "absorbed":
            weight = delta ** len(integrated)
        else:
            M = -np.expm1(-rates * delta) / rates
            weight = np.prod([(delta - m) / g if label in integrated else m
                              for label, g, m in zip("xyz", rates, M)])
        reference = nt.K_SYM[comps] * weight
        assert row["value"] == pytest.approx(reference, rel=1e-12, abs=1e-13)
