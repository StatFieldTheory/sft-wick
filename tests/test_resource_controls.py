"""Configuration-only eligibility and connected dry-run target checks."""

from dataclasses import replace

import pytest

from sft_wick.workflow.config import _parse_workflow
from sft_wick.workflow.resources import resource_controls


def _cfg(tmp_path):
    return _parse_workflow(
        {
            "system": {
                "field": {"n_components": 1},
                "linear": {"type": "diagonal", "gamma": [1.0]},
                "vertices": [],
                "noise": {
                    "kappa2": {
                        "type": "separable_translation",
                        "temporal": {
                            "type": "exponential",
                            "lam": 0.05,
                            "sigma_t": 0.3,
                        },
                        "spatial": {"type": "exponential", "sigma_x": 1.0},
                    }
                },
            },
            "expand": {"observable": ["phi(x)", "phi(y)"], "orders": [0]},
            "propagators": {"t_max": 1.0, "n_grid_t": 5},
            "sweep": {
                "positions_grid": {"x": [0.0], "y": [0.0]},
                "t_final_grid": [0.5],
                "component_pairs": [[0, 0]],
                "n_samples": 16,
            },
        },
        tmp_path,
    )


def _controls(cfg):
    return {entry["name"]: entry for entry in resource_controls(cfg)}


@pytest.mark.parametrize(
    "method,gl_status,qmc_status",
    [
        ("gauss_legendre", "conditional", "ignored"),
        ("qmc_scalar", "ignored", "ignored"),
        ("qmc_vectorized", "ignored", "ignored"),
    ],
)
def test_method_and_static_eligibility(tmp_path, method, gl_status, qmc_status):
    cfg = _cfg(tmp_path)
    cfg = replace(
        cfg,
        sweep=replace(
            cfg.sweep, method=method, gl_chunk_size=100, qmc_coupling_chunk_size=4
        ),
    )
    controls = _controls(cfg)
    assert controls["sweep.gl_chunk_size"]["status"] == gl_status
    assert controls["sweep.qmc_coupling_chunk_size"]["status"] == qmc_status
    if method == "gauss_legendre":
        assert "eager grid" in controls["sweep.gl_chunk_size"]["reason"]


def test_dynamic_feedback_never_calls_callback_and_cap_bypass(tmp_path):
    cfg = _cfg(tmp_path)

    def forbidden(*args):
        raise AssertionError("resource feedback called a coupling")

    cfg = replace(
        cfg,
        system=replace(cfg.system, vertices=[{"coupling": forbidden}]),
        sweep=replace(cfg.sweep, qmc_coupling_chunk_size=4, reuse_integrands=True),
    )
    assert _controls(cfg)["sweep.qmc_coupling_chunk_size"]["status"] == "conditional"
    cfg = replace(cfg, sweep=replace(cfg.sweep, qmc_coupling_chunk_size=16))
    assert (
        "at least n_samples"
        in _controls(cfg)["sweep.qmc_coupling_chunk_size"]["reason"]
    )


def test_three_parallel_scopes_and_small_sweep(tmp_path):
    cfg = _cfg(tmp_path)
    cfg = replace(
        cfg,
        sweep=replace(cfg.sweep, n_jobs=3),
        propagators=replace(cfg.propagators, n_jobs=2, c_closed_form_only=True),
    )
    controls = _controls(cfg)
    assert controls["sweep.n_jobs"]["status"] == "ignored"
    assert controls["propagators.n_jobs"]["status"] == "ignored"
    assert "not symbolic expansion" in controls["expand.n_jobs"]["scope"]
    cfg = replace(cfg, expand=replace(cfg.expand, n_jobs=2))
    assert _controls(cfg)["expand.n_jobs"]["status"] == "invalid"
    cfg = replace(
        cfg,
        expand=replace(cfg.expand, n_jobs=1),
        sweep=replace(cfg.sweep, t_final_grid=[0.2, 0.4, 0.6]),
    )
    assert _controls(cfg)["sweep.n_jobs"]["status"] == "conditional"
    assert "each worker" in _controls(cfg)["sweep.n_jobs"]["reason"]


def test_connected_dry_run_counts_connected_diagrams(tmp_path):
    from sft_wick.workflow.estimate import estimate_cost

    cfg = _cfg(tmp_path)
    cfg = replace(
        cfg,
        expand=replace(
            cfg.expand,
            observable=("phi(x)", "phi(y)", "phi(z)", "phi(w)"),
            target="connected",
        ),
        sweep=replace(
            cfg.sweep,
            positions_grid=dict.fromkeys("xyzw", [0.0]),
            component_pairs=[(0, 0, 0, 0)],
        ),
    )
    estimate = estimate_cost(cfg, bench_seconds=0.01)
    assert estimate.diagrams_per_order == {0: 0}
    assert estimate.t_diagram_per_order == {0: 0.0}


@pytest.mark.parametrize("method", ["qmc", "qmc_scalar", "qmc_vectorized", "nquad",
                                    "gauss_legendre"])
def test_the_report_names_the_integrator_the_run_takes(tmp_path, monkeypatch, method):
    """``qmc`` is the scalar loop on every cache ``Propagators.build`` makes,
    and the report used to class it with the batched paths."""
    import sft_wick.evaluate as ev
    from sft_wick.workflow.config import run_workflow
    from sft_wick.workflow.resources import runs_scalar_loop

    cfg = _cfg(tmp_path)
    cfg = replace(cfg, sweep=replace(cfg.sweep, method=method, n_samples=16, n_gauss=3,
                                     propagator_cache_bytes=4096,
                                     propagator_cache_contract="fixed_model",
                                     qmc_coupling_chunk_size=4,
                                     contraction_backend="einsum"))
    cfg = replace(cfg, expand=replace(cfg.expand, orders=(0, 2)), system=replace(
        cfg.system, vertices=[{"name": "F", "coupling": [[[0.3]]]}]))
    batched = []
    real = ev.DiagramIntegrand.integrate_moment_qmc_vectorized
    monkeypatch.setattr(
        ev.DiagramIntegrand, "integrate_moment_qmc_vectorized",
        lambda self, *a, **k: batched.append(1) or real(self, *a, **k))
    run_workflow(cfg, progress=False)
    controls = _controls(cfg)
    assert bool(batched) == (method == "qmc_vectorized")
    assert (controls["sweep.qmc_coupling_chunk_size"]["reason"].startswith(
        "method does not use")) == (method != "qmc_vectorized")
    scalar = method in ("qmc", "qmc_scalar", "nquad")
    assert runs_scalar_loop(method) is scalar
    assert (controls["sweep.propagator_cache_bytes"]["status"] == "ignored") is scalar
    assert (controls["sweep.contraction_backend"]["status"] == "ignored") is scalar


def test_estimate_counts_workers_as_joblib_does():
    import joblib

    from sft_wick.workflow.estimate import CostEstimate

    def estimate(n_jobs, points=9):
        return CostEstimate({0: 1}, points, 1, "gauss_legendre", 8, "closed_form:builtin",
                            True, 0, None, {0: 1.0}, sweep_n_jobs=n_jobs)

    for n_jobs in (-3, -2, -1, 2):
        assert estimate(n_jobs)._n_workers() == joblib.effective_n_jobs(n_jobs)
    assert "workers" in estimate(2).summary()
    assert "workers" not in estimate(2, points=2).summary()


def test_dry_run_takes_the_run_arguments(tmp_path, monkeypatch):
    """The dry run used to leave out the execution options, the homogeneity
    and the interpolation, and to build the whole full-grid table."""
    from sft_wick.workflow.config import run_workflow
    from sft_wick.workflow.estimate import estimate_cost
    from sft_wick.workflow.system import System

    cfg = _cfg(tmp_path)
    cfg = replace(
        cfg,
        expand=replace(cfg.expand, orders=(0, 2)),
        system=replace(cfg.system, vertices=[{"name": "F", "coupling": [[[0.3]]]}]),
        propagators=replace(cfg.propagators, r_max=3.0, n_grid_r=5, n_grid_t=8,
                            interp_method="cubic", c_closed_form=None),
        sweep=replace(cfg.sweep, method="gauss_legendre", n_gauss=3),
    )
    built = []
    real = System.propagators
    monkeypatch.setattr(System, "propagators",
                        lambda self, **k: built.append(k) or real(self, **k))
    estimate_cost(cfg, bench_seconds=0.01)
    assert all(k["interp_method"] == "cubic" and k["r_max"] == 3.0 for k in built)
    assert all(k["n_grid_r"] == 4 and k["n_grid_t"] == 6 for k in built)

    refused = replace(cfg, sweep=replace(
        cfg.sweep, contraction_backend="factorized", contraction_max_bytes=1))
    with pytest.raises(ValueError, match="contraction_max_bytes") as run:
        run_workflow(refused, progress=False)
    with pytest.raises(ValueError, match="contraction_max_bytes") as dry:
        estimate_cost(refused, bench_seconds=0.01)
    assert str(dry.value) == str(run.value)
