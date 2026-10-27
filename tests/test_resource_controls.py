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
