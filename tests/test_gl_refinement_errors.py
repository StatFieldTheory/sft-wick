"""GL convergence deltas retain coarse values and use signed group sums."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from sft_wick.workflow._version_worker import _normalize_rows, _workflow_rows
from sft_wick.workflow.config import (
    ConfigError, build_system, expand_kwargs, load_workflow_config, run_workflow,
)


QUICKSTART = Path(__file__).resolve().parents[1] / "examples" / "quickstart.yaml"


@pytest.fixture(scope="module")
def model():
    cfg = load_workflow_config(QUICKSTART)
    system = build_system(cfg.system)
    return system.expand(**expand_kwargs(cfg)), system.propagators(
        t_max=5, c_closed_form_only=True,
    )


def test_gl_delta_keeps_coarse_value_and_uses_same_domain_for_refinement(model):
    expansion, props = model
    settings = dict(
        method="gauss_legendre", positions={"x": 0.0, "y": 0.5},
        t_final=3.0, external_times={"x": 1.0}, gl_chunk_size=3,
    )
    coarse = expansion.evaluate(props, n_gauss=3, **settings)
    fine = expansion.evaluate(props, n_gauss=7, **settings)
    paired = expansion.evaluate(props, n_gauss=3, gl_error=True, **settings)
    assert paired.by_order == coarse.by_order
    assert paired.total == coarse.total
    assert paired.error == abs(coarse.total - fine.total)
    assert paired.error_by_order == {
        order: abs(value - fine.by_order[order])
        for order, value in coarse.by_order.items()
    }
    assert paired.error_by_order[2] > 0.0
    assert paired.error_by_vertex_type == {
        group: abs(value - fine.by_vertex_type[group])
        for group, value in coarse.by_vertex_type.items()
    }
    assert paired.to_dict()["error_kind"] == "gl_delta"
    for row, a, b in zip(paired.per_diagram, coarse.per_diagram, fine.per_diagram):
        assert row["value"] == a["value"]
        assert row["error"] == abs(a["value"] - b["value"])
        assert row["error_kind"] == "gl_delta"


@pytest.mark.parametrize("gl_error", [True, "enhanced"])
def test_gl_sweep_yaml_and_npz_keep_signed_refinement_totals(model, tmp_path, gl_error):
    expansion, props = model
    grid = dict(
        method="gauss_legendre", positions_grid={"x": [0.0], "y": [0.0, 0.5, 1.5]},
        t_final_grid=[3.0], component_tuples=[(0, 0)],
    )
    coarse = expansion.sweep(props, n_gauss=3, **grid)
    fine = expansion.sweep(props, n_gauss=7, **grid)
    paired = expansion.sweep(props, n_gauss=3, gl_error=gl_error, n_jobs=2, **grid)
    totals = paired.totals()
    delta = (coarse.totals()["value"] - fine.totals()["value"]).abs()
    diagnostic = "gl_delta" if gl_error == "enhanced" else "error"
    kind = "gl_estimate" if gl_error == "enhanced" else "gl_delta"
    np.testing.assert_array_equal(totals[diagnostic], delta)
    np.testing.assert_array_equal(totals["value"], coarse.totals()["value"])
    records = _normalize_rows(_workflow_rows(paired, totals))
    assert [row["error"] for row in records] == list(totals["error"])
    if gl_error == "enhanced":
        assert (totals["error"] >= delta).all()
        assert all("gl_delta" not in row["coordinates"] for row in records)

    raw = yaml.safe_load(QUICKSTART.read_text())
    raw["propagators"]["c_closed_form_only"] = True
    raw["sweep"].update(grid, n_gauss=3, gl_error=gl_error)
    del raw["sweep"]["component_pairs"]
    raw["sweep"]["component_tuples"] = [[0, 0]]
    path = tmp_path / "gl.yaml"
    npz = tmp_path / "gl.npz"
    raw["output"] = [{"path": str(npz), "type": "npz"}]
    path.write_text(yaml.safe_dump(raw))
    _, from_yaml = run_workflow(load_workflow_config(path), progress=False)
    assert from_yaml.equals(totals)
    with np.load(npz, allow_pickle=False) as out:
        np.testing.assert_array_equal(out[diagnostic], delta)
        assert set(out["error_kind"]) == {kind}


def test_gl_delta_sums_signed_values_before_taking_absolute_difference():
    from sft_wick.workflow.result import _refinement_group_error

    rows = (
        (dict(value=2.0, error_kind="gl_delta"), 3.0),
        (dict(value=-1.0, error_kind="gl_delta"), -2.0),
    )
    assert _refinement_group_error(rows) == (0.0, "gl_delta")


def test_gl_enhanced_keeps_raw_delta_and_adds_separate_residual(model):
    expansion, props = model
    settings = dict(
        method="gauss_legendre", positions={"x": 0.0, "y": 0.5},
        t_final=3.0, external_times={"x": 1.0}, gl_chunk_size=3,
    )
    middle = expansion.evaluate(props, n_gauss=5, **settings)
    fine = expansion.evaluate(props, n_gauss=7, **settings)
    raw = expansion.evaluate(props, n_gauss=3, gl_error=True, **settings)
    enhanced = expansion.evaluate(props, n_gauss=3, gl_error="enhanced", **settings)
    assert enhanced.by_order == raw.by_order
    summary = enhanced.to_dict()
    assert summary["error_kind"] == "gl_estimate"
    assert summary["gl_delta"] == raw.error
    assert summary["gl_delta_by_order"] == raw.error_by_order
    for order, error in enhanced.error_by_order.items():
        assert error >= raw.error_by_order[order] + abs(
            middle.by_order[order] - fine.by_order[order]
        )
    assert enhanced.error_by_order[2] > raw.error_by_order[2]
    for row, unenhanced in zip(enhanced.per_diagram, raw.per_diagram, strict=True):
        assert row["error_kind"] == "gl_estimate"
        assert row["gl_delta"] == unenhanced["error"]


def test_gl_enhanced_groups_signed_values_at_each_resolution():
    from sft_wick._gl_error import _GLRefinement
    from sft_wick.workflow.result import _enhanced_gl_error

    rows = (
        ({"value": 2.0}, _GLRefinement(3.0, 4.0, 1e-14, 2e-14)),
        ({"value": -1.0}, _GLRefinement(-2.0, -3.0, 1e-14, 2e-14)),
    )
    result = _enhanced_gl_error(rows)
    assert result.delta == 0.0
    assert result.residual == 0.0
    assert 8e-14 < result.roundoff < 9e-14
    assert result.error >= result.roundoff


def test_gl_enhanced_budget_accounts_for_final_weight_underflow():
    from math import ulp

    from sft_wick._gl_error import _GLRoundoff

    tiniest = ulp(0.0)
    weighted = np.zeros(4)
    unweighted = np.full(4, tiniest)
    accumulator = _GLRoundoff()
    accumulator.observe(weighted, 1, unweighted)
    assert accumulator.budget() >= 4 * tiniest
    accumulator = _GLRoundoff()
    accumulator.observe(weighted, 1, weighted)
    assert accumulator.budget() == 0.0


def test_gl_enhanced_keeps_representable_budget_for_large_cancelled_rules():
    from math import isfinite

    from sft_wick._gl_error import _GLRefinement
    from sft_wick.workflow.result import _enhanced_gl_error

    rows = (
        ({"value": 1e308}, _GLRefinement(1e308, 1e308, 0.0, 0.0)),
        ({"value": -1e308}, _GLRefinement(-1e308, -1e308, 0.0, 0.0)),
    )
    result = _enhanced_gl_error(rows)
    assert result.delta == 0.0
    assert result.residual == 0.0
    assert result.roundoff > 6e292
    assert isfinite(result.error)


def test_gl_error_requires_gauss_legendre(tmp_path):
    raw = yaml.safe_load(QUICKSTART.read_text())
    raw["sweep"]["gl_error"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="gl_error: requires gauss_legendre"):
        load_workflow_config(path)
