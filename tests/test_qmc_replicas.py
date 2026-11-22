"""Independent scrambles preserve the within-replica covariance of diagrams."""

import json
import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from sft_wick.workflow.config import (
    ConfigError,
    OutputConfig,
    build_system,
    expand_kwargs,
    load_workflow_config,
    run_workflow,
)
from sft_wick.workflow._version_worker import _normalize_rows, _workflow_rows
from sft_wick.workflow.estimate import estimate_cost
from sft_wick.workflow.resource_compare import compare_results, encode_results
from sft_wick.workflow.result import _replica_standard_error, _replica_sum_error


QUICKSTART = Path(__file__).resolve().parents[1] / "examples" / "quickstart.yaml"
N_SAMPLES = 64
REPLICAS = 4


@pytest.fixture(scope="module")
def setup():
    config = load_workflow_config(QUICKSTART)
    system = build_system(config.system)
    expansion = system.expand(**expand_kwargs(config))
    props = system.propagators(t_max=5.0, c_closed_form_only=True)
    return config, expansion, props


@pytest.mark.parametrize("method", ["qmc", "qmc_scalar", "qmc_vectorized"])
def test_group_standard_error_uses_matching_replica_sums(setup, method):
    _, expansion, props = setup
    args = dict(
        positions={"x": 0.0, "y": 0.5}, t_final=3.0,
        component_pair=(0, 0), method=method,
    )
    seed = 42
    children = [
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in np.random.SeedSequence(seed).spawn(REPLICAS)
    ]
    assert len(set(children)) == REPLICAS
    legacy = [
        expansion.evaluate(props, **args, seed=child, n_samples=N_SAMPLES // REPLICAS)
        for child in children
    ]
    result = expansion.evaluate(
        props, **args, seed=seed, n_samples=N_SAMPLES, qmc_replicas=REPLICAS,
    )
    second_order = [run.by_order[2] for run in legacy]
    expected_error = np.std(second_order, ddof=1) / np.sqrt(REPLICAS)
    assert expected_error > 0.0
    assert result.by_order[2] == pytest.approx(np.mean(second_order), rel=1e-14, abs=0.0)
    assert result.error_by_order[2] == pytest.approx(expected_error, rel=1e-14, abs=0.0)
    assert result.error_by_order[0] == 0.0
    assert result.to_dict()["error_kind"] == "rqmc"
    assert result.to_dict()["qmc_replicas"] == REPLICAS
    for i, row in enumerate(result.per_diagram):
        column = [run.per_diagram[i]["value"] for run in legacy]
        assert row["error_kind"] == "rqmc"
        assert row["value"] == pytest.approx(np.mean(column), rel=1e-14, abs=0.0)
        expected = np.std(np.subtract(column, column[0]), ddof=1) / np.sqrt(REPLICAS)
        assert row["error"] == pytest.approx(expected, rel=1e-14, abs=1e-25)


def test_sweep_npz_and_parallel_modes_use_the_same_replicas(setup, tmp_path):
    config, expansion, props = setup
    args = dict(
        positions_grid={"x": [0.0], "y": [0.0, 0.5, 1.5]}, t_final_grid=[3.0],
        component_tuples=[(0, 0)], n_samples=N_SAMPLES, qmc_replicas=REPLICAS,
    )
    serial = expansion.sweep(props, **args, seed=None)
    assert isinstance(serial.seed, int)
    for mode in ({"n_jobs": 2}, {"evaluate_n_jobs": 2}):
        parallel = expansion.sweep(props, **args, seed=serial.seed, **mode)
        assert parallel.rows == serial.rows
        assert parallel.totals().equals(serial.totals())
        assert parallel.by_vertex_type_totals().equals(serial.by_vertex_type_totals())
    out = tmp_path / "values.npz"
    parsed = replace(
        config,
        propagators=replace(config.propagators, c_closed_form_only=True),
        sweep=replace(config.sweep, positions_grid=args["positions_grid"],
                      t_final_grid=[3.0], component_pairs=[(0, 0)],
                      n_samples=N_SAMPLES, qmc_replicas=REPLICAS, seed=serial.seed),
        output=[OutputConfig(type="npz", path=out)],
    )
    sweep, totals = run_workflow(parsed, progress=False)
    assert sweep.rows == serial.rows
    records = _normalize_rows(_workflow_rows(sweep, totals))
    assert [row["error"] for row in records] == list(totals["error"])
    assert all("error_kind" not in row["coordinates"] for row in records)
    baseline = encode_results(sweep, totals)
    changed = deepcopy(baseline)
    changed["totals_records"][0]["error"] += 1e-8
    results = compare_results(baseline, changed, atol=1.1e-8)
    assert results["metadata_equal"] and results["within_tolerance"]
    assert not compare_results(baseline, changed)["within_tolerance"]
    sweep.rows.reverse()
    np.testing.assert_allclose(sweep.totals()["error"], totals["error"], rtol=1e-14)
    sweep.rows[0] = dict(sweep.rows[0])
    with pytest.raises(ValueError, match="original diagram rows"):
        sweep.totals()
    with np.load(out, allow_pickle=False) as result:
        np.testing.assert_array_equal(result["error_kind"], totals["error_kind"])
        np.testing.assert_array_equal(result["error"], totals["error"])


@pytest.mark.parametrize("replicas, n_samples, method", [
    (True, 64, "qmc_vectorized"),
    (1, 64, "qmc_vectorized"),
    (3, 64, "qmc_vectorized"),
    (4, 20, "qmc_vectorized"),
    (4, 64, "gauss_legendre"),
])
def test_configuration_refuses_invalid_replica_budgets(
    tmp_path, replicas, n_samples, method,
):
    raw = yaml.safe_load(QUICKSTART.read_text())
    raw["sweep"].update(qmc_replicas=replicas, n_samples=n_samples, method=method)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="qmc_replicas"):
        load_workflow_config(path)


def test_perfect_anticorrelation_cancels_before_measuring_spread():
    first = (2.0, 3.0, 5.0, 7.0)
    second = (-2.0, -3.0, -5.0, -7.0)
    assert _replica_standard_error(first) > 0.0
    assert _replica_standard_error(second) > 0.0
    assert _replica_sum_error((first, second), 4) == 0.0


def test_constant_contribution_cannot_hide_replica_spread():
    varying = (0.0, 1e-17, -1e-17, 2e-17)
    expected = np.std(varying, ddof=1) / 2
    measured = _replica_sum_error(((1.0,) * 4, varying), 4)
    assert measured == pytest.approx(expected, rel=1e-14, abs=0.0)
    # First replica sums to zero, second sums to one. Subtracting each
    # diagram's first replica before summing would lose the one.
    scale = 2.0**54
    assert _replica_sum_error(((scale, 1.0), (-scale, 0.0)), 2) == 0.5


@pytest.mark.parametrize("values, expected", [
    ((1e-200, 2e-200), 5e-201),
    ((1e200, 2e200), 5e199),
    ((-1e308, 1e308), 1e308),
])
def test_standard_error_remains_finite_at_extreme_scales(values, expected):
    assert _replica_standard_error(values) == pytest.approx(expected, rel=1e-14, abs=0)


@pytest.mark.filterwarnings("error")
def test_single_point_per_replica_skips_legacy_within_replica_se(setup):
    _, expansion, props = setup
    result = expansion.evaluate(
        props, positions={"x": 0.0, "y": 0.5}, t_final=3.0,
        component_pair=(0, 0), seed=7, n_samples=4, qmc_replicas=4,
    )
    assert result.error_by_order[2] > 0.0


@pytest.mark.filterwarnings("error")
def test_deterministic_replica_mean_stays_finite(setup):
    _, expansion, _ = setup
    props = expansion.system.propagators(
        t_max=5.0, c_closed_form=lambda *args: np.eye(2) * 1e308,
        c_closed_form_only=True,
    )
    result = expansion.evaluate(
        props, positions={"x": 0.0, "y": 0.5}, t_final=3.0, orders=[0],
        component_pair=(0, 0), n_samples=N_SAMPLES, qmc_replicas=REPLICAS,
    )
    assert result.by_order[0] == 1e308
    assert result.error_by_order[0] == 0.0


def test_yaml_and_cost_estimate_accept_non_power_of_two_total_budget(tmp_path):
    raw = yaml.safe_load(QUICKSTART.read_text())
    raw["propagators"]["c_closed_form_only"] = True
    raw["sweep"].update(qmc_replicas=3, n_samples=48)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_workflow_config(path)
    estimate = estimate_cost(config)
    assert estimate.n_samples == 48


def test_heavy_tail_status_preserves_estimate_and_refuses_group_error(
    setup, tmp_path, monkeypatch,
):
    import sft_wick.evaluate as ev

    _, expansion, props = setup
    args = dict(
        positions={"x": 0.0, "y": 0.5}, t_final=3.0, component_pair=(0, 0),
        n_samples=N_SAMPLES, qmc_replicas=REPLICAS, seed=42,
    )
    legacy = expansion.evaluate(props, **args)
    monkeypatch.setattr(ev, "_is_heavy_tail", lambda *_: True, raising=False)
    result = expansion.evaluate(props, **args, qmc_heavy_tail=True)
    assert result.by_order == legacy.by_order
    assert result.error_by_order[0] == 0.0
    assert math.isnan(result.error_by_order[2])
    assert math.isnan(result.error)
    assert result.to_dict()["error_kind"] == "heavy_tail"
    assert result.to_dict()["error"] is None
    assert result.to_dict()["error_by_order"][2] is None
    json.dumps(result.to_dict(), allow_nan=False)
    assert result.qmc_diagnostics
    for row in result.per_diagram:
        key = (row["order"], row["diagram_idx"])
        expected = "heavy_tail" if key in result.qmc_diagnostics else "rqmc"
        assert row["error_kind"] == expected

    raw = yaml.safe_load(QUICKSTART.read_text())
    raw["propagators"]["c_closed_form_only"] = True
    raw["sweep"].update(
        qmc_heavy_tail=True, qmc_replicas=REPLICAS, n_samples=N_SAMPLES, seed=42,
        positions_grid={"x": [0.0], "y": [0.5]}, t_final_grid=[3.0],
        component_pairs=[[0, 0]],
    )
    raw["output"] = [
        {"type": "npz", "path": str(tmp_path / "tail.npz")},
        {"type": "table", "format": "csv", "path": str(tmp_path / "tail.csv")},
    ]
    path = tmp_path / "tail.yaml"
    path.write_text(yaml.safe_dump(raw))
    sweep, totals = run_workflow(load_workflow_config(path), progress=False)
    assert list(totals["error_kind"]) == ["rqmc", "heavy_tail"]
    assert math.isnan(totals["error"].iloc[-1])
    with np.load(tmp_path / "tail.npz", allow_pickle=False) as out:
        assert math.isnan(out["error"][-1])
        assert out["error_kind"][-1] == "heavy_tail"
    assert ",,heavy_tail" in (tmp_path / "tail.csv").read_text()
    records = _normalize_rows(_workflow_rows(sweep, totals))
    assert records[-1]["error"] is None
    payload = encode_results(sweep, totals)
    assert payload["totals_records"][-1]["error"] is None
    assert compare_results(payload, payload)["exact"]
    malformed = deepcopy(payload)
    malformed["totals_records"][-1]["error"] = 0.0
    assert not compare_results(payload, malformed)["metadata_equal"]
    bad_totals = totals.copy()
    bad_totals.loc[bad_totals.index[-1], "error"] = 0.0
    with pytest.raises(ValueError, match="heavy_tail requires an unavailable error"):
        _workflow_rows(sweep, bad_totals)

    import matplotlib.pyplot as plt

    fig = sweep.plot(x="y", y="error")
    try:
        assert math.isnan(fig.axes[0].lines[-1].get_ydata()[0])
    finally:
        plt.close(fig)


@pytest.mark.parametrize("scale", [1e-250, 1.0, 1e250])
def test_weight_diagnostics_merge_signed_samples_across_scrambles(scale):
    from scipy.stats import skew

    from sft_wick._qmc_diagnostics import (
        _is_heavy_tail, _tail_diagnostics, _weight_moments,
    )

    weights = scale * np.array([[10.0, -1.0], [0.0, 0.0], [-2.0, 3.0]])
    estimates = tuple(row.mean() for row in weights)
    observed = _tail_diagnostics([_weight_moments(row) for row in weights], estimates)
    assert observed.max_weight_share == pytest.approx(10 / 16, rel=1e-14, abs=0.0)
    assert observed.effective_samples == pytest.approx(16**2 / 114, rel=1e-14, abs=0.0)
    reference = skew([4.5, 0.0, 0.5], bias=False)
    assert observed.scramble_skew == pytest.approx(reference, rel=1e-14, abs=0.0)
    assert _is_heavy_tail(observed)
    assert _tail_diagnostics([_weight_moments(np.zeros(2))], (0.0,)) is None


def test_heavy_tail_flag_requires_independent_scrambles(tmp_path):
    raw = yaml.safe_load(QUICKSTART.read_text())
    raw["sweep"]["qmc_heavy_tail"] = True
    path = tmp_path / "tail.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="qmc_heavy_tail: requires qmc_replicas"):
        load_workflow_config(path)
