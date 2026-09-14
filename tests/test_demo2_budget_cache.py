"""A saved demo2 channel must satisfy the current grid and accuracy target."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "examples/paper_assets/demo2_kappa4"))
import run_budget as budget


@pytest.fixture
def channel(monkeypatch, tmp_path):
    monkeypatch.setattr(budget, "CACHE", tmp_path)
    monkeypatch.setattr(
        budget, "reference_moments",
        lambda times, r, channels: {channels[0]: np.full((len(times), 3), r + 1)},
    )
    state = {"calls": 0, "relative_error": 0.0, "incomplete": False}

    def sweep(system, props, orders, vertex_types, rs, ts, pairs, method, **kw):
        state["calls"] += 1
        rows = [dict(y=r, t_final=t, a=a, b=b, vertex_type=vertex_types[0],
                     order=orders[0], value=(r + 1)*(1 + state["relative_error"]),
                     seconds=1.0)
                for r in rs for t in ts for a, b in pairs]
        df = pd.DataFrame(rows)
        return df.iloc[:0] if state["incomplete"] else df

    monkeypatch.setattr(budget, "sweep_rows", sweep)

    def run(rs=(0.0,), rtol=1e-4):
        return budget.verified_rows(None, None, 4, ["FK"], "fffk", rs, [1.0], rtol=rtol)

    return run, state, tmp_path


def test_cache_reuse_preserves_package_value(channel):
    run, state, _ = channel
    state["relative_error"] = 5e-5
    first = run()
    second = run()
    assert state["calls"] == 1
    pd.testing.assert_frame_equal(first, second)
    assert np.all(second.value != second.reference)


def test_changed_spatial_grid_requires_new_rows(channel):
    run, state, _ = channel
    run()
    result = run(rs=(0.0, 0.6))
    assert state["calls"] == 2
    assert len(result) == 6
    assert set(result.y) == {0.0, 0.6}


def test_stricter_accuracy_target_rechecks_cached_values(channel):
    run, state, _ = channel
    state["relative_error"] = 5e-5
    run()
    state["relative_error"] = 0.0
    result = run(rtol=1e-7)
    assert state["calls"] == 2
    np.testing.assert_array_equal(result.value, result.reference)


@pytest.mark.parametrize("damage", ["value", "nan", "missing", "duplicate", "channel"])
def test_inaccurate_or_incomplete_cache_is_recomputed(channel, damage):
    run, state, cache = channel
    run()
    path, = cache.glob("*.pkl")
    rows = pd.read_pickle(path)
    if damage == "value":
        rows.loc[0, "value"] = 2.0
    elif damage == "nan":
        rows.loc[0, "value"] = np.nan
    elif damage == "missing":
        rows = rows.iloc[:-1]
    elif damage == "duplicate":
        rows = pd.concat([rows.iloc[:2], rows.iloc[:1]], ignore_index=True)
    else:
        rows.loc[0, "vertex_type"] = "F"
    rows.to_pickle(path)
    result = run()
    assert state["calls"] == 2
    assert len(result) == 3
    np.testing.assert_array_equal(result.value, result.reference)


def test_empty_computation_cannot_pass_verification(channel):
    run, state, cache = channel
    state["incomplete"] = True
    with pytest.raises(RuntimeError, match="did not meet"):
        run()
    assert not list(cache.glob("*.pkl"))


@pytest.mark.parametrize("reference", [np.inf, np.nan])
def test_nonfinite_reference_cannot_certify_values(channel, monkeypatch, reference):
    run, _, cache = channel
    monkeypatch.setattr(
        budget, "reference_moments",
        lambda times, r, channels: {channels[0]: np.full((len(times), 3), reference)},
    )
    with pytest.raises(RuntimeError, match="did not meet"):
        run()
    assert not list(cache.glob("*.pkl"))
