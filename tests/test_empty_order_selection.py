r"""An order at which the expansion holds no diagram is a computed zero.

With a local cubic ``F`` alone, ``⟨φ_a(x) φ_b(y)⟩`` has no diagram at order
1 (five fields cannot pair).  Under the moment target a sweep that selected
only that order returned no row, and ``SweepResult.totals`` raised
``KeyError: 'x'`` after the propagators and the sweep had run: at L1, and
through ``--override sweep.orders=1``, whose up-front check accepted the
order.  A per-order scan hit it at every odd order.  Such an order now
gets a zero row per grid point, as under the connected target, and
``Expansion.evaluate`` lists it in ``by_order`` as 0.0.

The model is generic (distinct rates, ``F`` with entries of both signs and
no index symmetry, ``t_min = 0.3``, distinct points); order 0 is checked
against the closed-form covariance of
``tests/test_observable_operator_inputs.py``, which shares no code with the
package.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

import sft_wick as sw
from sft_wick.workflow import specs as sp
from sft_wick.workflow.cli import _parse_overrides
from sft_wick.workflow.config import load_workflow_config, run_workflow

from .test_observable_operator_inputs import (
    F_TENSOR,
    GAMMAS,
    LAM,
    SIGMA_T,
    SIGMA_X,
    T_FINAL,
    T_MIN,
    X,
    Y,
    _c_reference,
)

os.environ.setdefault("SFT_WICK_QUIET_CACHE", "1")

GL = dict(method="gauss_legendre", n_gauss=6)
TUPLES = [(1, 0), (1, 1)]


@pytest.fixture(scope="module")
def system():
    return sw.System(
        field=sp.FieldSpec("phi", n_components=2),
        linear=sp.DiagonalA(gamma=list(GAMMAS)),
        noise=sp.GaussianNoise(kappa2=sp.SeparableTranslation(
            temporal=sp.ExponentialTemporal(lam=LAM, sigma_t=SIGMA_T),
            spatial=sp.GaussianSpatial(sigma_x=SIGMA_X))),
        vertices=(sp.LocalVertex("F", F_TENSOR),),
        t_min=T_MIN,
    )


@pytest.fixture(scope="module")
def expansion(system):
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 1, 2],
                        progress=False)
    assert [len(exp.diagrams(o)) for o in (0, 1, 2)] == [1, 0, 6]
    return exp


@pytest.fixture(scope="module")
def props(system):
    return system.propagators(t_max=2.0, n_grid_t=12, c_closed_form_only=True,
                              progress=False)


def _sweep(expansion, props, **kw):
    return expansion.sweep(props, positions_grid={"x": [X], "y": [Y, -Y]},
                           t_final_grid=[T_FINAL], component_tuples=TUPLES,
                           **GL, **kw)


def _order_values(totals, order):
    rows = totals[totals["order"] == order]
    return sorted(zip(rows["y"], rows["a"], rows["b"], rows["value"]))


@pytest.mark.parametrize("orders", [1, [1]], ids=["int", "list"])
@pytest.mark.parametrize("reuse", [False, True], ids=["fresh", "reuse"])
def test_EO1_an_empty_order_alone_gives_zero_rows(expansion, props, orders,
                                                  reuse):
    totals = _sweep(expansion, props, orders=orders,
                    reuse_integrands=reuse).totals()
    assert len(totals) == 4  # two points of y times two tuples
    assert set(totals["order"]) == {1}
    assert (totals["value"] == 0.0).all()


def test_EO1_an_empty_order_next_to_others(expansion, props):
    sweep = _sweep(expansion, props, orders=[0, 1, 2])
    totals = sweep.totals()
    assert sorted(set(totals["order"])) == [0, 1, 2]
    assert all(v == 0.0 for *_, v in _order_values(totals, 1))
    for y, a, b, value in _order_values(totals, 0):
        want = _c_reference(a, b, T_FINAL, T_FINAL, X, y)
        assert value == pytest.approx(want, rel=1e-12, abs=1e-15), (y, a, b)
    placeholders = [r for r in sweep.rows if r["order"] == 1]
    assert len(placeholders) == 4
    assert all(r["diagram_idx"] is None and r["vertex_type"] is None
               for r in placeholders)


def test_EO1_evaluate_lists_the_empty_order(expansion, props):
    res = expansion.evaluate(props, positions={"x": X, "y": Y},
                             t_final=T_FINAL, component_pair=(1, 1), orders=1,
                             **GL)
    assert res.by_order == {1: 0.0} and res.total == 0.0
    assert res.per_diagram == []
    both = expansion.evaluate(props, positions={"x": X, "y": Y},
                              t_final=T_FINAL, component_pair=(1, 1),
                              orders=[0, 1], **GL)
    assert sorted(both.by_order) == [0, 1] and both.by_order[1] == 0.0
    assert both.by_order[0] == pytest.approx(
        _c_reference(1, 1, T_FINAL, T_FINAL), rel=1e-12, abs=0.0)


def test_EO2_a_channel_selection_keeps_the_empty_order(expansion, props):
    """Order 1 holds no diagram of any channel, so it is a zero next to the
    F channel of order 2.  A label no diagram at the selected orders
    carries is still refused."""
    totals = _sweep(expansion, props, orders=[1, 2], vertex_types="F").totals()
    assert sorted(set(totals["order"])) == [1, 2]
    assert all(v == 0.0 for *_, v in _order_values(totals, 1))
    assert all(v != 0.0 for *_, v in _order_values(totals, 2))
    with pytest.raises(ValueError, match=r"no diagram at order\(s\) \[1\]"):
        _sweep(expansion, props, orders=1, vertex_types="F")


def _config() -> dict:
    return {
        "system": {
            "field": {"name": "phi", "n_components": 2},
            "t_min": T_MIN,
            "linear": {"type": "diagonal", "gamma": list(GAMMAS)},
            "vertices": [{"name": "F", "coupling": F_TENSOR.tolist()}],
            "noise": {"kappa2": {
                "type": "separable_translation",
                "temporal": {"type": "exponential", "lam": LAM,
                             "sigma_t": SIGMA_T},
                "spatial": {"type": "gaussian", "sigma_x": SIGMA_X}}},
        },
        "expand": {"observable": ["phi_a(x)", "phi_b(y)"],
                   "orders": [0, 1, 2]},
        "propagators": {"t_max": 2.0, "n_grid_t": 12},
        "sweep": {"positions_grid": {"x": [X], "y": [Y]},
                  "t_final_grid": [T_FINAL],
                  "component_tuples": [list(t) for t in TUPLES], **GL},
    }


@pytest.mark.parametrize("order", [1, 3])
def test_EO3_cli_override_of_an_empty_order(tmp_path: Path, order):
    """``--override sweep.orders=N`` at an order with no diagram, as a
    per-order scan does; order 3 is the next odd order."""
    cfg = _config()
    cfg["expand"]["orders"] = [0, 1, 2, 3]
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    loaded = load_workflow_config(
        path, overrides=_parse_overrides([f"sweep.orders={order}"]))
    _sweep_result, totals = run_workflow(loaded, progress=False)
    assert list(totals["order"]) == [order, order]
    assert (totals["value"] == 0.0).all()
