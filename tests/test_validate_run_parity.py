"""``sft-wick validate`` reports what ``sft-wick run`` would refuse.

VR1 -- each refusal that used to come after the expansion, or in the sweep,
       stops ``validate``, ``run --dry-run`` and ``run`` with exit status 2,
       one line, and the same code and dotted path, before any stage runs.
VR2 -- a config next to each refusal passes, so the checks do not refuse
       what the run accepts.
VR3 -- which external points a response propagator joins at order 0.  The
       check before the run and the integrators' own refusal are compared
       directly, over observables, targets, selections and methods.
VR4 -- what only the diagrams decide stops ``run --dry-run`` and ``run``
       after the expansion, with status 2 and one line.  ``validate`` runs no
       expansion, so it passes these.
VR5 -- the closed-form C module is imported before the expansion, and the
       run uses the callable it exports.
VR6 -- a ``plot`` output names a column of the sweep; every column works.
VR7 -- every position on an axis must broadcast with every position on
       another axis; alternatives on the same axis need not broadcast.
VR8 -- validating a high-point observable takes one pairing per R
       assignment, with repeated points as with distinct ones, and still
       refuses the points an R joins.

Before these checks VR1 and VR5 ended with ``validate`` reporting the file
valid and ``run`` in a traceback, after the expansion, the propagator table
or the whole sweep.  The reference in VR3 is the refusal of
``sft_wick.evaluate``, reached through the public workflow; the reference in
VR5 is a run with the closed form passed to ``System.propagators``.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from math import perm
from pathlib import Path

import numpy as np
import pytest
import yaml

from sft_wick import perturbation
from sft_wick.workflow import _run_checks
from sft_wick.workflow.cli import main
from sft_wick.workflow.config import (ConfigError, _parse_workflow,
                                      check_before_run, run_workflow)
from sft_wick.workflow.system import System

ROOT = Path(__file__).resolve().parents[1]

DENSE_R = """
import numpy as np


def R_time(t1, t2):
    if t1 < t2:
        return np.zeros((2, 2))
    return np.array([[1.0, 0.3], [0.2, 1.0]]) * np.exp(-(t1 - t2))
"""

#: The closed form of the quick start's C at equal arguments of a toy model:
#: any callable of the documented signature serves the checks below.
C_MODULE = """
import numpy as np


def C_fn(n1, t1, n2, t2):
    return 0.1 * np.exp(-abs(t1 - t2)) * np.eye(2)
"""


#: A κ² of positions in R^3 times a matrix; no scalar position reads it.
VECTOR_KAPPA2 = """
import numpy as np

M = np.array({matrix})


def kappa2(n1, t1, n2, t2):
    if np.shape(n1) != (3,) or np.shape(n2) != (3,):
        raise TypeError("positions are vectors of three coordinates")
    apart = np.linalg.norm(np.subtract(n1, n2))
    m = M if min(t1, t2) >= {mixes_from} else np.diag(np.diag(M))
    return np.exp(-abs(t1 - t2) - apart) * m
"""


RATE_MODULE = """
import numpy as np


def gamma(t):
    return np.array([0.7 + 0.5 * np.sin(3.0 * t), 1.9])
"""


def _doc() -> dict:
    """The quick start at order 0, small enough to run in a test."""
    doc = yaml.safe_load((ROOT / "examples" / "quickstart.yaml").read_text())
    doc["expand"]["orders"] = [0]
    doc["propagators"].update(t_max=3.0, n_grid_t=6)
    doc["sweep"].update(n_samples=16, t_final_grid=[1.0, 2.0],
                        positions_grid={"x": [0.0], "y": [0.0, 0.5]})
    doc["output"] = []
    return doc


def _write(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def _file(tmp_path: Path, name: str, text: str) -> None:
    (tmp_path / name).write_text(text)


def _set(*path):
    """The key at ``path[:-1]`` set to the value ``path[-1]``."""
    def apply(doc, tmp_path):
        node = doc
        for step in path[:-2]:
            node = node[step]
        node[path[-2]] = path[-1]
    return apply


def _all(*steps):
    def apply(doc, tmp_path):
        for step in steps:
            step(doc, tmp_path)
    return apply


def _dense_r(doc, tmp_path):
    _file(tmp_path, "r.py", DENSE_R)
    doc["system"]["linear"] = {"type": "explicit", "R_time_module": "r.py",
                               "iso_R": False}
    doc["system"]["vertices"] = []


def _rate_spline(doc, tmp_path):
    """A callable rate on a spline ending at 4, and no C table to bound a
    time."""
    _file(tmp_path, "rate.py", RATE_MODULE)
    _file(tmp_path, "c_fn.py", C_MODULE)
    doc["system"]["linear"] = {"type": "diagonal", "gamma_module": "rate.py",
                               "t_max_cache": 4.0, "n_grid_cache": 201}
    doc["propagators"].update(c_closed_form_module="c_fn.py",
                              c_closed_form_only=True)


def _vector_kappa2(matrix, mixes_from=0.0, t_final=(1.0,), pinned=None):
    """The κ² above with ``matrix``, whose off-diagonal entries start at the
    time ``mixes_from``, read at two positions in R^3 and at ``t_final``, or
    at the ``pinned`` times of the two points."""
    def apply(doc, tmp_path):
        _file(tmp_path, "k2.py",
              VECTOR_KAPPA2.format(matrix=matrix, mixes_from=mixes_from))
        doc["system"]["noise"] = {"kappa2": {"type": "callable_module",
                                             "module": "k2.py",
                                             "attr": "kappa2"}}
        doc["sweep"]["positions_grid"] = {"x": [[0.3, -0.2, 0.9]],
                                          "y": [[-0.5, 0.4, 0.1]]}
        doc["sweep"]["t_final_grid"] = list(t_final)
        if pinned is not None:
            doc["sweep"]["external_times_grid"] = pinned
        doc["propagators"].update(t_max=1.0, n_grid_t=5)
    return apply


def _module(name: str, text: str, *steps):
    def apply(doc, tmp_path):
        _file(tmp_path, name, text)
        doc["propagators"]["c_closed_form_module"] = name
        for step in steps:
            step(doc, tmp_path)
    return apply


def _plot(**keys):
    return _set("output", [{"type": "plot", "path": "p.png", "x": "y", **keys}])


def _no_stage(*args, **kwargs):
    raise AssertionError("a stage ran for a config the checks should refuse")


_RESPONSE = _all(_set("expand", "observable", ["phi_a(x)", "psi_b(y)"]),
                 _set("sweep", "positions_grid", {"x": [0.0], "y": [0.5]}))

_ONE_POSITION = _set("sweep", "positions_grid", {"x": [0.0], "y": [0.0]})

#: name -> (change to the config, code, dotted path)
REFUSED = {
    "t_final_past_the_table": (_set("sweep", "t_final_grid", [1.0, 4.0]),
                               "invalid_value", "sweep.t_final_grid[1]"),
    "t_final_below_the_table": (
        _all(_set("system", "t_min", 0.5),
             _set("sweep", "t_final_grid", [0.25, 1.0])),
        "invalid_value", "sweep.t_final_grid[0]"),
    "pinned_time_past_the_table": (
        _set("sweep", "external_times_grid", {"x": [4.0], "y": [1.0]}),
        "invalid_value", "sweep.external_times_grid.x[0]"),
    "separation_past_the_table": (
        _all(_set("propagators", "r_max", 0.25),
             _set("propagators", "n_grid_r", 4)),
        "invalid_value", "sweep.positions_grid.y[1]"),
    "position_past_the_table": (
        _all(_set("propagators", "homogeneity", "general"),
             _set("propagators", "x_max", 0.25),
             _set("propagators", "n_grid_x", 4)),
        "invalid_value", "sweep.positions_grid.y[1]"),
    "pinned_time_past_the_rate_spline": (
        _all(_rate_spline, _RESPONSE, _ONE_POSITION,
             _set("sweep", "external_times_grid", {"x": [5.0], "y": [1.0]})),
        "invalid_value", "sweep.external_times_grid.x[0]"),
    "collect_topology_off_at_an_interacting_order": (
        _all(_set("expand", "orders", [0, 2]),
             _set("expand", "collect_topology", False)),
        "invalid_value", "expand.collect_topology"),
    "collect_topology_off_with_the_sweep_at_order_0": (
        _all(_set("expand", "orders", [0, 2]),
             _set("expand", "collect_topology", False),
             _set("sweep", "orders", [0])),
        "invalid_value", "expand.collect_topology"),
    "dense_R_under_diag_R": (
        _all(_dense_r, _set("expand", "diag_C", False),
             _set("propagators", "diag_C", False)),
        "invalid_value", "expand.diag_R"),
    "dense_R_under_diag_C": (
        _all(_dense_r, _set("expand", "diag_R", False)),
        "invalid_value", "expand.diag_C"),
    "dense_R_under_propagators_diag_C": (
        _all(_dense_r, _set("expand", "diag_R", False),
             _set("expand", "diag_C", False)),
        "invalid_value", "propagators.diag_C"),
    "mixing_kappa2_of_vectors_under_diag_C": (
        _vector_kappa2([[1.0, 0.3], [0.3, 0.8]]),
        "invalid_value", "expand.diag_C"),
    "mixing_kappa2_of_vectors_under_propagators_diag_C": (
        _all(_vector_kappa2([[1.0, 0.3], [0.3, 0.8]], mixes_from=0.6,
                            t_final=[0.3],
                            pinned={"x": [0.3, 1.0], "y": [1.0]}),
             _set("expand", "diag_C", False)),
        "invalid_value", "propagators.diag_C"),
    "empty_t_final_grid": (_set("sweep", "t_final_grid", []),
                           "invalid_value", "sweep.t_final_grid"),
    "empty_position_axis": (_set("sweep", "positions_grid", "y", []),
                            "invalid_value", "sweep.positions_grid.y"),
    "empty_time_axis": (
        _set("sweep", "external_times_grid", {"x": [], "y": [1.0]}),
        "invalid_value", "sweep.external_times_grid.x"),
    "repeated_t_final": (_set("sweep", "t_final_grid", [1.0, 1.0]),
                         "invalid_value", "sweep.t_final_grid"),
    "repeated_position": (_set("sweep", "positions_grid", "y", [0.5, 0.5]),
                          "invalid_value", "sweep.positions_grid.y"),
    "repeated_time": (
        _set("sweep", "external_times_grid", {"x": [1.0, 1.0]}),
        "invalid_value", "sweep.external_times_grid.x"),
    "pinned_and_integrated": (
        _all(_set("sweep", "external_times_grid", {"x": [1.0]}),
             _set("sweep", "integrate_over", ["x"])),
        "invalid_value", "sweep.integrate_over"),
    "pinned_and_integrated_all": (
        _all(_set("sweep", "external_times_grid", {"x": [1.0]}),
             _set("sweep", "integrate_over", "all")),
        "invalid_value", "sweep.integrate_over"),
    "response_leg_at_another_position": (
        _RESPONSE, "invalid_value", "sweep.positions_grid.y"),
    "response_leg_on_a_position_axis": (
        _all(_RESPONSE, _set("sweep", "positions_grid",
                             {"x": [0.0], "y": [0.0, 0.5]})),
        "invalid_value", "sweep.positions_grid.y"),
    "plot_x": (_plot(x="nope"), "invalid_value", "output[0].x"),
    "plot_y": (_plot(y="nope"), "invalid_value", "output[0].y"),
    "plot_hue": (_plot(hue="nope"), "invalid_value", "output[0].hue"),
    "plot_facet_col": (_plot(facet_col="nope"),
                       "invalid_value", "output[0].facet_col"),
    "plot_x_not_a_name": (_plot(x=["y"]), "invalid_value", "output[0].x"),
    "c_module_missing": (
        _set("propagators", "c_closed_form_module", "absent.py"),
        "read_error", "propagators.c_closed_form_module"),
    "c_module_syntax_error": (
        _module("c.py", "def (:\n"),
        "read_error", "propagators.c_closed_form_module"),
    "c_module_raises": (
        _module("c.py", "raise RuntimeError('at import')\n"),
        "read_error", "propagators.c_closed_form_module"),
    "c_module_no_attribute": (
        _module("c.py", "y = 1\n"),
        "read_error", "propagators.c_closed_form_attr"),
    "c_module_attribute_not_callable": (
        _module("c.py", "C_fn = 1\n"),
        "read_error", "propagators.c_closed_form_attr"),
    "c_closed_form_only_with_no_closed_form": (
        _all(_set("propagators", "c_closed_form", None),
             _set("propagators", "c_closed_form_only", True)),
        "invalid_value", "propagators.c_closed_form_only"),
    "c_closed_form_vectorized_alone": (
        _set("propagators", "c_closed_form_vectorized", True),
        "invalid_value", "propagators.c_closed_form_vectorized"),
    "t_max_at_t_min": (_set("propagators", "t_max", 0.0),
                       "invalid_value", "propagators.t_max"),
    "t_max_below_t_min": (_set("propagators", "t_max", -1.0),
                          "invalid_value", "propagators.t_max"),
    "lazy_table_of_three_time_nodes": (_set("propagators", "n_grid_t", 3),
                                       "invalid_value", "propagators.n_grid_t"),
    "number_next_to_a_vector_on_one_axis": (
        _set("sweep", "positions_grid", "y", [0.0, [0.5, 0.5]]),
        "invalid_value", "sweep.positions_grid.y[1]"),
    "number_next_to_a_vector_on_a_translation_table": (
        _all(_set("sweep", "positions_grid", {"x": [0.0], "y": [[0.5, 0.5]]}),
             _set("expand", "orders", [0, 2]),
             _set("propagators", "homogeneity", "translation"),
             _set("propagators", "r_max", 2.0),
             _set("propagators", "n_grid_r", 4)),
        "invalid_value", "sweep.positions_grid.y"),
    "vectors_of_two_lengths": (
        _set("sweep", "positions_grid",
             {"x": [[0.0, 0.0, 0.0]], "y": [[0.5, 0.5]]}),
        "invalid_value", "sweep.positions_grid"),
    "vector_on_a_general_table": (
        _all(_set("sweep", "positions_grid", {"x": [[0.0, 0.0]], "y": [[0.5, 0.5]]}),
             _set("propagators", "homogeneity", "general"),
             _set("propagators", "x_max", 2.0),
             _set("propagators", "n_grid_x", 4)),
        "invalid_value", "sweep.positions_grid.x"),
    "cubic_table_of_three_time_nodes": (
        _all(_set("propagators", "homogeneity", "translation"),
             _set("propagators", "r_max", 2.0),
             _set("propagators", "n_grid_r", 4),
             _set("propagators", "interp_method", "cubic"),
             _set("propagators", "n_grid_t", 3)),
        "invalid_value", "propagators.n_grid_t"),
    "quintic_table_of_five_separations": (
        _all(_set("propagators", "homogeneity", "translation"),
             _set("propagators", "r_max", 2.0),
             _set("propagators", "n_grid_r", 5),
             _set("propagators", "interp_method", "quintic")),
        "invalid_value", "propagators.n_grid_r"),
    "output_path_is_a_directory": (
        lambda doc, tmp_path: doc.update(
            output=[{"type": "table", "path": str(tmp_path)}]),
        "invalid_value", "output[0].path"),
    "the_file_of_an_npz_output_is_a_directory": (
        lambda doc, tmp_path: (
            (tmp_path / "results.npz").mkdir(),
            doc.update(output=[{"type": "npz", "path": "results"}])),
        "invalid_value", "output[0].path"),
    "the_file_of_a_plot_output_is_a_directory": (
        lambda doc, tmp_path: (
            (tmp_path / "figs.png").mkdir(),
            doc.update(output=[{"type": "plot", "path": "figs", "x": "y"}])),
        "invalid_value", "output[0].path"),
    "plot_suffix_with_no_format": (
        _plot(path="p.xyz"), "invalid_value", "output[0].path"),
    "output_path_below_a_file": (
        lambda doc, tmp_path: (
            _file(tmp_path, "plain.txt", ""),
            doc.update(output=[{"type": "npz", "path": "plain.txt/t.npz"}])),
        "invalid_value", "output[0].path"),
    "output_path_too_long": (
        _set("output", [{"type": "table", "path": "x" * 5000}]),
        "invalid_value", "output[0].path"),
    "cache_path_too_long": (_set("expand", "cache_path", "x" * 5000),
                            "invalid_value", "expand.cache_path"),
}


@pytest.mark.parametrize("name", REFUSED)
def test_VR1_every_command_refuses_before_any_stage(tmp_path, capsys,
                                                    monkeypatch, name):
    change, code, dotted = REFUSED[name]
    doc = _doc()
    change(doc, tmp_path)
    path = _write(tmp_path, doc)
    monkeypatch.setattr(System, "expand", _no_stage)
    monkeypatch.setattr(System, "propagators", _no_stage)
    monkeypatch.chdir(tmp_path)
    for command in (["validate", str(path)],
                    ["run", str(path), "--dry-run", "-q"],
                    ["run", str(path), "-q"]):
        status = main(command)
        lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.strip()]
        assert status == 2, command
        assert len(lines) == 1, lines
        assert lines[0].startswith(
            f"sft-wick: config error [{code}] {dotted}: "), lines[0]


# ---------------------------------------------------------------------------
# VR2 -- the neighbours of each refusal
# ---------------------------------------------------------------------------

ACCEPTED = {
    "kappa2_of_vectors_mixing_past_the_pinned_times_under_diag_C":
        _vector_kappa2([[1.0, 0.3], [0.3, 0.8]], mixes_from=0.6,
                       t_final=[1.0], pinned={"x": [0.5], "y": [0.5]}),
    "t_final_at_the_end_of_the_table": _set("sweep", "t_final_grid", [1.0, 3.0]),
    "unread_t_final_past_the_table": _all(
        _set("sweep", "t_final_grid", [9.0]),
        _set("sweep", "external_times_grid", {"x": [2.0], "y": [1.0]})),
    "separation_at_the_end_of_the_table": _all(
        _set("propagators", "r_max", 0.5),
        _set("propagators", "n_grid_r", 4)),
    "no_table_past_t_max": _all(
        _set("propagators", "c_closed_form_only", True),
        _set("sweep", "t_final_grid", [9.0])),
    "order_zero_of_phi_past_the_rate_spline": _all(
        _rate_spline, _set("sweep", "t_final_grid", [9.0])),
    "response_at_the_end_of_the_rate_spline": _all(
        _rate_spline, _RESPONSE, _ONE_POSITION,
        _set("sweep", "external_times_grid", {"x": [4.0], "y": [1.0]})),
    "collect_topology_off_at_order_0": _set("expand", "collect_topology", False),
    "dense_R_with_both_flags_off": _all(
        _dense_r, _set("expand", "diag_R", False),
        _set("expand", "diag_C", False), _set("propagators", "diag_C", False)),
    "propagators_diag_C_off_alone": _set("propagators", "diag_C", False),
    "distinct_values_on_every_axis": _all(
        _set("sweep", "positions_grid", {"x": [0.0, 0.25], "y": [0.5, 1.0]}),
        _set("sweep", "external_times_grid", {"x": [1.0, 2.0]})),
    "pinned_and_integrated_at_two_points": _all(
        _set("sweep", "external_times_grid", {"x": [1.0]}),
        _set("sweep", "integrate_over", ["y"])),
    "response_leg_at_one_position": _all(
        _set("expand", "observable", ["phi_a(x)", "psi_b(y)"]),
        _set("sweep", "positions_grid", {"x": [0.5], "y": [0.5]}),
        _set("sweep", "external_times_grid", {"x": [2.0], "y": [1.0]})),
    "two_physical_legs_apart": _set("sweep", "positions_grid",
                                    {"x": [0.0], "y": [0.5]}),
    "full_table_of_two_time_nodes": _all(
        _set("propagators", "n_grid_t", 2),
        _set("propagators", "homogeneity", "translation"),
        _set("propagators", "r_max", 2.0), _set("propagators", "n_grid_r", 4)),
    "lazy_table_of_four_time_nodes": _set("propagators", "n_grid_t", 4),
    "vectors_at_every_point": _set("sweep", "positions_grid",
                                   {"x": [[0.0, 0.0]], "y": [[0.0, 0.5]]}),
    "number_next_to_a_vector_on_a_table_per_separation": _set(
        "sweep", "positions_grid", {"x": [0.0], "y": [[0.5, 0.0, 0.0]]}),
    "number_next_to_a_vector_at_order_0_of_a_translation_table": _all(
        _set("sweep", "positions_grid", {"x": [0.0], "y": [[0.5, 0.5]]}),
        _set("propagators", "homogeneity", "translation"),
        _set("propagators", "r_max", 2.0), _set("propagators", "n_grid_r", 4)),
    "number_next_to_a_vector_on_a_rotation_table": _all(
        _set("sweep", "positions_grid", {"x": [0.0], "y": [[0.6, 0.0, 0.8]]}),
        _set("expand", "orders", [0, 2]),
        _set("propagators", "homogeneity", "rotation"),
        _set("propagators", "n_grid_cos", 4)),
    "vectors_of_lengths_that_broadcast": _set(
        "sweep", "positions_grid", {"x": [[0.0]], "y": [[0.5, 0.0]]}),
    "vectors_on_a_general_grid_without_a_table": _all(
        _set("sweep", "positions_grid", {"x": [[0.0, 0.0]], "y": [[0.5, 0.0]]}),
        _set("propagators", "homogeneity", "general"),
        _set("propagators", "x_max", 2.0), _set("propagators", "n_grid_x", 4),
        _set("propagators", "c_closed_form_only", True)),
    "cubic_table_of_four_nodes": _all(
        _set("propagators", "homogeneity", "translation"),
        _set("propagators", "r_max", 2.0), _set("propagators", "n_grid_r", 4),
        _set("propagators", "interp_method", "cubic"),
        _set("propagators", "n_grid_t", 4)),
    "npz_output_named_like_a_directory": lambda doc, tmp_path: (
        (tmp_path / "results").mkdir(),
        doc.update(output=[{"type": "npz", "path": "results"}])),
    "plot_output_named_like_a_directory": lambda doc, tmp_path: (
        (tmp_path / "figs").mkdir(),
        doc.update(output=[{"type": "plot", "path": "figs", "x": "y"}])),
    "plot_output_with_a_format_of_its_own": _plot(path="p.PDF"),
    "cache_paths_that_are_directories": lambda doc, tmp_path: (
        doc["expand"].update(cache_path=str(tmp_path)),
        doc["propagators"].update(cache_path=str(tmp_path))),
}


@pytest.mark.parametrize("name", ACCEPTED)
def test_VR2_the_neighbour_of_a_refusal_runs(tmp_path, capsys, monkeypatch,
                                             name):
    doc = _doc()
    ACCEPTED[name](doc, tmp_path)
    path = _write(tmp_path, doc)
    monkeypatch.chdir(tmp_path)
    for command in (["validate", str(path)], ["run", str(path), "-q"]):
        status = main(command)
        assert status == 0, (command, capsys.readouterr().err)


# ---------------------------------------------------------------------------
# VR3 -- the check against the integrators, at the boundary
# ---------------------------------------------------------------------------

def _outcome(doc: dict, tmp_path: Path, run: bool) -> str:
    """``"refused"`` when the config is refused for joining points at two
    positions, else ``"ran"``: by the check before the run, or by the run
    with that check bypassed."""
    cfg = _parse_workflow(copy.deepcopy(doc), base_dir=tmp_path)
    try:
        if run:
            run_workflow(cfg, progress=False)
        else:
            check_before_run(cfg)
    except ConfigError as err:
        assert err.path.startswith("sweep.positions_grid."), err
        return "refused"
    except ValueError as err:
        assert "different positions" in str(err), err
        return "refused"
    return "ran"


def _response_doc(observable, positions, *, target="moment",
                  method="gauss_legendre") -> dict:
    doc = _doc()
    doc["system"]["vertices"] = []
    doc["expand"].update(observable=observable, target=target)
    points = sorted({op[op.index("(") + 1:-1] for op in observable})
    letters = sorted({op[op.index("_") + 1] for op in observable})
    doc["sweep"].update(positions_grid={p: [positions.get(p, 0.0)]
                                        for p in points},
                        t_final_grid=[1.0], method=method, n_gauss=2)
    doc["sweep"].pop("component_pairs")
    doc["sweep"]["component_tuples"] = [[0] * len(observable)]
    assert letters
    return doc


OBSERVABLES = [
    ["phi_a(x)", "psi_b(y)"],
    ["psi_a(x)", "phi_b(y)"],
    ["phi_a(x)", "phi_b(y)"],
    ["psi_a(x)", "psi_b(y)"],
    ["phi_a(x)", "phi_b(y)", "psi_c(z)"],
    ["phi_a(x)", "phi_b(y)", "phi_c(z)", "psi_d(u)"],
    ["phi_a(x)", "phi_b(y)", "psi_c(z)", "psi_d(u)"],
    ["phi_a(x)", "psi_b(y)", "psi_c(z)", "psi_d(u)"],
    # No order-0 diagram: an equal-point R vanishes under Ito, and the two
    # crossed R form a causal loop.
    ["phi_a(x)", "psi_b(x)", "phi_c(y)", "psi_d(y)"],
]


def _apart(observable) -> list:
    """Each point moved off the others, one at a time, and none moved."""
    points = sorted({op[op.index("(") + 1:-1] for op in observable})
    return [{}] + [{p: 0.5} for p in points]


@pytest.mark.parametrize("target", ["moment", "connected"])
@pytest.mark.parametrize("observable", OBSERVABLES, ids=" ".join)
def test_VR3_the_check_refuses_exactly_what_the_integrators_refuse(
        tmp_path, observable, target):
    """Order 0, every point on or off the others, both targets.  A config
    the integrators refuse and the check passes would stop a sweep; the
    reverse would refuse a run that works."""
    for moved in _apart(observable):
        doc = _response_doc(observable, moved, target=target)
        before = _outcome(doc, tmp_path, run=False)
        with_no_check = _run_past_the_check(doc, tmp_path)
        assert before == with_no_check, (observable, moved, target)


def _run_past_the_check(doc: dict, tmp_path: Path) -> str:
    """The run of ``doc`` with the position check of the config layer off,
    so what answers is the integrators' own refusal."""
    import sft_wick.workflow._run_checks as rc

    saved = rc._refuse_joined_points
    rc._refuse_joined_points = lambda *args: None
    try:
        return _outcome(doc, tmp_path, run=True)
    finally:
        rc._refuse_joined_points = saved


@pytest.mark.parametrize("method", ["qmc", "qmc_vectorized", "gauss_legendre"])
def test_VR3_every_integrator_refuses_what_the_check_refuses(tmp_path, method):
    doc = _response_doc(["phi_a(x)", "psi_b(y)"], {"y": 0.5}, method=method)
    assert _outcome(doc, tmp_path, run=False) == "refused"
    assert _run_past_the_check(doc, tmp_path) == "refused"


@pytest.mark.parametrize("homogeneity,grid", [
    ("translation", {"r_max": 2.0, "n_grid_r": 4}),
    ("general", {"x_max": 2.0, "n_grid_x": 4}),
    ("translation", {}),
    ("general", {}),
])
def test_VR3_every_table_kind_refuses_what_the_check_refuses(tmp_path,
                                                             homogeneity, grid):
    doc = _response_doc(["phi_a(x)", "psi_b(y)"], {"y": 0.5})
    doc["propagators"].update(homogeneity=homogeneity, c_closed_form=None, **grid)
    assert _outcome(doc, tmp_path, run=False) == "refused"
    assert _run_past_the_check(doc, tmp_path) == "refused"


def test_VR3_a_selection_without_order_0_is_left_to_the_diagrams(tmp_path):
    """``sweep.orders`` without 0 selects no order-0 diagram, and
    ``vertex_types`` can exclude it, so the check before the run says
    nothing; which order holds a diagram is known after the expansion."""
    doc = _response_doc(["phi_a(x)", "psi_b(y)"], {"y": 0.5})
    doc["expand"]["orders"] = [0, 1]
    doc["sweep"]["orders"] = [1]
    assert _outcome(doc, tmp_path, run=False) == "ran"


def test_VR3_a_vertex_type_filter_reaches_the_walk_of_the_diagrams(tmp_path):
    doc = _doc()
    doc["expand"].update(observable=["phi_a(x)", "psi_b(y)"], orders=[0, 2])
    doc["sweep"].update(positions_grid={"x": [0.0], "y": [0.5]},
                        vertex_types=["F"])
    assert _outcome(doc, tmp_path, run=False) == "refused"
    assert _run_past_the_check(doc, tmp_path) == "refused"




# ---------------------------------------------------------------------------
# VR4 -- after the expansion
# ---------------------------------------------------------------------------

def test_VR4_a_conflict_at_an_interacting_order_stops_the_run_early(
        tmp_path, capsys, monkeypatch):
    doc = _doc()
    doc["expand"].update(observable=["phi_a(x)", "psi_b(y)"], orders=[2])
    doc["sweep"]["positions_grid"] = {"x": [0.0], "y": [0.5]}
    path = _write(tmp_path, doc)
    monkeypatch.setattr(System, "propagators", _no_stage)
    for command in (["run", str(path), "--dry-run", "-q"],
                    ["run", str(path), "-q"]):
        assert main(command) == 2
        lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.strip()]
        assert len(lines) == 1, lines
        assert lines[0].startswith(
            "sft-wick: config error [invalid_value] sweep.positions_grid.y: ")


def test_VR4_a_vertex_type_label_stops_the_run_early(tmp_path, capsys,
                                                     monkeypatch):
    doc = _doc()
    doc["sweep"]["vertex_types"] = ["FF"]
    path = _write(tmp_path, doc)
    assert main(["validate", str(path)]) == 0
    capsys.readouterr()
    monkeypatch.setattr(System, "propagators", _no_stage)
    for command in (["run", str(path), "--dry-run", "-q"],
                    ["run", str(path), "-q"]):
        assert main(command) == 2
        lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.strip()]
        assert len(lines) == 1, lines
        assert lines[0].startswith(
            "sft-wick: config error [invalid_value] sweep.vertex_types: ")


def test_VR4_an_order_without_diagrams_is_not_a_conflict(tmp_path, capsys):
    """Order 1 of a cubic theory holds no diagram of ``<phi psi>``, so the
    run has no response leg to place and reports a computed zero."""
    doc = _doc()
    doc["expand"].update(observable=["phi_a(x)", "psi_b(y)"], orders=[1])
    doc["sweep"].update(positions_grid={"x": [0.0], "y": [0.5]},
                        component_pairs=[[0, 0]], t_final_grid=[1.0])
    path = _write(tmp_path, doc)
    assert main(["run", str(path), "-q"]) == 0, capsys.readouterr().err


# ---------------------------------------------------------------------------
# VR5 -- the closed-form C module
# ---------------------------------------------------------------------------

def test_VR5_the_run_uses_the_callable_the_check_loaded(tmp_path):
    _file(tmp_path, "c.py", C_MODULE)
    doc = _doc()
    doc["propagators"].update(c_closed_form_module="c.py",
                              c_closed_form_only=True)
    cfg = _parse_workflow(doc, base_dir=tmp_path)
    check_before_run(cfg)
    _, totals = run_workflow(cfg, progress=False)
    # <phi_0 phi_0> at order 0 is C_00 of the module, 0.1 at equal times.
    assert np.allclose(totals["value"].to_numpy(), 0.1, rtol=1e-12, atol=0.0)


def test_VR5_a_run_imports_the_module_once(tmp_path, capsys, monkeypatch):
    """The module may load a table or compile at import, and ``run`` needs
    it in two checks and in the run; the old code imported it once."""
    log = tmp_path / "imports.log"
    _file(tmp_path, "c.py", f"open({str(log)!r}, 'a').write('x')\n" + C_MODULE)
    doc = _doc()
    doc["propagators"].update(c_closed_form_module="c.py",
                              c_closed_form_only=True)
    path = _write(tmp_path, doc)
    monkeypatch.chdir(tmp_path)
    assert main(["run", str(path), "-q"]) == 0, capsys.readouterr().err
    assert log.read_text() == "x"


def test_VR5_a_changed_module_is_imported_again(tmp_path):
    doc = _doc()
    doc["propagators"].update(c_closed_form_module="c.py",
                              c_closed_form_only=True)
    cfg = _parse_workflow(doc, base_dir=tmp_path)
    values = []
    for amplitude in ("0.1", "0.25"):
        _file(tmp_path, "c.py", C_MODULE.replace("0.1", amplitude))
        _, totals = run_workflow(cfg, progress=False)
        values.append(totals["value"].iloc[0])
    assert values == pytest.approx([0.1, 0.25], rel=1e-12, abs=0.0)


def test_VR5_a_module_error_keeps_the_type_it_had(tmp_path):
    """``except ImportError`` and ``except AttributeError`` around a run
    keep working; the error is a ``ConfigError`` as well."""
    _file(tmp_path, "c.py", "raise RuntimeError('at import')\n")
    doc = _doc()
    doc["propagators"]["c_closed_form_module"] = "c.py"
    cfg = _parse_workflow(doc, base_dir=tmp_path)
    with pytest.raises(ImportError) as info:
        check_before_run(cfg)
    assert isinstance(info.value, ConfigError)
    assert "RuntimeError" in str(info.value)

    _file(tmp_path, "c.py", "y = 1\n")
    with pytest.raises(AttributeError) as info:
        check_before_run(cfg)
    assert isinstance(info.value, ConfigError)


@pytest.mark.parametrize("text", ["raise RuntimeError('at import')\n",
                                  "no_callable_named_so = 1\n"],
                         ids=["raises", "no_attribute"])
def test_VR5_a_refused_module_leaves_sys_modules_as_it_was(tmp_path, text):
    """A module named like the standard library's must not stay in its
    place."""
    import colorsys
    import sys

    _file(tmp_path, "colorsys.py", text)
    doc = _doc()
    doc["propagators"]["c_closed_form_module"] = "colorsys.py"
    cfg = _parse_workflow(doc, base_dir=tmp_path)
    with pytest.raises(ConfigError):
        check_before_run(cfg)
    assert sys.modules["colorsys"] is colorsys

    _file(tmp_path, "half_loaded_c.py", text)
    doc["propagators"]["c_closed_form_module"] = "half_loaded_c.py"
    cfg = _parse_workflow(doc, base_dir=tmp_path)
    with pytest.raises(ConfigError):
        check_before_run(cfg)
    assert "half_loaded_c" not in sys.modules


# ---------------------------------------------------------------------------
# VR6 -- plot columns
# ---------------------------------------------------------------------------

COLUMNS = ["x", "y", "t_final", "a", "b", "order", "diagram_idx", "vertex_type",
           "n_cross_C", "value", "error"]


def test_VR6_every_column_of_the_sweep_can_be_plotted(tmp_path, capsys,
                                                      monkeypatch):
    doc = _doc()
    doc["output"] = [{"type": "plot", "path": f"{c}.png", "x": c,
                      "facet_col": c, "hue": c} for c in COLUMNS]
    doc["output"] += [{"type": "plot", "path": f"y_{c}.png", "x": "y", "y": c}
                      for c in COLUMNS]
    path = _write(tmp_path, doc)
    monkeypatch.chdir(tmp_path)
    assert main(["run", str(path), "-q"]) == 0, capsys.readouterr().err
    assert (tmp_path / "x.png").is_file()


def test_VR6_the_columns_follow_the_sweep(tmp_path):
    """A pinned point adds ``t_<point>``; a 3-point observable adds ``c``."""
    doc = _doc()
    doc["sweep"]["external_times_grid"] = {"x": [1.0]}
    doc["output"] = [{"type": "plot", "path": "p.png", "x": "t_x"}]
    check_before_run(_parse_workflow(doc, base_dir=tmp_path))

    doc = _doc()
    doc["output"] = [{"type": "plot", "path": "p.png", "x": "t_x"}]
    with pytest.raises(ConfigError) as info:
        check_before_run(_parse_workflow(doc, base_dir=tmp_path))
    assert info.value.path == "output[0].x"
    assert "t_final" in info.value.detail

    doc = _doc()
    doc["expand"].update(observable=["phi_a(x)", "phi_b(y)", "phi_c(z)"])
    doc["sweep"].update(component_tuples=[[0, 0, 0]],
                        positions_grid={"x": [0.0], "y": [0.0], "z": [0.5]})
    doc["sweep"].pop("component_pairs")
    doc["output"] = [{"type": "plot", "path": "p.png", "x": "c"}]
    check_before_run(_parse_workflow(doc, base_dir=tmp_path))


def test_VR6_a_misspelt_column_lists_the_column_meant(tmp_path):
    doc = _doc()
    doc["output"] = [{"type": "plot", "path": "p.png", "x": "t_finl"}]
    with pytest.raises(ConfigError) as info:
        check_before_run(_parse_workflow(doc, base_dir=tmp_path))
    assert info.value.suggestions == ("t_final",)


# ---------------------------------------------------------------------------
# VR7 -- all shapes of a Cartesian position grid
# ---------------------------------------------------------------------------

def test_VR7_later_position_shapes_are_refused_in_every_command(tmp_path, capsys):
    cases = [
        ({"x": [[0.0, 0.0], [0.0, 0.0, 0.0]],
          "y": [[0.1, 0.2]]}, "x[1]"),
        ({"x": [[0.0], [0.0, 0.0]],
          "y": [[0.1], [0.1, 0.2, 0.3]]}, "y[1]"),
    ]
    for grid, entry in cases:
        doc = _doc()
        doc["system"]["vertices"] = []
        doc["sweep"]["positions_grid"] = grid
        path = _write(tmp_path, doc)
        for command in (["validate", str(path)],
                        ["run", str(path), "--dry-run", "-q"],
                        ["run", str(path), "-q"]):
            status = main(command)
            lines = capsys.readouterr().err.splitlines()
            assert status == 2, command
            assert len(lines) == 1, lines
            assert lines[0].startswith(
                "sft-wick: config error [invalid_value] "
                f"sweep.positions_grid.{entry}: "), lines


def test_VR7_same_axis_alternatives_can_broadcast_separately(tmp_path, capsys):
    doc = _doc()
    doc["system"]["vertices"] = []
    doc["sweep"]["positions_grid"] = {
        "x": [[0.0, 0.0], [0.0, 0.0, 0.0]], "y": [[0.5]]}
    path = _write(tmp_path, doc)
    for command in (["validate", str(path)], ["run", str(path), "-q"]):
        status = main(command)
        assert status == 0, (command, capsys.readouterr().err)


# ---------------------------------------------------------------------------
# VR8 -- high-point validation does not enumerate Gaussian pairings
# ---------------------------------------------------------------------------

def _high_point_doc(fields: list, points: list, positions: list) -> dict:
    """An order-0 observable of the given fields at the given points."""
    doc = _doc()
    doc["system"]["vertices"] = []
    doc["expand"]["observable"] = [
        f"{field}_a({point})" for field, point in zip(fields, points)]
    doc["sweep"].pop("component_pairs")
    doc["sweep"].update(
        component_tuples=[[0] * len(fields)],
        positions_grid={p: [x] for p, x in zip(points, positions)})
    return doc


def _distinct(fields: list) -> list:
    return [f"p{i}" for i in range(len(fields))]


FACING = ["psi"] * 2 + ["phi"] * 12
FACING_POINTS = ["p0", "p1"] + ["p1"] * 6 + ["p0"] * 6


@pytest.mark.parametrize("fields, points, assignments", [
    (["phi"] * 16, _distinct(range(16)), 1),
    (["psi"] + ["phi"] * 13, _distinct(range(14)), 13),
    (["psi"] * 3 + ["phi"] * 11, _distinct(range(14)), perm(11, 3)),
    # Two psi at one point: the operators of each point do not count.
    (FACING, ["p0"] * 2 + ["p1"] * 6 + ["p2"] * 6, 3),
    # Each psi needs the other point, which closes a loop.
    (FACING, FACING_POINTS, 0),
])
def test_VR8_validate_takes_one_pairing_per_R_assignment(
        tmp_path, capsys, monkeypatch, fields, points, assignments):
    # 15!! and 13!! pairings took minutes and seconds; the count does not
    # depend on how busy the machine is.
    counts = []
    refuse = _run_checks._refuse_joined_points

    def counted(terms, sweep):
        terms = list(terms)
        counts.append(len(terms))
        return refuse(terms, sweep)

    monkeypatch.setattr(_run_checks, "_refuse_joined_points", counted)
    monkeypatch.setattr(perturbation, "_wick_contract_target", _no_stage)
    doc = _high_point_doc(fields, points, [0.0] * len(fields))
    assert main(["validate", str(_write(tmp_path, doc))]) == 0, capsys.readouterr().err
    assert counts == [assignments]


@pytest.mark.parametrize("seed", ["0", "3", "7"])
def test_VR8_the_pair_refused_does_not_follow_the_hash_seed(tmp_path, seed):
    """Six pairs are joined and apart, and the first in sorted order is named.
    Each seed runs in its own process: a process has one, and the old order
    named ``q0`` at about one seed in six."""
    points = [p for i in range(6) for p in (f"p{i}", f"q{i}")]
    positions = [x for i in range(6) for x in (float(i), i + 0.5)]
    doc = _high_point_doc(["psi", "phi"] * 6, points, positions)
    done = subprocess.run(
        [sys.executable, "-m", "sft_wick.workflow.cli", "validate",
         str(_write(tmp_path, doc))],
        capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONHASHSEED": seed,
             "PYTHONPATH": str(Path(_run_checks.__file__).parents[2])},
    )
    assert done.returncode == 2, done.stderr
    assert "sweep.positions_grid.q0" in done.stderr


@pytest.mark.parametrize("fields, points, positions, status", [
    # Pairing the response point with any physical point joins the two.
    (["psi"] + ["phi"] * 13, _distinct(range(14)), [0.5] + [0.0] * 13, 2),
    (["phi"] * 13 + ["psi"], _distinct(range(14)),
     [0.0] * 6 + [0.5] + [0.0] * 7, 2),
    (["psi"] * 3 + ["phi"] * 11, _distinct(range(14)),
     [0.0, 0.5] + [0.0] * 12, 2),
    # Repeated points: only the second physical point is apart.
    (["psi"] + ["phi"] * 13, ["p0"] + ["p1"] * 6 + ["p2"] * 7,
     [0.0] * 7 + [0.5] * 7, 2),
    # The two points face each other, and every R there vanishes.
    (FACING, FACING_POINTS, [0.0, 0.5] + [0.5] * 6 + [0.0] * 6, 0),
    # More response than physical fields: every pairing has a zero psi-psi.
    (["psi"] * 3 + ["phi"], _distinct(range(4)), [0.5, 0.0, 0.0, 0.0], 0),
])
def test_VR8_response_point_joined_in_some_pairing_is_refused(
        tmp_path, capsys, fields, points, positions, status):
    doc = _high_point_doc(fields, points, positions)
    assert main(["validate", str(_write(tmp_path, doc))]) == status
    err = capsys.readouterr().err
    assert ("a chain of response propagators joins them" in err) == bool(status)


@pytest.mark.parametrize("moved, outcome", [({"z": 0.5}, "refused"), ({}, "ran")])
def test_VR9_a_response_point_joined_at_an_interacting_order(
        tmp_path, moved, outcome):
    """Order 1 of ``<phi psi psi>`` has no diagram at order 0, so only the
    walk over R assignments sees that the vertex joins two external points.
    Validate passed this and the run refused it."""
    doc = _response_doc(["phi_a(x)", "psi_b(y)", "psi_c(z)"], moved)
    doc["system"]["vertices"] = _doc()["system"]["vertices"]
    doc["expand"]["orders"] = [1]
    assert _outcome(doc, tmp_path, run=False) == outcome
    assert _run_past_the_check(doc, tmp_path) == outcome
