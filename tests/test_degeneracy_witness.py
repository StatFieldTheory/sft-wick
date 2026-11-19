"""The degeneracy witness, ``tools/degeneracy_witness.py``, and its record.

Each corpus entry has a condition, read by a probe at the line it changes:
the state in which the entry changes what its function does.  For a
configuration and a mutant the witness reports one of three states:

* ``not exercised``: the mutated lines never run for the configuration, or
  they run and the condition never holds there;
* ``exercised, blind``: the condition holds on some call, and the
  configuration's values move by no more than its own tolerance;
* ``sees``: the condition holds and a value moves by more than the
  tolerance, or the mutant raises where the unmutated tree returns.

The toy tests run the witness on a small package in ``tmp_path``: one
mutant reads a pair of legs in the other order, so only off-diagonal
entries of a kernel move; others replace the test of an ``if``, raise, or
drop an R-cache mismatch.  The record tests read
``docs/verification/blind_spots.json``, written by ``python
tools/degeneracy_witness.py run --heavy``, against the expected cells below;
those are taken from the CHANGELOG and the demo READMEs, which say why
demos 1 to 3 did not show the 0.5.0 and 0.6.0 defects.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "verification" / "blind_spots.json"


def _load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    module_name = f"{name}_under_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dw():
    return _load_tool("degeneracy_witness")


# ---------------------------------------------------------------------------
# A toy package and its mutants
# ---------------------------------------------------------------------------

CORE = '''\
def pair_order(i, j):
    """The leg order a kernel is read at."""
    return (i, j)


def value(kernel, legs):
    total = 0.0
    for i, j in legs:
        a, b = pair_order(i, j)
        total += kernel[a][b]
    return total


def trace(kernel):
    return sum(kernel[a][a] for a in range(len(kernel)))


def clip(x, limit):
    if x > limit:
        return limit
    return x


class DiagramIntegrand:
    def __init__(self, diagram_term):
        self.diagram_term = diagram_term

    def _require_r_indices_match(self, cache):
        iso_R = _cache_r_type(cache)
        both = (_r_index_mismatch(self.diagram_term, False),
                _r_index_mismatch(self.diagram_term, True))
        if iso_R is not None and both[iso_R]:
            raise ValueError(both[iso_R])


def _cache_r_type(cache):
    return cache


def _r_index_mismatch(dt, iso_R):
    kind = dt.get(iso_R)
    if kind:
        return f"mismatch: {kind}"
    return None


def precheck(dt, cache):
    return _r_index_mismatch(dt, _cache_r_type(cache))
'''
#: name: (old, new, probe)
MUTANTS = {
    "leg-order": ("    return (i, j)\n", "    return (min(i, j), max(i, j))\n",
                  "pair_order", ("before", "return (i, j)", "i > j")),
    "clip": ("    if x > limit:\n", "    if False:\n", "clip",
             ("test", "if x > limit:", False)),
    "raise": ("    return x\n", "    raise ValueError('unclipped')\n", "clip",
              ("before", "return x", "True")),
    "r-cache": ("    if kind:\n", "    if False:\n", "_r_index_mismatch",
                ("test", "if kind:", False)),
}

CONFIGS = '''\
from toypkg import core

KERNELS = {"symmetric": [[1.0, 2.0], [2.0, 3.0]],
           "asymmetric": [[1.0, 2.0], [5.0, 3.0]]}


def tag_of(args):
    return list(args)


def run(name):
    if name == "trace":
        return {"trace": core.trace(KERNELS["asymmetric"])}
    if name == "diagonal":
        return {"diag": core.value(KERNELS["asymmetric"], [(1, 1), (0, 0)])}
    if name == "attributed":
        k = KERNELS["asymmetric"]
        return ({"r10": core.value(k, [(1, 0)]), "r00": core.value(k, [(0, 0)])},
                {"r10": [[1, 0]], "r00": [[0, 0]]})
    if name in ("clipped", "unclipped"):
        return {"x": core.clip(5.0 if name == "clipped" else 1.0, 3.0)}
    if name == "r_other_kind":
        core.DiagramIntegrand({False: "matrix"})._require_r_indices_match(True)
        return {"ok": 1.0}
    if name == "r_precheck":
        return {"flag": 1.0 if core.precheck({True: "scalar"}, True) else 0.0}
    return {"offdiag": core.value(KERNELS[name], [(1, 0)]),
            "diag": core.value(KERNELS[name], [(0, 0)])}
'''


@pytest.fixture
def toy(dw, tmp_path):
    head = tmp_path / "head"
    configs = tmp_path / "configs"
    trees = {"head": (head, CORE)}
    for name, (old, new, _fn, _probe) in MUTANTS.items():
        assert CORE.count(old) == 1, name
        trees[name] = (tmp_path / name, CORE.replace(old, new))
    for tree, text in trees.values():
        (tree / "toypkg").mkdir(parents=True)
        (tree / "toypkg" / "__init__.py").write_text("")
        (tree / "toypkg" / "core.py").write_text(text)
    configs.mkdir()
    (configs / "toy_configs.py").write_text(CONFIGS)
    head_file = head / "toypkg" / "core.py"

    def target(name, **site):
        _old, _new, function, probe = MUTANTS[name]
        return dw.MutantTree(
            label=f"toy-{name}", src=tmp_path / name, head_file=head_file,
            site=dw.Site(module="toypkg.core", function=function, probe=probe,
                         condition=f"toy {name}", **site))

    return dict(head=head, target=target, extra=(configs,))


def _config(dw, name):
    return dw.Config(id=name, kind="callable", target="toy_configs:run",
                     args={"name": name}, rel=1e-12, floor=0.0, demo="toy",
                     source="toy")


def _cells(dw, toy, names, target=None):
    target = target or toy["target"]("leg-order")
    cells = dw.witness(toy["head"], [target],
                       [_config(dw, n) for n in names],
                       extra_path=toy["extra"], jobs=1)
    return {c["config"]: c for c in cells}


def test_symmetric_config_is_blind_and_asymmetric_one_sees(dw, toy):
    cells = _cells(dw, toy, ["symmetric", "asymmetric"])
    assert cells["symmetric"]["state"] == dw.BLIND
    assert cells["asymmetric"]["state"] == dw.SEES
    for cell in cells.values():
        assert cell["line_hits"] > 0 and cell["condition_calls"] == 1
        assert cell["diverged_calls"] == 1
        assert cell["diverged_without_condition"] == 0
    assert cells["symmetric"]["max_abs_diff"] == 0.0
    assert cells["asymmetric"]["max_abs_diff"] == pytest.approx(3.0, abs=0)


def test_config_that_never_runs_the_mutated_line_is_not_exercised(dw, toy):
    cell = _cells(dw, toy, ["trace"])["trace"]
    assert cell["state"] == dw.NOT_EXERCISED
    assert cell["line_hits"] == 0 and cell["condition_calls"] == 0


def test_config_where_the_condition_never_holds_is_not_exercised(dw, toy):
    cell = _cells(dw, toy, ["diagonal"])["diagonal"]
    assert cell["line_hits"] > 0
    assert cell["condition_calls"] == 0 and cell["diverged_calls"] == 0
    assert cell["state"] == dw.NOT_EXERCISED


def test_an_if_probe_reads_the_unmutated_test(dw, toy):
    cells = _cells(dw, toy, ["clipped", "unclipped"], toy["target"]("clip"))
    assert cells["clipped"]["state"] == dw.SEES
    assert cells["clipped"]["condition_calls"] == 1
    assert cells["unclipped"]["line_hits"] > 0
    assert cells["unclipped"]["condition_calls"] == 0
    assert cells["unclipped"]["state"] == dw.NOT_EXERCISED


def test_a_mutant_that_raises_where_the_unmutated_tree_returns_sees(dw, toy):
    cells = _cells(dw, toy, ["unclipped", "clipped"], toy["target"]("raise"))
    assert cells["unclipped"]["state"] == dw.SEES
    assert cells["unclipped"]["seen_by"] == "mutant raises"
    assert "unclipped" in cells["unclipped"]["mutant_error"]
    assert cells["clipped"]["state"] == dw.NOT_EXERCISED


def test_calls_are_attributed_to_their_rows(dw, toy):
    target = toy["target"]("leg-order", attribution="toy_configs:tag_of")
    cell = _cells(dw, toy, ["attributed"], target)["attributed"]
    assert cell["rows"]["r10"]["state"] == dw.SEES
    assert cell["rows"]["r00"]["state"] == dw.NOT_EXERCISED
    assert not cell["rows"]["r00"]["condition"]
    assert cell["state"] == dw.SEES


def _with_probe(dw, toy, expression):
    target = toy["target"]("leg-order")
    site = dataclasses.replace(
        target.site, probe=("before", "return (i, j)", expression))
    return dataclasses.replace(target, site=site)


def test_a_probe_that_raises_counts_as_holding(dw, toy):
    target = _with_probe(dw, toy, "undefined_name > 0")
    cell = _cells(dw, toy, ["symmetric"], target)["symmetric"]
    assert cell["probe_errors"] == cell["calls"] == cell["condition_calls"]
    assert cell["state"] == dw.BLIND


def test_a_result_that_changes_with_the_condition_false_is_an_error(dw, toy):
    target = _with_probe(dw, toy, "False")
    cell = _cells(dw, toy, ["symmetric"], target)["symmetric"]
    assert cell["diverged_without_condition"] == 1
    assert cell["state"] == dw.ERROR
    assert "condition false" in cell["reason"]


def test_r_cache_decision_watches_only_the_kind_the_cache_has(dw, toy):
    target = toy["target"]("r-cache", decision="r_cache")
    cells = _cells(dw, toy, ["r_other_kind", "r_precheck"], target)
    other = cells["r_other_kind"]
    assert other["line_hits"] > 0 and other["condition_calls"] == 0
    assert other["state"] == dw.NOT_EXERCISED
    assert cells["r_precheck"]["condition_calls"] == 1
    assert cells["r_precheck"]["state"] == dw.SEES


def test_the_child_refuses_a_package_from_outside_its_tree(dw, toy, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="not from"):
        dw.evaluate(empty, [_config(dw, "asymmetric")],
                    extra_path=(toy["head"], *toy["extra"]), package="toypkg")


def test_toy_sentinel_is_installed_in_a_clean_child(dw, toy, tmp_path):
    """The child imports the toy package from the tree it is given, not
    from the parent's path."""
    decoy = tmp_path / "decoy" / "toypkg"
    decoy.mkdir(parents=True)
    (decoy / "__init__.py").write_text("raise ImportError('decoy')\n")
    (decoy / "core.py").write_text("raise ImportError\n")
    env = dict(os.environ, PYTHONPATH=str(decoy.parent))
    target = toy["target"]("leg-order")
    out = dw.evaluate(target.src, [_config(dw, "asymmetric")], target=target,
                      extra_path=toy["extra"], package="toypkg", env=env)
    (row,) = out
    assert row["error"] is None
    assert row["rows"]["offdiag"] == 2.0


# ---------------------------------------------------------------------------
# The pieces of the rule
# ---------------------------------------------------------------------------

def _rows(dw, head, mutant, *, tags=None, line_hits=1, rel=1e-10,
          floor=0.0):
    return dw.classify_rows(head=head, mutant=mutant,
                            row_tags=tags or {k: None for k in head},
                            condition_tags=[None], line_hits=line_hits,
                            rel=rel, floor=floor)


def test_a_row_is_judged_only_by_the_calls_attributed_to_it(dw):
    rows = dw.classify_rows(
        head={"FF": 1.0, "FK": 2.0}, mutant={"FF": 1.0, "FK": 2.5},
        row_tags={"FF": [[0, [["F", 2]]]], "FK": [[0, [["F", 1], ["K", 1]]]]},
        condition_tags=[[0, [["F", 1], ["K", 1]]]], line_hits=3, rel=1e-10,
        floor=0.0)
    assert rows["FF"]["state"] == dw.NOT_EXERCISED
    assert rows["FK"]["state"] == dw.SEES
    assert dw.aggregate([r["state"] for r in rows.values()]) == dw.SEES


def test_an_unattributed_condition_applies_to_every_row(dw):
    rows = _rows(dw, {"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0})
    assert {r["state"] for r in rows.values()} == {dw.BLIND}


def test_tolerance_is_relative_plus_a_floor_on_the_config_scale(dw):
    head = {"big": 1.0, "small": 0.0}
    rows = _rows(dw, head, {"big": 1.0, "small": 1e-13}, floor=1e-12)
    assert rows["small"]["state"] == dw.BLIND
    rows = _rows(dw, head, {"big": 1.0, "small": 1e-11}, floor=1e-12)
    assert rows["small"]["state"] == dw.SEES


def test_tolerance_has_a_relative_term(dw):
    rows = _rows(dw, {"v": 2.0}, {"v": 2.0 + 1e-4}, rel=1e-3, floor=1e-9)
    assert rows["v"]["tolerance"] == pytest.approx(2e-3 + 2e-9)
    assert rows["v"]["state"] == dw.BLIND


def test_a_move_equal_to_the_tolerance_is_blind(dw):
    rows = _rows(dw, {"v": 1.0}, {"v": 1.5}, rel=0.5)
    assert rows["v"]["abs_diff"] == rows["v"]["tolerance"] == 0.5
    assert rows["v"]["state"] == dw.BLIND


def test_the_floor_scale_can_be_per_order(dw):
    head = {"order=0|a=0": 1.0, "order=4|a=0": 1e-6}
    rows = dw.classify_rows(head, {**head, "order=4|a=0": 1e-6 * (1 + 1e-4)},
                            {k: None for k in head}, [None], 1, rel=1e-5,
                            floor=1e-5, scale="order")
    assert rows["order=4|a=0"]["scale"] == 1e-6
    assert rows["order=4|a=0"]["state"] == dw.SEES
    rows = dw.classify_rows(head, {**head, "order=4|a=0": 1e-6 * (1 + 1e-4)},
                            {k: None for k in head}, [None], 1, rel=1e-5,
                            floor=1e-5)
    assert rows["order=4|a=0"]["state"] == dw.BLIND


def test_no_line_hit_means_not_exercised_whatever_the_tags(dw):
    rows = _rows(dw, {"a": 1.0}, {"a": 1.0}, line_hits=0)
    assert rows["a"]["state"] == dw.NOT_EXERCISED


def test_a_value_that_moves_with_the_condition_off_is_an_error(dw):
    rows = dw.classify_rows({"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 1.5},
                            {"a": [[0, 1]], "b": [[0, 2]]}, [[0, 1]], 1,
                            rel=1e-10, floor=0.0)
    assert rows["a"]["state"] == dw.BLIND
    assert rows["b"]["state"] == dw.ERROR
    rows = _rows(dw, {"a": 1.0}, {"a": 2.0}, line_hits=0)
    assert rows["a"]["state"] == dw.ERROR


def test_a_row_missing_on_one_side_is_an_error(dw):
    rows = _rows(dw, {"a": 1.0, "b": 2.0}, {"a": 1.0})
    assert rows["b"]["state"] == dw.ERROR
    assert dw.aggregate(r["state"] for r in rows.values()) == dw.ERROR


def test_outcomes_compare_arrays_messages_and_declared_equivalents(dw):
    same = dw.sentinel_mod.same
    arr = np.array([1.0, 2.0])
    assert same(("return", arr), ("return", arr.copy()))
    assert not same(("return", arr), ("return", arr * 2))
    assert same(("return", np.array([np.nan])), ("return", np.array([np.nan])))
    assert not same(("raise", ValueError("a")), ("raise", ValueError("b")))
    assert same(("raise", ValueError("a")), ("raise", ValueError("a")))
    assert not same(("return", None), ("return", False))
    assert same(("return", None), ("return", False), [(None, False)])
    assert not same(("return", 0), ("return", None), [(None, False)])
    assert not same(("return", 1.0), ("raise", ValueError("1.0")))


def test_changed_lines_are_numbered_in_the_mutant_file(dw):
    head = "a\nb\nc\nd\n"
    mutant = "a\nx\ny\nb\nC\nd\n"
    assert dw.sentinel_mod.changed_lines(head, mutant) == {2, 3, 5}
    assert dw.sentinel_mod.changed_lines(head, "a\nb\nd\n") == {3}


def test_a_probe_anchor_must_name_one_statement(dw):
    text = "def f(x):\n    if x:\n        return 1\n    if x:\n        return 2\n"
    with pytest.raises(ValueError, match="2 statements"):
        dw.sentinel_mod.head_function(text, "f", {}, ("test", "if x:", False))
    with pytest.raises(ValueError, match="0 statements"):
        dw.sentinel_mod.head_function(text, "f", {}, ("before", "pass", "1"))


# ---------------------------------------------------------------------------
# The committed record
# ---------------------------------------------------------------------------

TARGETS = (
    "historical-shared-external-label-accepted",
    "historical-callable-coupling-at-one-leg-order",
    "historical-iso-c-trace",
    "historical-t-min-not-reaching-integrators",
    "historical-diagonal-flags-on-a-non-diagonal-system",
    "historical-r-group-takes-any-position",
    "historical-r-cache-mismatch-accepted",
    "historical-no-cut-at-a-pinned-external-time",
)
LEG_ORDER = "historical-callable-coupling-at-one-leg-order"

#: Demo 1 to 3 cells that are not ``not exercised``.  Every other demo 1 to
#: 3 cell is.  Reasons, per mutant:
#:
#: * leg order: the shipped kernels are a delta times a function symmetric
#:   in the leg points, so another leg order changes the call and not the
#:   value; demo 1 has no non-local vertex (CHANGELOG 0.5.0, "Callable
#:   non-local couplings"; examples/demo4/README.md, first paragraph).  At
#:   0.5.0 demo 2's R-contracted kappa3 was a quadrature with a point
#:   asymmetry of 2.3e-06 to 9.6e-05, and the fix moved demo 2's FK channel
#:   by 4.9e-06 at t = 1, below demo 2's 1e-4 (same entry); 0.6.1 replaced
#:   that kernel by an analytic one, and the record measures today's files.
#: * iso_C: no demo config sets ``iso_C`` (default False).
#: * t_min: "Every shipped example has ``t_min = 0``" (CHANGELOG 0.5.0).
#: * diagonal flags: demos 1 and 2 use separable specs and DiagonalA, which
#:   the probe does not read; demo 3's callable kappa2
#:   (examples/demo3/L2/kappa2_general.py) is a multiple of ``np.eye``, so
#:   the probe reads it and the condition never holds.
#: * R-group position: reached only through a response leg or a vertex with
#:   two response legs, which "no demo met" (CHANGELOG 0.6.0).
#: * R-cache: L1 builds the terms and the cache from one LinearOp
#:   (CHANGELOG 0.5.0, the R-cache refusal).
#: * kink at a pinned external: every demo config fixes all externals at
#:   one time, where the cut is the domain boundary (CHANGELOG 0.6.0, kink
#:   splitting).
#: * shared label: coincident points carry distinct labels "in every demo"
#:   (perturbation.py, check_distinct_external_labels).
EXPECTED_DEMO_1_3 = {
    ("demo2_config.yaml", LEG_ORDER): "exercised, blind",
    ("demo2/L2/config_FK.yaml", LEG_ORDER): "exercised, blind",
    ("demo3/config_FK.yaml", LEG_ORDER): "exercised, blind",
    ("demo3/config_F3K.yaml", LEG_ORDER): "exercised, blind",
}
DEMO_1_3 = ("demo1_config.yaml", "demo1/L2/config.yaml", "demo2_config.yaml",
            "demo2/L2/config_FF.yaml", "demo2/L2/config_FK.yaml",
            "demo3/config_FK.yaml", "demo3/config_F3K.yaml")
#: Demo 4 is asymmetric in points and components, so the leg order moves
#: its non-local channels (examples/demo4/README.md: 46.8 % at level A).
EXPECTED_DEMO_4_SEES = {LEG_ORDER}
WORST = ("error", "sees", "exercised, blind", "not exercised")

under_harness = pytest.mark.skipif(
    bool(os.environ.get("MUTATION_HARNESS")),
    reason="the mutation harness has the source under test")


@pytest.fixture(scope="module")
def record():
    if not RECORD.exists():
        pytest.fail(f"{RECORD.relative_to(ROOT)} is missing; run "
                    "python tools/degeneracy_witness.py run --heavy")
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _cell_map(record):
    return {(c["config"], c["mutant"]): c for c in record["cells"]}


@under_harness
def test_every_probe_names_one_statement_of_the_current_source(dw):
    import ast
    for label, site in dw.SITES.items():
        path = pathlib.Path(ROOT, "src", *site.module.split(".")).with_suffix(
            ".py")
        text = path.read_text(encoding="utf-8")
        node = next(n for n in ast.parse(text).body
                    if isinstance(n, ast.FunctionDef) and n.name == site.function)
        dw.sentinel_mod.instrument(node, text, site.probe)


@under_harness
def test_record_names_the_current_eight_entries(dw, record):
    assert tuple(m["label"] for m in record["mutants"]) == TARGETS
    assert record["corpus_sha256"] == dw.corpus_digest(dw.target_mutations())
    for mutant in record["mutants"]:
        assert mutant["condition"] == dw.SITES[mutant["label"]].condition


@under_harness
def test_record_holds_every_config_with_its_tolerance(dw, record):
    assert [c["id"] for c in record["configs"]] == [c.id for c in dw.CONFIGS]
    assert any(c["heavy"] for c in record["configs"])
    for stored, config in zip(record["configs"], dw.CONFIGS):
        assert (stored["rel"], stored["floor"], stored["scale"]) == (
            config.rel, config.floor, config.scale), config.id


@under_harness
def test_record_has_a_cell_per_config_and_mutant(record):
    cells = _cell_map(record)
    configs = [c["id"] for c in record["configs"]]
    assert set(DEMO_1_3) <= set(configs)
    assert len(cells) == len(record["cells"])
    assert set(cells) == {(c, m) for c in configs for m in TARGETS}
    assert {c["state"] for c in cells.values()} <= {
        "not exercised", "exercised, blind", "sees"}


def _scales(cell):
    out = {}
    for name, row in cell["rows"].items():
        key = ""
        if cell["tolerance"]["scale"] == "order":
            key = next((p for p in name.split("|") if p.startswith("order=")),
                       "")
        out[name] = key
    groups = {}
    for name, key in out.items():
        groups[key] = max(groups.get(key, 0.0), abs(cell["rows"][name]["head"]))
    return {name: groups[key] for name, key in out.items()}


@under_harness
def test_every_row_state_follows_from_its_values(record):
    for cell in record["cells"]:
        where = (cell["config"], cell["mutant"])
        if not cell["rows"]:
            assert cell.get("seen_by") == "mutant raises", where
            assert cell["condition_calls"] > 0, where
            continue
        rel, floor = cell["tolerance"]["rel"], cell["tolerance"]["floor"]
        scales = _scales(cell)
        for name, row in cell["rows"].items():
            assert row["abs_diff"] == abs(row["mutant"] - row["head"]), where
            assert row["scale"] == scales[name], where
            assert row["tolerance"] == (rel * abs(row["head"])
                                        + floor * row["scale"]), where
            if cell["line_hits"] == 0 or not row["condition"]:
                assert row["abs_diff"] == 0.0, (where, name)
                want = "not exercised"
            elif row["abs_diff"] <= row["tolerance"]:
                want = "exercised, blind"
            else:
                want = "sees"
            assert row["state"] == want, (where, name)
        worst = min(WORST.index(r["state"]) for r in cell["rows"].values())
        assert cell["state"] == WORST[worst], where


@under_harness
def test_demo_1_to_3_cells_match_the_notes(record):
    cells = _cell_map(record)
    for config in DEMO_1_3:
        for label in TARGETS:
            want = EXPECTED_DEMO_1_3.get((config, label), "not exercised")
            assert cells[(config, label)]["state"] == want, (config, label)


@under_harness
def test_demo_4_has_no_blind_cell(record):
    demo4 = [c["id"] for c in record["configs"] if c["demo"] == "demo4"]
    assert len(demo4) == 4
    cells = _cell_map(record)
    for config in demo4:
        for label in TARGETS:
            state = cells[(config, label)]["state"]
            assert state != "exercised, blind", (config, label)
            want = "sees" if label in EXPECTED_DEMO_4_SEES else "not exercised"
            assert state == want, (config, label)
    assert record["summary"]["demo4_blind"] == []


@under_harness
def test_record_summary_follows_from_the_cells(record):
    cells = _cell_map(record)
    summary = record["summary"]
    for config, missed in summary["not_seen"].items():
        assert missed == [m for m in TARGETS
                          if cells[(config, m)]["state"] != "sees"]
    for config, blind in summary["blind"].items():
        assert blind == [m for m in TARGETS
                         if cells[(config, m)]["state"] == "exercised, blind"]
    demo4 = {c["id"] for c in record["configs"] if c["demo"] == "demo4"}
    assert summary["demo4_blind"] == sorted(
        f"{c}: {m}" for (c, m), cell in cells.items()
        if c in demo4 and cell["state"] == "exercised, blind")
    assert summary["errors"] == []


@under_harness
def test_check_accepts_the_record_and_reports_tampering(dw, record, tmp_path):
    assert dw.check(RECORD) == []

    def problems(edit):
        stale = json.loads(json.dumps(record))
        edit(stale)
        path = tmp_path / "blind_spots.json"
        path.write_text(json.dumps(stale), encoding="utf-8")
        return dw.check(path)

    def blind_demo4(r):
        cell = next(c for c in r["cells"]
                    if c["config"] == "demo4/config_level_a.yaml"
                    and c["mutant"] == TARGETS[3])
        cell["state"] = "exercised, blind"

    def loose(r):
        for c in r["configs"]:
            c["rel"] = 10.0

    def drop_heavy(r):
        r["configs"] = [c for c in r["configs"] if not c["heavy"]]

    def summary_only(r):
        r["summary"]["blind"]["demo1_config.yaml"] = [LEG_ORDER]

    assert any("worst of its rows" in p for p in problems(blind_demo4))
    assert problems(summary_only) == [
        "the summary does not follow from the cells"]
    assert any("tolerance differs" in p for p in problems(loose))
    assert any("heavy ones included" in p for p in problems(drop_heavy))


def test_record_holds_no_local_path(record):
    text = RECORD.read_text(encoding="utf-8")
    assert "/Users/" not in text and "/private/" not in text
    assert "/tmp/" not in text and "/home/" not in text
