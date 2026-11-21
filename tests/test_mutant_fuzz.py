"""The kill rule and the forced draws of ``tools/mutant_fuzz.py``.

``tools/mutant_fuzz.py`` runs drawn cases of :mod:`validation.draw` under
the unmutated tree and under a copy with one corpus mutant applied, and
counts a kill per (seed, mode, integrator) from the two fuzz records.  The
tests here give it hand-written records and drawn cases; they start no
worker and import no sft_wick code.

The rule: a mutant is killed on a case when the unmutated tree passes
against the oracle and the mutant fails, or when the unmutated tree refuses
the case with the refusal the fix added and the mutant returns a value the
oracle rejects.  A refusal of another kind, an error, a value the oracle
accepts, or a failure made only of non-finite values is not a kill.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    module_name = f"{name}_under_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mf():
    return _load_tool("mutant_fuzz")


#: A compared row ``[tag, oracle, value, error, allowed, ratio]`` with a
#: finite value outside its tolerance.
REJECTED = [[0], 1.0, 2.0, 0.0, 1e-9, 1e9]


def _rec(seed, method, verdict, error=None, mode="default", route=None,
         rows=None):
    out = dict(seed=seed, method=method, verdict=verdict, mode=mode,
               route=route)
    if error is not None:
        out["error"] = error
    if verdict == "fail":
        out["rows"] = [REJECTED] if rows is None else rows
    return out


GL, QMC = "gauss_legendre", "qmc_vectorized"
REFUSAL = "ValueError: 2 external operators share the same spatial label 'x'"


def test_join_pairs_records_by_seed_mode_and_method(mf):
    head = [_rec(1, GL, "pass"), _rec(1, QMC, "pass"), _rec(2, GL, "pass"),
            _rec(2, GL, "pass", mode="shared-label")]
    mutant = [_rec(2, GL, "pass"), _rec(1, QMC, "fail"), _rec(1, GL, "pass"),
              _rec(2, GL, "fail", mode="shared-label")]
    rows = mf.join(head, mutant)
    keyed = {(r["seed"], r["mode"], r["method"]): r for r in rows}
    assert set(keyed) == {(1, "default", GL), (1, "default", QMC),
                          (2, "default", GL), (2, "shared-label", GL)}
    assert keyed[(1, "default", QMC)]["killed"]
    assert keyed[(2, "shared-label", GL)]["killed"]
    assert not keyed[(1, "default", GL)]["killed"]
    assert not keyed[(2, "default", GL)]["killed"]


def test_join_reports_a_record_missing_on_one_side(mf):
    rows = mf.join([_rec(1, GL, "pass"), _rec(3, GL, "pass")],
                   [_rec(1, GL, "fail")])
    keyed = {r["seed"]: r for r in rows}
    assert keyed[1]["killed"]
    assert not keyed[3]["killed"]
    assert keyed[3]["reason"] == "no mutant record"


def test_a_fail_over_a_baseline_pass_is_a_kill(mf):
    killed, reason = mf.kill_of(_rec(1, GL, "pass"), _rec(1, GL, "fail"))
    assert killed and reason == "value"


def test_a_fail_over_a_baseline_fail_is_not_a_kill(mf):
    killed, reason = mf.kill_of(_rec(1, GL, "fail"), _rec(1, GL, "fail"))
    assert not killed
    assert reason == "baseline fail"


@pytest.mark.parametrize("mutant", ["pass", "timeout", "harness-error"])
def test_a_baseline_pass_without_a_mutant_fail_is_not_a_kill(mf, mutant):
    killed, reason = mf.kill_of(_rec(1, GL, "pass"), _rec(1, GL, mutant))
    assert not killed
    assert reason == f"mutant {mutant}"


def test_a_baseline_refusal_and_a_rejected_mutant_value_is_a_kill(mf):
    killed, reason = mf.kill_of(_rec(1, GL, "refusal", REFUSAL),
                                _rec(1, GL, "fail"),
                                expect="share the same spatial label")
    assert killed and reason == "refusal"


@pytest.mark.parametrize("mutant", [
    _rec(1, GL, "refusal", "ValueError: the spatial points 'x' and 'y' are "
         "at different positions"),
    _rec(1, GL, "callback-error", "RuntimeError: boom"),
    _rec(1, GL, "pass"),
], ids=["another refusal", "another error", "accepted value"])
def test_a_baseline_refusal_without_a_rejected_value_is_not_a_kill(mf, mutant):
    killed, reason = mf.kill_of(_rec(1, GL, "refusal", REFUSAL), mutant,
                                expect="share the same spatial label")
    assert not killed
    assert reason == f"mutant {mutant['verdict']}"


def test_a_baseline_refusal_for_another_reason_is_not_a_kill(mf):
    killed, reason = mf.kill_of(_rec(1, GL, "refusal", REFUSAL),
                                _rec(1, GL, "fail"),
                                expect="off-diagonal component entries")
    assert not killed
    assert reason == "baseline refused otherwise"


def test_a_baseline_refusal_counts_only_for_a_refusal_type_mutant(mf):
    killed, reason = mf.kill_of(_rec(1, GL, "refusal", REFUSAL),
                                _rec(1, GL, "fail"), expect=None)
    assert not killed
    assert reason == "baseline refusal"


def test_every_target_label_has_a_route(mf):
    corpus = {m.label: m for m in mf.load_corpus()}
    assert len(mf.TARGETS) == 8
    for label, target in mf.TARGETS.items():
        assert corpus[label].kind == "historical", label
        assert target.mode in mf.MODES
        assert target.route in mf.ROUTES


def test_the_default_mode_leaves_the_draw_unchanged(mf):
    from validation.draw import draw
    for seed in (0, 7, 945):
        assert mf.force(draw(seed), "default") == draw(seed)


def _seed_with(mf, predicate, stop=400):
    from validation.draw import draw
    for seed in range(stop):
        case = draw(seed)
        if predicate(case):
            return case
    raise AssertionError("no seed found")


def test_shared_label_mode_puts_every_leg_at_one_point_and_time(mf):
    case = _seed_with(mf, lambda c: len(c.observable.legs) >= 2)
    forced = mf.force(case, "shared-label")
    obs = forced.observable
    assert len({obs.position(leg.label) for leg in obs.legs}) == 1
    assert not obs.times and not obs.integrate_over
    assert forced.max_weight >= 1
    one_leg = _seed_with(mf, lambda c: len(c.observable.legs) == 1)
    with pytest.raises(mf.ForceError):
        mf.force(one_leg, "shared-label")


def test_flag_modes_set_only_inadmissible_flags(mf):
    mixing = _seed_with(mf, lambda c: not c.flags["diag_C"])
    forced = mf.force(mixing, "diag-c")
    assert forced.flags["diag_C"] and forced.model == mixing.model
    diagonal = _seed_with(mf, lambda c: c.flags["diag_C"])
    with pytest.raises(mf.ForceError):
        mf.force(diagonal, "diag-c")
    distinct = _seed_with(mf, lambda c: len(set(c.model.gamma)) > 1)
    assert mf.force(distinct, "iso-r-on").flags["iso_R"]
    equal = _seed_with(mf, lambda c: len(set(c.model.gamma)) == 1
                       and c.model.n_components > 1)
    flags = mf.force(equal, "iso-r-off").flags
    assert not flags["iso_R"] and not flags["diag_R"]
    with pytest.raises(mf.ForceError):
        mf.force(equal, "iso-r-on")
    with pytest.raises(mf.ForceError):
        mf.force(distinct, "iso-r-off")


def test_multiplicative_apart_mode_moves_the_legs_apart(mf):
    case = _seed_with(mf, lambda c: c.model.multiplicative is not None
                      and len(c.observable.legs) >= 2, stop=2000)
    forced = mf.force(case, "multiplicative-apart")
    obs = forced.observable
    points = [np.asarray(obs.position(leg.label)) for leg in obs.legs]
    assert all(not np.allclose(points[0], p) for p in points[1:])
    assert mf.force(case, "multiplicative-apart") == forced
    plain = _seed_with(mf, lambda c: c.model.multiplicative is None)
    with pytest.raises(mf.ForceError):
        mf.force(plain, "multiplicative-apart")


def test_a_forced_case_has_an_admissible_control(mf):
    case = _seed_with(mf, lambda c: len(c.observable.legs) >= 2)
    control, spelling = mf.control_of(mf.force(case, "shared-label"),
                                      "shared-label")
    assert spelling == "distinct"
    assert control.flags == case.flags
    mixing = _seed_with(mf, lambda c: not c.flags["diag_C"])
    control, spelling = mf.control_of(mf.force(mixing, "diag-c"), "diag-c")
    assert control.flags == mixing.flags
    assert mf.control_of(case, "default") is None


def test_the_oracle_cache_must_lie_outside_the_tree(mf, tmp_path):
    with pytest.raises(SystemExit):
        mf.check_cache_dir(ROOT / "scratch-oracle")
    assert mf.check_cache_dir(tmp_path / "oracle") == tmp_path / "oracle"


def test_the_oracle_key_ignores_flags_and_spelling(mf):
    from validation.draw import draw
    case = _seed_with(mf, lambda c: len(set(c.model.gamma)) > 1)
    assert mf.oracle_key(case) == mf.oracle_key(mf.force(case, "iso-r-on"))
    assert mf.oracle_key(case) != mf.oracle_key(draw(case.seed + 1))


def test_join_keeps_both_trees_rows_at_the_mutant_worst_tag(mf):
    head = dict(_rec(1, GL, "pass"),
                rows=[[[0, 1], 2.0, 2.0, 0.0, 1e-9, 0.0],
                      [[1, 0], 3.0, 3.0, 0.0, 1e-9, 0.0]])
    mutant = dict(_rec(1, GL, "fail"), worst_tag=[1, 0], worst_ratio=1e9,
                  rows=[[[0, 1], 2.0, 2.0, 0.0, 1e-9, 0.0],
                        [[1, 0], 3.0, 4.0, 0.0, 1e-9, 1e9]])
    (row,) = mf.join([head], [mutant])
    assert row["killed"]
    assert row["mutant_row"] == dict(tag=[1, 0], oracle=3.0, value=4.0,
                                     error=0.0, allowed=1e-9, ratio=1e9)
    assert row["head_row"]["value"] == 3.0
    refused = mf.join([_rec(1, GL, "refusal", REFUSAL)], [mutant],
                      expect="share the same spatial label")
    assert refused[0]["killed"] and refused[0]["head_row"] is None


def test_saved_cases_are_read_back_as_they_ran(mf, tmp_path):
    import json

    case = _seed_with(mf, lambda c: len(c.observable.legs) >= 2)
    items, _ = mf.items_for("historical-shared-external-label-accepted",
                            [case.seed])
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(dict(
        label="historical-shared-external-label-accepted",
        items=[mf.item_to_dict(i) for i in items])))
    label, back = mf.items_from_file(path)
    assert label == "historical-shared-external-label-accepted"
    assert back == items
    assert [mf.oracle_key(i.case) for i in back] == [
        mf.oracle_key(i.case) for i in items]
    assert [i.mode for i in back] == ["shared-label", "shared-label/control"]


@pytest.mark.parametrize("label, removed, guard", [
    ("historical-diagonal-flags-on-a-non-diagonal-system",
     "ValueError: C of this system has off-diagonal component entries (from "
     "a dense R",
     "ValueError: κ² of the noise has off-diagonal component entries between "
     "the positions 0.1 and 0.4"),
    ("historical-r-group-takes-any-position",
     "ValueError: integrate_diagrams: the spatial points 'x' and 'y' are at "
     "different positions (0.4 and -0.7), and a chain of response",
     "ValueError: Expansion.evaluate: the external points 'x' and 'y' are at "
     "different positions (0.4 and -0.7), and a chain of response"),
], ids=["diagonal flag", "R-group"])
def test_a_refusal_of_an_unmutated_guard_is_not_the_entry_s(mf, label,
                                                           removed, guard):
    expect = mf.TARGETS[label].expect
    assert mf.kill_of(_rec(1, GL, "refusal", removed), _rec(1, GL, "fail"),
                      expect) == (True, "refusal")
    assert mf.kill_of(_rec(1, GL, "refusal", guard), _rec(1, GL, "fail"),
                      expect) == (False, "baseline refused otherwise")


def test_a_fail_made_only_of_non_finite_values_is_not_a_kill(mf):
    nan = [[0], 1.0, float("nan"), 0.0, 0.0, float("inf")]
    inf = [[1], 1.0, float("inf"), 0.0, 0.0, float("inf")]
    killed, reason = mf.kill_of(_rec(1, GL, "pass"),
                                _rec(1, GL, "fail", rows=[nan, inf]))
    assert not killed and reason == "mutant non-finite"
    killed, reason = mf.kill_of(_rec(1, GL, "refusal", REFUSAL),
                                _rec(1, GL, "fail", rows=[nan]),
                                expect="share the same spatial label")
    assert not killed and reason == "mutant non-finite"
    assert mf.kill_of(_rec(1, GL, "pass"),
                      _rec(1, GL, "fail", rows=[nan, REJECTED])) == (
        True, "value")


def test_join_keeps_routes_apart(mf):
    head = [_rec(1, GL, "pass", route="l1"), _rec(1, GL, "pass", route="l0")]
    mutant = [_rec(1, GL, "pass", route="l1"), _rec(1, GL, "fail", route="l0")]
    rows = {r["route"]: r for r in mf.join(head, mutant)}
    assert not rows["l1"]["killed"] and rows["l0"]["killed"]


def test_shared_seeds_run_once_under_the_unmutated_tree(mf):
    case = _seed_with(mf, lambda c: True)
    both = mf.Item(case, "default", "distinct", "l1", (GL, QMC))
    gl = mf.Item(case, "default", "distinct", "l1", (GL,))
    other = mf.Item(case, "default", "distinct", "l0", (GL,))
    merged = mf.unique_items([gl, both, gl, other])
    assert [(i.route, i.methods) for i in merged] == [("l1", (GL, QMC)),
                                                     ("l0", (GL,))]
    records = [dict(_rec(case.seed, m, "pass", route=r))
               for r in ("l1", "l0") for m in (GL, QMC)]
    kept = mf.records_for(records, [gl])
    assert [(r["route"], r["method"]) for r in kept] == [("l1", GL)]


def test_a_lost_oracle_row_is_reused_only_under_limits_no_larger(mf):
    reusable = mf.mutant_oracle.reusable
    assert reusable(dict(rows=[], scale=1.0), 10.0, 1.0)
    lost = dict(lost="oracle-timeout", limits=dict(timeout=600.0, max_rss=8.0))
    assert reusable(lost, 600.0, 8.0) and reusable(lost, 60.0, 2.0)
    assert not reusable(lost, 1200.0, 8.0)
    assert not reusable(lost, 600.0, 16.0)
    assert not reusable(dict(lost="oracle-error"), 1.0, 1.0)


#: The driver files: they and the tool modules they load import no sft_wick.
DRIVERS = ("mutant_fuzz", "mutant_oracle", "generated_kills",
           "mutation_check", "mutation_tree", "kill_matrix")


def _imported_names(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module or ""
        elif (isinstance(node, ast.Call) and node.args
              and isinstance(node.args[0], ast.Constant)
              and isinstance(node.args[0].value, str)
              and getattr(node.func, "attr", getattr(node.func, "id", ""))
              in ("import_module", "__import__")):
            yield node.args[0].value


@pytest.mark.parametrize("name", DRIVERS)
def test_the_driver_imports_no_sft_wick(name):
    path = ROOT / "tools" / f"{name}.py"
    for module in _imported_names(path):
        top = module.split(".")[0]
        assert top != "sft_wick", (name, module)
        assert module not in ("validation.lower", "validation.worker",
                              "mutant_lowering"), (name, module)


def test_loading_the_drivers_loads_no_sft_wick():
    code = (
        "import importlib.util, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "for name in ('mutant_fuzz', 'generated_kills'):\n"
        f"    spec = importlib.util.spec_from_file_location(name, {str(ROOT)!r}"
        " + f'/tools/{name}.py')\n"
        "    m = importlib.util.module_from_spec(spec)\n"
        "    sys.modules[name] = m\n"
        "    spec.loader.exec_module(m)\n"
        "mf = sys.modules['mutant_fuzz']\n"
        "items, _ = mf.items_for('historical-iso-c-trace', [4])\n"
        "[mf.oracle_key(i.case) for i in items]\n"
        "print(sorted(n for n in sys.modules if n.split('.')[0] == 'sft_wick'))\n")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "[]"
