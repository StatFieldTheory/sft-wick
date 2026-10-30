"""The ``sft-wick`` command line: config errors, ``validate``, ``schema``
and ``--print-resolved``.

CL1 -- a config error ends the command with exit status 2 and one line on
       stderr that names the error code, the dotted path and, for a
       misspelt key, the key meant.  The four misspellings below used to
       end in a traceback with status 1 (and, before 0.7.0, to run).
CL2 -- ``validate`` parses the config and builds the System, so it also
       reports what only the build checks; every shipped config passes.
CL3 -- ``schema`` prints the JSON Schema.
CL4 -- ``--print-resolved`` prints every key with the value the run uses.
CL5 -- ``--override`` sets a key the file leaves out; a malformed or
       misspelt override is a config error.
CL6 -- ``SFT_WICK_STRICT=0`` turns an unknown key in the file into a
       one-line warning and ignores the key, as releases before 0.7.0 did;
       overrides stay strict.
CL7 -- the Python API raises :class:`ConfigError`, a ``ValueError``; an
       override error is also a ``KeyError``, as before.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import yaml

from sft_wick.workflow import config as cf
from sft_wick.workflow.cli import main
from sft_wick.workflow.config import (ConfigError, ConfigWarning,
                                      load_workflow_config)
from sft_wick.workflow.schema import workflow_json_schema

ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = ROOT / "examples" / "quickstart.yaml"


def _quickstart() -> dict:
    return yaml.safe_load(QUICKSTART.read_text())


def _write(tmp_path: Path, doc: dict, name: str = "c.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def _misspelt(where: tuple, old: str, new: str) -> dict:
    doc = _quickstart()
    node = doc
    for step in where:
        node = node[step]
    node[new] = node.pop(old) if old in node else 0.5
    return doc


#: The misspellings that used to run, one per block: (block, key, typo).
WITNESSES = [
    ((("system",), "t_min", "t_mni")),
    ((("sweep",), "n_samples", "n_sampels")),
    ((("propagators",), "n_grid_t", "n_grid_tt")),
    (((), "sweep", "swep")),
    ((("expand",), "diag_C", "diag_c")),
    ((("system", "noise", "kappa2", "temporal"), "sigma_t", "sigma_tt")),
    ((("output", 0), "format", "fromat")),
]


def _dotted(where: tuple, key: str) -> str:
    out = ""
    for step in where + (key,):
        out += f"[{step}]" if isinstance(step, int) else (
            f".{step}" if out else str(step))
    return out


def _one_line(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == 1, lines
    return lines[0]


# ---------------------------------------------------------------------------
# CL1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("command", [["validate"], ["run", "--dry-run", "-q"]],
                         ids=["validate", "run"])
@pytest.mark.parametrize("where,key,typo", WITNESSES,
                         ids=[w[2] for w in WITNESSES])
def test_CL1_misspelt_key_exits_2_with_one_line(tmp_path, capsys, command,
                                                where, key, typo):
    path = _write(tmp_path, _misspelt(where, key, typo))
    status = main([command[0], str(path), *command[1:]])
    out = capsys.readouterr()
    assert status == 2
    line = _one_line(out.err)
    assert line.startswith("sft-wick: config error [unknown_key] ")
    assert f"{_dotted(where, typo)}: unknown key" in line
    assert f"'{key}'" in line
    assert "Traceback" not in out.err


def test_CL1_exit_status_reaches_the_shell(tmp_path):
    path = _write(tmp_path, _misspelt(("system",), "t_min", "t_mni"))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), env.get("PYTHONPATH", "")])
    done = subprocess.run(
        [sys.executable, "-m", "sft_wick.workflow.cli", "validate", str(path)],
        capture_output=True, text=True, env=env, cwd=tmp_path)
    assert done.returncode == 2
    assert _one_line(done.stderr) == (
        "sft-wick: config error [unknown_key] system.t_mni: unknown key; "
        "did you mean 't_min'?")


@pytest.mark.parametrize("command", [["validate"],
                                     ["validate", "--print-resolved"],
                                     ["run", "--dry-run", "-q"]],
                         ids=["validate", "print_resolved", "run"])
@pytest.mark.parametrize("head,grid", [
    ("", "positions_grid: {x: &g [0.0, *g], y: [0.5]}"),
    ("&r\n", "positions_grid: {x: [0.0, *r], y: [0.5]}"),
], ids=["list", "document"])
def test_CL1_a_file_whose_aliases_form_a_cycle_exits_2(tmp_path, capsys,
                                                       command, head, grid):
    """``validate`` used to pass these files; ``--print-resolved`` and
    ``run`` ended in a traceback."""
    text = QUICKSTART.read_text()
    old = "positions_grid: {x: [0.0], y: [0.0, 0.5, 1.5]}"
    assert old in text
    path = tmp_path / "c.yaml"
    path.write_text(head + text.replace(old, grid))
    status = main([command[0], str(path), *command[1:]])
    out = capsys.readouterr()
    assert status == 2
    assert _one_line(out.err) == (
        "sft-wick: config error [invalid_value] sweep.positions_grid.x[1]: "
        "an alias here makes a value hold itself")


def test_CL1_a_value_that_expands_is_one_short_line(tmp_path, capsys):
    """``seed`` holds a list of 2**18 pairs from one line of YAML."""
    levels = ", ".join(f"&g{i} [*g{i - 1}, *g{i - 1}]" for i in range(1, 19))
    text = QUICKSTART.read_text()
    assert "seed: 42" in text
    path = tmp_path / "c.yaml"
    path.write_text(text.replace(
        "seed: 42", f"seed: [&g0 [0.0, 1.0], {levels}]"))
    assert main(["validate", str(path)]) == 2
    line = _one_line(capsys.readouterr().err)
    assert line.startswith("sft-wick: config error [invalid_value] sweep.seed")
    assert len(line) < 2000


def test_CL1_a_value_error_is_one_line_too(tmp_path, capsys):
    doc = _quickstart()
    doc["sweep"]["method"] = "gauss_legendr"
    status = main(["validate", str(_write(tmp_path, doc))])
    line = _one_line(capsys.readouterr().err)
    assert status == 2
    assert line.startswith("sft-wick: config error [invalid_value] "
                           "sweep.method: ")
    assert "did you mean 'gauss_legendre'?" in line


def test_CL1_unreadable_files_are_config_errors(tmp_path, capsys):
    assert main(["validate", str(tmp_path / "missing.yaml")]) == 2
    assert "[read_error]" in _one_line(capsys.readouterr().err)
    bad = tmp_path / "bad.yaml"
    bad.write_text("system: [unclosed\n")
    assert main(["validate", str(bad)]) == 2
    assert "[read_error]" in _one_line(capsys.readouterr().err)
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    assert main(["validate", str(empty)]) == 2
    assert "[invalid_value] config: " in _one_line(capsys.readouterr().err)


# ---------------------------------------------------------------------------
# CL2
# ---------------------------------------------------------------------------

def _shipped_configs() -> list[Path]:
    paths = sorted(p for p in (ROOT / "examples").rglob("*.yaml")
                   if "sweep" in (yaml.safe_load(p.read_text()) or {}))
    paths.append(ROOT / "src" / "sft_wick" / "workflow" / "quickstart.yaml")
    return paths


@pytest.mark.parametrize("path", _shipped_configs(),
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_CL2_every_shipped_config_is_valid(capsys, path):
    assert main(["validate", str(path)]) == 0
    assert _one_line(capsys.readouterr().out) == f"[sft-wick] valid: {path}"


def test_CL2_validate_runs_the_build_checks(tmp_path, capsys):
    """A kernel parameter is checked when the System is built, which
    loading the file alone does not do."""
    doc = _quickstart()
    doc["system"]["noise"]["kappa2"]["temporal"]["sigma_t"] = 0.0
    path = _write(tmp_path, doc)
    load_workflow_config(path)  # loads
    assert main(["validate", str(path)]) == 2
    line = _one_line(capsys.readouterr().err)
    assert line.startswith("sft-wick: config error [invalid_value] ")
    assert "system.noise.kappa2.temporal.sigma_t: must be positive" in line


# ---------------------------------------------------------------------------
# CL3
# ---------------------------------------------------------------------------

def test_CL3_schema_command_prints_the_schema(capsys):
    assert main(["schema"]) == 0
    assert json.loads(capsys.readouterr().out) == workflow_json_schema()


# ---------------------------------------------------------------------------
# CL4
# ---------------------------------------------------------------------------

def _resolved(capsys, path: Path, *extra: str) -> dict:
    assert main(["validate", str(path), "--print-resolved", *extra]) == 0
    return yaml.safe_load(capsys.readouterr().out)


def _fields(cls) -> set:
    import dataclasses

    return {f.name for f in dataclasses.fields(cls)}


def test_CL4_print_resolved_fills_every_default(capsys):
    doc = _resolved(capsys, QUICKSTART)
    assert set(doc) == {"system", "expand", "propagators", "sweep", "output"}
    assert set(doc["expand"]) == _fields(cf.ExpandConfig)
    assert set(doc["propagators"]) == _fields(cf.PropagatorsConfig)
    assert set(doc["sweep"]) == (_fields(cf.SweepConfig) - {"component_pairs"}
                                 ) | {"component_tuples"}
    assert all(set(o) == _fields(cf.OutputConfig) for o in doc["output"])
    assert doc["sweep"]["method"] == "qmc_vectorized"
    assert doc["sweep"]["n_samples"] == 4096
    assert doc["sweep"]["component_tuples"] == [[0, 0], [1, 1]]
    assert doc["expand"]["target"] == "moment"
    assert doc["propagators"]["c_method"] == "auto"
    system = doc["system"]
    assert system["field"] == {"name": "phi", "n_components": 2}
    assert system["t_min"] == 0.0
    assert system["noise"]["sigma2"] is None
    assert system["linear"] == {"type": "diagonal", "gamma": [1.0, 1.0],
                                "n_grid_cache": 200, "t_max_cache": 100.0,
                                "t_min_cache": 0.0}
    vertex = system["vertices"][0]
    assert vertex["coupling_vectorized"] is False and vertex["rank"] is None
    assert system["nonlocal_vertices"] == []


def test_CL4_print_resolved_shows_derived_grids_and_overrides(tmp_path,
                                                              capsys):
    doc = _quickstart()
    del doc["propagators"]["n_grid_t"]
    doc["propagators"]["dt"] = 0.1
    path = _write(tmp_path, doc)
    out = _resolved(capsys, path, "--override", "sweep.seed=7")
    assert out["propagators"]["dt"] == 0.1
    assert out["propagators"]["n_grid_t"] == math.ceil(5.0 / 0.1)
    assert out["system"]["linear"]["n_grid_cache"] == math.ceil(100.0 / 0.1)
    assert out["sweep"]["seed"] == 7


@pytest.mark.parametrize("path", _shipped_configs(),
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_CL4_every_shipped_config_resolves(capsys, path):
    doc = _resolved(capsys, path)
    cfg = load_workflow_config(path)
    assert doc["sweep"]["t_final_grid"] == list(cfg.sweep.t_final_grid)
    assert doc["expand"]["orders"] == list(cfg.expand.orders)
    for vertex in doc["system"]["nonlocal_vertices"]:
        coupling = vertex["coupling"]
        assert isinstance(coupling, list) or coupling.startswith("<callable ")


def test_CL4_run_prints_the_resolved_config(tmp_path, capsys, monkeypatch):
    from sft_wick.workflow import estimate

    class _Estimate:
        def summary(self):
            return "[sft-wick] estimate: skipped"

    monkeypatch.setattr(estimate, "estimate_cost", lambda cfg: _Estimate())
    status = main(["run", str(QUICKSTART), "--dry-run", "-q",
                   "--print-resolved"])
    out = capsys.readouterr().out
    assert status == 0
    head, _, rest = out.partition("[sft-wick] resolved config:\n")
    assert rest
    body = rest.split("[sft-wick] system:")[0]
    assert yaml.safe_load(body)["sweep"]["method"] == "qmc_vectorized"


def _leaves(value, path: str = "") -> dict:
    """``{dotted path: value}`` of every value of a nested config."""
    if isinstance(value, dict):
        items = [(f"{path}.{k}" if path else str(k), v)
                 for k, v in value.items()]
    elif isinstance(value, list):
        items = [(f"{path}[{i}]", v) for i, v in enumerate(value)]
    else:
        return {path: value}
    return {k: v for key, item in items for k, v in _leaves(item, key).items()}


def test_CL4_run_prints_only_the_path_an_override_names(tmp_path, capsys,
                                                        monkeypatch):
    """``y`` is an alias of ``x`` in the file.  ``sft-wick run`` with an
    override of one entry of ``y`` prints that entry changed and nothing
    else, and prints what the file with the alias written out and the entry
    edited prints."""
    from sft_wick.workflow import estimate

    class _Estimate:
        def summary(self):
            return "[sft-wick] estimate: skipped"

    monkeypatch.setattr(estimate, "estimate_cost", lambda cfg: _Estimate())

    def printed(text: str, *extra: str) -> dict:
        path = tmp_path / "c.yaml"
        path.write_text(text)
        status = main(["run", str(path), "--dry-run", "-q",
                       "--print-resolved", *extra])
        out = capsys.readouterr().out
        assert status == 0
        body = out.partition("[sft-wick] resolved config:\n")[2]
        return yaml.safe_load(body.split("[sft-wick] system:")[0])

    text = QUICKSTART.read_text()
    grid = "positions_grid: {x: [0.0], y: [0.0, 0.5, 1.5]}"
    assert grid in text
    aliased = text.replace(grid, "positions_grid: {x: &g [0.0, 0.5, 1.5], y: *g}")
    edited = text.replace(
        grid, "positions_grid: {x: [0.0, 0.5, 1.5], y: [0.0, 0.7, 1.5]}")

    before = _leaves(printed(aliased))
    after = printed(aliased, "--override", "sweep.positions_grid.y[1]=0.7")
    changed = {k for k, v in _leaves(after).items() if before[k] != v}
    assert changed == {"sweep.positions_grid.y[1]"}
    assert after == printed(edited)


# ---------------------------------------------------------------------------
# CL5
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("override,field,value", [
    ("system.t_min=0.25", ("system", "t_min"), 0.25),
    ("expand.iso_C=true", ("expand", "iso_C"), True),
    ("propagators.c_method=gauss_legendre", ("propagators", "c_method"),
     "gauss_legendre"),
    ("expand.target=connected", ("expand", "target"), "connected"),
    ("sweep.method=gauss_legendre", ("sweep", "method"), "gauss_legendre"),
])
def test_CL5_override_sets_a_key_the_file_leaves_out(capsys, override, field,
                                                     value):
    """Each override here used to raise ``KeyError: override key ... does
    not exist in config`` (all but ``sweep.method``)."""
    assert main(["validate", str(QUICKSTART), "--override", override]) == 0
    cfg = load_workflow_config(QUICKSTART, overrides={".".join(field): value})
    block = getattr(cfg, field[0])
    assert getattr(block, field[1]) == value


def test_CL5_override_of_dt_matches_editing_the_file(tmp_path):
    doc = _quickstart()
    del doc["propagators"]["n_grid_t"]
    path = _write(tmp_path, doc)
    by_override = load_workflow_config(path, overrides={"propagators.dt": 0.1})
    doc["propagators"]["dt"] = 0.1
    by_edit = load_workflow_config(_write(tmp_path, doc, "edited.yaml"))
    assert by_override.propagators == by_edit.propagators
    assert by_override.system.linear == by_edit.system.linear


@pytest.mark.parametrize("override,code,fragment", [
    ("sweep.seed", "override_syntax", "KEY=VALUE"),
    ("sweep..seed=1", "override_syntax", "sweep..seed"),
    ("sweep.methd=nquad", "unknown_key", "did you mean 'method'?"),
    ("system.t_mni=0.5", "unknown_key", "did you mean 't_min'?"),
    ("output[3].path=x.md", "override_path", "output[3]"),
    ("system[0].linear=1", "override_path", "system[0]"),
    ("sweep.seed.x=1", "override_path", "sweep.seed"),
])
def test_CL5_bad_override_is_a_config_error(capsys, override, code, fragment):
    status = main(["validate", str(QUICKSTART), "--override", override])
    line = _one_line(capsys.readouterr().err)
    assert status == 2
    assert f"[{code}]" in line and fragment in line, line


# ---------------------------------------------------------------------------
# CL6
# ---------------------------------------------------------------------------

def test_CL6_strict_off_warns_and_ignores_the_key(tmp_path, capsys,
                                                  monkeypatch):
    monkeypatch.setenv("SFT_WICK_STRICT", "0")
    path = _write(tmp_path, _misspelt(("system",), "t_min", "t_mni"))
    assert main(["validate", str(path)]) == 0
    out = capsys.readouterr()
    assert _one_line(out.err) == (
        "sft-wick: config warning [unknown_key] system.t_mni: unknown key, "
        "ignored because SFT_WICK_STRICT=0; did you mean 't_min'?")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_workflow_config(path)
    assert [w.category for w in caught] == [ConfigWarning]
    clean = load_workflow_config(_write(tmp_path, _quickstart(), "clean.yaml"))
    assert cfg.system.t_min == clean.system.t_min == 0.0
    for block in ("expand", "propagators", "sweep", "output"):
        assert getattr(cfg, block) == getattr(clean, block), block
    assert cfg.system.linear == clean.system.linear


def test_CL6_strict_off_does_not_relax_overrides(capsys, monkeypatch):
    monkeypatch.setenv("SFT_WICK_STRICT", "0")
    status = main(["validate", str(QUICKSTART), "--override",
                   "system.t_mni=0.5"])
    assert status == 2
    assert "[unknown_key]" in _one_line(capsys.readouterr().err)


@pytest.mark.parametrize("setting", ["1", "true", "", "yes"])
def test_CL6_any_other_setting_is_strict(tmp_path, capsys, monkeypatch,
                                         setting):
    monkeypatch.setenv("SFT_WICK_STRICT", setting)
    path = _write(tmp_path, _misspelt(("system",), "t_min", "t_mni"))
    assert main(["validate", str(path)]) == 2
    capsys.readouterr()


# ---------------------------------------------------------------------------
# CL7
# ---------------------------------------------------------------------------

def test_CL7_python_api_error_types(tmp_path):
    path = _write(tmp_path, _misspelt(("system",), "t_min", "t_mni"))
    with pytest.raises(ConfigError) as info:
        load_workflow_config(path)
    err = info.value
    assert isinstance(err, ValueError)
    assert (err.code, err.path, err.suggestions) == (
        "unknown_key", "system.t_mni", ("t_min",))
    with pytest.raises(KeyError) as info:
        load_workflow_config(QUICKSTART, overrides={"output[4].path": "x"})
    assert isinstance(info.value, ConfigError)
    assert info.value.code == "override_path"
    assert str(info.value).startswith("output[4]: ")


def test_CL7_ties_list_every_nearest_key(tmp_path):
    """A misspelling as close to two keys as to the one meant names both,
    so the key meant is always among the suggestions."""
    doc = _quickstart()
    doc["propagators"]["n_grid_q"] = doc["propagators"].pop("n_grid_t")
    with pytest.raises(ConfigError) as info:
        load_workflow_config(_write(tmp_path, doc))
    assert set(info.value.suggestions) == {"n_grid_t", "n_grid_r",
                                           "n_grid_x"}
    copy.deepcopy(info.value)  # an error survives copying and pickling


# ---------------------------------------------------------------------------
# CL8 -- what validate and run report before anything is computed
# ---------------------------------------------------------------------------

def _gamma_module(tmp_path: Path) -> Path:
    module = tmp_path / "gm.py"
    module.write_text("def gamma(t):\n    return [1.0, 1.0]\n")
    return module


def _bad_inputs(tmp_path: Path) -> dict:
    """Configs refused only when the System is built or by a check that
    used to come after the expansion, each with a fragment of the line."""
    out = {}
    doc = _quickstart()
    doc["system"]["noise"]["kappa2"]["temporal"]["sigma_t"] = 0.0
    out["kernel_value"] = (doc, "sigma_t: must be positive")
    doc = _quickstart()
    doc["system"]["noise"]["kappa2"]["type"] = "separable_translaton"
    out["kappa2_type"] = (doc, "kappa2.type")
    doc = _quickstart()
    _gamma_module(tmp_path)
    doc["system"]["linear"] = {"type": "diagonal", "gamma_module": "gm.py",
                               "gamma_attr": "nope"}
    out["missing_attribute"] = (doc, "'nope'")
    doc = _quickstart()
    doc["sweep"]["n_samples"] = float("inf")
    out["infinite_integer"] = (doc, "sweep.n_samples")
    doc = _quickstart()
    (tmp_path / "empty.npy").write_bytes(b"")
    doc["system"]["vertices"][0] = {"name": "F", "coupling_path": "empty.npy"}
    out["empty_npy"] = (doc, "empty.npy")
    doc = _quickstart()
    doc["sweep"]["orders"] = [4]
    out["sweep_order"] = (doc, "sweep.orders: order(s) [4] are not in the expansion")
    doc = _quickstart()
    doc["sweep"]["component_pairs"] = [[0, 7]]
    out["component_index"] = (doc, "sweep.component_tuples")
    doc = _quickstart()
    doc["expand"]["n_jobs"] = 2
    doc["sweep"]["n_jobs"] = 2
    out["nested_jobs"] = (doc, "exactly one of")
    doc = _quickstart()
    doc["propagators"]["c_closed_form_module"] = "absent_c.py"
    out["closed_form_module"] = (doc, "absent_c.py")
    doc = _quickstart()
    doc["sweep"]["external_times_grid"] = {"x": [1.0], "z": [1.0]}
    out["external_point"] = (doc, "[unknown_point] sweep.external_times_grid.z")
    doc = _quickstart()
    doc["output"][0]["format"] = "markdwon"
    out["table_format"] = (doc, "did you mean 'markdown'?")
    doc = _quickstart()
    doc["propagators"]["homogeneity"] = "rotatoin"
    out["homogeneity"] = (doc, "did you mean 'rotation'?")
    doc = _quickstart()
    doc["system"]["noise"]["sigma2"] = {"type": ["constant"], "amplitude": 0.1}
    out["unhashable_type"] = (doc, "system.noise.sigma2")
    return out


_BAD_CASES = ("closed_form_module", "component_index", "empty_npy",
              "external_point", "homogeneity", "infinite_integer",
              "kappa2_type", "kernel_value", "missing_attribute",
              "nested_jobs", "sweep_order", "table_format",
              "unhashable_type")


@pytest.mark.parametrize("command", [["validate"], ["run", "--dry-run", "-q"]],
                         ids=["validate", "run"])
@pytest.mark.parametrize("case", _BAD_CASES)
def test_CL8_refused_before_anything_runs(tmp_path, capsys, monkeypatch,
                                          command, case):
    """Each of these used to pass ``validate`` or end ``run`` in a
    traceback (the build errors, a module without the attribute, an
    infinite integer, an empty .npy file, an unhashable type), or to stop
    ``run`` only after the expansion or the whole sweep (the order, the
    component index, the nested jobs, the closed-form module, a point of
    external_times_grid, the table format, the homogeneity)."""
    from sft_wick.workflow import estimate

    def no_estimate(cfg):  # the estimate must not be reached
        raise AssertionError("the config passed the checks")

    monkeypatch.setattr(estimate, "estimate_cost", no_estimate)
    cases = _bad_inputs(tmp_path)
    assert set(cases) == set(_BAD_CASES)
    doc, fragment = cases[case]
    path = _write(tmp_path, doc)
    status = main([command[0], str(path), *command[1:]])
    out = capsys.readouterr()
    assert status == 2, out
    line = _one_line(out.err)
    assert line.startswith("sft-wick: config error [") and fragment in line, line


def test_CL8_print_resolved_shows_the_diag_C_the_run_uses(capsys):
    """``propagators.diag_C: false`` also sets ``expand.diag_C`` false in
    the run; the record used to show it true."""
    doc = _resolved(capsys, QUICKSTART, "--override",
                    "propagators.diag_C=false")
    assert doc["propagators"]["diag_C"] is False
    assert doc["expand"]["diag_C"] is False


def test_CL8_strict_off_leaves_ignored_keys_out_of_the_record(
        tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SFT_WICK_STRICT", "0")
    doc = _quickstart()
    doc["system"]["linear"]["gamma_"] = 3.0
    doc["system"]["noise"]["kappa2"]["temporal"]["lamm"] = 9.0
    doc["system"]["vertices"][0]["ranc"] = 3
    doc["system"]["vertices"][0]["equal_time"] = "maybe"
    resolved = _resolved(capsys, _write(tmp_path, doc))
    system = resolved["system"]
    assert "gamma_" not in system["linear"]
    assert "lamm" not in system["noise"]["kappa2"]["temporal"]
    assert not {"ranc", "equal_time"} & set(system["vertices"][0])


# ---------------------------------------------------------------------------
# CL9 -- a malformed observable is a config error, in validate and in run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("command", [["validate"], ["run", "--dry-run", "-q"]],
                         ids=["validate", "run"])
@pytest.mark.parametrize("entry,path,fragment", [
    ([], "expand.observable[0]", "[]"),
    ({}, "expand.observable[0]", "{}"),
    (3, "expand.observable[0]", "3"),
    (None, "expand.observable[0]", "None"),
    ("phi_a", "expand.observable[0]", "'phi_a'"),
    ("phi_a(x", "expand.observable[0]", "'phi_a(x'"),
    ("wrong_a(x)", "expand.observable", "matches neither"),
    ("phi(x)", "expand.observable", "omits a component index"),
], ids=repr)
def test_CL9_malformed_observable_exits_2_with_one_line(tmp_path, capsys,
                                                        monkeypatch, command,
                                                        entry, path, fragment):
    from sft_wick.workflow import estimate

    def no_estimate(cfg):  # the estimate must not be reached
        raise AssertionError("the config passed the checks")

    monkeypatch.setattr(estimate, "estimate_cost", no_estimate)
    doc = _quickstart()
    doc["expand"]["observable"][0] = entry
    status = main([command[0], str(_write(tmp_path, doc)), *command[1:]])
    out = capsys.readouterr()
    assert status == 2, out
    line = _one_line(out.err)
    assert line.startswith(f"sft-wick: config error [invalid_value] {path}: ")
    assert fragment in line, line


@pytest.mark.parametrize("command", [("validate",), ("run", "--dry-run")],
                         ids=lambda c: "-".join(c))
@pytest.mark.parametrize("text,reason", [
    ('observable: [!!bool "x", "phi_b(y)"]', "the file cannot be read: 'x'"),
    ('observable: [!!int "", "phi_b(y)"]', "the file cannot be read: "),
    ('observable: [!!timestamp "x", "phi_b(y)"]', "the file cannot be read: "),
    ('observable: [!!int "x", "phi_b(y)"]', "invalid literal for int()"),
    ('observable: [2001-13-45, "phi_b(y)"]', "month must be in 1..12"),
    ('observable: [!!timestamp {=: x}, "phi_b(y)"]', "the file cannot be read: "),
    ("observable: [1" + ":0" * 180 + '.0, "phi_b(y)"]', "too large"),
    ("observable: " + "[" * 600 + "1" + "]" * 600, "nested too deeply to read"),
], ids=["bool-tag", "int-tag", "timestamp-tag", "int-word", "bad-date",
        "timestamp-mapping", "sexagesimal-overflow", "too-deep"])
def test_CL10_a_value_yaml_cannot_read_exits_2_with_one_line(
        tmp_path, capsys, command, text, reason):
    """PyYAML's constructors raise ``KeyError``, ``IndexError``,
    ``AttributeError``, ``TypeError``, ``ValueError`` or ``OverflowError``
    on such a value, and none of them is a ``yaml.YAMLError``."""
    path = tmp_path / "c.yaml"
    path.write_text(QUICKSTART.read_text().replace(
        'observable: ["phi_a(x)", "phi_b(y)"]', text))
    status = main([command[0], str(path), *command[1:]])
    out = capsys.readouterr()
    assert status == 2, out
    line = _one_line(out.err)
    assert line.startswith("sft-wick: config error [read_error] "), line
    assert reason in line, line


def test_CL10_a_file_of_another_encoding_exits_2_with_one_line(tmp_path, capsys):
    path = tmp_path / "c.yaml"
    path.write_bytes(QUICKSTART.read_bytes() + b"# caf\xe9\n")
    status = main(["validate", str(path)])
    out = capsys.readouterr()
    assert status == 2, out
    assert "config error [read_error] " in _one_line(out.err)
