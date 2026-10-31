"""Every refusal of the workflow parser is a ``ConfigError`` with a code and
a dotted path.

CE1 -- the keys whose refusal had no path, one by one.
CE2 -- a module a config names: one that Python cannot compile or run, one
       that is missing, a path or an attribute that is no string.  Each is a
       ``read_error`` with the dotted path of the key, and keeps the
       exception type it had for a caller of the Python API.
CE3 -- an ``.npy`` file ``np.load`` cannot read.
CE4 -- a refused value of the wrong type stays a ``TypeError``, and every
       error survives a pickle round trip (joblib workers raise them).
CE5 -- a message shows a value at a bounded size: an enormous integer, an
       alias diamond, a key.
CE6 -- the census: every key of two shipped configs, replaced by each of
       39 wrong values and deleted, gives a ``ConfigError`` with a path, or
       is accepted.  Run on the commit before this file, the same census of
       all shipped configs gave 29,363 plain exceptions out of 42,880 cases.
CE7 -- values the file can hold and the run cannot read are refused when the
       file is loaded: grid entries, grid sizes and table bounds.  A value
       that passes is returned as it was, so a config keeps its identity,
       and a key the method or the table does not read is not checked.
CE8 -- a nested list at a tensor key is measured before NumPy reads it.  A
       short file whose aliases expand past the limit is refused at once
       with the dotted path of the key; a tensor at the limit keeps its
       value, and a tensor in a ``.npy`` file has no limit.  A diamond past
       the first entry stays cheap, which lets the measure follow first
       entries only.  Text in a coupling is refused before NumPy stores it.
"""

from __future__ import annotations

import collections
import decimal
import fractions
import functools
import pickle
import signal
import sys
import time
import tracemalloc
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from sft_wick.workflow import config as cf
from sft_wick.workflow.cli import main
from sft_wick.workflow.config import (ConfigError, _parse_workflow, build_system,
                                      check_before_run, load_workflow_config)

ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = ROOT / "examples" / "quickstart.yaml"
#: A config with coupling modules next to it.
MODULES = ROOT / "examples" / "demo2" / "L2" / "config_FK.yaml"
HUGE = 10 ** 5000


def _case_id(value):
    """A test id: pytest prints an integer in full, which ``HUGE`` refuses."""
    return "huge" if value is HUGE else None


def _doc() -> dict:
    return yaml.safe_load(QUICKSTART.read_text())


def _write(tmp_path: Path, doc) -> Path:
    path = tmp_path / "c.yaml"
    text = doc if isinstance(doc, str) else yaml.safe_dump(doc, sort_keys=False)
    path.write_text(text)
    return path


def _refused(tmp_path: Path, doc: dict) -> ConfigError:
    """The refusal of ``doc``, read as a file would be from ``tmp_path``.
    YAML cannot write an integer past the digit limit, so no file is used."""
    with pytest.raises(ConfigError) as info:
        check_before_run(_parse_workflow(doc, tmp_path))
    return info.value


def _set(doc: dict, path: tuple, value) -> dict:
    node = doc
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = value
    return doc


# ---------------------------------------------------------------------------
# CE1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,value,dotted", [
    (("system", "linear", "type"), "nope", "system.linear.type"),
    (("output",), [{"type": "nope"}], "output[0].type"),
    (("output",), [{"type": "table"}, {"type": ["a"]}], "output[1].type"),
    (("output",), [{"path": "t.md"}], "output[0].type"),
    (("sweep", "gl_chunk_size"), "x", "sweep.gl_chunk_size"),
    (("sweep", "gl_chunk_size"), -1, "sweep.gl_chunk_size"),
    (("sweep", "qmc_coupling_chunk_size"), "x", "sweep.qmc_coupling_chunk_size"),
    (("sweep", "qmc_coupling_chunk_size"), 0, "sweep.qmc_coupling_chunk_size"),
    (("sweep", "propagator_cache_bytes"), "x", "sweep.propagator_cache_bytes"),
    (("sweep", "propagator_cache_bytes"), -1, "sweep.propagator_cache_bytes"),
    (("sweep", "propagator_cache_bytes"), 8, "sweep.propagator_cache_contract"),
    (("sweep", "propagator_cache_contract"), "x",
     "sweep.propagator_cache_contract"),
    (("sweep", "contraction_backend"), "x", "sweep.contraction_backend"),
    (("sweep", "contraction_max_bytes"), 0, "sweep.contraction_max_bytes"),
    (("sweep", "contraction_max_bytes"), "x", "sweep.contraction_max_bytes"),
    (("expand", "target"), "nope", "expand.target"),
    (("expand", "orders"), "x", "expand.orders"),
    (("sweep", "orders"), [0.5], "sweep.orders"),
    (("sweep", "vertex_types"), 3, "sweep.vertex_types"),
    (("sweep", "integrate_over"), 3, "sweep.integrate_over"),
    (("sweep", "component_pairs"), [[0, 9]], "sweep.component_tuples"),
    (("sweep", "orders"), [4], "sweep.orders"),
    (("system", "noise"), 3, "system.noise"),
    (("system", "field"), 3, "system.field"),
    (("expand",), 3, "expand"),
    (("output",), 3, "output"),
    (("propagators", "c_closed_form"), "x", "propagators.c_closed_form"),
    (("propagators", "c_method"), "x", "propagators.c_method"),
])
def test_CE1_a_refusal_names_the_key(tmp_path, capsys, path, value, dotted):
    doc = _set(_doc(), path, value)
    err = _refused(tmp_path, doc)
    assert (err.code, err.path) == ("invalid_value", dotted)
    assert "\n" not in str(err)
    assert not err.detail.startswith(dotted)

    status = main(["validate", str(_write(tmp_path, doc))])
    assert status == 2
    assert capsys.readouterr().err.startswith(
        f"sft-wick: config error [invalid_value] {dotted}: ")


def test_CE1_a_missing_section_or_key_names_it(tmp_path):
    doc = _doc()
    del doc["expand"]
    assert _refused(tmp_path, doc).path == "expand"
    doc = _doc()
    del doc["propagators"]["t_max"]
    assert _refused(tmp_path, doc).path == "propagators.t_max"
    doc = _doc()
    del doc["sweep"]["t_final_grid"]
    assert _refused(tmp_path, doc).path == "sweep.t_final_grid"
    doc = _doc()
    del doc["expand"]["observable"]
    assert _refused(tmp_path, doc).path == "expand.observable"
    doc = _doc()
    del doc["system"]["noise"]
    assert _refused(tmp_path, doc).path == "system.noise"


def test_CE1_a_vertex_path_is_a_dotted_path(tmp_path):
    """The name of the vertex is in the reason, not in the path, so the path
    can be written back as an override."""
    doc = _doc()
    doc["system"]["vertices"][0]["coupling"] = [1.0]
    err = _refused(tmp_path, doc)
    assert err.path == "system.vertices[0].coupling"
    assert "'F'" in err.detail


@pytest.mark.parametrize("raised,kind,detail", [
    (ValueError("a.b must be one"), ConfigError, "must be one"),
    (ValueError("a.b: not one"), ConfigError, "not one"),
    (ValueError("b must be one"), ConfigError, "must be one"),
    (ValueError("a: no such thing"), ConfigError, "no such thing"),
    (ValueError("a.bc must be one"), ConfigError, "a.bc must be one"),
    (ValueError("another reason"), ConfigError, "another reason"),
    (TypeError("b must be one"), cf._ConfigTypeError, "must be one"),
    (OverflowError("too large"), ConfigError, "too large"),
    (ZeroDivisionError("by zero"), ConfigError, "by zero"),
], ids=repr)
def test_CE1_a_plain_refusal_takes_the_path_of_the_block(raised, kind, detail):
    with pytest.raises(ConfigError) as info:
        with cf._refused_at("a.b"):
            raise raised
    assert type(info.value) is kind
    assert (info.value.path, info.value.detail) == ("a.b", detail)


def test_CE1_a_refusal_with_a_path_keeps_it():
    with pytest.raises(ConfigError) as info:
        with cf._refused_at("a.b"):
            raise ConfigError("unknown_key", "c.d", "why")
    assert (info.value.code, info.value.path) == ("unknown_key", "c.d")


def test_CE1_what_is_no_refusal_is_not_caught():
    with pytest.raises(KeyError):
        with cf._refused_at("a.b"):
            raise KeyError("a bug, not a value")


def test_CE1_a_misspelt_type_lists_the_type_meant(tmp_path):
    doc = _set(_doc(), ("system", "linear", "type"), "diagonl")
    assert _refused(tmp_path, doc).suggestions == ("diagonal",)
    doc = _set(_doc(), ("output",), [{"type": "tabel"}])
    assert _refused(tmp_path, doc).suggestions == ("table",)


# ---------------------------------------------------------------------------
# CE2
# ---------------------------------------------------------------------------

def _gamma(doc, name, attr=None):
    doc["system"]["linear"] = {"type": "diagonal", "gamma_module": name}
    if attr is not None:
        doc["system"]["linear"]["gamma_attr"] = attr


def _explicit(doc, name, attr=None):
    doc["system"]["linear"] = {"type": "explicit", "R_time_module": name}
    if attr is not None:
        doc["system"]["linear"]["R_time_attr"] = attr


def _vertex(doc, name, attr=None):
    doc["system"]["vertices"][0] = {"name": "F", "coupling_module": name}
    if attr is not None:
        doc["system"]["vertices"][0]["coupling_attr"] = attr


def _kappa2(doc, name, attr=None):
    doc["system"]["noise"]["kappa2"] = {"type": "callable_module", "module": name}
    if attr is not None:
        doc["system"]["noise"]["kappa2"]["attr"] = attr


def _sigma2(doc, name, attr=None):
    doc["system"]["noise"]["sigma2"] = {"type": "callable_module", "module": name}
    if attr is not None:
        doc["system"]["noise"]["sigma2"]["attr"] = attr


def _kernel(doc, name, attr=None):
    doc["system"]["noise"]["kappa2"]["temporal"] = {"type": "custom", "module": name}
    if attr is not None:
        doc["system"]["noise"]["kappa2"]["temporal"]["attr"] = attr


def _closed_form(doc, name, attr=None):
    doc["propagators"]["c_closed_form_module"] = name
    if attr is not None:
        doc["propagators"]["c_closed_form_attr"] = attr


#: how the config names a module -> (path of the key, path of its attribute)
MODULE_KEYS = {
    "gamma": (_gamma, "system.linear.gamma_module", "system.linear.gamma_attr"),
    "explicit": (_explicit, "system.linear.R_time_module",
                 "system.linear.R_time_attr"),
    "vertex": (_vertex, "system.vertices[0].coupling_module",
               "system.vertices[0].coupling_attr"),
    "kappa2": (_kappa2, "system.noise.kappa2.module", "system.noise.kappa2.attr"),
    "sigma2": (_sigma2, "system.noise.sigma2.module", "system.noise.sigma2.attr"),
    "kernel": (_kernel, "system.noise.kappa2.temporal.module",
               "system.noise.kappa2.temporal.attr"),
    "closed_form": (_closed_form, "propagators.c_closed_form_module",
                    "propagators.c_closed_form_attr"),
}

BROKEN = {
    "syntax_error": ("def (:\n", ImportError, "SyntaxError"),
    "raises": ("raise RuntimeError('at import')\n", ImportError, "RuntimeError"),
    "name_error": ("undefined_name\n", ImportError, "NameError"),
    "import_of_a_missing_module": ("import absent_module_xyz\n", ImportError,
                                   "absent_module_xyz"),
    "exits": ("import sys\nsys.exit(3)\n", ImportError, "SystemExit"),
}


@pytest.mark.parametrize("broken", BROKEN)
@pytest.mark.parametrize("key", MODULE_KEYS)
def test_CE2_a_module_python_cannot_import(tmp_path, capsys, key, broken):
    name_it, dotted, _ = MODULE_KEYS[key]
    text, kind, reason = BROKEN[broken]
    (tmp_path / "broken_mod.py").write_text(text)
    doc = _doc()
    name_it(doc, "broken_mod.py")
    err = _refused(tmp_path, doc)
    assert (err.code, err.path) == ("read_error", dotted)
    assert isinstance(err, kind)
    assert reason in str(err)
    assert "broken_mod" not in sys.modules
    assert main(["validate", str(_write(tmp_path, doc))]) == 2
    line = capsys.readouterr().err
    assert line.startswith(f"sft-wick: config error [read_error] {dotted}: ")
    assert len(line.strip().splitlines()) == 1


@pytest.mark.parametrize("key", MODULE_KEYS)
def test_CE2_a_missing_module_stays_an_OSError(tmp_path, key):
    name_it, dotted, _ = MODULE_KEYS[key]
    doc = _doc()
    name_it(doc, "absent_mod.py")
    err = _refused(tmp_path, doc)
    assert (err.code, err.path) == ("read_error", dotted)
    assert isinstance(err, FileNotFoundError)


@pytest.mark.parametrize("key", MODULE_KEYS)
def test_CE2_a_path_without_a_python_suffix(tmp_path, key):
    name_it, dotted, _ = MODULE_KEYS[key]
    (tmp_path / "notes.txt").write_text("x\n")
    doc = _doc()
    name_it(doc, "notes.txt")
    err = _refused(tmp_path, doc)
    assert (err.code, err.path) == ("read_error", dotted)
    assert isinstance(err, ImportError)


@pytest.mark.parametrize("value", [3, True, ["a"], {"a": 1}, HUGE, ""],
                         ids=_case_id)
@pytest.mark.parametrize("key", MODULE_KEYS)
def test_CE2_a_module_path_that_is_no_path(tmp_path, key, value):
    name_it, dotted, _ = MODULE_KEYS[key]
    doc = _doc()
    name_it(doc, value)
    err = _refused(tmp_path, doc)
    assert err.path == dotted
    assert err.code in ("invalid_value", "read_error")
    assert "\n" not in str(err)


@pytest.mark.parametrize("value", [3, True, ["a"], {"a": 1}, HUGE, ""],
                         ids=_case_id)
@pytest.mark.parametrize("key", MODULE_KEYS)
def test_CE2_an_attribute_that_is_no_name(tmp_path, key, value):
    name_it, _, attr_path = MODULE_KEYS[key]
    (tmp_path / "fine_mod.py").write_text("fn = len\n")
    doc = _doc()
    name_it(doc, "fine_mod.py", value)
    err = _refused(tmp_path, doc)
    assert err.path == attr_path
    assert "\n" not in str(err)


@pytest.mark.parametrize("key", MODULE_KEYS)
def test_CE2_a_module_without_the_attribute_names_the_attribute_key(tmp_path, key):
    name_it, _, attr_path = MODULE_KEYS[key]
    (tmp_path / "fine_mod.py").write_text("fn = len\n")
    doc = _doc()
    name_it(doc, "fine_mod.py", "absent")
    err = _refused(tmp_path, doc)
    assert (err.code, err.path) == ("read_error", attr_path)
    assert isinstance(err, AttributeError)


def test_CE2_a_default_attribute_the_module_lacks_names_the_module_key(tmp_path):
    """With no ``gamma_attr`` in the file the key to change is the module."""
    (tmp_path / "fine_mod.py").write_text("fn = len\n")
    doc = _doc()
    _gamma(doc, "fine_mod.py")
    err = _refused(tmp_path, doc)
    assert (err.code, err.path) == ("read_error", "system.linear.gamma_module")
    assert "'gamma'" in err.detail


# ---------------------------------------------------------------------------
# CE3
# ---------------------------------------------------------------------------

def _npy_coupling(doc: dict) -> dict:
    doc["system"]["vertices"][0] = {"name": "F", "coupling_path": "c.npy"}
    return doc


def test_CE3_a_file_numpy_takes_for_an_archive(tmp_path, capsys):
    (tmp_path / "c.npy").write_bytes(b"PK\x03\x04junk")
    doc = _npy_coupling(_doc())
    err = _refused(tmp_path, doc)
    assert (err.code, err.path) == ("read_error",
                                    "system.vertices[0].coupling_path")
    assert main(["validate", str(_write(tmp_path, doc))]) == 2
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize("error", [MemoryError("a shape too large"),
                                   zipfile.BadZipFile("not a zip file"),
                                   EOFError("no data"), OSError("disk")])
def test_CE3_each_failure_of_np_load(tmp_path, monkeypatch, error):
    (tmp_path / "c.npy").write_bytes(b"x")

    def load(*args, **kwargs):
        raise error

    monkeypatch.setattr(cf.np, "load", load)
    err = _refused(tmp_path, _npy_coupling(_doc()))
    assert (err.code, err.path) == ("read_error",
                                    "system.vertices[0].coupling_path")


# ---------------------------------------------------------------------------
# CE4
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,value", [
    (("sweep", "orders"), "x"),
    (("sweep", "vertex_types"), 3),
    (("sweep", "integrate_over"), 3),
    (("sweep", "gl_chunk_size"), "x"),
    (("sweep", "propagator_cache_bytes"), "x"),
])
def test_CE4_a_wrong_type_stays_a_TypeError(tmp_path, path, value):
    with pytest.raises(TypeError) as info:
        load_workflow_config(_write(tmp_path, _set(_doc(), path, value)))
    assert isinstance(info.value, ConfigError)
    assert info.value.path


@pytest.mark.parametrize("error", [
    cf.ConfigError("invalid_value", "a.b", "why", ("c",)),
    cf.OverrideError("override_path", "a.b", "why"),
    cf._ConfigTypeError("invalid_value", "a.b", "why"),
    cf._ModuleAttributeError("read_error", "a.b", "why"),
    cf._ModuleImportError("read_error", "a.b", "why"),
    cf._ModuleFileError("read_error", "a.b", "why"),
    cf._ModuleMissingError("read_error", "a.b", "why"),
], ids=lambda e: type(e).__name__)
def test_CE4_every_error_survives_a_pickle_round_trip(error):
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is type(error)
    assert (copy.code, copy.path, copy.detail, copy.suggestions) == (
        error.code, error.path, error.detail, error.suggestions)
    assert str(copy) == str(error)


# ---------------------------------------------------------------------------
# CE5
# ---------------------------------------------------------------------------

def test_CE5_an_enormous_integer_key(tmp_path):
    doc = _doc()
    doc["sweep"][HUGE] = 1
    err = _refused(tmp_path, doc)
    assert err.code == "unknown_key"
    assert err.path.startswith("sweep.")
    assert "Exceeds the limit" not in str(err)


@pytest.mark.parametrize("path", [
    ("sweep", "t_final_grid"), ("sweep", "n_samples"), ("expand", "orders"),
    ("sweep", "seed"), ("system", "linear", "gamma"), ("output",),
    ("sweep", "positions_grid", "y"), ("system", "field"),
])
def test_CE5_an_enormous_integer_value(tmp_path, path):
    err = _refused(tmp_path, _set(_doc(), path, HUGE))
    assert "Exceeds the limit" not in str(err)
    assert len(str(err)) < 400


def _diamond(depth: int) -> str:
    """YAML with an alias diamond: the value has ``2 ** depth`` leaves."""
    lines = ["a0: &a0 [1, 1]"]
    lines += [f"a{i}: &a{i} [*a{i - 1}, *a{i - 1}]" for i in range(1, depth)]
    return "\n".join(lines)


@pytest.mark.parametrize("key,template", [
    ("expand.observable", "expand: {{observable: {{a: *a{top}}}, orders: [0]}}"),
])
def test_CE5_an_alias_diamond_in_a_refused_value_is_cheap(key, template):
    """The message of a refused value used to hold ``repr`` of the value,
    made before it was known to be refused."""
    doc = yaml.safe_load(f"{_diamond(40)}\n{template.format(top=39)}")
    data = {k: v for k, v in doc.items() if not k.startswith("a")}
    with pytest.raises(ConfigError) as info:
        cf._parse_expand(data["expand"])
    assert info.value.path == key
    assert len(str(info.value)) < 400


@pytest.mark.parametrize("parse", [
    lambda v: cf._parse_expand({"observable": v, "orders": [0]}),
    lambda v: cf._parse_component_axis({"component_tuples": v}),
    lambda v: cf._parse_point_grid(v, "sweep.positions_grid"),
    lambda v: cf._integrate_over_selection({"a": v}),
])
def test_CE5_no_message_is_built_from_the_whole_of_a_diamond(parse):
    node = [1, 1]
    for _ in range(40):
        node = [node, node]
    started = time.perf_counter()
    with pytest.raises(ConfigError) as info:
        parse({"a": node})
    assert time.perf_counter() - started < 1.0
    assert len(str(info.value)) < 400


# ---------------------------------------------------------------------------
# CE6 -- the census
# ---------------------------------------------------------------------------

WRONG = [None, True, False, 0, 1, -1, 2, 1.5, -1.5, 1e308, float("inf"),
         float("nan"), HUGE, -HUGE, "", " ", "abc", "1e-3", "true", "\0",
         "x" * 300, "auto", [], [[]], [0], [1, 1], [1.5], ["a"], [None],
         [[1], [2, 3]], [True], [float("nan")], [HUGE], {}, {"a": 1},
         {"type": "nope"}, {1: 2}, {HUGE: 1}, {None: 1}]
_DELETED = object()
SECTIONS = ("system", "expand", "propagators", "sweep", "output")


def _paths(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield path + (key,)
            yield from _paths(value, path + (key,))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield path + (i,)
            yield from _paths(value, path + (i,))


def _changed(doc, path, value):
    """``doc`` with one value replaced, copying only the nodes on the way."""
    if not path:
        return value
    copied = doc.copy()
    if value is _DELETED and len(path) == 1:
        del copied[path[0]]
    else:
        copied[path[0]] = _changed(doc[path[0]], path[1:], value)
    return copied


@pytest.fixture
def _load_each_module_once(monkeypatch):
    monkeypatch.setattr(cf, "_load_callable_from_module",
                        functools.lru_cache(maxsize=None)(
                            cf._load_callable_from_module))


@pytest.mark.parametrize("config", [QUICKSTART, MODULES],
                         ids=["quickstart", "modules"])
def test_CE6_no_plain_exception_leaves_the_parser(config,
                                                  _load_each_module_once):
    doc = yaml.safe_load(config.read_text())
    plain, no_path, cases = [], [], 0
    for path in _paths(doc):
        for value in [_DELETED, *WRONG]:
            cases += 1
            data = _changed(doc, path, value)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    check_before_run(_parse_workflow(data, config.parent))
            except ConfigError as err:
                section = err.path.split(".")[0].split("[")[0]
                if section not in SECTIONS or "\n" in str(err):
                    no_path.append((path, str(err)[:200]))
            except Exception as exc:  # noqa: BLE001 -- recorded below
                plain.append((path, type(exc).__name__, str(exc)[:200]))
    assert cases > 2000
    assert not plain, (len(plain), plain[:10])
    assert not no_path, (len(no_path), no_path[:10])


# ---------------------------------------------------------------------------
# CE7 -- values the run cannot read
# ---------------------------------------------------------------------------

_GRID = ("sweep", "positions_grid", "y")
_TIMES = ("sweep", "t_final_grid")


@pytest.mark.parametrize("path,value,dotted", [
    (_TIMES, ["a"], "sweep.t_final_grid[0]"),
    (_TIMES, [1.0, None], "sweep.t_final_grid[1]"),
    (_TIMES, [[1.0]], "sweep.t_final_grid[0]"),
    (_TIMES, [float("nan")], "sweep.t_final_grid[0]"),
    (_TIMES, [float("inf")], "sweep.t_final_grid[0]"),
    (_TIMES, [True], "sweep.t_final_grid[0]"),
    (("sweep", "external_times_grid"), {"x": ["a"]},
     "sweep.external_times_grid.x[0]"),
    (_GRID, ["a"], "sweep.positions_grid.y[0]"),
    (_GRID, [0.0, None], "sweep.positions_grid.y[1]"),
    (_GRID, [float("nan")], "sweep.positions_grid.y[0]"),
    (_GRID, [float("inf")], "sweep.positions_grid.y[0]"),
    (_GRID, [[]], "sweep.positions_grid.y[0]"),
    (_GRID, [[0.0, "a"]], "sweep.positions_grid.y[0][1]"),
    (_GRID, [[0.0, float("nan")]], "sweep.positions_grid.y[0][1]"),
    (_GRID, [[[0.0]]], "sweep.positions_grid.y[0][0]"),
    (_GRID, [{"a": 1}], "sweep.positions_grid.y[0]"),
    (("sweep", "n_samples"), 0, "sweep.n_samples"),
    (("sweep", "n_samples"), -1, "sweep.n_samples"),
    (("sweep", "n_samples"), HUGE, "sweep.n_samples"),
    (("sweep", "n_gauss"), HUGE, "sweep.n_gauss"),
    (("propagators", "n_grid_t"), 1, "propagators.n_grid_t"),
    (("propagators", "n_grid_t"), 0, "propagators.n_grid_t"),
    (("propagators", "n_grid_t"), HUGE, "propagators.n_grid_t"),
    (("propagators", "c_n_gauss"), 1, "propagators.c_n_gauss"),
    (("propagators", "c_n_gauss"), HUGE, "propagators.c_n_gauss"),
    (("propagators", "t_max"), float("nan"), "propagators.t_max"),
    (("propagators", "t_max"), float("inf"), "propagators.t_max"),
    (("propagators", "interp_method"), "nope", "propagators.interp_method"),
    (("propagators", "interp_method"), None, "propagators.interp_method"),
    (("sweep", "seed"), -1, "sweep.seed"),
    (("propagators", "r_max"), "a", "propagators.r_max"),
    (("propagators", "r_max"), 0.0, "propagators.r_max"),
    (("propagators", "r_max"), -1.0, "propagators.r_max"),
    (("propagators", "r_max"), float("nan"), "propagators.r_max"),
    (("propagators", "x_max"), [1.0], "propagators.x_max"),
    (("propagators", "n_grid_r"), "a", "propagators.n_grid_r"),
    (("propagators", "n_grid_r"), 0, "propagators.n_grid_r"),
    (("propagators", "n_grid_r"), 1.5, "propagators.n_grid_r"),
    (("propagators", "n_grid_r"), True, "propagators.n_grid_r"),
    (("propagators", "n_grid_x"), 0, "propagators.n_grid_x"),
    (("propagators", "n_grid_cos"), -1, "propagators.n_grid_cos"),
    (("system", "t_min"), float("nan"), "system.t_min"),
    (("system", "t_min"), float("inf"), "system.t_min"),
    (("sweep", "n_jobs"), 0, "sweep.n_jobs"),
    (("expand", "n_jobs"), 0, "expand.n_jobs"),
    (("propagators", "n_jobs"), 0, "propagators.n_jobs"),
    (("expand", "cache_path"), 3, "expand.cache_path"),
    (("propagators", "cache_path"), ["a"], "propagators.cache_path"),
    (("propagators", "cache_path"), "a\0b", "propagators.cache_path"),
    (("output",), [{"type": "table", "path": 3}], "output[0].path"),
    (("output",), [{"type": "table", "path": ""}], "output[0].path"),
    (("output",), [{"type": "npz", "path": "a\0b"}], "output[0].path"),
], ids=_case_id)
def test_CE7_a_value_the_run_cannot_read(tmp_path, path, value, dotted):
    err = _refused(tmp_path, _set(_doc(), path, value))
    assert (err.code, err.path) == ("invalid_value", dotted)


@pytest.mark.parametrize("path,value", [
    (_TIMES, ["1e-3", "1e-1"]),  # how YAML reads 1e-3
    (_GRID, [0.0, "1e-3"]),
    (_GRID, [[0.0, 1.0], [1.0, 0.0]]),
    (_GRID, [0, 1]),
    (("sweep", "n_jobs"), -1),
    (("propagators", "interp_method"), "cubic"),
    (("propagators", "n_grid_t"), 2),
])
def test_CE7_a_value_the_run_can_read_is_kept(tmp_path, path, value):
    load_workflow_config(_write(tmp_path, _set(_doc(), path, value)))


@pytest.mark.parametrize("value", [0, -1], ids=["zero", "negative"])
def test_CE7_n_gauss_under_the_method_that_reads_it(tmp_path, value):
    doc = _set(_doc(), ("sweep", "method"), "gauss_legendre")
    assert _refused(tmp_path, _set(doc, ("sweep", "n_gauss"), value)).path == (
        "sweep.n_gauss")


def _value(doc: dict, tmp_path: Path) -> float:
    doc["expand"]["orders"] = [0]
    doc["output"] = []
    doc["sweep"].update(t_final_grid=[1.0], component_pairs=[[0, 0]])
    cfg = _parse_workflow(doc, tmp_path)
    return float(cf.run_workflow(cfg, progress=False)[1]["value"].iloc[0])


@pytest.mark.parametrize("method,changes", [
    ("gauss_legendre", {"seed": -1}),
    ("gauss_legendre", {"n_samples": 0}),
    ("gauss_legendre", {"n_samples": -3}),
    ("nquad", {"seed": -1, "n_samples": 0, "n_gauss": 0}),
    ("qmc_vectorized", {"n_gauss": 0}),
    ("qmc", {"n_gauss": -3}),
], ids=str)
def test_CE7_a_key_the_method_does_not_read_is_not_checked(tmp_path, method,
                                                           changes):
    """The keys of another method change no number, so a value of them is no
    reason to refuse the run."""
    doc = _set(_doc(), ("sweep", "method"), method)
    expected = _value(doc, tmp_path)
    doc = _set(_doc(), ("sweep", "method"), method)
    doc["sweep"].update(changes)
    assert _value(doc, tmp_path) == expected


@pytest.mark.parametrize("changes", [
    {"t_max": 0.0}, {"t_max": float("inf")}, {"t_max": float("nan")},
    {"n_grid_t": 1}, {"n_grid_t": 0}, {"n_grid_t": -5},
], ids=str)
def test_CE7_a_key_no_table_reads_is_not_checked(tmp_path, changes):
    """``c_closed_form_only`` builds no table."""
    doc = _set(_doc(), ("propagators", "c_closed_form_only"), True)
    expected = _value(doc, tmp_path)
    doc = _set(_doc(), ("propagators", "c_closed_form_only"), True)
    doc["propagators"].update(changes)
    assert _value(doc, tmp_path) == expected


def test_CE7_python_overrides_of_the_types_the_run_reads(tmp_path):
    """A named tuple, a 0-d array, a Fraction and a Decimal come from
    ``load_workflow_config(path, overrides=...)``, and each runs."""
    point = collections.namedtuple("Point", "a b c")
    path = _write(tmp_path, _doc())
    cfg = load_workflow_config(path, overrides={"sweep.positions_grid": {
        "x": [point(0.0, 0.0, 0.0)], "y": [point(0.5, 0.0, 0.0)]}})
    assert cfg.sweep.positions_grid["y"] == [point(0.5, 0.0, 0.0)]
    check_before_run(cfg)

    cfg = load_workflow_config(path, overrides={
        "propagators.r_max": np.array(3.0), "propagators.n_grid_r": 8,
        "sweep.t_final_grid": [np.array(1.0), fractions.Fraction(1, 2),
                               decimal.Decimal("1.5")]})
    assert cfg.sweep.t_final_grid == [1.0, 0.5, 1.5]
    check_before_run(cfg)


def test_CE7_a_number_written_as_YAML_reads_it_is_a_number(tmp_path):
    cfg = load_workflow_config(_write(tmp_path, _set(_doc(), _TIMES, ["1e-3"])))
    assert cfg.sweep.t_final_grid == [0.001]


def test_CE7_a_grid_of_the_python_api_is_read_as_before(tmp_path):
    """``load_workflow_config`` takes a NumPy array in place of a list."""
    grid = np.linspace(0.5, 2.5, 3)
    cfg = load_workflow_config(_write(tmp_path, _doc()),
                               overrides={"sweep.t_final_grid": grid})
    assert cfg.sweep.t_final_grid == list(grid)
    cfg = load_workflow_config(_write(tmp_path, _doc()), overrides={
        "sweep.positions_grid.y": np.eye(3)})
    assert all(np.array_equal(a, b)
               for a, b in zip(cfg.sweep.positions_grid["y"], np.eye(3)))


# ---------------------------------------------------------------------------
# CE8 -- the size of a tensor, before NumPy reads it
# ---------------------------------------------------------------------------

G0 = [[1.0, 0.0], [0.0, 1.0]]
G1 = [[[0.0, 0.25], [0.5, 0.0]], [[0.0, 0.0], [0.75, 0.0]]]
#: ``dotted key: (the value in a valid file, what places a value there)``
TENSOR_KEYS = {
    "system.vertices[0].coupling": (
        [[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.5], [0.5, 0.0]]],
        lambda system, v: system["vertices"][0].update(coupling=v)),
    "system.nonlocal_vertices[0].coupling": (
        [[1.0, 0.5], [0.5, 0.0]],
        lambda system, v: system.update(nonlocal_vertices=[
            {"name": "K", "order": 2, "coupling": v}])),
    "system.noise.sigma2.amplitude": (
        [[1.0, 0.25], [0.25, 2.0]],
        lambda system, v: system["noise"].update(
            sigma2={"type": "constant", "amplitude": v})),
    "system.noise.sigma2.g0": (
        G0,
        lambda system, v: system["noise"].update(
            sigma2={"type": "multiplicative", "g0": v, "g1": G1})),
    "system.noise.sigma2.g1": (
        G1,
        lambda system, v: system["noise"].update(
            sigma2={"type": "multiplicative", "g0": G0, "g1": v})),
}
#: Reads each key back from the built system.
BUILT = {
    "system.vertices[0].coupling": lambda s: s.vertices[0].coupling,
    "system.nonlocal_vertices[0].coupling":
        lambda s: s.nonlocal_vertices[0].coupling,
    "system.noise.sigma2.amplitude": lambda s: s.noise.sigma2.amplitude,
    "system.noise.sigma2.g0": lambda s: s.noise.sigma2.g0,
    "system.noise.sigma2.g1": lambda s: s.noise.sigma2.g1,
}


@pytest.fixture
def hang_fails():
    """A walk of a diamond would not end.  NumPy polls for signals between
    the lists it reads, so an alarm turns the walk into a failure.  The
    failure is a ``BaseException``, which no handler in the parser catches."""

    def expired(signum, frame):
        pytest.fail("still walking the value")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(5)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)


def _doubling(depth: int, leaves: tuple = (1.0, 1.0)) -> list:
    """Each list holds the one before it twice: ``2 ** depth`` leaves.  A
    dump writes the shared lists as aliases nested inside the value, as a
    file must, since a key at the top level for the anchors would be refused."""
    node = list(leaves)
    for _ in range(depth - 1):
        node = [node, node]
    return node


def _shaped(shape: tuple) -> list:
    """Nested lists of ``shape``, built without writing the entries out."""
    node = 0.0
    for size in reversed(shape):
        node = [node] * size
    return node


def _with_tensor(tmp_path: Path, key: str, value) -> Path:
    doc = _doc()
    TENSOR_KEYS[key][1](doc["system"], value)
    return _write(tmp_path, doc)


def _built(path: Path, key: str) -> np.ndarray:
    return BUILT[key](build_system(load_workflow_config(path).system))


@pytest.mark.parametrize("key", TENSOR_KEYS)
def test_CE8_validate_refuses_a_diamond_at_once(tmp_path, capsys, hang_fails,
                                                key):
    path = _with_tensor(tmp_path, key, _doubling(40))
    assert path.stat().st_size < 8000
    started = time.perf_counter()
    assert main(["validate", str(path)]) == 2
    assert time.perf_counter() - started < 1.0
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(f"sft-wick: config error [invalid_value] {key}: ")


@pytest.mark.parametrize("key", TENSOR_KEYS)
def test_CE8_the_error_has_the_code_and_the_path(tmp_path, hang_fails, key):
    path = _with_tensor(tmp_path, key, _doubling(40))
    with pytest.raises(ConfigError) as info:
        check_before_run(load_workflow_config(path))
    assert (info.value.code, info.value.path) == ("invalid_value", key)
    assert len(str(info.value)) < 300


@pytest.mark.parametrize("key", TENSOR_KEYS)
def test_CE8_a_tensor_keeps_its_value(tmp_path, key):
    tensor = TENSOR_KEYS[key][0]
    path = _with_tensor(tmp_path, key, tensor)
    assert np.array_equal(_built(path, key), np.array(tensor))


def test_CE8_a_tensor_with_aliased_rows_reads_as_the_one_written_out(tmp_path):
    row = [0.25, 0.25]
    key = "system.noise.sigma2.amplitude"
    path = _with_tensor(tmp_path, key, [row, row])
    assert "*id" in path.read_text()
    assert np.array_equal(_built(path, key), np.full((2, 2), 0.25))


def test_CE8_a_tensor_at_the_limit_passes_and_one_past_it_is_refused(
        tmp_path, monkeypatch):
    """The quick start holds one tensor, of 8 entries."""
    key = "system.vertices[0].coupling"
    tensor = TENSOR_KEYS[key][0]
    path = _with_tensor(tmp_path, key, tensor)
    monkeypatch.setattr(cf, "_MAX_TENSOR_ENTRIES", 8)
    assert np.array_equal(_built(path, key), np.array(tensor))
    monkeypatch.setattr(cf, "_MAX_TENSOR_ENTRIES", 7)
    with pytest.raises(ConfigError) as info:
        load_workflow_config(path)
    assert (info.value.code, info.value.path) == ("invalid_value", key)


def test_CE8_a_tensor_in_a_file_has_no_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(cf, "_MAX_TENSOR_ENTRIES", 1)
    tensor = np.array(TENSOR_KEYS["system.vertices[0].coupling"][0])
    np.save(tmp_path / "coupling.npy", tensor)
    doc = _doc()
    doc["system"]["vertices"][0] = {"name": "F", "coupling_path": "coupling.npy"}
    path = _write(tmp_path, doc)
    built = build_system(load_workflow_config(path).system).vertices[0].coupling
    assert np.array_equal(built, tensor)


@pytest.mark.parametrize("shape", [
    (1,), (1,) * 64, (10_000_000,), (3162, 3162), (215, 215, 215),
    (100, 100, 100), (10,) * 7, (3,) * 14, (2,) * 23,
], ids=lambda shape: f"{shape[0]}^{len(shape)}")
def test_CE8_the_shapes_a_system_can_ask_for_pass(shape):
    cf._refuse_oversized_tensor(_shaped(shape), "key")


@pytest.mark.parametrize("shape", [
    (10_000_001,), (3163, 3163), (216, 216, 216), (10,) * 8, (3,) * 15,
    (2,) * 24, (1,) * 40 + (2,) * 24,
], ids=lambda shape: f"{shape[-1]}^{len(shape)}")
def test_CE8_the_next_size_up_is_refused(shape):
    with pytest.raises(ConfigError) as info:
        cf._refuse_oversized_tensor(_shaped(shape), "key")
    assert info.value.path == "key"


@pytest.mark.parametrize("value", [
    [1.0, _doubling(40)],
    [[1.0], _doubling(40)],
    [[1.0, 1.0], _doubling(40)],
    [[[1.0, 1.0], [1.0, 1.0]], _doubling(40)],
    [[], _doubling(40)],
    [None, _doubling(40)],
    ["a", _doubling(40)],
    [{}, _doubling(40)],
    [{"k": _doubling(40)}, _doubling(40)],
], ids=["number", "short", "row", "matrix", "empty", "null", "text", "mapping",
        "mapping_holding_it"])
@pytest.mark.parametrize("key", TENSOR_KEYS)
def test_CE8_a_diamond_past_the_first_entry_stays_cheap(tmp_path, capsys,
                                                        hang_fails, key, value):
    """The measure follows first entries.  That bounds the walk because the
    first number ends NumPy's descent: a list below that depth is ragged and
    is not entered.  ``g0`` and ``g1`` report such a list at their block."""
    path = _with_tensor(tmp_path, key, value)
    started = time.perf_counter()
    assert main(["validate", str(path)]) == 2
    assert time.perf_counter() - started < 1.0
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1
    assert "[invalid_value] system." in lines[0]


@pytest.mark.parametrize("axes", [10, 60, 70])
def test_CE8_a_diamond_behind_axes_of_length_one_is_refused(tmp_path, hang_fails,
                                                            axes):
    """Past the axes the measure follows, NumPy refuses the list by itself."""
    value = _doubling(40)
    for _ in range(axes):
        value = [value]
    key = "system.vertices[0].coupling"
    with pytest.raises(ConfigError) as info:
        check_before_run(load_workflow_config(_with_tensor(tmp_path, key, value)))
    assert info.value.path == key


@pytest.mark.parametrize("leaves", [
    ("x" * 8, "x" * 8), (1.0, "x" * 8), (b"x" * 8, b"x" * 8),
], ids=["text", "text_after_a_number", "bytes"])
@pytest.mark.parametrize("key", [
    "system.vertices[0].coupling", "system.nonlocal_vertices[0].coupling",
])
def test_CE8_text_in_a_coupling_is_refused_before_numpy_stores_it(
        tmp_path, hang_fails, key, leaves):
    """NumPy stores every entry at the width of the longest string, so 4 kB
    of aliases around a long string asked for tens of GB.  Seventeen levels
    stay under the limit on what aliases add, so this guard is the one that
    answers, and eight characters keep a failure to a few MB."""
    path = _with_tensor(tmp_path, key, _doubling(17, leaves))
    tracemalloc.start()
    try:
        with pytest.raises(ConfigError) as info:
            load_workflow_config(path)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert (info.value.code, info.value.path) == ("invalid_value", key)
    assert "must be a tensor of finite numbers" in str(info.value)
    assert peak < 20 * 2 ** 20


def test_CE8_the_search_for_text_enters_each_list_once(hang_fails):
    assert not cf._holds_text(_doubling(400))
    assert cf._holds_text(_doubling(400, (1.0, "x")))
    assert cf._holds_text(((1.0, "x"),))
    loop: list = []
    loop.append(loop)
    assert not cf._holds_text(loop)


def test_CE8_a_tuple_is_measured_as_a_list_is():
    """A Python caller can pass tuples, which NumPy walks as it walks lists."""
    node = (1.0, 1.0)
    for _ in range(39):
        node = (node, node)
    with pytest.raises(ConfigError):
        cf._refuse_oversized_tensor(node, "key")


def test_CE8_a_list_holding_itself_ends_the_measure(hang_fails):
    """A Python caller reaches this without the check for cycles; NumPy
    refuses such a list by itself."""
    loop: list = []
    loop.append(loop)
    cf._refuse_oversized_tensor(loop, "key")


def test_CE8_an_array_is_not_measured(monkeypatch):
    monkeypatch.setattr(cf, "_MAX_TENSOR_ENTRIES", 1)
    cf._refuse_oversized_tensor(np.zeros((4, 4)), "key")
