"""Two corpora over the shipped workflow configs: misspelt keys and
overrides of keys a file leaves out.

CO1 -- every single-edit misspelling (a deletion, a transposition of two
       neighbours, a substitution) of every key in every shipped config is
       refused as an unknown key: the error is one line, names the dotted
       path of the misspelt key, and suggests the key that was misspelt.
       A misspelt point name in ``sweep.positions_grid`` is refused the
       same way.  An edit that gives another key the block accepts
       (``n_grid_t`` to ``n_grid_r``) or the empty string is left out,
       since no key check can tell it from an intended key.  The
       substitutions of all configs but the quick start are marked
       ``slow``; ``pytest -m slow tests/test_workflow_corpora.py`` runs
       them.
CO2 -- every key the schema allows in a block of a shipped config, which
       the file leaves out, can be set through an override, and the result
       is the result of writing the key into the file: the same parsed
       config, or the same refusal by a rule that ties two keys together.
CO4 -- an override changes only the path it names when the file shares a
       node through a YAML alias or a ``<<:`` merge: the result is that of
       the file with the sharing written out and the same edit made by
       hand.  Only the containers on the path are copied, so a file whose
       aliases expand far beyond its size stays that size.
CO5 -- a file where an alias makes a mapping or list hold itself is refused
       when it is loaded, with the path of that alias.  A node shared
       without a cycle stays accepted, however far it expands.
CO6 -- an error message shows a value at a bounded length, at every key of
       every shipped config: a value that expands through shared nodes
       gives a short message, and a short value prints as ``repr`` prints
       it.

Before the schema and the one-line errors, CO1 failed on every case (the
error was a ``ValueError`` listing the allowed keys, without the dotted
path of the key or a suggestion, and keys of the noise and kernel blocks
passed the parser) and CO2 raised ``KeyError`` for every key outside
``sweep``.  Before an override copied the containers on its path, the
cases of ``SHARED`` in CO4 wrote through the alias without an error.  A
file of CO5 passed ``sft-wick validate``, and ``--print-resolved`` and
``run`` then ended in a traceback.  The messages of CO6 expanded every
shared node, 6 MB per message at the depth used here.  The
parser's result is compared, not its own report: the reference is the
parsed :class:`~sft_wick.workflow.config.WorkflowConfig`, written out field
by field by :func:`_canonical` below.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import functools
import json
import string
import sys
import time
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import yaml

from sft_wick.workflow import config as cf
from sft_wick.workflow._messages import MAX_SHOWN_CHARS, shown, text_of
from sft_wick.workflow.config import ConfigError, load_workflow_config
from sft_wick.workflow.schema import workflow_json_schema

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = workflow_json_schema()
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)
#: Substitutions use the characters of YAML keys; the case-swapped original
#: character is added per position, so ``diag_C`` gives ``diag_c``.
ALPHABET = string.ascii_lowercase + string.digits + "_"
#: Mappings whose keys are point names rather than config keys.
POINT_GRIDS = {("sweep", "positions_grid"), ("sweep", "external_times_grid")}


def _shipped_configs() -> list[Path]:
    paths = sorted(p for p in (ROOT / "examples").rglob("*.yaml")
                   if "sweep" in (yaml.safe_load(p.read_text()) or {}))
    paths.append(ROOT / "src" / "sft_wick" / "workflow" / "quickstart.yaml")
    return paths


CONFIGS = _shipped_configs()
IDS = [str(p.relative_to(ROOT)) for p in CONFIGS]


@pytest.fixture(autouse=True)
def _load_each_module_once(monkeypatch):
    """Coupling and kernel modules are loaded once per (file, attribute):
    re-running a module for each of ~10^5 cases costs minutes, and which
    function object is returned does not touch the key checks."""
    monkeypatch.setattr(cf, "_load_callable_from_module",
                        functools.lru_cache(maxsize=None)(
                            cf._load_callable_from_module))


# ---------------------------------------------------------------------------
# The schema, walked alongside a config (written here, not taken from the
# package)
# ---------------------------------------------------------------------------

def _alternatives(node: dict) -> list[dict]:
    while "$ref" in node:
        node = SCHEMA["$defs"][node["$ref"].removeprefix("#/$defs/")]
    for key in ("anyOf", "oneOf"):
        if key in node:
            return [a for alt in node[key] for a in _alternatives(alt)]
    return [node]


def _object_for(node: dict, data: dict) -> dict | None:
    """The object schema ``data`` is read with: the variant its ``type``
    names, or the one whose ``type`` may be left out."""
    objects = [a for a in _alternatives(node) if a.get("type") == "object"]
    tagged = [a for a in objects if "const" in a.get("properties", {}).get(
        "type", {})]
    if not tagged:
        return objects[0] if objects else None
    if "type" in data:
        return next((a for a in tagged
                     if a["properties"]["type"]["const"] == data["type"]),
                    None)
    return next((a for a in tagged if "type" not in a.get("required", ())),
                None)


def _mappings(data, node=SCHEMA, path=()):
    """Yield ``(path, mapping, allowed, kind)`` for every mapping in a
    config: ``kind`` is ``"key"`` with the keys the schema allows there,
    or ``"point"`` for a grid keyed by point names."""
    if isinstance(data, dict):
        obj = _object_for(node, data)
        if obj is None:
            return
        if path[-2:] in POINT_GRIDS:
            yield path, data, None, "point"
            return
        props = obj.get("properties", {})
        yield path, data, set(props), "key"
        for key, value in data.items():
            if key in props:
                yield from _mappings(value, props[key], path + (key,))
    elif isinstance(data, list):
        arrays = [a for a in _alternatives(node) if a.get("type") == "array"]
        if arrays and "items" in arrays[0]:
            for i, value in enumerate(data):
                yield from _mappings(value, arrays[0]["items"], path + (i,))


def _dotted(path: tuple) -> str:
    out = ""
    for step in path:
        out += f"[{step}]" if isinstance(step, int) else (
            f".{step}" if out else str(step))
    return out


ALL_EDITS = ("delete", "swap", "substitute")


def _edits(key: str, kinds=ALL_EDITS) -> set:
    """The deletions, neighbour transpositions and substitutions of
    ``key`` named in ``kinds``."""
    out = set()
    for i in range(len(key)):
        if "delete" in kinds:
            out.add(key[:i] + key[i + 1:])
        if "substitute" in kinds:
            for c in ALPHABET + key[i].swapcase():
                out.add(key[:i] + c + key[i + 1:])
    if "swap" in kinds:
        for i in range(len(key) - 1):
            out.add(key[:i] + key[i + 1] + key[i] + key[i + 2:])
    out.discard(key)
    out.discard("")
    return out


def _rename(mapping: dict, old: str, new: str) -> None:
    items = [(new if k == old else k, v) for k, v in mapping.items()]
    mapping.clear()
    mapping.update(items)


def _observable_points(doc: dict) -> set:
    return {op.split("(")[1].rstrip(")") for op in doc["expand"]["observable"]}


# ---------------------------------------------------------------------------
# CO1 -- the typo corpus
# ---------------------------------------------------------------------------

#: The config whose whole corpus runs by default; the substitutions of the
#: other configs (95 % of their cases, about a minute) are marked slow.
FULL_BY_DEFAULT = ROOT / "examples" / "quickstart.yaml"


@pytest.mark.parametrize("path", CONFIGS, ids=IDS)
def test_CO1_every_single_edit_misspelling_is_refused(path):
    """Deletions and transpositions in every config; substitutions too in
    the quick start."""
    kinds = ALL_EDITS if path == FULL_BY_DEFAULT else ("delete", "swap")
    _check_typo_corpus(path, kinds)


@pytest.mark.slow
@pytest.mark.parametrize("path", [p for p in CONFIGS if p != FULL_BY_DEFAULT],
                         ids=[i for p, i in zip(CONFIGS, IDS)
                              if p != FULL_BY_DEFAULT])
def test_CO1_every_substitution_is_refused(path):
    """The rest of the corpus: substitutions in every other config."""
    _check_typo_corpus(path, ("substitute",))


def _check_typo_corpus(path: Path, kinds) -> None:
    doc = yaml.safe_load(path.read_text())
    pristine = copy.deepcopy(doc)
    points = _observable_points(doc)
    counts = {"cases": 0, "valid_key_edits": 0}
    failures = []
    for where, mapping, allowed, kind in list(_mappings(doc)):
        valid = allowed if kind == "key" else points
        code = "unknown_key" if kind == "key" else "unknown_point"
        for key in list(mapping):
            edits = sorted(_edits(str(key), kinds))
            schema_checked = False
            for edit in edits:
                if edit in valid or edit in mapping:
                    counts["valid_key_edits"] += 1
                    continue
                counts["cases"] += 1
                _rename(mapping, key, edit)
                try:
                    cf._parse_workflow(doc, base_dir=path.parent)
                    err = None
                except Exception as exc:  # noqa: BLE001 -- recorded below
                    err = exc
                if kind == "key" and not schema_checked:
                    # The schema refuses the key too (one edit per key; a
                    # schema run per case would take minutes).
                    assert not VALIDATOR.is_valid(doc), _dotted(
                        where + (edit,))
                    schema_checked = True
                _rename(mapping, edit, key)
                expected = _dotted(where + (edit,))
                if not (isinstance(err, ConfigError) and err.code == code
                        and err.path == expected and key in err.suggestions
                        and "\n" not in str(err)):
                    failures.append((expected, key, repr(err)))
    assert doc == pristine, "the parser wrote into its input"
    assert counts["cases"] > 300
    assert not failures, (len(failures), failures[:5])


# ---------------------------------------------------------------------------
# CO2 -- the override corpus
# ---------------------------------------------------------------------------

def _canonical(obj):
    """A parsed config, written out exactly: floats by ``float.hex``,
    arrays with dtype and shape, a loaded callable by its qualified name."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {type(obj).__name__: {f.name: _canonical(getattr(obj, f.name))
                                     for f in dataclasses.fields(obj)}}
    if isinstance(obj, dict):
        return {"dict": sorted([json.dumps(_canonical(k)), _canonical(v)]
                               for k, v in obj.items())}
    if isinstance(obj, (list, tuple)):
        return {type(obj).__name__: [_canonical(v) for v in obj]}
    if isinstance(obj, np.ndarray):
        return {"ndarray": [str(obj.dtype), list(obj.shape),
                            _canonical(obj.tolist())]}
    if isinstance(obj, np.generic):
        return {type(obj).__name__: _canonical(obj.item())}
    if isinstance(obj, float):
        return {"float": obj.hex()}
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, Path):
        return {"path": str(obj)}
    if callable(obj) and hasattr(obj, "__qualname__"):
        return {"callable": f"{obj.__module__}.{obj.__qualname__}"}
    return {"repr": repr(obj)}


def _outcome(load):
    try:
        return "loaded", json.dumps(_canonical(load()), sort_keys=True)
    except Exception as exc:  # noqa: BLE001 -- compared below
        return "refused", f"{type(exc).__name__}: {exc}", getattr(
            exc, "code", None)


def _absent_keys(doc: dict):
    """``(path, key, value)`` for every key the schema allows in a block of
    ``doc`` that the block leaves out; the value is the schema default, or
    null for a key without one."""
    for where, mapping, allowed, kind in _mappings(doc):
        if kind != "key":
            continue
        obj = _object_for(_node_at(where), mapping)
        for key in sorted(allowed - set(mapping)):
            prop = obj["properties"][key]
            yield where, key, copy.deepcopy(prop.get("default"))


def _node_at(path: tuple) -> dict:
    """The schema node of the value at ``path`` in a config."""
    node = SCHEMA
    for step in path:
        alts = _alternatives(node)
        if isinstance(step, int):
            node = next(a for a in alts if a.get("type") == "array")["items"]
        else:
            node = next(a["properties"][step] for a in alts
                        if step in a.get("properties", {}))
    return node


@pytest.mark.parametrize("path", CONFIGS, ids=IDS)
def test_CO2_absent_keys_are_set_by_override_as_by_editing(path):
    doc = yaml.safe_load(path.read_text())
    cases = list(_absent_keys(doc))
    assert cases
    loaded = 0
    for where, key, value in cases:
        dotted = _dotted(where + (key,))
        by_override = _outcome(lambda: load_workflow_config(
            path, overrides={dotted: value}))
        edited = copy.deepcopy(doc)
        node = edited
        for step in where:
            node = node[step]
        node[key] = copy.deepcopy(value)
        by_edit = _outcome(lambda: cf._parse_workflow(
            edited, base_dir=path.parent))
        assert by_override == by_edit, (dotted, by_override, by_edit)
        if by_override[0] == "refused":
            # A rule tying two keys together, never the key itself.
            assert by_override[2] not in ("unknown_key", "override_path"), (
                dotted, by_override)
        else:
            loaded += 1
            # What the parser accepts, the schema accepts.
            assert VALIDATOR.is_valid(edited), dotted
    assert loaded >= len(cases) // 2, (loaded, len(cases))


def test_CO2_override_creates_a_mapping_the_file_leaves_out():
    """``system.noise.sigma2`` is absent from the quick start; an override
    below it creates the mapping, as writing ``sigma2: {amplitude: ...}``
    into the file would."""
    path = ROOT / "examples" / "quickstart.yaml"
    by_override = load_workflow_config(
        path, overrides={"system.noise.sigma2.amplitude": 0.04})
    doc = yaml.safe_load(path.read_text())
    doc["system"]["noise"]["sigma2"] = {"amplitude": 0.04}
    by_edit = cf._parse_workflow(doc, base_dir=path.parent)
    assert _canonical(by_override) == _canonical(by_edit)
    assert by_override.system.noise["sigma2"] == {"amplitude": 0.04}


def test_CO2_override_follows_a_type_set_by_an_earlier_override():
    """Overrides apply in order, so a key of another ``type`` variant is
    accepted once an earlier override has switched the type."""
    path = ROOT / "examples" / "quickstart.yaml"
    temporal = "system.noise.kappa2.temporal"
    with pytest.raises(ConfigError) as info:
        load_workflow_config(path, overrides={f"{temporal}.module": "k.py"})
    assert info.value.code == "unknown_key"
    cfg = load_workflow_config(path, overrides={
        f"{temporal}.type": "gaussian", f"{temporal}.lam": 0.07})
    assert cfg.system.noise["kappa2"]["temporal"] == {
        "type": "gaussian", "lam": 0.07, "sigma_t": 0.3}


# ---------------------------------------------------------------------------
# CO3 -- a misspelling that gives another key of the block
# ---------------------------------------------------------------------------

class _Pairs(list):
    """A mapping written as ``(key, value)`` pairs, so a YAML file can hold
    one key twice, as a misspelt file does."""


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(
    _Pairs, lambda d, data: d.represent_mapping("tag:yaml.org,2002:map", data))


def _with_pairs(doc, where: tuple, mapping: _Pairs):
    if not where:
        return mapping
    out = copy.deepcopy(doc)
    node = out
    for step in where[:-1]:
        node = node[step]
    node[where[-1]] = mapping
    return out


@pytest.mark.parametrize("path", CONFIGS, ids=IDS)
def test_CO3_a_misspelling_that_gives_another_valid_key_is_refused(path):
    """CO1 leaves out the edits that give a key the block accepts (a point
    name swapped for another, ``n_grid_t`` for ``n_grid_r``).  Each is
    refused too: as a key written twice when the block holds the other key
    already, as a spatial grid key the run would not read, or as a missing
    ``t_max``.  The file is written as YAML with the key twice, and read
    with the loader of :func:`load_workflow_config`."""
    doc = yaml.safe_load(path.read_text())
    points = _observable_points(doc)
    cases = 0
    for where, mapping, allowed, kind in list(_mappings(doc)):
        valid = allowed if kind == "key" else points
        for key in list(mapping):
            for edit in sorted(_edits(str(key))):
                if not (edit in valid or edit in mapping):
                    continue
                cases += 1
                pairs = _Pairs((edit if k == key else k, v)
                               for k, v in mapping.items())
                text = yaml.dump(_with_pairs(doc, where, pairs), Dumper=_Dumper,
                                 sort_keys=False)
                with pytest.raises((ConfigError, ValueError)):
                    data = cf._load_yaml(text)
                    cfg = cf._parse_workflow(data, base_dir=path.parent)
                    cf.check_before_run(cfg)
    assert cases > 0


# ---------------------------------------------------------------------------
# CO4 -- an override through a node the file shares
# ---------------------------------------------------------------------------

SHARED_CONFIG = string.Template("""\
${head}system:
  field: {name: phi, n_components: 2}
  linear: {type: diagonal, gamma: [1.0, 1.5]}
  vertices:
$vertices
  noise:
    kappa2:
      type: separable_translation
      temporal: {type: exponential, lam: 0.05, sigma_t: 0.3}
      spatial: {type: exponential, sigma_x: 1.0}
expand:
  observable: ["phi_a(x)", "phi_b(y)"]
  orders: [0, 2]
propagators:
  t_max: 5.0
  n_grid_t: 60
  c_closed_form: auto
sweep:
$grids
  t_final_grid: [1.0, 5.0]
  component_pairs: [[0, 0], [1, 1]]
  n_samples: 64
  seed: 42
output:
$output
""")


def _vertex(name: str, *matrices: str, anchor: str = "") -> str:
    rows = "".join(f"\n        - {m}" for m in matrices)
    return f"    - {anchor}\n      name: {name}\n      coupling:{rows}"


#: Two different matrices, so a write into one does not look like the other.
M0, M1 = "[[0.0, 0.0], [0.0, 1.0]]", "[[0.0, 0.5], [0.5, 0.0]]"
VERTEX_F = _vertex("F", M0, M1)
GRIDS = "  positions_grid: {x: [0.0], y: [0.5, 1.5]}"
OUTPUT = "  - {type: table, format: markdown, path: a.md}"


def _shared_file(directory: Path, name: str, blocks: dict) -> Path:
    path = directory / f"{name}.yaml"
    path.write_text(SHARED_CONFIG.substitute(
        {"head": "", "vertices": VERTEX_F, "grids": GRIDS, "output": OUTPUT,
         **blocks}))
    return path


#: ``case: (blocks of the file, overrides, blocks of the same file with the
#: sharing written out and the edit made by hand)``.
SHARED = {
    "alias_of_a_mapping_override_at_the_alias": (
        {"grids": "  positions_grid: &grid {x: [0.0], y: [0.5, 1.5]}\n"
                  "  external_times_grid: *grid"},
        {"sweep.external_times_grid.y": [2.0, 3.0]},
        {"grids": "  positions_grid: {x: [0.0], y: [0.5, 1.5]}\n"
                  "  external_times_grid: {x: [0.0], y: [2.0, 3.0]}"},
    ),
    "alias_of_a_mapping_override_at_the_anchor": (
        {"grids": "  positions_grid: &grid {x: [0.0], y: [0.5, 1.5]}\n"
                  "  external_times_grid: *grid"},
        {"sweep.positions_grid.y": [2.0, 3.0]},
        {"grids": "  positions_grid: {x: [0.0], y: [2.0, 3.0]}\n"
                  "  external_times_grid: {x: [0.0], y: [0.5, 1.5]}"},
    ),
    "alias_of_a_list": (
        {"grids": "  positions_grid: {x: &g [0.0, 0.5, 1.5], y: *g}"},
        {"sweep.positions_grid.y[1]": 0.7},
        {"grids": "  positions_grid: {x: [0.0, 0.5, 1.5], y: [0.0, 0.7, 1.5]}"},
    ),
    "overrides_at_both_ends_of_an_alias": (
        {"grids": "  positions_grid: {x: &g [0.0, 0.5, 1.5], y: *g}"},
        {"sweep.positions_grid.y[1]": 0.7, "sweep.positions_grid.x[0]": 0.1},
        {"grids": "  positions_grid: {x: [0.1, 0.5, 1.5], y: [0.0, 0.7, 1.5]}"},
    ),
    "merged_mapping_override_at_the_merge": (
        {"grids": "  positions_grid: &grid {x: [0.0], y: [0.5, 1.5]}\n"
                  "  external_times_grid: {<<: *grid, y: [2.0, 3.0]}"},
        {"sweep.external_times_grid.x[0]": 4.0},
        {"grids": "  positions_grid: {x: [0.0], y: [0.5, 1.5]}\n"
                  "  external_times_grid: {x: [4.0], y: [2.0, 3.0]}"},
    ),
    "merged_mapping_override_at_the_source": (
        {"grids": "  positions_grid: &grid {x: [0.0], y: [0.5, 1.5]}\n"
                  "  external_times_grid: {<<: *grid, y: [2.0, 3.0]}"},
        {"sweep.positions_grid.x[0]": 4.0},
        {"grids": "  positions_grid: {x: [4.0], y: [0.5, 1.5]}\n"
                  "  external_times_grid: {x: [0.0], y: [2.0, 3.0]}"},
    ),
    "merged_list_of_mappings": (
        {"grids": "  positions_grid: &grid {x: [0.0], y: [0.5, 1.5]}\n"
                  "  external_times_grid: {<<: [*grid], y: [2.0, 3.0]}"},
        {"sweep.external_times_grid.x[0]": 4.0},
        {"grids": "  positions_grid: {x: [0.0], y: [0.5, 1.5]}\n"
                  "  external_times_grid: {x: [4.0], y: [2.0, 3.0]}"},
    ),
    "merged_vertex_coupling": (
        {"vertices": _vertex("F", M0, M1, anchor="&F")
            + "\n    - {<<: *F, name: K}"},
        {"system.vertices[1].coupling[1][0][1]": 0.25},
        {"vertices": VERTEX_F + "\n"
            + _vertex("K", M0, "[[0.0, 0.25], [0.5, 0.0]]")},
    ),
    "list_entry_that_is_an_alias": (
        {"output": "  - &table {type: table, format: markdown, path: a.md}\n"
                   "  - *table"},
        {"output[1].path": "b.md"},
        {"output": "  - {type: table, format: markdown, path: a.md}\n"
                   "  - {type: table, format: markdown, path: b.md}"},
    ),
    "list_entry_reached_through_an_alias": (
        {"vertices": _vertex("F", f"&m {M0}", "*m")},
        {"system.vertices[0].coupling[1][1][1]": 2.0},
        {"vertices": _vertex("F", M0, "[[0.0, 0.0], [0.0, 2.0]]")},
    ),
    "chain_of_aliases": (
        {"vertices": _vertex("F", "&m [&r [0.0, 1.0], *r]", "*m")},
        {"system.vertices[0].coupling[1][0][1]": 0.5},
        {"vertices": _vertex("F", "[[0.0, 1.0], [0.0, 1.0]]",
                             "[[0.0, 0.5], [0.0, 1.0]]")},
    ),
    "chain_of_merges_and_aliases": (
        {"vertices": _vertex("F", f"&m {M0}", "*m", anchor="&F")
            + "\n    - &K {<<: *F, name: K}\n    - {<<: *K, name: G}"},
        {"system.vertices[2].coupling[1][1][1]": 2.0},
        {"vertices": _vertex("F", M0, M0) + "\n" + _vertex("K", M0, M0) + "\n"
            + _vertex("G", M0, "[[0.0, 0.0], [0.0, 2.0]]")},
    ),
}

#: The same comparison where nothing is shared on the path: a scalar has no
#: container to write into, and ``<<:`` builds a mapping of its own.  These
#: passed before the containers were copied; they pin what the copy keeps.
NOT_SHARED = {
    "alias_of_a_scalar": (
        {"grids": "  positions_grid: {x: [&p 0.0], y: [*p, 1.5]}"},
        {"sweep.positions_grid.y[0]": 0.7},
        {"grids": "  positions_grid: {x: [0.0], y: [0.7, 1.5]}"},
    ),
    "key_of_a_merged_mapping": (
        {"output": "  - &table {type: table, format: markdown, path: a.md}\n"
                   "  - {<<: *table, path: b.md}"},
        {"output[1].format": "csv"},
        {"output": "  - {type: table, format: markdown, path: a.md}\n"
                   "  - {type: table, format: csv, path: b.md}"},
    ),
}


def _check_override_against_file_edit(directory: Path, case: tuple) -> None:
    blocks, overrides, written_out = case
    shared = _shared_file(directory, "shared", blocks)
    edited = _shared_file(directory, "edited", written_out)
    by_override = _outcome(lambda: load_workflow_config(
        shared, overrides=overrides))
    by_edit = _outcome(lambda: load_workflow_config(edited))
    assert by_edit[0] == "loaded", by_edit
    assert by_override == by_edit


@pytest.mark.parametrize("case", SHARED)
def test_CO4_an_override_changes_only_the_path_it_names(tmp_path, case):
    _check_override_against_file_edit(tmp_path, SHARED[case])


@pytest.mark.parametrize("case", NOT_SHARED)
def test_CO4_an_override_where_nothing_is_shared(tmp_path, case):
    _check_override_against_file_edit(tmp_path, NOT_SHARED[case])


def _containers(node, seen: dict | None = None) -> dict:
    """The mappings and lists reachable from ``node``, each counted once
    however many paths reach it."""
    seen = {} if seen is None else seen
    if isinstance(node, (dict, list)) and id(node) not in seen:
        seen[id(node)] = node
        for child in node.values() if isinstance(node, dict) else node:
            _containers(child, seen)
    return seen


def test_CO4_an_override_copies_the_path_and_not_the_document():
    """Each grid lists the one before it twice, so the last expands to
    2**16 pairs from 17 lines of YAML.  An override at the far end adds at
    most one container per step of its path, and every branch off the path
    stays the object the file built."""
    depth = 16
    lines = ["g0: &g0 [0.0, 1.0]"]
    lines += [f"g{i}: &g{i} [*g{i - 1}, *g{i - 1}]"
              for i in range(1, depth + 1)]
    doc = cf._load_yaml("sweep:\n  positions_grid:\n"
                        + "".join(f"    {line}\n" for line in lines))
    before = len(_containers(doc))
    off_path = doc["sweep"]["positions_grid"][f"g{depth - 1}"]
    steps = "[0]" * (depth + 1)
    cf._apply_override(doc, f"sweep.positions_grid.g{depth}{steps}", 9.0)

    grids = doc["sweep"]["positions_grid"]
    assert len(_containers(doc)) <= before + depth + 3
    assert grids[f"g{depth}"][1] is off_path
    assert grids[f"g{depth - 1}"] is off_path
    written, kept = grids[f"g{depth}"], off_path
    for _ in range(depth):
        written = written[0]
    for _ in range(depth - 1):
        kept = kept[0]
    assert written == [9.0, 1.0]
    assert kept == [0.0, 1.0]


def test_CO4_a_value_given_to_two_paths_stays_as_given(tmp_path):
    """A Python override value can sit at two paths, as a node of the file
    can; an override below one leaves the other path and the caller's list
    alone."""
    grid = [0.0, 0.5, 1.5]
    cfg = load_workflow_config(_shared_file(tmp_path, "plain", {}), overrides={
        "sweep.positions_grid.x": grid,
        "sweep.positions_grid.y": grid,
        "sweep.positions_grid.y[1]": 0.7,
    })
    assert cfg.sweep.positions_grid == {
        "x": [0.0, 0.5, 1.5], "y": [0.0, 0.7, 1.5]}
    assert grid == [0.0, 0.5, 1.5]


# ---------------------------------------------------------------------------
# CO5 -- a file whose aliases form a cycle
# ---------------------------------------------------------------------------

#: ``case: (blocks of the file, dotted path of the alias closing the cycle)``.
CYCLES = {
    "list_holds_itself": (
        {"grids": "  positions_grid: {x: &g [0.0, *g], y: [0.5]}"},
        "sweep.positions_grid.x[1]",
    ),
    "grid_holds_the_document": (
        {"head": "&r\n",
         "grids": "  positions_grid: {x: [0.0, *r], y: [0.5]}"},
        "sweep.positions_grid.x[1]",
    ),
    "first_of_two_aliases_in_the_file_is_named": (
        {"grids": "  positions_grid: {x: &g [*g, *g], y: [0.5]}"},
        "sweep.positions_grid.x[0]",
    ),
    "two_lists_hold_each_other": (
        {"grids": "  positions_grid: {x: &a [0.0, &b [1.0, *a]], y: [0.5]}"},
        "sweep.positions_grid.x[1][1]",
    ),
    "mapping_holds_itself": (
        {"grids": "  positions_grid: {x: [0.0, &m {k: *m}], y: [0.5]}"},
        "sweep.positions_grid.x[1].k",
    ),
    "merge_closes_the_cycle": (
        {"grids": "  positions_grid: {x: [0.0, &a {k: &b {<<: *a}}], y: [0.5]}"},
        "sweep.positions_grid.x[1].k.k",
    ),
    "ordered_mapping_holds_itself": (
        {"grids": "  positions_grid: {x: [0.0, &o !!omap [k: *o]], y: [0.5]}"},
        "sweep.positions_grid.x[1][0][1]",
    ),
    "cycle_in_an_entry_of_a_list_of_blocks": (
        {"output": "  - {type: table, format: markdown, path: a.md}\n"
                   "  - &p {type: plot, x: t, hue: [*p]}"},
        "output[1].hue[0]",
    ),
}


@pytest.mark.parametrize("case", CYCLES)
def test_CO5_a_value_that_holds_itself_is_refused(tmp_path, case):
    blocks, where = CYCLES[case]
    with pytest.raises(ConfigError) as info:
        load_workflow_config(_shared_file(tmp_path, "cycle", blocks))
    assert (info.value.code, info.value.path) == ("invalid_value", where)
    assert "\n" not in str(info.value)


def test_CO5_an_override_that_opens_the_cycle_is_a_file_edit(tmp_path):
    """The file is checked after the overrides, so an override that replaces
    the alias gives the result of replacing it in the file."""
    _check_override_against_file_edit(tmp_path, (
        {"grids": "  positions_grid: {x: &g [0.0, *g], y: [0.5]}"},
        {"sweep.positions_grid.x[1]": 0.7},
        {"grids": "  positions_grid: {x: [0.0, 0.7], y: [0.5]}"},
    ))


def test_CO5_an_override_value_that_holds_itself_is_refused(tmp_path):
    grid = [0.0]
    grid.append(grid)
    with pytest.raises(ConfigError) as info:
        load_workflow_config(_shared_file(tmp_path, "plain", {}),
                             overrides={"sweep.positions_grid.x": grid})
    assert info.value.path == "sweep.positions_grid.x[1]"


def test_CO5_a_node_shared_without_a_cycle_is_no_cycle(tmp_path):
    """Each list holds the one before it twice, so the last expands to
    2**24 pairs from one line of YAML.  A check that entered a list once per
    path to it would take seconds here and never end at twice the depth.
    The grid is refused for what it is, a position that is no vector of
    numbers."""
    depth = 24
    levels = ", ".join(f"&g{i} [*g{i - 1}, *g{i - 1}]"
                       for i in range(1, depth + 1))
    grids = f"  positions_grid: {{x: [&g0 [0.0, 1.0], {levels}], y: [0.5]}}"
    path = _shared_file(tmp_path, "shared", {"grids": grids})
    start = time.perf_counter()
    with pytest.raises(ConfigError) as info:
        load_workflow_config(path)
    assert time.perf_counter() - start < 2.0
    assert info.value.path == "sweep.positions_grid.x[1][0]"
    assert "must be a number" in info.value.detail


def test_CO5_a_chain_past_the_recursion_limit_is_walked():
    """A chain of aliases can be as long as the file, and nests no deeper
    in the YAML than a list of its links."""
    links = 3 * sys.getrecursionlimit()

    def chained(end: list) -> dict:
        for _ in range(links):
            end = [end]
        return {"a": end}

    cf._refuse_cycles(chained([0.0]))
    loop: list = []
    loop.append(loop)
    with pytest.raises(ConfigError) as info:
        cf._refuse_cycles(chained(loop))
    assert info.value.path == "a" + "[0]" * (links + 1)


def test_CO5_a_long_key_is_cut_in_the_path():
    """A path repeats a key once per level, so a key written once and used
    through an alias at each of 400 levels would give a message of 8 MB."""
    key = "k" * 20000
    top: dict = {}
    node = top
    for _ in range(400):
        node[key] = {}
        node = node[key]
    node[key] = top
    with pytest.raises(ConfigError) as info:
        cf._refuse_cycles(top)
    assert len(str(info.value)) < 401 * 50


# ---------------------------------------------------------------------------
# CO6 -- a value that expands, in an error message
# ---------------------------------------------------------------------------

class _WrittenOut(BaseException):
    """Not an ``Exception``, so no handler in the parser can swallow it."""


class _RefusesText(list):
    """A list that raises when written out.  A ``repr`` or ``str`` of a
    shared list is the expansion in question; this finds a site that makes
    one and drops the text, which the length of a message cannot show."""

    def __repr__(self) -> str:
        raise _WrittenOut


def _expanding(kind: type) -> list:
    """Each level holds the one before it twice: 2**18 pairs.  At 64 levels
    a parser check that walks every path would not end, and ``repr`` and
    NumPy run in C, where no timeout reaches them."""
    return functools.reduce(lambda inner, _: kind([inner, inner]), range(18),
                            kind([0.0, 1.0]))


EXPANDING = _expanding(_RefusesText)
#: Each shape reaches a different check: the value itself, a mapping, an
#: entry of a list, an entry below one, and a mapping in a list.
EXPANDING_SHAPES = (EXPANDING, {"k": EXPANDING}, [EXPANDING],
                    [0.0, [EXPANDING]], [{"k": EXPANDING}])
#: A message can show a few values and a few names.
MAX_MESSAGE_CHARS = 6 * MAX_SHOWN_CHARS
#: What ``sft-wick`` prints as one line; anything else is a traceback.
CONFIG_ERRORS = (ValueError, TypeError, ArithmeticError, OSError, EOFError,
                 ImportError, yaml.YAMLError)


def _with_expanding_values(doc: dict):
    """``(dotted key, config)`` with each shape at each key a block allows."""
    for where, mapping, allowed, kind in list(_mappings(doc)):
        for key in sorted(allowed) if kind == "key" else list(mapping):
            for value in EXPANDING_SHAPES:
                out = copy.deepcopy(doc)
                node = out
                for step in where:
                    node = node[step]
                node[key] = value
                yield _dotted(where + (key,)), out


def _check_messages(doc: dict, base_dir: Path) -> None:
    refused = 0
    for dotted, config in _with_expanding_values(doc):
        try:
            cf.check_before_run(cf._parse_workflow(config, base_dir=base_dir))
        except CONFIG_ERRORS as exc:
            refused += 1
            assert len(str(exc)) < MAX_MESSAGE_CHARS, (
                dotted, type(exc).__name__, str(exc)[:200])
    assert refused > 50


@pytest.mark.parametrize("path", CONFIGS, ids=IDS)
def test_CO6_a_value_that_expands_gives_a_short_message(path):
    _check_messages(yaml.safe_load(path.read_text()), path.parent)


#: ``case: (path of a block in the quick start, the block)``: blocks no
#: shipped config uses, so the keys they hold are in the corpus too.
UNSHIPPED_BLOCKS = {
    "multiplicative_noise": (("system", "noise", "sigma2"), {
        "type": "multiplicative", "g0": [[1.0, 0.0], [0.0, 1.0]],
        "g1": [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]}),
    "coupling_path": (("system", "vertices", 0),
                      {"name": "F", "coupling_path": "coupling.npy"}),
    "plot_and_npz": (("output",), [
        {"type": "plot", "path": "p.png", "x": "t_final"},
        {"type": "npz", "path": "r.npz"}]),
    "rotation_kernel": (("system", "noise", "kappa2"), {
        "type": "separable_rotation",
        "temporal": {"type": "exponential", "lam": 0.05, "sigma_t": 0.3},
        "angular": {"type": "legendre", "coeffs": [1.0]}}),
}


@pytest.mark.parametrize("case", UNSHIPPED_BLOCKS)
def test_CO6_a_value_that_expands_in_a_block_no_config_ships(tmp_path, case):
    np.save(tmp_path / "coupling.npy", np.zeros((2, 2, 2)))
    where, block = UNSHIPPED_BLOCKS[case]
    doc = yaml.safe_load(FULL_BY_DEFAULT.read_text())
    node = doc
    for step in where[:-1]:
        node = node[step]
    node[where[-1]] = block
    cf.check_before_run(cf._parse_workflow(doc, base_dir=tmp_path))
    _check_messages(doc, tmp_path)


@pytest.mark.parametrize("value", [
    None, True, 1.5, "phi_a(x)", [], {}, (), [1.0, "a", None], (1, 2),
    {"b": 1, "a": [2.0]}, [[0, 1], [1, 0]],
], ids=repr)
def test_CO6_a_short_value_prints_as_repr(value):
    assert shown(value) == repr(value)


@pytest.mark.parametrize("value", [
    None, 1, 1.5, True, "module.py", Path("a/b.py"), [1, 2], {"k": 1}, (1,),
], ids=repr)
def test_CO6_text_of_a_short_value_is_str(value):
    assert text_of(value) == str(value)


def test_CO6_text_of_a_value_that_expands_is_a_stand_in():
    assert text_of(EXPANDING) == "<_RefusesText of length 2>"


def test_CO6_the_limit_of_a_value_shown():
    """Both sides of the limit, for a string and for a list holding one."""
    at_limit = "x" * (MAX_SHOWN_CHARS - 2)  # and its two quotes
    assert shown(at_limit) == repr(at_limit)
    assert shown(at_limit + "x") == f"<str of length {MAX_SHOWN_CHARS - 1}>"
    # A list costs its brackets and two characters per entry.
    fits = ["x" * (MAX_SHOWN_CHARS - 6)]
    assert shown(fits) == repr(fits)
    assert shown(["x" * (MAX_SHOWN_CHARS - 5)]) == "<list of length 1>"
    # A mapping costs its braces and four characters per entry.
    fits = {"x" * (MAX_SHOWN_CHARS - 9): 1}
    assert shown(fits) == repr(fits)
    assert shown({"x" * (MAX_SHOWN_CHARS - 8): 1}) == "<dict of length 1>"


def test_CO6_a_long_value_gives_a_stand_in_at_a_bounded_cost():
    assert shown(EXPANDING) == "<_RefusesText of length 2>"
    assert shown(_expanding(list)) == "<list of length 2>"
    assert shown({"k": _expanding(list)}) == "<dict of length 1>"
    assert shown((_expanding(list),)) == "<tuple of length 1>"
    assert shown(list(range(10**6))) == "<list of length 1000000>"
    cycle: list = [0.0]
    cycle.append(cycle)
    assert shown(cycle) == "<list of length 2>"


def test_CO6_the_parser_shows_values_through_shown():
    """A new ``{value!r}`` in these modules would bring the expansion back,
    and only a file that reaches that line would show it."""
    for module in ("config.py", "schema.py", "expansion.py", "specs.py",
                   "system.py", "propagators.py"):
        tree = ast.parse((ROOT / "src" / "sft_wick" / "workflow" / module)
                         .read_text())
        lines = [node.lineno for node in ast.walk(tree)
                 if isinstance(node, ast.FormattedValue)
                 and node.conversion == ord("r")]
        assert not lines, (module, lines)
