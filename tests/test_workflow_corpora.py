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

Before the schema and the one-line errors, CO1 failed on every case (the
error was a ``ValueError`` listing the allowed keys, without the dotted
path of the key or a suggestion, and keys of the noise and kernel blocks
passed the parser) and CO2 raised ``KeyError`` for every key outside
``sweep``.  The parser's result is compared, not its own report: the
reference is the parsed :class:`~sft_wick.workflow.config.WorkflowConfig`,
written out field by field by :func:`_canonical` below.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import json
import string
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import yaml

from sft_wick.workflow import config as cf
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
