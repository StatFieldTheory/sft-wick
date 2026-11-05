"""The JSON Schema of the workflow YAML, against the config dataclasses and
the parser.

SC1 -- the document is a valid JSON Schema 2020-12.
SC2 -- the keys, defaults and required keys of each block with a config
       dataclass are that dataclass's fields.
SC3 -- every block and every ``type`` variant the schema describes holds
       exactly the keys the parser accepts there.  Both directions are
       checked on the parser itself: a block with every schema key passes
       the parser's key check, and one more key is refused as unknown.
SC4 -- each value enumeration is the set the parser accepts.
SC5 -- every shipped workflow config validates.

The reference for SC2 is the dataclass field sets; SC3 and SC4 run the
parser, so neither direction is read off the table the schema was built
from.  The schema is checked with ``jsonschema``, which shares no code
with the package.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import yaml

from sft_wick.workflow import config as cf
from sft_wick.workflow import specs as sp
from sft_wick.workflow.config import ConfigError, build_system, check_before_run
from sft_wick.workflow.schema import workflow_json_schema

ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = ROOT / "examples" / "quickstart.yaml"
BOGUS = "zz_not_a_key"


def _shipped_configs() -> list[Path]:
    paths = sorted(p for p in (ROOT / "examples").rglob("*.yaml")
                   if "sweep" in (yaml.safe_load(p.read_text()) or {}))
    paths.append(ROOT / "src" / "sft_wick" / "workflow" / "quickstart.yaml")
    return paths


# ---------------------------------------------------------------------------
# Walking the schema document (written here, not taken from the package)
# ---------------------------------------------------------------------------

def _resolve(node: dict, root: dict) -> dict:
    while "$ref" in node:
        name = node["$ref"].removeprefix("#/$defs/")
        node = root["$defs"][name]
    return node


def _alternatives(node: dict, root: dict) -> list[dict]:
    """The schemas a value at ``node`` may match, ``anyOf`` / ``oneOf``
    unfolded."""
    node = _resolve(node, root)
    for key in ("anyOf", "oneOf"):
        if key in node:
            return [a for alt in node[key] for a in _alternatives(alt, root)]
    return [node]


def _blocks(node: dict, root: dict, location: str = ""):
    """Yield ``(location, keys)`` for every fixed-key object the schema
    describes.  A ``type`` variant is written ``name<tag>``, an array
    entry ``name[]``.  A ``$ref`` names a value schema (the recursive
    tensor), which holds no block."""
    if "$ref" in node:
        return
    for alt in _alternatives(node, root):
        if alt.get("type") == "array" and "items" in alt:
            yield from _blocks(alt["items"], root, location + "[]")
        if alt.get("type") != "object" or "properties" not in alt:
            continue
        tag = alt["properties"].get("type", {}).get("const")
        here = f"{location}<{tag}>" if tag else location
        yield here, set(alt["properties"])
        for key, sub in alt["properties"].items():
            yield from _blocks(sub, root, f"{here}.{key}" if here else key)


@pytest.fixture(scope="module")
def schema() -> dict:
    return workflow_json_schema()


@pytest.fixture(scope="module")
def blocks(schema) -> dict:
    found = {}
    for location, keys in _blocks(schema, schema):
        assert location not in found, location
        found[location] = keys
    return found


# ---------------------------------------------------------------------------
# SC1 / SC5
# ---------------------------------------------------------------------------

def test_SC1_the_schema_is_a_valid_2020_12_document(schema):
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    jsonschema.Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("path", _shipped_configs(),
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_SC5_every_shipped_config_validates(schema, path):
    doc = yaml.safe_load(path.read_text())
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(doc))
    assert not errors, [e.message for e in errors]


# ---------------------------------------------------------------------------
# SC2 -- the config dataclasses are the source
# ---------------------------------------------------------------------------

def _names(cls) -> set:
    return {f.name for f in dataclasses.fields(cls)}


def _vertex_names(cls) -> set:
    return _names(cls) | {"coupling_path", "coupling_module", "coupling_attr"}


def test_SC2_block_keys_are_the_dataclass_fields(blocks):
    system = (_names(cf.SystemConfig) - {"field_name", "n_components",
                                         "base_dir"}) | {"field"}
    assert blocks[""] == _names(cf.WorkflowConfig)
    assert blocks["system"] == system
    assert blocks["system.field"] == _names(sp.FieldSpec)
    assert blocks["expand"] == _names(cf.ExpandConfig)
    assert blocks["propagators"] == _names(cf.PropagatorsConfig)
    assert blocks["sweep"] == _names(cf.SweepConfig) | {"component_tuples"}
    assert blocks["output[]"] == _names(cf.OutputConfig)
    assert blocks["system.vertices[]"] == _vertex_names(sp.LocalVertex)
    assert blocks["system.nonlocal_vertices[]"] == _vertex_names(
        sp.NonLocalVertex)


def _json_default(value):
    if isinstance(value, tuple):
        return [_json_default(v) for v in value]
    return value


@pytest.mark.parametrize("block,cls", [
    ("expand", cf.ExpandConfig),
    ("propagators", cf.PropagatorsConfig),
    ("sweep", cf.SweepConfig),
    ("output", cf.OutputConfig),
    ("field", sp.FieldSpec),
])
def test_SC2_defaults_and_required_keys_are_the_dataclass_ones(schema, block,
                                                               cls):
    props = schema["properties"]
    node = (props["system"]["properties"]["field"] if block == "field"
            else props[block])
    node = next(a for a in _alternatives(node, schema)
                if a.get("type") in ("object", "array"))
    if block == "output":
        node = _resolve(node["items"], schema)
    required = set(node.get("required", ()))
    for f in dataclasses.fields(cls):
        prop = node["properties"][f.name]
        if f.default is not dataclasses.MISSING:
            assert prop["default"] == _json_default(f.default), f.name
            assert f.name not in required, f.name
        elif f.default_factory is not dataclasses.MISSING:
            assert prop["default"] == f.default_factory(), f.name
        elif f.name == "component_pairs":
            # Required as one of two spellings, which JSON Schema cannot
            # tie to one key; the parser asks for either.
            assert f.name not in required
        else:
            assert f.name in required, f.name


# ---------------------------------------------------------------------------
# SC3 -- the schema's keys, block by block, against the parser
# ---------------------------------------------------------------------------

_LEGENDRE = {"type": "legendre", "coeffs": [1.0, 0.25]}


def _base() -> dict:
    return yaml.safe_load(QUICKSTART.read_text())


def _rotation(doc: dict) -> dict:
    kappa2 = doc["system"]["noise"]["kappa2"]
    doc["system"]["noise"]["kappa2"] = {
        "type": "separable_rotation", "temporal": kappa2["temporal"],
        "angular": dict(_LEGENDRE)}
    return doc


#: Every block the schema describes: where it sits in a config, and the
#: change to the quick-start config that puts a block of that kind there.
#: SC3 fails when the schema gains a block this table does not list.
_LOCATIONS = {
    "": ((), None),
    "system": (("system",), None),
    "system.field": (("system", "field"), None),
    "system.linear<diagonal>": (("system", "linear"), None),
    "system.linear<explicit>": (("system", "linear"), None),
    "system.noise": (("system", "noise"), None),
    "system.noise.kappa2<separable_translation>":
        (("system", "noise", "kappa2"), None),
    "system.noise.kappa2<separable_rotation>":
        (("system", "noise", "kappa2"), _rotation),
    "system.noise.kappa2<callable_module>":
        (("system", "noise", "kappa2"), None),
    "system.noise.kappa2<separable_translation>.temporal<exponential>":
        (("system", "noise", "kappa2", "temporal"), None),
    "system.noise.kappa2<separable_translation>.temporal<gaussian>":
        (("system", "noise", "kappa2", "temporal"), None),
    "system.noise.kappa2<separable_translation>.temporal<custom>":
        (("system", "noise", "kappa2", "temporal"), None),
    "system.noise.kappa2<separable_translation>.spatial<exponential>":
        (("system", "noise", "kappa2", "spatial"), None),
    "system.noise.kappa2<separable_translation>.spatial<gaussian>":
        (("system", "noise", "kappa2", "spatial"), None),
    "system.noise.kappa2<separable_translation>.spatial<custom>":
        (("system", "noise", "kappa2", "spatial"), None),
    "system.noise.kappa2<separable_rotation>.temporal<exponential>":
        (("system", "noise", "kappa2", "temporal"), _rotation),
    "system.noise.kappa2<separable_rotation>.temporal<gaussian>":
        (("system", "noise", "kappa2", "temporal"), _rotation),
    "system.noise.kappa2<separable_rotation>.temporal<custom>":
        (("system", "noise", "kappa2", "temporal"), _rotation),
    "system.noise.kappa2<separable_rotation>.angular<legendre>":
        (("system", "noise", "kappa2", "angular"), _rotation),
    "system.noise.kappa2<separable_rotation>.angular<custom>":
        (("system", "noise", "kappa2", "angular"), _rotation),
    "system.noise.sigma2<constant>": (("system", "noise", "sigma2"), None),
    "system.noise.sigma2<callable_module>":
        (("system", "noise", "sigma2"), None),
    "system.noise.sigma2<multiplicative>":
        (("system", "noise", "sigma2"), None),
    "system.vertices[]": (("system", "vertices", 0), None),
    "system.nonlocal_vertices[]": (("system", "nonlocal_vertices", 0), None),
    "expand": (("expand",), None),
    "propagators": (("propagators",), None),
    "sweep": (("sweep",), None),
    "output[]": (("output", 0), None),
}


def _placeholder(prop: dict, root: dict):
    """A value of the right JSON type for a schema property."""
    if "default" in prop:
        return copy.deepcopy(prop["default"])
    if "const" in prop:
        return prop["const"]
    if "enum" in prop:
        return prop["enum"][0]
    alts = [a for a in _alternatives(prop, root) if a.get("type") != "null"]
    kind = alts[0].get("type") if alts else None
    return {"number": 0.5, "integer": 2, "boolean": True, "string": "x",
            "array": [], "object": {}}.get(kind, 0.5)


def _variant_properties(schema: dict, location: str) -> dict:
    """The ``properties`` of the block at ``location``."""
    node, parts = schema, location.split(".") if location else []
    for part in parts:
        name, _, tag = part.partition("<")
        tag = tag.rstrip(">")
        name = name.removesuffix("[]")
        props = next(a["properties"] for a in _alternatives(node, schema)
                     if "properties" in a and name in a["properties"])
        node = props[name]
        if part.split("<")[0].endswith("[]"):
            node = next(a["items"] for a in _alternatives(node, schema)
                        if a.get("type") == "array")
        if tag:
            node = next(a for a in _alternatives(node, schema)
                        if a.get("properties", {}).get("type", {})
                        .get("const") == tag)
    return next(a for a in _alternatives(node, schema)
                if "properties" in a)["properties"]


def _full_block(schema: dict, location: str, base_block) -> dict:
    """The block at ``location`` holding every key the schema allows
    there: the quick-start value where it has one, else a placeholder."""
    props = _variant_properties(schema, location)
    base_block = base_block if isinstance(base_block, dict) else {}
    tag = location.rsplit("<", 1)[-1].rstrip(">") if location.endswith(
        ">") else None
    if tag and base_block.get("type", tag) != tag:
        base_block = {}
    return {key: copy.deepcopy(base_block[key]) if key in base_block
            else _placeholder(prop, schema) for key, prop in props.items()}


def _place(doc: dict, path: tuple, block: dict) -> dict:
    node = doc
    for step in path[:-1]:
        node = node[step]
    last = path[-1] if path else None
    if last is None:
        return block
    if isinstance(last, int):
        if not node:
            node.append(block)
        else:
            node[last] = block
    else:
        node[last] = block
    return doc


def _get(doc: dict, path: tuple):
    node = doc
    for step in path:
        try:
            node = node[step]
        except (KeyError, IndexError, TypeError):
            return None
    return node


def _parse(doc: dict):
    """Parse a config dict and build its System, as ``sft-wick validate``
    does; return the error raised, if any."""
    try:
        cfg = cf._parse_workflow(doc, base_dir=QUICKSTART.parent)
        build_system(cfg.system)
    except Exception as exc:  # noqa: BLE001 -- the test sorts them below
        return exc
    return None


def test_SC3_the_table_lists_every_block_of_the_schema(blocks):
    assert set(blocks) == set(_LOCATIONS)


@pytest.mark.parametrize("location", sorted(_LOCATIONS))
def test_SC3_schema_keys_are_the_parser_keys(schema, location):
    path, prepare = _LOCATIONS[location]
    doc = _base()
    doc["system"].setdefault("nonlocal_vertices", [])
    if prepare is not None:
        doc = prepare(doc)
    block = _full_block(schema, location, _get(doc, path))

    # Every schema key passes the parser's key check.
    err = _parse(_place(copy.deepcopy(doc), path, dict(block)))
    assert not (isinstance(err, ConfigError) and err.code == "unknown_key"), (
        f"{location}: the parser refuses a key the schema allows: {err}")

    # One more key is refused, and it is the only one refused, so the key
    # check above ran with the full set of schema keys in place.
    err = _parse(_place(copy.deepcopy(doc), path, {**block, BOGUS: 1}))
    dotted = ".".join(
        f"[{s}]" if isinstance(s, int) else s for s in path + (BOGUS,)
    ).replace(".[", "[")
    assert isinstance(err, ConfigError), (location, err)
    assert err.code == "unknown_key", (location, err)
    assert err.path == dotted, (location, err.path)
    assert "also unknown" not in str(err), (location, str(err))


# ---------------------------------------------------------------------------
# SC4 -- value enumerations
# ---------------------------------------------------------------------------

def _enum(schema: dict, location: str, key: str) -> list:
    prop = _variant_properties(schema, location)[key]
    values = [v for alt in _alternatives(prop, schema) for v in alt.get(
        "enum", [alt["const"]] if "const" in alt else [])]
    assert values, (location, key)
    return values


@pytest.mark.parametrize("location,key,near_miss", [
    ("expand", "target", "momnet"),
    ("propagators", "c_method", "gauss_legendr"),
    ("sweep", "method", "gauss_legendr"),
    ("sweep", "contraction_backend", "einsm"),
    ("sweep", "propagator_cache_contract", "fixed"),
    ("output[]", "type", "tabel"),
    ("output[]", "format", "markdwon"),
    ("propagators", "homogeneity", "rotatoin"),
])
def test_SC4_enumerations_are_what_the_parser_accepts(schema, location, key,
                                                      near_miss):
    """Each listed value loads, and a value one edit away does not."""
    path = tuple(p.removesuffix("[]") for p in location.split("."))
    path = ("output", 0) if location == "output[]" else path
    for value in _enum(schema, location, key) + [near_miss]:
        doc = _base()
        if location == "output[]":
            # npz and plot outputs require a path, and a plot an x.
            doc["output"][0].update(path="o.out", x="y")
        _get(doc, path)[key] = value
        try:
            cf._parse_workflow(doc, base_dir=QUICKSTART.parent)
            loaded = True
        except (ValueError, TypeError):
            loaded = False
        assert loaded is (value != near_miss), (key, value)


def test_SC4_table_formats_are_what_the_writer_accepts(schema):
    import pandas as pd

    frame = pd.DataFrame({"a": [1]})
    for fmt in _enum(schema, "output[]", "format"):
        assert cf._format_table(frame, fmt)
    with pytest.raises(ValueError):
        cf._format_table(frame, "markdwn")


@pytest.mark.parametrize("path,value", [
    (("system", "vertices"), None),
    (("system", "nonlocal_vertices"), None),
    (("system", "field"), None),
    (("output",), None),
])
def test_SC5_null_blocks_the_parser_accepts_validate(schema, path, value):
    """The parser reads a null list or field block as empty; the schema
    used to refuse it."""
    doc = _base()
    if path == ("system", "field"):
        # An empty field block means one component.
        doc["system"]["linear"]["gamma"] = [1.0]
        doc["system"]["vertices"] = None
    _get(doc, path[:-1])[path[-1]] = value
    cf._parse_workflow(copy.deepcopy(doc), base_dir=QUICKSTART.parent)
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(doc))
    assert not errors, [e.message for e in errors]


# ---------------------------------------------------------------------------
# SC6 -- keys one edit apart, and what refuses a misspelling between them
# ---------------------------------------------------------------------------

def _one_edit_apart(a: str, b: str) -> bool:
    """A deletion, an insertion, a substitution or a swap of neighbours
    turns ``a`` into ``b`` (written here, not taken from the package)."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1
                                  and a[diff[0]] == b[diff[1]]
                                  and a[diff[1]] == b[diff[0]])
    short, long_ = sorted((a, b), key=len)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


_REQUIRED = "the misspelt key is required, so it is missing"
_GRID = ("a spatial grid key the homogeneity does not read, or half of a "
         "pair, is refused (YC15)")
_SIMPLIFICATION = ("System.expand refuses a simplification whose condition "
                   "fails (YC17), and switching one off keeps the value")
#: Every pair of keys in one block that a single edit turns into each other,
#: with what refuses a misspelling of one into the other.  When both keys
#: are in the block the misspelling writes one key twice, which is refused
#: as a duplicate (YC14).
_COLLISIONS = {
    ("system.noise.sigma2<multiplicative>", "g0", "g1"): _REQUIRED,
    ("expand", "iso_R", "iso_C"): _SIMPLIFICATION,
    ("expand", "diag_R", "diag_C"): _SIMPLIFICATION,
    ("propagators", "t_max", "r_max"): _REQUIRED,
    ("propagators", "t_max", "x_max"): _REQUIRED,
    ("propagators", "n_grid_t", "n_grid_r"): _GRID,
    ("propagators", "n_grid_t", "n_grid_x"): _GRID,
    ("propagators", "r_max", "x_max"): _GRID,
    ("propagators", "n_grid_r", "n_grid_x"): _GRID,
    ("output[]", "x", "y"): "a plot requires x, checked at load (YC16)",
}


def test_SC6_every_pair_of_keys_one_edit_apart_is_listed(blocks):
    """A key added next to one a single edit away needs an entry here and
    a check that refuses the misspelling."""
    found = set()
    for location, keys in blocks.items():
        keys = sorted(keys)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if _one_edit_apart(a, b):
                    found.add((location, a, b))
    listed = {(loc, *sorted((a, b))) for loc, a, b in _COLLISIONS}
    assert found == listed


@pytest.mark.parametrize("block,key,value", [
    ("sweep", "t_final_grid", ["1e-3"]),
    ("propagators", "dt", "1e-1"),
    ("propagators", "t_max", "5"),
    ("expand", "diag_R", "no"),
    ("expand", "diag_R", 1),
    ("sweep", "n_samples", "1e3"),
    ("propagators", "c_closed_form", "Auto"),
])
def test_schema_takes_what_the_parser_reads(tmp_path, block, key, value):
    doc = yaml.safe_load(QUICKSTART.read_text())
    doc[block].pop("n_grid_t", None)
    doc[block][key] = value
    doc["sweep"]["t_final_grid"] = doc["sweep"]["t_final_grid"][:1]
    cf._parse_workflow(copy.deepcopy(doc), base_dir=tmp_path)
    jsonschema.validate(doc, workflow_json_schema())


@pytest.mark.parametrize("path,value", [
    (("sweep", "gl_chunk_size"), "100"),
    (("sweep", "qmc_coupling_chunk_size"), "100"),
    (("sweep", "contraction_max_bytes"), "1e9"),
    (("sweep", "orders"), ["0"]),
    (("expand", "orders"), "0"),
    (("sweep", "component_tuples"), [["0", 0]]),
    (("sweep", "n_samples"), "4096.5"),
    (("sweep", "seed"), "0.5"),
])
def test_schema_refuses_text_the_parser_refuses(tmp_path, path, value):
    """Text is taken at the integer keys that read it, in integral form.  The
    other integer keys share their reader with L1, which takes no text."""
    doc = yaml.safe_load(QUICKSTART.read_text())
    doc[path[0]].pop("component_pairs", None)
    doc[path[0]][path[1]] = value
    with pytest.raises(ConfigError):
        check_before_run(cf._parse_workflow(copy.deepcopy(doc), base_dir=tmp_path))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, workflow_json_schema())


def test_a_coupling_reads_numeric_text_as_the_other_tensors_do(tmp_path):
    """PyYAML reads ``1e-3`` as a string, in a file and in ``--override``."""
    doc = yaml.safe_load(QUICKSTART.read_text())
    tensor = doc["system"]["vertices"][0]["coupling"]
    while isinstance(tensor[0], list):
        tensor = tensor[0]
    tensor[0] = "1e-3"
    jsonschema.validate(doc, workflow_json_schema())
    system = cf._parse_workflow(copy.deepcopy(doc), base_dir=tmp_path).system
    assert np.ravel(system.vertices[0]["coupling"])[0] == 1e-3


@pytest.mark.parametrize("value", [0, 1, "maybe"])
def test_c_closed_form_words_are_one_list(tmp_path, value):
    doc = yaml.safe_load(QUICKSTART.read_text())
    doc["propagators"]["c_closed_form"] = value
    with pytest.raises(ConfigError):
        cf._parse_workflow(copy.deepcopy(doc), base_dir=tmp_path)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, workflow_json_schema())
