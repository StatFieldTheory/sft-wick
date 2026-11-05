"""JSON Schema of the workflow YAML, read from the config dataclasses.

:func:`workflow_json_schema` returns a JSON Schema (draft 2020-12) for the
files ``sft-wick run`` reads, and ``sft-wick schema`` prints it.  It gives
the keys of every block, their value types, enumerations and defaults, and
the keys a block requires; a block takes no other key.  A rule that ties
two keys together (``dt`` or ``n_grid_t``, one coupling source per vertex)
is the parser's alone, so a file can satisfy the schema and still be
refused: :func:`~sft_wick.workflow.config.load_workflow_config` decides.

Where the keys come from:

* the top level, ``system``, ``expand``, ``propagators``, ``sweep`` and
  each ``output`` entry: the fields of :class:`WorkflowConfig`,
  :class:`SystemConfig`, :class:`ExpandConfig`, :class:`PropagatorsConfig`,
  :class:`SweepConfig` and :class:`OutputConfig`, with their defaults.
  ``system.field`` stands for ``field_name`` and ``n_components``, and
  ``sweep.component_tuples`` is the current name of ``component_pairs``;
* the blocks inside ``system``, which have no config dataclass: the key
  tables the parser checks them with, and the defaults of the L1 specs
  they lower to (:class:`FieldSpec`, :class:`DiagonalA`,
  :class:`ExplicitR`, :class:`LocalVertex`, :class:`NonLocalVertex`,
  :class:`MultiplicativeImpulse`).

The same tree resolves ``--override`` paths (:func:`override_target`) and
fills the defaults that ``--print-resolved`` shows (:func:`resolved_config`).
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import types
import typing
from dataclasses import dataclass
from typing import Any

from sft_wick._connected import TARGETS
from sft_wick._contraction_admission import CONTRACTION_BACKENDS

from . import config as cf
from . import specs as sp
from ._messages import shown

SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

#: Marks a key without a default.
_NO_DEFAULT = object()


# =========================================================================
# The key tree
# =========================================================================


@dataclass(frozen=True)
class _Key:
    """One key of a block: the schema of its value (a JSON fragment or
    another node), whether the block requires it, and its default."""

    node: Any
    required: bool = False
    default: Any = _NO_DEFAULT
    nullable: bool = False
    doc: str = ""


@dataclass(frozen=True)
class _Block:
    """A mapping with a fixed set of keys."""

    keys: dict


@dataclass(frozen=True)
class _Tagged:
    """A mapping whose ``type`` key picks one of several blocks; ``default``
    is the type of a block that gives none (``None``: ``type`` required)."""

    variants: dict
    default: str | None


@dataclass(frozen=True)
class _MapOf:
    """A mapping from free names (point names) to values."""

    value: dict


@dataclass(frozen=True)
class _ListOf:
    """A list of blocks."""

    item: Any


def _any_case(words) -> str:
    """Alternatives of a regular expression matching ``words`` in any case;
    JSON Schema has no case flag."""
    return "|".join(
        "".join(f"[{c.upper()}{c.lower()}]" if c.isalpha() else c for c in word)
        for word in sorted(words)
    )


_MANTISSA = r"(\d[\d_]*(\.[\d_]*)?|\.\d[\d_]*)"
_NUMBER = {
    "anyOf": [
        {"type": "number"},
        # PyYAML reads ``1e-3``, with no decimal point, as a string, and the
        # parser's number reader converts it.
        {"type": "string", "pattern": rf"^\s*[+-]?{_MANTISSA}([eE][+-]?\d+)?\s*$"},
    ]
}
_INTEGER = {
    "anyOf": [
        {"type": "integer"},
        # What :func:`~sft_wick.workflow.config._integer` can read: nonzero
        # digits past the point need an exponent to be integral.
        {
            "type": "string",
            "pattern": (
                rf"^\s*[+-]?(\d[\d_]*(\.[0_]*)?|\.0[0_]*|{_MANTISSA}[eE][+-]?\d+)\s*$"
            ),
        },
    ]
}
#: The ``order`` of a non-local vertex.  A numeric string passes whatever its
#: value, as at every integer key.
_ORDER = {
    "anyOf": [
        {"type": "integer", "minimum": 1, "maximum": cf._MAX_VERTEX_ORDER},
        _INTEGER["anyOf"][1],
    ]
}
#: A key read by a check shared with L1, which takes no text.
_STRICT_INTEGER = {"type": "integer"}
#: What :func:`~sft_wick.workflow.config._as_bool` reads.
_BOOLEAN = {
    "anyOf": [
        {"type": "boolean"},
        {"enum": [0, 1]},
        {
            "type": "string",
            "pattern": "^\\s*(" + _any_case(cf._TRUTHY | cf._FALSY) + ")\\s*$",
        },
    ]
}
_STRING = {"type": "string"}
_TENSOR = {"$ref": "#/$defs/tensor"}
_NUMBERS = {"type": "array", "items": _NUMBER}
_STRINGS = {"type": "array", "items": _STRING}
_ORDERS = {"anyOf": [_STRICT_INTEGER, {"type": "array", "items": _STRICT_INTEGER}]}
#: A position: a number, or a vector of numbers for d-dim coordinates.
_POSITIONS = {
    "type": "array",
    "items": {"anyOf": [_NUMBER, _NUMBERS]},
}
_COMPONENT_TUPLES = {
    "type": "array",
    "items": {"type": "array", "items": _STRICT_INTEGER},
}


def _null(fragment: dict) -> dict:
    return {"anyOf": [{"type": "null"}, fragment]}


def _enum(values) -> dict:
    return {"enum": list(values)}


def _json_type(hint: Any) -> dict:
    """The JSON Schema of a dataclass field annotation; ``Any`` gives
    ``{}`` (any value)."""
    if hint is bool:
        return dict(_BOOLEAN)
    if hint is int:
        return dict(_INTEGER)
    if hint is float:
        return dict(_NUMBER)
    if hint is str:
        return dict(_STRING)
    origin = typing.get_origin(hint)
    if hint in (list, tuple) or origin in (list, tuple):
        return {"type": "array"}
    if hint is dict or origin is dict:
        return {"type": "object"}
    if origin in (types.UnionType, typing.Union):
        args = typing.get_args(hint)
        parts = [_json_type(a) for a in args if a is not type(None)]
        out = parts[0] if len(parts) == 1 else {"anyOf": parts}
        return _null(out) if type(None) in args else out
    return {}


def _dataclass_keys(cls, nodes: dict, *, skip=(), optional=()) -> dict:
    """``_Key`` per field of ``cls``: the default from the dataclass, the
    value schema from ``nodes`` or else from the annotation.  A field
    without a default is required unless listed in ``optional``."""
    hints = typing.get_type_hints(cls)
    keys = {}
    for f in dataclasses.fields(cls):
        if f.name in skip:
            continue
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:
            default = f.default_factory()
        else:
            default = _NO_DEFAULT
        node = nodes[f.name] if f.name in nodes else _json_type(hints[f.name])
        keys[f.name] = _Key(
            node,
            required=default is _NO_DEFAULT and f.name not in optional,
            default=default,
        )
    return keys


def _type_key(tag: str, default_tag: str | None) -> _Key:
    if tag == default_tag:
        return _Key({"const": tag}, default=tag)
    return _Key({"const": tag}, required=True)


def _table_block(allowed, keys: dict, tag: str | None = None, default_tag=None):
    """A block whose keys are the parser's table ``allowed``: each key
    takes its ``_Key`` from ``keys``, ``type`` its variant tag."""
    out = {}
    for name in allowed:
        out[name] = _type_key(tag, default_tag) if name == "type" else keys[name]
    return _Block(out)


def _default_of(cls, name: str):
    return next(f.default for f in dataclasses.fields(cls) if f.name == name)


# -- system ----------------------------------------------------------------


def _linear() -> _Tagged:
    diag, expl = sp.DiagonalA, sp.ExplicitR
    keys = {
        "gamma": _Key(
            {"anyOf": [_NUMBER, _NUMBERS]},
            doc="Decay rates, one per component or one for all; or give "
            "gamma_module.",
        ),
        "gamma_module": _Key(_STRING, doc="Path to a .py file exporting gamma(t)."),
        "gamma_attr": _Key(_STRING, doc="Default 'gamma' with gamma_module."),
        "dt": _Key(
            _null(_NUMBER),
            doc="Default propagators.dt; sets n_grid_cache when given.",
        ),
        "n_grid_cache": _Key(_INTEGER, default=_default_of(diag, "n_grid_cache")),
        "t_max_cache": _Key(_NUMBER, default=_default_of(diag, "t_max_cache")),
        "t_min_cache": _Key(_NUMBER, default=_default_of(diag, "t_min_cache")),
        "R_time_module": _Key(_STRING, required=True),
        # The parser removes the *_attr keys of a module it has loaded, so
        # their defaults are stated in the description: as schema defaults
        # --print-resolved would fill them in over the attribute used.
        "R_time_attr": _Key(_STRING, doc="Default 'R_time'."),
        "iso_R": _Key(_BOOLEAN, default=_default_of(expl, "iso_R")),
        "R_time_vectorized": _Key(_BOOLEAN, default=_default_of(expl, "vectorized")),
    }
    return _Tagged(
        {
            "diagonal": _table_block(
                cf._LINEAR_DIAGONAL_KEYS, keys, "diagonal", "diagonal"
            ),
            "explicit": _table_block(
                cf._LINEAR_EXPLICIT_KEYS, keys, "explicit", "diagonal"
            ),
        },
        default="diagonal",
    )


_KERNEL_PARAM_NODES = {
    "lam": _NUMBER,
    "sigma_t": _NUMBER,
    "sigma_x": _NUMBER,
    "coeffs": _NUMBERS,
}


def _kernel(axis: str) -> _Tagged:
    default = "exponential" if (axis, "exponential") in cf._KERNEL_PARAMS else None
    variants = {}
    for (a, kind), params in cf._KERNEL_PARAMS.items():
        if a == axis:
            keys = {p: _Key(_KERNEL_PARAM_NODES[p], required=True) for p in params}
            variants[kind] = _table_block(("type",) + params, keys, kind, default)
    custom = {
        "module": _Key(_STRING, required=True),
        "attr": _Key(_STRING, default=cf._KERNEL_BLOCK[axis]),
    }
    variants["custom"] = _table_block(cf._CUSTOM_KERNEL_KEYS, custom, "custom", default)
    return _Tagged(variants, default=default)


def _kappa2() -> _Tagged:
    variants = {}
    for kind, blocks in cf._SEPARABLE_KAPPA2.items():
        keys = {b: _Key(_kernel(axis), required=True) for b, axis in blocks}
        variants[kind] = _table_block(("type",) + tuple(keys), keys, kind)
    callable_keys = {
        "module": _Key(_STRING, required=True),
        "attr": _Key(_STRING, default="kappa2"),
    }
    variants["callable_module"] = _table_block(
        cf._KAPPA2_CALLABLE_KEYS, callable_keys, "callable_module"
    )
    return _Tagged(variants, default=None)


def _sigma2() -> _Tagged:
    mult = sp.MultiplicativeImpulse
    keys = {
        "amplitude": _Key(
            {"anyOf": [_NUMBER, {"type": "array", "items": _NUMBERS}]},
            required=True,
            doc="A number (amplitude * I_N) or a symmetric N x N matrix.",
        ),
        "module": _Key(_STRING, required=True),
        "attr": _Key(_STRING, default="sigma2"),
        "g0": _Key(_TENSOR, required=True),
        "g1": _Key(_TENSOR, required=True),
        "interpretation": _Key(
            _enum(("ito", "stratonovich")), default=_default_of(mult, "interpretation")
        ),
        "vertex_names": _Key(_STRINGS, default=_default_of(mult, "vertex_names")),
        "drift_names": _Key(_STRINGS, default=_default_of(mult, "drift_names")),
    }
    return _Tagged(
        {
            kind: _table_block(allowed, keys, kind, "constant")
            for kind, allowed in cf._SIGMA2_KEYS.items()
        },
        default="constant",
    )


def _vertex(cls) -> _Block:
    nodes = {
        "coupling": _TENSOR,
        "rank": _null(_STRICT_INTEGER),
        "order": _ORDER,
    }
    keys = _dataclass_keys(cls, nodes, optional=("coupling",))
    keys["coupling"] = dataclasses.replace(
        keys["coupling"],
        doc="An inline tensor; or give coupling_path or coupling_module.",
    )
    keys["coupling_path"] = _Key(_STRING, doc="Path to an .npy file.")
    keys["coupling_module"] = _Key(
        _STRING, doc="Path to a .py file exporting the coupling callable."
    )
    keys["coupling_attr"] = _Key(
        _STRING, doc="Default 'coupling_fn' with coupling_module."
    )
    return _Block(keys)


def _system() -> _Block:
    yaml_keys = {
        # A null block or list is read as empty.
        "field": _Key(_Block(_dataclass_keys(sp.FieldSpec, {})), nullable=True),
        "linear": _Key(_linear(), required=True),
        "noise": _Key(
            _table_block(
                cf._NOISE_KEYS,
                {
                    "kappa2": _Key(_kappa2(), required=True),
                    "sigma2": _Key(_sigma2(), nullable=True, default=None),
                },
            ),
            required=True,
        ),
        "vertices": _Key(_ListOf(_vertex(sp.LocalVertex)), nullable=True, default=[]),
        "nonlocal_vertices": _Key(
            _ListOf(_vertex(sp.NonLocalVertex)), nullable=True, default=[]
        ),
        "t_min": _Key(_NUMBER, default=_default_of(cf.SystemConfig, "t_min")),
    }
    # The keys follow SystemConfig: ``field`` stands for field_name and
    # n_components, and base_dir is not read from the file.
    keys = {}
    for f in dataclasses.fields(cf.SystemConfig):
        name = {"field_name": "field", "n_components": "field"}.get(f.name, f.name)
        if name != "base_dir" and name not in keys:
            keys[name] = yaml_keys[name]
    return _Block(keys)


# -- the other blocks --------------------------------------------------------


def _expand() -> _Block:
    nodes = {
        "observable": {"type": "array", "items": _STRING, "minItems": 1},
        "orders": _ORDERS,
        "iso_R": _null(_BOOLEAN),
        "cache_path": _null(_STRING),
        "target": _enum(TARGETS),
    }
    return _Block(_dataclass_keys(cf.ExpandConfig, nodes))


def _propagators() -> _Block:
    nodes = {
        "homogeneity": _null(_enum(cf.HOMOGENEITIES)),
        "r_max": _null(_NUMBER),
        "n_grid_r": _null(_INTEGER),
        "n_grid_cos": _null(_INTEGER),
        "x_max": _null(_NUMBER),
        "n_grid_x": _null(_INTEGER),
        "c_closed_form": {
            "anyOf": [
                {"type": ["boolean", "null"]},
                {
                    "type": "string",
                    "pattern": f"^({_any_case(['auto'])}|"
                    + "|".join(cf._CLOSED_FORM_OFF)
                    + ")$",
                },
            ]
        },
        "c_closed_form_module": _null(_STRING),
        "cache_path": _null(_STRING),
        "c_method": _enum(cf.C_METHODS),
        "interp_method": _enum(cf._INTERP_METHODS),
    }
    return _Block(_dataclass_keys(cf.PropagatorsConfig, nodes))


def _sweep() -> _Block:
    points = _MapOf(_POSITIONS)
    nodes = {
        "positions_grid": points,
        "t_final_grid": _NUMBERS,
        "component_pairs": _null(_COMPONENT_TUPLES),
        "orders": _null(_ORDERS),
        "vertex_types": _null({"anyOf": [_STRING, _STRINGS]}),
        "integrate_over": _null({"anyOf": [_STRING, _STRINGS]}),
        "method": _enum(cf.INTEGRATION_METHODS),
        "propagator_cache_contract": _null(_enum(("fixed_model",))),
        "contraction_backend": _enum(CONTRACTION_BACKENDS),
        "gl_chunk_size": _null(_STRICT_INTEGER),
        "qmc_coupling_chunk_size": _null(_STRICT_INTEGER),
        "propagator_cache_bytes": _null(_STRICT_INTEGER),
        "contraction_max_bytes": _STRICT_INTEGER,
    }
    keys = _dataclass_keys(cf.SweepConfig, nodes, optional=("component_pairs",))
    keys["external_times_grid"] = _Key(_MapOf(_NUMBERS), nullable=True, default=None)
    keys["component_pairs"] = dataclasses.replace(
        keys["component_pairs"], doc="The older name of component_tuples."
    )
    keys["component_tuples"] = _Key(
        _null(_COMPONENT_TUPLES),
        doc="Component indices, one per observable operator; required, or "
        "give component_pairs.",
    )
    return _Block(keys)


def _output() -> _Block:
    nodes = {
        "type": _enum(cf._OUTPUT_TYPES),
        "path": _null(_STRING),
        "format": _enum(cf._TABLE_FORMATS),
        "x": _null(_STRING),
        "hue": _null(_STRING),
        "facet_col": _null(_STRING),
    }
    return _Block(_dataclass_keys(cf.OutputConfig, nodes))


@functools.lru_cache(maxsize=1)
def _tree() -> _Block:
    blocks = {
        "system": _Key(_system(), required=True),
        "expand": _Key(_expand(), required=True),
        "propagators": _Key(_propagators(), required=True),
        "sweep": _Key(_sweep(), required=True),
        "output": _Key(_ListOf(_output()), nullable=True, default=[]),
    }
    names = [f.name for f in dataclasses.fields(cf.WorkflowConfig)]
    return _Block({name: blocks[name] for name in names})


# =========================================================================
# JSON Schema
# =========================================================================


def _json(value):
    if isinstance(value, tuple):
        return [_json(v) for v in value]
    if isinstance(value, list):
        return [_json(v) for v in value]
    return value


def _render(node) -> dict:
    if isinstance(node, dict):
        return copy.deepcopy(node)
    if isinstance(node, _Block):
        props = {name: _render_key(key) for name, key in node.keys.items()}
        out = {"type": "object", "properties": props, "additionalProperties": False}
        required = [name for name, key in node.keys.items() if key.required]
        if required:
            out["required"] = required
        return out
    if isinstance(node, _Tagged):
        return {"oneOf": [_render(block) for block in node.variants.values()]}
    if isinstance(node, _MapOf):
        return {"type": "object", "additionalProperties": _render(node.value)}
    if isinstance(node, _ListOf):
        return {"type": "array", "items": _render(node.item)}
    raise TypeError(f"not a schema node: {shown(node)}")


def _render_key(key: _Key) -> dict:
    out = _render(key.node)
    if key.nullable:
        out = _null(out)
    if key.doc:
        out["description"] = key.doc
    if key.default is not _NO_DEFAULT:
        out["default"] = _json(key.default)
    return out


def workflow_json_schema() -> dict:
    """The JSON Schema (draft 2020-12) of a workflow YAML file.

    A new dictionary on each call.  ``sft-wick schema`` prints it as JSON.
    """
    out = {
        "$schema": SCHEMA_DIALECT,
        "title": "sft-wick workflow config",
        "description": (
            "Keys, value types, enumerations, defaults and required keys of "
            "the YAML file sft-wick run reads.  Rules that tie two keys "
            "together are checked by the parser, which decides."
        ),
        "$defs": {
            "tensor": {
                "anyOf": [
                    _NUMBER,
                    {"type": "array", "items": {"$ref": "#/$defs/tensor"}},
                ]
            }
        },
    }
    out.update(_render(_tree()))
    return out


# =========================================================================
# Overrides
# =========================================================================


def _label(walked: str, step) -> str:
    if isinstance(step, int):
        return f"{walked}[{step}]"
    return f"{walked}.{step}" if walked else str(step)


def _variant(node: _Tagged, data: dict, walked: str) -> tuple:
    """The block of a ``type`` variant and its tag, read from ``data``."""
    tag = cf._tag(data, node.default)
    if tag not in node.variants:
        raise cf.OverrideError(
            "invalid_value",
            _label(walked, "type"),
            f"{shown(tag)} is not one of {cf._or_list(node.variants)}; set "
            f"{_label(walked, 'type')} first",
            cf._nearest(tag, node.variants) if isinstance(tag, str) else (),
        )
    return node.variants[tag], tag


def _child(node, data, step, walked: str):
    """The schema node of ``data[step]``, or raise an
    :class:`~sft_wick.workflow.config.OverrideError`."""
    here = _label(walked, step)
    if isinstance(step, int):
        if not isinstance(data, list):
            what = "a mapping" if isinstance(data, dict) else "a value"
            raise cf.OverrideError(
                "override_path", here, f"{walked or 'the config'} is {what}, not a list"
            )
        if not 0 <= step < len(data):
            raise cf.OverrideError(
                "override_path",
                here,
                f"{walked} has {len(data)} entr{'y' if len(data) == 1 else 'ies'}",
            )
        return node.item if isinstance(node, _ListOf) else {}
    if not isinstance(data, dict) or isinstance(node, (dict, _ListOf)):
        raise cf.OverrideError(
            "override_path", here, f"{walked} is a value, not a mapping of keys"
        )
    if isinstance(node, _MapOf):
        return node.value
    note = ""
    if isinstance(node, _Tagged):
        if step == "type":
            return _enum(node.variants)
        node, tag = _variant(node, data, walked)
        note = f" for type {shown(tag)}"
    key = node.keys.get(step)
    if key is None:
        raise cf.OverrideError(
            "unknown_key", here, f"unknown key{note}", cf._nearest(step, node.keys)
        )
    return key.node


def override_target(data: dict, path: list) -> tuple:
    """The container and the last step an override at ``path`` writes to.

    ``path`` is the parsed override key (``["output", 0, "path"]``).  The
    path is read against the schema tree and ``data`` together: a key the
    schema allows in a block is accepted whether or not ``data`` has it,
    and a mapping on the way that ``data`` leaves out (or sets to
    ``null``) is created.  A list entry is never created.

    Each mapping and list on the way is replaced in ``data`` by a shallow
    copy, so a write to the returned container changes this path alone: a
    node ``data`` holds at several paths, through a YAML alias or a ``<<:``
    merge, keeps its value at the others.  Everything off the path stays
    shared, so the cost does not grow with what the aliases expand to.
    """
    node, cur, walked = _tree(), data, ""
    for i, step in enumerate(path):
        child = _child(node, cur, step, walked)
        if i == len(path) - 1:
            return cur, step
        here = _label(walked, step)
        nxt = cur[step] if isinstance(step, int) else cur.get(step)
        if nxt is None:
            if not isinstance(child, (_Block, _Tagged, _MapOf)):
                raise cf.OverrideError(
                    "override_path",
                    here,
                    "is not set in the file, and an override cannot create "
                    "a list or a value below it",
                )
            nxt = cur[step] = {}
        elif isinstance(nxt, (dict, list)):
            nxt = cur[step] = copy.copy(nxt)
        cur, node, walked = nxt, child, here
    raise cf.OverrideError("override_syntax", "", "an override needs a key")


# =========================================================================
# The resolved config
# =========================================================================


#: Keys the parser puts in place of a module path it has loaded; the
#: schema names the path key instead.
_LOADED_KEYS = frozenset({"R_time"})


def _with_defaults(node, data):
    """``data`` with every key that has a default filled in, recursively.
    A key the schema does not allow, which ``SFT_WICK_STRICT=0`` let
    through without reading it, is left out; a loaded callable in place of
    its module path (``R_time``) is kept."""
    if isinstance(node, _ListOf) and isinstance(data, list):
        return [_with_defaults(node.item, v) for v in data]
    if not isinstance(data, dict) or not isinstance(node, (_Block, _Tagged)):
        return data
    if isinstance(node, _Tagged):
        tag = cf._tag(data, node.default)
        if tag not in node.variants:
            return dict(data)
        node = node.variants[tag]
    out = {}
    for name, value in data.items():
        key = node.keys.get(name)
        if key is not None:
            out[name] = _with_defaults(key.node, value)
        elif name in _LOADED_KEYS:
            out[name] = value
    for name, key in node.keys.items():
        if name not in out and key.default is not _NO_DEFAULT:
            out[name] = _with_defaults(key.node, copy.deepcopy(key.default))
    return out


def _plain(value):
    """``value`` as YAML-safe Python: arrays as nested lists, a loaded
    callable as ``"<callable module.name>"``."""
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if callable(value):
        name = getattr(value, "__qualname__", None)
        module = getattr(value, "__module__", None)
        label = f"{module}.{name}" if name and module else repr(value)
        return f"<callable {label}>"
    return str(value)


def _field_values(obj) -> dict:
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}


def resolved_config(cfg) -> dict:
    """``cfg`` laid out as a YAML file, with the value the run uses for
    every key: defaults filled in, derived values shown (``n_grid_t`` from
    ``propagators.dt``, ``system.linear.n_grid_cache`` from ``dt``,
    ``expand.diag_C`` false under ``propagators.diag_C: false``), module
    paths replaced by the callables loaded from them.

    It records a run; it is not meant to be loaded back (``dt`` and the
    ``n_grid_t`` derived from it are both present, and a loaded callable
    has no YAML spelling).
    """
    tree = _tree().keys
    system_keys = tree["system"].node.keys
    s = cfg.system
    system = {
        "field": {"name": s.field_name, "n_components": s.n_components},
        "t_min": s.t_min,
        "linear": _with_defaults(system_keys["linear"].node, dict(s.linear)),
        "noise": _with_defaults(system_keys["noise"].node, s.noise),
        "vertices": _with_defaults(system_keys["vertices"].node, list(s.vertices)),
        "nonlocal_vertices": _with_defaults(
            system_keys["nonlocal_vertices"].node, list(s.nonlocal_vertices)
        ),
    }
    sweep = {}
    for name, value in _field_values(cfg.sweep).items():
        sweep["component_tuples" if name == "component_pairs" else name] = value
    expand = _field_values(cfg.expand)
    expand["diag_C"] = cf.effective_expand_diag_C(cfg)
    doc = {
        "system": system,
        "expand": expand,
        "propagators": _field_values(cfg.propagators),
        "sweep": sweep,
        "output": [_field_values(o) for o in cfg.output],
    }
    return _plain(doc)
