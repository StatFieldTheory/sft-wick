"""YAML-driven configuration for the workflow API.

Lets users declare an entire ``System`` / ``Expansion`` /
``Propagators`` / sweep pipeline in a single YAML file, with fields
mapping 1:1 to the Python L1 API so the config is self-documenting.

Invoke via the CLI::

    sft-wick run examples/demo1_config.yaml

Or programmatically::

    from sft_wick.workflow.config import load_workflow_config, run_workflow

    cfg = load_workflow_config("examples/demo1_config.yaml")
    sweep, totals_df = run_workflow(cfg)

A key the parser does not read raises :class:`ConfigError` with the dotted
path of the key and the nearest valid keys; :mod:`.schema` gives the same
keys as a JSON Schema.
"""

from __future__ import annotations

import collections.abc
import importlib.util
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from dataclasses import fields as _dc_fields
from pathlib import Path
from typing import Any

import numpy as np

from sft_wick._dense_contraction import DEFAULT_CONTRACTION_MAX_BYTES
from sft_wick.evaluate import DEFAULT_GL_CHUNK_SIZE, INTEGRATION_METHODS

from ._messages import shown, text_of

# =========================================================================
# Internal dataclasses: a thin typed mirror of the L1 ``System`` spec
# =========================================================================


@dataclass(frozen=True)
class WorkflowConfig:
    """Top-level parsed config."""

    system: "SystemConfig"
    expand: "ExpandConfig"
    propagators: "PropagatorsConfig"
    sweep: "SweepConfig"
    output: list["OutputConfig"] = field(default_factory=list)


@dataclass(frozen=True)
class SystemConfig:
    field_name: str
    n_components: int
    linear: dict  # parsed at build time
    noise: dict
    vertices: list
    nonlocal_vertices: list = field(default_factory=list)
    t_min: float = 0.0
    # Resolved at parse time so that noise.kappa2.type='callable_module'
    # (and any future module-loaded specs in the system block) can resolve
    # paths relative to the YAML file at build time.
    base_dir: Path | None = None


@dataclass(frozen=True)
class ExpandConfig:
    observable: tuple
    orders: tuple
    response_phase: bool = True
    ito: bool = True
    collect_topology: bool = True
    iso_R: Any = None
    diag_R: bool = True
    diag_C: bool = True
    iso_C: bool = False
    cache_path: Any = None
    n_jobs: int = 1
    target: str = "moment"


@dataclass(frozen=True)
class PropagatorsConfig:
    t_max: float
    n_grid_t: int = 60
    dt: float | None = None
    homogeneity: Any = None
    r_max: Any = None
    n_grid_r: Any = None
    n_grid_cos: Any = None
    x_max: Any = None
    n_grid_x: Any = None
    n_jobs: int = 1
    #: ``'auto'`` (default): use the built-in closed form when the kernel
    #: family has one; ``null`` / ``false``: always run quadrature.  A
    #: ``c_closed_form_module`` takes precedence over both.
    c_closed_form: Any = "auto"
    c_closed_form_module: Any = None
    c_closed_form_attr: str = "C_fn"
    c_closed_form_only: bool = False
    c_closed_form_vectorized: bool = False
    cache_path: Any = None
    interp_method: str = "linear"
    #: 'auto' (GL with a converged node count for built-in kernels,
    #: dblquad for callable kernels) | 'gauss_legendre' | 'dblquad'
    c_method: str = "auto"
    c_n_gauss: int = 20  # nodes per dim under c_method='gauss_legendre'
    diag_C: bool = True  # set False to preserve off-diagonal C entries
    #                       (e.g. lensing kappa-gamma_+ cross, a dense R,
    #                       a mixing noise); the C tables then hold every
    #                       C_ab. When False, also
    #                       sets expand.diag_C=False so the symbolic
    #                       simplification keeps the (a, b) observable
    #                       indices distinct.  Without that step the
    #                       order-0 cross pair (a != b) collapses to 0
    #                       via the KroneckerDelta(a, b) inserted by
    #                       DiagramTerm.apply_diagonal.


@dataclass(frozen=True)
class SweepConfig:
    positions_grid: dict
    t_final_grid: list
    #: The component axis: tuples of component indices, one per observable
    #: operator, from YAML ``component_tuples`` or its older 2-point name
    #: ``component_pairs``.  Read it as :attr:`component_tuples`; the field
    #: keeps its old name so existing construction sites keep working.
    component_pairs: list
    orders: Any = None
    vertex_types: Any = None
    integrate_over: Any = None
    method: str = "qmc_vectorized"
    n_samples: int = 2**13
    #: Sobol seed; ``None`` (YAML ``null``) draws an unseeded sequence.
    seed: int | None = 42
    n_jobs: int = 1
    n_gauss: int = 8  # used only when method='gauss_legendre'
    #: ``{point: [times]}`` pinning externals at UNEQUAL times, swept as a
    #: further Cartesian axis (mirrors ``positions_grid``).  Omit to pin every
    #: external at ``t_final``, which is what every sweep did before and
    #: which makes any observable with a response leg identically 0, since
    #: Theta kills the R joining two externals at the same time.
    external_times_grid: dict | None = None
    reuse_integrands: bool = False
    gl_chunk_size: int | None = DEFAULT_GL_CHUNK_SIZE
    qmc_coupling_chunk_size: int | None = None

    propagator_cache_bytes: int | None = None
    propagator_cache_contract: str | None = None
    contraction_backend: str = "ordered"
    contraction_max_bytes: int = DEFAULT_CONTRACTION_MAX_BYTES

    @property
    def component_tuples(self) -> list:
        """The component axis (one index per observable operator)."""
        return self.component_pairs


@dataclass(frozen=True)
class OutputConfig:
    type: str  # "table" | "npz" | "plot"
    path: Any = None
    # Type-specific:
    format: str = "markdown"  # for table
    x: Any = None  # for plot
    y: str = "value"
    hue: Any = "order"
    facet_col: Any = None


# =========================================================================
# Config errors
# =========================================================================


class _ConfigMessage:
    """Fields and one-line text shared by :class:`ConfigError` and
    :class:`ConfigWarning`."""

    def __init__(self, code: str, path: str, detail: str, suggestions=()):
        self.code = code
        self.path = path
        self.detail = detail
        self.suggestions = tuple(suggestions)
        super().__init__(str(self))

    def __str__(self) -> str:
        text = f"{self.path}: {self.detail}" if self.path else self.detail
        if self.suggestions:
            text += f"; did you mean {_or_list(self.suggestions)}?"
        return " ".join(text.splitlines())

    def __reduce__(self):
        return (type(self), (self.code, self.path, self.detail, self.suggestions))


class ConfigError(_ConfigMessage, ValueError):
    """A workflow config the parser refuses.

    ``str(error)`` is one line.  ``code`` is one of the codes below,
    ``path`` the dotted path of the key or block (``system.t_mni``,
    ``output[0].format``; empty for the file as a whole), ``detail`` the
    reason, and ``suggestions`` the nearest valid names: every name at the
    smallest edit distance, so a single misspelling always lists the name
    meant.  ``sft-wick`` prints ``sft-wick: config error [code] <str>``
    and exits with status 2.

    ``unknown_key``
        A key the block does not read, in the file or in an override.
    ``unknown_point``
        A ``sweep.positions_grid`` name that ``expand.observable`` does not
        use.
    ``duplicate_key``
        A key written twice in one mapping of the file.
    ``invalid_value``
        Any other refusal of the parser (``sft-wick`` also prints a plain
        ``ValueError``, ``TypeError`` or ``OverflowError`` of the parser
        with this code).
    ``override_syntax``
        An override that is not ``KEY=VALUE`` with a readable key.
    ``override_path``
        An override path through a value, past the end of a list, or into
        a list the file leaves out.
    ``read_error``
        A file the config names cannot be read, or a module it names lacks
        the attribute.  Of the closed-form C module only the file is
        checked: the run imports it and raises a plain exception if that
        fails.  ``sft-wick`` also prints with this code the ``OSError``,
        ``ImportError`` or ``yaml.YAMLError`` that
        :func:`load_workflow_config` raises.
    """


class OverrideError(ConfigError, KeyError):
    """A refused override.  Also a ``KeyError``, which overrides raised
    before :class:`ConfigError` existed."""


class ConfigWarning(_ConfigMessage, UserWarning):
    """An unknown key ignored under ``SFT_WICK_STRICT=0``."""


#: ``SFT_WICK_STRICT`` values that turn an unknown key in a file into a
#: :class:`ConfigWarning`; this release only.
_LENIENT_WORDS = frozenset({"0", "false", "off", "no"})


def _strict_keys() -> bool:
    """False under ``SFT_WICK_STRICT=0``: an unknown key in a file is
    warned about and ignored, as releases before 0.7.0 ignored it."""
    raw = os.environ.get("SFT_WICK_STRICT", "").strip().lower()
    return raw not in _LENIENT_WORDS


def _or_list(names) -> str:
    """``'a'``, ``'a' or 'b'``, ``'a', 'b' or 'c'``."""
    quoted = [repr(str(n)) for n in names]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " or " + quoted[-1]


def _edit_distance(a: str, b: str, limit: int) -> int:
    """Optimal string alignment distance: an insertion, a deletion, a
    substitution or a swap of two neighbours each counts 1.

    Exact up to ``limit``; any larger distance is returned as
    ``limit + 1``.  The smallest entry of a row never decreases from one
    row to the next, so the scan stops once it exceeds ``limit``.
    """
    before, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(
                prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1])
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], before[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        before, prev = prev, cur
    return min(prev[-1], limit + 1)


def _nearest(name: Any, candidates) -> tuple:
    """Every candidate at the smallest edit distance from ``name``, when
    that distance is at most ``max(2, len(name) // 3)``.

    Ties are all returned, so a single misspelling, at distance 1 from the
    name meant, always lists it (``n_grid_q`` lists ``n_grid_r``,
    ``n_grid_t`` and ``n_grid_x``).
    """
    name = text_of(name)
    limit = max(2, len(name) // 3)
    best, found = limit + 1, []
    for cand in map(str, candidates):
        bound = min(best, limit)
        if abs(len(cand) - len(name)) > bound:
            continue
        d = _edit_distance(name, cand, bound)
        if d < best:
            best, found = d, [cand]
        elif d == best:
            found.append(cand)
    return tuple(sorted(found)) if best <= limit else ()


def _join(where: str, key: Any) -> str:
    """The dotted path of ``key`` in the block at ``where``."""
    return f"{where}.{key}" if where else str(key)


def _converted(convert, value: Any, where: str):
    """``convert(value)`` (``int`` or ``float``, as the parser has always
    read the key), raising a :class:`ConfigError` that names ``where``
    when the value does not convert (``.inf`` for an integer, a word for a
    number)."""
    try:
        return convert(value)
    except (TypeError, ValueError, OverflowError):
        kind = "an integer" if convert is int else "a number"
        detail = f"must be {kind}; got {shown(value)}"
        raise ConfigError("invalid_value", where, detail) from None


def _grid_points(span: float, dt: float, where: str) -> int:
    """``max(2, ceil(span / dt))``: the node count a ``dt`` gives."""
    try:
        return max(2, int(math.ceil(span / dt)))
    except (OverflowError, ValueError):
        detail = f"{shown(dt)} gives no finite grid over a span of {shown(span)}"
        raise ConfigError("invalid_value", where, detail) from None


def _tag(d: dict, default=None):
    """``d['type']``, or ``default`` when absent.  A value that cannot be
    hashed (a list) gives ``False``, which no type table holds, so it is
    refused as an unsupported type rather than with a ``TypeError``."""
    tag = d.get("type", default)
    try:
        hash(tag)
    except TypeError:
        return False
    return tag


class _ModuleAttributeError(ConfigError, AttributeError):
    """A module the config names lacks the callable it names.  Also an
    ``AttributeError``, which the loader raised before."""


# =========================================================================
# YAML → WorkflowConfig
# =========================================================================


def load_workflow_config(
    path: str | Path,
    overrides: dict | None = None,
) -> WorkflowConfig:
    """Load and validate a workflow YAML config.

    Args:
        path: path to a YAML file.
        overrides: optional ``{dotted.key: value}`` dict to patch
            the loaded config (e.g. ``{"sweep.seed": 7}``).  A field YAML
            writes as a list (a grid, the observable, the component axis)
            also takes a tuple or a NumPy array here, e.g.
            ``{"sweep.t_final_grid": np.linspace(1.0, 5.0, 9)}``.

    Returns:
        A :class:`WorkflowConfig` ready to pass to :func:`run_workflow`.

    Raises:
        ConfigError: the file or an override is refused; an override error
            is an :class:`OverrideError`, also a ``KeyError``.
        ValueError: a value the parser refuses with no code or path.  A
            :class:`ConfigError` is a ``ValueError`` too.
        TypeError: as ``ValueError``.
        OverflowError: as ``ValueError``.
        ImportError: a path the parser reads as a module that Python cannot
            load, or an import inside that module.
        OSError: the file, or a ``.py`` module the parser reads, cannot be
            opened.
        yaml.YAMLError: YAML cannot read the file.
    """
    try:
        import yaml  # noqa: F401  (read by _load_yaml; checked here first)
    except ImportError as e:
        raise ImportError(
            "YAML workflow configs require PyYAML.  "
            "Install with `pip install pyyaml`."
        ) from e

    path = Path(path)
    with path.open() as f:
        data = _load_yaml(f)
    _require_mapping(data)

    if overrides:
        for dotted_key, value in overrides.items():
            _apply_override(data, dotted_key, value)

    _refuse_cycles(data)
    return _parse_workflow(data, base_dir=path.parent)


_MERGE_TAG = "tag:yaml.org,2002:merge"
#: Stands among the keys of a mapping for ``<<``; no key in a file equals it.
_MERGE_KEY = object()
#: A path shows this much of a key: it repeats every alias on the way.
_MAX_KEY_TEXT = 40
#: What PyYAML's constructors raise on a value they cannot read, besides
#: ``yaml.YAMLError``.
_CONSTRUCTOR_ERRORS = (
    LookupError,
    AttributeError,
    TypeError,
    ValueError,
    ArithmeticError,
)


def _load_yaml(stream):
    """``yaml.safe_load(stream)``, refusing a key written twice in one
    mapping.

    YAML keeps the last of two equal keys, so a second ``n_samples``, or a
    misspelling that gives a key the block already holds (``y`` for ``x``,
    ``diag_C`` for ``diag_R``), replaced the first without a word.  A
    mapping merged in with ``<<:`` is checked too, also when it is written
    in place.  A key that overrides one brought in by the merge, or the
    earlier of two merged mappings, is YAML's merge, not a duplicate.

    What YAML cannot read is a ``yaml.YAMLError``, also where PyYAML raises
    something else, and comes before any repeat in the file.
    """
    import yaml

    try:
        return _load_checked(stream)
    except ConfigError:
        raise
    except RecursionError:
        raise yaml.YAMLError("nested too deeply to read") from None
    except _CONSTRUCTOR_ERRORS as exc:
        raise yaml.YAMLError(f"the file cannot be read: {exc}") from None


def _load_checked(stream):
    """Read the document as ``yaml.safe_load`` does, then compare its keys.

    The document is read once, by ``SafeLoader``, and the comparison uses
    the keys that read gave.  A key read apart from the document can differ
    from them (``flatten_mapping`` retags a shared key, a constructor added
    to ``SafeLoader`` is not seen, ``!!float nan`` by alias is one object).
    """
    import yaml

    written: dict = {}

    class Loader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            _keep_as_written(node, written)
            return super().construct_mapping(node, deep=deep)

    loader = Loader(stream)
    read = loader.constructed_objects  # construct_document starts a new one
    try:
        root = loader.get_single_node()
        if root is None:
            return None
        document = loader.construct_document(root)
    finally:
        loader.dispose()
    _refuse_repeated_keys(root, written, read)
    return document


def _keep_as_written(node, written: dict) -> None:
    """Keep ``node`` and the mappings merged into it, before they are merged.

    These are the mappings where a repeat loses a value.  ``flatten_mapping``
    rewrites them, and it recurses: an override of it would add a frame per
    link and halve the merge chain YAML can read.
    """
    import yaml

    pending = [node]
    while pending:
        mapping = pending.pop()
        if isinstance(mapping, yaml.MappingNode) and mapping not in written:
            written[mapping] = list(mapping.value)
            for key_node, value_node in mapping.value:
                if key_node.tag == _MERGE_TAG:
                    pending.extend(_merge_sources(value_node))


def _merge_sources(value_node) -> list:
    """The nodes a ``<<`` key merges in."""
    import yaml

    if isinstance(value_node, yaml.SequenceNode):
        return value_node.value
    return [value_node]


def _refuse_repeated_keys(root, written: dict, read: dict) -> None:
    """Refuse the first key in the file that repeats a key of its mapping.

    ``written`` holds the mappings as written and ``read`` the value of
    each key, so ``1`` repeats ``1.0`` and a key of a merged mapping is
    compared within that mapping: one overriding it is YAML's merge.  A key
    written as an alias has the place and the line of its anchor, since
    PyYAML keeps none for the alias, so such a repeat sorts at its anchor,
    and of two such repeats of one anchor the mapping that starts first is
    named.
    """
    repeats = []
    for mapping, entries in written.items():
        lines: dict = {}
        for key_node, _ in entries:
            key = _MERGE_KEY if key_node.tag == _MERGE_TAG else read[key_node]
            if key in lines:
                repeats.append((key_node, lines[key], mapping))
            else:
                lines[key] = key_node.start_mark.line + 1
    if not repeats:
        return
    key_node, first, mapping = min(
        repeats,
        key=lambda repeat: (repeat[0].start_mark.index,
                            repeat[2].start_mark.index),
    )
    fate = (
        "YAML merges both and the later wins a key they share; "
        "in one << list the earlier wins"
        if key_node.tag == _MERGE_TAG
        else "YAML would keep only the last"
    )
    raise ConfigError(
        "duplicate_key",
        _join(_path_of(root, mapping, written), _key_text(key_node)),
        f"written twice in one mapping, on lines {first} and "
        f"{key_node.start_mark.line + 1}; {fate}",
    )


def _key_text(key_node) -> str:
    """A key as written, cut short.  ``str`` of the value it reads as can be
    far larger than the file."""
    import yaml

    if key_node.tag == _MERGE_TAG:
        return "<<"
    if not isinstance(key_node, yaml.ScalarNode):
        return "<key>"
    return _cut_key(key_node.value)


def _cut_key(text: str) -> str:
    """A key as an error shows it."""
    return text if len(text) <= _MAX_KEY_TEXT else f"{text[:_MAX_KEY_TEXT]}..."


def _path_of(root, target, written: dict) -> str:
    """The dotted path of the mapping ``target``, by its first route in the
    file.  With a stack, for a long alias chain."""
    seen = set()
    stack = [(root, None)]
    while True:
        node, route = stack.pop()
        if node is target:
            break
        if node not in seen:
            seen.add(node)
            stack.extend(reversed(_children(node, route, written)))
    return _route_text(route)


def _route_text(route) -> str:
    """The dotted path of a chain of ``(route, part)`` links."""
    parts = []
    while route is not None:
        route, part = route
        parts.append(part)
    return "".join(reversed(parts)).removeprefix(".")


def _children(node, route, written: dict) -> list:
    """The nodes in ``node``, each with its route.  A route links back to
    the one before it, so the routes of a search share their parts.

    A mapping merged in has the path of the mapping it lands in, and so
    does a key that is no scalar.
    """
    import yaml

    if isinstance(node, yaml.SequenceNode):
        return [(item, (route, f"[{i}]")) for i, item in enumerate(node.value)]
    if not isinstance(node, yaml.MappingNode):
        return []
    children = []
    for key_node, value_node in written.get(node, node.value):
        if key_node.tag == _MERGE_TAG:
            children.extend(
                (source, route) for source in _merge_sources(value_node)
            )
            continue
        if not isinstance(key_node, yaml.ScalarNode):
            children.append((key_node, route))
        children.append((value_node, (route, f".{_key_text(key_node)}")))
    return children


_OVERRIDE_SEGMENT = re.compile(r"^(?P<name>[^.\[\]]+)(?P<idx>(?:\[\d+\])*)$")


def _override_path(dotted_key: str) -> list:
    """``"output[0].path"`` -> ``["output", 0, "path"]``.

    List indices are part of the documented ``--override`` syntax
    (``--override "output[0].path=results.md"``); before, the whole
    segment was looked up as a dict key and every such override raised.
    """
    path: list = []
    for part in str(dotted_key).split("."):
        m = _OVERRIDE_SEGMENT.match(part)
        if m is None:
            raise OverrideError(
                "override_syntax",
                str(dotted_key),
                f"cannot read the segment {shown(part)}; write name, or "
                f"name[index] for a list entry",
            )
        path.append(m.group("name"))
        path.extend(int(i) for i in re.findall(r"\[(\d+)\]", m.group("idx")))
    return path


#: Every key ``_parse_sweep`` and ``_parse_component_axis`` read.
_SWEEP_KEYS = frozenset(
    {
        "gl_chunk_size",
        "qmc_coupling_chunk_size",
        "reuse_integrands",
        "propagator_cache_bytes",
        "propagator_cache_contract",
        "contraction_backend",
        "contraction_max_bytes",
        "method",
        "n_samples",
        "seed",
        "n_jobs",
        "n_gauss",
        "orders",
        "vertex_types",
        "integrate_over",
        "positions_grid",
        "t_final_grid",
        "external_times_grid",
        "component_tuples",
        "component_pairs",
    }
)


def _apply_override(data: dict, dotted_key: str, value: Any) -> None:
    """Apply an ``"a.b.c"`` (or ``"a[0].b"``) override to a parsed config.

    The path is read against the workflow schema (:mod:`.schema`), not
    against the file: a key the schema allows is set even when the file
    leaves it out, and a mapping on the way that the file leaves out is
    created, as writing the key into the file would.  A key of a ``type``
    variant follows the ``type`` in the data, so an earlier override can
    switch it.  A key the schema does not allow raises ``unknown_key``; a
    step through a value, past the end of a list or into a list the file
    leaves out raises ``override_path``.

    Only the named path changes.  A node the file shares through a YAML
    alias or a ``<<:`` merge keeps its value at its other paths, as if the
    alias were written out.
    """
    from .schema import override_target

    parent, leaf = override_target(data, _override_path(dotted_key))
    parent[leaf] = value


def _refuse_cycles(data: dict) -> None:
    """Refuse a mapping or list that holds itself through a YAML alias.

    YAML allows ``x: &g [0.0, *g]``, and the parser reads such a value
    without looking inside it, so the file passed ``sft-wick validate`` and
    ended in a traceback later.  The dotted path of the alias that closes
    the cycle is reported.

    The walk uses a stack, since a chain of aliases can be as long as the
    file, and enters each container once, so a node shared many times costs
    one visit however far it expands.  An entry links to the one it was
    reached from; the path is written out only for the error.
    """
    containers = (dict, list, tuple)
    entered, finished = set(), set()
    stack = [(data, None, False)]
    while stack:
        item, route, leaving = stack.pop()
        if leaving:
            finished.add(id(item))
            continue
        if id(item) in finished:
            continue
        if id(item) in entered:
            raise ConfigError(
                "invalid_value",
                _route_text(route),
                "an alias here makes a value hold itself",
            )
        entered.add(id(item))
        stack.append((item, route, True))
        if isinstance(item, dict):
            steps = [
                (v, f".{_cut_key(str(k))}")
                for k, v in item.items()
                if isinstance(v, containers)
            ]
        else:
            steps = [
                (v, f"[{i}]") for i, v in enumerate(item) if isinstance(v, containers)
            ]
        stack.extend((value, (route, part), False) for value, part in reversed(steps))


#: Keys each block reads; a key outside its block's set raises.  The
#: ``expand``, ``propagators`` and ``output`` sets equal the fields of their
#: config dataclasses; ``tests/test_yaml_spec_coverage.py`` checks that.
_TOP_LEVEL_KEYS = ("system", "expand", "propagators", "sweep", "output")
_SYSTEM_KEYS = ("field", "t_min", "linear", "noise", "vertices", "nonlocal_vertices")
_FIELD_KEYS = ("name", "n_components")
_NOISE_KEYS = ("kappa2", "sigma2")
_EXPAND_KEYS = frozenset(
    {
        "observable",
        "orders",
        "response_phase",
        "ito",
        "collect_topology",
        "iso_R",
        "diag_R",
        "diag_C",
        "iso_C",
        "cache_path",
        "n_jobs",
        "target",
    }
)
_PROPAGATORS_KEYS = frozenset(
    {
        "t_max",
        "n_grid_t",
        "dt",
        "homogeneity",
        "r_max",
        "n_grid_r",
        "n_grid_cos",
        "x_max",
        "n_grid_x",
        "n_jobs",
        "c_closed_form",
        "c_closed_form_module",
        "c_closed_form_attr",
        "c_closed_form_only",
        "c_closed_form_vectorized",
        "cache_path",
        "interp_method",
        "c_method",
        "c_n_gauss",
        "diag_C",
    }
)
_OUTPUT_KEYS = frozenset({"type", "path", "format", "x", "y", "hue", "facet_col"})
#: Accepted values of ``propagators.c_method``, ``output[].type`` and
#: ``output[].format``; :mod:`.schema` lists the same.
_C_METHODS = ("auto", "dblquad", "gauss_legendre")
_OUTPUT_TYPES = ("table", "npz", "plot")
_TABLE_FORMATS = ("markdown", "csv", "plain")
#: Accepted values of ``propagators.homogeneity`` besides null.
_HOMOGENEITIES = ("translation", "rotation", "general")


def _check_position_labels(expand_cfg, sweep_cfg) -> None:
    """``sweep.positions_grid`` names exactly the observable's points.

    A point without a position sits at the default coordinate, so a
    misspelt or omitted name would place the intended point there and
    return the value for the wrong separation; a stray name would also add
    a column and repeat every row.  The L1 ``positions`` argument still
    accepts points the observable does not use, and still puts an omitted
    point at the default coordinate.
    """
    labels = _observable_points(expand_cfg)
    unknown = sorted(str(k) for k in sweep_cfg.positions_grid if k not in labels)
    if unknown:
        also = (
            f" (also unknown: {', '.join(unknown[1:])})" if len(unknown) > 1 else ""
        )
        raise ConfigError(
            "unknown_point",
            f"sweep.positions_grid.{unknown[0]}",
            f"a position for a point expand.observable does not name{also}; "
            f"its points are {sorted(labels)}",
            _nearest(unknown[0], labels),
        )
    times = sweep_cfg.external_times_grid or {}
    unknown = sorted(str(k) for k in times if k not in labels)
    if unknown:
        also = (
            f" (also unknown: {', '.join(unknown[1:])})" if len(unknown) > 1 else ""
        )
        raise ConfigError(
            "unknown_point",
            f"sweep.external_times_grid.{unknown[0]}",
            f"times for a point expand.observable does not name{also}; its "
            f"points are {sorted(labels)}",
            _nearest(unknown[0], labels),
        )
    missing = sorted(labels - set(sweep_cfg.positions_grid))
    if missing:
        raise ValueError(
            f"sweep.positions_grid: no position for {missing}, which "
            f"expand.observable names; a point without a position would sit "
            f"at the default coordinate 0.  Give every point a list, e.g. "
            f"{missing[0]}: [0.0]."
        )


def _observable_points(expand_cfg) -> set:
    """The point names of ``expand.observable`` (``x`` in ``phi_a(x)``);
    :func:`_check_observable_entries` checked that each entry parses."""
    from .system import _OBS_PATTERN

    return {_OBS_PATTERN.match(s).group("spatial") for s in expand_cfg.observable}


def _check_integrate_over(expand_cfg, sweep_cfg) -> None:
    """Every ``sweep.integrate_over`` name is a point of the observable.

    The integrators refuse an unknown name too, but only once the
    propagators are built.
    """
    names = sweep_cfg.integrate_over
    if names is None or names == "all":
        return
    labels = _observable_points(expand_cfg)
    unknown = sorted(set(names) - labels)
    if unknown:
        raise ValueError(
            f"sweep.integrate_over: names {unknown}, which expand.observable "
            f"does not use; its points are {sorted(labels)}.  Write all for "
            f"every point."
        )


def _integrate_over_selection(raw: Any):
    """``sweep.integrate_over``: ``None``, ``"all"``, or a list of point
    names.  One other string is one point name, as one string is one
    vertex-type label, so ``--override sweep.integrate_over=x`` works."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return "all" if raw == "all" else [str(raw)]
    usage = (
        f"sweep.integrate_over must be null, all, a point name or a list of "
        f"point names, e.g. [x]; got {shown(raw)}."
    )
    if isinstance(raw, (bytes, bytearray, collections.abc.Mapping)):
        raise TypeError(usage)
    try:
        names = list(raw)
    except TypeError:  # a number, a boolean, a 0-d array
        raise TypeError(usage) from None
    for name in names:
        if not isinstance(name, str):
            raise TypeError(
                f"sweep.integrate_over: the entry {shown(name)} in {shown(raw)} is not "
                f"a point name; quote a name YAML would read as a number or "
                f"a boolean, e.g. '1'."
            )
    return [str(name) for name in names]


def _as_list(value: Any, usage: str) -> list:
    """A field YAML writes as a list, as a list.

    A Python override (``load_workflow_config(path, overrides=...)``) may
    pass a tuple or a NumPy array, converted with ``list()``.  A string,
    bytes, a mapping and a scalar raise ``ValueError(usage)``: ``list()``
    would split the first two into characters and keep a mapping's keys.
    """
    if isinstance(value, (str, bytes, bytearray, collections.abc.Mapping)):
        raise ValueError(usage)
    try:
        return list(value)
    except TypeError:  # a number, a boolean, a 0-d array
        raise ValueError(usage) from None


def _parse_point_grid(raw: Any, where: str) -> dict:
    """``positions_grid`` / ``external_times_grid``: a mapping from point
    names (strings) to lists of values."""
    if not isinstance(raw, collections.abc.Mapping):
        raise ValueError(
            f"{where} must be a mapping from point names to lists of values, "
            f"e.g. {{x: [0.0], y: [0.5]}}; got {shown(raw)}."
        )
    out = {}
    for key, values in raw.items():
        if not isinstance(key, str):
            raise ValueError(
                f"{where}: the point name {shown(key)} is a {type(key).__name__}, "
                f"not a string.  YAML reads an unquoted 1, 2.5, on, off, yes, "
                f"no, true, false or null as a number, a boolean or null; "
                f"quote the name as expand.observable spells it, e.g. "
                f"'1': [0.0]."
            )
        out[key] = _as_list(
            values,
            f"{where}[{shown(key)}] must be a list of values, e.g. [0.0, 0.5]; "
            f"got {shown(values)}.",
        )
    return out


def _parse_workflow(data: dict, base_dir: Path) -> WorkflowConfig:
    _require_mapping(data)
    # Unknown keys first, so ``swep:`` is reported as a misspelling of
    # ``sweep`` rather than as a missing section.
    _reject_unknown_keys(data, _TOP_LEVEL_KEYS, "")
    system_d = _require_dict(data, "system")
    expand_d = _require_dict(data, "expand")
    props_d = _require_dict(data, "propagators")
    sweep_d = _require_dict(data, "sweep")
    output_d = data.get("output")
    if output_d is None:
        output_d = []
    if not isinstance(output_d, list):
        raise ValueError(
            f"output must be a list of entries, each a mapping with a "
            f"'type', e.g. [{{type: table}}]; got {shown(output_d)}.  Write a "
            f"single entry as a one-entry list."
        )

    # Extract the top-level dt before parsing system so the linear gamma-spline
    # cache can derive its own n_grid_cache from the same dt by default.
    default_dt = props_d.get("dt")
    if default_dt is not None:
        default_dt = _converted(float, default_dt, "propagators.dt")

    sweep_cfg = _parse_sweep(sweep_d)
    system_cfg = _parse_system(
        system_d, base_dir, default_dt=default_dt, t_max=props_d.get("t_max")
    )
    expand_cfg = _parse_expand(expand_d)
    _check_position_labels(expand_cfg, sweep_cfg)
    _check_integrate_over(expand_cfg, sweep_cfg)
    props_cfg = _parse_propagators(props_d, base_dir)
    output_cfgs = [_parse_output(o, i) for i, o in enumerate(output_d)]

    return WorkflowConfig(
        system=system_cfg,
        expand=expand_cfg,
        propagators=props_cfg,
        sweep=sweep_cfg,
        output=output_cfgs,
    )


def _require_mapping(data: Any) -> None:
    """The file holds a mapping (an empty file loads as ``None``)."""
    if not isinstance(data, dict):
        raise ConfigError(
            "invalid_value",
            "config",
            f"the file must hold a mapping with the sections "
            f"{', '.join(_TOP_LEVEL_KEYS[:-1])} and optionally output; got "
            f"{type(data).__name__}",
        )


def _require_dict(d: dict, key: str) -> dict:
    if key not in d:
        raise ValueError(f"config missing required top-level section '{key}'")
    if not isinstance(d[key], dict):
        raise ValueError(f"config section '{key}' must be a mapping")
    return d[key]


def _parse_system(
    d: dict,
    base_dir: Path,
    *,
    default_dt: float | None = None,
    t_max: Any = None,
) -> SystemConfig:
    _reject_unknown_keys(d, _SYSTEM_KEYS, "system")
    fld = d.get("field")
    if fld is None:
        fld = {}
    if not isinstance(fld, dict):
        raise ValueError(
            f"system.field must be a mapping with 'name' and 'n_components', "
            f"e.g. {{name: phi, n_components: 2}}; got {shown(fld)}."
        )
    _reject_unknown_keys(fld, _FIELD_KEYS, "system.field")
    for key in ("vertices", "nonlocal_vertices"):
        blocks = d.get(key)
        if blocks is not None and not isinstance(blocks, list):
            raise ValueError(
                f"system.{key} must be a list of vertex blocks; got "
                f"{shown(blocks)}.  Write a single vertex as a one-entry list."
            )
    name = fld.get("name", "phi")
    nc = _converted(int, fld.get("n_components", 1), "system.field.n_components")
    t_min = _converted(float, d.get("t_min", 0.0), "system.t_min")

    linear = d.get("linear")
    if linear is None:
        raise ValueError("system.linear is required")
    if not isinstance(linear, dict):
        raise ValueError(f"system.linear must be a mapping; got {shown(linear)}.")
    linear = _resolve_linear(
        dict(linear),
        base_dir,
        default_dt=default_dt,
        n_components=nc,
        t_min=t_min,
        t_max=t_max,
    )

    noise = d.get("noise")
    if noise is None:
        raise ValueError("system.noise is required")
    if isinstance(noise, dict):
        _reject_unknown_keys(noise, _NOISE_KEYS, "system.noise")
        _check_noise_keys(noise)

    # Validate each vertex block and resolve its coupling (file paths are
    # relative to the YAML file).
    vertices = [
        _parse_vertex(v, base_dir, kind="local", index=i, n_components=nc)
        for i, v in enumerate(d.get("vertices", []) or [])
    ]
    nonlocal_vertices = [
        _parse_vertex(v, base_dir, kind="nonlocal", index=i, n_components=nc)
        for i, v in enumerate(d.get("nonlocal_vertices", []) or [])
    ]
    names = [v["name"] for v in vertices + nonlocal_vertices]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(
            f"system: vertex name(s) {repeated} used more than once.  The "
            f"coupling values are keyed by vertex name, so names must be "
            f"unique across vertices and nonlocal_vertices."
        )

    return SystemConfig(
        field_name=name,
        n_components=nc,
        linear=linear,
        noise=noise,
        vertices=vertices,
        nonlocal_vertices=nonlocal_vertices,
        t_min=t_min,
        base_dir=base_dir,
    )


def _check_noise_keys(noise: dict) -> None:
    """Refuse an unknown key in the ``kappa2``, kernel and ``sigma2``
    blocks when the file is loaded.  :func:`build_system` reads their
    values, and used to be the first to look at their keys, so
    ``load_workflow_config`` accepted ``sigma_tt``.

    A block whose ``type`` is missing or unsupported is checked against
    the keys of every type, so a misspelt ``type`` key is still named;
    :func:`build_system` reports the type itself.
    """
    where = "system.noise.kappa2"
    kappa2 = noise.get("kappa2")
    if isinstance(kappa2, dict):
        kt = _tag(kappa2)
        if kt in _SEPARABLE_KAPPA2:
            blocks = _SEPARABLE_KAPPA2[kt]
            allowed = ("type",) + tuple(b for b, _ in blocks)
            _reject_unknown_keys(kappa2, allowed, where, f"type {shown(kt)}")
            for block, axis in blocks:
                if isinstance(kappa2.get(block), dict):
                    _check_kernel_keys(kappa2[block], axis)
        elif kt == "callable_module":
            _reject_unknown_keys(
                kappa2, _KAPPA2_CALLABLE_KEYS, where, f"type {shown(kt)}"
            )
        else:
            every = {"type", *_KAPPA2_CALLABLE_KEYS}
            for blocks in _SEPARABLE_KAPPA2.values():
                every.update(b for b, _ in blocks)
            _reject_unknown_keys(kappa2, every, where)
    sigma2 = noise.get("sigma2")
    if isinstance(sigma2, dict):
        st = _tag(sigma2, "constant")
        keys = _SIGMA2_KEYS.get(st)
        if keys is not None:
            _reject_unknown_keys(
                sigma2, keys, "system.noise.sigma2", f"type {shown(st)}"
            )
        else:
            every = set().union(*_SIGMA2_KEYS.values())
            _reject_unknown_keys(sigma2, every, "system.noise.sigma2")


def _check_kernel_keys(d: dict, axis: str) -> None:
    """The keys of one kernel block; see :func:`_check_noise_keys`."""
    where = f"system.noise.kappa2.{_KERNEL_BLOCK[axis]}"
    kt = _tag(d, "exponential")
    if kt == "custom":
        _reject_unknown_keys(d, _CUSTOM_KERNEL_KEYS, where, "type 'custom'")
    elif (axis, kt) in _KERNEL_PARAMS:
        allowed = ("type",) + _KERNEL_PARAMS[(axis, kt)]
        _reject_unknown_keys(d, allowed, where, f"type {shown(kt)}")
    else:
        every = set(_CUSTOM_KERNEL_KEYS)
        for (a, _), params in _KERNEL_PARAMS.items():
            if a == axis:
                every.update(params)
        _reject_unknown_keys(d, every, where)


#: The three ways a vertex block can give its coupling.
_COUPLING_SOURCES = ("coupling", "coupling_path", "coupling_module")
_VERTEX_FLAGS = ("coupling_vectorized", "equal_time", "already_R_contracted")


def _vertex_keys(cls) -> set:
    """The keys of a vertex block: every field of ``cls``, plus the
    coupling sources.  A field added to :class:`LocalVertex` /
    :class:`NonLocalVertex` is accepted without a change here."""
    return (
        {f.name for f in _dc_fields(cls)} | set(_COUPLING_SOURCES) | {"coupling_attr"}
    )


def _parse_vertex(
    v: Any, base_dir: Path, *, kind: str, index: int, n_components: int
) -> dict:
    """Validate one ``system.vertices[]`` (``kind='local'``) or
    ``system.nonlocal_vertices[]`` (``kind='nonlocal'``) entry and resolve
    its coupling to an array or a callable.

    Exactly one coupling source:
      ``coupling``          : inline tensor (nested YAML lists).
      ``coupling_path``     : path to an ``.npy`` file, loaded as a
                              numpy array.
      ``coupling_module``   : path to a ``.py`` module exporting an
                              attribute (``coupling_attr``, default
                              ``coupling_fn``) used as a callable
                              ``fn(n_list, t_list) -> tensor``.
                              Required for spacetime-dependent
                              non-local vertices like demo2's ``κ^{(3)}``.

    A tensor coupling must have every axis of length ``N``: shape
    ``(N,)*order`` for a non-local vertex, ``(N,)*n`` (``n >= 1``, the
    first axis the ψ leg) for a local one.  A tensor of another shape used
    to be indexed without an error.
    """
    from . import specs as sp

    block = "vertices" if kind == "local" else "nonlocal_vertices"
    cls = sp.LocalVertex if kind == "local" else sp.NonLocalVertex
    where = f"system.{block}[{index}]"
    if not isinstance(v, dict):
        raise ValueError(
            f"{where} must be a mapping with 'name' and a coupling; " f"got {shown(v)}."
        )
    allowed = _vertex_keys(cls)
    _reject_unknown_keys(v, allowed, where)
    # Under SFT_WICK_STRICT=0 an unknown key was only warned about; it is
    # dropped here, so it is neither read nor recorded.
    out = {k: val for k, val in v.items() if k in allowed}
    name = out.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{where}: 'name' (a non-empty string) is required.")
    where = f"{where} ({shown(name)})"
    sources = [k for k in _COUPLING_SOURCES if k in out]
    if len(sources) != 1:
        raise ValueError(
            f"{where}: give exactly one of {list(_COUPLING_SOURCES)}; "
            f"got {sources or 'none'}."
        )
    if "coupling_attr" in out and sources[0] != "coupling_module":
        raise ValueError(
            f"{where}: 'coupling_attr' names the callable in "
            f"'coupling_module' and does nothing with {shown(sources[0])}."
        )
    if kind == "nonlocal":
        order = out.get("order")
        if (
            isinstance(order, bool)
            or not isinstance(order, (int, np.integer))
            or order < 1
        ):
            raise ValueError(
                f"{where}: 'order' (m, the number of psi legs, an integer "
                f">= 1) is required; got {shown(order)}."
            )
        out["order"] = int(order)
    for flag in _VERTEX_FLAGS:
        if flag in out:
            out[flag] = _as_bool(out[flag], f"{where}.{flag}")

    if sources[0] == "coupling_path":
        npy = (base_dir / text_of(out.pop("coupling_path"))).resolve()
        try:
            out["coupling"] = np.load(npy)
        except (OSError, ValueError, EOFError) as exc:
            raise ConfigError(
                "read_error", f"{where}.coupling_path", f"cannot read {npy}: {exc}"
            ) from None
    elif sources[0] == "coupling_module":
        mod_path = (base_dir / text_of(out.pop("coupling_module"))).resolve()
        attr = text_of(out.pop("coupling_attr", "coupling_fn"))
        out["coupling"] = _load_callable_from_module(mod_path, attr)
    if callable(out["coupling"]):
        return out

    if out.get("coupling_vectorized"):
        raise ValueError(
            f"{where}: coupling_vectorized applies to a callable coupling "
            f"(coupling_module), not to a tensor."
        )
    try:
        arr = np.asarray(out["coupling"])
    except ValueError:
        arr = np.asarray(None)
    if not (np.issubdtype(arr.dtype, np.number) and np.all(np.isfinite(arr))):
        raise ValueError(f"{where}: the coupling must be a tensor of finite numbers.")
    n = int(n_components)
    if kind == "nonlocal":
        want = (n,) * out["order"]
    else:
        if arr.ndim == 0:
            raise ValueError(
                f"{where}: a local coupling F^(n) is a tensor with n >= 1 "
                f"axes, the first the psi leg; got a scalar."
            )
        want = (n,) * arr.ndim
    if arr.shape != want:
        raise ValueError(
            f"{where}: the coupling has shape {arr.shape}; with "
            f"n_components = {n} it must be {want}."
        )
    out["coupling"] = arr
    return out


def _build_vertex(cls, v: dict):
    """A :class:`LocalVertex` / :class:`NonLocalVertex` from a parsed vertex
    block.  Every dataclass field the block names is passed on, so no field
    of the class is out of reach from YAML."""
    c = v["coupling"]
    kwargs = {"coupling": c if callable(c) else np.asarray(c)}
    for f in _dc_fields(cls):
        if f.name != "coupling" and f.name in v:
            kwargs[f.name] = v[f.name]
    if "order" in kwargs:
        kwargs["order"] = int(kwargs["order"])
    for flag in _VERTEX_FLAGS:
        if flag in kwargs:
            kwargs[flag] = _as_bool(kwargs[flag], flag)
    return cls(**kwargs)


#: Keys of each ``system.linear`` type.
_LINEAR_DIAGONAL_KEYS = (
    "type",
    "gamma",
    "gamma_module",
    "gamma_attr",
    "dt",
    "n_grid_cache",
    "t_max_cache",
    "t_min_cache",
)
_LINEAR_EXPLICIT_KEYS = (
    "type",
    "R_time_module",
    "R_time_attr",
    "iso_R",
    "R_time_vectorized",
)


def _resolve_linear(
    lin: dict,
    base_dir: Path,
    *,
    default_dt: float | None = None,
    n_components: int | None = None,
    t_min: float = 0.0,
    t_max: Any = None,
) -> dict:
    """Resolve ``system.linear`` based on its ``type`` field.

    Supported lowerings:

    ``type: diagonal`` (default) -> :class:`sft_wick.workflow.specs.DiagonalA`

        ``gamma``: inline list of floats, or a 1D nested-list array.
        ``gamma_module``: path to a ``.py`` module exporting an attribute
            (default ``gamma``) used as a callable ``gamma(t) -> array(N)``.
            Required for spacetime-dependent linear drift such as the
            Sachs-saddle ``2 theta^(sa)(lambda)``.

        Discretization: the spline cache uses ``n_grid_cache`` points
        uniformly on ``[t_min_cache, t_max_cache]`` (defaults 0 and
        100). When the user provides ``dt`` (here or via
        ``propagators.dt``), it is converted to
        ``n_grid_cache = ceil((t_max_cache - t_min_cache) / dt)`` so a
        single ``dt`` controls every grid in the workflow. Providing both
        ``dt`` and ``n_grid_cache`` is rejected to avoid ambiguity.

    ``type: explicit`` -> :class:`sft_wick.workflow.specs.ExplicitR`

        Escape hatch for a closed-form R: the user supplies
        ``R(t1, t2)`` directly, so the wrapper bypasses the
        gamma-spline cache entirely. This unlocks YAML use cases the
        diagonal lowering can't express: a dense drift matrix,
        causal kernels with non-exponential decay, or pre-computed
        spline callables loaded from disk.

        ``R_time_module``: path to a ``.py`` module exporting an
            attribute (default ``R_time``) used as a callable
            ``R_time(t1, t2) -> float | (N, N)``. Must enforce
            causality (return 0 when ``t1 < t2``).
        ``iso_R``: ``true`` (default) for a scalar R, ``false`` for an
            ``(N, N)`` matrix.  The callable is called once at a causal
            and once at an acausal time pair inside ``[t_min, t_max]``:
            the value must have the declared shape and vanish for
            ``t1 < t2``.  A matrix R with off-diagonal entries also
            needs ``expand.diag_R: false`` and ``expand.diag_C: false``
            (``System.expand`` refuses it otherwise).
        ``R_time_vectorized``: ``true`` if the callable also accepts two
            arrays of shape ``(n,)`` (returning ``(n,)`` or
            ``(n, N, N)``); the batched integrators then call it once per
            batch of causal pairs.  Checked against the scalar calls at
            load time.  Default ``false``.

        γ-spline cache knobs (``gamma``, ``gamma_module``, ``dt``,
        ``n_grid_cache``, ``t_max_cache``, ``t_min_cache``) do not apply
        under this type and raise if specified.  The propagator is the
        user's callable, not a derived spline.
    """
    lt = lin.get("type", "diagonal")
    if lt == "explicit":
        return _resolve_linear_explicit(
            lin, base_dir, n_components=n_components, t_min=t_min, t_max=t_max
        )
    if lt == "diagonal":
        return _resolve_linear_diagonal(
            lin, base_dir, default_dt=default_dt, n_components=n_components
        )
    raise ValueError(
        f"Unsupported linear operator type {shown(lt)}.  "
        f"Supported: 'diagonal', 'explicit'."
    )


def _decay_rates(gamma: Any, n_components: int | None) -> list:
    """``system.linear.gamma``: one decay rate per component, or one for
    all (a single number, as ``--override system.linear.gamma=1.5``
    writes it, counts as one for all)."""
    where = "system.linear.gamma"
    if not isinstance(gamma, list):
        return [_yaml_number(gamma, where)]
    if not gamma:
        raise ValueError(f"{where} is an empty list.")
    rates = [_yaml_number(g, f"{where}[{i}]") for i, g in enumerate(gamma)]
    if n_components is not None and len(rates) not in (1, int(n_components)):
        raise ValueError(
            f"{where} has {len(rates)} rates; with n_components = "
            f"{n_components} give {n_components}, or one for all."
        )
    return rates


def _resolve_linear_diagonal(
    lin: dict,
    base_dir: Path,
    *,
    default_dt: float | None = None,
    n_components: int | None = None,
) -> dict:
    """Parse-time resolver for ``type: diagonal``."""
    _reject_unknown_keys(lin, _LINEAR_DIAGONAL_KEYS, "system.linear", "type 'diagonal'")
    if "gamma" in lin and "gamma_module" in lin:
        raise ValueError(
            "system.linear: provide exactly one of {'gamma', 'gamma_module'}"
        )
    if "gamma_module" in lin:
        mod_path = (base_dir / lin.pop("gamma_module")).resolve()
        attr = lin.pop("gamma_attr", "gamma")
        lin["gamma"] = _load_callable_from_module(mod_path, attr)
    elif "gamma" not in lin:
        raise ValueError(
            "system.linear (type 'diagonal') requires 'gamma', a list of "
            "decay rates (one per component, or one for all), or "
            "'gamma_module' for a time-dependent rate."
        )
    elif "gamma_attr" in lin:
        raise ValueError(
            "system.linear: 'gamma_attr' names the callable in "
            "'gamma_module' and does nothing with 'gamma'."
        )
    else:
        lin["gamma"] = _decay_rates(lin["gamma"], n_components)

    # dt -> n_grid_cache derivation. linear.dt overrides propagators.dt.
    linear_dt = lin.pop("dt", None)
    effective_dt = (
        _converted(float, linear_dt, "system.linear.dt")
        if linear_dt is not None
        else default_dt
    )
    if effective_dt is not None:
        if effective_dt <= 0.0:
            raise ValueError(
                f"system.linear.dt (or propagators.dt) must be positive, "
                f"got {effective_dt}"
            )
        if "n_grid_cache" in lin:
            raise ValueError(
                "system.linear: specify exactly one of {'dt' (here or in "
                "propagators), 'n_grid_cache'}; got both."
            )
        t_max_cache = _converted(
            float, lin.get("t_max_cache", 100.0), "system.linear.t_max_cache"
        )
        t_min_cache = _converted(
            float, lin.get("t_min_cache", 0.0), "system.linear.t_min_cache"
        )
        lin["n_grid_cache"] = _grid_points(
            t_max_cache - t_min_cache, effective_dt, "system.linear.dt"
        )
    return lin


def _resolve_linear_explicit(
    lin: dict,
    base_dir: Path,
    *,
    n_components: int | None = None,
    t_min: float = 0.0,
    t_max: Any = None,
) -> dict:
    """Parse-time resolver for ``type: explicit``.

    Loads ``R_time`` from a user module and rejects fields that only
    make sense under the diagonal-with-gamma-spline path. The module
    is registered for cross-process by-value serialisation by
    :func:`_load_callable_from_module`, so the loaded callable composes
    with ``propagators.n_jobs > 1`` / ``sweep.n_jobs > 1`` even when
    joblib reuses a worker pool across calls.

    ``iso_R: false`` declares an ``(N, N)`` matrix R (a dense drift); the
    callable is probed once for its shape and once for causality.
    """
    forbidden = (
        "gamma",
        "gamma_module",
        "gamma_attr",
        "dt",
        "n_grid_cache",
        "t_max_cache",
        "t_min_cache",
    )
    present = [k for k in forbidden if k in lin]
    if present:
        raise ValueError(
            f"system.linear.type='explicit' does not accept "
            f"gamma-spline fields {shown(present)}.  Use 'type: diagonal' for "
            f"those, or remove them under 'type: explicit'."
        )
    if "R_time" in lin:
        raise ValueError(
            "system.linear.type='explicit' does not support an inline "
            "'R_time' (a callable cannot be expressed in YAML).  Use "
            "'R_time_module' + 'R_time_attr' instead."
        )
    _reject_unknown_keys(lin, _LINEAR_EXPLICIT_KEYS, "system.linear", "type 'explicit'")
    if "R_time_module" not in lin:
        raise ValueError(
            "system.linear.type='explicit' requires "
            "'R_time_module: <relative path to .py file>'."
        )
    iso_r = _as_bool(lin.get("iso_R", True), "system.linear.iso_R")
    lin["iso_R"] = iso_r
    mod_path = (base_dir / lin.pop("R_time_module")).resolve()
    attr = lin.pop("R_time_attr", "R_time")
    lin["R_time"] = _load_callable_from_module(mod_path, attr)
    vectorized = _as_bool(
        lin.get("R_time_vectorized", False), "system.linear.R_time_vectorized"
    )
    lin["R_time_vectorized"] = vectorized
    _probe_R_time(lin["R_time"], iso_r, n_components, t_min, t_max)
    if vectorized:
        _probe_R_time_vectorized(lin["R_time"], t_min, t_max)
    return lin


def _probe_R_time_vectorized(R, t_min: float, t_max: Any) -> None:
    """Call a callable declared ``R_time_vectorized: true`` once on arrays
    of three causal time pairs and compare with its scalar calls.  A
    callable that only takes floats would otherwise fail inside the first
    integration, far from the YAML key that caused it."""
    where = "system.linear.R_time_vectorized"
    span = _probe_span(t_min, t_max)
    t1 = t_min + span * np.array([0.7, 0.9, 0.5])
    t2 = t_min + span * np.array([0.2, 0.1, 0.4])
    # The scalar reference comes first, from floats, so an R that writes
    # into its time arrays cannot change it; ``np.array`` copies each value,
    # since the list keeps it across the later calls and an R may fill and
    # return one buffer.
    ref = np.array(
        [np.array(R(float(a), float(b)), dtype=float) for a, b in zip(t1, t2)]
    )
    try:
        arr = np.asarray(R(t1, t2), dtype=float)
    except Exception as e:  # noqa: BLE001 (a user callable)
        raise ValueError(
            f"{where}: true, but R_time raised {type(e).__name__} on arrays "
            f"of shape (3,): {e}"
        ) from e
    if arr.shape != ref.shape:
        raise ValueError(
            f"{where}: true, but R_time on arrays of shape (3,) returned "
            f"shape {arr.shape}; the scalar calls give {ref.shape}."
        )
    if not np.allclose(arr, ref, rtol=1e-12, atol=0.0):
        raise ValueError(
            f"{where}: true, but R_time on arrays differs from its scalar "
            f"calls: {arr.tolist()} against {ref.tolist()}."
        )


def _probe_span(t_min: float, t_max: Any) -> float:
    """Length of the time window the R probes sample: ``t_max - t_min``,
    or 1 when ``t_max`` is missing, not a number or not above ``t_min``."""
    try:
        hi = float(t_max)
    except (TypeError, ValueError):
        hi = float("nan")
    return (hi - t_min) if (math.isfinite(hi) and hi > t_min) else 1.0


def _probe_R_time(
    R, iso_R: bool, n_components: int | None, t_min: float, t_max: Any
) -> None:
    """Call the user's R once at a causal and once at an acausal time pair
    inside ``[t_min, t_max]``: the value must have the shape ``iso_R``
    declares and must vanish when ``t1 < t2``.  A scalar returned where a
    matrix was declared (or the reverse) surfaced much later and as
    something else; an R written without the Heaviside surfaced not at
    all."""
    where = "system.linear.R_time_module"
    span = _probe_span(t_min, t_max)
    t1, t2 = t_min + 0.7 * span, t_min + 0.2 * span
    try:
        # ``np.array`` copies: ``fwd`` is kept while R is called again,
        # and an R may fill and return one buffer.
        fwd = np.array(R(t1, t2), dtype=float)
        bwd = np.asarray(R(t2, t1), dtype=float)
    except Exception as e:  # noqa: BLE001 (a user callable)
        raise ValueError(
            f"{where}: R_time({t1:g}, {t2:g}) raised " f"{type(e).__name__}: {e}"
        ) from e
    if iso_R:
        ok, want = fwd.shape == (), "a scalar"
    elif n_components is None:
        ok = fwd.ndim == 2 and fwd.shape[0] == fwd.shape[1]
        want = "a square matrix"
    else:
        n = int(n_components)
        ok, want = fwd.shape == (n, n), f"an ({n}, {n}) matrix"
    if not ok:
        raise ValueError(
            f"{where}: R_time({t1:g}, {t2:g}) has shape {fwd.shape}, but "
            f"iso_R: {str(iso_R).lower()} declares {want}."
        )
    if not np.all(np.isfinite(fwd)):
        raise ValueError(
            f"{where}: R_time({t1:g}, {t2:g}) is not finite: {fwd.tolist()}."
        )
    if bwd.shape != fwd.shape or np.any(bwd != 0.0):
        raise ValueError(
            f"{where}: R_time({t2:g}, {t1:g}) = {bwd.tolist()}, but the "
            f"response function is causal: R(t1, t2) = 0 for t1 < t2."
        )


def _load_callable_from_module(path: Path, attr: str):
    """Import ``path`` as a standalone module and return
    ``getattr(module, attr)``.

    Registers the module under its file basename (``path.stem``) and ensures
    ``path.parent`` is on both ``sys.path`` (for the current process) and
    the ``PYTHONPATH`` environment variable (for subprocess workers, e.g.
    joblib loky). Without the env-var step, a worker process started after
    this call cannot re-import the module by name, and unpickling a cache
    that holds the loaded callable raises ``BrokenProcessPool`` /
    ``ModuleNotFoundError``.
    """
    parent_dir = str(path.parent.resolve())
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    # Propagate parent_dir to subprocess workers via PYTHONPATH. loky
    # workers inherit os.environ but not the parent's sys.path mods,
    # so this is the durable channel.
    pp = os.environ.get("PYTHONPATH", "")
    pp_parts = pp.split(os.pathsep) if pp else []
    if parent_dir not in pp_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([parent_dir, *pp_parts])

    module_name = path.stem
    spec_obj = importlib.util.spec_from_file_location(module_name, path)
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(f"Cannot load coupling module {shown(path)}.")
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = module
    spec_obj.loader.exec_module(module)
    fn = getattr(module, attr, None)
    if fn is None or not callable(fn):
        raise _ModuleAttributeError(
            "read_error",
            "",
            f"module {shown(str(path))} has no callable attribute {shown(attr)}",
        )
    _register_module_by_value(module)
    return fn


def _register_module_by_value(module) -> None:
    """Register ``module`` for cross-process by-value serialisation.

    Modules loaded via :func:`importlib.util.spec_from_file_location`
    are not importable by name in subprocess workers. joblib's loky
    backend uses a persistent worker pool whose ``sys.path`` /
    ``PYTHONPATH`` is fixed at first-pool-creation time, so a worker
    spawned during an earlier test won't have a later test's
    ``tmp_path`` available.

    cloudpickle's ``register_pickle_by_value`` flips the encoding so
    the module's source is shipped inline with each task. Workers no
    longer need to import anything by name.

    Use joblib's bundled cloudpickle on older versions and the standalone
    dependency on joblib >= 1.6. Their registries are distinct, so prefer
    the bundled copy when present. Custom installs without either copy
    retain the import-by-name fallback.
    """
    try:
        from joblib.externals import cloudpickle
    except ImportError:  # joblib >= 1.6
        try:
            import cloudpickle
        except ImportError:  # pragma: no cover - nonstandard joblib install
            return
    try:
        cloudpickle.register_pickle_by_value(module)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        # cloudpickle rejects modules that are part of a package or
        # don't have a real source file. Either case means the worker
        # can already import the module by name, so by-value encoding
        # is not needed.
        pass


def _check_observable_entries(obs: list) -> None:
    """Each ``expand.observable`` entry is an operator string that parses.

    The other entry types used to reach the component check, which indexed
    them (``IndexError``, ``KeyError``, or a ``TypeError`` without a path),
    and a string that does not parse stopped the run at the expansion.  An
    empty list printed 0 for the moment of the identity, which is 1.
    Whether a field and component suit the system is
    :func:`check_before_run`'s to check.
    """
    from .system import _OBS_PATTERN

    if not obs:
        raise ConfigError(
            "invalid_value",
            "expand.observable",
            "must list at least one operator, e.g. ['phi_a(x)', 'phi_b(y)']",
        )
    for i, op in enumerate(obs):
        if not isinstance(op, str) or _OBS_PATTERN.match(op) is None:
            got = shown(op)
            if not isinstance(op, str):
                got = f"{got} ({type(op).__name__})"
            raise ConfigError(
                "invalid_value",
                f"expand.observable[{i}]",
                f"must be an operator string of the form "
                f"'name_component(spatial)', like 'phi_a(x)'; got {got}",
            )


def _parse_expand(d: dict) -> ExpandConfig:
    from .expansion import _order_selection

    _reject_unknown_keys(d, _EXPAND_KEYS, "expand")
    obs = d.get("observable")
    if obs is None:
        raise ValueError("expand.observable is required")
    if isinstance(obs, collections.abc.Set):
        raise ValueError(
            f"expand.observable: a set has no order, and the order of the "
            f"operators fixes which component index belongs to which; give "
            f"a list, e.g. ['phi_a(x)', 'phi_b(y)']; got {shown(obs)}."
        )
    obs = _as_list(
        obs,
        f"expand.observable must be a list of operators, e.g. "
        f"['phi_a(x)', 'phi_b(y)']; got {shown(obs)}.  A string is not split "
        f"into operators.",
    )
    # A NumPy array of operators gives np.str_ entries.
    obs = [str(o) if isinstance(o, str) else o for o in obs]
    _check_observable_entries(obs)
    # One integer is one order, as for sweep.orders.
    orders = _order_selection(d.get("orders"), "expand.orders", required=True)
    if orders is None:
        raise ValueError("expand.orders is required")
    from sft_wick._connected import validate_target

    validate_target(d.get("target", "moment"), obs)
    return ExpandConfig(
        observable=tuple(obs),
        orders=orders,
        response_phase=_flag(d, "response_phase", True, "expand"),
        ito=_flag(d, "ito", True, "expand"),
        collect_topology=_flag(d, "collect_topology", True, "expand"),
        iso_R=None if d.get("iso_R") is None else _flag(d, "iso_R", None, "expand"),
        diag_R=_flag(d, "diag_R", True, "expand"),
        diag_C=_flag(d, "diag_C", True, "expand"),
        iso_C=_flag(d, "iso_C", False, "expand"),
        cache_path=d.get("cache_path"),
        n_jobs=_converted(int, d.get("n_jobs", 1), "expand.n_jobs"),
        target=d.get("target", "moment"),
    )


def _parse_propagators(d: dict, base_dir: Path) -> PropagatorsConfig:
    _reject_unknown_keys(d, _PROPAGATORS_KEYS, "propagators")
    if "t_max" not in d:
        raise ValueError("propagators.t_max is required")
    module_spec = d.get("c_closed_form_module")
    if module_spec is not None:
        module_spec = str((base_dir / module_spec).resolve())

    t_max = _converted(float, d["t_max"], "propagators.t_max")
    dt = d.get("dt")
    has_n_grid_t = "n_grid_t" in d
    if dt is not None:
        dt = _converted(float, dt, "propagators.dt")
        if dt <= 0.0:
            raise ValueError(f"propagators.dt must be positive, got {dt}")
        if has_n_grid_t:
            raise ValueError(
                "propagators: specify exactly one of {'dt', 'n_grid_t'}; " "got both."
            )
        n_grid_t = _grid_points(t_max, dt, "propagators.dt")
    else:
        n_grid_t = _converted(int, d.get("n_grid_t", 60), "propagators.n_grid_t")

    c_closed_form = d.get("c_closed_form", "auto")
    if c_closed_form in (False, None, "none", "null", "off"):
        c_closed_form = None
    elif c_closed_form is True or text_of(c_closed_form).lower() == "auto":
        c_closed_form = "auto"
    else:
        raise ValueError(
            f"propagators.c_closed_form must be 'auto' or null/false; got "
            f"{shown(c_closed_form)}.  To supply your own closed form use "
            f"c_closed_form_module / c_closed_form_attr."
        )
    homogeneity = d.get("homogeneity")
    if homogeneity is not None and homogeneity not in _HOMOGENEITIES:
        raise ConfigError(
            "invalid_value",
            "propagators.homogeneity",
            f"must be null or one of {_or_list(_HOMOGENEITIES)}; "
            f"got {shown(homogeneity)}",
            _nearest(homogeneity, _HOMOGENEITIES),
        )
    c_method = text_of(d.get("c_method", "auto"))
    if c_method not in _C_METHODS:
        raise ValueError(
            f"propagators.c_method must be 'auto', 'dblquad' or "
            f"'gauss_legendre'; got {shown(c_method)}."
        )

    return PropagatorsConfig(
        t_max=t_max,
        n_grid_t=n_grid_t,
        dt=dt,
        homogeneity=d.get("homogeneity"),
        r_max=d.get("r_max"),
        n_grid_r=d.get("n_grid_r"),
        n_grid_cos=d.get("n_grid_cos"),
        x_max=d.get("x_max"),
        n_grid_x=d.get("n_grid_x"),
        n_jobs=_converted(int, d.get("n_jobs", 1), "propagators.n_jobs"),
        c_closed_form=c_closed_form,
        c_closed_form_module=module_spec,
        c_closed_form_attr=text_of(d.get("c_closed_form_attr", "C_fn")),
        c_closed_form_only=_flag(d, "c_closed_form_only", False, "propagators"),
        c_closed_form_vectorized=_flag(
            d, "c_closed_form_vectorized", False, "propagators"
        ),
        cache_path=d.get("cache_path"),
        interp_method=text_of(d.get("interp_method", "linear")),
        c_method=c_method,
        c_n_gauss=_converted(int, d.get("c_n_gauss", 20), "propagators.c_n_gauss"),
        diag_C=_flag(d, "diag_C", True, "propagators"),
    )


def _parse_component_axis(d: dict) -> list:
    """``sweep.component_tuples`` (or its older 2-point name
    ``sweep.component_pairs``) as a list of tuples.

    Only the shape of the YAML is checked here; the length of each tuple
    against the observable, and the index range against ``n_components``,
    are checked by :meth:`Expansion.sweep` (and before the expansion runs,
    by :func:`run_workflow` and the ``--dry-run`` estimate).
    """
    given = [k for k in ("component_tuples", "component_pairs") if d.get(k) is not None]
    if len(given) == 2:
        raise ValueError(
            "sweep: give component_tuples or component_pairs, not both "
            "(component_pairs is the older name of the same axis)."
        )
    if not given:
        raise ValueError(
            "sweep.component_tuples is required: a list of component-index "
            "tuples, one index per observable operator, e.g. [[0, 1, 1]] "
            "for a 3-point observable (component_pairs, e.g. [[0, 1]], is "
            "the older 2-point spelling)."
        )
    key = given[0]
    raw = d[key]
    usage = (
        f"sweep.{key} must be a non-empty list of component tuples, "
        f"e.g. [[0, 1]]; got {shown(raw)}."
    )
    entries = _as_list(raw, usage)
    if not entries:
        raise ValueError(usage)
    tuples = []
    for entry in entries:
        if isinstance(entry, (str, bytes, bytearray)) or not isinstance(
            entry, (collections.abc.Sequence, np.ndarray)
        ):
            raise ValueError(
                f"sweep.{key}: each entry is a list of component indices, "
                f"one per observable operator, e.g. [0, 1]; got {shown(entry)}."
            )
        tuples.append(tuple(entry))
    return tuples


def _parse_sweep(d: dict) -> SweepConfig:
    from sft_wick.evaluate import (
        _validate_gl_chunk_size,
        _validate_qmc_coupling_chunk_size,
    )

    from sft_wick.evaluate import _execution_options

    _reject_unknown_keys(d, _SWEEP_KEYS, "sweep")
    _execution_options(
        d.get("propagator_cache_bytes"),
        d.get("propagator_cache_contract"),
        d.get("contraction_backend", "ordered"),
        d.get("contraction_max_bytes", DEFAULT_CONTRACTION_MAX_BYTES),
    )
    from .expansion import _order_selection, _vertex_type_selection

    if "positions_grid" not in d:
        raise ValueError("sweep.positions_grid is required")
    if "t_final_grid" not in d:
        raise ValueError("sweep.t_final_grid is required")
    positions_grid = _parse_point_grid(d["positions_grid"], "sweep.positions_grid")
    t_final_grid = _as_list(
        d["t_final_grid"],
        f"sweep.t_final_grid must be a list of times, e.g. [1.0, 2.0]; "
        f"got {shown(d['t_final_grid'])}.",
    )
    external_times_grid = d.get("external_times_grid")
    if external_times_grid is not None:
        external_times_grid = (
            _parse_point_grid(external_times_grid, "sweep.external_times_grid")
            or None
        )
    # A single string is one label and a single integer one order; a
    # string used to be iterated character by character.
    orders = _order_selection(d.get("orders"), "sweep.orders")
    vertex_types = _vertex_type_selection(d.get("vertex_types"), "sweep.vertex_types")
    cps = _parse_component_axis(d)
    seed = d.get("seed", 42)
    method = text_of(d.get("method", "qmc_vectorized"))
    if method not in INTEGRATION_METHODS:
        # Checked here so that a misspelt method stops the load, not the
        # sweep after the expansion and the propagator table.
        raise ConfigError(
            "invalid_value",
            "sweep.method",
            f"must be one of {_or_list(INTEGRATION_METHODS)}; got {shown(method)}",
            _nearest(method, INTEGRATION_METHODS),
        )
    return SweepConfig(
        propagator_cache_bytes=d.get("propagator_cache_bytes"),
        propagator_cache_contract=d.get("propagator_cache_contract"),
        contraction_backend=d.get("contraction_backend", "ordered"),
        contraction_max_bytes=d.get("contraction_max_bytes", DEFAULT_CONTRACTION_MAX_BYTES),
        positions_grid=positions_grid,
        t_final_grid=t_final_grid,
        external_times_grid=external_times_grid,
        component_pairs=cps,
        orders=None if orders is None else list(orders),
        vertex_types=None if vertex_types is None else sorted(vertex_types),
        integrate_over=_integrate_over_selection(d.get("integrate_over")),
        method=method,
        n_samples=_converted(int, d.get("n_samples", 2**13), "sweep.n_samples"),
        seed=None if seed is None else _converted(int, seed, "sweep.seed"),
        n_jobs=_converted(int, d.get("n_jobs", 1), "sweep.n_jobs"),
        n_gauss=_converted(int, d.get("n_gauss", 8), "sweep.n_gauss"),
        reuse_integrands=_as_bool(
            d.get("reuse_integrands", False), "sweep.reuse_integrands"
        ),
        gl_chunk_size=_validate_gl_chunk_size(
            d.get("gl_chunk_size", DEFAULT_GL_CHUNK_SIZE)
        ),
        qmc_coupling_chunk_size=_validate_qmc_coupling_chunk_size(
            d.get("qmc_coupling_chunk_size")
        ),
    )


def _parse_output(d: dict, index: int = 0) -> OutputConfig:
    if not isinstance(d, dict):
        raise ValueError(
            f"output[{index}] must be a mapping with a 'type', e.g. "
            f"{{type: table}}; got {shown(d)}."
        )
    _reject_unknown_keys(d, _OUTPUT_KEYS, f"output[{index}]")
    if "type" not in d:
        raise ValueError("each output entry must specify a 'type'")
    t = d["type"]
    if t not in _OUTPUT_TYPES:
        raise ValueError(
            f"output type must be one of 'table', 'npz', 'plot'; " f"got {shown(t)}."
        )
    needs = {"npz": ("path",), "plot": ("path", "x")}.get(t, ())
    for key in needs:
        if d.get(key) is None:
            # Checked here: npz and plot outputs are written after the sweep.
            raise ConfigError(
                "invalid_value",
                f"output[{index}].{key}",
                f"output type {shown(t)} requires {shown(key)}",
            )
    fmt = text_of(d.get("format", "markdown"))
    if fmt not in _TABLE_FORMATS:
        # Checked here: the table is written only after the whole sweep.
        raise ConfigError(
            "invalid_value",
            f"output[{index}].format",
            f"must be one of {_or_list(_TABLE_FORMATS)}; got {shown(fmt)}",
            _nearest(fmt, _TABLE_FORMATS),
        )
    return OutputConfig(
        type=t,
        path=d.get("path"),
        format=fmt,
        x=d.get("x"),
        y=text_of(d.get("y", "value")),
        hue=d.get("hue", "order"),
        facet_col=d.get("facet_col"),
    )


# =========================================================================
# WorkflowConfig → L1 System
# =========================================================================


def build_system(cfg: SystemConfig):
    """Lower a parsed :class:`SystemConfig` to a
    :class:`sft_wick.System` instance."""
    from . import specs as sp
    from .system import System

    # Linear operator
    lin_d = dict(cfg.linear)
    lt = lin_d.pop("type", "diagonal")
    if lt == "diagonal":
        gamma = lin_d["gamma"]
        # Pass callables through to DiagonalA; only flatten static lists/arrays.
        gamma_arg = gamma if callable(gamma) else list(gamma)
        diag_kwargs = {"gamma": gamma_arg}
        for k in ("t_max_cache", "n_grid_cache", "t_min_cache"):
            if k in lin_d:
                diag_kwargs[k] = lin_d[k]
        linear = sp.DiagonalA(**diag_kwargs)
    elif lt == "explicit":
        # User-supplied R(t1, t2): bypass the gamma-spline cache entirely.
        # A scalar R (iso_R: true) or an N x N matrix (iso_R: false).
        linear = sp.ExplicitR(
            R_time=lin_d["R_time"],
            iso_R=_as_bool(lin_d.get("iso_R", True), "system.linear.iso_R"),
            vectorized=_as_bool(
                lin_d.get("R_time_vectorized", False), "system.linear.R_time_vectorized"
            ),
        )
    else:
        raise ValueError(
            f"Unsupported linear operator type {shown(lt)}.  "
            f"Supported: 'diagonal', 'explicit'."
        )

    # Noise
    noise = _build_noise(
        cfg.noise, base_dir=cfg.base_dir, n_components=cfg.n_components
    )

    vertices = [_build_vertex(sp.LocalVertex, v) for v in cfg.vertices]
    nonlocal_vertices = [
        _build_vertex(sp.NonLocalVertex, v) for v in cfg.nonlocal_vertices
    ]

    return System(
        field=sp.FieldSpec(cfg.field_name, n_components=cfg.n_components),
        linear=linear,
        noise=noise,
        vertices=tuple(vertices),
        nonlocal_vertices=tuple(nonlocal_vertices),
        t_min=cfg.t_min,
    )


#: Kernel blocks of each separable κ² type, and the axis each is built on.
_SEPARABLE_KAPPA2 = {
    "separable_translation": (("temporal", "time"), ("spatial", "space")),
    "separable_rotation": (("temporal", "time"), ("angular", "angular")),
}
#: Keys of ``kappa2`` with ``type: callable_module``.
_KAPPA2_CALLABLE_KEYS = ("type", "module", "attr")
#: Keys of ``sigma2`` by type.
_SIGMA2_KEYS = {
    "constant": ("type", "amplitude"),
    "callable_module": ("type", "module", "attr"),
    "multiplicative": (
        "type",
        "g0",
        "g1",
        "interpretation",
        "vertex_names",
        "drift_names",
    ),
}


def _build_noise(
    d: dict, base_dir: Path | None = None, n_components: int | None = None
):
    from . import specs as sp

    if not isinstance(d, dict) or not isinstance(d.get("kappa2"), dict):
        raise ValueError(
            "system.noise must be a mapping with a 'kappa2' block and an "
            "optional 'sigma2' block."
        )
    k2_d = dict(d["kappa2"])
    kt = _tag(k2_d)
    k2_d.pop("type", None)
    if kt in _SEPARABLE_KAPPA2:
        blocks = _SEPARABLE_KAPPA2[kt]
        _reject_unknown_keys(
            k2_d, [b for b, _ in blocks], "system.noise.kappa2", f"type {shown(kt)}"
        )
        missing = [b for b, _ in blocks if b not in k2_d]
        if missing:
            raise ValueError(
                f"system.noise.kappa2 of type {shown(kt)} requires the kernel "
                f"block(s) {missing}."
            )
        kernels = [
            _build_kernel(k2_d[b], axis=ax, base_dir=base_dir) for b, ax in blocks
        ]
        if kt == "separable_translation":
            kappa2 = sp.SeparableTranslation(temporal=kernels[0], spatial=kernels[1])
        else:
            kappa2 = sp.SeparableRotation(temporal=kernels[0], angular=kernels[1])
    elif kt == "callable_module":
        _reject_unknown_keys(
            k2_d,
            _KAPPA2_CALLABLE_KEYS[1:],
            "system.noise.kappa2",
            "type 'callable_module'",
        )
        if base_dir is None:
            raise ValueError(
                "noise.kappa2.type='callable_module' requires base_dir; "
                "the workflow loader should pass it through."
            )
        if "module" not in k2_d:
            raise ValueError(
                "noise.kappa2.type='callable_module' requires "
                "'module: <relative path to .py file>'."
            )
        mod_path = (base_dir / k2_d.pop("module")).resolve()
        attr = k2_d.pop("attr", "kappa2")
        fn = _load_callable_from_module(mod_path, attr)
        kappa2 = sp.GeneralKappa2(fn=fn)
    else:
        supported = (*_SEPARABLE_KAPPA2, "callable_module")
        raise ConfigError(
            "invalid_value",
            "system.noise.kappa2.type",
            f"unsupported type {shown(d['kappa2'].get('type'))}; supported: "
            f"{_or_list(supported)}",
            _nearest(kt, supported) if isinstance(kt, str) else (),
        )

    sigma2 = _build_sigma2(d.get("sigma2"), base_dir, n_components)
    return sp.GaussianNoise(kappa2=kappa2, sigma2=sigma2)


def _build_sigma2(sig_d: Any, base_dir: Path | None, n_components: int | None):
    """``system.noise.sigma2``: ``null``, ``{type: constant, amplitude}``
    with a number or an N x N matrix, ``{type: callable_module}``, or
    ``{type: multiplicative, g0, g1, interpretation}``."""
    from . import specs as sp

    if sig_d is None:
        return None
    where = "system.noise.sigma2"
    if not isinstance(sig_d, dict):
        raise ValueError(
            f"{where} must be null or a mapping such as "
            f"{{type: constant, amplitude: 0.01}}; got {shown(sig_d)}."
        )
    st = _tag(sig_d, "constant")
    if st in _SIGMA2_KEYS:
        _reject_unknown_keys(sig_d, _SIGMA2_KEYS[st], where, f"type {shown(st)}")
    if st == "constant":
        if "amplitude" not in sig_d:
            raise ValueError(
                f"{where}: type 'constant' requires 'amplitude': a number "
                f"(amplitude * I_N) or an N x N matrix as nested lists."
            )
        return sp.ConstantImpulse(
            amplitude=_sigma2_amplitude(
                sig_d["amplitude"], n_components, f"{where}.amplitude"
            )
        )
    if st == "callable_module":
        if base_dir is None:
            raise ValueError(
                "noise.sigma2.type='callable_module' requires base_dir; "
                "the workflow loader should pass it through."
            )
        if "module" not in sig_d:
            raise ValueError(
                "noise.sigma2.type='callable_module' requires "
                "'module: <relative path to .py file>'."
            )
        mod_path = (base_dir / sig_d["module"]).resolve()
        attr = sig_d.get("attr", "sigma2")
        fn = _load_callable_from_module(mod_path, attr)
        return sp.CustomImpulse(fn=fn)
    if st == "multiplicative":
        # White noise with amplitude g(phi) = g0 + g1 phi; see
        # MultiplicativeImpulse.  D0 = g0 g0^T enters C, the rest lowers to
        # local vertices.
        missing = [k for k in ("g0", "g1") if k not in sig_d]
        if missing:
            raise ValueError(
                f"{where}: type 'multiplicative' requires {missing}: "
                f"g0 as an N x M nested list, g1 as N x M x N."
            )
        names = {
            k: tuple(sig_d[k]) for k in ("vertex_names", "drift_names") if k in sig_d
        }
        return sp.MultiplicativeImpulse(
            g0=np.asarray(sig_d["g0"], dtype=float),
            g1=np.asarray(sig_d["g1"], dtype=float),
            interpretation=text_of(sig_d.get("interpretation", "ito")),
            **names,
        )
    raise ConfigError(
        "invalid_value",
        f"{where}.type",
        f"unsupported type {shown(sig_d.get('type'))}; "
        f"supported: {_or_list(_SIGMA2_KEYS)}",
        _nearest(st, _SIGMA2_KEYS) if isinstance(st, str) else (),
    )


def _sigma2_amplitude(raw: Any, n_components: int | None, where: str):
    """A white-noise amplitude: a number, or a symmetric N x N matrix."""
    if isinstance(raw, list):
        try:
            amp = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                f"{where} must be a number or an N x N matrix of numbers; "
                f"got {shown(raw)}."
            ) from None
    else:
        return _yaml_number(raw, where)
    n = n_components
    if (
        amp.ndim != 2
        or amp.shape[0] != amp.shape[1]
        or (n is not None and amp.shape != (n, n))
    ):
        size = f"{n} x {n}" if n is not None else "N x N"
        raise ValueError(
            f"{where} has shape {amp.shape}; a matrix amplitude must be "
            f"{size} (n_components = {n})."
        )
    if not np.all(np.isfinite(amp)):
        raise ValueError(f"{where} has a non-finite entry: {shown(raw)}.")
    scale = float(np.abs(amp).max())
    if float(np.abs(amp - amp.T).max()) > 1e-12 * scale:
        raise ValueError(
            f"{where} must be symmetric: it is the covariance of the white "
            f"noise, and the action term psi sigma2 psi / 2 reads only its "
            f"symmetric part.  Got {shown(raw)}."
        )
    return amp


#: Parameters of each built-in kernel of a separable κ², by (axis, type).
_KERNEL_PARAMS = {
    ("time", "exponential"): ("lam", "sigma_t"),
    ("time", "gaussian"): ("lam", "sigma_t"),
    ("space", "exponential"): ("sigma_x",),
    ("space", "gaussian"): ("sigma_x",),
    ("angular", "legendre"): ("coeffs",),
}
#: YAML block of each kernel axis; also the default ``attr`` of a
#: ``type: custom`` kernel.
_KERNEL_BLOCK = {"time": "temporal", "space": "spatial", "angular": "angular"}
#: Keys of a kernel with ``type: custom``.
_CUSTOM_KERNEL_KEYS = ("type", "module", "attr")


def _reject_unknown_keys(d: dict, allowed, where: str, note: str = "") -> None:
    """Refuse a key the block does not read: a misspelt key is otherwise
    ignored without a word, and its default used instead.

    ``where`` is the dotted path of the block (``""`` for the top level),
    ``note`` names the ``type`` whose keys ``allowed`` are.  The error
    names the first unknown key and the keys nearest it.  Under
    ``SFT_WICK_STRICT=0`` each unknown key is a :class:`ConfigWarning`
    instead, and the block is read as if the key were absent.
    """
    allowed = set(allowed)
    unknown = sorted((k for k in d if k not in allowed), key=str)
    if not unknown:
        return
    detail = "unknown key" + (f" for {note}" if note else "")
    if not _strict_keys():
        for key in unknown:
            warnings.warn(
                ConfigWarning(
                    "unknown_key",
                    _join(where, key),
                    f"{detail}, ignored because SFT_WICK_STRICT=0",
                    _nearest(key, allowed),
                ),
                stacklevel=3,
            )
        return
    if len(unknown) > 1:
        others = ", ".join(_join(where, k) for k in unknown[1:])
        detail += f" (also unknown: {others})"
    raise ConfigError(
        "unknown_key", _join(where, unknown[0]), detail, _nearest(unknown[0], allowed)
    )


def _yaml_number(value: Any, where: str):
    """A finite real number from YAML.

    PyYAML reads ``1e-3`` (no decimal point) as a string, so a numeric
    string is converted; any other non-number raises.  A number is
    returned unchanged, so existing specs keep their ``repr``.
    """
    if isinstance(value, bool):
        raise ValueError(f"{where} must be a number; got {shown(value)}.")
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise ValueError(f"{where} must be a number; got {shown(value)}.") from None
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{where} must be a number; got {shown(value)}.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{where} must be finite; got {shown(value)}.")
    return value


_TRUE_WORDS = {"true", "yes", "on", "1"}
_FALSE_WORDS = {"false", "no", "off", "0"}


def _flag(d: dict, key: str, default: bool | None, block: str) -> bool:
    """``d[key]`` (or ``default``) read as a flag; see :func:`_as_bool`."""
    return _as_bool(d.get(key, default), f"{block}.{key}")


def _as_bool(value: Any, where: str) -> bool:
    """A YAML flag as a bool.

    ``bool("false")`` is ``True``, so a quoted ``"false"`` used to switch a
    flag on; the words true/false, yes/no, on/off and 1/0 are read, and
    anything else raises.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    raise ValueError(f"{where} must be true or false; got {shown(value)}.")


def _build_kernel(d: dict, axis: str, base_dir: Path | None = None):
    """One kernel of a separable κ²: a built-in family, or ``type: custom``
    with ``module`` (+ ``attr``) naming a user callable, which becomes a
    :class:`~sft_wick.workflow.specs.CustomKernel`."""
    from . import specs as sp

    where = f"system.noise.kappa2.{_KERNEL_BLOCK[axis]}"
    if not isinstance(d, dict):
        raise ValueError(
            f"{where} must be a mapping such as {{type: exponential, ...}}; "
            f"got {shown(d)}."
        )
    kt = _tag(d, "exponential")
    if kt == "custom":
        _reject_unknown_keys(d, _CUSTOM_KERNEL_KEYS, where, "type 'custom'")
        if "module" not in d:
            raise ValueError(
                f"{where}: type 'custom' requires 'module: <path to a .py "
                f"file, relative to the YAML file>' exporting the kernel "
                f"callable as 'attr' (default {shown(_KERNEL_BLOCK[axis])})."
            )
        if base_dir is None:
            raise ValueError(
                f"{where}: type 'custom' needs the YAML file's directory to "
                f"resolve 'module'; the workflow loader passes it."
            )
        fn = _load_callable_from_module(
            (base_dir / text_of(d["module"])).resolve(),
            text_of(d.get("attr", _KERNEL_BLOCK[axis])),
        )
        return sp.CustomKernel(fn=fn)

    params = _KERNEL_PARAMS.get((axis, kt))
    if params is None:
        supported = sorted(t for (a, t) in _KERNEL_PARAMS if a == axis)
        raise ValueError(
            f"Unsupported {axis}-kernel type {shown(kt)} at {where}.  "
            f"Supported: {supported + ['custom']}."
        )
    _reject_unknown_keys(d, ("type",) + params, where, f"type {shown(kt)}")
    missing = [p for p in params if p not in d]
    if missing:
        raise ValueError(f"{where}: type {shown(kt)} requires {missing}.")
    if kt == "legendre":
        coeffs = d["coeffs"]
        if not isinstance(coeffs, list) or not coeffs:
            raise ValueError(
                f"{where}.coeffs must be a non-empty list [C_0, C_1, ...]; "
                f"got {shown(coeffs)}."
            )
        return sp.LegendreAngular(
            coeffs=[
                _yaml_number(c, f"{where}.coeffs[{i}]") for i, c in enumerate(coeffs)
            ]
        )
    vals = {p: _yaml_number(d[p], f"{where}.{p}") for p in params}
    for p in ("sigma_t", "sigma_x"):
        if p in vals and not float(vals[p]) > 0.0:
            raise ValueError(f"{where}.{p} must be positive; got {shown(vals[p])}.")
    cls = {
        ("time", "exponential"): sp.ExponentialTemporal,
        ("time", "gaussian"): sp.GaussianTemporal,
        ("space", "exponential"): sp.ExponentialSpatial,
        ("space", "gaussian"): sp.GaussianSpatial,
    }[(axis, kt)]
    return cls(**vals)


# =========================================================================
# Full runner
# =========================================================================


def check_before_run(cfg: WorkflowConfig, system=None):
    """The checks :func:`run_workflow` makes before the expansion, on
    ``cfg`` alone: the observable against the system (the parse
    :meth:`System.expand` makes), the component tuples against the
    observable and ``n_components``, the point names against the sweep's
    columns, ``sweep.orders`` against ``expand.orders``, the spatial grid
    keys, one parallel layer, and the closed-form C module on disk.  Builds
    the System unless ``system`` is given, and returns it.

    ``sft-wick validate`` and ``sft-wick run`` call it before anything is
    computed.  A ``sweep.vertex_types`` label is checked only against the
    diagrams, after the expansion.
    """
    from .expansion import (
        _check_sweep_columns,
        _resolve_component_tuples,
        component_columns,
    )
    from .system import _parse_observable

    system = build_system(cfg.system) if system is None else system
    try:
        _parse_observable(
            cfg.expand.observable, system, max(cfg.expand.orders, default=0)
        )
    except (ValueError, TypeError) as exc:
        raise ConfigError("invalid_value", "expand.observable", str(exc)) from None
    _resolve_component_tuples(
        None,
        cfg.sweep.component_tuples,
        tuple(cfg.expand.observable),
        system.n_components,
        "sweep.component_tuples",
    )
    _check_sweep_columns(
        cfg.sweep.positions_grid,
        cfg.sweep.external_times_grid or (),
        component_columns(len(cfg.expand.observable)),
    )
    if cfg.sweep.orders is not None:
        absent = [o for o in cfg.sweep.orders if o not in cfg.expand.orders]
        if absent:
            # The message Expansion._orders_list gives after the expansion.
            raise ValueError(
                f"sweep: order(s) {absent} are not in the expansion; it holds "
                f"orders {list(cfg.expand.orders)}."
            )
    # Parallelism layers cannot nest because joblib's loky backend does not
    # support nested process pools: ``sweep.n_jobs`` (over grid points) and
    # ``expand.n_jobs`` (over diagrams within a grid point).
    if int(cfg.expand.n_jobs) != 1 and int(cfg.sweep.n_jobs) != 1:
        raise ValueError(
            "Specify exactly one of {expand.n_jobs > 1, sweep.n_jobs > 1}; "
            "nested joblib loky pools are not supported."
        )
    _check_spatial_grid(cfg, system)
    module = cfg.propagators.c_closed_form_module
    if module is not None and not Path(module).is_file():
        raise FileNotFoundError(
            f"propagators.c_closed_form_module: no such file {module}"
        )
    return system


#: The spatial grid keys of ``propagators`` each homogeneity reads; the keys
#: of a pair are read only together (``Propagators.build``).
_SPATIAL_GRIDS = {
    "translation": ("r_max", "n_grid_r"),
    "rotation": ("n_grid_cos",),
    "general": ("x_max", "n_grid_x"),
}


def _check_spatial_grid(cfg: WorkflowConfig, system) -> None:
    """Refuse a spatial grid key the run would not read: one for another
    homogeneity, or one of a pair without the other.  ``n_grid_t``
    misspelt as ``n_grid_r`` or ``n_grid_x`` used to be ignored this way,
    leaving the default 60 time nodes."""
    props = cfg.propagators
    hom = props.homogeneity or system.homogeneity
    read = _SPATIAL_GRIDS.get(hom, ())
    given = [
        k
        for grid in _SPATIAL_GRIDS.values()
        for k in grid
        if getattr(props, k) is not None
    ]
    for key in given:
        owner = next(h for h, grid in _SPATIAL_GRIDS.items() if key in grid)
        if key not in read:
            detail = (
                f"is read only under homogeneity {shown(owner)}, and this run's "
                f"homogeneity is {shown(hom)}, so it would be ignored"
            )
        elif not all(k in given for k in read):
            other = next(k for k in read if k not in given)
            detail = (
                f"is read only together with propagators.{other}, so alone "
                f"it would be ignored"
            )
        else:
            continue
        raise ConfigError(
            "invalid_value",
            f"propagators.{key}",
            detail,
            _nearest(key, _PROPAGATORS_KEYS - {key}),
        )


def effective_expand_diag_C(cfg: WorkflowConfig) -> bool:
    """The ``diag_C`` the expansion runs with: ``propagators.diag_C:
    false`` also turns ``expand.diag_C`` off (see :func:`run_workflow`)."""
    return bool(cfg.expand.diag_C and cfg.propagators.diag_C)


def run_workflow(cfg: WorkflowConfig, progress: Any = None):
    """Execute the full pipeline: expand, build propagators, sweep,
    emit outputs.

    Returns ``(sweep, totals_dataframe)`` for programmatic use.

    ``progress`` is the :mod:`sft_wick.progress` setting for the three
    reported stages (expansion, propagator table, sweep): ``True`` /
    ``False`` / callable / ``None`` (inherit).
    """
    from sft_wick.progress import progress as _progress_scope
    from sft_wick.progress import stage

    with _progress_scope(progress):
        return _run_workflow(cfg, stage)


def _run_workflow(cfg: WorkflowConfig, stage):
    from sft_wick.evaluate import _execution_options

    execution = _execution_options(
        cfg.sweep.propagator_cache_bytes,
        cfg.sweep.propagator_cache_contract,
        cfg.sweep.contraction_backend,
        cfg.sweep.contraction_max_bytes,
    )

    # The component axis, the order selection and the parallel layers are
    # checked before the expensive stages run; Expansion.sweep checks them
    # again, but only after the expansion and the propagator table.
    system = check_before_run(cfg)

    # ``propagators.diag_C`` is the user-facing knob (the single
    # source of truth for "is C diagonal?"). The symbolic-side
    # ``expand.diag_C`` must agree: with ``propagators.diag_C=False``
    # the closed-form C returns a full (N, N) matrix per sample, but
    # ``expand.diag_C=True`` would collapse the observable (a, b)
    # index pair through ``KroneckerDelta(a, b)``, zeroing every
    # cross-component pair at order 0. Reject the contradictory
    # combination with a clear pointer instead of silently rounding
    # the result to zero.
    # A user who opted into off-diagonal C but left expand.diag_C at its
    # default (True) gets it synced, without setting the knob twice.
    expand_diag_C = effective_expand_diag_C(cfg)

    with stage("expansion", f"orders {list(cfg.expand.orders)}"):
        expansion = system.expand(
            observable=cfg.expand.observable,
            orders=cfg.expand.orders,
            response_phase=cfg.expand.response_phase,
            ito=cfg.expand.ito,
            collect_topology=cfg.expand.collect_topology,
            iso_R=cfg.expand.iso_R,
            diag_R=cfg.expand.diag_R,
            diag_C=expand_diag_C,
            iso_C=cfg.expand.iso_C,
            cache_path=cfg.expand.cache_path,
            target=getattr(cfg.expand, "target", "moment"),
        )
    counts = ", ".join(
        f"order {o}: {len(expansion.diagrams(o))}" for o in expansion.orders
    )
    _log(f"[sft-wick] diagrams -- {counts}")
    # An order or a vertex type the expansion does not hold is refused here,
    # before the propagator table is built.
    expansion._selection(cfg.sweep.orders, cfg.sweep.vertex_types, "sweep")

    c_fn = _load_c_closed_form(cfg.propagators)
    # User-supplied C_fn modules are loaded via
    # :func:`_load_callable_from_module`, which registers them under
    # their ``.py`` file's bare stem and adds the parent directory to
    # ``sys.path``, so the callable is importable in joblib loky
    # subprocesses. Combined with the module-level
    # ``_ClosedFormPropagatorCache`` class, this lets users opt into
    # parallel C-table builds (``propagators.n_jobs: -1``) when their
    # c_fn does heavy work per call.
    n_jobs = cfg.propagators.n_jobs
    with stage(
        "propagators",
        f"t_max={cfg.propagators.t_max}, " f"n_grid_t={cfg.propagators.n_grid_t}",
    ):
        props = system.propagators(
            t_max=cfg.propagators.t_max,
            n_grid_t=cfg.propagators.n_grid_t,
            homogeneity=cfg.propagators.homogeneity,
            r_max=cfg.propagators.r_max,
            n_grid_r=cfg.propagators.n_grid_r,
            n_grid_cos=cfg.propagators.n_grid_cos,
            x_max=cfg.propagators.x_max,
            n_grid_x=cfg.propagators.n_grid_x,
            n_jobs=n_jobs,
            c_closed_form=c_fn,
            cache_path=cfg.propagators.cache_path,
            interp_method=cfg.propagators.interp_method,
            c_closed_form_only=cfg.propagators.c_closed_form_only,
            c_closed_form_vectorized=cfg.propagators.c_closed_form_vectorized,
            c_method=cfg.propagators.c_method,
            c_n_gauss=cfg.propagators.c_n_gauss,
            diag_C=cfg.propagators.diag_C,
        )
    _log(
        f"[sft-wick] C propagator: {props.c_source}"
        + (", lazy per-separation table" if props.is_lazy else "")
    )

    # ``sweep.n_jobs`` (over Cartesian-product grid points) is forwarded to
    # ``expansion.sweep``; ``expand.n_jobs`` (over diagrams within a single
    # grid point) is forwarded as ``evaluate_n_jobs``.  check_before_run
    # refused the two together; :meth:`Expansion.sweep` enforces it too.
    with stage("sweep", f"method={cfg.sweep.method}"):
        sweep = expansion.sweep(
            props,
            **execution,
            positions_grid=cfg.sweep.positions_grid,
            t_final_grid=cfg.sweep.t_final_grid,
            external_times_grid=cfg.sweep.external_times_grid,
            component_tuples=cfg.sweep.component_tuples,
            orders=cfg.sweep.orders,
            vertex_types=cfg.sweep.vertex_types,
            integrate_over=cfg.sweep.integrate_over,
            method=cfg.sweep.method,
            n_samples=cfg.sweep.n_samples,
            seed=cfg.sweep.seed,
            n_jobs=cfg.sweep.n_jobs,
            evaluate_n_jobs=cfg.expand.n_jobs,
            n_gauss=cfg.sweep.n_gauss,
            reuse_integrands=cfg.sweep.reuse_integrands,
            gl_chunk_size=cfg.sweep.gl_chunk_size,
            **(
                {}
                if cfg.sweep.qmc_coupling_chunk_size is None
                else {"qmc_coupling_chunk_size": cfg.sweep.qmc_coupling_chunk_size}
            ),
        )

    totals = sweep.totals()
    for out in cfg.output:
        _emit_output(out, sweep, totals)

    return sweep, totals


def _log(msg: str) -> None:
    """Stage-level notice, printed only when progress reporting is on."""
    from sft_wick.progress import current_setting

    if current_setting() is True:
        print(msg, file=sys.stderr, flush=True)


def _load_c_closed_form(cfg: PropagatorsConfig):
    """The ``c_closed_form`` argument for :meth:`System.propagators`.

    A user module (``c_closed_form_module`` / ``c_closed_form_attr``)
    is imported and returned; otherwise the ``c_closed_form`` setting
    itself (``'auto'`` or ``None``) is passed through.
    """
    if cfg.c_closed_form_module is None:
        return cfg.c_closed_form
    path = Path(cfg.c_closed_form_module).resolve()

    # Register the module under its file basename and put the parent on
    # sys.path AND PYTHONPATH so that joblib loky workers can re-import
    # it when receiving tasks during parallel C-table builds, integration,
    # or sweep dispatch. See ``_load_callable_from_module`` for the same
    # pattern applied to coupling / gamma callables.
    parent_dir = str(path.parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    pp = os.environ.get("PYTHONPATH", "")
    pp_parts = pp.split(os.pathsep) if pp else []
    if parent_dir not in pp_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([parent_dir, *pp_parts])
    module_name = path.stem

    spec_obj = importlib.util.spec_from_file_location(module_name, path)
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(f"Cannot load c_closed_form_module {shown(path)}.")
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = module
    spec_obj.loader.exec_module(module)
    fn = getattr(module, cfg.c_closed_form_attr, None)
    if fn is None:
        raise AttributeError(
            f"Module {shown(path)} has no attribute "
            f"{shown(cfg.c_closed_form_attr)}."
        )
    _register_module_by_value(module)
    return fn


def _emit_output(out: OutputConfig, sweep, totals) -> None:
    if out.type == "table":
        payload = _format_table(totals, out.format)
        _write_or_print(payload, out.path)
    elif out.type == "npz":
        if out.path is None:
            raise ValueError("output type 'npz' requires a 'path'.")
        Path(out.path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out.path,
            **{col: totals[col].to_numpy() for col in totals.columns},
        )
    elif out.type == "plot":
        if out.path is None:
            raise ValueError("output type 'plot' requires a 'path'.")
        if out.x is None:
            raise ValueError("output type 'plot' requires 'x'.")
        fig = sweep.plot(
            x=out.x,
            y=out.y,
            hue=out.hue,
            facet_col=out.facet_col,
        )
        Path(out.path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.path, dpi=120, bbox_inches="tight")


def _format_table(df, fmt: str) -> str:
    if fmt == "markdown":
        return df.to_markdown(index=False)
    if fmt == "csv":
        return df.to_csv(index=False)
    if fmt == "plain":
        return df.to_string(index=False)
    raise ValueError(
        f"output.format must be 'markdown', 'csv', or 'plain'; got {shown(fmt)}."
    )


def _write_or_print(payload: str, path: Any) -> None:
    if path is None:
        print(payload)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(payload)
