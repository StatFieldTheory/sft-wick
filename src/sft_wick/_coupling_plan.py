"""Private ordered coupling lowering; plans contain structure, never values.

Lowering binds fixed operations and native-node metadata once. It retains
scalar builtin-sum and batched list-then-fold semantics, including exception
order. Each wrapper carries its caller's component-label witness; executable
cores share ordered native metadata modulo those labels. Numeric literal labels
are never renamed. Symbolic equality is not a cache key.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
import itertools as _itertools
from typing import Any, Callable
from types import (
    BuiltinFunctionType,
    FunctionType,
    GetSetDescriptorType,
    MemberDescriptorType,
    ModuleType,
)
from pathlib import Path
from weakref import WeakKeyDictionary, ref
import sys

import numpy as np

from .expressions import (
    Expr,
    ImaginaryUnit,
    KroneckerDelta,
    Product,
    Rational,
    Sum,
    Symbol,
)


@dataclass(frozen=True)
class _SymbolRecipe:
    name: str
    indices: tuple[Any, ...]
    literals: tuple[int | None, ...]
    slot: int
    read_slot: int


@dataclass(frozen=True)
class _IndexBindings:
    original: dict
    resolved: tuple[tuple | None, ...]
    labels: tuple[str, ...] = ()
    symbol_values: list | None = None


@dataclass(frozen=True)
class _LabelBindings:
    original: dict
    labels: tuple[str, ...]


_UNREAD = object()
_NATIVE_NUMPY_READ_HELPERS = {
    name: getattr(np, name)
    for name in (
        "asarray",
        "geterr",
        "iscomplexobj",
        "broadcast_to",
        "zeros",
        "ndindex",
    )
}


def native_numpy_c_type(kind, name):
    """An immutable native extension type, not a Python class naming itself NumPy."""
    return (
        type(kind) is type
        and kind.__module__ == "numpy"
        and kind.__name__ == name
        and type(getattr(kind, "__flags__", None)) is int
        and bool(kind.__flags__ & (1 << 8))  # CPython Py_TPFLAGS_IMMUTABLETYPE
    )


def _numpy_helper_provenance(name, function):
    """Reject hooks installed before this module imported NumPy as well."""
    if name in ("asarray", "zeros"):
        return (
            type(function) is BuiltinFunctionType
            and function.__module__ == "numpy"
            and function.__name__ == name
        )
    if name == "geterr":
        implementation = function if type(function) is FunctionType else None
    elif name == "ndindex":
        implementation = (
            function.__dict__.get("__init__")
            if type(function) is type
            and function.__module__ == "numpy"
            and function.__name__ == name
            else None
        )
        if (
            type(implementation) is not FunctionType
            or implementation.__globals__.get("ndindex") is not function
            or function.__bases__ != (object,)
            or "__getattribute__" in function.__dict__
            or "__getattr__" in function.__dict__
        ):
            return False
    else:
        implementation = (
            getattr(function, "__wrapped__", None)
            if native_numpy_c_type(type(function), "_ArrayFunctionDispatcher")
            else None
        )
    return (
        type(implementation) is FunctionType
        and type(implementation.__module__) is str
        and (
            implementation.__module__ == "numpy"
            or implementation.__module__.startswith("numpy.")
        )
        and Path(implementation.__code__.co_filename)
        .resolve()
        .is_relative_to(Path(np.__file__).resolve().parent)
    )


_NATIVE_NUMPY_READ_PROVENANCE = all(
    _numpy_helper_provenance(name, fn)
    for name, fn in _NATIVE_NUMPY_READ_HELPERS.items()
)
_NATIVE_NUMPY_READ_HELPERS = {
    name: fn if _numpy_helper_provenance(name, fn) else None
    for name, fn in _NATIVE_NUMPY_READ_HELPERS.items()
}
_NATIVE_NDINDEX_METHODS: dict[str, Any] = {
    name: (
        _NATIVE_NUMPY_READ_HELPERS["ndindex"].__dict__.get(name)
        if _NATIVE_NUMPY_READ_HELPERS["ndindex"] is not None
        else None
    )
    for name in ("__init__", "__iter__", "__next__")
}
_NATIVE_NDINDEX_PROVENANCE = all(
    type(fn) is FunctionType
    and type(fn.__module__) is str
    and fn.__module__.startswith("numpy.")
    and Path(fn.__code__.co_filename)
    .resolve()
    .is_relative_to(Path(np.__file__).resolve().parent)
    for fn in _NATIVE_NDINDEX_METHODS.values()
)
if not _NATIVE_NDINDEX_PROVENANCE:
    _NATIVE_NDINDEX_METHODS = {}


def _native_ndindex_fields():
    cls = _NATIVE_NUMPY_READ_HELPERS["ndindex"]
    if cls is None or not _NATIVE_NDINDEX_PROVENANCE:
        return {}
    fields = dict(cls.__dict__)
    root = Path(np.__file__).resolve().parent
    for value in fields.values():
        if type(value) is FunctionType:
            if not Path(value.__code__.co_filename).resolve().is_relative_to(root):
                return {}
        elif type(value) is GetSetDescriptorType:
            if value.__objclass__ is not cls:
                return {}
        elif type(value) is tuple:
            if len(value) > 128 or any(type(item) is not str for item in value):
                return {}
        elif type(value) not in (str, type(None), int, bool):
            return {}
    return fields


_NATIVE_NDINDEX_FIELDS = _native_ndindex_fields()
_NATIVE_NDINDEX_GLOBALS = (
    _NATIVE_NDINDEX_METHODS["__init__"].__globals__ if _NATIVE_NDINDEX_FIELDS else None
)
_NATIVE_NDINDEX_PRODUCT = (
    _itertools.product
    if type(_itertools.product) is type
    and _itertools.product.__module__ == "itertools"
    and _itertools.product.__name__ == "product"
    and _itertools.product.__flags__ & (1 << 8)
    and _NATIVE_NDINDEX_GLOBALS is not None
    and _NATIVE_NDINDEX_GLOBALS.get("product") is _itertools.product
    and "product" in _NATIVE_NDINDEX_METHODS["__init__"].__code__.co_names
    else None
)
_NDINDEX_ALLOWED_NAMES = {
    "__init__": {
        "len",
        "isinstance",
        "tuple",
        "min",
        "ValueError",
        "product",
        "map",
        "range",
        "_iter",
    },
    "__iter__": set(),
    "__next__": {"next", "_iter"},
}


def _native_ndindex_builtins():
    resolved = {}
    valid = True
    for method_name, method in _NATIVE_NDINDEX_METHODS.items():
        if not set(method.__code__.co_names) <= _NDINDEX_ALLOWED_NAMES[method_name]:
            valid = False
        for name in set(method.__code__.co_names) - {"product", "_iter"}:
            value = method.__globals__.get(name, method.__builtins__.get(name))
            if not (
                (
                    type(value) is BuiltinFunctionType
                    or (type(value) is type and value.__flags__ & (1 << 8))
                )
                and value.__module__ == "builtins"
                and value.__name__ == name
            ):
                valid = False
            else:
                resolved[(method_name, name)] = value
    return resolved if valid else None


_NATIVE_NDINDEX_BUILTINS = _native_ndindex_builtins()
if _NATIVE_NDINDEX_BUILTINS is None:
    _NATIVE_NDINDEX_PRODUCT = None
    _NATIVE_NDINDEX_BUILTINS = {}
_NATIVE_NDINDEX_CODES = {
    name: fn.__code__ for name, fn in _NATIVE_NDINDEX_METHODS.items()
}
if not _NATIVE_NDINDEX_FIELDS or _NATIVE_NDINDEX_PRODUCT is None:
    _NATIVE_NDINDEX_PROVENANCE = False
    _NATIVE_NDINDEX_METHODS = {}
    _NATIVE_NDINDEX_CODES = {}
    _NATIVE_NUMPY_READ_HELPERS["ndindex"] = None
    _NATIVE_NDINDEX_GLOBALS = None
    _NATIVE_NDINDEX_FIELDS = {}
    _NATIVE_NDINDEX_BUILTINS = {}
_MISSING_NDINDEX_FIELD = object()


def _native_small_numpy_helpers():
    geterr = _NATIVE_NUMPY_READ_HELPERS["geterr"]
    dispatcher = _NATIVE_NUMPY_READ_HELPERS["iscomplexobj"]
    if geterr is None or dispatcher is None:
        return None
    iscomplex = dispatcher.__wrapped__
    getter = geterr.__globals__.get("_get_extobj_dict")
    nx = iscomplex.__globals__.get("_nx")
    if (
        set(geterr.__code__.co_names) != {"_get_extobj_dict", "pop"}
        or type(getter) is not BuiltinFunctionType
        or getter.__module__
        not in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath")
        or getter.__name__ != "_get_extobj_dict"
        or type(nx) is not ModuleType
        or type(nx.__dict__.get("complexfloating")) is not type
    ):
        return None
    complex_type = nx.__dict__["complexfloating"]
    if not native_numpy_c_type(complex_type, "complexfloating") or not set(
        iscomplex.__code__.co_names
    ) <= {
        "dtype",
        "type",
        "AttributeError",
        "asarray",
        "issubclass",
        "_nx",
        "complexfloating",
    }:
        return None
    mappings = [
        (geterr.__globals__, "_get_extobj_dict", getter),
        (iscomplex.__globals__, "_nx", nx),
        (nx.__dict__, "complexfloating", complex_type),
    ]
    for name in ("AttributeError", "issubclass", "asarray"):
        value = iscomplex.__globals__.get(name, iscomplex.__builtins__.get(name))
        if name == "asarray":
            if value is not _NATIVE_NUMPY_READ_HELPERS["asarray"]:
                return None
        elif not (
            (
                type(value) is BuiltinFunctionType
                or (type(value) is type and value.__flags__ & (1 << 8))
            )
            and value.__module__ == "builtins"
            and value.__name__ == name
        ):
            return None
        namespace = (
            iscomplex.__globals__
            if name in iscomplex.__globals__
            else iscomplex.__builtins__
        )
        if namespace is not iscomplex.__globals__:
            mappings.append((iscomplex.__globals__, name, _MISSING_NDINDEX_FIELD))
        mappings.append((namespace, name, value))
    return (
        geterr,
        geterr.__code__,
        dispatcher,
        iscomplex,
        iscomplex.__code__,
        tuple(mappings),
    )


_NATIVE_SMALL_NUMPY = _native_small_numpy_helpers()


def _small_numpy_helpers_unchanged():
    if _NATIVE_SMALL_NUMPY is None:
        return False
    geterr, err_code, dispatcher, iscomplex, complex_code, mappings = (
        _NATIVE_SMALL_NUMPY
    )
    return (
        geterr.__code__ is err_code
        and dispatcher.__wrapped__ is iscomplex
        and iscomplex.__code__ is complex_code
        and all(
            namespace.get(name, _MISSING_NDINDEX_FIELD) is value
            for namespace, name, value in mappings
        )
    )


def native_symbol_reads():
    """No user code can change tensors between repeated native leaf reads."""
    if (
        not _NATIVE_NUMPY_READ_PROVENANCE
        or not _NATIVE_NDINDEX_PROVENANCE
        or not _small_numpy_helpers_unchanged()
        or _NATIVE_NDINDEX_GLOBALS.get("product") is not _NATIVE_NDINDEX_PRODUCT
        or any(
            _NATIVE_NDINDEX_METHODS[method].__globals__.get(
                name, _NATIVE_NDINDEX_METHODS[method].__builtins__.get(name)
            )
            is not value
            for (method, name), value in _NATIVE_NDINDEX_BUILTINS.items()
        )
        or any(
            getattr(np, name, None) is not fn
            for name, fn in _NATIVE_NUMPY_READ_HELPERS.items()
        )
        or len(np.ndindex.__dict__) != len(_NATIVE_NDINDEX_FIELDS)
        or any(
            np.ndindex.__dict__.get(name, _MISSING_NDINDEX_FIELD) is not value
            for name, value in _NATIVE_NDINDEX_FIELDS.items()
        )
        or any(
            np.ndindex.__dict__.get(name) is not fn
            or fn.__code__ is not _NATIVE_NDINDEX_CODES[name]
            for name, fn in _NATIVE_NDINDEX_METHODS.items()
        )
        or _indices is not _NATIVE_INDICES
        or _batched_symbol is not _NATIVE_BATCHED_SYMBOL
        or _delta is not _NATIVE_DELTA
        or _caller_label is not _NATIVE_CALLER_LABEL
    ):
        return False
    # Imported lazily: probe admission itself depends on this module.
    from . import _coupling_probe as probe

    return (
        probe._inert_float_errors is probe._NATIVE_INERT_FLOAT_ERRORS
        and probe._inert_float_errors()
    )


def _caller_label(label, labels):
    return labels[label] if type(label) is int else label


def native_bindings(values, index_map):
    """Inert bindings only; custom lookup/index/conversion hooks stay live."""
    return (
        type(values) is dict
        and all(
            type(k) is str and type(v) is np.ndarray and v.dtype.kind in "biufc"
            for k, v in values.items()
        )
        and type(index_map) is dict
        and all(type(k) is str and type(v) is int for k, v in index_map.items())
    )


def native_scalar_symbol_reads(values: dict, dtype: object) -> bool:
    """Scalar memoization only when conversion and folds cannot call hooks."""
    if (
        not _NATIVE_SCALAR_READ_BUILTINS_PROVENANCE
        or _NATIVE_SCALAR_TYPE(np) is not _NATIVE_SCALAR_MODULETYPE
        or _NATIVE_SCALAR_TYPE(sys) is not _NATIVE_SCALAR_MODULETYPE
    ):
        return False
    # Native member access bypasses an unknown DType's Python attribute hooks.
    if (
        _NATIVE_SCALAR_DTYPE_NUM is None
        or _NATIVE_SCALAR_TYPE(values) is not _NATIVE_SCALAR_DICTTYPE
    ):
        return False
    for value in values.values():
        if _NATIVE_SCALAR_TYPE(value) is not _NATIVE_SCALAR_ARRAYTYPE:
            return False
        typenum = _NATIVE_SCALAR_DTYPE_NUM.__get__(value.dtype)
        if typenum not in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 23):
            return False
    # Do not invoke a substituted any/all/getattr/type before declining it.
    for function, name, native in _NATIVE_SCALAR_READ_BUILTINS:
        if (
            function.__globals__.get(name, function.__builtins__.get(name))
            is not native
        ):
            return False
    # Read native module dictionaries: a missing attribute must not call a
    # module-level __getattr__ while this new admission decides to decline.
    for name, native in _NATIVE_NUMPY_READ_HELPERS.items():
        if np.__dict__.get(name) is not native:
            return False
    if (
        np is not _NATIVE_SCALAR_NUMPY
        or _NATIVE_SCALAR_ARRAYTYPE is None
        or np.__dict__.get("ndarray") is not _NATIVE_SCALAR_ARRAYTYPE
        or not _NATIVE_SCALAR_TYPES
        or _scalar_symbol is not _NATIVE_SCALAR_SYMBOL
        or not any(dtype is conversion for conversion in _NATIVE_SCALAR_CONVERSIONS)
    ):
        return False
    # Decline unsupported dtypes before paying the native execution checks.
    if not all(
        type(value) is _NATIVE_SCALAR_ARRAYTYPE
        and value.dtype.num in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 23)
        and any(value.dtype.type is scalar_type for scalar_type in _NATIVE_SCALAR_TYPES)
        and value.dtype.metadata is None
        and value.dtype.itemsize <= (16 if value.dtype.kind == "c" else 8)
        for value in values.values()
    ):
        return False
    return not (
        not _NATIVE_READ_TRACE_PROVENANCE
        or any(sys.__dict__.get(name) is not fn for name, fn in _NATIVE_READ_TRACE)
        or any(fn() is not None for _, fn in _NATIVE_READ_TRACE)
        or not native_symbol_reads()
        or any(
            function.__code__ is not code
            for function, code in _NATIVE_SCALAR_READ_CODES
        )
    )


def _indices(recipe: _SymbolRecipe, index_map: dict) -> tuple:
    labels = ()
    if type(index_map) is _IndexBindings:
        resolved = index_map.resolved[recipe.slot]
        if resolved is not None:
            return resolved
        # Missing bindings are raised only after the original value lookup.
        labels, index_map = index_map.labels, index_map.original
    elif type(index_map) is _LabelBindings:
        labels, index_map = index_map.labels, index_map.original
    out = []
    for label, literal in zip(recipe.indices, recipe.literals):
        label = _caller_label(label, labels)
        if label in index_map:
            out.append(index_map[label])
        elif literal is not None:
            out.append(literal)
        else:
            raise KeyError(
                f"Index '{label}' not found in index_map and is not a literal integer. "
                "Pass it via the fixed_indices argument of evaluate_coupling()."
            ) from None
    return tuple(out)


def _scalar_symbol(recipe: _SymbolRecipe, values: dict, index_map: dict):
    # The captured native type guard admits _IndexBindings; mypy cannot narrow it.
    memo = (
        index_map.symbol_values  # type: ignore[attr-defined]
        if recipe.indices and _NATIVE_SCALAR_TYPE(index_map) is _IndexBindings
        else None
    )
    if memo is not None:
        cached = memo[recipe.read_slot]
        if cached is not _UNREAD:
            return cached
    arr = values[recipe.name]
    val = arr[_indices(recipe, index_map)] if recipe.indices else arr
    val = np.asarray(val)
    if val.ndim != 0:
        if val.size == 1:
            val = val.reshape(())
        else:
            raise ValueError(
                f"coupling '{recipe.name}' carries "
                f"{len(recipe.indices)} component index/indices in this "
                f"diagram, so indexing its value left an array of shape "
                f"{np.shape(val)} rather than a scalar.  For a scalar "
                f"field (n_components=1) the vertex indices are stripped "
                f"entirely, so the coupling must be a 0-d value (e.g. "
                f"np.array(1j)), not a rank-{np.ndim(arr)} array."
            )
    value = complex(val) if np.iscomplexobj(arr) else float(val)
    if memo is not None:
        memo[recipe.read_slot] = value
    return value


def _batched_symbol(
    recipe: _SymbolRecipe, values: dict, index_map: dict, n_samples: int
):
    # No-index reshape can own a full sample copy for strided input. Keep its
    # original lifetime; indexed basic reads are only views or native scalars.
    memo = (
        index_map.symbol_values
        if recipe.indices and type(index_map) is _IndexBindings
        else None
    )
    if memo is not None:
        cached = memo[recipe.read_slot]
        if cached is not _UNREAD:
            return cached
    arr = values[recipe.name]
    if not recipe.indices:
        a = np.asarray(arr)
        if a.ndim == 0 or a.shape == (n_samples,):
            value = arr
        elif a.size == n_samples:
            value = a.reshape(n_samples)
        elif a.size == 1:
            value = a.reshape(())
        else:
            raise ValueError(
                f"coupling {recipe.name!r} carries no component index in "
                f"this term, so it must be a scalar per sample, but the "
                f"supplied array has shape {a.shape} (n_samples="
                f"{n_samples}).  A rank-m coupling reaches this branch "
                f"only in a single-component system, where every "
                f"component axis has length 1."
            )
    else:
        idx = _indices(recipe, index_map)
        if arr.ndim == len(idx) + 1:
            value = arr[(slice(None),) + idx]
        elif arr.ndim == len(idx):
            value = arr[idx]
        else:
            raise ValueError(
                f"Symbol {recipe.name!r} has rank {len(idx)} but the supplied "
                f"array has shape {arr.shape}; expected either rank or "
                f"rank+1 (with leading sample axis of length {n_samples})."
            )
    if memo is not None:
        memo[recipe.read_slot] = value
    return value


def _delta(labels: tuple, literals: tuple, index_map: Any):
    witness = ()
    if type(index_map) in (_IndexBindings, _LabelBindings):
        witness, index_map = index_map.labels, index_map.original
    resolved = []
    for label, literal in zip(labels, literals):
        label = _caller_label(label, witness)
        if label in index_map:
            resolved.append(int(index_map[label]))
        elif literal is not None:
            resolved.append(literal)
        else:
            raise KeyError(
                f"KroneckerDelta index {label!r} not found in index_map "
                f"and is not a literal integer. Pass it via the "
                f"fixed_indices argument of evaluate_coupling()."
            ) from None
    return 1.0 if resolved[0] == resolved[1] else 0.0


@dataclass(frozen=True)
class _PlanCore:
    scalar: Callable
    batched: Callable
    node_count: int
    recipes: tuple[_SymbolRecipe, ...]
    symbols: tuple[_SymbolRecipe, ...]
    tokens: tuple
    reuse_indices: bool
    reuse_symbol_reads: bool

    def bind_indices(self, index_map, labels=(), reuse_symbols=False):
        resolved = []
        for recipe in self.recipes:
            indices = []
            for label, literal in zip(recipe.indices, recipe.literals):
                label = _caller_label(label, labels)
                if label in index_map:
                    indices.append(index_map[label])
                elif literal is not None:
                    indices.append(literal)
                else:
                    indices = None
                    break
            resolved.append(None if indices is None else tuple(indices))
        # Custom iterators can yield conversion hooks after base-pin admission.
        inert_assignment = reuse_symbols and all(
            type(k) is str and type(v) is int for k, v in index_map.items()
        )
        memo = [_UNREAD] * len(self.symbols) if inert_assignment else None
        return _IndexBindings(index_map, tuple(resolved), labels, memo)


@dataclass(frozen=True)
class _CouplingPlan:
    expression: Expr
    core: _PlanCore
    labels: tuple[str, ...] = ()

    @property
    def symbols(self):
        # Probe-only caller metadata need not stay allocated in every clone.
        if not self.labels:
            return self.core.symbols
        return tuple(
            _SymbolRecipe(
                r.name,
                tuple(_caller_label(x, self.labels) for x in r.indices),
                r.literals,
                r.slot,
                r.read_slot,
            )
            for r in self.core.symbols
        )

    def scalar(self, expr, values, index_map):
        return self.core.scalar(expr, values, self._bindings(index_map))

    def batched(self, expr, values, index_map, n_samples):
        return self.core.batched(expr, values, self._bindings(index_map), n_samples)

    def _bindings(self, index_map):
        if not self.labels or type(index_map) is _IndexBindings:
            return index_map
        return _LabelBindings(index_map, self.labels)

    def bind_indices(self, index_map, reuse_symbols=False):
        return self.core.bind_indices(index_map, self.labels, reuse_symbols)

    def caller_label(self, label):
        return _caller_label(label, self.labels)

    @property
    def node_count(self):
        return self.core.node_count

    def __reduce__(self):
        # Rebuild the caller's wrapper, sharing no expression or numeric state.
        return compile_coupling_plan, (self.expression,)


def compile_coupling_plan(expression: Expr) -> _CouplingPlan | None:
    """Lower a bounded native tree, returning None for unsupported metadata.

    Preflight never raises the runtime unsupported-node exception. The caller
    then uses the original evaluator, preserving the first error in the tree.
    """
    # Very long literals depend on mutable sys.set_int_max_str_digits limits;
    # keep their conversion at runtime in the original interpreter.
    # Exact native-type checks below validate this heterogeneous node stream.
    nodes: list[Any] = []
    pending: list[tuple[Any, int]] = [(expression, 0)]
    while pending:
        node, depth = pending.pop()
        if depth > 32 or len(nodes) >= 2048:
            return None
        kind = type(node)
        if kind is Rational:
            if type(node.numerator) is not int or type(node.denominator) is not int:
                return None
        elif kind is Symbol:
            if type(node.name) is not str or type(node.indices) is not tuple:
                return None
            if any(
                type(label) is not str or len(label) > 256 for label in node.indices
            ):
                return None
        elif kind is KroneckerDelta:
            if (
                type(node.index1) is not str
                or type(node.index2) is not str
                or len(node.index1) > 256
                or len(node.index2) > 256
            ):
                return None
        elif kind in (Product, Sum):
            children = node.factors if kind is Product else node.terms
            if type(children) is not tuple or len(children) > 128:
                return None
            pending.extend((child, depth + 1) for child in reversed(children))
        elif kind is not ImaginaryUnit:
            return None
        nodes.append(node)
    if len(nodes) < 16:
        return None

    # Cache-only bounds are stricter than compiler admission. Oversized or
    # custom irrelevant metadata still gets the existing per-term compiler.
    token_list: list[tuple] = []
    token: tuple
    shareable = len(nodes) <= 512
    characters = integer_bits = 0
    for node in nodes:
        kind = type(node)
        if kind is Rational:
            token = ("r", node.numerator, node.denominator)
            integer_bits += node.numerator.bit_length() + node.denominator.bit_length()
        elif kind is Symbol:
            token = ("s", node.name, node.indices)
            characters += len(node.name) + sum(len(label) for label in node.indices)
            if (
                type(node.spatial_args) is tuple
                and len(node.spatial_args) <= 128
                and type(node.local) is bool
                and all(type(x) is str for x in node.spatial_args)
            ):
                token += (node.spatial_args, node.local)
                characters += sum(len(label) for label in node.spatial_args)
            else:
                shareable = False
            if len(node.indices) > 128:
                shareable = False
        elif kind is KroneckerDelta:
            token = ("d", node.index1, node.index2)
            characters += len(node.index1) + len(node.index2)
        elif kind is ImaginaryUnit:
            token = ("i",)
        else:
            token = (
                "p" if kind is Product else "a",
                len(node.factors if kind is Product else node.terms),
            )
        token_list.append(token)
    tokens = tuple(token_list)
    shareable = (
        shareable
        and characters <= 16384
        and integer_bits <= 4096
        and _scalar_symbol is _NATIVE_SCALAR_SYMBOL
        and _batched_symbol is _NATIVE_BATCHED_SYMBOL
        and _indices is _NATIVE_INDICES
        and _delta is _NATIVE_DELTA
        and _caller_label is _NATIVE_CALLER_LABEL
    )
    labels: tuple[str, ...] = ()
    if shareable:
        # Exact int parsing/spelling stays attached to literals; only other
        # component labels become slots. First occurrence fixes operand order.
        classified = {}
        witness: list[str] = []

        def label_slot(label):
            if label not in classified:
                try:
                    int(label)
                except ValueError:
                    classified[label] = len(witness)
                    witness.append(label)
                else:
                    classified[label] = label
            return classified[label]

        canonical = []
        for token in tokens:
            if token[0] == "s":
                token = (
                    token[0],
                    token[1],
                    tuple(label_slot(x) for x in token[2]),
                ) + token[3:]
            elif token[0] == "d":
                token = (token[0], *(label_slot(x) for x in token[1:]))
            canonical.append(token)
        labels = tuple(witness)
        core = _shared_core(tuple(canonical))
    else:
        core = _lower_core(tokens)
    plan = _CouplingPlan(expression, core, labels)
    identity = id(plan)
    _PLAN_PROVENANCE[identity] = (
        ref(plan, lambda _: _PLAN_PROVENANCE.pop(identity, None)),
        ref(expression),
        ref(core),
        labels,
    )
    return plan


@lru_cache(maxsize=32)
def _shared_core(tokens):
    # 32 * (<=512 nodes, <=16k characters, <=4096 integer bits), no arrays.
    return _lower_core(tokens)


_CORE_PROVENANCE: WeakKeyDictionary[Callable, tuple] = WeakKeyDictionary()
_PLAN_PROVENANCE: dict[int, tuple] = {}


def native_plan(plan):
    if type(plan) is not _CouplingPlan:
        return False
    if any(
        name in plan.__dict__ or _CouplingPlan.__dict__.get(name) is not method
        for name, method in _NATIVE_PLAN_METHODS.items()
    ):
        return False
    entry = _PLAN_PROVENANCE.get(id(plan))
    return (
        entry is not None
        and entry[0]() is plan
        and entry[1]() is plan.expression
        and entry[2]() is plan.core
        and entry[3] is plan.labels
        and native_core(plan.core)
    )


def native_core(core):
    if type(core) is not _PlanCore or type(core.scalar) is not FunctionType:
        return False
    entry = _CORE_PROVENANCE.get(core.scalar)
    return (
        entry is not None
        and entry[0]() is core
        and entry[1]() is core.batched
        and entry[2] is _NATIVE_SCALAR_SYMBOL
        and entry[3] is _NATIVE_BATCHED_SYMBOL
        and _indices is _NATIVE_INDICES
        and _delta is _NATIVE_DELTA
        and _caller_label is _NATIVE_CALLER_LABEL
        and "bind_indices" not in core.__dict__
        and _PlanCore.__dict__.get("bind_indices") is _NATIVE_CORE_BIND
    )


def _lower_core(tokens):
    recipes = []
    slots = {}
    symbols = {}
    symbol_count = 0
    position = 0

    def lower():
        nonlocal position, symbol_count
        token = tokens[position]
        position += 1
        kind = token[0]
        if kind == "r":
            numerator, denominator = token[1:]

            # Division remains at execution, preserving overflow/error order.
            def scalar(values, index_map):
                return numerator / denominator

            def batched(values, index_map, n_samples):
                return numerator / denominator

        elif kind == "i":

            def scalar(values, index_map):
                return 1j

            def batched(values, index_map, n_samples):
                return 1j

        elif kind == "s":
            name, indices = token[1:3]
            symbol_count += bool(indices)
            slot = slots.get(indices)
            if slot is None:
                slot = len(recipes)
                slots[indices] = slot
                recipes.append(
                    _SymbolRecipe(
                        name,
                        indices,
                        tuple(literal(x, 1) for x in indices),
                        slot,
                        len(symbols),
                    )
                )
            recipe = symbols.setdefault(
                (name, indices),
                _SymbolRecipe(
                    name, indices, recipes[slot].literals, slot, len(symbols)
                ),
            )
            scalar = partial(_scalar_symbol, recipe)
            batched = partial(_batched_symbol, recipe)
        elif kind == "d":
            labels = token[1:]
            literals = tuple(literal(x, 0) for x in labels)

            def scalar(values, index_map):
                return _delta(labels, literals, index_map)

            def batched(values, index_map, n_samples):
                return _delta(labels, literals, index_map)

        else:
            compiled = tuple(lower() for _ in range(token[1]))
            scalar_children = tuple(child[0] for child in compiled)
            batched_children = tuple(child[1] for child in compiled)
            if kind == "p":

                def scalar(values, index_map):
                    total = 1.0
                    for child in scalar_children:
                        total *= child(values, index_map)
                    return total

                def batched(values, index_map, n_samples):
                    total = 1.0
                    for child in batched_children:
                        total = total * child(values, index_map, n_samples)
                    return total

            else:

                def scalar(values, index_map):
                    return sum(child(values, index_map) for child in scalar_children)

                def batched(values, index_map, n_samples):
                    terms = [
                        child(values, index_map, n_samples)
                        for child in batched_children
                    ]
                    if not terms:
                        return 0.0
                    total = terms[0]
                    for child in terms[1:]:
                        total = total + child
                    return total

        return scalar, batched

    def literal(label, offset):
        if type(label) is int:
            return None
        try:
            return int(label) - offset
        except ValueError:
            return None

    scalar_node, batched_node = lower()

    def scalar(expr, values, index_map):
        return scalar_node(values, index_map)

    def batched(expr, values, index_map, n_samples):
        return batched_node(values, index_map, n_samples)

    reuse = (
        symbol_count >= 16
        and symbol_count >= 2 * len(recipes)
        and sum(len(r.indices) for r in recipes) <= 4096
        and all(len(r.indices) <= 128 for r in recipes)
    )
    core = _PlanCore(
        scalar,
        batched,
        len(tokens),
        tuple(recipes),
        tuple(symbols.values()),
        tokens,
        reuse,
        symbol_count >= 16 and symbol_count >= 2 * len(symbols) and len(tokens) <= 512,
    )
    _CORE_PROVENANCE[scalar] = (
        ref(core),
        ref(batched),
        _scalar_symbol,
        _batched_symbol,
    )
    return core


_NATIVE_INDICES = _indices
_NATIVE_SCALAR_SYMBOL = _scalar_symbol
_NATIVE_BATCHED_SYMBOL = _batched_symbol
_NATIVE_DELTA = _delta
_NATIVE_CALLER_LABEL = _caller_label
_NATIVE_SCALAR_TYPE = _IndexBindings.__class__
_NATIVE_SCALAR_MODULETYPE = ModuleType
_NATIVE_SCALAR_DICTTYPE: type[dict] = {}.__class__
_NATIVE_SCALAR_READ_CODES = tuple(
    (function, function.__code__)
    for function in (
        _scalar_symbol,
        _batched_symbol,
        _indices,
        _delta,
        _caller_label,
    )
)
_NATIVE_SCALAR_READ_BUILTINS = tuple(
    (function, name, function.__builtins__[name])
    for function, names in (
        (native_scalar_symbol_reads, ("any", "all", "getattr", "type")),
        (native_symbol_reads, ("type", "len", "any", "getattr")),
        (_scalar_symbol, ("type", "len", "float", "complex", "ValueError")),
        (_indices, ("type", "tuple", "zip", "KeyError")),
        (_delta, ("type", "zip", "int", "KeyError")),
        (_caller_label, ("type",)),
        (_lower_core, ("sum",)),
    )
    for name in names
)
_NATIVE_SCALAR_READ_BUILTINS_PROVENANCE = all(
    (
        type(native) is BuiltinFunctionType
        or (type(native) is type and native.__flags__ & (1 << 8))
    )
    and native.__module__ == "builtins"
    and native.__name__ == name
    for _, name, native in _NATIVE_SCALAR_READ_BUILTINS
)
if not _NATIVE_SCALAR_READ_BUILTINS_PROVENANCE:
    _NATIVE_SCALAR_READ_BUILTINS = ()
_NATIVE_SCALAR_CONVERSIONS = (
    (
        _scalar_symbol.__builtins__["float"],
        _scalar_symbol.__builtins__["complex"],
    )
    if _NATIVE_SCALAR_READ_BUILTINS_PROVENANCE
    else ()
)
_NATIVE_SCALAR_NUMPY = np
_NATIVE_SCALAR_ARRAYTYPE = (
    np.ndarray if native_numpy_c_type(np.ndarray, "ndarray") else None
)
_scalar_dtype_class = np.__dict__.get("dtype")
_scalar_dtype_meta = _NATIVE_SCALAR_TYPE(_scalar_dtype_class)
_NATIVE_SCALAR_DTYPE_NUM = (
    _scalar_dtype_class.__dict__.get("num")
    if _scalar_dtype_class is not None
    and (
        _scalar_dtype_meta is _NATIVE_SCALAR_TYPE
        or native_numpy_c_type(_scalar_dtype_meta, "_DTypeMeta")
    )
    and _scalar_dtype_class.__flags__ & (1 << 8)
    and _scalar_dtype_class.__module__ == "numpy"
    and _scalar_dtype_class.__name__ == "dtype"
    else None
)
if not (
    _NATIVE_SCALAR_DTYPE_NUM is not None
    and _NATIVE_SCALAR_TYPE(_NATIVE_SCALAR_DTYPE_NUM) is MemberDescriptorType
    and _NATIVE_SCALAR_DTYPE_NUM.__objclass__ is _scalar_dtype_class
    and _NATIVE_SCALAR_DTYPE_NUM.__name__ == "num"
):
    _NATIVE_SCALAR_DTYPE_NUM = None
del _scalar_dtype_class, _scalar_dtype_meta
_NATIVE_SCALAR_TYPES = tuple(
    getattr(np, name, None)
    for name in (
        "bool_",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    )
)
if not all(
    type(kind) is type and native_numpy_c_type(kind, kind.__name__)
    for kind in _NATIVE_SCALAR_TYPES
):
    _NATIVE_SCALAR_TYPES = ()
_NATIVE_READ_TRACE = tuple(
    (name, getattr(sys, name)) for name in ("gettrace", "getprofile")
)
_NATIVE_READ_TRACE_PROVENANCE = all(
    type(fn) is BuiltinFunctionType and fn.__module__ == "sys" and fn.__name__ == name
    for name, fn in _NATIVE_READ_TRACE
)
if not _NATIVE_READ_TRACE_PROVENANCE:
    _NATIVE_READ_TRACE = ()
_NATIVE_CORE_BIND = _PlanCore.bind_indices
_NATIVE_PLAN_METHODS = {
    name: _CouplingPlan.__dict__[name]
    for name in (
        "scalar",
        "batched",
        "_bindings",
        "bind_indices",
        "caller_label",
        "symbols",
    )
}
