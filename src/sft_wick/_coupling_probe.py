"""Sufficient proof that the native sample-zero scalar probe will succeed.

This is admission, not a replacement scalar evaluator. Declining preserves
scalar-first errors; a later batch fallback still evaluates the exact scalar.
"""

from math import isfinite, prod
from types import FunctionType
from typing import Any
import sys
import warnings
import re

import numpy as np

from ._coupling_plan import (
    _CouplingPlan,
    native_bindings,
    native_plan,
    native_symbol_reads,
    native_numpy_c_type,
)
from .expressions import Product, Propagator, Rational, Sum

_LIMIT = 1e100
_MAX_SCAN_ELEMENTS = 4096
_MAX_SCALAR_ELEMENTS = 4096
_WARNING_FUNCTIONS = (
    "showwarning",
    "_showwarnmsg",
    "_showwarnmsg_impl",
    "formatwarning",
    "_formatwarnmsg",
    "_formatwarnmsg_impl",
    "_get_context",
    "_get_filters",
)
# Python3.14 reexports its Python warning implementation from _py_warnings.
_WARNING_IMPL = sys.modules.get("_py_warnings", warnings)
_NATIVE_WARNINGS = {
    name: (
        fn
        if (
            type(fn) is FunctionType
            and fn.__module__ == _WARNING_IMPL.__name__
            and fn.__code__.co_filename
            in (_WARNING_IMPL.__file__, f"<frozen {_WARNING_IMPL.__name__}>")
        )
        else None
    )
    for name in _WARNING_FUNCTIONS
    if hasattr(_WARNING_IMPL, name)
    for fn in (getattr(_WARNING_IMPL, name, None),)
}
_NATIVE_WARNING_MESSAGE = _WARNING_IMPL.WarningMessage
_NATIVE_WARNING_CONTEXT = getattr(_WARNING_IMPL, "_global_context", None)
_PATTERN_TYPE = type(re.compile(""))
_NATIVE_ISFINITE: Any = np.isfinite
_NATIVE_ISFINITE_PROVENANCE = (
    native_numpy_c_type(type(_NATIVE_ISFINITE), "ufunc")
    and _NATIVE_ISFINITE.__name__ == "isfinite"
)
if not _NATIVE_ISFINITE_PROVENANCE:
    _NATIVE_ISFINITE = None
_FLOAT64_DTYPE: np.dtype = np.dtype("float64")


def _inert_float_errors():
    modes = np.geterr()
    if modes["under"] != "ignore" or any(
        x in ("call", "log", "print") for x in modes.values()
    ):
        return False
    if "warn" not in modes.values():
        return True
    # A hook may mutate tensors/pins and raise NotImplementedError before the
    # lazy scalar fallback. Retain the pre-batch seed whenever hooks exist.
    if (
        getattr(sys.flags, "context_aware_warnings", False)
        or getattr(warnings, "WarningMessage", None) is not _NATIVE_WARNING_MESSAGE
        or sys.stderr is not sys.__stderr__
    ):
        return False
    if any(
        native is None or getattr(warnings, name, None) is not native
        for name, native in _NATIVE_WARNINGS.items()
    ):
        return False
    if _NATIVE_WARNING_CONTEXT is not None and (
        _WARNING_IMPL._global_context is not _NATIVE_WARNING_CONTEXT
        or _NATIVE_WARNING_CONTEXT.log is not None
        or _WARNING_IMPL._use_context
    ):
        return False
    filters = getattr(warnings, "filters", None)
    return (
        type(filters) is list
        and len(filters) <= 64
        and all(
            type(item) is tuple
            and len(item) == 5
            and type(item[0]) is str
            and (item[1] is None or type(item[1]) in (str, _PATTERN_TYPE))
            and type(item[2]) is type
            and (item[3] is None or type(item[3]) in (str, _PATTERN_TYPE))
            and type(item[4]) is int
            for item in filters
        )
    )


def known_scalar_shape(term, values, fixed_indices, n_samples):
    from . import perturbation as pt

    if (
        _inert_float_errors is not _NATIVE_INERT_FLOAT_ERRORS
        or not _NATIVE_ISFINITE_PROVENANCE
        or getattr(np, "isfinite", None) is not _NATIVE_ISFINITE
        or type(term) is not pt.DiagramTerm
        or type(n_samples) is not int
        or n_samples < 1
    ):
        return None
    expr = term.coupling_sum
    if not (
        (
            type(expr) is Product
            and type(expr.factors) is tuple
            and len(expr.factors) >= 4
        )
        or (type(expr) is Sum and type(expr.terms) is tuple and len(expr.terms) >= 4)
    ):
        return None
    if not native_symbol_reads():
        return None
    for name, native in pt._NATIVE_COUPLING_METHODS.items():
        if name in term.__dict__ or pt.DiagramTerm.__dict__.get(name) is not native:
            # The plan cached_property is normally present in __dict__.
            if (
                name != "_coupling_execution_plan"
                or pt.DiagramTerm.__dict__.get(name) is not native
            ):
                return None
    if pt.compile_coupling_plan is not pt._NATIVE_COMPILE_PLAN:
        return None
    pins = {} if fixed_indices is None else fixed_indices
    if not native_bindings(values, pins):
        return None
    summation = term.summation_indices
    if type(summation) is not tuple or any(
        type(x) is not tuple
        or len(x) != 2
        or type(x[0]) is not str
        or type(x[1]) is not int
        or x[1] <= 0
        for x in summation
    ):
        return None
    if len({x[0] for x in summation}) != len(summation):
        return None
    if type(term.propagators) is not tuple or any(
        type(p) is not Propagator
        or any(
            x is not None and type(x) is not str for x in (p.index_left, p.index_right)
        )
        for p in term.propagators
    ):
        return None
    plan = term._coupling_execution_plan
    if (
        type(plan) is not _CouplingPlan
        or plan.expression is not expr
        or not native_plan(plan)
    ):
        return None
    prefactor = term.rational_prefactor
    if (
        type(prefactor) is not Rational
        or type(prefactor.numerator) is not int
        or type(prefactor.denominator) is not int
        or type(term.n_response) is not int
    ):
        return None
    try:
        pref = prefactor.numerator / prefactor.denominator
    except (OverflowError, ZeroDivisionError):
        return None
    if not isfinite(pref) or abs(pref) > _LIMIT or not _inert_float_errors():
        return None
    complex_output = term.n_response % 4 in (1, 3) or any(
        a.dtype.kind == "c" for a in values.values()
    )
    if not complex_output and any(token[0] == "i" for token in plan.core.tokens):
        return None
    prop_indices = term.propagator_indices
    coupling_indices = term.coupling_only_indices
    inner_count = prod(dim for _, dim in coupling_indices)
    if (
        len(prop_indices) > 32
        or prod(dim for _, dim in prop_indices) > _MAX_SCALAR_ELEMENTS
        or len(coupling_indices) > 32
        or prod(dim for _, dim in summation) > sys.maxsize
    ):
        return None
    intervals = {key: (value, value) for key, value in pins.items()}
    intervals.update({key: (0, dim - 1) for key, dim in summation})
    magnitudes = {}
    scanned_elements = 0
    for recipe in plan.symbols:
        if recipe.name not in values:
            return None
        arr = values[recipe.name]
        if (
            arr.dtype.itemsize > (16 if arr.dtype.kind == "c" else 8)
            or arr.size == 0
            or arr.ndim != len(recipe.indices)
        ):
            # No-index scalar accepts a size-one array, including singleton axes.
            if recipe.indices or arr.size != 1 or arr.dtype.itemsize > 8:
                return None
        for axis, (label, literal) in enumerate(zip(recipe.indices, recipe.literals)):
            interval = intervals.get(label)
            if interval is None:
                if literal is None:
                    return None
                interval = (literal, literal)
            if interval[0] < -arr.shape[axis] or interval[1] >= arr.shape[axis]:
                return None
        if recipe.name not in magnitudes:
            # A large tensor may have only a tiny pinned slice accessed by the
            # scalar probe. Bound both scan work and its temporary allocations.
            scanned_elements += arr.size
            if scanned_elements > _MAX_SCAN_ELEMENTS:
                return None
            # Cast before abs: abs(int64_min) in its original dtype overflows.
            # Test in the original dtype before widening: casting a signaling
            # NaN can raise/warn even though scalar float conversion would not.
            if not np.isfinite(arr.real).all() or not np.isfinite(arr.imag).all():
                return None
            real = arr.real.astype(_FLOAT64_DTYPE, copy=False)
            magnitude = float(abs(real).max())
            if arr.dtype.kind == "c":
                imag = arr.imag.astype(_FLOAT64_DTYPE, copy=False)
                magnitude = max(magnitude, float(abs(imag).max()))
            if magnitude > _LIMIT / 2:
                return None
            magnitudes[recipe.name] = max(1.0, 2 * magnitude)
    bounds = []
    for token in reversed(plan.core.tokens):
        kind = token[0]
        if kind == "s":
            bound = magnitudes[token[1]]
        elif kind == "r":
            try:
                bound = max(1.0, abs(token[1] / token[2]))
            except (OverflowError, ZeroDivisionError):
                return None
        elif kind == "d":
            for label in token[1:]:
                label = plan.caller_label(label)
                if label not in intervals:
                    try:
                        int(label)
                    except ValueError:
                        return None
            bound = 1.0
        elif kind == "i":
            bound = 1.0
        else:
            children = [bounds.pop() for _ in range(token[1])]
            bound = 1.0
            if kind == "p":
                for child in children:
                    bound *= 2 * child
                    if bound > _LIMIT:
                        return None
            else:
                bound = max(1.0, 2 * sum(children))
        if not isfinite(bound) or bound > _LIMIT:
            return None
        bounds.append(bound)
    if bounds[0] * max(1.0, abs(pref)) * inner_count > _LIMIT:
        return None
    return tuple(dim for _, dim in prop_indices)


_NATIVE_INERT_FLOAT_ERRORS = _inert_float_errors
