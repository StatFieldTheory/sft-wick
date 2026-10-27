"""Content identities for the native, callback-free propagator families.

This is deliberately a cache admission check, not a new specification
validator.  Unknown objects return ``None`` and keep the ordinary compute
path.  Reading an identity never calls a user callback, getter, repr or eq.
"""

from __future__ import annotations

import ast
import dataclasses
from functools import cached_property
import math
from pathlib import Path
import reprlib
from types import FunctionType, GetSetDescriptorType, MemberDescriptorType
from typing import Any

import numpy as np

from . import closed_forms, specs
from . import system as system_module
from ._native_protocol import native_protocol_unchanged
from .system import System

# Literal roles, tags and fields do not depend on mutable class metadata.
_SPEC_ROLES = (
    (specs, "FieldSpec", specs.FieldSpec, "field", ("name", "n_components")),
    (
        specs,
        "DiagonalA",
        specs.DiagonalA,
        "diagonal-a",
        ("gamma", "t_max_cache", "n_grid_cache", "t_min_cache"),
    ),
    (
        specs,
        "GaussianNoise",
        specs.GaussianNoise,
        "gaussian-noise",
        ("kappa2", "sigma2"),
    ),
    (
        specs,
        "SeparableTranslation",
        specs.SeparableTranslation,
        "separable-translation",
        ("temporal", "spatial"),
    ),
    (
        specs,
        "SeparableRotation",
        specs.SeparableRotation,
        "separable-rotation",
        ("temporal", "angular"),
    ),
    (
        specs,
        "ExponentialTemporal",
        specs.ExponentialTemporal,
        "exponential-temporal",
        ("lam", "sigma_t", "is_even", "has_diagonal_cusp"),
    ),
    (
        specs,
        "GaussianTemporal",
        specs.GaussianTemporal,
        "gaussian-temporal",
        ("lam", "sigma_t", "is_even", "has_diagonal_cusp"),
    ),
    (
        specs,
        "ExponentialSpatial",
        specs.ExponentialSpatial,
        "exponential-spatial",
        ("sigma_x",),
    ),
    (specs, "GaussianSpatial", specs.GaussianSpatial, "gaussian-spatial", ("sigma_x",)),
    (specs, "LegendreAngular", specs.LegendreAngular, "legendre-angular", ("coeffs",)),
    (
        specs,
        "ConstantImpulse",
        specs.ConstantImpulse,
        "constant-impulse",
        ("amplitude",),
    ),
    (
        closed_forms,
        "ClosedFormC",
        closed_forms.ClosedFormC,
        "closed-form-c",
        (
            "gamma",
            "lam",
            "sigma_t",
            "spatial",
            "sigma2",
            "t_min",
            "separable_translation",
            "has_diagonal_kink",
        ),
    ),
)
_SYSTEM_FIELDS = (
    "field",
    "linear",
    "noise",
    "vertices",
    "nonlocal_vertices",
    "t_min",
    "explicit_R",
)
_CLASS_ROLES = (
    (system_module, "System", System),
    *((module, name, cls) for module, name, cls, _tag, _fields in _SPEC_ROLES),
    (specs, "LinearOp", specs.LinearOp),
    (specs, "Kappa2", specs.Kappa2),
    (specs, "Sigma2", specs.Sigma2),
    (specs, "_StaticIsoR", specs._StaticIsoR),
    (specs, "_StaticMatR", specs._StaticMatR),
    (specs, "_SeparableTranslationKappa2", specs._SeparableTranslationKappa2),
    (specs, "_SeparableRotationKappa2", specs._SeparableRotationKappa2),
    (specs, "_ConstantImpulseIso", specs._ConstantImpulseIso),
    (specs, "_ConstantImpulseMat", specs._ConstantImpulseMat),
)
# copyreg caches the native layout as a list. These literal layouts never
# depend on mutable class display names or __slots__ metadata.
_SLOT_ROLES = (
    (specs._StaticIsoR, ("gamma",)),
    (specs._StaticMatR, ("gamma_arr", "_n")),
    (specs._SeparableTranslationKappa2, ("temporal", "spatial", "_n")),
    (specs._SeparableRotationKappa2, ("temporal", "angular", "_n")),
    (specs._ConstantImpulseIso, ("amp", "_n")),
    (specs._ConstantImpulseMat, ("mat",)),
)
_MODULE_ROLES = (
    (specs, "_is_scalar_pair", "sft_wick.workflow.specs", "specs.py"),
    *(
        (closed_forms, name, "sft_wick.workflow.closed_forms", "closed_forms.py")
        for name in (
            "_psi",
            "_dexp",
            "ou_exponential_phi",
            "ou_white_psi",
            "_resolve_positions",
            "_separation",
            "builtin_closed_form_for",
        )
    ),
)
_GENERATED_METHODS = frozenset(
    ("__init__", "__repr__", "__eq__", "__hash__", "__setattr__", "__delattr__")
)


def _slot_cache_native(cls: type, value: Any) -> bool:
    expected = next((names for native, names in _SLOT_ROLES if cls is native), ())
    return (
        type(value) is list
        and len(value) == len(expected)
        and all(
            type(item) is str and item == name for item, name in zip(value, expected)
        )
    )


def _functions(value: Any) -> tuple[FunctionType, ...] | None:
    """Exact native wrapper inspection; never test a user's truth value."""
    kind = type(value)
    if kind is FunctionType:
        return (value,)
    if kind is staticmethod or kind is classmethod:
        fn = value.__func__
        return (fn,) if type(fn) is FunctionType else None
    if kind is property:
        fns = tuple(fn for fn in (value.fget, value.fset, value.fdel) if fn is not None)
        return fns if all(type(fn) is FunctionType for fn in fns) else None
    if kind is cached_property:
        state = vars(value)
        fn = state.get("func")
        return (fn,) if type(fn) is FunctionType else None
    if callable(value):
        return None
    if not any(
        kind is native
        for native in (
            str,
            tuple,
            dict,
            int,
            float,
            bool,
            type(None),
            GetSetDescriptorType,
            MemberDescriptorType,
            dataclasses._DataclassParams,
        )
    ):
        return None
    return ()


def _source_role(node: ast.FunctionDef | ast.AsyncFunctionDef) -> type:
    names = [
        decorator.id for decorator in node.decorator_list if type(decorator) is ast.Name
    ]
    for name, role in (
        ("property", property),
        ("cached_property", cached_property),
        ("staticmethod", staticmethod),
        ("classmethod", classmethod),
    ):
        if name in names:
            return role
    return FunctionType


def _source_function(
    fn: FunctionType,
    module: str,
    path: Path,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    qualname: str,
) -> bool:
    if (
        type(fn.__module__) is not str
        or fn.__module__ != module
        or type(fn.__qualname__) is not str
    ):
        return False
    code = fn.__code__
    first_line = min(
        [node.lineno, *(decorator.lineno for decorator in node.decorator_list)]
    )
    return (
        code.co_filename == str(path)
        and code.co_name == node.name
        and code.co_firstlineno == first_line
        and fn.__qualname__ == qualname
    )


def _generated_function(
    fn: FunctionType, name: str, module: str, class_name: str
) -> bool:
    if name == "__replace__":
        original = vars(dataclasses).get("_replace")
        return fn is original and fn.__code__.co_filename == dataclasses.__file__
    if (
        name not in _GENERATED_METHODS
        or type(fn.__module__) is not str
        or fn.__module__ != module
        or type(fn.__name__) is not str
        or type(fn.__qualname__) is not str
    ):
        return False
    if fn.__name__ != name or fn.__qualname__ != class_name + "." + name:
        return False
    code = fn.__code__
    if code.co_filename == "<string>" and code.co_name == name:
        return True
    if name != "__repr__" or code.co_filename not in (
        dataclasses.__file__,
        reprlib.__file__,
    ):
        return False
    wrapped = vars(fn).get("__wrapped__")
    return (
        type(wrapped) is FunctionType
        and type(wrapped.__module__) is str
        and wrapped.__module__ == module
        and wrapped.__code__.co_filename == "<string>"
        and wrapped.__code__.co_name == "__repr__"
    )


def _native_function_metadata(fn: FunctionType) -> bool:
    """Decline unknown keyword metadata before snapshot iteration."""
    kwdefaults = fn.__kwdefaults__
    return kwdefaults is None or (
        type(kwdefaults) is dict and all(type(key) is str for key in kwdefaults)
    )


def _origins_native() -> bool:
    """Reject foreign replacements already present at the first lazy import."""
    if not all(
        native_protocol_unchanged(vars(module).get("_NATIVE_CALLABLE_PROTOCOL"))
        for module in (specs, system_module, closed_forms)
    ):
        return False
    base = Path(__file__).resolve().parent
    try:
        trees = {
            name: ast.parse((base / (name + ".py")).read_text())
            for name in ("specs", "system", "closed_forms")
        }
        for module, class_name, cls in _CLASS_ROLES:
            if type(cls) is not type:
                return False
            short = (
                "system"
                if module is system_module
                else "closed_forms" if module is closed_forms else "specs"
            )
            module_name = "sft_wick.workflow." + short
            if vars(module).get(class_name) is not cls:
                return False
            nodes = [
                node
                for node in trees[short].body
                if type(node) is ast.ClassDef and node.name == class_name
            ]
            if len(nodes) != 1:
                return False
            methods = {
                node.name: node
                for node in nodes[0].body
                if type(node) is ast.FunctionDef or type(node) is ast.AsyncFunctionDef
            }
            namespace = vars(cls)
            for name, node in methods.items():
                value = namespace.get(name)
                fns = _functions(value)
                if (
                    type(value) is not _source_role(node)
                    or fns is None
                    or len(fns) != 1
                ):
                    return False
                if not _source_function(
                    fns[0],
                    module_name,
                    base / (short + ".py"),
                    node,
                    class_name + "." + name,
                ):
                    return False
                if type(value) is cached_property:
                    attrname = vars(value).get("attrname")
                    if type(attrname) is not str or attrname != name:
                        return False
            for name, value in namespace.items():
                # copyreg lazily stores this cache during normal pickling,
                # including before the first identity-module import.
                if name == "__slotnames__":
                    if not _slot_cache_native(cls, value):
                        return False
                    continue
                fns = _functions(value)
                if fns is None or any(not _native_function_metadata(fn) for fn in fns):
                    return False
                if name in methods:
                    continue
                for fn in fns:
                    if name in ("__annotate__", "__annotate_func__"):
                        if (
                            type(fn.__module__) is not str
                            or fn.__module__ != module_name
                            or fn.__code__.co_filename != str(base / (short + ".py"))
                        ):
                            return False
                    elif not _generated_function(fn, name, module_name, class_name):
                        return False
        for module, name, module_name, filename in _MODULE_ROLES:
            fn = vars(module).get(name)
            nodes = [
                node
                for node in trees[filename[:-3]].body
                if type(node) is ast.FunctionDef and node.name == name
            ]
            if (
                type(fn) is not FunctionType
                or not _native_function_metadata(fn)
                or len(nodes) != 1
                or not _source_function(
                    fn, module_name, base / filename, nodes[0], name
                )
            ):
                return False
        return True
    except (OSError, SyntaxError):
        return False


def _function_state(fn: FunctionType) -> tuple:
    return (
        fn,
        fn.__code__,
        fn.__defaults__,
        None if fn.__kwdefaults__ is None else tuple(fn.__kwdefaults__.items()),
    )


_ORIGIN_NATIVE = _origins_native()
_PROTOCOLS = (
    tuple(
        (
            cls,
            {
                name: value
                for name, value in vars(cls).items()
                if name != "__slotnames__"
            },
        )
        for _module, _name, cls in _CLASS_ROLES
    )
    if _ORIGIN_NATIVE
    else ()
)
_BINDINGS = (
    (
        *_CLASS_ROLES,
        *(
            (module, name, vars(module).get(name))
            for module, name, _modname, _file in _MODULE_ROLES
        ),
    )
    if _ORIGIN_NATIVE
    else ()
)
_FUNCTION_STATES = (
    (
        tuple(
            _function_state(fn)
            for _cls, namespace in _PROTOCOLS
            for value in namespace.values()
            for fn in (_functions(value) or ())
        )
        + tuple(
            _function_state(value)
            for _module, _name, value in _BINDINGS
            if type(value) is FunctionType
        )
    )
    if _ORIGIN_NATIVE
    else ()
)
_DESCRIPTOR_STATES = tuple(
    (
        (value, tuple((name, vars(value).get(name)) for name in ("func", "attrname")))
        if type(value) is cached_property
        else (
            (
                value,
                tuple(
                    (name, getattr(value, name)) for name in ("fget", "fset", "fdel")
                ),
            )
            if type(value) is property
            else (value, (("__func__", value.__func__),))
        )
    )
    for _cls, namespace in _PROTOCOLS
    for value in namespace.values()
    if any(
        type(value) is kind
        for kind in (cached_property, property, staticmethod, classmethod)
    )
)
_NUMERIC_SCALARS = tuple(
    dtype.type
    for dtype in (
        np.dtype(code)
        for code in (
            "?",
            "b",
            "B",
            "h",
            "H",
            "i",
            "I",
            "l",
            "L",
            "q",
            "Q",
            "e",
            "f",
            "d",
            "g",
            "F",
            "D",
            "G",
        )
    )
)


def _protocols_unchanged() -> bool:
    if not _ORIGIN_NATIVE:
        return False
    if not all(
        native_protocol_unchanged(vars(module).get("_NATIVE_CALLABLE_PROTOCOL"))
        for module in (specs, system_module, closed_forms)
    ):
        return False
    if any(vars(module).get(name) is not value for module, name, value in _BINDINGS):
        return False
    for cls, namespace in _PROTOCOLS:
        current = vars(cls)
        # copyreg may add its native layout cache during ordinary pickling.
        extra = set(current) - set(namespace)
        if extra and not (
            extra == {"__slotnames__"}
            and _slot_cache_native(cls, current["__slotnames__"])
        ):
            return False
        if any(current.get(name) is not value for name, value in namespace.items()):
            return False
    for descriptor, attributes in _DESCRIPTOR_STATES:
        for name, original in attributes:
            current = (
                vars(descriptor).get(name)
                if type(descriptor) is cached_property
                else getattr(descriptor, name)
            )
            if current is not original:
                return False
    for fn, code, defaults, kwdefaults in _FUNCTION_STATES:
        if fn.__code__ is not code or fn.__defaults__ is not defaults:
            return False
        if not _native_function_metadata(fn):
            return False
        current_kw = fn.__kwdefaults__
        if kwdefaults is None:
            if current_kw is not None:
                return False
        elif (
            type(current_kw) is not dict
            or len(current_kw) != len(kwdefaults)
            or any(
                type(key) is not str or current_kw.get(key) is not value
                for key, value in kwdefaults
            )
        ):
            return False
    return True


class _Ineligible(Exception):
    """Private control flow; bypass caching without changing validation."""


def _native_state(value: Any, field_names: tuple[str, ...]) -> dict:
    state = object.__getattribute__(value, "__dict__")
    if type(state) is not dict or len(state) != len(field_names):
        raise _Ineligible
    if any(type(key) is not str for key in state) or set(state) != set(field_names):
        raise _Ineligible
    return state


def _encode(value: Any, active: set[int]) -> tuple:
    kind = type(value)
    if value is None:
        return ("none",)
    if kind is bool:
        return ("bool", value)
    if kind is int:
        return ("int", value)
    if kind is str:
        return ("str", value)
    if kind is float:
        if not math.isfinite(value):
            raise _Ineligible
        return ("float", value.hex())
    if kind is complex:
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise _Ineligible
        return ("complex", value.real.hex(), value.imag.hex())
    if any(kind is scalar for scalar in _NUMERIC_SCALARS):
        if not np.isfinite(value):
            raise _Ineligible
        return ("numpy-scalar", value.dtype.str, value.dtype.num, value.tobytes().hex())

    if kind is np.ndarray:
        dtype = value.dtype
        if (
            dtype.kind not in "biufc"
            or dtype.fields is not None
            or dtype.subdtype is not None
            or dtype.metadata is not None
            or not np.all(np.isfinite(value))
        ):
            raise _Ineligible
        return (
            "ndarray",
            dtype.str,
            dtype.num,
            value.shape,
            value.tobytes(order="C").hex(),
        )

    spec_role = next((role for role in _SPEC_ROLES if kind is role[2]), None)
    if kind is not list and kind is not tuple and spec_role is None:
        raise _Ineligible
    ident = id(value)
    if ident in active:
        raise _Ineligible
    active.add(ident)
    try:
        if kind is list or kind is tuple:
            return (
                "list" if kind is list else "tuple",
                tuple(_encode(v, active) for v in value),
            )
        if spec_role is None:
            raise _Ineligible
        field_names = spec_role[4]
        state = _native_state(value, field_names)
        return (
            "native-spec",
            spec_role[3],
            tuple((name, _encode(state[name], active)) for name in field_names),
        )
    finally:
        active.remove(ident)


def native_propagator_identity(system: Any) -> tuple | None:
    """Return a complete version-3 native identity, or bypass with ``None``.

    Only constant ``DiagonalA`` and the package's native separable kernels
    (optionally ``ConstantImpulse``) are admitted.  Interaction vertices and
    the field's symbolic name do not determine R/C and are excluded.  Full
    array contents are retained, independently of NumPy's display options.
    Mutable native arrays are read anew for each build.  Concurrent mutation
    during a build is outside the frozen specification contract.
    """
    if type(system) is not System or not _protocols_unchanged():
        return None
    try:
        state = object.__getattribute__(system, "__dict__")
        if type(state) is not dict or any(type(key) is not str for key in state):
            return None
        extra = set(state) - set(_SYSTEM_FIELDS)
        if extra not in (set(), {"_effective_linear"}):
            return None
        if any(name not in state for name in _SYSTEM_FIELDS):
            return None
        linear = state["linear"]
        if state["explicit_R"] is not None or type(linear) is not specs.DiagonalA:
            return None
        if "_effective_linear" in state and state["_effective_linear"] is not linear:
            return None
        field = state["field"]
        if type(field) is not specs.FieldSpec:
            return None
        field_state = _native_state(field, ("name", "n_components"))
        noise = state["noise"]
        if type(noise) is not specs.GaussianNoise:
            return None
        noise_state = _native_state(noise, ("kappa2", "sigma2"))
        if not any(
            type(noise_state["kappa2"]) is cls
            for cls in (specs.SeparableTranslation, specs.SeparableRotation)
        ):
            return None
        sigma2 = noise_state["sigma2"]
        if sigma2 is not None and type(sigma2) is not specs.ConstantImpulse:
            return None
        # Encode the exact raw inputs, without running is_iso_R or a getter.
        # The guarded native lowering determines iso_R from these same rates.
        active: set[int] = set()
        return (
            "sft-wick-native-propagator",
            "3",
            ("n_components", _encode(field_state["n_components"], active)),
            ("t_min", _encode(state["t_min"], active)),
            ("explicit_R", ("none",)),
            ("linear", _encode(linear, active)),
            ("noise", _encode(noise, active)),
        )
    except (_Ineligible, RecursionError):
        return None


def native_closed_form_identity(c_fn: Any) -> tuple | None:
    """Identity for no form or a guarded native built-in resolved form.

    The caller must additionally require that the original request is
    ``None`` or the exact native string ``'auto'``.  A user-provided
    ``ClosedFormC`` is still a callback choice rather than an inferred form.
    """
    if not _protocols_unchanged():
        return None
    if c_fn is None:
        return ("none",)
    if type(c_fn) is not closed_forms.ClosedFormC:
        return None
    try:
        return _encode(c_fn, set())
    except (_Ineligible, RecursionError):
        return None
