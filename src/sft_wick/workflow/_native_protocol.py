"""Definition-time callable witnesses for native propagator cache admission.

Only object identities are compared. No code constants, unknown values, user
equality, or annotations are evaluated. This guards callables and descriptor
metadata; it does not freeze arbitrary Python module globals.
"""

from __future__ import annotations

from functools import cached_property
from types import CodeType, FunctionType
from typing import NamedTuple, TypeGuard, cast

_MISSING = object()


class _FunctionWitness(NamedTuple):
    function: FunctionType
    code: CodeType
    defaults: object
    kwdefaults: tuple[tuple[str, object], ...] | None
    attributes: tuple[tuple[str, object], ...]
    closure: tuple[object, ...] | None


class _NativeProtocol(NamedTuple):
    namespace: dict[str, object]
    bindings: tuple[tuple[str, object], ...]
    members: tuple[tuple[type, str, object], ...]
    descriptors: tuple[tuple[cached_property, object, object], ...]
    functions: tuple[_FunctionWitness, ...]


def _plain_dict(value: object) -> TypeGuard[dict[str, object]]:
    return type(value) is dict and all(type(key) is str for key in value)


def _member_functions(value: object) -> tuple[FunctionType, ...] | None:
    if type(value) is FunctionType:
        return (value,)
    if type(value) is staticmethod or type(value) is classmethod:
        function = value.__func__
        return (function,) if type(function) is FunctionType else None
    if type(value) is property:
        functions = tuple(
            function
            for function in (value.fget, value.fset, value.fdel)
            if function is not None
        )
        return (
            cast(tuple[FunctionType, ...], functions)
            if all(type(fn) is FunctionType for fn in functions)
            else None
        )
    if type(value) is cached_property:
        state = object.__getattribute__(value, "__dict__")
        if not _plain_dict(state):
            return None
        cached_function = state.get("func", _MISSING)
        return (cached_function,) if type(cached_function) is FunctionType else None
    return ()


def _closure_contents(function: FunctionType) -> tuple[object, ...] | None:
    if function.__closure__ is None:
        return None
    result = []
    for cell in function.__closure__:
        try:
            result.append(cell.cell_contents)
        except ValueError:
            result.append(_MISSING)
    return tuple(result)


def capture_native_protocol(
    namespace: dict[str, object], class_names: tuple[str, ...]
) -> _NativeProtocol | None:
    """Capture actual definitions before callers can change public callables."""
    if not _plain_dict(namespace):
        return None
    module_name = namespace.get("__name__", _MISSING)
    if type(module_name) is not str:
        return None
    bindings: list[tuple[str, object]] = []
    members: list[tuple[type, str, object]] = []
    descriptors: list[tuple[cached_property, object, object]] = []
    pending: list[FunctionType] = []
    for name in class_names:
        cls = namespace.get(name, _MISSING)
        if type(cls) is not type:
            return None
        bindings.append((name, cls))
        for member_name, value in vars(cls).items():
            functions = _member_functions(value)
            if functions is None:
                return None
            if functions:
                members.append((cls, member_name, value))
                pending.extend(functions)
            if type(value) is cached_property:
                state = object.__getattribute__(value, "__dict__")
                descriptors.append(
                    (
                        value,
                        state.get("func", _MISSING),
                        state.get("attrname", _MISSING),
                    )
                )
    for name, value in namespace.items():
        if (
            type(value) is FunctionType
            and type(value.__module__) is str
            and value.__module__ == module_name
        ):
            bindings.append((name, value))
            pending.append(value)
    witnesses: list[_FunctionWitness] = []
    seen: set[int] = set()
    while pending:
        function = pending.pop()
        if id(function) in seen:
            continue
        seen.add(id(function))
        attributes = object.__getattribute__(function, "__dict__")
        kwdefaults = function.__kwdefaults__
        if not _plain_dict(attributes) or (
            kwdefaults is not None and not _plain_dict(kwdefaults)
        ):
            return None
        closure = _closure_contents(function)
        witnesses.append(
            _FunctionWitness(
                function,
                function.__code__,
                function.__defaults__,
                None if kwdefaults is None else tuple(kwdefaults.items()),
                tuple(attributes.items()),
                closure,
            )
        )
        wrapped = attributes.get("__wrapped__", _MISSING)
        if type(wrapped) is FunctionType:
            pending.append(wrapped)
        if closure is not None:
            pending.extend(value for value in closure if type(value) is FunctionType)
    return _NativeProtocol(
        namespace, tuple(bindings), tuple(members), tuple(descriptors), tuple(witnesses)
    )


def native_protocol_unchanged(witness: _NativeProtocol | None) -> bool:
    """Reject changed public native callables without evaluating their payloads."""
    if type(witness) is not _NativeProtocol or not _plain_dict(witness.namespace):
        return False
    if any(
        witness.namespace.get(name, _MISSING) is not original
        for name, original in witness.bindings
    ):
        return False
    for cls, name, original in witness.members:
        current = vars(cls)
        if any(type(key) is not str for key in current):
            return False
        if current.get(name, _MISSING) is not original:
            return False
    for descriptor, function, attrname in witness.descriptors:
        state = object.__getattribute__(descriptor, "__dict__")
        if not _plain_dict(state):
            return False
        if (
            state.get("func", _MISSING) is not function
            or state.get("attrname", _MISSING) is not attrname
        ):
            return False
    for function, code, defaults, kwdefaults, attributes, closure in witness.functions:
        if function.__code__ is not code or function.__defaults__ is not defaults:
            return False
        current_attributes = object.__getattribute__(function, "__dict__")
        if (
            not _plain_dict(current_attributes)
            or len(current_attributes) != len(attributes)
            or any(
                current_attributes.get(name, _MISSING) is not original
                for name, original in attributes
            )
        ):
            return False
        current_keywords = function.__kwdefaults__
        if kwdefaults is None:
            if current_keywords is not None:
                return False
        elif (
            not _plain_dict(current_keywords)
            or len(current_keywords) != len(kwdefaults)
            or any(
                current_keywords.get(name, _MISSING) is not original
                for name, original in kwdefaults
            )
        ):
            return False
        current_closure = _closure_contents(function)
        if closure is None:
            if current_closure is not None:
                return False
        elif (
            current_closure is None
            or len(current_closure) != len(closure)
            or any(
                value is not original
                for value, original in zip(current_closure, closure)
            )
        ):
            return False
    return True
