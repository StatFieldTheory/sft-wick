"""Explicit original-factor static contraction with an exact lazy array boundary.

Only private native monomial or same-sign Sum snapshots enter this backend. Reading the public
``coupling_array`` produces the original ordered ndarray and disables fusion.
The budget is conservative ndarray payload, never process RSS. No numeric input
is retained by the bounded structural einsum plan cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math
from threading import Lock
from types import FunctionType

import numpy as np

from . import perturbation as _pt
from . import _coupling_plan as _cp
from . import expressions as _expressions
from ._dense_contraction import (
    _DEFAULT_ERRORS,
    _SCAN_RESERVE,
    _SCAN_ELEMENTS,
    _factor_metadata,
    _native_dtype,
    _ordinary_label,
    _plan,
    _scan,
)
from .expressions import _NATIVE_NUMERIC_EXPRESSION_CLASSES

# Use definition-time class bindings, as the original numerical evaluators do.
# A missing/rebound public module export does not change these existing classes.
(_, _, Product, Sum, Symbol, Rational, ImaginaryUnit, Propagator) = (
    _NATIVE_NUMERIC_EXPRESSION_CLASSES
)


def _protocol(cls):
    namespace = dict(cls.__dict__)
    namespace.pop("__slotnames__", None)
    functions = []
    for member in namespace.values():
        if type(member) is FunctionType:
            candidates = ((member, None, member),)
        elif type(member) in (staticmethod, classmethod):
            candidates = ((member, "__func__", member.__func__),)
        elif type(member) is cached_property:
            candidates = ((member, "func", member.func),)
        elif type(member) is property:
            candidates = tuple(
                (member, name, getattr(member, name))
                for name in ("fget", "fset", "fdel")
            )
        else:
            candidates = ()
        for descriptor, attribute, function in candidates:
            if type(function) is FunctionType:
                functions.append(
                    (
                        descriptor,
                        attribute,
                        function,
                        function.__code__,
                        function.__defaults__,
                        dict(function.__kwdefaults__ or {}),
                    )
                )
    return namespace, tuple(functions)


def _unchanged(cls, witness):
    namespace, functions = witness
    actual = cls.__dict__
    if len(actual) not in (len(namespace), len(namespace) + 1):
        return False
    if any(type(name) is not str for name in actual):
        return False
    keys = set(actual)
    if "__slotnames__" in keys:
        slots = actual["__slotnames__"]
        if type(slots) is not list or slots:
            return False
        keys.remove("__slotnames__")
    if keys != namespace.keys() or any(
        actual[name] is not member for name, member in namespace.items()
    ):
        return False
    for descriptor, attribute, function, code, defaults, kwdefaults in functions:
        if (
            attribute is not None
            and getattr(descriptor, attribute) is not function
            or function.__code__ is not code
            or function.__defaults__ is not defaults
        ):
            return False
        kw = function.__kwdefaults__
        if kw is None:
            kw = {}
        elif type(kw) is not dict or any(type(key) is not str for key in kw):
            return False
        if kw.keys() != kwdefaults.keys() or any(
            kw[key] is not value for key, value in kwdefaults.items()
        ):
            return False
    return True


_TERM_PROTOCOL = _pt._NATIVE_NUMERIC_TERM_PROTOCOL
_TERM_BASES = _pt._NATIVE_NUMERIC_TERM_BASES
_NODE_PROTOCOLS = _expressions._NATIVE_NUMERIC_EXPRESSION_PROTOCOLS
_NODE_BASES = _expressions._NATIVE_NUMERIC_EXPRESSION_BASES
_ABC_PROTOCOL = _expressions._NATIVE_NUMERIC_ABC_PROTOCOL
_SYMBOLIC_PROTOCOL = _pt._NATIVE_NUMERIC_SYMBOLIC_PROTOCOL
_COUPLING_GLOBALS = _pt._NATIVE_NUMERIC_COUPLING_GLOBALS
_COUPLING_BUILTINS = _pt._NATIVE_NUMERIC_COUPLING_BUILTINS
_COUPLING_MISSING = _pt._NATIVE_NUMERIC_COUPLING_MISSING
(
    _PLANNER_FUNCTIONS,
    _PLANNER_GLOBALS,
    _PLANNER_BUILTINS,
    _PLANNER_MISSING,
    _SHARED_PLANNER,
    _SHARED_PLANNER_PROTOCOL,
    _PLANNER_CLASSES,
    _PLANNER_BASES,
) = _cp._NATIVE_NUMERIC_PLANNER_PROTOCOL
_CLASS = None
_CLASS_PROTOCOL = None
_CLASS_LOCK = Lock()


def _native_node_protocol():
    namespace, dump, available, meta, witness, bases, registries, globals_read = (
        _ABC_PROTOCOL
    )
    # Verify the metaclass before inspecting its classes or calling the native
    # dump. Compare only actual virtual registrations, not global cache tokens.
    if (
        not available
        or namespace.get("_get_dump") is not dump
        or not _unchanged(meta, witness)
        or meta.__bases__ is not bases
        or any(scope.get(name) is not value for scope, name, value in globals_read)
    ):
        return False
    if not all(_unchanged(cls, protocol) for cls, protocol in _NODE_PROTOCOLS):
        return False
    if any(cls.__bases__ is not old for cls, old in _NODE_BASES):
        return False
    for cls, registered in registries:
        actual = dump(cls)[0]
        if type(actual) is not set or len(actual) != len(registered):
            return False
        if frozenset(actual) != registered:
            return False
    return True


def _function_unchanged(function, witness):
    original, code, defaults, kwdefaults = witness
    if (
        function is not original
        or function.__code__ is not code
        or function.__defaults__ is not defaults
    ):
        return False
    actual = function.__kwdefaults__
    if actual is None:
        actual = {}
    elif type(actual) is not dict:
        return False
    if len(actual) != len(kwdefaults) or any(type(key) is not str for key in actual):
        return False
    return actual.keys() == kwdefaults.keys() and all(
        actual[key] is value for key, value in kwdefaults.items()
    )


def _native_term(term):
    if (
        type(term) is not _pt.DiagramTerm
        or not _unchanged(_pt.DiagramTerm, _TERM_PROTOCOL)
        or _pt.DiagramTerm.__bases__ is not _TERM_BASES
        or not _native_node_protocol()
        or not _function_unchanged(
            _pt.__dict__.get("_eval_symbolic"), _SYMBOLIC_PROTOCOL
        )
        or _pt.__dict__.get("compile_coupling_plan") is not _pt._NATIVE_COMPILE_PLAN
        or _cp.__dict__.get("compile_coupling_plan") is not _pt._NATIVE_COMPILE_PLAN
        or _cp.__dict__.get("_shared_core") is not _SHARED_PLANNER
        or not _function_unchanged(
            _SHARED_PLANNER.__dict__.get("__wrapped__"), _SHARED_PLANNER_PROTOCOL
        )
        or not all(_unchanged(cls, protocol) for cls, protocol in _PLANNER_CLASSES)
        or any(cls.__bases__ is not bases for cls, bases in _PLANNER_BASES)
        or any(
            namespace.get(name, _PLANNER_MISSING) is not value
            for namespace, name, value in _PLANNER_GLOBALS
        )
        or any(
            namespace.get(name) is not value
            for namespace, name, value in _PLANNER_BUILTINS
        )
        or any(
            namespace.get(name, _COUPLING_MISSING) is not value
            for namespace, name, value in _COUPLING_GLOBALS
        )
        or any(
            namespace.get(name) is not value
            for namespace, name, value in _COUPLING_BUILTINS
        )
        or type(object.__getattribute__(term, "__dict__")) is not dict
    ):
        return False
    state = object.__getattribute__(term, "__dict__")
    # Legacy evaluate_coupling calls any cached plan.scalar, even a custom
    # object, before its optional native-reuse guard. Provenance alone does not
    # certify generated function code/closures. Leave all pre-existing nonempty
    # plans on that original path rather than silently bypass their behavior.
    if state.get("_coupling_execution_plan") is not None:
        return False
    for name, function, code, defaults, kwdefaults in _PLANNER_FUNCTIONS:
        if not _function_unchanged(
            _cp.__dict__.get(name), (function, code, defaults, kwdefaults)
        ):
            return False
    summation = state.get("summation_indices")
    propagators = state.get("propagators")
    if (
        type(summation) is not tuple
        or any(
            type(item) is not tuple
            or len(item) != 2
            or not _ordinary_label(item[0])
            or type(item[1]) is not int
            or item[1] <= 0
            for item in summation
        )
        or type(propagators) is not tuple
        or any(
            type(prop) is not Propagator
            or any(
                value is not None and not _ordinary_label(value)
                for value in (prop.index_left, prop.index_right)
            )
            for prop in propagators
        )
    ):
        return False
    for name, member in _pt._NATIVE_COUPLING_METHODS.items():
        if _pt.DiagramTerm.__dict__.get(name) is not member:
            return False
        if name in state and name != "_coupling_execution_plan":
            return False
    return not any(
        name in state
        for name, member in _TERM_PROTOCOL[0].items()
        if type(member) in (FunctionType, staticmethod, classmethod, property)
    )


@dataclass(frozen=True)
class _Monomial:
    factors: tuple
    phase: float
    scans: tuple


@dataclass(frozen=True)
class _Snapshot:
    instruction: tuple
    arrays: tuple
    monomials: tuple
    prop_idx: tuple
    fixed: tuple
    prefactor: float | complex
    dtype: type
    e_psi: int
    nbytes: int


class _CoefficientOwner:
    """One latent ndarray binding shared by configured shallow views."""

    def __init__(self, array, snapshot=None, present=True):
        self.array = array
        self.snapshot = snapshot
        self.present = present
        self.lock = Lock()

    def __getstate__(self):
        with self.lock:
            return self.array, self.snapshot, self.present

    def __setstate__(self, state):
        self.array, self.snapshot, self.present = state
        self.lock = Lock()
        if self.snapshot is not None:
            for array in self.snapshot.arrays:
                array.flags.writeable = False


_OWNER_PROTOCOL = _protocol(_CoefficientOwner)


def share_factorized_owner(view, source):
    """Bind a configured shallow view to the same latent coefficient ndarray.

    Reading either view materializes once and disables fusion for both. Field
    replacement/deletion detaches only that instance, as with ordinary shallow
    copies of an ndarray field. This helper performs no public array read.
    """
    if type(view) is not _CLASS or type(source) is not _CLASS:
        return
    source_state = object.__getattribute__(source, "__dict__")
    view_state = object.__getattribute__(view, "__dict__")
    owner = source_state.get("_factorized_owner")
    if type(owner) is not _CoefficientOwner:
        return
    with owner.lock:
        view_state["_factorized_owner"] = owner
        view_state["_factorized_plan"] = owner.snapshot
        view_state["_factorized_lock"] = owner.lock
        if owner.present:
            view_state["coupling_array"] = owner.array
        else:
            view_state.pop("coupling_array", None)


def factorized_state(integrand):
    """Read authoritative private state for untimed acceptance instrumentation."""
    state = object.__getattribute__(integrand, "__dict__")
    owner = state.get("_factorized_owner")
    if type(owner) is not _CoefficientOwner:
        return state.get("coupling_array"), state.get("_factorized_plan")
    with owner.lock:
        return owner.array if owner.present else None, owner.snapshot


def _scalar(instruction, arrays, assignment):
    opcode = instruction[0]
    if opcode == "r":
        return instruction[1]
    if opcode == "i":
        return 1j
    if opcode == "s":
        array = arrays[instruction[1]]
        labels = instruction[2]
        value = array[tuple(assignment[label] for label in labels)] if labels else array
        return complex(value) if _native_dtype(array, np.complex128) else float(value)
    if opcode == "a":
        # Match both _eval_symbolic and the native scalar compiler, including
        # Python's version-specific compensated float/complex sum algorithm.
        return sum(_scalar(child, arrays, assignment) for child in instruction[1])
    value = 1.0
    for child in instruction[1]:
        value *= _scalar(child, arrays, assignment)
    return value


def _materialize(snapshot, max_bytes):
    shape = tuple(dim for _, dim in snapshot.prop_idx)
    planned = (
        snapshot.nbytes
        + 2 * math.prod(shape) * np.dtype(snapshot.dtype).itemsize
        + _SCAN_RESERVE
    )
    _require_materialization_budget(planned, max_bytes)
    output = np.zeros(shape, dtype=snapshot.dtype)
    names = tuple(name for name, _ in snapshot.prop_idx)
    fixed = dict(snapshot.fixed)
    for values in np.ndindex(*shape):
        assignment = {**fixed, **dict(zip(names, values))}
        output[values] = snapshot.dtype(
            _scalar(snapshot.instruction, snapshot.arrays, assignment)
        )
    return snapshot.prefactor * output


def _require_materialization_budget(planned, max_bytes):
    if type(max_bytes) is not int or max_bytes <= 0 or planned > max_bytes:
        raise ValueError(
            "factorized coefficient materialization requires a conservative "
            f"ndarray payload of {planned} bytes, exceeding "
            f"contraction_max_bytes={max_bytes}; increase this budget or rebuild "
            "this integrand with contraction_backend='ordered' explicitly to permit the original "
            "coefficient materialization"
        )


def check_materialization_budget(term, static_cv, max_bytes):
    """Reject a known oversized dense fallback without calling any getters.

    Only the explicit factorized builder uses this policy. Ordinary native raw
    component metadata determines the original coefficient shape. Unknown
    metadata retains its existing evaluator, and ordinary ordered/einsum
    construction does not call this helper. Two complex128 arrays conservatively
    cover coefficient preparation and its final prefactor multiplication.
    """
    if (
        type(term) is not _pt.DiagramTerm
        or not _unchanged(_pt.DiagramTerm, _TERM_PROTOCOL)
        or _pt.DiagramTerm.__bases__ is not _TERM_BASES
        or not _native_node_protocol()
    ):
        return
    state = object.__getattribute__(term, "__dict__")
    if type(state) is not dict:
        return
    summation = state.get("summation_indices")
    propagators = state.get("propagators")
    if type(summation) is not tuple or type(propagators) is not tuple:
        return
    indices = set()
    for prop in propagators:
        if type(prop) is not Propagator:
            return
        data = object.__getattribute__(prop, "__dict__")
        if type(data) is not dict:
            return
        for field in ("index_left", "index_right"):
            name = data.get(field)
            if name is not None:
                if type(name) is not str:
                    return
                indices.add(name)
    elements = 1
    names = set()
    for item in summation:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not int
            or item[1] <= 0
            or item[0] in names
        ):
            return
        names.add(item[0])
        if item[0] in indices:
            elements *= item[1]
    # static_cv is intentionally not inspected: custom unused values may have
    # conversion hooks. Complex128 is a conservative maximum native dtype.
    _require_materialization_budget(32 * elements + _SCAN_RESERVE, max_bytes)


def _bounds(scans, assignments, terms=1):
    upper = (assignments - 1).bit_length() + (terms - 1).bit_length()
    lower = 0
    for maximum, minimum, _ in scans:
        if maximum:
            upper += max(0, math.frexp(maximum)[1])
        if math.isfinite(minimum):
            lower += min(0, math.frexp(minimum)[1] - 53)
    return upper <= 1020 and lower >= -1000


def _component(array):
    if _native_dtype(array, np.float64):
        scanned = _scan(array)
        if scanned is None or scanned[2] != array.size:
            return None
        return array, 1 + 0j, scanned
    real = _scan(array.real)
    imag = _scan(array.imag)
    if real is None or imag is None:
        return None
    if real[2] == array.size and imag[2] == 0:
        return array.real, 1 + 0j, real
    if imag[2] == array.size and real[2] == 0:
        return array.imag, 1j, imag
    return None


def _uniform_sign(array):
    """A bounded-buffer sign proof, never a component-assignment tensor."""
    sign = 0
    iterator = np.nditer(
        array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="K",
        buffersize=_SCAN_ELEMENTS,
    )
    for chunk in iterator:
        if bool((chunk > 0).all()):
            current = 1
        elif bool((chunk < 0).all()):
            current = -1
        else:
            return None
        if sign and current != sign:
            return None
        sign = current
    return sign or None


def _contract(snapshot, fixed, cb, rp, jac, rb, e_psi, max_bytes):
    if (
        type(max_bytes) is not int
        or max_bytes <= 0
        or np.geterr() != _DEFAULT_ERRORS
        or type(e_psi) is not int
        or not 0 <= e_psi <= 1024
        or e_psi != snapshot.e_psi
        or (1j**e_psi) not in (1 + 0j, -1 + 0j, 1j, -1j)
        or type(fixed) is not dict
        or len(fixed) > 128
        or any(
            not _ordinary_label(k) or len(k) > 256 or type(v) is not int
            for k, v in fixed.items()
        )
        or type(cb) not in (list, tuple)
        or type(rb) not in (list, tuple)
        or not cb
        or type(rp) is not np.ndarray
        or not _native_dtype(rp, np.float64)
        or rp.ndim != 1
        or not rp.size
        or type(jac) is not np.ndarray
        or not _native_dtype(jac, np.float64)
        or jac.shape != rp.shape
    ):
        return None
    names = {name: (axis, dim) for axis, (name, dim) in enumerate(snapshot.prop_idx, 1)}
    samples = rp.size
    kernels = []
    for batches, response in ((cb, False), (rb, True)):
        for batch in batches:
            item = _factor_metadata(batch, names, fixed, samples, response=response)
            if item is None:
                return None
            kernels.append(item)
    borrowed = snapshot.nbytes + rp.nbytes + jac.nbytes + 8
    borrowed += sum(
        kernel[0].nbytes + (kernel[4].nbytes if kernel[4] is not None else 0)
        for kernel in kernels
    )
    paths = []
    scratch = 0
    for monomial in snapshot.monomials:
        if len(monomial.factors) + len(kernels) + 3 > 31:
            return None
        shapes = (
            *(factor[3] for factor in monomial.factors),
            *(kernel[3] for kernel in kernels),
            (),
            (samples,),
            (samples,),
        )
        labels = (
            *(factor[4] for factor in monomial.factors),
            *(kernel[2] for kernel in kernels),
            (),
            (0,),
            (0,),
        )
        covered = set().union(*(set(indices) for indices in labels))
        if covered != set(range(len(snapshot.prop_idx) + 1)):
            return None
        plan = _plan(shapes, labels)
        if plan is None:
            return None
        path, intermediate, _ = plan
        paths.append(path)
        scratch = max(
            scratch,
            8 * sum(math.prod(shape) for shape in shapes) + 16 * intermediate,
        )
    # Terms execute sequentially and release their scratch before the next.
    # Keep both the accumulated sample vector and the next term's output live.
    workspace = borrowed + scratch + 16 * samples + _SCAN_RESERVE
    if workspace > max_bytes:
        return None
    for kernel in kernels:
        if kernel[4] is not None and not bool(kernel[4].all()):
            return None
    common_scans = []
    for array in (*(kernel[0] for kernel in kernels), rp, jac):
        scanned = _scan(array)
        if scanned is None:
            return None
        common_scans.append(scanned)
    assignments = math.prod(dim for _, dim in snapshot.prop_idx)
    for monomial in snapshot.monomials:
        if not _bounds(
            (*monomial.scans, *common_scans), assignments, len(snapshot.monomials)
        ):
            return None
    total = None
    for monomial, path in zip(snapshot.monomials, paths):
        arguments = []
        for factor in monomial.factors:
            slot, imaginary, slices, _, indices = factor
            array = snapshot.arrays[slot]
            component = array.imag if imaginary else array.real
            arguments.extend((component[slices], list(indices)))
        for kernel in kernels:
            arguments.extend((kernel[0][kernel[1]], list(kernel[2])))
        arguments.extend(
            (
                np.array(monomial.phase, dtype=np.float64),
                [],
                rp,
                [0],
                jac,
                [0],
                [0],
            )
        )
        result = np.einsum(*arguments, optimize=list(path))
        if total is None:
            total = result
        else:
            np.add(total, result, out=total)
        del result, arguments
    return total


def _integrand_class():
    if _CLASS is not None:
        return _CLASS
    with _CLASS_LOCK:
        return _create_integrand_class()


def _create_integrand_class():
    global _CLASS, _CLASS_PROTOCOL
    if _CLASS is not None:
        return _CLASS
    from .evaluate import DiagramIntegrand

    class _FactorizedIntegrand(DiagramIntegrand):
        @property
        def coupling_array(self):
            state = object.__getattribute__(self, "__dict__")
            while True:
                owner = state.get("_factorized_owner")
                if owner is None:
                    if "coupling_array" not in state:
                        raise AttributeError("coupling_array")
                    return state["coupling_array"]
                with owner.lock:
                    if state.get("_factorized_owner") is not owner:
                        continue
                    if not owner.present:
                        raise AttributeError("coupling_array")
                    if owner.snapshot is not None:
                        owner.array = _materialize(
                            owner.snapshot, state.get("_contraction_max_bytes")
                        )
                        owner.snapshot = None
                    state["coupling_array"] = owner.array
                    state["_factorized_plan"] = None
                    return owner.array

        @coupling_array.setter
        def coupling_array(self, value):
            state = object.__getattribute__(self, "__dict__")
            while True:
                owner = state.get("_factorized_owner")
                if owner is None:
                    # The inherited dataclass constructor writes this field
                    # before the factory creates any latent owner.
                    state["coupling_array"] = value
                    return
                with owner.lock:
                    if state.get("_factorized_owner") is not owner:
                        continue
                    detached = _CoefficientOwner(value)
                    state["_factorized_owner"] = detached
                    state["_factorized_plan"] = None
                    state["_factorized_lock"] = detached.lock
                    state["coupling_array"] = value
                    return

        @coupling_array.deleter
        def coupling_array(self):
            state = object.__getattribute__(self, "__dict__")
            while True:
                owner = state.get("_factorized_owner")
                if owner is None:
                    if "coupling_array" not in state:
                        raise AttributeError("coupling_array")
                    del state["coupling_array"]
                    return
                with owner.lock:
                    if state.get("_factorized_owner") is not owner:
                        continue
                    if not owner.present:
                        raise AttributeError("coupling_array")
                    detached = _CoefficientOwner(None, present=False)
                    state["_factorized_owner"] = detached
                    state["_factorized_plan"] = None
                    state["_factorized_lock"] = detached.lock
                    state.pop("coupling_array", None)
                    return

        def __getattribute__(self, name):
            getter = DiagramIntegrand.__dict__.get("__getattribute__")
            if getter is None:
                return object.__getattribute__(self, name)
            # A later native-base customization must observe the genuine
            # materialized coefficient, rather than a private placeholder.
            return getter.__get__(self, type(self))(name)

        def __getstate__(self):
            state = object.__getattribute__(self, "__dict__")
            owner = state.get("_factorized_owner")
            if owner is None:
                return dict(state)
            with owner.lock:
                saved = dict(state)
                saved["_factorized_plan"] = owner.snapshot
                if owner.present:
                    saved["coupling_array"] = owner.array
                else:
                    saved.pop("coupling_array", None)
                saved.pop("_factorized_lock", None)
                return saved

        def __setstate__(self, state):
            object.__getattribute__(self, "__dict__").update(state)
            owner = state.get("_factorized_owner")
            if owner is not None:
                object.__setattr__(self, "_factorized_lock", owner.lock)

        def _static_values(
            self, c_batches, r_product, jacobians, *, r_batches=(), where=" (coupling)"
        ):
            if native_factorized_integrand(self):
                state = object.__getattribute__(self, "__dict__")
                while True:
                    owner = state["_factorized_owner"]
                    with owner.lock:
                        if state.get("_factorized_owner") is not owner:
                            continue
                        snapshot = owner.snapshot
                        if (
                            snapshot is not None
                            and self._contraction_backend == "factorized"
                            and self.diagram_term.propagator_indices
                            == snapshot.prop_idx
                        ):
                            values = _contract(
                                snapshot,
                                self.fixed_indices,
                                c_batches,
                                r_product,
                                jacobians,
                                r_batches,
                                self._e_psi,
                                self._contraction_max_bytes,
                            )
                            if values is not None:
                                return values
                        break
            return DiagramIntegrand._static_values(
                self, c_batches, r_product, jacobians, r_batches=r_batches, where=where
            )

    _FactorizedIntegrand.__qualname__ = "_FactorizedIntegrand"
    globals()["_FactorizedIntegrand"] = _FactorizedIntegrand
    _CLASS = _FactorizedIntegrand
    _CLASS_PROTOCOL = _protocol(_CLASS)
    return _CLASS


def __getattr__(name):
    # Standard pickle imports the private class by name in a fresh worker.
    if name == "_FactorizedIntegrand":
        return _integrand_class()
    raise AttributeError(name)


def native_factorized_integrand(integrand):
    """A root wrapper may configure only this unchanged private native protocol."""
    from . import evaluate as ev

    if (
        _CLASS is None
        or type(integrand) is not _CLASS
        or not _unchanged(_CLASS, _CLASS_PROTOCOL)
        or not ev._native_integrand_class()
    ):
        return False
    state = object.__getattribute__(integrand, "__dict__")
    if type(state) is not dict or ev._NATIVE_INTEGRAND_CALLABLES.intersection(state):
        return False
    owner = state.get("_factorized_owner")
    if (
        type(owner) is not _CoefficientOwner
        or not _unchanged(_CoefficientOwner, _OWNER_PROTOCOL)
        or type(object.__getattribute__(owner, "__dict__")) is not dict
        or not owner.present
        or type(owner.lock) is not type(_CLASS_LOCK)
    ):
        return False
    if any(
        name in state
        for name, member in _CLASS_PROTOCOL[0].items()
        if type(member) is FunctionType
    ):
        return False
    if not _native_term(state.get("diagram_term")):
        return False
    return all(
        current is native
        for current, native in zip(
            (
                ev._real_or_raise,
                ev.DiagramIntegrand._resolve_component,
                ev.DiagramIntegrand._r_factor,
                ev.DiagramIntegrand._c_product,
            ),
            ev._NATIVE_DENSE_HELPERS,
        )
    )


def try_build_factorized(term, spatial, static_cv, fixed, max_bytes):
    """Build an admitted factorization, or safely admit its dense fallback.

    A known oversized coefficient fallback raises before allocating that array.
    Increasing the explicit factorized budget or choosing ordered construction
    remains the caller's choice; the old backends do not gain this new policy.
    """
    result = _try_build_factorized(term, spatial, static_cv, fixed, max_bytes)
    if result is None:
        check_materialization_budget(term, static_cv, max_bytes)
    return result


def _try_build_factorized(term, spatial, static_cv, fixed, max_bytes):
    """Return a private lazy native integrand, otherwise decline with ``None``.

    Callers select this helper only for an explicit factorized backend. Admission
    never evaluates a user callable or the symbolic coefficient evaluator.
    """
    from . import evaluate as ev

    if (
        type(max_bytes) is not int
        or max_bytes <= _SCAN_RESERVE
        or np.geterr() != _DEFAULT_ERRORS
        or not ev._native_integrand_class()
        or not _native_term(term)
        or type(static_cv) is not dict
        or len(static_cv) > 128
        or any(type(name) is not str or len(name) > 256 for name in static_cv)
        or type(fixed) is not dict
        or len(fixed) > 128
        or any(
            not _ordinary_label(k) or len(k) > 256 or type(v) is not int
            for k, v in fixed.items()
        )
    ):
        return None
    for value in static_cv.values():
        if type(value) is not np.ndarray or not (
            _native_dtype(value, np.float64) or _native_dtype(value, np.complex128)
        ):
            return None
    state = object.__getattribute__(term, "__dict__")
    summation = state.get("summation_indices")
    propagators = state.get("propagators")
    if (
        type(summation) is not tuple
        or not 1 <= len(summation) <= 51
        or type(propagators) is not tuple
        or any(type(prop) is not Propagator for prop in propagators)
    ):
        return None
    names = {}
    for axis, item in enumerate(summation, 1):
        if (
            type(item) is not tuple
            or len(item) != 2
            or not _ordinary_label(item[0])
            or len(item[0]) > 256
            or type(item[1]) is not int
            or item[1] <= 0
            or item[0] in names
        ):
            return None
        names[item[0]] = axis, item[1]
    for prop in propagators:
        if any(
            value is not None and not _ordinary_label(value)
            for value in (prop.index_left, prop.index_right)
        ):
            return None
    prop_idx = term.propagator_indices
    if prop_idx != summation or math.prod(dim for _, dim in prop_idx) <= 1:
        return None
    rational = state.get("rational_prefactor")
    n_response = state.get("n_response")
    e_psi = state.get("n_external_response", 0)
    if (
        type(rational) is not Rational
        or type(rational.numerator) is not int
        or type(rational.denominator) is not int
        or not rational.denominator
        or type(n_response) is not int
        or not 0 <= n_response <= 1024
        or type(e_psi) is not int
        or not 0 <= e_psi <= 1024
    ):
        return None
    try:
        prefactor = [1.0, -1j, -1.0, 1j][n_response % 4] * (
            rational.numerator / rational.denominator
        )
    except (OverflowError, ZeroDivisionError):
        return None
    if not math.isfinite(abs(prefactor)) or prefactor == 0:
        return None
    used = {}
    nodes = [0]

    def lower(node, factors, depth=0):
        nodes[0] += 1
        if nodes[0] > 256 or depth > 32:
            return None
        kind = type(node)
        if kind is Rational:
            if (
                type(node.numerator) is not int
                or type(node.denominator) is not int
                or not node.denominator
            ):
                return None
            try:
                value = node.numerator / node.denominator
            except (OverflowError, ZeroDivisionError):
                return None
            if not math.isfinite(value) or value == 0:
                return None
            factors.append(("r", value))
            return "r", value
        if kind is ImaginaryUnit:
            factors.append(("i",))
            return ("i",)
        if kind is Symbol:
            if (
                type(node.name) is not str
                or len(node.name) > 256
                or type(node.indices) is not tuple
                or node.name not in static_cv
                or any(
                    not _ordinary_label(label) or len(label) > 256
                    for label in node.indices
                )
                or len(node.indices) > 32
            ):
                return None
            array = static_cv[node.name]
            if array.ndim != len(node.indices):
                return None
            slices = []
            indices = []
            shape = []
            for label, dim in zip(node.indices, array.shape):
                if label in names:
                    axis, size = names[label]
                    if dim != size:
                        return None
                    slices.append(slice(None))
                    indices.append(axis)
                    shape.append(dim)
                elif label in fixed and 0 <= fixed[label] < dim:
                    slices.append(fixed[label])
                else:
                    return None
            slot = used.setdefault(node.name, len(used))
            factors.append(("s", slot, tuple(slices), tuple(shape), tuple(indices)))
            return "s", slot, node.indices
        if kind is not Product or type(node.factors) is not tuple:
            return None
        if len(node.factors) > 64:
            return None
        children = []
        for child in node.factors:
            instruction = lower(child, factors, depth + 1)
            if instruction is None:
                return None
            children.append(instruction)
        return "p", tuple(children)

    expression = state.get("coupling_sum")
    groups = []
    summed = type(expression) is Sum
    if summed:
        if (
            type(expression.terms) is not tuple
            or not 1 <= len(expression.terms) <= 32
            or any(type(term) is not Product for term in expression.terms)
        ):
            return None
        nodes[0] = 1
        instructions = []
        for monomial in expression.terms:
            factors = []
            lowered = lower(monomial, factors, 1)
            if lowered is None or len(factors) > 24:
                return None
            instructions.append(lowered)
            groups.append(tuple(factors))
        instruction = "a", tuple(instructions)
    else:
        factors = []
        instruction = lower(expression, factors)
        if instruction is None or len(factors) > 24:
            return None
        groups.append(tuple(factors))
    if not used:
        return None
    nbytes = sum(static_cv[name].nbytes for name in used)
    if nbytes + _SCAN_RESERVE > max_bytes:
        return None
    components = []
    arrays = []
    for name in used:
        array = static_cv[name]
        owned = np.array(array, copy=True, order="K")
        owned.flags.writeable = False
        component = _component(owned)
        if component is None:
            return None
        arrays.append(owned)
        components.append((component[1], component[2]))
    assignments = math.prod(dim for _, dim in prop_idx)
    monomials = []
    sum_sign = None
    sign_proofs = {}
    for factors in groups:
        scalar_phase = complex(prefactor) * (1j**e_psi)
        operands = []
        scans = []
        for factor in factors:
            if factor[0] == "s":
                phase, scan = components[factor[1]]
                scalar_phase *= phase
                operands.append(
                    (factor[1], phase == 1j, factor[2], factor[3], factor[4])
                )
                scans.append(scan)
            elif factor[0] == "r":
                value = factor[1]
                scalar = np.array(value, dtype=np.float64)
                scalar.flags.writeable = False
                slot = len(arrays)
                arrays.append(scalar)
                operands.append((slot, False, (), (), ()))
                scans.append((abs(value), abs(value), 1))
                nbytes += scalar.nbytes
            else:
                scalar_phase *= 1j
        # Each term independently obeys the action phase. No cancellation
        # between imaginary terms may establish admission of the complete sum.
        if scalar_phase.imag != 0 or not math.isfinite(scalar_phase.real):
            return None
        phase = scalar_phase.real
        if summed:
            # Same-sign monomials exclude cross-term cancellation, including
            # the compensated-sum counterexample [1e16, 1, -1e16]. Kernels may
            # still cause cancellation during component contraction: this is
            # changed rounding, never an arbitrary-tolerance certificate.
            monomial_sign = 1 if phase > 0 else -1
            for slot, imaginary, slices, _, _ in operands:
                key = (
                    slot,
                    imaginary,
                    tuple(None if type(item) is slice else item for item in slices),
                )
                sign = sign_proofs.get(key)
                if sign is None:
                    array = arrays[slot]
                    component = array.imag if imaginary else array.real
                    sign = _uniform_sign(component[slices])
                    if sign is None:
                        return None
                    sign_proofs[key] = sign
                monomial_sign *= sign
            if sum_sign is not None and monomial_sign != sum_sign:
                return None
            sum_sign = monomial_sign
        scans.append((abs(phase), abs(phase), 1))
        if not _bounds(scans, assignments, len(groups)):
            return None
        monomials.append(_Monomial(tuple(operands), phase, tuple(scans)))
    dtype = (
        complex
        if isinstance(prefactor, complex)
        or any(_native_dtype(array, np.complex128) for array in static_cv.values())
        else float
    )
    nbytes += np.dtype(dtype).itemsize  # still-live public 0-d placeholder
    if nbytes + _SCAN_RESERVE > max_bytes:
        return None
    snapshot = _Snapshot(
        instruction,
        tuple(arrays),
        tuple(monomials),
        prop_idx,
        tuple(fixed.items()),
        prefactor,
        dtype,
        e_psi,
        nbytes,
    )
    cls = _integrand_class()
    if not _unchanged(cls, _CLASS_PROTOCOL):
        return None
    integrand = cls(
        diagram_term=term,
        spatial=spatial,
        coupling_array=np.zeros((), dtype=dtype),
        fixed_indices=dict(fixed),
        _contraction_backend="factorized",
        _contraction_max_bytes=max_bytes,
    )
    integrand_state = object.__getattribute__(integrand, "__dict__")
    owner = _CoefficientOwner(integrand_state["coupling_array"], snapshot)
    integrand_state["_factorized_owner"] = owner
    integrand_state["_factorized_plan"] = snapshot
    integrand_state["_factorized_lock"] = owner.lock
    return integrand
