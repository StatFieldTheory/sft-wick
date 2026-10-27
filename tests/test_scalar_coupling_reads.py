"""Scalar coupling read reuse: ordered results and native protocol fallback."""

import sys
from types import ModuleType

import numpy as np
import pytest

from sft_wick import _coupling_plan as cp
from sft_wick import perturbation as pt
from sft_wick.expressions import (
    KroneckerDelta,
    Product,
    Propagator,
    Rational,
    Sum,
    Symbol,
)


def term(phase=0, repetitions=24):
    expression = Sum(
        tuple(
            Product(
                (
                    Rational(1, 3),
                    Symbol("A", ("a", "i")),
                    Symbol("B", ("i", "j")),
                    KroneckerDelta("j", "0"),
                )
            )
            for _ in range(repetitions)
        )
    )
    return pt.DiagramTerm(
        (Propagator("C", "j", "j", "x", "y"),),
        expression,
        Rational(-2, 5),
        (),
        (("i", 2), ("j", 3)),
        phase,
    )


def values(dtype=float):
    return {
        "A": np.array([[0.125, 1.125], [2.125, 3.125]], dtype=dtype),
        "B": np.array([[0.375, 1.375, 2.375], [3.375, 4.375, 5.375]], dtype=dtype),
    }


def interpreter(dt, cv, pins):
    pref = dt.response_phase_factor() * (
        dt.rational_prefactor.numerator / dt.rational_prefactor.denominator
    )
    dtype = (
        complex
        if isinstance(pref, complex) or any(np.iscomplexobj(v) for v in cv.values())
        else float
    )
    out = np.zeros(3, dtype=dtype)
    for j in range(3):
        total = dtype(0)
        for i in range(2):
            total += dtype(
                pt._eval_symbolic(dt.coupling_sum, cv, {**pins, "i": i, "j": j})
            )
        out[j] = total
    return pref * out


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.int8,
        np.int64,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
        np.dtype(">f8"),
    ],
)
@pytest.mark.parametrize("phase", range(4))
def test_scalar_complete_query_bit_parity(dtype, phase):
    dt, cv, pins = term(phase), values(dtype), {"a": -1}
    with np.errstate(all="ignore"):
        actual, expected = dt.evaluate_coupling(cv, pins), interpreter(dt, cv, pins)
        assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
        cv["A"] = cv["A"] + 0.125
        pins["a"] = 0
        assert (
            dt.evaluate_coupling(cv, pins).tobytes()
            == interpreter(dt, cv, pins).tobytes()
        )


def test_instrumented_complete_query_read_witness(monkeypatch):
    # Mechanical witness only: profile observation explicitly disables the new
    # production trace guard in this isolated instance. Formal costs use neither.
    monkeypatch.setattr(cp, "_NATIVE_READ_TRACE", (), raising=False)
    dt, cv, calls = term(), values(), []
    code = cp._indices.__code__

    def profile(frame, event, arg):
        if event == "call" and frame.f_code is code:
            calls.append(1)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with np.errstate(all="ignore"):
            actual = dt.evaluate_coupling(cv, {"a": 0})
    finally:
        sys.setprofile(previous)
    assert len(calls) == 12  # 3 outer * 2 inner * 2 distinct indexed leaves.
    assert actual.tobytes() == interpreter(dt, cv, {"a": 0}).tobytes()


@pytest.mark.parametrize("name", ["float", "complex", "sum"])
def test_conversion_and_fold_hook_decline(monkeypatch, name):
    native = cp._scalar_symbol.__builtins__[name]
    monkeypatch.setattr(cp, name, lambda *args: native(*args), raising=False)
    with np.errstate(all="ignore"):
        assert not cp.native_scalar_symbol_reads(values(), float)


@pytest.mark.parametrize("name", ["sum", "float"])
def test_repeated_leaf_mutating_hook_matches_legacy_public_call(monkeypatch, name):
    dt, cv = term(), values()
    assert dt._coupling_execution_plan is not None
    initial = {key: value.copy() for key, value in cv.items()}
    builtin = cp._scalar_symbol.__builtins__[name]
    calls = []

    def hook(argument):
        if name == "sum":
            result = 0
            for k, value in enumerate(argument):
                result += value
                if k == 0:
                    cv["A"] = cv["A"] + 0.125
            calls.append(1)
            return result
        result = builtin(argument)
        calls.append(1)
        cv["A"] = cv["A"] + 0.125
        return result

    monkeypatch.setattr(cp, name, hook, raising=False)
    with np.errstate(all="ignore"):
        actual = dt.evaluate_coupling(cv, {"a": 0})
        actual_calls = len(calls)
        actual_inputs = {key: value.copy() for key, value in cv.items()}
        calls.clear()
        cv.clear()
        cv.update(initial)
        # Exact previous public binding policy: no scalar read memo.
        monkeypatch.setattr(pt, "native_scalar_symbol_reads", lambda *args: False)
        expected = dt.evaluate_coupling(cv, {"a": 0})
    assert actual.tobytes() == expected.tobytes() and len(calls) == actual_calls
    assert all(cv[key].tobytes() == actual_inputs[key].tobytes() for key in cv)


@pytest.mark.parametrize("name", ["type", "any", "all", "getattr"])
def test_new_guard_adds_no_custom_builtin_calls_on_legacy_route(monkeypatch, name):
    dt, cv = term(), values()
    assert dt._coupling_execution_plan is not None
    native = cp._scalar_symbol.__builtins__[name]
    calls = []

    def hook(*args, **kwargs):
        calls.append(1)
        return native(*args, **kwargs)

    monkeypatch.setattr(cp, name, hook, raising=False)
    with np.errstate(all="ignore"):
        actual = dt.evaluate_coupling(cv, {"a": 0})
        actual_count = len(calls)
        calls.clear()
        monkeypatch.setattr(pt, "native_scalar_symbol_reads", lambda *args: False)
        expected = dt.evaluate_coupling(cv, {"a": 0})
    assert actual.tobytes() == expected.tobytes()
    assert len(calls) == actual_count
    if name == "type":
        assert actual_count > 0


@pytest.mark.parametrize("module", [np, sys])
def test_new_guard_declines_module_subclass_without_attribute_hooks(module):
    cv, calls = values(), []

    class ObservedModule(ModuleType):
        def __getattribute__(self, name):
            calls.append(name)
            return super().__getattribute__(name)

    with np.errstate(all="ignore"):
        original = module.__class__
        module.__class__ = ObservedModule
        try:
            admitted = cp.native_scalar_symbol_reads(cv, float)
            observed = calls.copy()
        finally:
            module.__class__ = original
    assert not admitted and observed == []


@pytest.mark.parametrize("module,name", [(np, "asarray"), (sys, "gettrace")])
def test_new_guard_declines_missing_module_attribute_without_getattr(
    monkeypatch, module, name
):
    cv, calls = values(), []
    original = getattr(module, name)

    def hook(attribute):
        calls.append(attribute)
        if attribute == name:
            return original
        raise AttributeError(attribute)

    with np.errstate(all="ignore"), monkeypatch.context() as patches:
        patches.setattr(module, "__getattr__", hook, raising=False)
        patches.delattr(module, name)
        admitted = cp.native_scalar_symbol_reads(cv, float)
        observed = calls.copy()
    assert not admitted and observed == []


@pytest.mark.parametrize("change", ["symbol", "pin", "bounds", "rank"])
def test_public_first_error_matches_ordered_interpreter(change):
    dt, cv, pins = term(), values(), {"a": 0}
    if change == "symbol":
        del cv["A"]
    elif change == "pin":
        pins.clear()
    elif change == "bounds":
        pins["a"] = 3
    else:
        cv["A"] = np.ones((2, 2, 2))
    errors = []
    for operation in (
        lambda: dt.evaluate_coupling(cv, pins),
        lambda: interpreter(dt, cv, pins),
    ):
        with np.errstate(all="ignore"):
            try:
                operation()
            except (KeyError, IndexError, ValueError) as error:
                errors.append((type(error), str(error)))
    assert len(errors) == 2 and errors[0] == errors[1]


@pytest.mark.parametrize(
    "name", ["_scalar_symbol", "_indices", "_delta", "_caller_label", "_batched_symbol"]
)
def test_helper_in_place_code_mutation_declines(monkeypatch, name):
    original = getattr(cp, name)

    def replacement(*args):
        return 1

    monkeypatch.setattr(original, "__code__", replacement.__code__)
    with np.errstate(all="ignore"):
        assert not cp.native_scalar_symbol_reads(values(), float)


@pytest.mark.parametrize(
    "dtype",
    [
        np.longdouble,
        np.clongdouble,
        object,
        np.dtype(float, metadata={"owner": object()}),
    ],
)
def test_nonordinary_dtype_declines(dtype):
    with np.errstate(all="ignore"):
        cv = values(dtype)
        if cv["A"].dtype.kind not in "biufc":
            assert not cp.native_bindings(cv, {})
        elif cv["A"].dtype.metadata or not any(
            cv["A"].dtype.type is kind for kind in cp._NATIVE_SCALAR_TYPES
        ):
            assert not cp.native_scalar_symbol_reads(cv, float)


@pytest.mark.parametrize("layout", ["strided", "readonly", "broadcast", "alias"])
def test_plain_array_layouts_keep_complete_scalar_result(layout):
    cv, dt = values(), term()
    if layout == "strided":
        cv["A"] = cv["A"][::-1]
        cv["B"] = cv["B"][:, ::-1]
    elif layout == "readonly":
        cv["A"].flags.writeable = False
    elif layout == "broadcast":
        cv["A"] = np.broadcast_to(cv["A"][:1], (2, 2))
    else:
        base = np.arange(16.0)
        cv["A"] = base[:4].reshape(2, 2)
        cv["B"] = base[:6].reshape(2, 3)
    with np.errstate(all="ignore"):
        assert (
            dt.evaluate_coupling(cv, {"a": 1}).tobytes()
            == interpreter(dt, cv, {"a": 1}).tobytes()
        )


def test_no_index_scalar_is_not_stored_in_memo():
    expression = Sum(
        tuple(Product((Symbol("A"), Symbol("B", ("i",)))) for _ in range(24))
    )
    plan = cp.compile_coupling_plan(expression)
    bindings = plan.bind_indices({"i": 0}, reuse_symbols=True)
    cv = {"A": np.array(0.125), "B": np.array([2.0])}
    assert plan.scalar(expression, cv, bindings) == pt._eval_symbolic(
        expression, cv, {"i": 0}
    )
    for recipe in plan.core.symbols:
        if not recipe.indices:
            assert bindings.symbol_values[recipe.read_slot] is cp._UNREAD


@pytest.mark.parametrize("mode", ["trace", "profile"])
def test_observation_hooks_decline_only_new_route(mode):
    get, set_hook = (
        (sys.gettrace, sys.settrace)
        if mode == "trace"
        else (sys.getprofile, sys.setprofile)
    )
    previous = get()
    set_hook(lambda *args: None)
    try:
        with np.errstate(all="ignore"):
            assert not cp.native_scalar_symbol_reads(values(), float)
    finally:
        set_hook(previous)


def test_fake_ndarray_alias_never_admits_new_scalar_memo(monkeypatch):
    class Fake:
        dtype = np.dtype("float64")

        def __getitem__(self, indices):
            return 1.0

    monkeypatch.setattr(np, "ndarray", Fake)
    with np.errstate(all="ignore"):
        assert not cp.native_scalar_symbol_reads({"A": Fake()}, float)
