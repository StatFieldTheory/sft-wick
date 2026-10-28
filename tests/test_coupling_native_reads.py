"""Caller-label witnesses and invocation-local native reads against interpreter."""

import copy
from dataclasses import replace
import gc
import pickle
import subprocess
import sys
import warnings
import weakref

import numpy as np
import pytest

from sft_wick import _coupling_plan as cp
from sft_wick._coupling_probe import known_scalar_shape
from sft_wick.expressions import (
    KroneckerDelta,
    Product,
    Propagator,
    Rational,
    Sum,
    Symbol,
)
from sft_wick.perturbation import DiagramTerm, _eval_symbolic, _eval_symbolic_batched

#: The native read path reproduces NumPy's ``ndindex`` as implemented from
#: NumPy 2.3 (``itertools.product``).  Earlier NumPy uses ``nditer``; the path
#: is then disabled and the ordered interpreter runs, with the same values.
#: Read and call counts of the native path are checked only where it exists.
NATIVE_NDINDEX = pytest.mark.skipif(
    not cp._NATIVE_NDINDEX_PROVENANCE,
    reason="this NumPy's ndindex is not the itertools.product implementation",
)


def expression(a="a", i="i", j="j"):
    return Sum(
        tuple(
            Product(
                (
                    Rational(1, 3),
                    Symbol("A", (a, i)),
                    Symbol("B", (i, j)),
                    KroneckerDelta(j, "1"),
                )
            )
            for _ in range(24)
        )
    )


def term(a="a", i="i", j="j", phase=0):
    return DiagramTerm(
        (Propagator("C", j, j, "x", "y"),),
        expression(a, i, j),
        Rational(-2, 5),
        (),
        ((i, 2), (j, 3)),
        phase,
    )


def values(dtype=np.float64, batch=4):
    a = (np.arange(4).reshape(2, 2) + 0.125).astype(dtype)
    b = (np.arange(6).reshape(2, 3) + 0.375).astype(dtype)
    if np.dtype(dtype).kind == "c":
        a = a + 0.25j
        b = b - 0.5j
    return {"A": np.stack([a + k / 8 for k in range(batch)]) if batch else a, "B": b}


def oracle(dt, cv, batch, pins):
    # Original interpreter, explicit outer j and inner i; preserve fold dtype.
    dtype = (
        complex
        if dt.n_response % 2 or any(np.iscomplexobj(x) for x in cv.values())
        else float
    )
    a, i, j = dt._coupling_execution_plan.labels
    result = np.zeros((batch, 3), dtype=dtype)
    for jp in range(3):
        total = np.zeros(batch, dtype=dtype)
        for ip in range(2):
            m = {**pins, i: ip, j: jp}
            total = total + np.asarray(
                _eval_symbolic_batched(dt.coupling_sum, cv, m, batch), dtype=dtype
            )
        result[:, jp] = total
    return dt.response_phase_factor() * (-2 / 5) * result


def reads(fn):
    count = 0
    code = cp._indices.__code__

    def profile(frame, event, arg):
        nonlocal count
        if event == "call" and frame.f_code is code:
            count += 1

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        result = fn()
    finally:
        sys.setprofile(previous)
    return result, count


@pytest.mark.parametrize("phase", range(4))
@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128, np.int64]
)
def test_alpha_and_memo_match_ordered_interpreter(phase, dtype):
    left, right = term(phase=phase), term("external", "inner", "outer", phase)
    cv = values(dtype)
    assert left._coupling_execution_plan.core is right._coupling_execution_plan.core
    for dt, pins in ((left, {"a": -1}), (right, {"external": -1})):
        with np.errstate(all="ignore"):
            result, count = reads(lambda: dt.evaluate_coupling_batched(cv, 4, pins))
            expected = oracle(dt, cv, 4, pins)
        assert result.dtype == expected.dtype and result.tobytes() == expected.tobytes()
        if cp._NATIVE_NDINDEX_PROVENANCE:
            assert count == 12  # 3 outer * 2 inner * 2 distinct Symbol reads
        # New invocation and assignments observe replacement and mutable pins.
        pins[next(iter(pins))] = 0
        cv["A"] = cv["A"] + 0.125
        with np.errstate(all="ignore"):
            assert (
                dt.evaluate_coupling_batched(cv, 4, pins).tobytes()
                == oracle(dt, cv, 4, pins).tobytes()
            )


@pytest.mark.parametrize("layout", ["readonly", "strided", "broadcast", "alias"])
def test_plain_view_layouts_and_input_lifetime(layout):
    dt, cv = term(), values()
    if layout == "readonly":
        cv["A"].flags.writeable = False
    elif layout == "strided":
        cv["A"] = cv["A"][::-1]
        cv["B"] = cv["B"][:, ::-1]
    elif layout == "broadcast":
        cv["A"] = np.broadcast_to(cv["A"][:1], cv["A"].shape)
    else:
        shared = np.arange(32, dtype=float)
        cv["A"] = shared[:16].reshape(4, 2, 2)
        cv["B"] = shared[:6].reshape(2, 3)
    reference = weakref.ref(cv["A"])
    with np.errstate(all="ignore"):
        actual = dt.evaluate_coupling_batched(cv, 4, {"a": 1})
        assert actual.tobytes() == oracle(dt, cv, 4, {"a": 1}).tobytes()
    del cv
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("value", [0.0, -0.0, np.inf, -np.inf, np.nan])
def test_nonfinite_and_signed_zero_keep_arithmetic_occurrences(value):
    dt, cv = term(), values()
    cv["A"].fill(value)
    with np.errstate(all="ignore"):
        actual = dt.evaluate_coupling_batched(cv, 4, {"a": 0})
        expected = oracle(dt, cv, 4, {"a": 0})
    assert actual.tobytes() == expected.tobytes()


def test_empty_sample_axis():
    dt, cv = term(), values()
    cv["A"] = cv["A"][:0]
    with np.errstate(all="ignore"):
        assert dt.evaluate_coupling_batched(cv, 0, {"a": 1}).shape == (0, 3)


def test_no_index_reshape_copies_are_never_retained():
    expr = Sum(tuple(Product((Symbol("B", ("i",)), Symbol("A"))) for _ in range(24)))
    plan = cp.compile_coupling_plan(expr)
    cv = {"B": np.arange(1.0, 5.0), "A": np.arange(6.0).reshape(2, 3).T}
    bindings = plan.bind_indices({"i": 0}, reuse_symbols=True)
    with np.errstate(all="ignore"):
        actual = plan.batched(expr, cv, bindings, 6)
        expected = _eval_symbolic_batched(expr, cv, {"i": 0}, 6)
    assert actual.tobytes() == expected.tobytes()
    for recipe in plan.core.symbols:
        if not recipe.indices:
            assert bindings.symbol_values[recipe.read_slot] is cp._UNREAD


@pytest.mark.parametrize(
    "restore", [copy.deepcopy, lambda p: pickle.loads(pickle.dumps(p))]
)
def test_alpha_wrapper_serialization_and_provenance(restore):
    plan = term("external", "inner", "outer")._coupling_execution_plan
    cloned = restore(plan)
    assert cloned.core is plan.core and cloned.labels == plan.labels
    assert cloned.expression is not plan.expression and cp.native_plan(cloned)
    assert not cp.native_plan(replace(plan))
    assert not cp.native_plan(replace(plan, labels=tuple(reversed(plan.labels))))
    ref = weakref.ref(cloned)
    identity = id(cloned)
    del cloned
    gc.collect()
    assert ref() is None and identity not in cp._PLAN_PROVENANCE


def test_caller_label_hook_and_error_order_with_shared_core():
    class Logged(dict):
        def __contains__(self, key):
            calls.append(("contains", key))
            return super().__contains__(key)

        def __getitem__(self, key):
            calls.append(("get", key))
            return super().__getitem__(key)

    for labels in (("a", "i", "j"), ("external", "inner", "outer")):
        dt = term(*labels)
        plan = dt._coupling_execution_plan
        cv = values(batch=0)
        calls = []
        pins = Logged({labels[0]: 1, labels[1]: 0, labels[2]: 0})
        result = plan.scalar(dt.coupling_sum, cv, pins)
        actual_calls = calls.copy()
        calls.clear()
        expected = _eval_symbolic(dt.coupling_sum, cv, pins)
        assert result == expected and calls == actual_calls
        for interpreter in (plan.scalar, _eval_symbolic):
            calls.clear()
            with pytest.raises(KeyError, match=f"Index '{labels[0]}'"):
                interpreter(dt.coupling_sum, cv, Logged())
            assert calls == [("contains", labels[0])]


@pytest.mark.parametrize("literal", ["1", "01", "+1", "-1", "0", " 1"])
def test_literal_spellings_and_pins_remain_exact(literal):
    expr = Sum(
        tuple(
            Product((Symbol("A", (literal,)), KroneckerDelta(literal, "0")))
            for _ in range(24)
        )
    )
    plan = cp.compile_coupling_plan(expr)
    assert plan.labels == ()
    cv = {"A": np.arange(4.0)}
    pins = {literal: 2, "0": 2}
    assert plan.scalar(expr, cv, pins) == _eval_symbolic(expr, cv, pins)
    assert plan.batched(expr, cv, pins, 4) == _eval_symbolic_batched(expr, cv, pins, 4)


@pytest.mark.parametrize(
    "helper", ["asarray", "geterr", "iscomplexobj", "broadcast_to", "zeros", "ndindex"]
)
def test_numpy_replacement_declines_read_reuse(monkeypatch, helper):
    dt, cv = term(), values()
    native = getattr(np, helper)

    def custom(*args, **kwargs):
        return native(*args, **kwargs)

    monkeypatch.setattr(np, helper, custom)
    with np.errstate(all="ignore"):
        result, count = reads(lambda: dt.evaluate_coupling_batched(cv, 4, {"a": 1}))
        assert result.tobytes() == oracle(dt, cv, 4, {"a": 1}).tobytes()
        assert count == 288


def test_compiler_captured_custom_leaf_cannot_later_admit_memo(monkeypatch):
    native = cp._batched_symbol
    calls = []

    def custom(*args):
        calls.append(1)
        return native(*args)

    monkeypatch.setattr(cp, "_batched_symbol", custom)
    dt = term("new_external", "new_inner", "new_outer")
    plan = dt._coupling_execution_plan
    assert plan.labels == () and not cp.native_plan(plan)
    monkeypatch.setattr(cp, "_batched_symbol", native)
    with np.errstate(all="ignore"):
        dt.evaluate_coupling_batched(values(), 4, {"new_external": 1})
    assert len(calls) == 288


@pytest.mark.parametrize("handler", ["call", "log", "warning"])
def test_external_arithmetic_handler_mutations_keep_uncached_route(handler):
    # The first arithmetic warning replaces the tensor. Later Symbol reads
    # must observe that replacement, so read memoization is not admissible.
    expr = Sum(
        tuple(Product((Symbol("A", ("a",)), Symbol("B", ("a",)))) for _ in range(24))
    )
    dt = DiagramTerm((), expr, Rational(1), (), (("unused", 4),), 0)
    outcomes = []
    for compiled in (False, True):
        cv = {"A": np.array([np.finfo(float).max]), "B": np.array([2.0])}
        events = []

        def mutate(*args):
            events.append(1)
            cv["B"] = np.array([0.0])

        class Logger:
            def write(self, message):
                mutate(message)

        if not compiled:
            dt.__dict__["_coupling_execution_plan"] = None
        else:
            dt.__dict__.pop("_coupling_execution_plan", None)
            assert dt._coupling_execution_plan is not None
            assert dt._coupling_execution_plan.core.reuse_symbol_reads
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            if handler == "warning":
                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(warnings, "showwarning", mutate)
                    with np.errstate(over="warn", invalid="ignore", under="ignore"):
                        result = dt.evaluate_coupling_batched(cv, 3, {"a": 0})
            else:
                previous = np.geterrcall()
                np.seterrcall(mutate if handler == "call" else Logger())
                try:
                    with np.errstate(over=handler, invalid="ignore", under="ignore"):
                        result = dt.evaluate_coupling_batched(cv, 3, {"a": 0})
                finally:
                    np.seterrcall(previous)
        outcomes.append((result.tobytes(), len(events), cv["B"].tobytes()))
    assert outcomes[0] == outcomes[1] and outcomes[0][1] == 1


@pytest.mark.parametrize(
    "name", ["scalar", "batched", "_bindings", "bind_indices", "caller_label"]
)
@pytest.mark.parametrize("where", ["instance", "class"])
def test_wrapper_execution_overrides_are_not_native(monkeypatch, name, where):
    dt = term()
    plan = dt._coupling_execution_plan
    native = getattr(plan if where == "instance" else cp._CouplingPlan, name)
    if where == "instance":
        object.__setattr__(plan, name, native)
    else:
        monkeypatch.setattr(cp._CouplingPlan, name, lambda *a, **k: native(*a, **k))
    assert not cp.native_plan(plan)
    with np.errstate(all="ignore"):
        assert known_scalar_shape(dt, values(batch=0), {"a": 1}, 4) is None


def test_delta_mutation_is_not_hidden_by_read_memo(monkeypatch):
    expr = Sum(
        tuple(
            Product(
                (Symbol("A", ("i",)), KroneckerDelta("i", "i"), Symbol("A", ("i",)))
            )
            for _ in range(24)
        )
    )
    dt = DiagramTerm(
        (Propagator("C", "i", "i", "x", "y"),), expr, Rational(1), (), (("i", 4),), 0
    )
    plan = dt._coupling_execution_plan
    cv = {"A": np.arange(1.0, 5.0)}
    native = cp._delta
    first = True

    def delta(*args):
        nonlocal first
        if first:
            cv["A"] *= 2
            first = False
        return native(*args)

    monkeypatch.setattr(cp, "_delta", delta)
    with np.errstate(all="ignore"):
        actual = dt.evaluate_coupling_batched(cv, 1)
    assert actual.tolist() == [[94.0, 384.0, 864.0, 1536.0]]
    assert not cp.native_plan(plan) and not cp.native_symbol_reads()


@pytest.mark.parametrize("helper", ["asarray", "geterr", "ndindex", "dispatcher"])
def test_preimport_numpy_hook_is_not_native(helper):
    script = """
import numpy as np
native = np.asarray
def hook(*args, **kwargs):
    return native(*args, **kwargs)
HELPER_PLACEHOLDER
import sft_wick._coupling_plan as cp
with np.errstate(all="ignore"):
    assert not cp.native_symbol_reads()
print("preimport hook declined")
"""
    replacement = (
        "np.asarray = hook"
        if helper == "asarray"
        else (
            "def geterr():\n    return np._core._ufunc_config.geterr()\ngeterr.__module__ = None\nnp.geterr = geterr"
            if helper == "geterr"
            else (
                "native_next = np.ndindex.__next__\ndef custom_next(self):\n    return native_next(self)\nnp.ndindex.__next__ = custom_next"
                if helper == "ndindex"
                else "class _ArrayFunctionDispatcher:\n    __module__ = 'numpy'\n    __wrapped__ = np.iscomplexobj.__wrapped__\n    def __call__(self, *args, **kwargs):\n        raise RuntimeError('custom dispatcher invoked')\nnp.iscomplexobj = _ArrayFunctionDispatcher()"
            )
        )
    )
    script = script.replace("HELPER_PLACEHOLDER", replacement)
    result = subprocess.run(
        [sys.executable, "-B", "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "preimport hook declined" in result.stdout


@pytest.mark.parametrize(
    "helper", ["asarray", "geterr", "geterr_internal", "isfinite", "ndindex"]
)
def test_preimport_rejected_numpy_closure_is_not_retained(helper):
    script = """
import gc
import weakref
import numpy as np
helper = HELPER
owner, name = (np.ndindex, '__next__') if helper == 'ndindex' else (np, helper)
if helper == 'geterr_internal':
    owner, name = np.geterr.__globals__, '_get_extobj_dict'
native = owner[name] if type(owner) is dict else getattr(owner, name)
payload = np.ones(100)
ref = weakref.ref(payload)
def make_hook(data):
    def hook(*args, **kwargs):
        assert data.size == 100
        return native(*args, **kwargs)
    return hook
hook = make_hook(payload)
if type(owner) is dict:
    owner[name] = hook
else:
    setattr(owner, name, hook)
import sft_wick._coupling_plan as cp
import sft_wick._coupling_probe as probe
if type(owner) is dict:
    owner[name] = native
else:
    setattr(owner, name, native)
del payload, hook
gc.collect()
assert ref() is None
""".replace("HELPER", repr(helper))
    result = subprocess.run(
        [sys.executable, "-B", "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_numpy_iterator_method_override_declines_memo(monkeypatch):
    native = np.ndindex.__next__

    def custom(self):
        return native(self)

    monkeypatch.setattr(np.ndindex, "__next__", custom)
    with np.errstate(all="ignore"):
        result, count = reads(
            lambda: term().evaluate_coupling_batched(values(), 4, {"a": 1})
        )
    assert count == 288 and result.shape == (4, 3)


@NATIVE_NDINDEX
def test_probe_uses_original_labels_and_rejects_forged_wrapper():
    dt = term("external", "inner", "outer")
    cv = values(batch=0)
    with np.errstate(all="ignore"):
        assert known_scalar_shape(dt, cv, {"external": 1}, 4) == (3,)
        original = dt._coupling_execution_plan
        dt.__dict__["_coupling_execution_plan"] = replace(original)
        assert known_scalar_shape(dt, cv, {"external": 1}, 4) is None
