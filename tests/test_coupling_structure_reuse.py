"""Independent gates for shared cores, inert indices and native probe admission."""

import copy
import json
from pathlib import Path
import subprocess
import gc
import pickle
import sys
import weakref

import numpy as np
import pytest

from sft_wick._coupling_plan import _shared_core, compile_coupling_plan
from sft_wick._coupling_probe import known_scalar_shape
from sft_wick.evaluate import DynamicCouplingPromise
from sft_wick.expressions import (
    ImaginaryUnit,
    KroneckerDelta,
    Product,
    Propagator,
    Rational,
    Sum,
    Symbol,
)
from sft_wick.perturbation import DiagramTerm, _eval_symbolic


def expression():
    pair = (
        Product(
            (Rational(1, 2), Symbol("K", ("a", "j", "i")), Symbol("A", ("i", "j")))
        ),
        Product((Rational(-1, 3), Symbol("B", ("j",)), KroneckerDelta("i", "0"))),
    )
    return Sum(pair * 12)


def term(expr=None, phase=0):
    return DiagramTerm(
        (Propagator("C", "j", "j", "x", "y"),),
        expression() if expr is None else expr,
        Rational(-2, 5),
        (),
        (("i", 2), ("j", 3)),
        phase,
    )


def values(dtype=np.float64):
    a, j, i = np.indices((2, 3, 2))
    i2, j2 = np.indices((2, 3))
    return {
        "K": (100 * a + 10 * j + i).astype(dtype),
        "A": (1 + 2 * i2 + 3 * j2).astype(dtype),
        "B": (3 * (np.arange(3) + 1)).astype(dtype),
    }


def scalar_oracle(dt, cv, pins):
    # Explicit outer j / inner i contraction, separate from plan dispatch/binding.
    result = []
    dtype = (
        complex
        if dt.n_response % 2 or any(np.iscomplexobj(x) for x in cv.values())
        else float
    )
    for j in range(3):
        result.append(
            sum(
                dtype(_eval_symbolic(dt.coupling_sum, cv, {**pins, "j": j, "i": i}))
                for i in range(2)
            )
        )
    return -0.4 * (1.0, -1j, -1.0, 1j)[dt.n_response % 4] * np.array(result)


def promise(dt, cv, dtype=None, later_large=False, vectorized=True):
    counts = []
    k = cv.pop("K")

    def kernel(x, t):
        counts.append(t.shape)
        if vectorized:
            out = np.broadcast_to(k, (t.shape[1],) + k.shape).copy()
            if later_large:
                out[1:] = np.finfo(dtype).max
            return out
        return k.copy()

    kernel.vectorized = vectorized
    p = DynamicCouplingPromise(dt, cv, {"K": kernel}, {"K": ("x",)}, {"a": 1})
    return p, counts


def run(p, batch=3):
    return p.evaluate_at_batch({"x": np.linspace(0.1, 0.8, batch)}, {"x": 0.0}, batch)


def scalar_calls(fn):
    count = 0
    code = DiagramTerm.evaluate_coupling.__code__

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


def padded(expr):
    return Product((Rational(1),) * 16 + (expr,))


def test_dynamic_probe_does_not_invoke_replaced_isfinite(monkeypatch):
    dt = term()
    p, calls = promise(dt, values())
    expected = run(p)
    p, calls = promise(dt, values())
    hooks = []

    def isfinite(*args, **kwargs):
        hooks.append(1)
        raise RuntimeError("probe-only hook")

    monkeypatch.setattr(np, "isfinite", isfinite)
    actual = run(p)
    assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
    assert hooks == [] and calls == [(1, 3)]


def test_dynamic_probe_declines_geterr_internal_hook_before_call(monkeypatch):
    from sft_wick import _coupling_probe as probe

    outcomes = []
    for enabled in (False, True):
        with monkeypatch.context() as context:
            cv = values()
            p, calls = promise(term(), cv)
            hooks = []
            native = np.geterr.__globals__["_get_extobj_dict"]

            def getter():
                hooks.append(1)
                cv["A"] += 1
                return native()

            if not enabled:
                context.setattr(probe, "known_scalar_shape", lambda *_: None)
            context.setitem(np.geterr.__globals__, "_get_extobj_dict", getter)
            result = run(p)
            outcomes.append((result.dtype, result.tobytes(), hooks, calls))
    assert outcomes[0] == outcomes[1] and outcomes[0][2] == []


def test_dynamic_batch_zero_hook_preserves_sample_zero_seed(monkeypatch):
    from sft_wick import _coupling_probe as probe

    outcomes = []
    for enabled in (False, True):
        with monkeypatch.context() as context:
            cv = values()
            p, calls = promise(term(), cv)
            native = np.zeros
            hooks = []

            def zeros(shape, *args, **kwargs):
                if shape == (3, 3) and not hooks:
                    hooks.append(1)
                    cv["A"] *= 2
                    raise NotImplementedError("batch fallback after seed")
                return native(shape, *args, **kwargs)

            if not enabled:
                context.setattr(probe, "known_scalar_shape", lambda *_: None)
            context.setattr(np, "zeros", zeros)
            result = run(p)
            outcomes.append((result.dtype, result.tobytes(), hooks, calls))
    assert outcomes[0] == outcomes[1] and outcomes[0][2] == [1]


def test_dynamic_ndindex_constructor_mutation_preserves_scalar_first():
    # Changing __new__ mutates CPython's tp_new slot even after deleting the
    # field. Isolate each mode so this probe cannot damage the test process.
    outcomes = []
    for enabled in (False, True):
        script = """
import json
import sys
import numpy as np
sys.path.insert(0, TEST_UTILS)
from test_coupling_structure_reuse import term, values, promise, run
from sft_wick import _coupling_probe as probe
cv = values()
p, _ = promise(term(), cv)
hooks = []
def new(cls, *args):
    hooks.append(1)
    cv['A'] *= 2
    return object.__new__(cls)
if not ENABLED:
    probe.known_scalar_shape = lambda *_: None
np.ndindex.__new__ = staticmethod(new)
result = run(p)
print(json.dumps([result.dtype.str, result.tobytes().hex(), len(hooks)]))
""".replace("TEST_UTILS", repr(str(Path(__file__).parent))).replace(
            "ENABLED", repr(enabled)
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        outcomes.append(json.loads(result.stdout))
    assert outcomes[0] == outcomes[1] and outcomes[0][2] > 1


def test_exact_core_sharing_preserves_wrapper_expression():
    _shared_core.cache_clear()
    first, second = expression(), expression()
    a, b = compile_coupling_plan(first), compile_coupling_plan(second)
    assert a is not b and a.expression is first and b.expression is second
    assert a.core is b.core
    assert _shared_core.cache_info().hits == 1
    for restore in (copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))):
        restored = restore(b)
        assert restored.expression is not first and restored.core is a.core


@pytest.mark.parametrize(
    "other",
    [
        KroneckerDelta("b", "a"),
        Symbol("A", ("i",), ("y",), False),
        Symbol("A", ("i",), ("x",), True),
        Product((Symbol("B"), Symbol("A"))),
    ],
)
def test_ordered_metadata_never_uses_expr_equality(other):
    reference = (
        KroneckerDelta("a", "b")
        if isinstance(other, KroneckerDelta)
        else (
            Symbol("A", ("i",), ("x",), False)
            if isinstance(other, Symbol)
            else Product((Symbol("A"), Symbol("B")))
        )
    )
    left = compile_coupling_plan(padded(reference))
    right = compile_coupling_plan(padded(other))
    if isinstance(other, KroneckerDelta):
        # Alpha cores may share, while caller witnesses preserve operand order.
        assert left.core is right.core
        assert left.labels == ("a", "b") and right.labels == ("b", "a")
        for plan, missing in ((left, "a"), (right, "b")):
            with pytest.raises(KeyError, match=f"index '{missing}'"):
                plan.scalar(plan.expression, {}, {})
    else:
        assert left.core is not right.core


def test_custom_spatial_metadata_is_not_hashed_or_compared():
    class Explosive:
        def __hash__(self):
            raise AssertionError("custom hash")

        def __eq__(self, other):
            raise AssertionError("custom equality")

    expr = padded(Symbol("A", (), (Explosive(),)))
    a, b = compile_coupling_plan(expr), compile_coupling_plan(expr)
    assert a.core is not b.core
    assert a.scalar(expr, {"A": np.array(3.0)}, {}) == 3


def test_cache_bounds_and_no_expression_numeric_retention():
    _shared_core.cache_clear()
    expr = padded(Symbol("A"))
    ref = weakref.ref(expr)
    p = compile_coupling_plan(expr)
    arr = np.array(2.0)
    arr_ref = weakref.ref(arr)
    assert p.scalar(expr, {"A": arr}, {}) == 2
    del p, expr, arr
    gc.collect()
    assert ref() is None and arr_ref() is None
    for i in range(40):
        compile_coupling_plan(padded(Symbol(str(i))))
    assert _shared_core.cache_info().currsize == 32
    for expr in (padded(Symbol("x" * 17000)), padded(Rational(2**5000))):
        assert compile_coupling_plan(expr).core is not compile_coupling_plan(expr).core


@pytest.mark.parametrize("phase", range(4))
@pytest.mark.parametrize("dtype", [np.float32, np.complex64, np.float64, np.complex128])
def test_independent_mutable_indices_and_values(phase, dtype):
    dt, cv, pins = term(phase=phase), values(dtype), {"a": 1, "i": 100, "j": 100}
    assert dt._coupling_execution_plan.core.reuse_indices
    np.testing.assert_array_equal(
        dt.evaluate_coupling(cv, pins), scalar_oracle(dt, cv, pins)
    )
    # Free loops override pins, and changed fixed pins/current arrays stay live.
    pins["a"] = 0
    cv["K"] = cv["K"] + dtype(0.25)
    np.testing.assert_array_equal(
        dt.evaluate_coupling(cv, pins), scalar_oracle(dt, cv, pins)
    )
    if dtype in (np.float64, np.complex128):
        np.testing.assert_allclose(
            dt.evaluate_coupling_batched(cv, 3, pins),
            np.broadcast_to(scalar_oracle(dt, cv, pins), (3, 3)),
            rtol=2e-15,
            atol=2e-12,
        )


@pytest.mark.parametrize("batched", [False, True])
def test_missing_binding_remains_after_value_lookup(batched):
    expr = Sum(
        tuple(
            Product((Symbol("missing", ("z",)), Symbol("A", ("i",)))) for _ in range(24)
        )
    )
    dt = term(expr)
    cv = {"A": np.ones(2)}

    def fn():
        return (
            dt.evaluate_coupling_batched(cv, 3) if batched else dt.evaluate_coupling(cv)
        )

    with pytest.raises(KeyError, match="missing"):
        fn()
    cv["missing"] = np.ones(2)
    with pytest.raises(KeyError, match="Index 'z'"):
        fn()


@pytest.mark.parametrize("custom", ["mapping", "key", "pin", "array", "object"])
def test_custom_hooks_decline_binding(custom):
    from sft_wick._coupling_plan import native_bindings

    cv, pins = values(), {"a": 1}

    class CustomDict(dict):
        pass

    class CustomStr(str):
        pass

    class CustomInt(int):
        pass

    class CustomArray(np.ndarray):
        pass

    if custom == "mapping":
        cv = CustomDict(cv)
    elif custom == "key":
        pins = {CustomStr("a"): 1}
    elif custom == "pin":
        pins = {"a": CustomInt(1)}
    elif custom == "array":
        cv["K"] = cv["K"].view(CustomArray)
    else:
        cv["K"] = cv["K"].astype(object)
    assert not native_bindings(cv, pins)
    np.testing.assert_array_equal(
        term().evaluate_coupling(cv, pins), scalar_oracle(term(), cv, pins)
    )


@pytest.mark.parametrize("vectorized", [False, True])
def test_successful_probe_removed_and_callbacks_unchanged(vectorized):
    cv = values()
    dt = term()
    expected = -0.4 * 12 * np.array([200.5, 551, 961.5])
    p, counts = promise(dt, cv, vectorized=vectorized)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        out, count = scalar_calls(lambda: run(p))
    assert count == 0
    assert len(counts) == (1 if vectorized else 3)
    np.testing.assert_allclose(
        out, np.broadcast_to(expected, (3, 3)), rtol=2e-15, atol=2e-12
    )
    p.fixed_indices["a"] = 0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        out, count = scalar_calls(lambda: run(p, 7))
    assert count == 0 and out.shape == (7, 3)


@pytest.mark.parametrize("mode", ["raise", "warn", "call", "print", "log"])
def test_nonignore_underflow_keeps_probe(mode):
    dt, cv = term(), values()
    with np.errstate(under=mode):
        assert known_scalar_shape(dt, cv, {"a": 1}, 3) is None


@pytest.mark.parametrize(
    "invalid",
    [
        "nan",
        "inf",
        "pin",
        "missing",
        "shape",
        "static_batch",
        "huge",
        "imag",
        "zero_dim",
        "custom_array",
    ],
)
def test_invalid_or_custom_input_keeps_scalar_first(invalid):
    dt, cv, pins = term(), values(), {"a": 1}
    if invalid in ("nan", "inf"):
        cv["K"][0, 0, 0] = np.nan if invalid == "nan" else np.inf
    elif invalid == "pin":
        pins["a"] = 2
    elif invalid == "missing":
        del cv["A"]
    elif invalid == "shape":
        cv["A"] = np.ones((2, 3, 2))
    elif invalid == "static_batch":
        dt = term(padded(Symbol("A")))
        cv["A"] = np.ones(3)
    elif invalid == "huge":
        dt = term(padded(Rational(2**5000)))
    elif invalid == "imag":
        dt = term(padded(ImaginaryUnit()))
    elif invalid == "zero_dim":
        dt = DiagramTerm(
            dt.propagators,
            dt.coupling_sum,
            dt.rational_prefactor,
            (),
            (("i", 0), ("j", 3)),
            0,
        )
    else:

        class CustomArray(np.ndarray):
            pass

        cv["K"] = cv["K"].view(CustomArray)
    assert known_scalar_shape(dt, cv, pins, 3) is None


@pytest.mark.parametrize(
    "method",
    [
        "evaluate_coupling",
        "evaluate_coupling_batched",
        "response_phase_factor",
        "propagator_indices",
        "coupling_only_indices",
    ],
)
def test_method_overrides_decline_without_calling_them(method, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("eager override")

    monkeypatch.setattr(DiagramTerm, method, forbidden)
    assert known_scalar_shape(term(), values(), {"a": 1}, 3) is None


def test_exact_scalar_fallback_for_low_precision_with_external_handler():
    # Batchfloat32 overflows only in later samples; scalar Pythonfloat64 does not.
    dt, cv = term(), values(np.float32)
    baseline = scalar_oracle(dt, cv, {"a": 1})
    p, counts = promise(dt, cv, np.float32, later_large=True)

    def refuse_batch(err, flag):
        assert (
            known_scalar_shape(
                dt,
                {**p.static_values, "K": values(np.float32)["K"]},
                p.fixed_indices,
                3,
            )
            is None
        )
        p.static_values["A"] *= 2
        raise NotImplementedError("force native arithmetic fallback")

    with np.errstate(over="call"), np.errstate(call=refuse_batch):
        out, count = scalar_calls(lambda: run(p))
    assert count == 3 and len(counts) == 1
    np.testing.assert_array_equal(out[0], baseline)
    cv_later = values(np.float32)
    cv_later["K"].fill(np.finfo(np.float32).max)
    cv_later["A"] *= 2
    np.testing.assert_array_equal(out[1], scalar_oracle(dt, cv_later, {"a": 1}))
    np.testing.assert_array_equal(out[2], out[1])


def test_instance_plan_override_declines():
    dt = term()
    dt.__dict__["_coupling_execution_plan"] = object()
    assert known_scalar_shape(dt, values(), {"a": 1}, 3) is None


def test_empty_spatial_metadata_cannot_bypass_cache_bound():
    expr = padded(Symbol("A", (), ("",) * 100000))
    assert compile_coupling_plan(expr).core is not compile_coupling_plan(expr).core


@pytest.mark.parametrize("mode", ["warn", "raise", "call", "log"])
def test_signaling_nan_declines_without_cast_warning(mode):
    import warnings

    arr = np.array([0x7F800001] * 4, dtype=np.uint32).view(np.float32)
    dt = DiagramTerm((), padded(Symbol("A", ("i",))), Rational(1), (), (("i", 4),), 0)
    with np.errstate(invalid=mode), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert known_scalar_shape(dt, {"A": arr}, {}, 3) is None
    assert not caught


def test_custom_native_core_and_legacy_cached_plan_keep_scalar_first():
    from dataclasses import replace
    from types import SimpleNamespace
    from sft_wick._coupling_plan import _CouplingPlan

    def forbidden(*args):
        raise RuntimeError("custom scalar")

    for custom in ("native", "legacy"):
        dt = term()
        original = dt._coupling_execution_plan
        dt.__dict__["_coupling_execution_plan"] = (
            _CouplingPlan(dt.coupling_sum, replace(original.core, scalar=forbidden))
            if custom == "native"
            else SimpleNamespace(scalar=forbidden, batched=original.batched)
        )
        assert known_scalar_shape(dt, values(), {"a": 1}, 3) is None
        with pytest.raises(RuntimeError, match="custom scalar"):
            dt.evaluate_coupling(values(), {"a": 1})


def test_unallocatable_dimensions_decline_without_overflow():
    dt = DiagramTerm((), padded(Rational(1)), Rational(1), (), (("i", 10**500),), 0)
    assert known_scalar_shape(dt, {}, {}, 3) is None


def test_large_pinned_tensor_declines_before_finiteness_scan(monkeypatch):
    dt, cv = term(), values()
    cv["K"] = np.broadcast_to(cv["K"][:1], (1000000, 3, 2))

    def forbidden(*args, **kwargs):
        raise AssertionError("large array scanned")

    monkeypatch.setattr(np, "isfinite", forbidden)
    assert known_scalar_shape(dt, cv, {"a": 1}, 3) is None


def test_unallocatable_scalar_shape_declines():
    dt = DiagramTerm(
        (Propagator("C", "i", "i", "x", "y"),),
        padded(Symbol("A")),
        Rational(1),
        (),
        (("i", 2**61),),
        0,
    )
    assert known_scalar_shape(dt, {"A": np.array(1.0)}, {}, 3) is None


@pytest.mark.parametrize("target", ["showwarning", "_showwarnmsg_impl", "_get_context"])
def test_warning_hook_mutation_preserves_original_sample_zero(target, monkeypatch):
    import warnings

    dt, cv = term(), values(np.float32)
    baseline = scalar_oracle(dt, cv, {"a": 1})
    p, counts = promise(dt, cv, np.float32, later_large=True)

    def hook(*args, **kwargs):
        p.static_values["A"] *= 2
        p.fixed_indices["a"] = 0
        raise NotImplementedError("warning hook fallback")

    if not hasattr(warnings, target):
        pytest.skip("warning context helper only exists in Python3.14+")
    monkeypatch.setattr(warnings, target, hook)
    with np.errstate(over="warn"):
        out = run(p)
    assert len(counts) == 1
    np.testing.assert_array_equal(out[0], baseline)


def test_default_policy_optimization_in_fresh_process():
    import os
    import subprocess
    from pathlib import Path

    script = """
import sys
import numpy as np
from sft_wick._coupling_probe import _inert_float_errors
from sft_wick.expressions import Product,Symbol,Rational,Propagator
from sft_wick.perturbation import DiagramTerm
from sft_wick.evaluate import DynamicCouplingPromise
assert _inert_float_errors()
dt=DiagramTerm((Propagator("C","i","i","x","y"),),
 Product((Rational(1),)*16+(Symbol("A",("i",)),)),Rational(1),(),(("i",4),),0)
p=DynamicCouplingPromise(dt,{"A":np.arange(4.)},{},{},{})
calls=[]
def trace(frame,event,arg):
 if event=="call" and frame.f_code is DiagramTerm.evaluate_coupling.__code__: calls.append(1)
sys.setprofile(trace)
try: out=p.evaluate_at_batch({},{},3)
finally: sys.setprofile(None)
assert not calls
np.testing.assert_array_equal(out,np.broadcast_to(np.arange(4.),(3,4)))
"""
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
