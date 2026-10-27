"""Independent numerical and operational gates for ordered coupling lowering."""

import copy
import pickle

import numpy as np
import pytest

from sft_wick._coupling_plan import compile_coupling_plan
from sft_wick.expressions import (
    ImaginaryUnit,
    KroneckerDelta,
    Product,
    Propagator,
    Rational,
    Sum,
    Symbol,
)
from sft_wick.perturbation import DiagramTerm, _eval_symbolic, _eval_symbolic_batched


def large(expr):
    # Neutral scalar factors exercise compiled execution of otherwise tiny cases.
    return Product((Rational(1),) * 16 + (expr,))


def plan(expr):
    result = compile_coupling_plan(large(expr))
    assert result is not None
    return result


def term(expr, n_response=0):
    return DiagramTerm(
        (Propagator("C", "j", "j", "x", "y"),),
        large(expr),
        Rational(-2, 5),
        (),
        (("i", 2), ("j", 3)),
        n_response,
    )


def partial():
    return Sum(
        (
            Product(
                (Rational(1, 2), Symbol("K", ("a", "j", "i")), Symbol("A", ("i", "j")))
            ),
            Product((Rational(-1, 3), Symbol("B", ("j",)), KroneckerDelta("i", "0"))),
        )
    )


def bindings():
    a, j, i = np.indices((2, 3, 2))
    i2, j2 = np.indices((2, 3))
    return {
        "K": 100.0 * a + 10 * j + i,
        "A": 1.0 + 2 * i2 + 3 * j2,
        "B": 3.0 * (np.arange(3) + 1),
    }


@pytest.mark.parametrize("phase", range(4))
@pytest.mark.parametrize("batch", [1, 3, 7])
def test_independent_partial_contraction(phase, batch):
    dt = term(partial(), phase)
    cv = bindings()
    expected = -0.4 * (1.0, -1j, -1.0, 1j)[phase] * np.array([200.5, 551.0, 961.5])
    np.testing.assert_allclose(
        dt.evaluate_coupling(cv, {"a": 1}), expected, rtol=2e-15, atol=2e-13
    )
    cv["K"] = cv["K"][None] + np.arange(batch)[:, None, None, None]
    expected_batch = (
        -0.4
        * (1.0, -1j, -1.0, 1j)[phase]
        * (
            np.array([200.5, 551.0, 961.5])
            + np.arange(batch)[:, None] * [2.0, 5.0, 8.0]
        )
    )
    np.testing.assert_allclose(
        dt.evaluate_coupling_batched(cv, batch, {"a": 1}),
        expected_batch,
        rtol=2e-15,
        atol=2e-13,
    )
    assert dt._coupling_execution_plan is not None


def test_current_values_pins_dtype_and_batch_rank():
    dt = term(partial())
    cv = bindings()
    pins = {"a": 1}
    initial = dt.evaluate_coupling(cv, pins)
    cv["K"] += 2
    expected = initial - 0.8 * np.array([2.0, 5.0, 8.0])
    np.testing.assert_allclose(
        dt.evaluate_coupling(cv, pins), expected, rtol=2e-15, atol=2e-13
    )
    cv["K"] = bindings()["K"] + 1j
    assert dt.evaluate_coupling(cv, pins).dtype == np.complex128
    cv = bindings()
    cv["unused"] = np.array(1j)
    assert dt.evaluate_coupling(cv, pins).dtype == np.complex128
    del cv["unused"]
    assert dt.evaluate_coupling(cv, pins).dtype == np.float64
    pins["a"] = 0
    # K[a=0] subtracts 100; the sum of A/2 is [2,5,8].
    np.testing.assert_allclose(
        dt.evaluate_coupling(cv, pins),
        initial + 40 * np.array([2.0, 5.0, 8.0]),
        rtol=2e-15,
        atol=2e-13,
    )
    for batch in (7, 1, 3):
        np.testing.assert_allclose(
            dt.evaluate_coupling_batched(cv, batch, pins),
            np.broadcast_to(dt.evaluate_coupling(cv, pins), (batch, 3)),
            rtol=2e-15,
            atol=2e-13,
        )


@pytest.mark.parametrize("batched", [False, True])
def test_literal_conventions_and_live_pin_precedence(batched):
    expr = Product((Symbol("A", ("1",)), KroneckerDelta("1", "a")))
    p = plan(expr)
    values = {"A": np.array([3.0, 7.0])}

    def fn(pins):
        return (
            p.batched(expr, values, pins, 1)
            if batched
            else p.scalar(expr, values, pins)
        )

    assert fn({"a": 1}) == 3
    assert fn({"1": 1, "a": 1}) == 7
    assert fn({"a": 0}) == 0


@pytest.mark.parametrize("dtype", [np.float32, np.complex64, np.float64, np.complex128])
def test_distinct_scalar_and_batch_leaf_conversion(dtype):
    arr = np.array(
        0.123456789 + (0.2j if np.issubdtype(dtype, np.complexfloating) else 0),
        dtype=dtype,
    )
    expr = Product((Symbol("A"), Rational(1, 10)))
    p = plan(expr)
    scalar = complex(arr) if np.iscomplexobj(arr) else float(arr)
    assert p.scalar(expr, {"A": arr}, {}) == scalar * 0.1
    np.testing.assert_array_equal(p.batched(expr, {"A": arr}, {}, 3), (1.0 * arr) * 0.1)


@pytest.mark.parametrize("batched", [False, True])
def test_no_reassociation_short_circuit_or_nonfinite_suppression(batched):
    values = {"A": np.array(1e16), "B": np.array(-1e16), "C": np.array(1.0)}
    expr = Sum(tuple(Symbol(x) for x in "ABC"))
    p = plan(expr)
    assert (
        p.batched(expr, values, {}, 1) if batched else p.scalar(expr, values, {})
    ) == 1.0
    expr = Product(tuple(Symbol(x) for x in "ABC"))
    p = plan(expr)
    values = {"A": np.array(1e200), "B": np.array(1e-200), "C": np.array(1e-200)}
    assert (
        p.batched(expr, values, {}, 1) if batched else p.scalar(expr, values, {})
    ) == 1e-200
    expr = Product((Rational(0), Symbol("A")))
    p = plan(expr)
    with pytest.raises(KeyError, match="A"):
        (p.batched(expr, {}, {}, 1) if batched else p.scalar(expr, {}, {}))
    with np.errstate(invalid="ignore"):
        out = (
            p.batched(expr, {"A": np.array(np.inf)}, {}, 1)
            if batched
            else p.scalar(expr, {"A": np.array(np.inf)}, {})
        )
    assert np.isnan(out)


@pytest.mark.parametrize("batched", [False, True])
def test_ordered_equal_delta_errors(batched):
    for labels in (("a", "b"), ("b", "a")):
        expr = KroneckerDelta(*labels)
        p = plan(expr)
        with pytest.raises(KeyError, match=repr(labels[0])):
            (p.batched(expr, {}, {}, 1) if batched else p.scalar(expr, {}, {}))


@pytest.mark.parametrize("batched", [False, True])
def test_unknown_whole_tree_fallback_preserves_first_error(batched):
    unknown = Propagator("C", "i", "j", "x", "y")
    for children in ((Symbol("missing"), unknown), (unknown, Symbol("missing"))):
        expr = large(Sum(children))
        assert compile_coupling_plan(expr) is None
        dt = term(expr)
        expected = (
            KeyError
            if children[0] is not unknown
            else (NotImplementedError if batched else TypeError)
        )
        with pytest.raises(expected):
            (
                dt.evaluate_coupling_batched({}, 3)
                if batched
                else dt.evaluate_coupling({})
            )


def test_custom_nodes_and_metadata_are_not_inspected():
    class CustomSymbol(Symbol):
        def __getattribute__(self, name):
            if name in ("name", "indices"):
                raise AssertionError("compiler inspected custom node")
            return super().__getattribute__(name)

    expr = large(CustomSymbol("A"))
    assert compile_coupling_plan(expr) is None
    assert compile_coupling_plan(large(Symbol("A", ["i"]))) is None


def test_readonly_noncontiguous_input_is_not_modified():
    dt = term(partial())
    cv = bindings()
    cv["K"] = cv["K"][..., ::-1]
    before = {k: v.copy() for k, v in cv.items()}
    for v in cv.values():
        v.flags.writeable = False
    dt.evaluate_coupling(cv, {"a": 1})
    dt.evaluate_coupling_batched(cv, 7, {"a": 1})
    for k in cv:
        np.testing.assert_array_equal(cv[k], before[k])


@pytest.mark.parametrize(
    "restore", [lambda x: pickle.loads(pickle.dumps(x)), copy.deepcopy]
)
def test_compiled_term_serializes_and_reads_new_bindings(restore):
    dt = term(partial(), 3)
    cv = bindings()
    expected = dt.evaluate_coupling(cv, {"a": 1})
    restored = restore(dt)
    assert restored == dt
    np.testing.assert_array_equal(restored.evaluate_coupling(cv, {"a": 1}), expected)
    cv["K"] += 1
    np.testing.assert_array_equal(
        restored.evaluate_coupling(cv, {"a": 1}), dt.evaluate_coupling(cv, {"a": 1})
    )


def test_small_leaf_and_oversized_plans_decline():
    assert compile_coupling_plan(Symbol("A")) is None
    assert compile_coupling_plan(Product((Symbol("A"),))) is None
    assert compile_coupling_plan(Product((Rational(1),) * 129)) is None
    dt = DiagramTerm((), large(Symbol("A")), Rational(1), (), (), 0)
    assert dt._coupling_execution_plan is None
    assert dt.evaluate_coupling({"A": np.array(2.0)}) == 2.0


@pytest.mark.parametrize("batched", [False, True])
def test_nested_products_sums_match_recursive_reference(batched):
    expr = large(
        Product(
            (
                Sum((Symbol("A"), Symbol("B"))),
                Product(
                    (ImaginaryUnit(), Rational(-1, 3), Sum((Symbol("C"), Rational(1))))
                ),
            )
        )
    )
    cv = {"A": np.array(0.1), "B": np.array(0.2), "C": np.array(0.3)}
    p = compile_coupling_plan(expr)
    assert p is not None
    if batched:
        assert p.batched(expr, cv, {}, 1) == _eval_symbolic_batched(expr, cv, {}, 1)
    else:
        assert p.scalar(expr, cv, {}) == _eval_symbolic(expr, cv, {})


def test_scalar_builtin_sum_and_batched_fold_remain_distinct():
    expr = large(Sum(tuple(Symbol(x) for x in "ABC")))
    p = compile_coupling_plan(expr)
    cv = {"A": np.array(1e16), "B": np.array(1.0), "C": np.array(-1e16)}
    scalar_expected = sum(float(cv[x]) for x in "ABC")
    batch_expected = (cv["A"] + cv["B"]) + cv["C"]
    assert p.scalar(expr, cv, {}) == scalar_expected
    assert p.batched(expr, cv, {}, 1) == batch_expected


def test_batched_sum_reads_all_children_before_addition():
    events = []

    class Array(np.ndarray):
        def __add__(self, other):
            events.append("add")
            raise AssertionError("addition ran before all children were read")

    expr = large(Sum((Symbol("A"), Symbol("A"), Symbol("missing"))))
    p = compile_coupling_plan(expr)
    with pytest.raises(KeyError, match="missing"):
        p.batched(expr, {"A": np.ones(3).view(Array)}, {}, 3)
    assert events == []


def test_plan_does_not_keep_numeric_inputs_alive():
    import gc
    import weakref

    dt = term(partial())
    cv = bindings()
    refs = [weakref.ref(v) for v in cv.values()]
    dt.evaluate_coupling(cv, {"a": 1})
    assert dt._coupling_execution_plan is not None
    del cv
    gc.collect()
    assert all(ref() is None for ref in refs)


def _evaluate_worker(dt, cv):
    return dt, dt.evaluate_coupling(cv, {"a": 1})


def test_loky_compiled_term_roundtrip():
    from joblib import Parallel, delayed

    dt = term(partial())
    cv = bindings()
    expected = dt.evaluate_coupling(cv, {"a": 1})
    rows = Parallel(n_jobs=2, backend="loky")(
        delayed(_evaluate_worker)(dt, cv) for _ in range(2)
    )
    for restored, output in rows:
        np.testing.assert_array_equal(output, expected)
        cv2 = bindings()
        cv2["K"] += 2
        np.testing.assert_array_equal(
            restored.evaluate_coupling(cv2, {"a": 1}),
            dt.evaluate_coupling(cv2, {"a": 1}),
        )


def test_long_literal_conversion_uses_current_python_limit():
    import sys

    label = "9" * 700
    expr = large(KroneckerDelta(label, label))
    previous = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(640)
        assert compile_coupling_plan(expr) is None
        dt = term(expr)
        with pytest.raises(KeyError):
            dt.evaluate_coupling({})
        sys.set_int_max_str_digits(4300)
        np.testing.assert_array_equal(dt.evaluate_coupling({}), np.full(3, -0.8))
        np.testing.assert_array_equal(
            dt.evaluate_coupling_batched({}, 3), np.full((3, 3), -0.8)
        )
    finally:
        sys.set_int_max_str_digits(previous)


def test_custom_root_children_are_not_preinspected():
    class Children(tuple):
        def __len__(self):
            raise AssertionError("eligibility inspected custom sequence")

    for node in (Sum, Product):
        expr = node((Symbol("missing"),))
        attribute = "terms" if node is Sum else "factors"
        object.__setattr__(expr, attribute, Children((Symbol("missing"),)))
        assert type(getattr(expr, attribute)) is Children
        dt = term(Rational(1))
        # Adversarial native-node extension: initialization normally flattens tuples.
        dt = DiagramTerm(
            dt.propagators, expr, dt.rational_prefactor, (), dt.summation_indices, 0
        )
        with pytest.raises(KeyError, match="missing"):
            dt.evaluate_coupling({})
        with pytest.raises(KeyError, match="missing"):
            dt.evaluate_coupling_batched({}, 3)
