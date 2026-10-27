"""Original-factor candidate contracts, without weakening public arrays."""

from dataclasses import FrozenInstanceError, replace
import pickle
from types import SimpleNamespace
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from sft_wick._factorized_coupling import (
    factorized_state,
    native_factorized_integrand,
    share_factorized_owner,
    try_build_factorized,
)
from sft_wick.evaluate import DiagramIntegrand, analyze_spatial
from sft_wick.expressions import (
    ImaginaryUnit,
    KroneckerDelta,
    Product,
    Propagator,
    Rational,
    Sum,
    Symbol,
)
from sft_wick.perturbation import DiagramTerm

LIMIT = 64 * 1024**2


def _case(n=2, samples=4, e_psi=0, matrix_r=False):
    rng = np.random.default_rng(20261013 + n)
    labels = ("i", "j", "k", "a", "b", "c")
    props = [
        Propagator("C", "i", "a", "x", "y"),
        Propagator("C", "j", "b", "x", "y"),
        Propagator("C", "k", "c", "x", "y"),
    ]
    if matrix_r:
        props.append(Propagator("R", "i", "p", "x", "z"))
    term = DiagramTerm(
        tuple(props),
        Product((Symbol("F", labels[:3]), Symbol("G", labels[3:]))),
        Rational(3, 7),
        (),
        tuple((label, n) for label in labels),
        int(matrix_r),
        n_external_response=e_psi,
    )
    cv = {
        "F": rng.uniform(0.2, 0.8, (n, n, n)) * ((1j) ** (int(matrix_r) - e_psi)),
        "G": rng.uniform(0.3, 0.9, (n, n, n)),
    }
    fixed = {"p": n - 1}
    baseline = term.build_integrand(cv, fixed)
    candidate = try_build_factorized(term, analyze_spatial(term), cv, fixed, LIMIT)
    cb = [
        (rng.uniform(0.2, 0.9, (samples, n, n)), *pair)
        for pair in (("i", "a"), ("j", "b"), ("k", "c"))
    ]
    rb = (
        [
            (
                rng.uniform(0.2, 0.8, (samples, n, n)),
                "i",
                "p",
                np.ones((n, n), dtype=bool),
            )
        ]
        if matrix_r
        else []
    )
    rp = rng.uniform(0.3, 0.9, samples)
    jac = rng.uniform(0.3, 0.8, samples)
    return term, cv, baseline, candidate, cb, rp, jac, rb


def _stored(candidate):
    state = dict(object.__getattribute__(candidate, "__dict__"))
    if "_factorized_owner" in state:
        array, snapshot = factorized_state(candidate)
        if array is None:
            state.pop("coupling_array", None)
        else:
            state["coupling_array"] = array
        state["_factorized_plan"] = snapshot
    return state


def _values(integrand, cb, rp, jac, rb=()):
    return integrand._static_values(cb, rp, jac, r_batches=rb)


@pytest.mark.parametrize("n", [2, 8])
@pytest.mark.parametrize("e_psi", [0, 1, 2, 3])
@pytest.mark.parametrize("matrix_r", [False, True])
def test_original_factors_match_ordered_without_dense_coefficient(n, e_psi, matrix_r):
    _, _, baseline, candidate, cb, rp, jac, rb = _case(
        n=n, e_psi=e_psi, matrix_r=matrix_r
    )
    assert candidate is not None and native_factorized_integrand(candidate)
    assert _stored(candidate)["coupling_array"].shape == ()
    result = _values(candidate, cb, rp, jac, rb)
    np.testing.assert_allclose(
        result, _values(baseline, cb, rp, jac, rb), rtol=1e-12, atol=1e-14
    )
    assert _stored(candidate)["coupling_array"].shape == ()
    assert _stored(candidate)["_factorized_plan"] is not None


@pytest.mark.parametrize("unused_complex", [False, True])
@pytest.mark.parametrize("matrix_r", [False, True])
def test_public_materialization_is_exact_original_dtype_and_bytes(
    unused_complex, matrix_r
):
    term, cv, _, _, _, _, _, _ = _case(matrix_r=matrix_r)
    if unused_complex:
        cv["unused"] = np.array(1j, dtype=np.complex128)
    baseline = term.build_integrand(cv, {"p": 1})
    candidate = try_build_factorized(term, analyze_spatial(term), cv, {"p": 1}, LIMIT)
    materialized = candidate.coupling_array
    assert type(materialized) is np.ndarray
    assert materialized.dtype.type is baseline.coupling_array.dtype.type
    assert materialized.tobytes() == baseline.coupling_array.tobytes()
    assert candidate.coupling_array is materialized
    assert _stored(candidate)["_factorized_plan"] is None


def test_snapshots_do_not_borrow_inputs_and_late_fallback_does_not_reevaluate(
    monkeypatch,
):
    term, cv, baseline, candidate, cb, rp, jac, rb = _case()
    original = baseline.coupling_array.copy()
    cv["F"] *= 4
    cv["G"][:] = 7

    def forbidden(*args, **kwargs):
        raise AssertionError("snapshot fallback must not re-evaluate live term")

    monkeypatch.setattr(DiagramTerm, "evaluate_coupling", forbidden)
    # Force a safe runtime refusal without changing the materialization budget.
    cb[0] = (np.broadcast_to(np.eye(2), cb[0][0].shape), None, None)
    result = _values(candidate, cb, rp, jac, rb)
    assert candidate.coupling_array.tobytes() == original.tobytes()
    np.testing.assert_array_equal(result, _values(baseline, cb, rp, jac, rb))


def test_materialized_array_mutations_are_observed_by_ordered_fallback():
    _, _, baseline, candidate, cb, rp, jac, rb = _case()
    candidate.coupling_array[:] *= 3
    changed = replace(baseline, coupling_array=candidate.coupling_array.copy())
    np.testing.assert_array_equal(
        _values(candidate, cb, rp, jac, rb), _values(changed, cb, rp, jac, rb)
    )


def test_pickle_stays_lazy_and_owned_materialization_is_independent():
    _, _, baseline, candidate, cb, rp, jac, rb = _case()
    clone = pickle.loads(pickle.dumps(candidate))
    assert _stored(candidate)["coupling_array"].shape == ()
    assert _stored(clone)["coupling_array"].shape == ()
    np.testing.assert_array_equal(
        _values(candidate, cb, rp, jac, rb), _values(clone, cb, rp, jac, rb)
    )
    assert _stored(clone)["coupling_array"].shape == ()
    clone.coupling_array[:] = 2
    np.testing.assert_array_equal(candidate.coupling_array, baseline.coupling_array)
    assert not np.shares_memory(candidate.coupling_array, clone.coupling_array)


def test_build_time_coefficient_pins_and_live_propagator_pins():
    term, cv, _, _, cb, rp, jac, rb = _case(matrix_r=True)
    expr = Product((Symbol("F", ("p", "j", "k")), Symbol("G", ("a", "b", "c"))))
    term = replace(term, coupling_sum=expr)
    baseline = term.build_integrand(cv, {"p": 0})
    candidate = try_build_factorized(term, analyze_spatial(term), cv, {"p": 0}, LIMIT)
    candidate.fixed_indices["p"] = 1
    baseline.fixed_indices["p"] = 1
    np.testing.assert_allclose(
        _values(candidate, cb, rp, jac, rb),
        _values(baseline, cb, rp, jac, rb),
        rtol=1e-12,
        atol=1e-14,
    )
    assert _stored(candidate)["coupling_array"].shape == ()
    assert candidate.coupling_array.tobytes() == baseline.coupling_array.tobytes()


@pytest.mark.parametrize("kind", ["nested_sum", "delta", "literal", "coupling_only"])
def test_unsupported_expressions_decline_before_coefficient_build(kind):
    term, cv, _, _, _, _, _, _ = _case()
    expr = term.coupling_sum
    if kind == "nested_sum":
        expr = Product((Sum((expr, expr)), Symbol("G", ("a", "b", "c"))))
    elif kind == "delta":
        expr = Product((expr, KroneckerDelta("i", "j")))
    elif kind == "literal":
        expr = Product((Symbol("F", ("1", "j", "k")), Symbol("G", ("a", "b", "c"))))
    else:
        term = replace(term, summation_indices=term.summation_indices + (("q", 2),))
    term = replace(term, coupling_sum=expr)
    assert try_build_factorized(term, analyze_spatial(term), cv, {}, LIMIT) is None


@pytest.mark.parametrize(
    "kind",
    [
        "zero",
        "nan",
        "inf",
        "mixed_phase",
        "float32",
        "extended",
        "subclass",
        "callable",
    ],
)
def test_unsupported_numeric_couplings_decline(kind):
    term, cv, _, _, _, _, _, _ = _case()
    arr = cv["F"].real.copy()
    if kind in ("zero", "nan", "inf"):
        arr.flat[0] = {"zero": 0.0, "nan": np.nan, "inf": np.inf}[kind]
    elif kind == "mixed_phase":
        arr = arr.astype(np.complex128)
        arr.flat[0] += 1j
    elif kind == "float32":
        arr = arr.astype(np.float32)
    elif kind == "extended":
        arr = arr.astype(np.longdouble)
    elif kind == "subclass":

        class Array(np.ndarray):
            pass

        arr = arr.view(Array)
    else:

        def arr(*args):
            return np.ones((2, 2, 2))

    cv["F"] = arr
    assert try_build_factorized(term, analyze_spatial(term), cv, {}, LIMIT) is None


@pytest.mark.parametrize("scale", [1e308, 1e-300, 1e-200])
def test_extreme_subset_product_range_declines(scale):
    term, cv, _, _, _, _, _, _ = _case()
    cv["F"] = np.full((2, 2, 2), scale)
    cv["G"] = np.full((2, 2, 2), scale)
    assert try_build_factorized(term, analyze_spatial(term), cv, {}, LIMIT) is None


def test_budget_build_decline_and_runtime_fallback():
    term, cv, baseline, candidate, cb, rp, jac, rb = _case()
    with pytest.raises(ValueError, match="factorized coefficient materialization"):
        try_build_factorized(term, analyze_spatial(term), cv, {}, 1)
    object.__setattr__(candidate, "_contraction_max_bytes", 1)
    with pytest.raises(ValueError, match="factorized coefficient materialization"):
        _values(candidate, cb, rp, jac, rb)
    assert _stored(candidate)["_factorized_plan"] is not None
    object.__setattr__(candidate, "_contraction_max_bytes", LIMIT)
    assert candidate.coupling_array.tobytes() == baseline.coupling_array.tobytes()
    assert _stored(candidate)["_factorized_plan"] is None


@pytest.mark.parametrize(
    "kind", ["diagonal", "trace", "sparse_r", "nonfinite", "strict_err"]
)
def test_runtime_refusals_materialize_and_preserve_ordered_result(kind):
    _, _, baseline, candidate, cb, rp, jac, rb = _case(matrix_r=True)
    if kind == "diagonal":
        cb[0] = (cb[0][0][:, 0, :], "i", "a")
    elif kind == "trace":
        cb[0] = (np.broadcast_to(np.eye(2), cb[0][0].shape), None, None)
    elif kind == "sparse_r":
        rb[0][3][0, 0] = False
    elif kind == "nonfinite":
        cb[0][0][0, 0, 0] = np.inf
    if kind == "strict_err":
        with np.errstate(over="raise"):
            got = _values(candidate, cb, rp, jac, rb)
            expected = _values(baseline, cb, rp, jac, rb)
    else:
        got = _values(candidate, cb, rp, jac, rb)
        expected = _values(baseline, cb, rp, jac, rb)
    np.testing.assert_array_equal(got, expected)
    assert _stored(candidate)["_factorized_plan"] is None


def test_instance_protocol_override_and_base_class_override_decline(monkeypatch):
    _, _, baseline, candidate, cb, rp, jac, rb = _case()
    object.__setattr__(candidate, "_c_product", lambda *args: np.zeros(rp.size))
    assert not native_factorized_integrand(candidate)
    np.testing.assert_array_equal(
        _values(candidate, cb, rp, jac, rb), np.zeros(rp.size)
    )
    _, _, baseline, candidate, cb, rp, jac, rb = _case()
    monkeypatch.setattr(
        DiagramIntegrand, "_c_product", staticmethod(lambda *args: np.zeros(rp.size))
    )
    assert not native_factorized_integrand(candidate)
    np.testing.assert_array_equal(
        _values(candidate, cb, rp, jac, rb), np.zeros(rp.size)
    )


def test_nested_products_and_scalar_factors_materialize_exactly():
    term, cv, _, _, cb, rp, jac, rb = _case()
    inner = Product((Rational(2, 9), Symbol("F", ("i", "j", "k"))))
    outer = Product(())
    object.__setattr__(
        outer, "factors", (ImaginaryUnit(), inner, Symbol("G", ("a", "b", "c")))
    )
    term = replace(term, coupling_sum=outer, n_external_response=3)
    baseline = term.build_integrand(cv)
    candidate = try_build_factorized(term, analyze_spatial(term), cv, {}, LIMIT)
    assert candidate is not None
    np.testing.assert_allclose(
        _values(candidate, cb, rp, jac, rb),
        _values(baseline, cb, rp, jac, rb),
        rtol=1e-12,
        atol=1e-14,
    )
    assert candidate.coupling_array.tobytes() == baseline.coupling_array.tobytes()


def test_invalid_index_free_c_has_identical_ordered_error():
    _, _, baseline, candidate, cb, rp, jac, rb = _case()
    cb[0] = (cb[0][0], None, None)
    with pytest.raises(ValueError) as original:
        _values(baseline, cb, rp, jac, rb)
    with pytest.raises(ValueError) as factored:
        _values(candidate, cb, rp, jac, rb)
    assert str(factored.value) == str(original.value)


@pytest.mark.parametrize("throws", [False, True])
def test_preexisting_custom_cached_scalar_plan_retains_legacy_behavior(throws):
    term, cv, _, _, _, _, _, _ = _case()
    term = replace(
        term,
        coupling_sum=Product((Rational(1), Rational(1), *term.coupling_sum.factors)),
    )
    calls = []

    def scalar(*args):
        calls.append(args)
        if throws:
            raise RuntimeError("custom cached scalar plan")
        return 5

    object.__setattr__(term, "_coupling_execution_plan", SimpleNamespace(scalar=scalar))
    assert try_build_factorized(term, analyze_spatial(term), cv, {}, LIMIT) is None
    assert calls == []
    if throws:
        with pytest.raises(RuntimeError, match="custom cached scalar plan"):
            term.build_integrand(cv)
    else:
        actual = term.build_integrand(cv).coupling_array
        np.testing.assert_array_equal(actual, np.full((2,) * 6, (3 / 7) * 5))
    assert calls


def test_custom_compiler_is_not_bypassed(monkeypatch):
    from sft_wick import perturbation as pt

    term, cv, _, _, _, _, _, _ = _case()
    term = replace(
        term,
        coupling_sum=Product((Rational(1), Rational(1), *term.coupling_sum.factors)),
    )
    calls = []

    def compiler(expr):
        calls.append(expr)
        return SimpleNamespace(scalar=lambda *args: 7)

    monkeypatch.setattr(pt, "compile_coupling_plan", compiler)
    assert try_build_factorized(term, analyze_spatial(term), cv, {}, LIMIT) is None
    assert calls == []
    np.testing.assert_array_equal(
        term.build_integrand(cv).coupling_array, np.full((2,) * 6, (3 / 7) * 7)
    )
    assert calls == [term.coupling_sum]


def test_cached_native_plan_also_stays_on_original_path():
    from sft_wick._coupling_plan import compile_coupling_plan, native_plan

    term, cv, _, _, _, _, _, _ = _case()
    term = replace(
        term,
        coupling_sum=Product(
            (*(Rational(1) for _ in range(16)), *term.coupling_sum.factors)
        ),
    )
    cached = compile_coupling_plan(term.coupling_sum)
    assert cached is not None and native_plan(cached)
    object.__setattr__(term, "_coupling_execution_plan", cached)
    assert try_build_factorized(term, analyze_spatial(term), cv, {}, LIMIT) is None


def test_cached_none_does_not_disable_native_factorization():
    term, cv, _, _, _, _, _, _ = _case()
    term = replace(
        term,
        coupling_sum=Product((Rational(1), Rational(1), *term.coupling_sum.factors)),
    )
    baseline = term.build_integrand(cv)
    assert term._coupling_execution_plan is None
    candidate = try_build_factorized(term, analyze_spatial(term), cv, {}, LIMIT)
    assert candidate is not None
    assert candidate.coupling_array.tobytes() == baseline.coupling_array.tobytes()


def test_dataclass_replace_materialized_subclass_evaluates_and_pickles():
    _, _, baseline, candidate, cb, rp, jac, rb = _case()
    clone = replace(candidate)
    assert "_factorized_plan" not in _stored(clone)
    assert not native_factorized_integrand(clone)
    np.testing.assert_array_equal(
        _values(clone, cb, rp, jac, rb), _values(baseline, cb, rp, jac, rb)
    )
    restored = pickle.loads(pickle.dumps(clone))
    assert not native_factorized_integrand(restored)
    np.testing.assert_array_equal(
        _values(restored, cb, rp, jac, rb), _values(baseline, cb, rp, jac, rb)
    )


def test_large_six_axis_materialization_and_fallback_fail_before_allocation(
    monkeypatch,
):
    import sft_wick._factorized_coupling as helper
    from sft_wick import evaluate as ev

    n = 32
    labels = ("i", "j", "k", "a", "b", "c")
    term = DiagramTerm(
        tuple(
            Propagator("C", left, right, "x", "y")
            for left, right in zip(labels[:3], labels[3:])
        ),
        Product((Symbol("F", labels[:3]), Symbol("G", labels[3:]))),
        Rational(1),
        (),
        tuple((label, n) for label in labels),
        0,
    )
    cv = {"F": np.ones((n,) * 3), "G": np.ones((n,) * 3)}
    budget = 4 * 1024**2
    candidate = try_build_factorized(term, analyze_spatial(term), cv, {}, budget)
    assert candidate is not None
    allocations = []

    def forbidden(*args, **kwargs):
        allocations.append((args, kwargs))
        raise AssertionError("dense fallback was allocated before budget refusal")

    monkeypatch.setattr(helper.np, "zeros", forbidden)
    with pytest.raises(ValueError, match="exceeding contraction_max_bytes"):
        _ = candidate.coupling_array
    assert _stored(candidate)["coupling_array"].shape == ()
    assert _stored(candidate)["_factorized_plan"] is not None

    def public_read(integrand, *args, **kwargs):
        return integrand.coupling_array

    monkeypatch.setattr(ev, "_integrate_moment_dispatch", public_read)
    with pytest.raises(ValueError, match="rebuild this integrand"):
        ev.integrate_moment(
            candidate,
            1.0,
            object(),
            contraction_backend="ordered",
            contraction_max_bytes=budget,
        )
    assert _stored(candidate)["_factorized_plan"] is not None
    unsupported = replace(term, coupling_sum=Product((Sum((term.coupling_sum,)),)))
    with pytest.raises(ValueError, match="exceeding contraction_max_bytes"):
        try_build_factorized(unsupported, analyze_spatial(unsupported), cv, {}, budget)
    assert allocations == []


def test_custom_axis_property_is_not_guessed_by_fallback_budget(monkeypatch):
    n = 32
    labels = ("i", "j", "k", "a", "b", "c")
    term = DiagramTerm(
        tuple(
            Propagator("C", left, right, "x", "y")
            for left, right in zip(labels[:3], labels[3:])
        ),
        Product((Symbol("F", labels[:3]), Symbol("G", labels[3:]))),
        Rational(1),
        (),
        tuple((label, n) for label in labels),
        0,
    )
    cv = {"F": np.ones((n,) * 3), "G": np.ones((n,) * 3)}
    calls = []

    def custom_axes(self):
        calls.append(self)
        return ()

    monkeypatch.setattr(DiagramTerm, "propagator_indices", property(custom_axes))
    fixed = dict.fromkeys(labels, 0)
    assert (
        try_build_factorized(term, analyze_spatial(term), cv, fixed, 4 * 1024**2)
        is None
    )
    assert calls == []
    actual = term.build_integrand(cv, fixed).coupling_array
    assert actual.shape == () and actual.item() == 1.0
    assert calls == [term]


def test_late_base_attribute_hook_observes_real_array(monkeypatch):
    _, _, baseline, candidate, cb, rp, jac, rb = _case()

    def custom_read(self, name):
        value = object.__getattribute__(self, name)
        return value * 2 if name == "coupling_array" else value

    monkeypatch.setattr(
        DiagramIntegrand, "__getattribute__", custom_read, raising=False
    )
    assert not native_factorized_integrand(candidate)
    np.testing.assert_array_equal(
        _values(candidate, cb, rp, jac, rb), _values(baseline, cb, rp, jac, rb)
    )
    assert _stored(candidate)["_factorized_plan"] is None


@pytest.mark.parametrize(
    "hook",
    [
        "symbol_getter",
        "symbol_property",
        "expr_getter",
        "product_property",
        "term_code",
        "term_getter",
        "sum_property",
        "compiler_code",
        "compiler_kwargs",
        "symbolic_code",
    ],
)
def test_first_import_uses_definition_time_witness_for_public_build(hook):
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        from sft_wick.expressions import Expr, Product, Propagator, Rational, Symbol
        from sft_wick.expressions import Sum
        from sft_wick.perturbation import DiagramTerm
        from sft_wick.evaluate import analyze_spatial

        labels = ('i', 'j', 'k', 'a', 'b', 'c')
        term = DiagramTerm(
            tuple(Propagator('C', left, right, 'x', 'y')
                  for left, right in zip(labels[:3], labels[3:])),
            Product((Symbol('F', labels[:3]), Symbol('G', labels[3:]))),
            Rational(1), (), tuple((label, 2) for label in labels), 0)
        cv = {'F': np.full((2,) * 3, 2.), 'G': np.full((2,) * 3, 3.)}
        calls = []
        hook = sys.argv[1]

        def read_name(self):
            original = object.__getattribute__(self, '__dict__')['name']
            calls.append(original)
            return 'F' if len(calls) < 8 else 'G'

        def read(self, name):
            if type(self) is Symbol and name == 'name':
                return read_name(self)
            return object.__getattribute__(self, name)

        if hook == 'symbol_getter':
            Symbol.__getattribute__ = read
        elif hook == 'symbol_property':
            Symbol.name = property(read_name)
        elif hook == 'expr_getter':
            Expr.__getattribute__ = read
        elif hook == 'product_property':
            def factors(self):
                calls.append('factors')
                return object.__getattribute__(self, '__dict__')['factors']
            Product.factors = property(factors)
        elif hook == 'term_getter':
            def term_read(self, name):
                value = object.__getattribute__(self, name)
                if name == 'rational_prefactor':
                    calls.append('prefactor')
                    return Rational(2 * value.numerator, value.denominator)
                return value
            DiagramTerm.__getattribute__ = term_read
        elif hook == 'sum_property':
            from dataclasses import replace
            term = replace(term, coupling_sum=Sum((term.coupling_sum, term.coupling_sum)))
            def terms(self):
                calls.append('terms')
                return object.__getattribute__(self, '__dict__')['terms']
            Sum.terms = property(terms)
        elif hook in ('compiler_code', 'compiler_kwargs'):
            from dataclasses import replace
            from sft_wick import _coupling_plan as cp
            term = replace(term, coupling_sum=Product((Rational(1), Rational(1), *term.coupling_sum.factors)))
            object.__setattr__(term.coupling_sum, '_test_calls', calls)
            if hook == 'compiler_code':
                def compile_custom(expression):
                    object.__getattribute__(expression, '__dict__')['_test_calls'].append('compile')
                    return None
                cp.compile_coupling_plan.__code__ = compile_custom.__code__
            else:
                class BadDefaults(dict):
                    def __bool__(self):
                        raise AssertionError('custom kwdefaults must not be copied by admission')
                cp.compile_coupling_plan.__kwdefaults__ = BadDefaults()
        elif hook == 'symbolic_code':
            from sft_wick import perturbation as pt
            object.__setattr__(term.coupling_sum, '_test_calls', calls)
            def symbolic_custom(expr, symbol_values, index_map):
                object.__getattribute__(expr, '__dict__')['_test_calls'].append('symbolic')
                return 17.
            pt._eval_symbolic.__code__ = symbolic_custom.__code__
        else:
            object.__setattr__(term, '_test_calls', calls)
            def custom_evaluate(self, coupling_values, fixed_indices=None):
                object.__getattribute__(self, '__dict__')['_test_calls'].append('custom')
                return np.array(123.)
            DiagramTerm.evaluate_coupling.__code__ = custom_evaluate.__code__

        # Force a genuinely fresh optional helper import, even if a future
        # package initializer eagerly imports this optional implementation.
        sys.modules.pop('sft_wick._factorized_coupling', None)
        from sft_wick._factorized_coupling import try_build_factorized
        assert try_build_factorized(term, analyze_spatial(term), cv, {}, 64*1024**2) is None
        assert calls == [], calls
        from dataclasses import replace
        left = replace(term)
        right = replace(term)
        if hook == 'term_code':
            object.__setattr__(left, '_test_calls', calls)
            object.__setattr__(right, '_test_calls', calls)
        calls.clear()
        ordered = left.build_integrand(cv, contraction_backend='ordered').coupling_array
        original_calls = tuple(calls)
        calls.clear()
        selected = right.build_integrand(cv, contraction_backend='factorized').coupling_array
        assert selected.dtype.type is ordered.dtype.type
        assert selected.tobytes() == ordered.tobytes()
        assert tuple(calls) == original_calls
        assert original_calls or hook == 'compiler_kwargs'
        """
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, hook],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("terms", [2, 4, 8])
@pytest.mark.parametrize("e_psi", [0, 1, 2, 3])
@pytest.mark.parametrize("matrix_r", [False, True])
def test_sum_of_physical_permuted_products_stays_lazy_and_matches_ordered(
    terms, e_psi, matrix_r
):
    term, cv, _, _, cb, rp, jac, rb = _case(e_psi=e_psi, matrix_r=matrix_r)
    left = ("i", "j", "k")
    right = ("a", "b", "c")
    permutations = (
        (left, right),
        (("j", "i", "k"), right),
        (("k", "j", "i"), right),
        (left, ("b", "a", "c")),
        (left, ("c", "b", "a")),
        (("j", "k", "i"), ("b", "c", "a")),
        (("k", "i", "j"), ("c", "a", "b")),
        (("i", "k", "j"), ("a", "c", "b")),
    )
    expression = Sum(
        tuple(
            Product((Symbol("F", a), Symbol("G", b))) for a, b in permutations[:terms]
        )
    )
    term = replace(term, coupling_sum=expression)
    candidate = try_build_factorized(term, analyze_spatial(term), cv, {"p": 1}, LIMIT)
    baseline = replace(term).build_integrand(cv, {"p": 1})
    assert candidate is not None
    snapshot = _stored(candidate)["_factorized_plan"]
    assert len(snapshot.monomials) == terms and len(snapshot.arrays) == 2
    np.testing.assert_allclose(
        _values(candidate, cb, rp, jac, rb),
        _values(baseline, cb, rp, jac, rb),
        rtol=1e-12,
        atol=1e-14,
    )
    assert _stored(candidate)["coupling_array"].shape == ()
    assert candidate.coupling_array.tobytes() == baseline.coupling_array.tobytes()


@pytest.mark.parametrize("complex_dtype", [False, True])
def test_positive_sum_materialization_matches_builtin_compensation_and_compiler(
    complex_dtype,
):
    term, _, _, _, _, _, _, _ = _case()
    dtype = np.complex128 if complex_dtype else np.float64
    cv = {"F": np.ones((2,) * 3, dtype=dtype), "G": np.ones((2,) * 3, dtype=dtype)}
    magnitudes = [10**16, 1, 1, 1, 1, 1, 1, 1]
    terms = tuple(
        Product((Rational(m), *term.coupling_sum.factors)) for m in magnitudes
    )
    term = replace(term, coupling_sum=Sum(terms))
    candidate = try_build_factorized(term, analyze_spatial(term), cv, {}, LIMIT)
    baseline_term = replace(term)
    baseline = baseline_term.build_integrand(cv)
    assert candidate is not None and baseline_term._coupling_execution_plan is not None
    assert candidate.coupling_array.dtype.type is baseline.coupling_array.dtype.type
    assert candidate.coupling_array.tobytes() == baseline.coupling_array.tobytes()
    scalar_values = [
        complex(float(v)) if complex_dtype else float(v) for v in magnitudes
    ]
    naive = 0.0
    for value in scalar_values:
        naive += value
    compensated_since = (3, 14) if complex_dtype else (3, 12)
    if sys.version_info >= compensated_since:
        assert sum(scalar_values) != naive


@pytest.mark.parametrize(
    "kind", ["opposite_term", "mixed_tensor", "imaginary_cancellation"]
)
def test_sum_declines_cross_term_sign_and_phase_cancellation(kind):
    term, cv, _, _, _, _, _, _ = _case()
    first = term.coupling_sum
    second = Product((Rational(-1), *first.factors))
    if kind == "mixed_tensor":
        cv["F"] = cv["F"].copy()
        cv["F"].flat[0] *= -1
        second = first
    elif kind == "imaginary_cancellation":
        first = Product((ImaginaryUnit(), *first.factors))
        second = Product((Rational(-1), ImaginaryUnit(), *term.coupling_sum.factors))
    term = replace(term, coupling_sum=Sum((first, second)))
    assert try_build_factorized(term, analyze_spatial(term), cv, {}, LIMIT) is None


def test_known_compensated_cancellation_sum_retains_original_path():
    term, _, _, _, _, _, _, _ = _case()
    cv = {"F": np.ones((2,) * 3), "G": np.ones((2,) * 3)}
    magnitudes = [10**16, 1, -(10**16), 1, 1, 1, 1, 1]
    expression = Sum(
        tuple(Product((Rational(m), *term.coupling_sum.factors)) for m in magnitudes)
    )
    term = replace(term, coupling_sum=expression)
    assert try_build_factorized(term, analyze_spatial(term), cv, {}, LIMIT) is None
    actual = term.build_integrand(cv).coupling_array
    np.testing.assert_array_equal(
        actual, np.full((2,) * 6, sum(map(float, magnitudes)) * (3 / 7))
    )


@pytest.mark.parametrize("terms", [0, 33])
def test_sum_term_count_bound_declines_without_materializing(terms):
    term, cv, _, _, _, _, _, _ = _case()
    term = replace(
        term, coupling_sum=Sum(tuple(term.coupling_sum for _ in range(terms)))
    )
    assert try_build_factorized(term, analyze_spatial(term), cv, {}, LIMIT) is None


def test_small_prefactor_cannot_admit_raw_product_sum_overflow():
    term, cv, _, _, _, _, _, _ = _case()
    cv = {"F": np.full((2,) * 3, 1e200), "G": np.full((2,) * 3, 1e200)}
    term = replace(
        term,
        coupling_sum=Sum((term.coupling_sum,) * 8),
        rational_prefactor=Rational(1, 10**200),
    )
    assert try_build_factorized(term, analyze_spatial(term), cv, {}, LIMIT) is None


def test_sum_snapshots_pickle_without_materialization_and_array_mutation_is_live():
    term, cv, _, _, cb, rp, jac, rb = _case()
    term = replace(term, coupling_sum=Sum((term.coupling_sum,) * 8))
    candidate = try_build_factorized(term, analyze_spatial(term), cv, {}, LIMIT)
    assert candidate is not None
    clone = pickle.loads(pickle.dumps(candidate))
    assert _stored(candidate)["coupling_array"].shape == ()
    assert _stored(clone)["coupling_array"].shape == ()
    np.testing.assert_array_equal(
        _values(candidate, cb, rp, jac, rb), _values(clone, cb, rp, jac, rb)
    )
    clone.coupling_array[:] *= 0.5
    expected = replace(term).build_integrand(cv)
    expected.coupling_array[:] *= 0.5
    np.testing.assert_array_equal(
        _values(clone, cb, rp, jac, rb), _values(expected, cb, rp, jac, rb)
    )
    assert _stored(candidate)["coupling_array"].shape == ()


def _configured_view(source):
    view = object.__new__(type(source))
    object.__getattribute__(view, "__dict__").update(
        object.__getattribute__(source, "__dict__")
    )
    object.__setattr__(view, "_contraction_max_bytes", 32 * 1024**2)
    share_factorized_owner(view, source)
    return view


def test_configured_view_shares_public_read_and_inplace_mutation():
    _, _, baseline, candidate, cb, rp, jac, rb = _case()
    view = _configured_view(candidate)
    assert native_factorized_integrand(view)
    assert _stored(view)["coupling_array"].shape == ()
    np.testing.assert_allclose(
        _values(view, cb, rp, jac, rb),
        _values(baseline, cb, rp, jac, rb),
        rtol=1e-12,
        atol=1e-14,
    )
    candidate.coupling_array[:] *= 3
    baseline.coupling_array[:] *= 3
    assert _stored(view)["_factorized_plan"] is None
    assert view.coupling_array is candidate.coupling_array
    np.testing.assert_array_equal(
        _values(view, cb, rp, jac, rb), _values(baseline, cb, rp, jac, rb)
    )


@pytest.mark.parametrize("operation", ["replace", "delete"])
def test_frozen_public_field_semantics_match_ordered(operation):
    _, _, baseline, candidate, _, _, _, _ = _case()
    for integrand in (baseline, candidate):
        with pytest.raises(FrozenInstanceError):
            if operation == "replace":
                integrand.coupling_array = np.array(2.0)
            else:
                del integrand.coupling_array
    assert _stored(candidate)["coupling_array"].shape == ()


@pytest.mark.parametrize("operation", ["replace", "delete"])
def test_internal_field_bypass_detaches_only_current_binding(operation):
    _, _, baseline, candidate, cb, rp, jac, rb = _case()
    view = _configured_view(candidate)
    if operation == "replace":
        replacement = baseline.coupling_array.copy() * 3
        object.__setattr__(candidate, "coupling_array", replacement)
        assert candidate.coupling_array is replacement
        assert _stored(candidate)["_factorized_plan"] is None
    else:
        object.__delattr__(candidate, "coupling_array")
        with pytest.raises(AttributeError, match="coupling_array"):
            _ = candidate.coupling_array
        with pytest.raises(AttributeError, match="coupling_array"):
            _values(candidate, cb, rp, jac, rb)
    assert _stored(view)["_factorized_plan"] is not None
    np.testing.assert_allclose(
        _values(view, cb, rp, jac, rb),
        _values(baseline, cb, rp, jac, rb),
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_array_equal(view.coupling_array, baseline.coupling_array)


def test_pickle_preserves_shared_latent_owner_without_materializing():
    _, _, baseline, candidate, cb, rp, jac, rb = _case()
    view = _configured_view(candidate)
    original, restored_view = pickle.loads(pickle.dumps((candidate, view)))
    assert _stored(original)["coupling_array"].shape == ()
    assert _stored(restored_view)["coupling_array"].shape == ()
    assert (
        _stored(original)["_factorized_owner"]
        is _stored(restored_view)["_factorized_owner"]
    )
    restored_view.coupling_array[:] *= 3
    baseline.coupling_array[:] *= 3
    assert original.coupling_array is restored_view.coupling_array
    assert _stored(original)["_factorized_plan"] is None
    np.testing.assert_array_equal(
        _values(original, cb, rp, jac, rb), _values(baseline, cb, rp, jac, rb)
    )
    assert _stored(candidate)["coupling_array"].shape == ()


@pytest.mark.parametrize("registry", ["related", "unrelated"])
def test_first_import_abc_registry_semantics_match_public_ordered(registry):
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        from abc import ABC
        from dataclasses import replace
        from sft_wick.expressions import Product, Propagator, Rational, Symbol
        from sft_wick.perturbation import DiagramTerm
        from sft_wick.evaluate import analyze_spatial

        labels = ('i', 'j', 'k', 'a', 'b', 'c')
        term = DiagramTerm(
            tuple(Propagator('C', left, right, 'x', 'y')
                  for left, right in zip(labels[:3], labels[3:])),
            Product((Symbol('F', labels[:3]), Symbol('G', labels[3:]))),
            Rational(1), (), tuple((label, 2) for label in labels), 0)
        cv = {'F': np.full((2,) * 3, 2.), 'G': np.full((2,) * 3, 3.)}
        if sys.argv[1] == 'related':
            Rational.register(Symbol)
        else:
            class Independent(ABC):
                pass
            class Registered:
                pass
            Independent.register(Registered)
        sys.modules.pop('sft_wick._factorized_coupling', None)
        from sft_wick._factorized_coupling import try_build_factorized, factorized_state
        candidate = try_build_factorized(term, analyze_spatial(term), cv, {}, 64*1024**2)
        if sys.argv[1] == 'related':
            assert candidate is None
            errors = []
            for backend in ('ordered', 'factorized'):
                try:
                    replace(term).build_integrand(cv, contraction_backend=backend)
                except AttributeError as error:
                    errors.append(str(error))
                else:
                    raise AssertionError('registered Rational(Symbol) changes the old evaluator')
            assert errors[0] == errors[1] and 'numerator' in errors[0], errors
        else:
            assert candidate is not None and factorized_state(candidate)[1] is not None
            selected = replace(term).build_integrand(cv, contraction_backend='factorized')
            assert factorized_state(selected)[0].shape == ()
            assert factorized_state(selected)[1] is not None
            ordered = replace(term).build_integrand(cv, contraction_backend='ordered')
            assert selected.coupling_array.tobytes() == ordered.coupling_array.tobytes()
        """
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, registry],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("before_import", [False, True])
def test_deleted_compiler_binding_keeps_public_original_fallback(before_import):
    script = textwrap.dedent(
        """
    import sys
    import numpy as np
    import sft_wick._coupling_plan as cp
    from sft_wick.expressions import Product, Propagator, Rational, Symbol
    from sft_wick.perturbation import DiagramTerm
    from dataclasses import replace
    term = DiagramTerm((Propagator('C','i','j','u','x'),),
        Product((Symbol('A',('i','j')),Symbol('B',('i','j')))),
        Rational(1),('u',),(('i',2),('j',2)),0)
    cv = {'A':np.full((2,2),.2),'B':np.full((2,2),.3)}
    if sys.argv[1] == 'False':
        import sft_wick._factorized_coupling
    del cp.compile_coupling_plan
    reference = term.build_integrand(cv).coupling_array
    selected = replace(term).build_integrand(cv,contraction_backend='factorized').coupling_array
    assert selected.dtype.type is reference.dtype.type
    assert selected.tobytes() == reference.tobytes()
    """
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(before_import)],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", ["Sum", "ImaginaryUnit"])
def test_deleted_public_expression_export_keeps_existing_term_semantics(name):
    script = textwrap.dedent(
        """
    import sys
    import numpy as np
    import sft_wick.expressions as expr
    from sft_wick.expressions import Product, Propagator, Rational, Symbol
    from sft_wick.perturbation import DiagramTerm
    from dataclasses import replace
    term=DiagramTerm((Propagator('C','i','j','u','x'),),
        Product((Symbol('A',('i','j')),Symbol('B',('i','j')))),
        Rational(1),('u',),(('i',2),('j',2)),0)
    cv={'A':np.full((2,2),.2),'B':np.full((2,2),.3)}
    delattr(expr,sys.argv[1])
    reference=term.build_integrand(cv).coupling_array
    selected=replace(term).build_integrand(cv,contraction_backend='factorized').coupling_array
    assert selected.dtype.type is reference.dtype.type
    assert selected.tobytes()==reference.tobytes()
    """
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, name],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
