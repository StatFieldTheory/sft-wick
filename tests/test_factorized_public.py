"""Public preparation, complete results and explicit factorized boundaries."""

import numpy as np
import pytest
import yaml
import sft_wick.evaluate as ev
from sft_wick._factorized_coupling import native_factorized_integrand
from sft_wick.perturbation import DiagramTerm
from sft_wick.workflow.config import load_workflow_config, run_workflow
from sft_wick.workflow.resources import resource_controls
from .test_execution_controls import (
    system_case as system_case,
    _term,
    _values,
    _kw,
    _raw,
)


def _close(actual, expected):
    if type(expected) is dict:
        assert list(actual) == list(expected)
        for k in expected:
            _close(actual[k], expected[k])
    elif type(expected) in (list, tuple):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _close(a, b)
    elif type(expected) is float:
        assert type(actual) is float
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    else:
        assert type(actual) is type(expected) and actual == expected


@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized"])
def test_public_static_preparation_stays_lazy_and_matches_complete_result(
    system_case, method
):
    _, props, _ = system_case
    term = _term()
    base = term.build_integrand(_values())
    candidate = term.build_integrand(_values(), contraction_backend="factorized")
    assert native_factorized_integrand(candidate)
    assert object.__getattribute__(candidate, "__dict__")["coupling_array"].shape == ()
    expected = ev.integrate_moment(base, 1.0, props.cache, **_kw(method))
    actual = ev.integrate_moment(
        candidate, 1.0, props.cache, **_kw(method), contraction_backend="factorized"
    )
    _close(actual, expected)
    state = object.__getattribute__(candidate, "__dict__")
    assert state["coupling_array"].shape == ()
    assert state["_factorized_plan"] is not None


@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized"])
def test_public_diagram_batch_and_expansion_compare_all_rows(system_case, method):
    expansion, props, _ = system_case
    kwargs = _kw(method)
    _close(
        ev.integrate_diagrams(
            [_term()] * 2,
            _values(),
            1.0,
            props.cache,
            contraction_backend="factorized",
            **kwargs
        ),
        ev.integrate_diagrams([_term()] * 2, _values(), 1.0, props.cache, **kwargs),
    )
    opts = dict(
        positions={"x": 0.0, "y": 0.2},
        t_final=1.0,
        component_pair=(0, 1),
        method=method,
        n_samples=16,
        n_gauss=3,
        seed=17,
        gl_chunk_size=7,
    )
    expected = expansion.evaluate(props, **opts)
    actual = expansion.evaluate(props, contraction_backend="factorized", **opts)
    _close(actual.__dict__, expected.__dict__)


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("jobs", [1, 2])
def test_public_sweep_and_worker_pickle_compare_complete_rows(system_case, reuse, jobs):
    expansion, props, _ = system_case
    opts = dict(
        positions_grid={"x": [0.0], "y": [0.0, 0.2]},
        t_final_grid=[0.8],
        component_pairs=[(0, 1)],
        method="qmc_vectorized",
        n_samples=16,
        seed=17,
        reuse_integrands=reuse,
        n_jobs=jobs,
        progress=False,
    )
    expected = expansion.sweep(props, **opts)
    actual = expansion.sweep(props, contraction_backend="factorized", **opts)
    _close(actual.__dict__, expected.__dict__)


def test_public_custom_build_attribute_uses_original_two_argument_protocol(monkeypatch):
    term = _term()
    original = DiagramTerm.__getattribute__
    calls = []

    def getter(self, name):
        if name == "build_integrand":

            def custom(cv, fi):
                calls.append((cv, fi))
                return ("custom", cv, fi)

            return custom
        return original(self, name)

    monkeypatch.setattr(DiagramTerm, "__getattribute__", getter)
    values = _values()
    assert ev._build_execution_integrand(term, values, {"a": 1}, "factorized") == (
        "custom",
        values,
        {"a": 1},
    )
    assert len(calls) == 1


@pytest.mark.parametrize("method", ["qmc_scalar", "nquad"])
def test_scalar_feedback_keeps_factorized_materialization_budget_active(
    tmp_path, method
):
    raw = _raw()
    raw["sweep"].update(contraction_backend="factorized", method=method)
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_workflow_config(path)
    controls = {x["name"]: x for x in resource_controls(cfg)}
    assert controls["sweep.contraction_backend"]["status"] == "conditional"
    assert controls["sweep.contraction_max_bytes"]["status"] == "conditional"
    assert "materializ" in controls["sweep.contraction_backend"]["reason"]


def test_yaml_factorized_workflow_returns_complete_ordered_rows(tmp_path):
    raw = _raw()
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(raw))
    expected = run_workflow(load_workflow_config(path))
    raw["sweep"]["contraction_backend"] = "factorized"
    path.write_text(yaml.safe_dump(raw))
    actual = run_workflow(load_workflow_config(path))
    # Config and feedback intentionally expose the requested backend.
    _close(actual[0].__dict__, expected[0].__dict__)
    assert list(actual[1].columns) == list(expected[1].columns)
    _close(actual[1].to_dict("records"), expected[1].to_dict("records"))


@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized"])
def test_configured_views_preserve_public_array_ownership(
    monkeypatch, system_case, method
):
    from .test_execution_controls import _DenseC

    _, props, _ = system_case
    original_call = _DenseC.__call__
    results = []
    for backend in ("ordered", "factorized"):
        integrand = _term().build_integrand(_values(), contraction_backend=backend)
        mutated = []

        def callback(self, *args):
            if not mutated:
                mutated.append(True)
                integrand.coupling_array[:] *= 3
            return original_call(self, *args)

        monkeypatch.setattr(_DenseC, "__call__", callback)
        results.append(
            ev.integrate_moment(
                integrand,
                1.0,
                props.cache,
                contraction_backend=backend,
                contraction_max_bytes=32 * 1024**2,
                **_kw(method)
            )
        )
        assert mutated
    _close(results[1], results[0])


@pytest.mark.parametrize("backend", ["ordered", "einsum", "factorized"])
def test_public_coefficient_field_remains_frozen(backend):
    from dataclasses import FrozenInstanceError

    integrand = _term().build_integrand(_values(), contraction_backend=backend)
    with pytest.raises(FrozenInstanceError):
        integrand.coupling_array = np.ones((2, 2))
    with pytest.raises(FrozenInstanceError):
        del integrand.coupling_array


@pytest.mark.parametrize("backend", ["ordered", "einsum", "factorized"])
def test_builder_binds_explicit_execution_fields(backend):
    integrand = _term().build_integrand(
        _values(), contraction_backend=backend, contraction_max_bytes=32 * 1024**2
    )
    assert integrand._contraction_backend == backend
    assert integrand._contraction_max_bytes == 32 * 1024**2


def _record_factorized_preparation(monkeypatch):
    """Observe actual preparation without replacing any native class protocol."""
    original = ev.integrate_moment
    seen = []

    def observe(integrand, *args, **kwargs):
        state = object.__getattribute__(integrand, "__dict__")
        plan = state.get("_factorized_plan")
        owner = state.get("_factorized_owner")
        seen.append(
            (integrand._contraction_backend, integrand._contraction_max_bytes, plan)
        )
        if plan is not None:
            assert native_factorized_integrand(integrand)
            assert owner is not None and owner.array.shape == ()
            assert owner.snapshot is plan
        result = original(integrand, *args, **kwargs)
        if plan is not None:
            assert owner.array.shape == () and owner.snapshot is plan
        return result

    monkeypatch.setattr(ev, "integrate_moment", observe)
    return seen


@pytest.mark.parametrize("count,jobs", [(1, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized"])
def test_public_factorized_dispatch_prepares_serial_and_small_batches(
    monkeypatch, system_case, count, jobs, method
):
    _, props, _ = system_case
    budget = 2**25
    expected = ev.integrate_diagrams(
        [_term() for _ in range(count)], _values(), 1.0, props.cache, **_kw(method)
    )
    seen = _record_factorized_preparation(monkeypatch)
    actual = ev.integrate_diagrams(
        [_term() for _ in range(count)],
        _values(),
        1.0,
        props.cache,
        n_jobs=jobs,
        contraction_backend="factorized",
        contraction_max_bytes=budget,
        **_kw(method),
    )
    _close(actual, expected)
    assert len(seen) == count
    assert all(backend == "factorized" and limit == budget for backend, limit, _ in seen)
    assert all(plan is not None for _, _, plan in seen)


@pytest.mark.parametrize("route", ["evaluate", "sweep"])
@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized"])
def test_public_factorized_dispatch_prepares_unreused_expansion(
    monkeypatch, system_case, route, method
):
    expected_expansion, props, _ = system_case
    # Coefficient preparation may legitimately cache a dense symbolic plan on a
    # term. Give the two executions independent fresh expansion terms.
    candidate = expected_expansion.system.expand(
        ("phi_a(x)", "phi_b(y)"), orders=[2], diag_R=False, diag_C=False
    )
    budget = 2**25
    common = dict(orders=[2], method=method, n_samples=16, n_gauss=3, seed=17)

    def run(expansion, **execution):
        if route == "evaluate":
            return expansion.evaluate(
                props,
                positions={"x": 0.0, "y": 0.2},
                t_final=0.8,
                component_pair=(0, 1),
                **common,
                **execution,
            )
        return expansion.sweep(
            props,
            positions_grid={"x": [0.0], "y": [0.0, 0.2]},
            t_final_grid=[0.8],
            component_pairs=[(0, 1)],
            reuse_integrands=False,
            progress=False,
            **common,
            **execution,
        )

    expected = run(expected_expansion)
    seen = _record_factorized_preparation(monkeypatch)
    actual = run(
        candidate, contraction_backend="factorized", contraction_max_bytes=budget
    )
    _close(actual.__dict__, expected.__dict__)
    assert len(seen) == len(candidate.diagrams(2)) * (1 if route == "evaluate" else 2)
    assert all(backend == "factorized" and limit == budget for backend, limit, _ in seen)
    assert any(plan is not None for _, _, plan in seen)


@pytest.mark.parametrize("count,jobs", [(1, 1), (1, 2), (2, 2)])
def test_public_factorized_dispatch_restoration_budget_precedes_coefficient_work(
    system_case, count, jobs
):
    import sys

    _, props, _ = system_case
    coefficient_code = DiagramTerm.evaluate_coupling.__code__
    calls = []
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is coefficient_code:
            calls.append(frame.f_code)

    # Observing the native code object keeps the native builder/evaluator
    # bindings intact; replacing either would change the admission contract.
    sys.setprofile(observe)
    try:
        with pytest.raises(ValueError, match="coefficient materialization.*bytes"):
            ev.integrate_diagrams(
                [_term() for _ in range(count)],
                _values(),
                1.0,
                props.cache,
                n_jobs=jobs,
                contraction_backend="factorized",
                contraction_max_bytes=1,
                **_kw(),
            )
    finally:
        sys.setprofile(previous)
    assert calls == []


@pytest.mark.parametrize("route", ["evaluate", "sweep"])
def test_public_factorized_dispatch_expansion_respects_restoration_budget(
    system_case, route
):
    expansion, props, _ = system_case
    common = dict(
        orders=[2],
        method="qmc_vectorized",
        n_samples=16,
        seed=17,
        contraction_backend="factorized",
        contraction_max_bytes=1,
    )
    with pytest.raises(ValueError, match="coefficient materialization.*bytes"):
        if route == "evaluate":
            expansion.evaluate(
                props,
                positions={"x": 0.0, "y": 0.2},
                t_final=0.8,
                component_pair=(0, 1),
                **common,
            )
        else:
            expansion.sweep(
                props,
                positions_grid={"x": [0.0], "y": [0.0, 0.2]},
                t_final_grid=[0.8],
                component_pairs=[(0, 1)],
                reuse_integrands=False,
                progress=False,
                **common,
            )


def test_public_factorized_dispatch_real_diagram_worker_respects_budget(system_case):
    _, props, _ = system_case
    # Three terms exceed integrate_diagrams' serial shortcut even with n_jobs=2.
    with pytest.raises(ValueError, match="coefficient materialization.*bytes") as error:
        ev.integrate_diagrams(
            [_term() for _ in range(3)],
            _values(),
            1.0,
            props.cache,
            n_jobs=2,
            contraction_backend="factorized",
            contraction_max_bytes=1,
            **_kw(),
        )
    assert error.value.__cause__ is not None
    assert type(error.value.__cause__).__module__.startswith("joblib.")


@pytest.mark.parametrize("jobs,diagram_jobs", [(2, 1), (1, 2)])
def test_public_factorized_dispatch_real_sweep_worker_respects_budget(
    system_case, jobs, diagram_jobs
):
    expansion, props, _ = system_case
    assert len(expansion.diagrams(2)) >= 3
    # Three grid points test actual grid workers, while the other parameter
    # tests actual diagram workers; nested process pools remain unsupported.
    with pytest.raises(ValueError, match="coefficient materialization.*bytes") as error:
        expansion.sweep(
            props,
            positions_grid={"x": [0.0], "y": [0.0, 0.1, 0.2]},
            t_final_grid=[0.8],
            component_pairs=[(0, 1)],
            orders=[2],
            method="qmc_vectorized",
            n_samples=16,
            seed=17,
            n_jobs=jobs,
            evaluate_n_jobs=diagram_jobs,
            reuse_integrands=False,
            progress=False,
            contraction_backend="factorized",
            contraction_max_bytes=1,
        )
    assert error.value.__cause__ is not None
    assert type(error.value.__cause__).__module__.startswith("joblib.")
