"""Public opt-in controls, complete results and query-owned lifetime."""

from dataclasses import dataclass, fields, replace
import struct
import pickle
import weakref

import numpy as np
import pytest
import yaml

import sft_wick as sw
import sft_wick._dense_contraction as dense
import sft_wick.evaluate as ev
from sft_wick.evaluate import DiagramIntegrand, integrate_diagrams, integrate_moment
from sft_wick.expressions import Propagator, Rational, Symbol
from sft_wick.perturbation import DiagramTerm
from sft_wick.workflow.config import load_workflow_config, run_workflow
from sft_wick.workflow import resource_compare as rc
from sft_wick.workflow.resources import resource_controls


CACHE = dict(propagator_cache_bytes=2**20, propagator_cache_contract="fixed_model")
_INVALID = [
    {"propagator_cache_bytes": 10},
    {"propagator_cache_bytes": True, "propagator_cache_contract": "fixed_model"},
    {"contraction_backend": "auto"},
    {"contraction_max_bytes": 0},
]


def _signature(value):
    """Ordered metadata and exact numbers, independent of container aliasing."""
    if type(value) is dict:
        return (dict, tuple((key, _signature(item)) for key, item in value.items()))
    if type(value) in (list, tuple):
        return (type(value), tuple(_signature(item) for item in value))
    if type(value) is float:
        return (float, struct.pack("!d", value))
    if type(value) in (int, str, bool, type(None)):
        return (type(value), value)
    if isinstance(value, np.generic):
        return (type(value), value.dtype.str, value.tobytes())
    raise AssertionError(f"unsupported result type {type(value)}")


@dataclass
class _DenseR:
    def __call__(self, left, right):
        delta = left - right
        return np.exp(-delta) * np.array(
            [
                [np.cosh(0.2 * delta), np.sinh(0.2 * delta)],
                [np.sinh(0.2 * delta), np.cosh(0.2 * delta)],
            ]
        )


@dataclass
class _DenseC:
    scale: float = 1.0
    has_diagonal_kink = True

    def __call__(self, x1, t1, x2, t2):
        left, right = np.broadcast_arrays(np.asarray(t1), np.asarray(t2))
        minimum = np.minimum(left, right)
        rates = np.array([0.8, 1.2])
        eigen = np.exp(-np.abs(left - right)[..., None] * rates)
        eigen *= -np.expm1(-2 * minimum[..., None] * rates) / (2 * rates)
        out = np.empty(left.shape + (2, 2))
        out[..., 0, 0] = out[..., 1, 1] = 0.5 * (eigen[..., 0] + eigen[..., 1])
        out[..., 0, 1] = out[..., 1, 0] = 0.5 * (eigen[..., 0] - eigen[..., 1])
        return self.scale * out


@pytest.fixture
def admit_all(monkeypatch):
    """Run the backend on small fixtures, whatever the cost rule estimates."""
    monkeypatch.setattr(dense, "backend_is_cheaper", lambda backend, ordered: True)


@pytest.fixture
def system_case():
    kernel = _DenseC()
    system = sw.System(
        field=sw.FieldSpec("phi", 2),
        linear=sw.ExplicitR(R_time=_DenseR(), iso_R=False),
        vertices=[sw.LocalVertex("F", coupling=np.arange(1, 9).reshape(2, 2, 2) / 100)],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.1, sigma_t=0.3),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            )
        ),
    )
    props = system.propagators(
        t_max=1.3,
        c_closed_form=kernel,
        c_closed_form_only=True,
        c_closed_form_vectorized=True,
        diag_C=False,
        progress=False,
    )
    expansion = system.expand(
        ("phi_a(x)", "phi_b(y)"), orders=[0, 2], diag_R=False, diag_C=False
    )
    return expansion, props, kernel


def _term():
    # One integrated C leg gives a native batched static contraction with
    # independent i/j assignments; this is not a fake integrand dispatcher.
    return DiagramTerm(
        (Propagator("C", "i", "j", "u", "x"),),
        Symbol("A", ("i", "j")),
        Rational(1),
        ("u",),
        (("i", 2), ("j", 2)),
        0,
    )


def _values():
    return {"A": np.array([[0.2, 0.3], [0.4, 0.7]])}


def _kw(method="gauss_legendre"):
    return dict(
        method=method,
        n_gauss=4,
        n_samples=32,
        seed=17,
        positions={"x": 0.0},
        gl_chunk_size=3,
    )


def _record_banks(monkeypatch):
    banks = []
    original = ev.wrap_propagator_cache

    def record(cache, *args, **kwargs):
        wrapped = original(cache, *args, **kwargs)
        if wrapped is not cache:
            banks.append(wrapped)
        return wrapped

    monkeypatch.setattr(ev, "wrap_propagator_cache", record)
    return banks


class _Forbidden:
    def __getattr__(self, name):
        raise AssertionError(f"invalid execution choices touched user object: {name}")

    def __iter__(self):
        raise AssertionError("invalid execution choices inspected diagrams")


@pytest.mark.parametrize("route", ["moment", "diagrams", "evaluate", "sweep"])
@pytest.mark.parametrize("choices", _INVALID)
def test_invalid_public_options_fail_before_user_callbacks(system_case, route, choices):
    expansion, _, _ = system_case
    untouched = _Forbidden()
    with pytest.raises((TypeError, ValueError)):
        if route == "moment":
            integrate_moment(untouched, 1.0, untouched, **choices)
        elif route == "diagrams":
            integrate_diagrams(untouched, {}, 1.0, untouched, **choices)
        elif route == "evaluate":
            expansion.evaluate(
                untouched, positions={"x": 0.0, "y": 0.0}, t_final=1.0, **choices
            )
        else:
            expansion.sweep(
                untouched,
                positions_grid={"x": [0.0], "y": [0.0]},
                t_final_grid=[1.0],
                **choices,
            )


def test_public_diagram_batch_has_one_bank_exact_results_and_closes(
    monkeypatch, system_case
):
    _, props, _ = system_case
    terms = [_term()] * 3
    baseline = integrate_diagrams(terms, _values(), 1.0, props.cache, **_kw())
    banks = _record_banks(monkeypatch)
    monkeypatch.setattr(
        props.cache, "clear_cache", lambda: pytest.fail("source cleared")
    )
    cached = integrate_diagrams(terms, _values(), 1.0, props.cache, **_kw(), **CACHE)
    assert _signature(cached) == _signature(baseline)
    assert len(banks) == 1 and banks[0]._hits > 0
    assert banks[0]._closed and banks[0]._retained_bytes == 0
    assert not banks[0]._entries


def test_failed_public_moment_releases_only_its_local_bank(monkeypatch, system_case):
    _, props, _ = system_case
    banks = _record_banks(monkeypatch)
    monkeypatch.setattr(
        props.cache, "clear_cache", lambda: pytest.fail("source cleared")
    )

    class Raises:
        def integrate_moment_gauss_legendre(self, upper, cache, **kwargs):
            cache.R_matrix_batch(np.array([1.0]), np.array([0.2]))
            raise RuntimeError("fixture integration failure")

    with pytest.raises(RuntimeError, match="fixture integration failure"):
        integrate_moment(Raises(), 1.0, props.cache, method="gauss_legendre", **CACHE)
    assert len(banks) == 1 and banks[0]._closed
    assert not banks[0]._entries and banks[0]._retained_bytes == 0


@pytest.mark.parametrize("reuse", [False, True])
def test_public_serial_sweep_shares_one_bank_and_next_query_observes_model_change(
    monkeypatch, system_case, reuse
):
    expansion, props, kernel = system_case
    kwargs = dict(
        positions_grid={"x": [0.0], "y": [0.0, 0.2]},
        t_final_grid=[0.8, 1.0],
        component_pairs=[(0, 1)],
        method="qmc_vectorized",
        n_samples=32,
        seed=17,
        reuse_integrands=reuse,
        progress=False,
    )
    baseline = expansion.sweep(props, **kwargs)
    banks = _record_banks(monkeypatch)
    cached = expansion.sweep(props, **kwargs, **CACHE)
    assert _signature(cached.rows) == _signature(baseline.rows)
    assert len(banks) == 1 and banks[0]._hits > 0
    assert banks[0]._closed and not banks[0]._entries
    kernel.scale = 1.5  # Between complete queries, not within the promised scope.
    changed = expansion.sweep(props, **kwargs, **CACHE)
    reference = expansion.sweep(props, **kwargs)
    assert _signature(changed.rows) == _signature(reference.rows)
    assert changed.rows != cached.rows
    assert len(banks) == 2 and all(bank._closed for bank in banks)


@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized"])
def test_real_static_moment_admits_dense_backend_and_keeps_original_integrand(
    monkeypatch, admit_all, system_case, method
):
    _, props, _ = system_case
    integrand = _term().build_integrand(_values())
    baseline = integrate_moment(integrand, 1.0, props.cache, **_kw(method))
    admitted = []
    original = ev.try_static_contraction

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        admitted.append(result is not None)
        return result

    monkeypatch.setattr(ev, "try_static_contraction", observe)
    dense = integrate_moment(
        integrand, 1.0, props.cache, contraction_backend="einsum", **_kw(method)
    )
    np.testing.assert_allclose(dense, baseline, rtol=1e-12, atol=1e-14)
    assert admitted and any(admitted)
    assert integrand._contraction_backend == "ordered"


def test_public_dense_diagram_batch_admits_native_terms_with_complete_details(
    monkeypatch, admit_all, system_case
):
    _, props, _ = system_case
    terms = [_term()] * 3
    baseline = integrate_diagrams(terms, _values(), 1.0, props.cache, **_kw())
    admitted = []
    original = ev.try_static_contraction

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        admitted.append(result is not None)
        return result

    monkeypatch.setattr(ev, "try_static_contraction", observe)
    dense = integrate_diagrams(
        terms, _values(), 1.0, props.cache, contraction_backend="einsum", **_kw()
    )
    assert len(dense[1]) == len(baseline[1]) == 3
    np.testing.assert_allclose(dense[0], baseline[0], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(dense[1], baseline[1], rtol=1e-12, atol=1e-14)
    assert any(admitted)


@pytest.mark.parametrize("guard", ["diagram", "integrand"])
def test_public_custom_numerical_contract_retains_ordered_dispatch(
    monkeypatch, system_case, guard
):
    _, props, _ = system_case
    integrand = _term().build_integrand(_values())
    if guard == "diagram":

        class CustomTerm(DiagramTerm):
            pass

        custom = CustomTerm(
            **{
                field.name: getattr(integrand.diagram_term, field.name)
                for field in fields(DiagramTerm)
            }
        )
        integrand = replace(integrand, diagram_term=custom)
    elif guard == "integrand":

        class CustomIntegrand(DiagramIntegrand):
            pass

        integrand = CustomIntegrand(
            **{
                field.name: getattr(integrand, field.name)
                for field in fields(DiagramIntegrand)
            }
        )
        integrand = replace(integrand, _contraction_backend="einsum")
    baseline = integrate_moment(integrand, 1.0, props.cache, **_kw())
    monkeypatch.setattr(
        ev,
        "try_static_contraction",
        lambda *a, **k: pytest.fail("custom contract entered dense backend"),
    )
    candidate = integrate_moment(
        integrand, 1.0, props.cache, contraction_backend="einsum", **_kw()
    )
    assert _signature(candidate) == _signature(baseline)


@pytest.mark.parametrize("helper", ["_c_product", "_r_factor"])
@pytest.mark.parametrize("backend", ["ordered"])
def test_public_same_instance_helper_override_survives_backend_binding(
    monkeypatch, system_case, helper, backend
):
    _, props, _ = system_case
    integrand = _term().build_integrand(_values())
    original = getattr(integrand, helper)
    calls = []

    def customized(*args, **kwargs):
        calls.append(helper)
        return original(*args, **kwargs) + 0.25

    # A frozen dataclass can still carry an instance method override in its
    # dictionary. Public option binding must not discard it in a field clone.
    integrand.__dict__[helper] = customized
    baseline = integrate_moment(integrand, 1.0, props.cache, **_kw())
    assert calls
    calls.clear()
    monkeypatch.setattr(
        ev,
        "try_static_contraction",
        lambda *a, **k: pytest.fail("instance override entered dense backend"),
    )
    candidate = integrate_moment(
        integrand, 1.0, props.cache, contraction_backend=backend, **_kw()
    )
    assert calls and _signature(candidate) == _signature(baseline)
    assert integrand.__dict__[helper] is customized


def _raw():
    return {
        "system": {
            "field": {"n_components": 1},
            "linear": {"type": "diagonal", "gamma": [1.0]},
            "vertices": [{"name": "F", "coupling": [[[0.1]]]}],
            "noise": {
                "kappa2": {
                    "type": "separable_translation",
                    "temporal": {"type": "exponential", "lam": 0.1, "sigma_t": 0.3},
                    "spatial": {"type": "exponential", "sigma_x": 1.0},
                }
            },
        },
        "expand": {"observable": ["phi(x)", "phi(y)"], "orders": [0, 2]},
        "propagators": {"t_max": 1.1, "n_grid_t": 5},
        "sweep": {
            "positions_grid": {"x": [0.0], "y": [0.0]},
            "t_final_grid": [0.8, 1.0],
            "component_pairs": [[0, 0]],
            "method": "qmc_vectorized",
            "n_samples": 16,
            "seed": 17,
        },
    }


@pytest.mark.parametrize("choices", _INVALID)
def test_yaml_invalid_execution_options_are_rejected_before_module_import(
    tmp_path, choices
):
    raw = _raw()
    raw["sweep"].update(choices)
    module = tmp_path / "forbidden.py"
    module.write_text(
        "raise AssertionError('user module imported before validation')\n"
    )
    raw["system"]["vertices"] = [{"name": "F", "coupling_module": "forbidden.py"}]
    config = tmp_path / "invalid.yaml"
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises((TypeError, ValueError)):
        load_workflow_config(config)


def test_yaml_options_reach_public_sweep_and_cache_matches_complete_results(
    monkeypatch, tmp_path
):
    raw = _raw()
    config = tmp_path / "valid.yaml"
    config.write_text(yaml.safe_dump(raw))
    baseline, totals = run_workflow(load_workflow_config(config), progress=False)
    raw["sweep"].update(
        CACHE, contraction_backend="einsum", contraction_max_bytes=2**20
    )
    config.write_text(yaml.safe_dump(raw))
    configured = load_workflow_config(config)
    seen = []
    original = sw.Expansion.sweep

    def observe(self, props, **kwargs):
        seen.append(
            {
                key: kwargs[key]
                for key in (*CACHE, "contraction_backend", "contraction_max_bytes")
            }
        )
        return original(self, props, **kwargs)

    monkeypatch.setattr(sw.Expansion, "sweep", observe)
    actual, actual_totals = run_workflow(configured, progress=False)
    assert seen == [
        {**CACHE, "contraction_backend": "einsum", "contraction_max_bytes": 2**20}
    ]
    assert _signature(actual.rows) == _signature(baseline.rows)
    assert actual_totals.to_dict("records") == totals.to_dict("records")


@pytest.mark.parametrize(
    "changes",
    [
        {"propagator_cache_bytes": 10},
        {"propagator_cache_bytes": 10, "propagator_cache_contract": "fixed_model"},
    ],
)
def test_compare_rejects_missing_base_promise_or_variant_promise_before_worker(
    monkeypatch, tmp_path, changes
):
    raw = _raw()
    module = tmp_path / "forbidden.py"
    module.write_text("raise AssertionError('comparison imported a user module')\n")
    raw["system"]["vertices"] = [{"name": "F", "coupling_module": "forbidden.py"}]
    config, variants, report = (
        tmp_path / name for name in ("cfg.yaml", "variants.yaml", "report.json")
    )
    config.write_text(yaml.safe_dump(raw))
    variants.write_text(yaml.safe_dump({"cache": changes}))
    monkeypatch.setattr(rc, "_run_child", lambda *a, **k: pytest.fail("worker started"))
    with pytest.raises(ValueError):
        rc.compare_resources(config, variants, report)
    assert not report.exists()


def test_compare_six_controls_and_fixed_promise_are_checked_without_running_models():
    controls = {
        "reuse_integrands",
        "gl_chunk_size",
        "qmc_coupling_chunk_size",
        "propagator_cache_bytes",
        "contraction_backend",
        "contraction_max_bytes",
    }
    assert rc._CONTROLS == controls
    chosen = dict(
        reuse_integrands=True,
        gl_chunk_size=3,
        qmc_coupling_chunk_size=4,
        propagator_cache_bytes=10,
        contraction_backend="einsum",
        contraction_max_bytes=100,
    )
    rc._validate_variants({"all": chosen})
    raw = _raw()
    raw["sweep"]["propagator_cache_contract"] = "fixed_model"
    rc._validate_raw(raw)
    rc._validate_lookup_contract({**raw["sweep"], **chosen})
    assert raw["sweep"]["propagator_cache_contract"] == "fixed_model"


def test_resource_feedback_for_new_controls_never_calls_model_callbacks(tmp_path):
    raw = _raw()
    raw["sweep"].update(CACHE, contraction_backend="einsum")
    config = tmp_path / "feedback.yaml"
    config.write_text(yaml.safe_dump(raw))
    cfg = load_workflow_config(config)

    def forbidden(*args, **kwargs):
        pytest.fail("resource feedback invoked a model callback")

    cfg = replace(
        cfg,
        system=replace(
            cfg.system, vertices=[{"coupling": forbidden}], linear={"R_time": forbidden}
        ),
    )
    entries = {entry["name"]: entry for entry in resource_controls(cfg)}
    assert entries["sweep.propagator_cache_bytes"]["status"] == "conditional"
    assert entries["sweep.contraction_backend"]["status"] == "conditional"
    assert entries["sweep.propagator_cache_contract"]["status"] == "declared"


def test_public_parallel_grid_cache_serializes_with_exact_rows(system_case):
    expansion, props, _ = system_case
    kwargs = dict(
        positions_grid={"x": [0.0], "y": [0.0, 0.2, 0.4]},
        t_final_grid=[1.0],
        component_pairs=[(0, 1)],
        orders=[0, 2],
        method="qmc_vectorized",
        n_samples=32,
        seed=17,
        progress=False,
        reuse_integrands=True,
    )
    baseline = expansion.sweep(props, **kwargs)
    parallel = expansion.sweep(props, n_jobs=2, **kwargs, **CACHE)
    assert _signature(parallel.rows) == _signature(baseline.rows)


def test_native_pickle_keeps_class_admission_and_public_dense_backend(
    monkeypatch, admit_all, system_case
):
    _, props, _ = system_case
    integrand = pickle.loads(pickle.dumps(_term().build_integrand(_values())))
    assert ev._native_dense_integrand(integrand)
    original = ev.try_static_contraction
    admitted = []

    def observe(*args, **kwargs):
        value = original(*args, **kwargs)
        admitted.append(value is not None)
        return value

    monkeypatch.setattr(ev, "try_static_contraction", observe)
    baseline = integrate_moment(integrand, 1.0, props.cache, **_kw())
    result = integrate_moment(
        integrand, 1.0, props.cache, contraction_backend="einsum", **_kw()
    )
    assert any(admitted)
    np.testing.assert_allclose(result, baseline, rtol=1e-12, atol=1e-14)


def test_inline_native_integrand_lives_until_public_call_returns(
    monkeypatch, system_case
):
    _, props, _ = system_case
    reference = []
    finalized = []

    def make():
        integrand = _term().build_integrand(_values())
        reference.append(weakref.ref(integrand))
        weakref.finalize(integrand, finalized.append, True)
        return integrand

    def dispatch(**kwargs):
        assert reference[0]() is not None
        assert kwargs["integrand"] is not reference[0]()
        assert not finalized
        return (1.0, 0.0)

    monkeypatch.setattr(ev, "_integrate_moment_dispatch", dispatch)
    assert integrate_moment(make(), 1.0, props.cache, contraction_backend="einsum") == (
        1.0,
        0.0,
    )
    assert finalized == [True]
