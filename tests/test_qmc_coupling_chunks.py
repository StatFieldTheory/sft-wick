"""Coupling-only QMC partitions preserve draw order and whole-batch statistics.

The polynomial reference is elementary and does not reproduce the causal map.
The sampler stub exposes a deliberate non-divisor tail; expected statistics use
its independent row values, including the existing eight-batch error policy.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from sft_wick.evaluate import (
    DiagramIntegrand,
    integrate_diagrams,
    integrate_moment,
)
from sft_wick.expressions import Propagator, Rational, Symbol
from sft_wick.perturbation import DiagramTerm
from sft_wick.workflow.config import load_workflow_config, run_workflow
from tests.test_gl_chunked_integration import _YAML_BASE, _cumulant, POS3, T_MIN

OPTION = "qmc_coupling_chunk_size"


class UnitCovarianceCache:
    """Analytic C=1: a constant unit-variance random field.

    The native batch-cache handshake is valid because the scalar and batched
    methods below are the actual implementation; no fictitious spline is read.
    """

    model = SimpleNamespace(iso_R=True, diag_C=True, n_components=1)
    _c_splines = True

    def C_value(self, x1, t1, x2, t2):
        return np.ones((1, 1))

    def C_diagonal(self, x1, t1, x2=None, t2=None):
        return np.ones(1)

    def C_diagonal_batch(self, t1, t2):
        return np.ones((len(t1), 1))

    def R_time(self, t1, t2):
        return float(t1 > t2)

    def R_time_batch(self, t1, t2):
        return np.asarray(t1 > t2, dtype=float)

    def R_product(self, propagators, times):
        return float(np.prod([self.R_time(times[a], times[b]) for a, b in propagators]))


def _cache():
    return UnitCovarianceCache()


def _term(propagators=None):
    # C(s,s)=1 registers the internal label without changing the polynomial
    # integral. A bare K(s) has no direction-map entry in analyze_spatial.
    if propagators is None:
        propagators = (Propagator("C", None, None, "s", "s"),)
    return DiagramTerm(
        propagators=propagators,
        coupling_sum=Symbol("K", indices=(), spatial_args=("s",)),
        rational_prefactor=Rational(1, 1),
        integration_vars=("s",),
        summation_indices=(),
        n_response=0,
    )


def _integrand(callback, propagators=None):
    return _term(propagators).build_integrand({"K": callback})


@dataclass
class Polynomial:
    vectorized: bool = True
    calls: list = field(default_factory=list)

    def __call__(self, positions, times):
        times = np.asarray(times)
        self.calls.append(times.copy())
        return 3 * times[0] ** 2


def _run(ig, cap=None, n=17, seed=7, **kw):
    return ig.integrate_moment_qmc_vectorized(
        1.0,
        _cache(),
        n_samples=n,
        seed=seed,
        qmc_coupling_chunk_size=cap,
        **kw,
    )


def _install_draw(monkeypatch, points):
    from scipy.stats import qmc

    points = np.asarray(points, dtype=float).reshape(-1, 1)
    trace = {"construct": [], "draw": [], "arrays": []}

    class Sampler:
        def __init__(self, *, d, seed, bits):
            trace["construct"].append((d, seed, bits))
            assert d == 1

        def random(self, n):
            trace["draw"].append(n)
            assert n == len(points)
            out = points.copy()
            trace["arrays"].append(out)
            return out

    monkeypatch.setattr(qmc, "Sobol", Sampler)
    return trace


@pytest.mark.parametrize("cap", [None, 1, 3, 7, 16, 17, 18, 129, np.int64(3)])
@pytest.mark.parametrize("vectorized", [False, True])
def test_draw_rows_tail_and_eight_batch_statistics_are_preserved(
    monkeypatch, cap, vectorized
):
    points = np.arange(1, 18, dtype=float) / 19
    trace = _install_draw(monkeypatch, points)
    callback = Polynomial(vectorized)
    result = _run(_integrand(callback), cap)
    values = 3 * points**2
    batch_means = np.array([np.mean(values[2 * i : 2 * (i + 1)]) for i in range(8)])
    reference = (
        float(np.mean(values)),
        float(np.std(batch_means, ddof=1) / np.sqrt(8)),
    )
    np.testing.assert_array_equal(result, reference)
    assert trace["construct"] == [(1, 7, 64)]
    assert trace["draw"] == [17]
    np.testing.assert_array_equal(trace["arrays"][0][:, 0], points)
    if vectorized:
        limit = 17 if cap is None else min(int(cap), 17)
        lengths = [call.shape[1] for call in callback.calls]
        assert lengths == [min(limit, 17 - start) for start in range(0, 17, limit)]
        np.testing.assert_array_equal(
            np.concatenate([call[0] for call in callback.calls]), points
        )
    else:
        assert len(callback.calls) == 17  # No repeated sample-zero shape probe.
        np.testing.assert_array_equal([call[0] for call in callback.calls], points)


@pytest.mark.parametrize("cap", [1, 3, 129, 256, 257])
@pytest.mark.parametrize("seed", [0, 7, 20261011])
def test_real_sobol_stable_polynomial_has_exact_paired_result(cap, seed):
    eager = _run(_integrand(Polynomial()), None, n=256, seed=seed)
    chunked = _run(_integrand(Polynomial()), cap, n=256, seed=seed)
    np.testing.assert_array_equal(chunked, eager)
    assert abs(chunked[0] - 1.0) < 0.02  # Integral_0^1 3t^2 dt = 1.
    assert np.isfinite(chunked).all()


@pytest.mark.parametrize("n", [1, 8, 17, 10_003])
def test_small_and_large_requested_sample_counts_are_unchanged(monkeypatch, n):
    points = np.linspace(0.1, 0.9, n)
    trace = _install_draw(monkeypatch, points)
    callback = Polynomial()
    with np.errstate(invalid="ignore", divide="ignore"):
        result = _run(_integrand(callback), 129, n=n)
    np.testing.assert_allclose(result[0], np.mean(3 * points**2), rtol=0, atol=0)
    assert trace["draw"] == [n]
    assert sum(call.shape[1] for call in callback.calls) == n
    assert max(call.shape[1] for call in callback.calls) <= min(129, n)
    if n == 1:
        assert np.isnan(result[1])  # Existing ddof=1 policy remains observable.


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.5, "3", [], {}, np.bool_(True)])
@pytest.mark.parametrize("route", ["direct", "dispatcher"])
def test_cap_requires_positive_nonbool_integer(bad, route):
    ig = _integrand(Polynomial())
    with pytest.raises((ValueError, TypeError), match=OPTION):
        if route == "direct":
            _run(ig, bad)
        else:
            integrate_moment(
                ig,
                1.0,
                _cache(),
                method="qmc_vectorized",
                n_samples=8,
                qmc_coupling_chunk_size=bad,
            )


@pytest.mark.parametrize("method", ["qmc_scalar", "gauss_legendre", "nquad"])
def test_other_methods_validate_but_do_not_partition_dynamic_callbacks(method):
    callback = Polynomial(vectorized=False)
    ig = _integrand(callback)
    kw = dict(method=method, n_samples=8, seed=7, n_gauss=3)
    eager = integrate_moment(ig, 1.0, _cache(), **kw)
    eager_calls = [call.copy() for call in callback.calls]
    callback.calls.clear()
    got = integrate_moment(ig, 1.0, _cache(), qmc_coupling_chunk_size=1, **kw)
    np.testing.assert_array_equal(got, eager)
    assert len(callback.calls) == len(eager_calls)
    for actual, expected in zip(callback.calls, eager_calls):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises((ValueError, TypeError), match=OPTION):
        integrate_moment(ig, 1.0, _cache(), qmc_coupling_chunk_size=True, **kw)


def test_static_coupling_preserves_the_whole_batch_path(monkeypatch):
    trace = _install_draw(monkeypatch, np.linspace(0.1, 0.9, 17))
    ig = _term().build_integrand({"K": np.array(2.0)})
    eager = _run(ig, None)
    got = _run(ig, 3)
    np.testing.assert_array_equal(got, eager)
    assert trace["draw"] == [17, 17]
    np.testing.assert_array_equal(got, (2.0, 0.0))


@pytest.mark.parametrize("dynamic", [False, True])
def test_zero_dimensions_keep_the_existing_single_promise_call(dynamic):
    calls = []

    def kernel(n, t):
        calls.append(np.asarray(t).copy())
        return np.ones((1, 1, 1))

    exp, props = _cumulant(
        kernel if dynamic else np.ones((1, 1, 1)), n=1, absorbed=True
    )
    kw = dict(
        positions=POS3,
        t_final=1.4,
        component_pair=(0, 0, 0),
        orders=[1],
        method="qmc_vectorized",
        n_samples=17,
        seed=7,
    )
    eager = exp.evaluate(props, **kw)
    n_calls = len(calls)
    calls.clear()
    got = exp.evaluate(props, qmc_coupling_chunk_size=1, **kw)
    assert got.per_diagram == eager.per_diagram
    assert got.total == eager.total
    np.testing.assert_allclose(got.total, 1.0, rtol=2e-15, atol=0)
    assert len(calls) == n_calls


@pytest.mark.parametrize("cap", [1, 3, 129])
def test_full_reality_scale_is_checked_once_after_coupling_assembly(monkeypatch, cap):
    points = np.r_[np.full(8, 0.001), np.linspace(0.5, 1.0, 9)]
    _install_draw(monkeypatch, points)
    calls = []
    original = DiagramIntegrand._dynamic_values

    def observe(
        self, couplings, c_batches, r_product, jacobians=None, *, where="", r_batches=()
    ):
        calls.append((np.asarray(couplings).copy(), len(r_product)))
        return original(
            self,
            couplings,
            c_batches,
            r_product,
            jacobians,
            where=where,
            r_batches=r_batches,
        )

    monkeypatch.setattr(DiagramIntegrand, "_dynamic_values", observe)

    def tiny_roundoff(n, t):
        return np.asarray(t)[0] ** 8 + 1e-11j

    tiny_roundoff.vectorized = True
    eager = _run(_integrand(tiny_roundoff), None)
    got = _run(_integrand(tiny_roundoff), cap)
    np.testing.assert_array_equal(got, eager)
    assert [n for _, n in calls] == [17, 17]
    for couplings, _ in calls:
        np.testing.assert_allclose(couplings.real, points**8, rtol=0, atol=0)


@pytest.mark.parametrize("cap", [1, 3, 129])
def test_bad_phase_is_still_rejected_on_the_complete_batch(monkeypatch, cap):
    _install_draw(monkeypatch, np.linspace(0.1, 0.9, 17))

    def bad_phase(n, t):
        return np.asarray(t)[0] ** 2 + 1e-3j

    bad_phase.vectorized = True
    with pytest.raises(ValueError, match="imaginary|Im"):
        _run(_integrand(bad_phase), cap)


@pytest.mark.parametrize(
    "malformed", ["short", "scalar", "trailing", "dtype", "object"]
)
def test_malformed_custom_promise_blocks_fail_without_silent_assembly(malformed):
    ig = _integrand(Polynomial())
    calls = []

    def batch(label_t, label_x, n_samples):
        calls.append(n_samples)
        if len(calls) == 1:
            return np.ones(n_samples, dtype=float)
        if malformed == "short":
            return np.ones(1, dtype=float)
        if malformed == "scalar":
            return np.array(2.0)
        if malformed == "trailing":
            return np.ones((n_samples, 1), dtype=float)
        if malformed == "dtype":
            return np.ones(n_samples, dtype=complex)
        return np.full(n_samples, "bad", dtype=object)

    ig = replace(ig, dynamic_coupling=SimpleNamespace(evaluate_at_batch=batch))
    with pytest.raises(
        (TypeError, ValueError), match="shape|dtype|numeric|leading|batch"
    ):
        _run(ig, 3)
    assert calls == [3, 3]


def test_partition_protocol_retains_independent_row_views_and_fixed_positions(
    monkeypatch,
):
    points = np.linspace(0.1, 0.9, 17)
    _install_draw(monkeypatch, points)
    ig = _integrand(Polynomial())
    retained = []
    positions = []

    def batch(label_t, label_x, n_samples):
        retained.append(label_t)
        positions.append(label_x)
        return 3 * label_t["s"] ** 2

    ig = replace(ig, dynamic_coupling=SimpleNamespace(evaluate_at_batch=batch))
    _run(ig, 3, direction=0.37)
    assert len({id(labels) for labels in retained}) == 6
    np.testing.assert_array_equal(
        np.concatenate([labels["s"] for labels in retained]), points
    )
    assert all(labels["s"].flags.owndata is False for labels in retained)
    assert all(pos == {"s": 0.37} for pos in positions)


def test_batch_length_dependent_callbacks_are_outside_the_opt_in_contract(monkeypatch):
    points = np.linspace(0.1, 0.9, 17)
    _install_draw(monkeypatch, points)

    def global_length(n, t):
        return np.full(np.asarray(t).shape[1], np.asarray(t).shape[1], dtype=float)

    global_length.vectorized = True
    eager = _run(_integrand(global_length), None)
    got = _run(_integrand(global_length), 3)
    assert eager[0] == 17.0
    assert got[0] == (5 * 3 * 3 + 2 * 2) / 17
    assert got[0] != eager[0]


@pytest.mark.parametrize(
    "route", ["direct", "integrate_moment", "integrate_diagrams", "evaluate"]
)
def test_public_L0_L1_routes_forward_the_coupling_cap(route):
    calls = []

    def kernel(n, t):
        calls.append(np.asarray(t).shape[1])
        return np.ones((np.asarray(t).shape[1], 1, 1, 1))

    exp, props = _cumulant(kernel, n=1, vectorized=True)
    kwargs = dict(
        t_min=T_MIN, positions=POS3, n_samples=17, seed=7, qmc_coupling_chunk_size=3
    )
    terms = exp.diagrams(1)
    cv = exp.system.build_coupling_values()
    if route == "direct":
        value = sum(
            dt.build_integrand(cv).integrate_moment_qmc_vectorized(
                1.4, props.cache, **kwargs
            )[0]
            for dt in terms
        )
    elif route == "integrate_moment":
        value = sum(
            integrate_moment(
                dt.build_integrand(cv),
                1.4,
                props.cache,
                method="qmc_vectorized",
                **kwargs,
            )[0]
            for dt in terms
        )
    elif route == "integrate_diagrams":
        value = integrate_diagrams(
            terms, cv, 1.4, props.cache, method="qmc_vectorized", **kwargs
        )[0]
    else:
        value = exp.evaluate(
            props,
            positions=POS3,
            t_final=1.4,
            component_pair=(0, 0, 0),
            orders=[1],
            method="qmc_vectorized",
            n_samples=17,
            seed=7,
            qmc_coupling_chunk_size=3,
        ).total
    np.testing.assert_allclose(value, 1.2**3, rtol=2e-15, atol=0)
    assert calls and max(calls) <= 3 and 2 in calls


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("n_jobs,evaluate_n_jobs", [(1, 1), (2, 1), (1, 2)])
def test_sweep_prepared_and_parallel_routes_preserve_complete_rows(
    reuse, n_jobs, evaluate_n_jobs
):
    def kernel(n, t):
        return (1 + np.sum(np.asarray(t), axis=0))[:, None, None, None] * np.ones(
            (1, 1, 1)
        )

    exp, props = _cumulant(kernel, n=1, vectorized=True)
    kwargs = dict(
        positions_grid={k: [v] for k, v in POS3.items()},
        t_final_grid=[0.8, 1.4],
        component_tuples=[(0, 0, 0)],
        orders=[1],
        method="qmc_vectorized",
        n_samples=32,
        seed=7,
        progress=False,
    )
    eager = exp.sweep(props, **kwargs)
    got = exp.sweep(
        props,
        qmc_coupling_chunk_size=3,
        reuse_integrands=reuse,
        n_jobs=n_jobs,
        evaluate_n_jobs=evaluate_n_jobs,
        **kwargs,
    )
    assert got.rows == eager.rows
    assert all(np.isfinite(row["value"]) for row in got.rows)
    assert any(row["value"] != 0 for row in got.rows)


def test_integrate_diagrams_uses_real_joblib_workers_with_more_than_two_terms():
    callback = Polynomial()
    dt = _term()
    terms = [dt, dt, dt]
    kwargs = dict(
        method="qmc_vectorized", n_samples=32, seed=7, qmc_coupling_chunk_size=3
    )
    eager = integrate_diagrams(
        terms, {"K": callback}, 1.0, _cache(), n_jobs=1, **kwargs
    )
    actual = integrate_diagrams(
        terms, {"K": callback}, 1.0, _cache(), n_jobs=2, **kwargs
    )
    assert actual == eager
    assert len(actual[1]) == 3 and actual[0] != 0


def _yaml_config(tmp_path, cap=None):
    cfg = copy.deepcopy(_YAML_BASE)
    cfg["sweep"].update(method="qmc_vectorized", n_samples=32, seed=7)
    if cap is not None:
        cfg["sweep"][OPTION] = cap
    path = tmp_path / "qmc-coupling.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.mark.parametrize("bad", [True, False, 0, -1, 3.5, "8"])
def test_yaml_cap_rejects_invalid_values_without_coercion(tmp_path, bad):
    path = _yaml_config(tmp_path, bad)
    with pytest.raises((ValueError, TypeError), match=OPTION):
        load_workflow_config(path)


def test_yaml_default_none_and_configured_execution_preserve_results(tmp_path):
    default = load_workflow_config(_yaml_config(tmp_path))
    assert getattr(default.sweep, OPTION) is None
    eager, _ = run_workflow(default, progress=False)
    configured = load_workflow_config(_yaml_config(tmp_path, 3))
    assert getattr(configured.sweep, OPTION) == 3
    actual, _ = run_workflow(configured, progress=False)
    assert actual.rows == eager.rows


def test_cli_dry_run_reports_and_estimator_forwards_requested_cap(
    tmp_path, monkeypatch, capsys
):
    from sft_wick.workflow import Expansion
    from sft_wick.workflow.cli import main

    path = _yaml_config(tmp_path, 3)
    forwarded = []
    original = Expansion.evaluate

    def observe(self, *args, **kwargs):
        forwarded.append(kwargs.get(OPTION))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Expansion, "evaluate", observe)
    assert main(["run", str(path), "--dry-run", "--quiet"]) == 0
    output = capsys.readouterr().out
    assert f"{OPTION}=3" in output and "cost estimate" in output
    assert len(forwarded) >= 2 and set(forwarded) == {3}


def test_R_and_C_callbacks_still_receive_one_full_ordered_batch(monkeypatch):
    points = np.linspace(0.1, 0.9, 17)
    _install_draw(monkeypatch, points)
    events = []

    class Cache:
        model = SimpleNamespace(iso_R=True, diag_C=True, n_components=1)
        _c_splines = True

        def R_time_batch(self, t, s):
            events.append(("R", np.asarray(t).copy(), np.asarray(s).copy()))
            return np.ones(len(t))

        def C_diagonal_batch(self, t, s):
            events.append(("C", np.asarray(t).copy(), np.asarray(s).copy()))
            return np.full((len(t), 1), 2.0)

    propagators = (
        Propagator("R", None, None, "x", "s"),
        Propagator("C", None, None, "s", "y"),
    )
    ig = _integrand(Polynomial(), propagators)
    kwargs = dict(n_samples=17, seed=7)
    eager = ig.integrate_moment_qmc_vectorized(1.0, Cache(), **kwargs)
    eager_events = list(events)
    events.clear()
    got = ig.integrate_moment_qmc_vectorized(
        1.0, Cache(), qmc_coupling_chunk_size=3, **kwargs
    )
    np.testing.assert_array_equal(got, eager)
    assert [name for name, _, _ in events] == ["R", "C"]
    for actual, expected in zip(events, eager_events):
        assert actual[0] == expected[0]
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])
        assert len(actual[1]) == len(actual[2]) == 17
    np.testing.assert_array_equal(events[0][2], points)
    np.testing.assert_array_equal(events[1][1], points)


@pytest.mark.parametrize("cap", [1, 3, 129])
@pytest.mark.parametrize("kind", ["callable", "callable_batched"])
@pytest.mark.parametrize("integrate_over", [None, "all"])
def test_dense_matrix_response_and_covariance_keep_ordered_diagram_results(
    cap, kind, integrate_over
):
    from tests import test_matrix_r_batched as mb

    kw = dict(n_samples=128, seed=6, integrate_over=integrate_over)
    if integrate_over is None:
        kw["external_times"] = mb.EXT_TIMES
    eager = mb._dense_eval(kind, mb.OBS3, 1, (1, 0, 1), "qmc_vectorized", **kw)
    got = mb._dense_eval(
        kind,
        mb.OBS3,
        1,
        (1, 0, 1),
        "qmc_vectorized",
        qmc_coupling_chunk_size=cap,
        **kw,
    )
    # This nonsymmetric tensor has different time slopes on its ordered legs;
    # the dense non-normal response and off-diagonal C are independently
    # validated against scipy hand contractions in MB5's existing acceptance.
    assert len(got.per_diagram) == len(eager.per_diagram) > 0
    for actual, reference in zip(got.per_diagram, eager.per_diagram):
        assert actual.keys() == reference.keys()
        for key in actual:
            if key in {"value", "error"}:
                np.testing.assert_allclose(
                    actual[key], reference[key], rtol=2e-14, atol=2e-15
                )
            else:
                assert actual[key] == reference[key]
    np.testing.assert_allclose(got.total, eager.total, rtol=2e-14, atol=2e-15)
    assert (
        got.positions == eager.positions and got.component_pair == eager.component_pair
    )
    assert got.n_samples == eager.n_samples and got.seed == eager.seed


@pytest.mark.parametrize("cap", [1, 3, 7])
@pytest.mark.parametrize("kind", ["local_R", "local_raw", "all_R_zero", "invalid_C"])
def test_matrix_support_and_nonfinite_selection_remain_global(monkeypatch, cap, kind):
    points = np.linspace(0.1, 0.9, 17)
    _install_draw(monkeypatch, points)
    events = []

    class Cache:
        model = SimpleNamespace(iso_R=False, diag_C=False, n_components=2)
        _c_splines = True

        def R_matrix_batch(self, t, s):
            events.append(("R", len(t)))
            out = np.zeros((len(t), 2, 2))
            if kind != "all_R_zero":
                out[:, 0, 0] = 1.0
            if kind == "local_R":
                out[:3, 0, 0] = 0.0
            return out

        def C_matrix_batch(self, t, s):
            events.append(("C", len(t)))
            if kind in {"all_R_zero", "invalid_C"}:
                return np.ones((len(t), 1, 2, 2))  # Selected only for nonzero R.
            out = np.ones((len(t), 2, 2))
            out[:3, 0, 0] = np.nan
            return out

    def kernel(n, t):
        out = np.zeros((len(t[0]), 2, 2), dtype=complex)
        out[:, 0, 0] = 1j
        if kind == "local_raw":
            out[np.asarray(t)[0] <= points[2], 0, 0] = 0.0
        return out

    kernel.vectorized = True
    term = DiagramTerm(
        propagators=(
            Propagator("R", "i", "j", "x", "s"),
            Propagator("C", "i", "j", "s", "y"),
        ),
        coupling_sum=Symbol("K", indices=("i", "j"), spatial_args=("s",)),
        rational_prefactor=Rational(1, 1),
        integration_vars=("s",),
        summation_indices=(("i", 2), ("j", 2)),
        n_response=1,
    )
    ig = term.build_integrand({"K": kernel})
    kw = dict(n_samples=17, seed=7)
    if kind == "invalid_C":
        for bound in (None, cap):
            with pytest.raises(ValueError, match="C|shape"):
                ig.integrate_moment_qmc_vectorized(
                    1.0,
                    Cache(),
                    qmc_coupling_chunk_size=bound,
                    **kw,
                )
    else:
        with np.errstate(invalid="ignore"):
            eager = ig.integrate_moment_qmc_vectorized(1.0, Cache(), **kw)
            got = ig.integrate_moment_qmc_vectorized(
                1.0,
                Cache(),
                qmc_coupling_chunk_size=cap,
                **kw,
            )
        if kind == "all_R_zero":
            np.testing.assert_array_equal(got, (0.0, 0.0))
        else:
            assert np.isnan(got).all()  # Whole-support 0*NaN remains observable.
        np.testing.assert_array_equal(got, eager)
    assert events == [("R", 17), ("C", 17)] * 2


@pytest.mark.parametrize("cap", [1, 3, 129])
def test_each_matrix_component_keeps_its_own_reality_scale(cap):
    import sft_wick as sw
    from tests.test_gl_chunked_integration import AssignmentPhaseKappa

    exp, props = _cumulant(
        AssignmentPhaseKappa(),
        n=2,
        vectorized=True,
        equal_time=True,
        linear=sw.ExplicitR(
            R_time=lambda t, s: np.eye(2) if t >= s else np.zeros((2, 2)),
            iso_R=False,
        ),
    )
    for bound in (None, cap):
        with pytest.raises(ValueError, match="imaginary|Im"):
            exp.evaluate(
                props,
                positions=POS3,
                t_final=1.4,
                component_pair=(1, 1, 1),
                orders=[1],
                n_samples=128,
                seed=7,
                qmc_coupling_chunk_size=bound,
            )


@pytest.mark.parametrize("cap", [3, 129])
@pytest.mark.parametrize("equal_time", [False, True])
def test_fixed_vector_positions_keep_their_axes_on_three_row_blocks(cap, equal_time):
    from tests.test_gl_chunked_integration import K_RAW

    coordinates = {
        "x": np.array([0.0, 0.1, 0.2]),
        "y": np.array([0.3, 0.4, 0.5]),
        "z": np.array([-0.4, -0.3, -0.2]),
    }
    observed = []

    def kernel(n, t):
        observed.append(np.asarray(n).copy())
        assert n.shape == (3, len(t[0]), 3)
        for leg in range(3):
            np.testing.assert_array_equal(
                n[leg], np.broadcast_to(n[leg, 0], n[leg].shape)
            )
        return np.broadcast_to(K_RAW, (len(t[0]),) + K_RAW.shape)

    exp, props = _cumulant(kernel, vectorized=True, equal_time=equal_time)
    kw = dict(
        positions=coordinates,
        t_final=1.4,
        component_pair=(0, 1, 1),
        orders=[1],
        n_samples=128,
        seed=7,
    )
    eager = exp.evaluate(props, **kw)
    observed.clear()
    got = exp.evaluate(props, qmc_coupling_chunk_size=cap, **kw)
    np.testing.assert_array_equal(got.total, eager.total)
    np.testing.assert_array_equal(
        [(d["value"], d["error"]) for d in got.per_diagram],
        [(d["value"], d["error"]) for d in eager.per_diagram],
    )
    assert observed and all(a.shape[1] <= min(cap, 128) for a in observed)


@pytest.mark.parametrize("method", ["qmc", "qmc_vectorized"])
def test_none_keeps_legacy_custom_vectorized_method_signature(method):
    class LegacyIntegrand:
        def integrate_moment_qmc_vectorized(
            self,
            lambda_f,
            cache,
            *,
            t_min,
            direction,
            n_samples,
            seed,
            positions,
            integrate_over,
            external_times,
        ):
            return 7.0, 0.125

    np.testing.assert_array_equal(
        integrate_moment(LegacyIntegrand(), 1.0, _cache(), method=method),
        (7.0, 0.125),
    )


class NativeSelectedPromise:
    """Materialize the complete native tensor before selecting a scalar leaf.

    Row IDs connect a public QMC call to predetermined difficult native times.
    This is a custom promise protocol test, not an alternative causal mapper.
    Retained raw outputs deliberately make no claim about whole-call memory.
    """

    def __init__(self, law, times, selected=None, retain=False):
        self.law = law
        self.times = np.asarray(times).copy()
        self.selected = (0,) * law.order if selected is None else selected
        self.calls = []
        self.raw_blocks = []
        self.retain = retain

    def evaluate_at_batch(self, label_t, label_x, n_samples):
        row_ids = np.rint(label_t["s"] * (self.times.shape[1] + 1)).astype(int) - 1
        assert row_ids.shape == (n_samples,)
        self.calls.append(n_samples)
        time_batch = self.times[:, row_ids]
        full = self.law(np.zeros_like(time_batch), time_batch)
        if self.retain:
            self.raw_blocks.append(full.copy())
        # Budget and overflow checks have already covered EVERY component.
        return full[(slice(None),) + self.selected]


def _native_qmc(monkeypatch, law, times, cap, *, selected=None, retain=False):
    n = np.asarray(times).shape[1]
    trace = _install_draw(monkeypatch, np.arange(1, n + 1) / (n + 1))
    promise = NativeSelectedPromise(law, times, selected, retain)
    ig = replace(_integrand(Polynomial()), dynamic_coupling=promise)
    result = _run(ig, cap, n=n)
    assert trace["draw"] == [n]
    return result, promise


def _normal_native_times(n, order=3):
    row = np.arange(n, dtype=float) / max(1, n - 1)
    return np.array([0.4 + 0.2 * leg + 0.1 * row for leg in range(order)])


@pytest.mark.parametrize("cap", [1, 3, 129])
@pytest.mark.parametrize("complex_storage", [False, True])
def test_native_ou_budget_applies_to_every_actual_requested_tensor_batch(
    monkeypatch, cap, complex_storage
):
    from sft_wick.white_cumulant import OUWhiteCumulant

    n = 259
    amplitude = np.arange(1, 9, dtype=float).reshape((2,) * 3) / 7
    if complex_storage:
        amplitude = amplitude.astype(complex)  # Zero phase, 16-byte components.
    times = _normal_native_times(n)
    per_row_bytes = amplitude.nbytes
    law = OUWhiteCumulant(amplitude, [0.7, 1.3], max_output_bytes=cap * per_row_bytes)
    with pytest.raises(MemoryError, match="batch"):
        _native_qmc(monkeypatch, law, times, None)
    result, promise = _native_qmc(monkeypatch, law, times, cap)
    assert promise.calls == [min(cap, n - start) for start in range(0, n, cap)]
    assert all(size * per_row_bytes <= law.max_output_bytes for size in promise.calls)
    assert sum(promise.calls) == n
    control = OUWhiteCumulant(amplitude, [0.7, 1.3])
    eager, _ = _native_qmc(monkeypatch, control, times, None)
    np.testing.assert_array_equal(result, eager)
    assert np.isfinite(result).all() and result[0] != 0


@pytest.mark.parametrize("cap", [None, 1, 3, 8])
def test_native_ou_unselected_overflow_component_is_never_hidden_by_partitioning(
    monkeypatch, cap
):
    from sft_wick.white_cumulant import OUWhiteCumulant

    amplitude = np.zeros((2, 2))
    amplitude[0, 0] = 0.5  # Selected component stays finite, including the last row.
    amplitude[1, 1] = np.finfo(float).max
    times = np.full((2, 7), 0.1)
    times[:, -1] = np.nextafter(1.0, 2.0)
    law = OUWhiteCumulant(amplitude, 0.0)
    with pytest.raises(OverflowError, match="float64"):
        _native_qmc(monkeypatch, law, times, cap, selected=(0, 0))


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("cap", [None, 1, 3, 129])
def test_native_ou_late_invalid_time_remains_an_explicit_failure(
    monkeypatch, invalid, cap
):
    from sft_wick.white_cumulant import OUWhiteCumulant

    times = _normal_native_times(8)
    times[-1, -1] = invalid
    law = OUWhiteCumulant(np.ones((2,) * 3), [0.7, 1.3])
    with pytest.raises(ValueError, match="t_list.*finite"):
        _native_qmc(monkeypatch, law, times, cap)


@pytest.mark.parametrize("cap", [1, 3, 128, 129, 512])
@pytest.mark.parametrize("rates", [[0.7, 0.7], [0.7, 1.3]])
def test_native_ordinary_finite_rows_keep_exact_complete_result_across_caps(
    monkeypatch, cap, rates
):
    from sft_wick.white_cumulant import OUWhiteCumulant

    amplitude = np.arange(1, 9, dtype=float).reshape((2,) * 3) / 7
    times = _normal_native_times(259)
    law = OUWhiteCumulant(amplitude, rates)
    eager, eager_promise = _native_qmc(
        monkeypatch, law, times, None, selected=(0, 1, 0), retain=True
    )
    actual, promise = _native_qmc(
        monkeypatch, law, times, cap, selected=(0, 1, 0), retain=True
    )
    np.testing.assert_array_equal(actual, eager)
    np.testing.assert_array_equal(
        np.concatenate(promise.raw_blocks), np.concatenate(eager_promise.raw_blocks)
    )
    assert sum(promise.calls) == 259
    assert max(promise.calls) <= min(cap, 259)
    assert actual[0] > 0 and np.isfinite(actual).all()


@pytest.mark.parametrize("cap", [1, 3, 127, 128, 129, 512])
def test_native_mixed_normal_extreme_rows_match_180_digit_all_component_reference(
    monkeypatch, cap
):
    import itertools
    import warnings

    from sft_wick.white_cumulant import OUWhiteCumulant
    from tests.test_ou_white_cumulant import _high_precision_value

    amplitude = np.arange(1, 9, dtype=float).reshape((2,) * 3) / 7
    gamma = np.array([0.7, 1.3])
    lower = 0.2
    n = 259
    ordinary = np.array([0.8, 1.0, 0.9])
    extreme = np.array([1e308] * 3)
    extreme_rows = np.arange(n) % 7 == 3
    times = np.tile(ordinary[:, None], (1, n))
    times[:, extreme_rows] = extreme[:, None]
    references = []
    for column in (ordinary, extreme):
        tensor = np.empty_like(amplitude)
        for indices in itertools.product(range(2), repeat=3):
            tensor[indices] = _high_precision_value(
                amplitude[indices], gamma[list(indices)], column, lower
            )
        references.append(tensor)
    reference = np.stack([references[int(flag)] for flag in extreme_rows])
    law = OUWhiteCumulant(amplitude, gamma, t_min=lower)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        eager, eager_promise = _native_qmc(
            monkeypatch, law, times, None, selected=(0, 1, 0), retain=True
        )
        actual, promise = _native_qmc(
            monkeypatch, law, times, cap, selected=(0, 1, 0), retain=True
        )
    assert np.isfinite(reference).all() and np.all(reference > 0)
    np.testing.assert_allclose(
        np.concatenate(eager_promise.raw_blocks), reference, rtol=2e-12, atol=0
    )
    np.testing.assert_allclose(
        np.concatenate(promise.raw_blocks), reference, rtol=2e-12, atol=0
    )
    np.testing.assert_allclose(actual, eager, rtol=2e-12, atol=0)
    assert sum(promise.calls) == n and max(promise.calls) <= min(cap, n)


def test_raw_scalar_dtype_drift_is_outside_the_opt_in_contract_even_if_final_dtype_is_stable(
    monkeypatch,
):
    import warnings

    points = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.85, 0.9])
    _install_draw(monkeypatch, points)
    contracted_dtypes = []
    original = DiagramIntegrand._dynamic_values

    def observe(self, couplings, *args, **kwargs):
        contracted_dtypes.append(np.asarray(couplings).dtype)
        return original(self, couplings, *args, **kwargs)

    monkeypatch.setattr(DiagramIntegrand, "_dynamic_values", observe)

    def raw_dtype_changes(n, t):
        return 1.0 if t[0] < 0.5 else complex(1.0, 0.01)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", np.exceptions.ComplexWarning)
        eager = _run(_integrand(raw_dtype_changes), None, n=8)
    np.testing.assert_array_equal(eager, (1.0, 0.0))
    assert any(
        issubclass(item.category, np.exceptions.ComplexWarning) for item in caught
    )
    # A later group starts with a complex raw scalar; the existing promise now
    # preserves that imaginary part. Its FINAL contracted dtype is still the
    # same complex dtype, so the chunk assembly cannot diagnose raw instability.
    with pytest.raises(ValueError, match="imaginary|Im"):
        _run(_integrand(raw_dtype_changes), 3, n=8)
    assert len(contracted_dtypes) == 2
    assert contracted_dtypes[0] == contracted_dtypes[1]
    assert contracted_dtypes[0].kind == "c"
