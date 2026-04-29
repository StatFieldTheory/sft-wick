"""Tests for the spacetime-dependent (callable) coupling path.

Covers the ``DynamicCouplingPromise`` path in
``sft_wick.evaluate.DiagramIntegrand.integrate_moment_qmc_vectorized``
that is triggered when any ``coupling_values[name]`` is callable
(typically a non-local ``κ^{(m)}`` vertex like demo2's ``κ^{(3)}``).

Currently locked here:

* **DC1** -- ``NotImplementedError`` is raised when the dynamic
  coupling has propagator-indexed contraction (the v1 limitation
  documented at evaluate.py:2199-2206). Locking this contract
  prevents a future "helpful refactor" from silently dropping the
  guard before a proper implementation lands.
* **WF6** -- end-to-end FK (κ^{(3)}) integration: a constant
  callable κ^{(3)} routed through ``DynamicCouplingPromise`` must
  produce the same numerical result as the same constant tensor
  routed through the static fast path. This locks the
  static-vs-dynamic equivalence -- any future vectorisation of the
  per-sample loop (deferred A/B/C/D in the project todo) must
  preserve it.

Future tests in this file will cover the vectorised dynamic-coupling
path (deferred A/B/C/D).
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.evaluate import DynamicCouplingPromise


# Same demo1 closed-form C as the other test modules -- skipping
# dblquad on the demo1 OU kernel pulls each WF6/DC1 run from
# ~5-10s down to ~0.5s.
_DEMO1_CLOSED_FORM_SRC = (
    Path(__file__).resolve().parents[1]
    / "examples" / "demo1" / "c_closed_form.py"
)


def _load_demo1_C_fn():
    """Import ``C_fn`` from ``examples/demo1/c_closed_form.py``."""
    spec = _ilu.spec_from_file_location(
        "demo1_c_closed_form_dyn", _DEMO1_CLOSED_FORM_SRC,
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.C_fn


def _make_dynamic_kappa3_system():
    """Build a small System whose order-2 expansion contains FK
    diagrams driven by a callable ``κ^{(3)}``.

    Mirrors the structure of ``examples/demo2_config.yaml`` but
    keeps everything inline so the test does not depend on file
    layout.
    """
    N = 2
    F = np.zeros((N, N, N))
    F[0, 1, 1] = 1.0
    F[1, 0, 1] = 0.5
    F[1, 1, 0] = 0.5

    def kappa3_fn(n_list, t_list):
        # Component-diagonal κ^{(3)}_abc with a mild spacetime
        # envelope. The exact form is irrelevant for DC1 because we
        # patch the promise; we only need the build_integrand path
        # to recognise this as a dynamic coupling.
        n = np.asarray(n_list, dtype=float)
        t = np.asarray(t_list, dtype=float)
        envelope = float(
            np.exp(-abs(t[0] - t[1]) - abs(t[0] - t[2]))
            * np.exp(-abs(n[0] - n[1]) - abs(n[0] - n[2]))
        )
        K = np.zeros((N, N, N), dtype=complex)
        for a in range(N):
            K[a, a, a] = (1j / 6.0) * envelope
        return K

    return sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        vertices=[sw.LocalVertex("F", coupling=F)],
        nonlocal_vertices=[
            sw.NonLocalVertex("K", order=3, coupling=kappa3_fn),
        ],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            ),
        ),
    )


def test_DC1_prop_indexed_dynamic_raises_notimplemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dynamic-coupling QMC path raises ``NotImplementedError``
    when ``DynamicCouplingPromise.evaluate_at`` returns a non-scalar
    array (i.e. propagator-indexed dynamic coupling).

    See evaluate.py:2199-2206 for the contract. We trigger the
    branch by patching ``evaluate_at`` to return a 1-D array; the
    real prop-indexed path is more involved to set up and is
    blocked by the same NotImplementedError anyway, so we do not
    need a "real" prop-indexed diagram for the contract lock.
    """
    system = _make_dynamic_kappa3_system()
    expansion = system.expand(
        ("phi_a(x)", "phi_b(y)"),
        orders=[2],
    )
    props = system.propagators(
        t_max=2.0, n_grid_t=8, c_closed_form=_load_demo1_C_fn(),
    )

    # Pick the first FK diagram -- it routes through the dynamic
    # coupling path because K is a callable vertex.
    fk_dts = [
        dt for dt in expansion.dts_by_order[2]
        if expansion._vertex_type_label(dt) == "FK"
    ]
    assert fk_dts, "expected at least one FK diagram in the order-2 expansion"

    coupling_values = system.build_coupling_values()
    fixed_indices = {"a": 0, "b": 1}
    ig = fk_dts[0].build_integrand(coupling_values, fixed_indices=fixed_indices)
    assert ig.dynamic_coupling is not None, (
        "FK diagram must take the dynamic-coupling path"
    )

    # Force evaluate_coupling to return a non-scalar array so the
    # downstream ``coup0.ndim != 0`` probe inside
    # ``DynamicCouplingPromise.evaluate_at_batch`` raises. Patching
    # the contraction itself (rather than the dispatch helper) keeps
    # this test pinned to the real code path the
    # NotImplementedError exists to guard.
    real_evaluate_coupling = ig.diagram_term.evaluate_coupling

    def _fake_evaluate_coupling(self, *args, **kwargs):  # noqa: ARG001
        return np.zeros(2)  # ndim == 1 triggers the raise

    monkeypatch.setattr(
        type(ig.diagram_term), "evaluate_coupling", _fake_evaluate_coupling,
    )

    with pytest.raises(
        NotImplementedError,
        match="Dynamic coupling with propagator-indexed",
    ):
        ig.integrate_moment_qmc_vectorized(
            lambda_f=2.0,
            cache=props.cache,
            n_samples=2 ** 6,
            seed=0,
            positions={"x": 0.0, "y": 0.5},
        )


# =====================================================================
# WF6 -- end-to-end FK (kappa^{(3)}) integration: static vs dynamic
# =====================================================================


def _make_kappa3_system_with_K(K_coupling):
    """Build a 2-component System whose order-2 expansion contains
    FK diagrams driven by ``K``. ``K_coupling`` may be either a
    constant ``(N, N, N)`` tensor or a callable
    ``fn(n_list, t_list) -> (N, N, N)``.
    """
    N = 2
    F = np.zeros((N, N, N))
    F[0, 1, 1] = 1.0
    F[1, 0, 1] = 0.5
    F[1, 1, 0] = 0.5

    return sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        vertices=[sw.LocalVertex("F", coupling=F)],
        nonlocal_vertices=[
            sw.NonLocalVertex("K", order=3, coupling=K_coupling),
        ],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            ),
        ),
    )


def _integrate_FK_total(
    system: sw.System, *, t_max: float, n_samples: int, seed: int,
):
    """Expand at order 2, build the propagator cache, integrate the
    FK channel only, and return the summed value."""
    expansion = system.expand(
        ("phi_a(x)", "phi_b(y)"), orders=[2],
    )
    props = system.propagators(
        t_max=t_max, n_grid_t=20, c_closed_form=_load_demo1_C_fn(),
    )

    fk_dts = [
        dt for dt in expansion.dts_by_order[2]
        if expansion._vertex_type_label(dt) == "FK"
    ]
    assert fk_dts, "expected FK diagrams"

    coupling_values = system.build_coupling_values()
    fixed_indices = {"a": 0, "b": 1}

    from sft_wick.evaluate import integrate_diagrams

    total, _details = integrate_diagrams(
        fk_dts,
        coupling_values=coupling_values,
        lambda_f=t_max,
        cache=props.cache,
        method="qmc_vectorized",
        n_samples=n_samples,
        seed=seed,
        fixed_indices=fixed_indices,
        n_jobs=1,
    )
    return total


def test_WF6_kappa3_static_vs_dynamic_match() -> None:
    """A constant callable ``kappa^{(3)}`` routed via the dynamic-
    coupling path must produce the same FK total as the same
    constant tensor routed via the static fast path.

    Constructs both systems on top of the same physics
    (component-diagonal K with two non-zero entries), then runs
    them with identical seeds and grids. A single QMC sample
    sequence is therefore drawn the same way in both paths, so the
    only legitimate source of difference is the per-sample
    DynamicCouplingPromise overhead -- which must produce
    bit-identical numerics for a constant callable.
    """
    N = 2
    K_const = np.zeros((N, N, N))
    K_const[0, 0, 0] = 0.3
    K_const[1, 1, 1] = 0.5

    def K_callable(n_list, t_list):  # noqa: ARG001
        # Constant: ignore the spacetime arguments and return the
        # same tensor every sample. The MSR factor (i / 6) is
        # applied automatically by the wrapper -- pass the bare
        # tensor here, just like the static case.
        return K_const

    system_static = _make_kappa3_system_with_K(K_const)
    system_dynamic = _make_kappa3_system_with_K(K_callable)

    # Sanity: the static system's K passes through as an ndarray;
    # the dynamic system's K is a callable.
    nlv_static = system_static.nonlocal_vertices[0]
    nlv_dynamic = system_dynamic.nonlocal_vertices[0]
    assert not callable(nlv_static.coupling)
    assert callable(nlv_dynamic.coupling)

    common = dict(t_max=2.0, n_samples=2 ** 12, seed=20260428)
    total_static = _integrate_FK_total(system_static, **common)
    total_dynamic = _integrate_FK_total(system_dynamic, **common)

    # The static path is a vectorised single tensor multiplication.
    # The dynamic path computes a fresh tensor per sample. For a
    # constant callable the per-sample tensors are identical, so
    # the two paths' QMC integrands are identical sample-for-
    # sample. Therefore the totals must match to float64 noise.
    np.testing.assert_allclose(
        total_dynamic, total_static,
        rtol=1e-10, atol=1e-12,
        err_msg=(
            f"static FK total {total_static!r} disagrees with dynamic "
            f"FK total {total_dynamic!r}; static-vs-dynamic equivalence "
            f"is the WF6 contract that protects future vectorisation "
            f"work."
        ),
    )


def test_WF6_kappa3_dynamic_spacetime_dependence_changes_result() -> None:
    """Sanity guard for WF6: a *spacetime-dependent* callable
    ``kappa^{(3)}`` must produce a numerically different total from
    the constant version.

    Without this guard, ``test_WF6_kappa3_static_vs_dynamic_match``
    could pass against a buggy dynamic path that silently ignored
    the per-sample times/positions. Forcing the result to differ
    when the callable actually uses its arguments closes that hole.
    """
    N = 2
    K_const = np.zeros((N, N, N))
    K_const[0, 0, 0] = 0.3
    K_const[1, 1, 1] = 0.5

    def K_callable_spacetime(n_list, t_list):
        # Multiply the constant tensor by a non-trivial spacetime
        # envelope so the per-sample value actually varies.
        n = np.asarray(n_list, dtype=float)
        t = np.asarray(t_list, dtype=float)
        envelope = float(
            np.exp(-abs(t[0] - t[1]))
            * np.exp(-abs(n[0] - n[2]))
        )
        return envelope * K_const

    system_const = _make_kappa3_system_with_K(K_const)
    system_var = _make_kappa3_system_with_K(K_callable_spacetime)

    common = dict(t_max=2.0, n_samples=2 ** 12, seed=20260428)
    total_const = _integrate_FK_total(system_const, **common)
    total_var = _integrate_FK_total(system_var, **common)

    rel = abs(total_var - total_const) / (abs(total_const) + 1e-15)
    assert rel > 1e-3, (
        f"spacetime-dependent callable kappa^(3) gave the same total as "
        f"the constant tensor (rel diff {rel:.2e}); the dynamic path is "
        f"likely ignoring its per-sample arguments. "
        f"const={total_const!r}, var={total_var!r}"
    )


# =====================================================================
# WF7 -- vectorised callable contract (NonLocalVertex.coupling_vectorized)
# =====================================================================


def test_WF7_vectorized_callable_matches_per_sample() -> None:
    """A callable kappa^{(3)} marked
    ``NonLocalVertex(coupling_vectorized=True)`` must produce
    bit-identical FK totals to the same callable expressed as a
    per-sample function.

    The contract for vectorised callables is:

    * Inputs ``n_list`` and ``t_list`` arrive as shape
      ``(m_legs, n_samples)`` arrays (instead of length-m vectors).
    * Output is a tensor with leading axis ``n_samples``, i.e.
      shape ``(n_samples,) + (N,)*order``.

    The two paths share the same underlying physics (same bare
    kernel, same QMC seed, same diagram set), so any divergence
    larger than float64 roundoff is a real bug in the dispatch.
    """
    N = 2

    K_const = np.zeros((N, N, N))
    K_const[0, 0, 0] = 0.3
    K_const[1, 1, 1] = 0.5

    def K_per_sample(n_list, t_list):
        # Per-sample contract: 1-D length-m inputs, return (N, N, N).
        n = np.asarray(n_list, dtype=float)
        t = np.asarray(t_list, dtype=float)
        envelope = float(
            np.exp(-abs(t[0] - t[1])) * np.exp(-abs(n[0] - n[2]))
        )
        return envelope * K_const

    def K_vectorized(n_2d, t_2d):
        # Batched contract: (m, n_samples) inputs, return
        # (n_samples, N, N, N) -- the same envelope formula written
        # in a single ufunc-friendly expression.
        n = np.asarray(n_2d, dtype=float)
        t = np.asarray(t_2d, dtype=float)
        envelope = np.exp(-np.abs(t[0] - t[1])) * np.exp(-np.abs(n[0] - n[2]))
        return envelope[:, None, None, None] * K_const[None, :, :, :]

    system_per_sample = _make_kappa3_system_with_K(K_per_sample)
    # Replace the auto-built non-local vertex with one that opts in
    # to the vectorised contract. ``System`` is a frozen dataclass,
    # so we re-create it via dataclass.replace.
    import dataclasses
    system_vectorized = dataclasses.replace(
        _make_kappa3_system_with_K(K_vectorized),
        nonlocal_vertices=(
            sw.NonLocalVertex(
                name="K", order=3,
                coupling=K_vectorized,
                coupling_vectorized=True,
            ),
        ),
    )

    common = dict(t_max=2.0, n_samples=2 ** 12, seed=20260428)
    total_per_sample = _integrate_FK_total(system_per_sample, **common)
    total_vectorized = _integrate_FK_total(system_vectorized, **common)

    np.testing.assert_allclose(
        total_vectorized, total_per_sample,
        rtol=1e-10, atol=1e-12,
        err_msg=(
            f"vectorized FK total {total_vectorized!r} disagrees with "
            f"per-sample FK total {total_per_sample!r}; "
            f"NonLocalVertex(coupling_vectorized=True) dispatch is "
            f"the WF7 contract that opens up the user-facing fast "
            f"path for heavy callables."
        ),
    )


# =====================================================================
# WF8 -- d-dim (vector) positions through the dynamic-coupling promise
# =====================================================================


def _make_kappa3_system_with_K_handling_vector(N: int = 2):
    """A System whose K-callable accepts BOTH scalar and vector
    per-leg position shapes.  Used by WF8 to verify the d-dim path
    on both ``method='qmc_vectorized'`` and ``method='gauss_legendre'``.
    """
    F = np.zeros((N, N, N))
    F[0, 1, 1] = 1.0
    F[1, 0, 1] = 0.5
    F[1, 1, 0] = 0.5

    K_const = np.zeros((N, N, N))
    K_const[0, 0, 0] = 0.3
    K_const[1, 1, 1] = 0.5

    def K_callable(n_list, t_list):
        n = np.asarray(n_list, dtype=float)
        t = np.asarray(t_list, dtype=float)
        # For scalar per-leg positions: n.shape == (m,).
        # For d-dim per-leg positions: n.shape == (m, d).  Reduce
        # the d-dim case to scalar by taking the leg-norm so the
        # callable produces a finite real envelope in either case.
        if n.ndim == 2:
            n_scalar = np.linalg.norm(n, axis=-1)
        else:
            n_scalar = n
        envelope = float(
            np.exp(-abs(t[0] - t[1])) * np.exp(-abs(n_scalar[0] - n_scalar[2]))
        )
        return envelope * K_const

    return sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=sw.DiagonalA(gamma=[1.0] * N),
        vertices=[sw.LocalVertex("F", coupling=F)],
        nonlocal_vertices=[
            sw.NonLocalVertex("K", order=3, coupling=K_callable),
        ],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            ),
        ),
    )


@pytest.mark.parametrize(
    "method,kw",
    [
        ("qmc_vectorized", {"n_samples": 2 ** 10, "seed": 20260429}),
        ("gauss_legendre", {"n_gauss": 4}),
    ],
)
def test_WF8_dynamic_kappa3_with_3d_positions(method, kw):
    """Lift of the d-dim rejection in
    ``DynamicCouplingPromise.evaluate_at_batch`` (originally
    raised ``NotImplementedError("d-dim spatial positions not
    supported... K3 leg at label 'y_1' has position shape (3,)")``).

    Pre-fix: passing 3-D vector positions to a system that uses a
    callable kappa^(3) raised on both batch integrators (qmc_vectorized
    and gauss_legendre, both call into the same promise).

    Post-fix: the per-symbol broadcast path forwards vector positions
    as-is; the user callable sees per-leg slices of shape ``(m, d)``
    and is expected to handle them.

    Locks: the integration runs to completion and returns a finite
    nontrivial value on both batch integrators.
    """
    system = _make_kappa3_system_with_K_handling_vector()
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2])
    props = system.propagators(
        t_max=2.0, n_grid_t=20,
        c_closed_form=_load_demo1_C_fn(),
        c_closed_form_only=True,
        c_closed_form_vectorized=True,
    )

    result = expansion.evaluate(
        props,
        positions={
            "x": np.array([0.0, 0.0, 0.0]),   # 3-D: triggers the rejection pre-fix
            "y": np.array([0.5, 0.0, 0.0]),
        },
        t_final=2.0,
        component_pair=(0, 1),
        orders=[2],
        vertex_types={"FK"},
        method=method,
        **kw,
    )

    assert np.isfinite(result.total), (
        f"non-finite total {result.total!r} from method={method!r} "
        f"with 3-D vector positions"
    )
    assert abs(result.total) > 1e-15, (
        f"trivially-zero total {result.total!r} from method={method!r}; "
        f"the K-callable should give a nontrivial envelope at this grid"
    )
