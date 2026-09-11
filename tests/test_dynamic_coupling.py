"""Tests for the spacetime-dependent (callable) coupling path.

``DiagramTerm.build_integrand`` attaches a
``sft_wick.evaluate.DynamicCouplingPromise`` to the integrand when a
callable ``coupling_values[name]`` appears in the diagram's coupling
sum (typically a non-local ``κ^{(m)}`` vertex like demo2's
``κ^{(3)}``).  The ``DiagramIntegrand`` integrators use it as follows:

* ``integrate_moment_qmc_vectorized`` and
  ``integrate_moment_gauss_legendre`` call
  ``DynamicCouplingPromise.evaluate_at_batch`` once per diagram, with
  every sample;
* ``integrate_moment_qmc``, the scalar loop behind
  ``method='qmc_scalar'``, calls ``dynamic_coupling_array``, and
  through it ``DynamicCouplingPromise.evaluate_at``, once per sample;
* ``_evaluate_zero_dimensional``, which the qmc_vectorized,
  gauss_legendre and nquad integrators use for a diagram with no time
  integral left, calls ``evaluate_at_batch`` with one sample.

``integrate_moment_nquad`` otherwise raises ``NotImplementedError``
for a callable coupling.

Currently locked here:

* **DC1** -- propagator-indexed dynamic coupling (a surviving
  component index that lands on a C propagator, e.g. demo2's
  order-4 F³κ³ diagrams) agrees with the static-tensor path to
  machine precision.  A callable that ignores its arguments and
  returns a constant tensor is mathematically the static tensor, so
  the two routes must return bit-comparable numbers; this is the
  boundary test for the feature that replaced the pre-0.4.0
  ``NotImplementedError``.
* **DC2** -- scalar QMC integrates a callable coupling to closed-form
  values; ``DiagramIntegrand.evaluate`` refuses a callable coupling
  without an explicit ``coupling_array``; a coupling leg with no
  propagator is still given a position.
* **WF6** -- end-to-end FK (κ^{(3)}) integration: a constant
  callable κ^{(3)} routed through ``DynamicCouplingPromise`` must
  produce the same numerical result as the same constant tensor
  routed through the static fast path. This locks the
  static-vs-dynamic equivalence, which the batched evaluation in
  ``evaluate_at_batch`` must preserve.
* **WF7** -- a callable declared with ``coupling_vectorized=True``
  gives the same FK total as the same kernel under the per-sample
  contract.
* **WF8** -- 3-D vector leg positions integrate to a finite, nonzero
  FK total on qmc_vectorized and gauss_legendre.
* **WF9** -- the ``(n, t)`` argument shapes each contract receives for
  scalar, 2-D and 3-D positions on qmc_vectorized, gauss_legendre and
  qmc_scalar, and the agreement of the FK total across contracts and
  position kinds.
* **WF10** -- a per-sample callable is called once per sample and leg
  order on qmc_vectorized and gauss_legendre, also when the batched
  contraction falls back to the per-sample loop, and on qmc_scalar.
* **DC3** -- a single-component (N = 1) field with a constant callable
  coupling matches the static tensor on gauss_legendre and
  qmc_vectorized.
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.evaluate import DynamicCouplingPromise
from sft_wick.expressions import Symbol
from sft_wick.perturbation import DiagramTerm, Rational


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


def _make_constant_kappa3_system(*, dynamic: bool):
    """demo-2-shaped System whose order-4 F³κ³ diagrams carry a
    surviving propagator index, with a *constant* κ^{(3)}.

    ``already_R_contracted=True`` absorbs the three ψ-leg R
    propagators, which drops the order-4 FK time dimension from 6 to
    3 -- small enough for a tensor-product Gauss-Legendre rule in a
    unit test.  The R-absorption is structural (it depends on the
    vertex spec, not on whether the coupling is a tensor or a
    callable), so the static and dynamic systems produce the SAME
    diagrams and the SAME integration variables.

    ``dynamic=True`` wraps the identical tensor in a callable that
    ignores its arguments -- mathematically the static tensor, so
    the two routes must agree to machine precision.
    """
    N = 2
    F = np.zeros((N, N, N))
    F[0, 1, 1] = 1.0
    F[1, 0, 1] = 0.5
    F[1, 1, 0] = 0.5

    # Fully symmetric under leg exchange (as a cumulant must be) and
    # NOT component-diagonal, so the surviving ``i_0`` index really is
    # summed over rather than collapsing to a single term.
    K = np.zeros((N, N, N))
    K[0, 0, 0] = 3.0e-3
    K[1, 1, 1] = -1.0e-3
    for perm in ((0, 0, 1), (0, 1, 0), (1, 0, 0)):
        K[perm] = 7.0e-4
    for perm in ((0, 1, 1), (1, 0, 1), (1, 1, 0)):
        K[perm] = -4.0e-4

    if dynamic:
        def coupling(n_list, t_list):  # noqa: ARG001 -- constant on purpose
            return K
    else:
        coupling = K

    return sw.System(
        field=sw.FieldSpec("phi", n_components=N),
        linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        vertices=[sw.LocalVertex("F", coupling=F)],
        nonlocal_vertices=[
            sw.NonLocalVertex(
                "K", order=3, coupling=coupling, already_R_contracted=True,
            ),
        ],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            ),
        ),
    )


def _order4_fk_integrands(system, fixed_indices):
    """Order-4 FK diagrams of ``system``, as built integrands, keyed by
    the diagram's LaTeX form so the two systems can be matched up
    diagram-by-diagram rather than by list position."""
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[4])
    coupling_values = system.build_coupling_values()
    out = {}
    for dt in expansion.dts_by_order[4]:
        if expansion._vertex_type_label(dt) != "FK":
            continue
        key = dt.to_latex()
        out.setdefault(key, []).append(
            dt.build_integrand(coupling_values, fixed_indices=fixed_indices)
        )
    return out


@pytest.fixture(scope="module")
def dc1_setup():
    """Order-4 FK integrands for the static and dynamic constant-κ³
    systems, plus the propagator cache -- built once for both DC1 tests.

    The two DC1 tests need the *same* objects: the agreement test pairs
    static against dynamic, and the not-silently-zero companion
    re-integrates the prop-indexed subset of that same dynamic set.  The
    order-4 symbolic expansion costs ~3.1 s per system and was being run
    three times (static once, dynamic twice); here each runs exactly
    once.  Nothing that either test asserts happens in here: both
    expansions are still performed independently, the diagram sets are
    still compared in the test, and every integration -- the part
    actually under test -- still runs per test.

    The cache is built from the static system.  ``PropagatorModel`` is a
    function of ``field``/``linear``/``noise`` only, which the two
    systems share verbatim, and DC1 already fed one cache to both
    routes, so the cache is not part of either claim.
    """
    fixed_indices = {"a": 0, "b": 1}

    sw.reset_uid_counter()
    static_sys = _make_constant_kappa3_system(dynamic=False)
    static_igs = _order4_fk_integrands(static_sys, fixed_indices)

    sw.reset_uid_counter()
    dyn_sys = _make_constant_kappa3_system(dynamic=True)
    dyn_igs = _order4_fk_integrands(dyn_sys, fixed_indices)

    props = static_sys.propagators(
        t_max=2.0, n_grid_t=8, c_closed_form=_load_demo1_C_fn(),
    )
    return SimpleNamespace(
        static_igs=static_igs, dyn_igs=dyn_igs, cache=props.cache,
    )


def test_DC1_prop_indexed_dynamic_matches_static(dc1_setup) -> None:
    """A dynamic coupling whose contraction leaves a propagator index
    must agree with the same coupling supplied as a static tensor.

    This is demo2's blocked case: at order 4 the F³κ³ diagrams keep a
    single surviving index ``('i_0', 2)`` that sits on a C propagator,
    which the pre-0.4.0 ``DynamicCouplingPromise`` refused with
    ``NotImplementedError``.  A callable returning a constant tensor
    is the one configuration where both routes are legal, so their
    agreement is the boundary test for the feature.
    """
    static_igs = dc1_setup.static_igs
    dyn_igs = dc1_setup.dyn_igs

    assert set(static_igs) == set(dyn_igs), (
        "static and dynamic systems must produce the same diagram set"
    )
    n_prop_indexed = sum(
        1
        for igs in dyn_igs.values()
        for ig in igs
        if ig.diagram_term.propagator_indices
    )
    assert n_prop_indexed > 0, (
        "test is vacuous unless some order-4 FK diagram is prop-indexed"
    )

    kw = dict(
        lambda_f=1.5,
        cache=dc1_setup.cache,
        n_gauss=6,
        positions={"x": 0.0, "y": 0.5},
    )

    n_compared = 0
    for key in sorted(static_igs):
        for ig_s, ig_d in zip(static_igs[key], dyn_igs[key]):
            assert ig_s.dynamic_coupling is None
            assert ig_d.dynamic_coupling is not None
            v_s, _ = ig_s.integrate_moment_gauss_legendre(**kw)
            v_d, _ = ig_d.integrate_moment_gauss_legendre(**kw)
            assert v_d == pytest.approx(v_s, rel=1e-12, abs=1e-300), (
                f"static/dynamic mismatch on {key}: {v_s!r} vs {v_d!r}"
            )
            n_compared += 1
    assert n_compared == sum(len(v) for v in static_igs.values())


def test_DC1_prop_indexed_dynamic_is_not_silently_zero(dc1_setup) -> None:
    """Guard against the agreement test passing because BOTH routes
    return zero: at least one prop-indexed order-4 FK diagram must
    integrate to a non-negligible value.

    Shares the module fixture's dynamic integrands with the agreement
    test -- they are the same objects the agreement test compares, which
    is exactly what this guard has to speak about -- but integrates them
    itself.
    """
    igs = [
        ig
        for group in dc1_setup.dyn_igs.values()
        for ig in group
        if ig.diagram_term.propagator_indices
    ]
    assert igs, "no prop-indexed order-4 FK diagram to speak about"
    vals = [
        ig.integrate_moment_gauss_legendre(
            lambda_f=1.5, cache=dc1_setup.cache, n_gauss=6,
            positions={"x": 0.0, "y": 0.5},
        )[0]
        for ig in igs
    ]
    assert max(abs(v) for v in vals) > 1e-14, (
        f"all prop-indexed dynamic diagrams integrated to ~0: {vals}"
    )


def _dc2_cache():
    """A minimal REAL cache: no propagators in the term, so R_product over an
    empty tuple is 1 and the integrand is the coupling alone."""
    from sft_wick.evaluate import PropagatorCache, PropagatorModel
    model = PropagatorModel(R_time=lambda a, b: 1.0,
                            kappa2=lambda *a: np.array([[1.0]]),
                            n_components=1, diag_C=True, iso_R=True)
    return PropagatorCache(model, c_value_fn=lambda *a: np.array([[1.0]]))


def _dc2_integrand(fn):
    dt = DiagramTerm(
        propagators=(),
        coupling_sum=Symbol("K", indices=(), spatial_args=("s",)),
        rational_prefactor=Rational(1, 1),
        integration_vars=("s",),
        summation_indices=(),
        n_response=0,
    )
    ig = dt.build_integrand({"K": fn})
    assert ig.dynamic_coupling is not None
    return ig


@pytest.mark.parametrize("fn, exact", [
    (lambda n_list, t_list: 1.0,                        1.0),
    (lambda n_list, t_list: float(np.exp(-t_list[0])),  1.0 - np.exp(-1.0)),
    (lambda n_list, t_list: 3.0 * float(t_list[0]) ** 2, 1.0),
])
def test_DC2_scalar_qmc_integrates_the_callable_not_the_placeholder(fn, exact):
    """Scalar QMC must not silently integrate the dynamic placeholder zero.

    It used to satisfy that by REFUSING; it now satisfies it by computing the
    real thing.  Each case is a closed-form integral over ``s`` in [0, 1], so
    a path that fell back to the zeros placeholder would return 0.0 and a path
    that ignored the callable's arguments would return the wrong constant --
    neither can match all three.
    """
    val, _err = _dc2_integrand(fn).integrate_moment_qmc(
        lambda_f=1.0, cache=_dc2_cache(), n_samples=2 ** 14, seed=3)
    # abs=0.0 is a no-op here: the smallest ``exact`` is 1-e^-1 = 0.63, so
    # rel*expected = 6.3e-07 already dominates the 1e-12 default abs floor.
    assert val == pytest.approx(exact, rel=1e-6, abs=0.0)
    assert val != 0.0


def test_DC2_evaluate_still_refuses_a_callable_with_no_coupling_array():
    """The underlying protection stays: ``DiagramIntegrand.evaluate`` reads a
    zeros placeholder on the dynamic path, so calling it WITHOUT an explicit
    per-sample array would silently return 0.  The override is opt-in, and the
    refusal names the way out."""
    ig = _dc2_integrand(lambda n_list, t_list: 1.0)
    with pytest.raises(NotImplementedError, match="coupling_array"):
        ig.evaluate({"s": 0.5}, {}, _dc2_cache())


def test_DC2_a_coupling_leg_with_no_propagator_still_gets_a_position():
    """The leg ``s`` has no propagator attached, so it has no ``direction_map``
    entry.  Building the callable's position dict from that map alone raises a
    bare ``KeyError`` from inside ``evaluate_at`` -- the callable is asked for
    a leg nobody supplied a position for."""
    seen = []

    def fn(n_list, t_list):
        seen.append((tuple(np.ravel(n_list)), tuple(np.ravel(t_list))))
        return 1.0

    val, _ = _dc2_integrand(fn).integrate_moment_qmc(
        lambda_f=1.0, cache=_dc2_cache(), n_samples=64, seed=1)
    # abs=0.0 is a no-op here: expected is exactly 1.0, so rel*expected
    # = 1e-03 already dominates the 1e-12 default abs floor.
    assert val == pytest.approx(1.0, rel=1e-3, abs=0.0)
    assert seen, "the callable was never invoked"
    assert all(len(n) == 1 and len(t) == 1 for n, t in seen)


# =====================================================================
# WF6 -- end-to-end FK (kappa^{(3)}) integration: static vs dynamic
# =====================================================================


def _make_kappa3_system_with_K(
    K_coupling, *, coupling_vectorized: bool = False,
):
    """Build a 2-component System whose order-2 expansion contains
    FK diagrams driven by ``K``. ``K_coupling`` may be either a
    constant ``(N, N, N)`` tensor or a callable
    ``fn(n_list, t_list) -> (N, N, N)``; ``coupling_vectorized`` is
    passed to ``NonLocalVertex``.
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
            sw.NonLocalVertex(
                "K", order=3, coupling=K_coupling,
                coupling_vectorized=coupling_vectorized,
            ),
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


# =====================================================================
# WF9 -- the argument shapes a callable coupling receives
# =====================================================================


def _wf9_recording_kappa3(*, vectorized: bool, seen: list):
    """WF8's κ^(3) callable (same ``K_const``, same envelope) under the
    per-sample or the vectorised contract.  Every call appends
    ``(np.shape(n), np.shape(t))`` to ``seen``.

    The envelope depends on the positions only through each leg's norm,
    so a scalar position and a vector of the same length give the same
    value.
    """
    N = 2
    K_const = np.zeros((N, N, N))
    K_const[0, 0, 0] = 0.3
    K_const[1, 1, 1] = 0.5
    # Axes ahead of a position axis: legs, plus samples when vectorised.
    n_lead = 2 if vectorized else 1

    def K_callable(n_list, t_list):
        seen.append((np.shape(n_list), np.shape(t_list)))
        n = np.asarray(n_list, dtype=float)
        t = np.asarray(t_list, dtype=float)
        if n.ndim > n_lead:
            n = np.linalg.norm(n, axis=-1)
        envelope = np.exp(-np.abs(t[0] - t[1])) * np.exp(-np.abs(n[0] - n[2]))
        if vectorized:
            return envelope[:, None, None, None] * K_const[None, :, :, :]
        return float(envelope) * K_const

    return K_callable


#: Leg positions: x at the origin, y at distance 0.5 along the first axis.
_WF9_POSITIONS = {
    "scalar": {"x": 0.0, "y": 0.5},
    "2d": {"x": np.array([0.0, 0.0]), "y": np.array([0.5, 0.0])},
    "3d": {"x": np.array([0.0, 0.0, 0.0]), "y": np.array([0.5, 0.0, 0.0])},
}

#: method -> (integrator kwargs, samples per call of a vectorised callable).
_WF9_METHODS = {
    "qmc_vectorized": ({"n_samples": 2 ** 7, "seed": 20260429}, 2 ** 7),
    # n_gauss nodes in each of the FK diagrams' four time integrals.
    "gauss_legendre": ({"n_gauss": 4}, 4 ** 4),
    # evaluate_at runs a vectorised callable as a batch of one sample.
    "qmc_scalar": ({"n_samples": 2 ** 5, "seed": 20260429}, 1),
}


def _wf9_fk_total(
    *, vectorized: bool, positions: dict, method: str,
    seen: list | None = None,
):
    """Order-2 FK total with the recording κ^(3), and the ``(n, t)``
    shapes of every call the integration made.  Pass ``seen`` to follow
    the calls while the integration runs."""
    seen = [] if seen is None else seen
    system = _make_kappa3_system_with_K(
        _wf9_recording_kappa3(vectorized=vectorized, seen=seen),
        coupling_vectorized=vectorized,
    )
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2])
    props = system.propagators(
        t_max=2.0, n_grid_t=20,
        c_closed_form=_load_demo1_C_fn(),
        c_closed_form_only=True,
        c_closed_form_vectorized=True,
    )
    result = expansion.evaluate(
        props,
        positions=positions,
        t_final=2.0,
        component_pair=(0, 1),
        orders=[2],
        vertex_types={"FK"},
        method=method,
        # A worker process would record into its own copy of ``seen``.
        n_jobs=1,
        **_WF9_METHODS[method][0],
    )
    return result.total, seen


@pytest.mark.parametrize("method", list(_WF9_METHODS))
@pytest.mark.parametrize("pos_kind", list(_WF9_POSITIONS))
@pytest.mark.parametrize(
    "vectorized", [False, True], ids=["per_sample", "vectorized"],
)
def test_WF9_callable_coupling_argument_shapes(vectorized, pos_kind, method):
    """A callable κ^(3) receives the argument shapes that
    ``DynamicCouplingPromise.evaluate_at_batch`` documents.

    With m = 3 legs, the per-sample contract passes ``n`` of shape
    ``(m,)`` for scalar positions and ``(m, d)`` for d-dim positions;
    ``coupling_vectorized=True`` passes ``(m, n_samples)`` and
    ``(m, n_samples, d)``.  ``t`` is ``(m,)`` or ``(m, n_samples)`` in
    both cases.  ``method='qmc_scalar'`` runs a vectorised callable
    through ``DynamicCouplingPromise.evaluate_at`` one sample at a time,
    so there ``n_samples`` is 1.

    WF8's callable accepts ``(m,)`` and ``(m, d)`` alike and WF8 checks
    only for a finite nonzero total, so it passes whether or not vector
    positions reach the callable.  This callable records the shapes of
    every call.  d = 2 is included because with d = m = 3 an array with
    the leg and position axes swapped has the same shape.

    The envelope depends on the positions only through their norms, and
    all three position kinds put x at the origin and y at distance 0.5,
    so every contract and position kind must give the per-sample,
    scalar-position total of the same method.  Zeroing the vector
    positions inside the promise keeps every shape correct and changes
    the total, so this comparison catches a vector that arrives with the
    right shape and a wrong value.
    """
    positions = _WF9_POSITIONS[pos_kind]
    n_batch = _WF9_METHODS[method][1]
    lead = (3, n_batch) if vectorized else (3,)  # m = 3 legs
    expected = {(lead + np.shape(positions["x"]), lead)}

    total, seen = _wf9_fk_total(
        vectorized=vectorized, positions=positions, method=method,
    )
    assert set(seen) == expected, (
        f"method={method!r}, {pos_kind} positions: the callable received "
        f"(n, t) shapes {sorted(set(seen))}, expected {sorted(expected)}"
    )

    reference, _ = _wf9_fk_total(
        vectorized=False, positions=_WF9_POSITIONS["scalar"], method=method,
    )
    assert reference != 0.0, "test is vacuous if the reference total is zero"
    # abs=0.0: every total here is below 0.1, so rel * expected is under
    # 1e-13 and the default 1e-12 floor would loosen the check to more than
    # 1e-11 relative.
    assert total == pytest.approx(reference, rel=1e-12, abs=0.0)


# =====================================================================
# WF10 -- how often a per-sample callable coupling is called
# =====================================================================


#: (method, force ``evaluate_at_batch`` onto its per-sample fallback loop)
_WF10_CASES = [
    ("qmc_vectorized", False),
    ("gauss_legendre", False),
    ("qmc_vectorized", True),
    ("gauss_legendre", True),
    ("qmc_scalar", False),
]


@pytest.mark.parametrize(
    "method, fallback", _WF10_CASES,
    ids=[m + ("-fallback" if f else "") for m, f in _WF10_CASES],
)
def test_WF10_per_sample_callable_called_once_per_sample(
    method, fallback, monkeypatch,
):
    """A per-sample callable κ^(3) is called once per sample and leg
    order.

    On qmc_vectorized and gauss_legendre, ``evaluate_at_batch`` builds
    each leg order's tensor stack from one call per sample, and its
    sample-0 shape probe and per-sample fallback loop read that stack.
    The fallback runs when ``DiagramTerm.evaluate_coupling_batched``
    raises ``NotImplementedError``; the ``fallback`` cases force it.
    qmc_scalar calls the callable through ``evaluate_at``, also once
    per sample and leg order.

    Calls are counted per call of the promise method, with the leg
    orders read off the promise.  Both routes give the same count, so
    each case also checks which route ran.  On the fallback the FK
    total must equal the batched total to 1e-12, which a loop reading
    the wrong sample would fail.
    """
    reference = None
    batched_attempts: list = []
    if fallback:
        # The batched contraction's total, before it is disabled.
        reference, _ = _wf9_fk_total(
            vectorized=False, positions=_WF9_POSITIONS["scalar"],
            method=method,
        )

        def refuse_batched(self, *args, **kwargs):
            batched_attempts.append(1)
            raise NotImplementedError("WF10: forced per-sample fallback")

        monkeypatch.setattr(
            DiagramTerm, "evaluate_coupling_batched", refuse_batched,
        )

    seen: list = []
    # Per promise call: (method, leg orders, samples, callable calls).
    log: list = []
    batch_fn = DynamicCouplingPromise.evaluate_at_batch
    at_fn = DynamicCouplingPromise.evaluate_at

    def evaluate_at_batch(self, label_t, label_x, n_samples):
        first = len(seen)
        out = batch_fn(self, label_t, label_x, n_samples)
        log.append(("evaluate_at_batch", len(self.dynamic_values),
                    n_samples, len(seen) - first))
        return out

    def evaluate_at(self, times, positions):
        first = len(seen)
        out = at_fn(self, times, positions)
        log.append(("evaluate_at", len(self.dynamic_values), 1,
                    len(seen) - first))
        return out

    monkeypatch.setattr(
        DynamicCouplingPromise, "evaluate_at_batch", evaluate_at_batch,
    )
    monkeypatch.setattr(DynamicCouplingPromise, "evaluate_at", evaluate_at)

    total, _ = _wf9_fk_total(
        vectorized=False, positions=_WF9_POSITIONS["scalar"],
        method=method, seen=seen,
    )

    route = "evaluate_at" if method == "qmc_scalar" else "evaluate_at_batch"
    assert log, "the integration never reached DynamicCouplingPromise"
    assert {entry for entry, *_ in log} == {route}, (
        f"method={method!r} called {sorted({e for e, *_ in log})}, "
        f"expected {route}"
    )
    # Samples per promise call: all of them, or one for evaluate_at.
    n_batch = _WF9_METHODS[method][1]
    for entry, n_orders, n_samples, n_calls in log:
        assert n_samples == n_batch
        assert n_calls == n_orders * n_samples, (
            f"{entry} with {n_orders} leg orders and {n_samples} samples "
            f"called the callable {n_calls} times, expected "
            f"{n_orders * n_samples}"
        )
    assert len(seen) == sum(n_calls for *_, n_calls in log), (
        "the callable was called outside DynamicCouplingPromise"
    )

    if fallback:
        assert len(batched_attempts) == len(log)
        assert reference != 0.0, (
            "test is vacuous if the reference total is zero"
        )
        # abs=0.0 for the reason given in WF9.
        assert total == pytest.approx(reference, rel=1e-12, abs=0.0)


def _scalar_field_system(coupling):
    """A single-component (``n_components = 1``) system with a
    non-local kappa^3 vertex.

    N = 1 is the first thing a new user writes, and it takes a
    DIFFERENT symbolic path: with one component there is nothing to sum
    over, so the simplifier elides component indices entirely and every
    ``Symbol`` in the coupling comes out with ``indices = ()``.  The
    coupling array still has its rank-m shape, now all-ones.
    """
    F = np.zeros((1, 1, 1))
    F[0, 0, 0] = 0.7
    return sw.System(
        field=sw.FieldSpec("phi", n_components=1),
        linear=sw.DiagonalA(gamma=[1.0]),
        vertices=[sw.LocalVertex("F", coupling=F)],
        nonlocal_vertices=[sw.NonLocalVertex("K", 3, coupling=coupling)],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            ),
        ),
    )


@pytest.mark.parametrize("method", ["gauss_legendre", "qmc_vectorized"])
def test_DC3_scalar_field_dynamic_coupling_matches_static(method):
    """``n_components = 1`` with a CALLABLE coupling used to raise

        ValueError: input operand has more dimensions than allowed by
        the axis remapping

    from ``_sum_coupling_batched``.  With one component the simplifier
    drops every component index, so ``_eval_symbolic_batched`` took its
    ``not expr.indices`` branch and returned the coupling array whole --
    shape ``(n_samples, 1, 1, 1)`` -- where the caller expected
    ``(n_samples,)``.  The scalar evaluator has always tolerated a
    size-1 array of any shape here; the batched one did not.

    A constant callable is the static tensor, so the two must agree
    exactly.  N = 2 is covered by WF6; this is the N = 1 boundary.
    """
    K = np.full((1, 1, 1), 1e-3)

    sw.reset_uid_counter()
    static_sys = _scalar_field_system(K)
    sw.reset_uid_counter()
    dyn_sys = _scalar_field_system(lambda n_list, t_list: K)  # noqa: ARG005

    kw = dict(
        positions={"x": 0.0, "y": 0.5}, t_final=1.5, component_pair=(0, 0),
        orders=[2], vertex_types=["FK"], method=method, n_gauss=8,
        n_samples=2 ** 12, seed=0, n_jobs=1,
    )
    vals = []
    for system in (static_sys, dyn_sys):
        props = system.propagators(
            t_max=2.0, n_grid_t=8, c_closed_form=_load_demo1_C_fn(),
        )
        expansion = system.expand(("phi(x)", "phi(y)"), orders=[2])
        vals.append(expansion.evaluate(props, **kw).total)

    assert vals[0] != 0.0, "test is vacuous if the static value is zero"
    # abs=0.0: the compared value is 3.1e-04, so rel * expected is 3.1e-16
    # and the default 1e-12 floor would have enforced 3.3e-09 relative --
    # 3270x looser than the machine-precision agreement this test exists to
    # lock.  DC1 above already passes abs=1e-300 for the same reason.
    assert vals[1] == pytest.approx(vals[0], rel=1e-12, abs=0.0)
