"""End-to-end tests for d-dim spatial coordinates flowing through
the L1 ``Expansion.evaluate`` path.

Locked here:

* **DD1** -- ``Expansion.evaluate(positions={...})`` accepts
  arbitrary-dimensional vector positions. The integrand, the
  ``_resolve_group_x`` plumbing, and the ``C_at_batch`` translation
  branch must all reduce vector inputs to the right scalar
  ``r = ||x1 - x2||`` and produce the same total as a 1-D
  configuration with the same Euclidean separation.

The companion cache-layer tests (``S2b_translation_supports_3d_
vectors``, ``S6b_rotation_supports_3d_unit_vectors``,
``S8_general_full_grid_rejects_d_dim_input`` in
``test_deductive_numerics.py``) lock the propagator side; this file
focuses on the workflow-side surface where users actually touch the
contract.
"""
from __future__ import annotations

import importlib.util as _ilu
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import sft_wick as sw


_DEMO1_CLOSED_FORM_SRC = (
    Path(__file__).resolve().parents[1]
    / "examples" / "demo1" / "c_closed_form.py"
)


def _load_demo1_C_fn():
    spec = _ilu.spec_from_file_location(
        "demo1_c_closed_form_ddim", _DEMO1_CLOSED_FORM_SRC,
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.C_fn


def _build_demo1_system():
    F = np.zeros((2, 2, 2))
    F[0, 1, 1] = 1.0
    F[1, 0, 1] = 0.5
    F[1, 1, 0] = 0.5
    return sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            ),
        ),
    )


def test_DD1_expansion_evaluate_accepts_3d_positions() -> None:
    """``Expansion.evaluate`` with translation-invariant noise must
    give the same total for a 1-D pair (x=0, y=r) and a 3-D pair
    (x=(0,0,0), y=(r,0,0)) at the same Euclidean separation r.

    Locks the d-dim contract for the L1 layer: users can pass
    arbitrary-dimensional position vectors via the ``positions``
    kwarg and they reduce to the right scalar r downstream.
    """
    r = 0.7
    system = _build_demo1_system()
    expansion = system.expand(
        ("phi_a(x)", "phi_b(y)"), orders=[0, 2],
    )

    # Single propagator cache shared between both runs (the cache
    # itself is dimension-agnostic on the translation-mode path).
    props = system.propagators(
        t_max=2.0, n_grid_t=12, c_closed_form=_load_demo1_C_fn(),
    )

    # 1-D run: scalar positions, r encoded as the scalar separation.
    res_1d = expansion.evaluate(
        props,
        positions={"x": 0.0, "y": r},
        t_final=1.0,
        component_pair=(0, 0),
        orders=[0, 2],
        n_samples=2 ** 10,
        seed=20260428,
        integrate_over="all",
    )

    # 3-D run: vector positions whose Euclidean separation is r.
    res_3d = expansion.evaluate(
        props,
        positions={
            "x": np.array([0.0, 0.0, 0.0]),
            "y": np.array([r, 0.0, 0.0]),
        },
        t_final=1.0,
        component_pair=(0, 0),
        orders=[0, 2],
        n_samples=2 ** 10,
        seed=20260428,
        integrate_over="all",
    )

    np.testing.assert_allclose(
        res_3d.total, res_1d.total,
        rtol=1e-10, atol=1e-12,
        err_msg=(
            f"3-D positions total {res_3d.total!r} disagrees with "
            f"the equivalent 1-D scalar total {res_1d.total!r}; the "
            f"d-dim translation contract is broken."
        ),
    )


def test_DD2_expansion_evaluate_3d_separation_is_distance_only() -> None:
    """For translation-invariant noise the total must depend only
    on ||x - y||, not on the direction of the offset.

    Run the same expansion at two 3-D position configurations with
    different offset directions but the same Euclidean separation;
    the totals must agree to QMC noise (same seed).
    """
    r = 0.5
    system = _build_demo1_system()
    expansion = system.expand(
        ("phi_a(x)", "phi_b(y)"), orders=[0, 2],
    )
    props = system.propagators(
        t_max=2.0, n_grid_t=12, c_closed_form=_load_demo1_C_fn(),
    )

    common = dict(
        t_final=1.0,
        component_pair=(0, 0),
        orders=[0, 2],
        n_samples=2 ** 10,
        seed=20260428,
        integrate_over="all",
    )

    # Offset along +x axis.
    res_a = expansion.evaluate(
        props,
        positions={
            "x": np.array([0.0, 0.0, 0.0]),
            "y": np.array([r, 0.0, 0.0]),
        },
        **common,
    )
    # Same separation, offset along (1/sqrt(3), 1/sqrt(3), 1/sqrt(3)).
    diag = r / np.sqrt(3.0)
    res_b = expansion.evaluate(
        props,
        positions={
            "x": np.array([0.0, 0.0, 0.0]),
            "y": np.array([diag, diag, diag]),
        },
        **common,
    )

    np.testing.assert_allclose(
        res_b.total, res_a.total,
        rtol=1e-10, atol=1e-12,
        err_msg=(
            f"3-D rotated separation broke translation isotropy: "
            f"axis-aligned total {res_a.total!r} vs diagonal "
            f"total {res_b.total!r}."
        ),
    )


def test_DD3_sweep_with_3d_positions_aggregates_via_pandas_groupby() -> None:
    """Regression: ``Expansion.sweep(positions_grid={...})`` with
    d-dim vector positions must produce a SweepResult whose
    ``totals()`` (pandas groupby) does not crash with
    ``TypeError: unhashable type: 'list'``.

    Originally surfaced from a user config with
    ``positions_grid: {x: [[0.0, 0.0, 1.0]], y: [[0.16, 0, 0.99], ...]}``;
    pandas factorises group keys via a hash table that rejects
    list-typed cells, so the rows constructed in
    ``Expansion.sweep`` must coerce d-dim positions to tuples.
    """
    system = _build_demo1_system()
    expansion = system.expand(
        observable=("phi_a(x)", "phi_b(y)"), orders=(0,),
    )
    props = system.propagators(
        t_max=2.0, n_grid_t=10,
        c_closed_form=_load_demo1_C_fn(),
        c_closed_form_only=True,
        c_closed_form_vectorized=True,
    )
    sweep = expansion.sweep(
        props,
        positions_grid={
            "x": [np.array([0.0, 0.0, 0.0])],
            "y": [
                np.array([0.5, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
            ],
        },
        t_final_grid=[1.0],
        component_pairs=[(0, 0)],
        orders=(0,),
        method="gauss_legendre",
        n_gauss=4,
    )
    # The crash was in ``totals()``; make sure it succeeds and
    # returns one row per (positions, t_final, a, b, order) cell.
    df = sweep.totals()
    assert len(df) == 2, f"expected 2 rows, got {len(df)}: {df!r}"


def _batch_length_system():
    return sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[0.9, 1.3]),
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.5, sigma_t=0.4),
                spatial=sw.GaussianSpatial(sigma_x=1.0),
            ),
        ),
        t_min=0.2,
    )


@pytest.mark.parametrize("closed_form_only", [False, True])
@pytest.mark.parametrize(
    ("dim", "method", "knobs"),
    [
        (2, "gauss_legendre", {"n_gauss": 2}),
        (3, "gauss_legendre", {"n_gauss": 3}),
        (2, "qmc_vectorized", {"n_samples": 2}),
        (4, "qmc_vectorized", {"n_samples": 4}),
    ],
)
def test_DD4_fixed_vector_position_on_a_batch_of_equal_length(
    dim, method, knobs, closed_form_only
) -> None:
    """A batch of ``dim`` samples must not read a fixed ``dim``-vector
    position as one scalar position per sample.

    Integrating one external time gives a one-dimensional integral, so
    ``n_gauss`` GL nodes or ``n_samples`` QMC points form the whole
    batch.  ``C_at_batch`` reads a 1-D array of the batch's length as
    per-sample scalars; before the fix the eager GL and vectorised QMC
    paths handed it the fixed vector unchanged and returned a value
    10-35 % off.  The reference is the same separation as scalars.
    """
    system = _batch_length_system()
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0])
    kw = dict(t_max=3.0, n_grid_t=60, progress=False)
    if closed_form_only:
        kw.update(c_closed_form_only=True)
    props = system.propagators(**kw)
    x = np.linspace(-0.2, 0.4, dim)
    y = x + np.linspace(0.3, -0.8, dim)
    common = dict(
        t_final=2.0,
        component_pair=(1, 1),
        orders=[0],
        method=method,
        integrate_over={"x"},
        **knobs,
    )
    vector = expansion.evaluate(props, positions={"x": x, "y": y}, **common)
    scalar = expansion.evaluate(
        props,
        positions={"x": 0.37, "y": 0.37 + float(np.linalg.norm(y - x))},
        **common,
    )
    assert vector.total == pytest.approx(scalar.total, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("dim", [2, 4])
def test_DD5_two_point_qmc_fixed_vector_position_on_a_batch_of_equal_length(dim) -> None:
    """``integrate_two_point_qmc`` with ``n_samples == dim`` must read a
    fixed ``dim``-vector position as one point.  Before the fix it was
    8 % (dim 2) and 44 % (dim 4) off at order 1."""
    from sft_wick.evaluate import integrate_two_point_qmc

    F = np.array([[0.0, 0.4], [0.25, 0.1]])
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[0.9, 1.3]),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.5, sigma_t=0.4),
                spatial=sw.GaussianSpatial(sigma_x=1.0),
            ),
        ),
        t_min=0.2,
    )
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[1], progress=False)
    props = system.propagators(t_max=3.0, n_grid_t=60, progress=False)
    coupling = system.build_coupling_values()
    integrands = [
        dt.build_integrand(coupling, {"a": 0, "b": 1}) for dt in expansion.diagrams(1)
    ]
    x = np.linspace(-0.2, 0.4, dim)
    y = x + np.linspace(0.3, -0.8, dim)
    common = dict(t_f=2.0, cache=props.cache, t_min=0.2, n_samples=dim, seed=1)
    vector, _ = integrate_two_point_qmc(integrands, positions={"x": x, "y": y}, **common)
    scalar, _ = integrate_two_point_qmc(
        integrands,
        positions={"x": 0.37, "y": 0.37 + float(np.linalg.norm(y - x))},
        **common,
    )
    assert scalar != 0.0
    assert vector == pytest.approx(scalar, rel=1e-12, abs=0.0)


_DD6_T_MIN = 0.3
_DD6_G = np.array([[0.8, -0.35], [-0.35, 0.6]])


@dataclass(frozen=True)
class _InPlaceSourceC:
    """``C_ab = G_ab (t1 - t_min)(t2 - t_min) exp(-|x1 - x2|^2)``, written
    the way a memory-conscious hook might: ``x1 - x2`` is formed in place
    in the two position arguments.  A vector position has one more axis
    than a scalar one: ``(n, d)`` against ``(n,)`` per batch, ``(d,)``
    against a scalar per sample.  With ``in_place=False`` the same value
    is formed without writing, bit for bit: ``a + (-b)`` is ``a - b``."""

    vectorized: bool
    in_place: bool = True
    has_diagonal_kink = False

    def __call__(self, x1, t1, x2, t2):
        if self.in_place:
            diff = np.asarray(x1)
            minus = np.asarray(x2)
            np.negative(minus, out=minus)
            diff += minus
        else:
            diff = np.asarray(x1) - np.asarray(x2)
        is_vector = diff.ndim == (2 if self.vectorized else 1)
        dist2 = np.sum(diff * diff, axis=-1) if is_vector else diff * diff
        temporal = (np.asarray(t1) - _DD6_T_MIN) * (np.asarray(t2) - _DD6_T_MIN)
        return (temporal * np.exp(-dist2))[..., None, None] * _DD6_G


def _in_place_C_expansion(vectorized, in_place=True):
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[0.9, 1.3]),
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.5, sigma_t=0.4),
                spatial=sw.GaussianSpatial(sigma_x=1.0),
            ),
        ),
        t_min=_DD6_T_MIN,
    )
    expansion = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0], diag_C=False)
    props = system.propagators(
        t_max=2.0,
        c_closed_form=_InPlaceSourceC(vectorized, in_place),
        c_closed_form_only=True,
        c_closed_form_vectorized=vectorized,
        diag_C=False,
        progress=False,
    )
    return expansion, props


@pytest.mark.parametrize("vectorized", [True, False])
@pytest.mark.parametrize(
    ("dim", "n_gauss", "chunk"),
    [
        (3, 2, None),
        (3, 3, None),
        (3, 4, None),
        (2, 2, None),
        (4, 4, None),
        (3, 7, 4),
        (3, 6, 3),
    ],
    ids=["d3-n2", "d3-n3", "d3-n4", "d2-n2", "d4-n4", "d3-tail3", "d3-blocks3"],
)
def test_DD6_C_hook_may_write_into_fixed_vector_positions(
    dim, n_gauss, chunk, vectorized
) -> None:
    """A closed-form C that writes into its position arguments must work
    at every batch length.  A fixed ``dim``-vector on a batch of ``dim``
    samples used to reach the hook as a read-only broadcast view, so the
    in-place subtraction raised ``ValueError`` at that length only.  One
    integrated external time makes the GL batch ``n_gauss`` long (blocks
    of ``chunk`` nodes when one is given).  The reference is the integral
    over the linear time factor, ``G_01 exp(-|x - y|^2) L^3 / 2``, which
    GL integrates exactly; the caller's position arrays must come back
    unchanged."""
    expansion, props = _in_place_C_expansion(vectorized)
    x = np.linspace(-0.2, 0.4, dim)
    y = x + np.linspace(0.3, -0.8, dim)
    x_before, y_before = x.copy(), y.copy()
    L = 1.1
    got = expansion.evaluate(
        props,
        positions={"x": x, "y": y},
        t_final=_DD6_T_MIN + L,
        component_pair=(0, 1),
        orders=[0],
        integrate_over={"x"},
        method="gauss_legendre",
        n_gauss=n_gauss,
        gl_chunk_size=chunk,
    )
    separation2 = float(np.dot(x_before - y_before, x_before - y_before))
    ref = _DD6_G[0, 1] * np.exp(-separation2) * L**3 / 2
    assert got.total == pytest.approx(ref, rel=2e-14, abs=0.0)
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(y, y_before)


@pytest.mark.parametrize("dim", [2, 4])
def test_DD6b_qmc_C_hook_may_write_into_fixed_vector_positions(dim) -> None:
    """The vectorised QMC path with ``n_samples == dim`` hands the hook
    the same fixed vector, which must be writable there too.  The
    reference is the same separation as scalars, as in DD4."""
    expansion, props = _in_place_C_expansion(True)
    x = np.linspace(-0.2, 0.4, dim)
    y = x + np.linspace(0.3, -0.8, dim)
    common = dict(
        t_final=_DD6_T_MIN + 1.1,
        component_pair=(0, 1),
        orders=[0],
        integrate_over={"x"},
        method="qmc_vectorized",
        n_samples=dim,
        seed=3,
    )
    vector = expansion.evaluate(props, positions={"x": x, "y": y}, **common)
    scalar = expansion.evaluate(
        props,
        positions={"x": 0.37, "y": 0.37 + float(np.linalg.norm(y - x))},
        **common,
    )
    assert scalar.total != 0.0
    assert vector.total == pytest.approx(scalar.total, rel=1e-12, abs=0.0)


def _dd6_reference(x, y, L, n_integrated):
    """``G_01 exp(-|x - y|^2)`` times the time factor ``(t1 - t_min)(t2 -
    t_min)``: ``L^2`` at ``t1 = t2 = t_min + L``, its integral ``L^3 / 2``
    over one external time."""
    d = np.atleast_1d(np.asarray(x, dtype=float) - np.asarray(y, dtype=float))
    time_factor = L**2 if n_integrated == 0 else L**3 / 2
    return _DD6_G[0, 1] * np.exp(-float(np.dot(d, d))) * time_factor


@pytest.mark.parametrize("method", ["qmc", "qmc_scalar", "nquad"])
def test_DD6c_per_sample_integrators_hand_the_C_hook_copies(method) -> None:
    """The per-sample integrators reach the hook through
    ``PropagatorCache.C_value``, which used to pass the caller's position
    array itself.  The in-place hook then rewrote it, so every later
    sample saw other positions: -0.14713 on ``qmc`` and ``qmc_scalar``
    and -0.14680 on ``nquad`` against -0.10545, with no error, and
    ``nquad`` returned the caller's ``x`` changed.  Both evaluations with
    the same positions must equal the reading hook's value bit for bit
    and the exact ``G_01 exp(-|x - y|^2) L^3 / 2``."""
    x = np.linspace(-0.2, 0.4, 3)
    y = x + np.linspace(0.3, -0.8, 3)
    x_before, y_before = x.copy(), y.copy()
    L = 1.1
    common = dict(
        t_final=_DD6_T_MIN + L,
        component_pair=(0, 1),
        orders=[0],
        integrate_over={"x"},
        method=method,
    )
    if method != "nquad":
        common.update(n_samples=64, seed=3)
    expansion, props = _in_place_C_expansion(False)
    positions = {"x": x, "y": y}
    totals = [
        expansion.evaluate(props, positions=positions, **common).total
        for _ in range(2)
    ]
    _, reading_props = _in_place_C_expansion(False, in_place=False)
    reading = expansion.evaluate(
        reading_props, positions={"x": x_before, "y": y_before}, **common
    ).total
    assert totals == [reading, reading]
    ref = _dd6_reference(x_before, y_before, L, n_integrated=1)
    assert reading == pytest.approx(ref, rel=1e-9, abs=0.0)
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(y, y_before)


# One QMC sample has no sample variance; NumPy warns while computing it.
@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize("vectorized", [True, False])
@pytest.mark.parametrize(
    "case", ["zero-dimensional", "gl-n_gauss-1", "qmc-n_samples-1"]
)
def test_DD6d_C_hook_may_write_into_a_one_vector_on_a_batch_of_one(
    case, vectorized
) -> None:
    """At batch length ``n = d = 1`` the vectorised hook used to receive
    the caller's own ``(1,)`` array, or an internal one kept between
    evaluations, so a second evaluation with the same positions changed:
    -0.29547 then -0.41929 with no integrated time.  Batch length one is
    every zero-dimensional evaluation, GL with ``n_gauss = 1`` and QMC
    with ``n_samples = 1``.  Both evaluations must equal the reading
    hook's value bit for bit and, where GL or no integral is involved,
    the exact value (one GL node integrates the linear time factor
    exactly)."""
    x = np.array([0.1])
    y = np.array([0.7])
    L = 1.1
    kw = {
        "zero-dimensional": {},
        "gl-n_gauss-1": dict(
            integrate_over={"x"}, method="gauss_legendre", n_gauss=1
        ),
        "qmc-n_samples-1": dict(
            integrate_over={"x"}, method="qmc_vectorized", n_samples=1, seed=3
        ),
    }[case]
    common = dict(t_final=_DD6_T_MIN + L, component_pair=(0, 1), orders=[0], **kw)
    expansion, props = _in_place_C_expansion(vectorized)
    positions = {"x": x, "y": y}
    totals = [
        expansion.evaluate(props, positions=positions, **common).total
        for _ in range(2)
    ]
    _, reading_props = _in_place_C_expansion(vectorized, in_place=False)
    reading = expansion.evaluate(
        reading_props, positions={"x": x.copy(), "y": y.copy()}, **common
    ).total
    assert totals == [reading, reading]
    if case != "qmc-n_samples-1":
        n_integrated = 0 if case == "zero-dimensional" else 1
        ref = _dd6_reference(x, y, L, n_integrated)
        assert reading == pytest.approx(ref, rel=2e-15, abs=0.0)
    np.testing.assert_array_equal(x, [0.1])
    np.testing.assert_array_equal(y, [0.7])


@dataclass(frozen=True)
class _TimeShiftC:
    """Vectorised ``C_ab = G_ab s1 s2 exp(-(x1 - x2)^2)`` with ``s = t -
    t_min``, formed in place in the time arguments when ``in_place``
    (``t1 -= t_min``), by subtraction otherwise; the two agree bit for
    bit."""

    in_place: bool
    has_diagonal_kink = False

    def __call__(self, x1, t1, x2, t2):
        if self.in_place:
            s1 = np.asarray(t1)
            s1 -= _DD6_T_MIN
            s2 = np.asarray(t2)
            s2 -= _DD6_T_MIN
        else:
            s1 = np.asarray(t1) - _DD6_T_MIN
            s2 = np.asarray(t2) - _DD6_T_MIN
        d = np.asarray(x1) - np.asarray(x2)
        return (s1 * s2 * np.exp(-d * d))[..., None, None] * _DD6_G


@pytest.mark.parametrize(
    "kw",
    [
        dict(method="gauss_legendre", n_gauss=6, gl_chunk_size=None),
        dict(method="gauss_legendre", n_gauss=6, gl_chunk_size=50),
        dict(method="qmc_vectorized", n_samples=256, seed=1),
    ],
    ids=["gl-eager", "gl-blocks50", "qmc"],
)
def test_DD7_vectorised_C_hook_may_write_into_its_time_arguments(kw) -> None:
    """The vectorised hook used to receive the integrator's time batches,
    which order 2 reads again for the other C and R factors, so ``t1 -=
    t_min`` inside the hook changed the integral with no error: 0.06517
    against 0.14215 on GL at ``n_gauss = 6``, eager or in blocks, and
    0.06537 against 0.14244 on QMC.  The writing and the reading hook must
    agree bit for bit, twice in a row.  Distinct rates, couplings of both
    signs, distinct points and ``t_min != 0``."""
    F = np.zeros((2, 2, 2))
    F[0, 1, 1] = 0.7
    F[1, 0, 1] = -0.4
    F[1, 1, 0] = -0.4
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[0.9, 1.3]),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=0.5, sigma_t=0.4),
                spatial=sw.GaussianSpatial(sigma_x=1.0),
            ),
        ),
        t_min=_DD6_T_MIN,
    )
    expansion = system.expand(
        ("phi_a(x)", "phi_b(y)"), orders=[2], diag_C=False, progress=False
    )

    def total(in_place):
        props = system.propagators(
            t_max=2.0,
            c_closed_form=_TimeShiftC(in_place),
            c_closed_form_only=True,
            c_closed_form_vectorized=True,
            diag_C=False,
            progress=False,
        )
        return expansion.evaluate(
            props,
            positions={"x": 0.1, "y": -0.5},
            t_final=1.6,
            component_pair=(0, 1),
            orders=[2],
            **kw,
        ).total

    reading = total(False)
    assert reading != 0.0
    assert [total(True), total(True)] == [reading, reading]
