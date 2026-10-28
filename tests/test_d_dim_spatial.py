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
    against a scalar per sample."""

    vectorized: bool
    has_diagonal_kink = False

    def __call__(self, x1, t1, x2, t2):
        diff = np.asarray(x1)
        minus = np.asarray(x2)
        np.negative(minus, out=minus)
        diff += minus
        is_vector = diff.ndim == (2 if self.vectorized else 1)
        dist2 = np.sum(diff * diff, axis=-1) if is_vector else diff * diff
        temporal = (np.asarray(t1) - _DD6_T_MIN) * (np.asarray(t2) - _DD6_T_MIN)
        return (temporal * np.exp(-dist2))[..., None, None] * _DD6_G


def _in_place_C_expansion(vectorized):
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
        c_closed_form=_InPlaceSourceC(vectorized),
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
