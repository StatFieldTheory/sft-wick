"""Tests for the ``propagators.diag_C: false`` L2 knob.

Downstream consumers (e.g. canoes' weak-lensing pipeline) need the
full ``N x N`` closed-form C matrix to read off cross-component
entries -- the lensing observable is a 3-component field
``phi = (kappa, gamma_+, gamma_x)`` and the user requests pairs like
``(0, 1) = kappa-gamma_+``. Before this patch the L2 evaluator
silently mapped every off-diagonal pair to its on-diagonal projection
because ``System.build_propagator_model`` hardcoded ``diag_C=True``.

Acceptance criteria (from
``PROMPT_FOR_SFT_WICK_nondiagonal_C_support.md``):

* ``diag_C: true`` (default) keeps the existing diagonal-only path
  exactly -- regression-safe.
* ``diag_C: false`` returns ``C[a, b]`` for every requested
  ``(a, b)``, including ``a != b``.
* ``iso_R: true`` must remain independent of ``diag_C: false`` -- the
  R side stays scalar while the C side is matrix-valued.
* The 4 unit tests below cover all four corners of the matrix.

The tests use a hand-stubbed 3x3 closed-form C with distinct values
in every (a, b) slot. We exercise the full L2 pipeline (YAML load +
``run_workflow``) at order 0 only because that is where the diagonal
projection bug bites hardest: with no internal integration the order-0
C matrix entry IS the answer, so a wrong projection is a 100 percent
relative error.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml

import sft_wick as sw
from sft_wick.evaluate import PropagatorCache
from sft_wick.workflow.config import (
    load_workflow_config,
    run_workflow,
)


# Stub closed-form C: a fixed 3x3 matrix with every (a, b) slot
# distinct so misrouting is detectable. The matrix is symmetric and
# the diagonal entries are distinct from each other and from the
# off-diagonal entries. The same matrix is returned for every
# (n1, t1, n2, t2) so the order-0 result is exactly ``C_STUB[a, b]``
# up to a known prefactor from the propagator evaluation.
_STUB_C_MODULE = textwrap.dedent("""
    import numpy as np

    # Distinct values per (a, b) so misrouting is detectable. Symmetric
    # so the test cross-checks ``(a, b)`` against ``(b, a)`` when
    # useful.
    _C_STUB = np.array([
        [11.0, 12.0, 13.0],
        [12.0, 22.0, 23.0],
        [13.0, 23.0, 33.0],
    ])

    def C_fn(n1, t1, n2, t2):
        return _C_STUB.copy()
""")


# 3-component field with a trivial F vertex (zeros, so order-2 is
# zero) and a benign noise spec; the test focuses on order 0 where
# the C matrix entry directly drives the result.
_BASE_YAML = textwrap.dedent("""
    system:
      field: {name: phi, n_components: 3}
      linear: {type: diagonal, gamma: [1.0, 1.0, 1.0]}
      vertices: []
      nonlocal_vertices: []
      noise:
        kappa2:
          type: separable_translation
          temporal: {type: exponential, lam: 1.0, sigma_t: 1.0}
          spatial:  {type: exponential, sigma_x: 1.0}
        sigma2: null

    expand:
      observable: ["phi_a(x)", "phi_b(y)"]
      orders: [0]

    propagators:
      t_max: 1.0
      n_grid_t: 4
      c_closed_form_module: ./stub_c.py
      c_closed_form_attr: C_fn
      c_closed_form_only: true
      c_closed_form_vectorized: false

    sweep:
      positions_grid:
        x: [0.0]
        y: [0.0]
      t_final_grid: [1.0]
      component_pairs: PLACEHOLDER
      orders: [0]
      method: qmc_vectorized
      n_samples: 16
      seed: 0
""")


def _write_config(tmp: Path, pairs, *, diag_C: bool | None = None) -> Path:
    """Materialise a YAML config with the given ``component_pairs``
    and optional ``propagators.diag_C`` override."""
    (tmp / "stub_c.py").write_text(_STUB_C_MODULE)
    data = yaml.safe_load(_BASE_YAML)
    data["sweep"]["component_pairs"] = [list(p) for p in pairs]
    if diag_C is not None:
        data["propagators"]["diag_C"] = diag_C
    path = tmp / "c.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _pair_value(totals, pair: tuple[int, int]) -> float:
    """Pull the order-0 value for ``(a, b)`` out of the
    ``sweep.totals()`` DataFrame."""
    df = totals[(totals["a"] == pair[0]) & (totals["b"] == pair[1])]
    return float(df["value"].sum())


# =====================================================================
# T1 -- diag_C=True (default) preserves current diagonal-projection.
# =====================================================================


def test_diag_C_true_preserves_current_behaviour(tmp_path: Path) -> None:
    """With ``diag_C: true`` (the default), the on-diagonal pair must
    return ``C[a, a]`` exactly while off-diagonal pairs must collapse
    to zero -- the long-standing behaviour we are explicitly
    preserving.

    The collapse to zero (not to ``C[0, 0]``) is the symbolic
    ``apply_diagonal`` short-circuit at work: with ``diag_C=True``
    every observable C propagator is enforced as ``delta_{a, b} C``;
    the ``KroneckerDelta(a, b)`` factor evaluates to 0 when the
    observable pins ``(a, b) = (0, 1)``. Downstream workarounds (the
    canoes_pipeline ``SFT_WICK_OFFDIAG_TARGET`` env-var trick) read
    the on-diagonal pair value and post-hoc relabel it; they never
    request the genuine off-diagonal pair in the diag_C=True mode."""
    path = _write_config(tmp_path, [(0, 0), (0, 1)], diag_C=True)
    cfg = load_workflow_config(path)
    _sweep, totals = run_workflow(cfg)

    v_00 = _pair_value(totals, (0, 0))
    v_01 = _pair_value(totals, (0, 1))

    # On-diagonal pair: R(t, t) is 1 under iso_R + gamma=1, the stub
    # ``C[0, 0] = 11.0`` is the entire integrand at order 0.
    assert v_00 == pytest.approx(11.0, rel=1e-12, abs=0.0), (
        f"diag_C=True: (0, 0) must read C[0, 0] = 11.0; got {v_00:.6e}"
    )
    # Off-diagonal pair: the symbolic ``KroneckerDelta(a, b)`` from
    # ``apply_diagonal`` zeros the contribution.
    assert v_01 == pytest.approx(0.0, abs=1e-12), (
        f"diag_C=True: (0, 1) must zero out via KroneckerDelta; "
        f"got {v_01:.6e}"
    )


# =====================================================================
# T2 -- diag_C=False returns the off-diagonal C[a, b] entries.
# =====================================================================


def test_diag_C_false_returns_off_diagonal(tmp_path: Path) -> None:
    """With ``diag_C: false`` the L2 evaluator must read the full
    ``(N, N)`` matrix per sample and extract ``C[a, b]`` for every
    requested pair -- including ``a != b``."""
    path = _write_config(tmp_path, [(0, 1), (1, 2)], diag_C=False)
    cfg = load_workflow_config(path)
    _sweep, totals = run_workflow(cfg)

    v_01 = _pair_value(totals, (0, 1))
    v_12 = _pair_value(totals, (1, 2))

    assert v_01 == pytest.approx(12.0, rel=1e-12, abs=0.0), (
        f"diag_C=False: (0, 1) must read C[0, 1] = 12.0; got {v_01:.6e}"
    )
    assert v_12 == pytest.approx(23.0, rel=1e-12, abs=0.0), (
        f"diag_C=False: (1, 2) must read C[1, 2] = 23.0; got {v_12:.6e}"
    )


# =====================================================================
# T3 -- mixed on-/off-diagonal pairs in a single sweep.
# =====================================================================


def test_mixed_diag_offdiag_pairs_in_one_sweep(tmp_path: Path) -> None:
    """Single sweep, four pairs covering both on- and off-diagonal
    requests: ``(0, 0), (0, 1), (1, 1), (1, 2)``. Each pair must
    extract the corresponding C matrix entry independently."""
    pairs = [(0, 0), (0, 1), (1, 1), (1, 2)]
    path = _write_config(tmp_path, pairs, diag_C=False)
    cfg = load_workflow_config(path)
    _sweep, totals = run_workflow(cfg)

    expected = {
        (0, 0): 11.0,
        (0, 1): 12.0,
        (1, 1): 22.0,
        (1, 2): 23.0,
    }
    for pair, want in expected.items():
        got = _pair_value(totals, pair)
        assert got == pytest.approx(want, rel=1e-12, abs=0.0), (
            f"pair {pair}: want C[{pair[0]}, {pair[1]}] = {want}; "
            f"got {got:.6e}"
        )


# =====================================================================
# T4 -- iso_R: true is independent of diag_C: false.
# =====================================================================


def test_iso_R_independent_of_diag_C(tmp_path: Path) -> None:
    """The R-side ``iso_R`` flag (scalar R) must remain compatible
    with the C-side ``diag_C: false`` (matrix C). Acceptance criterion
    (3) in the prompt: ``iso_R: true + diag_C: false`` must be a
    legal combination and must produce the full ``(N, N)`` C through
    the pipeline.

    The base YAML uses ``linear: {type: diagonal, gamma: [...]}``
    which lowers to a scalar-R isotropic model (``System.iso_R``
    returns True). We just verify the run does not error out, the
    cache reports ``iso_R=True``, and the off-diagonal pair extracts
    the right C entry."""
    path = _write_config(tmp_path, [(0, 1)], diag_C=False)
    cfg = load_workflow_config(path)

    # iso_R is derived from system.linear; the base YAML uses
    # ``diagonal``, which is scalar-R isotropic.
    from sft_wick.workflow.config import build_system

    system = build_system(cfg.system)
    assert system.iso_R is True

    _sweep, totals = run_workflow(cfg)
    v_01 = _pair_value(totals, (0, 1))

    # The C matrix entry must come through unchanged: iso_R does not
    # touch the C side when diag_C is False.
    assert v_01 == pytest.approx(12.0, rel=1e-12, abs=0.0), (
        f"iso_R=True + diag_C=False: (0, 1) must read C[0, 1] = 12.0; "
        f"got {v_01:.6e}"
    )


# =====================================================================
# T5 -- diag_C=False without c_closed_form_only tabulates every C_ab.
# =====================================================================


def test_diag_C_false_tables_hold_every_entry(tmp_path: Path) -> None:
    """``diag_C: false`` with the spline tables used to be refused, because
    the tables held ``C_aa`` only.  They now hold every ``C_ab``: a table
    built from the stub closed form returns its off-diagonal entries, in
    both orders of the pair."""
    (tmp_path / "stub_c.py").write_text(_STUB_C_MODULE)
    data = yaml.safe_load(_BASE_YAML)
    # Strip the c_closed_form_only opt-in; the user still provides
    # c_closed_form_module, so a spline table is built from it.
    data["propagators"]["c_closed_form_only"] = False
    data["propagators"]["diag_C"] = False
    data["sweep"]["component_pairs"] = [[0, 1], [1, 2], [2, 0]]
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(data))

    cfg = load_workflow_config(tmp_path / "c.yaml")
    _sweep, totals = run_workflow(cfg)
    for pair, want in {(0, 1): 12.0, (1, 2): 23.0, (2, 0): 13.0}.items():
        got = _pair_value(totals, pair)
        assert got == pytest.approx(want, rel=1e-12, abs=0.0), (
            f"pair {pair}: want C[{pair[0]}, {pair[1]}] = {want}; "
            f"got {got:.6e}"
        )


N_OFF = 2


def _c_with_off_diagonal(n1, t1, n2, t2):
    """``C_01 = 0.3 C_00``, as a user closed form may write it."""
    size = np.exp(-abs(t1 - t2)) * (1.0 - np.exp(-2.0 * min(t1, t2)))
    return size * np.array([[1.0, 0.3], [0.3, 0.8]])


def _two_component_system():
    return sw.System(
        field=sw.FieldSpec("phi", N_OFF),
        linear=sw.DiagonalA(gamma=[0.8, 1.7]),
        noise=sw.GaussianNoise(
            kappa2=sw.SeparableTranslation(
                temporal=sw.ExponentialTemporal(lam=1.0, sigma_t=0.5),
                spatial=sw.ExponentialSpatial(sigma_x=1.0),
            )
        ),
    )


def test_a_user_c_is_read_as_one_c_under_diag_c():
    """The scalar read kept ``C_01`` where the batched read and the tables
    had dropped it: a factor 4.9 between integrators at L1 up to this fix."""
    model = _two_component_system().build_propagator_model(diag_C=True)
    cache = PropagatorCache(model, c_value_fn=_c_with_off_diagonal)
    full = _c_with_off_diagonal(0.0, 1.5, 0.3, 0.9)
    np.testing.assert_array_equal(
        cache.C_value(0.0, 1.5, 0.3, 0.9), np.diag(np.diag(full))
    )


@pytest.mark.parametrize("method", ["qmc_scalar", "qmc_vectorized"])
def test_every_integrator_reads_the_diagonal_of_a_closed_form_at_l1(method):
    """Order 0 at ``(0, 1)`` under an expansion that keeps the indices.  The
    scalar loop returned ``C_01`` here and the batched path 0 at order 2."""
    system = _two_component_system()
    props = system.propagators(
        t_max=2.0,
        c_closed_form=_c_with_off_diagonal,
        c_closed_form_only=True,
        progress=False,
    )
    total = (
        system.expand(("phi_a(x)", "phi_b(y)"), orders=[0], diag_C=False)
        .evaluate(
            props,
            positions={"x": 0.2, "y": 0.9},
            t_final=1.5,
            component_pair=(0, 1),
            orders=[0],
            method=method,
            n_samples=16,
        )
        .total
    )
    assert total == 0.0


def test_a_direct_build_refuses_what_system_propagators_refuses():
    """``Propagators.build`` built a table with ``C_01 = 0`` for a mixing
    white noise: order 0 of ``<phi_0 phi_1>`` read 0 against 0.0738.  The
    t_max check reads the same place, so one call covers both refusals."""
    kappa2 = sw.SeparableTranslation(
        temporal=sw.ExponentialTemporal(lam=1.0, sigma_t=0.5),
        spatial=sw.ExponentialSpatial(sigma_x=1.0),
    )
    mixing = sw.System(
        field=sw.FieldSpec("phi", 2),
        linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        noise=sw.GaussianNoise(
            kappa2=kappa2,
            sigma2=sw.ConstantImpulse(np.array([[1.0, 0.5], [0.5, 1.0]])),
        ),
    )
    with pytest.raises(ValueError, match="off-diagonal"):
        sw.Propagators.build(mixing, t_max=3.0, n_grid_t=6)
    built = sw.Propagators.build(mixing, t_max=3.0, n_grid_t=6, diag_C=False)
    assert built.cache.C_value(0.0, 1.0, 0.0, 1.0)[0, 1] != 0.0
    rising = sw.System(
        field=sw.FieldSpec("phi", 1),
        linear=sw.DiagonalA(gamma=lambda t: np.array([1.0 + t]), t_max_cache=2.0),
        noise=sw.GaussianNoise(kappa2=kappa2),
    )
    with pytest.raises(ValueError, match="t_max_cache"):
        sw.Propagators.build(rising, t_max=3.0, n_grid_t=6)
