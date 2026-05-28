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
    assert v_00 == pytest.approx(11.0, rel=1e-12), (
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

    assert v_01 == pytest.approx(12.0, rel=1e-12), (
        f"diag_C=False: (0, 1) must read C[0, 1] = 12.0; got {v_01:.6e}"
    )
    assert v_12 == pytest.approx(23.0, rel=1e-12), (
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
        assert got == pytest.approx(want, rel=1e-12), (
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
    assert v_01 == pytest.approx(12.0, rel=1e-12), (
        f"iso_R=True + diag_C=False: (0, 1) must read C[0, 1] = 12.0; "
        f"got {v_01:.6e}"
    )


# =====================================================================
# T5 (guard) -- diag_C=False without c_closed_form_only must error.
# =====================================================================


def test_diag_C_false_without_closed_form_only_raises(tmp_path: Path) -> None:
    """Spline-table paths only fill diagonal entries; pairing
    ``diag_C: false`` with a non-closed-form-only build silently
    drops off-diagonals at lookup time. Reject early at the
    propagator-build site with a clear message."""
    (tmp_path / "stub_c.py").write_text(_STUB_C_MODULE)
    data = yaml.safe_load(_BASE_YAML)
    # Strip the c_closed_form_only opt-in; the user still provides
    # c_closed_form_module so a spline table would be built.
    data["propagators"]["c_closed_form_only"] = False
    data["propagators"]["diag_C"] = False
    data["sweep"]["component_pairs"] = [[0, 1]]
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(data))

    cfg = load_workflow_config(tmp_path / "c.yaml")
    with pytest.raises(ValueError, match="diag_C=False requires c_closed_form_only"):
        run_workflow(cfg)
