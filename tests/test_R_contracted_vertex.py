"""Tests for ``NonLocalVertex(already_R_contracted=True)`` foundation.

Covers the Phase-1 deliverables:

  1. ``NonLocalVertex.already_R_contracted`` accepts ``False`` (default)
     and ``True`` as a boolean flag.
  2. ``already_R_contracted=True`` together with ``equal_time=True`` is
     rejected at construction (vacuous combination).
  3. The YAML config layer round-trips ``already_R_contracted`` through
     the ``nonlocal_vertices`` block.
  4. ``System.build_action()`` raises ``NotImplementedError`` (with a
     pointer to the design note) when an R-contracted vertex is present
     — locking the Phase-1 safeguard in place until the Phase-2
     dispatch lands.
  5. ``sft_wick.build_R_contracted_callable`` brute-force-computes the
     R-contracted κ³ on a small grid and recovers the analytical
     limit case ``R = 1`` (returning ``κ³`` integrated over χ).

The dispatch-level equivalence test (paired raw vs R_contracted run
on demo2 FK) is deferred to Phase 2; see
``docs/notes/R_contracted_nonlocal_vertex.md`` §4.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

import sft_wick as sw
from sft_wick.workflow import build_R_contracted_callable
from sft_wick.workflow.config import build_system, load_workflow_config


# ---------------------------------------------------------------------------
# 1. NonLocalVertex.already_R_contracted schema
# ---------------------------------------------------------------------------


def _trivial_callable():
    def fn(n_list, t_list):  # noqa: ARG001
        return np.zeros((2, 2, 2))
    return fn


def test_already_R_contracted_default_is_false():
    nv = sw.NonLocalVertex(name="K", order=3, coupling=_trivial_callable())
    assert nv.already_R_contracted is False


def test_already_R_contracted_accepts_true():
    nv = sw.NonLocalVertex(
        name="K", order=3, coupling=_trivial_callable(),
        already_R_contracted=True,
    )
    assert nv.already_R_contracted is True


def test_already_R_contracted_rejects_equal_time_combo():
    with pytest.raises(ValueError, match="mutually exclusive"):
        sw.NonLocalVertex(
            name="K", order=3, coupling=_trivial_callable(),
            already_R_contracted=True,
            equal_time=True,
        )


# ---------------------------------------------------------------------------
# 2. YAML round-trip
# ---------------------------------------------------------------------------


_YAML_TEMPLATE = """
system:
  field: {{name: phi, n_components: 2}}
  linear: {{type: diagonal, gamma: [1.0, 1.0]}}
  vertices: []
  nonlocal_vertices:
    - name: K
      order: 3
      coupling_module: {coupling_module}
      coupling_attr: coupling_fn
      already_R_contracted: {already_R_contracted}
  noise:
    kappa2:
      type: separable_translation
      temporal: {{type: exponential, lam: 0.05, sigma_t: 0.3}}
      spatial:  {{type: exponential, sigma_x: 1.0}}
    sigma2: null

expand:
  observable: ["phi_a(x)", "phi_b(y)"]
  orders: [0]

propagators:
  t_max: 5.0
  n_grid_t: 16

sweep:
  positions_grid: {{x: [0.0], y: [0.5]}}
  t_final_grid: [1.0]
  component_pairs: [[0, 0]]
  orders: [0]

output: []
"""


_KAPPA3_MODULE = textwrap.dedent(
    """
    import numpy as np

    def coupling_fn(n_list, t_list):  # noqa: ARG001
        return np.zeros((2, 2, 2), dtype=float)
    """
)


@pytest.fixture
def _kappa3_module(tmp_path: Path) -> Path:
    p = tmp_path / "k3_stub.py"
    p.write_text(_KAPPA3_MODULE)
    return p


def _write_yaml(tmp_path: Path, kappa3_module: Path, already_R_contracted: str) -> Path:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(_YAML_TEMPLATE.format(
        coupling_module=str(kappa3_module),
        already_R_contracted=already_R_contracted,
    ))
    return cfg_path


def test_yaml_round_trip_default_false(tmp_path: Path, _kappa3_module: Path):
    cfg_path = _write_yaml(tmp_path, _kappa3_module, already_R_contracted="false")
    wf = load_workflow_config(cfg_path)
    system = build_system(wf.system)
    assert system.nonlocal_vertices[0].already_R_contracted is False


def test_yaml_round_trip_true(tmp_path: Path, _kappa3_module: Path):
    cfg_path = _write_yaml(tmp_path, _kappa3_module, already_R_contracted="true")
    wf = load_workflow_config(cfg_path)
    system = build_system(wf.system)
    assert system.nonlocal_vertices[0].already_R_contracted is True


def test_yaml_missing_already_R_contracted_defaults_false(
    tmp_path: Path, _kappa3_module: Path,
):
    """YAML configs predating this feature must still parse and
    produce ``already_R_contracted is False`` automatically."""
    cfg_path = tmp_path / "cfg_no_flag.yaml"
    cfg_path.write_text(_YAML_TEMPLATE.format(
        coupling_module=str(_kappa3_module),
        already_R_contracted="false",
    ).replace("      already_R_contracted: false\n", ""))
    wf = load_workflow_config(cfg_path)
    system = build_system(wf.system)
    assert system.nonlocal_vertices[0].already_R_contracted is False


# ---------------------------------------------------------------------------
# 3. Build-action correctness for both raw and R-contracted vertices
# ---------------------------------------------------------------------------


def _system_with_K(already_R_contracted: bool):
    F = np.zeros((2, 2, 2))
    F[0, 1, 1] = 1.0
    return sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        vertices=[sw.LocalVertex("F", coupling=F)],
        nonlocal_vertices=[sw.NonLocalVertex(
            name="K", order=3, coupling=_trivial_callable(),
            already_R_contracted=already_R_contracted,
        )],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
            spatial=sw.ExponentialSpatial(sigma_x=1.0),
        )),
    )


def test_build_action_propagates_flag_to_raw_Vertex():
    """The ``already_R_contracted`` flag must reach the L0 ``Vertex``
    so the diagram-graph dispatch can identify which vertex's R-legs
    to absorb."""
    system = _system_with_K(already_R_contracted=True)
    action = system.build_action()
    # F (local) + K (non-local) = 2 raw vertices.
    assert len(action.vertices) == 2
    [k_vertex] = [v for v in action.vertices if not v.local]
    assert k_vertex.already_R_contracted is True
    [f_vertex] = [v for v in action.vertices if v.local]
    assert f_vertex.already_R_contracted is False


def test_build_action_default_raw_vertex():
    """Default ``already_R_contracted=False`` propagates as False."""
    system = _system_with_K(already_R_contracted=False)
    action = system.build_action()
    [k_vertex] = [v for v in action.vertices if not v.local]
    assert k_vertex.already_R_contracted is False


# ---------------------------------------------------------------------------
# 4. Phase-2 equivalence: raw vs already_R_contracted produce identical
# diagram values when the user-supplied callable is the exact analytical
# R-contraction of the raw kernel.
#
# Setup: raw κ³ is a constant tensor (independent of leg times). Then
# the R-contracted form factors exactly:
#
#   κ³_R(γ; λ_1', λ_2', λ_3') = K · ∏_i (1 - exp(-γ λ_i')) / γ
#
# Both pipelines should produce IDENTICAL diagram values to machine
# precision. Any disagreement would isolate a bug in the
# R-absorption dispatch (NOT in the brute-force reference quadrature).
# ---------------------------------------------------------------------------


def _equivalence_system(k3_callable, already_R_contracted: bool, *, gamma: float):
    F = np.zeros((2, 2, 2))
    F[0, 1, 1] = 1.0
    F[1, 0, 1] = F[1, 1, 0] = 0.5
    return sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[gamma, gamma]),
        vertices=[sw.LocalVertex("F", coupling=F)],
        nonlocal_vertices=[sw.NonLocalVertex(
            name="K", order=3, coupling=k3_callable,
            already_R_contracted=already_R_contracted,
        )],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.1, sigma_t=0.3),
            spatial=sw.ExponentialSpatial(sigma_x=1.0),
        )),
    )


def test_raw_vs_R_contracted_agree_at_machine_precision():
    gamma = 1.0
    K_tensor = np.zeros((2, 2, 2))
    for a in range(2):
        for b in range(2):
            for c in range(2):
                # Parity-restricted constant κ³ — non-zero on the (0,1)
                # cross-pair which is the only non-trivial FK channel for
                # this F.
                K_tensor[a, b, c] = float((a + b + c) % 2 == 1)

    def raw_k3(n_list, t_list):  # noqa: ARG001
        return K_tensor.copy()

    def k3_R_analytical(n_list, t_list_outer):  # noqa: ARG001
        t = np.asarray(t_list_outer, dtype=float)
        factor = float(np.prod((1.0 - np.exp(-gamma * t)) / gamma))
        return K_tensor * factor

    sweep_kwargs = dict(
        positions_grid={"x": [0.0], "y": [0.5]},
        t_final_grid=[1.0, 2.0, 3.0],
        component_pairs=[(0, 1)],
        orders=[2],
        vertex_types=["FK"],
        method="gauss_legendre",
        n_gauss=10,
    )

    def _run(k3_callable, already_R_contracted):
        sys_ = _equivalence_system(k3_callable, already_R_contracted, gamma=gamma)
        exp_ = sys_.expand(("phi_a(x)", "phi_b(y)"), orders=[2])
        props_ = sys_.propagators(t_max=5.0, n_grid_t=24)
        return exp_.sweep(props_, **sweep_kwargs).totals()

    raw_df = _run(raw_k3, already_R_contracted=False)
    rc_df = _run(k3_R_analytical, already_R_contracted=True)

    merged = raw_df.merge(
        rc_df, on=["x", "y", "t_final", "a", "b", "order"],
        suffixes=("_raw", "_rc"),
    )
    assert len(merged) == 3
    # Both pipelines must agree to ~ machine precision on every cell.
    # We use rtol=1e-12 — a generous bound given observed 1.8e-15.
    np.testing.assert_allclose(
        merged["value_rc"].to_numpy(),
        merged["value_raw"].to_numpy(),
        rtol=1e-12, atol=1e-14,
    )
    # The headline value must be non-zero (otherwise the test would be
    # trivially passing).
    assert (merged["value_raw"].abs() > 0.01).all()


def test_R_contracted_diagram_carries_r_absorbed_pairs():
    """Structural check: the FK-channel DiagramTerms emitted under
    ``already_R_contracted=True`` carry non-empty ``r_absorbed_pairs``
    with one entry per κ leg (m=3 for κ³)."""
    K_tensor = np.zeros((2, 2, 2))
    K_tensor[0, 0, 1] = K_tensor[0, 1, 0] = K_tensor[1, 0, 0] = 1.0

    def k3(n_list, t_list):  # noqa: ARG001
        return K_tensor.copy()

    sys_R = _equivalence_system(k3, already_R_contracted=True, gamma=1.0)
    exp_R = sys_R.expand(("phi_a(x)", "phi_b(y)"), orders=[2])

    fk_dts = exp_R.by_vertex_type(2).get("FK", [])
    assert fk_dts, "expected FK diagrams at order 2"
    # Every FK diagram must absorb exactly m=3 R-propagators (one per K leg).
    for dt in fk_dts:
        assert len(dt.r_absorbed_pairs) == 3, (
            f"expected 3 absorbed R-pairs (K has 3 ψ legs), "
            f"got {len(dt.r_absorbed_pairs)}: {dt.r_absorbed_pairs}"
        )
        # Each absorbed leg must also appear in equal_time_aliases so its
        # time integration variable is dropped.
        absorbed_legs = {leg for _, leg in dt.r_absorbed_pairs}
        aliased_legs = {leg for leg, _ in dt.equal_time_aliases}
        assert absorbed_legs <= aliased_legs, (
            f"absorbed legs {absorbed_legs} not all in alias map "
            f"{aliased_legs}"
        )


def test_raw_path_carries_no_r_absorbed_pairs():
    """Default raw path must NOT populate r_absorbed_pairs."""
    K_tensor = np.zeros((2, 2, 2))
    K_tensor[0, 0, 1] = 1.0

    def k3(n_list, t_list):  # noqa: ARG001
        return K_tensor.copy()

    sys_raw = _equivalence_system(k3, already_R_contracted=False, gamma=1.0)
    exp_raw = sys_raw.expand(("phi_a(x)", "phi_b(y)"), orders=[2])
    for dt in exp_raw.diagrams(2):
        assert dt.r_absorbed_pairs == ()


# ---------------------------------------------------------------------------
# 5. Vectorised callable + already_R_contracted
#
# The leg-alias / partner-lookup machinery threads batched ``t_list``
# arrays through the equal-time-alias dispatch in
# ``evaluate.py::DynamicCouplingPromise.evaluate_at_batch``. Since
# ``already_R_contracted=True`` re-uses that exact dispatch (just with
# a per-leg, rather than per-vertex, alias map), the vectorised contract
# should "just work". This test pins it down.
# ---------------------------------------------------------------------------


def _vec_equivalence_system(k3_callable, *, already_R_contracted: bool,
                            coupling_vectorized: bool, gamma: float):
    """Variant of ``_equivalence_system`` that exposes the
    ``coupling_vectorized`` knob."""
    import dataclasses
    base = _equivalence_system(k3_callable, already_R_contracted, gamma=gamma)
    # Replace the auto-built non-local vertex with one carrying the
    # ``coupling_vectorized`` flag.
    return dataclasses.replace(
        base,
        nonlocal_vertices=(
            sw.NonLocalVertex(
                name="K", order=3,
                coupling=k3_callable,
                coupling_vectorized=coupling_vectorized,
                already_R_contracted=already_R_contracted,
            ),
        ),
    )


def test_vectorised_R_contracted_matches_per_sample():
    """Four-way machine-precision equivalence:

        raw + per-sample
        raw + vectorised
        already_R_contracted + per-sample
        already_R_contracted + vectorised

    All four must agree on the F+K order-2 (0, 1)-channel headline.
    The constant-κ³ + analytical-R-contraction setup makes Fubini
    exact term-by-term, so any disagreement isolates a dispatch bug.
    """
    gamma = 1.0
    K_tensor = np.zeros((2, 2, 2))
    for a in range(2):
        for b in range(2):
            for c in range(2):
                K_tensor[a, b, c] = float((a + b + c) % 2 == 1)

    # Raw κ³ — independent of leg times.
    def raw_k3_per_sample(n_list, t_list):  # noqa: ARG001
        return K_tensor.copy()

    def raw_k3_vectorized(n_2d, t_2d):  # noqa: ARG001
        n_samples = np.asarray(t_2d).shape[1]
        return np.broadcast_to(
            K_tensor[None, :, :, :], (n_samples, 2, 2, 2),
        ).copy()

    # R-contracted form (exact closed form for constant raw κ³).
    def k3_R_per_sample(n_list, t_list_outer):  # noqa: ARG001
        t = np.asarray(t_list_outer, dtype=float)
        factor = float(np.prod((1.0 - np.exp(-gamma * t)) / gamma))
        return K_tensor * factor

    def k3_R_vectorized(n_2d, t_2d_outer):  # noqa: ARG001
        # t_2d_outer shape (3, n_samples)
        t = np.asarray(t_2d_outer, dtype=float)
        # ∏_i (1-exp(-γ t_i))/γ along the leg axis ⇒ (n_samples,)
        factors = np.prod((1.0 - np.exp(-gamma * t)) / gamma, axis=0)
        return K_tensor[None, :, :, :] * factors[:, None, None, None]

    sweep_kwargs = dict(
        positions_grid={"x": [0.0], "y": [0.5]},
        t_final_grid=[1.0, 2.0, 3.0],
        component_pairs=[(0, 1)],
        orders=[2],
        vertex_types=["FK"],
        method="gauss_legendre",
        n_gauss=10,
    )

    def _run(system):
        exp_ = system.expand(("phi_a(x)", "phi_b(y)"), orders=[2])
        props_ = system.propagators(t_max=5.0, n_grid_t=24)
        return exp_.sweep(props_, **sweep_kwargs).totals()

    df_raw_ps = _run(_vec_equivalence_system(
        raw_k3_per_sample, already_R_contracted=False,
        coupling_vectorized=False, gamma=gamma,
    ))
    df_raw_vec = _run(_vec_equivalence_system(
        raw_k3_vectorized, already_R_contracted=False,
        coupling_vectorized=True, gamma=gamma,
    ))
    df_rc_ps = _run(_vec_equivalence_system(
        k3_R_per_sample, already_R_contracted=True,
        coupling_vectorized=False, gamma=gamma,
    ))
    df_rc_vec = _run(_vec_equivalence_system(
        k3_R_vectorized, already_R_contracted=True,
        coupling_vectorized=True, gamma=gamma,
    ))

    # Reference: raw per-sample.
    ref = df_raw_ps["value"].to_numpy()
    # All four must equal the reference (headlines are non-zero, so
    # we can use rtol meaningfully).
    assert (np.abs(ref) > 0.01).all(), (
        "headline values must be non-zero for a meaningful comparison"
    )

    for label, df_other in [
        ("raw vectorised", df_raw_vec),
        ("R_contracted per-sample", df_rc_ps),
        ("R_contracted vectorised", df_rc_vec),
    ]:
        other = df_other["value"].to_numpy()
        np.testing.assert_allclose(
            other, ref, rtol=1e-12, atol=1e-14,
            err_msg=f"{label} disagrees with raw per-sample reference",
        )


# ---------------------------------------------------------------------------
# 4. build_R_contracted_callable smoke + analytical-limit check
# ---------------------------------------------------------------------------


def test_build_R_contracted_callable_R_equals_unity():
    """When R ≡ 1 on the grid, the R-contracted callable reduces to
    the plain m-leg integral of the raw κ^(m) over the χ-grid."""

    def raw_fn(n_list, t_list):  # noqa: ARG001
        # Simple separable κ³ for an exact analytical check.
        t = np.asarray(t_list, dtype=float)
        return np.eye(2)[..., None].repeat(2, axis=2) * float(np.prod(np.exp(-t)))

    chi_grid = np.linspace(0.0, 5.0, 81)

    contracted = build_R_contracted_callable(
        raw_coupling_fn=raw_fn,
        R_time=lambda t_outer, t_inner: 1.0,  # unity R, non-causal
        chi_grid=chi_grid,
        order=3,
        n_components=2,
        causal=False,
    )

    # ∫∫∫ exp(-χ_1 - χ_2 - χ_3) dχ on [0, 5]^3 = (1 - e^{-5})^3.
    expected_scalar = (1.0 - np.exp(-5.0)) ** 3
    result = contracted([0, 0, 0], [0.0, 0.0, 0.0])
    assert result.shape == (2, 2, 2)
    # The tensor structure mirrors raw_fn's: diagonal κ³. Trapezoid
    # quadrature on the exponential at h=5/80 carries an O(h²)≈4e-3
    # error per axis; the 3-leg integral compounds it to ~1e-3.
    diag_val = result[0, 0, 0]
    np.testing.assert_allclose(diag_val, expected_scalar, rtol=2e-3)


def test_build_R_contracted_callable_causal_short_circuit():
    """Causal R should make outer-times below the grid minimum return
    zero (no χ-samples survive)."""

    def raw_fn(n_list, t_list):  # noqa: ARG001
        return np.ones((2, 2, 2))

    chi_grid = np.linspace(1.0, 5.0, 11)
    contracted = build_R_contracted_callable(
        raw_coupling_fn=raw_fn,
        R_time=lambda t_outer, t_inner: 1.0,
        chi_grid=chi_grid,
        order=3,
        causal=True,
    )

    # Every outer time is below the χ-grid minimum (1.0), so every
    # χ-sample short-circuits to R=0.
    result = contracted([0, 0, 0], [0.5, 0.5, 0.5])
    np.testing.assert_array_equal(result, np.zeros((2, 2, 2)))


def test_build_R_contracted_callable_rejects_bad_inputs():
    def raw_fn(n_list, t_list):  # noqa: ARG001
        return np.zeros((2, 2, 2))

    with pytest.raises(ValueError, match="chi_grid"):
        build_R_contracted_callable(raw_fn, lambda a, b: 1.0, chi_grid=[1.0])

    with pytest.raises(ValueError, match="order"):
        build_R_contracted_callable(
            raw_fn, lambda a, b: 1.0,
            chi_grid=np.linspace(0, 1, 5), order=0,
        )

    contracted = build_R_contracted_callable(
        raw_fn, lambda a, b: 1.0,
        chi_grid=np.linspace(0, 1, 5), order=3, causal=False,
    )
    with pytest.raises(ValueError, match="t_list_outer"):
        contracted([0, 0, 0], [0.5, 0.5])  # wrong length
