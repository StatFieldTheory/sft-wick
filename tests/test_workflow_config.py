"""Tests for the YAML/CLI (L2) layer of the workflow API.

CF1 — a demo1-style YAML builds a :class:`System` structurally
      identical to the hand-constructed demo1 system.
CF2 — demo2-style YAML with a ``NonLocalVertex`` produces the
      right vertex shape.
CF3 — full-config ``run_workflow`` produces the same
      ``sweep.totals()`` DataFrame as a Python-only L1 flow.
CF4 — ``--override sweep.seed=…`` patches the correct field.
CF5 — malformed YAML raises a readable error.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml

import sft_wick as sw
from sft_wick.workflow.config import (
    build_system,
    load_workflow_config,
    run_workflow,
)


# --- Shared minimal YAML blobs (written to tmp_path per test) -------- #


_DEMO1_YAML = textwrap.dedent("""
    system:
      field: {name: phi, n_components: 2}
      linear: {type: diagonal, gamma: [1.0, 1.0]}
      vertices:
        - name: F
          coupling:
            - [[0.0, 0.0], [0.0, 1.0]]
            - [[0.0, 0.5], [0.5, 0.0]]
      nonlocal_vertices: []
      noise:
        kappa2:
          type: separable_translation
          temporal: {type: exponential, lam: 0.05, sigma_t: 0.3}
          spatial:  {type: exponential, sigma_x: 1.0}
        sigma2: null

    expand:
      observable: ["phi_a(x)", "phi_b(y)"]
      orders: [0, 2]

    propagators:
      t_max: 2.0
      n_grid_t: 12

    sweep:
      integrate_over: all        # keep the all-integrated convention
                                 # so we can compare against the L1
                                 # reference cleanly
      positions_grid:
        x: [0.0]
        y: [0.0, 0.5]
      t_final_grid: [1.0]
      component_pairs: [[0, 0]]
      orders: [0, 2]
      n_samples: 256
      seed: 42
""")


_DEMO2_YAML = textwrap.dedent("""
    system:
      field: {name: phi, n_components: 2}
      linear: {type: diagonal, gamma: [1.0, 1.0]}
      vertices:
        - name: F
          coupling:
            - [[0.0, 0.0], [0.0, 1.0]]
            - [[0.0, 0.5], [0.5, 0.0]]
      nonlocal_vertices:
        - name: K
          order: 3
          coupling:
            - [[0.0, 0.0], [0.0, 0.0]]
            - [[0.0, 0.0], [0.0, 0.0]]
      noise:
        kappa2:
          type: separable_translation
          temporal: {type: exponential, lam: 0.05, sigma_t: 0.3}
          spatial:  {type: exponential, sigma_x: 1.0}

    expand:
      observable: ["phi_a(x)", "phi_b(y)"]
      orders: [2]

    propagators:
      t_max: 3.0
      n_grid_t: 20

    sweep:
      positions_grid:
        x: [0.0]
        y: [0.5]
      t_final_grid: [1.0]
      component_pairs: [[0, 0]]
      vertex_types: [F]
      n_samples: 256
      seed: 1
""")


def _write(tmp: Path, name: str, body: str) -> Path:
    p = tmp / name
    p.write_text(body)
    return p


# =====================================================================
# CF1 — YAML → System structure
# =====================================================================


def test_CF1_yaml_builds_demo1_system(tmp_path: Path) -> None:
    cfg = load_workflow_config(_write(tmp_path, "c.yaml", _DEMO1_YAML))
    system = build_system(cfg.system)

    assert system.n_components == 2
    assert isinstance(system.linear, sw.DiagonalA)
    np.testing.assert_allclose(system.linear.gamma, [1.0, 1.0])
    assert len(system.vertices) == 1
    assert system.vertices[0].name == "F"
    # Bare F tensor: F_{0,1,1}=1, F_{1,0,1}=F_{1,1,0}=0.5
    arr = np.asarray(system.vertices[0].coupling)
    assert arr.shape == (2, 2, 2)
    assert arr[0, 1, 1] == pytest.approx(1.0)
    assert arr[1, 0, 1] == pytest.approx(0.5)
    assert arr[1, 1, 0] == pytest.approx(0.5)
    # Noise kappa2 is the separable-translation variant
    assert isinstance(system.noise.kappa2, sw.SeparableTranslation)
    assert system.homogeneity == "translation"


# =====================================================================
# CF2 — YAML with non-local vertex
# =====================================================================


def test_CF2_yaml_non_local_vertex(tmp_path: Path) -> None:
    cfg = load_workflow_config(_write(tmp_path, "c.yaml", _DEMO2_YAML))
    system = build_system(cfg.system)
    assert len(system.nonlocal_vertices) == 1
    nv = system.nonlocal_vertices[0]
    assert nv.name == "K"
    assert nv.order == 3
    assert np.asarray(nv.coupling).shape == (2, 2, 2)

    # The classification pipeline yields 6 F + 2 FK at order 2 —
    # same as test_WF3, via the CLI-built System.
    exp = system.expand(cfg.expand.observable, orders=cfg.expand.orders)
    groups = exp.by_vertex_type(2)
    assert len(groups.get("F", [])) == 6
    assert len(groups.get("FK", [])) == 2


# =====================================================================
# CF3 — full run matches a Python-only L1 reference
# =====================================================================


def test_CF3_full_run_matches_L1_reference(tmp_path: Path) -> None:
    cfg = load_workflow_config(_write(tmp_path, "c.yaml", _DEMO1_YAML))
    sweep_yaml, totals_yaml = run_workflow(cfg)

    # Hand-built reference
    F = np.zeros((2, 2, 2))
    F[0, 1, 1] = 1.0
    F[1, 0, 1] = 0.5
    F[1, 1, 0] = 0.5
    system = sw.System(
        field=sw.FieldSpec("phi", n_components=2),
        linear=sw.DiagonalA(gamma=[1.0, 1.0]),
        vertices=[sw.LocalVertex("F", coupling=F)],
        noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
            temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
            spatial=sw.ExponentialSpatial(sigma_x=1.0),
        )),
    )
    exp = system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2])
    props = system.propagators(t_max=2.0, n_grid_t=12)
    sweep_ref = exp.sweep(
        props,
        positions_grid={"x": [0.0], "y": [0.0, 0.5]},
        t_final_grid=[1.0],
        component_pairs=[(0, 0)],
        orders=[0, 2],
        integrate_over="all",
        n_samples=256, seed=42,
    )
    totals_ref = sweep_ref.totals()

    # Compare value-by-value (aligned by index after sorting)
    key = ["x", "y", "t_final", "a", "b", "order"]
    yaml_sorted = totals_yaml.sort_values(key).reset_index(drop=True)
    ref_sorted = totals_ref.sort_values(key).reset_index(drop=True)
    np.testing.assert_allclose(
        yaml_sorted["value"].to_numpy(),
        ref_sorted["value"].to_numpy(),
        rtol=1e-10,
    )


# =====================================================================
# CF4 — override
# =====================================================================


def test_CF4_override_patches_field(tmp_path: Path) -> None:
    path = _write(tmp_path, "c.yaml", _DEMO1_YAML)
    cfg0 = load_workflow_config(path)
    assert cfg0.sweep.seed == 42
    cfg1 = load_workflow_config(path, overrides={"sweep.seed": 7})
    assert cfg1.sweep.seed == 7


# =====================================================================
# CF5 — malformed input
# =====================================================================


def test_CF5_missing_required_section_raises(tmp_path: Path) -> None:
    # Config lacks ``expand`` section.
    bad = textwrap.dedent("""
        system:
          field: {name: phi, n_components: 1}
          linear: {type: diagonal, gamma: [1.0]}
          vertices: []
          noise:
            kappa2:
              type: separable_translation
              temporal: {type: exponential, lam: 0.05, sigma_t: 0.3}
              spatial:  {type: exponential, sigma_x: 1.0}
        propagators: {t_max: 1.0}
        sweep:
          positions_grid: {x: [0.0]}
          t_final_grid: [1.0]
          component_pairs: [[0, 0]]
    """)
    with pytest.raises(ValueError, match="expand"):
        load_workflow_config(_write(tmp_path, "c.yaml", bad))


def test_CF5_unknown_kappa2_type_raises(tmp_path: Path) -> None:
    bad_yaml = yaml.safe_load(_DEMO1_YAML)
    bad_yaml["system"]["noise"]["kappa2"]["type"] = "not_a_real_type"
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(bad_yaml))
    cfg = load_workflow_config(tmp_path / "c.yaml")
    with pytest.raises(ValueError, match="kappa2.type"):
        build_system(cfg.system)
