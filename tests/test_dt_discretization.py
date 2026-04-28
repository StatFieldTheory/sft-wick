"""Convergence smoke test for the ``propagators.dt`` knob.

The contract documented in ``docs/user_guide/discretization.rst`` is that
when the source-cumulant ``kappa2`` is smooth on the t-grid, halving ``dt``
should change observables only at order ``dt`` (i.e. by a small fraction).
This test verifies that contract on the demo1 separable-translation noise
(Gaussian temporal kernel, sigma_t = 0.3, much larger than the dt values
tested).
"""
from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml

from sft_wick.workflow.config import load_workflow_config, run_workflow


# Same demo1 closed-form C as ``test_workflow_config.py`` -- avoids
# the ~30-100x dblquad cost on every parametric run_workflow call.
_DEMO1_CLOSED_FORM_SRC = (
    Path(__file__).resolve().parents[1]
    / "examples" / "demo1" / "c_closed_form.py"
)


_DT_CONVERGENCE_YAML = textwrap.dedent("""
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
      orders: [0]

    propagators:
      t_max: 2.0
      dt: 0.1                # n_grid_t = 20
      homogeneity: translation
      c_closed_form_module: ./demo1_c_closed_form.py
      c_closed_form_attr: C_fn

    sweep:
      integrate_over: all
      positions_grid:
        x: [0.0]
        y: [0.5]
      t_final_grid: [1.0]
      component_pairs: [[0, 0]]
      orders: [0]
      n_samples: 256
      seed: 42
""")


@pytest.fixture(autouse=True)
def _stage_closed_form(tmp_path: Path) -> None:
    """Stage the demo1 closed-form C module next to each test's
    ``tmp_path`` so the YAML's relative reference resolves."""
    target = tmp_path / "demo1_c_closed_form.py"
    if not target.exists():
        shutil.copy(_DEMO1_CLOSED_FORM_SRC, target)


def _run_with_dt(tmp_path: Path, dt: float) -> float:
    cfg_yaml = yaml.safe_load(_DT_CONVERGENCE_YAML)
    cfg_yaml["propagators"]["dt"] = dt
    p = tmp_path / f"c_dt{dt}.yaml"
    p.write_text(yaml.safe_dump(cfg_yaml))
    cfg = load_workflow_config(p)
    _, totals = run_workflow(cfg)
    # One row at order 0 with the (0,0) component pair.
    return float(totals.loc[totals["order"] == 0, "value"].iloc[0])


def test_dt_convergence_smooth_kappa2(tmp_path: Path) -> None:
    """Halving dt should not change a smooth-noise observable at O(1)."""
    val_coarse = _run_with_dt(tmp_path, dt=0.1)   # n_grid_t = 20
    val_fine = _run_with_dt(tmp_path, dt=0.05)    # n_grid_t = 40
    # Order-0 (Gaussian) observable should agree well because the noise has
    # sigma_t = 0.3, comfortably resolved by either dt.
    assert val_coarse != 0.0, "test fixture produced a trivial zero observable"
    rel_diff = abs(val_fine - val_coarse) / abs(val_coarse)
    assert rel_diff < 5e-2, (
        f"dt-halving moved the smooth-kappa2 observable by {rel_diff:.3%}, "
        f"expected <5%; coarse={val_coarse:.6e}, fine={val_fine:.6e}"
    )


def test_dt_zero_or_negative_is_rejected(tmp_path: Path) -> None:
    cfg_yaml = yaml.safe_load(_DT_CONVERGENCE_YAML)
    cfg_yaml["propagators"]["dt"] = 0.0
    p = tmp_path / "c_bad.yaml"
    p.write_text(yaml.safe_dump(cfg_yaml))
    with pytest.raises(ValueError, match="positive"):
        load_workflow_config(p)
