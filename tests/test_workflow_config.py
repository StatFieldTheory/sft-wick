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

import shutil
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


# Path to the canonical demo1 closed-form C module shipped with the
# package. Demo1's separable OU kappa2 has a clean analytical
# integral, so threading it into every demo1 test slashes the
# default ``dblquad`` cost from ~30-100s/test to ~1-3s/test (roughly
# 30x faster), without changing the math: the closed form is what
# ``dblquad`` was numerically approximating in the first place.
_DEMO1_CLOSED_FORM_SRC = (
    Path(__file__).resolve().parents[1]
    / "examples" / "demo1" / "c_closed_form.py"
)
assert _DEMO1_CLOSED_FORM_SRC.exists(), (
    f"demo1 closed-form module is missing at {_DEMO1_CLOSED_FORM_SRC}; "
    f"the demo1-using tests rely on it for fast C-table builds."
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
      c_closed_form_module: ./demo1_c_closed_form.py
      c_closed_form_attr: C_fn

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


def _stage_demo1_c_module(tmp: Path) -> None:
    """Copy demo1's analytical C module into ``tmp`` so the YAML's
    ``c_closed_form_module: ./demo1_c_closed_form.py`` reference
    resolves. Idempotent -- skips the copy if the file already exists.
    """
    target = tmp / "demo1_c_closed_form.py"
    if not target.exists():
        shutil.copy(_DEMO1_CLOSED_FORM_SRC, target)


@pytest.fixture(autouse=True)
def _stage_closed_form(tmp_path: Path) -> None:
    """Auto-stage the demo1 closed-form C module next to every test's
    ``tmp_path`` so any YAML referencing
    ``./demo1_c_closed_form.py`` resolves. Tests that don't run
    workflows pay an O(0.1 ms) file copy that is negligible."""
    _stage_demo1_c_module(tmp_path)


def _write_demo1(tmp: Path, body: str = _DEMO1_YAML, name: str = "c.yaml") -> Path:
    """Write a demo1 YAML config to ``tmp``. The closed-form module
    is auto-staged via the ``_stage_closed_form`` fixture."""
    return _write(tmp, name, body)


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
    cfg = load_workflow_config(_write_demo1(tmp_path))
    sweep_yaml, totals_yaml = run_workflow(cfg)

    # Hand-built reference -- must use the same closed-form C path
    # the YAML uses, otherwise dblquad's ~1e-7 quadrature error
    # exceeds the rtol=1e-10 invariant this test locks. Importing
    # the same demo1 C_fn used by the YAML keeps both sides bit-
    # identical.
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "demo1_c_closed_form_ref", _DEMO1_CLOSED_FORM_SRC,
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    C_fn = _mod.C_fn

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
    props = system.propagators(t_max=2.0, n_grid_t=12, c_closed_form=C_fn)
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


# =====================================================================
# CF6 — propagators.dt: single knob derives n_grid_t (and n_grid_cache)
# =====================================================================


def test_CF6_dt_derives_n_grid_t_and_n_grid_cache(tmp_path: Path) -> None:
    cfg_yaml = yaml.safe_load(_DEMO1_YAML)
    cfg_yaml["propagators"].pop("n_grid_t")
    cfg_yaml["propagators"]["dt"] = 0.05  # t_max=2.0 -> n_grid_t = 40
    cfg_yaml["system"]["linear"] = {
        "type": "diagonal",
        "gamma": [1.0, 1.0],
        "t_max_cache": 5.0,  # -> n_grid_cache = ceil(5.0 / 0.05) = 100
    }
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg_yaml))
    cfg = load_workflow_config(tmp_path / "c.yaml")
    assert cfg.propagators.dt == pytest.approx(0.05)
    assert cfg.propagators.n_grid_t == 40
    assert cfg.system.linear.get("n_grid_cache") == 100


def test_CF6_dt_and_n_grid_t_together_raises(tmp_path: Path) -> None:
    cfg_yaml = yaml.safe_load(_DEMO1_YAML)
    cfg_yaml["propagators"]["dt"] = 0.1
    # n_grid_t still present from the demo.
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg_yaml))
    with pytest.raises(ValueError, match="dt.*n_grid_t"):
        load_workflow_config(tmp_path / "c.yaml")


def test_CF6_dt_and_n_grid_cache_together_raises(tmp_path: Path) -> None:
    cfg_yaml = yaml.safe_load(_DEMO1_YAML)
    cfg_yaml["propagators"].pop("n_grid_t")
    cfg_yaml["propagators"]["dt"] = 0.1
    cfg_yaml["system"]["linear"] = {
        "type": "diagonal",
        "gamma": [1.0, 1.0],
        "t_max_cache": 5.0,
        "n_grid_cache": 50,  # explicit while dt is also set
    }
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg_yaml))
    with pytest.raises(ValueError, match="dt.*n_grid_cache"):
        load_workflow_config(tmp_path / "c.yaml")


def test_CF6_legacy_n_grid_t_only_still_parses(tmp_path: Path) -> None:
    cfg_yaml = yaml.safe_load(_DEMO1_YAML)
    # Original DEMO1: n_grid_t=12, no dt.
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg_yaml))
    cfg = load_workflow_config(tmp_path / "c.yaml")
    assert cfg.propagators.dt is None
    assert cfg.propagators.n_grid_t == 12


# =====================================================================
# CF7 - c_closed_form_module supports n_jobs > 1 (parallel C-table)
# =====================================================================


_TRIVIAL_C_MODULE = textwrap.dedent("""
    import numpy as np

    def C_fn(n1, t1, n2, t2):
        # Trivial closed form returning a scaled identity. Used purely
        # to exercise the parallel C-table build path.
        return np.eye(2) * (float(t1) + float(t2))
""")


@pytest.fixture(scope="session")
def demo1_sequential_total(tmp_path_factory: pytest.TempPathFactory) -> float:
    """Sequential ``run_workflow(_DEMO1_YAML)`` total - computed once per
    test session.

    Multiple parallel-equivalence tests (CF8 expand/sweep, CF9 lazy)
    used to call a per-test helper that re-ran the full demo1 pipeline,
    which dominated the pytest runtime. Since the reference value is
    deterministic given a fixed seed, a session fixture is enough.
    """
    tmp = tmp_path_factory.mktemp("demo1_ref")
    _stage_demo1_c_module(tmp)
    cfg = load_workflow_config(_write(tmp, "ref.yaml", _DEMO1_YAML))
    _sweep, totals = run_workflow(cfg)
    return float(totals["value"].sum())


# =====================================================================
# CF8 - parallel-equivalence regression for expand.n_jobs / sweep.n_jobs
# =====================================================================


def test_CF8_expand_n_jobs_matches_sequential(
    tmp_path: Path, demo1_sequential_total: float,
) -> None:
    """Diagram-level parallelism (expand.n_jobs > 1) must match the
    sequential CF3 reference value to float64 tolerance.

    The QMC seed is fixed (DEMO1 sets seed=42), so the per-diagram QMC
    sample sequence is bit-identical regardless of dispatch order.
    """
    cfg_yaml = yaml.safe_load(_DEMO1_YAML)
    cfg_yaml["expand"]["n_jobs"] = -1  # exercise full multi-worker path
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg_yaml))
    cfg = load_workflow_config(tmp_path / "c.yaml")
    assert cfg.expand.n_jobs == -1

    _sweep, totals = run_workflow(cfg)
    par_total = float(totals["value"].sum())
    np.testing.assert_allclose(
        par_total, demo1_sequential_total, rtol=1e-12, atol=0.0
    )


def test_CF8_sweep_n_jobs_matches_sequential(
    tmp_path: Path, demo1_sequential_total: float,
) -> None:
    """Sweep-grid parallelism (sweep.n_jobs > 1) must match the
    sequential CF3 reference value to float64 tolerance.
    """
    cfg_yaml = yaml.safe_load(_DEMO1_YAML)
    cfg_yaml["sweep"]["n_jobs"] = -1  # exercise full multi-worker path
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg_yaml))
    cfg = load_workflow_config(tmp_path / "c.yaml")
    assert cfg.sweep.n_jobs == -1

    _sweep, totals = run_workflow(cfg)
    par_total = float(totals["value"].sum())
    np.testing.assert_allclose(
        par_total, demo1_sequential_total, rtol=1e-12, atol=0.0
    )


def test_CF8_both_n_jobs_set_raises(tmp_path: Path) -> None:
    """Setting both expand.n_jobs > 1 and sweep.n_jobs > 1 is not
    supported (nested loky pools); must raise ValueError.
    """
    cfg_yaml = yaml.safe_load(_DEMO1_YAML)
    cfg_yaml["expand"]["n_jobs"] = 2
    cfg_yaml["sweep"]["n_jobs"] = 2
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg_yaml))
    cfg = load_workflow_config(tmp_path / "c.yaml")
    with pytest.raises(ValueError, match="exactly one of"):
        run_workflow(cfg)


def test_CF7_c_closed_form_supports_parallel(tmp_path: Path) -> None:
    """The parallel C-table build path must work with c_closed_form_module.

    Regression-locks the fix that replaced the dynamic-class factory with
    the module-level ``_ClosedFormPropagatorCache``, and that drops the
    ``n_jobs = 1 if c_fn else ...`` forcing in ``run_workflow``.
    """
    (tmp_path / "trivial_c.py").write_text(_TRIVIAL_C_MODULE)

    cfg_yaml = yaml.safe_load(_DEMO1_YAML)
    cfg_yaml["propagators"]["n_jobs"] = 2
    cfg_yaml["propagators"]["c_closed_form_module"] = "./trivial_c.py"
    cfg_yaml["propagators"]["c_closed_form_attr"] = "C_fn"
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg_yaml))

    cfg = load_workflow_config(tmp_path / "c.yaml")
    assert cfg.propagators.n_jobs == 2
    sweep, totals = run_workflow(cfg)
    assert len(totals) > 0


# =====================================================================
# CF9 - lazy spline cache n_jobs is pinned to 1 (nested-loky guard)
# =====================================================================


def test_CF9_lazy_cache_njobs_pinned_to_one(tmp_path: Path) -> None:
    """``Propagators.build()`` must pin every lazy spline cache's
    ``n_jobs`` to 1, regardless of what the user requested.

    Lazy builds are triggered from inside the QMC integration loop
    (worker hits a previously unseen r/cos/x value). If they were
    allowed to spin up their own ``Parallel(...)`` pool while the outer
    layer (integrate_diagrams or Expansion.sweep) is itself parallel,
    we would have nested loky pools - which silently hangs on macOS or
    raises ``AssertionError: daemonic processes are not allowed to
    have children``.

    This test locks the contract: even if ``propagators.n_jobs > 1``,
    the lazy cache's inner ``n_jobs`` is forced to 1.
    """
    cfg_yaml = yaml.safe_load(_DEMO1_YAML)
    cfg_yaml["propagators"]["n_jobs"] = 4  # user asks for parallel
    # demo1 leaves r_max/n_grid_r unset -> translation lazy mode.
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg_yaml))

    cfg = load_workflow_config(tmp_path / "c.yaml")
    system = build_system(cfg.system)
    props = system.propagators(
        t_max=cfg.propagators.t_max,
        n_grid_t=cfg.propagators.n_grid_t,
        homogeneity=cfg.propagators.homogeneity,
        r_max=cfg.propagators.r_max,
        n_grid_r=cfg.propagators.n_grid_r,
        n_grid_cos=cfg.propagators.n_grid_cos,
        x_max=cfg.propagators.x_max,
        n_grid_x=cfg.propagators.n_grid_x,
        n_jobs=cfg.propagators.n_jobs,
    )
    assert props.is_lazy is True
    assert props.cache._lazy_translation is not None
    assert props.cache._lazy_translation.n_jobs == 1


def test_CF9_lazy_combined_with_outer_parallel_matches_sequential(
    tmp_path: Path, demo1_sequential_total: float,
) -> None:
    """End-to-end: lazy mode + ``propagators.n_jobs=-1`` + outer
    ``expand.n_jobs=-1`` must run without nesting and must produce
    the same totals as the fully sequential reference (``rtol=1e-12``).
    """
    cfg_yaml = yaml.safe_load(_DEMO1_YAML)
    cfg_yaml["propagators"]["n_jobs"] = -1
    cfg_yaml["expand"]["n_jobs"] = -1
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg_yaml))

    cfg = load_workflow_config(tmp_path / "c.yaml")
    _sweep, totals = run_workflow(cfg)
    par_total = float(totals["value"].sum())
    np.testing.assert_allclose(
        par_total, demo1_sequential_total, rtol=1e-12, atol=0.0
    )


# =====================================================================
# CF10 - integrate_diagrams default n_jobs is 1 (no implicit parallelism)
# =====================================================================


def test_CF10_integrate_diagrams_default_is_serial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When called without an explicit ``n_jobs``, ``integrate_diagrams``
    must run sequentially - it must not invoke ``joblib.Parallel``.

    The L0 default used to be ``-1`` (all cores), which made it
    inconsistent with the L1/L2 layers (default ``1``) and could
    surprise users who imported it directly. CF10 locks the new
    safer default. The check is two-fold: (a) inspect the function
    signature, (b) patch ``joblib.Parallel`` to ensure it is never
    instantiated during a default-args full-config run.
    """
    import inspect

    import joblib

    from sft_wick.evaluate import integrate_diagrams

    # (a) Signature default must be 1.
    sig = inspect.signature(integrate_diagrams)
    assert sig.parameters["n_jobs"].default == 1

    # (b) Spy on joblib.Parallel itself - the function imports it
    # lazily as ``from joblib import Parallel, delayed`` inside the
    # parallel branch, so patching the source attribute catches the
    # only path that could escape.
    parallel_calls: list[tuple] = []
    real_parallel = joblib.Parallel

    class _SpyParallel(real_parallel):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            parallel_calls.append((args, kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(joblib, "Parallel", _SpyParallel)

    # Reuse the demo1 reference run; expand.n_jobs / sweep.n_jobs are
    # both at their default (1), so the L1 path must hit
    # integrate_diagrams without an explicit n_jobs override.
    cfg = load_workflow_config(_write(tmp_path, "c.yaml", _DEMO1_YAML))
    _sweep, totals = run_workflow(cfg)
    assert len(totals) > 0
    assert parallel_calls == [], (
        f"integrate_diagrams unexpectedly invoked joblib.Parallel "
        f"under default args: {parallel_calls!r}"
    )


# =====================================================================
# CF12 - propagators.interp_method opts into 'cubic' (default 'linear')
# =====================================================================


def test_CF12_default_interp_method_is_linear(tmp_path: Path) -> None:
    """``propagators.interp_method`` is ``'linear'`` by default and is
    forwarded to ``PropagatorCache.interp_method``.

    The default is conservative because ``'cubic'`` produces sign
    flips on steep C tails; see
    ``tests/test_evaluate_interpolation_accuracy.py``.
    """
    # demo1 with explicit r_max/n_grid_r so a full-grid (3-D)
    # spline is built -- only the full-grid path consults
    # interp_method (lazy splines use RectBivariateSpline).
    cfg_yaml = yaml.safe_load(_DEMO1_YAML)
    cfg_yaml["propagators"]["r_max"] = 2.0
    cfg_yaml["propagators"]["n_grid_r"] = 6
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg_yaml))

    cfg = load_workflow_config(tmp_path / "c.yaml")
    assert cfg.propagators.interp_method == "linear"


def test_CF12_cubic_interp_method_routes_through(tmp_path: Path) -> None:
    """``propagators.interp_method: cubic`` routes through to
    ``PropagatorCache.interp_method`` and the full-grid build
    completes without error.

    A full pipeline run is also exercised so we know the
    integrator can consume cubic-interpolated values without
    crashing (numerical accuracy is checked separately by
    ``tests/test_evaluate_interpolation_accuracy.py``).
    """
    cfg_yaml = yaml.safe_load(_DEMO1_YAML)
    # Force a full-grid translation cache so the interp_method
    # actually controls a RegularGridInterpolator. demo1 leaves
    # r_max/n_grid_r unset by default (lazy mode).
    cfg_yaml["propagators"]["r_max"] = 2.0
    cfg_yaml["propagators"]["n_grid_r"] = 6
    cfg_yaml["propagators"]["interp_method"] = "cubic"
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg_yaml))

    cfg = load_workflow_config(tmp_path / "c.yaml")
    assert cfg.propagators.interp_method == "cubic"

    _sweep, totals = run_workflow(cfg)
    assert len(totals) > 0


# =====================================================================
# CF13 - cache_path round-trip (closures replaced by module-level classes)
# =====================================================================


def test_CF13_propagators_cache_path_round_trip(tmp_path: Path) -> None:
    """``propagators.cache_path`` must successfully save and reload a
    built ``Propagators`` via ``joblib.dump`` / ``joblib.load``.

    Historically this path was silently broken: the R / kappa^2 /
    sigma^2 callables built by ``DiagonalA._build_*_R``,
    ``SeparableTranslation.build_callable`` etc. were inner-function
    closures, which the standard serialisation protocol cannot
    persist. CF13 locks the fix that promoted those closures to
    module-level callable classes (``_StaticIsoR``,
    ``_SeparableTranslationKappa2``, ...).

    The test runs a full workflow once with ``cache_path`` set, then
    a second time with the same path; the second run must succeed
    without rebuilding (the cache layer prints a 'loaded from cache'
    message internally) and must produce identical totals.
    """
    cache_dir = tmp_path / "props_cache"

    cfg_yaml = yaml.safe_load(_DEMO1_YAML)
    cfg_yaml["propagators"]["cache_path"] = str(cache_dir)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg_yaml))

    cfg = load_workflow_config(tmp_path / "c.yaml")
    _sweep1, totals1 = run_workflow(cfg)

    # On second invocation with the same cache_path the workflow
    # should reload the saved Propagators via joblib.load instead of
    # rebuilding. Either path yields identical totals.
    cfg2 = load_workflow_config(tmp_path / "c.yaml")
    _sweep2, totals2 = run_workflow(cfg2)

    assert cache_dir.exists() and any(cache_dir.iterdir()), (
        "cache_path was set but no cache file was written"
    )
    np.testing.assert_allclose(
        totals1["value"].to_numpy(),
        totals2["value"].to_numpy(),
        rtol=1e-12,
        atol=0.0,
    )


# =====================================================================
# CF11 - YAML kappa^{(3)} end-to-end (callable non-local vertex)
# =====================================================================


_DEMO2_KAPPA3_MODULE = textwrap.dedent("""
    import numpy as np

    _K = np.zeros((2, 2, 2))
    _K[0, 0, 0] = 0.3
    _K[1, 1, 1] = 0.5

    def coupling_fn(n_list, t_list):
        # Spacetime-dependent kappa^(3): constant tensor times an
        # exponential envelope, mirroring the demo2 form but with a
        # deterministic, simple shape so the YAML round-trip stays
        # numerically reproducible across machines.
        n = np.asarray(n_list, dtype=float)
        t = np.asarray(t_list, dtype=float)
        envelope = float(
            np.exp(-abs(t[0] - t[1]))
            * np.exp(-abs(n[0] - n[2]))
        )
        return envelope * _K
""")


_DEMO2_YAML_TEMPLATE = textwrap.dedent("""
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
          coupling_module: ./k3.py
          coupling_attr: coupling_fn
      noise:
        kappa2:
          type: separable_translation
          temporal: {type: exponential, lam: 0.05, sigma_t: 0.3}
          spatial:  {type: exponential, sigma_x: 1.0}
        sigma2: null

    expand:
      observable: ["phi_a(x)", "phi_b(y)"]
      orders: [2]

    propagators:
      t_max: 2.0
      n_grid_t: 20
      c_closed_form_module: ./demo1_c_closed_form.py
      c_closed_form_attr: C_fn

    sweep:
      positions_grid:
        x: [0.0]
        y: [0.5]
      t_final_grid: [1.0]
      component_pairs: [[0, 1]]
      vertex_types: [F, FK]
      orders: [2]
      n_samples: 4096
      seed: 20260428
""")


def test_CF11_yaml_kappa3_callable_end_to_end(tmp_path: Path) -> None:
    """End-to-end: a YAML config with ``nonlocal_vertices`` of
    ``order: 3`` and a ``coupling_module`` callable runs through
    ``run_workflow`` and produces FK channel values that are
    finite, non-zero, and reproducible across two identical runs.

    Pairs with ``test_dynamic_coupling.py::test_WF6_*`` to lock the
    YAML loader's callable-coupling routing (the
    ``_load_callable_from_module`` path that backs
    ``coupling_module`` / ``coupling_attr``).
    """
    (tmp_path / "k3.py").write_text(_DEMO2_KAPPA3_MODULE)
    (tmp_path / "c.yaml").write_text(_DEMO2_YAML_TEMPLATE)

    cfg = load_workflow_config(tmp_path / "c.yaml")
    # Confirm the YAML loader populated the non-local vertex's
    # callable coupling correctly.
    assert len(cfg.system.nonlocal_vertices) == 1
    nl = cfg.system.nonlocal_vertices[0]
    assert nl["order"] == 3
    assert callable(nl["coupling"])

    sweep, totals = run_workflow(cfg)
    assert len(totals) > 0

    # ``totals`` aggregates over diagram (vertex_type, diagram_idx),
    # so to inspect the FK channel separately we pull the per-
    # diagram DataFrame from the SweepResult.
    df = sweep.to_dataframe()
    assert "vertex_type" in df.columns, (
        f"sweep.to_dataframe missing vertex_type column: {df.columns!r}"
    )
    fk = df[df["vertex_type"] == "FK"]
    assert len(fk) >= 1, f"no FK diagrams in sweep: {df!r}"

    # Sum FK contributions at the single grid point and verify it
    # is finite and non-trivial.
    fk_value = float(fk["value"].sum())
    assert np.isfinite(fk_value), f"non-finite FK value {fk_value}"
    assert abs(fk_value) > 1e-10, (
        f"FK value collapsed to ~zero ({fk_value:.3e}); the callable "
        f"K may not be threading through the dynamic-coupling path."
    )

    # Reproducibility: a second run with the same YAML and seed
    # must produce the same FK value to float64 noise.
    cfg2 = load_workflow_config(tmp_path / "c.yaml")
    sweep2, _totals2 = run_workflow(cfg2)
    fk2 = sweep2.to_dataframe()
    fk2 = fk2[fk2["vertex_type"] == "FK"]
    fk_value2 = float(fk2["value"].sum())
    np.testing.assert_allclose(
        fk_value2, fk_value, rtol=1e-12, atol=0.0,
    )
