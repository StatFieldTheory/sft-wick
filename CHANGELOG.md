# Changelog

## Unreleased

### Features

- **`NonLocalVertex(equal_time=True)`** — opt-in mode for non-local
  vertices whose `coupling` callable encodes an **equal-shell** /
  **equal-time** connected cumulant (one common cosmology use case:
  `canoes.sachs.compute_kappa3_zeta_table` which returns the
  equal-shell bispectrum `ζ_eq(γ_12, γ_23, γ_31; λ)` for the LSS
  matter / potential 3-point function).

  When `equal_time=True`, the `m` time legs of that vertex collapse
  into a **single** integration variable while the `m` spatial legs
  remain independent (so the callable still receives `m` distinct
  positions and only one shared time replicated `m` times in
  `t_list`).  This matches the action-level relation
  `κ^(m)(z_1,...,z_m) ≈ δ(λ_1−λ_2) … δ(λ_{m−1}−λ_m) · κ_eq(n_1,...,n_m; λ)`
  used in single-shell / Limber-like cosmological approximations.

  The default `equal_time=False` is unchanged — the full
  cross-spacetime cumulant `κ^(m)(z_1,...,z_m)` with `m` independent
  `(n, λ)` integrations is still the canonical contract.

  Failure mode this guards against: passing an equal-shell ζ as
  `coupling` without `equal_time=True` lets sft-wick faithfully
  sweep `m` independent times, contributing a spurious `(t_max)^(m−1)`
  factor of integration measure (e.g. `~5.6×10^6` for `m=3` and
  `t_max=2360` Mpc).  This was the diagnosis-of-record for the
  STF_lensing path-integral lensing Order-2 FK ⟨κκ⟩ bug.

  Implementation spans `workflow/specs.py`, `workflow/system.py`,
  `workflow/config.py`, `vertices.py`, `perturbation.py`, and
  `evaluate.py`.  Threaded through as
  `Vertex(equal_time=...)` → `VertexInstance.equal_time_aliases` →
  `DiagramTerm.equal_time_aliases` →
  `SpatialStructure.equal_time_aliases`, with time-alias resolution
  in `_times` of the QMC + Gauss-Legendre paths and in
  `DiagramIntegrand.evaluate`.  Diagram topology (R / C propagator
  routing, direction groups) is unchanged — only the time-integration
  Jacobian and the callable's `t_list` payload differ.

  YAML usage::

      nonlocal_vertices:
        - name: K
          order: 3
          coupling_module: kappa3_callable.py
          coupling_attr: coupling_fn
          equal_time: true

  See `docs/user_guide/workflow.rst` for the full discussion and
  `docs/notes/equal_time_nonlocal_vertex.md` for the design note.
  Regression coverage: `tests/test_equal_time_nonlocal.py`
  (11 tests covering alias plumbing, static evaluation, dynamic
  callables, and Jacobian ratio).

- **`NonLocalVertex(already_R_contracted=True)`** — opt-in mode where
  the user-supplied callable returns the **R-contracted** form
  `κ^(m)_R(γ; z_1', …, z_m') := ∫ ∏ R(z_i', z_i) · κ^(m) dz_1…dz_m`
  evaluated at the **partner** (outer) spacetime points rather than
  the κ's own legs. When set, sft-wick

  * tags the m R-propagators attached to this vertex's ψ legs as
    **absorbed** — they remain in the diagram graph (so direction
    groups continue to identify the leg with its partner) but
    contribute a factor of 1 instead of the usual `R_time` value;
  * aliases each leg's time onto its Wick partner's time via the
    existing `equal_time_aliases` machinery, so the m leg-time
    integration variables drop out of the simplex;
  * feeds the callable `(n_list, t_list)` with the **partner**
    coordinates instead of the leg's own.

  Cuts integration dimensionality by `m` per such vertex (e.g. an
  order-2 F+K diagram becomes 1-D after absorbing K's 3 legs). The
  surviving integrand is also **smoother** because the narrow-kernel
  peak has been folded into κ^(m)_R, so Gauss-Legendre converges
  exponentially with far fewer nodes. Motivating use case: canoes'
  squeezed κ³ at `ℓ_max = 5000, z_s = 1100`, where the raw kernel
  has diagonal width `Δχ ~ 2 Mpc/h` and the per-leg χ-integration
  would otherwise demand `~10^11` quadrature points.

  Mutually exclusive with `equal_time=True` (rejected at
  construction). Default `already_R_contracted=False` is unchanged.

  Implementation spans `workflow/specs.py`, `workflow/system.py`,
  `workflow/config.py`, `vertices.py`, `perturbation.py`, and
  `evaluate.py`. Threaded through as
  `Vertex(already_R_contracted=...)` →
  `DiagramTerm.r_absorbed_pairs` (collected per-routing in
  `_collect_r_absorbed_pairs`, merged into `equal_time_aliases`) →
  `SpatialStructure.r_absorbed_pairs`, with R-factor skipping
  centralised in `evaluate.py::_kept_r_propagators` (used by all
  five R-product sites: scalar / matrix per-sample, QMC-vectorised,
  GL tensor-product, integrate-over external).

  YAML usage::

      nonlocal_vertices:
        - name: K
          order: 3
          coupling_module: kappa3_R_callable.py
          coupling_attr: coupling_fn
          already_R_contracted: true

  Companion utility `sft_wick.build_R_contracted_callable` wraps a
  raw κ^(m) + R kernel into the R-contracted form via brute-force
  trapezoid quadrature on a user χ-grid — for validation comparand
  use; production callers should supply the analytical / pre-
  tabulated κ^(m)_R directly (e.g. canoes' FFTlog-of-W chain).

  See `docs/user_guide/workflow.rst` and
  `docs/notes/R_contracted_nonlocal_vertex.md` for full
  documentation. Regression coverage:
  `tests/test_R_contracted_vertex.py` (15 tests including the
  machine-precision constant-κ³ equivalence at `rtol=1e-12` and the
  four-way per-sample × vectorised contract check).

### Performance

- **Test-suite wall time cut ~3.5×** (from ~18 min to ~5 min for the
  full 275-test suite on M-series).  The profile showed two tests
  accounting for 80 % of runtime — both were building expensive
  C-propagator caches that the test's tolerance did not actually
  require:
  - `test_CF3_full_run_matches_L1_reference` (585 s → 72 s): this is
    an equivalence test (YAML pipeline vs. direct Python — rtol=1e-10
    is bit-identity), so the propagator grid only needs internal
    consistency, not physical precision.  `n_grid_t` 30 → 12,
    `t_max` 3.0 → 2.0, `n_samples` 1024 → 256.
  - `test_T4_time_dependent_gamma_end_to_end` (302 s → 24 s): an
    order-0 smoke test whose single C-evaluation did not justify a
    30² dblquad grid.  `n_grid_t` 30 → 10, `n_grid_cache` 120 → 40,
    `t_max` 2.0 → 1.5.
  - `test_W3_white_noise_absorbed_into_translation_spline`
    (97 s → 32 s): lazy-spline cache cost dominated — `n_grid_t`
    25 → 15 (quadratic win); `n_samples` 2¹³ → 2¹¹ (negligible since
    QMC is not the bottleneck here).
  - `test_spline_matches_split_dblquad` (77 s → 54 s):
    `n_grid` 30 → 25 with worst-case-O(h⁴) tolerance relaxed from
    2e-3 → 3e-3 (still within 2× the theoretical bound).
  - `test_WF4_end_to_end_matches_validate_phase5` (44 s → 18 s): the
    sweep covered 4 × 2 grid points but the reference values only
    pinned 3 × 1 of them — dropped `y=2.5` and `t_final=1.0`.

- **`integrate_diagrams` default `n_jobs` flipped from `1` to `-1`.**
  The guard `len(diagram_terms) <= 2 → sequential fallback` already
  prevents joblib's ~1 s startup from dominating on trivial batches,
  so the new default is a free speedup for any order-2+ batch
  evaluation.  Explicit `n_jobs=1` callers (determinism-comparison
  tests like C5) are unaffected.

### Cleanups

- Removed **dead `System.explicit_C` field** (declared but never read
  anywhere in the codebase; replaced by the `c_closed_form` kwarg on
  `System.propagators()` and `Propagators.build()`).  Docstring for
  `System.explicit_R` now points to `ExplicitR` as the preferred
  structured alternative.
- Removed 10 unused imports flagged by ruff F401 across
  `diagrams.py`, `drawing.py`, `evaluate.py`, `perturbation.py`,
  `vertices.py`, `workflow/expansion.py`, `workflow/result.py`,
  `workflow/specs.py`.
- Removed 3 orphaned local assignments flagged by F841
  (`evaluate.py:ids`, `perturbation.py:sum_idx_dims`,
  `simplify.py:ref_flipped`).

### Features

- **Time-dependent linear operator** (`workflow/specs.py::DiagonalA`).
  `gamma` may now be a callable ``γ(t) -> np.ndarray(shape=(N,))``;
  the wrapper pre-computes ``Γ_a(t) = ∫_0^t γ_a(τ) dτ`` on a grid
  and caches it as a cubic spline for O(1) R lookups.  Diagonal
  case only — full-matrix time-dependent A still requires
  `ExplicitR` with a user-supplied time-ordered matrix exponential.
  New tests in `tests/test_diagonal_A_time_dependent.py`.

- **Dynamic coupling evaluation for non-local vertices**
  (`evaluate.py::DynamicCouplingPromise` +
  `perturbation.py::DiagramTerm.build_integrand`).  `coupling_values`
  now accepts callables (`fn(n_list, t_list) -> (N,)*m tensor`) for
  spacetime-dependent κ^{(n)}.  Demo2's FK channel is now a one-line
  `expansion.evaluate(vertex_types={'FK'})` — no bespoke integrator.
  See `examples/demo2/validate_FK_dynamic.py`.

- **Observable convention kwarg** `integrate_over` on
  `integrate_moment` / `Expansion.evaluate` / `.sweep` — per-external
  control over fixed-time (`None`, default, matches ⟨φ(t_f) φ(t_f)⟩
  physics convention) vs time-integrated (`"all"` or a subset).
  The previous default was time-integrated; callers that want that
  now pass `integrate_over="all"` explicitly.

- **L2 — YAML config + CLI**.  `sft-wick run config.yaml` executes
  an end-to-end workflow (expand → propagators → sweep → output);
  `--override key=value` patches fields without editing the file.
  `c_closed_form_module` and vertex `coupling_module` fields wire
  user Python modules in.  Full round-trip tests in
  `tests/test_workflow_config.py` (CF1–CF5).  Example configs:
  `examples/demo1_config.yaml`, `examples/demo2_config.yaml`.

- **L1 — high-level workflow API**: `System` (physics spec),
  `Expansion` (diagram-level inspection + per-channel integration),
  `Propagators` (thin cache wrapper), `Result` / `SweepResult`
  (pandas-backed structured output).  Bare F/κ^{(n)} tensors —
  wrapper applies MSR factors `-(i^n)/n!` automatically.
  `expansion.by_vertex_type(order)` classifies diagrams by
  F/K composition (demo2-style FF/FK split).  New tests
  `tests/test_workflow.py` (WF1–WF5).

- **Parallelised lazy propagator spline build** (`evaluate.py`).
  `_LazyTimeSplineCache._build` now uses `joblib.Parallel` when
  `n_jobs != 1` (previously silently ignored for lazy mode).
  `examples/demo1/validate_R_C_derivation.py` went from 400 s → 39 s
  on M-series (10.3× speedup) with the `n_jobs=-1` plumbing.

### Bug fixes

- **`DiagramIntegrand.evaluate` scalar path now honours
  `fixed_indices`** (evaluate.py:1322, 1424).  Previously
  `_resolve_component(il, {})` was passed `{}` instead of
  `self.fixed_indices`, so observable component labels (`'a'`,
  `'b'`) failed to resolve and the code summed over all
  components — an N-factor overcounting at order 0.  The
  vectorised path already handled `fi` correctly; this brings the
  scalar path in sync.

### Bug fixes

- **Fixed incorrect C-propagator component resolution in numerical integration**
  (`evaluate.py`).  When `build_integrand()` was called with `fixed_indices`
  (e.g. `{'a': 1, 'b': 1}` for selecting a specific correlator component),
  the fixed index names (like `'b'`) were not passed through to the QMC/GL
  integration routines.  `_resolve_component('b', idx_map)` failed to
  resolve the index, causing the integrator to **sum over all field
  components** instead of selecting the fixed one.  For `N_comp = 2` with
  `iso_C`, this produced a factor-of-2 overcounting per unresolved
  C-propagator index, making perturbative corrections ~1.3–1.8× too large
  depending on the diagram topology.

  **Fix:** `DiagramIntegrand` now stores the `fixed_indices` dict and
  both `integrate_two_point_qmc` and `integrate_moment_qmc_vectorized`
  merge it into the index map before calling `_resolve_component`.

  Affected files:
  - `sft_wick/evaluate.py` — added `fixed_indices` field to
    `DiagramIntegrand`; updated `integrate_two_point_qmc` and
    `integrate_moment_qmc_vectorized` to use `ig.fixed_indices`.
  - `sft_wick/perturbation.py` — `build_integrand()` now passes
    `fixed_indices` to the `DiagramIntegrand` constructor.

- **Fixed coupling symmetry bug in `compute_moment_numerical`**
  (`perturbation.py`).  The canonical grouping of component routings
  reused reference propagators and static coupling symbols, causing
  asymmetric coupling tensors (e.g. `F[1,0,1]=1, F[1,1,0]=0`) to
  produce wrong results.  Symmetric tensors were unaffected.

  **Fix:** Each component routing now produces its own `DiagramTerm`
  with routing-specific propagators.  This removes the incorrect
  canonical grouping but increases the number of diagram terms.
