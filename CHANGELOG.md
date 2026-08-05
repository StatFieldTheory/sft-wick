# Changelog

## Unreleased

> **Numbers move.** Several fixes below change results that were previously
> wrong. If you have pinned values, re-pin them against the corrected output
> rather than loosening tolerances — every change is documented with its
> measured size. The two that move the most output: the causal **lower**
> bounds (any observable with an external response leg) and the **C-table
> diagonal** fix (every tadpole, i.e. every interacting order).

### Added

- **`propagators_from_cache`** wraps a hand-built `PropagatorCache` as a
  `Propagators`, so the L1 workflow API can be driven by a cache that does not
  come from a `System` — which the spectral one does not, since its `R` and
  `C` come from a spectrum rather than the system's `kappa2`. Without it the
  whole workflow layer was closed to `sft_wick.spectral`.

- **`sft_wick.spectral` — disorder-averaged (spectral) propagators.**
  `SpectralDensity` + `spectral_cache(density, D, shift=...)` build a
  `PropagatorCache` whose `R` and `C` are superpositions of Ornstein-Uhlenbeck
  propagators over a spectrum `rho(h)`:

      R*(t,t')   = Theta(t-t') <exp(-h (t-t'))>_rho
      C*(t1,t2)  = <(D/h) (exp(-h|t1-t2|) - exp(-h(t1+t2-2 t_min)))>_rho

  This is the disorder-averaged limit: the effective single-site problem is
  scalar, so the `O(N^rank)` coupling-index contraction that dominates a
  per-instance matrix calculation disappears. `R*` is a superposition of
  decays, i.e. genuinely non-Markovian.

  with the initial condition `x(t_min) = 0`. A zero rate is admitted and takes
  the finite free-diffusion limit `2 D (min(t1,t2) - t_min)`; a negative rate
  is rejected, and round-off negatives from `eigvalsh` are clamped.

  Construct the density with `SpectralDensity.from_samples` (empirical spectra,
  reduced by equal-mass binning),
  `.from_callable` (Gauss-Legendre over an analytic density), or `.delta`
  (a single rate, which reduces every formula to the plain OU one — the
  boundary check the tests are built on).

  Both propagators are evaluated **exactly, never tabulated**. Tabulating `C`
  on a `(t1,t2)` grid and splining it reintroduces the diagonal ridge — `C`
  has a derivative jump of exactly `-2D` on `t1 == t2`, and every tadpole
  evaluates `C(s,s)` there. Measured on a sampled Marchenko-Pastur spectrum:
  a hand-rolled `RectBivariateSpline` gives 23% relative error on the diagonal
  and does not converge (0.2306 at n_grid=30, 0.2228 at 60), against 2.4e-04
  at 64 spectral nodes and 1.3e-05 at 256.

- **Two-time observables from the declarative API.** `Expansion.sweep` takes
  `external_times_grid={point: [times]}` — the same shape as `positions_grid`,
  swept as a further Cartesian axis — and `Expansion.evaluate` /
  `integrate_diagrams` take `external_times=`. The YAML `sweep:` block accepts
  `external_times_grid` too. Result rows gain a `t_<point>` column per named
  point, and `SweepResult.totals()` groups by them.

  The **response field is now nameable in an observable**: `"psi_b(y)"`
  alongside `"phi_a(x)"`. Without it there was no way to ask for `R(t,t')` at
  all — the parser accepted only the physical field.

  Together these make `R(t,t')` and `C(t,t')`, the DMFT order parameters,
  reachable declaratively. Previously every external was pinned at a single
  `t_final`, and since Θ kills an R joining two externals at the same time,
  *every* observable carrying a response leg came back identically 0.
  Omitting `external_times_grid` reproduces the old rows exactly.

- **`external_times={point: time}`** on `integrate_moment_{qmc,qmc_vectorized,
  gauss_legendre,nquad}`, the `integrate_moment` dispatcher,
  `_evaluate_zero_dimensional` and `integrate_two_point_qmc`. Every integrator
  previously pinned *all* fixed externals at a single `lambda_f`, so `R(t,t')`
  and `C(t,t')` were unreachable — and because Θ kills an R joining two
  externals at the same time, **every observable with an external response leg
  came back identically 0** at every order through all five backends.
  `external_times=None` reproduces the old numbers bit-for-bit.
- **Time-dependent (callable) *local* couplings.** The vertex's single
  spacetime point was discarded, so a callable local coupling was rejected.
- **`PropagatorCache(model, c_value_fn=...)`** — a public L0 hook for supplying
  C directly, without subclassing and overriding semi-private methods.

### Fixed

- **The C cache keyed ndarray arguments on `id()`.** `id()` is unique only
  among *live* objects, so a freed position array's address can be recycled and
  the cache then returns one separation's `C` for another. Now keyed on
  contents. I could not force such a collision in 200k attempts, so the
  practical reachability is low — but the key was wrong, and a value key is
  correct and cheap here.
- **`PropagatorCache.C_diagonal` was position-blind when a spatial table was
  present.** It consulted the legacy time-only table first while `C_value`
  consults the spatial one first, so the two accessors disagreed by ~38 % with
  both tables built. Same precedence in both now.
- `sft_wick.spectral`: `average()` rejected a transposed layout instead of
  contracting the wrong axis; `SpectralDensity` became comparable and hashable
  (the generated dataclass `__eq__` raised on ndarray fields);
  `C_diagonal_batch` appends the component axis rather than inserting it at
  position 1 (identical for 1-D times, wrong for 2-D); `clear_cache()` keeps
  the batch-C capability instead of demoting every backend to the scalar loop;
  and `noise_D` / `n_components` / `t_min` / `n_nodes` are validated.

- **Causal *lower* bounds from external response legs were dropped entirely.**
  Every bound-builder wrote `if earlier in int_vars: upper_bounds[...]`, which
  silently discards any ordering whose *earlier* endpoint is external — exactly
  the orderings an external ψ leg creates, since `<φ(u) ψ(y)> = R(u,y)` forces
  `t_u >= t_y`. The O(g) response was integrated over `[t_min, t_x]` instead of
  `[t_y, t_x]`: up to 5x wrong, and −11.25 instead of exactly 0 at `t_y = t_x`.
  Lower bounds are now also transitively closed along internal edges, and
  inverted intervals are clamped so `nquad` never integrates backwards (which
  returned a negative volume, sign-inverting the O(g²) response).
- **A swept external's causal bounds are now carried, not refused.** With
  `integrate_over`, a bound from a swept point is variable-valued. Dropping it
  is not a correctness error — Θ already zeroes the integrand there — but it is
  a quadrature error for a fixed-node rule, because Θ leaves a jump *inside*
  the domain: `gauss_legendre` measured **22.5 % wrong at the library default
  `n_gauss=8`**, converging only as O(1/n). Now exact at n=8. The same applies
  to orderings between two swept externals (**29 %** at n=8) and, since they
  can be mediated by an internal vertex or involve a fixed external, the whole
  ordering graph is closed transitively rather than by edge kind.
- **Retardation (Θ) is applied at diagram evaluation.** `R_time` is the raw
  Θ-stripped model accessor; the three consumers that multiply R factors now
  apply Θ with a strict `t_left > t_right` test. An R joining two *fixed*
  externals has no integration domain to enforce it, so an order-0 response
  returned the unbounded acausal `exp(+μ(t_y−t_x))`, and 1 instead of 0 at
  equal times.
- **The C builder contracted only `diag(R)`** instead of `R κ Rᵀ` — 57 %
  Frobenius error for a dense drift.
- **Eighteen complex→real projection sites** took `abs()` of a complex value or
  `.real` of an imaginary one, destroying the sign (or zeroing the value) of
  any diagram whose observable carries external response legs. All now route
  through a checked projection using the reality theorem: with
  `n_R = Σ_v n_ψ(v) + E_ψ`, a diagram equals `i^(−E_ψ) ×` a real number.
- **The C table did not converge on its diagonal.** `C(t1,t2)` integrates up to
  `min(t1,t2)`, so `∂C/∂t1` jumps by exactly `−σ²(t)` across `t1 == t2`; a
  tensor-product spline is C² and cannot represent that ridge. Measured
  mid-cell relative error **22.3 % at `n_grid=41`, still 21.4 % at 321**
  (absolute error O(h) against O(h⁴) off the diagonal) — and *every tadpole*
  evaluates `C(s,s)`, exactly on it. The grid's own `i == j` entries are now
  harvested into a separate interpolator, for the legacy table, the lazy
  spatial builder and the three full spatial grids. This **changes every
  interacting order**: `examples/demo1`'s YAML-workflow sweep moves in 42 of
  48 rows by up to 6.1e-3 relative. (The demo1 *notebook*, which produces the
  paper figures, is unaffected — it uses a closed-form analytical cache.)
- **`integrate_two_point_qmc` ignored spatial separation at order 0** and, above
  order 0, ignored a spatial C table in favour of the single-`t_ref` κ² ratio.
  The first returned the coincident-point value at *every* separation (a factor
  e² at r = 2σ); the second was 8.2 % off at order 2 for a kernel whose
  correlation length grows with time.
- `DiagramIntegrand.evaluate` silently returned 0 for a callable coupling, and
  a scalar-field coupling of the wrong rank raised an opaque `TypeError`.
- The reality projection ran *before* the negligibility guard on three
  prop-indexed sites, so a coupling entry that is float noise around zero
  raised instead of being skipped.

### Changed

- `ito=False` is documented as a **symbolic** switch. It keeps the equal-point
  R terms in the expression tree but does not change any number: Θ(0)=0 and the
  absent Stratonovich functional Jacobian cancel exactly, and for additive noise
  the Itô and Stratonovich answers coincide. Do **not** "fix" this by setting
  Θ(0)=1/2 without emitting the Jacobian — for the linear vertex that adds a
  spurious term growing without bound in the final time (measured 200 %/400 %/
  800 % of the exact answer at T = 4/8/16).
- `PropagatorCache.R_time`'s docstring is now the single authoritative
  statement of the Θ convention, which was previously described three
  inconsistent ways.

### Internal

- `PropagatorCache` stays picklable with a C table built: the diagonal
  interpolator serialises only its nodes and values, since
  `scipy.interpolate.CubicSpline` is not picklable on scipy ≥ 1.18. This keeps
  `propagators.cache_path`, `integrate_diagrams(n_jobs>1)` and
  `Expansion.sweep(n_jobs>1)` working.

### Fixed (details of the first item above)

- **`DiagramIntegrand.integration_bounds` returned wrong integration bounds.**
  `scipy.integrate.nquad` calls `ranges[i]` with the **outer** integration
  variables `int_vars[i+1:]`; the code indexed them as if they were the inner
  ones. Two silent failure modes followed: at 2 variables the bound collapsed
  to the literal `1.0`; from 3 variables the guard could *pass* and return the
  wrong outer variable. Every diagram containing a vertex-to-vertex response
  chain `R(y_i, y_j)` was affected (44/44 order-3 diagrams of the quartic
  Langevin model). The correct mapping — `later_args[index(src) - i - 1]` —
  was already present in `integrate_moment_nquad`; the public helper had never
  been migrated to it. **The five production integrators build their own
  bounds and were never affected**, so no supported computation changes.

- **The C-propagator builder used only the diagonal of a matrix `R_time`.**
  Both quadrature paths (`_C_value_direct`, `_C_value_direct_gl`) contracted
  `R[a,a] κ[a,b] R[b,b]` instead of the matrix triple product
  `C = ∫∫ R(t1,λ1) κ(λ1,λ2) R(t2,λ2)ᵀ`. That is correct only when the linear
  operator is diagonal in the chosen basis; for a dense drift (e.g.
  `A = H + λ` with `H = XᵀX/N`) it gave a 57 % Frobenius error, wrong signs on
  6 of 9 matrix elements, and exact zeros where the true value was nonzero.
  `iso_R=True` results are bit-for-bit unchanged.

- **`abs()` projections silently destroyed signs.** Four sites returned
  `result.real if result.imag == 0 else abs(result)`, and six more took
  `complex(coeff).real`, annihilating an imaginary coefficient. All thirteen
  now route through `_real_or_raise`, which applies the structural `i**E_psi`
  rotation and *raises* rather than guessing when the result is still not
  real. See the reality theorem on `DiagramTerm.observable_phase_factor`.
  Consequence: observables carrying external response legs (response
  functions) are now correct where the four backends previously disagreed by
  factors of `-1` and `0`; a mis-specified action now errors instead of
  returning a plausible wrong number.

- **`DiagramIntegrand.evaluate` silently returned 0 for a callable coupling.**
  It reads the static `coupling_array`, a zeros placeholder on the dynamic
  path. It now raises `NotImplementedError` naming the backends that do
  support callables.

- **A wrong-rank coupling for a scalar field raised an opaque `TypeError`.**
  It now raises a `ValueError` naming the symbol and the expected rank, and
  accepts a size-1 array of any shape.

### Added

- **Time-dependent (callable) *local* couplings.** A local vertex has exactly
  one spacetime argument, so evaluating its coupling there is unambiguous —
  but that point was discarded when building the coupling `Symbol`, so
  `build_integrand` had no coordinates to pass and refused every callable.
  The point is now recorded (`Symbol.spatial_args`, with `Symbol.local=True`
  suppressing it in LaTeX so rendering of constant couplings is unchanged).
  This is what expanding about a time-dependent mean-field trajectory
  requires: the vertices then carry couplings ∝ `wbar(t)^k`. Verified against
  a closed form to 3.7e-12, with the constant-callable path bit-identical to
  the ndarray path.

- **`PropagatorCache(..., c_value_fn=...)`** — a public L0 hook to supply `C`
  directly, bypassing the `∫∫ R κ R` construction. Necessary for any
  disorder-averaged (DMFT) propagator pair, where `⟨R κ R⟩ ≠ ⟨R⟩ κ ⟨R⟩` means
  no noise cumulant can reproduce the correct `C`. Previously this required
  subclassing and overriding `_C_value_direct`.

- **`DiagramTerm.n_external_response`** (`E_psi`) and
  **`DiagramTerm.observable_phase_factor()`** (`i**E_psi`), plus
  `DiagramIntegrand.expected_phase`, exposing the phase structure that makes
  the reality projection well posed.

- **`_collect_symbol_occurrences`** and a guard in `build_integrand`: a
  callable coupling occurring at genuinely different point sets (two copies of
  one vertex at order ≥ 2) is now refused instead of being evaluated at the
  first occurrence's coordinates, which was measured 4.06× wrong. Permutations
  of a single point set — the symmetrised non-local coupling sum — are still
  accepted, preserving the κ⁽ᵐ⁾ feature.

### Tests

- `tests/test_msr_numerics_regressions.py` — 17 tests pinning each of the
  above against independently derived ground truth (stationary Fokker-Planck
  series, the exactly-solvable `D/(mu+k)` model, closed-form matrix `C`).
  15 of them fail on the previous code.

## 0.2.0 — 2026-07-25

First release on PyPI (`pip install sft-wick`).

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
