# Changelog

## Unreleased

### Propagator-indexed dynamic couplings (demo2's order-4 F³κ³)

- **`DynamicCouplingPromise.evaluate_at_batch` no longer refuses a
  contraction that leaves a propagator index.**  A callable κ^(m) whose
  contraction against the surrounding Wick structure leaves a surviving
  component index on a C propagator used to raise
  `NotImplementedError: Dynamic coupling with propagator-indexed
  contraction is not yet supported`.  It now returns
  `(n_samples,) + prop_shape`, and the integrators contract it against
  the C-propagator product one index assignment at a time — the same
  `np.ndindex(prop_shape)` loop the static-tensor branch has always
  used, with the scalar coefficient promoted to a per-sample array.
  The contraction is centralised in
  `DiagramIntegrand._dynamic_values`, called from all three consumers
  (QMC-vectorised, Gauss-Legendre, and the zero-dimensional path); the
  batched C lookup is hoisted out of the index loop, since it does not
  depend on the component assignment.

  This is what blocked the exact evaluation of demo2's order-4 F³κ³
  channel: 30 diagrams, of which those with `propagator_indices ==
  (('i_0', 2),)` hit the guard.

  **Size of the change:** zero on every path that worked before — a
  fully contracted (scalar) callable takes the same branch and returns
  bit-identical numbers; the 236 tests of the integrator suite are
  unchanged.  What was previously an exception is now a number.

  Locked by `DC1` in `tests/test_dynamic_coupling.py`, which replaces
  the test that pinned the `NotImplementedError`: a callable that
  ignores its arguments and returns a constant tensor is
  mathematically the static tensor, so the two routes are compared
  diagram-by-diagram on the order-4 F³κ³ set and agree to `rel=1e-12`
  (observed: exact equality).  A companion test asserts the comparison
  is not vacuous — at least one prop-indexed diagram integrates to a
  non-negligible value.

### Breaking: external operators may no longer share a spatial label

`("phi_a(x)", "phi_b(x)")` — the natural spelling of an equal-point
correlator — now raises `ValueError` at both `System.expand` (L1) and
`compute_moment` (L0), **at interacting orders only**.  It previously
returned a number, correct at order 0 and wrong once vertices are
present.

**Order 0 is exempt, and the exemption is measured rather than
assumed.**  The free-theory contraction keeps every routing at
coincident labels — `⟨φ_a φ_b φ_c φ_d⟩` gives
`C_ab C_cd + C_ac C_bd + C_ad C_bc` either way, and the scalar
`⟨ψφφφ⟩` gives `3 R(x,x) C(x,x)` against the distinct-label
`R(x,w)C(y,z) + R(y,w)C(x,z) + R(z,w)C(x,y)`.  The loss appears only
once vertices exist, when the downstream diagram-isomorphism pass merges
topologies that the external collapse has made isomorphic.  The
equal-point Itô tests depend on the exemption and are unchanged; a new
test pins it (`CE2`).

The refusal at order ≥ 1 is deliberately conservative — demo2's order-2
`FK` channel is *right* at coincident labels while its `F` channel is
low by 2, and distinguishing them per-channel is exactly the analysis
this refusal exists to avoid.  One test moved: `test_latex_local_coupling`
used a repeated label at order 1 while testing symbol flags, and now
uses distinct ones.

**Migration:** give each external a distinct label and set them to the
same point through `positions`:

```python
system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2])
expansion.evaluate(props, positions={"x": 0.0, "y": 0.0}, ...)
```

Coincident external *points* are, and always were, fully supported;
it is the shared *label* that is not.  Demo 1, demo 2 and the paper all
use distinct labels, so nothing in this repository changes.

**What was wrong.**  The spatial Wick engine enumerates topologies keyed
by spatial label and recovers the operator-level count with a
multiplicity that excludes observable points, assuming one operator
each.  Measured on demo2's system at t = 3.48 with both externals at
position 0: order 0 agreed exactly, the order-2 `F` channel was **low by
a factor 2** (1.838e-04 against 3.677e-04), and the order-2 `FK` channel
was exactly right.  Not even uniform between channels.

**Why it is refused rather than repaired.**  What the collapse loses is
not a factor but a *sum over the assignments of external operators to
legs*, each with its own component-index routing:

```
("phi_a(x)", "phi_b(y)", "phi_c(z)") -> K_abc + K_acb + K_bac + K_bca + K_cab + K_cba
("phi_a(x)", "phi_b(x)", "phi_c(x)") -> K_abc
```

`6 · K_abc` reproduces that sum only for a K symmetric under all six
permutations.  demo2's κ³ is component-diagonal and the two-point F case
had `a = b`, which is why the observed discrepancies were clean integers
and why the bug survived — a multiplicity "fix" would have replaced a
silent wrong answer with a subtler one, since the API accepts arbitrary
bare tensors.  Supporting the spelling properly means keeping external
operators distinguishable through the label-keyed enumeration in
`wick.wick_contract_spatial`; that is a change to the multiplicity core
and is left as follow-up work.  The reasoning is recorded in
`tests/test_coincident_external_labels.py` so it is not "optimised" back
later.

### Fixed: single-component systems with a callable coupling

`n_components = 1` plus a callable (dynamic) coupling raised
`ValueError: input operand has more dimensions than allowed by the axis
remapping` on every integrator.  With one component there is nothing to
sum over, so the simplifier elides every component index and
`_eval_symbolic_batched` took its `not expr.indices` branch, returning
the coupling array whole — `(n_samples, 1, 1, 1)` where `(n_samples,)`
was expected.  The scalar `_eval_symbolic` had always tolerated "a
size-1 array in any shape"; the batched path now does too.  A static
tensor at `N = 1` was unaffected, as was `N ≥ 2`.  Locked by `DC3`
(both integrators, dynamic against static to `rel=1e-12`).

**Verification that neither of the two changes above moved a published
number.**  Both are expected to be exactly inert for demo 1: one adds a
refusal on a spelling demo 1 does not use, the other repairs an `N = 1`
branch demo 1 never enters.  Measured rather than assumed —
`examples/demo1/L2/config.yaml` re-run before and after, all
**2016 / 2016** `(y, t_final, a, b, order)` totals **bit-identical**,
max relative change **0.0** against the v0.3.0 comparison's 3.3e-14
floating-point baseline.

### Demo 2 hardening

Every item below is a number that moved, with its measured size.

- **The order-4 F³κ³ channel of demo 2 is now computed, not estimated.**
  It was the blocked case above.  With the R-contracted κ³ the effective
  time dimension of its 30 diagrams is 3 (it is 6 with the raw kernel —
  measured, via `analyze_spatial().time_integration_vars`), so a
  tensor-product Gauss-Legendre rule does it in seconds per grid point.

  The equal-time estimate it replaces was built by collapsing κ³ to a
  constant `24 α λ² σ_t²` and calibrating on FK at order 2.  That
  calibration ratio is 0.42–0.64 for the FK-type partner-time
  configuration `(t′, s, s)` but 1.08–1.50 for three distinct partner
  times — which is what the F³κ³ diagrams have — so it was a
  factor-of-≈2 quantity.

  **Measured effect.**  At `t = 15, r = 0`, `ξ₀₁`: the exact channel is
  **5.47e-05** (GL10; 5.41e-05 at GL14) against the old estimate's
  8.10e-05.  Over the 18 times at `r = 0` this takes χ² of
  (simulation − theory) from **340.7 to 44.2**, the mean pull from
  **+3.31 to +1.09**, and the largest residual from **9.36e-05 to
  3.88e-05** (7.0σ → 2.9σ).

  **It also contributes where the 0.3.0 budget assumed it could not.**
  `F³κ³` was taken to vanish for `ξ₀₀` and `ξ₁₁` by `φ₁ → −φ₁` parity.
  That parity is broken by the deformation itself (`η̃ = η + α(η² − λ)`
  is not odd in `η` — which is the entire reason `ξ₀₁ ≠ 0`); the order-2
  FK channel does vanish there, but for the narrower reason that its two
  diagrams' index structure does.  Computed for all three pairs rather
  than assumed: **4.53e-06 for `ξ₀₀`** and 4.34e-06 for `ξ₁₁` at
  `t = 5.44` (converged to 0.1 %), i.e. 13 % of the `ξ₀₀` residual.

- **The residual's perturbative order is now measured, not asserted.**
  Scaling the quadratic drift by `s` scales each channel by a known
  power of `s`.  At `t = 15, r = 0`, all at dt = 0.02 so the step-size
  bias cancels out of the `s`-dependence: `s = 0.5` (20M realisations)
  gives a residual of 9.83e-06 ± 2.74e-06 and `s = 1` (2M) gives
  7.28e-05 ± 1.00e-05.  Those fit a **pure `s³` law with χ² = 0.06 for
  1 dof** — `c₃ = 7.38e-05 ± 0.91e-05` — so the residual *is* an order-4
  effect, deduced from the simulation and the order-2 channel alone.
  The exactly-computed `F³κ³` then agrees at **2.1σ** (1.8σ against the
  un-extrapolated dt = 0.02 residual; the spread between the two is
  whether the step-size bias is extrapolated away, and that bias is
  itself only a 1.1σ measurement).

  `s = 1.5` is excluded from the fit and kept as a measured boundary of
  validity: its residual *exceeds* the leading term (1.04e-03 against
  5.16e-04) and **4.5 % of trajectories blow up** against 6e-05 at
  `s = 1` — a 780× jump — so the reported mean is conditioned on
  survivors with the largest excursions removed.  At `s = 0.5` the
  blow-up fraction is **zero in 20 million realisations**, so the
  cleaner of the two fit points carries no conditioning at all.

- **`κ⁽⁵⁾` is excluded as the home of the remaining 2σ.**  The cumulant
  ladder gives `κ⁽⁵⁾ = 7.1 %` of `κ⁽³⁾`, and the lowest `κ⁽⁵⁾` channel
  enters at the same perturbative order with the same `s³` scaling — but
  through **6 diagrams against 30** (both enumerated), putting it at
  ~1e-06, an order of magnitude below the residual's uncertainty.
  Estimated rather than computed: a 5-leg R-contracted kernel does not
  exist, and per-diagram magnitudes and the `−iᵐ/m!` convention differ
  between `m = 3` and `m = 5`.

- **FFFF was Sobol QMC and is now Gauss-Legendre.**  At 32768 samples
  `FFFF_00(t = 15, r = 0)` gave 7.47, 3.22, 2.80, 5.05e-05 across four
  seeds — 46 % scatter on a mean of 4.6e-05.  Against the converged
  GL14 the QMC column was wrong by up to **4.84e-05 absolute (121 %)**,
  as large as the residuals it was used to interpret.  GL10 vs GL14 is
  **2.20e-06** absolute, well below them.

- **The full budget** is rewritten in
  `examples/paper_assets/demo2_kappa4/budget.md`; the interpretation, in
  a fixed five-part shape, is new at `examples/demo2/INTERPRETATION.md`.

- **`examples/demo2/k3_R_coupling.py` claimed "Accuracy ~1e-6 relative".
  It is 1e-4.**  1.7e-6 is what the FK-type configuration `(t′, s, s)`
  achieves — the only one it had ever been checked at.  Measured against
  cusp-aware 3-D adaptive quadrature of the raw leg integral: 1.4e-4 at
  coincident and at three-distinct partner times, 5.4e-4 at very short
  times, 2.6e-3 at `t′ = 0.1` (where the kernel is 100× below its
  plateau, so 1.7e-8 absolute).  The README's 1e-4 was right.  The
  docstring now carries the measured table and the
  `already_R_contracted` contract's silent assumptions.

- **`k4_R_contracted.py`'s "~1e-3"** holds over the range that matters
  (2.6e-4 to 2.2e-3) and degrades to 9.9e-3 at `t′ = 0.2`, same cause.
  Its `t′ = 50` row cannot be checked by QMC at all — a Sobol rule on
  the `50⁴` box has 80 % seed scatter — so what is checked there instead
  is that `K_R` has saturated, and it has.

- **`examples/demo2/L2/reproduce_figures.py` plotted the wrong error
  bars.**  `mc_err = sqrt(ξ²/n)` is the error on a mean of ξ, not on a
  mean of per-realisation products; the latter has variance
  `<φ_a²><φ_b²> + ξ²`.  Measured against the 2M-realisation runs that do
  accumulate sums of squares, the old formula is **0.029 of the true SEM
  for ξ₀₁ (low by 34×)** and 0.48–0.63 for ξ₀₀.  Replaced by the
  Isserlis estimate, which is 0.87–1.00 of the measured SEM over
  `t ∈ [0.6, 50]`.

- **Off-grid separations biased the simulation high.**  The simulation
  measures on a grid of pitch `σ_x/5 = 0.2` and reports off-grid `r` by
  `np.interp`; on a convex profile that overestimates by +0.36 % at
  r = 0.25, +0.50 % at r = 0.5, +0.39 % at r = 0.75.  The "+3.7σ" rows of
  the ξ₀₀ r = 0.5 tables were this, not physics.  The budget now
  evaluates the theory at the simulation's own grid sites and combines
  them with the same weights, so an off-grid row is compared like for
  like, and reports a genuine grid site (r = 0.4) alongside.
  `sim_dt_study.py` additionally stores the un-interpolated grid profile
  (`x_grid`, `xi_sites`, `xi_err_sites`) so future runs need no
  correction.

- **`sim_dt_study.py` gained `--F_scale`** (multiplies the quadratic
  drift only) and explicit blow-up accounting (`n_attempted`,
  `n_blown`).  This is the discriminating experiment for the residual's
  perturbative order: each channel scales as a known power of the
  coupling amplitude, so `residual(s) = c₃s³ + c₅s⁵` and `c₃` is
  directly comparable with the computed F³κ³ without assuming the
  order-4 calculation is right.  See
  `examples/paper_assets/demo2_kappa4/fscale_fit.py`.

- **`tests/test_demo2_kernels.py` is new** — nothing in `tests/` pinned
  a demo-2 number before.  `k3_R` and `k4_R` against quadrature of the
  raw leg integral at coincident / FK-type / split / short / long
  partner times, with tolerances taken from the measurements above; the
  raw-κ³-vs-`already_R_contracted` boundary test the feature never had
  on a NON-constant kernel (`test_R_contracted_vertex.py` only ever
  compared them on a constant κ³, where the leg integral factorises and
  any time-structure-dependent aliasing bug cancels); the
  `already_R_contracted` contract's silent assumptions; one pinned FK
  value and one pinned order-0 value; and the single-site cumulant
  ladder against its generating function.

- **The cumulant ladder is a truncation, and now says so.**
  `κ_n = n! [u^n/(2n) + (λ/2)u^{n-2}]` with `u = 2αλ` reproduces demo2's
  hand-derived κ², κ³ and κ⁴ exactly and continues: κ⁵ is **7.1 %** of
  κ³ and κ⁶ is 2.6 %.  demo2's action stops at κ⁴, so the lowest κ⁵
  channel (F³κ⁵) sits at the same perturbative order as F³κ³ and scales
  the same way in the coupling amplitude.  It is therefore inside the
  residual and inside the fitted `c₃`.  This was nowhere in the 0.3.0
  budget.

### Fixed: the `auto` C-quadrature choice was decided by a wall-clock race

`PropagatorCache._gl_is_cheaper` timed one Gauss-Legendre call against
one `dblquad` call and took the winner.  Both rules are verified
converged by `select_gl_node_count` before it ran, so the race could
never produce a wrong value — but it made the *choice* depend on machine
load, and with it `Propagators.c_source` and the spline table.  Two runs
of the same config on a busy and an idle machine could resolve
differently, which is a poor property for a package that sells
reproducibility; it also made
`test_direct_calls_before_any_build_use_dblquad_under_auto` fail
intermittently (observed failing in a full-suite run made while a
28-core sweep was running, passing in isolation and on a quiet machine).

The decision is now a function of the inputs alone: prefer
Gauss-Legendre whenever a converged node count exists, fall back to
`dblquad` only when none does.

**Size of the change:** bounded by the agreement of the two rules at the
node count actually resolved, since that is all the race ever chose
between — GL vs dblquad at the deep corner: **9.1e-16** at `t_max = 3`
(n = 20), **1.2e-10** at 15 (n = 20), **2.1e-09** at 50 (n = 30),
**3.6e-09** at 100 (n = 45).  All inside the 1e-8 selection tolerance.
No demo can be affected at all: they set `c_closed_form_only`, which
short-circuits before this path.  Verified by running the previously
flaky test five times under 24-way CPU load (5/5 pass) and by the 87
tests of `test_propagator_dispatch.py`,
`test_closed_form_dispatch_boundaries.py`,
`test_c_propagator_gauss_legendre.py` and `test_deductive_numerics.py`.

**And it is faster, not slower.**  Per `_C_value_direct` call at the
deep corner on the demo1 kernel, comparing the two rules *at the node
count `auto` resolves to* (best of 5, one core): 5.7 vs 7.1 ms at
`t_max = 5` (n = 20), 5.7 vs 39.5 at 15, 12.5 vs 88.0 at 30 (n = 30),
12.5 vs 144.4 at 50, 27.9 vs 209.2 at 100 (n = 45) — Gauss-Legendre
wins by **1.3× to 11.6×**, because adaptive cost grows with the interval
while a fixed-`n` tensor rule is flat and the required `n` grows only
slowly.  The one regime the race ever chose `dblquad` for is a smooth
kernel over a short horizon, where unconditional GL costs ~1.3×.  If
that ever matters it should return as a *threshold on the converged n*,
not a timing measurement.

### Documented: unstable cache keys for callable spec fields

`cache_path` keys the symbolic expansion and the propagator table on
`repr()` of the spec, and `repr()` of a plain function embeds its memory
address — so a callable passed to `CustomKernel`, `GeneralKappa2`,
`CustomImpulse`, `ExplicitR` or `DiagonalA.gamma` changes the key every
process and the cache never hits.  A module-level `def` is affected
exactly as much as a `lambda`; what fixes it is a small frozen dataclass
with `__call__`, whose `repr` is its field values (and which pickles
cleanly for joblib besides).

No behaviour change: an unstable `repr` can only cause a cache **miss**,
never a wrong hit, so the current hashing is fail-safe and is left
alone.  Keying on `(module, qualname)` would make it stable at the cost
of introducing the one failure mode that does not exist today — two
functions sharing a qualname colliding onto a *wrong* cached expansion —
so it is documented rather than "fixed".  **Vertex couplings are
unaffected**: they never enter the key, correctly, since the symbolic
enumeration does not depend on coupling values.  Noted on
`CustomKernel`.

### Demo 1: the order-4 channel was resting on one QMC seed

- At `(0,0), r = 0.5, t = 15`, order 4 across six Sobol seeds gives
  3.93, 4.17, 1.93, 1.56, 4.07, 2.45e-5 — **39 % scatter**.  The
  published value is the `seed = 42` draw, 3.93e-5; the converged
  Gauss-Legendre value is **2.330e-5** (GL14 vs GL18 agree to 0.15 %),
  so the figure's order-4 curve was **69 % high** there, and 74 % high
  at `r = 0`.  Order 4 is 10.0 % of order 2 at t = 15, not the 17.3 %
  the published draw implies.  The fixed-seed regression check could not
  see any of this.

- At `t = 100` **neither** method is converged: QMC scatters 134 % and
  the tensor-product rule still moves 12 % between GL14 and GL18,
  because the integrand lives within `~1/γ` and `~σ_t` of the upper
  corner of a simplex of size `t_f`.  Both channels have saturated well
  before then, which is what makes the late-time points usable at all.

- **Order 2 was not converged either, at the largest times** — this was
  not in the brief and is easy to miss, because order 2 is a 2-D
  integrand and "2-D integrals are fine" is the natural assumption.  At
  `t = 100, r = 2.5, (1,1)` the converged value is **3.10410e-05**
  (GL64, GL96, GL128 and 2²² Sobol agree to 5 digits); the published
  QMC 2¹⁵ seed-42 draw gives 3.72854e-05, **+20 %**, and scatters 13 %
  across seeds.

- **The sweep is now `gauss_legendre` with `n_gauss: 24`.**  The node
  count matters and the obvious choice is wrong: order 4 costs `n⁴`, so
  the temptation is to keep `n` small — but it is *order 2* that needs
  the finer grid at large `t_f`, and `n = 14` gives it only 196 nodes.
  Its order-2 error at `t = 100` is **cell-dependent and non-monotone**:
  −14.8 % at `r = 2.5, (1,1)`, +11.9 % at `r = 0, (0,0)`, +3.1 % at
  `r = 0.5, (0,0)`, −2.2 % at `r = 2.5, (0,0)` (each against GL64).  So
  `n = 14` would have traded an order-4 error for an order-2 one of
  either sign.  `n = 24` beats the QMC it
  replaces on both orders at every grid point; worst-case residual
  error **+1.3 % (order 2) and +2 % (order 4) at `t = 100`**, below
  0.1 % by `t = 15`.  Cost ~15 min on 28 cores against ~2 min, paid
  once, for a deterministic answer with a measurable error.

- **Effect on the published curve**, measured over all 2016
  `(y, t_final, a, b, order)` totals: summed over orders, the plotted
  `xi_ab(r, t)` moves by at most **1.67 %** (median 7.8e-06).  Per
  order: order 0 unchanged to machine precision (max rel 0.00 %),
  order 2 up to 15.7 % (7.9e-05 absolute, at `t = 100`), order 4 up to
  5.0e-05 absolute — the largest order-4 point,
  `t = 23.7, r = 0, (0,0)`, goes from **8.996e-05 to 3.972e-05**.  Full
  convergence tables in `examples/demo1/L2/INTEGRATION_ERROR.md`.

## 0.3.0 — 2026-09-02

> **This is the version the CPC paper (arXiv:2606.19480, revised) refers
> to.**  v0.2.0 is what the referees ran; everything below is what changed
> for the revision, plus the fixes that had accumulated on `main` since
> July.

> **Numbers move.** Several fixes below change results that were previously
> wrong. If you have pinned values, re-pin them against the corrected output
> rather than loosening tolerances — every change is documented with its
> measured size. The two that move the most output: the causal **lower**
> bounds (any observable with an external response leg) and the **C-table
> diagonal** fix (every tadpole, i.e. every interacting order).

### Referee revision (CPC round 1): a fast default path, progress, a quick start

The CPC referee ran the README quick start on a laptop and got no output
within an hour.  Every number in this section is measured on the same
machine (Apple M3 Ultra, one core unless stated); the "laptop proxy" rows
are the same runs under `taskpolicy -c background` with
`OMP_NUM_THREADS=1`.

- **Built-in closed-form C** (`sft_wick.workflow.closed_forms`).  For a
  diagonal constant drift driven by separable translation-invariant noise
  with an exponential temporal kernel — demo1, demo2 and the README family —
  `C = ∫∫ R κ² R` is now evaluated analytically, with an optional constant
  white-noise impulse and per-component `γ`.  `System.propagators()` and the
  YAML key `propagators.c_closed_form: auto` (the default) select it
  automatically; `null` forces quadrature and a `c_closed_form_module` still
  overrides it.  The formula is written with non-positive exponents only, so
  it does not overflow at `γ t ≳ 350` where the textbook `exp(2γt)·exp(−γ(t₁+t₂))`
  form does, and the removable singularities at `γ = 1/σ_t` and `γ = 0` are
  evaluated through `expm1`.  `examples/demo1_config.yaml`,
  `examples/demo2_config.yaml` and the demo L2 configs no longer need the
  `c_closed_form_module` hook (the module files stay as references).

  Validated at the dispatch boundary rather than at convenient points
  (`tests/test_closed_form_dispatch_boundaries.py`): closed form, Gauss-Legendre
  and dblquad are evaluated directly at the same cells — the diagonal ridge
  `t₁ = t₂`, `t → t_min`, `t = t_max`, `γ = 1/σ_t` exactly and to within
  1e-7/1e-9 of it, `r = 0` and `r = r_max`, per-component `γ`, `t_min ≠ 0`, a
  white-noise impulse — and agree to **1e-10** (vs GL) and **1e-8** (vs
  dblquad).  The boundary sweep caught two things a point check would not:
  `DiagonalA` lowers per-component rates that agree to `np.allclose`'s 1e-5
  into a single-rate `R` (the closed form now mirrors the model actually
  built, so the two cannot disagree at that tolerance), and the un-split
  dblquad path was only ~2e-6 accurate at the cusp (below).

- **Separable kernels build one temporal table.**  In lazy translation mode
  `C(r; t₁, t₂) = κ_x(r)·C(0; t₁, t₂)`, so the `n_grid_t²` quadrature grid is
  built once and rescaled per separation instead of being redone for every
  distinct `r` (4× fewer quadrature calls for the README sweep, 12×+ for a
  positions sweep).  With an even temporal kernel and diagonal `C` only the
  upper triangle of the `(t₁, t₂)` grid is evaluated (another 2×).  Both are
  exact up to rounding: identical to the per-`r` full build to 1e-12 under
  Gauss-Legendre (`tests/test_propagator_dispatch.py`).  Off for user
  closed forms / `c_value_fn` (nothing is known about their structure) and
  with a white-noise impulse (its `r`-independent term breaks the scaling).

- **`c_method` default `dblquad` → `auto`.**  `auto` runs Gauss-Legendre
  with a node count chosen by `select_gl_node_count`: starting from
  `c_n_gauss = 20` the rule is refined (×1.5) until it agrees with its own
  refinement to 1e-8 at the table's extreme cells (the corner `(t_max, t_max)`,
  its mid-table neighbours, the thinnest strip), falling back to `dblquad`
  if no count up to ~100 converges.  User callables (`GeneralKappa2`,
  `CustomKernel`, `CustomImpulse`, `ExplicitR`, callable `γ`) go straight
  to `dblquad`.  Why not simply "GL at 20": at fixed node count the
  accuracy degrades with `(γ + 1/σ_t)·t_max`.  Measured on the demo1
  kernel (`γ = 1`, `σ_t = 0.3`, cell `(t, t)`, relative to the closed form):

  | t   | split dblquad | GL n=20 | GL auto (n) |
  |-----|---------------|---------|-------------|
  | 5   | 7 ms, 2e-16   | 5.6 ms, 1e-15 | 5.8 ms, 1e-15 (20) |
  | 15  | 38 ms, 3e-16  | 5.9 ms, 1e-10 | 6.0 ms, 1e-10 (20) |
  | 30  | 86 ms, 1e-15  | 5.7 ms, 4e-7  | 13 ms, 2e-12 (30) |
  | 50  | 145 ms, 8e-13 | 5.9 ms, 2e-4  | 13 ms, 2e-9 (30) |
  | 100 | 212 ms, 8e-10 | 5.9 ms, **1e-2** | 30 ms, 3e-9 (45) |

  For a Gaussian temporal kernel (`σ_t = 0.3`) fixed n=20 is already 2e-4
  off at `t = 15`; `auto` picks 45 there (30 ms, 2e-11) and hands `t ≥ 30`
  to dblquad.  On the demo1 / demo2 kernels none of this runs: the closed
  form does.  The L0 default (`PropagatorCache(c_method="auto")`) resolves
  at the first table build; a direct `C_value` call before any build uses
  dblquad.

- **The dblquad path splits the rectangle at the `λ₁ = λ₂` cusp** (the
  same three pieces the Gauss-Legendre path uses).  Stationary `κ²` has a
  `|λ₁ − λ₂|` cusp there; integrating across it made scipy's adaptive rule
  both slow — the referee's run: 74 ms per call rising to 245 ms with `t`
  — and inaccurate, reporting roundoff and delivering 2e-6 relative error
  at `epsrel = 1e-10` against the closed form.  Split: 7 ms per call at
  `t = 5` (38 ms at 15, 212 ms at 100) and 1e-10 to 1e-15.  **Numbers
  move** by up to ~2e-6 relative on the dblquad path, toward the exact
  value.

- **Progress reporting** (`sft_wick.progress`).  Three stages get bars —
  expansion, C-table build (per cell, per `r`), sweep (grid points ×
  diagrams) — with elapsed / ETA, through `tqdm` when installed
  (`pip install sft-wick[progress]`) and otherwise as plain stderr lines;
  bars stay silent for loops under one second and by default when stderr
  is not a terminal.  CLI: stage banners with timings, `--quiet`, and
  `SFT_WICK_PROGRESS=0` / `=1` to silence / force.  L1:
  `progress=True|False|callable` on `System.expand`, `System.propagators`,
  `Expansion.sweep`.  Never changes a number (tested).

- **`sft-wick run --dry-run` prints a cost estimate**: diagrams per order,
  grid points, distinct separations, the resolved C source, the number of
  quadrature calls the chosen path will make, and a rough wall-clock from
  a one-second micro-benchmark of one C call and one diagram evaluation
  per order (`sft_wick.workflow.estimate`).

- **`examples/quickstart.yaml` and `sft-wick quickstart`.**  Demo1 physics
  at orders 0 and 2, three separations, two times, 4096 samples, markdown
  table to stdout.  `sft-wick quickstart` writes the file to the current
  directory and runs it.  The README "Quick Start — L2" is now this file.
  A GitHub Actions job (`examples-time-gate`) runs it under `timeout 600`
  and `examples/demo1_config.yaml` under `timeout 1200` on a stock
  `ubuntu-latest` runner.

- **Validation catalogue** `docs/verification/catalog.rst`, generated by
  `tools/gen_test_catalog.py` from the collected suite (`pytest
  --collect-only`) plus a curated table of what each file checks, against
  which reference (brute-force Wick, closed form, alternative backend,
  simulation cache, …) and at what tolerance.  `make -C docs html`
  regenerates it and `tests/test_catalog_current.py` fails when it is
  stale or a test file has no entry.  The README's "460 tests" and the
  verification page's "275 tests" were both wrong; the catalogue carries
  the measured count.

- **Fixed: `DiagramTerm.to_latex()` printed a local coupling's point.**
  The index-relabelling passes rebuilt `Symbol`s without the `local` flag,
  so the symmetrised partner of a local `F` rendered as
  `F_{i1 i0 i2}(y_0)` (Table 1 of the paper).  Regression test
  `tests/test_latex_local_coupling.py`.

- Docstrings in `sft_wick.spectral` referred to a directory outside the
  repository; rewritten to state the caveat itself.

- **Paper-figure regression check.**  `examples/demo1/L2/config.yaml`
  (the sweep behind the paper's Gaussian-noise figures: 672 grid points ×
  71 diagrams, 32768 samples) run on v0.2.0 and on this release agrees to
  a maximum relative difference of **3.3e-14** over all 2016
  `(y, t, a, b, order)` totals — the closed-form C path is untouched by
  the C-table diagonal fix, as the July changelog claimed; now verified
  rather than assumed (`examples/paper_assets/README.md`).

- **Fixed: `output: {type: npz, path: results/x.npz}` failed when the
  directory did not exist** (`np.savez` does not create it; the table
  writer already did).  The same for `plot` outputs.

- **`c_method: auto` also compares cost.** When both rules are converged
  for the horizon, one call of each at the deep corner decides: on a
  smooth kernel over a short horizon adaptive dblquad needs a few hundred
  evaluations (3.8 ms) and beats the 20-node tensor rule (5.6 ms); with a
  cusp or a long horizon the tensor rule wins by 5-30×.  (Without this,
  a smooth-kernel regression test took 73 s instead of 28 s.)

### Demo 2 (non-Gaussian noise) corrections

Re-deriving and re-running the κ⁽³⁾ example for the referee turned up
three things the paper's demo-2 figures inherit; all are fixed in
`examples/demo2` and quantified in `examples/paper_assets/demo2_kappa4`.

- **κ⁽³⁾ was missing its `α³` term.**  For `η̃ = η + α(η² − λ)` the third
  cumulant is `κ⁽³⁾(1,2,3) = 2αλ²[k₁₃k₂₃ + k₁₂k₂₃ + k₁₂k₁₃] + 8α³λ³ k₁₂k₂₃k₁₃`;
  the second term (the connected three-point function of `η² − λ`) is
  2.4 % of the first at coincidence and was absent from `k3_coupling.py`
  and from the `6αλ²` line of the κ³ cross-check figure.  The simulated
  third moment, 9.21e-3, is `6αλ² + 8α³λ³`, not `6αλ² = 9.00e-3`.  The
  FK channel of ξ₀₁ moves by +1.2 % (t = 1) to +0.6 % (t ≥ 3).

- **The FK quadrature was not converged beyond t ≈ 10.**  The raw kernel
  is narrow in the relative leg times, and the fixed 8-node tensor rule
  on the 4-D integral gives, for ξ₀₁ at r = 0, 4.95e-4 at t = 15 and
  1.85e-4 at t = 50 against the converged plateau 3.44e-4 (n = 12/16/20
  give 4.23/3.86/3.71e-4 at t = 15; QMC is worse).  The remedy is the
  package's own `already_R_contracted` vertex: `examples/demo2/k3_R_coupling.py`
  integrates the three legs inside the callable (analytically in the
  common time, composite Gauss-Legendre on the two relative times with
  each cusp aligned to the grid), so the FK integral is one-dimensional
  and a 32-node rule agrees with 64 to 1e-4 at every t.  `config_FK.yaml`
  now uses it.  The R-contracted kernel is validated against adaptive
  3-D quadrature and QMC of the raw leg integral to ~1e-4.

- **ξ₀₀ / ξ₁₁ used the single-kernel `λ_eff` approximation.**  The exact
  effective covariance of `η̃` is `λ k + 2α²λ² k²` -- the second piece
  has HALF the correlation time and length -- so replacing it by
  `λ(1 + 2α²λ) k` over-counts its contribution to C by 70 %: +1.8e-4 on
  ξ₀₀ at large t (1.5 % of the signal, 2σ of the 200k-realisation
  simulation).  The budget uses the exact form (a sum of two built-in
  closed forms) for order 0, FF and FFFF.

- **κ⁽⁴⁾ enters ξ₀₀ / ξ₁₁ but not ξ₀₁.**  `κ⁽⁴⁾ = 4α²λ³ Σ_paths k k k +
  16α⁴λ⁴ Σ_cycles k k k k` (12 Hamiltonian paths and 3 cycles of K₄;
  `examples/paper_assets/demo2_kappa4/k4_coupling.py`, checked against
  Monte Carlo).  F·κ⁽⁴⁾ at order 2 vanishes by ψ/φ counting; the leading
  term is F·F·κ⁽⁴⁾ at order 3 (three pure-R⁶ diagrams).  The Gaussian
  theory with F has a `φ₁ → −φ₁` symmetry that only ODD cumulants break,
  so FF, FFFF and FFK4 are identically zero for ξ₀₁ -- the κ⁽³⁾ signal
  there is clean, and its residual can only be closed by F³κ⁽³⁾ and
  higher odd terms.

- **The error budget** (`examples/paper_assets/demo2_kappa4/budget.md`,
  2M realisations per step size at Δt = 0.02 and 0.01, extrapolated):
  ξ₀₁ − FK is 0 within 1σ for t ≤ 1 and +24 % of FK (9e-5, 6-7σ) for
  t ≥ 5; the order-4 F³κ⁽³⁾ channel, estimated by collapsing κ⁽³⁾ to an
  equal-time constant calibrated on FK, is 2 % / 12 % / 24 % of FK at
  t = 1 / 3.5 / 15 -- the residual is truncation, not κ⁽⁴⁾ (zero there by
  symmetry), Δt or Monte-Carlo noise.  ξ₀₀ with the exact C_eff, FFK4
  and FFFF agrees within ±2σ except at t ≈ 5 (+3σ).

### Added

- **`sft_wick.selfconsistency` — the fixed-point driver.**
  `solve_self_consistency(initial, step, ...)` iterates, mixes, and is honest
  about what happened. The package could compute a self-energy but nothing in
  it iterated, so every DMFT-style use was one pass of a loop the caller wrote
  by hand, usually without convergence diagnostics.

  The Dyson solve is **deliberately not** attempted: it is model-specific and
  is genuinely an integral-equation solve rather than a diagram evaluation, so
  a wrong general one would be worse than none. You supply it as `step`.

  It never returns a bare state — a non-converged iteration looks exactly like
  a converged one if you only print the last state. The result carries
  `converged`, the residual history, and a `reason` in
  `converged / diverged / oscillating / max_iter`; `bool(result)` is
  `converged`. Four failure modes it is specifically built to avoid reporting
  as success are documented in the module docstring and in
  `docs/api/selfconsistency.rst`, each with a regression test. Linear mixing
  only — no Anderson acceleration, which matters near a transition.

- **Callable (spacetime-dependent) couplings now work with MATRIX-valued
  response propagators.** This combination was previously computable by *no*
  backend at any cumulant order: `method='qmc'` refused callables, while
  `qmc_vectorized` and `gauss_legendre` refuse matrix `R`. Worse, the scalar
  loop's error message named the other two, which refuse for the *other*
  reason — a loop of three `NotImplementedError`s with no exit. It is the
  natural disorder-averaged-with-components configuration.

  The scalar loop is in fact the natural home for the per-sample callable
  contract, since it already visits one sample at a time.
  `DiagramIntegrand.evaluate()` takes a `coupling_array=` override, and
  `dynamic_coupling_array()` materialises it. Verified against
  `qmc_vectorized` where both are legal, against the static-tensor path at the
  same point (exact agreement in both), and against three closed-form
  one-dimensional integrals.

- **`SpectralDensity.average(f, node_axis=-1)`.** When the array `f` returns is
  square, which axis indexes the spectral nodes is ambiguous by shape and no
  check can resolve it; `node_axis=` states it at the call site. Note that in
  exactly that square case a *wrong* `node_axis` returns a wrong number rather
  than raising — the parameter is a declaration the array cannot confirm.

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

- **`_mix` coerced the state's leaf through `np.asarray`.** The container
  branches preserve dict / list / tuple / namedtuple, but the leaf did not, so
  a state whose leaves are array-*like* rather than `ndarray` — a JAX or torch
  array — reached `step` as its own type on iteration 1 and as a plain host
  `ndarray` from iteration 2 onward, silently moving a device array off the
  device. Only `damping > 0` reached the coercion, so it was invisible in the
  default configuration and appeared exactly when the caller took the module's
  own advice to add damping.

- **The C-value memo lost an ndarray subclass's identity.**
  `np.ascontiguousarray` downcasts a subclass to a base array, so a masked
  array's `tobytes` override never ran and the mask vanished from the key: two
  positions differing only in their mask shared one entry. `tobytes()` already
  serialises in C order whatever the memory layout, so the
  `ascontiguousarray` call bought nothing and cost this.

- **Object-dtype positions were keyed on recyclable pointers.** An object
  array's buffer is raw `PyObject*` values, and an address is unique only among
  *live* objects — the same defect as keying on `id()`. Those are no longer
  memoised at all, and the refusal is decided from inside the recursion, so a
  nested object array is caught too.

- **The memo entry was frozen in place rather than copied.**
  `_C_value_direct` ends in `np.asarray`, the identity for a float64 array, so
  marking the result read-only flipped the flag on the *user's* object and
  broke any `c_value_fn` returning a reused buffer or a module-level constant.
  Copied first now; the memo is still handed out read-only, since it is the
  cache entry itself and a caller doing `C += x` would rewrite every later
  lookup.

- **The memo rode into every parallel worker.** All three
  `precompute_C_table_*` builders dispatch a closure referencing `self`, so a
  full memo was serialised into each worker payload — tens of megabytes,
  growing as `N^2`, and useless to the worker, which recomputes what it needs.
  `__getstate__` empties it.

- **`evaluate_at` ignored `coupling_vectorized=True`,** handing a callable
  declared under the batched contract the per-sample one instead. Some batched
  callables broadcast and return a plausible *wrong* shape rather than raising.

- **A coupling leg that no propagator attaches to** has no `direction_map`
  entry, and all backends raised a bare `KeyError` when the callable asked for
  its position.

- **`SpectralDensity.average` did not validate `node_axis`.** Out-of-range
  values died with a bare `IndexError` from the shape lookup, and
  `node_axis=True` sailed past the length check (`shape[True]` is `shape[1]`)
  only to die inside `tensordot`. Neither names the parameter at fault.

- **`max_abs_distance` subtracted in the input's integer dtype,** so the
  difference wrapped modulo `2**nbits` and a state far from the fixed point
  could report a distance small enough to satisfy `tol` — convergence declared
  by overflow.

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
- `sft_wick.spectral`: `average()` gives a diagnostic error for a transposed
  layout instead of numpy's shape error — a better message, not new safety:
  that layout already raised. The genuinely ambiguous case, `k == n_nodes`,
  is indistinguishable by shape and is documented rather than claimed fixed; `SpectralDensity` became comparable and hashable
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

- **The spatially-uniform-cache warning moved to `Expansion.sweep`.** A cache
  with no spatial table returns the same `C` at every separation, so a
  positions sweep yields a column of identical numbers — correct for a
  disorder-averaged single-site cache, a silent mistake otherwise. The
  question is "does this *sweep* cover more than one position configuration",
  which is answerable only where the grid is. Asked per grid point instead it
  was silent on a sweep over a single position key and fired on every ordinary
  single-separation `evaluate(x != y)` — and a warning that fires on normal
  use is worse than none.

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
