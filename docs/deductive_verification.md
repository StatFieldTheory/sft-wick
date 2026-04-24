# Deductive verification of sft-wick

This document summarises the deductive test programme that accompanies
the simulation-based demo verification in
[`examples/demo1`](../examples/demo1) and [`examples/demo2`](../examples/demo2).

Where the demo notebooks verify the package inductively (sim and theory
agree within Monte-Carlo noise over a specific parameter range), this
programme pits each elementary transformation in the pipeline against an
independently derivable ground truth, so any disagreement points to the
offending module.

---

## Scope

Two phases, run in sequence:

1. **Phase 1 — symbolic expansion** (`tests/test_deductive_expansion.py`).
   For the simplest representative actions — a scalar single-vertex
   cubic theory (Case A) and the same action augmented with a non-local
   ψψψ vertex (Case B) — enumerate Wick pairings by hand, classify each
   by vanishing rule, verify multiplicities and coefficients, and
   cross-check canonicalisation against `pynauty` (when installed) and a
   brute-force permutation search.  Every comparison is an *exact*
   equality — no numerical tolerance.

2. **Phase 2 — numerical evaluation** (`tests/test_deductive_numerics.py`).
   With Phase 1 as a proven symbolic baseline, check the propagator
   formulae, integration methods, and dimensional scaling against
   analytical identities.  Tolerances are chosen from the known
   convergence rates of each method.

The brute-force reference lives in `tests/brute_wick.py` — ~300 lines,
**zero imports from `sft_wick`**, using only `itertools`, `fractions`,
and `collections`.  It is the ground truth against which every
`sft_wick` output is tested.

---

## Phase 1 — test matrix

| ID  | Scope                              | Module under test                                              |
|-----|------------------------------------|----------------------------------------------------------------|
| T1  | pairing enumeration count          | `wick.generate_valid_pairings`                                 |
| T2  | per-pairing vanishing rules        | `wick`, `propagators.contract_pair`, Itô, causal DFS           |
| T3  | operator-level Wick sum            | `wick.wick_contract`                                           |
| T4  | spatial multiplicity               | `wick._compute_multiplicity`, symmetry factor formula          |
| T5  | engine agreement (op vs spatial)   | hybrid engine, `_enumerate_component_routings`                 |
| T6  | Taylor / multinomial prefactor     | `perturbation.compute_moment` prefactor assembly               |
| T7  | response phase $(-i)^{n_R}$        | `expressions.apply_response_phase`, `DiagramTerm`              |
| T8  | canonical-form agreement           | `simplify._canonical_diagram_form`, `_canonical_key_nauty`     |
| T9  | diagram-collection soundness       | `simplify.collect_by_diagram`, `_match_propagators_after_spatial` |
| T10 | individual simplification passes   | `simplify._flatten`, `_absorb_rationals`, `_eliminate_zeros`   |
| T11 | non-local vertex instantiation     | `Vertex(local=False)`, `VertexInstance.instantiate`            |
| T12 | KK vanishing by leg count          | `wick_contract_spatial`, leg-count check                       |
| T13 | FK coupling-sum collapse           | `DiagramTerm.evaluate_coupling`, summation index handling      |
| T14 | α=0 regression                     | whole pipeline, consistency check                              |
| T15 | N-invariance of spatial topology   | `wick_contract_spatial` factors spatial ≠ component            |
| T16 | summation-index dimensions scale with N | `DiagramTerm.summation_indices` bookkeeping                |
| T17 | hand-computed ``evaluate_coupling`` at N=2 | `DiagramTerm.evaluate_coupling`, coupling-sum collapse      |
| T18 | sparse F vanishing pattern         | component-index routing through vertices                       |
| T19 | engine agreement at N ∈ {2, 3}     | `_enumerate_component_routings`                                |
| T20 | FK selection rule at N=3           | `Vertex(local=False)` + coupling-sum collapse at N>2           |
| T21 | non-iso ↔ iso consistency          | full `R_{ij}`/`C_{ij}` evaluation contracts back to iso scalar  |

### Key numerical outcomes (Case A, order 2)

By brute enumeration:

- Candidate pairings: **105** = (7!!).
- After ψ-ψ filter: **90** remain.
- After Itô R(x,x)=0 filter: **42** remain.
- After causal R-loop filter: **30** remain — the survivors.
- These 30 partition into **12 labelled spatial topologies**.
- Under integration-variable relabelling (z₀ ↔ z₁), these **merge into 6
  distinct Feynman diagrams** — the "order 2" count reported by demo 1.

sft-wick's `compute_moment(order=2)` produces:

- `diagram_terms(2)` → 6 `DiagramTerm` records (matches hand count).
- `diagrams_by_order[2]` → 30 `DiagramInfo` records (matches the labelled
  pairing count from brute enumeration).

### Key numerical outcomes (Case B, order 2)

- FK pair class: 12 surviving pairings → 6 labelled topologies → 2
  Feynman diagrams.
- FF pair class: same 30/12/6 as Case A.
- KK pair class: 0 survivors (leg count forbids, brute says so too).

### Non-local vertex coverage

Before this test suite, **zero** tests exercised the `Vertex(local=False)`
code path.  T11–T13 and T20 collectively add:

- `VertexInstance.instantiate` for non-local vertices (T11: each field
  gets a fresh spatial variable; local vs non-local distinction tested
  directly).
- `compute_moment` with a non-local vertex in `Action` (T12 and T13).
- `evaluate_coupling` on a `DiagramTerm` whose coupling has a symbolic
  spatial-argument list (T13 at N=2, T20 at N=3).

### Multi-component coverage (T15–T20)

These exercise the component-index routing path, which is separate
from the spatial topology path and carries almost all of the
combinatorial complexity at N > 1.

**Hand-derived formulas for demo-1's F tensor at N = 2.**  For each of
the 6 FF diagrams at order 2 we derived a closed form for
`evaluate_coupling` purely from the Wick-contraction structure and the
ψφφ vertex's symmetry in its two φ-legs.  Let
`S(c) = Σ_i F[c, i, i]`, `T1(a,b) = Σ_{i,j} F[a,i,j] F[b,i,j]`,
`T2(a,b) = Σ_{i,j} F[a,i,j] F[b,j,i]`,
`U(a,b) = F[a, 0, b] + F[a, b, 0]` (simplified using S(1)=0 for demo-1's F),
`W(a,b) = 4 Σ_{i,j} F[a,i,j] F[i,b,j]`.  Then:

| diagram | evaluate_coupling at (a, b) |
|---------|----------------------------|
| 0 (double tadpole, C(z₀,z₀) + C(z₁,z₁)) | `S(a) · S(b)` |
| 1 (bubble, two C(z₀,z₁))               | `T1(a,b) + T2(a,b)` |
| 2 (C on y-side)                        | `U(a,b)` |
| 3 (crossed, C(y,z₁) + C(z₀,z₁))        | `W(a,b)` |
| 4 (C on x-side, a↔b of 2)              | `U(b,a)` |
| 5 (crossed, a↔b of 3)                  | `W(b,a)` |

This table is a direct consequence of MSR structure — no sft-wick
implementation choice enters.  T17's 15 parametrised assertions check
all six diagrams at all three independent (a, b) combinations.

**Sparse-F sanity (T18).**  For `F[0,0,0] = 1, all others 0`:
- `xi_ab` at order 2 must vanish for any (a, b) ≠ (0, 0);
- at (0, 0), the six diagrams take hand-derivable values `[1, 2, 2, 4, 2, 4]`
  from the formulas above (with F[0,0,0]=1).

**Engine agreement at N > 1 (T19).**  The operator-level and spatial-
level Wick engines share almost no code at N > 1: the spatial path
goes through `_enumerate_component_routings`, which reconstructs
component-index assignments from a spatial representative and the
pairing.  T19 compares the set of Feynman-diagram canonical forms
produced by both paths at N = 2 and N = 3.

**Non-local at higher N (T20).**  Extends T13 to N = 3.  With
component-diagonal `K[a,a,a] = k_a` only, the FK contribution to
`xi_{ab}` equals `F[a, b, b] · k_b + F[b, a, a] · k_a`.  For demo-1's F:
non-zero only at (0, 1), (1, 0); the selection rule still forbids
diagonal pairs.

**Non-iso ↔ iso consistency (T21).**  Previously *every* numerical
test used `iso_R=True, iso_C=True` to collapse the propagator
component structure, leaving the full `R_{ij}` / `C_{ij}` routing
untested.  T21 fills this hole: it calls `compute_moment` without iso
flags (so each propagator leg carries its own component index — at
N=2 a 4-propagator diagram's `evaluate_coupling` returns a shape
(2,2,2,2,2,2) = 64-entry tensor), then contracts each propagator
`(index_left, index_right)` pair with a Kronecker δ (the trivially
diagonal + isotropic case).  The resulting scalar must equal the
iso-flag pipeline's output.  This verifies that the symbolic iso
rewrite is a *proper simplification* — it never drops or mis-routes
an index relative to keeping full structure.  T21 runs this
consistency check on all 6 order-2 diagrams × 3 observable pairs.

**Off-diagonal κ² → non-diagonal C (N9).**  Tests the numerical
routing of non-trivial cross-component correlations through
`PropagatorCache.C_value`.  Supplies a κ² with an off-diagonal 2×2
matrix coefficient (symmetric positive-definite, off-diag 0.3), then
verifies sft-wick's returned C matrix has the expected non-zero
off-diagonal entries to within 1e-4 relative.  Ground truth via
domain-split dblquad on each of the 4 matrix entries.  This is the
numerical counterpart to T21: T21 tests the *symbolic* non-iso
machinery, N9 tests the *numerical* non-diagonal machinery.

### Phase 3 — full Feynman-diagram integration (P1–P3)

Tests ``DiagramIntegrand`` + ``scipy.nquad(make_scipy_integrand)`` and
``integrate_moment_qmc`` against closed-form references.  Uses the
**double-tadpole** diagram at order 2, whose integrand factorises:
the contribution decomposes into independent 1-D integrals per
vertex, each evaluable to machine precision via ``scipy.quad``.

Closed-form references (for demo-1's F tensor at (a=b=0), N=2):

- ``A(t) = ∫_0^t R(t,τ) C(τ,τ) dτ``
  — per-vertex 1-D integral.
- ``B(λ) = ∫_0^λ A(t) dt``
  — outer integral over external time (for moments).
- Fixed-time ξ(r=0, t_f) = ``N² × coupling × A(t_f)²``.
- Time-integrated moment at ``lambda_f`` = ``N² × coupling × B(λ)²``.

The factor ``N²`` = 4 at N=2 comes from the trace over component
indices implicit in the C-propagator contraction (even with
``iso_C=True``).  ``coupling`` for this diagram at (a=b=0) with
demo-1's F equals 1 (= ``S(0)²``) from the hand-derived formulas in
T17.

**Design choice: closed-form C cache in the test.**  Sft-wick's
native ``PropagatorCache`` either ``dblquad``s every C evaluation
(QMC at 2^14 samples takes ~42 min per test) or pre-tabulates C on a
spline grid (~85 s setup, precision floor ≈ 5e-4 from cubic
interpolation).  The Phase-3 tests use a drop-in analytical cache
that returns closed-form C in microseconds and preserves
machine-precision bounds.  This *isolates the QMC machinery under
test* — sft-wick's spline path is already tested in N7
(``TestSplineTable``), and the dblquad path in
``TestPropagatorCacheCValue``.

P1 uses ``scipy.nquad`` on the `make_scipy_integrand` closure with
``rel < 1e-6`` — verifies the fixed-time evaluation path end-to-end
against the factorised ``A²`` reference.

P2 runs ``integrate_moment_qmc`` at ``n_samples=2^14`` and asserts
agreement within the reported 3σ error band, and ``rel < 5e-3`` as a
sanity floor.

P3 runs QMC at ``n_samples ∈ {2^10, 2^12, 2^14}`` and asserts the
error (``|got − closed_form|``) decreases by at least a factor of 2
across the scan — a regression guard for the Sobol sampling logic
without assuming a precise O(1/N) convergence rate.

**What Phase 3 does not cover.**  Only the double-tadpole diagram is
checked.  Other order-2 diagrams (bubble, chain with internal R,
mixed-vertex FK) have coupled temporal integrations whose closed
forms are messier — they fall back to demo-1/demo-2's inductive
sim-based verification.  Extending Phase 3 to those would add ~3
test cases, each with a 2-D ``scipy.dblquad`` reference.

### Phase 4 — alternative-path consistency (C1–C5)

sft-wick exposes several *alternative* entry points (faster or
parallelised versions of the defaults).  Before Phase 4 these were
untested:

- ``compute_moment_numerical``: nauty-based alternative to the
  hybrid ``compute_moment``.  Enables order-6 calculations and an
  optional ``n_jobs`` parallelisation for the canonicalisation +
  component-routing steps.
- ``integrate_moment_qmc_vectorized``: batch-vectorised QMC (no
  Python loop over samples) — much faster for large n_samples.
- ``integrate_two_point_qmc``: specialised QMC for fixed-time
  2-point correlators that supports **per-point spatial
  positions** (used by demo-1's ``xi_pert``).
- ``integrate_diagrams(n_jobs=-1)``: parallel batch evaluation of
  multiple diagrams via joblib.

Phase 4's pattern: each alternative is pitted against a tested
baseline, and the two must agree.

**C1 — two symbolic engines** integrated-total equality.
``compute_moment`` groups Wick pairings into canonical Feynman
topologies (6 DTs at Case A order 2), while
``compute_moment_numerical`` uses nauty + exhaustive component-
routing enumeration (30 DTs).  The *per-DT* count is structurally
different by design, but the integrated sum over all diagrams must
agree — observed ~5e-7 relative difference (within QMC precision).
This is the load-bearing check that both symbolic engines produce
the same physical observable.

**C2 & C5 — parallel-vs-serial bit-identity.**  joblib's default
backend preserves result ordering on these deterministic workloads,
so ``n_jobs=-1`` must yield bit-identical diagram-term lists and
integrated totals to ``n_jobs=1``.  Any discrepancy would indicate
either an ordering bug or a non-deterministic RNG path leaking into
the parallel branch.

**C3 — scalar vs vectorised QMC.** ``integrate_moment_qmc`` runs a
Python loop over Sobol samples; ``integrate_moment_qmc_vectorized``
uses batched array operations on the same Sobol sequence.  Same
seed → **bit-identical** result (both to the integrand value and
to the standard-error estimate).

**C4 — two 2-point integrators.** ``integrate_two_point_qmc`` is a
specialised fast path; ``scipy.nquad(make_scipy_integrand(...))``
uses a completely different adaptive-quadrature backend.  Must
match to QMC precision (~1e-7 observed).

Phase 4 closes the coverage gap for alternative entry points.  Any
future optimisation added alongside an existing path (e.g. a
further-vectorised QMC, a CUDA backend, a memoised symbolic engine)
should come with a corresponding C-series consistency test — the
pattern is now established.

**Important caveat — equivalence vs substitutability.**  C1–C5 verify
that the alternative paths agree on the *specific test case used*
(scalar observable at the iso_R/iso_C simplification, `r=0`,
`fixed_indices={a:0,b:0}`).  They do **not** prove full input/output
equivalence: the alternatives are not all drop-in replacements.
Known capability differences:

- ``compute_moment`` and ``compute_moment_numerical`` have different
  signatures.  ``_numerical`` requires ``coupling_values`` upfront and
  returns only ``DiagramTerm`` dicts; it has no ``response_phase`` or
  ``collect_topology`` flags and no access to the symbolic expression
  tree (``PerturbativeResult.order_terms``).  If you need the
  symbolic expression for LaTeX rendering or post-hoc diagonal
  simplification, you must use ``compute_moment``.
- ``integrate_moment_qmc_vectorized`` requires the cache to support
  batched C evaluation (``PropagatorCache.precompute_C_table`` first,
  or a cache with ``C_diagonal_batch``).  The scalar
  ``integrate_moment_qmc`` works on any cache.
- ``integrate_two_point_qmc`` takes per-point spatial ``positions``
  (suitable for 2-point functions where ``x`` and ``y`` differ); it
  is a higher-level interface than ``make_scipy_integrand``, which
  expects per-direction group values.  C4 happens to compare them
  at ``r=0`` where both accept identical input.

**Benchmark summary** (N=2, order 2, on this machine):

| use case | fastest path | relative speedup |
|---|---|---|
| symbolic expansion (order ≤ 2) | tie (both sub-ms) | — |
| symbolic expansion (order ≥ 4) | ``compute_moment_numerical`` | dramatic (per docs) |
| single-diagram QMC (n=2^16) | ``integrate_moment_qmc_vectorized`` | **208×** over scalar loop |
| batch over diagrams | ``integrate_diagrams(n_jobs=-1)`` | ~1.6× on 6 diagrams, grows with batch |
| 2-point with non-trivial spatial r | ``integrate_two_point_qmc`` | only path that supports it |

The 208× speedup from vectorised QMC is the most impactful single
finding of the audit.

### Consequence — dispatcher default changed

As a direct result of the benchmark, the default ``method='qmc'`` in
``integrate_moment`` (and therefore in ``integrate_diagrams``) was
changed to **auto-select** the vectorised path when the cache
supports batch C evaluation, falling back to scalar only if not.
Previously, ``method='qmc'`` always ran the scalar loop — forfeiting
the 208× speedup for any user whose cache *could* batch.

New behaviour:

- ``method='qmc'`` (default) → vectorised if
  ``cache._c_splines is not None``, else scalar.  Covers
  ``PropagatorCache`` after ``precompute_C_table``, and any custom
  cache that sets the ``_c_splines`` sentinel (e.g. the analytical
  cache used in demo-1/demo-2 and in the Phase-3/4 tests).
- ``method='qmc_vectorized'`` → force vectorised (errors if cache
  doesn't batch).
- ``method='qmc_scalar'`` → force scalar (explicit opt-out, for
  debugging or non-batch caches where you'd rather not fall back
  silently).
- ``method='nquad'`` → unchanged.

**Benchmark verification** on the 6-diagram test workload:

| call | old default | new default | speedup |
|---|---|---|---|
| single diagram, n=2^16 | 1035 ms | 5 ms | **208×** |
| batch over 6 diagrams, n=2^14 | 1568 ms | 9 ms | **180×** |

Results are **bit-identical** between old and new defaults (same seed,
same sampling logic — just batched instead of looped), so the change
is a pure performance improvement with no numerical side-effect.

**C6** (added to ``TestAlternativePathConsistency``) codifies this
behaviour: verifies the dispatcher picks the vectorised path when
``_c_splines is not None``, falls back to scalar on a proxy cache
without batch support, and ``method='qmc_scalar'`` still routes to
the scalar implementation for explicit opt-out.

---

## Phase 2 — test matrix

| ID  | Scope                                | Method                                                      |
|-----|--------------------------------------|-------------------------------------------------------------|
| N1  | closed-form C vs adaptive quadrature | dblquad at eps=1e-8                                         |
| N2  | Gauss-Legendre convergence           | monotone error decrease at n_gauss ∈ {20, 40, 80}           |
| N3  | stationarity limit                   | $C(t,t) \to \lambda/[\gamma(\gamma+1/\sigma_t)]$ as t → ∞   |
| N4  | dimensional scaling                  | C ∝ λ; C_stat ∝ 1/γ(γ+1/σ_t) as γ varies                    |
| N5  | α=0 two-kernel vs single-kernel      | identical to machine precision when α=0                     |
| N6  | two-kernel κ²_eff shape              | B-term closed form vs direct dblquad integration            |
| N7  | spline-table acceleration            | `precompute_C_table` spline values vs closed form           |
| N8  | Itô flag end-to-end                  | `compute_moment(ito=True/False)` on equal-point φψ observable |
| N9  | off-diagonal κ² → non-diagonal C     | `PropagatorCache.C_value` with `diag_C=False` and cross-component kappa2 |
| P1  | `scipy.nquad(make_scipy_integrand)` = N²·coupling·A² | double-tadpole fixed-time ξ, closed-form reference to 1e-6 |
| P2  | `integrate_moment_qmc` = N²·coupling·B² | same diagram time-integrated moment, QMC within 3σ |
| P3  | QMC convergence monotone vs n_samples | Sobol QMC regression guard |
| C1  | `compute_moment` vs `compute_moment_numerical` | 2 independent symbolic engines produce equal integrated totals |
| C2  | `compute_moment_numerical` parallel vs serial | joblib parallelisation preserves DiagramTerms bit-identically |
| C3  | `integrate_moment_qmc` vs `integrate_moment_qmc_vectorized` | scalar-loop vs batch QMC, same seed → bit-identical |
| C4  | `integrate_two_point_qmc` vs `scipy.nquad(make_scipy_integrand)` | alternative 2-point QMC matches independent integrator |
| C5  | `integrate_diagrams(n_jobs=1)` vs `(n_jobs=-1)` | parallel batch evaluation bit-identical to serial |
| C6  | `integrate_moment(method='qmc')` dispatcher auto-selects fastest | verifies vectorised is picked when cache supports batch, scalar fallback otherwise |

### Tolerances

- **Closed-form vs analytical**: exact to machine precision (1e-15).
- **dblquad vs closed form on cusped kernels** (|τ₁−τ₂| in exponent):
  1e-4 relative.  The `|·|` cusp at τ₁=τ₂ limits adaptive quadrature
  to ~1e-5 without domain splitting.
- **Tensor-product GL on cusped kernels**: 5e-3 relative at n=80
  (sub-algebraic convergence).  Convergence-rate test rather than
  absolute-precision test.
- **Spline interpolation**: 5e-3 relative, matching cubic-spline order.

---

## Phases 5–8 — workflow layer and new code paths

Phases 5–8 were added alongside the L1 workflow API.  Each follows
the same deductive discipline as Phases 1–4 — an independent
reference (analytic formula, equivalent-path comparison, or
closed-form limit) is checked for every new code path.  The
summaries below are high-level; the per-test matrix and tolerances
live in `docs/verification/index.rst` (rendered Sphinx) and the
test files themselves.

- **Phase 5 — spatial homogeneity** (`TestSpatialAwareCache`,
  S1–S7).  Three `precompute_C_table_*` variants (translation,
  rotation, general) validated against each other and against
  closed-form limits.  Cross-group C scaling and lazy-mode build
  verified.
- **Phase 6 — white-noise component** (`TestWhiteNoiseComponent`,
  W1–W3).  `GaussianNoise.sigma2` adds δ(t-t')σ²(t) to κ²;
  linearity (smooth + white = sum) and absorption into the
  translation spline are checked.
- **Phase 7 — dynamic non-local coupling + workflow round-trip**.
  `tests/test_workflow.py` (WF1–WF5) and
  `tests/test_workflow_config.py` (CF1–CF6).  The L1 wrapper's
  `System`/`Expansion`/`Propagators`/`SweepResult` chain is compared
  to `validate_phase5.py` reference values (WF4).  YAML round-trip
  tests (CF1–CF6) check that YAML → Python → numerical sweep
  matches the equivalent pure-Python L1 pipeline to rtol=1e-10 —
  this is an equivalence test (bit-identity), not a precision
  test.  Demo2's FK channel via spacetime-dependent κ^(3) is
  verified end-to-end via `examples/demo2/validate_FK_dynamic.py`
  (0.48 % rel agreement vs hand-coded reference).
- **Phase 8 — time-dependent linear operator** (T1–T4).
  `DiagonalA(gamma=callable)` pre-computes Γ(t) = ∫γ(τ)dτ as a
  cubic spline.  Checked against static analogues (T1, T1b),
  closed-form linear γ (T2), causality (T3), and end-to-end
  System round-trip (T4).

Full test count post-Phase-8: **275 tests**, full suite
~3.5 min on M-series.

---

## Running the tests

All Phase 1/2 tests, fast (<1 s for expansion, a few seconds for numerics):

```
pytest tests/test_deductive_expansion.py tests/test_deductive_numerics.py -v
```

With coverage instrumentation:

```
pytest --cov=src/sft_wick tests/ --cov-report=term-missing
```

The previously-unexercised `_enumerate_component_routings`,
`_canonical_key_nauty`, and non-local-vertex-instantiation branches are
now covered.

---

## Interpretation of a failure

Because every test isolates a single transformation, a failure
diagnoses a specific module:

- **T2 fails** → `contract_pair` or the vanishing-rule logic in `wick.py`.
- **T4 or T5 fails** → multiplicity formula in `_compute_multiplicity`
  or component-routing logic in `_enumerate_component_routings`.
- **T8 fails** → `_canonical_diagram_form` (brute permutation search)
  or `_canonical_key_nauty` (nauty-based) disagree on a pair of
  topologies — one of them has a bug.
- **T9 fails** → `collect_by_diagram` is creating spurious partitions
  (either merging non-isomorphic topologies or splitting one).
- **T13 fails** → `DiagramTerm.evaluate_coupling` mishandles the
  coupling sum for a spatially-dependent coupling (non-local vertex).
- **N1/N6 fails** → the analytical C-propagator formula disagrees with
  numerical quadrature of the same integrand — the closed form has a bug.

---

## What this does not cover

- Order 4 and higher combinatorics.  Brute enumeration at order 4 is
  tractable (6!! = 10395 pairings for the 2-point observable) but its
  test turnaround is seconds, not milliseconds; left for a follow-up.
- Multi-component field theories at order ≥ 4.  The new T15–T20 block
  covers multi-component at order 2 (the representative case); order-4
  index-routing correctness is still tested only inductively via demo 1.
- Order-4 FF diagrams' exact coupling collapse (cf.  `FF4_groups` in
  `analysis.ipynb`).  Currently tested only via the inductive sim
  comparison.

These are documented as future scope rather than acknowledged limitations:
every one is amenable to the same brute-force pattern if the problem
size is managed carefully.
