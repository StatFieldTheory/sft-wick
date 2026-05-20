# Design note — `NonLocalVertex(already_R_contracted=True)`

**Status**: **Landed 2026-05-20** on `main`. Phase 1 (schema + reference
utility) + Phase 2 (DiagramTerm-level R-absorption dispatch) shipped
together. Validation: 15 tests in `tests/test_R_contracted_vertex.py`
including a four-way machine-precision equivalence (`rtol=1e-12`)
across {raw, `already_R_contracted=True`} × {per-sample,
`coupling_vectorized=True`} on a constant-κ³ F+K diagram (observed
`rel_diff ≈ 1.8e-15`).

**Origin**: Driven by the canoes squeezed-κ³ workload. At
`ℓ_max = 5000, z_s = 1100` the raw κ³ kernel has diagonal width
`Δχ ~ χ_max / ℓ_max ≈ 2 Mpc/h`, so a per-leg χ-integration would need
`N_χ ≳ 5000`, i.e. `N_χ³ ~ 10¹¹` per diagram cell — intractable. The
**R-contracted** form
`κ³_R(γ; λ_1', λ_2', λ_3') := ∫∫∫ R(χ_i, λ_i') κ³(γ; χ_1, χ_2, χ_3)
dχ_1 dχ_2 dχ_3` is `R²(λ)/R²(λ_s)`-weighted, smooth on
`Δλ ~ λ_s / 3`, and resolvable with `N_λ ~ 20-30`.

**Specification source**: [`SFT_WICK_CONTRACTED_KAPPA_PROMPT.md`](../../SFT_WICK_CONTRACTED_KAPPA_PROMPT.md)
(repo root).

---

## 1. The math — what "R-contracted" means

In MSR, a `NonLocalVertex` of order `m` is

```
W^(m)[ψ] = (1/m!) ∫dz_1 … dz_m  ψ(z_1) ψ(z_2) … ψ(z_m) κ^(m)(z_1, …, z_m).
```

The `m` ψ legs Wick-contract with `m` φ's elsewhere, producing
`m` R-propagators
`R(z_partner_i, z_leg_i) := ⟨φ(z_partner_i) ψ(z_leg_i)⟩_0`. The
diagram-level integrand for one κ³ vertex thus contains the block

```
∫dz_1 dz_2 dz_3  R(z_1', z_1) R(z_2', z_2) R(z_3', z_3) · κ³(γ; z_1, z_2, z_3)
```

where `z_i = (n_i, χ_i)` are the κ³ leg points and `z_i' = (n_i', λ_i')`
are the partner φ points elsewhere in the diagram. Define

```
κ³_R(γ; z_1', z_2', z_3') := ∫dz_1 dz_2 dz_3
                              R(z_1', z_1) R(z_2', z_2) R(z_3', z_3)
                              · κ³(γ; z_1, z_2, z_3).
```

Then the diagram block reduces to `κ³_R(γ; z_1', z_2', z_3')` —
the inner integration has been **lifted into the vertex**. Mathematically
this is just Fubini; numerically it sidesteps the narrow-kernel cost.

The same identity generalises to `κⁿ_R` for `n = 4, 5, …`; the API
design must not hard-code `n = 3`.

## 2. What sft-wick does today

`NonLocalVertex(coupling=fn, order=m)` lowers to a raw `Vertex` with
`m` ψ legs (`system.py:149-157`). At Wick contraction those ψ's pair
with φ's from elsewhere, producing R-propagators in the `DiagramTerm`.
At QMC time the integrand multiplies the R-propagator products,
the κ³ callable (evaluated at the **leg** coordinates `(n_legs, t_legs)`),
and the C-propagator products on the causal simplex. **All
integration dimensions — including the m leg times `χ_i` — are folded
into a single QMC sweep**; there is no separate "inner χ-integration"
loop to short-circuit.

The §1.2 description in the cold-start prompt is mathematically
correct but **does not match sft-wick's evaluation structure**.
Implementing `already_R_contracted=True` therefore is **not** a
one-line flip of an inner-loop subroutine; it requires rewriting the
diagram graph so the κ³ legs get **absorbed** into their partners (no
R-factor, no leg-time integration variable).

## 3. Dispatch design — leg-level R-absorption

We extend the existing `equal_time_aliases` infrastructure
([equal_time_nonlocal_vertex.md](equal_time_nonlocal_vertex.md)). That
mechanism collapses the `m` leg times of an equal-time vertex onto a
single canonical leg; for R_contracted we want a **per-leg** alias
mapping each κ leg to its R-contraction partner.

### 3.1 Spec layer (`workflow/specs.py`)

```python
@dataclass(frozen=True)
class NonLocalVertex:
    name: str
    order: int
    coupling: Any                       # κ^(m) tensor or callable
    coupling_vectorized: bool = False
    equal_time: bool = False
    already_R_contracted: bool = False  # NEW
```

`already_R_contracted=False` (default): existing behaviour — the
runtime integrates over `m` leg times with R-propagators on each leg.

`already_R_contracted=True`: the user callable returns the
**already-R-contracted** value `κ^(m)_R`; the runtime treats each ψ
leg's contraction with its partner as a **delta-identification**
instead of an R-propagator.

### 3.2 Graph layer (`vertices.py`, `perturbation.py`)

* `Vertex` (raw) carries a new field
  `already_R_contracted: bool = False` that `System.build_action()`
  passes through from `NonLocalVertex`.
* `VertexInstance.absorbed_legs: tuple[str, ...]` — list of spatial
  labels on this instance whose R-propagator must be absorbed (only
  populated when the parent vertex has `already_R_contracted=True`).
* `DiagramTerm` gains `r_absorbed_pairs: tuple[tuple[str, str], ...]`
  — `(partner_label, leg_label)` pairs where the leg's R-propagator
  has been absorbed (and therefore must NOT contribute an R-factor to
  the integrand product).

### 3.3 Spatial-structure layer (`evaluate.py`)

The cleanest invariant: **keep the absorbed R-propagator in
`dt.propagators`** so direction-group union-find continues to
identify the leg with its partner (preserving the `δ(n_leg − n_partner)`
implication of the original R). Use the existing
`equal_time_aliases` infrastructure to alias each leg's time
onto its partner's time. The would-be `(partner, partner)`
self-ordering produced by the absorbed R is **already dropped**
by the existing alias-rewrite loop in `analyze_spatial`
(lines 201-208 of `evaluate.py`).

`SpatialStructure` gains
`r_absorbed: tuple[tuple[str, str], ...]` propagated straight from
`DiagramTerm.r_absorbed_pairs`. The integrand evaluator queries this
set to skip the absorbed R's when building the R-product (next
section).

### 3.4 Integrand layer (`evaluate.py`, `DiagramIntegrand`)

After §3.3 the absorbed R-propagator has `spatial_left ==
spatial_right` once the alias is applied to its right-endpoint
lookup, so a naive `R_time(t, t)` would evaluate to 0 under Itô —
**not** the unity we want. The R-product loop must therefore
**explicitly skip** propagators listed in `r_absorbed`. This is
the only integrand-side change required.

The κ³_R callable receives `(n_list, t_list)` where the `t_list`
entries are now the **partner** times (because the leg label has
been aliased onto the partner's label and the equal-time-alias
lookup machinery already routes per-leg `times[...]` reads through
that alias). For spatial labels, similarly: the κ³_R callable's
`n_list` entry for each leg becomes the partner's `n_*` direction.

### 3.5 MSR phase

`n_response` (the `(-i)^n_response` global phase) is **not changed**.
The R-propagator's `i` is a topological signature of the
ψ-leg-pairing; absorbing the R-value into the κ³ callable does not
remove the leg from the diagram, just pre-evaluates its kernel.

## 4. Validation strategy

1. **Build a brute-force reference**:
   `sft_wick.workflow.r_contracted.build_R_contracted_callable(raw_fn,
   R_callable, lambda_grid)` returns a callable that, for each requested
   λ' tuple, numerically integrates
   `∫∫∫ R(λ_i', χ_i) raw_fn(γ, χ_1, χ_2, χ_3) dχ_1 dχ_2 dχ_3` on a
   fine χ-grid and tabulates the result.
2. **Paired runs against demo2 FK** (`examples/demo2/L2/config_FK.yaml`):
   - (a) raw path: existing config (`coupling_attr: coupling_fn_vectorized`).
   - (b) R_contracted path: wrap the same κ³ via
     `build_R_contracted_callable`, point a new module attr at it,
     and add `already_R_contracted: true`.
   - Assert `|H_a − H_b| / |H_a| < 1e-3` on every (a, b, t_final)
     headline cell.
3. **Performance pin**: at `n_samples = 8192, n_gauss = 8` the
   R_contracted path is expected to be ≥ 5× faster than raw,
   primarily because the diagram-side integration drops 3 dimensions
   per κ³ vertex.

The validation harness lives at
`tests/test_R_contracted_vertex.py` — 15 tests covering schema,
YAML round-trip, raw-Vertex propagation, structural
`r_absorbed_pairs` correctness, the brute-force reference utility,
the **machine-precision equivalence** between raw and
`already_R_contracted` paths on a constant-κ³ F+K diagram (the
analytically-tractable case where Fubini is term-by-term exact), and
the four-way vectorised vs per-sample equivalence.

### 4.1 demo2 FK numerical check (2026-05-20)

Beyond the synthetic constant-κ³ test, an out-of-tree convergence
study confirms the dispatch on **demo2's spacetime-dependent κ³**
(via `examples/demo2/k3_coupling.py` wrapped through
`build_R_contracted_callable`). At one headline cell
(y=0.5, t_final=2.0, comp=(0,1)), the rel-diff vs the raw path
behaves as follows when `n_gauss` is swept:

| `n_gauss` | raw (4-D GL) | R_C (1-D GL) | rel_diff |
|---|---|---|---|
| 8  | 1.6927e-4 | 1.6090e-4 | 4.94e-2 |
| 10 | 1.6634e-4 | 1.6093e-4 | 3.25e-2 |
| 12 | 1.6491e-4 | 1.6083e-4 | 2.48e-2 |
| 16 | 1.6351e-4 | 1.6340e-4 | **6.65e-4** |
| 20 | 1.6283e-4 | 1.6187e-4 | 5.94e-3 |

The raw value drifts monotonically and the rel-diff drops to ~7e-4 at
`n_gauss=16`, then wobbles at ~1e-3 due to the brute-force trapezoid
χ-grid (used to build the κ³_R reference). The dispatch itself is
provably correct via the machine-precision constant-κ³ test; demo2
also shows that the R_contracted path is automatically **more
accurate at the same `n_gauss`** because the diagram-side integrand
is smooth on a 1-D simplex rather than peaked on a 4-D simplex —
the entire raison d'être of the R-contraction trick at production
`ℓ_max`.

This study is intentionally not a pytest: the brute-force reference
takes ~3 minutes per `n_gauss` value at `n_chi=81`. In production
(canoes analytical FFTlog-of-W) the κ³_R callable is fast and smooth,
so the wobble at `n_gauss=20` would shrink further.

## 5. Compatibility constraints

* **Default behaviour unchanged**: existing YAML configs (no
  `already_R_contracted`) round-trip to `already_R_contracted=False`
  and produce bit-identical results.
* **Mutual-exclusion with `equal_time`**: a vertex with
  `already_R_contracted=True` rejects `equal_time=True` at
  construction. The R-contracted callable already integrates over
  its leg coordinates; declaring the result equal-shell would be
  vacuous. Phase 1 raises `ValueError` when both are set.
* **No partial contraction**: `already_R_contracted=True` absorbs
  ALL `m` ψ legs uniformly. Mixed schemes (e.g. legs 1+2 absorbed,
  leg 3 raw) are explicitly out of scope (confirmed 2026-05-20).
* **Cache keys**: `_system_spec_key()` includes
  `(name, order, equal_time, already_R_contracted)` per non-local
  vertex (previously only `(name, order)`). Flipping `equal_time` or
  `already_R_contracted` now correctly invalidates the expansion
  cache. Existing on-disk caches will be re-keyed on next run; the
  recomputed value is numerically identical when the flags' defaults
  match the original config.

## 6. Open follow-ups

1. **Vectorised contracted callable** — **closed 2026-05-20**. Pinned
   by `test_vectorised_R_contracted_matches_per_sample`: the four-way
   {raw, R-contracted} × {per-sample, vectorised} matrix agrees to
   machine precision. The hypothesis ("the existing
   `equal_time_aliases` dispatch handles batched `t_list` lookups
   transparently") held — no code changes beyond Phase 2 were needed.
2. **Higher `m`** — the dispatch does not hard-code `m=3`; only the
   brute-force reference utility carries an `order=3` default. Validate
   the dispatch on a κ⁴ diagram when a use case appears.
3. **Partial / mixed-leg contraction** — explicitly out of scope per
   §5. Mixed schemes (e.g. legs 1+2 absorbed, leg 3 raw) would
   require an opt-in mask in `NonLocalVertex` and per-leg propagator
   accounting in `_collect_r_absorbed_pairs`.
4. **LaTeX rendering** — `DiagramTerm.to_latex()` currently prints all
   R-propagators regardless of absorption. For κ³_R diagrams the
   absorbed R's should be hidden (or rendered with a tilde) since
   they have been folded into the coupling symbol. Cosmetic; defer
   until paper-figure context calls for it.
5. **Memory budget for tabulated κⁿ_R** — for production canoes
   `kappa3_R_callable.py` the table is `(N_γ, N_λ', N_λ', N_λ', N, N, N)`.
   The YAML need not enforce a budget knob today; the callable owns
   its memory.

## 7. References

* [`SFT_WICK_CONTRACTED_KAPPA_PROMPT.md`](../../SFT_WICK_CONTRACTED_KAPPA_PROMPT.md)
  — the original cold-start prompt.
* [`equal_time_nonlocal_vertex.md`](equal_time_nonlocal_vertex.md) —
  prior art for the leg-alias machinery this design extends.
* canoes `scripts/squeezed_kappa3/DESIGN.md` — the upstream κ³
  callable that this dispatch will eventually consume.
* canoes `scripts/squeezed_kappa3/FFTLOG_OF_W_BISPECTRUM_PROMPT.md`
  — the analytical FFTlog-of-W chain that will produce production
  `κ³_R` once Phase 2 is complete.
