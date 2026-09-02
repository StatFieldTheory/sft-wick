# Demo 3 — what is validated, how, and what is not

All numbers below were produced on base commit **`ac7f201`** ("feat:
propagator-indexed dynamic couplings"), `sft-wick` 0.3.0, on an M3 Ultra
(28 cores).  The parallel `demo2-hardening` branch is changing
`src/sft_wick/`; anything quoted here that later disagrees is
attributable to that.

## 1. The physics

The driving field is a **filtered Poisson (shot) process**.  Independently
for each component `a`, events `(x_k, s_k)` are drawn from a Poisson
process of rate `ν` per unit length × time over the whole line and the
whole past, and

    η_a(x, t) = Σ_k h · w(x − x_k) · g(t − s_k) − ⟨·⟩,
    g(τ) = Θ(τ) e^{−τ/σ_t},   w(x) = e^{−|x|/σ_x}.

Campbell's theorem gives **every** cumulant in closed form, all of the
same shape, and all factoring through the single source point `(x′, s)`:

    κ_m(z_1 … z_m) = δ_{a_1…a_m} · ν h^m · X_m(x_1…x_m) · T_m(t_1…t_m),
    X_m = ∫dx′ Π_i w(x_i − x′),      T_m = ∫ds Π_i g(t_i − s).

**What makes it non-Gaussian, and by how much.**  With `n ≡ ν σ_t σ_x`,

    skewness = (4√2/9)/√n ≈ 0.6285/√n,     excess kurtosis = 1/(2n),

and `n → ∞` is the Gaussian limit.  `n` is the *only* knob: `h` is fixed
to hold `κ₂ = n h²/2` constant, so a sweep in `n` moves the non-Gaussian
channels and leaves the Gaussian sector — and every other scale —
untouched.  The headline point is `n = 1` (skewness 0.63, comparable to
demo 2's 0.76), with `σ_t = 0.5` and `γ = 1`: the same order of magnitude,
so there is no scale separation to strain the quadrature, but `γ ≠ 1/σ_t`
so the closed forms stay away from their removable singularity.

**The observable.**  With `N = 2` and `F[0,1,1] = 1`, `F[1,0,1] =
F[1,1,0] = ½` (demo 2's structure, deliberately), the drift is invariant
under `φ₁ → −φ₁`.  Were the noise law also invariant under `η₁ → −η₁`,
the cross-correlator `ξ₀₁ = ⟨φ₀(x,t) φ₁(y,t)⟩` would vanish identically.
Shot noise is skewed, so `ξ₀₁` is driven by the **odd** cumulants of `η₁`
alone: order 0, FF, FFFF and the entire `κ⁴` channel cancel.  There is no
Gaussian background to subtract — the observable *is* the non-Gaussian
signal.

**Why the R-contracted vertex is exact here.**  Because every cumulant
factors through one source time, the `m` leg integrals commute with it and
collapse to a single one-dimensional integral,

    K_R(t′; x′) = ∫ Π_i du_i R(t′_i, u_i) κ_m(u; x′) = ν h^m X_m(x′) T̃_m(t′),
    T̃_m(t′) = ∫ ds Π_i J(t′_i, s),   J(t′,s) = ∫_0^{t′} R(t′,u) g(u−s) du,

with `J` elementary and `T̃_m` a `2^m`-term closed form.  The construction
is **m-agnostic** — which is what makes the neglected-cumulant ladder
computable rather than merely boundable (§3).  Demo 2's R-contracted `κ³`
needs a cusp-aligned composite rule accurate only to ~1e-6.

## 2. Validation ledger

| claim | independent reference | agreement achieved | what limits it |
|---|---|---|---|
| `X_m` (spatial overlap), m = 2…6 | adaptive quadrature of `∫dx′ Π w` | ≤ 1e-15 rel | float round-off |
| `κ₂, κ₃` formulas | direct Monte Carlo of the **event process** | within 4σ, SE < 10 % of signal | MC statistics |
| `κ₄` (connected) | same | within 4σ | MC statistics |
| `κ₂ = ν h² X₂ T̃₂` | the package's own `ClosedFormC` (independent derivation) | 1.2e-16 rel | float round-off |
| `T̃_m`, m = 3 | 3-D adaptive quadrature of the raw leg integral, split by which `u_i` is smallest | 1e-15 rel | reference tolerance |
| `T̃_m`, m = 4 | 4-D tensor GL on the same min-split regions | 1e-12 rel | reference convergence |
| `t_tilde` branch dispatch | the **same** closed form in 60-digit `mpmath`, at 9 values of `γ/a` × 8 extreme `t′` corners | ≤ 1e-10 rel everywhere, incl. `γ = 1/σ_t` exactly | by construction (§3) |
| **Level A: `⟨φ³⟩` package vs closed form** | closed form (level A is a *single* diagram) | **1.4e-16** rel, coincident and at unequal positions | float round-off |
| **Level A: connected `⟨φ⁴⟩`** | same | **6.6e-16** rel | float round-off |
| `already_R_contracted` on a non-constant kernel | the same observable with the **raw** `κ³` vertex (3-D quadrature) | 2e-4 rel at QMC `2^18` | the raw path's own convergence |
| Level A vs **event-exact simulation**, `⟨φ³⟩(t)` | simulation with *no* Δt and *no* spatial discretisation | all pulls ≤ 0.64σ over 1.2e6 realisations | Monte Carlo (0.9 % of signal) |
| Level A, connected `⟨φ⁴⟩(t)` | same | all pulls ≤ 1.12σ | Monte Carlo (3.9 % of signal) |
| `1/√n` law | closed form and simulation across `n ∈ {0.25, 1, 4}` | ratio **4.0000** vs 4.0000 predicted | exact |
| **Level B: order-2 `Fκ³` channel** | an independent 1-D integral: because the two components are driven by *independent* event processes, one of the two first-order terms vanishes identically and `ξ₀₁^{FK}(t) = s∫_0^t R(t,u) K_R(u,u,t) du` | ≤ 7e-7 rel (limited by the quoted reference digits) | — |
| Level B order-2, integrator independence | Gauss-Legendre `n = 16/32/64` vs QMC `2^18` | 2e-9 rel | — |
| Level B order-4 (`F³κ³`, `F³κ⁵`) | node-count convergence on **this demo's worst cell** (`t = 3`), `n_gauss` 8→22 | monotone; `n = 12` is within **1.1e-5** of `n = 22` for `F³κ³` and 4e-8 for `F³κ⁵` | GL convergence |
| **Level B: `ξ₀₁(t)` vs simulation** | ETD simulation, 6 independent seeds (6 × 4e5) | all six times within **0.90σ**, relative deviations ≤ **0.79 %**, χ²/dof 0.29–1.54 | Monte Carlo |
| Level B: `ξ₀₁(r)` vs simulation | same, sites placed exactly at each `r` | all pulls ≤ 1.35σ | Monte Carlo |
| Level B: `ξ₀₀` (even sector) | same, against order 0 + FF + `F²κ⁴` | all pulls ≤ 0.9σ | Monte Carlo + order-4 truncation |
| ETDRK2 discretisation | paired Δt study at **identical events** (same seed, same `n_real`, so the same events are drawn) | ratios 4.0–5.1 vs the 4× expected for `O(Δt²)`; residual is **0.01×** the MC error | — |
| L2 (YAML/CLI) vs L1 (Python) | `sft-wick run config_FK.yaml` / `config_F3K.yaml` against `level_b.py` | ≤ 2e-6 rel (quoted-precision limited) | — |

Monte-Carlo caveat that applies throughout: within one panel every point
shares the same realisations, so residuals are **strongly correlated**.  A
coherent offset across a curve is *one* fluctuation, not one per point.

## 3. Error budget

**Absent by construction** — these are not small, they are zero:

| source | why it is absent |
|---|---|
| spatial discretisation | the drift has no spatial derivative, so sites couple only through the noise; a finite set of sites is an *exact* realisation |
| interpolation in `r` | sites are placed exactly at the plotted separations (this is what biases demo 2's off-grid `r` by +0.5 %) |
| time stepping (level A) | the linear response to an exponential pulse is analytic per event |
| `C`-propagator quadrature | `κ²` is `SeparableTranslation` × `ExponentialTemporal`, so the built-in closed form applies (`c_source == "closed_form:builtin"`); no `dblquad`, no spline table, and the GL-vs-`dblquad` dispatch is never reached — which also means demo 3 is immune to the timing race in `_gl_is_cheaper` that can make `c_source` load-dependent for quadrature-path systems |
| leg-integral quadrature | the `m` leg integrals are done analytically (§1) |
| contamination of level A by other cumulants | a `κ^(m′)` vertex with `m′ ≠ m` cannot balance the legs |

**Present, and sized:**

| source | size | how it is controlled |
|---|---|---|
| event-window truncation | **7.6e-11** relative (m = 3) | analytic: the m-th cumulant is an m-fold product of pulse profiles, so the discarded region costs `e^{−mL/σ_x}` — *m* times faster than the field's own tail. The subtracted mean uses the **same truncated window**, so it contributes no bias at all |
| `T̃_m` closed-form cancellation | ≤ 1e-12 by construction | the closed form reports its own conditioning (`ε·Σ|term|/|result|`, including the `exp` argument amplification) and the dispatcher recomputes any sample above tolerance by quadrature. Verified against `mpmath` to be an upper bound |
| order-4 Gauss-Legendre (`F³κ³`, `F³κ⁵`) | 1.1e-5 rel at `n_gauss = 12` (vs `n = 22`), i.e. **8e-7 of the total** | measured on this demo's own worst cell (`t = 3`) rather than inherited; the sequence is monotone in `n`, so the limit is trustworthy |
| order-2 Gauss-Legendre (`Fκ³`) | machine precision | already converged at `n_gauss = 8`; the configs use 32. The "low order needs the finer grid at large `t_f`" trap that caught demo 1 does not bite here — demo 3's largest `t_f` is 3, not 100, so the integrand never develops a narrow peak relative to the simplex |
| **truncation of the `F` series** | `F³κ³` = 7.9 % of `Fκ³` at `t = 3` — **computed, not estimated** | computed exactly (30 diagrams, 3 time-integration variables) |
| **neglected-cumulant ladder** | `F³κ⁵` = 0.094 % of `Fκ³` — **computed** (6 diagrams) | closed-form ladder ratio `κ_m/κ₃ = h^{m−3}(3/m)²`; `h` is free, so it can be shrunk at will |
| uncomputed `O(F⁵)` | ≈ 0.63 % of `Fκ³` at `t = 3` | geometric estimate `(F³κ³/Fκ³)²`; falls as `s²` |
| ETDRK2 discretisation (level B) | **0.01×** the MC error | paired Δt study at identical events; convergence ratio 4.0–5.1 confirms `O(Δt²)` |
| Monte Carlo, level A | 0.9 % (`⟨φ³⟩`), 3.9 % (connected `⟨φ⁴⟩`) | batch scatter over ≥ 20 batches |
| Monte Carlo, level B | 0.7–1.0 % on `ξ₀₁` from 6 × 4e5 realisations | inverse-variance combination, with χ²/dof reported — see below |
| blow-up of the quadratic drift | **0 diverged trajectories out of 7.3e6 integrated** | every trajectory the script integrates is checked — the `t` sweep, the paired Δt study, the separation sweep and the amplitude scan — and the count is aggregated, not quoted per run |

**The `ξ₀₁` estimator is heavy-tailed, and that has to be handled
explicitly.**  `ξ₀₁` is a product of two heavy-tailed fields, and at small
`t` — where the field has accumulated fewest events and is furthest from
Gaussian — a *single seed out of six* can land several σ from the rest.
In one set of six seeds the per-`t` χ²/dof reached **3.4 at `t = 0.5`**,
falling to 1.3 at `t = 3`, with a seed-scatter / batch-scatter error ratio
of 1.82 → 1.07.  It is tempting to read that as the batch-scatter error
being optimistic by a `t`-dependent factor — which is what we first
concluded, and it is **wrong**.  Removing the single most deviant seed
takes χ²/dof to 1.30 and the ratio to 0.99 at `t = 0.5`, and to 1.02 /
1.00 at `t = 1`: the apparent inflation was one outlying seed.  With only
6 seeds the ratio carries ≈ 30 % uncertainty of its own, so 1.82 and 1.41
are not even distinguishable.  The shipped run (a different seed set)
gives χ²/dof 0.29–1.54 throughout, i.e. no anomaly at all.

The consequences for how demo 3 reports level B: seeds are combined by
**inverse variance** (an outlying seed carries a correspondingly large
error and is down-weighted automatically) rather than by a plain scatter,
and `level_b.py` prints χ²/dof and a drop-the-most-deviant-seed column
alongside every number, so the reader can judge the error bars instead of
trusting them.  A separate, real bug was found the same way: the per-batch
means were being averaged with **equal weight**, so a realisation count
that did not divide evenly left a remainder batch of eight realisations
carrying the same weight as thirteen thousand — a 20 % shift.  Batches are
now size-weighted (`simulate_b._weighted`), locked by
`test_control_variate_weights_batches_by_size`.  (Checked against demo 2 by the `demo2-hardening` session:
its `ξ₀₁` errors show no such behaviour — median seed-scatter ratio 1.06,
χ²/dof 0.34–1.62 — plausibly because its AR(1)-driven field is far closer
to Gaussian at small `t` than a shot-noise field with few events.  The
diagnosis of the mechanism stands; it simply does not bite there.)

**A warning that belongs in the budget.** `F³κ³` and `F³κ⁵` enter at the
same order and carry the same `F³` scaling, so the amplitude-scaling test
**cannot** distinguish them.  Only computing one of them can.  Do not read
the fitted `s³` exponent as evidence that `κ³` alone is responsible.

## 4. What is NOT established

* **The `O(F⁵)` remainder is estimated, not computed** (≈ 0.6 % at
  `t = 3`).  It is bounded only by the geometric argument and by the
  observed `s³` scaling of the residual.
* **Level B is not exact**, unlike level A.  It is a truncated series
  compared against a simulation whose `F` term is discretised.
* **The `1/√n` law is exhibited at three values of `n`**, not fitted over
  a range.
* **The amplitude-scaling exponent is not a precision measurement.**  The
  residual after subtracting `Fκ³` is consistent with `F³` but is
  MC-dominated at the amplitudes used; and it could not distinguish
  `F³κ³` from `F³κ⁵` even if it were precise.
* **`ξ₀₀` is truncated at order 3** (order 0 + FF + `F²κ⁴`).  The omitted
  order-4 terms are ~5e-5, below the MC error (~5e-4), but they are not
  computed.
* **The raw-vs-R-contracted check reaches 2e-4, not machine precision** —
  the raw path's 3-D integrand kinks on the `u_i = u_j` planes, so tensor
  GL converges only at order ≈ 2 there (measured 1.97) and QMC is the
  honest comparand.
* **`γ = 1/σ_t` is validated but not used.**  The demo runs at
  `|γ−a|/max(γ,a) = 0.5`.

### Known package defects and limitations (all pre-existing on `ac7f201`)

1. **Coincident external spatial labels.**  Repeating a label across
   external operators loses pairing multiplicity.  For the level-A
   3-point diagram, distinct labels give the coupling sum
   `K_abc + K_acb + K_bac + K_bca + K_cab + K_cba` while
   `("phi_a(x)","phi_b(x)","phi_c(x)")` gives `K_abc` alone — exactly
   `1/6` here.  A blanket factor 6 would be *wrong* for a coupling that is
   not symmetric under all six index permutations, so the hardened branch
   **refuses** the spelling rather than repairing it.  Demo 3 uses
   distinct labels with equal positions throughout; no number is affected.
2. **`n_components = 1` with a callable coupling raises** in
   `_sum_coupling_batched`.  Demo 3 uses `N = 2` throughout.
3. **`Expansion.sweep` is 2-point only** (`for (a, b) in
   component_pairs`), so level A cannot be driven from the L2 CLI even
   though `Expansion.evaluate` accepts a triple.  Both YAML configs are
   therefore level B.
4. **`to_feynman_diagram()` UID collision** for the 3-point / order-1 /
   `K3`-only expansion: observable operators and the vertex instance are
   both allocated uids 0,1,2.  Affects *rendering only* — the level-A
   diagram cannot be drawn, while its numbers are exact.  Level B renders
   42/42.
5. **L2 spatial-kernel gap.**  `config.py::_build_kernel(axis="space")`
   accepts only `exponential` and `gaussian`, and demo 3's envelope
   `σ_x(1+r/σ_x)e^{−r/σ_x}` is neither, so `κ²` cannot be declared as
   `SeparableTranslation` in YAML.  The configs use the general
   `callable_module` hatch plus a closed-form `C`; the L1 API needs
   neither.
6. **Plain functions in kernel spec fields silently defeat the caches.**
   `_system_spec_key` builds the expansion key from `repr(noise.kappa2)`
   and `repr(linear)` (and `_minimal_propagator_spec` likewise), and
   `repr()` of *any* plain function embeds its memory address — so the key
   changes on every process.  This affects every spec field that can hold
   a function: `CustomKernel`, `GeneralKappa2`, `CustomImpulse`,
   `ExplicitR`, and a callable `DiagonalA.gamma`.  A module-level `def` is
   affected **exactly as much as a lambda**; the fix is a callable
   *object* with a stable `repr`, i.e. a frozen dataclass with
   `__call__`, which is what demo 3 uses (16.7 s → 1.5 s on the second
   call).  Non-local **vertex couplings are not affected** — they enter
   the key only as `(name, order, equal_time, already_R_contracted)`,
   correctly, since the symbolic enumeration does not depend on coupling
   values.  The failure is fail-safe: an unstable `repr` can only cause a
   cache *miss*, never a wrong hit.  (Diagnosis and the cross-process
   measurements are the `demo2-hardening` session's; this demo's own
   speed-up came entirely from the `CustomKernel` change.)

## 5. The case for the paper

Demo 3 is the demonstration demo 2 was trying to be.  Its non-Gaussianity
is controlled by a single dimensionless number with the rest of the model
held fixed, so the claim "this channel is non-Gaussian" is testable by
watching it vanish as `1/√n` while the Gaussian channels do not move.  Its
free-field limit is not a weak consistency check but an **exact** one: the
`m`-point function is a single diagram equal to a closed form, reproduced
by the package to 1.4e-16 and by a simulation that has no discretisation
error of any kind, only Monte-Carlo error that is quoted.  And where demo
2 could only estimate the leading correction and had to leave the
cumulant ladder as an unexamined assumption, demo 3 computes both — the
`F³κ³` correction and the `F³κ⁵` ladder term — leaving a single, named,
geometrically-bounded `O(F⁵)` remainder below 1 %.  The whole
demonstration reproduces in well under an hour of CPU.

## 6. Timings (M3 Ultra, 28 cores)

| step | 28 cores | single core |
|---|---|---|
| `pytest tests/test_demo3_*.py` (220 tests) | 9 s | 9 s |
| `sft-wick run config_FK.yaml` | 0.6 s | 0.6 s |
| `sft-wick run config_F3K.yaml` (cold; includes the order-4 enumeration) | 178 s | 178 s |
| order-4 enumeration, `κ³ + κ⁵` (cached afterwards) | 143 s | 143 s |
| order-4 evaluation, 36 diagrams × 3 time points | **2.9 s** | **18.0 s** |
| `level_b.py` theory section (cache warm) | 11 s | ~35 s |
| `level_a.py` (1.2e6 realisations) | 180 s | 180 s |
| `level_b.py` full (2.4e6 + 0.9e6 + 1e6 + 3e6 realisations) | ~6 min | ~6.5 min |

The **only** step that uses more than one core is the order-4 diagram
integration (`n_jobs=-1`, 36 independent 3-D Gauss-Legendre integrals),
worth 6.2×; the enumeration, the propagators and every simulation are
single-core as quoted.  End to end (both levels, all figures, the full
test suite and both CLI configs) the demonstration reproduces in **about
twelve minutes** of wall clock, or about twenty on one core.
