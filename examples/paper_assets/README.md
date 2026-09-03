# Assets for the CPC paper (referee revision, sft-wick 0.3.0)

Everything here was produced on the release candidate of 0.3.0 (Apple M3
Ultra, 28 cores; conda env `sft-wick`, Python 3.14).  Each directory has a
script that regenerates its contents.

## `table1/` — Table 1 and the order-1 diagrams

`generate_table1.py` reproduces the scaling table with three corrections
to the submitted version: for the cubic vertex `F_abc φ_a φ_b ψ_c` the
operator count is **3n + m**, not 2n + m; the observable alternates with
the parity of n (⟨φ_a(x)φ_b(y)φ_c(z)⟩ for odd n, ⟨φ_a(x)φ_b(y)⟩ for even
n), so the table now carries an observable column; and **the odd-order
diagram counts change**, because the submitted ones were produced with
two externals sharing the spatial label `x`.  Wall-clock is to three
significant digits (best of three, single core).

| Order n | Observable | Operators 3n+m | Raw pairings (3n+m−1)!! | Distinct diagrams | Wall-clock (s) | Reduction |
|---|---|---|---|---|---|---|
| 1 | ⟨φ_a(x)φ_b(y)φ_c(z)⟩ | 6 | 15 | 6 | see below | 2 |
| 2 | ⟨φ_a(x)φ_b(y)⟩ | 8 | 105 | 6 | see below | 18 |
| 3 | ⟨φ_a(x)φ_b(y)φ_c(z)⟩ | 12 | 10395 | 80 | see below | 130 |

The submitted table gave 4 and 75 at orders 1 and 3 (reductions 4 and
139).  Those were the artefact of the label collapse that 0.4.0 made a
`ValueError`: same-label externals lose the sum over assignments of
externals to legs.  `3n+m` and the raw pairing count are unaffected —
`m` is 3 either way — and order 2 never used a repeated label, so its
row is unchanged.  See the 0.4.2 CHANGELOG entry, and
`tests/test_coincident_external_labels.py::test_CE3_*`, which pins these
counts so the asset cannot silently drift again.

**Wall-clock is deliberately not quoted here.**  It must be re-measured
on an otherwise idle machine: the numbers this script last emitted came
off a heavily loaded one, and the order-3 cell now times an 80-diagram
enumeration rather than a 75-diagram one, so it is not comparable to the
submitted value even after re-measurement.  Run the script and take the
column from `table1.md`.

Files: `tab_scaling.tex` (drop-in replacement for the table in
`main.tex`; the paper repository's `generate_figures.py::generate_table`
should take its header from it), `table1.md`.

The six order-1 diagrams of row 1 are three pairs, one per choice of
which external carries the `R` leg, each pair splitting into a tree
(`R(x;w)C(y;w)C(w;z)`, coupling `F_{i₀i₁i₂} + F_{i₁i₀i₂}`, pairing
multiplicity 2) and a tadpole (`R(x;w)C(y;z)C(w;w)`, coupling
`F_{i₀i₁i₂}`, multiplicity 1).  Wick's theorem pairs the six operators in
fifteen ways; the six that pair the vertex's ψ with one of its own φ's
vanish under the Itô prescription (`R(x, x) = 0`), leaving nine, which
group into these six topologies with multiplicities 2, 1, 2, 1, 1, 2,
folded into the rational prefactors.  ⟨ψψ⟩ = 0 removes nothing at order
1 — the expression contains a single ψ, so no ψψ pairing exists; it
starts pruning at order 2.  (Checked by re-running with `ito=False`,
which restores all fifteen.)  Files:
`order1_diagram_{1..6}.tex` (TikZ, the styles of
`fig_example_diagrams_order2.pdf`: blue solid C, red dashed R, circles
for external points, squares for vertices), `order1_diagram_{k}_standalone.pdf`
(compiled), `order1_diagrams.pdf` (matplotlib grid with multiplicities),
`order1_diagrams.md` (each term's full LaTeX).

## `demo2_kappa4/` — the κ⁽³⁾ example, publication grade, with an error budget

Parameters throughout: α = 0.6, λ = 0.05, σ_t = 0.3, σ_x = 1, γ = 1,
N = 2, F[0,1,1] = 1, F[1,0,1] = F[1,1,0] = 1/2.

### What was wrong in the submitted version

Re-deriving the example turned up four things, all fixed in
`examples/demo2` (see the 0.3.0 CHANGELOG):

1. **κ⁽³⁾ lacked its α³ term.**  ⟨η̃³⟩ = 6αλ² + 8α³λ³, not 6αλ²; the
   simulated third moment (9.211e-3 over 2M realisations) is the former
   (9.216e-3).  Spatially, κ⁽³⁾ = 2αλ²[k₁₃k₂₃ + k₁₂k₂₃ + k₁₂k₁₃] +
   8α³λ³k₁₂k₂₃k₁₃.  FK moves by +1.2 % (t = 1) to +0.6 % (t ≥ 3).
2. **The FK quadrature was not converged for t ≳ 10.**  The raw kernel is
   narrow in the relative leg times; the 8-node tensor rule on the 4-D
   integral (config_FK.yaml, v1) is 10 % high at t = 3.5, 45 % high at
   t = 15 and 46 % low at t = 50 (ξ₀₁, r = 0).  Remedy: the package's
   `already_R_contracted` vertex with `examples/demo2/k3_R_coupling.py`,
   which integrates the three legs inside the callable (analytically in
   the common time, composite Gauss-Legendre on the two relative times
   with every cusp aligned to the grid).  The FK integral is then
   one-dimensional; 32 nodes agree with 64 to 1e-4 at every t and with
   the raw kernel's 40-node rule (t = 1) to 5e-4.  Validated against
   QMC of the raw leg integral to 1e-4.
3. **ξ₀₀ / ξ₁₁ used the single-kernel λ_eff approximation.**  The exact
   effective covariance is λk + 2α²λ²k² (second piece: half the
   correlation time and length); λ_eff k over-counts it and is +1.8e-4
   high on ξ₀₀ at large t (1.5 %, 2σ of the 200k simulation).  Now the
   exact form, a sum of two built-in closed forms
   (`examples/demo2/L2/c_closed_form.py::C_fn_eff_exact`).
4. **The simulation cache was run at Δt = 0.05.**  The first three
   measurement times sit 3–8 steps in and the recorded times were
   rounded to the step.  The budget uses new runs at Δt = 0.02 and 0.01
   (`sim_dt_study.py`, 20 seeds × 100 000 realisations each = 2M per
   step, measured at exactly the theory times, Richardson-extrapolated
   with Heun's O(Δt²)); the Δt bias itself is below the Monte-Carlo
   error at t ≥ 0.4.

### κ⁽⁴⁾

`k4_coupling.py`: κ⁽⁴⁾ = 4α²λ³ Σ_{12 Hamiltonian paths} k k k +
16α⁴λ⁴ Σ_{3 cycles} k k k k (Wick on Gaussian η; checked against Monte
Carlo: single site 2.199e-3 vs 2.1989e-3, four distinct times 2.50e-5 vs
2.46e-5 ± QMC).  Counting ψ and φ legs, **F·κ⁽⁴⁾ at order 2 vanishes** for
⟨φφ⟩ (2 + n_F − 4 n_K must be a non-negative even number); the leading
term is **F·F·κ⁽⁴⁾ at order 3**, three pure-R⁶ diagrams, evaluated with
the R-contracted `k4_R_contracted.py` (12 + 3 terms, each in
coordinates that align its cusps; 12-node outer rule, checked against 16
to 2e-2 at t = 15 and 5e-4 at t ≤ 3.5).  And, because the Gaussian
theory with this F has a φ₁ → −φ₁ symmetry that only odd cumulants
break, **FF, FFFF and FFK4 are identically zero for ξ₀₁**: the κ⁽³⁾
signal there is clean, and κ⁽⁴⁾ cannot touch its residual.  κ⁽⁴⁾ does
enter ξ₀₀ / ξ₁₁: 5e-6 at t = 1, 1.9e-5 for t ≥ 3.5 (r = 0).

### The error budget (`budget.md`, `budget.npz`, `run_budget.py`)

Theory channels on the simulation's time grid (18 times, 0.1 → 50) and
separations (12): order 0 and FF with the exact C_eff (QMC, 32768
samples), FK (R-contracted, converged), FFK4 (r = 0, 0.5), FFFF (order 4,
64 diagrams, r = 0, 0.5).  Simulation: 2M realisations per step size,
Δt → 0 extrapolated; Monte-Carlo error 1.3e-5 on ξ₀₁ and 1.9e-5 on ξ₀₀
at large t.

* **ξ₀₁ (r = 0)**: simulation − FK is 0 within 1σ for t ≤ 1, then
  grows to +5.2e-5 at t = 3.5 and +8–9e-5 (24 % of FK, 6–7σ) for
  t ≥ 5.  This is **truncation**: the order-4 F³κ⁽³⁾ channel, which the
  dynamic-coupling path cannot evaluate (a κ⁽³⁾ index sits on a C
  propagator), estimated with κ⁽³⁾ collapsed to an equal-time constant
  and calibrated on FK at order 2, is 3e-6 (2 % of FK) at t = 1, 4.1e-5
  (12 %) at t = 3.5 and 8.1e-5 (24 %) at t = 15 — the size, sign and
  t-dependence of the residual.  The estimate carries a ~50 %
  uncertainty of its own (calibration ratio 0.42–0.64).  Finite Δt and
  the Monte-Carlo error are an order of magnitude smaller.  The
  submitted figure's ~8e-5 residual at t ≈ 3 was a coincidence of the
  un-converged quadrature (+10 %), the missing α³ term (−1 %) and this
  truncation.
* **ξ₀₀ (r = 0)**: with the exact C_eff, simulation − (0 + FF) is
  +2e-5 to +1.2e-4 (up to 6σ); adding FFK4 (2e-5) and FFFF (4–9e-5)
  leaves −4e-5 … +7e-5, i.e. within ±2σ except t = 5–5.4 (+3σ), where the
  order-4 FFFK channel (not computed for ξ₀₀) is the natural remainder.
  The λ_eff approximation of the submitted version is −1.0e-4 to
  −1.5e-4 off for t ≥ 1.
* Full tables per pair and separation, χ² per channel, quadrature
  checks and the noise-cumulant cross-check are in `budget.md`.

### Figures (paper-ready, matplotlib rcParams of the demo2 notebook)

* `xi01_vs_time.pdf` — ξ₀₁(r = 0, t): simulation vs converged FK, with
  the v1 8-node rule for comparison and a residual panel.
* `xi01_vs_r.pdf` — ξ₀₁(r) at t = 3.48 and 15.
* `xi00_vs_time.pdf` — ξ₀₀(r = 0, t): simulation vs 0 + FF (+ FFK4 +
  FFFF), residual panel including the λ_eff approximation.
* `fig_fk_diagrams.tex`, `fk_diagram_{1,2}.tex` / `_standalone.pdf` —
  the two FK diagrams (TikZ); `fk_diagrams.md` their LaTeX terms.

### Timings (28 workers unless stated)

Theory: 0 + FF 9 s, FK (R-contracted, GL32, all 648 grid points) 7 s,
FFK4 (R-contracted, GL12, 108 points) 224 s, FFFF 35 s.  Simulation:
100 000 realisations take 100 s at Δt = 0.02 and 195 s at Δt = 0.01 on
one core (`sim_dt_study.py`); the 40 runs were done in parallel.
`examples/demo2/L2/reproduce_figures.py` on the corrected configs: 39 s
wall.

## Regression check of the demo1 paper figures

`examples/demo1/L2/config.yaml` (the sweep behind `xi_vs_time`,
`comparison_multi_time`, `xi_vs_order`: 672 grid points × 71 diagrams,
32768 QMC samples) was run on v0.2.0 (what the referees had) and on the
0.3.0 release candidate.  Over all 2016 `(y, t_final, a, b, order)`
totals the maximum relative change is **3.3e-14** (order 2; 1.2e-14 at
order 0, 2.8e-14 at order 4) — floating-point noise.  The July changelog
claimed the closed-form path was unaffected by the C-table diagonal fix;
it is.  Wall-clock 4.6 min (28 workers) on the release candidate.

## Timings of the shipped examples

| Config | M3 Ultra, one core | Laptop proxy (`taskpolicy -c background`, `OMP_NUM_THREADS=1`) |
|---|---|---|
| `sft-wick quickstart` | 2 s | 10 s |
| `examples/demo1_config.yaml` (= README v1 quick start, orders 0–4, 8192 samples) | 50 s | 5.2 min |
| `examples/demo1/L2/config.yaml` | 4.6 min (28 workers) | — |
| `examples/demo2/L2/*.yaml` + figures | 39 s (28 workers) | — |

For comparison, v0.2.0 on the README v1 quick start: dblquad C table
at 74–245 ms per call × 14 400 calls, not finished after 4 min, projected
> 1 h — the referee's experience.
