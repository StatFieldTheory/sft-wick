# Demo 2 — what it demonstrates, and what it does not

*Companion to `examples/paper_assets/demo2_kappa4/budget.md`, which
carries every number quoted here.  Written in a fixed shape so it can be
set against demo 3's.*

---

## 1. The physics

Two coupled fields `φ = (φ₀, φ₁)` on a line, driven by noise that is
**not Gaussian**:

```
∂ₜφ₀ = −φ₀ + φ₁² + η̃₀        η̃ₐ = ηₐ + α(ηₐ² − λ)
∂ₜφ₁ = −φ₁ + φ₀φ₁ + η̃₁       η  = Ornstein–Uhlenbeck, σ_t = 0.3, σ_x = 1
```

with `α = 0.6`, `λ = 0.05`, `γ = 1`.  The quadratic deformation of the
noise is what makes the problem non-Gaussian: it gives `η̃` a full ladder
of cumulants, `κ⁽²⁾ = 5.180e-02`, `κ⁽³⁾ = 9.216e-03`,
`κ⁽⁴⁾ = 2.199e-03`, `κ⁽⁵⁾ = 6.573e-04`, … .  Everything a Gaussian
theory can produce is even; the odd cumulants are the signal.

**The observable that isolates it is `ξ₀₁ = ⟨φ₀(0)φ₁(r)⟩`.**  Order 0,
the `F·F` channel, the `κ⁽⁴⁾` channel and `F⁴` all vanish there, so
`ξ₀₁` is *nothing but* non-Gaussianity: at leading order it is the
`F × κ⁽³⁾` diagram pair, 3.44e-04 at `t = 15, r = 0`, against a `ξ₀₀` of
1.23e-02 that is 96 % order 0.  This is the point of the demo.  The
deformation is a small perturbation of the noise — it shifts the
variance by `2α²λ = 3.6 %` — and `ξ₀₁` is 2.8 % of `ξ₀₀`, but **100 % of
`ξ₀₁` is non-Gaussian**: there is no Gaussian contribution to subtract.
That is what makes it a clean test, and what makes a 1e-05 residual in
it worth arguing about.

---

## 2. Validation ledger

| claim | independent reference | agreement achieved | what limits it |
|---|---|---|---|
| `κ⁽³⁾ = 6αλ² + 8α³λ³` | exact Isserlis algebra | rel. 3.3e-16 | machine precision |
| `κ⁽⁴⁾ = 48α²λ³ + 48α⁴λ⁴` | exact Isserlis algebra | rel. 2.3e-15 | machine precision |
| the whole ladder `κₙ = n![uⁿ/2n + (λ/2)uⁿ⁻²]`, `u = 2αλ` | generating function; 40M-sample MC | exact for n ≤ 4; 0.6 % for `κ⁽⁵⁾` | MC statistics of the reference |
| `κ⁽²⁾`, `κ⁽³⁾`, `κ⁽⁴⁾` as *simulated* | the simulator's own recorded moments, 2M × 18 times | 3e-4, 5e-4, 1e-3 relative | simulation MC |
| R-contracted `κ⁽³⁾` kernel (`k3_R_coupling.py`) | cusp-aware 3-D adaptive quadrature of the raw leg integral | **1e-4** over the range that matters; 1.7e-6 at the FK configuration; 2.6e-3 at `t′ = 0.1` | fixed composite panel edges when `t′ < σ_t`; harmless (1.7e-08 absolute) |
| R-contracted `κ⁽⁴⁾` kernel | randomised-Sobol QMC of the raw 4-leg integral | 2.6e-4 – 2.2e-3; 9.9e-3 at `t′ = 0.2` | the reference's own scatter; at `t′ = 50` QMC has 80 % seed scatter and cannot test it at all |
| `already_R_contracted` route, constant `κ⁽³⁾` | analytic factorisation | 1.8e-15 | machine precision |
| `already_R_contracted` route, **real** `κ⁽³⁾` | the raw kernel through the 4-D rule, refined | raw converges *onto* it monotonically: 2.9e-3 → 9.9e-4 → 4.1e-4 as GL10 → GL16 → GL22 | the raw rule's own convergence (2.7e-4 at GL22) — the residual disagreement *is* that |
| FK channel quadrature | GL32 vs GL64 | 8.3e-5 relative | — |
| FFK4 channel quadrature | GL12 vs GL16 | 1.8e-2 at `t = 15` = **3.4e-07 absolute** | — |
| **FFFK** channel quadrature | GL8 vs GL10 vs GL14 | 0.1 % to `t = 5.4`; **1.1 % (6.1e-07) at `t = 15`**; 9.2 % at `t = 50` | the 3-D rule loses the peak at large `t_f`; the channel saturates, so the late-time value is the saturated one |
| FFFF channel quadrature | GL10 vs GL14 | 2.2e-06 absolute | replaced 32768-sample Sobol QMC, which was wrong by **4.8e-05 (121 %)** — as large as the residuals it interpreted |
| simulation step size | dt = 0.02 vs 0.01, Richardson | step = 1.55e-05 ± 1.4e-05 at `t = 15, r = 0` | **not resolved at these statistics** (1.1σ) — see §3 |
| simulation MC error | sample SEM from accumulated sums of squares | Isserlis estimate is 0.87–1.00 of it | the field's non-Gaussianity (≤ 13 %) |
| off-grid separations | theory interpolated with the *same* `np.interp` weights | removes a +0.5 % (+3.7σ) bias exactly | — |

---

## 3. Error budget

At `t = 15, r = 0` (the worst point), `ξ₀₁`:

| | value | evidence class |
|---|---|---|
| simulation, dt = 0.02 | 4.169e-04 ± 1.00e-05 | measured |
| simulation, Δt → 0 | 4.376e-04 ± 1.30e-05 | measured + extrapolated |
| FK (order 2) | 3.441e-04 | **computed** |
| FFFK (order 4, F³κ⁽³⁾) | 5.47e-05 (GL10) / 5.41e-05 (GL14) | **computed** |
| theory total | 3.988e-04 | |
| **residual** | **+3.88e-05 (+2.9σ)** | |

Over the 18 times at `r = 0`, adding the exactly-computed order-4
channel takes **χ² from 340.7 to 44.2**, the mean pull from **+3.31 to
+1.09**, and the largest residual from **9.36e-05 to 3.88e-05**.

**Attribution, and its evidence class.**  The residual is an order-4
effect — *deductively, not by assumption*.  Scaling the quadratic drift
by `s` scales each channel by a known power of `s`, and the residual
measured at `s = 0.5` (20M realisations) and `s = 1` (2M) fits a **pure
`s³` law with χ² = 0.06 for 1 dof**, giving `c₃ = 7.38e-05 ± 0.91e-05`.
That uses only the simulation and the validated order-2 channel; it
assumes nothing about the order-4 calculation.  The computed `F³κ⁽³⁾`
= 5.47e-05 then agrees with it at **2.1σ**, and with the raw dt = 0.02
residual at 1.8σ.

The spread between 1.8σ and 2.9σ is *not* physics: it is whether the
step-size bias is extrapolated away or treated as zero.  That bias is
itself only a 1.1σ measurement at these statistics, so both numbers are
honest and neither is better founded than the other.

**No further channel is required, and the obvious candidate is
excluded.**  demo2's action truncates the cumulant ladder at `κ⁽⁴⁾`, so
`F³κ⁽⁵⁾` — the same perturbative order, the same `s³` scaling — sits
inside both the residual and the fitted `c₃`.  It is too small to
matter: `κ⁽⁵⁾` is 7.1 % of `κ⁽³⁾`, and it enters through **6 diagrams
against 30** (both counts enumerated), putting the channel at ~1e-06 —
an order of magnitude below the residual's uncertainty.  *Estimated,
not computed*: the per-diagram magnitudes and the `−iᵐ/m!` convention
differ between `m = 3` and `m = 5`, so this is a sizing argument, but it
points firmly the wrong way for the `κ⁽⁵⁾` hypothesis.

---

## 4. What is NOT established

- **The 2σ is not resolved.**  The computed `F³κ⁽³⁾` accounts for the
  residual to within 2σ, not within its errors.  Closing it would need
  either better simulation statistics at `s = 1` or the order-6
  `F⁵κ⁽³⁾` channel, neither of which is here.

- **`F³κ⁽⁵⁾` is estimated, not computed.**  Confirming ~1e-06 needs a
  5-leg R-contracted kernel that does not exist.

- **The late-time rows are quadrature-limited, and the theory saturates
  there.**  Beyond `t ≈ 20` the FFFK and FFFF rules lose the integrand's
  peak (9.2 % and 7.4 % at the worst cells).  Both channels saturate
  well before that, so the physical value is the saturated one — but do
  not read a trend across the last rows.

- **The series is asymptotic, and the model blows up.**  Roughly 6e-05
  of trajectories per realisation reach `|φ| > 10⁶` in finite time and
  are discarded, so the simulation reports a *conditioned* mean while
  the theory is an asymptotic series with no such conditioning.  A
  rare-excursion contribution at the 1e-05 level is not testable from
  the stored sums.  **D2 sharpens this rather than removing it**: at
  `s = 0.5` the blow-up fraction is **exactly zero in 20 million
  realisations**, so the cleanest of the two fit points carries no
  conditioning at all — which is why the `s³` law can be trusted.  At
  `s = 1.5` it is 4.5 %, and there the residual exceeds the leading
  term: that is the demonstrated boundary of validity, not a
  measurement.

- **Only `κ⁽³⁾` and `κ⁽⁴⁾` are in the action.**  The truncation is a
  choice, now with a size attached (§3), not a property of the model.

- **One assumption of the 0.3.0 budget was simply wrong** and is worth
  recording as a caution: `F³κ⁽³⁾` was assumed to vanish for `ξ₀₀` and
  `ξ₁₁` by `φ₁ → −φ₁` parity.  That parity is broken by the deformation
  itself.  Computed rather than assumed, the channel is 4.5e-06 there —
  13 % of the `ξ₀₀` residual.

---

## 5. The case for the paper

Demo 2 is the only demo in which the diagrammatic machinery is doing
something that cannot be checked by inspection: an observable whose
entire value comes from a third noise cumulant, computed through a
non-local vertex whose leg integrals are contracted analytically, and
verified against a simulation at the 1e-05 level.  It exercises the
parts of the package nothing else does — `already_R_contracted`,
propagator-indexed dynamic couplings, the exact two-kernel `C_eff` — and
each of those was wrong at least once, which is the argument for the
demo existing at all.

What it now demonstrates, that the 0.3.0 version did not: the
perturbative order of a residual can be **measured**, by scaling the
coupling and reading off the power, and the answer agrees with the
diagram sum.  That is a stronger claim than "theory matches simulation"
and it is the one worth making.

The cost of the caveats is a 2σ residual we cannot presently close, a
late-time regime that is quadrature- rather than physics-limited, and a
conditioned mean standing in for an asymptotic series.  None of these
threatens the demonstration; all of them are stated with sizes, which is
the standard the rest of the paper should be held to.
