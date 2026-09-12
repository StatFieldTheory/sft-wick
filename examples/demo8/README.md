# Demo 8 — time-dependent coefficients and non-exponential dynamics

Demos 1-5 all have a constant, diagonal drift and an exponential (OU)
temporal kernel, so `R(t, s) = e^{−γ(t−s)}` and every closed form in the
package applies.  This demo drives the four routes that leave that family:

| model | what leaves the family |
|---|---|
| (a) rate | `DiagonalA(gamma=callable)`: `γ(t) = [1 + 0.5 sin t, 0.6 + 0.3 cos 2t]`, unequal and time-varying, `t_min = −1.3` and `0.7` |
| (b) oscillator | `ExplicitR` with the response of `ẍ + 2ζω ẋ + ω² x` (`ω = 1.7`, `ζ = 0.3`), which oscillates and changes sign |
| (c) kernels | `CustomKernel`: Matérn-3/2 and a damped cosine; `GaussianTemporal` |
| (d) white | `CustomImpulse` with an amplitude `s_a(t)` varying in time |

Two components throughout, a local `F_abc` with no index symmetry (and a
cubic `G_abcd` in model b), distinct points `x = 0`, `y = 0.7`, `z = −0.4`,
off-diagonal component tuples, orders 0-2.

```
(a)  dφ_a = (−γ_a(t) φ_a + F_abc φ_b φ_c + η_a) dt,        η OU, λ = 0.5, σ_t = 0.6
(b)  ẍ_a + 2ζω ẋ_a + ω² x_a = F_abc x_b x_c + G_abcd x_b x_c x_d + ξ_a,
                                  ⟨ξ_a(x,t) ξ_b(y,t')⟩ = δ_ab s_a K_w(x−y) δ(t−t')
(c)  dφ_a = (−γ_a φ_a + F_abc φ_b φ_c + η_a) dt,           η Matérn-3/2 | damped cosine | Gaussian
(d)  dφ_a = (−γ_a φ_a + F_abc φ_b φ_c + η_a) dt + dW_a,    ⟨dW_a dW_b⟩ = δ_ab s_a(t) K_w(x−y) dt
```

## What is compared with what

The reference is the moment hierarchy of a finite Markov embedding, whose
generator depends on time; it is integrated with `scipy.integrate.solve_ivp`
(DOP853, `rtol = 1e-12`) and shares no code with the package
(`time8_reference.py`, on `examples/reference/ito_moments.py`).  Its
coefficient of `F^k` is order `k` of the package.

| model | embedding |
|---|---|
| (a) | `(φ, η)`, `η` an OU process started in its stationary law, drift `−γ_a(t)` |
| (b) | `(x, v)`, white force; two tags count the quadratic and the cubic vertex |
| (c) | `(φ, η, ξ)` for Matérn-3/2 (`η̇ = ξ`, `ξ̇ = −a²η − 2aξ + √(4a³λ) w`) and `(φ, z₁, z₂)` for the damped cosine (`ż = [[−a, −ω], [ω, −a]] z + √(2aλ) w`) |
| (d) | `(φ, η)` with a diffusion coefficient `s_a(t)` |

`GaussianTemporal` has no finite embedding (its spectral density is not
rational), so it has no hierarchy.  What stands in for one, and how far
each piece is exact:

| piece | status |
|---|---|
| `C(t₁, t₂)` for this kernel and an exponential R | exact, in closed form (`gaussian_kernel_C`, `erf`); checked against `dblquad` to 1e-11 |
| the order-1 and order-2 diagrams written out from the perturbative solution of the SDE and Wick's theorem (`HandContraction`) | exact as formulas — checked against the hierarchy for the exponential kernel, which does have an embedding (1e-10) |
| their time integrals | Gauss-Legendre, converged: 48 against 64 nodes per dimension agree to 1e-14 |
| time-translation invariance | an exact identity of the model, but blind to any error that shifts with `t_min` |

The Gaussian rows are therefore exact to a converged quadrature rather than
to a closed form.  The Wick combinatorics behind them is written out in the
reference module and shares no code with the package.

C reaches the package two ways.  **exact C** hands it
`time8_reference.ExactC`, the defining integrals `∫∫ R κ² R (+ ∫ R σ² R)`
evaluated by Gauss-Legendre at every point the integrators ask for, so a row
measures the diagrams and R alone; that C is checked against the hierarchy's
two-time order-0 moment for every model (≤ 2e-12).  **table** is the
package's own route: quadrature on an `n_grid_t × n_grid_t` grid and a
spline lookup.

A matrix R — what component-dependent rates give — runs on the scalar loops
only (`nquad`, `qmc_scalar`, `qmc`).  Each rate-dependent model is therefore
run twice: at unequal rates on those, and at equal rates on Gauss-Legendre
and `qmc_vectorized`.

## Results

Worst relative difference from the reference over the component tuples and
orders of each block (`time8_results.json`).

### (a) a rate varying in time, `t_min ∈ {−1.3, 0.7}`

| route | worst |
|---|---|
| exact C, `nquad` (matrix R) | 5.9e-09 |
| exact C, `qmc_scalar` and `qmc` 2¹¹ (matrix R) | 4.3e-05 |
| exact C, `gauss_legendre` 12 (scalar R) | 2.3e-13 |
| exact C, `nquad` (scalar R) | 1.3e-08 |
| exact C, `qmc_vectorized` 2¹⁴ (scalar R) | 4.3e-07 |
| table `n_grid_t = 21` / `41`, `nquad` (matrix R) | 9.7e-06 / 1.9e-07 |
| table `n_grid_t = 21` / `41`, `gauss_legendre` 12 (scalar R) | 2.1e-05 / 1.3e-05 |

`gauss_legendre` and `qmc_vectorized` refuse the matrix R; the table row on
`gauss_legendre` does not converge, for the reason in "Limits" (an OU κ²
cusps, and the table cache does not declare it).

`Γ_a(t) = ∫γ_a` is built on the rate-cache grid, and the moments converge in
its spacing `h` (worst of the tadpole and order-2 `⟨φ_a φ_b⟩`, exact C):

| `h` | 0.50 (the `DiagonalA` default) | 0.015 | 0.0036 |
|---|---|---|---|
| now | 6.1e-06 | 4.2e-12 | 1.3e-13 |
| with the trapezoid Γ this branch replaced | 1.3e-03 | 1.1e-06 | — |

### (b) a damped oscillator, `R(τ) = e^{−ζωτ} sin(ω_d τ)/ω_d`

`R` first crosses zero at `τ = π/ω_d = 1.94`, inside the window
(`span = 3.1`).  Channels: `F` and `FG` of `⟨x_a⟩`, order 0, `G`, `FF` and
`GG` of `⟨x_a x_b⟩`, `F` of `⟨x_a x_b x_c⟩`.

| route | worst |
|---|---|
| exact C, `gauss_legendre` 12 | 6.5e-08 |
| exact C, `gauss_legendre` 16 / 20 | 5.7e-10 / 1.9e-15 |
| exact C, `nquad` | 2.7e-06 |
| exact C, `qmc_vectorized` 2¹⁴ | 4.4e-05 |
| exact C, `qmc_scalar` 2¹¹ | 2.5e-03 |
| table `n_grid_t = 21` / `41`, `gauss_legendre` 12 | 6.4e-05 / 1.6e-06 |
| unequal external times, `qmc_vectorized` 2¹⁶ | 3.2e-12 (orders 0, 1), 6.2e-08 (order 2) |
| unequal external times, `nquad` | 3.7e-08 (order 1), 7.1e-07 (order 2) |
| unequal external times, `gauss_legendre` 12 / 28 | 1.6e-03 / 4.4e-05 (see "Limits") |

### (c) custom temporal kernels

| kernel, rates | exact C | table `n_grid_t = 21` | table `41` |
|---|---|---|---|
| Matérn-3/2, equal (`gauss_legendre` 12) | 1.5e-13 | 4.8e-07 | 4.4e-08 |
| Matérn-3/2, unequal (`nquad`) | 1.4e-09 | 7.8e-07 | 5.4e-08 |
| damped cosine, equal (`gauss_legendre` 12) | 1.3e-13 | 2.5e-05 | 2.2e-05 |
| damped cosine, unequal (`nquad`) | 1.0e-08 | 2.3e-06 | 3.2e-07 |
| Gaussian, equal (`gauss_legendre` 12) | 2.4e-13 | 2.8e-06 | 1.7e-07 |
| Gaussian, unequal (`nquad`) | 9.5e-15 | 3.2e-06 | 1.6e-07 |

`qmc_vectorized` at 2¹⁴ reaches 6.4e-07 (equal rates) and `qmc_scalar` at
2¹¹ 1.4e-04 (unequal) on the same exact C.

The damped-cosine table row does not converge: the limit is not the table
but the integrator (see "Limits").  The Gaussian rows are measured against
the hand contraction, the others against the hierarchy.

Time-translation invariance (`GaussianTemporal`, table C, `t_min` shifted by
−1.1, 0 and +0.9 with the observation time): 3.3e-16 at orders 0 and 2.

### (d) a white-noise amplitude varying in time

| route | worst |
|---|---|
| exact C, `gauss_legendre` 12 (scalar R) | 1.4e-13 |
| exact C, `nquad` (scalar R) | 1.9e-07 |
| exact C, `qmc_vectorized` 2¹⁴ | 8.3e-07 |
| exact C, `nquad` (matrix R) | 7.2e-08 |
| exact C, `qmc_scalar` 2¹¹ (matrix R) | 9.9e-05 |

The package's own table against the hierarchy, order-2 `⟨φ_0(x) φ_1(y)⟩`
and the order-1 tadpole:

| `n_grid_t` | 11 | 21 | 41 | 81 |
|---|---|---|---|---|
| order 2, `qmc_vectorized` 2¹⁴ | 8.3e-03 | 1.9e-03 | 4.7e-04 | 1.2e-04 |
| order 2, `gauss_legendre` 12 | 7.5e-03 | 1.9e-03 | 3.4e-04 | 2.6e-04 |
| order 1, `⟨φ_0⟩` | 1.2e-05 | 6.9e-07 | 1.5e-08 | 1.9e-09 |
| order 0, `⟨φ_1(x) φ_1(y)⟩` | 7.7e-05 | 7.4e-14 | 7.4e-14 | 7.4e-14 |

Order 0 reads `C(t_f, t_f)`, which comes from the table's own diagonal: it
is exact wherever `t_f` is a grid node (it is not at `n_grid_t = 11`).  The
order-2 rows read C off the diagonal and converge as the step squared.

The table's C itself, by distance `δ` from the time diagonal (largest
relative departure from the exact C over `t ∈ [0.6, 2.1]`):

| `n_grid_t` | `δ = 0` | `δ = 10⁻⁶` | `δ = 10⁻²` | `δ = 0.1` | `δ = 0.4` |
|---|---|---|---|---|---|
| 21 | 1.7e-06 | 1.8e-02 | 1.1e-02 | 1.8e-02 | 4.2e-03 |
| 41 | 9.0e-08 | 8.5e-03 | 8.8e-03 | 3.2e-03 | 1.5e-05 |
| 81 | 6.2e-09 | 3.3e-03 | 5.9e-03 | 1.6e-04 | 4.7e-08 |

## Limits, measured

- **A matrix R runs on the scalar loops only.**  `gauss_legendre` and
  `qmc_vectorized` refuse one, so the unequal-rate half of models (a), (c)
  and (d) is run on `nquad` / `qmc_scalar` / `qmc`.  The tests carrying
  those routes skip with a message and start checking as soon as the
  refusal goes.
- **The domain is cut where an internal time crosses a *fixed external*
  time** (was a limit until 2026-09-12).  A C propagator with one end at
  each is kinked there when C is (white noise, or a κ² with a `|Δt|`
  cusp), and a constant is not an ordering between two variables, so the
  pair split could not express it; the integrators cut the variable's range
  at that time instead.  It costs nothing while every external sits at
  `t_final` — the crossing is then the domain boundary — and shows up with
  `external_times`.  Model (b), order-1 `G` channel at `x: 2.60, y: 3.50`,
  against the hierarchy:

  | | 8 | 12 | 16 | 24 | 32 | 48 nodes |
  |---|---|---|---|---|---|---|
  | unequal external times, before | 4.5e-04 | 3.1e-05 | 4.7e-07 | 5.7e-06 | 2.9e-06 | 2.2e-07 |
  | unequal external times, now | 4.5e-07 | 4.4e-13 | 6.8e-15 | 5.8e-15 | 9.1e-15 | 1.1e-14 |

  where the same channel with both externals at `t_final` reached 1.5e-14
  by 16 nodes either way.  `qmc_vectorized` reached 2.4e-12 at 2¹⁶ on the
  same integrand and `nquad` 2.7e-08 before the cut, so the value was right
  and only the rate was lost.  The order-2 `F` and `G` channels behave the
  same (2.6e-04 and 6.6e-04 at 16 nodes before, 1.1e-15 and 2.6e-13 now).
  The cost is one integration per cut: 2.88 pieces per diagram over those
  three channels, against 1.38 before.  The same integral written out by
  hand converges once its domain is split at the crossing (5.9e-15 at 60
  nodes, 6.8e-15 at 120) and not otherwise (1.0e-07 at 60, 6.3e-10 at 240);
  `time8_run.py` prints both.
- **A κ² with a `|Δt|` cusp is declared as kinking C** (was a limit until
  2026-09-12).  `_c_has_diagonal_kink` was true when the model carried
  `sigma2`, or when a closed-form C said so; a quadrature-table cache whose
  κ² has a cusp (the damped cosine here, and the OU kernel of demos 1-5)
  said nothing, so Gauss-Legendre did not split at the C diagonal.  It now
  asks the question the C quadrature already asks of the kernel
  (`PropagatorCache._kappa2_has_diagonal_cusp`: the built-in kernels
  declare `has_diagonal_cusp`, any other callable is probed from one-sided
  differences).  Order-2 `⟨φ_0(x) φ_1(y)⟩` on the package's table at
  `n_grid_t = 41`, against the hierarchy:

  | kernel | | GL 12 | GL 20 | `qmc_vectorized` 2¹⁶ |
  |---|---|---|---|---|
  | damped cosine | before | 1.4e-05 | 1.7e-06 | 1.9e-07 |
  | | now | 9.5e-07 | 1.5e-07 | |
  | exponential (OU) | before | 7.5e-06 | 1.2e-06 | 4.8e-08 |
  | | now | 3.8e-07 | 1.2e-07 | |

  Gauss-Legendre now lands where QMC does — on the table's own error.  A
  Gaussian kernel is differentiable at `Δt = 0`, leaves C smooth and is kept
  out of the split.  The cost is one integration per unordered C pair: 1.33
  pieces per diagram here, against 1.00.  The `exact C` rows of this demo
  declared `has_diagonal_kink=True` all along and never lost the rate.
- **The C table cannot follow a white-noise kink.**  `C(t₁, t₂)` has a
  derivative jump of `σ²` on `t₁ = t₂`; the table's tensor-product spline is
  C², so just off the diagonal it is wrong by `O(h)` (table above) and the
  moments that read C there converge as `O(h²)` instead of at spline order.
  On the diagonal itself a separate 1-D spline keeps `O(h⁴)`, so tadpoles
  and the equal-time correlator are unaffected.  Where a closed form exists
  (`c_closed_form`), it removes the whole effect.
- **The rate-cache grid.**  A callable `DiagonalA.gamma` is integrated on
  `[t_min_cache, t_max_cache]` with `n_grid_cache` nodes; at the defaults
  (`t_max_cache = 100`, 200 nodes, `h = 0.50`) a rate varying on a unit time
  scale gives R to 4.2e-04 and the moments to 5.8e-06.  Choose
  `t_max_cache` near the horizon actually needed.

## Files

| file | contents |
|---|---|
| `time8_params.py` | parameters and the model functions (`γ(t)`, the oscillator response, the kernels, `σ²(t)`); numpy only |
| `time8_reference.py` | the time-dependent moment hierarchy, the free C by quadrature, the erf C for a Gaussian kernel, and the hand contraction; no sft-wick code |
| `time8_system.py` | the four `System` objects and the two ways to supply C |
| `time8_run.py` | every comparison above; writes `time8_results.json` |

```bash
conda activate sft-wick
cd examples/demo8
python time8_run.py               # ~12 min
python time8_run.py --items ad --quick
pytest ../../tests/test_demo8_reference.py \
       ../../tests/test_demo8_time_dependent_rate.py \
       ../../tests/test_demo8_oscillator.py \
       ../../tests/test_demo8_kernels.py \
       ../../tests/test_demo8_white_noise_table.py -q
```
