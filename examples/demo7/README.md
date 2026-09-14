# Demo 7: observables in space, angle and time

`R` is local in space, `R(x, t; z, s) = δ(x − z) R(t − s)`, and every local
vertex acts at one point.  An n-point function at observation points
`{x_i}` therefore involves the noise only at those points, and the fields
there obey a finite-dimensional Itô SDE whose noise covariance is the
spatial kernel at the points' separations.  One reference covers
translation, rotation and general homogeneity, any spatial kernel and any
dimension: the moment hierarchy of that embedding (`space7_reference.py`,
on `examples/reference/ito_moments.py`).

```
dφ_a/dt = −γ_a φ_a + F_abc φ_b φ_c + G_abcd φ_b φ_c φ_d + η_a(x,t) + ξ_a(t),
                                              t ≥ t_min = 0.4,  φ(t_min) = 0
⟨η_a(x,t) η_b(y,t')⟩ = δ_ab λ e^{−|t−t'|/σ_t} κ_x(x, y)     (coloured)
⟨ξ_a ξ_b⟩ = S_ab δ(t − t')                                  (optional, dense S)
```

`F` and `G` have no index symmetry, the decay rates differ per component in
the matrix-R variants, and the observation points and times are all
distinct.  The embedding gives `η_{a,i}` as an Ornstein-Uhlenbeck process
at each point with stationary covariance `Σ^a_ij = λ κ_x(x_i, x_j)`, so the
reference needs no diagrams, no propagators and no sft-wick code.  Tags
`(k_F, k_G)` count the vertices, so each coefficient is one channel of the
package (`F`, `G`, `FG`).

## What is checked

| item | what | route |
|---|---|---|
| (a) | `⟨φ_a(x, t) φ_b(y, t')⟩` at orders 0-2, `a ≠ b`, `r = 1.3`, `t ≠ t'` in **both** orders, through `external_times` | `gauss_legendre`, `qmc_vectorized`, `qmc_scalar`, `nquad`; scalar R, matrix R, and matrix R with a component-mixing white noise |
| (b) | `SeparableRotation` with a **four-coefficient** `LegendreAngular` kernel at three angles (35°, 80°, 150°), orders 0-2 | the package's own rotation tables (lazy, per `cos θ`) |
| (c) | `GaussianSpatial` and a `CustomKernel` (damped cosine, negative at `r = 1.3`) through the translation tables, and a `GeneralKappa2` with no spatial symmetry and a different amplitude, correlation time and width per component | `gauss_legendre`, `qmc_vectorized`, `qmc_scalar` |
| (d) | 3-D positions with a callable `κ³`: compound-Poisson shot noise in `R³`, orders 0-2, off-diagonal component pairs | `gauss_legendre`, `qmc_vectorized`, `qmc_scalar` |
| (e) | `integrate_over='all'` and a one-point subset at `N = 2` with `a ≠ b`, order 2; and the three-point function at order 2 | `gauss_legendre`, `qmc_vectorized`, `qmc_scalar` |

The reference is compared **channel by channel**, not only on the total:
the package's `Result.by_vertex_type` against the hierarchy's tags.

### Item (d): shot noise in three dimensions

Events `(z_k, s_k)`, `z_k ∈ R³`, of rate `ν` per unit volume per unit time
drive component `a` at `x` by `h_a w_a(x − z_k) g_a(t − s_k)` with
`w_a(u) = e^{−|u|²/(2 s_a²)}`.  Campbell's theorem makes every cumulant a
Gaussian overlap integral in closed form,

```
κ^(m)_{a…}(x_1…x_m; t_1…t_m) = ν Π_j h_{a_j} · X_a(x) · G_a(t),
X_a(x) = ∫d³z Π_j w_{a_j}(x_j − z) = (2π/P)^{3/2} e^{−½ Σ_j w_j |x_j − x̄|²},
```

so the `κ³` vertex is a callable evaluated at 3-D leg positions
(`(m, n_samples, 3)` under the vectorised contract) and C is an exact
closed form of the 3-vector separation (`diag_C=False`: the events drive
both components).  White pulses use the raw `equal_time` vertex,
exponential pulses the `already_R_contracted` `K_R`.

## Results

Worst relative difference from the hierarchy over the component tuples, the
channels and (item a) the two time orders; `results.json` and
`shot3d_results.json` hold every row. All listed scalar- and matrix-R
routes are supported, including the batched backends at interacting orders.

**(a) `⟨φ_a(x, 1.7) φ_b(y, 1.05)⟩` and its time-reversed partner** (6 cases
per cell: 3 component pairs × 2 time orders; `gauss_legendre` n = 48,
`qmc_vectorized` 2¹⁶, `qmc_scalar` 2¹¹, 2¹² for the white variant):

| noise | order | `gauss_legendre` | `qmc_vectorized` | `qmc_scalar` | `nquad` |
|---|---|---|---|---|---|
| coloured, scalar R | 0 | 0 | 0 | 0 | 0 |
| coloured, scalar R | 1 | 1.4e-15 | 7.4e-16 | 5.9e-11 | 3.6e-16 |
| coloured, scalar R | 2 | 4.9e-15 | 9.7e-10 | 5.9e-06 | 9.9e-16 |
| coloured, matrix R | 0 | 1.2e-16 | 1.2e-16 | 1.2e-16 | 1.2e-16 |
| coloured, matrix R | 1 | 1.8e-15 | 6.6e-16 | 1.1e-10 | 6.6e-16 |
| coloured, matrix R | 2 | 4.7e-15 | 1.4e-09 | 2.2e-05 | 6.7e-16 |
| + mixing white, matrix R | 0 | 7.7e-16 | 7.7e-16 | 7.7e-16 | 7.7e-16 |
| + mixing white, matrix R | 1 | 2.7e-15 | 8.1e-16 | 9.9e-12 | 1.0e-15 |
| + mixing white, matrix R | 2 | 5.7e-15 | 3.3e-07 | 3.4e-05 | 1.5e-15 |

Order 0 of the first block is exactly 0 on both sides for `a ≠ b`: with
`κ² ∝ I_N` and a diagonal R, C is diagonal.  The mixing white noise makes
`C_{01}` non-zero, and with a rate per component it is asymmetric under
swapping the two times (by 37 % at these times), so the order-0 row of the
last block does check the routing of the two external times.

**(b) rotation, `Σ_{ℓ≤3} C_ℓ P_ℓ(cos θ)`** at 35°, 80° and 150°, where the
kernel is +0.92, +0.18 and −0.13 (it changes sign with direction; every
earlier test used `coeffs=[1.0]`, where C does not depend on direction):

| R | order | `gauss_legendre` (32) | `qmc_vectorized` (2¹⁴) | `qmc_scalar` (2¹⁰ / 2¹¹) |
|---|---|---|---|---|
| scalar | 0 | 1.7e-10 | | |
| scalar | 1 | 2.6e-08 | | 2.7e-08 |
| scalar | 2 | 2.6e-07 | 6.9e-07 | |
| matrix | 0 | | | 1.1e-08 |
| matrix | 1 | 1.9e-07 | | 2.0e-07 |
| matrix | 2 | | 3.5e-06 | 2.4e-04 |

**(c) the package's own C tables** at `r = 1.3` (`gauss_legendre` n = 32,
`qmc_vectorized` 2¹⁴, `qmc_scalar` 2¹¹):

| κ² | order 0 | order 1 | order 2 | order 2, QMC |
|---|---|---|---|---|
| `GaussianSpatial` (GL tables, n_grid_t = 48) | 1.7e-10 | 2.6e-08 | 5.9e-08 | 1.9e-07 |
| `CustomKernel`, damped cosine (dblquad tables) | 1.7e-10 | 2.6e-08 | 1.2e-07 | 2.5e-07 |
| `GeneralKappa2` (GL tables, n_grid_t = 40) | 4.3e-09 | 4.9e-08 | 1.1e-07 | 2.0e-07 |
| `GeneralKappa2`, matrix R (`qmc_scalar`) | 1.3e-08 | 1.9e-07 | 4.1e-05 | 5.0e-07 |

**(d) 3-D positions with a callable κ³** (`gauss_legendre` n = 16,
`qmc_vectorized` 2¹⁴, `qmc_scalar` 2¹¹; the order-2 column is the `F` and
`FK3` channels, the 3-point column the `F` and `K3` channels):

| pulses | observable | `gauss_legendre` | `qmc_vectorized` | `qmc_scalar` |
|---|---|---|---|---|
| white, raw `equal_time` κ³ | order 0 | 1.8e-16 | 1.8e-16 | 1.8e-16 |
| white | order 2 | 4.2e-16 | 1.1e-05 | 1.8e-04 |
| white | 3-point, order 1 | 6.3e-16 | 1.6e-11 | 2.5e-10 |
| exponential, `already_R_contracted` | order 0 | 3.5e-15 | 3.5e-15 | 3.5e-15 |
| exponential | order 2 | 6.2e-15 | 6.2e-08 | 7.9e-06 |
| exponential | 3-point, order 1 | 2.1e-14 | 2.8e-12 | 8.0e-11 |
| white, matrix R | order 0 | 1.8e-16 | 1.8e-16 | 1.8e-16 |
| white, matrix R | order 2 | 9.2e-16 | 1.2e-05 | 2.0e-04 |
| white, matrix R | 3-point, order 1 | 1.0e-15 | 2.5e-11 | 4.7e-10 |

**(e) `integrate_over` and the three-point function** (`gauss_legendre`
n = 24, `qmc_vectorized` 2¹⁶ / 2¹⁴, `qmc_scalar` 2¹²; three component
tuples per cell):

| observable | order | `gauss_legendre` | `qmc_vectorized` | `qmc_scalar` |
|---|---|---|---|---|
| `integrate_over='all'` | 0 / 1 / 2 | 1.5e-15 / 1.8e-15 / 1.2e-15 | 3.8e-10 / 2.6e-08 / 2.8e-05 | 3.3e-07 / 3.2e-06 / 1.2e-03 |
| `integrate_over={'x'}` | 0 / 1 / 2 | 4.1e-16 / 9.9e-16 / 8.9e-16 | 4.1e-16 / 5.2e-10 / 1.1e-06 | 1.0e-12 / 6.0e-07 / 2.2e-04 |
| `integrate_over`, matrix R | 0 / 1 / 2 | 1.4e-15 / 1.4e-15 / 1.2e-15 | 5.0e-10 / 1.2e-07 / 4.9e-05 | 3.2e-07 / 6.7e-05 / 1.5e-03 |
| 3-point `⟨φ_a(x) φ_b(y) φ_c(z)⟩` | 1 / 2 | 9.8e-16 / 2.2e-15 | 5.8e-12 / 1.4e-07 | |
| 3-point, matrix R | 1 / 2 | 2.5e-16 / 1.8e-15 | | 1.8e-10 / 2.0e-05 |

The three-point function has no `FF` or `GG` diagrams at order 2 (an odd
number of fields), so that order is the `FG` channel alone; the hierarchy
gives 0 for the other two tags and the package produces no diagram for
them.

The tables in this section match the saved results, including the
revalidated C-table paths and the domain cuts (see "Limits, measured").
The cuts improve the integrated-observable Gauss-Legendre results: `integrate_over='all'`
at order 2 is 1.2e-15 in the table against 2.2e-09 before the cuts, and
`integrate_over={'x'}` at order 2 is 8.9e-16 against 1.1e-07.

## Limits, measured

- **Gauss-Legendre converged as `n^-4` on a two-time integrand** until
  2026-09-12, when the two kinks behind it were declared and cut.  C's
  third derivative jumps on its time diagonal (the `|Δt|` cusp of the OU
  kernel, through `∂²C/∂t₁∂t₂ = R κ² R`), and where an internal time
  crosses a *fixed* external time that line sits inside the domain as soon
  as `t ≠ t'`.  On the order-1 channel of `⟨φ_0(x,t) φ_1(y,t')⟩`, coloured
  scalar R:

  | | 8 | 12 | 16 | 24 | 32 nodes |
  |---|---|---|---|---|---|
  | before | 4.1e-05 | 8.8e-06 | 2.9e-06 | 5.9e-07 | 1.9e-07 |
  | now | 3.6e-16 | 0.0 | 3.6e-16 | 1.8e-16 | 5.4e-16 |

  and on the order-2 channel 6.1e-06 → 6.2e-14 at 8 nodes.  The same
  configuration with both externals at one time is unchanged at order 1
  (the kink is on the boundary) and goes from 5.7e-06 to 1.7e-10 at 8 nodes
  at order 2, where the kink is between two internal times.  The
  white-noise, matrix-R variant went from 4.0e-3, 1.0e-3, 2.6e-4, 6.6e-5 at
  8, 16, 32, 64 nodes (`n^-2`) to machine precision at 8.  The cost is
  2.1 pieces per diagram on the coloured configuration and 2.9 on the
  two-time white-noise one.  The tables above are the regenerated run.
- **Historical QMC floor with 30-bit Sobol points.** Both routes stalled
  at 1.4e-9 relative on that channel, the same
  value for every seed and for 2¹² to 2¹⁸ samples, while `nquad` and the
  reference agree to 3.6e-16.  scipy's Sobol points carry 30 bits by
  default, so the sample mean has a left-Riemann bias of
  `−(f(1) − f(0))/2^31`.  Measured directly on `∫_0^1 e^{2u} du`: −9.31e-10
  at 2¹⁴, 2¹⁸ and 2²⁰ for two seeds, which is that formula to three
  digits; with `bits=64` the error is 0 at 2¹⁸.  It is a floor, not a rate,
  and it only shows where the QMC statistical error is already below 1e-8.
  The package now uses `bits=64`; the current results above include this fix.
- **Matrix R is supported** by all four listed backends. The former
  batched-backend restriction was already removed in 0.6.0; the numerical
  rows in item (a) replace the obsolete "refused" cells.
- **Table resolution** can dominate the quadrature-table configurations.
  The earlier `(t1,t2)` rotation tables gave order-1 errors of 4.5e-7 at
  `n_grid_t = 24` and 6.7e-8 at 48, at fixed `n_gauss = 32`. The current
  min-time/lag table results are reported in items (b) and (c).
- The reference's own two-time propagation and integrated fields are
  checked against quadrature of the same free theory in
  `tests/test_demo7_space.py` (1e-10 and 1e-6).

## Files

| file | contents |
|---|---|
| `space7_reference.py` | the moment hierarchy at the observation points: two-time propagation, integrated fields, the spatial kernels written out; numpy + `ito_moments` only |
| `space7_model.py` | the Gaussian model: `F`, `G`, the kernels, and one `Config` per noise variant (system, propagators, reference covariance) |
| `space7_run.py` | items (a), (b), (c), (e) → `results.json` |
| `space7_shot.py` | the 3-D shot noise: closed forms and its own hierarchy; numpy + `ito_moments` only |
| `space7_shot_system.py` | the sft-wick side of the shot noise (callables, system, propagators) |
| `space7_shot3d.py` | item (d) → `shot3d_results.json` |

```bash
conda activate sft-wick
cd examples/demo7
python space7_run.py            # items (a), (b), (c), (e)
python space7_shot3d.py         # item (d)
pytest ../../tests/test_demo7_space.py ../../tests/test_demo7_shot3d.py -q
```
