# Demo 5: white noise on every integrator, additive and multiplicative

Demos 1-4 drive the field with coloured noise or with pulses; none uses the
package's white-noise path (`GaussianNoise(sigma2=…)`), none starts at
`t_min ≠ 0`, and none runs the scalar-loop integrators or `iso_C=True`.
Each of those hid a defect fixed on 2026-09-11, and this demo runs all of
them against an exact reference.

## Part A: additive white noise (`run.py`)

```
dφ_a = (−γ φ_a + F_abc φ_b φ_c + η_a) dt + dW_a,     t ≥ t_min = 0.5,  φ(t_min) = 0
⟨η_a(x,t) η_b(y,t')⟩ = δ_ab λ e^{−|t−t'|/σ_t} e^{−(x−y)²/(2σ_x²)}
⟨dW_a dW_b⟩ = S_ab dt        (ConstantImpulse, the same at every point)
```

White noise makes `C = ∫ R S R` kinked on its time diagonal; the coloured
part keeps distinct points distinct.  Three variants take three code paths:

| variant | `S` | flags |
|---|---|---|
| mixing | dense | `diag_C=False` |
| diagonal | `diag(S)` | `diag_C=True`: the diag-C fast path of the scalar loops |
| isotropic | `s · 1` | `diag_C=True`, `iso_C=True` |

Observables: `⟨φ_a(x)⟩` at orders 1 and 3; `⟨φ_a(x) φ_b(y)⟩` at orders 0,
2, 4 for all four pairs; `⟨φ_a(x) φ_b(y) φ_c(z)⟩` at order 1.  Gauss-Legendre
runs everything; `qmc_vectorized` orders ≤ 2; `qmc_scalar` and `qmc` order 2
of `⟨φ_0 φ_1⟩` and the tadpole; `nquad` the tadpole at orders 1 and 3 and
`⟨φ_a φ_b⟩` at order 2.

Reference: the moment hierarchy of the Markov embedding `(φ, η)` at the
observation points, with `η` an Ornstein-Uhlenbeck process started in its
stationary law (`white_reference.py`, on `examples/reference/ito_moments.py`; no
sft-wick code).  Its coefficient of `F^k` is order `k` of the package.

Worst relative difference over the component tuples (`results.json`):

| | mixing | diagonal | isotropic |
|---|---|---|---|
| Gauss-Legendre, orders 0-3 (12 nodes) | 6.6e-16 | 9.2e-16 | 5.6e-16 |
| Gauss-Legendre, order 4 (12 nodes) | 9.0e-16 | 1.4e-15 | 1.3e-15 |
| `nquad`, orders 1 / 2 / 3 | 4.3e-16 / 8.3e-16 / 2.0e-16 | 1.9e-16 / 3.2e-16 / 3.7e-16 | 4.0e-16 / 1.7e-16 / 5.0e-16 |
| `qmc_vectorized` (2¹⁴), orders 1 / 2 | 2.5e-7 / 9.4e-7 | 1.1e-7 / 1.0e-6 | 1.1e-7 / 9.6e-7 |
| `qmc_scalar`, `qmc` (2¹¹), orders 1 / 2 | 2.5e-7 / 2.7e-6 | 1.1e-7 / 6.7e-6 | 1.1e-7 / 6.7e-6 |

`run.py --c-quadrature` takes C from quadrature tables instead of the
closed form; the mixing variant then tabulates every `C_ab`, which is the
route `System.propagators(diag_C=False)` used to refuse without one.  At 31
time points it agrees with the hierarchy to 1.2e-8 (order 0), 8.8e-8
(order 1) and 2.1e-5 / 1.1e-3 (order 2, the two component pairs).

The same script on older code:

| code | what fails |
|---|---|
| `facf556` (the merged branches, before the fixes of 40a9288 and b1495bb) | isotropic variant: 1.0, 3.0 and 7.0 relative (factors 2, 4, 8) on every integrator (`iso_C`); Gauss-Legendre at orders 1-4: 5.8e-4 to 3.5e-3 (`t_min` not passed on, and the white-noise kink) |
| `3cc7115` (before the diag-C fast-path fix) | diagonal variant, `qmc_scalar` and `qmc` at order 2: 64 % |
| `7034888` (before `nquad` split at kinks) | `nquad` at order 2, every variant and pair: 2.2e-8 to 1.4e-7, in 1.0-2.7 s per pair (now 0.4-0.6 s); the order-3 tadpole (mixing): 6.9e-8, in 16 s (now 5.2 s) |

## Part B: multiplicative white noise, L0 (`multiplicative.py`)

```
dφ_a = (−γ_a φ_a + F_abc φ_b φ_c) dt + (g_ad + g'_adc φ_c) dW_d      (Itô, one point)
```

The noise term `½ ψ_a ψ_b S_ab(φ)` of the MSR action, `S = G Gᵀ`, gives C
(`S⁰`) and two local vertices with two ψ legs, ψψφ (`S¹`) and ψψφφ (`S²`),
each with the factor `−i²/2! = ½`; an order-1 ψψ vertex with `½ S⁰`
reproduces C to 1e-12, which fixes that convention.  The L1 workflow cannot
build a vertex with two ψ legs, so this part runs `compute_moment` and
`integrate_diagrams` directly, on a C from the built-in closed form.

With distinct rates `γ = (0.6, 1.6)` R is a diagonal matrix, and in the FG
channel the two ψ legs of the ψψφ vertex contract with the two φ legs of
the F vertex: two R propagators between the same two points, the diagrams
of the repeated-pair fix (63fc842).  Every integrator runs the matrix-R
case; `gauss_legendre` and `qmc_vectorized` did so only after the matrix-R
change, and raised `NotImplementedError` before it.

Reference: the same hierarchy with a state-dependent diffusion, order by
order in `F` and `g'`.  Under Itô the GG channel has no diagram and `⟨φ_a⟩`
has no `g'`-odd term; the hierarchy agrees.

| | scalar R, GL | matrix R, GL | matrix R, `nquad` (orders ≤ 1) | matrix R, `qmc_vectorized` (2¹³) | matrix R, `qmc_scalar` (2¹³) |
|---|---|---|---|---|---|
| worst over all tags and components | 2.4e-16 | 6.6e-16 | 6.2e-16 | 2.6e-5 | 2.6e-5 |
| seconds | 0.0 | 0.1 | 0.2 | 2.2 | 20.5 |

The two QMC columns are the same Sobol points: they agree to 2.4e-16 and
differ only in speed.

On `3cc7115`, before the repeated-pair fix, the matrix-R `qmc_scalar` run is
off by 2.7e-3 to 9.1e-3 in the FG channel and 2.3e-2 to 6.7e-2 in FH (the
ψψφφ vertex's two ψ legs meet the F vertex's two φ legs); the other
columns are unchanged.

## Part C: multiplicative white noise at L1, Itô and Stratonovich (`white_l1_multiplicative.py`)

```
dφ_a = (−γ_a φ_a + η_a + F_abc φ_b φ_c) dt + g_ak(φ) (∘) dW_k,  k = 0, 1, 2
g_ak(φ) = g0_ak + g1_akb φ_b,   one point,  t ≥ t_min = 0.4,  φ(t_min) = 0
η: the coloured noise of part A (λ = 0.3, σ_t = 0.7)
```

Two components, three Wiener processes, no symmetry in `F`, `g0` or `g1`.
The L1 system declares the white noise as
`MultiplicativeImpulse(g0, g1, interpretation)`: `D0 = g0 g0ᵀ` is the white
noise of C (dense, so `diag_C=False`), the rest of `D(φ) = g(φ) g(φ)ᵀ`
becomes the local vertices `G` (ψψφ) and `H` (ψψφφ) with the MSR factor
`−i²/2! = ½`, and `interpretation='stratonovich'` adds the noise-induced
drift `½ Σ_jk g_jk ∂_j g_ik = b + Lφ` as a source `B` (ψ) and a linear
vertex `L` (ψφ) with the factor `−i`.

Tags `(k_F, k_g)` count powers of `F` and `g1`: `F` (1, 0), `G` and `B`
(0, 1), `H` and `L` (0, 2).  Every vertex has weight ≥ 1, so a tag of
weight `w` collects vertex orders up to `w`; the script sums the package's
diagrams of each tag and compares with that tag's coefficient in the exact
hierarchy.  `⟨φ φ⟩` vanishes at odd weight and `⟨φ⟩`, `⟨φ φ φ⟩` at even
weight, so the three observables together cover vertex orders 0-4.

Reference: the moment hierarchy of the same Markov embedding, with the
generator in Hörmander form `L = f·∇ + ½ Σ_k (g_k·∇)(g_k·∇)`
(`white_hormander_reference.py` on `examples/reference/hormander_moments.py`;
no sft-wick code).  The noise-induced drift is never formed there, so the
package's conversion is not checked against itself.

Worst relative difference over the tags, component tuples and observables
(`white_l1_multiplicative_results.json`); the same table for both
interpretations:

| run | vertex orders | worst |
|---|---|---|
| scalar R, Gauss-Legendre (12 nodes) | 0-3 (`⟨φ⟩`, `⟨φφφ⟩`), 0-4 (`⟨φφ⟩`) | 9.0e-16 |
| scalar R, `qmc_vectorized` (2¹⁴) | 0-2 | 1.0e-6 |
| matrix R, `qmc_scalar` (2¹²) | 0-2 (`⟨φφ⟩`), 0-3 (`⟨φ⟩`) | 1.4e-5 (order 2), 1.1e-3 (order 3) |
| matrix R, `nquad` | 0-2 | 2.3e-7 |

Distinct times at one point, `⟨φ_a(t_min + 0.9) φ_b(t_min + 1.5)⟩`, orders
0-3:

| integrator | worst |
|---|---|
| `nquad` | 5.2e-8 |
| `qmc_vectorized` (2¹⁶) | 2.0e-7 |
| Gauss-Legendre (12 nodes) | 2.0e-3 |

Gauss-Legendre is algebraic on the two-time integrand: an internal time
crosses the earlier external time inside the domain, where C is kinked, and
the GL kink split pairs internal variables only.  Measured on the FG
channel: 2.0e-3 at 12 nodes, 1.1e-3 at 24, 1.8e-4 at 48, 5.4e-5 at 80.  The
FF channel behaves the same way, so this is a property of two external
times with white noise, not of the noise vertices.

What the run would have shown on older code:

| code | what fails |
|---|---|
| `7034888` (this branch's base) | `MultiplicativeImpulse` does not exist; the L0 route can express the Itô model only, and `ito=False` returns the Itô value in silence: for the ψψφ vertex at order 1 it returns 0 on all four integrators, while the Stratonovich coefficient of that tag is +6.886e-02 (a = 0) and −5.325e-02 (a = 1) |

Margins of the comparison, measured by mutating the lowering (worst
relative difference the Gauss-Legendre check then sees, tolerance 1e-9):

| mutation | Itô | Stratonovich |
|---|---|---|
| noise-induced drift dropped | 4e-16 | 1.0 |
| drift factor 1 instead of ½ | 4e-16 | 1.0 |
| drift contracted on the wrong `g1` slot | 4e-16 | 2.9e-1 |
| MSR factor 1 instead of ½ on the `D(φ)` vertices | 1.0 | 6.6e-1 |
| MSR factor `−i` instead of ½ | raises (the reality check) | raises |

## Limits, measured

- `nquad` at order 4 (64 four-dimensional diagrams) does not finish one
  evaluation of `⟨φ_0 φ_1⟩` in 600 s, split or not, so order 4 runs on
  Gauss-Legendre only.  An earlier version of this README said that order
  2 did not finish in 10 minutes; on `7034888` it finishes in 1.0-2.7 s
  (the `7034888` row of the table above).
- A matrix R runs on every integrator.  At the same Sobol points the
  batched loop is 11 times faster than the scalar one here (2.2 s vs
  20.5 s at 2¹³), and Gauss-Legendre reaches 6.6e-16 in 0.1 s.
- The numerical layer evaluates every R at equal times as 0, so it computes
  the Itô SDE whatever `ito=` says.  Under `ito=False` the extra
  equal-point R on a vertex with one ψ leg is still evaluated (it gives 0,
  which is exact: it and the Stratonovich Jacobian cancel, and the package
  emits neither); on a vertex with two or more ψ legs, or between two
  external operators, it now raises instead of returning the Itô value.  A
  Stratonovich model is computed in its Itô form (part C).
- Gauss-Legendre converges algebraically when the two external points are
  at different times (part C); `nquad` and QMC do not.
- Part C is one SDE per point: `Expansion.evaluate` refuses external points
  at different positions that a response chain joins, and since 0.5.x so do
  the L0 entry points, for any system rather than this noise alone.

## Files

| file | contents |
|---|---|
| `white_model.py` | parameters, the three variants, the `System` and its propagators |
| `white_reference.py` | the moment hierarchy of the Markov embedding |
| `run.py` | part A |
| `multiplicative.py` | part B |
| `white_l1_multiplicative.py` | part C: the L1 system, the per-tag comparison |
| `white_hormander_reference.py` | part C's reference: the generator in Hörmander form |

```bash
conda activate sft-wick
cd examples/demo5
python run.py                      # ~12 min, 10 of them the mixing variant's order 4
python multiplicative.py           # ~30 s
python white_l1_multiplicative.py  # ~20 min, 17 of them the two Gauss-Legendre runs
pytest ../../tests/test_demo5_white_noise.py -q
pytest ../../tests/test_multiplicative_noise_l1.py -q
```
