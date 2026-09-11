# Demo 5 — white noise on every integrator, additive and multiplicative

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
of `⟨φ_0 φ_1⟩` and the tadpole; `nquad` the tadpole.

Reference: the moment hierarchy of the Markov embedding `(φ, η)` at the
observation points, with `η` an Ornstein-Uhlenbeck process started in its
stationary law (`white_reference.py`, on `examples/reference/ito_moments.py`; no
sft-wick code).  Its coefficient of `F^k` is order `k` of the package.

Worst relative difference over the component tuples (`results.json`):

| | mixing | diagonal | isotropic |
|---|---|---|---|
| Gauss-Legendre, orders 0-3 (12 nodes) | 6.6e-16 | 9.2e-16 | 5.6e-16 |
| Gauss-Legendre, order 4 (12 nodes) | 9.0e-16 | 1.4e-15 | 1.3e-15 |
| `nquad`, order 1 | 4.3e-16 | 1.9e-16 | 4.0e-16 |
| `qmc_vectorized` (2¹⁴), orders 1 / 2 | 2.5e-7 / 9.4e-7 | 1.1e-7 / 1.0e-6 | 1.1e-7 / 9.6e-7 |
| `qmc_scalar`, `qmc` (2¹¹), orders 1 / 2 | 2.5e-7 / 2.7e-6 | 1.1e-7 / 6.7e-6 | 1.1e-7 / 6.7e-6 |

The same script on older code:

| code | what fails |
|---|---|
| `facf556` (the merged branches, before the fixes of 40a9288 and b1495bb) | isotropic variant: 1.0, 3.0 and 7.0 relative (factors 2, 4, 8) on every integrator (`iso_C`); Gauss-Legendre at orders 1-4: 5.8e-4 to 3.5e-3 (`t_min` not passed on, and the white-noise kink) |
| `3cc7115` (before the diag-C fast-path fix) | diagonal variant, `qmc_scalar` and `qmc` at order 2: 64 % |

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
of the repeated-pair fix (63fc842).  Matrix R runs on the scalar loops only.

Reference: the same hierarchy with a state-dependent diffusion, order by
order in `F` and `g'`.  Under Itô the GG channel has no diagram and `⟨φ_a⟩`
has no `g'`-odd term; the hierarchy agrees.

| | scalar R, Gauss-Legendre | matrix R, `nquad` (orders ≤ 1) | matrix R, `qmc_scalar` (2¹³) |
|---|---|---|---|
| worst over all tags and components | 2.4e-16 | 6.2e-16 | 2.6e-5 |

On `3cc7115`, before the repeated-pair fix, the matrix-R `qmc_scalar` run is
off by 2.7e-3 to 9.1e-3 in the FG channel and 2.3e-2 to 6.7e-2 in FH (the
ψψφφ vertex's two ψ legs meet the F vertex's two φ legs); the other
columns are unchanged.

## Limits, measured

- `nquad` on the order-2 white-noise integrand did not finish one
  evaluation in 10 minutes; adaptive quadrature meets the kink everywhere.
  It runs the tadpole only.
- A matrix R runs on the scalar loops only (`qmc_scalar`, `qmc`, `nquad`).
- The package is Itô numerically whatever `ito=` says: with `ito=False` the
  extra equal-time terms are kept but evaluate to 0, because the retarded
  R vanishes at equal times.  A Stratonovich model needs its noise-induced
  drift written into `F`.

## Files

| file | contents |
|---|---|
| `white_model.py` | parameters, the three variants, the `System` and its propagators |
| `white_reference.py` | the moment hierarchy of the Markov embedding |
| `run.py` | part A |
| `multiplicative.py` | part B |

```bash
conda activate sft-wick
cd examples/demo5
python run.py              # ~12 min, 10 of them the mixing variant's order 4
python multiplicative.py   # ~30 s
pytest ../../tests/test_demo5_white_noise.py -q
```
