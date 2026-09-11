# Demo 4 — compound-Poisson noise asymmetric in points and components

A non-Gaussian example built so that the cumulants have no symmetry beyond
the one every cumulant has.  In demos 1-3 every cumulant was `δ_{a…}` times
a function symmetric in its leg points; that is the configuration in which
the leg-order defect of 0.4.2 (fixed in PR #11) gave the right answer.  On
the code before that fix this demo's level A is 47 % off.

```
dφ_a/dt = −γ φ_a + F_abc φ_b φ_c + η_a,          φ(0) = 0
η_a(x,t) = Σ_k h_a w_a(x − z_k) g_a(t − s_k) − ⟨·⟩,  events ~ Poisson(ν)
w_a(u) = exp(−u²/(2 s_a²)),    g_a = δ (white)  or  Θ(τ) e^{−τ/τ_a} (exponential)
```

Every event drives both components, with its own amplitude `h_a` (one of
them negative), width `s_a` and decay time `τ_a`.  Campbell's theorem gives
every cumulant as one source-point integral:

```
κ^(m)_{a₁…a_m}(x, t) = ν Π_j h_{a_j} · X_a(x) · G_a(t)
```

which is symmetric only when (component, point) pairs are permuted
together.  `F` has no index symmetry either.

| parameter | value |
|---|---|
| ν, γ | 1.5, 1.0 |
| h, s, τ | (1.0, −0.7), (0.6, 1.1), (0.3, 2.0) |
| points x, y, z, w | 0.0, 0.8, −0.5, 1.3 |
| observation time T | 1.7 |

## What is compared with what

| tier | package route | reference |
|---|---|---|
| level A, `F = 0` | the order-1 `κ^(m)` diagram, `m = 3, 4`, for every component tuple: R-contracted callable, raw callable (leg integrals done by the package; `equal_time` for white pulses), unequal external times | the closed form `K_R = ν Π h · X · T̃` (`noise.py`; `T̃` checked against direct quadrature to 1e-12), and the moment hierarchy |
| level B, `F ≠ 0` | `⟨φ_a(x) φ_b(y)⟩` for all four pairs, channel by channel (order 0, FK3, FF, FFK4), and the tadpole `⟨φ_a(x)⟩` | the moment hierarchy |

The moment hierarchy (`reference.py`, built on
`examples/reference/ito_moments.py`) is the exact generator of the Markov
process at the observation points: the fields for white pulses, the fields
and the pulses (a Poisson-driven Ornstein-Uhlenbeck process, started in its
stationary law) for exponential ones.  It is solved order by order in `F`
and in the cumulant order, so each coefficient is one channel of the
package.  It uses no sft-wick code.

The noise mixes the components, so C has off-diagonal entries; the
propagators come from the closed form (`c_closed_form_only=True`,
`diag_C=False`), which is `K_R` at `m = 2`.

## Results

Maximum relative difference over the component tuples (`level_a.py`,
`level_b.py`; stored in `level_a_results.json`, `level_b_results.json`):

| level A | white pulses | exponential pulses |
|---|---|---|
| 3-point, R-contracted vs closed form | 1.4e-16 | 4.7e-15 |
| 3-point, raw vs closed form | 6.6e-16 (GL 16) | 2.9e-5 (QMC 2¹⁸) |
| 3-point, hierarchy vs closed form | 5.0e-16 | 2.3e-14 |
| 3-point at unequal times, R-contracted | 2.3e-16 | 3.7e-16 |
| connected 4-point, R-contracted | 4.4e-16 | 1.8e-14 |
| connected 4-point, hierarchy | 5.9e-16 | 1.5e-13 |

| level B, package vs hierarchy | white pulses (GL 16) | exponential pulses (GL 32 / 24) |
|---|---|---|
| order 0 | 1.1e-16 | 4.2e-16 |
| FK3 | 4.5e-16 | 2.8e-15 |
| FF | 7.4e-16 | 5.6e-15 |
| FFK4 | 1.3e-15 | 4.0e-9 |
| tadpole `⟨φ_a⟩` | 2.1e-16 | 1.9e-15 |

On `817375f` (0.4.2 plus documentation, before the leg-order fix) level A is
46.8 % off on every route: the R-contracted and raw white-pulse routes and
the R-contracted exponential route.

## Limits, measured

- The raw exponential kernel `G_a(t)` depends on the smallest leg time, so
  its integrand is kinked where two leg times cross.  Gauss-Legendre
  converges as `n^-2` on it: 5.7e-2, 1.5e-2, 3.7e-3, 1.7e-3 at 16, 32,
  64, 96 nodes; QMC reaches 8.3e-4 at 2¹⁴ samples and 2.9e-5 at 2¹⁸.  The
  R-contracted route is exact.
- The R-contracted `K_R` also depends on the smallest partner time.  In
  FFK4 two partners are F-vertex times, so that channel keeps an algebraic
  rate (3.1e-8 at 16 nodes, 5.4e-10 at 32) for exponential pulses; for
  white pulses level B uses the raw `equal_time` vertex, which is smooth.
- White noise kinks C on its time diagonal.  Gauss-Legendre splits the
  domain there (see the CHANGELOG); without the split, white-pulse FF was
  7.9e-5 off at 64 nodes.

## Files

| file | contents |
|---|---|
| `noise.py` | parameters, the closed forms (`K_R`, `T̃`, C), the hierarchy's jump moments; numpy only |
| `system.py` | the sft-wick `System`, the coupling callables and the closed-form C |
| `reference.py` | the moment hierarchy at the observation points |
| `level_a.py`, `level_b.py` | the two tiers |

```bash
conda activate sft-wick
cd examples/demo4
python level_a.py      # ~5 s
python level_b.py      # ~2 s
pytest ../../tests/test_demo4_asymmetric_noise.py -q
```
