# Demo 6: repeated and static non-local vertices, cubic and quartic drift

Five structures the package supports and no test had evaluated: two copies
of one non-local vertex in a diagram, a static (ndarray) non-local coupling,
a non-local vertex at `m = 5`, an `m = 2` one, and a quartic local vertex
next to a cubic one.

```
dφ_a = (−γ_a φ_a + F_abc φ_b φ_c + G_abcd φ_b φ_c φ_d + X_a + η_a) dt + dW_a
φ(t_min) = 0,   t_min = 0.4,   N = 2
```

| source | how it enters the package | cumulants |
|---|---|---|
| `dW` | `CustomImpulse` σ², the C propagator | `σ²_ab(x,y) = Q_ab ∫dz w_a(x−z) w_b(y−z)`, `w_a = e^{−u²/2ℓ_a²}` |
| `X`, a force constant in space and time | static ndarray vertices `X2 … X5` | `κ^(m)`, symmetric with entries of both signs |
| `η`, white jumps whose jump vector does not depend on the point | static `equal_time` ndarray vertices `W2`, `J3 … J5` | `ν E[J^{⊗m}] δ(t_1−t_2)…` |
| demo 4's compound-Poisson pulses (parts D and F) | callable vertices, raw and R-contracted | Campbell's theorem, `examples/demo4/poisson_noise.py` |

Nothing here is symmetric beyond what the physics forces: the rates,
white-noise widths and amplitudes differ between the components (one
amplitude is negative), `F` and `G` have no index symmetry, the cumulant
tensors are symmetric in their indices (a cumulant is) with entries of both
signs, and the external points and times are all distinct.

| parameter | value |
|---|---|
| γ | (0.8, 1.3), or (1.05, 1.05) where a scalar R is needed |
| t_min | 0.4 |
| Q, ℓ | ((0.5, −0.2), (−0.2, 0.35)), (0.7, 1.2) |
| F, G | no index symmetry; see `vertex6_model.py` |
| κ^(m), ν E[J^{⊗m}] | symmetric, entries of both signs |

## References

Every number is compared against one or two references that share no code
with the package:

- **closed forms** (`vertex6_model.py`): at `F = G = 0` the moment is a sum
  over set partitions of the external points into cumulant blocks, each
  block a static `κ^(m) Π_j g_{a_j}(t_j)` or an equal-time
  `κ_eq^(m) ∫ ds Π_j e^{−γ_{a_j}(t_j−s)}`; the C propagator and the C of an
  `m = 2` vertex likewise;
- **the moment hierarchy** (`vertex6_reference.py`, on
  `examples/reference/ito_moments.py`): the generator of the Markov process
  at the observation points, solved order by order in the vertices.  Each
  vertex species has its own tag, so a tag is exactly one channel of the
  package.  Observation times may differ: the variables of a point are
  frozen after its time, and the intervals are solved one after another
  (`solve_multitime`, which works for any `PolySDE`, including demo 4's
  hierarchy);
- **Campbell's theorem** for parts D and F, through demo 4's
  `poisson_noise.py`.

## Parts

| part | script | what it evaluates |
|---|---|---|
| A | `vertex6_repeated.py` | two copies of one cubic vertex: the order-2 six-point function, static and `equal_time`, every component tuple |
| B | `vertex6_interacting.py` | `F ≠ 0` with static cumulants: order 0, `F X3`, `F F`, `F F X4` of `⟨φ_a φ_b⟩`; with `--two-copies`, `F X3 X3` and `F J3 J3` of the five-point function |
| C | `vertex6_cubic.py` | a quartic `G_{a;bcd} ψ φ φ φ` and a cubic `F` vertex in one system, orders 2-3 of `⟨φ_a⟩` and `⟨φ_a φ_b⟩` |
| D | `vertex6_high_cumulants.py` | `m = 4` and `m = 5` on every route: R-contracted callable, raw callable, static, static `equal_time` |
| F | `vertex6_gaussian_vertex.py` | an `m = 2` vertex: its order-1 contribution is the C of that noise, and its channels with `F` |

Demo 4 gains one script from the same work,
`examples/demo4/poisson_level_b_order4.py`: the order-4 `F³κ³` channel of
`⟨φ_a(x) φ_b(y)⟩` (30 diagrams) against demo 4's hierarchy at tag `F³ μ¹`.

## Results

Worst relative difference over the component tuples; the results files hold
every row.

### Part A: two copies of one vertex (`repeated_results.json`)

The order-2 six-point function at six distinct points and six distinct
times, against the closed form (the ten splits) and the hierarchy.

| vertex | route | rates | tuples | vs closed form | vs hierarchy |
|---|---|---|---|---|---|
| static `X3` | Gauss-Legendre 8 | 1.05, 1.05 | 64 | 5.9e-16 | 9.8e-16 |
| static `X3` | `qmc_vectorized` 2¹⁴ | 1.05, 1.05 | 3 | 6.7e-05 | 6.7e-05 |
| static `X3` | `qmc_scalar`, `qmc` 2¹⁶ | 0.8, 1.3 | 3 | 1.2e-05 | 1.2e-05 |
| `equal_time` `J3` | Gauss-Legendre 12 | 1.05, 1.05 | 64 | 9.7e-16 | 2.4e-15 |
| `equal_time` `J3` | `qmc_vectorized` 2¹⁴ | 1.05, 1.05 | 3 | 1.3e-07 | 1.3e-07 |
| `equal_time` `J3` | `nquad` | 1.05, 1.05 | 3 | 4.1e-16 | 2.2e-16 |
| `equal_time` `J3` | `qmc_scalar`, `qmc` 2¹¹ | 0.8, 1.3 | 3 | 2.6e-05 | 2.6e-05 |
| `equal_time` `J3` | `nquad` | 0.8, 1.3 | 1 | 1.1e-16 | 2.3e-16 |

The hierarchy and the closed form agree to 5.7e-16 (static) and 2.2e-15
(equal-time).  **The `equal_time` rows are the ones that fail on `7034888`**:
there the ten contractions were merged into one diagram carrying the time
structure of the first, 40.4 % and 26.1 % off at these times (fixed in this
branch; see `tests/test_equal_time_nonlocal.py`).

### Part B: static cumulants with `F ≠ 0` (`interacting_results.json`)

Against the hierarchy at the tag with the same vertices, at distinct points
and times (three component pairs; five component tuples for the five-point
channels):

| channel | times | route | rates | worst |
|---|---|---|---|---|
| order 0 | distinct | Gauss-Legendre 8 | 1.05, 1.05 | 0.0 |
| `F X3` | distinct | Gauss-Legendre 8 | 1.05, 1.05 | 3.8e-16 |
| `F X3` | distinct | `qmc_vectorized` 2¹⁶ | 1.05, 1.05 | 8.1e-06 |
| `F X3` | distinct | `qmc_scalar` 2¹⁴ | 0.8, 1.3 | 7.9e-05 |
| `F F X4` | distinct | Gauss-Legendre 8 | 1.05, 1.05 | 4.3e-16 |
| `F F X4` | distinct | `qmc_scalar` 2¹⁴ | 0.8, 1.3 | 2.5e-04 |
| `F F` | equal | Gauss-Legendre 12 | 1.05, 1.05 | 5.4e-16 |
| `F F` | distinct | Gauss-Legendre 16 | 1.05, 1.05 | 6.3e-16 |
| `F F` | distinct | `qmc_vectorized` 2¹⁸ | 1.05, 1.05 | 1.7e-08 |
| `F X3 X3` (5-point, order 3) | distinct | Gauss-Legendre 8 | 1.05, 1.05 | 1.6e-15 |
| `F X3 X3` (5-point, order 3) | distinct | `qmc_vectorized` 2¹⁸ | 1.05, 1.05 | 9.2e-08 |
| `F J3 J3` (5-point, order 3) | distinct | `qmc_vectorized` 2¹⁸ | 1.05, 1.05 | 1.7e-06 |
| `F J3 J3` (5-point, order 3) | distinct | Gauss-Legendre 8 | 1.05, 1.05 | 1.0e-12 |

The two five-point channels are two copies of one vertex with an
interaction: 5 diagrams of `F X3 X3` and 35 of `F J3 J3`.  The static one
has no C propagator and no equal-time vertex; the equal-time one has
parents that include fixed external points, the kink that is cut since
2026-09-12 (it was 5.5e-02 on the same route before the cut, and QMC was
the route for it).

### Part E: demo 4 at order 4 (`examples/demo4/level_b_order4_results.json`)

`F³κ³` of `⟨φ_a(x) φ_b(y)⟩`, 30 diagrams, against demo 4's hierarchy at tag
`F³ μ¹`, all four component pairs:

| pulses | route | n_gauss | worst | convergence |
|---|---|---|---|---|
| white | raw `equal_time` κ³ | 12 | 7.5e-16 | 8: 3.5e-12, 12: 3.8e-16 |
| exponential | R-contracted `K_R` | 24 | 2.2e-09 | 12: 1.0e-07, 16: 1.9e-08, 24: 1.8e-09, 32: 3.4e-10 |

### Part C: a quartic and a cubic local vertex (`cubic_results.json`)

Channel by channel against the hierarchy with a cubic drift term.
`⟨φ_a⟩` has one external point, so its channels have no external time to
cross:

| observable | channel (order) | route | worst |
|---|---|---|---|
| `⟨φ_a⟩` | `F G` (2) | Gauss-Legendre 12 | 2.9e-16 |
| `⟨φ_a⟩` | `F F F` (3) | Gauss-Legendre 12 | 6.9e-16 |
| `⟨φ_a⟩` | `F G G` (3) | Gauss-Legendre 12 | 1.0e-15 |

`⟨φ_a φ_b⟩`, at equal and at distinct external times (the order-3 channels
have 68 and 44 diagrams and take the cheaper settings):

| channel (order) | GL, equal times | GL, distinct times | GL, distinct, before the cut | QMC, distinct times |
|---|---|---|---|---|
| `F F` (2) | 5.8e-16 (12) | 3.1e-16 (16) | 7.5e-04 | 1.7e-08 (2¹⁸) |
| `G G` (2) | 6.7e-16 (12) | 5.5e-15 (16) | 3.3e-02 | 3.0e-06 (2¹⁸) |
| `F F G` (3) | 2.6e-15 (10) | 2.0e-15 (10) | 1.6e-03 | 9.3e-05 (2¹⁴) |
| `G G G` (3) | 1.4e-13 (10) | 3.5e-15 (10) | 1.2e-03 | 4.9e-04 (2¹⁴) |

("before the cut" is the same run before the domain was cut at a kink
against a fixed external time; see "Limits, measured" below.)

### Part D: `m = 4` and `m = 5` on every route (`high_cumulants_results.json`)

Level A (`F = 0`): the connected `m`-point function, every component tuple
(16 at `m = 4`, 32 at `m = 5`) where the route is cheap, four otherwise.
demo 4's compound-Poisson noise against Campbell's closed form:

| m | pulses | route | equal times | distinct times |
|---|---|---|---|---|
| 4 | white | R-contracted | 5.4e-16 | 6.6e-16 |
| 4 | white | raw (`equal_time`), GL 16 | 1.0e-15 | |
| 4 | exponential | R-contracted | 2.2e-14 | 7.5e-15 |
| 4 | exponential | raw, `qmc_vectorized` 2¹⁸ | 2.5e-05 | |
| 4 | exponential | raw, Gauss-Legendre 16 | 1.2e-01 | |
| 5 | white | R-contracted | 2.5e-15 | 2.8e-15 |
| 5 | white | raw (`equal_time`), GL 16 | 3.6e-15 | |
| 5 | exponential | R-contracted | 8.8e-13 | 3.4e-14 |
| 5 | exponential | raw, `qmc_vectorized` 2¹⁸ | 1.7e-03 | |

demo 4's hierarchy reproduces Campbell's closed form to 1.9e-13 at equal
times and 3.2e-14 at distinct ones (the latter through the frozen-variable
solver).  demo 6's static vertices, at distinct times, against the closed
form:

| m | vertex | route | rates | worst |
|---|---|---|---|---|
| 4, 5 | `X4`, `X5` | Gauss-Legendre 8 | 1.05, 1.05 | 2.9e-16 |
| 4, 5 | `X4`, `X5` | `qmc_vectorized` 2¹⁶ | 1.05, 1.05 | 1.2e-06 |
| 4, 5 | `X4`, `X5` | `qmc_scalar` 2¹⁴ | 0.8, 1.3 | 4.9e-06 |
| 4, 5 | `J4`, `J5` | Gauss-Legendre 8 | 1.05, 1.05 | 5.2e-16 |
| 4, 5 | `J4`, `J5` | `nquad` | both | 4.7e-16 |
| 4, 5 | `J4`, `J5` | `qmc_scalar`, `qmc_vectorized` | both | 1.2e-09 |

demo 6's hierarchy reproduces the same closed form to 1.3e-15.

### Part F: the `m = 2` vertex (`gaussian_vertex_results.json`)

Order 1, which must equal the C of that noise exactly, at distinct points
and distinct times:

| kernel | route | rates | worst |
|---|---|---|---|
| static `X2`, white `W2` | Gauss-Legendre 8, `nquad` | 1.05, 1.05 | 2.1e-16 |
| static `X2`, white `W2` | `nquad` | 0.8, 1.3 | 2.9e-16 |
| static `X2`, white `W2` | `qmc_vectorized` 2¹⁴ | 1.05, 1.05 | 5.8e-09 |
| static `X2`, white `W2` | `qmc_scalar`, `qmc` 2¹⁴ | 0.8, 1.3 | 1.6e-08 |
| demo 4, R-contracted (white, exponential) | Gauss-Legendre 16 | | 0.0 |
| demo 4, raw (white, `equal_time`) | Gauss-Legendre 16 | | 2.0e-16 |
| demo 4, raw (exponential) | Gauss-Legendre 32 / `qmc_vectorized` 2¹⁶ | | 2.4e-15 / 4.4e-06 |

With `F ≠ 0`, against the hierarchy's tags:

| channel | times | route | worst |
|---|---|---|---|
| `F K2`, `F W2` in `⟨φ_a⟩` | | Gauss-Legendre 12 | 7.5e-16 |
| `F F K2`, `F F W2` in `⟨φ_a φ_b⟩` | equal | Gauss-Legendre 12 | 7.0e-16 |
| `F F K2`, `F F W2` in `⟨φ_a φ_b⟩` | distinct | Gauss-Legendre 16 | 5.6e-16 |
| `F F K2`, `F F W2` in `⟨φ_a φ_b⟩` | distinct | `qmc_vectorized` 2¹⁸ | 1.4e-07 |

The raw exponential row was 2.1e-04 and the distinct-time `F F K2` /
`F F W2` row 8.5e-04 before the domain was cut at a kink against a fixed
external time.

## Limits, measured

- **The kink at a fixed external time is cut** (was a limit until
  2026-09-12).  `integrate_moment_gauss_legendre` splits the domain where
  two *internal* times whose order the causal structure leaves free cross;
  a C propagator with a kinked diagonal (white noise) or an equal-time
  vertex with two parents can put the same kink between an internal time
  and an external point held at its own time, where a constant is not an
  ordering between two variables.  The variable's range is now cut at that
  time instead.  Measured on the `F F` channel of `⟨φ_0(x) φ_1(y)⟩` against
  the hierarchy:

  | external times | 8 | 16 | 24 | 32 nodes | `qmc_vectorized` 2¹⁸ |
  |---|---|---|---|---|---|
  | equal (1.9, 1.9) | 6.5e-14 | 5.4e-16 | 1.3e-15 | 4.4e-15 | 7.9e-08 |
  | distinct (1.9, 1.3), before | 4.2e-03 | 6.9e-04 | 4.8e-04 | 1.6e-04 | 7.7e-09 |
  | distinct (1.9, 1.3), now | 6.3e-16 | 6.3e-16 | 3.1e-16 | 1.9e-15 | |

  The cost is one integration per cut: 2.0 pieces per diagram on this
  channel, and none at all with every external at one time, where the kink
  is the boundary of the domain.  Channels with no C propagator and no
  equal-time vertex (`F X3`, `F F X4`, `F X3 X3`, every level-A `m`-point
  function) converged exponentially at distinct times before the cut as
  well.  QMC is unaffected throughout.
- **A matrix R runs on the scalar loops only.**  Distinct rates make R a
  diagonal matrix; `gauss_legendre` and `qmc_vectorized` refuse it, so those
  configurations run on `qmc_scalar`, `qmc` and `nquad`
  (`vertex6_repeated.py --pending` records the refusals).
- **The raw exponential kernel is kinked and peaked.**  `κ^(m)` of demo 4's
  exponential pulses depends on the smallest leg time, and the fast
  component (τ = 0.3) makes it sharp; Gauss-Legendre converges
  algebraically and costs `n^m` evaluations per leg order, of which there
  are up to `m!`.  At `m = 4`, the all-fast component tuple is 4.6e-01 off
  at 8 nodes, 1.2e-01 at 16 and 5.3e-02 at 24, while QMC reaches 1.4e-03 at
  2¹⁶ and 1.2e-05 at 2¹⁸.  The R-contracted route is exact.
- **The order-4 expansion of a four-point observable did not finish in 20
  minutes** (`F` and a cubic non-local vertex, `⟨φ_a φ_b φ_c φ_d⟩`), so the
  two-copies-with-an-interaction channel is `F X3 X3` at order 3 of the
  five-point function, whose expansion takes about a minute.

## Files

| file | contents |
|---|---|
| `vertex6_model.py` | parameters, the cumulant tensors, the closed forms; numpy only |
| `vertex6_reference.py` | the moment hierarchy, with several observation times |
| `vertex6_system.py` | the `System`, the closed-form C, the vertex composition of a diagram |
| `vertex6_repeated.py`, `vertex6_interacting.py`, `vertex6_cubic.py`, `vertex6_high_cumulants.py`, `vertex6_gaussian_vertex.py` | parts A, B, C, D, F |

```bash
conda activate sft-wick
cd examples/demo6
python vertex6_repeated.py          # part A
python vertex6_interacting.py       # part B (--two-copies for the five-point channel)
python vertex6_cubic.py             # part C
python vertex6_high_cumulants.py    # part D
python vertex6_gaussian_vertex.py   # part F
cd ../demo4 && python poisson_level_b_order4.py
pytest ../../tests/test_demo6_static_vertices.py \
       ../../tests/test_demo6_cubic_quartic.py \
       ../../tests/test_demo6_high_cumulants.py \
       ../../tests/test_demo4_order4_channel.py -q
```
