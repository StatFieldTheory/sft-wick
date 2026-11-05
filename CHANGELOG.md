# Changelog

## Unreleased

### Fixed: rates a relative test tells apart were called equal

`DiagonalA` lowered rates to one scalar R, component 0's, when `np.allclose` called
them equal.  Its absolute floor of 1e-8 called small rates equal whatever their
ratio, and its relative tolerance of 1e-5 left a 5e-6 relative error in the rate.
At `gamma = [1e-9, 5e-9]`, `t = 1e9`, R of the second component was too large by a
factor 55 and C by a factor 4.32.  At `[1e-9, 0]`, C came out at 0.43 of its value.
The result depended on which rate was listed first.  Rates are now equal when they
differ by at most 1e-9 of the first, the tolerance of an index-free C, with no floor.
The reference is the OU solution written out in the test.  Rates that differ by more
than 1e-9 now take a matrix R, which costs more than a scalar one.  From a difference
of 1e-9 to one of about 1e-5, an explicit `iso_R=True` now raises, as `iso_C=True` does
from a difference that depends on the time and the table (about 2e-9 at `t = 1.5`, 4e-8
at `t = 0.05`), since neither system is what those flags say: float32-rounded rates
reach this.  `sft-wick validate` and `sft-wick run` refuse `iso_R` at `expand.iso_R`
with status 2 when a selected diagram reads an R; an R absorbed into an
`already_R_contracted` vertex is not read.  They refuse `iso_C` at `expand.iso_C` from
exact C at one point, the last `t_final`.  The integrators read the table at every
time, so close to that refusal `validate` can pass and the run end in a `ValueError`,
status 1, with no key named.

### Fixed: wrong numbers at L0

| Case | Before | Now |
|---|---|---|
| A propagator label that names nothing (`fixed_indices={}`, N = 2, order 1) | 0.19525, the trace; the four pinned values are 0.04149, 0.05457, -0.01220, 0.00263 | `KeyError` on every integrator |
| A flagged and a plain vertex sharing a coupling name, order 2 | -134.08 against -182.91 (`equal_time`), -57.30 against -113.02 (`already_R_contracted`) | both engines give the hand count |
| `integrate_moment` with positions on a cache with no table | 0.51419, against `C(x, y)` 0.11473 | 0.11473 on all four methods |
| A time-only table asked for a C it does not hold | `C(x, x)`, 0.31946 against 0.10371; a factor e on a σ² of position; a factor 25 at `x = y = 2` | `ValueError` on every integrator, `DiagramIntegrand.evaluate` and `make_scipy_integrand`, unless κ² and σ² read there as where the table was built, at six times across its range |
| `dblquad` C at or below `t_min` with an R that is not causal | -0.12028, and -1.82e17 at `C(0.1, -40)` | 0 on every method |
| L0 integrals on a model with `t_min` other than 0 | 0.0709183 against 0.0729964, or a table refusal | the integrals read the model's `t_min` |
| `integration_bounds` with no `t_min` under a model with `t_min = 2` | 1.46834 against 0.326142, a default of 0 | `t_min` is required; the method holds no cache to read it from |
| `precompute_C_table` a second time on one cache | the grid read the old table, so a rebuild at another `direction` kept C there under the new label: 0.31946 against 7.98644, past the time-only refusal, and the position it did hold was refused | the old table is cleared first |
| A time-only table beside a far stronger component | a component 1e14 weaker could depend on position unseen: a factor 3.08 at r = 1.5, 90 at r = 3 | each entry is compared on its own scale |
| Two times on a time-only table at a large time origin | a lag up to `1e-9 max(1, abs(t))` read the diagonal line at the mean time: 0.33333 against 0.21594 at t = 1e9, lag 0.9, and at t = 1e6 with gamma = 1000 | the tolerance is `1e-9` of the table's span, or 8 units of rounding of the times |
| `clear_cache` on a time-only table | the rules of the removed table stayed, so `c_method_resolved` reported them and a later C took them: 3.1e-2 off where a fresh cache is 1.4e-12 off | they go with the table; a spatial table keeps its own |
| A half table on a one-sided kernel that passed the probe | 58 and 73 percent off, translation; 0.1044 against 0.1545, general; 22 percent on a kernel of short reach; 0.79377 against 0.57343 by the separation read first | half a table only for a declared kernel (see below) |
| A translation table on noise not even in `x - y`, such as a moving pattern `κ²(x - y - v (t - t'))` | the orientation read first: 0.889096 against 0.437489, a factor 2.03; order 2 of `<phi phi>` 2.5385 against 2.1283 | `ValueError` at each separation the table reads, full or lazy; `homogeneity='general'` holds both orientations |
| `nquad` on an undeclared closed-form C and a cache with no table | the kink probe read the ambient `direction`, where `C_value` reads the points: `C ∝ x1 x2` ran unsplit at `direction = 0`, 9.4e-07 off under an estimate of 1.5e-08 | the probe reads C where the integrator does; on a time-only table, at the table's position |
| `make_scipy_integrand` on a callable coupling with `directions` keyed by point | `KeyError`, while `evaluate` takes that spelling | takes it too |
| `collect_by_diagram` on a hand-built sum with a delta | -0.81172 against -1.67758 | every occurrence of a dummy index is renamed, and a merge that is not injective is declined |
| `collect_by_diagram` on `F_ij (R_ai R_bi + R_aj R_bi)`, one index on both legs against two | 813 against a direct sum of 797 on integer tables, 895 before the fix in the row above | an index that meets itself is kept as `i -> i`, and a merge is made only when the renaming maps every propagator of one term onto the other; a C at one point is matched either way round, so `C_jk(w, w) C_ak(x, w)` and `C_jk(w, w) C_aj(x, w)` still merge; the matching is searched depth first and stops at the first renaming that completes, so seven C at one point take 0.1 ms, not 0.22 s and 320 MB |
| `compute_moment` at a `psi(3) phi(3) chi(2)` vertex, order 2, both engines | -215.21 against -129.21 for the Wick sum before collection | an index is renamed only onto one summed over the same range |
| `collect_by_diagram` on `sum_i G_ai (R_ai + R_ia)` with `a` free | 8 against a direct sum of 12 on integer tables | only an index summed by an enclosing `SumOverIndex` is renamed; a free index is fixed, so a hand-built sum with no `SumOverIndex` is no longer merged |
| `collect_by_diagram` on `sum_{i<2} sum_{j<3} F_ij (R_ai + R_aj)` | 186 against 231 | an index is renamed only onto one summed over the same range |
| `collect_by_diagram` on `sum_ij delta_ia F_ij (R_ai + R_aj)`, the delta outside the inner sum | 14 against 18 | a factor outside the inner sum that carries a dummy is renamed with each term |
| `collect_by_diagram` on `sum_{i<3} F_i (2 + R_ii(x, x))`, a term of the inner sum with no propagator | the coupling dropped from that term: 6 against -4 on integer tables | the term keeps the outer factors |
| `collect_by_diagram` on `sum_ij G_ij H(i) (R_ai + R_aj)`, where `i` is also a free point | the index swap moved the point: -0.79652 against -0.67733 | indices and points are renamed separately |
| An `already_R_contracted` vertex with a phi leg | one term 0.0 against 2.0 | `Vertex` refuses a phi leg there |
| `evaluate_coupling_batched` on `I K` with a real K | zeros with a `ComplexWarning`; the scalar entry raised | both keep the factor: `2j`, `3j` |

`t_min` of an L0 integrator now defaults to `None`, the model's value, and an explicit
value that differs from the model's raises `ValueError`.  A conflict in one position
group now raises the position `ValueError` on every integrator, with or without a
table, on a diagram that reads no position too: the order-0 `⟨φ(x) ψ(y)⟩` at two
positions returned 0.0 at L0, and `qmc_vectorized` and `gauss_legendre` returned a number
at order 1.  A leg with no label at all still gives the trace.

### Fixed: the rule of C, the kink probe and the time-only rescale

| Case | Before | Now |
| --- | --- | --- |
| A `GeneralKappa2.fn` object with a field named `temporal`, whose time shape changes with position | taken for separable: the origin's rule at every pair, 0.73 off at gamma 0.5, T 20, width 1e-3, with no warning | a private flag of the separable kernels decides, and the pair finds its rule |
| A separable kernel not finite at r = 0 | `dblquad` at every separation: 5.3e-16 against 4.6e-4 at sigma_t 1e-3, T 30 | the pair finds its rule, and `c_source` is `quadrature:per_position` |
| The same kernel raising at r = 0 | `ZeroDivisionError` on the first read of a lazy table | the table falls back to quadrature per separation |
| A weak cusp beside a strong smooth entry of κ² | read as smooth: order 2 was 1.3e-3 off at 8 nodes where the same kernel with equal entries is 1.5e-5 off | each entry is judged on its own scale |
| An undeclared closed-form C with every external pinned past `t_final` | probed up to `t_final` only: at `t_final = t_min` no split, 8.3e-3 off at 16 nodes | probed over the domain, up to the latest pinned time |
| A κ² that collapses a one-element array, on a time-only table at distinct positions | `integrate_two_point_qmc` wrapped the positions: 46 to 94 percent off, or `IndexError` | the kernel receives the positions as given |
| A per-build `c_method` of a build, failed or not | stayed on the instance: 44 percent off at 4 nodes on later direct cells, and an instance set to `dblquad` took it too | a per-build method reaches the cells of its build alone, whatever the instance's setting |
| A kernel feature at interior times under `c_method='auto'` | the rule was read at five cells: with `gamma = 5`, `T = 16`, a burst at `t = 14` gave up to 1.94 off at width 0.02 and 4e-5 at width 0.2, no warning | the rule is read at 23 cells (see below): the node count rises, or `dblquad` with its warning |
| A kernel not symmetric in time under `c_method='auto'` | no cell with `t1 < t2` was read: 5e-2 and 1e-1 off on a ridge at lag 8, no warning, on a valid covariance of a moving pattern too | those cells are read |
| Separations or angles below the key of a lazy table | a key of 10 decimals put them on one table, so both took the value of the one read first: order 2 was 23 percent low at positions of order 1e-13 and below, on every integrator; an angle below 1e-5 collapsed in any units; 1.02e-9 and 1.04e-9 collapsed too | a key of 13 significant digits, the separation in general mode too, and for rotation the distances between unit vectors in place of a cosine |
| `integrate_two_point_qmc` on a time-only table where kappa2 at `t_ref` is at most 1e-30 | 0.0 against 3.7e-6 where noise has decayed by `t_ref`, 0.0 against 2.5e-32 at an amplitude of 1e-31, and 0.0 against 0.0115 where it has ended | the ratio has no floor, and a kappa2 that is 0 at `t_ref` and not throughout is refused; a component with no noise keeps 0 |
| A chain of aliases, an equal-time leg as partner of an absorbed leg | `KeyError` on `qmc_scalar` and `nquad` with a callable coupling | every leg reads the end of its chain |
| `DiagramIntegrand.to_latex` with an absorbed R | printed R factors the integrand does not hold | prints the factors it multiplies |
| `ou_exponential_phi` on a batch with one small time | numpy overflow warnings on correct input | the small-rate branch reads its own elements |

The rule search reads 23 cells: the five extreme ones, the mirrors of their two
off-diagonal cells, ten more on the diagonal and three lags on both sides.  A kernel
that fails at the extreme cells stops there, and one that passes reads all 23.  The diagonal covers a rate times the
range up to about 100, and a feature beyond that can still go unread.

The scalar loop (`qmc_scalar`, `qmc`, `nquad`) on a closed-form-only cache costs about
25 percent more CPU: 78.9 s against 63.1 s on demo 5's matrix-R run on `qmc_scalar` at
`2^12` samples.

A lazy table now builds a second table for separations that differ by more than 1e-12
relative, where the old key collapsed them; the rescaled separable path pays nothing
for that.  The key has no floor at the rounding of the positions: separations up to
9 ulps of the coordinates shared the r = 0 table, so at x = 1e10 and a width of 2e-6 the
ratio to C(x, x) was 1 against 1e-16 at 9 ulps, and moving the origin changed it.  Two
positions meant to coincide that differ by rounding now cost a table of their own.

`dblquad` runs with `epsabs=0` and `epsrel=1.49e-8` under whatever `quad_opts` gives.
scipy's default `epsabs=1.49e-8` stopped the outer integral on its first samples of a
small C, and `quad_opts` cannot be set from L1 or L2:

| Kernel, C(T, T) / amplitude through `dblquad` | amplitude 1 | amplitude 1e-9 | now |
|---|---|---|---|
| gamma 0.5, T 200, ridge and kink | 0.455744 | 1.481339 | 0.455744 at both |
| gamma 5, T 60, ridge and kink | 0.0054113 | 0.0228534 | 0.0054113 at both |

On OU and Gaussian kernels of width 0.5 at `gamma = 0.5`, a `dblquad` cell costs 8 to
25 ms at `T = 5`, as before.  At `T = 30` it costs 58 to 70 ms on OU against 44 to 45 ms
before, and 193 to 262 ms on Gaussian against 124 to 170 ms.  An
entry of C that cancels to 0 cannot meet a relative tolerance, so scipy warns on it
unless `quad_opts` gives an `epsabs`.  A ridge narrower than the samples can still be
missed at any amplitude, as the fallback warning says.

### Changed: half a C table only for a kernel that declares it

A table was filled from `C_ab(t1, t2) = C_ba(t2, t1)`, and `C(x2, x1)` from
`C(x1, x2)`, when κ² and σ² passed a numerical test at five time pairs.  No finite set
of reads can establish that: a one-sided kernel that ends below 0.37 of the span reads
0 on both sides at every lag read, and at a separation where the spatial factor is 0
the test compares 0 with 0.  The test is gone, with `_probe_c_transpose`,
`_c_half_grid`, the general table's swapped slice and the lazy table's filing at
`(x2, x1)`.  Half a table now takes `symmetric_in_time` on the kernel, which the
built-in even temporal kernels declare, and with `diag_C=False` a σ² from
`ConstantImpulse` or none.  Anything else gets every cell, twice the cells of a half
table: a `CustomKernel` or `GeneralKappa2` at `diag_C=False`, and every general table.
No value of a declared kernel moves.

### Fixed: wrong numbers at L1 and L2

| Case | Before | Now |
|---|---|---|
| `diag_C=True` on a user closed form with off-diagonal entries | 0.60289 on `qmc_scalar`, 0.12403 on `qmc_vectorized`, 0.12476 on `gauss_legendre`, a factor 4.9 | every read takes the diagonal |
| `Propagators.build` called directly on a C with off-diagonal entries | order 0 of `<phi_0 phi_1>` 0.0 against 0.0738, order 2 -0.885 against -0.582 | `ValueError`, as from `System.propagators` |
| `propagators.dt` with `t_min` other than 0 | `ceil(t_max / dt)` nodes: 1.5 to 10 percent off at `t_min = -10`, 20 times too many at `95..100` | `ceil((t_max - t_min) / dt)` nodes |
| `ConstantImpulse` not symmetric at L1 | `C_01 = 0.2050`, `C_10 = 0` against 0.1025 for both | `ValueError` at L1 as at L2 |
| A coupling of the wrong shape at L1 | entries outside the leading block dropped | `ValueError` |
| A negative kernel width at L1 | a kernel growing with `abs(dt)`, a factor 52 | `ValueError` naming the width |
| A point left out of `positions`, or misspelt, beside a located one | 30 percent off at order 0, 45 at order 2 | `ValueError` |
| `n_samples: 4096.9`, `seed: 1.5`, `t_max: true` in YAML | truncated, or read as 1 | refused at the key |

The positions check is narrower than a required key: an extra name beside a position
for every point, `{}`, all positions 0 with a point left out, a point sharing an R
group with a located one and a cache with no spatial table stay accepted, because
each gave the right number.  A callable coupling reads the positions, so a point left
out is still refused for it on a cache with no spatial table.  At L1 a 0-d coupling at `N = 1` and equal rates of any
length stay too.  `CustomImpulse` and a callable rate of the wrong length are not
checked.

Newly refused at L1, as at their YAML keys: `ConstantImpulse` of another `N` or not
square, a kernel width that is not positive, unequal rates of the wrong length,
`NonLocalVertex` of an order that is not an integer of at least 1.  Also newly refused:
an `ExponentialTemporal` width whose reciprocal overflows, below about 5.57e-309.  The
closed form returned NaN there, and a 40-node Gauss-Legendre rule gives 0.

Under a constant `gamma` the YAML parser now reads `n_grid_cache`, refusing a value
below 2, and requires `t_max_cache` and `t_min_cache` to be finite numbers, so `.inf`
is refused; it did not read them.  It does not compare the two times, and L1 checks none of the three under a
constant `gamma`.

`examples/demo1/c_closed_form.py` took `|sum(n1) - sum(n2)|` as the separation of two
vectors, 0 for `[1, 0]` and `[0, 1]`; it now takes the norm.  The demos use scalar
positions, so no demo value moves.

### Fixed: a wrong refusal of an R or a `t_max`

`system.linear.R_time_module` refused an R that does not vanish for `t1 < t2`.  The
integrators apply Θ, and at `gamma = 0.9`, `t_min = 0.4` such an R gives 1.5701585792e-01
at L1, the value of the clamped R.  The probe now reads the shape of that value only.
`ExplicitR` and the user guide state the one contract.  `System.propagators` compared
`t_max` with `t_max_cache` with no slack, and refused `10 * (1 + 5e-13)`, which R reads.
It now uses the table slack, and `_outside_table`, `_refuse_past_spline` and this check
share `_table_slack`.

### Fixed: `sft-wick validate` passed a joined point at an interacting order

F vertex, `<phi psi psi>`, order 1, `x = y = 0`, `z = 0.5`: `validate` exited 0 and `run`
exited 2.  The check read order 0 only, which has no diagram for three operators.  It
now walks the R assignments at the selected orders, only when two points of the grid
are apart and the observable has a response leg or a local vertex has two, and stops
at the first diagram it refuses.  A config with nothing to refuse pays for the whole
walk.  A connected target is decided after the
expansion, since the choice depends on the C pairings, so `validate` does not see it.

### Fixed: the closed form of an OU C lost its digits at small rates

`ou_exponential_phi` divides by `γ + a` a sum that cancels to `(γ + a) t²`, and
`_psi` returned 1 below 1e-12, dropping a term that the division then lifts.
`c_closed_form='auto'` took it with no check, while the quadrature beside it was
right.  At `γ = 0`, `(t1, t2) = (2, 1)`, against an 80-digit integral: 8.9e-5 off at
`σ_t = 1e12` and 5.0e-1 off at `1e13`.  At `γ = 0`, `σ_t = inf` it returned NaN.
Below `(γ + a) t = 1` it now sums positive terms (`_phi_at_small_rates`): the square
of the domain is `t²` times a divided difference of `ψ`, taken from its Taylor series
with no `0 / 0` at `γ = a`.  `_psi` keeps its term, and `_dexp` forms a difference of
two exponentials through `expm1` when the rates are close.  Against the defining
integral at 40 digits: within 1e-13 at each cell tried, on both sides of the change
of form.  On demo 2 no value moves.  Its `FF` channel at order 2 under
Gauss-Legendre at 16 nodes takes 3.69 s against 3.33 s, and the L2 sweep of
`config_FF.yaml` at `n_jobs = 1` 83.4 s against 111.8 s, medians of three runs.

### Fixed: a time-only table carried a C it does not hold

The refusal of a time-only C table read κ² alone, at one equal time, and compared
the two ends with each other.  It now compares κ² and σ² at the two ends with those
at the table's own position, `direction` of `precompute_C_table`, at 36 pairs of
times across the range, with or without `positions`.  `integrate_two_point_qmc`
went around it at order 0 with `diag_C=False`, returning `C(x, x)`, a factor 3.08 at
`r = 1.5`; it now takes its κ² ratio there too, and refuses what the ratio cannot
carry, off-diagonal entries and σ² that reads differently at the two ends (0.0
against 0.13233).  Its ratio is now taken against the table's position.  It reads
positions through `_resolve_group_x`, so vector positions reach a time-only table,
where they raised a bare numpy error.

### Fixed: `response_phase=False` printed two conventions

`order(n)` came without the phase and `to_latex` with it.  `DiagramTerm` now carries a
display field `response_phase` (not compared), read by `to_latex` and
`to_latex_evaluated`.  `evaluate_coupling` and the integrators always use the physical
phase, so `response_phase` changes no number.

### Changed: other refusals

- `apply_response_phase` raises `ValueError` on a product over a sum whose terms hold
  different numbers of R.  It used to skip the phase (-1) or apply the wrong one.
- The batched integrators and `integrate_two_point_qmc` raise `ValueError` on a total
  that is not finite, as `qmc_scalar` does.  They used to return it.
- A closed-form C that returns a complex array is refused, a zero imaginary part
  included, on both ways of attaching it.  One kept the real part and one the complex
  value.  `sft-wick validate` does not read a user closed form, so at L2 this passes
  `validate` and ends in the exception on `run` and `run --dry-run`, status 1.
- `System.expand`, `System.propagators`, `coupling_vectorized` and `ExplicitR.iso_R`
  raise `TypeError` on anything but a bool, `0` or `1`.  A word such as `'false'` read
  as true; `None`, `1.0`, `2` and `np.array(True)` also ran.  `iso_R=None` on
  `System.expand` stays, and means the system's own.
- `propagators.c_closed_form: 0` is refused, as by the schema.
- A `PropagatorCache` pickled by an earlier release fails on load, where the first read
  raised `AttributeError`.  A lazy, empty or time-only table raises `ValueError`.  A
  closed-form cache, whose class is gone, and a full spatial table, whose interpolator
  changed slots, raise `AttributeError` while unpickling.  A pickled cache does not
  cross this release; `cache_path` does, by rebuilding.
- `ConstantImpulse` refuses a `0 x 0` amplitude, and shows a refused amplitude at full
  precision: the array repr rounded `0.500000001` to `0.5` and printed a symmetric
  matrix as asymmetric.
- A file nested past the limit is refused before `compare-resources` starts a worker.
  A refused variant now reads `variant <name>: <rule text>`.

### Changed: words and numbers read in one way

- `--override` reads a value as the value of a key in a file, with the file's loader, so
  `017`, `0x10`, `1:30`, `1e3`, `Off`, `no`, `NONE` and `.inf` mean what they mean in a
  file.  That includes a comment: `run #3.csv` reads as `run`.  `|` and `>` read as an
  empty string, and `2026-11-01` as a date, which a typed key refuses.  `none` is now the
  string `none` on an ordinary key.  A value that reads as a list or mapping stays a
  string, and so does an empty value or one that is only a comment, which an ordinary key
  then refuses as before, so `system.vertices=$UNSET` deletes nothing.  `null` and `~`
  read as null; `---` and `...` are strings, as in a file.
- `SFT_WICK_QUIET_CACHE`, `SFT_WICK_PROGRESS`, `SFT_WICK_STRICT` and the
  `reuse_integrands` of a `compare-resources` variant read one word list, which also
  takes `yes`, `no`, 1 and 0 there.  `false`, `OFF` and ` no ` no longer silence the cache tip, and a
  word outside the list, such as `xyz`, reads as unset.
- `sft-wick schema` accepts what the parser reads: `dt: 1e-1`, numeric strings, flag words
  in any case.  The other way round, the parser now refuses `true` or `false` inside a
  coupling, `sigma2.amplitude`, `g0` or `g1` tensor, which it read as 1 or 0 while the
  schema refused it.  Numeric strings pass the schema only at keys that read them, and
  at an integer key only without a fractional part or with an exponent, so `'1e-1'`
  passes there and the parser refuses it.  `order`, `rank`, `orders`, component indices,
  `gl_chunk_size`, `qmc_coupling_chunk_size`, `propagator_cache_bytes` and
  `contraction_max_bytes` share their check with L1, which takes an integer only.  The
  schema gives them the type `integer`, which in JSON Schema includes `8.0`, so `8.0` at
  these keys passes the schema and `sft-wick validate` refuses it.
- The YAML integer reader takes `1000`, `1000.0`, `'1e3'` and `'1000'` as 1000, at the
  keys that use it, `n_grid_r`, `n_grid_cos` and `n_grid_x` among them.  A coupling reads
  numeric strings as `sigma2.amplitude`, `g0` and `g1` do, so `1e-3`, which PyYAML reads
  as a string, works in a coupling in a file and in `--override`.
- The resource report classes `qmc`, `qmc_scalar` and `nquad` as the scalar loop, which is
  what they run, and the run's summary line no longer prints `qmc_coupling_chunk_size`
  under them; the report still gives it as ignored.
- `--dry-run` counts both time orders of a general C table: 324 cells at
  `n_grid_t = 8`, `n_grid_x = 3`, now 576.  The build fills 702 cells there, so the
  estimate is still low.
- `--dry-run` takes its arguments from the run, so a config that fails with
  `contraction_max_bytes=1` fails the dry run.  It no longer builds the full table:
  22.5 s to 17.0 s on a 20 x 20 x 9 cubic table with a Gaussian kernel of width 0.3,
  and 1.8 s now at width 2.  The worker count matches `joblib` at
  `n_jobs = -3, -2, -1, 2`.
- Any of the six BLAS thread variables silences the thread tip.
- `c_closed_form_only=True` with the built-in closed form takes the batched read whatever
  `c_closed_form_vectorized` says: 31.7 s to 0.21 s on 80 diagrams at N = 2, order 3,
  `n_gauss = 10`.
  Totals do not change.

### Fixed: display and small numeric differences

- The TikZ backend drew parallel propagators on one line, so a one-loop diagram read as
  a tree.  Parallel edges now bend, with the end tangent of the matplotlib curve, and
  two self-loops at a node point apart.  `DiagramRenderer.PROP_STYLES` and the arrow end
  come from the shared style and layout.  Matplotlib output is unchanged on 321 diagrams.
- `LaTeXFormatter` renames propagators on the tree instead of the text, so a coupling
  named `KC` no longer becomes `KG` under `C -> G`, and swapping two names works.  A
  one-index propagator shows its index.  `LaTeXFormatter.format` takes an `Expr` and
  raises `TypeError` on anything else: a `DiagramTerm` or a `PerturbativeResult`, which
  the text replacement also renamed, would keep the default names with no error.
- `FeynmanDiagram.from_pairing` no longer stores `full_label` on external nodes;
  `render_labels.py` spells it.  The spelled text differs from the stored text only for
  an empty spatial argument, `\phi_{a}` against `\phi_{a}()`.
- The reality floor is one rule: `1e-292(1+i)` passed the scalar path and raised on the
  batched one.  `_isotropic_C` judges each point by its own size, so the verdict does
  not depend on the batch.
- `integrate_two_point_qmc` reported a NaN error at `n_samples` 2 to 7, and a NaN that
  spread through a sum of diagrams.  All three QMC paths share one error rule.

### Changed: a κ² that is not separable finds its rules on the workers

Under `'auto'` a full table asks one rule per pair of positions.  That stage ran in
the parent whatever `n_jobs`.  It now goes through `n_jobs`, and the warnings stay in
the parent.  It does not grow with `n_grid_t`, so its share of a serial build falls as
the table grows.

A general table on a smooth kernel, which the old route built with `dblquad`, now costs
more to build.  Serial, N = 2, a Gaussian kernel in `dt` and `dx`, rule `n = 20`, `t_max
= 2`; an off-grid cell agrees with the old value to about 1e-15:

| `(n_grid_t, n_grid_x)` | build | rule stage | share | `dblquad` before | ratio |
|---|---|---|---|---|---|
| (5, 3) | 8.9 s | 6.7 s | 75 % | 1.1 s | 8.2 |
| (5, 5) | 24.3 s | 18.4 s | 76 % | 2.6 s | 9.5 |
| (10, 5) | 40.4 s | 18.4 s | 45 % | 9.0 s | 4.5 |
| (20, 5) | 108 s | 18.2 s | 17 % | 34.8 s | 3.1 |
| (40, 3) | 143 s | 6.7 s | 4.7 % | 50.8 s | 2.8 |

The ratio reached 23 at N = 1 on (5, 5): 23.8 s against 1.0 s.  With `diag_C=False` on
(10, 3) at `t_max = 8` it is 1.5: 12.5 s against 8.4 s.

### Changed: one implementation per rule

No value changes here: 97 integrator cases, 93 table cases and 2800 symbolic cases are
identical to the start of this work, numerical terms compared with their coupling
points removed.

`DiagramIntegrand.observable_phase` and `.expected_phase` are removed, as is the private
`_ClosedFormPropagatorCache`: nothing read the two properties, and
`DiagramTerm.observable_phase_factor` is the phase in use.  A closed form is
`PropagatorCache(model, c_value_fn=c_fn)`.  `generate_valid_pairings` and
`evaluate_pairing` are no longer importable from `sft_wick.perturbation`; they stay in
`sft_wick.wick`.  `integrate_moment`, `Expr`, `Product`, `Rational`, `Sum` and `Symbol`
are no longer importable from `sft_wick.workflow.expansion`.

**Breaking (`compute_moment_numerical`).**  The unread `coupling_values` and
`fixed_indices` are gone.  A fourth positional argument raises `TypeError`.  Terms now
hold coupling symbols with their legs, so a callable coupling evaluates on numerical
terms, where it raised `ValueError`, and agrees with the 3! routings by hand to 1e-10.
`wick_contract_grouped` is removed and `wick_contract` takes `keep`.  Both engines share
one sequence and term builder, and one pairing loop.  One causal-domain statement serves QMC, Gauss-Legendre,
`nquad` and `integration_bounds`.  The constants and helpers of one rule have one home:
`i_power`, `canonical_edge`, `_msr_factor`, `vertex_type_label`, `is_literal_component`,
`mixed_weights`, `C_METHODS`, `HOMOGENEITIES`, `TARGETS`, `CONTRACTION_BACKENDS`,
`DEFAULT_N_GRID_T`, `DEFAULT_C_N_GAUSS`, `LAZY_KEY_DIGITS`.

### Fixed: under `c_method='auto'` a C no node count converged on came from an unchecked `dblquad`

When `select_gl_node_count` found no converged count, `auto` took `dblquad`, whose
first samples miss a ridge or a boundary layer far narrower than the interval.
It reports a small error and the value can be off by any factor, with
`GaussianTemporal` on default settings, against a 1-D `mpmath` integral at
`(gamma, sigma_t, T)`:

| Case | `dblquad` value or error | Now |
|---|---|---|
| `(0.5, 0.02, 20)` | 3.9e-1 | 6.7e-12 |
| `(1, 0.01, 5)` | 1.6e-1 | 9.4e-12 |
| `(0.5, 0.002, 20)` | 5.8e-10, true 5.0e-3 | 5.8e-12 |
| `(300, 0.3, 30)` | 2.5e-11, true 1.1e-5 | 1.0e-12 |
| `(3000, 0.3, 30)` | 1.0e-87, true 1.1e-7 | 4.6e-13 |
| `(0.5, 1e-5, 20)` | not run | 1.2e-11 |

The fallback is now Gauss-Legendre on panels halved towards the ends of each
interval, 8 nodes a panel, at the depth `select_graded_levels` finds converged to
the 1e-8 that `select_gl_node_count` asks for.  `c_source` reports
`quadrature:graded(levels=12)`.  A one-component build at `n_grid_t = 4`, `t_max = 32`,
`gamma = 0.3`, width 0.01, with its first lookup takes 124 s at 12 levels.
A kernel on which no depth converges gets `dblquad` with a `UserWarning`.  A route
that resolves to a node count does not change.  An explicit `c_method='dblquad'`
remains unchecked.

### Fixed: under `c_method='auto'` a user callable kernel and a C before any table came from an unchecked `dblquad`

`System.propagators` sent any callable (`CustomKernel`, `GeneralKappa2`,
`CustomImpulse`, `ExplicitR`, a callable `gamma`) straight to `dblquad`, and a
direct `C_value` before any table, with a legacy `precompute_C_table`, took it as
well.  Both now run the steps a built-in kernel runs: Gauss-Legendre with a
converged count, the graded rule, `dblquad` with a `UserWarning`.  A direct
evaluation runs them for its positions and for the first power of 2 above `t -
t_min`, as for a table reaching that horizon, and every later cell at those
positions below that horizon shares the rule.  Where they find no rule, the cell's
own steps decide.  So its rule does not depend on the calls before it, and
`c_method_resolved` is `'auto'` until something resolves it.  An explicit
`n_gauss` with no method takes Gauss-Legendre at that count.

Relative error of `C_aa` on a `CustomKernel` equal to a Gaussian of width
`sigma_t`, `gamma = (0.1, 0.3)`, `[2, 32]`, table at the deep cell, against an
`erf`-reduced `mpmath` integral, and of a direct cell before any table at
`gamma = 0.1`, `t = 30`:

| `sigma_t` | `dblquad` table | Table now | Rule now | Direct before | Direct now |
|---|---|---|---|---|---|
| 2 | 3.6e-14 | 6.0e-10 | `n=20` | 1.3e-15 | 6.0e-10 |
| 0.3 | 2.4e-8 | 5.4e-12 | `n=68` | 2.3e-12 | 1.7e-12 |
| 0.05 | 3.3e-1 | 1.8e-11 | graded | 3.3e-1 | 1.2e-11 |
| 0.01 | 1.0 | 1.3e-11 | graded | 9.7e-1 | 2.6e-12 |

The first cell at a position costs 0.8 s on a smooth kernel, 9.1 s at width 0.3
and 113 s on the two narrow rows, and a later cell 7 ms, 80 ms and 0.65 s, where
`dblquad` took 70 to 390 ms and on the narrow rows 2 to 3 s and was wrong.  A cell
on which no rule converges pays for all the steps each time; a table
or an explicit `c_method` avoids that.

The graded rule's test was blind inside the interval: its depths differ only at
the ends, so a kink or a step at a lag of 2 passed at 12 levels, off by 1.5e-5 and
4.6e-3 with no warning, and a ridge of width 0.01 at a lag of 3 by 1.2e-1.  This
reached user kernels only.  The depth found is now compared with the same panels at
12 nodes.  All three end in `dblquad` with the warning.  There the
triangle is right to 9.3e-10 and the step to 1.9e-6, and the ridge is still off by
0.85, as it was before.

A non-separable kernel now has a rule per pair of positions, on `translation`,
`rotation` and `general` tables, full or lazy, and a pair where no rule converges
takes `dblquad` there alone, with a warning that names it.  With a time width of 0.3
at the origin and 0.01 at `x = 1`, `dblquad` was off by 0.97 and is now right to
2.6e-12.  One function decides the rule, from the pair and the horizon: full tables
ask it per pair, lazy tables per position, direct cells per cell.  A direct cell
inside a table takes the rule the table took there.  The graded rule is not finer than a
count: 8 nodes on a quarter of the interval missed an oscillation that 30 nodes on
all of it resolved, 3.3e-5 off at those positions with no warning.  `c_source` names
the deepest rule of a table, or `quadrature:per_position` when only some pairs take
`dblquad`.  On a 5 by 5 `general` grid, with a Paciorek-Schervish
kernel of time width 0.3 that narrows at the interior node `x = 0.5`, against a 1-D
integral at `T = 5`:

| Width at the node | Rule of the grid's ends | `dblquad` | Now | Build, 4 workers |
|---|---|---|---|---|
| 0.03 | 1.4e-3 at `n=30` | 1.7e-13 | 1.4e-9 at `n=68` | 17 s |
| 0.02 | 5.9e-3 at `n=30` | 5.6e-11 | 7.7e-12 at 12 levels | 122 s |
| 0.01 | 3.6e-2 at `n=30` | 2.2e-1 | 7.8e-12 at 12 levels | 122 s |
| 0.003 | 2.1e-1 at `n=30` | 8.6e-1 | 2.2e-12 at 12 levels | 122 s |

`c_source` names the
rule at the origin on a lazy table, or `quadrature:per_position` when the kernel
raises on the scalar probe, vanishes at the origin, or no rule converges there.  An
explicit per-build method keeps its rule at every position.  `sft-wick run --dry-run`
ended in the kernel's traceback on a kernel that raises on the scalar probe, where
`run` succeeded.  It now prints an estimate timed at a position of the sweep, and
says, for more than one separation or grid position, that working out the rule is
not in that estimate.

A separable kernel with a white-noise `sigma2`, `ConstantImpulse` included, also
searches its rule once per pair of a full table and once per position of a lazy
one, because the `sigma2` term does not scale with the spatial factor.  No value
moves.  A rotation table at `n_grid_t = 11`, `n_grid_cos = 5`, `t_max = 5`, N = 2,
rates (0.4, 0.7), a `GaussianTemporal` kernel and three Legendre terms, with
`diag_C=False`, builds in 8.6 s with `ConstantImpulse(0.05 I)` and in 4.7 s with no
`sigma2`; that includes the integral of `sigma2` itself.

`PropagatorCache.resolve_c_method` gained `positions`.  A subclass override with the
old `(self, t_max)` signature keeps working: it is called with `t_max` alone, and it
decides the rule of a lazy table too.  `PropagatorCache.graded_levels_resolved` is the
largest depth of the last table, as `n_gauss_resolved` is the largest count.
`resolve_c_method` no longer changes what those report.  `_C_value_direct(...,
method='graded')` with no depth takes the depth in force, else the first of
`_GRADED_LEVELS`.

The check judges each entry of C at each pair of positions on its own scale.  With
one scale a weak component passed against a floor of 1e-6 of a stronger one: at
`gamma = (0.05, 3e5)`, `sigma_t = 2`, `T = 30`, `C_11` came back 0 under `n=20`, true
1.1e-11, with no warning, and it is now 1.6e-9 off under the graded rule.  When R mixes components an entry is
a sum that can cancel, so two rules may differ there by rounding: a difference up to
32 eps of the sizes of that entry's own terms does not count, which costs one more
rule at each probe cell on such a system.  When that allowance is above the tolerance
times an entry, a warning says once that the entry cannot be checked.  An entry
the kernel does not feed is 0 and needs no rule: a zero kernel resolves its rule
in 0.8 s at a horizon of 5, with no warning.  At `gamma = (1e-3, 3e6)`, `sigma_t = 1` no depth converges, so that
case ends in the warning, where it was a silent 0.

`c_n_gauss` of 74 or more under `auto` is tested once, where it passed the cap of 110
untested and a smooth kernel went to the graded rule.

The check compares two rules.  A feature narrower than the node spacing that both
miss passes it: a pulse in absolute time of width 0.01 under a background a thousand
times weaker gave `n=20`, 3.9e-6 off with no warning, the same value `dblquad` gave.
That covers the diagonal too: a spike of width 3e-8 on a kernel of unit width over
`T = 5` gives `n=20`, 0.59 off with no warning, the value `dblquad` gives, where
widths down to 1e-7 go to the graded rule.  White noise is the spelling for such a
kernel.

The demo tables under `auto` were `dblquad` and are now `n=20`.  They cost less to
build: 16.0 s to 6.7 s, 5.0 s to 4.3 s, 16.7 s to 5.2 s, 7.9 s to 5.4 s, 4.9 s to
3.8 s and 18.9 s to 12.0 s on demo 8 rate, oscillator, Matern, damped cosine, white
and demo 7 damped cosine.  Values at 36 off-grid cells move by at most 3.8e-14, except
Matern at 3.2e-8, where `dblquad` was the one off: against the exact C at three cells
it had 1.9e-9 and `n=20` has 5.6e-16.  The only table in a demo ledger built on these
routes is the demo 7 `CustomKernel` row, which moves by 1.3e-15.

### Fixed: an undeclared closed form on a model with `sigma2` could lose its split

A kink rule of this release sent such a closed form to the probe, which needed the
jump at two of three times in the last two thirds of the range.  With noise
`exp(-(s/tau)^2)` the cusp has died out by then, so Gauss-Legendre went unsplit
with no signal.  On order-2 `F`, `g = 1`, against a hand-written Picard iteration,
at 24 nodes: 5.1e-4 at `T = 4`, `tau = 0.6` and 1.8e-2 at `T = 30`, `tau = 3`,
where it had been 8.4e-15 and 2.8e-8.  They are back at 8.2e-15 and 2.8e-8.  A model with `sigma2` now splits an undeclared closed form whatever the
probe would say, and the probe takes five times, adding 0.05 and 0.15 of the range,
which gives the same two values with no `sigma2` on the model.  A pulse narrower than
about 1e-3 of the range that lies between the times is still missed when the model
does not carry it.

### Fixed: the cusp probe of a κ² read two fixed times

`_probe_diagonal_cusp` read `t_min + 0.7` and `t_min + 2.3` whatever the range, so a
kernel off at both, such as noise over by `t = 0.6`, reported no cusp and took no
split.  It now reads five times across the range the integrator or quadrature asks
about, at steps scaled to it.  On a range of 3 it keeps the two old times.

### Fixed: a closed form narrow across the diagonal was not split

The probe tests for a jump in a derivative, so a rounded cusp and a narrow smooth
peak passed as smooth and Gauss-Legendre stayed on one piece.  On order-2 `F` at
`T = 2`, 24 nodes, against the same iteration, unsplit against split:

| C | Unsplit | Split |
|---|---|---|
| rounded `exp(-sqrt(dt^2 + eps^2))`, `eps = 1e-4` | 8.2e-4 | 5.4e-8 |
| rounded, `eps = 1e-7` | 8.3e-4 | 1.6e-13 |
| Gaussian in `dt`, width 0.1 | 1.5e-4 | 3.0e-9 |
| Gaussian in `dt`, width 0.01 | 8.2e-2 | 8.1e-4 |

The probe now also compares a 32-node rule on the range with the same count on each
side of a cut at `t1 = t2`, and asks for a split when the integral moves by more than
1e-10.  The time of that line is a node of the rule on the range, because an unsplit
tensor rule has nodes on `t1 = t2`, which read a feature there at full height however
narrow it is.  A cusp, a ridge or a value at exactly `t1 = t2` is therefore split at any
width.  Order 2 of `F` at `T = 3` with a cusp or a ridge of width 1e-7 was off by
2.9e-1, 1.5e-1, 7.7e-2 and 3.9e-2 at 8, 16, 32 and 64 nodes with no error, and is off
by about 1e-15 at 16.  A C with nothing
on the diagonal that varies too fast for the 32-node rule is split as well, and there
the split can cost accuracy: `1 / ((1 + t1) (1 + t2))` from `T = 30` and
`0.1 exp(-(t1 + t2)) cosh((t1 - t2) / 2) + 0.01` from `T = 155` to `579`, where the
same transient at `T` from 5 to 154 and from 580 is left alone.  At `T = 30` at 16 nodes that
transient is 5.4e-12 off unsplit and was 2.55e-7 off split.

### Fixed: an undeclared kink of a closed-form C was integrated without a split

A user closed-form C with a `|t1 - t2|` cusp and no `has_diagonal_kink` went
through Gauss-Legendre and `nquad` unsplit, with no signal.  On order-2 `F` with
an OU C under white noise, against the exact Itô hierarchy, Gauss-Legendre at 24
nodes was off by 1.8e-4 to 9.5e-3 relative over `gamma` from 0.01 to 1 and
`t_final` from 0.05 to 30, and `nquad` by 1.3e-7.  The split gives 1e-15 to 6e-11.
The undeclared callable is now probed at the times and points of the diagram for
a jump in its first, second or third derivative across `t1 = t2`, under the
scalar or the batched contract, entry by entry.  The third derivative is what a
hand-written C of an exponential kernel carries, as the closed forms of demos
1-3 do; those were split only through the next fix's defect.  With the two fixes
above the probe adds 780 calls on a smooth C and 300 on a kinked one (2 and 1
batched) per pair of points and
per `t_final`, and per grid point of a sweep with `n_jobs > 1`, where each task has
its own copy of the cache.  `has_diagonal_kink` skips it.  At order 2 with 16 `y`, 2
`t_final` and 4 nodes, on a scalar C of 200 us a read, 7168 reads became 26968
serially and 35968 on 4 workers, and 1.52 s became 5.56 s serially.  It misses a cusp below
about 1e-6 of the size of C, which `has_diagonal_kink = True` covers.  Where it misses a third
derivative of a hand-written exponential kernel, with `sigma_t` far above the
range, the value is off by at most 4e-9 at 12 nodes and 2.7e-10 at 24.

### Fixed: a closed-form C was judged by the κ² and `has_diagonal_kink = False` was ignored

`_kappa2_kinks_c` read κ² on a closed-form cache, which never uses it, although
its docstring said such a cache answers False.  A smooth closed form on a cusped
κ² was split, and a callable setting `has_diagonal_kink = False` was split too,
and the split costs wall clock and changes no value beyond the node-count error.
A declaration now decides, `False` included, and a closed form is never judged by
the κ² it replaces.  The `sigma2` of the model still counts when nothing is
declared (see above).

### Fixed: a lower cut carried past a swept external left a kink in the domain

`integrate_over` with a swept external `s`, a variable `v` cut at a pinned time
`t*` and a later `w` with `v <= w` and `s <= w` leaves `max(s, t*)` in `w`'s lower
limit, which `_kink_constant_cuts` missed because `s` is not below `v`.  On a
three-point `F` observable at order 3 with one external swept and two pinned,
Gauss-Legendre was off by 1.5e-5 relative at 10 nodes, 1.8e-6 at 20 and 1.6e-7 at
28, against a three-time Itô hierarchy; it is now 1.5e-14 at 10 nodes.  The other
kink rules hold on the same observable: pinned, one swept and two swept externals
reach 1e-14 at 10 nodes.

### Fixed: `PerturbativeResult.total` applied the response phase twice

`compute_moment` applied `(-i)^n_R` to each order and again to their sum when
two or more orders were nonzero, so a term in `total` carried `(-i)^(2 n_R)`.
On a two-component `F` vertex at order 2 with seeded tables the sum of the
orders is -74.30 and `total` gave +75.04; on a linear vertex at order 1 it is
0.849+0.488i and `total` gave 1.338.  `total` is now the sum of the order terms.
`order_terms`, `DiagramTerm` and every integrated value read neither path and
do not change.  The expansion cache held the old `total` in `raw_result`, and
the cache schema bump of this release already rejects those entries.

### Fixed: `diagonal_propagators` kept a sum whose index one term eliminated

Under `diag_C`, `diag_R` or their isotropic forms, a delta constraint removes a
summation index from one additive term.  The expression walker dropped a sum
only when every term had lost the index, so in `sum_i (G_i C_ai + H_ii C_bb)`
the first term came back multiplied by N: 9.65967 against 5.42918 at N = 3.
`order(n)`, `order_terms` and `total` were wrong from order 1 on three and four
points, with `F` on `<phi phi phi>` and `<phi phi psi>` and with a quartic vertex
on `<phi phi phi phi>`, by up to 1.8 of the larger value on random tables at N = 2
and 3.  They were wrong from order 2 on `<phi phi>` with `F` and on `<phi>` with
`F` and `L`: -18.8346 against -8.89954 (ratio 2.12) at N = 2, ratio 2.99 at N = 3
and 1.94 under `iso_C`, `iso_R` at N = 2.  Order 1 on one and two points was right.
`DiagramTerm`s, `to_latex` of a term and the integrators were right.  It also
dropped a sum whose index no term ever used, so `sum_i C_bb` lost its factor N.
Sums are now held per additive term and only an eliminated index removes one;
`DiagramTerm.apply_diagonal` and the walker share `diagonal_index_sub`, and
`simplify.strip_isotropic` is the one statement of what an isotropy flag does to
a propagator.  `apply_diagonal` now applies it also to a term with no index to
substitute, so on a diagram with an absorbed R the printed text shows `R` where it
showed `R_{aa}`; no value changes.  `diagonal_propagators` raises `ValueError`
when a summed index or integrated point is bound twice in one product, or a sum
rebinds a name inside it, since the terms hold their sums at one level.
`compute_moment` never builds that shape.  It also expands a product of sums into
additive terms, and raises `ValueError` past 2**20 terms: two sums of 1024 terms take
27.7 s and 1.4 GB where they took 0.02 s, and 1025 are refused.  The largest product
`compute_moment` built in the cases measured had 99 terms (orders up to 4, N = 2), at
3 to 12 percent more time.

### Fixed: a callable coupling lost its positions on a cache with no spatial table

`qmc`, `qmc_vectorized`, `gauss_legendre` and `nquad` took a leg's position from
its direction group only when the cache had a spatial table.  With no table,
such as `spectral_cache` or a time-only table, every leg sat at the ambient
`direction`, 0 by default, whatever `positions` said, while
`integrate_two_point_qmc` read `positions`.  One resolver now gives a leg its
position and time on every evaluator: the position of its direction group,
else the ambient `direction`.  C still reads a table with no spatial structure
by time, and the warning of `Expansion.sweep` says so.

Measured on the order `F kappa3` term of `<phi(x) phi(y)>`, a diagram with no C,
with `kappa3 = (1 + n_1) exp(-t_1 - t_2 - t_3)`, `F = 0.5`, `gamma = 1`, `x = y = 0.7`,
`t_final = 1.5`: `0.028551` before, `0.048536` now, 41 % low.  The reference is the
closed form `2 F kappa A(T) I(T)`, with `A(t) = t exp(-t)` and `I` a 1-D integral,
confirmed by integrating the model with `eta = exp(-t) xi` as an ODE in `xi`
(relative difference 1.4e-13).  At `x = 3e6, y = -2e6` with
`g = 1 + n_1 + n_2 + n_3`, the old value was `0.028` and the reference `42826`.

The same resolver ends two crashes on hand-built L0 terms: `qmc_vectorized` and
`gauss_legendre` raised `TypeError` on a vector `direction`, and these two and
`integrate_two_point_qmc` raised `KeyError` on a leg with no propagator.  A leg
that is no external point and no integration variable now raises `ValueError`
on every evaluator; the batched ones used to put it at `t_final`.  Two points of
one direction group at two positions are refused for a callable coupling on any
cache, as they are on a spatial table.

**Breaking (`DiagramIntegrand.dynamic_coupling_array`).**  The signature went from
`(times, directions, default_position=0.0)` to `(times, leg_positions)`, keyed by
leg rather than by direction variable.  A call with two positional arguments in the
old form raises `KeyError`, and one with `default_position` or `directions=` raises
`TypeError`.

### Fixed: a spectral C was kinked on its diagonal and the time split did not know

`SpectralPropagatorCache` computes C from the density, with `|t1 - t2|` in it, so
C has a first-derivative jump of `-2D` on `t1 = t2`.  The cache did not declare
it, so `gauss_legendre` and `nquad` did not cut the time domain there and
`gauss_legendre` converged at `n^-2` with no error.  The cache now declares
`has_diagonal_kink` (true when `D > 0`), and `_c_has_diagonal_kink` reads it, so
every integrator takes the answer from one predicate.

Measured on order 2 of `<phi_0 phi_1>` with a non-symmetric `F`, rates
`(0.4, 1.1, 3.0)`, `D = 0.35`, `t_min = 0.5`, against a hand-written iteration of
the equation of motion that shares no code with the integrators.  At
`t_final = 3.5`, `gauss_legendre` went from 7.4e-3 to 5.9e-7 at `n = 8`, from
1.3e-4 to 2.4e-14 at `n = 64`, and `nquad` from 8.1e-8 to 2.4e-14.  At
`t_final = 60`, `n = 48` went from 5.3e-3 to 8.3e-11.  Below the rule's resolution
of `exp(-h t)` the split does not help: at `t_final = 60`, `n = 8` is off by 0.39
unsplit and 0.56 split.  QMC never split and does not change, and order 0 has no
time domain.

### Fixed: `c_method='auto'` built its tables with `c_n_gauss` nodes, not the converged count

`select_gl_node_count` found the count, `c_source` reported it, and the table
builders then overrode it with `c_n_gauss`, on every path (translation, rotation,
general, lazy or on a grid).  With `GaussianTemporal(0.05, 0.3)`,
`DiagonalA([0.1, 0.3])`, `t_min = 2`, `t_max = 32`, `c_source` said `n=68` and
the cells used 20: `C_aa` at `(32, 32)` was off by 2.1e-3 and 5.7e-3 against an
independent one-dimensional integral, and the order-0 correlator at `r = 0.7` by
8.8e-4 and 3.8e-3 against `n = 200`.  Those errors are now 1e-11 or below where a
count converges, and one count serves the table, `c_source`, `n_gauss_resolved`, later direct
evaluations and the `--dry-run` timing.  Under `'auto'`, `c_n_gauss` is only
where the refinement starts; an explicit `c_method='gauss_legendre'` keeps
`c_n_gauss` with no convergence check.  The right count and the rule search of 23
cells cost time: a two-component table with `sigma_t = 0.3`, `t_max = 32`,
`n_grid_t = 8` at `n=68` went from 3.1 s to 12.6 s.

A table moves only where the converged count exceeds `c_n_gauss`.  No route the
demos ship moves: demos 1-6 use closed forms or an explicit rule, and the `'auto'`
tables of demos 7 and 8 resolve to 20.  Forcing quadrature on the demo 1 kernel
moves the table by up to 2.4e-2 at `t_max = 100` (45 nodes) and the demo 2 kernels
by up to 4.3e-4 at `t_max = 50` (30 nodes).  The cache schema goes to 4, so a
persisted table built with the wrong count is rebuilt.

### Fixed: `collect_topology=False` returned 0 at interacting orders

**Breaking (`System.expand`, `expand.collect_topology` and
`PerturbativeResult.diagram_terms`).**  The operator path builds the expression
and no `DiagramTerm` above order 0, and L1, L2 and `integrate_diagrams` took
the empty list for a computed zero.  An order 2 of 0.000257579 came out as 0
with no warning.  `System.expand` now raises `ValueError` at an order above 0
with `collect_topology=False`, and `validate`, `run --dry-run` and `run` report
it as one line at `expand.collect_topology`.  `diagram_terms` raises
`ValueError` at an order that has an expression and no diagrams, so an order
that is 0 still gives an empty list.  `diagram_terms_by_order` has no key for such an
order, so that reading it raises `KeyError` rather than handing `integrate_diagrams`
an empty list.  Order 0 is unchanged, and `compute_moment` at L0 keeps the flag and
still gives the expression, which has no numerical evaluation above order 0.

### Removed: the `ito` switch

**Breaking (every layer).**  `ito=` is gone from `compute_moment`,
`compute_moment_numerical`, `System.expand`, `wick_contract`,
`wick_contract_spatial`, `evaluate_pairing`, `contract_pair` and
`generate_valid_pairings`, and `expand.ito` from the YAML, where it is now an
unknown key.  Passing it raises `TypeError`, `ito=True`
included.  `--print-resolved` no longer prints `expand.ito: true`; for the
eleven configurations shipped with the package that line is the only change in
the resolved configuration.  The engine is Ito: `ito=False` changed the expression only, and the
numerics applied Theta(0)=0 whatever it said and refused where the value
depended on it.  A Stratonovich SDE with multiplicative noise is reached through
`MultiplicativeImpulse(interpretation='stratonovich')`, which is unchanged.

The arguments after `order` in `compute_moment` and `compute_moment_numerical`, and
after `operators` in `wick_contract`, are now keyword-only, so a positional `ito`
raises instead of being read as the next flag.

A `DiagramTerm` built by hand can still hold an equal-point R or a loop of R.
The refusals of `ito=False` are gone with it.  An equal-point R gives 0, unless a C
propagator of the term lies past the table and is read first, which raises the usual
`ValueError`: on every integrator on a translation table, and on `qmc_vectorized`, `qmc`
and `gauss_legendre` on a time-only table when the term has an integration variable.  A
loop raises `ValueError` naming its points.

`FeynmanDiagram.from_pairing` raises `ValueError` on a pair that vanishes, which
it used to leave out of the diagram.

The flag left the expansion cache key.  The schema bump of this release rejects
every earlier entry, so existing expansions are recomputed.

### Fixed: a closed R loop raised `KeyError` or returned 0, by integrator

**Breaking (a `DiagramTerm` with a closed R loop).**  `analyze_spatial`, which
every integrator passes through, now raises `ValueError` naming the points of
the loop.  A loop of two or three R over integrated points raised a bare
`KeyError` from `integrate_moment`, from each `integrate_moment_*` method and
from `integrate_two_point_qmc`.  With a free variable beside the loop,
`qmc_vectorized` and `gauss_legendre` returned 0.0 and the others raised
`KeyError`.  A loop through an external point returned 0.0 everywhere and is
refused too.  The equal-time case is covered by the same
check: an L0 `Vertex(fields=[phi, psi], local=False, equal_time=True)` with `F`
gives four terms for `<phi>` at order 2, and one, `R(y_1, y_0) R(y_0, y_2)` with
`y_2` at the time of `y_1`, raised `KeyError: 'y_1'` on all four integrators.  It
closes in time and not in labels, so the Wick engines keep it, and integrating
that order now raises `ValueError` where it raised `KeyError`.  The Wick engines
drop every other loop.  An equal-point R keeps its value of 0.

### Changed: one vanishing rule and one R enumerator in the Wick engines

`wick.r_vanishes` is now the only statement of the Ito and causal rules, and
both engines call it: an equal-point R is the shortest closed loop.
`wick.r_classes` enumerates the R assignments between labels, lazily, and is
the only enumeration of them.  `wick.wick_contract_spatial` crosses each class
with its C pairings between labels, and `generate_valid_pairings` lifts it to
operator assignments and crosses those with C pairings between operators.

**Breaking (`generate_valid_pairings`).**  `operators` is required, the memo is
gone, and the pairings come class by class.  Its `ito` defaulted to `False`, so
a call that passed `operators` and no flag kept equal-point R and causal loops.
That call now prunes them with no error: one phi and one psi at `x` went from
one pairing to none.  `wick_contract` and `compute_moment` put the pairings back
in operator order, so their output is unchanged.

The first pairing of ten phi and ten psi takes 0.08 ms and no table.  With nine
phi and nine psi at 18 distinct points, 362880 pairings, `wick_contract` went from
6.3 s and 1254 MB to 2.8 s and 459 MB, since a repeated operator pair now shares one
propagator.  An enumeration in which nothing survives, one phi and one psi at each
of ten points, went from 6.4 s and 1.9 GB to 4.0 s and 72 MB.

**Breaking (L0 API).**  The API reference documents `sft_wick.wick`, and
`wick_contract_spatial` now returns `(propagators, pairing)` per topology and
no longer takes `vertex_points`.  Its multiplicity went unread and differed
from the operator-level count; the component routings of a topology give that
count.  `SpatialTopology` is gone, with the private `_compute_multiplicity`,
`_has_r_cycle` and module cache `_r_cycle_cache`.  `wick_contract_grouped` is
removed: it had no caller and was not exported.  `wick.contracted_pairings` gives
each pairing of `generate_valid_pairings` in operator order with its propagators,
and `wick.r_class_pairings` gives one complete pairing per R class.

### Fixed: later vector shapes passed validation, and high-point moments slowed it

The position check read only the first shape on each sweep axis, so
`x: [[0, 0], [0, 0, 0]]` with `y: [[0.1, 0.2]]` passed `validate` and
ended in a traceback in `run` and `run --dry-run`.  Every shape is now checked
against the shapes on other axes, with a single-line error and status 2 at
the incompatible entry.  Alternatives on one axis can still have different
lengths when they each broadcast with the other axes, such as a length-one
vector at the other point.

Validation also enumerated every order-zero Gaussian pairing of the
observable, 13!! of them at 14 points, to find the points a response
propagator joins.  The checks read only the R of a diagram, so it now takes
one pairing per R assignment from the Wick engine, repeated points included,
and the engine's rule decides which assignments vanish.  Its cost is the
number of assignments: 13 at 1 response and 13 physical legs, 40320 at 8 and
8.  When several pairs sit apart, the error names the first in sorted order,
where it followed `PYTHONHASHSEED`, in `validate`, `run` and
`Expansion.evaluate`.  The expansion performed by a run is unchanged.

### Fixed: aliases that expand past what a walk finishes hung `run` and `--print-resolved`

**Breaking (a file whose aliases add more than 10^6 values, or nested past
100 levels).**  A list that holds the one before it twice doubles at each
level, and YAML keeps one object, so 748 bytes at a key nothing reads
(`output[0].x`, or `system.linear.t_max_cache` under a constant `gamma`)
passed `validate`.  `validate --print-resolved` then took 20 s and 1.6 GB at
20 levels, doubling with each level, and `run` did the same in `repr` of the
cache key.  Many vertices sharing one tensor, each under the limit on one
tensor, expanded the same way, and a position repeated through aliases was
read once per alias.  The loader now measures the file as a tree, entering
each container once, and refuses what aliases add past 10^6 values with
`invalid_value` and the dotted path where the value spreads out.  A value
written out, and pairs merged with `<<`, do not count.  `compare-resources`
refuses the same files.  A file nested past 100 levels, written out or
through aliases, is refused by the config loader: a chain of 500 aliases
passed `validate` and `--print-resolved` ended in a `RecursionError`.

### Fixed: a chain of YAML merges hung `sft-wick validate`

**Breaking (a file whose merges add more than 10^6 pairs).**  PyYAML copies
a merged mapping's pairs into each mapping it lands in, so a file whose
mappings each merge the one before twice took 5.0 s at 22 links from 1.2
kB, and the time doubled with each link, inside `yaml.safe_load` and
before any package code ran.  The parser now counts what each merge will add
before PyYAML does it, and a file past 10^6 added pairs is refused with
`invalid_value` and the dotted path of the `<<` key, whatever its length: a
chain of 400 links stops at 0.19 s, at the link that passes the limit.
The count follows PyYAML's walk, so it matches the lists it builds, also for
a merge that reaches a mapping still being merged.  `validate`, `run` and
`compare-resources` read files the same way.

### Fixed: a module named in a config changed `PYTHONPATH` for later processes

Loading a module named in a config put its folder at the front of
`PYTHONPATH` for the rest of the process.  Every process started afterwards
read that folder before the standard library, so a `colorsys.py` beside the
config replaced the real module there, and `test_catalog_is_current` failed
when it ran after a test that loads such a file.  Workers did not need it: a
loky worker takes `sys.path` from the parent, and the entry was gone from
theirs once they started.  `PYTHONPATH` now stays as it was.  The folder is
still added to the end of `sys.path`, so workers started after the load find
what the module imports from it.  A program that started its own process after loading a
config and relied on the entry must add it itself.

### Fixed: an alias diamond at a tensor key hung `sft-wick validate`

**Breaking (a tensor of more than 10^7 entries written in a config).**
`np.asarray` visits every path of a nested list, so a file of a few lines
whose lists each hold the one before twice took 3.1 s at 24 levels, and the
time doubled with each level.  `coupling` of a local or
non-local vertex and `amplitude`, `g0` and `g1` of `system.noise.sigma2` are
now measured along their first entries before NumPy reads them, and a tensor
of more than 10^7 entries is refused with `invalid_value` and the dotted path
of the key.  The first entries bound the entries NumPy visits, since it does
not descend past the depth of the first number.  A tensor in a `.npy` file
(`coupling_path`) has no limit.  PyYAML takes minutes to read a tensor written
out at the limit.

A `coupling` holding text kept its refusal and its message, and is now
refused before NumPy stores it.  NumPy gives every entry the width of the
longest string, so a short file of aliases around a long string took memory
that grew with each level.

### Fixed: an R read past the cumulative-rate spline returned a number

**Breaking (a time past the spline of a callable `gamma`).**
`System.propagators` keeps `t_max` inside `[t_min_cache, t_max_cache]`, and a C
table bounds every time by `t_max`.  Under `c_closed_form_only` nothing bounded
a time, so R extrapolated the spline with no error.  With
`gamma = 0.7 + 0.5 sin(3t)` and `t_max_cache = 3`, against `quad` of the rate:

| read | returned | reference |
|---|---|---|
| `R(3, 1)` | 0.24986 | 0.24986 |
| `R(4, 1)` | 0.16473 | 0.16623 |
| `R(6, 1)` | 6.5e-14 | 0.03976 |

A rate linear in `t` hides this: the spline of `Gamma` extrapolates a quadratic
exactly.  A table built by hand on such a model past the spline was wrong in
the same way.

R now raises `ValueError` on a causal read past either end of the spline, at
any level.  `Expansion.evaluate` and `Expansion.sweep` raise before anything is
read, a sweep before its first point.  `sft-wick validate` and `run` stop before
the expansion with status 2 and one line that names the entry.

The R callable refuses only a read: an acausal pair is 0 wherever it sits, and a
value within rounding of an end is read at that end.  The checks before a run
go by the request, by one rule at every level, since `validate` has no diagrams:
a request reads R if a leg is a response field, or if it selects an order above
0 in a system with a vertex.  They then count every time it names, so a pinned
`psi` later than its `phi`, an integrated point that a pinned point caps, and an
order that holds no diagram are refused at a time no R reaches, and lowering
`t_final` or raising `t_max_cache` clears it.  Order 0 of `phi` legs holds no R
and is not checked, `t_final` with every external point pinned is not read, and
a constant `gamma` or an `ExplicitR` has no spline.  The limits are the spline's
own breakpoints, so propagators cached on disk carry them and `_CACHE_SCHEMA`
does not change.

The check costs about 0.6 us on a scalar R call of 6 us, and about 11% on a
batch of 8,192, medians of three runs.  In-range numbers do not move: the 156 rows of the shipped
configs and 3,030 in-range R values, ends included, are bit-identical to the
previous commit.

### Fixed: a C lookup outside its table returned a number

**Breaking (a time or position the table does not cover).**  A `t_final`
above `t_max` was never compared with the table, a time below `t_min` was not
checked at all, and the limits of a full spatial grid were never read.  A lazy
table clamped to its edge and a full-grid table extrapolated, so each returned
a value with no error, and `sft-wick validate` and `run` both exited 0.
Measured against a `dblquad` of the integral that defines C, on tables with
`t_max = 2`, `r_max = 1` and `x_max = 1`:

| request | returned | reference |
|---|---|---|
| `t_final = 5`, lazy table | 0.15158 | 0.17080 |
| `t_final = 5`, full grid | 0.23699 | 0.17080 |
| `t_final = -0.5`, `t_min = 0`, full grid | -0.01518 | 0 |
| `r = 3` | -0.04423 | 0.002318 |
| `r = 50` | -1.747 | 7.1e-29 |
| `x = y = -3` | 2.819 | 0.00986 |

`sft-wick validate` and `run` now stop before the expansion with status 2 and
one line that names the entry, such as `sweep.t_final_grid[1]` or
`sweep.positions_grid.y[0]`.  `Expansion.evaluate` and `Expansion.sweep` raise
`ValueError`, a sweep before its first point.  `PropagatorCache.C_at_batch`,
`C_diagonal_batch` and `C_matrix_batch` refuse such a read too, and so does
`C_value` on a spatial table.  A hand-built cache with only a time table
stays with its accessors at L1, since `C_value` goes to direct quadrature past
the table, which is correct.

A value within rounding of a limit is read at the limit.  `t_final` is compared
only when a point is left free: with every external point pinned it is not
read, and such a run is unchanged.  There is no limit on cos on a rotation
table, on the position on a lazy table, or on anything under
`c_closed_form_only`, which reads no table.

In-range numbers do not move.  All 10 shipped YAML configs (with the grids
trimmed to their first and last values and at most 512 samples, so the limits
stay in) and 3,464 lookups on four table kinds are bit-identical to the previous
commit.
`test_S3_translation_vs_general_agree_on_invariant_input` read one point past
its own table; it now samples inside it.

### Changed: `sft-wick validate` reports what `sft-wick run` would refuse

**Breaking (a value the run could not use, and some it used by accident).**

`sft-wick validate`, `sft-wick run` and `sft-wick run --dry-run` make these
checks before any stage, each with exit status 2, one line, a code and the
dotted path.  Each of these passed `validate` and ended in a traceback with
status 1, in the expansion, the propagator table, the sweep or after it:

* `expand.diag_R`, `expand.diag_C` and `propagators.diag_C` where R or C has
  off-diagonal entries.
* An empty axis, or a value listed twice, on `sweep.t_final_grid`,
  `sweep.positions_grid` or `sweep.external_times_grid`.  Under `--dry-run`
  an empty axis ended in an `IndexError`, and a repeated value passed.
* A point that `sweep.external_times_grid` pins and `sweep.integrate_over`
  names.
* Points a response propagator joins at order 0, at two positions:
  `['phi_a(x)', 'psi_b(y)']` with `x` and `y` apart.  `validate` contracts
  the observable at order 0 for this, which takes milliseconds.
* A `plot` output whose `x`, `y`, `hue` or `facet_col` is no column of the
  sweep.
* `propagators.c_closed_form_module`, now imported before the expansion: a
  module Python cannot import, or without a callable
  `c_closed_form_attr`, is a `read_error`.  An attribute that is not
  callable used to fail when the table was built.
* `c_closed_form_only` with no closed form and `c_closed_form_vectorized`
  without it.  Where a C table is built: `propagators.t_max` not above
  `system.t_min`, `n_grid_t` below 4 when no spatial grid is given (the
  table of each separation is a cubic spline), and a full table with fewer
  nodes on an axis than `interp_method` needs (2 for `slinear`, 4 for
  `cubic` and `pchip`, 6 for `quintic`).  A run of `<phi psi>` alone, which
  reads no C, passed with such an `n_grid_t` and is refused too.
* Positions the run cannot combine: a number next to a vector on one axis,
  shapes that do not broadcast, a vector on a `general` full table, and a
  number next to a vector on a `translation` full table at an interacting
  order.
* An `output[].path` or a `cache_path` that cannot be written to: a name too
  long, a parent that is a file, and for an output a directory where the
  file would go (`np.savez` adds `.npz`, and `savefig` the default format
  to a name with no suffix).  A `plot` path with a suffix `savefig` has no
  format for.

These values are refused when the file is loaded:

* An entry of `sweep.t_final_grid` or `sweep.external_times_grid` that is no
  finite number, and an entry of `sweep.positions_grid` that is neither a
  finite number nor a non-empty vector of them.  Some of these ran: a NaN
  time at an order that does not read it, a `null` or NaN position on a
  closed-form path, `true` for 1.0.  A number as YAML reads `1e-3`, a
  string, is read as a number; it used to fail in some runs.
* `propagators.r_max` or `x_max` that is no positive finite number, and
  `n_grid_r`, `n_grid_cos` or `n_grid_x` that is no integer from 1.  A
  negative `r_max` built a table and read past it.
* `sweep.n_samples` below 1 or a negative `sweep.seed` under a QMC method,
  `sweep.n_gauss` below 1 under `gauss_legendre`, `propagators.n_grid_t`
  below 2 or a `t_max` that is not finite unless `c_closed_form_only` builds
  no table, `c_n_gauss` below 2, `system.field.n_components` below 1, a
  `system.t_min` that is not finite, an `n_jobs` of 0, and an integer too
  large for an array.  A key the method does not read stays unchecked.
  `n_samples: 0` ran at an order with no sampled diagram.  `n_jobs: 0` ran
  where a layer stayed serial, which depends on the grid and the diagrams;
  it means nothing to joblib, so it is refused at every size.
* `propagators.interp_method` that is no method of
  `scipy.interpolate.RegularGridInterpolator`.
* `output[].path`, `expand.cache_path` or `propagators.cache_path` that is
  no path.

A `sweep.vertex_types` label, and points a chain of response propagators
joins through a vertex, need the diagrams.  `sft-wick run` makes these two
checks after the expansion, before the propagator table, with status 2 and
one line; they used to end in a traceback, the second in the sweep.
`validate` runs no expansion past order 0 and passes a label.  It finds joined
points by walking the R assignments, except under `target: connected`, which it
passes.

Some failures still show only where they happen, such as a number that is
finite in the file and overflows in an integrand, a directory the process
may not write to, or a table too large for memory.

### Changed: every refusal of the parser has a code and a dotted path

`load_workflow_config` and `check_before_run` raise `ConfigError` for every
refused value, so `except ConfigError` covers them and `sft-wick` prints the
path.  Over 39 wrong values for each key of the 11 shipped configs, 42,880
cases, 29,363 raised a plain exception with no code or path and 284 a
`ConfigError` with no path; none does now.  A config file that cannot be
opened still raises `OSError` and one YAML cannot read `yaml.YAMLError`,
which have no key to name.

* A message is `path: reason`, so the text of many changed:
  `sweep.t_final_grid must be a list` is `sweep.t_final_grid: must be a
  list`.  An error keeps the type a caller caught: a value of the wrong type
  is a `TypeError` as well, a module that cannot be read an `ImportError` or
  an `OSError`, and a module without the attribute an `AttributeError`.
* `system.linear.type`, `output[].type`, `sweep.gl_chunk_size`,
  `qmc_coupling_chunk_size`, `propagator_cache_bytes`,
  `propagator_cache_contract`, `contraction_backend`,
  `contraction_max_bytes`, `expand.target` and the sections of the file
  name their key.  A misspelt `system.linear.type` or `output[].type` lists
  the type meant.
* The path of a vertex is a dotted path, `system.vertices[0].coupling`,
  with the name of the vertex in the reason.
* A module a config names is a `read_error` at the key that names it,
  whatever stops the import: a syntax error (a traceback, since `SyntaxError`
  was not caught), any exception it raises, and `sys.exit`, which ended the
  process.  It no longer stays in `sys.modules` half loaded.  A path or an
  attribute that is no string is refused at its key.
* An `.npy` file on which `np.load` raises `zipfile.BadZipFile` or
  `MemoryError` is a `read_error`; both ended in a traceback.
* An integer past the digit limit of `str`, as a value or as a key, is shown
  as a stand-in.  It replaced the message with that limit's own.

### Fixed: `compare-resources` walked a shared node once per path

`_known_files` recursed, so a long chain of aliases ended in a
`RecursionError`, and it walked every path to a shared node, so aliases used
twice per level took time exponential in their depth.  It uses a stack and
visits each node once.

### Fixed: an empty observable returned 0 at L1

`System.expand([], orders=[0])` gave an expansion that evaluated to 0, where
the moment of the identity is 1.  It raises `ValueError`, as the YAML loader
refuses `expand.observable: []`.

### Fixed: an error message could hold the expanded size of a value

A message showing a value printed its `repr`, which expands every path to
a node shared through YAML aliases, so the message of a refused value of a few
lines doubled with each level of sharing.  The
messages of `sweep.positions_grid`, `external_times_grid`, `t_final_grid`,
`component_tuples` and `expand.observable` were built on every load, so a
valid grid of 24 levels took 5.0 s to load.  `str()` of a list where the
parser expects text (`sweep.method`, `output[0].format`, `coupling_module`)
cost the same without showing it.  A value up to 1000 characters prints as
before, and a longer one as a stand-in such as `<list of length 2>`.  The
alias limit above now refuses the grid of 24 levels in 8 ms, and one of 400
levels in 58 ms.  A
tensor key such as `coupling` still builds the array the file describes, at
its expanded size, up to 10^7 entries.

### Fixed: a file whose aliases form a cycle passed `validate`

**Breaking (a file where a value holds itself).**  With
`positions_grid: {x: &g [0.0, *g], y: [0.5]}` the parser read the value
without looking inside it, so `sft-wick validate` passed the file,
`--print-resolved` ended in a `RecursionError` and `run` in a `ValueError`
or `TypeError` traceback.  A mapping or list that holds itself is refused
when the file is loaded, after the overrides, with `invalid_value` and the
dotted path of the alias that closes the cycle.  A node shared without a
cycle stays accepted.

### Fixed: an override through a YAML alias changed every place sharing the node

**Breaking (an override through a list or mapping the file shares).**  An
override wrote into the object YAML built, and an alias or a `<<:` merge
makes that object shared.  With `positions_grid: {x: &g [0.0, 0.5], y: *g}`,
`--override "sweep.positions_grid.y[1]=0.7"` also changed `x[1]`, and with
`output: [&o {...}, *o]`, `--override "output[1].path=b.md"` also changed
`output[0].path`, without an error.  An override now changes only the path
it names: the result is that of the file with the alias written out and
the same edit made.  It copies the mappings and lists on the path and no
others, so the cost does not grow with what the aliases expand to.  A
Python override value given at two paths is kept as given.

### Fixed: `compare-resources` kept the last of a key written twice in a merged mapping

`sft-wick compare-resources` refuses a key written twice in its config and
variants files, but a mapping merged in with `<<:` and written in place
was flattened without the check, so `case: {<<: {a: 1, a: 2}}` gave
`a: 2`.  It now reads both files with the loader of the workflow, so the
error gives the dotted path of the key, and overrides that the merge
defines stay.  A value that holds itself through aliases is refused with
`invalid YAML`; one that did so through a list ended in a `RecursionError`.

### Fixed: a quoted `"false"` switched a flag on

**Breaking (a value that is not a flag).**  `expand.response_phase`,
`collect_topology`, `diag_R`, `diag_C`, `iso_C` and
`propagators.c_closed_form_only`, `c_closed_form_vectorized` and `diag_C`
were read with Python's `bool()`, so a quoted `"false"`, `"no"` or `"off"`
switched the flag on without an error (`expand.iso_C`, `propagators.diag_C`).
They are read like the other flags of the file: `true` and `false`, `yes`
and `no`, `on` and `off`, `1` and `0`; any other value, `null` and `2`
included, is refused.  `expand.iso_R` is `null` or such a flag; a string
used to reach `System.expand` as written.

### Changed: a misspelling that gives another key of the block is refused

**Breaking (a file with a key written twice, or with a grid key the run
does not read).**

* A key written twice in one mapping is refused with the code
  `duplicate_key`, its dotted path and both line numbers (a key written as
  an alias gives the line of its anchor).  YAML keeps the
  last, so a second `n_samples`, or a misspelling that gives a key the
  block already holds (`y` for `x`), replaced the first without a message.
  A mapping merged in with `<<:` is checked as well, also when it is
  written in place (`<<: {t_min: 0.5, t_min: 0.0}`, a list of such
  mappings, or `<<: &anchor {...}`), and so is `<<` itself.  YAML merges
  two `<<` keys and the later wins a key they share; in one list the
  earlier wins, so `<<: *a` and then `<<: *b` is `<<: [*b, *a]`.  A key
  that overrides one brought in by the merge, or the earlier of two merged
  mappings, is not a duplicate.
* A spatial grid key of `propagators` that the run would not read is
  refused before the run: one for another homogeneity, or `r_max` without
  `n_grid_r` (`x_max` without `n_grid_x`, and the reverse).  `n_grid_t`
  misspelt as `n_grid_r` or `n_grid_x` used to leave the default 60 time
  nodes, and `sft-wick validate` reported the file valid.
* `npz` and `plot` outputs require `path`, and `plot` requires `x`, when
  the file is loaded; they were checked after the sweep.

`tests/test_workflow_schema.py` (SC6) lists every pair of keys in one block
that a single edit turns into each other, with what refuses a misspelling
between them.  One pair is not refused when only one of its keys is in the
file: `iso_R` and `iso_C`, and likewise `diag_R` and `diag_C`.  Each states
a simplification that `System.expand` refuses when its condition fails, and
switching one off keeps the value, so such a misspelling stops the run or
leaves the number unchanged.

### Added: a JSON Schema of the workflow files, `sft-wick validate` and `--print-resolved`

`sft_wick.workflow.schema.workflow_json_schema()` returns a JSON Schema
(draft 2020-12) of the workflow YAML, and `sft-wick schema` prints it: the
keys of every block, their value types, enumerations and defaults, and the
keys a block requires.  The keys and defaults of `expand`, `propagators`,
`sweep` and each `output` entry are the fields of their config dataclasses;
the top level and `system` follow `WorkflowConfig` and `SystemConfig`, and
the blocks inside `system` take their keys from the parser's key tables and
their defaults from the L1 specs they lower to.  Every shipped config
validates against it.  A rule that ties two keys together (`dt` or
`n_grid_t`, one coupling source per vertex) stays the parser's.

`sft-wick validate CONFIG` parses the config, applies any `--override`,
builds the System and makes the checks `run_workflow` makes before the
expansion (`check_before_run`), without running anything.  `sft-wick run`
makes the same checks first.  `--print-resolved` (on `run` and
`validate`) prints the config with the value the run uses for every key:
defaults filled in and derived values such as `n_grid_t` from `dt` shown.

### Changed: a refused config is one line with the dotted path and the nearest keys, exit status 2

**Breaking (the message, and the exit status of the CLI).**  The parser
raises `sft_wick.workflow.ConfigError`, a `ValueError` with the fields
`code`, `path` and `suggestions`.  Its message is one line: the dotted path
of the key (`system.t_mni`, `output[0].fromat`), the reason, and every valid
name at the smallest edit distance, so a single misspelling always lists
the name meant (`system.t_mni: unknown key; did you mean 't_min'?`).  The
0.7.0 message `system: unknown key(s) ['t_mni']; allowed: [...]` is gone.
`sft-wick run` and `sft-wick validate` print `sft-wick: config error [code]
...` on standard error and exit with status 2, where they printed a
traceback and exited with status 1; a parser `ValueError` or `TypeError`
without a code is printed as `invalid_value`, and a file that cannot be
read as `read_error`.  The codes are listed in the workflow guide.

Also when the file is loaded:

* The keys of the `kappa2`, kernel and `sigma2` blocks are checked.  0.7.0
  checked them only when the System was built, so `load_workflow_config`
  accepted `system.noise.kappa2.temporal.sigma_tt`.
* A misspelt section (`swep:`) is reported as an unknown key with a
  suggestion, where it was reported as a missing `sweep` section.
* `sweep.method` is checked, with a suggestion for `gauss_legendr`.  It
  used to fail at the sweep stage, after the expansion and the propagator
  table had been built.
* A point name in `sweep.positions_grid` that the observable does not use
  has the code `unknown_point` and a suggestion.  A point name in
  `sweep.external_times_grid` is checked the same way; it used to fail at
  the sweep stage.
* `output[].format` and `propagators.homogeneity` are checked, with a
  suggestion.  A misspelt table format used to fail after the whole sweep
  had run.
* Each entry of `expand.observable` must be an operator string that
  parses, like `phi_a(x)`; the error names `expand.observable[i]`.  `[]`
  and `{}` used to end in an `IndexError` and a `KeyError`, a number in a
  message without a path, and `phi_a` passed `sft-wick validate`.  A
  `FieldOperator` given through a Python override is refused too: it failed
  in the component check of the run; `System.expand` takes them at L1.
  An empty `expand.observable` is refused; the run printed 0 for the moment
  of the identity, which is 1.
* A value YAML cannot read (`!!bool x`, `!!int ""`, `!!int x`, `!!timestamp
  x`, the date `2001-13-45`, a sexagesimal float too large for a float), a
  file that is not in the encoding it is read with, and a file nested too
  deeply are a `read_error` with status 2.  They ended in a `KeyError`, an
  `IndexError`, an `AttributeError`, a `RecursionError`, or a one-line
  `invalid_value`; in `sft-wick compare-resources` the float ended in an
  `OverflowError`.  In Python all of them raise `yaml.YAMLError`.
* `n_samples: .inf`, a `dt` too small for a finite grid, a `kappa2` `type`
  that is a list and an empty `.npy` file are refused with a dotted path;
  they used to end in an `OverflowError`, a `TypeError` without a path
  or an `EOFError`.  A module without the attribute named is a `read_error`;
  it used to end in an `AttributeError`, and the `ConfigError` it raises now
  is one too.
* `run_workflow` checks the order selection, the parallel layers and the
  closed-form C module before the expansion, and parses the observable as
  `System.expand` does.  A field
  the system lacks (`wrong_a(x)`), a missing component index (`phi(x)`
  with two components) or, at interacting orders, two operators at one
  spatial label stop `sft-wick validate` and `sft-wick run` with
  `expand.observable` in the error.  A point named like a column of the
  sweep (`phi_a(a)`) stops them too, with the refusal `Expansion.sweep`
  makes after the expansion.
* An `--override` without `=` is a config error with status 2; it was a
  `SystemExit` with status 1.
* A diagram no longer reads operator uids, and `System.expand` no longer
  resets them.  The reset ran after the observable was built or parsed, so
  the vertices took uids it held and `Expansion.plot` raised `UID collision`
  on the records holding such a vertex, for example each record at order 1
  of a first expansion with one quartic vertex, and records of order n + 1
  after an expansion to order n.  `FeynmanDiagram.from_pairing` now finds
  each node by its position in the pairing, since a uid need not be
  unique.  Operators pickled in another process and expansions
  cached by an earlier version draw, and two observable operators with one
  uid, which drew a wrong graph without an error, draw two points.  Numbers
  never depended on uids.

### Changed: `--override` sets a key the file leaves out

`--override` and `load_workflow_config(path, overrides=...)` read the key
against the schema instead of the file.  A key the schema allows is set
when the file leaves it out (`system.t_min`, `expand.iso_C`,
`propagators.dt`); 0.7.0 accepted only the optional `sweep` keys and raised
`KeyError: override key ... does not exist in config` for the rest.  A
mapping on the way that the file leaves out is created
(`system.noise.sigma2.amplitude`), and a key of a `type` variant follows the
`type` in the file or one set by an earlier override.  The result is the
result of writing the key into the file: over the shipped configs there are
628 cases of a block leaving out a key the schema allows, and with the key
set to its default the parsed config, or the refusal, is the same either
way (`tests/test_workflow_corpora.py`, CO2).  An
override error is an `OverrideError`, a `ConfigError` that is also a
`KeyError`, so `except KeyError` still catches it; its message changed.

### Added: `SFT_WICK_STRICT=0`, for this release only

With `SFT_WICK_STRICT=0` in the environment, an unknown key in a workflow
file is a `ConfigWarning` (one line on standard error from the CLI) and is
ignored, as releases before 0.7.0 ignored it.  It is meant for moving old
configs over and will be removed in a later release.  Overrides stay
strict.

### Changed: test dependencies

`jsonschema>=4.0` joins the `dev` extra; the tests check the schema with
it.  The package does not import it.

### Changed: repository

* `tests/test_repo_hygiene.py` refuses any three-digit task number (`T`, an
  optional hyphen, three digits) and the subtask IDs that extend it, where it
  used to refuse only numbers below 100, also when a slash follows the
  number, and identifiers of planning notes kept outside the repository.  It
  checks tracked file names (RH9) and the commit messages reachable from
  `HEAD` (RH7).  `flagged_commit_messages` and `flagged_commit_content` take
  `git log` revision arguments, so the messages, added paths and added lines
  of a set of commits can be checked before a push; RH8 runs both on a
  scratch repository.

## 0.7.0 — 2026-10-29

This minor release adds execution and memory controls for coupling
evaluation and integration, three opt-in observables (connected targets,
finite Gaussian quadratic observables and analytic OU white cumulants), a
resource comparison command, vectorised response callables and a recipe for
changing the integration variable.  Default settings keep the 0.6.1 rules,
methods and resolutions; values move only where noted below.

It also fixes defects that returned a wrong number without an error: a
string `vertex_types`, observable labels spelled like the generated names,
numeric component labels in L0 Kronecker deltas, set observables, two
Gauss-Legendre position and domain cases, and user callables that write
into their arguments or return one buffer.  Several inputs now raise: a
workflow key or point the parser does not read, names and component tuples
that read two ways, a numeric label outside `1..n_components`, the keys
view of a mapping that is not a dict, a non-bool `equal_time` or
`already_R_contracted`, a custom cache's batch method that writes into its
time arguments, and writes into a `SpectralDensity`.  Each is marked
**Breaking** below; most of these inputs used to give a wrong number
without an error.

### Added: execution and memory controls

New keywords on `Expansion.evaluate` / `Expansion.sweep`, mirrored under
`sweep:` in YAML (`docs/user_guide/workflow.rst`, `docs/user_guide/resources.rst`):

* `gl_chunk_size` (default `4096`): evaluate Gauss-Legendre tensor nodes in
  blocks of at most this many, over every causal and kink piece.  A grid of
  at most one block is the eager grid, bit for bit; `None` keeps the eager
  grid at any size.  On a 331 776-node order-2 grid and a 117 649-node
  order-4 grid, the default took the same wall time as the eager grid and
  cut the peak RSS from 812 to 127 MB and from 1272 to 212 MB, with the
  same totals (`tools/benchmark_gl_chunk_size.py`).  On larger grids the
  block sums are added with `math.fsum`, so a GL value can change in its
  last digits, and vectorised hooks (`coupling_vectorized`,
  `R_time_vectorized`, `c_closed_form_vectorized`) receive blocks of at most
  4096 rows, so they must treat each sample independently of the batch.
  `--override sweep.gl_chunk_size=null` selects the eager grid on a file
  that omits the key.  Non-finite block totals combine as the eager sum
  over the nodes does (NaN, or the one infinity), and an exception raised
  while a block is evaluated propagates, an `OverflowError` from a user
  callable included.
* `qmc_coupling_chunk_size` (default `None`): build dynamic coupling tensors
  on `qmc_vectorized` in row blocks.  The callable must be sample-separable.
* `reuse_integrands` (default `False`): prepare static coupling tensors and
  spatial analysis once per component tuple within one sweep.
* `contraction_backend` (`"ordered"` default, `"einsum"`, `"factorized"`) and
  `contraction_max_bytes` (default 64 MiB): how a coupling tensor is
  contracted, and the workspace budget above which a contraction is refused
  rather than allocated.  `einsum` and `factorized` run a diagram call only
  when their result obeys the bound the ordered fold obeys,
  `|result - exact| <= 1.01 (m + q + A T) u S` with `q` the rationals of one
  product that have no exact float64 value, every entry the ordered fold
  multiplies is finite and free of overflow, and every array the call holds
  at one time fits `contraction_max_bytes` (Python objects aside).
  Otherwise `einsum` declines and the ordered fold runs; `factorized` first
  materializes the coefficient, which raises `ValueError` when its payload
  exceeds `contraction_max_bytes`, so every budget below 256 KiB raises
  there, where `ordered` and `einsum` return a value.  On 330 diagrams captured from
  `Expansion.evaluate` (`N` = 2, 3 and 5, orders 2 to 4), `einsum` ran on 204
  to 310 of them at 2.2x to 103x the ordered speed per call, and
  `factorized` on 2 to 4, whose coupling sums have no monomials of opposite
  sign (`docs/user_guide/resources.rst`).
* `propagator_cache_bytes` (default `None`, off) with
  `propagator_cache_contract: fixed_model`: a byte-bounded cache of
  propagator lookups, valid only while the model and its callbacks stay fixed.

`sft-wick run` and `run --dry-run` report each requested control.
`sft-wick compare-resources CONFIG --variants VARIANTS --json REPORT` times and
memory-profiles resource variants of one workflow in fresh processes, checks
that every variant returns the same rows within declared tolerances, and never
modifies the configuration.

### Added: opt-in observables

* `System.expand(..., target="connected")` (also `compute_moment`,
  `compute_moment_numerical`, YAML `expand.target`) keeps only the connected
  contributions.  The default `target="moment"` is unchanged
  (`docs/user_guide/connected_targets.rst`).  Under it, a `vertex_types`
  label that a diagram at the requested orders could carry, and none does,
  is a computed zero.
* `GaussianQuadratic`, `GaussianQuadraticFamily`,
  `GaussianQuadraticStatistics` and `GaussianSpectralTail` give the cumulants
  and characteristic function of `Y = b + l·z + zᵀAz` for a finite Gaussian
  `z` (`docs/user_guide/gaussian_quadratic.rst`).  A mixed cumulant or joint
  coefficient of a family is the float64 contraction when a rigorous bound
  puts its rounding error within `1e-12` of its magnitude, and otherwise the
  correctly rounded exact value of the prepared data; a zero is returned
  only when it is proved.  The exact evaluation raises `ArithmeticError`
  above its work or memory limit.  Preparation uses an exactly symmetric
  `A` unchanged, subnormal entries included, and replaces any other by its
  symmetric part with each entry rounded once to float64; it raises
  `ValueError` only when every entry of a declared matrix is below
  `2**-1022` and an entry of its symmetric part is not a float64 value.
* `OUWhiteCumulant` evaluates the response contraction of a static
  equal-time noise cumulant under a constant diagonal OU response in closed
  form, removing one time integration per absorbed vertex
  (`docs/user_guide/ou_white_cumulant.rst`).

### Changed: faster coupling evaluation and integration

Coupling expressions with repeated component assignments are compiled once per
diagram into an ordered execution plan; repeated tensor reads, scalar reads
and propagator lookups within a batch are reused; `import sft_wick` no longer
imports Matplotlib until `drawing` or `DiagramRenderer` is used.  These paths
keep the arithmetic order of 0.6.1 and are compared with it by the test
suite at the byte level where the operation order is unchanged.

### Fixed: Gauss-Legendre values with one integrated and one pinned external

With `integrate_over` covering one external point and another pinned at a
time inside the domain, Gauss-Legendre and `nquad` did not split the domain
at the C kink against the pinned time, so GL converged only algebraically.
The split is now made (`docs/user_guide/workflow.rst`, "Declared kinks
against a fixed external time").  On a Brownian C with `y` pinned at
`t_min + 0.45`, the GL error at 12 nodes falls from 2.0e-3 to round-off.
**GL and `nquad` values in this regime change**; other configurations are
unchanged to 4.5e-16.

### Fixed: a fixed vector position on a batch of equal length

`C_at_batch` reads a 1-D position array whose length equals the batch as
one scalar position per sample.  The eager Gauss-Legendre path, vectorised
QMC and `integrate_two_point_qmc` passed a fixed d-vector position
unchanged, so a batch of exactly d samples read it as d scalar positions.  On a 3-D pair with one integrated external time, GL at
`n_gauss = 3` was 25 % off and QMC at `n_samples = 2` and `4` (2-D and 4-D
pairs) 29 % and 35 % off; `integrate_two_point_qmc` at order 1 was 8 % and
44 % off.  Every batch lookup now broadcasts such a vector to `(n, d)` first
(`evaluate.py::_C_at_fixed_positions`); values change only when the batch
length equals the position dimension.  Locked by
`tests/test_d_dim_spatial.py::test_DD4_*` and `test_DD5_*`, which fail
without the fix.

### Changed: a key or point name the parser does not read raises

**Breaking.**  A workflow file used to accept any key, and a misspelt one
fell back to its default without a message: `sweep.gl_chunk_sise` kept the
default bound, `system.t_minn` integrated from 0, `expand.diag_c: false`
kept the diagonal simplification, and `system.noise.sigma_2` dropped the
white noise (with `amplitude: 0.3` added to the quickstart model, the value
came out 23 times too small).  Every block now
refuses a key it does not read with `ValueError: <block>: unknown key(s)
[...]; allowed: [...]`: the top level, `system`, `system.field`,
`system.noise`, `expand`, `propagators`, `sweep` and each `output` entry
(the kernel, `sigma2`, `linear` and vertex blocks already did).  A
top-level key used only to hold YAML anchors or notes is refused too.

`sweep.positions_grid` refuses a point that `expand.observable` does not
name when the file is loaded.  A misspelt point used to stay at the
default position: `positions_grid: {x: [0], yy: [0.45]}` returned the value
at zero separation, 57 % high on the quickstart model, and added a `yy`
column.  A point the observable names and the grid omits sat at the default
coordinate 0; YAML now requires every one.  A grid key that YAML reads as a
number or a boolean (an unquoted `1` or `on`) is refused with advice to
quote it, and a block of the wrong shape (a mapping for `output`, a scalar
for `vertices`, a string observable) is refused with the block's name.
`sweep.integrate_over` is checked at load, and one string other than `all`
is one point name, so `--override sweep.integrate_over=x` runs; at L1 write
a list.  The L1 `positions` argument of `Expansion.evaluate` and
`Expansion.sweep` still accepts points the observable does not use, so one
positions dict can serve several observables; a point given no position
takes the default there.  All shipped configurations load unchanged
(`tests/test_yaml_spec_coverage.py`, YC11 and YC12).

### Fixed: a string `vertex_types` was read one character at a time

`Expansion.evaluate` and `Expansion.sweep` filtered diagrams with
`set(vertex_types)`, so `vertex_types="FK"`, which YAML `vertex_types: FK`
and `--override sweep.vertex_types=FK` produce, selected the label `F` and
dropped `FK`.  On a two-component model (cubic `F`, constant `κ³`, rates 0.8
and 1.3, `t_min = 0.3`) the run returned the F channel, -1.0286e-2, for the
FK channel, -2.0545e-1, with no error; the FK value agrees with a 1-D quad of
the channel to 2.1e-12.  **Breaking**: the selectors follow one rule at L1,
in `System.expand` and in YAML.

* One string is one label; a list, tuple, set, array, pandas Index or other
  iterable of strings is a set of labels.  A mapping raises `TypeError`
  (pass `list(exp.by_vertex_type(order))`), and so do bytes, a scalar and a
  non-string entry.
* One integer is one order.  A boolean, a string, a float, a negative order
  and a non-integer entry raise: `"2"` used to select order 2, `2.5` order 2,
  `b"2"` order 50, and `System.expand(obs, orders=2)` raised.
* An empty selection raises `ValueError`.  Under the moment target, an order
  outside `expand.orders` and a label that no diagram at the selected orders
  carries raise before any integral, naming what is present; they used to
  fail after the work, and a misspelt label next to a valid one dropped its
  channel without a message.  An order of the expansion that holds no
  diagram gives zero rows, as under the connected target; it used to end in
  `KeyError: 'x'` in `totals()` after the propagators were built.
* YAML applies the rules when the file is loaded, and the run and the
  dry-run estimate check the selection against the expansion before the
  propagators are built.

### Fixed: observable labels spelled like the generated names

The expansion names its vertex points `y_0, y_1, ...` and its component
indices `i_0, i_1, ...`, and the Wick engines key operators by these
strings, so an observable label spelled the same way was merged with the
vertex point or index of that name.  On a generic model whose order-2 value
is 2.5355e-2 (exact Itô hierarchy), `(y_0, y_1)` returned 0, `(x, y_0)` and
three other spellings raised about a response chain joining two points, and
the L0 component labels `(i_0, i_1)` gave 0.0740 and 0.0672.  The generators
now skip the observable's labels (`IndexContext(reserved=...)`, which a
caller that instantiates vertex copies itself must pass too).  Every
spelling agrees with `(x, y)` to 1e-12 or better (4e-16 at order 2);
observables without such labels
give the same diagrams, expressions and cache keys as before.

### Changed: names that read two ways raise

**Breaking.**  Each of these returned a number without an error.

* Two observable operators that share a component letter carry one index,
  so a component tuple giving them different values took the later one:
  `<phi_a(x) phi_a(y)>` at `(0, 1)` returned the `(1, 1)` values (0.05272 at
  order 0, where the value is 0).  `Expansion.evaluate`, `Expansion.sweep`
  and the YAML component axis refuse such a tuple.
* Of two vertices with one name, the second coupling replaced the first for
  both (-0.01099 against 0.01867 at order 2).  `System` refuses a repeated
  name.
* With vertices `F`, `K` and `FK`, the `F K` and `FK` channels were both
  labelled `"FK"` (-0.06248 against -0.05974 for `F K` alone).  `System`
  refuses vertex names for which two sets of vertices share a label, when
  it is built.
* An L0 `positions` key spelled like a vertex point (`y_0`) was read as that
  point's position; the resolvers read external points only.
* `FieldSpec("psi")` gave the physical and the response field one name, so
  a `phi` and a `psi` operator of one label shared a cache key; it raises.

### Fixed: literal component labels in Kronecker deltas (L0)

A numeric component label in an L0 observable, as in `phi("1", "x")`,
counts from 1 for coupling symbols and propagator legs, but the
`KroneckerDelta` that `diag_C`, `diag_R`, `iso_C` and `iso_R` insert between
two external labels read it from 0.  On a two-component OU process (rates
0.7 and 1.9, white noise diag(0.8, 1.3), `t_min = 0.4`), with both L0
engines and every integrator:

* `<phi_1 phi_b>` under `diag_C`: 0 at b = 0 (exact 0.315), and 0.315 at
  b = 1 (exact 0); `<phi_1 psi_b>` under `diag_R` the same with 0.657.
* Four points under `diag_C`: 0 at (b, d) = (0, 1) and (1, 0) (exact 3.3e-3
  and 5.4e-3).
* Three points at first order in a quadratic drift: -0.138 against -0.330
  from the exact hierarchy.

Every resolver now reads the label one way (`sft_wick.indices.literal_component`).
**Breaking (L0)**: `compute_moment` and `compute_moment_numerical` refuse a
numeric label outside `1..n_components`, also when `fixed_indices` pins it;
`"0"` used to wrap to the last component.  L1 strings and YAML accept only
letter labels and were not affected.

### Fixed: `System.expand` with `FieldOperator` observables or a set

The `FieldOperator` branch of `System.expand` raised `AttributeError` on
every call, although the docstring offered the form.  It now gives the
expansion and the cache key of the string spelling, and refuses an
operator of another field, a missing label on a multi-component field, a
non-string label and a numeric literal label.  A set or frozenset
observable came in hash order, so `PYTHONHASHSEED` decided which component
of the tuple belonged to which operator (-0.06211 or -0.14544 at `(0, 1)`);
it raises `ValueError`.  A dict's keys view keeps the dict's insertion order
and is accepted, also through an `OrderedDict`, a dict subclass or a
`MappingProxyType`.  **Breaking**: the keys view of any other mapping
raises, since the package cannot check its order; that includes one whose
order is fixed, such as `ChainMap(...).keys()`, which 0.6.1 accepted with
the right value.  Pass `list(mapping.keys())`.

### Fixed: user callables that write into their arguments or return one buffer

Each site below returned a wrong value without an error for a user callable
that wrote into an array the package reads again, or returned one buffer
that the package kept across a later call.  Callables that do neither give
the same values as before, apart from the `t_max` conversion in the last
bullet of the first group.

Propagator hooks (L0, and L1 or L2 where a hook reaches it):

* A closed-form C hook (`c_value_fn`, a `c_closed_form`, an override of
  `_C_value_direct`) received arrays the package reads again: the caller's
  positions on the per-sample integrators, a table builder's position for
  every cell, the time batches of the vectorised path.  A hook forming
  `x1 - x2` in place gave -0.14713 on `qmc` against -0.10545 at d = 3; a
  vectorised hook doing `t1 -= t_min` gave 0.06517 against 0.14215 at order
  2.  Hooks now receive copies of array and list arguments (shallow ones).
* A `GeneralKappa2` or `CustomImpulse` callable that wrote into its
  positions changed every later quadrature node: at 3-D positions C_00
  came out about 1e-9 (`gauss_legendre`) and 0.0 (`dblquad`) against
  5.85e-2, and the caller's
  position came back changed.  It now receives copies, at about 4 % per GL C
  value with 3-D positions.
* A vectorised closed-form C that filled and returned one buffer made every
  earlier C factor equal to the last one: 0.28329 against 0.15921 under
  `diag_C` at order 2, and 0.054947 against 0.043179 without it.
* An L0 coupling callable that returned one buffer gave both κ³ copies of a
  six-point function the last copy's tensor, 20 % off, on `gauss_legendre`
  and `qmc_vectorized` for a vectorised callable and on the per-sample
  `qmc` loop for either contract.  L1 wraps each callable in its MSR
  factor, which allocates, and was not affected.
* A κ² that returned one buffer gave the diagonal-cusp probe three equal
  values, so `dblquad` integrated a `|Δt|` cusp without the split (7.9e-8
  relative instead of 1.9e-16, 38 times slower).  `integrate_two_point_qmc`
  divided κ² at a separation by the κ² at zero separation it had kept, so
  C_11 at separation 0.8 came out 2.23 times (e^0.8) too large at order 0
  and 2.20 times at order 2.
* The Gauss-Legendre C quadrature kept `R(t1, λ1)` while it called
  `R(t2, λ2)`: a dense `ExplicitR` returning one buffer gave C_00 48 % high
  without white noise and 62 % high with it, and C_10 of the wrong sign in
  both.  `dblquad` was not affected.  The YAML R probe tested `R(t1, t2)`
  after `R(t2, t1)` had overwritten it, so an R that is not finite at causal
  times (NaN or inf) loaded when it reused a buffer; the causality check
  reads the acausal value and was not affected.
* `PropagatorCache.C_value` with object-dtype positions returned the C
  hook's own array (`c_value_fn` at L0,
  `System.propagators(c_closed_form=..., c_closed_form_only=True).cache` at
  L1), so a hook reusing a buffer changed an earlier result (C_01 from
  -0.0235 to -0.0075).  The integrators consume each lookup at once and
  were not affected.
* `select_gl_node_count`, `resolve_c_method` and the translation, rotation
  and general table builders passed the caller's `t_max` object to the
  `R_time` hook, and every C table builder kept it as the table range.  A
  0-d array `t_max` and an R writing into its time argument moved `t_max`
  from 2.1 to 1.58, chose `dblquad` instead of Gauss-Legendre and changed C
  by up to 23 % (L0).  A 0-d `t_max` that the caller raised after the build
  moved the range with it: the legacy table then read its spline outside
  the grid (-3.5 % at t = (2.8, 2.6) with `t_max = 2.1`), and on a spatial
  table `Expansion.evaluate` skipped its horizon check and returned the
  clamped value instead of raising `ValueError`.  `t_max` is now converted
  to a float, so a float32 `t_max` gives a float64 time grid, which moves C
  by up to a few 1e-8 relative.

Spec wrappers (L1 and L2):

* A callable `DiagonalA` rate that filled and returned one buffer gave every
  row of the rate grid behind the Γ spline the rate at `t_max_cache`.  On a
  two-component rate, R(2.7, 1.1) came out 0.1264 and 0.4643 against the
  exact 0.1389 and 0.2666; in `Expansion.evaluate` order 0 moved by -8.9 %
  and +53 %, order 2 at (1, 1) by +129 %, and order 2 at (0, 1) changed
  sign.  Through YAML `gamma_module`, a rate whose components meet only at
  `t_max_cache` made a matrix R scalar, so both components got the response
  of the rate at `t_max_cache` held constant (R(2.7, 1.1) = 0.2903 for
  both, against the exact 0.1344 and 0.2224).
* `SeparableTranslation` and `SeparableRotation` given raw callables (not
  `CustomKernel`) kept the temporal value while they called the spatial or
  angular kernel.  Two kernels filling one buffer made κ² the square of the
  second factor (0.3282 against 0.0767).  The order-0 C at distinct points
  came out 76 % (C_00) and 78 % (C_11) high on the lazy translation table,
  6.5 % and 8.2 % high on the full table, and 1.5 % low (C_00) in rotation;
  order 2 moved by up to 141 %.  YAML `type: custom` and the built-in
  kernels return floats and were not affected.

Custom propagator caches (L1 through `propagators_from_cache`, and L0).
The figures are measured on the two-component systems of
`tests/test_custom_cache_aliasing.py` and
`tests/test_trusted_cache_delegates.py`; on another system the same
mechanisms give other sizes.

* The batched integrators (`gauss_legendre`, `qmc_vectorized`,
  `integrate_two_point_qmc`, the zero-dimensional branch) keep one batch per
  C and matrix-R factor while they ask for the next.  A cache that overrides
  a batch method, or is not a `PropagatorCache`, and reuses one output
  buffer gave every kept factor the last batch: 2.8 times the order-2 value
  through `C_at_batch`, +16.5 % through `R_matrix_batch`, -1.2 % through
  `C_diagonal_batch`, -5.5 % and -12.5 % in the zero-dimensional branch.
  The outputs of batch methods other than the package's own are now copied.
* `gauss_legendre`, `qmc_vectorized` and `integrate_two_point_qmc` passed
  time batches they read again, so a batch method forming the lag in place
  (`t1 -= t2`) changed every later factor: -72 % through `R_matrix_batch`,
  -59 % through `R_time_batch`, -63 % in `integrate_two_point_qmc`.
  **Breaking (custom caches)**: those batches are now read-only, and such a
  write raises `ValueError`, as does a Cython memoryview over them declared
  without `const`.  So is every other time array the package passes to a
  batch method: those of the zero-dimensional branch and of
  `PropagatorCache.C_value` (the order-0 diagram on every integrator, every
  diagram on `qmc_scalar` and `nquad`), and the arguments that the package's
  `C_at_batch`, `C_diagonal_batch` and `C_matrix_batch` pass on to methods a
  subclass can override (`C_diagonal_batch` or `C_matrix_batch` in the
  translation fallback of `C_at_batch`, `_lazy_lookup`,
  `_closed_form_at_batch_diag`, `_on_diagonal`), as read-only views where
  they are the caller's arrays.  A write there changed a value only through
  `_on_diagonal`, after which `C_diagonal_batch` and `C_matrix_batch` read
  the times again: at coincident times the zero-dimensional seven-point
  function came out -61.6 % through `C_diagonal_batch` and -99.4 % through
  `C_matrix_batch` on `gauss_legendre`, `qmc_vectorized` and `nquad`
  (`qmc_scalar` looks C up with float times, which such a write leaves
  unchanged).  On a direct call the write changed the caller's arrays
  (`[1.9, 1.7]` became `[0.45, 0.5]` through the translation fallback).
  The legacy-table lookups of a direct `C_value` or `C_diagonal` also read
  the times again after `_on_diagonal`: with 0-d array times at 1.4 the
  write moved the caller's `t1` to 0, and C_00 came out -82.4 %, the value
  at (0.7, 0.7).  These pass read-only views too.
* A custom `C_at_batch`, `C_value` or `C_diagonal` received the caller's
  position arrays, one object for both ends of a C inside a direction group.
  One forming the separation in place zeroed the caller's positions or
  turned `x` into `x - y`: 3.2 times the order-2 value through `C_at_batch`,
  and at orders 0 and 2 through `C_value` -43 % on `gauss_legendre` and 2.5
  times the value on `qmc_scalar`.  A custom `C_value` or `C_diagonal` now
  receives copies of array and list positions, and every `C_at_batch`, the
  package's own included, receives float64 array copies; the integrators
  pass the package's own `C_value` and `C_diagonal` the caller's objects.
  A direct `C_at_batch` passed the caller's position arrays on to an
  overridden `_lazy_lookup` or `_closed_form_at_batch_diag`: one forming the
  separation in place turned the caller's `x` into `x - y`, and a second
  call on the same arrays came out up to 90 % off.  These overrides now
  receive copies; the package's own receive the arrays themselves.
* On the legacy table, `C_value` and `C_diagonal` returned the array of an
  overridden `_C_value_from_table` or `_C_diagonal_from_table` as it was.
  One filling and returning one buffer changed an earlier C_00 from 0.1589
  to 0.0499 at the next lookup.  The integrators take each entry before the
  next lookup and were not affected.  The result is now copied when the
  lookup is not the package's own.
* The copies above did not reach a subclass that keeps
  `SpectralPropagatorCache.C_value` or `PropagatorCache.C_at_batch` and
  overrides a method they call.  `C_value` passes its positions to
  `_c_matrix`: one forming the separation in place turned the caller's `x`
  into `x - y` at order 0, so at orders 0 and 2 in one call the total came
  out -16.8 % on `gauss_legendre` and -17.0 % on `qmc_scalar` at component
  pair (1, 1), -49 % at (1, 0), and the second point of an `nquad` sweep
  -90.0 %.  `C_at_batch` returns the array of `_assemble_C`, `_lazy_lookup`
  or `_closed_form_at_batch_diag`: any of them returning one buffer gave
  every kept C factor the last one, at order 2 +181.8 % (mixing κ²) and
  +179.9 % (diagonal κ²) on `gauss_legendre`, `qmc_vectorized` and
  `integrate_two_point_qmc`, with and without the lookup bank.  Each now
  counts as the package's own only while those methods are too; otherwise
  its positions or its output are copied.  On a direct call, `C_value` also
  handed `_c_matrix` the caller's positions and returned its array: a second
  call on the same `x` came out -90.0 %, and a `_c_matrix` filling one buffer
  changed the first result by -50.6 % at the next call.  It now hands an
  overridden `_c_matrix` copies and returns a copy.
* `DiagramIntegrand.evaluate` handed `R_product` the times dict that its C
  lookups read next, the caller's own dict on a direct call.  An
  `R_product` that moved every time down by `t_min` after reading them gave
  order 0 at the shifted times, -27.6 % on every integrator; at orders 0
  and 2, -57.1 % on `qmc_scalar` and -9.2 % in `integrate_two_point_qmc`.
  One that removed the two ends of each R raised `KeyError` at order 2 on
  `qmc_scalar`.  An `R_product` other than the package's own now receives a
  copy of the dict.

Spectral densities, self-consistency, R-contracted vertices:

* `SpectralDensity.from_callable` called `rho(h)` and then built the nodes
  from the same `h`.  A `rho` shifting its argument by 0.2 moved every node
  by 0.2, and one filling and returning its argument made the nodes the
  density values: the order-0 C_11 through `spectral_cache` came out 51 %
  high and 12.8 times too large.  `rho` now receives a copy.  **Breaking
  (`SpectralDensity`)**: a `rho` that reorders its argument in place raises
  `ValueError`.  That includes a `rho` that reversed its argument and
  returned the density at the reversed points, which gave the right value
  before because the Gauss-Legendre weights are symmetric; every other
  in-place reordering was wrong.
* `SpectralDensity.average` passed the density's own nodes to `f`.  An `f`
  doubling its argument in place moved the nodes of the frozen, hashed
  density, so a later average gave 0.0157 against 0.0907 and a cache built
  earlier gave 0.108 of its order-0 value.  `f` now receives a copy.
  **Breaking (`SpectralDensity`)**: the stored nodes and weights are
  read-only, also after pickle, joblib and `copy.deepcopy`, so a direct
  write (`dens.nodes[0] = ...`, `dens.nodes *= 2`) raises `ValueError`.
* `solve_self_consistency` handed a user distance the live states, the
  callback the live state after mixing, and iterated on the caller's
  `initial` itself.  On a linear contraction that converges in 27
  iterations, a distance subtracting in place diverged after 6, one sorting
  in place reported convergence 2.8 from the fixed point, and a callback
  rounding in place ended at `max_iter`.  A step that fills and returns its
  own buffer, started from that buffer, reported convergence after one
  iteration with a residual of 0.  The distance and the callback now
  receive copies and the loop starts from a copy of `initial`, so a callback
  can no longer steer the iteration by writing into the state.
* `build_R_contracted_callable` passed the caller's positions to each of its
  raw-kernel calls and kept `chi_grid` by reference.  A raw kernel centring
  its positions in place made the order-1 three-point function at L1
  (`already_R_contracted=True`) 566 times too small, and a change to
  `chi_grid` after the build changed every later value.  The wrapper now
  keeps its own copies.

### Changed: the propagator disk cache covers built-in specs only

`System.propagators(cache_path=...)` now keys tables on the complete content
of built-in linear and noise specs.  A system with `CustomKernel`,
`GeneralKappa2`, `CustomImpulse`, `ExplicitR`, a callable `gamma` or a user
closed form computes its tables normally and writes no cache file, because a
callable has no version identity.  Up to 0.6.1 such systems were cached
under a `repr()` key.

### Fixed: small coupling coefficients were dropped

Indexed static and dynamic batch contractions skipped coefficients whose
magnitude was below `1e-20`.  A reciprocal rescaling of a small coefficient
and a large kernel therefore lost its contribution.  Only exact zeros are now
skipped.

### Fixed: serial and small-batch paths ignored the contraction options

`Expansion.evaluate`, `reuse_integrands=False` and the serial shortcut for at
most two diagrams built integrands without forwarding `contraction_backend`
and `contraction_max_bytes`, so they used the dense path and skipped the
budget refusal.  Every path now uses one integrand builder.

### Changed: documentation and repository

* The README and the documentation link the repository and the paper
  (arXiv:2606.19480); `docs/citing.rst` gives the BibTeX entry and the Zenodo
  DOIs, and the theory section points to the paper.
* `tests/test_repo_hygiene.py` fails if a tracked file is an agent working
  file or development record, or contains a local absolute path or a task ID.
  `.gitignore` and the sdist excludes cover `.agents/`, `.agent-state/` and
  `docs/research/`.

### Added: changing the integration variable (documentation)

`docs/user_guide/time_variable.rst` gives the rescaling that poses a system
in another variable `u`, `t = f(u)`, without a change in the package: the
power of `J = dt/du` that each ingredient carries (response, `DiagonalA`
rate, C, `κ²`, white `σ²`, local and non-local vertices, externals).  On an
Einstein-de Sitter lensing toy with a source at z = 1100, the order-2
channel posed in the affine parameter is 17 per cent off at 16 nodes per
time variable; posed in the comoving distance it is converged to 4.5e-10 at
12.  `tests/test_time_variable_recipe.py` checks every row of the table
against the same system in `t` (agreement 5e-14 for C, 4.6e-12 for
diagrams) and that a wrong factor in any row is detected.  Proposed by
unify_wl (option (i) of its sft-wick proposal).

### Fixed: a non-bool `equal_time` or `already_R_contracted` raises

`Vertex` and `NonLocalVertex` stored `bool(equal_time)`.  A tuple of leg
groups, the spelling of a partially equal-time vertex the package does not
support, became `True` and put every leg at one time; a unify_wl probe got
0.0 against 54.88 with no error.  Both flags now accept a Python or NumPy
bool and raise `TypeError` otherwise.  `already_R_contracted` had the same
coercion and gets the same check.  **Breaking**: a non-bool value that
0.6.1 accepted, such as `1` or a tuple, raises.

### Added: vectorised response callables

`PropagatorCache.R_time_batch` called `model.R_time` once per causal sample
through `np.vectorize`.  In a weak-lensing workload (unify_wl, scalar
`ExplicitR` backed by a spline) these calls took 94 to 98.5 per cent of the
CPU time of a Gauss-Legendre evaluation.

* A response callable with the attribute `vectorized = True` accepts two
  arrays of shape `(n,)` and returns `(n,)` (scalar R) or `(n, N, N)`.
  `R_time_batch` and `R_matrix_batch` call it once on the causal pairs of a
  batch.  It must still accept two floats.
* The four built-in R classes of `DiagonalA` are vectorised, so every system
  with a `DiagonalA` drift uses this path on `gauss_legendre` and
  `qmc_vectorized` without a change.
* `ExplicitR(vectorized=True)` and the YAML key
  `system.linear.R_time_vectorized: true` declare a user callable as
  vectorised.  The YAML loader calls it on arrays once and compares with its
  scalar calls, which it makes first, on floats, and copies, so an R that
  reuses one buffer or uses its time arrays as scratch space loads.
* A callable without the flag is now called once per distinct causal pair
  for a scalar R, as `R_matrix_batch` already did for a matrix R.

Values are unchanged bit for bit: `tests/test_vectorized_r_time.py` compares
the array calls with the scalar calls, and channel totals of the vectorised
built-ins with the same R behind a scalar-only wrapper, with `==`.

## 0.6.1 — 2026-09-14

This patch release improves C-table accuracy near coincident times, completes
the integration-domain split for two swept external times, and corrects and
independently verifies demo2's higher-order budget. It also supports joblib
1.6's standalone cloudpickle and excludes local task files and generated
caches from source distributions.

The public API, Wick contraction rules, diagram combinatorics and MSR factors
are unchanged. Numerical results can change on the affected table and
integration paths. Workflow disk caches from earlier schemas are rebuilt
automatically on first use, including cached expansions. Closed-form-only C
lookups bypass the table change. The detailed evidence and measured limits
follow below; intermediate demo2 results are explicitly marked as superseded.

The repaired implementation passed the full CI suite on Python 3.10, 3.11
and 3.12 (2194 passed, 8 skipped and 19 slow tests deselected per version),
plus distribution and example runtime checks. The independent demo2 reference
checks the saved perturbative channels; agreement with a finite-order
simulation comparison is a separate question from their numerical accuracy.

### Fixed: hook serialization with joblib 1.6

Joblib 1.6 uses standalone cloudpickle instead of `joblib.externals.cloudpickle`.
The demo2 analytic helper and YAML hook registration now support both layouts,
preferring the bundled copy on older joblib so registration reaches the same
serializer used by its workers. This fixes demo2 test collection and prevents
silent loss of by-value hook registration with an already running worker pool.
The regression test transports a hook after deleting its source file; the
independent subprocess kernel check also avoids the removed import path.
No scientific formula or numerical tolerance changes in this repair.

### Fixed: source distribution excludes local task tools and example caches

The sdist now explicitly excludes `.agents/`, `.agent-state/` and generated
example pickle files. A local build previously bundled the untracked task
handoff skill and demo2 refinement caches. The wheel contents and runtime
dependencies are unchanged.

### Fixed: min-time/lag table boundaries and the general earlier-first lookup

The general full-grid builder separately tabulates the earlier-first `E`
when a covariance cannot be recovered by transposition, including user C
callables.  That branch incorrectly exchanged the spatial arguments again
at lookup.  For the rank-one Gaussian field `phi(x,t)=(t+x*t**2)*Z`, a query
at `(x,t;y,s)=(0,0.25;1,0.75)` returned 0.234375 instead of 0.328125, a
28.57 % error even at a grid node.  Explicit E tables now retain their
spatial order; transposed L tables still exchange the spatial arguments.

The rectangular min-time/lag grid also evaluated physical kernels up to
`2*t_max-t_min`, although the requested horizon is `t_max`.  At `t_max=1`,
all four table paths called a bounded response at 1.25 and raised.  They
now evaluate only the physical triangle, adding three in-domain samples
on its shortest edges.  Auxiliary values outside the triangle are cubic
continuations of the computed table values, without invoking R, noise or
user C there.  The original physical grid nodes and the lag-zero boundary
are retained.  Default full-grid linear interpolation uses physical
`(t1,t2)` cells, splitting a diagonal cell into its two time orders.  Its
weights therefore combine physical vertices only: negative auxiliary
continuations cannot turn a positive covariance negative.  This also
retains the off-diagonal accuracy of the previous linear time grid.
At 41 time nodes this uses 864 physical samples per time
order and spatial key, down from 1681.  Cache schema 3 rejects tables built
with either defect.  `tests/test_min_lag_boundaries.py` checks the spatial
order, bounded kernels, convergence at the horizon and both temporal
corners, positivity under steep decay, and invalidation of the previous
schema.

Full-grid queries now read physical time vertices directly and interpolate
only the remaining spatial axes, evaluating one time-order branch per query.
This removes repeated generic interpolation without changing the rule:
30,000 random queries across one to three spatial axes, including spatial
extrapolation and both swap conventions, are bit-identical to the prior
implementation. Scalar, empty, NaN and serialized queries were also checked.
The accuracy benefit should be distinguished from query cost: the split
lookup still does more work than the single interpolator used in 0.6.0.

Demo 7's C-table and rotation results were remeasured after this repair.
The component-label regression snapshots now reflect the min/lag table;
a separate two-state OU covariance checks their interpolation error at
the same Sobol points, independently of the package's C implementation.

### Fixed: demo 2 covariance, kernel quadrature and unverified budget values

The first regenerated budget still had large numerical errors. At r=0,t=50,
the old FF QMC values were 4–5% low, FFFF GL10 was about 41% low, and FFFK_01
was about 21% low against a new independent moment reference. The reported
22.6% GL10/GL14 difference was not a bound on the error of either rule.
FFFK also used bare-lambda C instead of the exact two-kernel effective C.

The example κ³ and κ⁴ R-contracted callables now integrate their exponential
pieces analytically over all raw-leg time orderings, declaring partner-time
coincidences for the existing outer-domain splitter. Short intervals use a
stable matrix series; coincident rates use a small matrix exponential.
The helper is serialized by value with external hooks. Neither the package's
Wick rules, diagram combinatorics nor MSR factors are changed by this repair.

`examples/reference/demo2_moments.py` supplies independent polynomial Itô
moment equations for spatially correlated replicated OU noise. Cumulant
scaling in replica count separates FF, FK, FFK4, FFFF, FFFK and the omitted
F³κ⁵ contribution. The budget retains actual L1 evaluations and refines
them against this reference, requiring relative error ≤1e-4 for FFFK/FFK4
and ≤1e-3 for FFFF on every saved nonzero cell. FF uses GL48 and FFFK uses
exact C_eff. Refined stage caches fingerprint their scientific inputs.
On reuse they also recheck the full requested grid, finite values and the
current accuracy target. Missing or duplicated rows, wrong channels and
values that fail a stricter target trigger recomputation; empty output or
a nonfinite reference cannot certify a stage. The saved budget was separately
checked cell by cell and retains its original computed values.

Kernel tests use raw adaptive/QMC integrals and short-time, constant-kernel,
permutation and resonant-rate limits. Separate moment tests check all three
FFFK/FFK4 component pairs, spatial separation and covariance limits.
`make_figures.py` now generates `INTERPRETATION.md` as well as the budget;
numerical documentation checks compare their error claims with the saved
independent references. Quadrature accuracy, omitted cumulants and simulation
conditioning are reported separately.

### Fixed: two externals swept by `integrate_over` were not paired in the kink split

Two externals swept by `integrate_over` are drawn by
`_swept_external_order`, which reads the causal structure.  The
Gauss-Legendre and `nquad` split paired a swept external with an integration
variable but never with a second swept external, and the extra ordering the
split adds did not reach `_swept_external_order` in the first place, so the
kink between the two swept times stayed inside the domain.  It was the largest
error left in the kink split: seven orders of magnitude behind the paired case.

`_kink_pairs` now admits a pair of swept externals unless the full causal
closure already orders them (a chain through a fixed external orders them too,
which `_later_sets` does not follow); `_kink_orientations` drops an orientation
that would make that closure cyclic; and both integrators pass the orderings
the split added to `_swept_external_order`, so the extra edge reaches the
sampler.

Measured on demo 7 (`N = 2`, `integrate_over="all"`, order 2, against the exact
Itô hierarchy), `gauss_legendre`:

| channel | nodes | before | now |
|---|---|---|---|
| `ab = (0, 1)` | 12 | 5.85e-08 | 5.86e-15 |
| `ab = (0, 1)` | 16 | 1.09e-08 | 2.16e-15 |
| `ab = (1, 1)` | 12 | 5.84e-08 | 6.40e-15 |

The one-swept-and-one-pinned channel is unchanged (1.09e-15 at 16 nodes), and
a diagram whose externals all share one time takes the unsplit mapping to the
bit.  The split costs one piece per resolved pair: the `integrate_over="all"`
diagrams go from 2.79 to 3.93 pieces per diagram on demo 7.  Locked by the
rewritten `test_kink_split_external_time.py::test_two_swept_externals_are_a_pair`
and the tightened `test_demo7_space.py::test_integrate_over_at_order_2`
(`1e-6` to `1e-12`).

### Fixed: the C table crossed the kink in the band around the time diagonal

`C(t1, t2)` has a derivative discontinuity across `t1 == t2` whenever the
model's C is declared kinked: white noise puts one in the first derivative,
and since 0.6.0 a `kappa2` with a `|dt|` cusp (the OU family of demos 1 to 5)
puts one in the third.  Every C table interpolated on a tensor-product
`(t1, t2)` grid, whose cells straddle that diagonal, so the table lost four
orders of magnitude within one grid spacing of it.  Exactly-equal times were
routed to a separate 1-D spline of the diagonal and were accurate; the band
`|t1 - t2| ~ h` around it, not the diagonal, carried the table's worst error.
The committed test included off-grid points but none in
`0 < |t1-t2| < h`, so it missed the worst band.

The tables are now tabulated in `(s, u) = (min(t1,t2), |t1-t2|)`, which moves
the kink onto the grid edge `u = 0` where no cell straddles anything.  `(s,u)`
folds the two time orders onto one point, so each stored entry is the
later-first value `L_ab(s,u) = C_ab(s+u, s)`, and the other order is recovered
from the covariance `C_ab(t1,t2) = C_ba(t2,t1)`; where that transposition
cannot supply the earlier-first half (general spatial keys, and a deliberately
one-sided kernel) it is tabulated separately. The translation, rotation and
general spatial builders and their lazy caches take the chart; the legacy
`precompute_C_table` API retains its old layout. `workflow/cache.py::hash_spec` mixes a schema
tag so an old on-disk table in `(t1,t2)` coordinates is never read back.

Measured on the mixing system of `tests/test_offdiagonal_c_tables.py`
(N = 2, `t_min = 0.4`, span 1.9, `n_grid_t = 41`, `h = 0.0475`), worst
relative error against the closed form, by distance from the diagonal:

| offset | before | now |
|---|---|---|
| 0.00 h | 3.1e-08 | 3.1e-08 |
| 0.10 h | 3.7e-03 | 2.3e-07 |
| 0.50 h | 1.1e-02 | 4.1e-07 |
| 1.00 h | 3.5e-03 | 3.4e-08 |
| 8.00 h | 3.9e-06 | 8.3e-08 |

The band error becomes the smooth interior level and the grid convergence is
restored.  Demo 8's order-2 `(0, 1)` channel against its independent moment
hierarchy, `gauss_legendre`: 9.5e-06 at `n_grid_t = 11` and 5.9e-08 at 31
(QMC sits at its own 1e-05 floor at both).  Locked by the new
`test_full_table_stays_accurate_between_grid_nodes_near_the_diagonal` in
`tests/test_offdiagonal_c_tables.py`, which fails on 0.6.0 at 1.1e-02.

The WF4 golden file was re-pinned: all six values moved by 2.5e-04 to 9.6e-04
relative, inside its sampler's own scatter, and order 0 moved because it
integrates C over the swept external times and so reads the band.

### Initial regeneration: demo 2's budget and demo 7's results

Two committed assets were left stale by the 0.6.0 domain cuts and are now
regenerated on the current code.

In the September 12 first pass, `examples/paper_assets/demo2_kappa4/budget.npz` and `budget.md` were rebuilt
by `run_budget.py` then `make_figures.py`.  The cuts make the
Gauss-Legendre channels expensive: the FFK4 and FFFK stages take 7655 s and
4607 s on 28 workers, most of the budget's cost.  The header note that said
the numbers predated the split is gone, and `examples/demo2/INTERPRETATION.md`
no longer said the budget was not re-run. These demo2 values were subsequently
superseded by the independently checked reconstruction described above.

`examples/demo7/results.json` and `shot3d_results.json` were rebuilt by
`space7_run.py` and `space7_shot3d.py`, and the README tables and their
"Limits, measured" paragraphs were updated.  The `integrate_over='all'`
order-2 Gauss-Legendre row moves from 2.2e-09 to 1.2e-15 and the
`integrate_over={'x'}` order-2 row from 1.1e-07 to 8.9e-16.

## 0.6.0 — 2026-09-12

> **Eight inputs that returned a wrong number without an error, seven
> limitations removed, and three demos built to look for them.**  Breaking:
> two points of one R-connected group may not sit at different positions,
> and `ito=False` raises where it returned the Itô value for multiplicative
> noise.  Values move where they were least converged: every Gauss-Legendre
> integral on a cusped κ² (demo 1: 6e-07 at `t = 1` to 3.9e-04 at `t = 15`,
> towards the converged value) and every QMC draw (the Sobol points are
> 64-bit now, which redraws the sequence).  Demo 1's and demo 2's L2
> figures were regenerated; demo 2's error budget was regenerated in the
> following release (see 0.6.1).

> **Five limitations removed, seven more defects that returned a wrong
> number without an error, and three demos built to look for them.**  Each
> limitation was removed with the demo that exercises it, and every demo is
> checked against a reference that shares no code with the package.

### Fixed: two copies of an `equal_time` vertex shared one contraction's time structure

`compute_moment` (`collect_topology=True`, the path `System.expand` takes)
merges contractions whose propagator graphs are isomorphic under a
relabeling of the internal points.  Every leg of a non-local vertex is its
own point, so such a relabeling could move a leg from one equal-time copy of
a vertex to another, and the merged diagram keeps the propagators and the
`equal_time_aliases` of its first contraction: every other contraction was
integrated with the legs of the first one sharing a time.

Measured on two copies of a static `equal_time` κ³ (N = 2, F = 0,
`t_min = 0.4`, one rate), the order-2 six-point function at six distinct
points and six distinct times, against the closed form (the sum over the ten
splits of the six points into two triples):

| (a…f) | 0.5.0 | now | error |
|---|---|---|---|
| (0,1,1,0,1,0) | 4.999688e-03 | 8.392107e-03 | 40.4 % |
| (1,0,1,1,0,1) | 5.884800e-03 | 7.960211e-03 | 26.1 % |

`gauss_legendre`, `nquad` and QMC agreed with each other on the wrong value.
At equal external times with one rate every split has the same integral, so
the defect did not show there; `compute_moment_numerical`, which keeps every
contraction as its own term, was right.

`_canonical_diagram_form` now takes the equal-time leg groups and adds a
clique of `("E", u, v)` edges per group, so only relabelings that map groups
onto groups merge contractions.  Expansions with at most one equal-time copy
are unchanged (demo 4's white-pulse expansions keep 8/3/94, 7/3 and 100
diagrams); the six-point expansion splits into 10 diagrams, one per split.
Locked by 9 cases in `tests/test_equal_time_nonlocal.py`, all of which fail
on 0.5.0.

### Fixed: the copies of one vertex shared a leg structure

The same family, one level down and not limited to `equal_time`.  The
coupling sum of a merged diagram routes the legs between the copies term by
term.  The order-2 six-point function with two κ³ vertices is one
`DiagramTerm` whose 360 terms spread six leg labels over 10 partitions and
36 leg orders.  The leg structure of an `equal_time` vertex (its legs share a
time) and of an `already_R_contracted` one (its legs take their partners'
times and drop their R factors) was read off the vertex instances.
Every term was integrated with the measure of one routing.  `compute_moment`
now splits such a record by leg structure, one `DiagramTerm` per structure
with its own aliases and absorbed pairs
(`perturbation._split_by_instance_structure`); with no such vertex it returns
its input, so existing diagrams are bit-identical.

Order-2 six-point function at distinct external times, N = 3, static
couplings, `gauss_legendre`, against a numpy hand contraction:

| vertices | before | now | error |
|---|---|---|---|
| two `equal_time` copies | 1.4618e-03 | 3.6838e-03 | 60 % |
| `already_R_contracted` + plain | −9.8994e-03 | 2.5995e-01 | 104 % |
| `equal_time` + plain | −4.2364e-04 | 9.7227e-03 | 104 % |

The two fixes are complementary rather than redundant: the canonical form
keeps the equal-time copies apart at the merge, and the split covers what it
does not describe.  Instrumented over both suites, the split is called 160
times and separates structures once, into 20 of them, in the
`already_R_contracted` case.  Locked by `tests/test_callable_vertex_copies.py`
(55 cases, VC0-VC4); 44 fail on 0.5.0.

### Fixed: an off-diagonal C was truncated to its diagonal in the batched lookups

At L0 a model with `diag_C=False` and a spatial table returned the diagonal
from every batched C lookup, while the scalar loop fell back to direct
quadrature per sample and was right.  Measured on an order-1 tadpole
`⟨φ_0⟩ = F_0bc ∫ R C_bc`, which reads C at `b ≠ c`:

| backend | before | now |
|---|---|---|
| `gauss_legendre` | 1.34778e-01 | 1.53391e-01 |
| `qmc_vectorized` | 1.34779e-01 | 1.53391e-01 |
| `qmc_scalar` | 1.53391e-01 | 1.53391e-01 |

1.53391e-01 is the closed form.  Three backends, two answers, no error.  A
cache that keeps off-diagonal entries while offering only
`C_diagonal_batch` is now refused rather than truncated.

### Added: an off-diagonal C by quadrature, without a user closed form

`C = ∫∫ R κ² R + ∫ R σ² R` has off-diagonal component entries when R is
dense (`ExplicitR(iso_R=False)`), when κ² mixes components (`GeneralKappa2`)
or when σ² is a matrix; the tables held `C_aa` only, so
`System.propagators(diag_C=False)` was refused unless the caller supplied a
closed form.  Every table now stores the pairs its model needs, across the
lazy and full-grid builders and the white-noise impulse, and half of each is
filled by `C_ab(t1, t2) = C_ba(t2, t1)` where a numerical check says κ² and
σ² allow it.

Against the exact C, away from the time diagonal: 3.3e-05, 1.8e-07, 1.6e-08,
1.4e-09 at 11, 21, 41 and 81 time points.  Within one grid spacing of the
diagonal the white-noise kink limits it to 5.3e-02 … 3.5e-03, digit for
digit what a diagonal table gives on the same system.  Build cost per table
(N = 2, cells and seconds, diagonal → full): built-in kernels 861/8.1 s →
861/6.4 s under Gauss-Legendre; a user callable κ² 1681/14.2 s → 861/5.7 s.
Locked by `tests/test_offdiagonal_c_tables.py` (42 cases) against the
Lyapunov equation of the Markov embedding.

### Added: the batched integrators evaluate a matrix-valued R

`gauss_legendre` and `qmc_vectorized` raised `NotImplementedError` for a
matrix-valued R (`DiagonalA` with component-dependent rates,
`ExplicitR(iso_R=False)`), so those systems had neither the batched speed nor
Gauss-Legendre's convergence.  Each kept R propagator now becomes one
`(n_samples, N, N)` batch (`PropagatorCache.R_matrix_batch`), whose entries
are selected per component assignment exactly as C's are; the selection, the
C product and the sum over assignments live in one static and one dynamic
helper shared by `qmc_vectorized`, `gauss_legendre` and
`integrate_two_point_qmc`, so the paths cannot drift.  Scalar-R values are
unchanged bit for bit (28 recorded cases).

| case | Gauss-Legendre | `nquad` | `qmc_vectorized` |
|---|---|---|---|
| `DiagonalA` (0.6, 1.6), orders 0-2, 1-, 2-, 3-point | 1.2e-15 | 1e-12 (vs GL) | 2.4e-16 (vs `qmc_scalar`) |
| non-normal dense R, same set | 6.8e-15 | 1e-12 (vs GL) | " |
| demo 5 part B, all channels | 6.6e-16 | 6.2e-16 | 2.6e-5 (2¹³) |

Speed at equal accuracy: demo 5 part B 20.5 s → 1.9 s (`qmc_scalar` →
`qmc_vectorized`, same points), 0.07 s on Gauss-Legendre; a dense-R order-2
two-point channel 4.03 s → 0.05 s.  `integrate_two_point_qmc` also raised on
a matrix R and **returned 0** for a callable coupling, whose
`coupling_array` is a zeros placeholder; both are fixed.  Locked by
`tests/test_matrix_r_batched.py` (53 cases), 47 of which fail on 0.5.0.

### Fixed: an absorbed R with a matrix R

An R absorbed into an `already_R_contracted` vertex stands in for the
Kronecker delta between its partner's component and its leg's.  Only
`diag_R=True` / `iso_R=True` applied that delta, and they apply it to every
R, which zeroes a non-diagonal R's off-diagonal entries; without them the
R-cache guard refused the terms.  On the FK channel of `⟨φ_aφ_b⟩` with a
non-normal drift, `diag_R=True` gives −2.905200629e-02 against the exact
−7.663980261e-03, 280 % off.  `compute_moment` now writes the partner's
index on the absorbed leg, on the absorbed R alone, so
`already_R_contracted` works with a matrix R on every backend (raw versus
absorbed versus closed form agree to 1.6e-15).

### Fixed: `ito=False` returned the Itô value for multiplicative noise

The numerical layer evaluates every R at equal times as 0, so it computes the
Itô SDE whatever `ito=` says.  That is exact for an equal-point R on a local
vertex with one ψ leg, where it and the Stratonovich functional Jacobian
cancel and the package emits neither.  On a vertex with two or more ψ legs
the term is the drift `Θ(0) ∂_b D_ab(φ)`, which no Jacobian cancels and whose
Stratonovich value needs `g`, not `D`.  Both that case and an equal-point R
between two external operators now raise.  Measured on `⟨φ_a⟩` at order 1
with a ψψφ vertex on four integrators: 0.4.x returned 0 on all four, where
the Stratonovich coefficient of that tag is +6.886e-02 and −5.325e-02.

### Added: multiplicative white noise at L1, Itô and Stratonovich

The L1 workflow could not build a local vertex with two response legs, so
multiplicative noise existed only at L0.
`GaussianNoise(sigma2=MultiplicativeImpulse(g0, g1, interpretation))` now
declares white noise with amplitude `g_ak(φ) = g0_ak + g1_akb φ_b`;
`D = g gᵀ` splits into the white noise of C (`D0`), the ψψφ and ψψφφ
vertices (MSR factor `−i²/2! = ½`), and, under `interpretation='stratonovich'`,
the noise-induced drift as a ψ source and a ψφ vertex (factor `−i`), so R and
C stay those of the Itô part and every integrator runs it.  YAML:
`noise.sigma2.type: multiplicative`.  Checked per bookkeeping tag against the
moment hierarchy of the same embedding with the generator in Hörmander form
(`examples/reference/hormander_moments.py`, which never forms the drift):
9.0e-16 on Gauss-Legendre at orders 0-4, 1.0e-6 on `qmc_vectorized`,
2.3e-7 on `nquad` with a matrix R.  Mutating the lowering moves the checked
values by 0.29 to 1.0.

### Added: `nquad` evaluates callable couplings

`method='nquad'` refused a spacetime-dependent coupling.  It and
`make_scipy_integrand` now materialise the coupling at every point the
quadrature visits, as the scalar QMC loop does.  Locked by
`tests/test_nquad_callable_coupling.py` (38 cases): against `gauss_legendre`
to 1e-10 on plain, `equal_time` and `already_R_contracted` κ³ with both
contracts, and demo 4's level A against Campbell's closed form to 1.1e-15.

### Fixed: a callable coupling at two sets of points was refused

`DiagramTerm.build_integrand` raised "a callable coupling at more than one
point set is not supported yet" for two copies of one callable vertex.  Each
`(name, spatial_args)` pair already gets its own symbol, so each copy is now
evaluated at its own points, on every backend.

### Added: callable local couplings at L1 and in YAML

`LocalVertex` took only a tensor; a callable raised `TypeError` from
`System.build_coupling_values`, although L0 has evaluated one since F6.
`LocalVertex(coupling=fn, rank=n)` lowers `fn` with the MSR factor `−i`
applied to its output, under both calling contracts, and YAML takes `rank`
and `coupling_vectorized`.  Checked against the exact hierarchy of a Markov
embedding with a deterministic state `u = e^{−λt}` for
`F_abc(x, t) = F0_abc (1 + βx) e^{−λt}` at N = 2: 1e-15 on `gauss_legendre`,
9e-16 on `nquad`, 3.4e-7 on `qmc_vectorized`.  On 0.5.0, 89 of the 100 cases
fail.

### Improved: `nquad` splits at kinks, and coupling callables can declare kinks

`integrate_moment_nquad` now splits the time domain where the integrand is
kinked, with the recursion `integrate_moment_gauss_legendre` uses.  Adaptive
quadrature used to stop short of its tolerance along a kink:

| `nquad`, demo 5 part A | before | now |
|---|---|---|
| `⟨φ_a φ_b⟩` order 2, three variants, four pairs | 2.2e-8 to 1.4e-7, 1.0-2.7 s | ≤ 8.3e-16, 0.4-0.6 s |
| `⟨φ_0⟩` order 3, mixing | 6.9e-8, 16 s | 2.0e-16, 5.2 s |

A coupling callable can set `has_coincident_time_kinks = True`: it is kinked
wherever two of its time arguments coincide (a `min` over them).
Gauss-Legendre and `nquad` then also split at the pairs of those arguments
the causal structure leaves unordered: the leg times of a raw vertex, the
partner times of an `already_R_contracted` one.  The attribute is read
through `__wrapped__`, so the MSR factor wrapper passes it on.  Demo 4's
kernels declare it; against Campbell's closed form and the hierarchy:

| channel | before, 16 / 32 nodes | now, 12 / 16 nodes | pieces |
|---|---|---|---|
| raw κ³, 3-point function | 5.7e-2 / 1.5e-2 | 2.2e-14 / 2.2e-14 | 6 |
| R-contracted κ⁴, FFK4 | 4.3e-8 / 7.4e-10 | 2.6e-15 / 6.3e-15 | 2 |

QMC is not split: at equal cost the split did not reduce the RMS error over
16 seeds for 2^10 to 2^16 samples.  Values without a declaration are
bit-identical.

### Added: sweeps over n-point observables, and the L1 specs YAML could not express

`Expansion.sweep` unpacked every entry of `component_pairs` as a pair, so a
3- or 4-point observable could be evaluated one tuple at a time and neither
swept nor run from YAML.  The component axis now takes one index per
observable operator, as `component_tuples`; a 2-point sweep's rows and
columns are unchanged.  Demo 4 ships `config_level_a.yaml` and
`config_level_a_4pt.yaml`, which reproduce `level_a.py` for every component
triple and quadruple (4.7e-15 against Campbell's `K_R`, and every sweep row
equals `Expansion.evaluate` exactly).

Four L1 specs had no YAML spelling: a `CustomKernel` on the temporal,
spatial or angular axis, a matrix-valued `ExplicitR`, and a symmetric
`N × N` `ConstantImpulse`.  Each was checked against the equivalent L1
`System` (identical to the last bit) and against an independent reference
(1.3e-15 to 8.4e-16).  What used to pass silently now raises: a component
tuple longer than the observable returned the truncated value, a repeated
tuple or grid value doubled its `totals()` row, a 1-D σ² amplitude was
accepted, a non-symmetric one gave ⟨φ₀(x)φ₁(y)⟩ ≠ ⟨φ₁(x)φ₀(y)⟩,
`already_R_contracted: "false"` was read as `True`, a (3,3,3) coupling at
N = 2 used its (2,2,2) corner, and a second vertex named `F` was dropped.

### Fixed: the Sobol samplers used scipy's 30-bit points

`scipy.stats.qmc.Sobol` defaults to `bits=30`, so its points are multiples of
2⁻³⁰ and the sample mean of a smooth integrand carries a left-Riemann bias
`−(f(1) − f(0)) / 2³¹`: the same value for every seed and every sample
count, a floor rather than a rate.  Demo 7 measured it at 1.38e-9 relative on
the order-1 channel of `⟨φ_0(x,t) φ_1(y,t′)⟩`, where `nquad`,
Gauss-Legendre and the exact hierarchy agree to 3.6e-16; both QMC backends
returned the same wrong digits from 2¹² to 2¹⁸ samples.  The three samplers
in `evaluate.py` now pass `bits=64`, and on that channel both QMC routes
agree with the hierarchy to better than 1e-11.  Every QMC number moves by
about 1e-9.  Locked by `tests/test_qmc_sobol_bits.py` (SB1-SB3), including
an AST guard that requires `bits=64` at every `Sobol` construction in
`src/`.

### Added: demo 6, repeated and static non-local vertices, cubic plus quartic drift

`examples/demo6/`: five structures the package supports and no test had
evaluated: two copies of one non-local vertex, a static (ndarray) non-local
coupling, `m = 5`, `m = 2`, and a quartic local vertex next to a cubic one.
References: the closed form of the model, Campbell's theorem through demo 4,
and a moment hierarchy that observes at several times by freezing each point
after its own time (`solve_multitime`).

| part | worst relative difference |
|---|---|
| two copies, 6-point order 2, all 64 component tuples | 5.9e-16 (static, GL), 9.7e-16 (`equal_time`, GL), 1.2e-05 to 2.6e-05 (QMC) |
| static cumulants with F | 4.3e-16; `F X3 X3` of the five-point function 1.6e-15 |
| quartic `G ψφφφ` next to cubic `F`, orders 2-3 | 1.0e-15 (one external point), 1.4e-13 (equal times) |
| `m = 4`, `m = 5` on four routes | 5.4e-16 to 8.8e-13 |
| an `m = 2` vertex: order 1 reproduces the C of that noise | 0.0 to 2.9e-16 |

`examples/demo4/poisson_level_b_order4.py` adds demo 4's order-4 `F³κ³`
channel (30 diagrams), which demo 2 could only estimate: 7.5e-16 (white
pulses, GL 12) and 2.2e-09 (exponential, GL 24) against the hierarchy at tag
`F³ μ¹`.

### Added: demo 7, observables in space, angle and time

R is local in space and the local vertices act at one point, so an n-point
function at the observation points reduces to a finite-dimensional Itô SDE
there whose noise covariance is the spatial kernel at their separations.  One
reference therefore covers translation, rotation and general homogeneity, any
spatial kernel and any dimension; it adds two-time propagation and integrated
fields for `integrate_over`, both checked against quadrature.

What had no exact check before: the two-time `⟨φ_a(x, t) φ_b(y, t′)⟩` at
`a ≠ b`, `r ≠ 0`, `t ≠ t′`; `LegendreAngular` with more than one coefficient
(every earlier test used `coeffs=[1.0]`, where C does not depend on
direction); `CustomKernel` and `GeneralKappa2` through the package's own C
tables; 3-D positions with a callable κ³; `integrate_over` at N = 2 with
`a ≠ b`; the three-point function at order 2.  Worst relative difference from
the hierarchy: 3.9e-08 (two-time, GL), 2.6e-07 (four Legendre coefficients),
6.5e-08 (Gaussian tables), 4.2e-16 (3-D shot noise, GL), 2.2e-09
(`integrate_over`), 1.1e-07 (three-point at order 2).  No route the package
accepts returned a wrong number.

### Breaking: two points of one R-connected group may not be at different positions

R carries `δ(n − n')`, so the points an R chain joins are one spatial point
with one coordinate.  `DiagramIntegrand._resolve_group_x` took whichever of
the named points a `frozenset` yielded first.  Two externals share a group
only through an external response leg (`("phi_a(x)", "psi_b(y)")`) or a local
vertex with two or more ψ legs (what `MultiplicativeImpulse` lowers to);
`("phi_a(x)", "phi_b(y)")` never does, which is why no demo met it.

Measured at N = 2, γ = (0.8, 1.15), an asymmetric F and a position-dependent
closed-form C, on order 2 of `⟨φ_a(x, 3.0) ψ_b(y, 1.2)⟩` at
`{'x': 0.0, 'y': 0.9}`:

| the pick | value |
|---|---|
| x (0.0) | 1.1042554919e-01 |
| y (0.9) | 4.9123696275e-02 |

A factor 2.25 apart, and each is bit for bit the value of a different
one-position configuration: the request for two positions was answered as a
one-position question.  Renaming the externals to `('u', 'v')` flips which
one comes out, and so does `PYTHONHASHSEED`.  The same program on the same
input returned either number.

At two positions the diagram is a delta function rather than a value, so it
is refused rather than computed (`evaluate.py::_require_one_position_per_group`):
from `_resolve_group_x` (the four `integrate_moment_*` and the
zero-dimensional path), from `integrate_diagrams` before any joblib worker
starts, unconditionally from `integrate_two_point_qmc` (whose own
last-external-wins pick is gone with it), and from
`DiagramIntegrand.evaluate` for the point-keyed `directions` spelling.
`Expansion.evaluate`'s `_guard_single_site` still fires first for a
`MultiplicativeImpulse`, with its noise-specific message.  Locked by
`tests/test_group_position_conflict.py` (33 tests, 20 of which fail before
the fix), including the old resolver restored under a monkeypatch to show
which of the two answers it gave.

### Fixed: a callable `DiagonalA` rate was integrated to O(h²)

`DiagonalA(gamma=callable)` builds `R_aa(t1, t2) = exp(-(Γ_a(t1) − Γ_a(t2)))`
from `Γ_a` sampled on the cache grid, and `Γ_a` was the cumulative trapezoid
rule of `γ_a`: second order in the node spacing, where everything around it is
fourth order or exact.  At the default grid (200 nodes on [0, 100], spacing
0.50) and `γ(t) = [1 + 0.5 sin t, 0.6 + 0.3 cos 2t]`, R was 2.0e-2 off, and
the error fell by 4.0 per halving of the spacing.

`Γ_a` is now the exact integral of the cubic-spline interpolant of `γ_a`
(`CubicSpline(...).antiderivative()`, a piecewise quartic): 4.2e-4 at the
default grid, 3.3e-5 at spacing 0.25, 1.3e-6 at 0.12.  A constant or linear
rate is still integrated exactly.  Locked by T5 and T5b in
`tests/test_diagonal_A_time_dependent.py`, which fail on the previous code
(2.0e-2 against a bound of 1e-3, and a halving ratio of 4.0 against a bound
of 8).

### Added: demo 8, time-dependent coefficients and non-exponential dynamics

`examples/demo8/`: the four routes that leave the constant-diagonal-drift,
exponential-kernel family every earlier demo stays in: a callable `γ(t)`
with unequal, time-varying components and `t_min` of either sign; the
response of a damped oscillator through `ExplicitR`, which oscillates and
changes sign; Matérn-3/2, damped-cosine and Gaussian temporal kernels through
`CustomKernel`; and a white-noise amplitude varying in time through
`CustomImpulse`.  The reference is the moment hierarchy of a finite Markov
embedding whose generator depends on time, integrated with `solve_ivp`
(DOP853, `rtol = 1e-12`), sharing no code with the package.  The Gaussian
kernel has no finite embedding, so it is checked against a closed-form C
(`erf`, against `dblquad` to 1e-11), a hand contraction of the order-1 and
order-2 diagrams, and time-translation invariance.

| route | worst relative difference |
|---|---|
| callable `γ(t)`, exact C, `gauss_legendre` 12 (scalar R) | 2.3e-13 |
| the same on `nquad`, matrix R | 5.9e-09 |
| the same through quadrature tables, `n_grid_t` = 21 / 41 | 9.7e-06 / 1.9e-07 |
| damped-oscillator `ExplicitR`, `gauss_legendre` 12 / 16 / 20 | 6.5e-08 / 5.7e-10 / 1.9e-15 |
| the same at unequal external times, `qmc_vectorized` 2¹⁶ | 3.2e-12 (orders 0-1), 6.2e-08 (order 2) |

The Γ-spline entry above is what this demo found: at the `DiagonalA` default
spacing the callable-rate route was 6.1e-06 off where it is now 1.3e-13 at
spacing 0.0036, and 1.3e-03 with the trapezoid rule it replaced.

### Fixed: the domain is cut at a kink against a fixed external time

`_kink_pairs` splits the domain by ordering two integration times, and all
three kink sources (the ends of a C propagator kinked on its diagonal, the
parents of a multi-ψ vertex, a coupling callable declaring
`has_coincident_time_kinks`) can put the kink between an integration time
and an external pinned at its own time instead.  `u = t*` with `t*` constant
is not an ordering, so it was skipped: with every external at `t_final` the
kink is the domain boundary and nothing is lost; at distinct external times
Gauss-Legendre fell from spectral to algebraic.  The values were right.

The variable's range `[t_min, min(parents)]` is now cut in two at `t*`
(`_extra_bounds`), and the cut is carried twice over: `v ≥ t*` holds every
later variable above `t*` (else the piece's outer variable has a range that
collapses to zero width there), and `v ≤ t*` leaves `min(parents, t*)`, so a
variable parent takes the same cut.  Worst relative difference against the
demos' exact references:

| case | nodes | before | now |
|---|---|---|---|
| demo 7, two-time matrix R + mixing white, order 1 | 8 / 32 | 4.0e-03 / 2.6e-04 | 3.0e-16 / 1.2e-15 |
| demo 6, `F F` at distinct times | 16 | 7.5e-04 | 6.3e-16 |
| demo 6, `F J3 J3` five-point (equal-time vertex) | 8 | 5.5e-02 | 1.0e-12 |
| demo 8 (b), oscillator order-1 `G`, two times | 12 | 4.3e-05 | 4.4e-13 |
| demo 5 part C (MN5), two-time | 12 | 1.5e-03 | 6.4e-16 |
| demo 4, raw exponential 3-point, three external times | 16 | 1.4e-03 | 4.8e-15 |

`nquad` gains where it was reaching its tolerance the hard way (MN5:
1.5e-08 in 22.5 s to 6.4e-16 in 1.0 s).  Cost, one integration per cut: 2.0
to 2.9 pieces per diagram on demos 5-8, 25 against 7 on demo 4's three-leg
case, and none when the externals share a time, where 44 measured values
stay bit-identical.  Locked by `tests/test_kink_split_external_time.py`
(29 cases), which fails on 0.5.0 at 4.5e-03.

### Fixed: a κ² with a `|Δt|` cusp is declared as kinking C

`C = ∫∫ R κ² R` gives `∂²C/∂t1∂t2 = R κ² R`, so a cusped kernel jumps C's
third derivative on its time diagonal, where white noise jumps the first.
Only white noise and a closed form declaring `has_diagonal_kink` were known,
and the OU kernel of demos 1-5 was not.  `_c_has_diagonal_kink` now asks the
question the C quadrature already asks of the kernel (the built-in kernels
declare `has_diagonal_cusp`, any other callable is probed); a Gaussian
kernel is kept out.  A kink against an external swept by `integrate_over` is
ordered rather than cut, which the same work required.

| case | nodes | before | now |
|---|---|---|---|
| demo 8 (c), damped cosine / OU, order 2 on the C table | 12 | 1.4e-05 / 7.5e-06 | 9.5e-07 / 3.8e-07 |
| demo 7, coloured two-time, order 1 / order 2 | 8 | 4.1e-05 / 6.1e-06 | 3.6e-16 / 6.2e-14 |
| demo 7, coloured at one external time, order 2 | 8 | 5.7e-06 | 1.7e-10 |
| demo 7, `integrate_over={'x'}`, order 2 (coloured / white) | 16 | 4.6e-07 / 1.3e-03 | 1.3e-15 / 1.3e-15 |

**Every Gauss-Legendre value on a cusped κ² moves**, towards the converged
one: demo 1's sweep by 6e-07 (t = 1) to 9.7e-03 (t = 100) relative, at 3.03
pieces per diagram and 2.3 times the wall clock
(`examples/demo1/L2/INTEGRATION_ERROR.md` records it cell by cell); demo 2
the same way.  Both demos' L2 figures were regenerated with the split in
place; demo 2's error budget was regenerated in the following development
line (see 0.6.1).  Locked by `tests/test_gl_white_noise_kinks.py`.

### Added: `propagators_from_cache` at the top level

`sft_wick.workflow.__all__` has listed `propagators_from_cache` since the
propagator cache landed, but `sft_wick` did not re-export it, so
`docs/api/index.rst`'s claim that the top-level package re-exports the whole
public API was false for that one name.  It is now `sft_wick.propagators_from_cache`,
and the API index says which names live only on their module
(`workflow.cache.hash_spec`, `load_or_compute`, `workflow.config.run_workflow`).

### Added: a committed file still holds the text its generator writes

Demo 2's error budget, table 1 with its order-1 diagrams, and demo 3's
diagram README are written by scripts.  Editing one by hand looks like an
ordinary documentation change, and the next run of the generator writes the
old text back.  Re-running a generator in a test costs minutes to an hour and
some need caches that are not in the repository, so
`tests/test_generated_docs_in_sync.py` (GD1-GD3) parses the generator
instead: every literal prose run it writes must appear verbatim in its
committed output.  Interpolated values are not checked, only the fixed text
around them; docstrings, strings inside a `raise` and strings handed to a
plotting call are skipped.  GD2 fails on the drift it was written for.

## 0.5.0 — 2026-09-11

> **Eight defects that returned a wrong number without an error, one crash,
> a Gauss-Legendre rule that handles white noise, and two demos checked
> against exact references.**  Breaking: `System.expand` refuses `diag_R` /
> `diag_C` when the system's R or C has off-diagonal entries,
> `System.propagators` refuses `diag_C` with an off-diagonal C and a `t_max`
> above `t_max_cache`, and the L0 evaluators refuse R indices that
> contradict the propagator cache.  Values change for callable non-local
> couplings that are not symmetric in their leg points, for `iso_C=True`
> with N ≥ 2, for `System.t_min ≠ 0`, and on the scalar loops for pinned
> observable labels under `diag_C`.  Demos 1-3 reproduce their stored
> numbers.

### Fixed: a callable non-local coupling was evaluated at one leg order

A non-local vertex enters a diagram through a coupling sum with one term per
assignment of its legs to the fields they contract with.  Each term keeps
every component index on its own leg, `K_{σ(abc)}(σ(y))`, and the symbolic
expansion was right.  The numerical path was not: `DiagramTerm.build_integrand`
passed a callable coupling the legs of the first term only, and every term
read that one tensor with its own permuted indices.  sft-wick computed
`Σ_σ κ_{σ(abc)}(y)` where `Σ_σ κ_{σ(abc)}(σ(y))` is required.

The two agree when the kernel is symmetric under a permutation of its leg
points at fixed component indices, and differ in general otherwise.  A
cumulant is symmetric only when (index, point) pairs are permuted together,
so a multi-component cumulant whose value depends on the leg points was
mis-evaluated, for example a κ³ with spin-2 components.  Constant (ndarray)
couplings were not affected.  Every route that accepts a callable shared the
defect: `gauss_legendre`, `qmc_vectorized`, `qmc_scalar` (also with a
matrix-valued R), `equal_time` vertices, and the zero-dimensional path that
`already_R_contracted` vertices reach.  The 0.3.0 entry below says that
permutations of one point set "are still accepted"; accepting them was this
defect.

Each distinct leg order of a callable symbol in a coupling sum now gets a
symbol of its own (`K@0`, `K@1`, ...; `_split_callable_occurrences` in
`perturbation.py`).  The callable is evaluated at each leg order, and each
term reads the tensor of its own.  `DynamicCouplingPromise` and the
integrators are unchanged.  A callable vertex at two different point sets
(two copies at order ≥ 2) is still refused.

Relative difference from an independent numpy hand contraction (N = 3,
R = Θ, the generic kernel of the new tests; component triple (0,1,2) at
order 1, (a,b) = (0,1) for the order-2 FK channel of `<φ_a(x) φ_b(z)>`):

| case | 0.4.2 | now |
|---|---|---|
| order 1, `gauss_legendre` | 6.4e-01 | 1.3e-15 |
| order 1, `qmc_vectorized` | 1.3 | 1.8e-15 |
| order 1, `qmc_scalar` | 1.3 | 2.8e-15 |
| order 1, `equal_time` | 6.3e-01 | 1.8e-16 |
| order 1, `already_R_contracted` | 6.8e-01 | 1.3e-16 |
| order 2, FK channel | 5.5e-01 | 2.1e-16 |
| order 2, FK channel, cumulant-symmetric kernel | 5.3e-01 | 1.0e-15 |

**Every value computed through a callable non-local coupling that is not
point-symmetric changes.**  The kernels shipped in `examples/` are cumulants
of the form `δ_{a₁…a_m} f(z₁, …, z_m)` with `f` symmetric in the leg points:
a term is non-zero only when every leg carries the same component, and
permuting its legs then permutes the arguments of a symmetric function, so
every leg order needs the same tensor.  They move only by their numerical
asymmetry, measured at representative partner points:

| callable | max relative point asymmetry |
|---|---|
| demo 2 raw κ³; demo 3 raw κ³, κ⁴; `paper_assets/demo2_kappa4` raw κ⁴ | 3.3e-16 |
| demo 3 R-contracted κ³, κ⁴, κ⁵ | 4.4e-15 |
| `paper_assets/demo2_kappa4/k4_R_contracted.py` | 5.3e-06 |
| demo 2 R-contracted κ³ (`k3_R_coupling.py`, `k3_R_contracted.py`) | 2.3e-06 to 9.6e-05 |

The last two are quadratures accurate to about 1e-4 whose integration
coordinates single out one leg.  Recomputed on 0.4.2 and on this version,
with the settings of `examples/paper_assets/demo2_kappa4/run_budget.py`
(the 0.4.2 values reproduce the stored `budget.npz`), demo 2's `ξ₀₁`
channels change by 4.9e-06 (FK, t = 1), 3.2e-06 (FK, t = 15) and 4.0e-07
(F³κ³, t = 15), below their quoted digits and their quadrature errors.
Below t = 1 the FK change grows as the channel shrinks: 2.3e-05 relative at
t = 0.16 and 6.1e-05 (1.3e-11 absolute) at t = 0.1.
Demo 3's level-A values are bit-identical and its level-B channels change
by at most 7e-16.  Demo 1 has no non-local vertex.  No existing test
changed.

Cost: a diagram in which `k` distinct leg orders occur calls the callable
`k` times where it called it once (`k ≤ m!`, 6 for κ³), per sample under
the per-sample contract and per integrand under `coupling_vectorized=True`.
Demo 2's F²κ⁴ channel (`k ≤ 24`) went from 3 to 30 callable calls per
Gauss-Legendre node, and the `ffk4_rc16` stage of
`examples/paper_assets/demo2_kappa4/run_budget.py` from 35 s to 323 s; the
timings in that directory's README predate the change.

Locked by `tests/test_nonlocal_leg_order.py` (71 cases, LO0-LO6): the routes
above, both callable contracts, a single-component field, the L0
`compute_moment` route, and the FK channel with a plain and an
`already_R_contracted` K, each against the hand contraction to 1e-12.  Every
case whose kernel is not point-symmetric also requires the old value to
differ from the correct one by more than 1e-2.  On 0.4.2, 42 cases fail; the
29 that pass are the two point-symmetric controls, the kernel-symmetry checks
and the static-coupling control.

### Fixed: a pinned observable component label was ignored in the diag_C fast path

`DiagramIntegrand.evaluate` routes to `_evaluate_diag_fast` when the cache
model is `iso_R` + `diag_C` and the diagram carries propagator summation
indices.  That path resolved a C propagator's component label with
`_resolve_component(idx_name, {})` -- an empty map -- whenever the label was
not one of those summation indices.  An observable label pinned through
`fixed_indices` (the `a`, `b` of `<φ_a(x) φ_b(y)>`) therefore resolved to
`None`, and the propagator contributed `c_diag.sum()`, every component,
instead of `c_diag[fixed_indices[label]]`.  `_evaluate_general` and the
batched backends merge `fixed_indices` into the index map and were right.

Only the scalar loop reached it: `method='qmc_scalar'`; `method='qmc'`,
because `integrate_moment` auto-selects the scalar loop whenever
`_cache_supports_batch_c(cache)` is false, as it is for a cache holding only
spatial (rotation / translation / general) tables and no legacy `_c_splines`;
`method='nquad'`; a direct `DiagramIntegrand.evaluate`; and
`make_scipy_integrand`.  The L1 `Expansion.evaluate` and YAML `sweep.method`
default is `qmc_vectorized`, which is correct.

Order-2 `<φ_a(x) φ_b(y)>` for a 2-component field with one cubic local vertex
and rotation-homogeneous noise -- 4 of its 6 diagram terms carry `a` or `b`
on a C propagator alongside a summation index.  `t_final = 1`, `n_samples =
2**10`, `seed = 3`, the fixture of `tests/test_diag_fast_component_labels.py`:

| (a, b) | `qmc_scalar` / `qmc`, 0.4.2 | all backends, now | 0.4.2 error |
|---|---|---|---|
| (0, 0) | 2.834850e-02 | 2.416321e-02 | 17.3% |
| (0, 1) | 2.809094e-02 | 1.581779e-02 | 77.6% |
| (1, 1) | 9.275227e-03 | 7.065476e-03 | 31.3% |

`qmc_vectorized` and `gauss_legendre` returned the right column on 0.4.2 and
are unchanged.  The scalar loop now agrees with `qmc_vectorized` to round-off
at the same seed (at most 1 ulp, 1.4e-16 relative, across the three pairs);
`gauss_legendre` and `nquad`, different rules, agree to 1.3e-05 and 9.4e-05
relative.

The same loop read `il` and ignored `ir`.  Under `diag_C`,
`C_{ab} = δ_{ab} c_diag[a]`, so the two legs are tied by a Kronecker delta.
`apply_diagonal(diag_C=True)` merges them into one name, so the L1 pipeline
never exposed this, but a term expanded without `diag_C` and evaluated
against a `diag_C` cache still arrives with `il != ir`, and dropping the
delta keeps a cross-component term that `_evaluate_general` and
`_select_C_batch` both set to zero.  The fast path now resolves both legs and
carries the delta across all four leg classes: two summation axes (equal or
distinct), a summation axis and a pinned label, and two pinned labels.  Of 66
leg pairings on the fixture's terms, 39 disagreed with `_evaluate_general` on
0.4.2 and none do now.

**No published number changes.**  Every script, notebook and YAML under
`examples/` was checked.  The L2 configs use `gauss_legendre` or
`qmc_vectorized` (or omit `method:`, whose default is `qmc_vectorized`);
`examples/demo1/validate_phase5.py` passes `method='qmc_vectorized'`;
`demo3/level_a.py` and `level_b.py` pass `method='gauss_legendre'`;
`paper_assets/demo2_kappa4/run_budget.py` passes `gauss_legendre` at every
call site; `paper_assets/table1/generate_table1.py` is symbolic and
integrates nothing; and every `expansion.evaluate` / `expansion.sweep` call
in `examples/` omits `method`, taking the `qmc_vectorized` default.  The
hand-written `integrate_gl` in `demo1/analysis.ipynb` and
`demo2/analysis.ipynb` is the only caller of `DiagramIntegrand.evaluate`, and
only in its `d == 0` branch: across the 106 (demo 1, orders 2 and 4) and 112
(demo 2, FF orders 2 and 4, FK order 2) integrands those notebooks feed it,
none has zero time-integration variables, so the branch never fires.  The
`method='qmc'` in `demo1/analysis.ipynb` is that notebook's own switch onto
the batched `integrate_two_point_qmc`.

`integrate_moment(method='qmc')` still treats a spatial-table cache as not
batch-capable, so it keeps routing such a cache to the scalar loop.  That is
now a cost in speed only -- measured at 125x on the fixture above (7.59 s vs
0.06 s), with a bit-identical result.  `_cache_supports_batch_c` is a
deliberate handshake on `_c_splines` that custom caches opt into and out of,
and widening it would silently override an explicit opt-out and change which
backend every existing `method='qmc'` caller runs on.  The limitation is
documented on `integrate_moment` instead; pass `method='qmc_vectorized'`
explicitly for spatial caches.

### Fixed: L0 terms evaluated against a cache with the other kind of R

`compute_moment` writes an R propagator with two component indices, `R_{a i}`
(the default), with one repeated index (`diag_R=True`), or with none
(`iso_R=True`).  A `PropagatorCache` holds either a scalar R (`model.iso_R`,
meaning `R_ab = δ_ab R`) or an N × N matrix R.  The evaluators read an R
propagator's indices only when R is a matrix, and they skip the factor of an R
absorbed into an `already_R_contracted` vertex.  Three pairings therefore
returned a wrong number without an error:

| terms | cache R | what the evaluators did |
|---|---|---|
| two indices (the L0 default flags) | scalar | dropped `δ_ab` and summed the leg components |
| no index (`iso_R=True`) | matrix | evaluated each R as `trace(R)`, N times too large for R proportional to the identity |
| an absorbed R with two indices | either | summed the leg components |

Measured on order-1 `<φ_a(x1) φ_b(x2) φ_c(z)>` with N = 3, one non-local ψψψ
vertex with a static random coupling `K = (i/6) A`, R = Θ and every time 1.
Each R factor is 1 on the integration domain, so the correct value is the
coupling contraction `(1/6) Σ_β A[β(0), β(1), β(2)]` = −0.589070:

| pairing | qmc_scalar | gauss_legendre | qmc_vectorized | nquad |
|---|---|---|---|---|
| two indices, scalar R | −2.193211 | −2.193211 | −2.193211 | −2.193211 |
| no index, matrix R | −15.904889 | refused | refused | −15.904889 |
| absorbed, two indices, either R | −2.193211 | −2.193211 | −2.193211 | −2.193211 |

−2.193211 is `Σ_ijk A_ijk` and −15.904889 is 27 times the correct value.
`integrate_two_point_qmc` also returned −2.193211 for the first pairing.

The three pairings now raise `ValueError`, naming the flags that match the
cache, at every entry point that evaluates R: `integrate_diagrams`,
`integrate_moment`, the four `DiagramIntegrand.integrate_moment_*` methods,
`DiagramIntegrand.evaluate` and `integrate_two_point_qmc`.  With those flags
every backend returns the correct value on the example above.

The terms are not repaired.  `dt.apply_diagonal(diag_R=True, iso_R=True)` is
exact for a scalar R (it reproduces `compute_moment(iso_R=True)` term for
term), but only `integrate_diagrams` still holds the terms.  An integrand
carries its coupling already evaluated on the terms' indices (or a callable's
promise), so a repair there would need a second implementation of the delta;
refusing at every entry point keeps one input from succeeding at one entry
point and failing at another.  With one component the delta is 1 and the
trace is R's only entry, so
N = 1 is not checked.  Each integrand's verdict is computed once; the check
left in `DiagramIntegrand.evaluate` measured 0.4% of `qmc_scalar` time on an
order-2 diagram with two C propagators and 4% on a constant integrand of three
R factors.

L1 builds the terms and the cache from the same `LinearOp`, so
`Expansion.evaluate` met these pairings only through `System.expand` flags
that contradict it (`iso_R=False` with `diag_R=False` on a scalar-R system,
`iso_R=True` on a matrix-R system); those raise now as well.

Locked by `tests/test_r_cache_mismatch.py` (80 cases): the correct value for
matching pairs on every backend and entry point, each refusal above, the
order-0 response function `<φ_a ψ_b> = δ_ab`, N = 1, the L1 overrides, and a
refusal on the `n_jobs` path before any joblib worker starts.

### Fixed: two R propagators between the same two points shared one set of component indices

With a matrix-valued R (`iso_R=False`), the matrix-R product loop
(`DiagramIntegrand._evaluate_r_product_general`) resolved each R factor's
component indices by looking the factor up through its
`(spatial_left, spatial_right)` endpoints, and `_find_r_propagator` returned
the **first** R propagator with those endpoints.
Endpoints do not identify an R propagator.  A local vertex with two ψ legs
puts two of them between the same two internal points, carrying different
indices, `R_{i₁i₃}(y₀,y₁) R_{i₂i₄}(y₀,y₁)`; both factors then read the first
match's indices and the diagram evaluated `R[j,l] * R[j,l]` where
`R[j,l] * R[k,m]` is required.  Wrong value, no error.

Only the matrix-R path was affected: the scalar path multiplies R through
`PropagatorCache.R_product`, which carries no indices at all.  Such a diagram
is reachable only from the raw L0 API (`compute_moment` +
`integrate_diagrams`), because the L1/L2 workflow cannot build a local vertex
with two response legs.

Measured on N = 2 with R = Θ·1 (so `diag_R=True` is exact and its expansion is
the reference), a ψφφ (`F`) and a ψψφ (`G`) local vertex, seed-7 normal
couplings, observable `<φ_a(x) φ_b(y)>` at order 2, `qmc_scalar` with 2¹⁰
samples and seed 5.  Eleven diagrams; nine were already right, and the two
carrying a repeated pair — 7 and 9 — were not:

| quantity | 0.4.2 | now | `diag_R=True` |
|---|---|---|---|
| (a,b) = (0,0), diagram 7 | +6.859817e-02 | -6.059916e-03 | -6.059916e-03 |
| (a,b) = (0,0), diagram 9 | +6.859817e-02 | -6.059916e-03 | -6.059916e-03 |
| (a,b) = (0,0), total | +1.752026e-01 | +2.588640e-02 | +2.588640e-02 |
| (a,b) = (0,1), diagram 7 | +1.383161e-01 | +1.159565e-01 | +1.159565e-01 |
| (a,b) = (0,1), diagram 9 | -1.215400e-01 | +1.713376e-02 | +1.713376e-02 |
| (a,b) = (0,1), total | -5.969186e-02 | +5.662232e-02 | +5.662232e-02 |

The totals were wrong by 5.8 and 2.1 times their own magnitude, and the
(0,1) total had the wrong sign.  Against a generic non-symmetric R
(`[[0.7,-0.4],[0.25,1.3]]`, the matrix of the new tests) at fixed times
t = (0.9, 0.8, 0.6, 0.3), the same two diagrams were wrong by factors of
3.46, 3.46, 0.15 and 2.51 relative to a numpy hand contraction.

`_evaluate_r_product_general` now iterates over the R `Propagator` objects
themselves — `_kept_r_propagator_objects`, the object-level counterpart of
`_kept_r_propagators`, which filters absorbed pairs by endpoint exactly as
that helper does — and reads each factor's own `index_left` / `index_right`.
`_find_r_propagator` had no other caller and is gone.  Diagrams without a
repeated pair are unaffected, bit for bit: there the first match *is* the
factor.

Locked by `tests/test_matrix_r_index_and_zero_dim.py`: the repeated pair is
shown to exist and to carry distinct indices, both diagrams are checked
against a numpy contraction written from the diagram's own propagator list
(with the old endpoint-lookup value computed alongside and required to
differ by more than 1e-2), the nine unaffected diagrams are required to match
*both*, and the R = 1 expansion is required to reproduce `diag_R=True` on
every diagram.

### Fixed: a callable coupling with a matrix-valued R crashed on a zero-dimensional diagram

`DiagramIntegrand._evaluate_zero_dimensional` handles a diagram with no
surviving time-integration variable, which is what an
`already_R_contracted` vertex reaches when its absorbed R legs alias onto
fixed external points.  Its dynamic-coupling branch multiplied
`cache.R_time_batch`, which is scalar-only
(`np.vectorize(model.R_time, otypes=[float])`), so a matrix-valued R raised

    ValueError: setting an array element with a sequence.

Nothing refused it first.  `integrate_moment_qmc_vectorized` and
`integrate_moment_gauss_legendre` do carry an explicit `NotImplementedError`
for matrix R, and `integrate_moment_nquad` refuses callable couplings, but
all three checks sit *after* their `n_total == 0` early return, so none of
them fires for a zero-dimensional integrand.

Measured on N = 2 with R = Θ·1, one non-local `already_R_contracted` ψψ
vertex, observable `<φ_a(x₁) φ_b(x₂) φ_c(x₃) ψ_d(x₄)>` at order 1 with
`diag_R=True` (three diagrams, all zero-dimensional), fixed indices
(a,b,c,d) = (0,1,0,0) and t_{x₄} = 0.5:

| coupling | qmc_scalar | qmc_vectorized | gauss_legendre | nquad |
|---|---|---|---|---|
| `K = 0.5·1`, 0.4.2 and now | +2.000000 | +2.000000 | +2.000000 | +2.000000 |
| callable `K`, 0.4.2 | +2.000000 | `ValueError` | `ValueError` | `ValueError` |
| callable `K`, now | +2.000000 | +2.000000 | +2.000000 | +2.000000 |

+2 is the closed form: the δ from the ψ_d leg kills the b diagram (b = 1,
d = 0) and leaves two diagrams of `(K + K) R[0,0] = (0.5 + 0.5) × 1 = 1`
each, the two absorbed R factors contributing 1.

For a matrix R the branch now materialises the coupling for that single
point and goes through the index-aware `DiagramIntegrand.evaluate`, the way
`integrate_moment_qmc`'s own `n_total == 0` branch already did.  Moving the
refusals ahead of the early return was the alternative and is the worse one:
it would turn a confusing exception into a clear one while leaving the value
uncomputable on three backends out of four, when `qmc_scalar` already
computes it — and there is nothing to refuse, since a zero-dimensional
integrand is a single point, which is exactly what the scalar evaluation
handles.  All four backends now agree instead of one computing and three
raising.  The scalar-R zero-dimensional path still goes through
`R_time_batch` and is unchanged.

Locked by `tests/test_matrix_r_index_and_zero_dim.py`: the three diagrams are
shown to be zero-dimensional, each of the four backends is checked against
the closed form per diagram (`[1, 0, 1]`) under both coupling contracts, the
callable is required to match the static tensor exactly, and the scalar-R
path is checked to still agree with itself.

### Fixed: L1 defaults that dropped off-diagonal R and C entries

`System.expand` and `System.propagators` default to `diag_R=True` and
`diag_C=True`, which keep only the diagonal of R and C.  Nothing checked
that the system had no off-diagonal entries, and three inputs have them:

| input | what was dropped |
|---|---|
| a dense drift, `ExplicitR(iso_R=False)` | the off-diagonal entries of R and of C |
| a component-mixing κ², `GeneralKappa2` | the off-diagonal entries of C |
| a component-mixing white noise, a `ConstantImpulse` or `CustomImpulse` matrix | the off-diagonal entries of C |

Measured on a non-normal 2 × 2 drift with exponential noise and a linear
local vertex, against the Lyapunov equation of the Markov embedding:
order-0 `⟨φ_0 φ_1⟩` came out 0 instead of 0.1178; at order 1, `(0,0)` was
0.1123 instead of 0.2028 and `(1,1)` had the wrong sign; at order 2, `(0,0)`
had the wrong sign.  With an off-diagonal `ConstantImpulse` the tabulated C
was 60 % off.

Both methods now look for off-diagonal entries in R, κ² and σ² (from the
spec type where it decides the matter, otherwise by evaluating the callable
at a few time and position pairs) and raise `ValueError` naming the flags
to use: `diag_R=False`, `diag_C=False`, and propagators built with
`diag_C=False`, `c_closed_form_only=True` and a closed form that returns the
full C (the quadrature tables hold `C_aa` only).  With those flags the dense
example matches the Lyapunov reference to 1e-10 at order 0.  A kernel that
cannot be evaluated at the probe's scalar positions is not refused.
`DiagonalA` with distinct rates, separable noise and diagonal white noise
are not refused.

### Fixed: `DiagonalA` with a callable rate

Two defects, each giving a wrong R without an error:

- The choice between a scalar and a matrix R compared the components of
  `gamma` at t = 0 and t = 1 only.  For `gamma(t) = [1, 1 + 0.8 sin πt]` it
  chose a scalar R with component 0's rate: `R_11(2.5, 0.2)` was 0.1003
  instead of 0.0816.  The comparison now uses every node of the spline
  grid, evaluated once and shared with the spline.
- The cumulative-rate spline started at t = 0 whatever `System.t_min` was,
  so R below 0 was extrapolated: `R(0.5, −2)` came out 13 times too large
  and `R(−0.5, −3)` 1.4e3 times.  `DiagonalA` gains `t_min_cache`
  (default 0), `System` extends the grid down to its own `t_min` at the
  same node spacing, and `System.propagators` refuses a `t_max` above
  `t_max_cache`, where the spline would also extrapolate.  The YAML
  `system.linear` block accepts `t_min_cache`.

Both are locked by `tests/test_l1_structure_guards.py` (19 cases), with
the entry above: scipy `quad` of the rate for R, the Lyapunov equation for
the dense system, and a refusal or an acceptance for each structure.

### Fixed: `System.t_min` did not reach the integrators

`Expansion.evaluate`, and so `Expansion.sweep` and `sft-wick run`, called
`integrate_diagrams` without `t_min`, so every time integral started at 0
while the propagators were built from `t_min`.  With stationary noise and a
constant drift, the system started at `t_min = 0.75` and observed at
`T + 0.75` is the system started at 0 and observed at `T`.  With `T = 2`, a
static κ³ gave an order-1 `⟨φ_a φ_b φ_c⟩` 1.26878 times too large; a local F
at order 2, whose tadpoles read `C(τ, τ)` below `t_min`, was 2.2e-4 off with
the closed-form C, which is 0 there.  The L0 route,
`integrate_diagrams(..., t_min=...)`, was right.  `Expansion.evaluate` now
passes the system's `t_min` and refuses a propagator cache built for
another one.  Every shipped example has `t_min = 0`.

Locked by `tests/test_system_t_min.py`: the identity to 1e-10 on
`gauss_legendre` and `qmc_vectorized` for a local and a non-local vertex,
through `sweep`, and the refusal.

### Fixed: `iso_C=True` multiplied every C propagator by N

`compute_moment(..., iso_C=True)` strips the equal component indices of a
C propagator and keeps any index sum in the coupling, so what remains of the
propagator is `c` in `C_ab = δ_ab c`.  Every evaluator multiplied by the
trace `N c` instead.  At N = 2, with isotropic noise and a generic F,
order-0 `⟨φ_0 φ_0⟩` and order-1 `⟨φ_0⟩` came out 2.00000 times too large and
order 2 4.00000 times, on every backend and at L0 and L1; `iso_C=False`, the
L1 default, was right, and N = 1 is unaffected.  `TestFeynmanDiagramQMC` in
`tests/test_deductive_numerics.py` had the factor `N²` written into its
reference values, and `docs/deductive_verification.md` explained it as a
trace over components; both are corrected.

An index-free C now contributes `c` (`_isotropic_C` in `evaluate.py`), and
a C that is not a multiple of the identity raises `ValueError`.  Locked by
`tests/test_iso_c_value.py`: `iso_C=True` equals `iso_C=False` to 1e-10 at
orders 0-2 on four backends, and the refusal.

### Improved: Gauss-Legendre splits the time domain where the integrand is kinked

Tensor-product Gauss-Legendre converges exponentially on a smooth integrand
and as `n^-2` on a kinked one.  Two structures kink a diagram's integrand
inside the domain, where two internal times that the causal structure
leaves unordered cross:

- the two ends of a C propagator, when C has a derivative jump on its time
  diagonal, which white noise gives (`C = ∫ R σ² R`);
- two parents of one time variable, whose upper bound `min(parents)`
  changes branch: a vertex with several ψ legs at one time (an
  `equal_time` non-local vertex, or a local vertex with two ψ legs).

`integrate_moment_gauss_legendre` now integrates each consistent order of
such pairs as its own causal sub-simplex and adds the results, and repeats
the split inside each sub-simplex until no such pair is left: ordering one
pair can place a variable below two others that are still unordered (at
order 4 of demo 5 a single split left 3.1e-3).  Measured on demo 4 against
the exact moment hierarchy, `⟨φ_0(x) φ_1(y)⟩`:

| channel | before, 8 / 16 / 64 nodes | now, 8 / 16 nodes |
|---|---|---|
| white pulses, FF | 4.5e-3 / 1.2e-3 / 7.9e-5 | 2.0e-13 / 7.0e-16 |
| white pulses, FFK4 (`equal_time` κ⁴) | 2.7e-3 at 16 | 5.4e-12 / 1.1e-15 |
| exponential pulses, FF | 1.2e-5 / 8.2e-7 / 3.5e-9 | 6.9e-11 / 1.1e-15 |

The C split applies when the model carries `sigma2`, or when a closed-form
C declares `has_diagonal_kink = True` (demo 4's does: exponential pulses
give C a jump in a higher derivative); the parent split applies whatever
the noise.  The cost is one integration per consistent order: 2 for one
pair, up to `k!` for `k` mutually unordered times.  A kink inside a
coupling callable, such as the `min` over partner times of an R-contracted
cumulant, is invisible to the package and keeps the algebraic rate
(demo 4's exponential FFK4: 3.1e-8 at 16 nodes, 5.4e-10 at 32).  QMC and
`nquad` are unchanged.  Locked by `tests/test_gl_white_noise_kinks.py`:
the orientations are the linear extensions of the causal order, and
white-noise order 2 matches the hierarchy to 1e-11 at 16 nodes.

### Added: demo 4, cumulants asymmetric in points and in components

`examples/demo4/`: compound-Poisson noise whose events drive both
components with their own amplitude (one negative), width and time profile
(white or exponential pulses), so every cumulant is symmetric only under
joint permutations of (component, point) pairs; `F` has no index symmetry.
In demos 1-3 every cumulant was `δ_{a…}` times a point-symmetric function,
the configuration in which the leg-order defect of 0.4.2 gave the right
answer.

- Level A (`F = 0`): the package's 3- and 4-point functions, for every
  component tuple at distinct points and also at unequal times, against
  Campbell's closed form: 1.4e-16 to 1.8e-14 on the R-contracted routes and
  the white-pulse raw route; the raw exponential kernel is kinked where two
  leg times cross, and QMC at 2^18 samples reaches 2.9e-5.
- Level B (`F ≠ 0`): `⟨φ_a(x) φ_b(y)⟩` channel by channel (order 0, FK3, FF,
  FFK4) and the tadpole, against an exact moment hierarchy of the Markov
  process at the observation points (`examples/demo4/poisson_reference.py` on
  `examples/reference/ito_moments.py`, no sft-wick code): at most 1.3e-15
  for white pulses and 5.6e-15 for exponential pulses, except FFK4 at
  4.0e-9 (a kink inside the R-contracted callable).

On `817375f`, before the leg-order fix, level A is 46.8 % off on every route.
Locked by `tests/test_demo4_asymmetric_noise.py`.

### Added: demo 5, white noise on every integrator, additive and multiplicative

`examples/demo5/`, against the exact moment hierarchy of the Markov
embedding (`examples/demo5/white_reference.py`, no sft-wick code):

- Part A (`run.py`): white noise from a `ConstantImpulse` matrix plus
  coloured noise with a Gaussian envelope, a quadratic drift with no index
  symmetry, `t_min = 0.5`, in three variants that take the
  `diag_C=False`, `diag_C=True` (the diag-C fast path of the scalar loops)
  and `iso_C=True` paths.  `⟨φ_a⟩` at orders 1 and 3, `⟨φ_a φ_b⟩` at orders
  0, 2, 4 and `⟨φ_a φ_b φ_c⟩` at order 1.  Gauss-Legendre agrees to 1.4e-15
  at every order up to 4, `nquad` (the tadpole) to 4.3e-16, the QMC routes
  to their sampling error (1e-7 to 7e-6).
- Part B (`multiplicative.py`): multiplicative white noise at L0, with the
  local ψψφ and ψψφφ vertices the L1 workflow cannot build (factor
  `−i²/2! = ½`, checked by making an order-1 ψψ vertex reproduce C).
  Scalar R on Gauss-Legendre agrees to 2.4e-16; matrix R, where the F and
  ψψφ vertices share two R propagators, to 6.2e-16 on `nquad` and 2.6e-5 on
  `qmc_scalar`.

Run on older code, the same scripts fail where the fixes of 2026-09-11
say they should: on `facf556` the `iso_C` variant is off by factors 2, 4
and 8 on every integrator and Gauss-Legendre by 5.8e-4 to 3.5e-3 (`t_min`,
the white-noise kink); on `3cc7115` the diagonal variant's scalar loops
are 64 % off at order 2 (diag-C fast path) and part B's matrix-R channels
2.7e-3 to 6.7e-2 (repeated R pair).  Locked by
`tests/test_demo5_white_noise.py`.

### Performance: a per-sample callable coupling is called once per sample

`DynamicCouplingPromise.evaluate_at_batch`, which `qmc_vectorized` and
`gauss_legendre` use for a callable non-local coupling, called a callable
under the per-sample contract `n_samples + 1` times per integrand and leg
order: once at sample 0 to learn the shape of the contracted coupling, then
at every sample to build the tensor stack.  When the batched contraction fell
back to the per-sample loop, that loop called it again for samples 1 to
`n_samples - 1`.  The stack is now built first, and the shape probe, the
batched contraction and the fallback loop read it: `n_samples` calls per
integrand and leg order on both paths.

Calls of a recording callable on the N = 2 system of WF9 and WF10 (order-2
FK channel, 2 integrands with 3 leg orders each):

| route | before | after |
|---|---|---|
| `qmc_vectorized` (256 samples), `gauss_legendre` (`n_gauss=4`) | 1542 | 1536 |
| the same, batched contraction forced to fall back | 3072 | 1536 |
| `qmc_scalar` (32 samples), through `evaluate_at` | 192 | 192 |

The FK totals are bit-identical before and after on `qmc_vectorized` and
`gauss_legendre`, with and without the fallback, and on `qmc_scalar` and
`qmc`.  A callable declared with `coupling_vectorized=True` is unaffected.

Locked by WF10 in `tests/test_dynamic_coupling.py` (5 cases): the calls per
`evaluate_at_batch` or `evaluate_at` call, the route each method takes, and,
on the forced fallback, the FK total against the batched one to 1e-12.  On
the previous code the four `evaluate_at_batch` cases fail and the
`qmc_scalar` case passes; with the fallback loop reading sample `s - 1` for
sample `s`, the two fallback cases fail.

### Tests: the argument shapes a callable coupling receives

`tests/test_dynamic_coupling.py` gains WF9 (18 cases), which records the
arguments a callable non-local coupling receives and checks their shapes on
`gauss_legendre`, `qmc_vectorized` and `qmc_scalar`, for scalar, 2-D and 3-D
leg positions under both contracts.  Positions arrive as `(m,)` or `(m, d)`
per sample and as `(m, n_samples)` or `(m, n_samples, d)` under
`coupling_vectorized=True`; times as `(m,)` or `(m, n_samples)`.
`qmc_scalar` passes a vectorised callable one sample at a time, so there
`n_samples` is 1.  Each case also requires the FK total to equal the
per-sample, scalar-position total of the same method to 1e-12.

WF8, which runs the same system with 3-D positions, checks only for a finite
nonzero total.  It still passed with the positions reduced to their norms
before they reached the callable; WF9 fails in its 12 vector cases under that
change, and in the same 12 when the positions are zeroed.

### Documentation: stale demo 1 figures

- `examples/demo1/L2/INTEGRATION_ERROR.md` quoted the Gauss-Legendre 18-node
  values of two order-4 cells (`(0,0)`, r = 0 and `(1,1)`, r = 0.5, both at
  t = 15) in its "converged (GL24)" column; the 24-node values are 3.9691e-5
  and 2.2166e-5, not 3.9721e-5 and 2.2175e-5.  The conclusions drawn from
  them (74 % and 42 % errors of the seed-42 draw) do not change.
- `examples/demo1/validate_wrapper.py` compared its time-integrated sweep
  with a table copied from `validate_phase5.py` when that script still
  reported time-integrated moments; since the QMC sampling fixes of
  0.3.0-0.4.x its order-2 and order-4 rows differed by 7.6e-4 to 5.4e-3
  relative, and the script failed.  The table now holds this script's own
  output; the L1-versus-raw-API equivalence is tested by
  `tests/test_workflow.py::test_WF4_end_to_end_matches_validate_phase5`.
- `docs/user_guide/workflow.rst` referred to a `ConstantMatrix` linear
  operator that does not exist; a dense R is `ExplicitR(iso_R=False)`.

## 0.4.2 — 2026-09-03

> **One `src/` fix, a documentation catch-up, and the test suite's tolerances
> made honest.**  An earlier draft of this entry said "paper assets and
> documentation only — no `src/` changes"; a readiness audit then found a
> silent factor-2 wrong answer on an exported API, so that framing was wrong
> and has been corrected rather than kept.  Four of the items below are
> corrections to claims made in the 0.4.0 and 0.4.1 entries themselves.

### Fixed: `compute_moment_numerical` bypassed the coincident-label guard

0.4.0 made two external operators sharing a spatial label a `ValueError` and
described the coverage as "both `System.expand` (L1) and `compute_moment`
(L0)".  **L0 exports two engines.**  `compute_moment_numerical` — the nauty
path documented as the faster drop-in for order ≥ 4 — was left open, and it
drives the identical label-keyed `wick_contract_spatial`, so it inherited the
identical collapse.

Measured on a demo2-shaped system (N=2, one cubic vertex, `lam=0.05`,
`sigma_t=0.3`, `sigma_x=1.0`, t_f = 3.48, both externals at position 0, QMC
2¹⁴ seed 42):

| call | observable | result |
|---|---|---|
| `compute_moment` | `('x','y')` | 1.47018966e-03 |
| `compute_moment_numerical` | `('x','y')` | 1.47018966e-03 — agree bit-for-bit |
| `compute_moment_numerical` | `('x','x')` | 7.35094832e-04 — **ratio exactly 0.500000** |
| `compute_moment` | `('x','x')` | `ValueError` |

So the silently-wrong factor 2 that the 0.4.0 entry calls out was still
reachable through the engine the documentation advertised as "same output", and it
continued past the guard to order 4 (7560 `DiagramTerm`s).  `docs/deductive_
verification.md` check C1 cross-validates the two engines as "2 independent
symbolic engines produce equal integrated totals", which this input class made
vacuous.

Exposure was narrow, which is why it survived: L1 and L2 never call the
numerical engine, and it appears in one test file and no example or paper
asset.  The rationale was already written into
`test_CE1_repeated_external_label_is_refused_at_L0` — "the raw API is a public
entry point and an L1 check would not protect it" — and simply was not
followed through to the second engine.

Locked by `CE4`, including a parametrised check that both engines guard
identically at orders 0, 1 and 2: whatever they do, they must do the same
thing, or C1 cross-validates two engines that disagree about what is legal.

### Fixed: the Table 1 asset did not run on the version the paper cites

`examples/paper_assets/table1/generate_table1.py` built its odd-order
observable as `phi_a(x) phi_b(x) phi_c(y)` — two externals sharing the
spatial label `x`.  0.4.0 made exactly that spelling a `ValueError` at
interacting orders, so from 0.4.0 onward the script raised at orders 1
and 3.  A referee re-running the asset against the cited version got a
traceback, which is the worst available outcome for a reproducibility
artefact.

**The counts change, and the published ones were the artefact.**  With a
distinct label per external the script runs and gives

| Order | Diagrams, submitted | Diagrams, correct | Reduction |
|---|---|---|---|
| 1 | 4 | **6** | 4 → **2** |
| 2 | 6 | 6 (unchanged — `m = 2` already used distinct labels) | 18 |
| 3 | 75 | **80** | 139 → **130** |

This is the collapse this repository documented one release earlier and
did not then apply to its own asset: same-label externals lose the sum
over assignments of externals to legs.  The six order-1 diagrams are
three pairs, one per choice of which external carries the `R` leg, each
pair splitting into a tree and a tadpole — visible directly in the
regenerated `order1_diagrams.md`, where the trees carry
`F_{i₀i₁i₂} + F_{i₁i₀i₂}` and the tadpoles `F_{i₀i₁i₂}` alone.  `3n+m`
and the raw pairing count `(3n{+}m{-}1)!!` are unchanged: `m` is 3 either
way.

The script no longer hardcodes any count.  Its `assert len(dts) == 4` is
what turned a recount into a traceback, and the prose in the generated
`order1_diagrams.md` said "exactly four" as a literal; both now derive
from the data, so the next such change shows up as a different number
rather than a crash.  All six standalone TikZ diagrams and the matplotlib
rendering are regenerated; the four tracked `_standalone.pdf` files had
been compiled from the superseded four-diagram version.

**Wall-clock is not comparable across this change** and is still to be
re-measured on an idle machine: the order-3 cell now times an 80-diagram
enumeration rather than a 75-diagram one, so the column changed meaning
as well as magnitude.

### Fixed: every `approx(rel=)` site now states its `abs=` (issue #5)

`pytest.approx(x, rel=r)` compares against `max(r * expected, abs)` with `abs`
defaulting to **1e-12**, so wherever the compared quantity is small the floor —
not the written `rel` — is what is enforced.  0.4.1 fixed the two sites where
that made an assertion unable to fail, and documented 22 more as weakened.
This release closes the rest.

**78 sites** wrote an explicit `rel` without an `abs`.  They split cleanly, and
the split is what made this safe to do at all:

* **56 were no-ops.**  At a site whose `rel` is already in force,
  `max(rel * expected, 0.0) == rel * expected`, so adding `abs=0.0` cannot
  change behaviour — only make the intent explicit.
* **22 were the weakened ones**, and adding `abs=0.0` tightens them, by up to
  nine orders of magnitude.

**All 22 pass at the tolerance they were written to assert, and none needed a
loosened `rel` or a reverse-engineered `abs`.**  Every one was measured rather
than merely re-run: the largest true deviation found anywhere in the group is
9.0e-16, and several sites are bit-identical.  The two most extreme — the
`t_tilde` auto-dispatch repairs in `test_demo3_shot_noise.py`, written
`rel=1e-15` but enforcing 8.1e-07 — turn out to compare a value against its own
source, because the `auto` path assigns the quadrature result elementwise; they
agree exactly, to 0.0.  So the floor had been masking agreement far better than
the tests claimed, not covering up disagreement.

**The measured result.**  Before and after, over the same 207 runtime sites:

| bucket | 0.4.1 | 0.4.2 |
|---|---|---|
| `rel` genuinely in force | 108 | 52 |
| explicit `abs=` | 75 | **153** (120 of them `abs=0.0`) |
| compares against 0 on the default floor | 2 | 2 |
| **weakened by the floor** | **22** | **0** |
| vacuous | 0 | 0 |
| total | 207 | 207 |

The 56 no-ops moved from "rel in force" to "explicit `abs=`" without changing
what they enforce; the 22 moved from "weakened" to enforcing what they say.

Three sites in `test_spectral.py` keep a deliberate non-zero `abs`, and
correctly: `R` is structurally exactly zero for acausal and equal times, and
`C` vanishes at `t1 = t2 = 0`, so a relative tolerance is meaningless there.
That is the distinction the guard below is meant to force an author to make.

**A guard, so this cannot come back**: `test_every_approx_rel_site_states_its
_abs` walks the suite's AST and fails on any `approx(..., rel=...)` with no
`abs=`.  It is static by design — the runtime auditor can only tell you a site
was weakened after the magnitudes happened to come out small, whereas the AST
rule refuses the site before it ever runs.

### Fixed: the 0.4.1 approx breakdown did not add up

0.4.1's CHANGELOG and catalogue both read "Over 207 runtime sites: 108 have
`rel` genuinely in force, 29 pass an explicit `abs=`, 15 compare against 0.0,
22 are weakened, and 2 were vacuous" — presented as an exhaustive partition.
**108 + 29 + 15 + 22 + 2 = 176.**  Thirty-one sites were in no category.

The 29 and 15 were carried over from an earlier measurement of a 173-site
suite, where `15 + 29 + 108 + 21 = 173` does sum; the total was updated to 207
and the weakened count to 22, and those two were not.  `tests/test_catalog_
current.py` could not catch it: the figures were hardcoded prose inside
`tools/gen_test_catalog.py`, so the test regenerated the catalogue from the
same literals it then compared against — structurally incapable of noticing
they were wrong.

**The fix is not a corrected number but a corrected mechanism.**
`tools/approx_audit.py` now writes `docs/verification/approx_audit_summary
.json` during the instrumented run, asserting as it does so that the buckets
partition the total exactly, and `tools/gen_test_catalog.py` renders the
paragraph from that file.  The bucket priority is recorded in the JSON, so the
partition is reproducible rather than a convention someone has to guess.  A
figure that describes a measurement is now read from that measurement.

### Fixed: a published timing could not notice that its subject had changed

The README's row for `examples/demo1/L2/config.yaml` read "4.6 min (28
workers), 32768 samples".  That number was never wrong — nobody's machine got
slower.  Its **subject** was replaced: the config became `method:
gauss_legendre` with `n_gauss: 24`, under which a sample count is not a
meaningful quantity at all.  A caveat about hardware would not have caught it,
because the hardware was not the variable.

The measurements cannot be automated — the table spans three machines and
needs an idle one, so generating it at build time would replace three careful
numbers with one machine-dependent number.  The **invalidation** can be.
`test_readme_timings_still_describe_these_configs` pins the integrator, its
resolution and the worker count for every config the table names, so a config
change turns a silently-wrong row into a failure that names itself.

### Fixed: two figures misquoted from demo 3's own stored results

- The level-A `m = 4` agreement is **6.6e-16**, the maximum over the nine
  times in `level_a_results.npz`.  0.4.0 and 0.4.1 quoted **4.9e-16**,
  which is the `t = 3` cell.  `INTERPRETATION.md` had the maximum right;
  the CHANGELOG did not.
- The ETDRK2 `O(Δt²)` convergence ratios span **4.0–6.9**, not the
  4.0–5.1 quoted in both files.  The 6.9 cell is not anomalous
  convergence: its Δt differences are ~1e-8, so the ratio there is
  noise-dominated.  Quoting a range that excluded it made the agreement
  look tidier than the data.

Neither affects a released number, and both were found by re-deriving the
figures from the `.npz` files rather than by reading the prose.

## 0.4.1 — 2026-09-02

> **Test-suite hardening only — no `src/` changes, no behaviour change.**  The
> version the CPC paper cites, because it is the first in which
> `docs/verification/catalog.rst` states the tolerances the suite *enforces*
> rather than the ones it was written to assert.

### Fixed: two assertions that could not fail

`pytest.approx(x, rel=...)` compares against `max(rel * expected, abs)` with
`abs` defaulting to **1e-12**, so wherever the compared quantity is small the
floor — not the written `rel` — is enforced, and below 1e-12 the assertion
passes for any value.  This cannot be read off the source, since it depends on
each site's runtime magnitude; `tools/approx_audit.py` measures it during a
normal pytest run.

Over 207 runtime sites: 108 have `rel` genuinely in force, 75 pass an explicit
`abs=`, 2 compare against `0.0` on the default floor (where the floor is the
intent), **22 are weakened** (enforcing 2.0e-12 to 8.1e-07 relative), and
**2 were vacuous**:

> **Corrected in 0.4.2.**  As shipped, this paragraph read "29 pass an explicit
> `abs=`, 15 compare against `0.0`", which does not add up: 108 + 29 + 15 + 22
> + 2 = 176 against a stated total of 207.  Those two figures were carried over
> from an earlier 173-site measurement.  The numbers above are the re-measured
> partition; 108, 22, the 2.0e-12–8.1e-07 range and the two vacuous sites were
> always right.  See the 0.4.2 entry for why the catalogue no longer restates
> figures it can read.


- `tests/test_demo3_shot_noise.py` — the dispatch check at the removable
  singularity `γ = 1/σ_t`, called "the boundary check in its strongest form" by
  its own docstring.  At the `(m=3, t'=[40, 2, 0.05])` corner the compared value
  is 2.3e-37, so the floor enforced **4.3e+24** relative.  **Shipped in v0.4.0.**
- `tests/test_closed_form_dispatch_boundaries.py` — compared `Φ(1e4, 9.9e3)`
  against `np.exp(-γ·100)·0.0 + Φ(1e4, 9.9e3)`; that first term is exactly
  `0.0`, so both sides were the same expression.

**In both the code was independently verified correct and only the test was
empty**: the two dispatch branches agree to rel 0.0 at all eight corners, and
`Φ` is T-independent to 2e-16 in the regime the second covered — a regime no
shipped example reaches (largest `γt` across all configs is 100, against 1e4).
No released number is affected; the suite's *coverage* was overstated, not its
results.

Replacements are checks, not relaxations: the deep-domain assertion now tests
T-independence of `Φ(T, T−d)` (a mutant with 1e-9 drift per decade of T fails it
and passed the old form), and the boundary check enforces `rel=1e-12, abs=0.0`,
which the branches meet at rel 0.0.

### Changed

- `docs/verification/catalog.rst` carries the measured breakdown above and
  records that the two vacuous rows were **not enforced in v0.4.0**.
- `tools/approx_audit.py` is kept in the repository rather than treated as a
  one-off: static review had missed this class entirely, and this makes it a
  measurement that can be re-run in one command.

The 22 weakened sites are documented but not fixed, and no lint guard yet stops
new bare `rel=` sites appearing — tracked in issue #5.

## 0.4.0 — 2026-09-02

> **This is the version the CPC paper refers to.**  v0.3.0 predates the
> exact order-4 calculation in demo 2 and all of demo 3, so neither
> demo's headline result can be reproduced against it.
>
> **One breaking change**: external operators may no longer share a
> spatial label — `("phi_a(x)", "phi_b(x)")` now raises rather than
> silently losing pairing multiplicity.  Write distinct labels with
> equal positions instead: `("phi_a(x)", "phi_b(y)")` with
> `positions={"x": 0.0, "y": 0.0}`.  See the section below for why a
> multiplicity factor would have been wrong rather than merely absent.
>
> **Numbers move** in demo 1's L2 sweep (its order-4 channel was resting
> on a single QMC seed) and, at the 1e-9 level and below, wherever the
> `auto` C-quadrature dispatch previously raced.  Both are sized in place.

### Propagator-indexed dynamic couplings (demo2's order-4 F³κ³)

- **`DynamicCouplingPromise.evaluate_at_batch` no longer refuses a
  contraction that leaves a propagator index.**  A callable κ^(m) whose
  contraction against the surrounding Wick structure leaves a surviving
  component index on a C propagator used to raise
  `NotImplementedError: Dynamic coupling with propagator-indexed
  contraction is not yet supported`.  It now returns
  `(n_samples,) + prop_shape`, and the integrators contract it against
  the C-propagator product one index assignment at a time — the same
  `np.ndindex(prop_shape)` loop the static-tensor branch has always
  used, with the scalar coefficient promoted to a per-sample array.
  The contraction is centralised in
  `DiagramIntegrand._dynamic_values`, called from all three consumers
  (QMC-vectorised, Gauss-Legendre, and the zero-dimensional path); the
  batched C lookup is hoisted out of the index loop, since it does not
  depend on the component assignment.

  This is what blocked the exact evaluation of demo2's order-4 F³κ³
  channel: 30 diagrams, of which those with `propagator_indices ==
  (('i_0', 2),)` hit the guard.

  **Size of the change:** zero on every path that worked before — a
  fully contracted (scalar) callable takes the same branch and returns
  bit-identical numbers; the 236 tests of the integrator suite are
  unchanged.  What was previously an exception is now a number.

  Locked by `DC1` in `tests/test_dynamic_coupling.py`, which replaces
  the test that pinned the `NotImplementedError`: a callable that
  ignores its arguments and returns a constant tensor is
  mathematically the static tensor, so the two routes are compared
  diagram-by-diagram on the order-4 F³κ³ set and agree to `rel=1e-12`
  (observed: exact equality).  A companion test asserts the comparison
  is not vacuous — at least one prop-indexed diagram integrates to a
  non-negligible value.

### Breaking: external operators may no longer share a spatial label

`("phi_a(x)", "phi_b(x)")` — the natural spelling of an equal-point
correlator — now raises `ValueError` at both `System.expand` (L1) and
`compute_moment` (L0), **at interacting orders only**.  It previously
returned a number, correct at order 0 and wrong once vertices are
present.

**Order 0 is exempt, and the exemption is measured rather than
assumed.**  The free-theory contraction keeps every routing at
coincident labels — `⟨φ_a φ_b φ_c φ_d⟩` gives
`C_ab C_cd + C_ac C_bd + C_ad C_bc` either way, and the scalar
`⟨ψφφφ⟩` gives `3 R(x,x) C(x,x)` against the distinct-label
`R(x,w)C(y,z) + R(y,w)C(x,z) + R(z,w)C(x,y)`.  The loss appears only
once vertices exist, when the downstream diagram-isomorphism pass merges
topologies that the external collapse has made isomorphic.  The
equal-point Itô tests depend on the exemption and are unchanged; a new
test pins it (`CE2`).

The refusal at order ≥ 1 is deliberately conservative — demo2's order-2
`FK` channel is *right* at coincident labels while its `F` channel is
low by 2, and distinguishing them per-channel is exactly the analysis
this refusal exists to avoid.  One test moved: `test_latex_local_coupling`
used a repeated label at order 1 while testing symbol flags, and now
uses distinct ones.

**Migration:** give each external a distinct label and set them to the
same point through `positions`:

```python
system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2])
expansion.evaluate(props, positions={"x": 0.0, "y": 0.0}, ...)
```

Coincident external *points* are, and always were, fully supported;
it is the shared *label* that is not.  Demo 1 and demo 2 use distinct
labels and are unaffected.

**Correction (0.4.2):** this entry originally continued "and the paper
all use distinct labels, so nothing in this repository changes."  That
was wrong, and wrong because the sweep behind it was incomplete:
`examples/paper_assets/table1/generate_table1.py` builds its odd-order
observable as `phi_a(x) phi_b(x) phi_c(y)` and therefore raised on the
very version this entry ships in.  See the 0.4.2 entry for the recount
it forced.

**What was wrong.**  The spatial Wick engine enumerates topologies keyed
by spatial label and recovers the operator-level count with a
multiplicity that excludes observable points, assuming one operator
each.  Measured on demo2's system at t = 3.48 with both externals at
position 0: order 0 agreed exactly, the order-2 `F` channel was **low by
a factor 2** (1.838e-04 against 3.677e-04), and the order-2 `FK` channel
was exactly right.  Not even uniform between channels.

**Why it is refused rather than repaired.**  What the collapse loses is
not a factor but a *sum over the assignments of external operators to
legs*, each with its own component-index routing:

```
("phi_a(x)", "phi_b(y)", "phi_c(z)") -> K_abc + K_acb + K_bac + K_bca + K_cab + K_cba
("phi_a(x)", "phi_b(x)", "phi_c(x)") -> K_abc
```

`6 · K_abc` reproduces that sum only for a K symmetric under all six
permutations.  demo2's κ³ is component-diagonal and the two-point F case
had `a = b`, which is why the observed discrepancies were clean integers
and why the bug survived — a multiplicity "fix" would have replaced a
silent wrong answer with a subtler one, since the API accepts arbitrary
bare tensors.  Supporting the spelling properly means keeping external
operators distinguishable through the label-keyed enumeration in
`wick.wick_contract_spatial`; that is a change to the multiplicity core
and is left as follow-up work.  The reasoning is recorded in
`tests/test_coincident_external_labels.py` so it is not "optimised" back
later.

### Fixed: single-component systems with a callable coupling

`n_components = 1` plus a callable (dynamic) coupling raised
`ValueError: input operand has more dimensions than allowed by the axis
remapping` on every integrator.  With one component there is nothing to
sum over, so the simplifier elides every component index and
`_eval_symbolic_batched` took its `not expr.indices` branch, returning
the coupling array whole — `(n_samples, 1, 1, 1)` where `(n_samples,)`
was expected.  The scalar `_eval_symbolic` had always tolerated "a
size-1 array in any shape"; the batched path now does too.  A static
tensor at `N = 1` was unaffected, as was `N ≥ 2`.  Locked by `DC3`
(both integrators, dynamic against static to `rel=1e-12`).

**Verification that neither of the two changes above moved a published
number.**  Both are expected to be exactly inert for demo 1: one adds a
refusal on a spelling demo 1 does not use, the other repairs an `N = 1`
branch demo 1 never enters.  Measured rather than assumed —
`examples/demo1/L2/config.yaml` re-run before and after, all
**2016 / 2016** `(y, t_final, a, b, order)` totals **bit-identical**,
max relative change **0.0** against the v0.3.0 comparison's 3.3e-14
floating-point baseline.

### Demo 2 hardening

Every item below is a number that moved, with its measured size.

- **The order-4 F³κ³ channel of demo 2 is now computed, not estimated.**
  It was the blocked case above.  With the R-contracted κ³ the effective
  time dimension of its 30 diagrams is 3 (it is 6 with the raw kernel —
  measured, via `analyze_spatial().time_integration_vars`), so a
  tensor-product Gauss-Legendre rule does it in seconds per grid point.

  The equal-time estimate it replaces was built by collapsing κ³ to a
  constant `24 α λ² σ_t²` and calibrating on FK at order 2.  That
  calibration ratio is 0.42–0.64 for the FK-type partner-time
  configuration `(t′, s, s)` but 1.08–1.50 for three distinct partner
  times — which is what the F³κ³ diagrams have — so it was a
  factor-of-≈2 quantity.

  **Measured effect.**  At `t = 15, r = 0`, `ξ₀₁`: the exact channel is
  **5.47e-05** (GL10; 5.41e-05 at GL14) against the old estimate's
  8.10e-05.  Over the 18 times at `r = 0` this takes χ² of
  (simulation − theory) from **340.7 to 44.2**, the mean pull from
  **+3.31 to +1.09**, and the largest residual from **9.36e-05 to
  3.88e-05** (7.0σ → 2.9σ).

  **It also contributes where the 0.3.0 budget assumed it could not.**
  `F³κ³` was taken to vanish for `ξ₀₀` and `ξ₁₁` by `φ₁ → −φ₁` parity.
  That parity is broken by the deformation itself (`η̃ = η + α(η² − λ)`
  is not odd in `η` — which is the entire reason `ξ₀₁ ≠ 0`); the order-2
  FK channel does vanish there, but for the narrower reason that its two
  diagrams' index structure does.  Computed for all three pairs rather
  than assumed: **4.53e-06 for `ξ₀₀`** and 4.34e-06 for `ξ₁₁` at
  `t = 5.44` (converged to 0.1 %), i.e. 13 % of the `ξ₀₀` residual.

- **The residual's perturbative order is now measured, not asserted.**
  Scaling the quadratic drift by `s` scales each channel by a known
  power of `s`.  At `t = 15, r = 0`, all at dt = 0.02 so the step-size
  bias cancels out of the `s`-dependence: `s = 0.5` (20M realisations)
  gives a residual of 9.83e-06 ± 2.74e-06 and `s = 1` (2M) gives
  7.28e-05 ± 1.00e-05.  Those fit a **pure `s³` law with χ² = 0.06 for
  1 dof** — `c₃ = 7.38e-05 ± 0.91e-05` — so the residual *is* an order-4
  effect, deduced from the simulation and the order-2 channel alone.
  The exactly-computed `F³κ³` then agrees at **2.1σ** (1.8σ against the
  un-extrapolated dt = 0.02 residual; the spread between the two is
  whether the step-size bias is extrapolated away, and that bias is
  itself only a 1.1σ measurement).

  `s = 1.5` is excluded from the fit and kept as a measured boundary of
  validity: its residual *exceeds* the leading term (1.04e-03 against
  5.16e-04) and **4.5 % of trajectories blow up** against 6e-05 at
  `s = 1` — a 780× jump — so the reported mean is conditioned on
  survivors with the largest excursions removed.  At `s = 0.5` the
  blow-up fraction is **zero in 20 million realisations**, so the
  cleaner of the two fit points carries no conditioning at all.

- **`κ⁽⁵⁾` is excluded as the home of the remaining 2σ.**  The cumulant
  ladder gives `κ⁽⁵⁾ = 7.1 %` of `κ⁽³⁾`, and the lowest `κ⁽⁵⁾` channel
  enters at the same perturbative order with the same `s³` scaling — but
  through **6 diagrams against 30** (both enumerated), putting it at
  ~1e-06, an order of magnitude below the residual's uncertainty.
  Estimated rather than computed: a 5-leg R-contracted kernel does not
  exist, and per-diagram magnitudes and the `−iᵐ/m!` convention differ
  between `m = 3` and `m = 5`.

- **FFFF was Sobol QMC and is now Gauss-Legendre.**  At 32768 samples
  `FFFF_00(t = 15, r = 0)` gave 7.47, 3.22, 2.80, 5.05e-05 across four
  seeds — 46 % scatter on a mean of 4.6e-05.  Against the converged
  GL14 the QMC column was wrong by up to **4.84e-05 absolute (121 %)**,
  as large as the residuals it was used to interpret.  GL10 vs GL14 is
  **2.20e-06** absolute, well below them.

- **The full budget** is rewritten in
  `examples/paper_assets/demo2_kappa4/budget.md`; the interpretation, in
  a fixed five-part shape, is new at `examples/demo2/INTERPRETATION.md`.

- **`examples/demo2/k3_R_coupling.py` claimed "Accuracy ~1e-6 relative".
  It is 1e-4.**  1.7e-6 is what the FK-type configuration `(t′, s, s)`
  achieves — the only one it had ever been checked at.  Measured against
  cusp-aware 3-D adaptive quadrature of the raw leg integral: 1.4e-4 at
  coincident and at three-distinct partner times, 5.4e-4 at very short
  times, 2.6e-3 at `t′ = 0.1` (where the kernel is 100× below its
  plateau, so 1.7e-8 absolute).  The README's 1e-4 was right.  The
  docstring now carries the measured table and the
  `already_R_contracted` contract's silent assumptions.

- **`k4_R_contracted.py`'s "~1e-3"** holds over the range that matters
  (2.6e-4 to 2.2e-3) and degrades to 9.9e-3 at `t′ = 0.2`, same cause.
  Its `t′ = 50` row cannot be checked by QMC at all — a Sobol rule on
  the `50⁴` box has 80 % seed scatter — so what is checked there instead
  is that `K_R` has saturated, and it has.

- **`examples/demo2/L2/reproduce_figures.py` plotted the wrong error
  bars.**  `mc_err = sqrt(ξ²/n)` is the error on a mean of ξ, not on a
  mean of per-realisation products; the latter has variance
  `<φ_a²><φ_b²> + ξ²`.  Measured against the 2M-realisation runs that do
  accumulate sums of squares, the old formula is **0.029 of the true SEM
  for ξ₀₁ (low by 34×)** and 0.48–0.63 for ξ₀₀.  Replaced by the
  Isserlis estimate, which is 0.87–1.00 of the measured SEM over
  `t ∈ [0.6, 50]`.

- **Off-grid separations biased the simulation high.**  The simulation
  measures on a grid of pitch `σ_x/5 = 0.2` and reports off-grid `r` by
  `np.interp`; on a convex profile that overestimates by +0.36 % at
  r = 0.25, +0.50 % at r = 0.5, +0.39 % at r = 0.75.  The "+3.7σ" rows of
  the ξ₀₀ r = 0.5 tables were this, not physics.  The budget now
  evaluates the theory at the simulation's own grid sites and combines
  them with the same weights, so an off-grid row is compared like for
  like, and reports a genuine grid site (r = 0.4) alongside.
  `sim_dt_study.py` additionally stores the un-interpolated grid profile
  (`x_grid`, `xi_sites`, `xi_err_sites`) so future runs need no
  correction.

- **`sim_dt_study.py` gained `--F_scale`** (multiplies the quadratic
  drift only) and explicit blow-up accounting (`n_attempted`,
  `n_blown`).  This is the discriminating experiment for the residual's
  perturbative order: each channel scales as a known power of the
  coupling amplitude, so `residual(s) = c₃s³ + c₅s⁵` and `c₃` is
  directly comparable with the computed F³κ³ without assuming the
  order-4 calculation is right.  See
  `examples/paper_assets/demo2_kappa4/fscale_fit.py`.

- **`tests/test_demo2_kernels.py` is new** — nothing in `tests/` pinned
  a demo-2 number before.  `k3_R` and `k4_R` against quadrature of the
  raw leg integral at coincident / FK-type / split / short / long
  partner times, with tolerances taken from the measurements above; the
  raw-κ³-vs-`already_R_contracted` boundary test the feature never had
  on a NON-constant kernel (`test_R_contracted_vertex.py` only ever
  compared them on a constant κ³, where the leg integral factorises and
  any time-structure-dependent aliasing bug cancels); the
  `already_R_contracted` contract's silent assumptions; one pinned FK
  value and one pinned order-0 value; and the single-site cumulant
  ladder against its generating function.

- **The cumulant ladder is a truncation, and now says so.**
  `κ_n = n! [u^n/(2n) + (λ/2)u^{n-2}]` with `u = 2αλ` reproduces demo2's
  hand-derived κ², κ³ and κ⁴ exactly and continues: κ⁵ is **7.1 %** of
  κ³ and κ⁶ is 2.6 %.  demo2's action stops at κ⁴, so the lowest κ⁵
  channel (F³κ⁵) sits at the same perturbative order as F³κ³ and scales
  the same way in the coupling amplitude.  It is therefore inside the
  residual and inside the fitted `c₃`.  This was nowhere in the 0.3.0
  budget.

### Fixed: the `auto` C-quadrature choice was decided by a wall-clock race

`PropagatorCache._gl_is_cheaper` timed one Gauss-Legendre call against
one `dblquad` call and took the winner.  Both rules are verified
converged by `select_gl_node_count` before it ran, so the race could
never produce a wrong value — but it made the *choice* depend on machine
load, and with it `Propagators.c_source` and the spline table.  Two runs
of the same config on a busy and an idle machine could resolve
differently, which is a poor property for a package that sells
reproducibility; it also made
`test_direct_calls_before_any_build_use_dblquad_under_auto` fail
intermittently (observed failing in a full-suite run made while a
28-core sweep was running, passing in isolation and on a quiet machine).

The decision is now a function of the inputs alone: prefer
Gauss-Legendre whenever a converged node count exists, fall back to
`dblquad` only when none does.

**Size of the change:** bounded by the agreement of the two rules at the
node count actually resolved, since that is all the race ever chose
between — GL vs dblquad at the deep corner: **9.1e-16** at `t_max = 3`
(n = 20), **1.2e-10** at 15 (n = 20), **2.1e-09** at 50 (n = 30),
**3.6e-09** at 100 (n = 45).  All inside the 1e-8 selection tolerance.
No demo can be affected at all: they set `c_closed_form_only`, which
short-circuits before this path.  Verified by running the previously
flaky test five times under 24-way CPU load (5/5 pass) and by the 87
tests of `test_propagator_dispatch.py`,
`test_closed_form_dispatch_boundaries.py`,
`test_c_propagator_gauss_legendre.py` and `test_deductive_numerics.py`.

**And it is faster, not slower.**  Per `_C_value_direct` call at the
deep corner on the demo1 kernel, comparing the two rules *at the node
count `auto` resolves to* (best of 5, one core): 5.7 vs 7.1 ms at
`t_max = 5` (n = 20), 5.7 vs 39.5 at 15, 12.5 vs 88.0 at 30 (n = 30),
12.5 vs 144.4 at 50, 27.9 vs 209.2 at 100 (n = 45) — Gauss-Legendre
wins by **1.3× to 11.6×**, because adaptive cost grows with the interval
while a fixed-`n` tensor rule is flat and the required `n` grows only
slowly.  The one regime the race ever chose `dblquad` for is a smooth
kernel over a short horizon, where unconditional GL costs ~1.3×.  If
that ever matters it should return as a *threshold on the converged n*,
not a timing measurement.

### Documented: unstable cache keys for callable spec fields

`cache_path` keys the symbolic expansion and the propagator table on
`repr()` of the spec, and `repr()` of a plain function embeds its memory
address — so a callable passed to `CustomKernel`, `GeneralKappa2`,
`CustomImpulse`, `ExplicitR` or `DiagonalA.gamma` changes the key every
process and the cache never hits.  A module-level `def` is affected
exactly as much as a `lambda`; what fixes it is a small frozen dataclass
with `__call__`, whose `repr` is its field values (and which pickles
cleanly for joblib besides).

No behaviour change: an unstable `repr` can only cause a cache **miss**,
never a wrong hit, so the current hashing is fail-safe and is left
alone.  Keying on `(module, qualname)` would make it stable at the cost
of introducing the one failure mode that does not exist today — two
functions sharing a qualname colliding onto a *wrong* cached expansion —
so it is documented rather than "fixed".  **Vertex couplings are
unaffected**: they never enter the key, correctly, since the symbolic
enumeration does not depend on coupling values.  Noted on
`CustomKernel`.

### Demo 1: the order-4 channel was resting on one QMC seed

- At `(0,0), r = 0.5, t = 15`, order 4 across six Sobol seeds gives
  3.93, 4.17, 1.93, 1.56, 4.07, 2.45e-5 — **39 % scatter**.  The
  published value is the `seed = 42` draw, 3.93e-5; the converged
  Gauss-Legendre value is **2.330e-5** (GL14 vs GL18 agree to 0.15 %),
  so the figure's order-4 curve was **69 % high** there, and 74 % high
  at `r = 0`.  Order 4 is 10.0 % of order 2 at t = 15, not the 17.3 %
  the published draw implies.  The fixed-seed regression check could not
  see any of this.

- At `t = 100` **neither** method is converged: QMC scatters 134 % and
  the tensor-product rule still moves 12 % between GL14 and GL18,
  because the integrand lives within `~1/γ` and `~σ_t` of the upper
  corner of a simplex of size `t_f`.  Both channels have saturated well
  before then, which is what makes the late-time points usable at all.

- **Order 2 was not converged either, at the largest times** — this was
  not in the brief and is easy to miss, because order 2 is a 2-D
  integrand and "2-D integrals are fine" is the natural assumption.  At
  `t = 100, r = 2.5, (1,1)` the converged value is **3.10410e-05**
  (GL64, GL96, GL128 and 2²² Sobol agree to 5 digits); the published
  QMC 2¹⁵ seed-42 draw gives 3.72854e-05, **+20 %**, and scatters 13 %
  across seeds.

- **The sweep is now `gauss_legendre` with `n_gauss: 24`.**  The node
  count matters and the obvious choice is wrong: order 4 costs `n⁴`, so
  the temptation is to keep `n` small — but it is *order 2* that needs
  the finer grid at large `t_f`, and `n = 14` gives it only 196 nodes.
  Its order-2 error at `t = 100` is **cell-dependent and non-monotone**:
  −14.8 % at `r = 2.5, (1,1)`, +11.9 % at `r = 0, (0,0)`, +3.1 % at
  `r = 0.5, (0,0)`, −2.2 % at `r = 2.5, (0,0)` (each against GL64).  So
  `n = 14` would have traded an order-4 error for an order-2 one of
  either sign.  `n = 24` beats the QMC it
  replaces on both orders at every grid point; worst-case residual
  error **+1.3 % (order 2) and +2 % (order 4) at `t = 100`**, below
  0.1 % by `t = 15`.  Cost ~15 min on 28 cores against ~2 min, paid
  once, for a deterministic answer with a measurable error.

- **Effect on the published curve**, measured over all 2016
  `(y, t_final, a, b, order)` totals: summed over orders, the plotted
  `xi_ab(r, t)` moves by at most **1.67 %** (median 7.8e-06).  Per
  order: order 0 unchanged to machine precision (max rel 0.00 %),
  order 2 up to 15.7 % (7.9e-05 absolute, at `t = 100`), order 4 up to
  5.0e-05 absolute — the largest order-4 point,
  `t = 23.7, r = 0, (0,0)`, goes from **8.996e-05 to 3.972e-05**.  Full
  convergence tables in `examples/demo1/L2/INTEGRATION_ERROR.md`.

### Demo 3 (filtered Poisson noise): a non-Gaussian example that is exact at F = 0

`examples/demo3/` adds a third worked example whose driving field is a
**filtered Poisson (shot) process**, `η = Σ_k h w(x−x_k) g(t−s_k)`.
Campbell's theorem gives every cumulant in closed form, which buys three
things demo 2 cannot offer.

**One knob.**  With `n ≡ ν σ_t σ_x` the shape of the one-point law depends
on `n` alone — skewness `(4√2/9)/√n`, excess kurtosis `1/(2n)` — and `h`
is fixed to hold `κ₂ = n h²/2` constant.  A sweep in `n` therefore moves
the non-Gaussian channels and *nothing else*; the observed ratio between
`n = 0.25` and `n = 4` is 4.0000 against 4.0000 predicted.

**An R-contracted vertex that is closed form, at any m.**  Every cumulant
factors through a single source point, so the `m` leg integrals of
`NonLocalVertex(already_R_contracted=True)` collapse to one
one-dimensional integral over the source time with a `2^m` closed form.
Demo 2's κ³ needs a cusp-aligned composite rule accurate to ~1e-6; this is
exact, and being *m*-agnostic it makes the neglected-cumulant ladder
computable rather than merely boundable.

**A level-A test that is exact.**  With `F = 0` the series terminates: the
`m`-point connected function is a single diagram equal to `K_R`, and no
`κ^(m′)` with `m′ ≠ m` can balance the legs, so nothing else can mix in.
The package reproduces the closed form to **1.4e-16** at `m = 3` and
**6.6e-16** at `m = 4`, coincident and at unequal positions (maxima over
the nine times in `examples/demo3/level_a_results.npz`; 4.9e-16 is the
`m = 4` value at `t = 3`, and was quoted here in 0.4.0/0.4.1 as though it
were the maximum).  The
reference simulation has **no discretisation error of any kind** — the
drift carries no spatial derivative, so a finite set of sites is exact and
sites sit exactly at the plotted separations; and the linear response to
an exponential pulse is analytic per event, so there is no time stepping.
The only approximation is the event window, bounded analytically at
7.6e-11.  All pulls are ≤ 0.64σ over 1.2e6 realisations.

**Level B computes the truncation instead of estimating it.**  `ξ₀₁` is
driven by the odd cumulants alone (order 0, FF, FFFF and the whole κ⁴
channel cancel identically under `φ₁ → −φ₁`), and all three leading
channels are evaluated: `Fκ³` (order 2), `F³κ³` (order 4, 30 diagrams) and
`F³κ⁵` (order 4, 6 diagrams — the cumulant-ladder term, which carries the
same `F³` scaling and so *cannot* be separated by an amplitude-scaling
test).  At `s = 0.2` they are 92.6 %, 7.3 % and 0.094 % of the total at
`t = 3`, leaving a geometric `O(F⁵)` remainder of 0.63 %.  Against an
exponential-time-differencing simulation over 6 seeds, every time agrees
within **0.90σ**; a paired Δt study at identical events gives `O(Δt²)`
convergence ratios of 4.0–6.9 with a residual 0.01× the Monte-Carlo error
(0.4.0/0.4.1 quoted 4.0–5.1, which excluded the second time, where the
ratio is 6.9 because the Δt differences there are ~1e-8 and the ratio is
noise-dominated rather than better than second order);
and no trajectory of 7.3e6 diverged.

The full validation ledger, error budget and the list of what is *not*
established are in `examples/demo3/INTERPRETATION.md`.  Two L2 configs
(`config_FK.yaml`, `config_F3K.yaml`) drive the same physics through the
CLI.  Locked by `tests/test_demo3_shot_noise.py` (the cumulants and the
`T̃_m` branch dispatch, against Monte Carlo of the event process,
`m`-dimensional quadrature of the raw leg integral, the package's own
`ClosedFormC`, and 60-digit `mpmath`) and `tests/test_demo3_levels.py`
(level A through the package).

Three limitations were found and are documented rather than worked around:
`Expansion.sweep` unpacks `component_pairs` as exactly two, so level A
cannot be driven from the L2 CLI; `to_feynman_diagram` hits a UID
collision on the 3-point / order-1 / single-non-local-vertex expansion, so
that diagram cannot be *drawn* (its numbers are exact); and
`config.py::_build_kernel(axis="space")` accepts only `exponential` and
`gaussian`, so demo 3's `σ_x(1+r/σ_x)e^{−r/σ_x}` envelope cannot be
declared as a `SeparableTranslation` in YAML.

## 0.3.0 — 2026-09-02

> **This is the version the CPC paper (arXiv:2606.19480, revised) refers
> to.**  v0.2.0 is what the referees ran; everything below is what changed
> for the revision, plus the fixes that had accumulated on `main` since
> July.

> **Numbers move.** Several fixes below change results that were previously
> wrong. If you have pinned values, re-pin them against the corrected output
> rather than loosening tolerances — every change is documented with its
> measured size. The two that move the most output: the causal **lower**
> bounds (any observable with an external response leg) and the **C-table
> diagonal** fix (every tadpole, i.e. every interacting order).

### Referee revision (CPC round 1): a fast default path, progress, a quick start

The CPC referee ran the README quick start on a laptop and got no output
within an hour.  Every number in this section is measured on the same
machine (Apple M3 Ultra, one core unless stated); the "laptop proxy" rows
are the same runs under `taskpolicy -c background` with
`OMP_NUM_THREADS=1`.

- **Built-in closed-form C** (`sft_wick.workflow.closed_forms`).  For a
  diagonal constant drift driven by separable translation-invariant noise
  with an exponential temporal kernel — demo1, demo2 and the README family —
  `C = ∫∫ R κ² R` is now evaluated analytically, with an optional constant
  white-noise impulse and per-component `γ`.  `System.propagators()` and the
  YAML key `propagators.c_closed_form: auto` (the default) select it
  automatically; `null` forces quadrature and a `c_closed_form_module` still
  overrides it.  The formula is written with non-positive exponents only, so
  it does not overflow at `γ t ≳ 350` where the textbook `exp(2γt)·exp(−γ(t₁+t₂))`
  form does, and the removable singularities at `γ = 1/σ_t` and `γ = 0` are
  evaluated through `expm1`.  `examples/demo1_config.yaml`,
  `examples/demo2_config.yaml` and the demo L2 configs no longer need the
  `c_closed_form_module` hook (the module files stay as references).

  Validated at the dispatch boundary rather than at convenient points
  (`tests/test_closed_form_dispatch_boundaries.py`): closed form, Gauss-Legendre
  and dblquad are evaluated directly at the same cells — the diagonal ridge
  `t₁ = t₂`, `t → t_min`, `t = t_max`, `γ = 1/σ_t` exactly and to within
  1e-7/1e-9 of it, `r = 0` and `r = r_max`, per-component `γ`, `t_min ≠ 0`, a
  white-noise impulse — and agree to **1e-10** (vs GL) and **1e-8** (vs
  dblquad).  The boundary sweep caught two things a point check would not:
  `DiagonalA` lowers per-component rates that agree to `np.allclose`'s 1e-5
  into a single-rate `R` (the closed form now mirrors the model actually
  built, so the two cannot disagree at that tolerance), and the un-split
  dblquad path was only ~2e-6 accurate at the cusp (below).

- **Separable kernels build one temporal table.**  In lazy translation mode
  `C(r; t₁, t₂) = κ_x(r)·C(0; t₁, t₂)`, so the `n_grid_t²` quadrature grid is
  built once and rescaled per separation instead of being redone for every
  distinct `r` (4× fewer quadrature calls for the README sweep, 12×+ for a
  positions sweep).  With an even temporal kernel and diagonal `C` only the
  upper triangle of the `(t₁, t₂)` grid is evaluated (another 2×).  Both are
  exact up to rounding: identical to the per-`r` full build to 1e-12 under
  Gauss-Legendre (`tests/test_propagator_dispatch.py`).  Off for user
  closed forms / `c_value_fn` (nothing is known about their structure) and
  with a white-noise impulse (its `r`-independent term breaks the scaling).

- **`c_method` default `dblquad` → `auto`.**  `auto` runs Gauss-Legendre
  with a node count chosen by `select_gl_node_count`: starting from
  `c_n_gauss = 20` the rule is refined (×1.5) until it agrees with its own
  refinement to 1e-8 at the table's extreme cells (the corner `(t_max, t_max)`,
  its mid-table neighbours, the thinnest strip), falling back to `dblquad`
  if no count up to ~100 converges.  User callables (`GeneralKappa2`,
  `CustomKernel`, `CustomImpulse`, `ExplicitR`, callable `γ`) go straight
  to `dblquad`.  Why not simply "GL at 20": at fixed node count the
  accuracy degrades with `(γ + 1/σ_t)·t_max`.  Measured on the demo1
  kernel (`γ = 1`, `σ_t = 0.3`, cell `(t, t)`, relative to the closed form):

  | t   | split dblquad | GL n=20 | GL auto (n) |
  |-----|---------------|---------|-------------|
  | 5   | 7 ms, 2e-16   | 5.6 ms, 1e-15 | 5.8 ms, 1e-15 (20) |
  | 15  | 38 ms, 3e-16  | 5.9 ms, 1e-10 | 6.0 ms, 1e-10 (20) |
  | 30  | 86 ms, 1e-15  | 5.7 ms, 4e-7  | 13 ms, 2e-12 (30) |
  | 50  | 145 ms, 8e-13 | 5.9 ms, 2e-4  | 13 ms, 2e-9 (30) |
  | 100 | 212 ms, 8e-10 | 5.9 ms, **1e-2** | 30 ms, 3e-9 (45) |

  For a Gaussian temporal kernel (`σ_t = 0.3`) fixed n=20 is already 2e-4
  off at `t = 15`; `auto` picks 45 there (30 ms, 2e-11) and hands `t ≥ 30`
  to dblquad.  On the demo1 / demo2 kernels none of this runs: the closed
  form does.  The L0 default (`PropagatorCache(c_method="auto")`) resolves
  at the first table build; a direct `C_value` call before any build uses
  dblquad.

- **The dblquad path splits the rectangle at the `λ₁ = λ₂` cusp** (the
  same three pieces the Gauss-Legendre path uses).  Stationary `κ²` has a
  `|λ₁ − λ₂|` cusp there; integrating across it made scipy's adaptive rule
  both slow — the referee's run: 74 ms per call rising to 245 ms with `t`
  — and inaccurate, reporting roundoff and delivering 2e-6 relative error
  at `epsrel = 1e-10` against the closed form.  Split: 7 ms per call at
  `t = 5` (38 ms at 15, 212 ms at 100) and 1e-10 to 1e-15.  **Numbers
  move** by up to ~2e-6 relative on the dblquad path, toward the exact
  value.

- **Progress reporting** (`sft_wick.progress`).  Three stages get bars —
  expansion, C-table build (per cell, per `r`), sweep (grid points ×
  diagrams) — with elapsed / ETA, through `tqdm` when installed
  (`pip install sft-wick[progress]`) and otherwise as plain stderr lines;
  bars stay silent for loops under one second and by default when stderr
  is not a terminal.  CLI: stage banners with timings, `--quiet`, and
  `SFT_WICK_PROGRESS=0` / `=1` to silence / force.  L1:
  `progress=True|False|callable` on `System.expand`, `System.propagators`,
  `Expansion.sweep`.  Never changes a number (tested).

- **`sft-wick run --dry-run` prints a cost estimate**: diagrams per order,
  grid points, distinct separations, the resolved C source, the number of
  quadrature calls the chosen path will make, and a rough wall-clock from
  a one-second micro-benchmark of one C call and one diagram evaluation
  per order (`sft_wick.workflow.estimate`).

- **`examples/quickstart.yaml` and `sft-wick quickstart`.**  Demo1 physics
  at orders 0 and 2, three separations, two times, 4096 samples, markdown
  table to stdout.  `sft-wick quickstart` writes the file to the current
  directory and runs it.  The README "Quick Start — L2" is now this file.
  A GitHub Actions job (`examples-time-gate`) runs it under `timeout 600`
  and `examples/demo1_config.yaml` under `timeout 1200` on a stock
  `ubuntu-latest` runner.

- **Validation catalogue** `docs/verification/catalog.rst`, generated by
  `tools/gen_test_catalog.py` from the collected suite (`pytest
  --collect-only`) plus a curated table of what each file checks, against
  which reference (brute-force Wick, closed form, alternative backend,
  simulation cache, …) and at what tolerance.  `make -C docs html`
  regenerates it and `tests/test_catalog_current.py` fails when it is
  stale or a test file has no entry.  The README's "460 tests" and the
  verification page's "275 tests" were both wrong; the catalogue carries
  the measured count.

- **Fixed: `DiagramTerm.to_latex()` printed a local coupling's point.**
  The index-relabelling passes rebuilt `Symbol`s without the `local` flag,
  so the symmetrised partner of a local `F` rendered as
  `F_{i1 i0 i2}(y_0)` (Table 1 of the paper).  Regression test
  `tests/test_latex_local_coupling.py`.

- Docstrings in `sft_wick.spectral` referred to a directory outside the
  repository; rewritten to state the caveat itself.

- **Paper-figure regression check.**  `examples/demo1/L2/config.yaml`
  (the sweep behind the paper's Gaussian-noise figures: 672 grid points ×
  71 diagrams, 32768 samples) run on v0.2.0 and on this release agrees to
  a maximum relative difference of **3.3e-14** over all 2016
  `(y, t, a, b, order)` totals — the closed-form C path is untouched by
  the C-table diagonal fix, as the July changelog claimed; now verified
  rather than assumed (`examples/paper_assets/README.md`).

- **Fixed: `output: {type: npz, path: results/x.npz}` failed when the
  directory did not exist** (`np.savez` does not create it; the table
  writer already did).  The same for `plot` outputs.

- **`c_method: auto` also compares cost.** When both rules are converged
  for the horizon, one call of each at the deep corner decides: on a
  smooth kernel over a short horizon adaptive dblquad needs a few hundred
  evaluations (3.8 ms) and beats the 20-node tensor rule (5.6 ms); with a
  cusp or a long horizon the tensor rule wins by 5-30×.  (Without this,
  a smooth-kernel regression test took 73 s instead of 28 s.)

### Demo 2 (non-Gaussian noise) corrections

Re-deriving and re-running the κ⁽³⁾ example for the referee turned up
three things the paper's demo-2 figures inherit; all are fixed in
`examples/demo2` and quantified in `examples/paper_assets/demo2_kappa4`.

- **κ⁽³⁾ was missing its `α³` term.**  For `η̃ = η + α(η² − λ)` the third
  cumulant is `κ⁽³⁾(1,2,3) = 2αλ²[k₁₃k₂₃ + k₁₂k₂₃ + k₁₂k₁₃] + 8α³λ³ k₁₂k₂₃k₁₃`;
  the second term (the connected three-point function of `η² − λ`) is
  2.4 % of the first at coincidence and was absent from `k3_coupling.py`
  and from the `6αλ²` line of the κ³ cross-check figure.  The simulated
  third moment, 9.21e-3, is `6αλ² + 8α³λ³`, not `6αλ² = 9.00e-3`.  The
  FK channel of ξ₀₁ moves by +1.2 % (t = 1) to +0.6 % (t ≥ 3).

- **The FK quadrature was not converged beyond t ≈ 10.**  The raw kernel
  is narrow in the relative leg times, and the fixed 8-node tensor rule
  on the 4-D integral gives, for ξ₀₁ at r = 0, 4.95e-4 at t = 15 and
  1.85e-4 at t = 50 against the converged plateau 3.44e-4 (n = 12/16/20
  give 4.23/3.86/3.71e-4 at t = 15; QMC is worse).  The remedy is the
  package's own `already_R_contracted` vertex: `examples/demo2/k3_R_coupling.py`
  integrates the three legs inside the callable (analytically in the
  common time, composite Gauss-Legendre on the two relative times with
  each cusp aligned to the grid), so the FK integral is one-dimensional
  and a 32-node rule agrees with 64 to 1e-4 at every t.  `config_FK.yaml`
  now uses it.  The R-contracted kernel is validated against adaptive
  3-D quadrature and QMC of the raw leg integral to ~1e-4.

- **ξ₀₀ / ξ₁₁ used the single-kernel `λ_eff` approximation.**  The exact
  effective covariance of `η̃` is `λ k + 2α²λ² k²` -- the second piece
  has HALF the correlation time and length -- so replacing it by
  `λ(1 + 2α²λ) k` over-counts its contribution to C by 70 %: +1.8e-4 on
  ξ₀₀ at large t (1.5 % of the signal, 2σ of the 200k-realisation
  simulation).  The budget uses the exact form (a sum of two built-in
  closed forms) for order 0, FF and FFFF.

- **κ⁽⁴⁾ enters ξ₀₀ / ξ₁₁ but not ξ₀₁.**  `κ⁽⁴⁾ = 4α²λ³ Σ_paths k k k +
  16α⁴λ⁴ Σ_cycles k k k k` (12 Hamiltonian paths and 3 cycles of K₄;
  `examples/paper_assets/demo2_kappa4/k4_coupling.py`, checked against
  Monte Carlo).  F·κ⁽⁴⁾ at order 2 vanishes by ψ/φ counting; the leading
  term is F·F·κ⁽⁴⁾ at order 3 (three pure-R⁶ diagrams).  The Gaussian
  theory with F has a `φ₁ → −φ₁` symmetry that only ODD cumulants break,
  so FF, FFFF and FFK4 are identically zero for ξ₀₁ -- the κ⁽³⁾ signal
  there is clean, and its residual can only be closed by F³κ⁽³⁾ and
  higher odd terms.

- **The error budget** (`examples/paper_assets/demo2_kappa4/budget.md`,
  2M realisations per step size at Δt = 0.02 and 0.01, extrapolated):
  ξ₀₁ − FK is 0 within 1σ for t ≤ 1 and +24 % of FK (9e-5, 6-7σ) for
  t ≥ 5; the order-4 F³κ⁽³⁾ channel, estimated by collapsing κ⁽³⁾ to an
  equal-time constant calibrated on FK, is 2 % / 12 % / 24 % of FK at
  t = 1 / 3.5 / 15 -- the residual is truncation, not κ⁽⁴⁾ (zero there by
  symmetry), Δt or Monte-Carlo noise.  ξ₀₀ with the exact C_eff, FFK4
  and FFFF agrees within ±2σ except at t ≈ 5 (+3σ).

### Added

- **`sft_wick.selfconsistency` — the fixed-point driver.**
  `solve_self_consistency(initial, step, ...)` iterates, mixes, and is honest
  about what happened. The package could compute a self-energy but nothing in
  it iterated, so every DMFT-style use was one pass of a loop the caller wrote
  by hand, usually without convergence diagnostics.

  The Dyson solve is **deliberately not** attempted: it is model-specific and
  is genuinely an integral-equation solve rather than a diagram evaluation, so
  a wrong general one would be worse than none. You supply it as `step`.

  It never returns a bare state — a non-converged iteration looks exactly like
  a converged one if you only print the last state. The result carries
  `converged`, the residual history, and a `reason` in
  `converged / diverged / oscillating / max_iter`; `bool(result)` is
  `converged`. Four failure modes it is specifically built to avoid reporting
  as success are documented in the module docstring and in
  `docs/api/selfconsistency.rst`, each with a regression test. Linear mixing
  only — no Anderson acceleration, which matters near a transition.

- **Callable (spacetime-dependent) couplings now work with MATRIX-valued
  response propagators.** This combination was previously computable by *no*
  backend at any cumulant order: `method='qmc'` refused callables, while
  `qmc_vectorized` and `gauss_legendre` refuse matrix `R`. Worse, the scalar
  loop's error message named the other two, which refuse for the *other*
  reason — a loop of three `NotImplementedError`s with no exit. It is the
  natural disorder-averaged-with-components configuration.

  The scalar loop is in fact the natural home for the per-sample callable
  contract, since it already visits one sample at a time.
  `DiagramIntegrand.evaluate()` takes a `coupling_array=` override, and
  `dynamic_coupling_array()` materialises it. Verified against
  `qmc_vectorized` where both are legal, against the static-tensor path at the
  same point (exact agreement in both), and against three closed-form
  one-dimensional integrals.

- **`SpectralDensity.average(f, node_axis=-1)`.** When the array `f` returns is
  square, which axis indexes the spectral nodes is ambiguous by shape and no
  check can resolve it; `node_axis=` states it at the call site. Note that in
  exactly that square case a *wrong* `node_axis` returns a wrong number rather
  than raising — the parameter is a declaration the array cannot confirm.

- **`propagators_from_cache`** wraps a hand-built `PropagatorCache` as a
  `Propagators`, so the L1 workflow API can be driven by a cache that does not
  come from a `System` — which the spectral one does not, since its `R` and
  `C` come from a spectrum rather than the system's `kappa2`. Without it the
  whole workflow layer was closed to `sft_wick.spectral`.

- **`sft_wick.spectral` — disorder-averaged (spectral) propagators.**
  `SpectralDensity` + `spectral_cache(density, D, shift=...)` build a
  `PropagatorCache` whose `R` and `C` are superpositions of Ornstein-Uhlenbeck
  propagators over a spectrum `rho(h)`:

      R*(t,t')   = Theta(t-t') <exp(-h (t-t'))>_rho
      C*(t1,t2)  = <(D/h) (exp(-h|t1-t2|) - exp(-h(t1+t2-2 t_min)))>_rho

  This is the disorder-averaged limit: the effective single-site problem is
  scalar, so the `O(N^rank)` coupling-index contraction that dominates a
  per-instance matrix calculation disappears. `R*` is a superposition of
  decays, i.e. genuinely non-Markovian.

  with the initial condition `x(t_min) = 0`. A zero rate is admitted and takes
  the finite free-diffusion limit `2 D (min(t1,t2) - t_min)`; a negative rate
  is rejected, and round-off negatives from `eigvalsh` are clamped.

  Construct the density with `SpectralDensity.from_samples` (empirical spectra,
  reduced by equal-mass binning),
  `.from_callable` (Gauss-Legendre over an analytic density), or `.delta`
  (a single rate, which reduces every formula to the plain OU one — the
  boundary check the tests are built on).

  Both propagators are evaluated **exactly, never tabulated**. Tabulating `C`
  on a `(t1,t2)` grid and splining it reintroduces the diagonal ridge — `C`
  has a derivative jump of exactly `-2D` on `t1 == t2`, and every tadpole
  evaluates `C(s,s)` there. Measured on a sampled Marchenko-Pastur spectrum:
  a hand-rolled `RectBivariateSpline` gives 23% relative error on the diagonal
  and does not converge (0.2306 at n_grid=30, 0.2228 at 60), against 2.4e-04
  at 64 spectral nodes and 1.3e-05 at 256.

- **Two-time observables from the declarative API.** `Expansion.sweep` takes
  `external_times_grid={point: [times]}` — the same shape as `positions_grid`,
  swept as a further Cartesian axis — and `Expansion.evaluate` /
  `integrate_diagrams` take `external_times=`. The YAML `sweep:` block accepts
  `external_times_grid` too. Result rows gain a `t_<point>` column per named
  point, and `SweepResult.totals()` groups by them.

  The **response field is now nameable in an observable**: `"psi_b(y)"`
  alongside `"phi_a(x)"`. Without it there was no way to ask for `R(t,t')` at
  all — the parser accepted only the physical field.

  Together these make `R(t,t')` and `C(t,t')`, the DMFT order parameters,
  reachable declaratively. Previously every external was pinned at a single
  `t_final`, and since Θ kills an R joining two externals at the same time,
  *every* observable carrying a response leg came back identically 0.
  Omitting `external_times_grid` reproduces the old rows exactly.

- **`external_times={point: time}`** on `integrate_moment_{qmc,qmc_vectorized,
  gauss_legendre,nquad}`, the `integrate_moment` dispatcher,
  `_evaluate_zero_dimensional` and `integrate_two_point_qmc`. Every integrator
  previously pinned *all* fixed externals at a single `lambda_f`, so `R(t,t')`
  and `C(t,t')` were unreachable — and because Θ kills an R joining two
  externals at the same time, **every observable with an external response leg
  came back identically 0** at every order through all five backends.
  `external_times=None` reproduces the old numbers bit-for-bit.
- **Time-dependent (callable) *local* couplings.** The vertex's single
  spacetime point was discarded, so a callable local coupling was rejected.
- **`PropagatorCache(model, c_value_fn=...)`** — a public L0 hook for supplying
  C directly, without subclassing and overriding semi-private methods.

### Fixed

- **`_mix` coerced the state's leaf through `np.asarray`.** The container
  branches preserve dict / list / tuple / namedtuple, but the leaf did not, so
  a state whose leaves are array-*like* rather than `ndarray` — a JAX or torch
  array — reached `step` as its own type on iteration 1 and as a plain host
  `ndarray` from iteration 2 onward, silently moving a device array off the
  device. Only `damping > 0` reached the coercion, so it was invisible in the
  default configuration and appeared exactly when the caller took the module's
  own advice to add damping.

- **The C-value memo lost an ndarray subclass's identity.**
  `np.ascontiguousarray` downcasts a subclass to a base array, so a masked
  array's `tobytes` override never ran and the mask vanished from the key: two
  positions differing only in their mask shared one entry. `tobytes()` already
  serialises in C order whatever the memory layout, so the
  `ascontiguousarray` call bought nothing and cost this.

- **Object-dtype positions were keyed on recyclable pointers.** An object
  array's buffer is raw `PyObject*` values, and an address is unique only among
  *live* objects — the same defect as keying on `id()`. Those are no longer
  memoised at all, and the refusal is decided from inside the recursion, so a
  nested object array is caught too.

- **The memo entry was frozen in place rather than copied.**
  `_C_value_direct` ends in `np.asarray`, the identity for a float64 array, so
  marking the result read-only flipped the flag on the *user's* object and
  broke any `c_value_fn` returning a reused buffer or a module-level constant.
  Copied first now; the memo is still handed out read-only, since it is the
  cache entry itself and a caller doing `C += x` would rewrite every later
  lookup.

- **The memo rode into every parallel worker.** All three
  `precompute_C_table_*` builders dispatch a closure referencing `self`, so a
  full memo was serialised into each worker payload — tens of megabytes,
  growing as `N^2`, and useless to the worker, which recomputes what it needs.
  `__getstate__` empties it.

- **`evaluate_at` ignored `coupling_vectorized=True`,** handing a callable
  declared under the batched contract the per-sample one instead. Some batched
  callables broadcast and return a plausible *wrong* shape rather than raising.

- **A coupling leg that no propagator attaches to** has no `direction_map`
  entry, and all backends raised a bare `KeyError` when the callable asked for
  its position.

- **`SpectralDensity.average` did not validate `node_axis`.** Out-of-range
  values died with a bare `IndexError` from the shape lookup, and
  `node_axis=True` sailed past the length check (`shape[True]` is `shape[1]`)
  only to die inside `tensordot`. Neither names the parameter at fault.

- **`max_abs_distance` subtracted in the input's integer dtype,** so the
  difference wrapped modulo `2**nbits` and a state far from the fixed point
  could report a distance small enough to satisfy `tol` — convergence declared
  by overflow.

- **The C cache keyed ndarray arguments on `id()`.** `id()` is unique only
  among *live* objects, so a freed position array's address can be recycled and
  the cache then returns one separation's `C` for another. Now keyed on
  contents. I could not force such a collision in 200k attempts, so the
  practical reachability is low — but the key was wrong, and a value key is
  correct and cheap here.
- **`PropagatorCache.C_diagonal` was position-blind when a spatial table was
  present.** It consulted the legacy time-only table first while `C_value`
  consults the spatial one first, so the two accessors disagreed by ~38 % with
  both tables built. Same precedence in both now.
- `sft_wick.spectral`: `average()` gives a diagnostic error for a transposed
  layout instead of numpy's shape error — a better message, not new safety:
  that layout already raised. The genuinely ambiguous case, `k == n_nodes`,
  is indistinguishable by shape and is documented rather than claimed fixed; `SpectralDensity` became comparable and hashable
  (the generated dataclass `__eq__` raised on ndarray fields);
  `C_diagonal_batch` appends the component axis rather than inserting it at
  position 1 (identical for 1-D times, wrong for 2-D); `clear_cache()` keeps
  the batch-C capability instead of demoting every backend to the scalar loop;
  and `noise_D` / `n_components` / `t_min` / `n_nodes` are validated.

- **Causal *lower* bounds from external response legs were dropped entirely.**
  Every bound-builder wrote `if earlier in int_vars: upper_bounds[...]`, which
  silently discards any ordering whose *earlier* endpoint is external — exactly
  the orderings an external ψ leg creates, since `<φ(u) ψ(y)> = R(u,y)` forces
  `t_u >= t_y`. The O(g) response was integrated over `[t_min, t_x]` instead of
  `[t_y, t_x]`: up to 5x wrong, and −11.25 instead of exactly 0 at `t_y = t_x`.
  Lower bounds are now also transitively closed along internal edges, and
  inverted intervals are clamped so `nquad` never integrates backwards (which
  returned a negative volume, sign-inverting the O(g²) response).
- **A swept external's causal bounds are now carried, not refused.** With
  `integrate_over`, a bound from a swept point is variable-valued. Dropping it
  is not a correctness error — Θ already zeroes the integrand there — but it is
  a quadrature error for a fixed-node rule, because Θ leaves a jump *inside*
  the domain: `gauss_legendre` measured **22.5 % wrong at the library default
  `n_gauss=8`**, converging only as O(1/n). Now exact at n=8. The same applies
  to orderings between two swept externals (**29 %** at n=8) and, since they
  can be mediated by an internal vertex or involve a fixed external, the whole
  ordering graph is closed transitively rather than by edge kind.
- **Retardation (Θ) is applied at diagram evaluation.** `R_time` is the raw
  Θ-stripped model accessor; the three consumers that multiply R factors now
  apply Θ with a strict `t_left > t_right` test. An R joining two *fixed*
  externals has no integration domain to enforce it, so an order-0 response
  returned the unbounded acausal `exp(+μ(t_y−t_x))`, and 1 instead of 0 at
  equal times.
- **The C builder contracted only `diag(R)`** instead of `R κ Rᵀ` — 57 %
  Frobenius error for a dense drift.
- **Eighteen complex→real projection sites** took `abs()` of a complex value or
  `.real` of an imaginary one, destroying the sign (or zeroing the value) of
  any diagram whose observable carries external response legs. All now route
  through a checked projection using the reality theorem: with
  `n_R = Σ_v n_ψ(v) + E_ψ`, a diagram equals `i^(−E_ψ) ×` a real number.
- **The C table did not converge on its diagonal.** `C(t1,t2)` integrates up to
  `min(t1,t2)`, so `∂C/∂t1` jumps by exactly `−σ²(t)` across `t1 == t2`; a
  tensor-product spline is C² and cannot represent that ridge. Measured
  mid-cell relative error **22.3 % at `n_grid=41`, still 21.4 % at 321**
  (absolute error O(h) against O(h⁴) off the diagonal) — and *every tadpole*
  evaluates `C(s,s)`, exactly on it. The grid's own `i == j` entries are now
  harvested into a separate interpolator, for the legacy table, the lazy
  spatial builder and the three full spatial grids. This **changes every
  interacting order**: `examples/demo1`'s YAML-workflow sweep moves in 42 of
  48 rows by up to 6.1e-3 relative. (The demo1 *notebook*, which produces the
  paper figures, is unaffected — it uses a closed-form analytical cache.)
- **`integrate_two_point_qmc` ignored spatial separation at order 0** and, above
  order 0, ignored a spatial C table in favour of the single-`t_ref` κ² ratio.
  The first returned the coincident-point value at *every* separation (a factor
  e² at r = 2σ); the second was 8.2 % off at order 2 for a kernel whose
  correlation length grows with time.
- `DiagramIntegrand.evaluate` silently returned 0 for a callable coupling, and
  a scalar-field coupling of the wrong rank raised an opaque `TypeError`.
- The reality projection ran *before* the negligibility guard on three
  prop-indexed sites, so a coupling entry that is float noise around zero
  raised instead of being skipped.

### Changed

- **The spatially-uniform-cache warning moved to `Expansion.sweep`.** A cache
  with no spatial table returns the same `C` at every separation, so a
  positions sweep yields a column of identical numbers — correct for a
  disorder-averaged single-site cache, a silent mistake otherwise. The
  question is "does this *sweep* cover more than one position configuration",
  which is answerable only where the grid is. Asked per grid point instead it
  was silent on a sweep over a single position key and fired on every ordinary
  single-separation `evaluate(x != y)` — and a warning that fires on normal
  use is worse than none.

- `ito=False` is documented as a **symbolic** switch. It keeps the equal-point
  R terms in the expression tree but does not change any number: Θ(0)=0 and the
  absent Stratonovich functional Jacobian cancel exactly, and for additive noise
  the Itô and Stratonovich answers coincide. Do **not** "fix" this by setting
  Θ(0)=1/2 without emitting the Jacobian — for the linear vertex that adds a
  spurious term growing without bound in the final time (measured 200 %/400 %/
  800 % of the exact answer at T = 4/8/16).
- `PropagatorCache.R_time`'s docstring is now the single authoritative
  statement of the Θ convention, which was previously described three
  inconsistent ways.

### Internal

- `PropagatorCache` stays picklable with a C table built: the diagonal
  interpolator serialises only its nodes and values, since
  `scipy.interpolate.CubicSpline` is not picklable on scipy ≥ 1.18. This keeps
  `propagators.cache_path`, `integrate_diagrams(n_jobs>1)` and
  `Expansion.sweep(n_jobs>1)` working.

### Fixed (details of the first item above)

- **`DiagramIntegrand.integration_bounds` returned wrong integration bounds.**
  `scipy.integrate.nquad` calls `ranges[i]` with the **outer** integration
  variables `int_vars[i+1:]`; the code indexed them as if they were the inner
  ones. Two silent failure modes followed: at 2 variables the bound collapsed
  to the literal `1.0`; from 3 variables the guard could *pass* and return the
  wrong outer variable. Every diagram containing a vertex-to-vertex response
  chain `R(y_i, y_j)` was affected (44/44 order-3 diagrams of the quartic
  Langevin model). The correct mapping — `later_args[index(src) - i - 1]` —
  was already present in `integrate_moment_nquad`; the public helper had never
  been migrated to it. **The five production integrators build their own
  bounds and were never affected**, so no supported computation changes.

- **The C-propagator builder used only the diagonal of a matrix `R_time`.**
  Both quadrature paths (`_C_value_direct`, `_C_value_direct_gl`) contracted
  `R[a,a] κ[a,b] R[b,b]` instead of the matrix triple product
  `C = ∫∫ R(t1,λ1) κ(λ1,λ2) R(t2,λ2)ᵀ`. That is correct only when the linear
  operator is diagonal in the chosen basis; for a dense drift (e.g.
  `A = H + λ` with `H = XᵀX/N`) it gave a 57 % Frobenius error, wrong signs on
  6 of 9 matrix elements, and exact zeros where the true value was nonzero.
  `iso_R=True` results are bit-for-bit unchanged.

- **`abs()` projections silently destroyed signs.** Four sites returned
  `result.real if result.imag == 0 else abs(result)`, and six more took
  `complex(coeff).real`, annihilating an imaginary coefficient. All thirteen
  now route through `_real_or_raise`, which applies the structural `i**E_psi`
  rotation and *raises* rather than guessing when the result is still not
  real. See the reality theorem on `DiagramTerm.observable_phase_factor`.
  Consequence: observables carrying external response legs (response
  functions) are now correct where the four backends previously disagreed by
  factors of `-1` and `0`; a mis-specified action now errors instead of
  returning a plausible wrong number.

- **`DiagramIntegrand.evaluate` silently returned 0 for a callable coupling.**
  It reads the static `coupling_array`, a zeros placeholder on the dynamic
  path. It now raises `NotImplementedError` naming the backends that do
  support callables.

- **A wrong-rank coupling for a scalar field raised an opaque `TypeError`.**
  It now raises a `ValueError` naming the symbol and the expected rank, and
  accepts a size-1 array of any shape.

### Added

- **Time-dependent (callable) *local* couplings.** A local vertex has exactly
  one spacetime argument, so evaluating its coupling there is unambiguous —
  but that point was discarded when building the coupling `Symbol`, so
  `build_integrand` had no coordinates to pass and refused every callable.
  The point is now recorded (`Symbol.spatial_args`, with `Symbol.local=True`
  suppressing it in LaTeX so rendering of constant couplings is unchanged).
  This is what expanding about a time-dependent mean-field trajectory
  requires: the vertices then carry couplings ∝ `wbar(t)^k`. Verified against
  a closed form to 3.7e-12, with the constant-callable path bit-identical to
  the ndarray path.

- **`PropagatorCache(..., c_value_fn=...)`** — a public L0 hook to supply `C`
  directly, bypassing the `∫∫ R κ R` construction. Necessary for any
  disorder-averaged (DMFT) propagator pair, where `⟨R κ R⟩ ≠ ⟨R⟩ κ ⟨R⟩` means
  no noise cumulant can reproduce the correct `C`. Previously this required
  subclassing and overriding `_C_value_direct`.

- **`DiagramTerm.n_external_response`** (`E_psi`) and
  **`DiagramTerm.observable_phase_factor()`** (`i**E_psi`), plus
  `DiagramIntegrand.expected_phase`, exposing the phase structure that makes
  the reality projection well posed.

- **`_collect_symbol_occurrences`** and a guard in `build_integrand`: a
  callable coupling occurring at genuinely different point sets (two copies of
  one vertex at order ≥ 2) is now refused instead of being evaluated at the
  first occurrence's coordinates, which was measured 4.06× wrong. Permutations
  of a single point set — the symmetrised non-local coupling sum — are still
  accepted, preserving the κ⁽ᵐ⁾ feature.

  **Corrected after 0.4.2:** accepting those permutations was not correct.
  Every permutation was still evaluated at the first occurrence's leg order,
  which is right only for a kernel symmetric under a permutation of its leg
  points at fixed component indices.  See the Unreleased entry.

### Tests

- `tests/test_msr_numerics_regressions.py` — 17 tests pinning each of the
  above against independently derived ground truth (stationary Fokker-Planck
  series, the exactly-solvable `D/(mu+k)` model, closed-form matrix `C`).
  15 of them fail on the previous code.

## 0.2.0 — 2026-07-25

First release on PyPI (`pip install sft-wick`).

### Features

- **`NonLocalVertex(equal_time=True)`** — opt-in mode for non-local
  vertices whose `coupling` callable encodes an **equal-shell** /
  **equal-time** connected cumulant (one common cosmology use case:
  `canoes.sachs.compute_kappa3_zeta_table` which returns the
  equal-shell bispectrum `ζ_eq(γ_12, γ_23, γ_31; λ)` for the LSS
  matter / potential 3-point function).

  When `equal_time=True`, the `m` time legs of that vertex collapse
  into a **single** integration variable while the `m` spatial legs
  remain independent (so the callable still receives `m` distinct
  positions and only one shared time replicated `m` times in
  `t_list`).  This matches the action-level relation
  `κ^(m)(z_1,...,z_m) ≈ δ(λ_1−λ_2) … δ(λ_{m−1}−λ_m) · κ_eq(n_1,...,n_m; λ)`
  used in single-shell / Limber-like cosmological approximations.

  The default `equal_time=False` is unchanged — the full
  cross-spacetime cumulant `κ^(m)(z_1,...,z_m)` with `m` independent
  `(n, λ)` integrations is still the canonical contract.

  Failure mode this guards against: passing an equal-shell ζ as
  `coupling` without `equal_time=True` lets sft-wick faithfully
  sweep `m` independent times, contributing a spurious `(t_max)^(m−1)`
  factor of integration measure (e.g. `~5.6×10^6` for `m=3` and
  `t_max=2360` Mpc).  This was the diagnosis-of-record for the
  STF_lensing path-integral lensing Order-2 FK ⟨κκ⟩ bug.

  Implementation spans `workflow/specs.py`, `workflow/system.py`,
  `workflow/config.py`, `vertices.py`, `perturbation.py`, and
  `evaluate.py`.  Threaded through as
  `Vertex(equal_time=...)` → `VertexInstance.equal_time_aliases` →
  `DiagramTerm.equal_time_aliases` →
  `SpatialStructure.equal_time_aliases`, with time-alias resolution
  in `_times` of the QMC + Gauss-Legendre paths and in
  `DiagramIntegrand.evaluate`.  Diagram topology (R / C propagator
  routing, direction groups) is unchanged — only the time-integration
  Jacobian and the callable's `t_list` payload differ.

  YAML usage::

      nonlocal_vertices:
        - name: K
          order: 3
          coupling_module: kappa3_callable.py
          coupling_attr: coupling_fn
          equal_time: true

  See `docs/user_guide/workflow.rst` for the full discussion and
  `docs/notes/equal_time_nonlocal_vertex.md` for the design note.
  Regression coverage: `tests/test_equal_time_nonlocal.py`
  (11 tests covering alias plumbing, static evaluation, dynamic
  callables, and Jacobian ratio).

- **`NonLocalVertex(already_R_contracted=True)`** — opt-in mode where
  the user-supplied callable returns the **R-contracted** form
  `κ^(m)_R(γ; z_1', …, z_m') := ∫ ∏ R(z_i', z_i) · κ^(m) dz_1…dz_m`
  evaluated at the **partner** (outer) spacetime points rather than
  the κ's own legs. When set, sft-wick

  * tags the m R-propagators attached to this vertex's ψ legs as
    **absorbed** — they remain in the diagram graph (so direction
    groups continue to identify the leg with its partner) but
    contribute a factor of 1 instead of the usual `R_time` value;
  * aliases each leg's time onto its Wick partner's time via the
    existing `equal_time_aliases` machinery, so the m leg-time
    integration variables drop out of the simplex;
  * feeds the callable `(n_list, t_list)` with the **partner**
    coordinates instead of the leg's own.

  Cuts integration dimensionality by `m` per such vertex (e.g. an
  order-2 F+K diagram becomes 1-D after absorbing K's 3 legs). The
  surviving integrand is also **smoother** because the narrow-kernel
  peak has been folded into κ^(m)_R, so Gauss-Legendre converges
  exponentially with far fewer nodes. Motivating use case: canoes'
  squeezed κ³ at `ℓ_max = 5000, z_s = 1100`, where the raw kernel
  has diagonal width `Δχ ~ 2 Mpc/h` and the per-leg χ-integration
  would otherwise demand `~10^11` quadrature points.

  Mutually exclusive with `equal_time=True` (rejected at
  construction). Default `already_R_contracted=False` is unchanged.

  Implementation spans `workflow/specs.py`, `workflow/system.py`,
  `workflow/config.py`, `vertices.py`, `perturbation.py`, and
  `evaluate.py`. Threaded through as
  `Vertex(already_R_contracted=...)` →
  `DiagramTerm.r_absorbed_pairs` (collected per-routing in
  `_collect_r_absorbed_pairs`, merged into `equal_time_aliases`) →
  `SpatialStructure.r_absorbed_pairs`, with R-factor skipping
  centralised in `evaluate.py::_kept_r_propagators` (used by all
  five R-product sites: scalar / matrix per-sample, QMC-vectorised,
  GL tensor-product, integrate-over external).

  YAML usage::

      nonlocal_vertices:
        - name: K
          order: 3
          coupling_module: kappa3_R_callable.py
          coupling_attr: coupling_fn
          already_R_contracted: true

  Companion utility `sft_wick.build_R_contracted_callable` wraps a
  raw κ^(m) + R kernel into the R-contracted form via brute-force
  trapezoid quadrature on a user χ-grid — for validation comparand
  use; production callers should supply the analytical / pre-
  tabulated κ^(m)_R directly (e.g. canoes' FFTlog-of-W chain).

  See `docs/user_guide/workflow.rst` and
  `docs/notes/R_contracted_nonlocal_vertex.md` for full
  documentation. Regression coverage:
  `tests/test_R_contracted_vertex.py` (15 tests including the
  machine-precision constant-κ³ equivalence at `rtol=1e-12` and the
  four-way per-sample × vectorised contract check).

### Performance

- **Test-suite wall time cut ~3.5×** (from ~18 min to ~5 min for the
  full 275-test suite on M-series).  The profile showed two tests
  accounting for 80 % of runtime — both were building expensive
  C-propagator caches that the test's tolerance did not actually
  require:
  - `test_CF3_full_run_matches_L1_reference` (585 s → 72 s): this is
    an equivalence test (YAML pipeline vs. direct Python — rtol=1e-10
    is bit-identity), so the propagator grid only needs internal
    consistency, not physical precision.  `n_grid_t` 30 → 12,
    `t_max` 3.0 → 2.0, `n_samples` 1024 → 256.
  - `test_T4_time_dependent_gamma_end_to_end` (302 s → 24 s): an
    order-0 smoke test whose single C-evaluation did not justify a
    30² dblquad grid.  `n_grid_t` 30 → 10, `n_grid_cache` 120 → 40,
    `t_max` 2.0 → 1.5.
  - `test_W3_white_noise_absorbed_into_translation_spline`
    (97 s → 32 s): lazy-spline cache cost dominated — `n_grid_t`
    25 → 15 (quadratic win); `n_samples` 2¹³ → 2¹¹ (negligible since
    QMC is not the bottleneck here).
  - `test_spline_matches_split_dblquad` (77 s → 54 s):
    `n_grid` 30 → 25 with worst-case-O(h⁴) tolerance relaxed from
    2e-3 → 3e-3 (still within 2× the theoretical bound).
  - `test_WF4_end_to_end_matches_validate_phase5` (44 s → 18 s): the
    sweep covered 4 × 2 grid points but the reference values only
    pinned 3 × 1 of them — dropped `y=2.5` and `t_final=1.0`.

- **`integrate_diagrams` default `n_jobs` flipped from `1` to `-1`.**
  The guard `len(diagram_terms) <= 2 → sequential fallback` already
  prevents joblib's ~1 s startup from dominating on trivial batches,
  so the new default is a free speedup for any order-2+ batch
  evaluation.  Explicit `n_jobs=1` callers (determinism-comparison
  tests like C5) are unaffected.

### Cleanups

- Removed **dead `System.explicit_C` field** (declared but never read
  anywhere in the codebase; replaced by the `c_closed_form` kwarg on
  `System.propagators()` and `Propagators.build()`).  Docstring for
  `System.explicit_R` now points to `ExplicitR` as the preferred
  structured alternative.
- Removed 10 unused imports flagged by ruff F401 across
  `diagrams.py`, `drawing.py`, `evaluate.py`, `perturbation.py`,
  `vertices.py`, `workflow/expansion.py`, `workflow/result.py`,
  `workflow/specs.py`.
- Removed 3 orphaned local assignments flagged by F841
  (`evaluate.py:ids`, `perturbation.py:sum_idx_dims`,
  `simplify.py:ref_flipped`).

### Features

- **Time-dependent linear operator** (`workflow/specs.py::DiagonalA`).
  `gamma` may now be a callable ``γ(t) -> np.ndarray(shape=(N,))``;
  the wrapper pre-computes ``Γ_a(t) = ∫_0^t γ_a(τ) dτ`` on a grid
  and caches it as a cubic spline for O(1) R lookups.  Diagonal
  case only — full-matrix time-dependent A still requires
  `ExplicitR` with a user-supplied time-ordered matrix exponential.
  New tests in `tests/test_diagonal_A_time_dependent.py`.

- **Dynamic coupling evaluation for non-local vertices**
  (`evaluate.py::DynamicCouplingPromise` +
  `perturbation.py::DiagramTerm.build_integrand`).  `coupling_values`
  now accepts callables (`fn(n_list, t_list) -> (N,)*m tensor`) for
  spacetime-dependent κ^{(n)}.  Demo2's FK channel is now a one-line
  `expansion.evaluate(vertex_types={'FK'})` — no bespoke integrator.
  See `examples/demo2/validate_FK_dynamic.py`.

- **Observable convention kwarg** `integrate_over` on
  `integrate_moment` / `Expansion.evaluate` / `.sweep` — per-external
  control over fixed-time (`None`, default, matches ⟨φ(t_f) φ(t_f)⟩
  physics convention) vs time-integrated (`"all"` or a subset).
  The previous default was time-integrated; callers that want that
  now pass `integrate_over="all"` explicitly.

- **L2 — YAML config + CLI**.  `sft-wick run config.yaml` executes
  an end-to-end workflow (expand → propagators → sweep → output);
  `--override key=value` patches fields without editing the file.
  `c_closed_form_module` and vertex `coupling_module` fields wire
  user Python modules in.  Full round-trip tests in
  `tests/test_workflow_config.py` (CF1–CF5).  Example configs:
  `examples/demo1_config.yaml`, `examples/demo2_config.yaml`.

- **L1 — high-level workflow API**: `System` (physics spec),
  `Expansion` (diagram-level inspection + per-channel integration),
  `Propagators` (thin cache wrapper), `Result` / `SweepResult`
  (pandas-backed structured output).  Bare F/κ^{(n)} tensors —
  wrapper applies MSR factors `-(i^n)/n!` automatically.
  `expansion.by_vertex_type(order)` classifies diagrams by
  F/K composition (demo2-style FF/FK split).  New tests
  `tests/test_workflow.py` (WF1–WF5).

- **Parallelised lazy propagator spline build** (`evaluate.py`).
  `_LazyTimeSplineCache._build` now uses `joblib.Parallel` when
  `n_jobs != 1` (previously silently ignored for lazy mode).
  `examples/demo1/validate_R_C_derivation.py` went from 400 s → 39 s
  on M-series (10.3× speedup) with the `n_jobs=-1` plumbing.

### Bug fixes

- **`DiagramIntegrand.evaluate` scalar path now honours
  `fixed_indices`** (evaluate.py:1322, 1424).  Previously
  `_resolve_component(il, {})` was passed `{}` instead of
  `self.fixed_indices`, so observable component labels (`'a'`,
  `'b'`) failed to resolve and the code summed over all
  components — an N-factor overcounting at order 0.  The
  vectorised path already handled `fi` correctly; this brings the
  scalar path in sync.

### Bug fixes

- **Fixed incorrect C-propagator component resolution in numerical integration**
  (`evaluate.py`).  When `build_integrand()` was called with `fixed_indices`
  (e.g. `{'a': 1, 'b': 1}` for selecting a specific correlator component),
  the fixed index names (like `'b'`) were not passed through to the QMC/GL
  integration routines.  `_resolve_component('b', idx_map)` failed to
  resolve the index, causing the integrator to **sum over all field
  components** instead of selecting the fixed one.  For `N_comp = 2` with
  `iso_C`, this produced a factor-of-2 overcounting per unresolved
  C-propagator index, making perturbative corrections ~1.3–1.8× too large
  depending on the diagram topology.

  **Fix:** `DiagramIntegrand` now stores the `fixed_indices` dict and
  both `integrate_two_point_qmc` and `integrate_moment_qmc_vectorized`
  merge it into the index map before calling `_resolve_component`.

  Affected files:
  - `sft_wick/evaluate.py` — added `fixed_indices` field to
    `DiagramIntegrand`; updated `integrate_two_point_qmc` and
    `integrate_moment_qmc_vectorized` to use `ig.fixed_indices`.
  - `sft_wick/perturbation.py` — `build_integrand()` now passes
    `fixed_indices` to the `DiagramIntegrand` constructor.

- **Fixed coupling symmetry bug in `compute_moment_numerical`**
  (`perturbation.py`).  The canonical grouping of component routings
  reused reference propagators and static coupling symbols, causing
  asymmetric coupling tensors (e.g. `F[1,0,1]=1, F[1,1,0]=0`) to
  produce wrong results.  Symmetric tensors were unaffected.

  **Fix:** Each component routing now produces its own `DiagramTerm`
  with routing-specific propagators.  This removes the incorrect
  canonical grouping but increases the number of diagram terms.
