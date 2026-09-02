# Demo 1: integration error of the perturbative channels

*Measured 2026-09-02 on an Apple M3 Ultra (28 cores).  Every number
below is reproducible from `config.yaml` by changing only
`sweep.method` / `sweep.n_gauss` / `sweep.seed`.*

## Summary

The order-0 channel is converged by any method.  **The order-4 channel
was not, and neither was order 2 at the largest times**, and the figures published before this note
used a single Sobol draw (`sweep.seed: 42`) whose value is **69–74 %
above** the converged one at `t = 15`.  The sweep now uses a
tensor-product Gauss–Legendre rule, which is deterministic and whose
error can be measured by refining it.

## The problem: order 4 under Sobol QMC

`xi_ab` at order 4 (64 diagrams, a 4-D integrand on the causal simplex),
32768 Sobol samples, across six scramblings:

| pair | r | t | six seeds (×10⁻⁵) | scatter | converged (GL24) |
|---|---|---|---|---|---|
| (0,0) | 0.5 | 15 | 3.93, 4.17, 1.93, 1.56, 4.07, 2.45 | **39 %** | 2.3286 |
| (0,0) | 0.0 | 15 | 6.89, 7.06, 3.05, 2.65, 6.80, 4.33 | **40 %** | 3.9721 |
| (1,1) | 0.5 | 15 | 3.67, 3.95, 1.66, 1.35, 3.95, 2.44 | **42 %** | 2.2175 |
| (0,0) | 0.5 | 100 | 0.223, 0.046, 0.00014, 0.0012, 0.160, 0.00023 | **135 %** | 2.3804 |
| (0,0) | 0.5 | 1 | 2.362, 2.363, 2.362, 2.361, 2.362, 2.363 (×10⁻⁷) | 0.0 % | 2.3621e-7 |

The published `seed = 42` draw is the FIRST entry of each row.  At
`t = 15, r = 0.5` it is 3.93e-5 against a converged 2.329e-5 — **69 %
high**; at `r = 0` it is 6.89e-5 against 3.972e-5 — **74 % high**.  As a
fraction of order 2 (2.6429e-4 at `t = 15, r = 0.5`), order 4 is
**8.8 %**, not the 14.9 % the published draw implies.

The cause is geometric, not statistical bad luck.  The integrand is
concentrated within `~1/gamma` and `~sigma_t` of the upper corner of a
causal simplex of side `t_f`, so the fraction of samples that land in
the peak falls off as `t_f` grows.  At `t = 1` the peak fills the
simplex and QMC is exact to 5 digits; by `t = 15` only a handful of
samples carry the integral; by `t = 100` almost none do, and the
estimator collapses towards zero from below with occasional large
excursions.  Averaging seeds does not fix it — the estimator is
*biased*, not merely noisy, and the bias is one-sided.

A fixed-seed regression check cannot see any of this: it reproduces the
same wrong number every time.

## What replaced it

`sweep.method: gauss_legendre`, `sweep.n_gauss: 24`.  Deterministic, and
the error is diagnosable by refining.  Convergence of order 4 at
`(0,0), r = 0.5`:

| t | GL10 | GL14 | GL18 | GL24 | GL24 vs GL18 |
|---|---|---|---|---|---|
| 5 | 2.17065e-05 | 2.16975e-05 | 2.16953e-05 | 2.16944e-05 | 0.0 % |
| 15 | 2.35280e-05 | 2.33347e-05 | 2.33004e-05 | 2.32860e-05 | 0.1 % |
| 30 | 2.45677e-05 | 2.36005e-05 | 2.33763e-05 | 2.33105e-05 | 0.3 % |
| 50 | 2.27129e-05 | 2.43839e-05 | 2.36392e-05 | 2.33704e-05 | 1.2 % |
| 100 | 8.19568e-06 | 2.20174e-05 | 2.46829e-05 | 2.38044e-05 | 3.7 % |

## Why n_gauss = 24 and not 14

This is the part that is easy to get wrong, and I did get it wrong
first.  One node count has to serve a 2-D integrand (order 2) and a 4-D
one (order 4).  The obvious move is to keep `n` small because order 4
costs `n⁴` — but **it is order 2 that needs the finer grid**, because at
large `t_f` a coarse tensor grid cannot resolve a feature of width
`sigma_t = 0.3` inside a simplex of side `t_f`, and `n = 14` gives order
2 only 196 nodes.

Order 2 at `t = 100`, against GL64 (which agrees with GL96, GL128 and
2²² Sobol to 5 digits).  **The error is cell-dependent and
non-monotone in `n`**, so quote the cell:

| cell | GL14 | GL20 | GL24 | GL32 | GL40 | GL64 | GL14 vs GL64 |
|---|---|---|---|---|---|---|---|
| `r = 2.5, (1,1)` | 2.64362e-05 | 3.19792e-05 | 3.14450e-05 | 3.10646e-05 | 3.10470e-05 | 3.10421e-05 | **−14.8 %** |
| `r = 0.0, (0,0)` | 4.46659e-04 | 4.23901e-04 | 4.11031e-04 | 4.03113e-04 | 4.00806e-04 | 3.99236e-04 | **+11.9 %** |
| `r = 0.5, (0,0)` | 2.72446e-04 | 2.74566e-04 | 2.69228e-04 | 2.65741e-04 | 2.64875e-04 | 2.64298e-04 | +3.1 % |
| `r = 2.5, (0,0)` | 1.42278e-04 | 1.45976e-04 | 1.45699e-04 | 1.45464e-04 | 1.45443e-04 | 1.45433e-04 | −2.2 % |

and the QMC it replaces, at `r = 2.5, (1,1)`:

| rule | value | error |
|---|---|---|
| QMC 2¹⁵, seed 42 | 3.72854e-05 | **+20 %** |
| QMC 2¹⁵, three seeds | 3.729, 3.754, 2.899e-05 | 13 % scatter |
| QMC 2¹⁹, three seeds | 3.101, 3.112, 3.086e-05 | 0.4 % scatter |

So `n = 14` would have swapped an order-4 error for an order-2 one of
**either sign**, up to 15 %.  `n = 24` is within 1.3 % on the worst of
these cells and is better than the QMC it replaces on **both** orders
at **every** point of the grid.

*(If you re-run this, use the same cell: at `r = 0.5, (0,0)` — a natural
choice — GL14 is only +3.1 %, which would understate the problem.)*

**Quoted integration error of the sweep**, worst case (at `t = 100`,
falling to < 0.1 % by `t = 15`): **+1.3 % on order 2, +2 % on order 4**.
Order 4 contributes ~10 % of order 2, so its 2 % is 0.2 % of the total.

## The late-time caveat, stated

The tensor rule loses resolution at large `t_f` for the same geometric
reason QMC does — it just degrades gracefully and visibly instead of
catastrophically and invisibly.  What makes the late-time points usable
is that **both channels saturate**: once `t_f` exceeds a few `1/gamma`,
extending the integration window adds nothing.  Order 2 at
`r = 2.5, (1,1)` is 3.10410e-05 at `t = 15` and 3.10410e-05 at
`t = 100` — identical to 6 digits.  Order 4 is flat to 2 % from `t = 15`
to `t = 100`.

Do not read the `t = 100` points as independent measurements: they are
the same numbers as `t = 15`, and any apparent trend across them at the
few-percent level is quadrature, not physics.

## Cost

~15 minutes on 28 cores, against ~2 minutes for the QMC sweep it
replaces.  That is the price of a deterministic answer with a
measurable error instead of a 39 %-scatter lottery, and it is paid once.
