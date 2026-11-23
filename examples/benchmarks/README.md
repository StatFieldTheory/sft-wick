# Connected cumulant solvers

`diagram_hierarchy.py` compares Wick diagrams and an independent Itô
moment hierarchy for the same connected coefficient. The fixture has one
site, unequal drift rates, independent Gaussian forcing on each component
and a sparse quadratic vertex coupling component zero to every other
component. All observed legs are component zero at the end of a finite
window. The ratio of the window to the temporal width is (1/2) for
the colored kernels. This fixes the regime in which the measured costs
can be compared.

Run from the package checkout with its development environment active.

```bash
python examples/benchmarks/diagram_hierarchy.py --output /tmp/cumulant-smoke.json
python examples/benchmarks/diagram_hierarchy.py --full --output /tmp/cumulant-grid.json
```

The smoke run uses two physical components, second perturbative order
and a two-point cumulant. The full grid spans physical dimension
(D\in\{2,4,8,12\}), perturbative order (k\in\{1,2\}) and cumulant
order (n\in\{2,3,4\}). Each grid includes white noise, exponential
(OU), Matérn-3/2 and Gaussian temporal forcing.

## Accuracy and timing

The diagram worker uses direct built-in covariance and Gauss–Legendre
quadrature. Each order is checked against a finer rule before timing.
The hierarchy worker loads the independent polynomial generator under
`examples/reference`. White, OU and Matérn-3/2 have exact Markov
representations using zero, one and two additional states per component.
The Gaussian worker uses Gauss–Hermite quadrature on its spectral density.
Each pair of nonzero frequencies gives a stationary two-state real
oscillator. The starting mode count is chosen from a uniform finite-lag
kernel error bound, and the cumulant is checked against two more modes.
The nominal hierarchy state dimension is (D(1+K)) for (K) real
auxiliary states per component.

Each timed backend runs in its own serial process. Repeats alternate
the execution order. The output records model and solver setup, evaluation,
whole-process wall time, peak resident memory, refinement history and
paired numerical errors. Solver imports occur before the setup clock.
Pairs agree when their difference is at most (10^{-8}) times the
hierarchy coefficient plus 64 machine epsilons times the cancellation
scale. Structural zeros and timeout or nonconverged cells are retained in
the output. Compare timings only for matched nonzero cells. The per-worker
timeout can be changed with `--timeout`.
