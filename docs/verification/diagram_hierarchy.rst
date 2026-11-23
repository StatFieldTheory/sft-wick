Diagram and moment hierarchy timings
====================================

These timings apply to one site with unequal diagonal rates, independent
additive Gaussian forcing and a quadratic hub coupling. The coefficient
is a connected cumulant of component zero with all legs at the end of
the time window. For colored drivers the window is half the temporal width.
The benchmark uses direct built-in covariance and Gauss–Legendre for the
diagrams, and an independent polynomial moment solver for the hierarchy.

Each solver, cell and repeat uses a fresh serial process. Solver imports
precede the clock for model setup and numerical evaluation. The table
shows the ratio of the two medians for this combined time. Ratios above
one favor diagrams on this statistic. A crossover bracket requires a
reversal between two cells. At each end of the bracket the two solver
repeat ranges are disjoint.

Of 96 cells, 94 passed the numerical checks
and 46 of those had a nonzero coefficient. The largest paired
relative difference was 9.75e-12. The two remaining
cells are Gaussian four-point cumulants at second order with physical
dimensions 8 and 12. The hierarchy reached the 25-second worker limit
during calibration. At D = 8 the six-mode calculation completed and its
eight-mode refinement timed out. At D = 12 the six-mode calculation
timed out. Both diagram calculations completed.

.. list-table:: Hierarchy time / diagram time
   :header-rows: 1
   :widths: 14 9 9 15 15 15 15

   * - Temporal kernel
     - Order
     - Legs
     - D = 2
     - D = 4
     - D = 8
     - D = 12
   * - White
     - 2
     - 2
     - 0.103
     - 0.103
     - 0.166
     - 0.261
   * - White
     - 1
     - 3
     - 0.25
     - 0.279
     - 0.368
     - 0.537
   * - White
     - 2
     - 4
     - 0.0234
     - 0.0244
     - 0.0447
     - 0.084
   * - OU
     - 2
     - 2
     - 0.147
     - 0.33
     - 1.35
     - 3.12
   * - OU
     - 1
     - 3
     - 0.34
     - 0.517
     - 1.29
     - 2.44
   * - OU
     - 2
     - 4
     - 0.0421
     - 0.128
     - 0.735
     - 2.06
   * - Matérn-3/2
     - 2
     - 2
     - 0.171
     - 0.721
     - 3.52
     - 8.1
   * - Matérn-3/2
     - 1
     - 3
     - 0.377
     - 0.775
     - 2.53
     - 5.66
   * - Matérn-3/2
     - 2
     - 4
     - 0.0765
     - 0.391
     - 2.09
     - 5.52
   * - Gaussian
     - 2
     - 2
     - 2.3
     - 12.2
     - 61.9
     - 162
   * - Gaussian
     - 1
     - 3
     - 3.32
     - 12.2
     - 48
     - 122
   * - Gaussian
     - 2
     - 4
     - 3.84
     - 26
     - Unavailable
     - Unavailable

Observed crossovers
-------------------

* OU, order 2, 2 legs, between effective hierarchy dimensions 8 and 16.
* OU, order 1, 3 legs, between effective hierarchy dimensions 8 and 16.
* OU, order 2, 4 legs, between effective hierarchy dimensions 16 and 24.
* Matérn-3/2, order 2, 2 legs, between effective hierarchy dimensions 12 and 24.
* Matérn-3/2, order 1, 3 legs, between effective hierarchy dimensions 12 and 24.
* Matérn-3/2, order 2, 4 legs, between effective hierarchy dimensions 12 and 24.

The effective hierarchy dimension is :math:`D(1+K)` with :math:`K` real
auxiliary states per component. The table uses the timed hierarchy
resolution. A method can lead at all measured dimensions for a given
coefficient, in which case the grid establishes no crossing for it.
Repeat ranges overlap in cells where these measurements resolve neither
method as faster. Different observation times, multiple sites, dense
drift, multiplicative noise and higher orders need their own timings.

Run from the development checkout to reproduce the grid.

.. code-block:: bash

   python examples/benchmarks/diagram_hierarchy.py --full --timeout 25 --output /tmp/cumulants.json

The result records every calibration and timing sample, the selected
resolution, wall and compute times, peak resident memory and any worker
that exceeded its time limit. The compute ratios above were measured on
arm64 macOS 26.5 with Python 3.14.3, NumPy 2.4.2 and SciPy 1.17.1.
