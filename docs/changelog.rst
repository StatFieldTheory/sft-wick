Changelog
=========

.. note::

   The authoritative changelog is `CHANGELOG.md
   <https://github.com/StatFieldTheory/sft-wick/blob/main/CHANGELOG.md>`_ at
   the repository root, which records every change together with its
   measured size.  This page is a condensed summary of it; where the two
   differ, the Markdown file is correct.


Version 0.4.2 (unreleased)
--------------------------

*Paper assets and documentation only — no* ``src/`` *changes, no behaviour
change.*  Both items are corrections to claims made in the 0.4.0 and 0.4.1
entries themselves.

Fixed
~~~~~

- **The Table 1 paper asset did not run on the version the paper cites.**
  ``examples/paper_assets/table1/generate_table1.py`` built its odd-order
  observable as ``phi_a(x) phi_b(x) phi_c(y)`` — two externals sharing the
  spatial label ``x``.  0.4.0 made exactly that spelling a ``ValueError``
  at interacting orders, so from 0.4.0 onward the script raised at orders 1
  and 3.  With a distinct label per external it runs, and the diagram
  counts change: order 1 goes from 4 to **6**, order 2 is unchanged at 6
  (``m = 2`` already used distinct labels), and order 3 goes from 75 to
  **80**.  This is the same collapse the 0.4.0 entry documents — same-label
  externals lose the sum over assignments of externals to legs.  The script
  no longer hardcodes any count, and the standalone TikZ diagrams and the
  matplotlib rendering are regenerated.  The wall-clock column is not
  comparable across this change and is still to be re-measured.

- **Two figures were misquoted from demo 3's own stored results.**  The
  level-A ``m = 4`` agreement is **6.6e-16**, the maximum over the nine
  times in ``level_a_results.npz`` — 0.4.0 and 0.4.1 quoted 4.9e-16, which
  is the ``t = 3`` cell.  The ETDRK2 ``O(Δt²)`` convergence ratios span
  **4.0–6.9**, not the 4.0–5.1 quoted in both; the 6.9 cell is
  noise-dominated rather than anomalous convergence.  Neither affects a
  released number.


Version 0.4.1 — 2026-09-02
--------------------------

*Test-suite hardening only — no* ``src/`` *changes, no behaviour change.*

Fixed
~~~~~

- **Two assertions that could not fail.**  ``pytest.approx(x, rel=...)``
  compares against ``max(rel * expected, abs)`` with ``abs`` defaulting to
  **1e-12**, so wherever the compared quantity is small the floor — not the
  written ``rel`` — is what is enforced.  This cannot be read off the
  source, since it depends on each site's runtime magnitude;
  ``tools/approx_audit.py`` measures it during a normal pytest run.  Over
  207 runtime sites: 108 have ``rel`` genuinely in force, 29 pass an
  explicit ``abs=``, 15 compare against ``0.0``, **22 are weakened**, and
  **2 were vacuous** — one in ``tests/test_demo3_shot_noise.py`` (shipped
  in v0.4.0) and one in ``tests/test_closed_form_dispatch_boundaries.py``.
  In both the code was independently verified correct and only the test was
  empty, so no released number is affected: the suite's *coverage* was
  overstated, not its results.  The replacements are checks, not
  relaxations.

Changed
~~~~~~~

- :doc:`/verification/catalog` carries the measured breakdown above and
  records that the two vacuous rows were **not enforced in v0.4.0**.
- ``tools/approx_audit.py`` is kept in the repository rather than treated
  as a one-off, so the measurement can be re-run in one command.

The 22 weakened sites are documented but not fixed, and no lint guard yet
stops new bare ``rel=`` sites appearing.


Version 0.4.0 — 2026-09-02
--------------------------

*This is the version the CPC paper refers to.*  v0.3.0 predates the exact
order-4 calculation in demo 2 and all of demo 3, so neither demo's headline
result can be reproduced against it.

.. warning::

   **Breaking: external operators may no longer share a spatial label.**
   ``("phi_a(x)", "phi_b(x)")`` — the natural spelling of an equal-point
   correlator — now raises ``ValueError`` at both ``System.expand`` (L1)
   and ``compute_moment`` (L0), **at interacting orders only**.  It
   previously returned a number, correct at order 0 and wrong once vertices
   are present: measured on demo 2's system, the order-2 ``F`` channel was
   low by a factor 2 while the order-2 ``FK`` channel was exactly right.

   Order 0 is exempt, and the exemption is measured rather than assumed.

   **Migration** — give each external a distinct label and set them to the
   same point through ``positions``:

   .. code-block:: python

      system.expand(("phi_a(x)", "phi_b(y)"), orders=[0, 2])
      expansion.evaluate(props, positions={"x": 0.0, "y": 0.0}, ...)

   Coincident external *points* are, and always were, fully supported; it
   is the shared *label* that is not.  What the collapse loses is not a
   factor but a sum over the assignments of external operators to legs,
   each with its own component-index routing, so a multiplicity "fix" would
   have replaced a silent wrong answer with a subtler one — which is why
   the spelling is refused rather than repaired.

Added
~~~~~

- **Propagator-indexed dynamic couplings.**  A callable ``κ^(m)`` whose
  contraction against the surrounding Wick structure leaves a surviving
  component index on a C propagator used to raise ``NotImplementedError``;
  the integrators now contract it against the C-propagator product one
  index assignment at a time.  This is what blocked the exact evaluation of
  demo 2's order-4 ``F³κ³`` channel (30 diagrams).  Every path that worked
  before returns bit-identical numbers.

- **Demo 3 (filtered Poisson noise)** — ``examples/demo3/``, a third worked
  example whose driving field is a shot process, so Campbell's theorem
  gives every cumulant in closed form.  The one-point law's shape depends
  on ``n = ν σ_t σ_x`` alone, the R-contracted vertex is closed form at any
  ``m``, and with ``F = 0`` the series terminates: the package reproduces
  the closed form to **1.4e-16** at ``m = 3`` and **6.6e-16** at ``m = 4``.
  The reference simulation has no discretisation error of any kind.

Fixed
~~~~~

- **Single-component systems with a callable coupling.**  ``n_components =
  1`` plus a callable (dynamic) coupling raised ``ValueError: input operand
  has more dimensions than allowed by the axis remapping`` on every
  integrator.  A static tensor at ``N = 1`` was unaffected, as was
  ``N ≥ 2``.

- **The** ``auto`` **C-quadrature choice was decided by a wall-clock race.**
  ``PropagatorCache._gl_is_cheaper`` timed one Gauss-Legendre call against
  one ``dblquad`` call and took the winner, so the resolved
  ``Propagators.c_source`` and spline table depended on machine load.  Both
  rules are verified converged before the choice is made, so the race could
  never produce a wrong value; the decision is now a function of the inputs
  alone (prefer Gauss-Legendre whenever a converged node count exists).
  The change is bounded by the two rules' agreement — 9.1e-16 at
  ``t_max = 3`` to 3.6e-09 at 100, all inside the 1e-8 selection tolerance
  — and no demo can be affected at all, since they set
  ``c_closed_form_only``.

Changed — numbers move
~~~~~~~~~~~~~~~~~~~~~~

- **Demo 2's order-4** ``F³κ³`` **channel is computed, not estimated.**
  Over the 18 times at ``r = 0`` this takes χ² of (simulation − theory)
  from **340.7 to 44.2**, the mean pull from +3.31 to +1.09, and the
  largest residual from 9.36e-05 to 3.88e-05.  The residual's perturbative
  order is now measured rather than asserted (a coupling-amplitude scan
  fits a pure ``s³`` law), ``FFFF`` moved from Sobol QMC — 46 % scatter
  across seeds — to Gauss-Legendre, the Monte-Carlo error bars in
  ``reproduce_figures.py`` used a formula for the wrong quantity and now
  use the Isserlis estimate, and off-grid separations no longer bias the
  simulation high.  ``tests/test_demo2_kernels.py`` is new: nothing in
  ``tests/`` pinned a demo-2 number before.

- **Demo 1's order-4 channel was resting on one QMC seed** — 39 % scatter
  across six seeds, with the published value 69 % high at ``r = 0.5,
  t = 15`` against the converged Gauss-Legendre value.  Order 2 was not
  converged at the largest times either (+20 % at ``t = 100``).  The sweep
  is now ``gauss_legendre`` with ``n_gauss: 24``; summed over orders, the
  plotted ``ξ_ab(r, t)`` moves by at most **1.67 %**.  Full convergence
  tables in ``examples/demo1/L2/INTEGRATION_ERROR.md``.

Documented
~~~~~~~~~~

- **Unstable cache keys for callable spec fields.**  ``cache_path`` keys
  the symbolic expansion and the propagator table on ``repr()`` of the
  spec, and ``repr()`` of a plain function embeds its memory address, so a
  callable passed to ``CustomKernel``, ``GeneralKappa2``,
  ``CustomImpulse``, ``ExplicitR`` or ``DiagonalA.gamma`` changes the key
  every process and never hits the cache.  A small frozen dataclass with
  ``__call__`` fixes it.  No behaviour change: an unstable ``repr`` can
  only cause a cache **miss**, never a wrong hit.  Vertex couplings are
  unaffected — they never enter the key.


Version 0.3.0 — 2026-09-02
--------------------------

*Referee revision for CPC round 1* — a fast default path, progress
reporting and a quick start — plus the fixes that had accumulated on
``main`` since July.

.. warning::

   **Numbers move.**  Several fixes below change results that were
   previously wrong.  If you have pinned values, re-pin them against the
   corrected output rather than loosening tolerances.  The two that move
   the most output are the causal **lower** bounds (any observable with an
   external response leg) and the **C-table diagonal** fix (every tadpole,
   i.e. every interacting order).

Added
~~~~~

- **Built-in closed-form C** (``sft_wick.workflow.closed_forms``) for a
  diagonal constant drift driven by separable translation-invariant noise
  with an exponential temporal kernel — the demo 1, demo 2 and README
  family.  Selected automatically by ``propagators.c_closed_form: auto``;
  ``null`` forces quadrature.  Validated at the dispatch boundary rather
  than at convenient points, agreeing to **1e-10** against Gauss-Legendre
  and **1e-8** against ``dblquad``.

- **A faster default C quadrature** — ``c_method`` moved from
  ``dblquad`` to ``auto``, with the node count
  chosen by ``select_gl_node_count``.  Separable kernels build one temporal
  table and rescale it per separation instead of redoing the quadrature
  grid for every distinct ``r``.  The ``dblquad`` path now splits the
  rectangle at the ``λ₁ = λ₂`` cusp, which makes it both faster and more
  accurate — numbers on that path move by up to ~2e-6 relative, toward the
  exact value.

- **Progress reporting** (``sft_wick.progress``) for expansion, C-table
  build and sweep, through ``tqdm`` when installed and plain stderr lines
  otherwise; CLI ``--quiet`` and ``SFT_WICK_PROGRESS=0|1``; and
  ``progress=True|False|callable`` on the L1 entry points.  It never
  changes a number.

- **A cost estimate before committing to a run** — ``sft-wick run
  --dry-run`` reports diagrams per order, grid points, the resolved C
  source and a rough wall-clock.  ``examples/quickstart.yaml`` and
  ``sft-wick quickstart`` give a run that finishes in seconds.

- **Validation catalogue** :doc:`/verification/catalog`, generated by
  ``tools/gen_test_catalog.py`` from the collected suite and kept current
  by ``tests/test_catalog_current.py``.

- **The fixed-point driver** ``sft_wick.selfconsistency``, which never
  returns a bare state: the result carries ``converged``, the residual
  history and a ``reason`` in ``converged / diverged / oscillating /
  max_iter``.  The Dyson solve is deliberately left to the caller.

- **Disorder-averaged (spectral) propagators**, ``sft_wick.spectral`` —
  built as superpositions of Ornstein-Uhlenbeck propagators over a spectrum
  and evaluated exactly rather than tabulated (tabulating ``C`` would
  reintroduce the diagonal ridge).  ``propagators_from_cache`` lets the L1
  workflow be driven by such a hand-built cache.

- **Two-time observables from the declarative API**:
  ``external_times_grid`` on ``Expansion.sweep``, ``external_times`` on
  ``Expansion.evaluate`` and the integrators, and ``"psi_b(y)"`` as a
  nameable response external.  Every integrator previously pinned all
  fixed externals at a single time, and since Θ kills an R joining two
  externals at the same time, **every observable with an external response
  leg came back identically 0**.  Omitting the new arguments reproduces the
  old numbers bit-for-bit.

- **Callable couplings with matrix-valued response propagators**, which
  no backend could compute before, and time-dependent (callable) *local*
  couplings.

Fixed
~~~~~

- **Causal lower bounds from external response legs were dropped
  entirely.**  Every bound-builder discarded an ordering whose earlier
  endpoint is external — exactly the orderings an external ψ leg creates —
  so the O(g) response was integrated over ``[t_min, t_x]`` instead of
  ``[t_y, t_x]``: up to 5× wrong.  Bounds are now transitively closed and
  inverted intervals clamped.

- **The C table did not converge on its diagonal.**  ``C(t₁, t₂)``
  integrates up to ``min(t₁, t₂)``, so ``∂C/∂t₁`` jumps across
  ``t₁ = t₂``, and a tensor-product spline is C² and cannot represent that
  ridge: **22.3 % relative error at n_grid = 41, still 21.4 % at 321** —
  and *every tadpole* evaluates ``C(s, s)``, exactly on it.  The grid's own
  diagonal entries are now harvested into a separate interpolator.  This
  changes every interacting order.

- **The C builder used only the diagonal of a matrix R** instead of the
  matrix triple product ``R κ Rᵀ`` — 57 % Frobenius error for a dense drift,
  with wrong signs on 6 of 9 matrix elements.  ``iso_R=True`` results are
  bit-for-bit unchanged.

- **Complex→real projection sites silently destroyed signs.**  They took
  ``abs()`` of a complex value or ``.real`` of an imaginary one, so the
  sign (or the value) of any diagram whose observable carries external
  response legs was lost.  All now route through a checked projection that
  raises rather than guessing.

- **Retardation (Θ) is applied at diagram evaluation**, so an order-0
  response no longer returns the unbounded acausal form, and equal times
  give 0 rather than 1.

- ``integrate_two_point_qmc`` ignored spatial separation at order 0 and, at
  higher orders, ignored a spatial C table.

- Assorted cache and container defects: the C-value memo keyed ndarray
  arguments on ``id()`` and lost an ndarray subclass's identity, froze its
  entry in place rather than copying, and rode into every parallel worker;
  ``C_diagonal`` was position-blind when a spatial table was present (~38 %
  disagreement with ``C_value``); ``evaluate_at`` ignored
  ``coupling_vectorized=True``.

Demo 2 (non-Gaussian noise) corrections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **The third cumulant was missing its** ``α³`` **term.**  ``κ⁽³⁾``
  includes the connected three-point function of ``η² − λ``, 2.4 % of the
  leading term at coincidence, which was absent from ``k3_coupling.py``.
  The FK channel of ξ₀₁ moves by +1.2 % (t = 1) to +0.6 % (t ≥ 3).

- **The FK quadrature was not converged beyond t ≈ 10.**  The raw kernel is
  narrow in the relative leg times; the remedy is the package's own
  ``already_R_contracted`` vertex (``examples/demo2/k3_R_coupling.py``),
  which integrates the three legs inside the callable and makes the FK
  integral one-dimensional.

- **ξ₀₀ / ξ₁₁ used the single-kernel** ``λ_eff`` **approximation.**  The exact
  effective covariance of ``η̃`` is ``λ k + 2α²λ² k²``, whose second piece
  has half the correlation time and length, so the approximation
  over-counts its contribution to C by 70 %.  The budget now uses the exact
  form.

- **The fourth cumulant enters ξ₀₀ / ξ₁₁ but not ξ₀₁.**  The Gaussian
  theory with F has a ``φ₁ → −φ₁`` symmetry that only odd cumulants break,
  so ``κ⁽⁴⁾`` cannot contribute to ξ₀₁.


Version 0.2.0
-------------

*First release on PyPI —* ``pip install sft-wick``.

Builds two higher-level API layers and a numerical evaluation pipeline
on top of the L0 symbolic core.

Added
~~~~~

- **L1 workflow API** (``System``, ``Expansion``, ``Propagators``,
  ``Result``, ``SweepResult``) — immutable, high-level interface for
  specifying a physics system and expanding / evaluating field moments,
  with single-point ``evaluate`` and grid ``sweep``.
- **L2 YAML + CLI** (``sft-wick run config.yaml``) — fully declarative,
  reproducible workflows; ``--override key=value`` for parameter scans;
  every L1 constructor argument maps 1:1 onto a YAML field.
- **Numerical evaluation pipeline** (``evaluate.py``) — propagator
  caches, three spatial-homogeneity modes (translation / rotation /
  general), and three integrators: Sobol QMC (default), tensor-product
  Gauss–Legendre, and adaptive ``nquad``.
- **Dynamic (spacetime-dependent) couplings** for non-local vertices,
  with per-sample and vectorised contracts.
- **R-contracted non-local vertices** (``already_R_contracted=True``)
  that cut integration dimensionality for narrow-kernel ``κ^(m)``.
- Closed-form ``C`` hook (``c_closed_form``) that bypasses ``dblquad``
  for separable kernels, and a Gauss–Legendre ``C`` builder
  (``c_method="gauss_legendre"``).
- ``integrate_over`` observable convention (fixed-time 2-point
  correlator vs time-integrated moment).
- Time-dependent linear drift via ``γ(t)`` callables and a fully
  explicit closed-form ``R(t₁, t₂)`` escape hatch.
- Parallel evaluation via joblib across propagator builds, expansion,
  and sweeps.
- ``compute_moment_numerical`` using nauty graph isomorphism for
  canonical labelling.

Packaging
~~~~~~~~~

- Published to PyPI; installable via ``pip install sft-wick``.
- Continuous-integration workflow (tests on Python 3.10–3.12 plus a
  packaging check) and a Trusted-Publishing release workflow.


Version 0.1.0
-------------

*Initial release.*

- Core Wick contraction engine with MSR-optimised pairing enumeration
- Custom symbolic expression tree (no SymPy dependency)
- Exact rational arithmetic via ``fractions.Fraction``
- Local and non-local interaction vertices
- Perturbative expansion via ``compute_moment()`` to arbitrary order
- Multi-pass expression simplification pipeline
- Feynman diagram generation (``networkx.MultiGraph``)
- Matplotlib-based diagram rendering
- Configurable LaTeX output with ``LaTeXFormatter``
- Support for scalar and multi-component fields
- 46 tests covering all modules
