Verification
============

sft-wick's correctness rests on two complementary layers of evidence:

1. A **deductive** test suite (Phases 1–8 plus the later feature
   suites; every file, its reference and its collected test count are
   listed in the generated :doc:`catalog`) that pits each elementary
   transformation in the package against an independent reference —
   any failure points at a specific module.
2. Two **inductive** end-to-end demos (Gaussian and non-Gaussian
   driving) that compare the package's full pipeline to direct
   Langevin simulation over a large parameter grid.

Both layers are necessary: the deductive suite proves the machinery
is right *per step*; the demos confirm the output matches physics
on non-trivial problems.

Overview — why deductive tests prove correctness
------------------------------------------------

Most scientific software is validated **inductively**: run it on a
case where you know the expected answer (comparison to simulation, a
known limit, published numbers), see they agree.  This is a necessary
sanity check but not a proof — you have shown only that *in the
tested regime*, the answer happens to be right.  A bug could still
hide in any untested regime.

Deductive verification takes a different stance.  For each elementary
transformation the code performs — expanding products, applying
Wick's theorem, collapsing component-index sums, integrating a
diagram numerically — construct an **independent** reference and check
they agree *exactly* (or within a precisely-defined tolerance set by
the numerical method, not by convenience).  A pass at this level does
not just say "the final answer looks plausible"; it says *every
step* is doing what it should, and any future regression in a
specific step will surface as a specific failing test.

The suite has eight phases.  Phases 1–4 cover the symbolic + numerical
core; Phases 5–8 were added alongside the L1/L2 wrapper layer to cover
spatial homogeneity, white-noise components, dynamic non-local
couplings, time-dependent linear operators, and the end-to-end
workflow and YAML-config layers.

Phase 1 — Symbolic expansion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*File: ``tests/test_deductive_expansion.py``, ~80 tests, <1 s.*

The operator-level Wick engine, simplification passes,
diagram-canonicalisation, and index-routing logic are compared
against a brute-force reference (``tests/brute_wick.py``) that has
**zero imports from sft-wick** and therefore constitutes an
independent re-implementation of the same rules.

For the simplest non-trivial case — order 2, scalar / two-component
field, single cubic vertex — the brute reference enumerates all 105
candidate pairings and classifies each by vanishing reason:

=====================================  ======
ψ-ψ contractions (auto-zero)            15
R(x,x) killed by Itô                    48
Causal R-loops (DFS cycle check)        12
Surviving pairings                      30
Labelled spatial topologies              12
Canonical Feynman diagrams (after       6
vertex-relabeling)
=====================================  ======

Every one of those integers has a dedicated test that cross-checks
it against sft-wick's output.  Topics covered:

- pairing count ``(2n−1)!!`` for n=2..5 (T1)
- per-pairing vanishing rules (T2)
- operator-level Wick sum vs brute propagator-product sum (T3)
- spatial topology multiplicities (T4)
- **two independent compute_moment paths** (collect_topology=False
  vs True) produce the same Feynman topology set (T5)
- rational prefactor :math:`(-1)^n / n! \times \mathrm{multinomial}` (T6)
- response-phase :math:`(-i)^{n_R}` (T7)
- two canonical-form algorithms (nauty vs brute permutation search)
  induce the same equivalence relation (T8)
- diagram-collection soundness — no false merges, no false splits (T9)
- individual simplification passes (T10)
- non-local vertex instantiation and leg-count vanishing (T11, T12)
- FK coupling-sum collapse at N=2 and N=3 (T13, T20)
- α=0 regression between Case A and Case B (T14)
- multi-component coverage at N∈{1,2,3} with hand-derived closed
  forms for all six order-2 FF diagrams (T15–T19)
- non-iso propagator consistency (T21)
- ``apply_diagonal`` retains ``KroneckerDelta(a, b)`` for external
  observable indices so cross-pair (a != b) order-0 evaluates to 0
  under ``diag_C=True`` (regression lock for a long-standing bug
  that demo1 / FF tests didn't catch because they only exercise
  diagonal observable pairs).

Phase 2 — Propagator numerics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*File: ``tests/test_deductive_numerics.py``, ~22 tests, ~50 s.*

The closed-form :math:`C(t, t')` formula and the package's
``PropagatorCache`` are verified against domain-split
``scipy.dblquad`` references that reach 1e-12 relative precision.

- closed-form C vs adaptive-quadrature (domain-split to kill the
  cusp at :math:`\tau_1=\tau_2`, N1)
- Gauss-Legendre and trapezoidal quadrature convergence (N2)
- stationarity limit :math:`C \to \lambda / [\gamma(\gamma+1/\sigma_t)]` (N3)
- dimensional scaling in :math:`\lambda` and :math:`\gamma` (N4)
- α=0 regression for the two-kernel form (N5)
- two-kernel :math:`\kappa^{(2)}_{\mathrm{eff}}` shape at α>0 (N6)
- spline-table (``precompute_C_table``) interpolation error (N7)
- Itô flag end-to-end (N8)
- off-diagonal :math:`\kappa^{(2)}` → non-diagonal C propagator (N9)

Phase 3 — Full diagram evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*~3 tests, ~1 s.*

For the double-tadpole diagram (where the time integral factorises
into two independent 1-D integrals, each evaluable to machine
precision via ``scipy.quad``), QMC and ``scipy.nquad`` are compared
to the closed-form reference to 1e-6 relative (P1).  The QMC path
is also compared to the moment closed-form :math:`N^2 \times B^2`
within its self-reported 3σ error band (P2), and the Sobol
convergence rate is checked across a scan of ``n_samples`` (P3).

Phase 4 — Alternative-path consistency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*~7 tests, ~17 s.*

sft-wick exposes multiple paths for the same computation (a nauty-
based symbolic engine, a vectorised QMC, a parallel batch).  Each
alternative is pitted against a tested baseline — two independent
implementations producing the same answer is strong deductive
evidence that both are correct.

=====================================  ================================
Alternative                            Baseline
=====================================  ================================
``compute_moment_numerical``           ``compute_moment`` (C1)
parallel ``n_jobs=-1``                 serial ``n_jobs=1`` (C2, C5)
``integrate_moment_qmc_vectorized``    ``integrate_moment_qmc`` (C3)
``integrate_two_point_qmc``            ``scipy.nquad`` reference (C4)
dispatcher auto-selection              explicit method paths (C6)
=====================================  ================================

A benchmark-driven outcome of this phase: the default
``integrate_moment(method='qmc')`` now auto-selects the vectorised
path whenever the cache supports batch C evaluation, yielding a
**208× speedup** over the previous scalar default with bit-identical
results.

Phase 5 — Spatial coordinates (homogeneity modes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*File: ``tests/test_deductive_numerics.py``, ``TestSpatialAwareCache``
(S1–S8 + S2b/S6b) and ``tests/test_d_dim_spatial.py`` (DD1, DD2).*

The spatial caches add three homogeneity-dependent code paths —
translation (1-D or d-dim, reduces to ``r = ||x1 - x2||``),
rotation (radial + Legendre, accepts d-dim unit vectors), and
general (full 2-D, scalar-x only on the full-grid path).  Each
has its own ``precompute_C_table_*`` builder and its own spline
backend.  Phase 5 verifies that:

- translation and general give bit-identical answers on
  translation-invariant inputs (S3);
- rotation reduces to closed-form on the direct-radial kernel (S6);
- bubble-diagram cross-group C scaling
  :math:`\xi(r)/\xi(0) = e^{-r/\sigma_x}^{n_{\mathrm{cross}}}` holds
  under both eager and lazy cache builds (S5, S7);
- translation accepts d=3 vector positions and reduces them to the
  right scalar ``r`` (S2b);
- rotation accepts d=3 unit-vector inputs (S6b);
- general full-grid mode raises ``NotImplementedError`` on vector
  inputs and directs the user to lazy mode (S8);
- ``Expansion.evaluate(positions=...)`` end-to-end produces the
  same total for 1-D scalar and equivalent 3-D vector
  configurations (DD1), and is rotation-isotropic for
  translation-invariant noise (DD2).

Phase 6 — White-noise component
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*``TestWhiteNoiseComponent``, W1–W3.*

``GaussianNoise.sigma2`` adds a δ(t-t')·σ²(t) white-noise term to
κ².  Phase 6 verifies linearity (W2 — smooth + white = sum of both
evaluated separately) and that the white-noise piece is correctly
absorbed into the lazy translation spline alongside the smooth
kernel (W3).

Phase 7 — Dynamic non-local coupling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*``tests/test_workflow_config.py``, ``tests/test_dynamic_coupling.py``,
and demo2 validation.*

Non-local vertices with callable coupling tensors are routed
through :class:`~sft_wick.evaluate.DynamicCouplingPromise` inside
``integrate_moment_qmc_vectorized``. Two contracts are supported:

* ``fn(n_list, t_list) → tensor`` -- per-sample call (default).
* ``fn(n_2d, t_2d) → (n_samples, ...)`` when the user opts in via
  ``NonLocalVertex(coupling_vectorized=True)`` -- one call per
  integrand, useful for heavy callables.

Locked invariants:

* **DC1** -- prop-indexed dynamic coupling raises
  ``NotImplementedError`` (the v1 limitation).
* **WF6** -- a constant callable κ^(3) routed through the dynamic
  path produces the same FK total as the same constant tensor on
  the static fast path (rtol 1e-10).
* **WF7** -- per-sample and vectorised callables produce
  bit-identical totals (rtol 1e-10).
* Validated against demo2's FK channel to 0.48 % relative agreement
  against the hand-coded reference integrator.

Phase 8 — Time-dependent linear operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*``tests/test_diagonal_A_time_dependent.py``, T1–T4.*

``DiagonalA(gamma=callable)`` accepts a time-dependent γ(t) and
pre-computes :math:`\Gamma_a(t) = \int_0^t \gamma_a(\tau) d\tau` on a
grid as a cubic spline.  Phase 8 checks:

- callable-constant γ matches the static list input (T1, T1b);
- linear γ(t) = g₀ + g₁·t agrees with its closed-form Γ(t) (T2);
- causality R(t₁, t₂)=0 for t₁<t₂ is preserved (T3);
- end-to-end :class:`~sft_wick.workflow.System` round-trip works
  with the time-dependent path (T4).

Phases WF + CF — Workflow and YAML round-trip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*``tests/test_workflow.py`` (WF1–WF5) and
``tests/test_workflow_config.py`` (CF1–CF5).*

WF tests check that :class:`~sft_wick.workflow.Expansion.sweep`
reproduces validate_phase5.py reference values to 1e-5 relative
(WF4), that :meth:`by_vertex_type` classifies demo2's 6 FF + 2 FK
order-2 diagrams correctly (WF3), and that
:class:`~sft_wick.workflow.SweepResult.totals` has the expected
DataFrame schema (WF5).

CF tests verify that a YAML config → Python :class:`System`
round-trip produces a structurally identical system (CF1, CF2), and
that the full YAML-driven ``run_workflow`` agrees with the equivalent
L1 Python pipeline to rtol=1e-10 (CF3 — a bit-identity equivalence
check).  CF4 exercises ``--override``, CF5 covers malformed-input
errors.

What the deductive suite buys you
---------------------------------

275 passing deductive tests do **not** prove that every physical
observable is computed correctly in every regime — that is an
infinite claim no test suite can make.  But they do prove:

- Every known failure mode at the tested scale has a specific test
  catching it, so a future bug cannot silently pass through.
- Any future regression in a specific transformation produces a
  specific failing test that points at the broken module.
- The multiple alternative paths (fast, parallel, numerical) all
  return the same answer as the default, and the dispatcher picks
  the fastest compatible path automatically.
- Symbolic and numerical layers are decoupled: Phase 1 does not
  depend on Phase 2 passing and vice versa, so a numerical bug
  cannot mask a symbolic bug or vice versa.

End-to-end validation — the two demos
-------------------------------------

For full **inductive** validation against direct simulation on
physically non-trivial problems, see the two demos in
``examples/``:

**examples/demo1** — Gaussian driving
  Two-component Langevin system with quadratic self-interactions,
  Gaussian :math:`\eta` with exponential space-time covariance,
  compared against a Heun-integrated Monte Carlo simulation with
  :math:`10^5` realisations.  Perturbative expansion to order 4
  (0th, 2nd, 4th FF).  Agreement within the MC error envelope
  across the full :math:`(r, t)` grid.  Open:
  ``examples/demo1/analysis_combined.ipynb``.

**examples/demo2** — Non-Gaussian driving, non-local MSR vertex
  Same Langevin system with a quadratic deformation of the noise
  :math:`\tilde\eta = \eta + \alpha(\eta^2 - \lambda)`.  The
  resulting :math:`\kappa^{(3)}` introduces a non-local ψψψ
  interaction vertex, splitting the order-2 diagrams into
  **FF** (6) + **FK** (2) + **KK** (0, selection-rule forbidden).
  Demonstrates:

  - the non-local vertex infrastructure
    (:class:`~sft_wick.vertices.Vertex` with ``local=False``)
  - a coupling-mediated **selection rule** forcing FK to contribute
    only to the cross correlator :math:`\xi_{12}`
  - the first-principle two-kernel
    :math:`\kappa^{(2)}_{\mathrm{eff}} = \lambda \kappa_k +
    2\alpha^2\lambda^2 \kappa_k^2` closing the diagonal residual at
    the expected 1:1 level.

  Open: ``examples/demo2/analysis.ipynb``.

Running the tests
-----------------

.. code-block:: bash

   # Deductive core (fast, <10 s — Phase 1 only)
   pytest tests/test_deductive_expansion.py -v

   # Full numerical deductive suite (includes Phases 2–7, ~2 min)
   pytest tests/test_deductive_numerics.py -v

   # Workflow + config + time-dependent-A tests
   pytest tests/test_workflow.py tests/test_workflow_config.py \
          tests/test_diagonal_A_time_dependent.py -v

   # Everything (275 tests, ~3.5 min on M-series)
   pytest tests/ -v

   # Demo notebooks (minutes)
   cd examples/demo1 && python run_simulation.py --n_real 5000 && \
       jupyter nbconvert --to notebook --execute analysis_combined.ipynb

   cd examples/demo2 && python run_simulation.py --n_real 200000 --alpha 0.6 && \
       jupyter nbconvert --to notebook --execute analysis.ipynb

For the detailed per-test matrix, tolerances, and design rationale,
see :download:`deductive_verification.md <../deductive_verification.md>`
at the project root — it is the authoritative reference that test
code links back to when explaining a specific check.
