Workflow API (L1 + L2)
======================

The :mod:`sft_wick.workflow` subpackage is the recommended entry
point for typical use.  It encapsulates the full pipeline — physics
spec → perturbative expansion → propagator cache → numerical sweep —
into five immutable types and two orchestration helpers.

Which layer to use
------------------

**Default to L2 (YAML + CLI).**  A config file is the best starting
point for any new analysis: it is reproducible, diff-able, easy to
share, identical on a laptop and a cluster, and supports quick
parameter scans via ``--override``.  The L1 Python API is the same
machinery — every YAML field maps 1:1 onto an L1 constructor
argument — so you can always drop down to Python if you need to
compose multiple systems or attach custom post-processing.

Use **L1 Python** when you need to:

- drive the sweep from inside a larger script (e.g. a grid search
  that builds dozens of systems programmatically)
- attach non-trivial pre- or post-processing that isn't expressible
  as a YAML ``output`` block
- iterate rapidly in a notebook where the YAML round-trip is friction

Drop to the **raw L0 API** (``Field``, ``Vertex``, ``Action``,
``compute_moment``, ``PropagatorCache``, ``DiagramIntegrand``) only
when you need fine-grained control over Itô prescription,
diagonal/isotropic simplification flags, or contraction engines
that the L1 defaults don't expose.

Quick start — L2 (config file)
------------------------------

A minimal but complete config covering the demo1 two-component
system at orders 0, 2, 4:

.. code-block:: yaml

   # demo1_config.yaml
   system:
     field: {name: phi, n_components: 2}
     linear: {type: diagonal, gamma: [1.0, 1.0]}
     vertices:
       - name: F
         coupling:                              # bare F tensor —
           - [[0.0, 0.0], [0.0, 1.0]]           # MSR factor applied
           - [[0.0, 0.5], [0.5, 0.0]]           # automatically
     noise:
       kappa2:
         type: separable_translation
         temporal: {type: exponential, lam: 0.05, sigma_t: 0.3}
         spatial:  {type: exponential, sigma_x: 1.0}

   expand:
     observable: ["phi_a(x)", "phi_b(y)"]
     orders: [0, 2, 4]

   propagators: {t_max: 15.0, n_grid_t: 60}

   sweep:
     positions_grid: {x: [0.0], y: [0.0, 0.5, 1.0, 2.5]}
     t_final_grid: [1.0, 15.0]
     component_pairs: [[0, 0], [1, 1]]
     n_samples: 8192
     seed: 42

   output:
     - {type: table, format: markdown, path: results.md}
     - {type: npz, path: results.npz}
     - {type: plot, x: y, hue: order, facet_col: t_final, path: result.png}

Run it:

.. code-block:: bash

   sft-wick run demo1_config.yaml                  # full pipeline
   sft-wick run demo1_config.yaml --override sweep.seed=7
   sft-wick run demo1_config.yaml --dry-run        # validate + summarize

The CLI is registered on install; ``--override key=value`` patches
any leaf field (safe scalar coercion — no ``eval``).  Larger
examples exercising non-local vertices, closed-form :math:`C`
hooks, and dynamic couplings live in ``examples/demo1_config.yaml``
and ``examples/demo2_config.yaml``.

Quick start — L1 (Python)
-------------------------

Same workflow written in Python — use this when embedding in a
larger script:

.. code-block:: python

   import numpy as np
   import sft_wick as sw

   F = np.zeros((2, 2, 2))
   F[0, 1, 1] = 1.0
   F[1, 0, 1] = F[1, 1, 0] = 0.5

   system = sw.System(
       field=sw.FieldSpec("phi", n_components=2),
       linear=sw.DiagonalA(gamma=[1.0, 1.0]),
       vertices=[sw.LocalVertex("F", coupling=F)],   # bare F
       noise=sw.GaussianNoise(kappa2=sw.SeparableTranslation(
           temporal=sw.ExponentialTemporal(lam=0.05, sigma_t=0.3),
           spatial=sw.ExponentialSpatial(sigma_x=1.0),
       )),
   )

   expansion = system.expand(("phi_a(x)", "phi_b(y)"),
                             orders=[0, 2, 4])
   props = system.propagators(t_max=15.0, n_grid_t=60)

   sweep = expansion.sweep(
       props,
       positions_grid={"x": [0.0], "y": [0.0, 0.5, 1.0, 2.5]},
       t_final_grid=[1.0, 15.0],
       component_pairs=[(0, 0), (1, 1)],
   )

   print(sweep.totals())    # long-format pandas DataFrame

.. warning::

   **Every external operator needs its own spatial label.**  Write the
   equal-point correlator as ``("phi_a(x)", "phi_b(y)")`` with
   ``positions={"x": 0.0, "y": 0.0}`` — *not* as
   ``("phi_a(x)", "phi_b(x)")``, which raises ``ValueError``.

   The label is the name of an integration/external coordinate, not the
   point itself: coincident *points* are fully supported, and every demo
   uses them.  A shared *label* is not, because the spatial contraction
   is keyed by label, so operators sharing one are collapsed without
   their distinct component-index routings.  Before 0.3.1 that produced
   a silently wrong number at interacting orders (a factor 2 in demo2's
   order-2 ``F`` channel); it is now refused.

The five headline types
-----------------------

:class:`~sft_wick.workflow.System` — physics spec
   Combines field, linear operator (for :math:`R`), noise (for
   :math:`C`), and vertex lists.  Methods ``expand`` and
   ``propagators`` lower to the computational layer.

:class:`~sft_wick.workflow.Expansion` — diagram-level view
   What ``compute_moment`` returned, wrapped for inspection:
   ``diagrams(order)``, ``summary()``, ``by_vertex_type(order)``.
   Also the numerical entry points ``evaluate`` (single point) and
   ``sweep`` (grid).

:class:`~sft_wick.workflow.Propagators` — cache wrapper
   Thin wrapper over :class:`~sft_wick.evaluate.PropagatorCache`.
   Pass a ``c_closed_form=`` hook to ``System.propagators`` when a
   closed-form :math:`C(n_1, t_1, n_2, t_2)` is known — skips the
   dblquad grid build entirely.

:class:`~sft_wick.workflow.Result`, :class:`~sft_wick.workflow.SweepResult`
   Pandas-backed structured output.  ``SweepResult.totals()`` is a
   long-format DataFrame keyed by ``(x, y, t_final, a, b, order)``;
   ``SweepResult.plot(x=…, hue=…, facet_col=…)`` gives a quick
   diagnostic figure.

MSR convention — the one gotcha
-------------------------------

L1 applies the MSR prefactors for you:

- **Local** :math:`F^{(n)}`: multiplied by :math:`-i`
- **Non-local** :math:`\kappa^{(m)}`: multiplied by
  :math:`-i^m / m!` (e.g. :math:`+i/6` for :math:`m=3`)

So users pass **bare** tensors:

.. code-block:: python

   # CORRECT — pass the physical F, wrapper applies −i
   sw.LocalVertex("F", coupling=F)

   # WRONG — pre-multiplied, will double-apply the factor
   sw.LocalVertex("F", coupling=-1j * F)

The raw API (``compute_moment``, ``DiagramTerm.evaluate_coupling``)
does NOT apply these factors automatically — raw callers must pass
the pre-multiplied tensor.

Observable convention — ``integrate_over``
------------------------------------------

Both ``Expansion.evaluate`` and ``Expansion.sweep`` accept an
``integrate_over`` keyword:

``None`` (default)
    All external points fixed at ``t_final`` — the physics 2-point
    correlator :math:`\langle \varphi(t_f)\,\varphi(t_f) \rangle`.

``"all"``
    All external times integrated over ``[t_min, t_final]`` — the
    time-integrated moment
    :math:`\langle \int\!\varphi \cdot \int\!\varphi \rangle`.  Use
    for line-of-sight or weak-lensing observables.

``{"x", "y", ...}``
    Explicit subset of external-point names to integrate; the rest
    are fixed at ``t_final``.

Time-dependent linear operator
------------------------------

:class:`~sft_wick.workflow.DiagonalA` accepts either a list of N
floats (constant γ) or a callable ``γ(t) → np.ndarray(shape=(N,))``:

.. code-block:: python

   # Time-dependent γ (e.g. cosmological expansion rate)
   system = sw.System(
       field=sw.FieldSpec("phi", n_components=1),
       linear=sw.DiagonalA(
           gamma=lambda t: np.array([1.0 + 0.1 * np.sin(t)]),
           t_max_cache=20.0, n_grid_cache=400,
       ),
       ...
   )

Internally the wrapper pre-computes
:math:`\Gamma_a(t) = \int_0^t \gamma_a(\tau) d\tau` on the grid and
caches it as a cubic spline for O(1) R lookups.  Scalar closed-form
``R(t_1, t_2)`` callables that the spline-cache lowering cannot
express are supported via :class:`~sft_wick.workflow.ExplicitR`.
From YAML this is the ``linear.type: explicit`` route::

   system:
     linear:
       type: explicit
       R_time_module: ./R_time.py     # exports R_time(t1, t2)
       R_time_attr:   R_time           # default 'R_time'
       iso_R:         true             # required: YAML explicit R is scalar

The module is loaded at config-parse time and registered for
cross-process by-value serialisation, so the explicit-R callable
composes cleanly with ``propagators.n_jobs > 1``,
``expand.n_jobs > 1``, and ``sweep.n_jobs > 1`` (same machinery
``c_closed_form_module`` and ``coupling_module`` use).

Matrix-valued R is still available from the lower-level Python APIs,
where callers can choose scalar-loop integration paths explicitly.  The
L2 YAML numerical wrapper currently rejects ``linear.type: explicit``
with ``iso_R: false`` rather than silently routing a matrix R through a
scalar-only vectorised integrator.

Dynamic coupling (spacetime-dependent κ^(m))
--------------------------------------------

For non-local vertices whose coupling tensor depends on the sample
points, pass a callable to
:class:`~sft_wick.workflow.NonLocalVertex`. Two equivalent contracts
are supported.

**Per-sample contract** (default, simplest to write)::

   def k3_coupling(n_list, t_list):
       """``n_list``, ``t_list`` are length-m sequences (one entry
       per ψ-leg).  Return a shape ``(N,)*3`` tensor."""
       ...

   system = sw.System(
       ...,
       nonlocal_vertices=[
           sw.NonLocalVertex("K", order=3, coupling=k3_coupling),
       ],
   )

The runtime calls ``k3_coupling`` once per QMC sample.

**Vectorised contract** (opt-in, fast for heavy callables)::

   def k3_coupling(n_2d, t_2d):
       """``n_2d``, ``t_2d`` are shape ``(m, n_samples)`` arrays.
       Return a tensor of shape ``(n_samples,) + (N,)*3``."""
       ...

   sw.NonLocalVertex(
       "K", order=3, coupling=k3_coupling, coupling_vectorized=True,
   )

The runtime calls ``k3_coupling`` exactly once per integrand,
amortising the callable's overhead across all samples. This is the
right form when the function does heavy work that vectorises well
(special functions, ufuncs, BLAS).  For cheap functions
(`numpy.exp` of a few scalars) the per-sample contract has lower
total overhead.

The static fast path is used automatically when no callable is
passed.  Both dynamic contracts route through
:class:`~sft_wick.evaluate.DynamicCouplingPromise.evaluate_at_batch`,
which dispatches per-symbol based on the ``vectorized`` flag --
mixing both contracts on different symbols within the same diagram
is supported.

.. note::

   Spacetime-dependent callables currently support only **scalar**
   leg positions (1-D translation, or sphere-direction unit vectors
   reduced to ``cos θ``).  d-dim vector positions on the legs raise
   ``NotImplementedError`` from inside the dynamic-coupling QMC
   path; use a constant-tensor coupling for that vertex if you need
   d-dim spatial coordinates.

Equal-time (single-shell) cumulants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default :class:`~sft_wick.workflow.NonLocalVertex` contract treats
the vertex's coupling as the **full cross-spacetime** cumulant
``κ^(m)(z_1, …, z_m)`` and integrates over ``m`` independent
``(n, λ)`` arguments --- the literal MSR action
``W^(m)[ψ] = (1/m!) ∫ dz_1 … dz_m  ψ(z_1) … ψ(z_m) κ^(m)``.

Many applications (especially cosmological structure / lensing in the
single-shell / Limber-like regime) only have an **equal-shell** or
**equal-time** cumulant available, e.g. the equal-shell bispectrum
``ζ_eq(γ_12, γ_23, γ_31; λ)`` from a CCL-like pipeline.  The
action-level relation between the two is::

   κ^(m)(z_1, ..., z_m)  ≈  δ(λ_1 − λ_2) · ... · δ(λ_{m−1} − λ_m)
                              · κ_eq(n_1, ..., n_m;  λ)

Passing ``ζ_eq`` as the ``coupling`` callable *without* declaring the
δ-structure makes sft-wick faithfully sweep ``m`` independent times
over ``[t_min, t_final]``, contributing a spurious ``(t_final − t_min)^(m−1)``
factor of integration measure --- the answer is wrong by orders of
magnitude.

Use ``equal_time=True`` to declare the equal-time / single-shell
form::

   sw.NonLocalVertex(
       "K", order=3, coupling=k3_equal_shell_callable,
       equal_time=True,
   )

When set, sft-wick

* collapses the ``m`` time-integration legs of this vertex into a
  **single** integration variable (one ``∫dλ_K`` covering all ``m``
  ψ-legs of one vertex insertion);
* keeps the ``m`` spatial legs independent --- the callable still
  receives ``m`` distinct positions, because ``ζ_eq`` IS a function
  of ``m`` independent angular coordinates;
* fills the callable's ``t_list`` with the *same* time value
  replicated ``m`` times.

YAML form::

   nonlocal_vertices:
     - name: K
       order: 3
       coupling_module: kappa3_callable.py
       coupling_attr: coupling_fn
       coupling_vectorized: false
       equal_time: true

Propagator topology, R / C wiring, direction groups, and the response
phase are all unchanged by ``equal_time``; only the time-integration
Jacobian and the per-leg ``t_list`` payload differ.

When NOT to use ``equal_time``: if your callable returns the genuine
cross-spacetime cumulant ``κ^(m)(z_1, …, z_m)`` (e.g. an unequal-time
matter bispectrum with explicit linear-growth scaling
``D(λ_1) D(λ_2) D(λ_3)`` and a stationary template), keep
``equal_time=False`` (the default) --- otherwise the ``m`` independent
``λ`` integrations that the cross-spacetime form requires would be
incorrectly collapsed.

Already-R-contracted vertices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default :class:`~sft_wick.workflow.NonLocalVertex` returns the
**bare** ``κ^(m)`` at the m leg points and lets sft-wick handle every
R-propagator that lands on a leg via the Wick contraction. For
**narrow-kernel** ``κ^(m)`` (canonical case: canoes' squeezed κ³ at
``ℓ_max → ∞``, with diagonal width ``Δχ ~ χ_max / ℓ_max``), the m
leg-time integrations on the causal simplex demand prohibitively
many quadrature nodes. The trick is to recognise the math identity

.. math::

   \kappa^{(m)}_R(\gamma; z_1', \ldots, z_m')
       \;:=\; \int dz_1 \cdots dz_m \,
                \prod_{i=1}^m R(z_i', z_i) \,
                \kappa^{(m)}(\gamma; z_1, \ldots, z_m),

i.e. **fold the m R-propagators on the κ legs into the coupling
callable itself**. Pass ``already_R_contracted=True`` and supply a
callable that returns ``κ^(m)_R`` (the partner-time R-contracted
form) instead of the bare ``κ^(m)``::

   sw.NonLocalVertex(
       "K", order=3, coupling=k3_R_callable,
       already_R_contracted=True,
   )

When set, sft-wick

* tags the m R-propagators attached to this vertex's ψ legs as
  **absorbed** — they remain in the diagram graph (so direction
  groups continue to identify the leg with its partner) but
  contribute a factor of **1** instead of the usual ``R_time`` value;
* aliases each leg's time onto its Wick partner's time via the
  existing ``equal_time_aliases`` machinery, so the m leg-time
  integration variables drop out of the simplex;
* feeds the callable ``(n_list, t_list)`` where the entries are now
  the **partner (outer) spacetime points** ``(z_1', …, z_m')``,
  not the leg's own ``(z_1, …, z_m)``.

The dimensionality of the diagram-side integration drops by ``m`` per
R-contracted vertex (e.g. an order-2 F+K diagram on a 4-D simplex
becomes 1-D after absorption). Beyond the speedup, the surviving
integrand is **smoother** — the narrow-kernel peak has been folded
into the (presumably smooth) ``κ^(m)_R``, so Gauss-Legendre converges
exponentially with far fewer nodes.

YAML form::

   nonlocal_vertices:
     - name: K
       order: 3
       coupling_module: kappa3_R_callable.py
       coupling_attr: coupling_fn
       coupling_vectorized: false
       already_R_contracted: true

The same flag is also exposed at the **L0 surface** as
``Vertex(fields=[psi]*m, coupling='K', local=False,
already_R_contracted=True)`` for users driving
:func:`~sft_wick.perturbation.compute_moment` /
:func:`~sft_wick.evaluate.integrate_moment` directly.  See
:doc:`vertices_and_actions` → *Non-local vertex flags* for the L0
ergonomics.

Building the R-contracted callable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For validation, sft-wick ships a brute-force wrapper that turns a raw
``κ^(m)`` callable + an R kernel into a numerical ``κ^(m)_R`` via
tensor-product trapezoid on a user-supplied χ-grid::

   from sft_wick import build_R_contracted_callable

   k3_R = build_R_contracted_callable(
       raw_coupling_fn=raw_k3,
       R_time=R_kernel,
       chi_grid=np.linspace(0.0, t_max, n_chi),
       order=3,
       n_components=N,
       causal=True,        # use causal Heaviside on R
   )

This is slow (O(n_chi^m) raw-callable evaluations per outer point);
its purpose is to validate the dispatch against a known raw kernel.
**For production work, supply an analytical / pre-tabulated**
``κ^(m)_R`` **callable** — e.g. canoes' FFTlog-of-W chain.

Compatibility:

* ``already_R_contracted=True`` rejects ``equal_time=True`` at
  construction (vacuous combination — the R-contracted callable has
  already integrated over its leg coordinates).
* All existing knobs (``coupling_vectorized``, MSR factor, response
  phase, diagonal / isotropic simplification) work unchanged.
* The dispatch is **mathematically equivalent** to the raw path when
  both pipelines receive analytically-equivalent inputs — locked at
  machine precision (``rtol=1e-12``, observed ``rel_diff ≈ 1.8e-15``)
  by the constant-κ³ equivalence test in
  ``tests/test_R_contracted_vertex.py``. The two paths are **not
  bit-identical**: raw integrates the diagram on a higher-dimensional
  causal simplex (m extra leg-time variables per absorbed vertex), so
  the floating-point sums use different node sets and the last few
  bits of float64 necessarily differ. Convergence under refinement
  (e.g. demo2 FK at ``n_gauss=16``: ``rel_diff ≈ 7e-4``) confirms the
  dispatch on real spacetime-dependent κ³ — see
  ``docs/notes/R_contracted_nonlocal_vertex.md`` §4.1.

Design details and the equivalence-validation evidence on demo2 FK
live at ``docs/notes/R_contracted_nonlocal_vertex.md``.

d-dim spatial coordinates
-------------------------

Static (constant-tensor) couplings combined with translation- or
rotation-invariant noise accept arbitrary-dimensional position
vectors:

.. code-block:: python

   r = 0.7
   exp.evaluate(
       props,
       positions={
           "x": np.array([0.0, 0.0, 0.0]),
           "y": np.array([r, 0.0, 0.0]),
       },
       t_final=1.0,
       component_pair=(0, 0),
       orders=[0, 2],
       integrate_over="all",
   )

* **translation**: the wrapper reduces the input to ``r = ||x1 -
  x2||`` (Euclidean norm), so the cache shape stays
  ``(t1, t2, r)`` -- 3-D regardless of the ambient dimension.
* **rotation**: ``_rotation_cos(n1, n2)`` works on unit vectors of
  any dimension (it only uses ``np.dot`` and ``np.linalg.norm``).
* **general**: lazy mode supports d-dim via dict-keyed memoisation
  (one 2-D ``(t1, t2)`` spline per distinct ``(x1, x2)`` pair). The
  full-grid path raises ``NotImplementedError`` because a d-dim grid
  would inflate the spline to ``(2 + 2d)``-D, with ``n_grid_x **
  (2d)`` build calls.

YAML configuration reference (L2)
---------------------------------

The YAML schema mirrors the L1 constructor signatures one-for-one
— anything expressible in L1 Python has a YAML equivalent. This
section is the **complete reference** for every key the parser
recognises.

Top-level structure
~~~~~~~~~~~~~~~~~~~

Every config has five top-level blocks (``output`` is optional):

.. code-block:: yaml

   system: {...}        # physics spec: field, linear op, noise, vertices
   expand: {...}        # diagram enumeration: observable, orders, flags
   propagators: {...}   # C-cache build: t_max, grid, c_method, hooks
   sweep: {...}         # numerical evaluation: grid, integrator, parallel
   output: [...]        # optional: emit table / npz / plot artefacts

Annotated full example
~~~~~~~~~~~~~~~~~~~~~~

Every key the parser recognises, with its default value and a
one-line note. Drop unused keys; only the marked **required**
fields must be present.

.. code-block:: yaml

   # ============================================================
   # system  --  physics specification
   # ============================================================
   system:
     field:
       name: phi             # required: field-symbol prefix used in observables
       n_components: 2       # required: number of internal components N

     linear:
       type: diagonal        # 'diagonal' (DiagonalA) | 'explicit' (ExplicitR)
       # ---- type: diagonal -- A_{ab} = -γ_a δ_{ab} ----
       gamma: [1.0, 1.0]     # length-N constant rates  -- OR
       # gamma_module: ./drift.py        # dotted path to callable γ(t)→array(N)
       # gamma_attr: gamma_fn            # default 'gamma_fn'
       # t_max_cache: 20.0               # γ-spline cache horizon (default = propagators.t_max)
       # n_grid_cache: 400               # γ-spline node count (default ceil(t_max_cache/dt))
       # ---- type: explicit -- user-supplied scalar R(t1, t2) ----
       # R_time_module: ./R_time.py      # path to a .py exporting R_time(t1, t2) -> float
       # R_time_attr:   R_time           # default 'R_time'
       # iso_R:         true             # required for YAML explicit R

     noise:
       kappa2:
         type: separable_translation     # 'separable_translation' | 'separable_rotation'
                                         # | 'callable_module'
         temporal: {type: exponential, lam: 0.05, sigma_t: 0.3}
                                         # 'exponential' | 'gaussian'
         spatial:  {type: exponential, sigma_x: 1.0}
                                         # 'exponential' | 'gaussian'
         # angular: {type: legendre, coeffs: [1.0, 0.2]}  # separable_rotation
       sigma2: null                      # optional δ-correlated white-noise floor
                                         # 'constant'        -> {type: constant, amplitude: 0.01}
                                         # 'callable_module' -> {type: callable_module,
                                         #                       module: ./sigma2.py, attr: sigma2}
                                         # callable signature: sigma2(n1, lam, n2) -> (N, N)

     vertices:                           # zero or more local F vertices (bare F tensor)
       - name: F
         coupling: [[[0.0,0.0],[0.0,1.0]], [[0.0,0.5],[0.5,0.0]]]
         # OR
         # coupling_path: ./F.npy        # path to .npy file (relative to YAML)
         # OR (for spacetime-dependent F)
         # coupling_module: ./F_dynamic.py
         # coupling_attr:   coupling_fn
         # coupling_vectorized: false

     nonlocal_vertices: []               # zero or more non-local κ^(m) vertices
       # - name: K
       #   order: 3                      # required: m, the number of ψ-legs
       #   coupling_module: ./k3.py
       #   coupling_attr:   coupling_fn  # default 'coupling_fn'
       #   coupling_vectorized: false    # set true if attr is the batched contract

     t_min: 0.0                          # lower bound for time integration domain

   # ============================================================
   # expand  --  perturbative diagram enumeration
   # ============================================================
   expand:
     observable: ["phi_a(x)", "phi_b(y)"]   # required
     orders:     [0, 2]                     # required
     response_phase:    true     # multiply by (-i)^n_response (MSR)
     ito:               true     # Itô prescription (R(x,x) = 0)
     collect_topology:  true     # spatial-level Wick contraction (faster, default)
     diag_R:            true     # diagonal-R simplification
     diag_C:            true     # diagonal-C simplification
     iso_R:             null     # null = same as diag_R; true = also strip indices
     iso_C:             false
     cache_path:        null     # optional dir/file for joblib expansion cache
     n_jobs:            1        # parallel diagrams per grid point (>1 forbids sweep.n_jobs>1)

   # ============================================================
   # propagators  --  C-cache build
   # ============================================================
   propagators:
     t_max:              50.0    # required: upper time bound
     n_grid_t:           60      # time-grid resolution (or set dt instead)
     # dt:               0.5     # alternative single-knob: n_grid_t = ceil(t_max/dt)

     # spatial-cache modes (all optional; lazy mode if unset)
     homogeneity:        null    # auto-inferred from noise.kappa2.type
     r_max:              null    # translation full-grid: |x1-x2| upper bound
     n_grid_r:           null
     n_grid_cos:         null    # rotation full-grid: cos θ resolution
     x_max:              null    # general full-grid: positions hypercube edge
     n_grid_x:           null

     # closed-form C hook (machine precision; bypasses the cache)
     c_closed_form_module:    null   # ./c_closed_form.py
     c_closed_form_attr:      C_fn   # callable name in the module
     c_closed_form_only:      false  # skip spline cache entirely
     c_closed_form_vectorized:false  # c_fn accepts (n,) arrays, returns (n, N, N)

     # quadrature for the inner ∫ R κ² R when no closed form is given
     c_method:           dblquad      # 'dblquad' | 'gauss_legendre'
     c_n_gauss:          20           # GL nodes/dim under c_method='gauss_legendre'

     interp_method:      linear       # 'linear' | 'cubic'
     n_jobs:             1
     cache_path:         null         # dir for joblib cache

   # ============================================================
   # sweep  --  numerical evaluation grid
   # ============================================================
   sweep:
     positions_grid:                    # required: each list is swept independently
       x: [0.0]
       y: [0.0, 0.5, 1.0, 2.5]
     t_final_grid: [1.0, 5.0, 15.0]    # required
     component_pairs: [[0, 0], [1, 1]] # required

     # Optional: pin external points at UNEQUAL times, swept as a further
     # Cartesian axis (same shape as positions_grid).  Needed for two-time
     # observables such as R(t, t') and C(t, t') -- with every external at one
     # time, Theta kills the R joining them and any observable carrying a
     # response leg is identically 0.  Result rows gain a ``t_<point>`` column.
     external_times_grid:
       x: [4.0]
       y: [1.0, 2.0, 3.0]

     orders:           null            # optional subset of expand.orders
     vertex_types:     null            # optional subset of {"F","FF","FK","K",…}
     integrate_over:   null            # null | "all" | ["x"] | ...

     method:           qmc_vectorized  # 'qmc_vectorized' | 'qmc' | 'qmc_scalar'
                                       # | 'gauss_legendre' | 'nquad'
     n_samples:        8192            # Sobol samples (QMC paths only)
     seed:             42              # Sobol seed (QMC paths only)
     n_gauss:          8               # GL nodes/dim (gauss_legendre only)

     n_jobs:           1               # parallel over grid points
                                       # (mutually exclusive with expand.n_jobs > 1)

   # ============================================================
   # output  --  optional artefact emitters (list of plugins)
   # ============================================================
   output:
     - {type: table, format: markdown, path: results.md}
     - {type: npz,   path: results.npz}
     - {type: plot,  path: result.png, x: y, hue: order, facet_col: t_final}

Section reference: ``system``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 36

   * - Field
     - Type
     - Default
     - Notes
   * - ``field.name``
     - ``str``
     - **required**
     - Field-symbol prefix used in ``expand.observable`` (e.g. ``phi``)
   * - ``field.n_components``
     - ``int``
     - **required**
     - N — number of internal components per field
   * - ``linear.type``
     - ``str``
     - ``"diagonal"``
     - ``"diagonal"`` -> :class:`~sft_wick.workflow.DiagonalA`; ``"explicit"`` -> :class:`~sft_wick.workflow.ExplicitR`. ``ConstantMatrix`` still needs L1 Python.
   * - ``linear.gamma``
     - ``list[float]`` of length N
     - **required** unless ``gamma_module`` set (``type: diagonal``)
     - Constant decay rates :math:`\gamma_a`
   * - ``linear.gamma_module``
     - ``str`` (path)
     - ``null``
     - (``type: diagonal``) dotted-path / relative-path to a ``.py`` file exporting a callable ``γ(t) → ndarray(N,)``
   * - ``linear.gamma_attr``
     - ``str``
     - ``"gamma_fn"``
     - Attribute name in ``gamma_module``
   * - ``linear.t_max_cache``
     - ``float``
     - ``propagators.t_max``
     - (``type: diagonal``) horizon for the :math:`\Gamma_a(t) = \int_0^t \gamma_a` cumulative-integral spline
   * - ``linear.n_grid_cache``
     - ``int``
     - derived from ``dt``
     - (``type: diagonal``) spline node count for cumulative Γ
   * - ``linear.R_time_module``
     - ``str`` (path)
     - **required when** ``type: explicit``
     - Path to a ``.py`` file exporting a scalar callable ``R_time(t1, t2) -> float``. Must enforce causality (return 0 when ``t1 < t2``).
   * - ``linear.R_time_attr``
     - ``str``
     - ``"R_time"``
     - Attribute name in ``R_time_module``
   * - ``linear.iso_R``
     - ``bool``
     - ``true``
     - (``type: explicit``) must be ``true``. Matrix-valued R is a lower-level Python API feature, not an L2 YAML mode.
   * - ``noise.kappa2.type``
     - ``str``
     - **required**
     - ``"separable_translation"`` (1-D r), ``"separable_rotation"`` (Legendre angular), ``"callable_module"``
   * - ``noise.kappa2.temporal``
     - block
     - **required** (separable variants)
     - ``{type: exponential|gaussian, lam, sigma_t}`` — see :doc:`expressions`
   * - ``noise.kappa2.spatial``
     - block
     - **required** (``separable_translation``)
     - ``{type: exponential|gaussian, sigma_x}``. For ``separable_rotation``, use ``angular: {type: legendre, coeffs: [...]}`` instead.
   * - ``noise.sigma2``
     - block or ``null``
     - ``null``
     - Optional δ-correlated white-noise variance.  Two flavours: ``{type: constant, amplitude: 0.01}`` for a scalar / spacetime-independent impulse, **or** ``{type: callable_module, module: ./fn.py, attr: sigma2}`` for a user-supplied ``sigma2(n1, lam, n2) → (N, N)`` (mirrors the ``kappa2.callable_module`` pattern; the spec is wrapped via ``CustomImpulse``)
   * - ``vertices``
     - list of blocks
     - ``[]``
     - Local F vertices; see :ref:`vertex spec <vertex-spec>` below
   * - ``nonlocal_vertices``
     - list of blocks
     - ``[]``
     - Non-local κ^(m) vertices; same vertex-spec
   * - ``t_min``
     - ``float``
     - ``0.0``
     - Lower bound for time integration

.. _vertex-spec:

**Vertex spec** — one of the following must be present in each ``vertices[]`` / ``nonlocal_vertices[]`` entry to define the coupling tensor:

.. list-table::
   :header-rows: 1
   :widths: 28 50 22

   * - Form
     - Meaning
     - When to use
   * - ``coupling: [[...]]``
     - Inline nested-list tensor literal
     - Small, ≤ 4 components
   * - ``coupling_path: ./F.npy``
     - Path to ``.npy`` file (loaded as ``np.ndarray``)
     - Larger or precomputed tensors
   * - ``coupling_module: ./fn.py`` + ``coupling_attr: name``
     - Spacetime-dependent callable (per-sample contract by default)
     - When the coupling depends on leg positions/times (e.g. demo2's :math:`\kappa^{(3)}`)
   * - + ``coupling_vectorized: true``
     - Switches the loaded callable to the batched contract
     - When the callable does heavy work that vectorises well

Non-local vertices additionally **require** ``order: <m>`` (the number
of ψ-legs) and accept two specialisation flags:

.. list-table::
   :header-rows: 1
   :widths: 22 56 22

   * - Flag
     - Meaning
     - When to use
   * - ``equal_time: true``
     - Callable returns the equal-shell cumulant
       :math:`\zeta_{eq}(n_1,\ldots,n_m;\lambda)` (one shared time
       across all m legs)
     - Cosmological equal-shell bispectrum, single-shell limber
   * - ``already_R_contracted: true``
     - Callable returns the R-contracted form
       :math:`\kappa^{(m)}_R(\gamma; z_1', \ldots, z_m')` evaluated at
       the **partner** spacetime points; sft-wick absorbs the m
       R-propagators on this vertex's legs (no per-leg integration)
     - Narrow-kernel :math:`\kappa^{(m)}` where the m leg-time
       integrations are too expensive (e.g. squeezed κ³ at
       :math:`\ell_{max} \to \infty`)

The two flags are **mutually exclusive** — combining them is rejected
at construction time.

Section reference: ``expand``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 16 16 44

   * - Field
     - Type
     - Default
     - Notes
   * - ``observable``
     - ``list[str]``
     - **required**
     - e.g. ``["phi_a(x)", "phi_b(y)"]``; component letters are MSR-fixed-indices, spatial args are the keys later swept by ``sweep.positions_grid``
   * - ``orders``
     - ``list[int]``
     - **required**
     - e.g. ``[0, 2, 4]``
   * - ``response_phase``
     - ``bool``
     - ``true``
     - Multiply each diagram by :math:`(-i)^{n_R}` (MSR). Disable to inspect raw symbolic output
   * - ``ito``
     - ``bool``
     - ``true``
     - Itô prescription: :math:`R(x, x) = 0` and causal R-loops vanish
   * - ``collect_topology``
     - ``bool``
     - ``true``
     - Spatial-level Wick contraction (avoids combinatorial explosion). Disable only for debugging
   * - ``diag_R`` / ``diag_C``
     - ``bool``
     - ``true``
     - Apply diagonal-propagator simplification (collapses index sums where R/C is component-diagonal)
   * - ``iso_R`` / ``iso_C``
     - ``bool`` or ``null``
     - ``null`` / ``false``
     - Strip equal component indices from R/C entirely (treats diagonal entries as a single scalar)
   * - ``cache_path``
     - ``str`` or ``null``
     - ``null``
     - Directory or ``.joblib`` file for caching the symbolic expansion
   * - ``n_jobs``
     - ``int``
     - ``1``
     - Parallelise over diagrams within each grid point (mutually exclusive with ``sweep.n_jobs > 1``)

.. _two-time-observables:

Two-time observables: ``R(t, t')`` and ``C(t, t')``
----------------------------------------------------

The response propagator is retarded, so ``<phi(u) psi(y)> = R(u, y)`` vanishes
when its two legs sit at the same time.  A sweep pins every external at
``t_final`` unless told otherwise, which means an observable carrying a
response leg evaluates to **exactly zero everywhere**.  Two things are needed
to get a non-trivial answer:

1. Name the **response field** in the observable — ``psi_b(y)`` alongside
   ``phi_a(x)``.
2. Separate the external times with ``external_times_grid``.

.. code-block:: yaml

   expand:
     observable: ["phi_a(x)", "psi_b(y)"]
     orders: [0, 2]

   sweep:
     positions_grid: {x: [0.0], y: [0.0]}
     t_final_grid: [4.0]
     external_times_grid: {x: [4.0], y: [1.0, 2.0, 3.0]}
     component_pairs: [[0, 0]]

Each named point contributes a ``t_<point>`` column to the result rows, and
``SweepResult.totals()`` groups by them.  For the free theory the order-0 term
is the bare propagator ``exp(-gamma (t - t'))``:

.. code-block:: text

   |   t_x |   t_y |     value |
   |     4 |     1 | 0.0497871 |
   |     4 |     2 | 0.135335  |
   |     4 |     3 | 0.367879  |

Three requests are rejected rather than silently mis-answered: a time beyond
the propagator table's ``t_max`` (which would clamp to the table edge), an
empty list in the grid (which would produce no rows), and a point whose
``t_<point>`` column would collide with an existing one.  A response
observable whose externals are all at the same time warns.


Section reference: ``propagators``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 18 16 38

   * - Field
     - Type
     - Default
     - Notes
   * - ``t_max``
     - ``float``
     - **required**
     - Upper time bound. Sets the C-cache time grid
   * - ``n_grid_t`` / ``dt``
     - ``int`` / ``float``
     - ``60`` / derived
     - Pick **one**. ``dt`` derives ``n_grid_t = ceil(t_max/dt)``. See :doc:`discretization`
   * - ``homogeneity``
     - ``str``
     - auto
     - Override inferred ``"translation"`` / ``"rotation"`` / ``"general"``
   * - ``r_max`` + ``n_grid_r``
     - ``float`` + ``int``
     - ``null``
     - Translation full-grid mode (both required to leave lazy mode)
   * - ``n_grid_cos``
     - ``int``
     - ``null``
     - Rotation full-grid mode
   * - ``x_max`` + ``n_grid_x``
     - ``float`` + ``int``
     - ``null``
     - General full-grid mode
   * - ``c_closed_form_module``
     - ``str``
     - ``null``
     - Path to ``.py`` module exporting a closed-form ``C_fn(n1, t1, n2, t2) → (N, N)``
   * - ``c_closed_form_attr``
     - ``str``
     - ``"C_fn"``
     - Attribute name in the module
   * - ``c_closed_form_only``
     - ``bool``
     - ``false``
     - Skip the spline cache entirely; route every C lookup through ``C_fn`` for machine precision
   * - ``c_closed_form_vectorized``
     - ``bool``
     - ``false``
     - ``C_fn`` accepts ``(n,)``-shaped time/position arrays and returns ``(n, N, N)`` (only with ``c_closed_form_only: true``)
   * - ``c_method``
     - ``str``
     - ``"dblquad"``
     - Quadrature for the inner :math:`\int R\kappa^2 R`. ``"dblquad"`` (adaptive) or ``"gauss_legendre"`` (18-100× faster on piecewise-analytic κ²). See :ref:`integrator-choice` below
   * - ``c_n_gauss``
     - ``int``
     - ``20``
     - GL nodes per dimension when ``c_method: gauss_legendre`` (cost ``c_n_gauss²`` per sub-region)
   * - ``interp_method``
     - ``str``
     - ``"linear"``
     - Spline kind: ``"linear"`` (monotone, default) or ``"cubic"`` (O(h⁴) on smooth grids; ignored under ``c_closed_form_only``)
   * - ``n_jobs``
     - ``int``
     - ``1``
     - Parallel workers for the cache build (``-1`` = all cores via joblib)
   * - ``cache_path``
     - ``str``
     - ``null``
     - Directory or file for caching the built propagator

Section reference: ``sweep``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 22 14 40

   * - Field
     - Type
     - Default
     - Notes
   * - ``positions_grid``
     - ``dict[str, list[float] | list[ndarray]]``
     - **required**
     - One key per spatial argument in ``expand.observable``. Cartesian product
   * - ``t_final_grid``
     - ``list[float]``
     - **required**
     - External upper-time values to evaluate at.  Also the default time for
       any external point not named in ``external_times_grid``
   * - ``external_times_grid``
     - ``dict[str, list[float]]``
     - optional
     - Pin external points at unequal times, swept as a further Cartesian
       axis.  Required for two-time observables — see
       :ref:`two-time-observables`
   * - ``component_pairs``
     - ``list[[a, b]]``
     - **required**
     - Component-index pairs for the observable
   * - ``orders``
     - ``list[int]`` or ``null``
     - ``null``
     - Optional subset of ``expand.orders``
   * - ``vertex_types``
     - ``list[str]`` or ``null``
     - ``null``
     - Filter diagrams by vertex composition (e.g. ``["FK"]`` for demo2's cross-term)
   * - ``integrate_over``
     - ``null`` / ``"all"`` / ``list[str]``
     - ``null``
     - External-time integration: ``null`` = fixed at ``t_final``, ``"all"`` = integrate every external, list = explicit subset
   * - ``method``
     - ``str``
     - ``"qmc_vectorized"``
     - ``"qmc_vectorized"`` / ``"qmc"`` / ``"qmc_scalar"`` / ``"gauss_legendre"`` / ``"nquad"``. See :ref:`integrator-choice` below
   * - ``n_samples``
     - ``int``
     - ``8192``
     - Sobol sample count (QMC paths only)
   * - ``seed``
     - ``int``
     - ``42``
     - Sobol seed (QMC paths only)
   * - ``n_gauss``
     - ``int``
     - ``8``
     - GL nodes per dimension (``method: gauss_legendre`` only). Cost scales as ``n_gauss^d``
   * - ``n_jobs``
     - ``int``
     - ``1``
     - Parallelise across grid points (``-1`` = all cores). Mutually exclusive with ``expand.n_jobs > 1``

Section reference: ``output`` (optional list of plugins)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each list entry is a ``{type: <kind>, path: ..., ...}`` dict. Three kinds today:

.. code-block:: yaml

   output:
     - {type: table,  path: results.md,  format: markdown}     # or csv
     - {type: npz,    path: results.npz}                       # raw arrays
     - {type: plot,   path: result.png,
        x: y, hue: order, facet_col: t_final}                  # matplotlib facet grid

.. _integrator-choice:

Choosing an integrator (``sweep.method`` and ``propagators.c_method``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two layers in the pipeline run numerical quadrature:

1. **C-propagator construction** (``propagators.c_method``) — the
   inner :math:`\int_0^{t_1}\!\int_0^{t_2} R(t_1,\lambda_1)\,
   \kappa^2(\lambda_1,\lambda_2)\,R(t_2,\lambda_2)\,d\lambda` for
   each :math:`(t_1, t_2, n_1, n_2)` cache cell. Skipped entirely
   when you supply ``c_closed_form_module``.
2. **Diagram evaluation** (``sweep.method``) — the outer
   :math:`d`-dimensional time integral over the causal simplex,
   one evaluation per ``(positions, t_final, component_pair)``
   grid point.

For both layers the choice is the same trade-off: **adaptive
robustness** (``dblquad`` / ``nquad``) vs **smooth-integrand
speed** (``gauss_legendre``).

.. tip::

   **Gauss-Legendre is not Gaussian/exponential-only.** It gives
   exponential convergence on **any function analytic on the
   integration domain** — polynomials, rationals, exponentials,
   trigonometrics, products and compositions thereof.

   When the integrand is only **piecewise** analytic (e.g. an
   ``|λ1 − λ2|`` cusp on the diagonal of κ²), splitting the
   domain along the non-smooth boundary recovers full GL
   convergence on each piece. The package's
   ``c_method='gauss_legendre'`` does this split automatically.

   GL is **not** the right tool for: (a) discontinuous
   integrands, (b) high-frequency oscillations (need Filon or
   Levin), (c) integrable singularities like :math:`1/\sqrt{x}`
   (need Gauss-Jacobi or tanh-sinh). None of these arise in
   typical sft-wick diagrams; the standard exp / Gaussian /
   Legendre kernels are all GL-friendly.

Decision matrix:

.. list-table::
   :header-rows: 1
   :widths: 24 18 22 36

   * - Layer
     - Method
     - Best for
     - Trade-off
   * - ``propagators.c_closed_form``
     - ``auto`` (default)
     - Diagonal constant drift + separable exponential-temporal noise
       (the demo1 / demo2 family), optional constant white noise
     - **No quadrature at all**: the built-in closed form fills the
       table in milliseconds.  ``null`` forces quadrature; a
       ``c_closed_form_module`` supplies your own.
   * - ``propagators.c_method``
     - ``auto`` (default)
     - Any of the package's own kernels (exponential / Gaussian /
       Legendre) when no closed form applies
     - Gauss-Legendre with the node count refined until the rule is
       converged at the table's extreme cells (20 → 30 → 45 → 68), so
       6-30 ms / cell; ``dblquad`` if no count converges.  User
       callables always get ``dblquad``.
   * -
     - ``gauss_legendre``
     - Piecewise-analytic κ² (single ``|Δt|`` cusp on the diagonal)
     - Fixed ``c_n_gauss`` nodes, ~6 ms / cell at 20; accuracy
       degrades with ``(γ + 1/σ_t) t_max`` (1e-12 at ``t_max=15``,
       2e-2 at 100 for demo1's kernel), so prefer ``auto``
   * -
     - ``dblquad``
     - Any κ² (robust, adaptive)
     - 7 ms / cell at ``t_max ≈ 5`` rising to ~200 ms at 100 (the
       rectangle is split at the diagonal cusp; un-split it was
       74-245 ms and only ~2e-6 accurate)
   * - ``sweep.method``
     - ``qmc_vectorized`` (default)
     - High-d diagrams (``d ≥ 6``), non-smooth integrands, or when stochastic error bars are wanted
     - ``~ 1/√n_samples`` bias decay; can severely under-resolve narrow peaks at large ``t_final``
   * -
     - ``gauss_legendre``
     - Smooth integrands at ``d ≤ 5`` (the typical sft-wick case)
     - **Exponential convergence** in ``n_gauss``; deterministic; cost scales as ``n_gauss^d``
   * -
     - ``nquad``
     - Adaptive 1-3D fallback when GL nodes are insufficient
     - Slow; raises ``NotImplementedError`` on dynamic-coupling diagrams
   * -
     - ``qmc`` / ``qmc_scalar``
     - Compatibility / single-sample debugging
     - Slow scalar Python loop; not for production sweeps

User-Python hooks
~~~~~~~~~~~~~~~~~

The YAML block can defer to user Python in six places — each
loads a callable from a ``.py`` module relative to the YAML file
and registers it for joblib's worker-safe by-value module
loading (so it composes cleanly with any of the
``n_jobs > 1`` knobs):

.. list-table::
   :header-rows: 1
   :widths: 36 22 42

   * - YAML key
     - Callable signature
     - Purpose
   * - ``system.linear.gamma_module``
     - ``γ(t) → ndarray(N,)``
     - Time-dependent linear drift (replaces ``linear.gamma`` under ``linear.type: diagonal``)
   * - ``system.linear.R_time_module``
     - ``R_time(t1, t2) → float``
     - Scalar closed-form ``R`` (escape hatch via ``linear.type: explicit``; bypasses the γ-spline cache)
   * - ``system.noise.kappa2.type: callable_module``
     - ``κ²(n1,t1,n2,t2) → (N, N)``
     - Non-separable noise correlator (replaces ``separable_translation``/``separable_rotation``)
   * - ``system.noise.sigma2.type: callable_module``
     - ``σ²(n1,lam,n2) → (N, N)``
     - Spacetime-varying δ-correlated white-noise impulse (the alternative to ``sigma2.type: constant``)
   * - ``system.vertices[].coupling_module``
     - ``F(n,t) → (N,)*n_legs``
     - Spacetime-dependent local F (rare; ``coupling`` literal is the common case)
   * - ``system.nonlocal_vertices[].coupling_module``
     - ``κ^(m)(n_list,t_list) → (N,)*m`` (per-sample) **or** ``κ^(m)(n_2d,t_2d) → (n_samples,)+(N,)*m`` (vectorised; opt-in via ``coupling_vectorized: true``). With ``already_R_contracted: true`` the callable returns the partner-time R-contracted ``κ^(m)_R`` and sft-wick skips the m leg-time integrations.
     - Spacetime-dependent non-local coupling (e.g. demo2's :math:`\kappa^{(3)}`)
   * - ``propagators.c_closed_form_module``
     - ``C_fn(n1,t1,n2,t2) → (N,N)``  or vectorised ``(n,N,N)`` (opt-in via ``c_closed_form_vectorized: true``)
     - Closed-form C lookup (skips dblquad / GL entirely)

Parameter scans from the shell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--override key=value`` patches any leaf field in the parsed
config (safe scalar coercion — no ``eval``):

.. code-block:: bash

   # vary one knob over a list
   for seed in 0 1 2 3 4 5; do
       sft-wick run demo1_config.yaml \
           --override "sweep.seed=$seed" \
           --override "output[0].path=results_seed$seed.md"
   done

   # bump GL precision for a high-t_f sweep
   sft-wick run demo2/L2/config_FK.yaml \
       --override sweep.n_gauss=12

   # validate without running
   sft-wick run demo1_config.yaml --dry-run

Homogeneity modes
-----------------

The :class:`~sft_wick.workflow.Kappa2` variant chosen in ``noise``
determines ``System.homogeneity``, which in turn selects the spatial
code path:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - ``Kappa2`` variant
     - ``System.homogeneity``
     - Spatial cache path
   * - ``SeparableTranslation``
     - ``"translation"``
     - 1-D separable; ``precompute_C_table_translation``
   * - ``SeparableRotation``
     - ``"rotation"``
     - Radial + Legendre; ``precompute_C_table_rotation``
   * - ``GeneralKappa2``
     - ``"general"``
     - Full 2-D spatial; ``precompute_C_table_general``

The homogeneity is inferred from the spec automatically; users
normally don't set it directly.
