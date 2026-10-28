Workflow API (L1 + L2)
======================

The :mod:`sft_wick.workflow` subpackage is the recommended entry
point for typical use.  It wraps the full pipeline (physics spec →
perturbative expansion → propagator cache → numerical sweep) into five
immutable types and two orchestration helpers.

Which layer to use
------------------

**Default to L2 (YAML + CLI).**  A config file is reproducible,
diff-able, easy to share, identical on a laptop and a cluster, and
supports quick parameter scans via ``--override``.  The L1 Python API
is the same machinery: every YAML field maps 1:1 onto an L1
constructor argument.  Drop down to Python to compose multiple systems
or attach custom post-processing.

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

Quick start: L2 (config file)
-----------------------------

A minimal but complete config covering the demo1 two-component
system at orders 0, 2, 4:

.. code-block:: yaml

   # demo1_config.yaml
   system:
     field: {name: phi, n_components: 2}
     linear: {type: diagonal, gamma: [1.0, 1.0]}
     vertices:
       - name: F
         coupling:                              # bare F tensor;
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
any leaf field (safe scalar coercion, no ``eval``).  Larger
examples exercising non-local vertices, closed-form :math:`C`
hooks, and dynamic couplings live in ``examples/demo1_config.yaml``
and ``examples/demo2_config.yaml``.

Quick start: L1 (Python)
------------------------

Same workflow written in Python.  Use this when embedding in a larger
script:

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
   ``positions={"x": 0.0, "y": 0.0}``, *not* as
   ``("phi_a(x)", "phi_b(x)")``, which raises ``ValueError`` at
   interacting orders (order 0 is exempt).

   The label is the name of an integration/external coordinate, not the
   point itself: coincident *points* are fully supported, and every demo
   uses them.  A shared *label* is not, because the spatial contraction
   is keyed by label, so operators sharing one are collapsed without
   their distinct component-index routings.  Before 0.4.0 that produced
   a silently wrong number at interacting orders (a factor 2 in demo2's
   order-2 ``F`` channel); it is now refused.

The five headline types
-----------------------

:class:`~sft_wick.workflow.System`: physics spec
   Combines field, linear operator (for :math:`R`), noise (for
   :math:`C`), and vertex lists.  Methods ``expand`` and
   ``propagators`` lower to the computational layer.

:class:`~sft_wick.workflow.Expansion`: diagram-level view
   What ``compute_moment`` returned, wrapped for inspection:
   ``diagrams(order)``, ``summary()``, ``by_vertex_type(order)``.
   Also the numerical entry points ``evaluate`` (single point) and
   ``sweep`` (grid).

:class:`~sft_wick.workflow.Propagators`: cache wrapper
   Thin wrapper over :class:`~sft_wick.evaluate.PropagatorCache`.
   Pass a ``c_closed_form=`` hook to ``System.propagators`` when a
   closed-form :math:`C(n_1, t_1, n_2, t_2)` is known; it skips the
   dblquad grid build entirely.

:class:`~sft_wick.workflow.Result`, :class:`~sft_wick.workflow.SweepResult`
   Pandas-backed structured output.  ``SweepResult.totals()`` is a
   long-format DataFrame keyed by ``(x, y, t_final, a, b, order)``;
   ``SweepResult.plot(x=…, hue=…, facet_col=…)`` gives a quick
   diagnostic figure.

MSR convention
--------------

L1 applies the MSR prefactors for you:

- **Local** :math:`F^{(n)}`: multiplied by :math:`-i`
- **Non-local** :math:`\kappa^{(m)}`: multiplied by
  :math:`-i^m / m!` (e.g. :math:`+i/6` for :math:`m=3`)

So users pass **bare** tensors:

.. code-block:: python

   # CORRECT: pass the physical F, wrapper applies −i
   sw.LocalVertex("F", coupling=F)

   # WRONG: pre-multiplied, will double-apply the factor
   sw.LocalVertex("F", coupling=-1j * F)

The raw API (``compute_moment``, ``DiagramTerm.evaluate_coupling``)
does NOT apply these factors automatically.  Raw callers must pass the
pre-multiplied tensor.

Multiplicative white noise, Itô and Stratonovich
------------------------------------------------

White noise whose amplitude depends on the field is declared as the
system's :math:`\sigma^2`:

.. code-block:: python

   system = sw.System(
       field=sw.FieldSpec("phi", n_components=2),
       linear=sw.DiagonalA(gamma=[1.1, 1.1]),
       vertices=[sw.LocalVertex("F", coupling=F)],
       noise=sw.GaussianNoise(
           kappa2=...,                       # the coloured noise, if any
           sigma2=sw.MultiplicativeImpulse(
               g0=g0,                        # (N, M): the amplitude at φ = 0
               g1=g1,                        # (N, M, N): dg_ak/dφ_b
               interpretation="stratonovich",   # or "ito" (default)
           )),
   )

for the SDE

.. math::

   d\varphi_a = \bigl(\dots\bigr)\,dt
       + g_{ak}(\varphi)\,(\circ)\,dW_k , \qquad
   g_{ak}(\varphi) = g^{(0)}_{ak} + g^{(1)}_{akb}\,\varphi_b ,

with ``M`` independent Wiener processes.  The noise covariance
:math:`D(\varphi) = g(\varphi) g(\varphi)^{\mathsf T}` splits into a part
that C carries and a part that becomes vertices:

.. list-table::
   :header-rows: 1
   :widths: 12 20 30 14 12

   * - Vertex
     - Legs
     - Bare coupling
     - MSR factor
     - power of ``g1``
   * - (none)
     - (none)
     - ``D0 = g0 g0ᵀ`` → the white noise of C
     - (none)
     - 0
   * - ``G``
     - ψ ψ φ
     - ``D1_abc = g0_ak g1_bkc + g1_akc g0_bk``
     - ``1/2``
     - 1
   * - ``H``
     - ψ ψ φ φ
     - ``D2_abce = g1_akc g1_bke``
     - ``1/2``
     - 2
   * - ``B``
     - ψ
     - ``b_i = ½ g1_ikj g0_jk`` (Stratonovich)
     - ``−i``
     - 1
   * - ``L``
     - ψ φ
     - ``L_ic = ½ g1_ikj g1_jkc`` (Stratonovich)
     - ``−i``
     - 2

The MSR action carries :math:`\tfrac12 \psi_a \psi_b D_{ab}(\varphi)`, so
the two-ψ vertices take :math:`-i^2/2! = 1/2`, applied by
``System.build_coupling_values`` exactly as the :math:`-i` of a
``LocalVertex`` is.  ``interpretation='stratonovich'`` converts the SDE to
its Itô form: the extra drift
:math:`\tfrac12 \sum_{jk} g_{jk}\,\partial_j g_{ik} = b + L\varphi`
becomes the drift vertices ``B`` and ``L``, so R and C stay those of the
Itô part and every integrator that runs the Itô model runs the
Stratonovich one.  (Folding ``L`` into the linear operator would resum it,
but a generic ``L`` is not diagonal, and a dense R runs only on the
scalar-loop integrators and has no closed-form C.)  Rename the vertices
with ``vertex_names=`` and ``drift_names=`` if they collide with your own;
a vertex whose coupling is identically zero is not generated.

**Orders count vertices**, whatever power of ``g1`` a vertex carries: at
:math:`g_1^2`, ``H`` sits at order 1 and the channels labelled ``G``,
``BG`` and ``B`` at order 2 (and ``L`` at order 1 under Stratonovich).  Use
``Expansion.by_vertex_type`` / ``vertex_types=`` to collect a fixed power.
A label is the set of a diagram's vertex names, sorted and concatenated, so
two ``G`` vertices are labelled ``G`` and one ``B`` with one ``G`` is
``BG``.

Under the moment target ``vertex_types=`` refuses a label no diagram at the
selected orders carries, and many vertex sets give no diagram (an odd
number of fields, a vanishing response loop), so pass the labels of the
power that ``Expansion.by_vertex_type(order)`` holds, as
``examples/demo5/white_l1_multiplicative.py`` does.

Expand with ``ito=True`` (the default): the numerical layer evaluates the
Itô form, and under ``ito=False`` it refuses the equal-point R of a
two-ψ vertex rather than return the Itô value (see
:doc:`perturbative_expansion`).

Two further properties of this spec:

* ``D0`` reaches C as a ``ConstantImpulse(D0)`` would, so the built-in
  closed form covers it (``DiagonalA`` + separable exponential κ² +
  white noise).  ``D0`` is dense in general, so pass ``diag_C=False`` to
  ``System.expand`` and to ``System.propagators``, as for any
  component-mixing white noise.
* The vertices are local, so the model is one SDE per spatial point.
  ``Expansion.evaluate`` refuses external points at different positions
  that a chain of response propagators joins: the answer would depend on
  how the noise at different points is correlated, which local vertices do
  not describe.  Distinct labels and distinct ``external_times`` at one
  position are fine.

  The L0 entry points refuse the same spelling, for any system rather than
  this noise alone (``evaluate.py::_require_one_position_per_group``): R
  carries ``δ(n − n')``, so the points an R chain joins are one spatial
  point and have one coordinate.  The other way to write such a chain is
  an observable with an external response leg, ``("phi_a(x)",
  "psi_b(y)")``, which reaches L0 with no multiplicative noise in
  sight.

A quadratic term in ``g`` is not supported; it would add vertices with two
ψ legs and three and four φ legs, and a quadratic and a cubic
noise-induced drift.

``examples/demo5/white_l1_multiplicative.py`` runs this route against an
exact reference; the YAML form is ``noise.sigma2.type: multiplicative``.

Observable convention: ``integrate_over``
-----------------------------------------

Both ``Expansion.evaluate`` and ``Expansion.sweep`` accept an
``integrate_over`` keyword:

``None`` (default)
    All external points fixed at ``t_final``: the physics 2-point
    correlator :math:`\langle \varphi(t_f)\,\varphi(t_f) \rangle`.

``"all"``
    All external times integrated over ``[t_min, t_final]``: the
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
:math:`\Gamma_a(t) = \int_{t_c}^t \gamma_a(\tau) d\tau` on a grid from
:math:`t_c` = ``t_min_cache`` (default 0; ``System`` extends it down to
its ``t_min``) to ``t_max_cache`` (default 100) and caches it as a cubic
spline for O(1) R lookups; ``System.propagators`` refuses a ``t_max``
above ``t_max_cache``.  Scalar closed-form
``R(t_1, t_2)`` callables that the spline-cache lowering cannot
express are supported via :class:`~sft_wick.workflow.ExplicitR`.
From YAML this is the ``linear.type: explicit`` route::

   system:
     linear:
       type: explicit
       R_time_module: ./R_time.py     # exports R_time(t1, t2)
       R_time_attr:   R_time           # default 'R_time'
       iso_R:         true             # true: a scalar R; false: an (N, N) matrix
       R_time_vectorized: false        # true: R_time also accepts arrays

With ``R_time_vectorized: true`` the callable must also accept two arrays
of shape ``(n,)`` and return shape ``(n,)`` (scalar R) or ``(n, N, N)``.
The batched integrators (``gauss_legendre``, ``qmc_vectorized``) then call
it once per batch of causal time pairs.  Without the flag they call it once
per distinct causal pair, which on a tensor-product grid can dominate the
run time when the callable is expensive (a spline, a numerical solve).  The
callable must still accept two floats: the C quadrature and the scalar
integrators call it that way.  It is called once on arrays at load time and
compared with its scalar calls.  The built-in R of ``type: diagonal`` is
vectorised already.

The module is loaded at config-parse time and registered for
cross-process by-value serialisation, so the explicit-R callable
composes cleanly with ``propagators.n_jobs > 1``,
``expand.n_jobs > 1``, and ``sweep.n_jobs > 1`` (same machinery
``c_closed_form_module`` and ``coupling_module`` use).

``iso_R: false`` declares a matrix-valued R, the response of a dense
drift: a time-ordered matrix exponential rather than a per-component
decay.  At load time the callable is called once at a causal and once at
an acausal time pair inside ``[t_min, t_max]``: the value must have the
declared shape and must vanish for ``t1 < t2``.  A dense R needs three
more settings, because C is then dense too: ``expand.diag_R: false`` and
``expand.diag_C: false`` (``System.expand`` refuses the diagonal
simplifications when R or C has off-diagonal entries), and
``propagators.diag_C: false``, so the tables hold every ``C_ab``.  Every
``sweep.method`` evaluates a matrix R.

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

The runtime calls ``k3_coupling`` for every integration point (a
QMC sample or a Gauss-Legendre node) and leg order (see *Leg
order* below).

**Vectorised contract** (opt-in, fast for heavy callables)::

   def k3_coupling(n_2d, t_2d):
       """``n_2d``, ``t_2d`` are shape ``(m, n_samples)`` arrays.
       Return a tensor of shape ``(n_samples,) + (N,)*3``."""
       ...

   sw.NonLocalVertex(
       "K", order=3, coupling=k3_coupling, coupling_vectorized=True,
   )

With the batched integrators (``method='qmc_vectorized'`` or
``'gauss_legendre'``), the runtime calls ``k3_coupling`` once per
integrand and leg order,
amortising the callable's overhead across all samples. This is the
right form when the function does heavy work that vectorises well
(special functions, ufuncs, BLAS).  For cheap functions
(`numpy.exp` of a few scalars) the per-sample contract has lower
total overhead.
Under the scalar QMC loop (``method='qmc_scalar'``, and ``'qmc'``
when :func:`~sft_wick.evaluate.integrate_moment` selects that loop),
the runtime calls ``k3_coupling`` once per QMC sample and leg order,
as a batch of one (``n_samples = 1``).  ``method='nquad'`` does the
same at every point its adaptive quadrature visits; it refused a
callable coupling up to 0.5.0.

The static fast path is used automatically when no callable is
passed.  The batched integrators call
:meth:`~sft_wick.evaluate.DynamicCouplingPromise.evaluate_at_batch`
and the scalar QMC loop calls
:meth:`~sft_wick.evaluate.DynamicCouplingPromise.evaluate_at`; both
dispatch per symbol on the ``vectorized`` flag, so
mixing both contracts on different symbols within the same diagram
is supported.

**Leg order.**  Axis ``l`` of the returned tensor belongs to the leg
at ``n_list[l]``, ``t_list[l]``.  A diagram sums the vertex over every
assignment of its legs to the fields they contract with.  Each term
calls the callable with its own leg order, so the callable is evaluated
once per distinct leg order in the diagram (at most ``m!``) and needs
no permutation symmetry of its own.

**Several copies of one vertex.**  At order ``n >= 2`` a diagram can
hold two or more copies of the same vertex, at different points, and
its coupling sum also routes the legs of the copies between them.  Each
``(copy, leg order)`` is evaluated at its own points, so the callable is
called once per distinct leg tuple in the diagram.  Up to 0.5.0 a
callable at more than one set of points was refused.

**Local vertices.**  A :class:`~sft_wick.workflow.LocalVertex` takes the
same two contracts for a spacetime-dependent ``F^(n)(x, t)``; the vertex
has a single point, so ``n_list`` and ``t_list`` have length 1
(``(1, n_samples)`` under ``coupling_vectorized=True``) and the returned
tensor has shape ``(N,)*n``::

   sw.LocalVertex("F", coupling=f_of_x_t, rank=3)

``rank`` is required for a callable, because the number of legs cannot
be read off one; it is checked against the tensor when a tensor is
passed.
The MSR factor ``-i`` is applied to the callable's output, as it is to a
tensor.

.. note::

   With scalar positions, ``n_list`` has shape ``(m,)`` and ``n_2d``
   has shape ``(m, n_samples)``.  With d-dim vector positions they
   gain a trailing axis, ``(m, d)`` and ``(m, n_samples, d)``; the
   shapes of ``t_list`` and ``t_2d`` do not change.

Equal-time (single-shell) cumulants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default :class:`~sft_wick.workflow.NonLocalVertex` contract treats
the vertex's coupling as the **full cross-spacetime** cumulant
``κ^(m)(z_1, …, z_m)`` and integrates over ``m`` independent
``(n, λ)`` arguments, the literal MSR action
``W^(m)[ψ] = (1/m!) ∫ dz_1 … dz_m  ψ(z_1) … ψ(z_m) κ^(m)``.

Many applications (especially cosmological structure / lensing in the
single-shell / Limber-like regime) only have an **equal-shell** or
**equal-time** cumulant available, e.g. the equal-shell bispectrum
``ζ_eq(γ_12, γ_23, γ_31; λ)`` from a CCL-like pipeline.  The
action-level relation between the two is::

   κ^(m)(z_1, ..., z_m)  ≈  δ(λ_1 − λ_2) · ... · δ(λ_{m−1} − λ_m)
                              · κ_eq(n_1, ..., n_m;  λ)

Passing ``ζ_eq`` as the ``coupling`` callable *without* declaring the
δ-structure makes sft-wick sweep ``m`` independent times over
``[t_min, t_final]``, contributing a spurious
``(t_final − t_min)^(m−1)`` factor of integration measure.  The answer
is then wrong by orders of magnitude.

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
* keeps the ``m`` spatial legs independent: the callable still
  receives ``m`` distinct positions, because ``ζ_eq`` is a function
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
``equal_time=False`` (the default).  Otherwise the ``m`` independent
``λ`` integrations that the cross-spacetime form requires would be
collapsed.

Already-R-contracted vertices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default :class:`~sft_wick.workflow.NonLocalVertex` returns the
**bare** ``κ^(m)`` at the m leg points and lets sft-wick handle every
R-propagator that lands on a leg via the Wick contraction. For
**narrow-kernel** ``κ^(m)`` (canonical case: canoes' squeezed κ³ at
``ℓ_max → ∞``, with diagonal width ``Δχ ~ χ_max / ℓ_max``), the m
leg-time integrations on the causal simplex demand prohibitively
many quadrature nodes. The identity used is

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
  **absorbed**: they remain in the diagram graph (so direction
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
becomes 1-D after absorption). The surviving integrand is also
smoother: the narrow-kernel peak has been folded into the (presumably
smooth) ``κ^(m)_R``, so Gauss-Legendre converges exponentially with far
fewer nodes.

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

This is slow (O(n_chi^m) raw-callable evaluations per outer point).
Use it to validate the dispatch against a known raw kernel.
**For production work, supply an analytical / pre-tabulated**
``κ^(m)_R`` **callable**, e.g. canoes' FFTlog-of-W chain.

Compatibility:

* ``already_R_contracted=True`` rejects ``equal_time=True`` at
  construction: the R-contracted callable has already integrated over
  its leg coordinates.
* All existing knobs (``coupling_vectorized``, MSR factor, response
  phase, diagonal / isotropic simplification) work unchanged.
* The dispatch is **mathematically equivalent** to the raw path when
  both pipelines receive analytically-equivalent inputs, locked at
  machine precision (``rtol=1e-12``, observed ``rel_diff ≈ 1.8e-15``)
  by the constant-κ³ equivalence test in
  ``tests/test_R_contracted_vertex.py``. The two paths are **not
  bit-identical**: raw integrates the diagram on a higher-dimensional
  causal simplex (m extra leg-time variables per absorbed vertex), so
  the floating-point sums use different node sets and the last few
  bits of float64 necessarily differ. Convergence under refinement
  (e.g. demo2 FK at ``n_gauss=16``: ``rel_diff ≈ 7e-4``) confirms the
  dispatch on real spacetime-dependent κ³; see
  ``docs/notes/R_contracted_nonlocal_vertex.md`` §4.1.

Design details and the equivalence-validation evidence on demo2 FK
live at ``docs/notes/R_contracted_nonlocal_vertex.md``.

.. _declaring-kinks:

Declaring kinks
~~~~~~~~~~~~~~~

Tensor-product Gauss-Legendre converges exponentially on an integrand
that is smooth on the causal simplex, and as ``n^-2`` on one with a kink
(a jump in the first derivative) inside it; adaptive ``nquad`` stops short
of its tolerance along a kink.  ``method='gauss_legendre'`` and
``method='nquad'`` therefore split the time domain where two times that the
causal structure leaves unordered cross and the integrand is kinked there.
Each consistent order of such pairs is integrated as its own causal
sub-simplex, the split repeats inside each piece until no pair is left, and
the pieces are added.  Three kinds of pair are split:

* two parents of one time variable (a vertex with several ψ legs at one
  time), whose upper bound ``min(parents)`` changes branch where they
  cross; found from the diagram;
* the two ends of a C propagator when C is kinked on its time diagonal:
  white noise (``GaussianNoise(sigma2=...)``) and a κ² with a ``|Δt|`` cusp
  (the exponential/OU kernel, a damped cosine, any callable the kernel
  probe finds cusped, since ``C = ∫∫ R κ² R`` inherits the cusp in its
  third derivative) are found from the model, and a closed-form C callable
  declares it with ``has_diagonal_kink = True``;
* two time arguments of a coupling callable that declares
  ``has_coincident_time_kinks = True``, meaning that it is kinked wherever
  two of its time arguments coincide, for example through a ``min`` over
  them.

Each of the three can also put the kink between an integration time and an
external point pinned at its own time (``external_times``, or ``t_final``
when the externals sit at different times).  A constant is not an ordering
between two variables, so that one is not a pair: the variable's range
``[t_min, min(parents)]`` is cut in two, ``[t*, min(parents)]`` and
``[t_min, min(t*, parents)]``, and the cut is carried to the variables the
orderings put on the same side of it.  With every external at one time the
crossing is the boundary of the domain and no cut is made, so only unequal
external times pay for it.

Splitting one kink can expose another: a piece bounds a variable by
``min(parents, t*)``, which is kinked where a parent crosses ``t*``, so the
parent takes the same cut.  The split therefore repeats until no pair and
no cut is left.

The package cannot see inside a callable, so the last two are declared on
the callable object, for example as a class attribute of a frozen
dataclass::

   @dataclass(frozen=True)
   class RawKappa3:
       """κ³ of exponential pulses; depends on min(t_list)."""
       rates: tuple
       has_coincident_time_kinks = True

       def __call__(self, n_list, t_list):
           ...

Which time arguments are paired depends on the vertex:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - vertex
     - time arguments paired
   * - raw (the default)
     - its own leg times
   * - ``already_R_contracted=True``
     - the partner times its legs alias onto
   * - ``equal_time=True``
     - none: the legs share one time

The attribute is read through the MSR factor wrapper and through any
wrapper that sets ``__wrapped__``, so it works at L1, from YAML
(``coupling_module``) and at L0.  Declaring a kink that is not there costs
time, not accuracy.  ``gauss_legendre`` and ``nquad`` both split at a
declared kink; ``qmc_vectorized`` does not (see the decision matrix
below).

The cost is one integration per consistent order: 2 for one pair, up to
``k!`` for ``k`` mutually unordered times.  Measured on demo 4
(``examples/demo4/README.md``), exponential pulses, against exact
references (maximum relative difference over the component tuples):

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - channel
     - not split, 16 / 32 nodes
     - split, 12 / 16 nodes
     - pieces
   * - raw κ³, 3-point function
     - 5.7e-2 / 1.5e-2
     - 2.2e-14 / 2.2e-14
     - 6
   * - R-contracted κ⁴, FFK4 of ``⟨φ_a φ_b⟩``
     - 4.3e-8 / 7.4e-10
     - 2.6e-15 / 6.3e-15
     - 2, in the diagrams where two F times are partners

The cut at a fixed external time costs one more integration per cut.  The
worst case measured is demo 4's raw exponential 3-point function at
external times (1.7, 1.2, 0.6), where three leg times are each cut against
three distinct external times: 25 pieces per diagram, against 7 for the
leg-order split alone, and 4.8e-15 at 16 nodes where it was 1.4e-3 (2.1e-4
at 64).
Demo 6's ``F F`` channel at two external times takes 2.0 pieces per
diagram, demo 7's two-time channel 2.9, demo 8's oscillator channels 2.88
against 1.38.

An external time swept by ``integrate_over`` is drawn before every internal
variable, so it pairs like an integration variable: a kink against one is
ordered, not cut.  Two swept externals are not a pair (their relative order
comes from the causal structure alone).

Limits:

* The declaration covers kinks where two time arguments coincide, not a
  kink along another curve.
* QMC is not split.  At equal cost the split did not reduce the error for
  2^10 to 2^16 samples: on demo 4's raw kernel (6 pieces) it was 4 to 10
  times larger, and on demo 5's white-noise channel (2 pieces) it was
  larger up to 2^14 samples and 3.8 times smaller at 2^16.

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
  ``(s, u, r)``, with ``s = min(t1,t2)`` and ``u = abs(t1-t2)``,
  3-D regardless of the ambient dimension.
* **rotation**: ``_rotation_cos(n1, n2)`` works on unit vectors of
  any dimension (it only uses ``np.dot`` and ``np.linalg.norm``).
* **general**: lazy mode supports d-dim via dict-keyed memoisation
  (2-D ``(s, u)`` splines per distinct ``(x1, x2)`` pair, retaining
  both time orders). The
  full-grid path raises ``NotImplementedError`` because a d-dim grid
  would inflate the spline to ``(2 + 2d)``-D, with ``n_grid_x **
  (2d)`` build calls.

YAML configuration reference (L2)
---------------------------------

The YAML schema mirrors the L1 constructor signatures one-for-one:
anything expressible in L1 Python has a YAML equivalent. This section
is the **complete reference** for every key the parser recognises.

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
fields must be present.  A key the parser does not read raises a
``ValueError`` that names it and lists the keys its block accepts, so
a misspelt key cannot fall back to its default.

.. code-block:: yaml

   # ============================================================
   # system: physics specification
   # ============================================================
   system:
     field:
       name: phi             # required: field-symbol prefix used in observables
       n_components: 2       # required: number of internal components N

     linear:
       type: diagonal        # 'diagonal' (DiagonalA) | 'explicit' (ExplicitR)
       # ---- type: diagonal (A_{ab} = -γ_a δ_{ab}) ----
       gamma: [1.0, 1.0]     # length-N constant rates, OR
       # gamma_module: ./drift.py        # dotted path to callable γ(t)→array(N)
       # gamma_attr: gamma               # default 'gamma'
       # t_max_cache: 20.0               # γ-spline upper end (default 100.0); propagators.t_max may not exceed it
       # t_min_cache: 0.0                # γ-spline lower end (default 0.0); System extends it down to t_min
       # n_grid_cache: 400               # γ-spline node count (default ceil((t_max_cache - t_min_cache)/dt), else 200)
       # ---- type: explicit (user-supplied R(t1, t2)) ----
       # R_time_module: ./R_time.py      # path to a .py exporting R_time(t1, t2)
       # R_time_attr:   R_time           # default 'R_time'
       # iso_R:         true             # true: a scalar R; false: an (N, N) matrix

     noise:
       kappa2:
         type: separable_translation     # 'separable_translation' | 'separable_rotation'
                                         # | 'callable_module'
         temporal: {type: exponential, lam: 0.05, sigma_t: 0.3}
                                         # 'exponential' | 'gaussian' | 'custom'
         spatial:  {type: exponential, sigma_x: 1.0}
                                         # 'exponential' | 'gaussian' | 'custom'
         # temporal: {type: custom, module: ./kernels.py, attr: temporal}
         #                                 # any callable κ(Δt) -> float; 'attr'
         #                                 # defaults to the block's name
         # angular: {type: legendre, coeffs: [1.0, 0.2]}  # separable_rotation
                                         # 'legendre' | 'custom'
       sigma2: null                      # optional δ-correlated white-noise floor
                                         # 'constant'        -> {type: constant, amplitude: 0.01}
                                         #                      amplitude: a number (× I_N), or a
                                         #                      symmetric N × N nested list
                                         # 'callable_module' -> {type: callable_module,
                                         #                       module: ./sigma2.py, attr: sigma2}
                                         # callable signature: sigma2(n1, lam, n2) -> (N, N)

     vertices:                           # zero or more local F vertices (bare F tensor)
       - name: F                         # unique across vertices and nonlocal_vertices
         coupling: [[[0.0,0.0],[0.0,1.0]], [[0.0,0.5],[0.5,0.0]]]   # shape (N,)*n
         # OR
         # coupling_path: ./F.npy        # path to .npy file (relative to YAML)
         # OR (for spacetime-dependent F)
         # coupling_module: ./F_dynamic.py
         # coupling_attr:   coupling_fn
         # rank: 3                       # required for a callable coupling
         # coupling_vectorized: false

     nonlocal_vertices: []               # zero or more non-local κ^(m) vertices
       # - name: K
       #   order: 3                      # required: m, the number of ψ-legs
       #   coupling_module: ./k3.py      # or coupling / coupling_path, shape (N,)*m
       #   coupling_attr:   coupling_fn  # default 'coupling_fn'
       #   coupling_vectorized: false    # set true if attr is the batched contract
       #   equal_time: false             # one shared time for the m legs
       #   already_R_contracted: false   # the callable returns κ^(m)_R

     t_min: 0.0                          # lower bound for time integration domain

   # ============================================================
   # expand: perturbative diagram enumeration
   # ============================================================
   expand:
     observable: ["phi_a(x)", "phi_b(y)"]   # required: a list of operators
     orders:     [0, 2]                     # required: a list, or one integer
     target:            moment   # 'moment' | 'connected' (connected_targets.rst)
     response_phase:    true     # multiply by (-i)^n_response (MSR)
     ito:               true     # Itô prescription (R(x,x) = 0)
     collect_topology:  true     # spatial-level Wick contraction (faster, default)
     diag_R:            true     # diagonal-R simplification; raises if R has off-diagonal entries
     diag_C:            true     # diagonal-C simplification; raises if C has off-diagonal entries
     iso_R:             null     # null = inferred from the linear operator; true = also strip indices
     iso_C:             false    # true needs C_ab = δ_ab c (contributes c; raises otherwise)
     cache_path:        null     # optional dir/file for joblib expansion cache
     n_jobs:            1        # parallel diagrams per grid point (>1 forbids sweep.n_jobs>1)

   # ============================================================
   # propagators: C-cache build
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
     # 'auto': built-in closed form when the kernel family has one
     # (diagonal constant drift + separable exponential-temporal noise,
     # optional constant white noise); null forces quadrature
     c_closed_form:           auto
     c_closed_form_module:    null   # ./c_closed_form.py
     c_closed_form_attr:      C_fn   # callable name in the module
     c_closed_form_only:      false  # skip spline cache entirely
     c_closed_form_vectorized: false # c_fn accepts (n,) arrays, returns (n, N, N)

     # quadrature for the inner ∫ R κ² R when no closed form is given
     c_method:           auto         # 'auto' | 'dblquad' | 'gauss_legendre'
     c_n_gauss:          20           # GL nodes/dim under c_method='gauss_legendre'

     diag_C:             true         # false: tabulate every C_ab, for a C with
                                      # off-diagonal entries (dense R, mixing
                                      # kappa2, matrix sigma2); sets expand.diag_C

     interp_method:      linear       # 'linear' | 'cubic'
     n_jobs:             1
     cache_path:         null         # dir for joblib cache

   # ============================================================
   # sweep: numerical evaluation grid
   # ============================================================
   sweep:
     positions_grid:                    # required: one list per point of the observable,
       x: [0.0]                         # each swept independently; quote a point name
       y: [0.0, 0.5, 1.0, 2.5]          # YAML would read as a number or a boolean ('1', 'on')
     t_final_grid: [1.0, 5.0, 15.0]    # required
     component_tuples: [[0, 0], [1, 1]]
                                       # required: one component index per observable
                                       # operator, so [[0, 1, 1], ...] for a 3-point
                                       # observable.
     # component_pairs: [[0, 0], [1, 1]]   # the older name of the same axis; give one
                                           # or the other

     # Optional: pin external points at UNEQUAL times, swept as a further
     # Cartesian axis (same shape as positions_grid).  Needed for two-time
     # observables such as R(t, t') and C(t, t'); with every external at one
     # time, Theta kills the R joining them and any observable carrying a
     # response leg is identically 0.  Result rows gain a ``t_<point>`` column.
     external_times_grid:
       x: [4.0]
       y: [1.0, 2.0, 3.0]

     orders:           null            # optional subset of expand.orders: one integer
                                       # or a list
     vertex_types:     null            # optional labels, one string (F) or a list
                                       # ([F, FK]); a label joins the sorted names of
                                       # a diagram's distinct vertices, so two F
                                       # vertices give F.  A label no diagram carries
                                       # raises and names the labels present.
     integrate_over:   null            # null | all | one point name (x) | a list
                                       # ([x, y]); a name the observable does not
                                       # use raises when the file is loaded

     method:           qmc_vectorized  # 'qmc_vectorized' | 'qmc' | 'qmc_scalar'
                                       # | 'gauss_legendre' | 'nquad'
     n_samples:        8192            # Sobol samples (QMC paths only)
     seed:             42              # Sobol seed (QMC paths only)
     n_gauss:          8               # GL nodes/dim (gauss_legendre only)
     gl_chunk_size:    4096            # GL nodes per batch; null = eager grid
     qmc_coupling_chunk_size: null     # opt-in dynamic coupling batches only
     reuse_integrands: false           # opt-in static preparation per component tuple

     # Lookup reuse and contraction order (see "Explicit lookup reuse and
     # contraction order" below, and resources.rst)
     propagator_cache_bytes:    null      # byte budget for repeated propagator lookups;
                                          # null or 0 disables the cache
     propagator_cache_contract: null      # 'fixed_model', required by a positive budget
     contraction_backend:       ordered   # 'ordered' | 'einsum' | 'factorized'
     contraction_max_bytes:     67108864  # workspace budget of einsum / factorized (64 MiB)

     n_jobs:           1               # parallel over grid points
                                       # (mutually exclusive with expand.n_jobs > 1)

   # ============================================================
   # output: optional artefact emitters (list of plugins)
   # ============================================================
   output:
     - {type: table, format: markdown, path: results.md}
     - {type: npz,   path: results.npz}
     - {type: plot,  path: result.png, x: y, y: value, hue: order, facet_col: t_final}

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
     - N, the number of internal components per field
   * - ``linear.type``
     - ``str``
     - ``"diagonal"``
     - ``"diagonal"`` -> :class:`~sft_wick.workflow.DiagonalA`; ``"explicit"`` -> :class:`~sft_wick.workflow.ExplicitR`, scalar or matrix-valued (``iso_R``).
   * - ``linear.gamma``
     - ``list[float]`` of length N (or 1)
     - **required** unless ``gamma_module`` set (``type: diagonal``)
     - Constant decay rates :math:`\gamma_a`.  One rate stands for every component; a bare number is read as such a list, so ``--override system.linear.gamma=1.5`` works
   * - ``linear.gamma_module``
     - ``str`` (path)
     - ``null``
     - (``type: diagonal``) dotted-path / relative-path to a ``.py`` file exporting a callable ``γ(t) → ndarray(N,)``
   * - ``linear.gamma_attr``
     - ``str``
     - ``"gamma"``
     - Attribute name in ``gamma_module``
   * - ``linear.t_max_cache``
     - ``float``
     - ``100.0``
     - (``type: diagonal``) upper end of the cumulative-integral spline :math:`\Gamma_a(t) = \int_{t_c}^t \gamma_a`, :math:`t_c` = ``t_min_cache``.  ``System.propagators`` refuses a ``t_max`` above it
   * - ``linear.t_min_cache``
     - ``float``
     - ``0.0``
     - (``type: diagonal``) lower end of the Γ spline; ``System`` extends it down to ``system.t_min``
   * - ``linear.n_grid_cache``
     - ``int``
     - derived from ``dt``, else ``200``
     - (``type: diagonal``) spline node count for cumulative Γ, ``ceil((t_max_cache - t_min_cache) / dt)`` when ``dt`` is set
   * - ``linear.R_time_module``
     - ``str`` (path)
     - **required when** ``type: explicit``
     - Path to a ``.py`` file exporting a callable ``R_time(t1, t2) -> float``, or ``-> (N, N)`` under ``iso_R: false``. Must enforce causality (return 0 when ``t1 < t2``).
   * - ``linear.R_time_attr``
     - ``str``
     - ``"R_time"``
     - Attribute name in ``R_time_module``
   * - ``linear.iso_R``
     - ``bool``
     - ``true``
     - (``type: explicit``) ``true``: the callable returns a scalar; ``false``: it returns an ``(N, N)`` matrix.  Checked for shape and for causality when the config is loaded
   * - ``linear.R_time_vectorized``
     - ``bool``
     - ``false``
     - (``type: explicit``) ``true``: the callable also accepts two arrays of shape ``(n,)`` and returns ``(n,)`` or ``(n, N, N)``; the batched integrators call it once per batch.  Checked against the scalar calls when the config is loaded
   * - ``noise.kappa2.type``
     - ``str``
     - **required**
     - ``"separable_translation"`` (1-D r), ``"separable_rotation"`` (Legendre angular), ``"callable_module"``
   * - ``noise.kappa2.temporal``
     - block
     - **required** (separable variants)
     - ``{type: exponential|gaussian, lam, sigma_t}``, or ``{type: custom, module: ./kernels.py, attr: temporal}`` for any callable ``κ(Δt) → float`` (lowered to :class:`~sft_wick.workflow.CustomKernel`); see :doc:`expressions`
   * - ``noise.kappa2.spatial``
     - block
     - **required** (``separable_translation``)
     - ``{type: exponential|gaussian, sigma_x}``, or ``{type: custom, module: ..., attr: spatial}`` for any callable ``κ(r) → float``. For ``separable_rotation``, use ``angular: {type: legendre, coeffs: [...]}`` or ``{type: custom, ...}`` (``κ(cos θ) → float``) instead.  A custom block's ``attr`` defaults to the block's own name.
   * - ``noise.sigma2``
     - block or ``null``
     - ``null``
     - Optional δ-correlated white-noise variance.  Three flavours: ``{type: constant, amplitude: 0.01}``, where the amplitude is a number (read as ``amplitude × I_N``) or a symmetric ``N × N`` nested list (the matrix form mixes the components, so it needs ``propagators.diag_C: false``); ``{type: callable_module, module: ./fn.py, attr: sigma2}`` for a user-supplied ``sigma2(n1, lam, n2) → (N, N)`` (mirrors the ``kappa2.callable_module`` pattern; the spec is wrapped via ``CustomImpulse``); or ``{type: multiplicative, g0: [[...]], g1: [[[...]]], interpretation: ito|stratonovich}`` for white noise whose amplitude depends on φ (:class:`~sft_wick.workflow.MultiplicativeImpulse`; optional ``vertex_names``, ``drift_names``), whose ``D0 = g0 g0ᵀ`` is dense in general and then needs ``propagators.diag_C: false``
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

**Vertex spec.**  Every ``vertices[]`` / ``nonlocal_vertices[]`` entry needs a ``name``, unique across both lists (the coupling values are keyed by it), and exactly one of the following to define the coupling.  A tensor coupling must have every axis of length N: ``(N,)*order`` for a non-local vertex, ``(N,)*n`` for a local one whose first axis is the ψ leg.

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

The two flags are **mutually exclusive**: combining them is rejected
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
     - e.g. ``["phi_a(x)", "phi_b(y)"]``; component letters are MSR-fixed-indices, spatial args are the keys later swept by ``sweep.positions_grid``.  Operators that share a component letter carry one index, so a component tuple must give them the same value
   * - ``orders``
     - ``int`` or ``list[int]``
     - **required**
     - e.g. ``[0, 2, 4]``; one integer is one order
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
     - Apply diagonal-propagator simplification (collapses index sums where R/C is component-diagonal).  Raises ``ValueError`` when R or C has off-diagonal entries (from a dense R, a component-mixing κ² or a white-noise matrix); for C, set ``propagators.diag_C: false``, which sets this flag with it and tabulates every ``C_ab``
   * - ``iso_R`` / ``iso_C``
     - ``bool`` or ``null``
     - ``null`` / ``false``
     - Strip equal component indices from R/C entirely (treats diagonal entries as a single scalar).  ``iso_R: null`` is inferred from the linear operator.  ``iso_C: true`` requires ``C_ab = δ_ab c``, contributes ``c`` per C propagator, and raises ``ValueError`` otherwise
   * - ``cache_path``
     - ``str`` or ``null``
     - ``null``
     - Directory or ``.joblib`` file for caching the symbolic expansion
   * - ``n_jobs``
     - ``int``
     - ``1``
     - Parallelise over diagrams within each grid point (mutually exclusive with ``sweep.n_jobs > 1``)
   * - ``target``
     - ``str``
     - ``"moment"``
     - ``"moment"`` or ``"connected"``, which keeps the incidence-connected diagrams; see :doc:`connected_targets`

.. _two-time-observables:

Two-time observables: ``R(t, t')`` and ``C(t, t')``
----------------------------------------------------

The response propagator is retarded, so ``<phi(u) psi(y)> = R(u, y)`` vanishes
when its two legs sit at the same time.  A sweep pins every external at
``t_final`` unless told otherwise, which means an observable carrying a
response leg evaluates to **exactly zero everywhere**.  Two things are needed
to get a non-trivial answer:

1. Name the **response field** in the observable: ``psi_b(y)`` alongside
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

.. _n-point-sweeps:

n-point observables: ``component_tuples``
-----------------------------------------

An observable of three or four operators is swept like a 2-point one; its
component axis carries one index per operator:

.. code-block:: yaml

   expand:
     observable: ["phi_a(x)", "phi_b(y)", "phi_c(z)"]
     orders: [1]

   sweep:
     positions_grid: {x: [0.0], y: [0.8], z: [-0.5]}
     t_final_grid: [1.7]
     component_tuples: [[0, 1, 1], [1, 1, 1]]

Each result row carries the tuple in the columns ``a``, ``b``, ``c``, ...,
named by position, whatever index letters the observable uses, so a 2-point
sweep keeps its ``a``, ``b``.  ``SweepResult.totals()`` groups by them.
The same argument is available in Python as
``Expansion.sweep(component_tuples=[(0, 1, 1), ...])``;
``component_pairs`` is its older name and still takes 2-point pairs.

Refused rather than mis-answered: a tuple whose length differs from the
number of operators, an index outside ``0 .. N-1``, a tuple or a grid value
listed twice (``totals()`` would add the copies together), and a spatial
label that collides with a component column.

``examples/demo4/config_level_a.yaml`` and ``config_level_a_4pt.yaml`` run
demo 4's free-field 3- and 4-point functions this way, for every component
triple and quadruple.


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
   * - ``c_closed_form``
     - ``str``
     - ``"auto"``
     - How C is obtained without quadrature. ``"auto"`` (default) uses the built-in closed form when the kernel family has one (diagonal constant drift + separable exponential-temporal noise, optionally a constant white-noise impulse) and falls through to ``c_method`` otherwise; ``null`` forces quadrature. A ``c_closed_form_module`` supplies your own closed form and takes precedence over both. See :ref:`integrator-choice` below
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
     - ``"auto"``
     - Quadrature for the inner :math:`\int R\kappa^2 R`, used only when no closed form applies. ``"auto"`` (default) is Gauss-Legendre for the package's own kernels, with the node count refined from ``c_n_gauss`` until the rule is converged at the table's extreme cells, and ``"dblquad"`` for user callables; ``"gauss_legendre"`` forces fixed-``c_n_gauss`` GL (18-100× faster on piecewise-analytic κ²) and ``"dblquad"`` forces the adaptive rule. See :ref:`integrator-choice` below
   * - ``c_n_gauss``
     - ``int``
     - ``20``
     - GL nodes per dimension when ``c_method: gauss_legendre`` (cost ``c_n_gauss²`` per sub-region)
   * - ``diag_C``
     - ``bool``
     - ``true``
     - ``false`` keeps the off-diagonal entries of C: every table holds all :math:`N^2` entries :math:`C_{ab}` and every lookup returns them. Needed when C is not component-diagonal (a dense R, a component-mixing :math:`\kappa^2`, a matrix :math:`\sigma^2`), for which ``true`` is refused. No closed form is required: a Gauss-Legendre cell costs what a diagonal one costs, a ``dblquad`` cell :math:`N^2` adaptive integrals instead of :math:`N`. Sets ``expand.diag_C`` to ``false`` with it
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
     - Directory or file for caching the built propagator. A schema change invalidates older workflow caches automatically; upgrading to 0.6.1 recomputes cached propagators and expansions on first use. See :doc:`discretization`

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
     - One key per spatial argument in ``expand.observable``, each with a list; a point the observable does not name, and a point it names but the grid omits, are refused when the file is loaded.  Quote a name YAML would read as a number or a boolean (``'1'``, ``'on'``).  Cartesian product
   * - ``t_final_grid``
     - ``list[float]``
     - **required**
     - External upper-time values to evaluate at.  Also the default time for
       any external point not named in ``external_times_grid``
   * - ``external_times_grid``
     - ``dict[str, list[float]]``
     - optional
     - Pin external points at unequal times, swept as a further Cartesian
       axis.  Required for two-time observables; see
       :ref:`two-time-observables`
   * - ``component_tuples``
     - ``list[list[int]]``
     - **required**
     - One component index per observable operator: ``[[0, 0], [1, 1]]`` for a 2-point observable, ``[[0, 1, 1]]`` for a 3-point one.  Each tuple is a further Cartesian axis and fills the result's component columns ``a``, ``b``, ``c``, ...  ``component_pairs`` is the older name of the same axis; give one or the other.  See :ref:`n-point-sweeps`
   * - ``orders``
     - ``int``, ``list[int]`` or ``null``
     - ``null``
     - Optional subset of ``expand.orders``; one integer is one order, so ``--override sweep.orders=2`` works.  An order the expansion does not hold raises before the propagators are built.  An order at which the expansion holds no diagram (order 1 with only a cubic vertex) gives one row per grid point with value 0
   * - ``vertex_types``
     - ``str``, ``list[str]`` or ``null``
     - ``null``
     - Filter diagrams by vertex composition (e.g. ``["FK"]`` for demo2's cross-term).  One string is one label, so ``vertex_types: FK`` and ``--override sweep.vertex_types=FK`` select the FK channel.  A label joins the sorted names of a diagram's distinct vertices (two F vertices give ``F``; order 0 gives ``""``).  A label no diagram at the selected orders carries raises before the propagators are built, and the message lists the labels present.  Under ``target: connected`` a label such a diagram could carry (the sorted names of at most that many vertices) is accepted and gives the zero coefficient the connected target reports.  ``System`` refuses vertex names that give two sets one label (``F``, ``K`` and ``FK``) and two vertices with one name
   * - ``integrate_over``
     - ``null`` / ``"all"`` / ``str`` / ``list[str]``
     - ``null``
     - External-time integration: ``null`` = fixed at ``t_final``, ``"all"`` = integrate every external, list = explicit subset.  One string other than ``all`` is one point name, so ``integrate_over: x`` and ``--override sweep.integrate_over=x`` integrate the time of ``x``.  A name ``expand.observable`` does not use is refused when the file is loaded
   * - ``method``
     - ``str``
     - ``"qmc_vectorized"``
     - ``"qmc_vectorized"`` / ``"qmc"`` / ``"qmc_scalar"`` / ``"gauss_legendre"`` / ``"nquad"``. See :ref:`integrator-choice` below
   * - ``n_samples``
     - ``int``
     - ``8192``
     - Sobol sample count (QMC paths only)
   * - ``seed``
     - ``int`` or ``null``
     - ``42``
     - Sobol seed (QMC paths only); ``null`` draws an unseeded sequence
   * - ``n_gauss``
     - ``int``
     - ``8``
     - GL nodes per dimension (``method: gauss_legendre`` only). Cost scales as ``n_gauss^d`` per consistent order of the kinked time pairs (2 for one pair, up to ``k!`` for ``k`` mutually unordered times; see :ref:`integrator-choice`)
   * - ``n_jobs``
     - ``int``
     - ``1``
     - Parallelise across grid points (``-1`` = all cores). Mutually exclusive with ``expand.n_jobs > 1``
   * - ``gl_chunk_size``
     - positive ``int`` or ``null``
     - ``4096``
     - Maximum GL nodes per batch. A grid of at most this many nodes is the eager grid; a larger one is generated and evaluated in blocks, including every causal/kink piece. ``null`` keeps the eager grid at any size. Used only with ``method: gauss_legendre``; see :ref:`gl-batch-memory`
   * - ``reuse_integrands``
     - ``bool``
     - ``false``
     - Prepare static coupling tensors and spatial analysis once per component tuple within this sweep. Dynamic diagrams keep the ordinary preparation path. See below for the memory trade-off
   * - ``qmc_coupling_chunk_size``
     - positive ``int`` or ``null``
     - ``null``
     - Maximum rows per dynamic coupling construction on vectorized QMC paths. Requires sample-separable raw callbacks with stable dtype and trailing shape. R/C and final statistics remain whole; see below

Reusing static integrands in a sweep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For configuration feedback and an explicit comparison of time/memory choices,
see :doc:`resources`.

Set ``sweep.reuse_integrands: true`` in YAML, or pass
``reuse_integrands=True`` to ``Expansion.sweep``, when many positions or
times share the same static coupling tensors. Each requested component tuple
gets its own prepared integrands, retaining the exact diagram weights, ordered
tensor slots, equal-time aliases and absorbed response lines. Propagators,
coordinates, integration nodes and numerical error estimates are still evaluated
at each grid point. The option works with either sweep or diagram parallelism;
the existing restriction against nested pools remains.

The bindings last for one sweep call. Later calls rebuild them from the current
couplings, so edits to input arrays between sweeps are observed. Coupling callables
are evaluated through the ordinary path at the current sample points, and their
results are never cached. Keep model tensors fixed during a sweep.

Preparation now retains coefficient arrays for all requested component tuples
instead of one point at a time. Memory therefore grows with the component-axis
size and each diagram's surviving tensor dimensions; process workers may carry
additional copies. A short sweep or expensive integration may see little benefit.
The default remains ``false``. Compare full timings including preparation at the
same integrator settings before enabling it for a large model.

Indexed static and dynamic batch-contraction paths no longer prune coefficients
solely because their magnitude is below ``1e-20``. Only exact-zero entries are
skipped by the sparsity check; a reciprocal rescaling of a small coefficient and
a large kernel therefore retains its contribution. This correction applies with
or without ``reuse_integrands``. Existing MSR phase projection tolerances remain.

Reusing coupling expression structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Large native coupling sums with repeated component assignments prepare a private
ordered execution plan on the first evaluation of each diagram term. It retains
symbol names, index rules and operation order; each evaluation reads the current
coupling tensors and component pins. Scalar and batched arithmetic retain their
existing conversion and summation rules. No callback results or propagator values
are retained by this plan, and no user option is required.

Leaf-only and small expressions, one-assignment contractions, custom expression
nodes and long integer labels keep the ordinary evaluator. Each immutable term
owns its expression wrapper, which rebuilds on pickle. A bounded process-local
cache shares executable cores for matching ordered native metadata after
parameterising component labels. Each wrapper keeps the caller's label mapping;
literal spellings, symbol names, spatial metadata and operation order stay
explicit. This shares execution metadata while every diagram and integration
remains separate. The cache retains no caller tensors or pins.

Within large plans, repeated ordered symbol-index tuples are resolved once per
component assignment when dictionaries, integer pins and numeric arrays have no
custom lookup or indexing hooks. Large batched plans can also reuse indexed
native views or scalars within that assignment, reading each at its first
original occurrence. Every multiplication and addition still occurs in its
original order. No-index reshape results are excluded because they may own
large copies. Changed tensors and pins are read on every new invocation;
custom bindings or arithmetic handlers retain the existing path and errors.

Large scalar plans can also reuse successfully converted indexed reads within
one component assignment for ordinary native NumPy numeric dtypes. Every scalar
shape check and conversion still occurs at the first original occurrence, and
all arithmetic keeps its original order. No-index Symbols stay live. A new
assignment or call reads current inputs again. Extended/custom dtypes, dtype
metadata, custom conversion or arithmetic hooks, and active trace/profile hooks
keep the preceding scalar read policy. The assignment memo retains only
converted Python scalars; numeric arrays remain live inputs on each new call.

Dynamic batches can avoid the sample-zero scalar contraction when conservative
native checks prove its scalar shape, indexing and finite arithmetic will succeed.
The proof limits full-tensor scanning and scalar output size to 4096 elements
respectively; a large tensor with a small pinned slice therefore keeps the old
probe. Custom floating-point/warning handlers, nonfinite values, uncertain shapes,
unsupported nodes and changed NumPy helper protocols also retain scalar-first
evaluation. Unrecognised helper implementations conservatively decline these
optimisations. Batch fallback uses
the exact scalar result and reuses the already materialised tensor stacks.

These mechanisms complement static sweep reuse and bounded GL batches. Their
preparation, numeric checks and metadata have a cost: whole integration timings
and cold/warm measurements still determine their benefit for a workload.

.. _gl-batch-memory:

Bounding Gauss-Legendre batch memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Gauss-Legendre integral over ``d`` time variables visits ``n_gauss^d``
nodes, and the eager grid holds coordinates, mapped times and propagator
arrays for all of them at once, so its memory grows with the same power.
``sweep.gl_chunk_size`` (default ``4096``) caps the nodes held at once.  A
grid of at most ``gl_chunk_size`` nodes is the eager grid, with the same
value to the last bit; a larger grid is evaluated in blocks of that many
nodes.  ``null`` / ``None`` keeps the eager grid at any size.  The option is
accepted by ``Expansion.evaluate`` and ``Expansion.sweep``, and by the L0
``integrate_moment``, ``integrate_diagrams`` and
``DiagramIntegrand.integrate_moment_gauss_legendre``.  It must be a positive
integer.  Other integration methods validate the option but do not use it.
The CLI summary displays the GL bound, and the ``--dry-run`` timing probes
use it for warm-up and measured evaluations.

The default rarely needs changing.  Lower it for wide dynamic couplings:
each row of a block holds one raw coupling tensor per dynamic coupling
symbol, ``N^m`` complex entries for a callable ``κ^(m)``, so with ``N^m``
in the thousands a block of 4096 rows is itself large.  Lower it only as far
as the memory requires, because each block carries a fixed Python cost: on
an order-4 grid of 4096 nodes per diagram, a bound of 1024 took 10 % longer
than eager GL and 256 took 40 % longer, and on an order-4 sweep 512 took
twice as long and 64 ten times as long.  Bounds of 4096, 16384 and 65536
took the same wall time as eager GL to within 5 % in every measurement,
with one thread per process:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Workload (``integrate_over: all``)
     - Bound
     - Wall time
     - Peak RSS
   * - order 2, ``N = 2``, ``n_gauss = 24`` (331 776 nodes)
     - eager / 65536 / 4096
     - 18.2 / 18.0 / 18.4 s
     - 812 / 452 / 127 MB
   * - order 4, ``N = 2``, ``n_gauss = 7`` (117 649 nodes)
     - eager / 65536 / 4096
     - 263 / 265 / 263 s
     - 1272 / 775 / 212 MB

Both rows returned the same total for every bound.  The script is
``tools/benchmark_gl_chunk_size.py``.

The bound removes the out-of-memory failure that used to stop a grid sized
far beyond the machine, so such a grid now runs until it finishes.
``sft-wick run --dry-run`` times one diagram evaluation per order at one
grid point and prints the projected cost before anything runs.

The tensor grid is generated directly in blocks, so node coordinates, mapped
times, propagator arrays and dynamic coupling batches have at most
``gl_chunk_size`` rows. This bounds their temporary workspace; it does not
bound the model, propagator caches or static component tensors. Every one of
the ``n_gauss^d`` nodes and every kink partition is still evaluated. Small
blocks repeat dispatch work and can increase wall time. Parallel workers each
hold their own block arrays, so worker count still affects total memory.

The same causal mapping, Jacobians, equal-time and absorbed aliases, external
time limits and component contractions apply within each block. Changing
reduction order can change the rounded result; block totals use
``math.fsum``, while each block retains the existing NumPy reduction. Compare
cancelling observables with a suitable absolute tolerance. The returned GL
error remains ``0.0`` as a deterministic placeholder; it does not certify
quadrature convergence.

Batch evaluation is dispatched separately for each block. User-provided
vectorised R, C and coupling hooks must define each sample independently of
batch length or invocation count.
Dynamic coupling phase checks preserve the scale of the entire final kink
partition, separately per component assignment, without retaining sample arrays
or reevaluating samples solely for phase validation. Fixed vector positions are
explicitly broadcast so a
three-node block cannot reinterpret a fixed three-dimensional position as three
scalar positions.

One-dimensional GL rules are model-independent and read-only. A process-local
LRU retains at most 32 rules keyed by node count; it stores no tensor-product
grid, geometry or model data. This is a bound on rule count, rather than bytes.
Rules can be reused across sweep calls; static integrand bindings retain their
one-sweep lifetime described above.

Declared kinks against a fixed external time are also split when the other
endpoint is an integrated external point, including an absorbed vertex's
partner time. Multiple integrated external predecessors in a causal ``max``
lower bound are also ordered, and lower cuts are propagated to resolve the resulting
branch changes. This corrects omitted splits in the previous GL and nquad
paths independently of ``gl_chunk_size``. It can change their values at fixed
resolution for these cases; the quadrature rule on each resulting piece is
unchanged.

Bounding dynamic coupling stacks in QMC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``sweep.qmc_coupling_chunk_size: 512`` alongside
``sweep.method: qmc_vectorized`` in YAML, or pass
``qmc_coupling_chunk_size=512`` to ``Expansion.evaluate`` or
``Expansion.sweep``. The L0 ``integrate_moment``, ``integrate_diagrams`` and
``DiagramIntegrand.integrate_moment_qmc_vectorized`` accept the same option.
A positive Python or NumPy integer is required; booleans are rejected.
``null`` / ``None`` keeps the original single coupling call. Static
couplings, integrals with no time variables and a bound at least
``n_samples`` also keep their original path. ``method: qmc`` uses the option
only when its cache selects vectorized QMC. Scalar QMC, GL and nquad validate
the option but do not use it.

This option partitions only construction of dynamic symbol tensors and their
coupling contractions. It assembles one complete contracted coupling array,
then uses the original whole-batch R/C support selection, component phase
checks, Jacobians, causal masking, mean and eight-batch error estimate.
The same single Sobol draw, row order, external times, equal-time aliases and
fixed spatial coordinates are retained. It does not partition propagator
lookups or substitute an average of block means.

Each **raw symbol callback** must compute every sample independently of batch
length, other samples and invocation count, with stable dtype and trailing
shape. Model values, component pins and callback state must remain fixed for
the evaluation. ``vectorized=True`` only declares batch support; it does
not establish these additional properties. Batch normalization, callbacks
that branch on batch length and stateful invocation-dependent kernels are
outside this opt-in contract. Contracted block outputs are checked for numeric
dtype, exact row count and stable dtype/trailing shape before copying. These
checks cannot prove raw callback stability: the built-in promise's complex
output can conceal a raw dtype change.

Vectorized callbacks are called once per symbol per block, in chunk-major
order rather than the original symbol-major order. Nonvectorized callbacks
still evaluate each sample for each active symbol, but their grouping and
exception order can change; symbolic sample-zero shape probes can repeat per
block. Input time views remain independently addressable and are not recycled
through an overwritten scratch buffer. Callbacks retaining inputs or outputs
can prevent memory reclamation.

The bound limits rows in raw dynamic symbol stacks. Full Sobol coordinates,
mapped times, R/C arrays, the contracted coupling output and final values
remain allocated, so this is **not a bound on total memory or RSS**. Native
``OUWhiteCumulant.max_output_bytes`` is checked against each actual requested
batch; a smaller request can fit when the original whole-batch request was
rejected. Its component overflow guards still apply to every request. Native
OU ordinary/scaled arithmetic admission can depend on the requested batch,
so arbitrary partitions can change floating-point rounding. No universal
bitwise equality is promised.

Smaller blocks repeat dispatch, allocations and contractions and can increase
wall time. Benchmark an appropriate block size for the actual workload;
there is no automatic activation. Each parallel worker owns its arrays, and
worker count still affects total memory. The CLI displays a configured QMC
bound and forwards it to dry-run evaluations. Those timing probes use fewer
samples than a large production run and do not certify its memory peak.

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

1. **C-propagator construction** (``propagators.c_method``): the
   inner :math:`\int_0^{t_1}\!\int_0^{t_2} R(t_1,\lambda_1)\,
   \kappa^2(\lambda_1,\lambda_2)\,R(t_2,\lambda_2)\,d\lambda` for
   each :math:`(t_1, t_2, n_1, n_2)` cache cell. Skipped entirely
   when a closed form applies: the built-in one under
   ``c_closed_form: auto`` (the default), or your own via
   ``c_closed_form_module``.
2. **Diagram evaluation** (``sweep.method``): the outer
   :math:`d`-dimensional time integral over the causal simplex,
   one evaluation per ``(positions, t_final, component_pair)``
   grid point.

For both layers the choice is the same trade-off: **adaptive
robustness** (``dblquad`` / ``nquad``) vs **smooth-integrand
speed** (``gauss_legendre``).

.. tip::

   **Gauss-Legendre is not Gaussian/exponential-only.** It gives
   exponential convergence on **any function analytic on the
   integration domain**: polynomials, rationals, exponentials,
   trigonometrics, products and compositions thereof.

   When the integrand is only **piecewise** analytic (e.g. an
   ``|λ1 − λ2|`` cusp on the diagonal of κ²), splitting the
   domain along the non-smooth boundary recovers full GL
   convergence on each piece. The package's
   ``c_method='gauss_legendre'`` does this split automatically for
   the C table.  ``sweep.method='gauss_legendre'`` does it for the
   diagram's time domain, where a kinked C (white noise, a κ² with a
   ``|Δt|`` cusp, or a closed-form C that declares
   ``has_diagonal_kink``) or a vertex with several ψ legs at one time
   kinks the integrand, and keeps exponential convergence.  A kink
   inside a coupling callable, such as a ``min`` over partner times in
   an R-contracted cumulant, is split once the callable declares
   ``has_coincident_time_kinks`` and keeps the algebraic rate
   otherwise (see :ref:`declaring-kinks`).

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
     - High-d diagrams (``d ≥ 6``), a kink the package cannot see (an undeclared one inside a coupling callable), or when stochastic error bars are wanted
     - ``~ 1/√n_samples`` bias decay; can severely under-resolve narrow peaks at large ``t_final``; not split at kinks
   * -
     - ``gauss_legendre``
     - Smooth integrands at ``d ≤ 5`` (the typical sft-wick case), and kinked-C, several-ψ-leg or declared coupling kinks, which it splits out against another time or against a fixed external one (:ref:`declaring-kinks`)
     - **Exponential convergence** in ``n_gauss``; deterministic; cost ``n_gauss^d`` per piece (2 per kink pair or cut, up to ``k!`` for ``k`` mutually unordered times)
   * -
     - ``nquad``
     - Adaptive 1-3D fallback when GL nodes are insufficient
     - Slow; split at kinks like ``gauss_legendre``; a callable coupling is evaluated once per quadrature point and symbol, so cost grows with the adaptive subdivision
   * -
     - ``qmc`` / ``qmc_scalar``
     - Compatibility / single-sample debugging
     - Slow scalar Python loop; not for production sweeps

Every ``sweep.method`` accepts a matrix-valued R (``DiagonalA`` with
component-dependent rates, ``ExplicitR(iso_R=False)``).  The batched
methods select R's component entries per index assignment on the whole
batch of nodes, as they do for C.  Before, ``gauss_legendre`` and
``qmc_vectorized`` raised ``NotImplementedError`` for a matrix R, and only
``qmc`` / ``qmc_scalar`` and ``nquad`` evaluated it.

User-Python hooks
~~~~~~~~~~~~~~~~~

The YAML block can defer to user Python in eight places.  Each loads
a callable from a ``.py`` module relative to the YAML file and
registers it for joblib's worker-safe by-value module loading, so it
composes cleanly with any of the ``n_jobs > 1`` knobs:

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
     - ``R_time(t1, t2) → float`` or ``→ (N, N)`` (matrix; opt-in via ``linear.iso_R: false``)
     - Closed-form ``R`` (escape hatch via ``linear.type: explicit``; bypasses the γ-spline cache)
   * - ``system.noise.kappa2.type: callable_module``
     - ``κ²(n1,t1,n2,t2) → (N, N)``
     - Non-separable noise correlator (replaces ``separable_translation``/``separable_rotation``)
   * - ``system.noise.kappa2.<block>.type: custom``
     - ``κ(Δt) → float``, ``κ(r) → float`` or ``κ(cos θ) → float``
     - One kernel of a separable κ² outside the built-in families (demo 3's spatial envelope, a sum of exponentials); ``module`` + ``attr``, whose default is the block's own name
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
config (safe scalar coercion, no ``eval``):

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

The homogeneity is inferred from the spec automatically and does not
normally need to be set directly.

Explicit lookup reuse and contraction order
-------------------------------------------

The ``sweep`` YAML block and :meth:`~sft_wick.workflow.Expansion.evaluate` /
:meth:`~sft_wick.workflow.Expansion.sweep` accept
``propagator_cache_bytes``, ``propagator_cache_contract``,
``contraction_backend`` and ``contraction_max_bytes``. Defaults preserve the
ordered evaluator and disable lookup caching. See :doc:`resources` for the
fixed-model promise, admitted dense and original-factor subsets, memory boundaries and complete
result comparison. These options do not change sampling or quadrature nodes.

``contraction_backend: einsum`` and ``contraction_backend: factorized``
evaluate each static diagram's sum over component assignments in pairwise
contractions instead of one assignment at a time; ``factorized`` contracts
the original coupling tensors of each monomial and does not build the
coefficient tensor. Their results lie within the error bound stated in
:doc:`resources`, which the ordered fold also satisfies, and differ from the
ordered results in the last bits. A diagram call runs on the ordered fold
when a backend cannot keep that bound, would raise a different error, or is
estimated to be slower; small diagrams therefore keep the ordered fold.
``factorized`` prepares a sum of monomials only when no two monomials have
opposite signs at any component assignment, because monomials that cancel
would break the bound. On a 330-diagram test set at 1024 to 65536 samples,
``einsum`` ran on 204 to 310 of the diagrams and its static contraction was
2.2 to 103 times faster; ``factorized`` ran on 2 to 4 of them, 2.8 to 118
times faster, the other coupling sums having monomials of opposite sign.
The end-to-end gain is smaller (0.95 to 1.85 times the ordered speed in the
measured runs) because most of an evaluation is spent outside the static
contraction. :doc:`resources` lists the admission rules, the cost rule, the
workspace budget and the measured speeds.

Persisted propagator tables share only supported, unchanged native linear and
noise identities. Complete array bytes are keyed; explicit response and user
closed-form callbacks bypass shared disk caching. Their arbitrary external
state cannot be automatically identified. Query-local lookup caching remains
controlled by its separate ``fixed_model`` promise.
