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
caches it as a cubic spline for O(1) R lookups.  Full-matrix
time-dependent A is currently supported only via
:class:`~sft_wick.workflow.ExplicitR` with a user-supplied
time-ordered matrix exponential.

Dynamic coupling (spacetime-dependent κ^(m))
--------------------------------------------

For non-local vertices whose coupling tensor depends on the sample
points, pass a callable to
:class:`~sft_wick.workflow.NonLocalVertex`:

.. code-block:: python

   def k3_coupling(n_list, t_list):
       """Shape (N,)*3 tensor at the three ψ-leg spacetime points."""
       ...

   system = sw.System(
       ...,
       nonlocal_vertices=[
           sw.NonLocalVertex("K", order=3, coupling=k3_coupling),
       ],
   )

The runtime detects the callable in ``build_integrand`` and routes
it through :class:`~sft_wick.evaluate.DynamicCouplingPromise` for
per-sample materialisation inside the QMC loop.  If no callable is
passed, the static fast path is used automatically.

YAML schema reference (L2 details)
----------------------------------

The YAML schema mirrors the L1 constructor signatures exactly, so
anything expressible in L1 Python is expressible in YAML.  Top-level
keys:

``system``
    ``field``, ``linear``, ``noise``, ``vertices``,
    ``nonlocal_vertices``, ``t_min``.  Each nested object uses a
    ``type:`` discriminator plus its constructor args (e.g.
    ``linear: {type: diagonal, gamma: [...]}`` →
    :class:`~sft_wick.workflow.DiagonalA`).

``expand``
    ``observable: [...]``, ``orders: [...]``, optional
    ``cache_path``, optional ``n_jobs`` (parallelise per diagram
    inside each grid point — see :doc:`parallelism`).

``propagators``
    ``t_max``, optional ``n_grid_t`` *or* ``dt`` (single-knob
    discretization, see :doc:`discretization`), ``n_jobs``,
    optional ``c_closed_form_module`` + ``c_closed_form_attr``,
    ``cache_path``, ``interp_method`` (``'linear'`` default,
    ``'cubic'`` opt-in).

``sweep``
    ``positions_grid``, ``t_final_grid``, ``component_pairs``,
    optional ``integrate_over``, ``vertex_types``, ``orders``,
    ``method``, ``n_samples``, ``seed``, ``n_jobs``
    (parallelise across grid points — mutually exclusive with
    ``expand.n_jobs > 1``).

``output`` (optional)
    A list of output plugins.  Current plugin types: ``table``
    (markdown / csv), ``npz``, ``plot`` (matplotlib grid).

Hooks for user Python code (without editing the CLI):

- ``propagators.c_closed_form_module`` + ``c_closed_form_attr`` —
  dotted path to a module exporting a
  ``C_fn(n1, t1, n2, t2) → value`` callable.  The loaded module
  is registered with cloudpickle for by-value cross-process
  serialisation, so it composes cleanly with ``propagators.n_jobs
  > 1``, ``expand.n_jobs > 1``, and ``sweep.n_jobs > 1`` even when
  joblib reuses a worker pool across calls.
- ``nonlocal_vertices[].coupling_module`` +
  ``nonlocal_vertices[].coupling_attr`` — dotted path to a
  dynamic-coupling ``fn(n_list, t_list) → tensor`` callable
  (e.g. demo2's ``κ^{(3)}``).
- ``system.linear.gamma_module`` — dotted path to a callable
  ``γ(t) → array(N)`` for time-dependent linear drift.
- ``noise.kappa2.type: callable_module`` — dotted path to a
  ``κ²(n1, t1, n2, t2) → (N, N)`` callable for non-separable
  noise correlators.

Parameter scans from the shell:

.. code-block:: bash

   for seed in 0 1 2 3 4 5; do
       sft-wick run demo1_config.yaml \
           --override "sweep.seed=$seed" \
           --override "output[0].path=results_seed$seed.md"
   done

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
