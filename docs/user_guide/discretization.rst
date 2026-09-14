Time discretization (``dt``)
============================

The L2 YAML config accepts a single top-level ``propagators.dt`` knob that
controls every internal time-grid resolution coherently. This page covers
its physical meaning and how it interacts with the ``noise.kappa2`` /
``noise.sigma2`` source-cumulant split.

What ``dt`` means
-----------------

``dt`` is a numerical resolution setting used to derive grid sizes. It
does not change the noise model or automatically replace unresolved
colored noise with white noise. The two grids it controls are:

* ``propagators.n_grid_t``: number of points along each axis of the
  ``C(t1, t2)`` interpolation table built by :meth:`PropagatorCache
  <sft_wick.evaluate.PropagatorCache>`. Derived as
  ``max(2, ceil(t_max / dt))``. The physical time nodes include both
  endpoints, so their spacing is ``(t_max - t_min) / (n_grid_t - 1)``;
  it is not exactly ``dt``.

* ``system.linear.n_grid_cache``: number of points along the
  cumulative-Γ spline used by :class:`DiagonalA
  <sft_wick.workflow.specs.DiagonalA>` for time-dependent linear drift.
  Derived as ``max(2, ceil((t_max_cache - t_min_cache) / dt))``.

Specifying ``dt`` and ``n_grid_t`` (or ``dt`` and ``n_grid_cache``)
together is rejected at parse time.

Example
-------

::

    propagators:
      t_max: 100.0
      dt: 1.0          # -> n_grid_t = 100
      homogeneity: rotation

    system:
      linear:
        type: diagonal
        gamma_module: ./gamma.py
        t_max_cache: 200.0   # -> n_grid_cache = 200, sharing the same dt

A finer resolution needs ``dt: 0.5``, which doubles both grid sizes.
The ``linear`` block can override the default with its own ``dt:`` field
when the gamma cache needs different resolution from the C-table.

How the C table handles coincident times
----------------------------------------

Since 0.6.1, the translation, rotation and general spatial builders store
their time axes as ``s = min(t1,t2)`` and ``u = abs(t1-t2)``. This places
the time-diagonal kink at the boundary ``u = 0``. Both time orders are
retained, using covariance transposition where applicable. Callers still
supply the physical times ``t1`` and ``t2``.

Only ``t_min <= s <= s + u <= t_max`` is sampled from physical kernels.
Auxiliary spline values beyond that triangle are continued from the table
values. Full-grid ``interp_method: linear`` combines physical time-cell
vertices and splits cells crossed by the diagonal. Cubic and lazy spline
lookups use the transformed time coordinates. Closed-form-only lookups
(``c_closed_form_only: true``) bypass these tables, and the legacy L0
``precompute_C_table`` retains its original layout.

Workflow disk caches carry a schema identifier. Upgrading to 0.6.1 causes
older cached propagators and expansions to be recomputed on first use;
no manual deletion is needed. This applies to caches managed by the
workflow API, rather than arbitrary objects loaded directly with joblib.


When to use ``kappa2`` vs ``sigma2``
------------------------------------

The two source-cumulant slots contribute additively to the covariance:

* ``noise.kappa2`` describes noise with a finite temporal correlation
  kernel. Resolve its relevant scales by refining the grid and quadrature.

* ``noise.sigma2`` describes a delta-correlated component through
  ``sigma2(t) delta(t1 - t2)``. It can be an independent physical noise
  source or a separately justified white-noise approximation.

If both slots approximate parts of the same spectrum, routing each mode
to exactly one avoids double-counting. For example, a model may build
``kappa2`` from a multipole-resolved
angular power spectrum truncated at some ``ell_cut``, then compute
``sigma2`` from the *complement* of that resolved range in k-space (e.g.,
integrating the matter power spectrum from ``k_cut(chi) = (ell_cut + 0.5)
/ chi`` upward).

Choosing ``dt``
---------------

A defensible default is ``dt = t_max / 60`` (the legacy ``n_grid_t = 60``
default, written out). Halve it for convergence checks while keeping the
physical model fixed. The convergence rate depends on kernel smoothness,
interpolation and the outer integrator; no universal power of ``dt`` is
guaranteed.

Convergence
-----------

If halving ``dt`` changes the observable substantially, continue the grid
refinement and check the inner quadrature and outer integration accuracy
separately. Compare with a closed form when one is available. Replacing a
colored component by ``sigma2`` changes the physical model and requires
its own approximation check; it is not a numerical convergence repair.


Parallelism layers
------------------

The L2 YAML workflow exposes three independent ``n_jobs`` knobs, one per
layer of the pipeline. Each layer can run sequentially or via joblib's
loky backend; choose the layer that has the most independent units in
your workload.

============================  =================================  ==========================  ====================================================
Layer                         YAML knob                          Parallel unit               When to use
============================  =================================  ==========================  ====================================================
Propagator C-table build      ``propagators.n_jobs``             one physical table sample   Always: runs once before sweep, large grid.
Diagram QMC integration       ``expand.n_jobs``                  one Feynman diagram         Many diagrams per grid point (typical at orders >= 2).
Sweep grid                    ``sweep.n_jobs``                   one grid point              Sweep grid is large; few diagrams per point.
============================  =================================  ==========================  ====================================================

Defaults are all ``1`` (sequential, backward-compatible). Pass ``-1`` to
use all CPU cores.

.. important::

   ``expand.n_jobs > 1`` and ``sweep.n_jobs > 1`` are **mutually
   exclusive**: joblib's loky backend does not support nested process
   pools. The L2 workflow raises ``ValueError`` if both are set. Pick
   whichever layer parallelises a larger workload for your case.

   ``propagators.n_jobs`` is independent (runs before sweep) and can
   always be ``-1`` regardless of the other two.

Mechanics
~~~~~~~~~

* ``expand.n_jobs`` routes ``Expansion.evaluate`` through
  :func:`sft_wick.evaluate.integrate_diagrams`, which dispatches each
  ``DiagramTerm`` to a worker. Sequential fallback when
  ``len(diagrams) <= 2`` to avoid the ~1 s loky startup overhead.
* ``sweep.n_jobs`` parallelises the Cartesian product of
  ``positions × t_final × component_tuples`` in :meth:`Expansion.sweep`,
  one grid point per worker. Each worker calls ``evaluate`` with
  ``n_jobs=1`` (no nested pool).
* User-supplied callable modules (``noise.kappa2.callable_module``,
  ``vertices.coupling_module``, ``propagators.c_closed_form_module``)
  are loaded under their ``.py`` file's bare basename and added to
  ``sys.path``, which makes them picklable across loky boundaries. See
  :func:`sft_wick.workflow.config._load_callable_from_module` and
  :func:`sft_wick.workflow.config._load_c_closed_form`.

Verification
~~~~~~~~~~~~

Tests ``CF8_expand_n_jobs_matches_sequential`` and
``CF8_sweep_n_jobs_matches_sequential`` lock in float64 bit-identity of
the parallel paths against the sequential reference (when ``seed`` is
fixed, which is the default for all sweeps).
