Time discretization (``dt``)
============================

The L2 YAML config accepts a single top-level ``propagators.dt`` knob that
controls every internal time-grid resolution coherently. This page explains
its physical meaning and how it interacts with the ``noise.kappa2`` /
``noise.sigma2`` source-cumulant split.

What ``dt`` means
-----------------

``dt`` is the smallest temporal correlation length that the framework can
resolve. The two numerical grids it controls are:

* ``propagators.n_grid_t`` — number of points along each axis of the
  ``C(t1, t2)`` interpolation table built by :meth:`PropagatorCache
  <sft_wick.evaluate.PropagatorCache>`. Derived as
  ``ceil(t_max / dt)``.

* ``system.linear.n_grid_cache`` — number of points along the
  cumulative-Γ spline used by :class:`DiagonalA
  <sft_wick.workflow.specs.DiagonalA>` for time-dependent linear drift.
  Derived as ``ceil(t_max_cache / dt)``.

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

A finer resolution simply needs ``dt: 0.5`` (which doubles both grid sizes).
The ``linear`` block can override the default with its own ``dt:`` field
when the gamma cache needs different resolution from the C-table.

When to use ``kappa2`` vs ``sigma2``
------------------------------------

The two source-cumulant slots are *complementary*, not additive:

* ``noise.kappa2`` carries modes whose temporal correlation length is
  resolvable by the time grid (correlation length ``>> dt``). These show
  up as smooth functions of ``(t1, t2)``.

* ``noise.sigma2`` carries the white-noise complement: modes whose
  temporal correlation length is below ``dt`` are observationally
  indistinguishable from a delta-in-time and contribute through the
  noise-amplitude term ``sigma2(t) delta(t1 - t2)``.

Routing each mode to exactly one of the two slots avoids double-counting.
A typical workflow first builds ``kappa2`` from a multipole-resolved
angular power spectrum truncated at some ``ell_cut``, then computes
``sigma2`` from the *complement* of that resolved range in k-space (e.g.,
integrating the matter power spectrum from ``k_cut(chi) = (ell_cut + 0.5)
/ chi`` upward).

Choosing ``dt``
---------------

A defensible default is ``dt = t_max / 60`` (the legacy ``n_grid_t = 60``
default, made explicit). Halve it for convergence checks; observables
should change only at order ``dt`` if all the unresolved power has been
moved to ``sigma2``.

Convergence
-----------

If halving ``dt`` causes the observable to jump by an O(1) fraction, the
source spectrum still has unresolved smooth power being captured by
``kappa2`` interpolation at borderline scales. Move that power to
``sigma2`` (raise ``ell_cut`` or, equivalently, lower the k-space cut)
until the observable is dt-stable.


Parallelism layers
------------------

The L2 YAML workflow exposes three independent ``n_jobs`` knobs, one per
layer of the pipeline. Each layer can run sequentially or via joblib's
loky backend; choose the layer that has the most independent units in
your workload.

============================  =================================  ==========================  ====================================================
Layer                         YAML knob                          Parallel unit               When to use
============================  =================================  ==========================  ====================================================
Propagator C-table build      ``propagators.n_jobs``             one ``(t1, t2, cos)`` cell  Always — runs once before sweep, large grid.
Diagram QMC integration       ``expand.n_jobs``                  one Feynman diagram         Many diagrams per grid point (typical at orders >= 2).
Sweep grid                    ``sweep.n_jobs``                   one grid point              Sweep grid is large; few diagrams per point.
============================  =================================  ==========================  ====================================================

Defaults are all ``1`` (sequential, backward-compatible). Pass ``-1`` to
use all CPU cores.

.. important::

   ``expand.n_jobs > 1`` and ``sweep.n_jobs > 1`` are **mutually
   exclusive** — joblib's loky backend does not support nested process
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
  ``positions × t_final × component_pairs`` in :meth:`Expansion.sweep`,
  one grid point per worker. Each worker calls ``evaluate`` with
  ``n_jobs=1`` (no nested pool).
* User-supplied callable modules (``noise.kappa2.callable_module``,
  ``vertices.coupling_module``, ``propagators.c_closed_form_module``)
  are loaded under their ``.py`` file's bare basename and added to
  ``sys.path`` — picklable across loky boundaries. See
  :func:`sft_wick.workflow.config._load_callable_from_module` and
  :func:`sft_wick.workflow.config._load_c_closed_form`.

Verification
~~~~~~~~~~~~

Tests ``CF8_expand_n_jobs_matches_sequential`` and
``CF8_sweep_n_jobs_matches_sequential`` lock in float64 bit-identity of
the parallel paths against the sequential reference (when ``seed`` is
fixed, which is the default for all sweeps).
