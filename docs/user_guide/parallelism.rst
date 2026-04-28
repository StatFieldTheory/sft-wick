Parallelism (``n_jobs``)
========================

The L2 YAML config exposes three independent parallelism knobs, plus an
implicit fourth layer that runs at expansion time.  This page explains
which knob to use, the **mutual-exclusion rules** that prevent nested
process pools, and the BLAS thread-oversubscription pitfall that bites
every multi-process Python numerics pipeline.

The four parallelism layers
---------------------------

==========================================  ============================  =========================
Layer                                       YAML knob                      What gets parallelised
==========================================  ============================  =========================
**L1 — C-table builder**                    ``propagators.n_jobs``        Grid points
                                                                          ``(t1, t2[, r/cos/x])``
                                                                          inside
                                                                          ``precompute_C_table_*``
**L2 — diagram-term integration**           ``expand.n_jobs``             Per-diagram QMC integration
                                                                          inside
                                                                          ``integrate_diagrams``
**L3 — sweep grid points**                  ``sweep.n_jobs``              Cartesian product of
                                                                          ``positions × t_final ×
                                                                          component_pairs``
**L4 — topology canonicalisation**          (no YAML key yet)             Inside
                                                                          ``compute_moment_numerical``
                                                                          when the topology list is
                                                                          long
==========================================  ============================  =========================

All four layers use ``joblib.Parallel(backend='loky')``.  Each defaults
to ``1`` (sequential).  ``-1`` means *all CPU cores*.

Choosing where to parallelise
-----------------------------

* If your **sweep grid is large** (many positions × t_final × component
  pairs), set ``sweep.n_jobs: -1``.  This is the most common case.
* If your **expansion has many diagrams per grid point** (typical at
  perturbative order ≥ 2), set ``expand.n_jobs: -1`` and leave
  ``sweep.n_jobs: 1``.  Useful when the sweep grid is small but
  individual evaluations are expensive.
* If the dominant cost is the **C-table pre-build** (slow user
  ``c_closed_form_module``, dense ``r_max`` / ``n_grid_r`` grid), set
  ``propagators.n_jobs: -1``.  This finishes before any sweep starts,
  so it composes safely with either L2 or L3 parallelism.

Mutual exclusion (no nested pools)
----------------------------------

joblib's ``loky`` backend cannot nest: a worker process inside one pool
cannot itself open another pool.  The library enforces this by raising
``ValueError`` whenever you would request both:

.. code-block:: text

    expand.n_jobs > 1   AND   sweep.n_jobs > 1     -> ValueError

For the L1 × {L2, L3} combination there is **no error** because the
guard is automatic: when the C-table goes into lazy mode (``r_max`` /
``n_grid_r`` etc. unset, the default), the library pins the lazy
spline cache's inner ``n_jobs`` to ``1`` after
``Propagators.build()`` completes.  That removes the only path along
which an L2 or L3 worker could trigger a nested pool.  Full-grid mode
runs to completion before any sweep starts, so there is no nesting
risk there either.

BLAS thread oversubscription
----------------------------

NumPy and SciPy operations inside each loky worker -- matrix products,
``scipy.integrate.dblquad``, ``RectBivariateSpline.__call__`` -- are
themselves multi-threaded by default through OpenBLAS or MKL.  Without
intervention, ``N`` worker processes each running ``N`` BLAS threads
yields ``N**2`` total threads, leading to lock contention and a slowdown
*worse* than running serially.

**Always set the following before invoking sft-wick with any
``n_jobs > 1``:**

.. code-block:: bash

    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OMP_NUM_THREADS=1

The CLI prints a one-line tip when it detects ``n_jobs > 1`` without
those variables set; you can disable the tip by setting any of them
explicitly (even to ``1`` is enough -- the tip suppresses itself once
it sees a value).
