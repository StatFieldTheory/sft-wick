``sft_wick.selfconsistency``
============================

The fixed-point iteration for a self-consistent (DMFT-style) solution:
propagators define a self-energy, the self-energy defines new propagators,
repeat.

What this is, and what it is not
--------------------------------

A self-consistent solution needs three pieces:

1. the self-energy from the current propagators,
2. new propagators from that self-energy — a Dyson / integral-equation solve,
3. the iteration itself, with the judgement about when it has converged.

sft-wick computes (1); that is what the whole package is for. This module
supplies (3).

.. important::

   **(2) is deliberately absent.** It is model-specific and is genuinely an
   integral-equation solve rather than a diagram evaluation, and a wrong
   general Dyson solver would be worse than none. You supply it as the body of
   ``step``.

Why the result is not a bare state
----------------------------------

A fixed-point iteration that has *not* converged looks exactly like one that
has, if you only print the last state. :func:`~sft_wick.solve_self_consistency`
therefore never returns a bare state — it returns a
:class:`~sft_wick.SelfConsistencyResult` carrying ``converged``, the full
residual history, and a ``reason`` distinguishing the ways it can fail:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - ``reason``
     - meaning
   * - ``"converged"``
     - the residual :math:`\|F(x) - x\|` fell below ``tol``
   * - ``"diverged"``
     - the residual is growing fast enough that more iterations will not help
   * - ``"oscillating"``
     - the state keeps returning to where it was while the residual stops
       improving — **use damping**
   * - ``"max_iter"``
     - none of the above; it simply ran out

``bool(result)`` is ``converged``, so ``if not result: ...`` is the natural
way to handle a failure.

.. code-block:: python

   from sft_wick import solve_self_consistency

   def step(state):
       sigma = self_energy_from_diagrams(state)   # sft-wick
       return dyson_solve(sigma)                  # yours

   result = solve_self_consistency(initial, step, tol=1e-8, damping=0.3)
   if not result:
       raise RuntimeError(result.summary())
   R, C = result.state

Four things it is careful about
-------------------------------

Each is a way to report a solution that was never found, and each has a
regression test.

**Damping does not shrink the residual.** The state moves by only
:math:`(1-d)` of the step, so a residual read off the *movement* falls as
damping rises — turn damping up and any iteration "converges" sooner, at a
point that is not a fixed point. The residual always measures the step,
:math:`\|F(x) - x\|`.

**``step`` cannot fake a zero residual by mutating.** It is handed a *copy* of
the state, and the proposal is copied before becoming the new state. Without
the second copy a ``step`` returning a buffer it reuses — ``buf[:] = ...;
return buf``, the standard preallocated-output idiom — would have the next
call overwrite the state in place *before* the residual was taken, comparing
the buffer with itself.

**A cycle is a statement about the state, not the residual.** A two-cycle has
a *constant* residual, indistinguishable from a stall by residual shape alone;
a *growing* residual is divergence, not oscillation. Cycles are found by
asking whether the state returned somewhere it had already been — but an
alternating *contraction* does that too, so the residual must also have
stopped improving.

**Growth is measured against a recent window,** not the best residual ever
seen, which would condemn any run that drifts away from a repelling start
before converging. "Begin at the non-interacting solution and find the
interacting one" is exactly that shape.

Limitations
-----------

* **Linear mixing only** — no Anderson acceleration, no Newton step. This is
  the standard first choice and is fine near a stable fixed point, but it
  converges slowly near a transition. Implement acceleration as a ``step``
  wrapper if you need it.
* Cycles of period 2 through 5 are detected; an exact cycle of period 6 or
  more is reported ``"max_iter"``, so that verdict does not strictly mean
  "more iterations would help".
* ``tol`` is an **absolute** tolerance on the state, so choose it against the
  scale of your state.

States
------

Anything ``step`` accepts and the distance function can compare: an array, a
scalar, or nested dicts / lists / tuples / namedtuples of those — which covers
the ``(R, C)`` pairs and ``{name: array}`` dicts a DMFT state usually is.
Container *and* leaf types are preserved across iterations, so ``step``
receives what it returned and ``result.state``'s type does not depend on
``damping``; a device array (JAX, torch) stays on its device.

:func:`~sft_wick.max_abs_distance` compares complex arrays by **modulus**, not
by real part, and promotes integer states before subtracting so the difference
cannot wrap.

.. automodule:: sft_wick.selfconsistency
   :members:
   :undoc-members:
   :show-inheritance:
