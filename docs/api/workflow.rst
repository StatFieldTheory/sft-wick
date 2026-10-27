``sft_wick.workflow``: High-Level Workflow API (L1 + L2)
===========================================================

The ``workflow`` subpackage is the user-facing layer.  To "declare
physics → expand → integrate → inspect", start with these types.  Drop
down to the raw API only when a concrete requirement demands it.

Top-level system object
-----------------------

.. automodule:: sft_wick.workflow.system
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Specification objects
---------------------

.. automodule:: sft_wick.workflow.specs
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Expansion
---------

.. automodule:: sft_wick.workflow.expansion
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Propagators
-----------

.. automodule:: sft_wick.workflow.propagators
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

R-contracted non-local coupling
-------------------------------

.. automodule:: sft_wick.workflow.r_contracted
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Propagator disk cache
---------------------

.. automodule:: sft_wick.workflow.cache
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Result & SweepResult
--------------------

.. automodule:: sft_wick.workflow.result
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

YAML + CLI (L2)
---------------

.. automodule:: sft_wick.workflow.resources
   :members:

.. automodule:: sft_wick.workflow.resource_compare
   :members:

.. automodule:: sft_wick.workflow.config
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. automodule:: sft_wick.workflow.cli
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Explicit numerical resource options
-----------------------------------

``Expansion.evaluate`` and ``Expansion.sweep`` expose these keyword options;
the YAML layer uses the same names under ``sweep``. The raw integration entry
points forward the same options to the integrand. Defaults retain the previous
evaluation path.

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Option
     - Default
     - Contract
   * - ``propagator_cache_bytes``
     - ``None``
     - Nonnegative integer or null; zero/null disable. Positive budgets require
       ``propagator_cache_contract="fixed_model"``.
   * - ``propagator_cache_contract``
     - ``None``
     - Explicit fixed-model caller promise; callbacks/data are stable and
       lookup results depend only on arguments during the cache lifetime.
   * - ``contraction_backend``
     - ``"ordered"``
     - ``"ordered"``, ``"einsum"`` or ``"factorized"``; native static fusion is opt-in.
   * - ``contraction_max_bytes``
     - ``67108864``
     - Positive integer controlling array-payload admission. ``factorized`` also
       checks known native coefficient restoration and can raise before allocation.

The propagator cache owns its stored native arrays and returns independent
arrays on hits. Its byte cap covers retained key/array payloads, with Python
metadata, temporary copies and all other model/integration allocations outside
the cap. One serial sweep or evaluate call owns a cache lifetime. Workers use
independent task/batch-local caches and serialization clears active entries;
contents are not shared across invocations. Declaring ``fixed_model`` does not verify callback purity
or make mutable model data safe to change during evaluation.

``einsum`` and ``factorized`` evaluate each static diagram's sum over
component assignments as pairwise contractions. A result lies within the
error bound ``1.01 (m + A T) u S`` of the exact sum of the same float64
inputs, which the ordered fold also satisfies; ``S`` is built from the summed
coefficient the ordered fold uses. A diagram call runs on the ordered fold
when an input it reads, or an entry the ordered fold multiplies and the
backend leaves out (by a zero of a diagonal C table, or before an R support
excludes the assignment), is not finite, a product could overflow or underflow,
a coefficient fails the ordered phase check, NumPy's underflow policy is not
``"ignore"``, the workspace exceeds ``contraction_max_bytes``, or the cost
rule estimates the backend to be slower. ``factorized`` prepares a monomial,
or a sum of monomials that have no opposite signs at any component
assignment, from owned snapshots of the original coupling tensors; other
sums use the ordered preparation. Errors, inf and NaN behave as on the
ordered fold. Replacing package methods at run time is unsupported. See
:doc:`../user_guide/resources` for the bound, the admission and cost rules,
the measured speeds and executable YAML examples.

Reading the public ``coupling_array`` restores the original ndarray using the
existing ordered product and builtin sum evaluation, then disables fusion; that
read and scalar or unsupported batch fallback can exceed the configured budget
and raise. Rebuild with ``ordered`` or increase the budget when materialization
is intended. Unknown dynamic/custom shapes are outside this check. Native disk
propagator identities use full content; explicit or unversioned callbacks bypass
shared persisted tables. See :doc:`../user_guide/resources` for these contracts.
