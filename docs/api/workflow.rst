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

The dense backend validates per-component coefficient phase before contraction.
It admits only its documented native static dense subset, retains ordered
fallbacks, and can change accumulation rounding. It provides no runtime global
rounding certificate. Declare observable-specific tolerances in the explicit
resource comparison and require complete-result acceptance before interpreting
its time/memory ratios. See :doc:`../user_guide/resources` for eligibility,
lifetimes, memory-accounting boundaries and executable YAML examples.

``factorized`` preparation owns original factor snapshots for monomials or
bounded native sums of monomials (at most 32 terms in a top-level sum). Each
term must independently pass phase and finite-value admission; fusion cannot
be admitted by cross-term imaginary cancellation. Multi-term sums also require
uniform nonzero signs in each monomial's selected array components and matching
term signs after scalar phases, excluding cross-monomial real cancellation.
Mixed or opposing signs retain the ordered path; this additional sign constraint
does not apply to single monomials. Native kernel batches still require finite
values and admitted scales. Complete-result rounding validation remains required
after these coefficient checks. Existing compiled coefficient plans retain their original path;
use fresh workflow preparation for cost comparison.

Reading the public ``coupling_array`` restores the original ndarray using the
existing ordered product and builtin sum evaluation, then disables fusion; that
read and scalar or unsupported batch fallback can exceed the configured budget
and raise. Rebuild with ``ordered`` or increase the budget when materialization
is intended. Unknown dynamic/custom shapes are outside this check. Native disk
propagator identities use full content; explicit or unversioned callbacks bypass
shared persisted tables. See :doc:`../user_guide/resources` for these contracts.
