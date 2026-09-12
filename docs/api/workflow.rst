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
