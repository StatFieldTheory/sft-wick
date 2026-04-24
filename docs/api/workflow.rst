``sft_wick.workflow`` --- High-Level Workflow API (L1 + L2)
===========================================================

The ``workflow`` subpackage is the user-facing layer.  Users who need
to "declare physics → expand → integrate → inspect" should reach for
these types first; drop down to the raw API only if a concrete
requirement demands it.

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
