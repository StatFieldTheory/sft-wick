``sft_wick.drawing``: Diagram Rendering
==========================================

Import lifecycle
----------------

Importing ``sft_wick`` for numerical or symbolic work does not load
Matplotlib. The public ``sft_wick.DiagramRenderer`` class and
``sft_wick.drawing`` module load on their first ordinary attribute access
or explicit import, and return the same real objects as a direct submodule
import. Existing drawing methods continue to load rendering when called.

``dir(sft_wick)`` advertises both names before loading. Star imports and
``inspect.getmembers(sft_wick)`` request exports and therefore load plotting.
Before first access, direct module-dictionary inspection (``vars`` or
``__dict__``) does not contain the deferred exports. Plotting initialization
and import errors occur when rendering is requested; failed imports can be
retried. Choose the Matplotlib backend before that request if required.

This removes unnecessary plotting startup work from non-rendering calls.
It does not change numerical arithmetic or reduce the plotting work needed
by the first render. Startup savings depend on the platform and font cache.

.. automodule:: sft_wick.drawing
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
