``sft_wick.spectral``
=====================

Disorder-averaged (spectral) propagators: ``R`` and ``C`` as superpositions of
Ornstein–Uhlenbeck propagators over a spectrum :math:`\rho(h)`.

Use this when the object of interest is the *ensemble-averaged* theory rather
than one instance of the disorder. The effective single-site problem is
scalar, so the :math:`O(N^{\text{rank}})` coupling-index contraction that
dominates a per-instance matrix calculation disappears.

.. warning::

   Above order 0 this is an **annealed** substitution, not a controlled
   quenched average — see the module docstring. Exact at order 0; measured
   35 % off above it.

.. automodule:: sft_wick.spectral
   :members:
   :undoc-members:
   :show-inheritance:
