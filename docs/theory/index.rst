Theoretical Background
======================

.. important::

   The full theoretical treatment is in the paper
   `arXiv:2606.19480 <https://arxiv.org/abs/2606.19480>`_
   (Z. Zhang, *sft-wick: A formalism and package for Feynman-diagram
   expansion and evaluation in stochastic field theories*).  It derives
   the MSR action for multi-component fields with non-Gaussian noise,
   the diagram rules and the topology-first enumeration that sft-wick
   implements, and validates the predictions against direct Langevin
   simulation.  The pages below are a summary for orientation; read the
   paper for the derivations, and cite it if you use the package
   (:doc:`/citing`).

This section presents the physics underlying **sft-wick**: the MSR
path-integral formalism for stochastic differential equations, Wick's
theorem for Gaussian integrals, and the Feynman diagram rules that
arise from the perturbative expansion.

Readers already comfortable with MSR perturbation theory may want to
skip ahead to the :doc:`/getting_started` tutorial.

.. toctree::
   :maxdepth: 2

   msr_formalism
   wick_theorem
   feynman_diagrams
