sft-wick: Wick Contractions for Statistical Field Theory
=========================================================

**sft-wick** automates the computation of perturbative expansions in the
path-integral formalism for stochastic differential equations.  Given an
observable and an interaction action, it applies Wick's theorem to express
arbitrary field moments in terms of two-point propagators --- the correlation
function :math:`C` and the response function :math:`R`.

.. math::

   \langle \mathcal{O} \rangle_S
   = \sum_{n=0}^{N} \frac{(-1)^n}{n!}\,
     \langle \mathcal{O}\, S_{\mathrm{int}}^{\,n} \rangle_{S_0}

Key Features
------------

- **No SymPy dependency** --- custom lightweight expression tree with exact
  rational arithmetic via ``fractions.Fraction``.
- **MSR-optimised contraction** --- exploits the constraint
  :math:`\langle\psi\,\psi\rangle = 0` to skip vanishing pairings entirely.
- **Feynman diagram generation** --- ``networkx``-based graph representation
  with ``matplotlib`` rendering (correlation :math:`C` as solid blue lines,
  response :math:`R` as dashed red arrows).
- **LaTeX output** --- every expression renders to publication-ready LaTeX,
  with configurable propagator names.
- **Immutable expressions** --- frozen dataclasses that are hashable and safe
  for use in sets and dictionaries.
- **Diagram-based simplification** --- terms with isomorphic Feynman diagrams
  (under dummy-variable relabeling and graph isomorphism) are automatically
  grouped, factoring out propagators and summing coupling coefficients with
  permuted indices.  Handles spatial-variable relabeling and C-propagator
  symmetry.

Quick Example
-------------

.. code-block:: python

   from sft_wick import Field, Vertex, Action, compute_moment

   # Define scalar fields
   phi = Field('phi', 'physical')
   psi = Field('psi', 'response')

   # Compute <psi(x) phi(x) phi(x) phi(x)>_{S_0}
   obs = [psi('x'), phi('x'), phi('x'), phi('x')]
   result = compute_moment(obs, Action(vertices=[]), order=0)
   print(result.order(0).to_latex())
   # Output: 3 R(x, x) C(x, x)

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   theory/index
   getting_started
   user_guide/index
   examples/index
   api/index
   contributing
   changelog

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
