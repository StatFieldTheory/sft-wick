Getting Started
===============

This tutorial walks through a complete perturbative calculation from
scratch: how to define fields, build an interaction action, compute
moments, inspect Feynman diagrams, and export results to LaTeX.


Step 1: Install and Import
-----------------------------

.. code-block:: bash

   pip install -e ".[dev]"

.. code-block:: python

   from sft_wick import (
       Field, Vertex, Action, compute_moment, LaTeXFormatter
   )


Step 2: Define Fields
------------------------

Every calculation starts by declaring the field species.  In the MSR
formalism there are two types: **physical** (:math:`\phi`) and
**response** (:math:`\psi`).

.. code-block:: python

   # Scalar fields (single component)
   phi = Field('phi', 'physical')
   psi = Field('psi', 'response')

For multi-component fields, specify ``n_components``:

.. code-block:: python

   phi = Field('phi', 'physical', n_components=3)
   psi = Field('psi', 'response', n_components=3)


Step 3: Create an Observable
-------------------------------

An **observable** is a list of field operators: concrete field
instances with bound component indices and spatial arguments.

.. code-block:: python

   # Scalar fields: phi('x') creates phi(x)
   phi = Field('phi', 'physical')
   psi = Field('psi', 'response')
   obs = [psi('y'), phi('x'), phi('x'), phi('x')]

For multi-component fields, pass ``(component_index, spatial_arg)``:

.. code-block:: python

   phi3 = Field('phi', 'physical', n_components=3)
   psi3 = Field('psi', 'response', n_components=3)
   obs3 = [psi3('a', 'w'), phi3('b', 'x'), phi3('c', 'y'), phi3('d', 'z')]


Step 4: Compute the Zeroth-Order Moment
------------------------------------------

At zeroth order, only the free propagators contribute (no interactions):

.. code-block:: python

   result = compute_moment(obs, Action(vertices=[]), order=0)
   print(result.order(0).to_latex())

.. code-block:: text

   -3 \mathrm{i} R(x, y) C(x, x)

The factor of 3 arises because :math:`\psi` can pair with any of the
three :math:`\phi`'s (giving :math:`R`), and the remaining two pair
together (giving :math:`C`).  The factor :math:`-\mathrm{i}` is the response
phase of Step 9.  With ``psi('x')`` the result is 0, because
:math:`R(x,x) = 0`.  See :doc:`theory/wick_theorem` for the combinatorial
explanation.


Step 5: Define an Interaction Vertex
---------------------------------------

Vertices represent terms in the interaction action
:math:`S_{\mathrm{int}}`.  A **local vertex** puts all fields at the
same spatial point:

.. code-block:: python

   # S_int = integral F_{ijk} phi_i(z) phi_j(z) psi_k(z) dz
   phi = Field('phi', 'physical', n_components=3)
   psi = Field('psi', 'response', n_components=3)
   v = Vertex(fields=[phi, phi, psi], coupling='F')

A **non-local vertex** gives each field its own spatial argument:

.. code-block:: python

   # S_int = iint K_{ij}(z, z') psi_i(z) psi_j(z') dz dz'
   v_nl = Vertex(fields=[psi, psi], coupling='K', local=False)


Step 6: Compute to First Order
----------------------------------

Wrap vertices in an :class:`~sft_wick.action.Action` and request a
higher-order expansion:

.. code-block:: python

   action = Action(vertices=[v])
   obs = [phi('a', 'x'), phi('b', 'y'), phi('c', 'z')]

   result = compute_moment(obs, action, order=1)

   # Zeroth order
   print("Order 0:", result.order(0).to_latex())

   # First order
   print("Order 1:", result.order(1).to_latex())

   # All orders combined
   print("Total:", result.total.to_latex())

The operators need distinct spatial labels at interacting orders; two
externals at one point take distinct labels and the same position.  A moment
is nonzero only when the fields pair up, so this three-point function starts
at order 1, and a four-point function has no order-1 term.


Step 7: Inspect Feynman Diagrams
------------------------------------

Every non-vanishing contraction is stored as a
:class:`~sft_wick.perturbation.DiagramInfo`.  Convert to a graph and
inspect topology:

.. code-block:: python

   for d_info in result.diagrams_by_order[1]:
       fd = d_info.to_feynman_diagram()
       print(fd.summary())
       print(f"  Loops: {fd.n_loops}, Connected: {fd.is_connected}")

Render diagrams with ``matplotlib``:

.. code-block:: python

   # Draw all diagrams at order 1
   result.draw_diagrams(order=1)

   # Draw everything
   result.draw_diagrams()


Step 8: Format as LaTeX
--------------------------

Every expression has a ``.to_latex()`` method:

.. code-block:: python

   print(result.order(0).to_latex())

For custom propagator names (e.g. *G* instead of *C*):

.. code-block:: python

   fmt = LaTeXFormatter(propagator_names={
       'C': 'G',
       'R': r'R^{\mathrm{ret}}'
   })
   print(fmt.format(result.order(0)))

Generate a complete ``align`` environment for a paper:

.. code-block:: python

   print(fmt.format_aligned(result.order_terms))


Step 9: Conventions: Itô and Response Phase
-----------------------------------------------

Two physics conventions apply:

**Itô prescription**: the response propagator vanishes at equal points,
:math:`R(x,x) = 0`, and causal R-loops vanish.  The engine has no other
prescription.  A Stratonovich SDE with multiplicative noise is computed in its
Itô form through :class:`~sft_wick.workflow.specs.MultiplicativeImpulse`; see
:doc:`user_guide/perturbative_expansion`.

**Response phase** (``response_phase=True``): each term is multiplied by
:math:`(-\mathrm{i})^n` where *n* counts response propagators,
implementing :math:`\langle\phi\,\psi\rangle = -\mathrm{i}\,R`.  The flag
changes the symbolic output only; integrated numbers always carry the phase.

.. code-block:: python

   # Disable to get raw R propagators:
   result = compute_moment(obs, action, order=1, response_phase=False)

   # Or apply the phase manually:
   from sft_wick import apply_response_phase
   phased = apply_response_phase(raw_expr)

**Diagram-based collection** (``collect_topology=True``): terms whose
Feynman diagrams are isomorphic (under dummy-variable relabeling and
C-propagator symmetry) are grouped.  Propagators are factored out with
canonical component indices, and coupling coefficients are summed with
permuted indices.  At second order and above, spatial-variable
relabeling further reduces the number of distinct terms.

.. code-block:: python

   # Contract at the operator level; no DiagramTerm is built:
   result = compute_moment(obs, action, order=1, collect_topology=False)


Next Steps
----------

- :doc:`user_guide/index`: detailed coverage of every feature
- :doc:`examples/index`: worked examples with multi-component fields,
  non-local vertices, and higher orders
- :doc:`api/index`: complete API reference
