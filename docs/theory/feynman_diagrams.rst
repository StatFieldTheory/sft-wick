Feynman Diagram Rules
=====================

Every non-vanishing Wick contraction corresponds to a **Feynman
diagram** --- a graph that encodes the topology of the contraction.
**sft-wick** automatically constructs these diagrams as
:class:`~sft_wick.diagrams.FeynmanDiagram` objects backed by a
``networkx.MultiGraph``.


Elements of a Diagram
---------------------

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Symbol
     - Element
     - Description
   * - |bullet|
     - External point
     - An observable field operator (e.g. :math:`\phi_a(x)`).
       Rendered as a filled circle.
   * - |square|
     - Interaction vertex
     - A term from :math:`S_{\mathrm{int}}` (e.g. :math:`F_{ijk}
       \phi_i\phi_j\psi_k`).  Rendered as a filled square.
   * - |blueline|
     - *C* propagator
     - Correlation :math:`C_{ij}(x,x') = \langle\phi_i(x)\,
       \phi_j(x')\rangle_{S_0}`.  Drawn as a **solid blue** line.
   * - |redline|
     - *R* propagator
     - Response :math:`R_{ij}(x,x') = \langle\phi_i(x)\,
       \psi_j(x')\rangle_{S_0}`.  Drawn as a **dashed red** arrow
       (pointing from :math:`\psi` to :math:`\phi`).

.. |bullet| unicode:: U+25CF
.. |square| unicode:: U+25A0
.. |blueline| replace:: ---
.. |redline| unicode:: U+2192 .. right arrow


Constructing a Diagram from a Pairing
--------------------------------------

Given an observable :math:`\mathcal{O}` and an expansion of
:math:`S_{\mathrm{int}}^n`, the Wick contraction produces a set of
pairings.  Each pairing maps to a diagram as follows:

1. **Create nodes:** one external node per observable operator and
   one vertex node per interaction vertex instance.
2. **Create edges:** for each contracted pair :math:`(i,j)`, draw a
   propagator edge between the nodes that own operators *i* and *j*.

The class method :meth:`~sft_wick.diagrams.FeynmanDiagram.from_pairing`
performs this construction automatically.


Reading Off Expressions
-----------------------

To reconstruct the algebraic expression from a diagram:

- Each **vertex** :math:`v` contributes:

  - A coupling factor (e.g. :math:`F_{ijk}`)
  - Integration(s) over internal spatial variable(s)
    (:math:`\int \mathrm{d}y\,\ldots`)
  - Summation(s) over internal component indices
    (:math:`\sum_{i=1}^N \ldots`)

- Each **propagator edge** contributes a two-point function:
  :math:`C_{ij}(x,x')` or :math:`R_{ij}(x,x')`.

- The overall prefactor for an order-\ *n* diagram is
  :math:`(-1)^n / n!` times the multinomial coefficient from the
  vertex combination.


Topological Properties
----------------------

**Loop count:**

.. math::

   L = E - V + C

where :math:`E` is the number of edges (propagators), :math:`V` the
number of nodes (external + vertex), and :math:`C` the number of
connected components.  Accessed via the
:attr:`~sft_wick.diagrams.FeynmanDiagram.n_loops` property.

**Connectivity:**

A diagram is **connected** if every node can be reached from every
other node.  Disconnected diagrams factorise into independent sub-diagrams.
Checked via :attr:`~sft_wick.diagrams.FeynmanDiagram.is_connected`.

**Self-loops (tadpoles):**

When a propagator connects a node to itself (e.g. :math:`C(x,x)` at
a vertex), it forms a tadpole.  These are rendered as small loops in the
diagram.


Rendering
---------

Use :class:`~sft_wick.drawing.DiagramRenderer` (or the convenience
method :meth:`~sft_wick.perturbation.PerturbativeResult.draw_diagrams`)
to visualise diagrams:

.. code-block:: python

   result = compute_moment(obs, action, order=1)

   # Quick visualisation
   result.draw_diagrams(order=1)

   # Manual rendering with custom layout
   from sft_wick import DiagramRenderer
   renderer = DiagramRenderer(figsize=(8, 6))
   for d_info in result.diagrams_by_order[1]:
       fd = d_info.to_feynman_diagram()
       renderer.draw(fd, title=fd.summary())

See :doc:`/user_guide/diagrams` for the full usage guide.
