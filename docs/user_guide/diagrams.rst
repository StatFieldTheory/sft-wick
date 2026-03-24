Feynman Diagrams
================

**sft-wick** represents Feynman diagrams as ``networkx.MultiGraph``
objects and renders them with ``matplotlib``.


The ``FeynmanDiagram`` Class
----------------------------

:class:`~sft_wick.diagrams.FeynmanDiagram` wraps a ``networkx.MultiGraph``
with domain-specific convenience methods.

**Node types:**

- **External** (``node_type="external"``) --- observable field operators.
  Rendered as filled circles.
- **Vertex** (``node_type="vertex"``) --- interaction vertices from
  :math:`S_{\mathrm{int}}`.  Rendered as filled squares.

**Edge attributes (propagators):**

- ``kind`` --- ``"C"`` (correlation) or ``"R"`` (response)
- ``index_left``, ``index_right`` --- component indices
- ``spatial_left``, ``spatial_right`` --- spatial arguments


Building Diagrams
-----------------

Diagrams are typically constructed from a Wick pairing via the class
method :meth:`~sft_wick.diagrams.FeynmanDiagram.from_pairing`:

.. code-block:: python

   for d_info in result.diagrams_by_order[1]:
       fd = d_info.to_feynman_diagram()

This maps each operator UID to a graph node, creates all external and
vertex nodes, and adds propagator edges for each contracted pair.

You can also build diagrams manually:

.. code-block:: python

   from sft_wick import FeynmanDiagram

   fd = FeynmanDiagram()
   n1 = fd.add_external_point("phi(x)", "physical", spatial="x")
   n2 = fd.add_external_point("phi(y)", "physical", spatial="y")
   fd.add_propagator(n1, n2, kind="C", spatial_left="x", spatial_right="y")


Topological Properties
----------------------

.. code-block:: python

   fd.n_loops         # E - V + connected_components
   fd.is_connected    # True if the graph is connected
   fd.external_nodes  # list of external node IDs
   fd.vertex_nodes    # list of vertex node IDs
   fd.summary()       # one-line text description


Rendering with ``DiagramRenderer``
-----------------------------------

:class:`~sft_wick.drawing.DiagramRenderer` handles matplotlib drawing.

**Single diagram:**

.. code-block:: python

   from sft_wick import DiagramRenderer

   renderer = DiagramRenderer(figsize=(8, 6))
   renderer.draw(fd, title="My diagram")

**Multiple diagrams in a grid:**

.. code-block:: python

   fd_list = [d.to_feynman_diagram() for d in result.diagrams_by_order[1]]
   renderer.draw_all(fd_list, ncols=3, suptitle="Order-1 Diagrams")


Quick Visualisation
-------------------

The convenience method on
:class:`~sft_wick.perturbation.PerturbativeResult` handles everything
in one call:

.. code-block:: python

   result.draw_diagrams()            # all orders
   result.draw_diagrams(order=1)     # specific order


Visual Conventions
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Element
     - Rendering
   * - *C* propagator
     - Solid blue line
   * - *R* propagator
     - Dashed red arrow (from :math:`\psi` to :math:`\phi`)
   * - External point
     - Filled black circle with label above
   * - Vertex
     - Filled black square with coupling label below
   * - Self-loop (tadpole)
     - Small circle at the node


Layout Algorithm
----------------

The renderer uses a hybrid layout:

1. ``networkx.spring_layout`` as a starting point
2. External nodes are placed on a circle at radius 1.5
3. Vertex nodes are scaled inward to 50% of the spring-layout positions

This produces readable diagrams for typical MSR perturbation-theory
topologies.
