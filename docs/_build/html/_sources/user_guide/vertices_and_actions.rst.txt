Interaction Vertices and Actions
================================

Vertices define the nonlinear terms in the interaction action
:math:`S_{\mathrm{int}}`.  An :class:`~sft_wick.action.Action` collects
one or more vertices and handles the combinatorial expansion
:math:`S_{\mathrm{int}}^n` during perturbative calculations.


Defining a Vertex
-----------------

A :class:`~sft_wick.vertices.Vertex` is a frozen dataclass with three
attributes:

``fields``
   Sequence of :class:`~sft_wick.fields.Field` objects that participate
   in the vertex.

``coupling``
   Name of the coupling constant (e.g. ``'F'``, ``'g'``, ``'K'``).

``local``
   Whether all fields share the same spatial argument (default ``True``).


Local Vertices
^^^^^^^^^^^^^^

In a **local** vertex every field is evaluated at the same spatial point
and a single spatial integration wraps the term:

.. math::

   S_{\mathrm{int}}^{(\text{local})}
   = \int \mathrm{d}x\; F_{ijk}\,\phi_i(x)\,\phi_j(x)\,\psi_k(x)

.. code-block:: python

   from sft_wick import Field, Vertex

   phi = Field('phi', 'physical', n_components=3)
   psi = Field('psi', 'response', n_components=3)

   v = Vertex(fields=[phi, phi, psi], coupling='F')
   # v.local is True by default


Non-Local Vertices
^^^^^^^^^^^^^^^^^^

In a **non-local** vertex each field gets its own spatial argument and
multiple integrations appear:

.. math::

   S_{\mathrm{int}}^{(\text{non-local})}
   = \iint \mathrm{d}x\,\mathrm{d}x'\;
     K_{ij}(x,x')\,\psi_i(x)\,\psi_j(x')

.. code-block:: python

   v_nl = Vertex(fields=[psi, psi], coupling='K', local=False)


Building an Action
------------------

An :class:`~sft_wick.action.Action` is simply a list of vertex
templates:

.. code-block:: python

   from sft_wick import Action

   action = Action(vertices=[v, v_nl])

You can also pass an empty list when you only need zeroth-order
(free-propagator) results:

.. code-block:: python

   free_action = Action(vertices=[])


Multinomial Expansion
---------------------

When computing order *n*, the expansion
:math:`S_{\mathrm{int}}^n = (v_0 + v_1 + \cdots)^n` is generated
by :meth:`~sft_wick.action.Action.all_vertex_combinations`.  For each
combination of *n* vertices (with repetition), it yields the
corresponding **multinomial coefficient**:

.. math::

   \frac{n!}{n_0!\,n_1!\,\cdots}

where :math:`n_k` is the number of times vertex :math:`k` appears.


Vertex Instantiation
--------------------

During expansion, each vertex template is turned into a
:class:`~sft_wick.vertices.VertexInstance` with fresh, non-overlapping
internal indices.  An :class:`~sft_wick.indices.IndexContext` tracks
the counters to ensure no collisions between copies:

- **Component indices:** ``i_0``, ``i_1``, ``i_2``, ...
- **Spatial variables:** ``y_0``, ``y_1``, ``y_2``, ...

This is handled automatically by
:func:`~sft_wick.perturbation.compute_moment`; you only need to interact
with ``VertexInstance`` directly if you are using the low-level API.
