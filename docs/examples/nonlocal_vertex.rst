Non-Local Interactions
======================

Non-local vertices allow the coupling to depend on multiple spatial
arguments.  This is useful for modelling spatially extended interactions,
kernels in Fourier space, or non-Markovian dynamics.


A Non-Local Kernel
------------------

Consider
:math:`S_{\mathrm{int}} = \iint K_{ij}(x,x')\,\psi_i(x)\,\psi_j(x')\,\mathrm{d}x\,\mathrm{d}x'`:

.. code-block:: python

   from sft_wick import Field, Vertex, Action, compute_moment, reset_uid_counter

   reset_uid_counter()
   phi = Field('phi', 'physical', n_components=2)
   psi = Field('psi', 'response', n_components=2)

   v_nl = Vertex(fields=[psi, psi], coupling='K', local=False)
   action = Action(vertices=[v_nl])

   obs = [phi('a', 'x'), phi('b', 'y')]
   result = compute_moment(obs, action, order=1)
   print(result.order(1).to_latex())

Because the vertex is non-local:

- Each :math:`\psi` in the vertex gets its own spatial variable
  (:math:`y_0` and :math:`y_1`)
- The coupling symbol carries spatial arguments:
  :math:`K_{i_0 i_1}(y_0, y_1)`
- Two spatial integrations appear:
  :math:`\int\mathrm{d}y_0\int\mathrm{d}y_1`


Mixing Local and Non-Local Vertices
-------------------------------------

Actions can contain both local and non-local vertices:

.. code-block:: python

   reset_uid_counter()
   v_local = Vertex(fields=[phi, phi, psi], coupling='F')
   v_nonlocal = Vertex(fields=[psi, psi], coupling='K', local=False)

   action = Action(vertices=[v_local, v_nonlocal])

   obs = [phi('a', 'x'), phi('b', 'y')]
   result = compute_moment(obs, action, order=1)

At first order, each vertex contributes separately (one diagram set
from the local vertex, another from the non-local vertex).  At second
order and beyond, *cross terms* appear where different vertex types mix.


Comparing Local vs. Non-Local Output
--------------------------------------

The key differences in the symbolic output:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Feature
     - Difference
   * - Spatial variables
     - Local: all fields share one variable.
       Non-local: each field gets its own.
   * - Coupling symbol
     - Local: ``F_{ijk}`` (no spatial args).
       Non-local: ``K_{ij}(y_0, y_1)``.
   * - Integrations
     - Local: single :math:`\int\mathrm{d}y_0`.
       Non-local: one per spatial variable.
