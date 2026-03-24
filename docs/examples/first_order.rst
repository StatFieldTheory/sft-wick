First-Order Perturbation Theory
================================

First-order calculations introduce a single power of
:math:`S_{\mathrm{int}}`, adding interaction vertices to the
contraction.


Scalar Cubic Vertex
-------------------

Consider :math:`S_{\mathrm{int}} = \int g\,\phi(z)\,\psi(z)\,\mathrm{d}z`
and the two-point function :math:`\langle\phi(x)\,\phi(y)\rangle_S`:

.. code-block:: python

   from sft_wick import Field, Vertex, Action, compute_moment, reset_uid_counter

   reset_uid_counter()
   phi = Field('phi', 'physical')
   psi = Field('psi', 'response')

   v = Vertex(fields=[phi, psi], coupling='g')
   action = Action(vertices=[v])

   obs = [phi('x'), phi('y')]
   result = compute_moment(obs, action, order=1)

   print("Order 0:")
   print(result.order(0).to_latex())
   print()
   print("Order 1:")
   print(result.order(1).to_latex())

.. code-block:: text

   Order 0:
   C(x, y)

   Order 1:
   <first-order correction involving R and C propagators with integration>

The first-order correction involves an integration over the internal
spatial variable :math:`y_0` and products of R and C propagators.


Multi-Component Cubic Vertex
-----------------------------

For :math:`S_{\mathrm{int}} = \int F_{ijk}\,\phi_i(z)\,\phi_j(z)\,\psi_k(z)\,\mathrm{d}z`:

.. code-block:: python

   reset_uid_counter()
   phi = Field('phi', 'physical', n_components=3)
   psi = Field('psi', 'response', n_components=3)

   v = Vertex(fields=[phi, phi, psi], coupling='F')
   action = Action(vertices=[v])

   obs = [psi('a', 'x'), phi('b', 'x'), phi('c', 'x'), phi('d', 'x')]
   result = compute_moment(obs, action, order=1)

   print("Order 0:")
   print(result.order(0).to_latex())
   print()
   print("Order 1:")
   print(result.order(1).to_latex())

The first-order terms now include summations over internal component
indices :math:`i_0, i_1, i_2` and the coupling tensor :math:`F_{i_0 i_1 i_2}`.


Inspecting Diagrams
-------------------

Each non-vanishing contraction at first order produces a diagram:

.. code-block:: python

   for d_info in result.diagrams_by_order[1]:
       fd = d_info.to_feynman_diagram()
       print(fd.summary())
       print(f"  Loops: {fd.n_loops}, Connected: {fd.is_connected}")

   # Visualise
   result.draw_diagrams(order=1)

First-order diagrams have one interaction vertex (square) connected to
the external points (circles) by propagator lines.
