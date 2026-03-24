Multi-Component Fields
======================

When fields carry component indices, propagators and couplings gain
index structure, and the output includes summations over internal
indices.


Four-Point Function with Component Indices
-------------------------------------------

:math:`\langle\phi_a(x)\,\phi_b(y)\,\phi_c(z)\,\phi_d(w)\rangle_{S_0}`:

.. code-block:: python

   from sft_wick import Field, Action, compute_moment, reset_uid_counter

   reset_uid_counter()
   phi = Field('phi', 'physical', n_components=3)

   obs = [phi('a', 'x'), phi('b', 'y'), phi('c', 'z'), phi('d', 'w')]
   result = compute_moment(obs, Action(vertices=[]), order=0)
   print(result.order(0).to_latex())

.. code-block:: text

   C_{ab}(x, y) C_{cd}(z, w) + C_{ac}(x, z) C_{bd}(y, w) + C_{ad}(x, w) C_{bc}(y, z)

The three pairings now carry component indices matching the external
operators.


Mixed Physical and Response Fields
------------------------------------

:math:`\langle\psi_a(x)\,\phi_b(x)\,\phi_c(x)\,\phi_d(x)\rangle_{S_0}`:

.. code-block:: python

   reset_uid_counter()
   phi = Field('phi', 'physical', n_components=3)
   psi = Field('psi', 'response', n_components=3)

   obs = [psi('a', 'x'), phi('b', 'x'), phi('c', 'x'), phi('d', 'x')]
   result = compute_moment(obs, Action(vertices=[]), order=0)
   print(result.order(0).to_latex())

Each :math:`R` propagator now has the convention that the physical-field
index is on the left: :math:`R_{ba}(x,x)` means
:math:`\langle\phi_b(x)\,\psi_a(x)\rangle_{S_0}`.


First-Order with Coupling Tensor
---------------------------------

With a vertex :math:`\int F_{ijk}\,\phi_i\,\phi_j\,\psi_k\,\mathrm{d}z`:

.. code-block:: python

   reset_uid_counter()
   v = Vertex(fields=[phi, phi, psi], coupling='F')
   action = Action(vertices=[v])

   obs = [phi('a', 'x'), phi('b', 'y')]
   result = compute_moment(obs, action, order=1)
   print(result.order(1).to_latex())

The output includes:

- Summation :math:`\sum_{i_0}\sum_{i_1}\sum_{i_2}` over internal
  component indices
- The coupling tensor :math:`F_{i_0 i_1 i_2}`
- Integration :math:`\int\mathrm{d}y_0` over the vertex position
- Products of :math:`C` and :math:`R` propagators with the appropriate
  index combinations
