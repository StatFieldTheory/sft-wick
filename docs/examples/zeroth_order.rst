Zeroth-Order Moments
====================

At zeroth order no interaction vertices contribute --- the result is
purely the Wick contraction of the observable under the free action
:math:`S_0`.


Scalar Two-Point Function
--------------------------

The simplest non-trivial example: :math:`\langle\phi(x)\,\phi(y)\rangle_{S_0}`.

.. code-block:: python

   from sft_wick import Field, Action, compute_moment, reset_uid_counter

   reset_uid_counter()
   phi = Field('phi', 'physical')

   obs = [phi('x'), phi('y')]
   result = compute_moment(obs, Action(vertices=[]), order=0)
   print(result.order(0).to_latex())

.. code-block:: text

   C(x, y)

Only one pairing exists, and it produces the correlation propagator.


Mixed-Field Four-Point Function
-------------------------------

:math:`\langle\psi(x)\,\phi(x)\,\phi(x)\,\phi(x)\rangle_{S_0}`:

.. code-block:: python

   reset_uid_counter()
   phi = Field('phi', 'physical')
   psi = Field('psi', 'response')

   obs = [psi('x'), phi('x'), phi('x'), phi('x')]
   result = compute_moment(obs, Action(vertices=[]), order=0)
   print(result.order(0).to_latex())

.. code-block:: text

   3 R(x, x) C(x, x)

**Why the factor of 3?**  The single :math:`\psi` must pair with one
of the three :math:`\phi`'s (3 choices, each giving :math:`R(x,x)`),
and the remaining two :math:`\phi`'s pair together
(:math:`(2-1)!! = 1` way, giving :math:`C(x,x)`).


Pure Physical Four-Point Function
---------------------------------

:math:`\langle\phi(x)\,\phi(y)\,\phi(z)\,\phi(w)\rangle_{S_0}`:

.. code-block:: python

   reset_uid_counter()
   phi = Field('phi', 'physical')

   obs = [phi('x'), phi('y'), phi('z'), phi('w')]
   result = compute_moment(obs, Action(vertices=[]), order=0)
   print(result.order(0).to_latex())

.. code-block:: text

   C(x, y) C(z, w) + C(x, z) C(y, w) + C(x, w) C(y, z)

The three pairings correspond to the three ways of pairing four items:
:math:`(2 \times 4 - 1)!! = 3`.


Vanishing Cases
---------------

An **odd** number of operators always gives zero:

.. code-block:: python

   obs = [phi('x'), phi('y'), phi('z')]
   result = compute_moment(obs, Action(vertices=[]), order=0)
   print(result.order(0).to_latex())
   # 0

More :math:`\psi`'s than :math:`\phi`'s also gives zero (not enough
physical fields to absorb all response fields):

.. code-block:: python

   obs = [psi('x'), psi('y'), phi('z')]
   result = compute_moment(obs, Action(vertices=[]), order=0)
   print(result.order(0).to_latex())
   # 0
