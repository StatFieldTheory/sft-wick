Wick's Theorem
==============

Wick's theorem is the combinatorial identity that allows expectation
values of products of fields under a Gaussian measure to be written as
sums over all **complete pairings** of two-point functions.


Statement
---------

For :math:`2n` field operators and a Gaussian (free) action :math:`S_0`:

.. math::

   \langle \Phi_1\,\Phi_2\,\cdots\,\Phi_{2n} \rangle_{S_0}
   = \sum_{\text{pairings}}
     \prod_{\text{pairs } (i,j)}
     \langle \Phi_i\,\Phi_j \rangle_{S_0}

The sum runs over all *complete pairings* --- every operator must be
paired with exactly one other.  If the total number of operators is
odd, the result vanishes.


Counting Pairings
-----------------

For :math:`2n` items, the number of complete pairings is the
**double factorial**:

.. math::

   (2n-1)!! = 1 \times 3 \times 5 \times \cdots \times (2n-1)

For example:

- 2 operators: 1 pairing
- 4 operators: 3 pairings
- 6 operators: 15 pairings
- 8 operators: 105 pairings

The function :func:`~sft_wick.wick.generate_all_pairings` enumerates
these.


The MSR Constraint
------------------

In the MSR formalism, the :math:`\psi`--:math:`\psi` contraction
vanishes:

.. math::

   \langle \psi_i(x)\,\psi_j(x') \rangle_{S_0} = 0

This has a powerful combinatorial consequence: **every response field**
:math:`\psi` **must be paired with a physical field** :math:`\phi`,
producing a response propagator :math:`R`.  The remaining
:math:`\phi`'s then pair among themselves, producing correlation
propagators :math:`C`.

This constraint dramatically reduces the number of non-vanishing
pairings.  If there are :math:`n_\phi` physical operators and
:math:`n_\psi` response operators:

- **Feasibility check:** :math:`n_\psi \le n_\phi` and
  :math:`(n_\phi - n_\psi)` must be even.
- **Valid pairings:** choose which :math:`n_\psi` of the :math:`\phi`'s
  pair with the :math:`\psi`'s, assign them in all :math:`n_\psi!`
  permutations, then pair the remaining :math:`\phi`'s giving
  :math:`(n_\phi - n_\psi - 1)!!` sub-pairings each.

The function :func:`~sft_wick.wick.generate_valid_pairings` implements
this optimised enumeration, skipping all vanishing :math:`\psi`--:math:`\psi`
pairings at construction time rather than generating and discarding them.


Worked Example
--------------

Consider the four-operator expectation value:

.. math::

   \langle \psi(x)\,\phi(x)\,\phi(x)\,\phi(x) \rangle_{S_0}

We have :math:`n_\psi = 1` and :math:`n_\phi = 3`, so the single
:math:`\psi` must pair with one of the three :math:`\phi`'s (producing
:math:`R`), and the remaining two :math:`\phi`'s pair together
(producing :math:`C`).  There are :math:`\binom{3}{1} = 3` ways to
choose which :math:`\phi` partners the :math:`\psi`, and
:math:`(2-1)!! = 1` way to pair the remaining two :math:`\phi`'s.
This gives **3 pairings**, all contributing :math:`R(x,x)\,C(x,x)`:

.. math::

   \langle \psi(x)\,\phi(x)\,\phi(x)\,\phi(x) \rangle_{S_0}
   = 3\,R(x,x)\,C(x,x)

In **sft-wick**:

.. code-block:: python

   from sft_wick import Field, Action, compute_moment

   phi = Field('phi', 'physical')
   psi = Field('psi', 'response')

   obs = [psi('x'), phi('x'), phi('x'), phi('x')]
   result = compute_moment(obs, Action(vertices=[]), order=0)
   print(result.order(0).to_latex())
   # 3 R(x, x) C(x, x)


Low-Level Access
----------------

For direct control over the Wick contraction machinery (without the
perturbative wrapper), use:

- :func:`~sft_wick.wick.wick_contract` --- applies Wick's theorem to
  any list of field operators and returns the symbolic sum plus the
  list of surviving pairings.
- :func:`~sft_wick.propagators.contract_pair` --- contracts a single
  pair of operators into a propagator (or ``None`` if it vanishes).
