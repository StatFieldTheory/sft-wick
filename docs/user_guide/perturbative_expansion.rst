Computing Perturbative Expansions
=================================

The function :func:`~sft_wick.perturbation.compute_moment` is the main
entry point for perturbative calculations.


Basic Usage
-----------

.. code-block:: python

   from sft_wick import Field, Vertex, Action, compute_moment

   phi = Field('phi', 'physical', n_components=3)
   psi = Field('psi', 'response', n_components=3)

   v = Vertex(fields=[phi, phi, psi], coupling='F')
   action = Action(vertices=[v])

   obs = [psi('a', 'x'), phi('b', 'x'), phi('c', 'x'), phi('d', 'x')]
   result = compute_moment(obs, action, order=2)

This computes:

.. math::

   \langle \mathcal{O} \rangle_S
   = \sum_{n=0}^{2} \frac{(-1)^n}{n!}\,
     \langle \mathcal{O}\, S_{\mathrm{int}}^{\,n} \rangle_{S_0}


The ``PerturbativeResult`` Object
----------------------------------

:func:`~sft_wick.perturbation.compute_moment` returns a
:class:`~sft_wick.perturbation.PerturbativeResult` with three main
attributes:

``order_terms``
   A ``dict[int, Expr]`` mapping perturbative order to the simplified
   expression at that order.

``total``
   A single :class:`~sft_wick.expressions.Expr` summing all non-zero
   order contributions.

``diagrams_by_order``
   A ``dict[int, list[DiagramInfo]]`` storing the Feynman diagram
   records for each order.


Accessing Individual Orders
---------------------------

.. code-block:: python

   # Single order
   expr_0 = result.order(0)
   expr_1 = result.order(1)

   # LaTeX output
   print(expr_0.to_latex())
   print(expr_1.to_latex())

   # Full result (all orders)
   print(result.to_latex())


What Happens Internally
-----------------------

For each order *n* from 0 to the requested maximum:

1. **Multinomial expansion:** ``Action.all_vertex_combinations(n)``
   generates all ways to pick *n* vertices with the associated
   multinomial coefficient.

2. **Vertex instantiation:** each selected vertex is converted into a
   :class:`~sft_wick.vertices.VertexInstance` with fresh internal
   indices via :class:`~sft_wick.indices.IndexContext`.

3. **Wick contraction:** the observable operators and vertex operators
   are concatenated and contracted.  When ``collect_topology=True``
   (the default), the **spatial-level engine**
   :func:`~sft_wick.wick.wick_contract_spatial` is used: it enumerates
   *spatial topologies* (R-edge and C-edge assignments between spatial
   points) rather than individual operator-level pairings, and computes
   a **multiplicity** for each topology.  This avoids the combinatorial
   explosion from component-index routing at higher orders.  When
   ``collect_topology=False``, the operator-level engine
   :func:`~sft_wick.wick.wick_contract` is used instead, enumerating
   all non-vanishing pairings explicitly.

4. **Expression assembly:** each non-zero contraction (or spatial
   topology with its multiplicity) is multiplied by the coupling
   symbols, wrapped in spatial integrals and component summations, and
   scaled by the prefactor
   :math:`(-1)^n / n! \times \text{multinomial coeff}`.

5. **Diagram-based grouping** (``collect_topology=True``): terms whose
   Feynman diagrams are isomorphic (under dummy-variable relabeling
   and graph isomorphism) are grouped, factoring out propagators with
   canonical indices and summing coupling coefficients.

6. **Simplification:** the grouped expression is passed through
   :func:`~sft_wick.simplify.simplify` (flatten, absorb rationals,
   eliminate zeros, collect like terms).

7. **Response phase** (``response_phase=True``): each term is
   multiplied by :math:`(-\mathrm{i})^n`.


Sign and Factorial
------------------

The prefactor for order *n* with multinomial coefficient *M* is:

.. math::

   \frac{(-1)^n}{n!} \times M

For a single vertex type (:math:`M = 1`), the familiar alternating-sign
factorial series is recovered.


Itô Prescription and Causality
-------------------------------

When ``ito=True`` (the default), two physics-motivated rules eliminate
vanishing contributions at contraction time:

**Equal-point R vanishes:**
:math:`R(x,x) = 0` --- the Itô discretisation convention
:math:`\Theta(0) = 0`.  This eliminates self-response contractions and
intra-vertex tadpoles in local vertices.

**Causal R-loops vanish:**
Any closed loop of response propagators
:math:`R(a,b)\,R(b,c)\cdots R(z,a) = 0`,
since the retarded propagator :math:`R \propto \Theta(t - t')` would
require a cyclic time ordering :math:`t_a > t_b > \cdots > t_a`, which
is impossible.

Pass ``ito=False`` to keep all R terms symbolic.  The numerical layer
applies :math:`\Theta(0)=0` at every R evaluation, so it computes the
Itô SDE.  Whether that is also the value ``ito=False`` asks for depends
on where the equal-point R sits, and the numerical layer applies this
rule:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Equal-point ``R(y, y)``
     - What happens
   * - on a local vertex with **one** ψ leg (a drift term
       :math:`\psi_a f_a(\varphi)`)
     - Evaluated (it gives 0).  The term is
       :math:`\Theta(0)\,\partial_a f_a`, which the Jacobian of the
       Stratonovich path integral
       :math:`-\tfrac12\int\mathrm{d}s\,\partial f/\partial\varphi`
       cancels; sft-wick emits neither, which is exact, so the number
       equals the ``ito=True`` one.
   * - on a local vertex with **two or more** ψ legs (a
       :math:`\varphi`-dependent noise covariance: multiplicative noise)
     - ``ValueError``.  The term is the drift
       :math:`\Theta(0)\,\partial_b D_{ab}(\varphi)`, which no Jacobian
       cancels; the Itô and Stratonovich moments differ, and the
       Stratonovich value needs the amplitude :math:`g` of
       :math:`D = g g^{\mathsf T}`, which the diagram does not carry.
   * - between two **external** operators at one label
     - ``ValueError``.  It is the equal-time response
       :math:`\Theta(0)` itself: 0 under Itô, 1/2 under Stratonovich.

The refusals come from
:meth:`~sft_wick.perturbation.DiagramTerm.build_integrand` and
:func:`~sft_wick.evaluate.integrate_diagrams`, so every integrator
raises the same message rather than returning the Itô value under
another name.

A Stratonovich SDE is therefore computed in its Itô form, with the
noise-induced drift
:math:`\tfrac12\sum_{jk} g_{jk}\,\partial_j g_{ik}` written into the
drift, and expanded with ``ito=True``.  At L1
:class:`~sft_wick.workflow.MultiplicativeImpulse` with
``interpretation='stratonovich'`` does that conversion; see
:doc:`workflow`.

Do **not** set :math:`\Theta(0)=1/2` without the Jacobian: for the
linear vertex that adds a spurious :math:`-k\,C(T,T)\,T/2`, 200 % /
400 % / 800 % of the exact answer at :math:`T = 4/8/16`
(``test_F15_ito_false_changes_the_expression_not_the_number``).


Response Phase Convention
-------------------------

When ``response_phase=True`` (the default), each term in the result is
multiplied by :math:`(-\mathrm{i})^n` where *n* is the number of
response propagators in that term.  This implements the MSR convention:

.. math::

   \langle \phi(a)\,\psi(b) \rangle = -\mathrm{i}\,R(a,b)

The phase is applied **after** simplification so that like-term
collection is unaffected.  Pass ``response_phase=False`` to get raw
*R* propagators.

The phase can also be applied manually to any expression:

.. code-block:: python

   from sft_wick import apply_response_phase
   phased = apply_response_phase(raw_expr)


Diagram-Based Term Collection
-------------------------------

When ``collect_topology=True`` (the default), the spatial-level
contraction engine first enumerates spatial topologies with
multiplicities, dramatically reducing the number of terms.  These
terms are then grouped by Feynman diagram isomorphism: two diagrams
are considered isomorphic when there exists a relabeling of dummy
integration variables (and, for C propagators, a spatial-argument
swap exploiting :math:`C(x,y) = C(y,x)`) that maps one propagator
set onto the other.

The algorithm computes a **canonical graph form** for each term by
trying all permutations of internal spatial variables.  For *n*
integration variables this costs :math:`O(n!)` --- fast for the
practical range :math:`n \le 4`.

Propagators are factored out with canonical component indices, and the
coupling coefficients are summed with indices appropriately permuted.
For example, if two Wick pairings produce:

.. math::

   F_{i_0 i_1 i_2}\,R_{a\,i_2}(x,y_0)\,C_{i_0 i_1}(y_0,y_0)
   + F_{i_0 i_2 i_1}\,R_{a\,i_1}(x,y_0)\,C_{i_0 i_2}(y_0,y_0)

the second term's internal indices are relabelled to match the first,
yielding:

.. math::

   \bigl(F_{i_0 i_1 i_2} + F_{i_0 i_2 i_1}\bigr)\,
   R_{a\,i_2}(x,y_0)\,C_{i_0 i_1}(y_0,y_0)

At second order and above, spatial-variable relabeling (e.g. swapping
:math:`y_0 \leftrightarrow y_1` for two copies of the same vertex)
merges additional equivalent pairings, further reducing the number of
terms.

Pass ``collect_topology=False`` to keep all pairings expanded
individually.  The function can also be called directly:

.. code-block:: python

   from sft_wick import collect_by_diagram
   collected = collect_by_diagram(raw_expr)

   # Backward-compatible alias:
   from sft_wick import collect_by_topology  # same function


Zeroth-Order Calculations
-------------------------

At order 0, no vertices are involved --- the result is purely the Wick
contraction of the observable under the free action:

.. code-block:: python

   result = compute_moment(obs, Action(vertices=[]), order=0)
   # Only result.order(0) is non-trivial
