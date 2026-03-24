Advanced Usage
==============

This page covers lower-level API access and more complex calculations.


Direct Wick Contraction
-----------------------

For manual contraction without the perturbative machinery, use
:func:`~sft_wick.wick.wick_contract`:

.. code-block:: python

   from sft_wick import Field, wick_contract, reset_uid_counter

   reset_uid_counter()
   phi = Field('phi', 'physical', n_components=3)

   ops = [phi('a', 'x'), phi('b', 'y'), phi('c', 'z'), phi('d', 'w')]
   expr, pairings = wick_contract(ops)

   print(f"Number of pairings: {len(pairings)}")
   print(expr.to_latex())

.. code-block:: text

   Number of pairings: 3
   C_{ab}(x, y) C_{cd}(z, w) + C_{ac}(x, z) C_{bd}(y, w) + C_{ad}(x, w) C_{bc}(y, z)


Single Pair Contraction
-----------------------

:func:`~sft_wick.propagators.contract_pair` contracts exactly two
operators:

.. code-block:: python

   from sft_wick import Field, contract_pair, reset_uid_counter

   reset_uid_counter()
   phi = Field('phi', 'physical', n_components=3)
   psi = Field('psi', 'response', n_components=3)

   # phi-phi -> C
   prop = contract_pair(phi('a', 'x'), phi('b', 'y'))
   print(prop.to_latex())       # C_{ab}(x, y)

   # phi-psi -> R
   prop = contract_pair(phi('a', 'x'), psi('b', 'y'))
   print(prop.to_latex())       # R_{ab}(x, y)

   # psi-psi -> None (vanishes)
   prop = contract_pair(psi('a', 'x'), psi('b', 'y'))
   print(prop)                  # None


Second-Order Perturbation Theory
---------------------------------

Higher orders work identically --- just increase the ``order`` argument:

.. code-block:: python

   from sft_wick import Field, Vertex, Action, compute_moment, reset_uid_counter

   reset_uid_counter()
   phi = Field('phi', 'physical')
   psi = Field('psi', 'response')

   v = Vertex(fields=[phi, psi], coupling='g')
   action = Action(vertices=[v])

   obs = [phi('x'), phi('y')]
   result = compute_moment(obs, action, order=2)

   for n in range(3):
       n_diagrams = len(result.diagrams_by_order.get(n, []))
       print(f"Order {n}: {n_diagrams} diagrams")
       if n_diagrams > 0:
           print(f"  {result.order(n).to_latex()}")

At second order, the multinomial expansion produces
:math:`S_{\mathrm{int}}^2 / 2!`, and the prefactor
:math:`(-1)^2 / 2! = 1/2` appears.


Diagram Topology Analysis
--------------------------

Access the full ``networkx.MultiGraph`` for custom analysis:

.. code-block:: python

   for d_info in result.diagrams_by_order[1]:
       fd = d_info.to_feynman_diagram()

       # Topology
       print(f"Loops: {fd.n_loops}")
       print(f"Connected: {fd.is_connected}")

       # Raw networkx access
       g = fd.graph
       print(f"Nodes: {g.number_of_nodes()}")
       print(f"Edges: {g.number_of_edges()}")

       # Node details
       for node, data in g.nodes(data=True):
           print(f"  {node}: {data}")

       # Edge details
       for u, v, key, data in g.edges(keys=True, data=True):
           print(f"  {u} -- {v}: {data['kind']}")


Itô Prescription
-----------------

By default, ``ito=True`` eliminates equal-point response propagators
and causal R-loops:

.. code-block:: python

   from sft_wick import Field, Vertex, Action, compute_moment, reset_uid_counter

   reset_uid_counter()
   phi = Field('phi', 'physical')
   psi = Field('psi', 'response')

   # <psi(x) phi(x)^3> — all at the same point
   obs = [psi('x'), phi('x'), phi('x'), phi('x')]

   # With Itô (default): R(x,x) = 0, entire result vanishes
   result = compute_moment(obs, Action(vertices=[]), order=0)
   print(result.order(0).to_latex())  # 0

   # Without Itô: 3 R(x,x) C(x,x)
   result_raw = compute_moment(
       obs, Action(vertices=[]), order=0, ito=False, response_phase=False
   )
   print(result_raw.order(0).to_latex())  # 3 R(x, x) C(x, x)

Causal R-loops (e.g. :math:`R(a,b)\,R(b,a) = 0`) are also detected
and eliminated at any cycle length:

.. code-block:: python

   # At first order, some pairings produce R(x,z)R(z,x)
   reset_uid_counter()
   v = Vertex(fields=[phi, psi], coupling='g')
   result_ito = compute_moment(
       [phi('x'), phi('y')], Action([v]), order=1, response_phase=False
   )
   result_no_ito = compute_moment(
       [phi('x'), phi('y')], Action([v]), order=1,
       ito=False, response_phase=False
   )
   print(f"Diagrams with Itô: {len(result_ito.diagrams_by_order[1])}")
   print(f"Diagrams without:  {len(result_no_ito.diagrams_by_order[1])}")


Response Phase Convention
--------------------------

The MSR convention :math:`\langle\phi\,\psi\rangle = -\mathrm{i}\,R`
is applied by default (``response_phase=True``).  Each term is
multiplied by :math:`(-\mathrm{i})^n` where *n* is the number of R
propagators:

.. code-block:: python

   from sft_wick import apply_response_phase

   reset_uid_counter()
   obs = [phi('x'), psi('y')]

   # With phase (default)
   result = compute_moment(obs, Action(vertices=[]), order=0, ito=False)
   print(result.order(0).to_latex())  # -1 i R(x, y)

   # Without phase
   result_raw = compute_moment(
       obs, Action(vertices=[]), order=0, ito=False, response_phase=False
   )
   print(result_raw.order(0).to_latex())  # R(x, y)

   # Manual application
   print(apply_response_phase(result_raw.order(0)).to_latex())


Diagram-Based Term Collection
-------------------------------

By default (``collect_topology=True``), terms whose Feynman diagrams
are isomorphic (under dummy-variable relabeling) are grouped.  Compare
the raw vs. collected output:

.. code-block:: python

   from sft_wick import Field, Vertex, Action, compute_moment, reset_uid_counter

   reset_uid_counter()
   phi = Field('phi', 'physical', n_components=3)
   psi = Field('psi', 'response', n_components=3)
   v = Vertex(fields=[phi, phi, psi], coupling='F')
   obs = [phi('a', 'x'), phi('b', 'y')]

   # Without diagram collection: many expanded terms at order 2
   reset_uid_counter()
   r_raw = compute_moment(obs, Action([v]), order=2,
                          response_phase=False, collect_topology=False)

   # With diagram collection (default): grouped terms
   reset_uid_counter()
   r_col = compute_moment(obs, Action([v]), order=2,
                          response_phase=False)

The collected form factors out propagators with canonical component
indices and produces sums of permuted coupling tensors (e.g.
:math:`F_{i_0 i_1 i_2} + F_{i_0 i_2 i_1}`).  At second order,
spatial-variable relabeling (:math:`y_0 \leftrightarrow y_1`) further
reduces the number of distinct terms.


Working with Expression Trees
------------------------------

Expressions can be inspected programmatically:

.. code-block:: python

   from sft_wick.expressions import Sum, Product, Propagator, Rational, ImaginaryUnit

   expr = result.order(1)

   if isinstance(expr, Sum):
       print(f"Sum of {len(expr.terms)} terms")
       for term in expr.terms:
           if isinstance(term, Product):
               for factor in term.factors:
                   if isinstance(factor, Propagator):
                       print(f"  Propagator: {factor.kind}")
                   elif isinstance(factor, Rational):
                       print(f"  Coefficient: {factor.numerator}/{factor.denominator}")
                   elif isinstance(factor, ImaginaryUnit):
                       print(f"  Imaginary unit: i")
