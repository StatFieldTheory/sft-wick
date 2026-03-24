Expression Simplification
=========================

The :func:`~sft_wick.simplify.simplify` function applies a multi-pass
pipeline to clean up the raw output of Wick contractions.  It is called
automatically inside :func:`~sft_wick.perturbation.compute_moment`, but
you can also call it directly on any expression.

.. code-block:: python

   from sft_wick import simplify
   simplified = simplify(raw_expr)


The Pipeline
------------

``simplify()`` applies four passes in sequence:

Pass 1 --- Flatten
^^^^^^^^^^^^^^^^^^

Recursively flattens nested :class:`~sft_wick.expressions.Sum` and
:class:`~sft_wick.expressions.Product` structures:

- ``Sum(Sum(a, b), c)`` becomes ``Sum(a, b, c)``
- ``Product(Product(a, b), c)`` becomes ``Product(a, b, c)``

Also descends into the bodies of
:class:`~sft_wick.expressions.IntegralOver` and
:class:`~sft_wick.expressions.SumOverIndex`.

Pass 2 --- Absorb Rationals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In any Product, multiplies all :class:`~sft_wick.expressions.Rational`
factors into a single coefficient:

- ``Product(Rational(1,2), Rational(3,1), C)`` becomes
  ``Product(Rational(3,2), C)``
- If the total coefficient is 0, the whole product becomes ``ZERO``.
- If the total coefficient is 1 and only one other factor remains,
  the product is unwrapped.

Pass 3 --- Eliminate Zeros
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Removes zero terms from sums
- Collapses any product containing a zero factor to ``ZERO``
- Removes ``Rational(1)`` factors from products
- Collapses wrappers whose body is zero

Pass 4 --- Collect Like Terms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Groups sum terms by their **structural signature** (the non-coefficient
part) and sums the rational coefficients within each group:

For example:

- ``2 C(x,y) R(x,z) + 3 C(x,y) R(x,z)`` becomes
  ``5 C(x,y) R(x,z)``

The signature includes propagator kinds, indices, spatial arguments,
symbol names, integral variables, and summation indices --- all in
canonical order.  Terms that cannot be matched are left unchanged.


Term Signatures
---------------

The internal function ``_term_signature()`` extracts a hashable
signature from each term for grouping.  It handles:

- Bare propagators
- Products of rationals, propagators, symbols, integrals, and
  summations

Products containing nested sums or other complex sub-expressions are
marked as "ungroupable" and passed through without merging.


Diagram-Based Collection
--------------------------

:func:`~sft_wick.simplify.collect_by_diagram` is a separate pass
(not part of ``simplify()``) that groups terms by **Feynman diagram
isomorphism**.  Two terms are equivalent if their propagator graphs
are isomorphic under relabeling of dummy integration variables and
summation indices.

The algorithm computes a **canonical graph form** for each term by
trying all permutations of internal spatial variables (integration
variables), using sorted C-edge endpoints to account for the symmetry
:math:`C(x, y) = C(y, x)`.  Terms with the same canonical form are
merged: the propagators are factored out with canonical component
indices, and the coupling coefficients are summed with indices (and
spatial arguments for non-local couplings) permuted to match.

This handles three cases that matter in practice:

- **Within-vertex index routing**: different Wick pairings that share
  the same spatial topology but differ in component-index assignments
  are collected, producing sums like
  :math:`(F_{i_0 i_1 i_2} + F_{i_0 i_2 i_1})`.
- **Cross-vertex spatial relabeling**: at order :math:`n \ge 2` with
  the same vertex type, swapping vertex-copy spatial variables
  (e.g. :math:`y_0 \leftrightarrow y_1`) is a valid relabeling that
  merges additional equivalent pairings.
- **C propagator symmetry**: pairings that differ only by the
  orientation of a correlation propagator
  (:math:`C(a, b)` vs :math:`C(b, a)`) are identified.

.. code-block:: python

   from sft_wick import collect_by_diagram
   collected = collect_by_diagram(simplified_expr)

   # Backward-compatible alias:
   from sft_wick import collect_by_topology  # same function


When Simplification Runs
-------------------------

:func:`~sft_wick.perturbation.compute_moment` applies the following
pipeline to each order expression:

1. **Wick contraction** --- when ``collect_topology=True`` (default),
   uses the spatial-level engine which enumerates spatial topologies
   with multiplicities, then groups by canonical diagram form.
   When ``collect_topology=False``, uses the operator-level engine
   followed by ``collect_by_diagram``.
2. **simplify** --- flatten, absorb rationals, eliminate zeros, collect
   like terms
3. **apply_response_phase** (``response_phase=True``) --- multiply by
   :math:`(-\mathrm{i})^n`

You do not normally need to call these yourself, but they are all
exposed in the public API for manual use.
