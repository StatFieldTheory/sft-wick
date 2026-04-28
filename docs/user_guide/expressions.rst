The Expression Tree
===================

**sft-wick** uses a custom lightweight expression tree instead of
relying on SymPy.  All expression types are **frozen dataclasses**
--- immutable and hashable --- with exact rational arithmetic via
``fractions.Fraction``.


Design Philosophy
-----------------

- **No external symbolic algebra dependency.**  The expression tree is
  self-contained in :mod:`sft_wick.expressions`.
- **Immutability.**  Once created, an expression cannot be mutated.
  This makes expressions safe to use as dictionary keys, in sets, and
  in concurrent code.
- **Exact arithmetic.**  Coefficients are represented as
  :class:`~sft_wick.expressions.Rational` numbers backed by
  ``fractions.Fraction``, avoiding floating-point drift.
- **LaTeX rendering.**  Every expression type implements ``to_latex()``.


The ``Expr`` Base Class
-----------------------

All expression types inherit from :class:`~sft_wick.expressions.Expr`,
which provides operator overloads:

- ``expr + expr`` --- creates a :class:`~sft_wick.expressions.Sum`
- ``expr * expr`` --- creates a :class:`~sft_wick.expressions.Product`
- ``-expr`` --- multiplies by :math:`-1`
- ``expr - expr`` --- addition of negation
- ``repr(expr)`` --- delegates to ``to_latex()``


Numeric: ``Rational``
---------------------

:class:`~sft_wick.expressions.Rational` stores exact rational numbers
with automatic normalisation (positive denominator, reduced form):

.. code-block:: python

   from sft_wick.expressions import Rational, ZERO, ONE

   r = Rational(3, 4)       # 3/4
   r.to_latex()              # '\\frac{3}{4}'
   r.is_zero                 # False
   r.is_one                  # False
   r.to_fraction()           # Fraction(3, 4)

   # Arithmetic stays exact
   Rational(1, 3) * Rational(2, 5)  # Rational(2, 15)

Constants: ``ZERO``, ``ONE``, ``MINUS_ONE``.


Symbolic: ``Symbol``
--------------------

:class:`~sft_wick.expressions.Symbol` represents named tensors and
coupling constants with optional indices and spatial arguments:

.. code-block:: python

   from sft_wick.expressions import Symbol

   Symbol('F', ('i', 'j', 'k'))               # F_{ijk}
   Symbol('K', ('i', 'j'), ('y_0', 'y_1'))    # K_{ij}(y_0, y_1)


Propagators
-----------

:class:`~sft_wick.expressions.Propagator` stores a two-point function:

.. code-block:: python

   from sft_wick.expressions import Propagator

   c = Propagator('C', 'a', 'b', 'x', 'y')  # C_{ab}(x, y)
   r = Propagator('R', 'a', 'b', 'x', 'y')  # R_{ab}(x, y)

- ``kind`` --- ``'C'`` (correlation) or ``'R'`` (response)
- For **R**, the physical field is always on the left by convention.
- For scalar fields, ``index_left`` and ``index_right`` are ``None``.


Composite: ``Sum`` and ``Product``
----------------------------------

:class:`~sft_wick.expressions.Sum` and
:class:`~sft_wick.expressions.Product` hold tuples of sub-expressions.
Nested structures are **automatically flattened** at construction time:

.. code-block:: python

   from sft_wick.expressions import Sum, Product

   # Sum(Sum(a, b), c) becomes Sum(a, b, c)
   # Product(Product(a, b), c) becomes Product(a, b, c)


Wrappers: ``SumOverIndex`` and ``IntegralOver``
------------------------------------------------

These wrap an inner expression with a summation or integration:

.. code-block:: python

   from sft_wick.expressions import SumOverIndex, IntegralOver

   # sum_{i=1}^{3} body
   SumOverIndex('i', 3, body)

   # integral dy body
   IntegralOver('y', body)

They are inserted automatically by
:func:`~sft_wick.perturbation.compute_moment` when vertices introduce
internal indices or spatial variables.


Delta Functions
---------------

:class:`~sft_wick.expressions.KroneckerDelta` and
:class:`~sft_wick.expressions.DiracDelta` represent component and
spatial delta functions respectively.  Their equality and hashing are
**order-independent**: :math:`\delta_{ij} = \delta_{ji}`.

``KroneckerDelta`` factors are inserted automatically by
:meth:`DiagramTerm.apply_diagonal
<sft_wick.perturbation.DiagramTerm.apply_diagonal>` whenever
``diag_R=True`` or ``diag_C=True`` constraints merge a pair of
**external** (non-summation) indices: the simplifier collapses
``C[a, b] -> delta_{a, b} C[a, a]`` so that pinning the
observable indices to different values via ``fixed_indices``
correctly returns 0.  ``_eval_symbolic`` evaluates a
``KroneckerDelta`` to ``1.0`` when both indices resolve to the
same integer (via ``index_map`` or as a literal numeric label),
and ``0.0`` otherwise.


Immutability and Hashability
----------------------------

Because every expression is a frozen dataclass, you can use them freely
as dictionary keys or in sets:

.. code-block:: python

   expr_set = {c, r, c}   # {C_{ab}(x,y), R_{ab}(x,y)}
   expr_dict = {c: 1.0}   # works


Imaginary Unit: ``ImaginaryUnit``
---------------------------------

:class:`~sft_wick.expressions.ImaginaryUnit` represents the imaginary
unit :math:`\mathrm{i}`, used in the response phase convention.  A
module-level constant ``I`` is provided:

.. code-block:: python

   from sft_wick import I, ImaginaryUnit

   I.to_latex()          # '\\mathrm{i}'
   I == ImaginaryUnit()  # True

The function :func:`~sft_wick.expressions.apply_response_phase`
multiplies each term by :math:`(-\mathrm{i})^n` where *n* counts
the response propagators, implementing
:math:`\langle\phi\,\psi\rangle = -\mathrm{i}\,R`.
