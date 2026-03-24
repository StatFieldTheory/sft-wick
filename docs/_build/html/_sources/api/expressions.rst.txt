``sft_wick.expressions`` --- Symbolic Expression Tree
=====================================================

All expression types are frozen dataclasses --- immutable and hashable.
Uses exact rational arithmetic via ``fractions.Fraction``.

.. autosummary::
   :nosignatures:

   ~sft_wick.expressions.Expr
   ~sft_wick.expressions.Rational
   ~sft_wick.expressions.Symbol
   ~sft_wick.expressions.Propagator
   ~sft_wick.expressions.Sum
   ~sft_wick.expressions.Product
   ~sft_wick.expressions.SumOverIndex
   ~sft_wick.expressions.IntegralOver
   ~sft_wick.expressions.KroneckerDelta
   ~sft_wick.expressions.DiracDelta
   ~sft_wick.expressions.ImaginaryUnit
   ~sft_wick.expressions.apply_response_phase

.. automodule:: sft_wick.expressions
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
