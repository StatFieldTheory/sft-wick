"""LaTeX output formatting for Wick contraction results."""

from __future__ import annotations

from dataclasses import replace

from .expressions import (
    Expr,
    IntegralOver,
    Product,
    Propagator,
    Sum,
    SumOverIndex,
)


class LaTeXFormatter:
    """Configurable LaTeX output for expressions.

    Allows overriding propagator display names, e.g. rendering *C* as *G*
    or *R* as :math:`R^{\\mathrm{ret}}`.

    Args:
        propagator_names: Mapping from internal kind (``"C"``, ``"R"``)
            to the desired LaTeX display name.  Defaults to identity.
    """

    def __init__(
        self,
        propagator_names: dict[str, str] | None = None,
    ) -> None:
        self.propagator_names = propagator_names or {"C": "C", "R": "R"}

    def format(self, expr: Expr) -> str:
        """Format an expression as LaTeX, applying name substitutions.

        Args:
            expr: The expression to format.

        Returns:
            A LaTeX string with propagator names replaced.

        Raises:
            TypeError: if ``expr`` is not an :class:`Expr`, such as a
                ``DiagramTerm``, whose ``to_latex`` would ignore the names.
        """
        if not isinstance(expr, Expr):
            raise TypeError(
                f"LaTeXFormatter.format takes an Expr, not "
                f"{type(expr).__name__}; call its to_latex() for the "
                f"default propagator names."
            )
        return self._with_display_names(expr).to_latex()

    def _with_display_names(self, expr: Expr) -> Expr:
        """``expr`` with each propagator kind replaced by its display name."""
        renamed = self._with_display_names
        if isinstance(expr, Propagator):
            kind = self.propagator_names.get(expr.kind, expr.kind)
            return replace(expr, kind=kind)
        if isinstance(expr, Sum):
            return Sum(tuple(renamed(t) for t in expr.terms))
        if isinstance(expr, Product):
            return Product(tuple(renamed(f) for f in expr.factors))
        if isinstance(expr, SumOverIndex):
            return replace(expr, body=renamed(expr.body))
        if isinstance(expr, IntegralOver):
            return replace(expr, body=renamed(expr.body))
        return expr

    def format_equation(self, lhs: str, expr: Expr) -> str:
        """Format as a LaTeX equation: ``lhs = rhs``.

        Args:
            lhs: Left-hand side string (e.g. ``r"\\langle O \\rangle"``).
            expr: Right-hand side expression.

        Returns:
            A string of the form ``"lhs = <formatted rhs>"``.
        """
        rhs = self.format(expr)
        return rf"{lhs} = {rhs}"

    def format_aligned(self, terms: dict[int, Expr]) -> str:
        """Format order-by-order as a LaTeX ``align`` environment.

        Args:
            terms: Mapping from perturbative order to expression,
                typically :attr:`PerturbativeResult.order_terms`.

        Returns:
            A complete ``\\begin{align}...\\end{align}`` LaTeX block.
        """
        lines: list[str] = []
        for n in sorted(terms.keys()):
            expr = terms[n]
            rhs = self.format(expr)
            lines.append(rf"  O({n}) &= {rhs} \\")
        body = "\n".join(lines)
        return rf"""\begin{{align}}
{body}
\end{{align}}"""
