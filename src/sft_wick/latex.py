"""LaTeX output formatting for Wick contraction results."""

from __future__ import annotations

from .expressions import Expr


class LaTeXFormatter:
    """Configurable LaTeX output for expressions.

    Allows overriding propagator display names and formatting options.
    """

    def __init__(
        self,
        propagator_names: dict[str, str] | None = None,
    ) -> None:
        self.propagator_names = propagator_names or {"C": "C", "R": "R"}

    def format(self, expr: Expr) -> str:
        """Format an expression as LaTeX, applying name substitutions."""
        raw = expr.to_latex()
        for key, display in self.propagator_names.items():
            if key != display:
                raw = raw.replace(f"{key}_{{", f"{display}_{{")
                raw = raw.replace(f"{key}(", f"{display}(")
        return raw

    def format_equation(self, lhs: str, expr: Expr) -> str:
        """Format as a LaTeX equation: lhs = rhs."""
        rhs = self.format(expr)
        return rf"{lhs} = {rhs}"

    def format_aligned(self, terms: dict[int, Expr]) -> str:
        """Format order-by-order as a LaTeX align environment."""
        lines: list[str] = []
        for n in sorted(terms.keys()):
            expr = terms[n]
            rhs = self.format(expr)
            lines.append(rf"  O({n}) &= {rhs} \\")
        body = "\n".join(lines)
        return rf"""\begin{{align}}
{body}
\end{{align}}"""
