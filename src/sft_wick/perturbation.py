"""Perturbative expansion driver.

Computes <O>_S = sum_{n=0}^{N} (-1)^n / n! * <O * S_int^n>_{S_0}
using Wick's theorem for each order.
"""

from __future__ import annotations

from math import factorial
from typing import TYPE_CHECKING

from .action import Action
from .expressions import (
    ZERO,
    Expr,
    IntegralOver,
    Product,
    Rational,
    Sum,
    SumOverIndex,
)
from .fields import FieldOperator
from .indices import IndexContext
from .simplify import simplify
from .vertices import VertexInstance
from .wick import Pairing, wick_contract

if TYPE_CHECKING:
    pass


class PerturbativeResult:
    """Container for the result of a perturbative calculation."""

    def __init__(
        self,
        order_terms: dict[int, Expr],
        total: Expr,
        diagrams_by_order: dict[int, list[DiagramInfo]],
    ) -> None:
        self.order_terms = order_terms
        self.total = total
        self.diagrams_by_order = diagrams_by_order

    def order(self, n: int) -> Expr:
        """Get the contribution at a specific perturbative order."""
        return self.order_terms.get(n, ZERO)

    def to_latex(self) -> str:
        """Generate LaTeX for the full result, order by order."""
        parts: list[str] = []
        for n in sorted(self.order_terms.keys()):
            expr = self.order_terms[n]
            if not _is_zero(expr):
                parts.append(f"O({n}): {expr.to_latex()}")
        return "\n".join(parts)

    def draw_diagrams(self, order: int | None = None, **kwargs) -> None:
        """Draw Feynman diagrams. If order is specified, only draw that order."""
        from .drawing import DiagramRenderer

        renderer = DiagramRenderer(**kwargs)
        if order is not None:
            diagrams = self.diagrams_by_order.get(order, [])
        else:
            diagrams = []
            for o in sorted(self.diagrams_by_order.keys()):
                diagrams.extend(self.diagrams_by_order[o])

        if not diagrams:
            print("No diagrams to draw.")
            return

        from .diagrams import FeynmanDiagram

        fd_list = [d.to_feynman_diagram() for d in diagrams]
        renderer.draw_all(fd_list)

    def __repr__(self) -> str:
        return self.to_latex()


class DiagramInfo:
    """Lightweight record of a Feynman diagram for deferred construction."""

    def __init__(
        self,
        observable_ops: list[FieldOperator],
        vertex_instances: list[VertexInstance],
        pairing: Pairing,
        coefficient: Rational,
        order: int,
    ) -> None:
        self.observable_ops = observable_ops
        self.vertex_instances = vertex_instances
        self.pairing = pairing
        self.coefficient = coefficient
        self.order = order

    def to_feynman_diagram(self):
        from .diagrams import FeynmanDiagram

        return FeynmanDiagram.from_pairing(
            self.observable_ops,
            self.vertex_instances,
            self.pairing,
        )


def compute_moment(
    observable: list[FieldOperator],
    action: Action,
    order: int,
) -> PerturbativeResult:
    """Compute <O>_S up to the given perturbative order.

    <O>_S = sum_{n=0}^{order} (-1)^n / n! * <O * S_int^n>_{S_0}

    In the MSR formalism, the partition function Z = 1, so there
    is no denominator to worry about.
    """
    order_terms: dict[int, Expr] = {}
    diagrams_by_order: dict[int, list[DiagramInfo]] = {}

    for n in range(order + 1):
        sign = (-1) ** n
        fact = factorial(n)
        order_exprs: list[Expr] = []
        order_diagrams: list[DiagramInfo] = []

        if n == 0:
            # Zeroth order: just <O>_{S_0}
            wick_result, pairings = wick_contract(observable)
            order_exprs.append(wick_result)
            for p in pairings:
                order_diagrams.append(
                    DiagramInfo(observable, [], p, Rational(1), 0)
                )
        else:
            # Expand S_int^n using multinomial theorem
            for vertex_seq, multinomial_coeff in action.all_vertex_combinations(n):
                idx_ctx = IndexContext()

                # Instantiate each vertex copy
                vertex_instances = [
                    VertexInstance.instantiate(v, idx_ctx, copy_id=k)
                    for k, v in enumerate(vertex_seq)
                ]

                # Collect all operators: observable + vertex fields
                all_ops = list(observable)
                for vi in vertex_instances:
                    all_ops.extend(vi.field_operators)

                # Apply Wick's theorem
                wick_result, pairings = wick_contract(all_ops)

                if _is_zero(wick_result):
                    continue

                # Build prefactor: (-1)^n / n! * multinomial_coeff
                prefactor = Rational(sign * multinomial_coeff, fact)

                # Build coupling product
                coupling_factors: list[Expr] = [prefactor]
                for vi in vertex_instances:
                    coupling_factors.append(vi.coupling_symbol)

                # The full term wraps wick_result with couplings, integrals, sums
                coupling_product = Product(tuple(coupling_factors))
                term: Expr = Product((coupling_product, wick_result))

                # Wrap with integrals over internal spatial variables
                for vi in vertex_instances:
                    for var in vi.spatial_variables:
                        term = IntegralOver(var, term)

                # Wrap with summations over internal component indices
                for vi in vertex_instances:
                    for comp_idx in vi.component_indices:
                        field_for_idx = None
                        for op in vi.field_operators:
                            if op.component_index == comp_idx:
                                field_for_idx = op.field
                                break
                        if field_for_idx is not None:
                            term = SumOverIndex(
                                comp_idx, field_for_idx.n_components, term
                            )

                order_exprs.append(term)

                for p in pairings:
                    order_diagrams.append(
                        DiagramInfo(
                            observable,
                            vertex_instances,
                            p,
                            prefactor,
                            n,
                        )
                    )

        if order_exprs:
            raw = order_exprs[0] if len(order_exprs) == 1 else Sum(tuple(order_exprs))
            order_terms[n] = simplify(raw)
        else:
            order_terms[n] = ZERO

        diagrams_by_order[n] = order_diagrams

    # Total
    all_terms = [order_terms[n] for n in range(order + 1) if not _is_zero(order_terms[n])]
    if not all_terms:
        total = ZERO
    elif len(all_terms) == 1:
        total = all_terms[0]
    else:
        total = simplify(Sum(tuple(all_terms)))

    return PerturbativeResult(order_terms, total, diagrams_by_order)


def _is_zero(expr: Expr) -> bool:
    return isinstance(expr, Rational) and expr.is_zero
