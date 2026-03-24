"""Action definition and multinomial expansion for perturbation theory."""

from __future__ import annotations

from itertools import combinations_with_replacement
from math import factorial
from typing import Iterator

from .vertices import Vertex


class Action:
    r"""The interaction action :math:`S_{\mathrm{int}}` as a sum of vertex terms.

    The free action :math:`S_0` is implicit --- it defines the propagator
    rules (C and R).  The ``Action`` stores only the interaction vertices.

    Args:
        vertices: List of :class:`~sft_wick.vertices.Vertex` templates
            that make up :math:`S_{\mathrm{int}}`.

    Attributes:
        vertices: The stored list of vertex templates.
    """

    def __init__(self, vertices: list[Vertex]) -> None:
        self.vertices = list(vertices)

    def all_vertex_combinations(
        self, order: int
    ) -> Iterator[tuple[tuple[Vertex, ...], int]]:
        """Generate all ways to pick `order` vertices (with repetition).

        S_int^n = (v_0 + v_1 + ...)^n expands via the multinomial theorem.

        Yields:
            (vertex_sequence, multinomial_coefficient) where vertex_sequence
            has length `order` and multinomial_coefficient = n! / (n_0! * n_1! * ...).
        """
        if order == 0:
            yield (), 1
            return

        k = len(self.vertices)
        for combo in combinations_with_replacement(range(k), order):
            vertex_seq = tuple(self.vertices[i] for i in combo)
            # Compute multinomial coefficient
            counts = [combo.count(i) for i in range(k)]
            coeff = factorial(order)
            for c in counts:
                coeff //= factorial(c)
            yield vertex_seq, coeff
