"""sft-wick: Wick's theorem contractions for statistical field theory.

Usage:
    from sft_wick import Field, Vertex, Action, compute_moment

    phi = Field('phi', 'physical', n_components=3)
    psi = Field('psi', 'response', n_components=3)

    v = Vertex(fields=[phi, phi, psi], coupling='F')
    action = Action(vertices=[v])

    obs = [psi('a', 'x'), phi('b', 'x'), phi('c', 'x'), phi('d', 'x')]
    result = compute_moment(obs, action, order=1)
    print(result.to_latex())
"""

from .action import Action
from .diagrams import FeynmanDiagram
from .drawing import DiagramRenderer
from .expressions import (
    ZERO,
    ONE,
    Expr,
    IntegralOver,
    Product,
    Propagator,
    Rational,
    Sum,
    SumOverIndex,
    Symbol,
)
from .fields import Field, FieldOperator, FieldType, reset_uid_counter
from .indices import IndexContext
from .latex import LaTeXFormatter
from .perturbation import PerturbativeResult, compute_moment
from .propagators import contract_pair
from .simplify import simplify
from .vertices import Vertex, VertexInstance
from .wick import wick_contract

__all__ = [
    "Action",
    "DiagramRenderer",
    "Expr",
    "FeynmanDiagram",
    "Field",
    "FieldOperator",
    "FieldType",
    "IndexContext",
    "IntegralOver",
    "LaTeXFormatter",
    "ONE",
    "PerturbativeResult",
    "Product",
    "Propagator",
    "Rational",
    "Sum",
    "SumOverIndex",
    "Symbol",
    "Vertex",
    "VertexInstance",
    "ZERO",
    "compute_moment",
    "contract_pair",
    "reset_uid_counter",
    "simplify",
    "wick_contract",
]
