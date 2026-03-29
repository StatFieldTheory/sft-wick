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
    I,
    ZERO,
    ONE,
    Expr,
    ImaginaryUnit,
    IntegralOver,
    Product,
    Propagator,
    Rational,
    Sum,
    SumOverIndex,
    Symbol,
    apply_response_phase,
)
from .fields import Field, FieldOperator, FieldType, reset_uid_counter
from .indices import IndexContext
from .latex import LaTeXFormatter
from .evaluate import (
    DiagramIntegrand,
    PropagatorCache,
    PropagatorModel,
    SpatialStructure,
    analyze_spatial,
    integrate_moment,
)
from .perturbation import DiagramTerm, PerturbativeResult, compute_moment
from .propagators import contract_pair
from .simplify import collect_by_diagram, collect_by_topology, diagonal_propagators, simplify
from .vertices import Vertex, VertexInstance
from .wick import wick_contract

__all__ = [
    "Action",
    "DiagramIntegrand",
    "DiagramRenderer",
    "DiagramTerm",
    "Expr",
    "FeynmanDiagram",
    "Field",
    "FieldOperator",
    "FieldType",
    "I",
    "ImaginaryUnit",
    "IndexContext",
    "IntegralOver",
    "LaTeXFormatter",
    "ONE",
    "PerturbativeResult",
    "Product",
    "Propagator",
    "PropagatorCache",
    "PropagatorModel",
    "Rational",
    "Sum",
    "SpatialStructure",
    "SumOverIndex",
    "Symbol",
    "Vertex",
    "VertexInstance",
    "ZERO",
    "analyze_spatial",
    "integrate_moment",
    "apply_response_phase",
    "collect_by_diagram",
    "collect_by_topology",
    "compute_moment",
    "contract_pair",
    "diagonal_propagators",
    "reset_uid_counter",
    "simplify",
    "wick_contract",
]
