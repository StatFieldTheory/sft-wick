"""Vertex definitions and vertex instantiation.

A Vertex is a template for an interaction term in S_int.
A VertexInstance is a concrete instantiation with freshly assigned indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .expressions import Symbol
from .fields import Field, FieldOperator
from .indices import IndexContext


@dataclass(frozen=True)
class Vertex:
    """A vertex (interaction term) in S_int.

    Examples:
        # Local vertex: int F_{ijk} phi_i(x) phi_j(x) psi_k(x) dx
        Vertex(fields=[phi, phi, psi], coupling='F')

        # Non-local vertex: iint K_{ij}(x,x') psi_i(x) psi_j(x') dx dx'
        Vertex(fields=[psi, psi], coupling='K', local=False)
    """

    fields: tuple[Field, ...]
    coupling: str
    local: bool = True
    equal_time: bool = False  # non-local only; collapse m time legs into one
    # When True, every ψ leg of this non-local vertex has had its
    # R-propagator pre-integrated into the coupling callable
    # (``κ^(m)_R``). At DiagramTerm construction the R-propagators
    # attached to this vertex's legs are tagged for absorption: each
    # leg's time / direction is aliased onto its Wick partner's, and
    # the integrand R-product loop skips those R-factors. Always
    # ``False`` for local vertices (a local vertex's legs sit at one
    # shared spacetime point, so there is nothing to absorb across).
    already_R_contracted: bool = False

    def __init__(
        self,
        fields: Sequence[Field],
        coupling: str,
        local: bool = True,
        equal_time: bool = False,
        already_R_contracted: bool = False,
    ):
        if equal_time and local:
            raise ValueError(
                "equal_time=True is only valid for non-local vertices "
                "(local=False). A local vertex already shares one "
                "spatial / time leg across all fields, so there is "
                "nothing to collapse."
            )
        if already_R_contracted and local:
            raise ValueError(
                "already_R_contracted=True is only valid for non-local "
                "vertices (local=False). A local vertex's legs sit at a "
                "single spacetime point, so R-absorption is vacuous."
            )
        if already_R_contracted and equal_time:
            raise ValueError(
                "already_R_contracted=True and equal_time=True are "
                "mutually exclusive on a non-local vertex. The "
                "R-contracted callable has already integrated over its "
                "leg coordinates; declaring the result equal-shell "
                "would be vacuous."
            )
        object.__setattr__(self, "fields", tuple(fields))
        object.__setattr__(self, "coupling", coupling)
        object.__setattr__(self, "local", local)
        object.__setattr__(self, "equal_time", bool(equal_time))
        object.__setattr__(self, "already_R_contracted", bool(already_R_contracted))

    @property
    def n_fields(self) -> int:
        return len(self.fields)

    @property
    def n_physical(self) -> int:
        return sum(1 for f in self.fields if f.is_physical)

    @property
    def n_response(self) -> int:
        return sum(1 for f in self.fields if f.is_response)


@dataclass
class VertexInstance:
    """A concrete instance of a vertex with freshly-assigned indices.

    Created during perturbative expansion when we instantiate copies of S_int.
    """

    vertex: Vertex
    field_operators: list[FieldOperator]
    coupling_symbol: Symbol
    spatial_variables: list[str]
    component_indices: list[str]
    copy_id: int
    # For an equal_time non-local vertex: each non-representative leg
    # spatial label maps to the canonical representative (the first
    # leg) so downstream time integration collapses m legs into one.
    # Stored as a tuple of (non_rep, rep) pairs to match the invariant
    # used by ``DiagramTerm.equal_time_aliases`` and
    # ``SpatialStructure.equal_time_aliases``. Empty tuple for local or
    # non-equal-time vertices.
    equal_time_aliases: tuple[tuple[str, str], ...] = ()

    @classmethod
    def instantiate(
        cls,
        vertex: Vertex,
        idx_ctx: IndexContext,
        copy_id: int,
    ) -> VertexInstance:
        """Create a concrete vertex instance with fresh indices.

        For a local vertex: all fields share one spatial variable.
        For a non-local vertex: each field gets its own spatial variable.
        """
        operators: list[FieldOperator] = []
        comp_indices: list[str] = []
        spatial_vars: list[str] = []

        if vertex.local:
            # All fields share the same spatial argument
            shared_spatial = idx_ctx.fresh_spatial_variable()
            spatial_vars.append(shared_spatial)

            for f in vertex.fields:
                if f.is_scalar:
                    op = FieldOperator(
                        field=f,
                        component_index=None,
                        spatial_arg=shared_spatial,
                        uid=-1,  # placeholder, will be set below
                    )
                else:
                    comp_idx = idx_ctx.fresh_component_index()
                    comp_indices.append(comp_idx)
                    op = FieldOperator(
                        field=f,
                        component_index=comp_idx,
                        spatial_arg=shared_spatial,
                        uid=-1,
                    )
                operators.append(op)
        else:
            # Each field gets its own spatial argument
            for f in vertex.fields:
                spatial = idx_ctx.fresh_spatial_variable()
                spatial_vars.append(spatial)

                if f.is_scalar:
                    op = FieldOperator(
                        field=f,
                        component_index=None,
                        spatial_arg=spatial,
                        uid=-1,
                    )
                else:
                    comp_idx = idx_ctx.fresh_component_index()
                    comp_indices.append(comp_idx)
                    op = FieldOperator(
                        field=f,
                        component_index=comp_idx,
                        spatial_arg=spatial,
                        uid=-1,
                    )
                operators.append(op)

        # Assign unique UIDs
        from .fields import _uid_counter

        final_operators: list[FieldOperator] = []
        for op in operators:
            final_op = FieldOperator(
                field=op.field,
                component_index=op.component_index,
                spatial_arg=op.spatial_arg,
                uid=next(_uid_counter),
            )
            final_operators.append(final_op)

        # Build coupling symbol with appropriate indices and spatial args
        if vertex.local:
            # A local vertex has exactly ONE spacetime argument — the shared
            # point created above.  Recording it (a) makes two copies of the
            # same vertex distinguishable at order >= 2, and (b) is what lets a
            # callable, time-dependent coupling c(t) be evaluated at the right
            # place.  ``local=True`` suppresses it in LaTeX so rendering is
            # unchanged for constant couplings.
            coupling_symbol = Symbol(
                name=vertex.coupling,
                indices=tuple(comp_indices),
                spatial_args=(spatial_vars[0],),
                local=True,
            )
        else:
            coupling_symbol = Symbol(
                name=vertex.coupling,
                indices=tuple(comp_indices),
                spatial_args=tuple(spatial_vars),
            )

        alias_pairs: list[tuple[str, str]] = []
        if vertex.equal_time and len(spatial_vars) > 1:
            rep = spatial_vars[0]
            for var in spatial_vars[1:]:
                alias_pairs.append((var, rep))

        return cls(
            vertex=vertex,
            field_operators=final_operators,
            coupling_symbol=coupling_symbol,
            spatial_variables=spatial_vars,
            component_indices=comp_indices,
            copy_id=copy_id,
            equal_time_aliases=tuple(alias_pairs),
        )
